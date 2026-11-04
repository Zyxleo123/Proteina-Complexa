#!/usr/bin/env python3
"""Is the dG proxy data-hungry? A learning curve over the NUMBER OF TRAINING TARGETS.

`cpseq_meanscore_report.py` leaves the proxy at within-target rho 0.257 (29 % of the 0.896
ceiling) and the open recommendation was "collect more targets". That is a guess until the
curve is measured, so this measures it: train on K of the 19 available non-test targets,
K in {5, 10, 15, 19}, and score the held-out one.

**Targets and rows are confounded** -- 5 targets is ~500 rows and 19 is ~1900 -- and the
two levers available are different things (more targets, versus more scored sequences per
target). So three arms are run, not one:

* `targets`      -- K targets, ALL their rows. The natural "what if I collect more
                    targets" curve, with volume growing alongside diversity.
* `rows_matched` -- K targets, subsampled to a FIXED total row count. Diversity varies,
                    volume does not. If this rises, target diversity pays for itself.
* `volume`       -- all 19 targets, varying the total row count. Volume varies, diversity
                    does not. If this rises, more scored sequences pay.

The subsets are **nested** (one random permutation per draw, truncated at each K), so the
curve is paired across K and the differences are not contaminated by which targets happened
to be drawn. Averaged over `--draws` permutations per held-out target, and over all 20
held-out targets.

A **shuffled-label control** runs at every K and must stay flat at ~0. If it shows a trend
the harness is manufacturing one and no other row means anything
([[benchmark-needs-positional-control]]).

The predictor is the ridge, because that is what the sweep says to ship: it beats the
22.6k-param network at every aggregator and loss. It is also deterministic, so the only
sampling noise here is which targets and rows were drawn, which is exactly what the draws
average over.

Feature centring note: `extra_matrix` centres the MPNN columns within each target,
including the test target, using feature values only and no labels. That is legitimate --
at triage time every candidate's features are in hand before anything is folded -- but it
is transduction, not pure induction, and is stated so nobody has to rediscover it.

Energies are Rosetta ref2015 REU, NOT measured binding free energies.

Usage:
    python script_utils/cpseq_meanscore_learning_curve.py \
        --dataset .../dataset_v3.json \
        --extra-features .../mpnn_feats_bb_v3.json .../mpnn_feats_ar_v3.json \
        --json .../learning_curve_v3.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_meanscore_baseline import AA, CHARGE, KD  # noqa: E402
from cpseq_meanscore_scorer import (  # noqa: E402
    aggregate_label,
    extra_matrix,
    load_extra_features,
)

ALPHAS = np.logspace(-2, 4, 25)
K_GRID = (2, 3, 5, 10, 15, 19)   # 5/10/15/19 requested; 2 and 3 pin the bend
# Fixed row budget for the diversity-at-constant-volume arm. The smallest target has 74
# rows, so 5 targets can always supply 70 each; anything larger would make K=5 infeasible
# and silently turn the arm into a different comparison.
ROWS_FIXED = 350
VOLUME_GRID = (350, 700, 1400, 0)   # 0 = every available row


def comp_delta(r) -> list[float]:
    s, o = r["sequence"], r["original_sequence"]
    d = [(s.count(a) - o.count(a)) / len(s) for a in AA]
    return d + [
        st.mean(KD.get(c, 0.0) for c in s) - st.mean(KD.get(c, 0.0) for c in o),
        sum(CHARGE.get(c, 0.0) for c in s) - sum(CHARGE.get(c, 0.0) for c in o),
        float(r.get("n_mutations") or 0),
    ]


def fit_and_score(X, y, parents, train_idx, test_idx, shuffle_labels=False, rng=None,
                  offsets=None):
    """Ridge on `train_idx`, within-target Spearman on `test_idx`. None if degenerate.

    `offsets` supplies a per-target label offset estimated from EVERY available row of
    that target instead of from the subsample. That matters for the fixed-row-budget arm:
    at K=19 the budget leaves ~18 rows/target, so the subsample mean has SE ~0.89 REU
    against ~0.45 REU at 70 rows. Noisier centring injects noise into the training label
    and would masquerade as "diversity hurts", so the arm is run both ways and the
    difference is the centring cost.
    """
    Xtr, ytr = X[train_idx], y[train_idx].copy()
    ptr = parents[train_idx]
    if len(set(ptr.tolist())) < 2 or len(train_idx) < 30:
        return None
    # Per-target centring of the LABEL: the model must learn within-target contrast.
    for p in set(ptr.tolist()):
        m = ptr == p
        ytr[m] -= offsets[p] if offsets is not None else ytr[m].mean()
        if shuffle_labels:
            ytr[m] = rng.permutation(ytr[m])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    model = RidgeCV(alphas=ALPHAS)
    model.fit((Xtr - mu) / sd, ytr)
    pred = model.predict((X[test_idx] - mu) / sd)
    if len(set(pred.tolist())) < 2:
        return None
    rho = spearmanr(y[test_idx], pred).statistic
    return float(rho) if rho == rho else None


def pick_rows(idx_by_target, chosen, total, rng):
    """Up to `total` rows spread evenly over `chosen` targets; all rows if total <= 0."""
    if total <= 0:
        return np.concatenate([idx_by_target[t] for t in chosen])
    per = max(1, total // len(chosen))
    out = []
    for t in chosen:
        pool = idx_by_target[t]
        take = min(per, len(pool))
        out.append(rng.choice(pool, take, replace=False))
    return np.concatenate(out)


def boot_ci(vals, n=5000, seed=0):
    rng = np.random.RandomState(seed)
    v = np.asarray(vals, dtype=float)
    draws = [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def saturating(k, a, b):
    """rho = a * k / (k + b): rises, then flattens at asymptote `a`."""
    return a * k / (k + b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--extra-features", nargs="*", default=[], type=Path)
    ap.add_argument("--agg", default="trimmed20")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    payload = json.loads(args.dataset.read_text())
    rows = [r for r in payload["rows"] if r.get("arm") != "reference"]
    rows = [r for r in rows if r.get("original_sequence")
            and len(r["sequence"]) == len(r["original_sequence"])
            and any(a != b for a, b in zip(r["sequence"], r["original_sequence"]))]

    names: list[str] = []
    if args.extra_features:
        merged, names, dropped, have = load_extra_features(args.extra_features, rows)
        rows = [r for r, h in zip(rows, have) if h]
        if dropped:
            print(f"dropped features: {dropped}")

    parents = np.array([r["parent_id"] for r in rows])
    targets = sorted(set(parents.tolist()))
    y = np.array([aggregate_label(r["fold_dG"], args.agg) for r in rows])
    Xc = np.array([comp_delta(r) for r in rows], dtype=float)
    feats = {"composition_delta": Xc}
    if names:
        feats["composition_delta + mpnn"] = np.hstack(
            [Xc, extra_matrix(merged, names, rows, parents.tolist())])

    idx_by_target = {t: np.nonzero(parents == t)[0] for t in targets}
    # Offsets estimated from every row of a target, for the centring-cost decomposition.
    full_offsets = {t: float(y[idx_by_target[t]].mean()) for t in targets}
    min_rows = min(len(v) for v in idx_by_target.values())
    # A fixed row budget is only honest where every chosen target can supply its share.
    k_rows_ok = tuple(k for k in K_GRID if k * min_rows >= ROWS_FIXED)
    print(f"dataset  {args.dataset}")
    print(f"rows     {len(rows)} over {len(targets)} targets "
          f"(median {int(np.median([len(v) for v in idx_by_target.values()]))} rows/target)")
    print(f"label    {args.agg}; metric within-target Spearman, whole targets held out")
    print(f"draws    {args.draws} nested permutations per held-out target")
    print(f"K grid   {K_GRID}   volume grid {VOLUME_GRID}")
    print(f"rows_matched budget {ROWS_FIXED}; feasible only at K in {k_rows_ok} "
          f"(smallest target has {min_rows} rows)\n")

    # arm -> mode -> level -> {test_target: mean rho over draws}
    res: dict = {}

    def record(arm, mode, level, t, vals):
        res.setdefault(arm, {}).setdefault(mode, {}).setdefault(str(level), {})[t] = (
            float(np.mean(vals)) if vals else float("nan"))

    for arm, X in feats.items():
        for ti, t in enumerate(targets):
            pool = [u for u in targets if u != t]
            test_idx = idx_by_target[t]
            acc: dict = {("targets", k): [] for k in K_GRID}
            acc.update({("rows_matched", k): [] for k in k_rows_ok})
            acc.update({("rows_matched_fullcenter", k): [] for k in k_rows_ok})
            acc.update({("volume", v): [] for v in VOLUME_GRID})
            acc.update({("shuffled", k): [] for k in K_GRID})
            for d in range(args.draws):
                rng = np.random.RandomState(1_000_003 * (ti + 1) + d)
                perm = list(rng.permutation(pool))
                for k in K_GRID:
                    chosen = perm[:k]
                    tr_all = np.concatenate([idx_by_target[u] for u in chosen])
                    r1 = fit_and_score(X, y, parents, tr_all, test_idx)
                    if r1 is not None:
                        acc[("targets", k)].append(r1)
                    if k in k_rows_ok:
                        tr_fix = pick_rows(idx_by_target, chosen, ROWS_FIXED, rng)
                        r2 = fit_and_score(X, y, parents, tr_fix, test_idx)
                        if r2 is not None:
                            acc[("rows_matched", k)].append(r2)
                        r2b = fit_and_score(X, y, parents, tr_fix, test_idx,
                                            offsets=full_offsets)
                        if r2b is not None:
                            acc[("rows_matched_fullcenter", k)].append(r2b)
                    r3 = fit_and_score(X, y, parents, tr_all, test_idx,
                                       shuffle_labels=True, rng=rng)
                    if r3 is not None:
                        acc[("shuffled", k)].append(r3)
                for v in VOLUME_GRID:
                    tr_v = pick_rows(idx_by_target, pool, v, rng)
                    r4 = fit_and_score(X, y, parents, tr_v, test_idx)
                    if r4 is not None:
                        acc[("volume", v)].append(r4)
            for (mode, level), vals in acc.items():
                record(arm, mode, level, t, vals)

    def curve(arm, mode, levels):
        out = []
        for lv in levels:
            per_t = res[arm][mode][str(lv)]
            vals = [v for v in per_t.values() if v == v]
            lo, hi = boot_ci(vals) if vals else (float("nan"),) * 2
            out.append((lv, float(np.mean(vals)), lo, hi, len(vals)))
        return out

    ceiling = 0.896
    for arm in feats:
        print("=" * 78)
        print(f"ARM: {arm}")
        print("=" * 78)
        print(f"\n  A. targets  (K targets, ALL rows -- diversity AND volume grow)")
        print(f"     {'K':>3s} {'~rows':>7s} {'rho':>7s} {'95% CI':>17s} {'%ceil':>6s}")
        natural = curve(arm, "targets", K_GRID)
        med_rows = int(np.median([len(v) for v in idx_by_target.values()]))
        for k, m, lo, hi, n in natural:
            print(f"     {k:3d} {k*med_rows:7d} {m:7.3f}   [{lo:+.3f},{hi:+.3f}] "
                  f"{100*m/ceiling:6.1f}")

        print(f"\n  B. rows_matched  (K targets, total rows fixed at {ROWS_FIXED} "
              f"-- diversity ONLY)")
        print(f"      'fullctr' re-centres labels from ALL of a target's rows, so the")
        print(f"     difference from 'rho' is the cost of estimating the offset from few.")
        print(f"     {'K':>3s} {'rows/tgt':>9s} {'rho':>7s} {'95% CI':>17s} "
              f"{'fullctr':>8s} {'ctr cost':>9s}")
        fc = {k: m for k, m, _, _, _ in curve(arm, "rows_matched_fullcenter", k_rows_ok)}
        for k, m, lo, hi, n in curve(arm, "rows_matched", k_rows_ok):
            print(f"     {k:3d} {ROWS_FIXED//k:9d} {m:7.3f}   [{lo:+.3f},{hi:+.3f}] "
                  f"{fc[k]:8.3f} {m - fc[k]:+9.3f}")

        print(f"\n  C. volume  (all 19 targets, total rows varied -- volume ONLY)")
        print(f"     {'rows':>7s} {'rho':>7s} {'95% CI':>17s}")
        for v, m, lo, hi, n in curve(arm, "volume", VOLUME_GRID):
            lab = "all" if v == 0 else str(v)
            print(f"     {lab:>7s} {m:7.3f}   [{lo:+.3f},{hi:+.3f}]")

        # The row axis is the lever this experiment identifies, so fit it as well.
        vol_pts = curve(arm, "volume", VOLUME_GRID)
        n_all = len(rows)
        vx = np.array([n_all if v == 0 else v for v, _, _, _, _ in vol_pts], dtype=float)
        vy = np.array([m for _, m, _, _, _ in vol_pts], dtype=float)
        order = np.argsort(vx)
        vx, vy = vx[order], vy[order]
        try:
            (va, vb), _ = curve_fit(saturating, vx, vy, p0=[0.3, 300.0], maxfev=40000)
            vres = vy - saturating(vx, va, vb)
            print(f"\n  ROW-COUNT FIT  rho = a*R/(R+b):  a (asymptote) = {va:.3f}, "
                  f"b = {vb:.0f} rows")
            print(f"     residuals {np.array2string(vres, precision=3)}  "
                  f"(max |resid| {np.max(np.abs(vres)):.3f})")
            print(f"     at the current {n_all} rows: {saturating(n_all, va, vb):.3f}  "
                  f"({100*saturating(n_all, va, vb)/va:.0f}% of its own asymptote)")
            for mult in (2, 4, 10):
                print(f"     {mult}x the scored sequences ({mult*n_all} rows) -> "
                      f"{saturating(mult*n_all, va, vb):.3f}")
            print(f"     Only 4 points and the curve is still rising at the right edge, "
                  f"so the\n     asymptote is the least trustworthy number here.")
            res.setdefault("_vol_fit", {})[arm] = {
                "asymptote": float(va), "b_rows": float(vb),
                "max_resid": float(np.max(np.abs(vres))),
                "at_current": float(saturating(n_all, va, vb)),
                "proj_2x": float(saturating(2 * n_all, va, vb)),
                "proj_4x": float(saturating(4 * n_all, va, vb)),
                "proj_10x": float(saturating(10 * n_all, va, vb))}
        except Exception as exc:
            print(f"\n  row-count fit failed: {exc}")

        ctrl = curve(arm, "shuffled", K_GRID)
        print(f"\n  CONTROL (labels shuffled within each training target; must be flat ~0)")
        print("     " + "  ".join(f"K={k}:{m:+.3f}" for k, m, _, _, _ in ctrl))
        cmax = max(abs(m) for _, m, _, _, _ in ctrl)
        ctrl_ok = cmax < 0.05
        print(f"     max |rho| {cmax:.3f} -- "
              f"{'OK' if ctrl_ok else 'BROKEN: the harness manufactures a trend'}")

        # Paired test on the span that matters: K=5 -> K=19, same held-out targets.
        a = res[arm]["targets"]["5"]
        b = res[arm]["targets"]["19"]
        common = sorted(set(a) & set(b))
        d = np.array([b[t] - a[t] for t in common if a[t] == a[t] and b[t] == b[t]])
        lo, hi = boot_ci(d)
        try:
            _, p = wilcoxon(d)
        except Exception:
            p = float("nan")
        print(f"\n  PAIRED K=5 -> K=19 (per held-out target)")
        print(f"     mean delta {d.mean():+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  "
              f"p={p:.4f}  {int((d > 0).sum())}/{len(d)} targets up")

        # Extrapolation, on the natural curve only.
        ks = np.array([k for k, _, _, _, _ in natural], dtype=float)
        ms = np.array([m for _, m, _, _, _ in natural], dtype=float)
        try:
            (a_hat, b_hat), _ = curve_fit(saturating, ks, ms,
                                          p0=[0.3, 1.0], maxfev=20000)
            resid = ms - saturating(ks, a_hat, b_hat)
            print(f"\n  SATURATING FIT  rho = a*K/(K+b):  a (asymptote) = {a_hat:.3f}, "
                  f"b = {b_hat:.2f}")
            print(f"     residuals {np.array2string(resid, precision=3)}  "
                  f"(max |resid| {np.max(np.abs(resid)):.3f})")
            print(f"     projected: K=40 -> {saturating(40, a_hat, b_hat):.3f}   "
                  f"K=92 -> {saturating(92, a_hat, b_hat):.3f}")
            print(f"     EXTRAPOLATION FROM 4 POINTS, K<=19. Treat as an order of "
                  f"magnitude, not a forecast.")
            res.setdefault("_fit", {})[arm] = {
                "asymptote": float(a_hat), "b": float(b_hat),
                "max_resid": float(np.max(np.abs(resid))),
                "proj_K40": float(saturating(40, a_hat, b_hat)),
                "proj_K92": float(saturating(92, a_hat, b_hat))}
        except Exception as exc:
            print(f"\n  saturating fit failed: {exc}")
        print()

    # ---- verdict -------------------------------------------------------------------
    arm = "composition_delta + mpnn" if "composition_delta + mpnn" in feats \
        else "composition_delta"
    nat = {k: m for k, m, _, _, _ in curve(arm, "targets", K_GRID)}
    rm = {k: m for k, m, _, _, _ in curve(arm, "rows_matched", k_rows_ok)}
    rmf = {k: m for k, m, _, _, _ in curve(arm, "rows_matched_fullcenter", k_rows_ok)}
    vol = {v: m for v, m, _, _, _ in curve(arm, "volume", VOLUME_GRID)}
    span_nat = nat[19] - nat[5]
    k_lo, k_hi = min(k_rows_ok), max(k_rows_ok)
    span_div = rm[k_hi] - rm[k_lo]
    span_div_fc = rmf[k_hi] - rmf[k_lo]
    span_vol = vol[0] - vol[350]

    v350 = {v: m for v, m, _, _, _ in curve(arm, "volume", VOLUME_GRID)}[350]
    rm19 = rm[max(k_rows_ok)]
    print("=" * 78)
    print("CONSISTENCY CHECK")
    print("=" * 78)
    print(f"  volume@350 rows and rows_matched@K={max(k_rows_ok)} are the SAME")
    print(f"  configuration (19 targets x 18 rows) reached by two independent code")
    print(f"  paths: {v350:.3f} vs {rm19:.3f}, difference {abs(v350 - rm19):.4f}.")
    print(f"  {'OK' if abs(v350 - rm19) < 0.01 else 'MISMATCH -- one of the two arms is wrong'}")

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  natural K=5 -> 19 (diversity + volume) : {span_nat:+.3f}")
    print(f"  diversity alone, rows fixed            : {span_div:+.3f}  "
          f"(K={k_lo}->{k_hi})")
    print(f"    same, with offsets from all rows     : {span_div_fc:+.3f}  "
          f"(centring cost {span_div - span_div_fc:+.3f})")
    print(f"  volume alone, 19 targets               : {span_vol:+.3f}")
    if span_nat < 0.02:
        verdict = "saturated: more targets of this kind will not help"
        print(f"\n  SATURATED. Going 5 -> 19 targets moves rho by only {span_nat:+.3f}.")
        print(f"  The proxy is NOT data-hungry in targets, so collecting more targets")
        print(f"  like these is the wrong investment. The limit is the information in")
        print(f"  the features, not the amount of it.")
    elif span_div > span_vol:
        verdict = "data-hungry in TARGET DIVERSITY"
        print(f"\n  DATA-HUNGRY IN DIVERSITY. At a fixed row budget, going 5 -> 19")
        print(f"  targets is worth {span_div:+.3f}, more than the {span_vol:+.3f} from")
        print(f"  adding rows at fixed diversity. Collect more TARGETS.")
    else:
        verdict = "data-hungry in ROWS (sequences per target)"
        print(f"\n  DATA-HUNGRY IN VOLUME. Adding rows at fixed diversity is worth")
        print(f"  {span_vol:+.3f} against {span_div:+.3f} for diversity at fixed rows.")
        print(f"  Score more SEQUENCES on the targets already in hand.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "dataset": str(args.dataset), "agg": args.agg,
            "feature_names": names, "draws": args.draws,
            "k_grid": list(K_GRID), "rows_fixed": ROWS_FIXED,
            "volume_grid": list(VOLUME_GRID),
            "per_target": res, "verdict": verdict,
            "spans": {"natural": span_nat, "diversity": span_div,
                      "diversity_fullcenter": span_div_fc, "volume": span_vol},
        }, indent=1, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
