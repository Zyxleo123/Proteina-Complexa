#!/usr/bin/env python3
"""Does the proxy save folds? The deployment number, not a correlation.

`cpseq_meanscore_report.py` ends on within-target Spearman rho = 0.257 for the winning
ridge, which is 29 % of the label's ceiling. That is the right number for "is there
signal", and the wrong number for "should the search use it". The search does not consume
a correlation: it has ~100 candidates for a target, can afford to fold M of them at
~238 s each, and wants the best dG it can get for that budget.

So this measures exactly that. Leave-one-target-out, the proxy ranks the held-out target's
candidates, the top M are "folded", and what is reported is:

* **best-of-M dG** achieved by proxy selection vs random selection vs an oracle;
* **compute multiplier** -- the number of RANDOM folds needed to match what the proxy
  found in M, which is the saving in the only unit that matters;
* **recall@M** of the true top decile.

Three deliberate controls, because [[benchmark-needs-positional-control]] is the house
rule: a `shuffled` predictor must come out at multiplier ~1.0, an `oracle` must come out at
~n/M, and `n_mutations` is the trivial heuristic the proxy has to beat. If the shuffled arm
does not land on 1.0, the harness is wrong and no other row means anything.

**The oracle is computed split-half** (select on one half of the ~10 folds, evaluate on the
other) because an oracle that selects on the same noisy label it is scored against is
picking the luckiest seed, not the best peptide, and reads far too high. Proxy and random
are both min-of-M-noisy-labels over different subsets, so noise enters them identically and
comparing them directly is fair.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies. Lower is
better throughout, so "best" means minimum.

Usage:
    python script_utils/cpseq_meanscore_triage.py \
        --dataset /zfsauton/scratch/yixiz/cpseq/meanscore/dataset_v3.json \
        --extra-features .../mpnn_feats_bb_v3.json .../mpnn_feats_ar_v3.json \
        --json /zfsauton/scratch/yixiz/cpseq/meanscore/triage_v3.json
    python script_utils/cpseq_meanscore_triage.py --self-test
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_meanscore_baseline import AA, CHARGE, KD  # noqa: E402
from cpseq_meanscore_scorer import (  # noqa: E402
    aggregate_label,
    extra_matrix,
    load_extra_features,
)

ALPHAS = np.logspace(-2, 4, 25)
BUDGETS = (5, 10, 20, 30, 50)
N_SHUFFLE = 200      # shuffled-control draws per target; shrinks its SE by sqrt(N)
TOP_DECILE = 0.10


def comp_delta(r) -> list[float]:
    """Composition difference from the reference peptide, plus mutation count."""
    s, o = r["sequence"], r["original_sequence"]
    d = [(s.count(a) - o.count(a)) / len(s) for a in AA]
    return d + [
        st.mean(KD.get(c, 0.0) for c in s) - st.mean(KD.get(c, 0.0) for c in o),
        sum(CHARGE.get(c, 0.0) for c in s) - sum(CHARGE.get(c, 0.0) for c in o),
        float(r.get("n_mutations") or 0),
    ]


def best_random_curve(y: np.ndarray) -> np.ndarray:
    """E[min of m labels drawn without replacement], EXACTLY, for every m = 1..n.

    No Monte Carlo: for a sorted ascending y, the probability that y_i (1-based) is the
    minimum of an m-subset is C(n-i, m-1)/C(n, m) -- the other m-1 picks must all come
    from the n-i entries above it. Computed in log space so n=100, m=50 does not overflow.

    Returns a length-(n+1) array indexed by m, with index 0 unused (set to +inf).
    """
    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)
    lg = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])  # log k!

    curve = np.full(n + 1, np.inf)
    i = np.arange(1, n + 1)
    for m in range(1, n + 1):
        # log C(n-i, m-1) - log C(n, m), vectorised over i
        num = np.full(n, -np.inf)
        rem = n - i
        ok = (m - 1) <= rem
        num[ok] = lg[rem[ok]] - lg[m - 1] - lg[rem[ok] - (m - 1)]
        w = np.exp(num - (lg[n] - lg[m] - lg[n - m]))
        curve[m] = float(w @ y)
    return curve


def compute_multiplier(curve: np.ndarray, achieved: float, m: int) -> float:
    """How many RANDOM folds match `achieved`? Returned as a multiple of m.

    `curve` is monotone non-increasing in m, so the first index that reaches `achieved`
    is the answer. Capped at n: if folding everything does not match the proxy, the proxy
    beat the full enumeration on this label and n/m is the honest floor on the saving.

    **Never average this across targets.** It is a steeply nonlinear function of
    `achieved`, so E[multiplier] != multiplier(E[achieved]) by a wide margin: a lucky
    random draw needs many extra folds to match because the curve is flat out there,
    while an unlucky one can only save down to 1/m. The mean is therefore biased upward
    even for a random selector -- measured at 1.79 where the truth is 1.0, which is what
    the shuffled control in `self_test` exists to catch. Aggregate with the MEDIAN (a
    monotone transform commutes with it), or transform the aggregate via
    `aggregate_multiplier`.
    """
    n = len(curve) - 1
    hit = np.nonzero(curve[1:] <= achieved)[0]
    return (int(hit[0]) + 1) / m if len(hit) else n / m


def aggregate_multiplier(curves: list[np.ndarray], achieved_mean: float,
                         m: int) -> float:
    """Portfolio multiplier: averaging the random curves FIRST, then inverting.

    This is the "across my targets, the proxy's mean best-of-M equals a random
    best-of-how-many" reading. Transforming the aggregate rather than aggregating the
    transform makes it exactly 1.0 for a random selector, by construction.
    """
    n_min = min(len(c) - 1 for c in curves)
    mean_curve = np.array([float(np.mean([c[mm] for c in curves]))
                           for mm in range(1, n_min + 1)])
    hit = np.nonzero(mean_curve <= achieved_mean)[0]
    return (int(hit[0]) + 1) / m if len(hit) else n_min / m


def evaluate_target(scores: np.ndarray, y: np.ndarray, y_a: np.ndarray,
                    y_b: np.ndarray) -> dict:
    """One held-out target. `scores` predict dG (lower = better), as does `y`."""
    n = len(y)
    curve = best_random_curve(y)
    curve_b = best_random_curve(y_b)
    order = np.argsort(scores, kind="stable")
    k_top = max(1, int(round(n * TOP_DECILE)))
    true_top = set(np.argsort(y, kind="stable")[:k_top].tolist())
    out = {"n_candidates": n, "k_top_decile": k_top, "budgets": {}}
    for m in BUDGETS:
        if m >= n:
            continue
        sel = order[:m]
        achieved = float(y[sel].min())
        rand = float(curve[m])
        hits = len(true_top & set(sel.tolist()))
        out["budgets"][str(m)] = {
            "best_proxy": achieved,
            "best_random": rand,
            "gain_REU": rand - achieved,            # positive = proxy is better
            "multiplier": compute_multiplier(curve, achieved, m),
            "recall_top_decile": hits / k_top,
            "enrichment": (hits / m) / (k_top / n) if m else float("nan"),
        }
    # Split-half oracle: pick on half A, score on half B, so it cannot select on the
    # noise it is graded against.
    out["oracle_split_half"] = {}
    for m in BUDGETS:
        if m >= n:
            continue
        sel = np.argsort(y_a, kind="stable")[:m]
        out["oracle_split_half"][str(m)] = {
            "best": float(y_b[sel].min()),
            "best_random": float(curve_b[m]),
        }
    return out


def loto_scores(X, y, parents, targets, min_seq=4):
    """Predicted dG for each target's candidates, from a ridge fit on the other targets."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    parents = np.asarray(parents)
    preds = {}
    for t in targets:
        te, tr = parents == t, parents != t
        if te.sum() < min_seq:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        ytr = y[tr].copy()
        for p in set(parents[tr]):          # centre each training target: the model must
            msk = parents[tr] == p          # learn within-target contrast, not target id
            ytr[msk] -= ytr[msk].mean()
        model = RidgeCV(alphas=ALPHAS)
        model.fit((X[tr] - mu) / sd, ytr)
        preds[t] = model.predict((X[te] - mu) / sd)
    return preds


def boot_ci(vals, n=5000, seed=0):
    rng = np.random.RandomState(seed)
    v = np.asarray(vals, dtype=float)
    draws = [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def self_test() -> int:
    """The harness must put a shuffled predictor at 1.0 and an oracle near n/m."""
    rng = np.random.RandomState(0)
    n = 100
    y = rng.normal(-20, 4, n)
    fails = []

    perfect = evaluate_target(y.copy(), y, y, y)
    for m, d in perfect["budgets"].items():
        if abs(d["recall_top_decile"] - min(1.0, int(m) / perfect["k_top_decile"])) > 1e-9:
            fails.append(f"oracle recall wrong at m={m}: {d['recall_top_decile']}")
        if d["multiplier"] < 0.8 * (n / int(m)) - 1e-6:
            fails.append(f"oracle multiplier too low at m={m}: {d['multiplier']:.2f} "
                         f"(expected ~{n/int(m):.1f})")

    shuf, achieved = [], []
    for s in range(200):
        r = np.random.RandomState(100 + s)
        sc = y.copy()
        r.shuffle(sc)
        res = evaluate_target(sc, y, y, y)
        shuf.append(res["budgets"]["20"]["multiplier"])
        achieved.append(res["budgets"]["20"]["best_proxy"])
    med, mean = float(np.median(shuf)), float(np.mean(shuf))
    agg = aggregate_multiplier([best_random_curve(y)], float(np.mean(achieved)), 20)
    if not 0.75 <= med <= 1.35:
        fails.append(f"shuffled MEDIAN multiplier should be ~1.0, got {med:.3f}")
    if not 0.9 <= agg <= 1.15:
        fails.append(f"shuffled AGGREGATE multiplier should be ~1.0, got {agg:.3f}")
    if mean <= 1.35:
        fails.append(f"the mean multiplier is supposed to be biased high (that is why "
                     f"the median is used); got {mean:.3f} -- recheck the reasoning")

    anti = evaluate_target(-y, y, y, y)
    if anti["budgets"]["20"]["recall_top_decile"] > 0.01:
        fails.append("anti-correlated predictor recalled top decile")

    print("SELF-TEST")
    print(f"  oracle   multiplier @m=20 : {perfect['budgets']['20']['multiplier']:.2f} "
          f"(expect ~{n/20:.1f})   recall {perfect['budgets']['20']['recall_top_decile']:.2f}")
    print(f"  shuffled median mult@m=20 : {med:.3f} (expect ~1.0, 200 shuffles)")
    print(f"  shuffled aggreg mult@m=20 : {agg:.3f} (expect ~1.0)")
    print(f"  shuffled MEAN   mult@m=20 : {mean:.3f} (expect >1.3 -- biased by "
          f"construction, which is why it is not reported)")
    print(f"  anti     recall     @m=20 : {anti['budgets']['20']['recall_top_decile']:.2f} "
          f"(expect 0.00)")
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        return 1
    print("  all controls pass")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--extra-features", nargs="*", default=[], type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--agg", default="trimmed20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())
    if not args.dataset:
        raise SystemExit("--dataset is required unless --self-test")

    payload = json.loads(args.dataset.read_text())
    rows = [r for r in payload["rows"] if r.get("arm") != "reference"]
    rows = [r for r in rows if r.get("original_sequence")
            and len(r["sequence"]) == len(r["original_sequence"])
            and any(a != b for a, b in zip(r["sequence"], r["original_sequence"]))]

    names: list[str] = []
    merged = {}
    if args.extra_features:
        merged, names, dropped, have = load_extra_features(args.extra_features, rows)
        rows = [r for r, h in zip(rows, have) if h]
        if dropped:
            print(f"dropped features: {dropped}")

    parents = [r["parent_id"] for r in rows]
    targets = sorted(set(parents))
    y = np.array([aggregate_label(r["fold_dG"], args.agg) for r in rows])
    # Independent halves of the fold list, for the honest oracle.
    y_a = np.array([aggregate_label(r["fold_dG"][0::2], args.agg) for r in rows])
    y_b = np.array([aggregate_label(r["fold_dG"][1::2], args.agg) for r in rows])

    Xc = np.array([comp_delta(r) for r in rows], dtype=float)
    feats = {"composition_delta": Xc}
    if names:
        Xe = extra_matrix(merged, names, rows, parents)
        feats["composition_delta + mpnn"] = np.hstack([Xc, Xe])
    n_mut = np.array([[float(r.get("n_mutations") or 0)] for r in rows])

    print(f"dataset   {args.dataset}")
    print(f"rows      {len(rows)} over {len(targets)} targets")
    print(f"label     {args.agg}   (lower REU = better)")
    print(f"features  {', '.join(names) if names else 'sequence only'}\n")

    arms: dict[str, dict] = {}
    pa = np.asarray(parents)

    for arm_name, X in feats.items():
        arms[arm_name] = loto_scores(X, y, pa, targets)
    # Controls.
    arms["n_mutations"] = {t: n_mut[pa == t, 0] for t in targets}

    curves = {t: best_random_curve(y[pa == t]) for t in targets}
    results: dict[str, dict] = {}
    for arm_name, preds in arms.items():
        per_t = {}
        for t in targets:
            if t not in preds:
                continue
            m = pa == t
            per_t[t] = evaluate_target(np.asarray(preds[t], dtype=float), y[m],
                                       y_a[m], y_b[m])
        results[arm_name] = per_t

    # The control, averaged over many shuffles. ONE shuffled draw per target is an
    # unbiased but very noisy estimate of zero -- with 20 targets it lands anywhere in
    # +/-1 REU, which is the same size as the effect being tested and would leave the
    # control unable to rule anything out. Averaging N_SHUFFLE draws shrinks its standard
    # error by sqrt(N). The linear quantities are averaged and the multiplier is then
    # recomputed from the averaged best-of-M, never averaged itself (see
    # `compute_multiplier`).
    ctrl_per_t = {}
    for t in targets:
        msk = pa == t
        yt, yat, ybt = y[msk], y_a[msk], y_b[msk]
        draws = []
        for sh in range(N_SHUFFLE):
            r = np.random.RandomState(10_000 * (args.seed + 1) + sh)
            draws.append(evaluate_target(r.permutation(yt), yt, yat, ybt))
        merged = {"n_candidates": draws[0]["n_candidates"],
                  "k_top_decile": draws[0]["k_top_decile"],
                  "n_shuffles": N_SHUFFLE,
                  "oracle_split_half": draws[0]["oracle_split_half"], "budgets": {}}
        for key in draws[0]["budgets"]:
            avg = {k: float(np.mean([d["budgets"][key][k] for d in draws]))
                   for k in ("best_proxy", "best_random", "gain_REU",
                             "recall_top_decile", "enrichment")}
            avg["multiplier"] = compute_multiplier(curves[t], avg["best_proxy"], int(key))
            merged["budgets"][key] = avg
        ctrl_per_t[t] = merged
    results["shuffled (control)"] = ctrl_per_t

    print("FOLD-BUDGET TRIAGE   (leave-one-target-out, averaged over 20 targets)")
    print("  'mult~' = aggregate multiplier (mean best-of-M inverted through the mean")
    print("            random curve); 'multM' = median per-target multiplier. The MEAN")
    print("            multiplier is biased high and is deliberately not shown.")
    print("  'gain' = REU the proxy's best-of-M beats a random best-of-M by.\n")
    hdr = f"  {'arm':26s} {'M':>3s} {'best':>7s} {'rand':>7s} {'gain':>6s} " \
          f"{'mult~':>5s} {'multM':>5s} {'recall':>7s} {'enrich':>6s}"
    print(hdr)
    summary: dict[str, dict] = {}
    for arm_name, per_t in results.items():
        summary[arm_name] = {}
        for m in BUDGETS:
            key = str(m)
            vals = [v["budgets"][key] for v in per_t.values() if key in v["budgets"]]
            if not vals:
                continue
            row = {k: float(np.mean([v[k] for v in vals])) for k in
                   ("best_proxy", "best_random", "gain_REU",
                    "recall_top_decile", "enrichment")}
            row["multiplier_median"] = float(np.median([v["multiplier"] for v in vals]))
            row["multiplier_agg"] = aggregate_multiplier(
                [curves[t] for t in per_t], row["best_proxy"], m)
            row["gain_ci"] = list(boot_ci([v["gain_REU"] for v in vals]))
            summary[arm_name][key] = row
            print(f"  {arm_name:26s} {m:3d} {row['best_proxy']:7.2f} "
                  f"{row['best_random']:7.2f} {row['gain_REU']:+6.2f} "
                  f"{row['multiplier_agg']:5.2f} {row['multiplier_median']:5.2f} "
                  f"{row['recall_top_decile']:7.2f} {row['enrichment']:6.2f}")
        print()

    # The honest oracle, from the split-half protocol.
    best_arm = "composition_delta + mpnn" if "composition_delta + mpnn" in results \
        else "composition_delta"
    print("HEADROOM   (split-half oracle: selects on half the folds, scored on the other)")
    print(f"  {'M':>3s} {'oracle':>7s} {'proxy':>7s} {'random':>7s} {'captured':>9s}")
    headroom = {}
    for m in BUDGETS:
        key = str(m)
        per_t = results[best_arm]
        orc = [v["oracle_split_half"][key]["best"] for v in per_t.values()
               if key in v["oracle_split_half"]]
        orr = [v["oracle_split_half"][key]["best_random"] for v in per_t.values()
               if key in v["oracle_split_half"]]
        if not orc or key not in summary[best_arm]:
            continue
        o, r0 = float(np.mean(orc)), float(np.mean(orr))
        p = summary[best_arm][key]["best_proxy"]
        rnd = summary[best_arm][key]["best_random"]
        frac = (rnd - p) / (rnd - o) if abs(rnd - o) > 1e-9 else float("nan")
        headroom[key] = {"oracle": o, "oracle_random": r0, "proxy": p,
                         "random": rnd, "fraction_captured": frac}
        print(f"  {m:3d} {o:7.2f} {p:7.2f} {rnd:7.2f} {100*frac:8.1f}%")

    # Paired test: does the proxy beat random, target by target, at a realistic budget?
    print("\nPAIRED PER-TARGET TEST   (proxy vs random best-of-M, M=20)")
    verdicts = {}
    for arm_name in results:
        per_t = results[arm_name]
        d = np.array([v["budgets"]["20"]["gain_REU"] for v in per_t.values()
                      if "20" in v["budgets"]])
        if len(d) < 5:
            continue
        lo, hi = boot_ci(d)
        try:
            _, p = wilcoxon(d)
        except Exception:
            p = float("nan")
        n_up = int((d > 0).sum())
        verdicts[arm_name] = {"mean_gain": float(d.mean()), "ci": [lo, hi],
                              "wilcoxon_p": float(p), "n_improved": n_up,
                              "n_targets": len(d)}
        print(f"  {arm_name:26s} gain {d.mean():+6.2f} REU  CI [{lo:+.2f},{hi:+.2f}]  "
              f"p={p:.4f}  {n_up}/{len(d)} targets up")

    # The decisive comparison is proxy vs the CONTROL, paired per target: both are
    # measured against the same analytic random baseline, so any residual bias in that
    # baseline cancels in the difference.
    pc = {}
    if "shuffled (control)" in results and best_arm in results:
        common = sorted(set(results[best_arm]) & set(results["shuffled (control)"]))
        gp = np.array([results[best_arm][t]["budgets"]["20"]["gain_REU"] for t in common])
        gc = np.array([results["shuffled (control)"][t]["budgets"]["20"]["gain_REU"]
                       for t in common])
        dd = gp - gc
        lo2, hi2 = boot_ci(dd)
        try:
            _, p2 = wilcoxon(dd)
        except Exception:
            p2 = float("nan")
        pc = {"mean": float(dd.mean()), "ci": [lo2, hi2], "wilcoxon_p": float(p2),
              "n_improved": int((dd > 0).sum()), "n_targets": len(dd)}
        print(f"\nPROXY vs CONTROL, paired per target (M=20) -- the decisive test")
        print(f"  mean advantage over the shuffled control: {dd.mean():+.2f} REU")
        print(f"  95% CI [{lo2:+.2f},{hi2:+.2f}]   p={p2:.4f}   "
              f"{int((dd > 0).sum())}/{len(dd)} targets up")

    ctrl = verdicts.get("shuffled (control)", {})
    # The control passes when its own CI includes zero: a skill-free selector must be
    # indistinguishable from the analytic random baseline. A fixed REU threshold would
    # be arbitrary; this scales with the data's own spread.
    cci = ctrl.get("ci", [-9, 9])
    ok = cci[0] <= 0 <= cci[1]
    print("\nCONTROL CHECK")
    print(f"  shuffled arm gain {ctrl.get('mean_gain', float('nan')):+.2f} REU  "
          f"CI [{cci[0]:+.2f},{cci[1]:+.2f}]  ({N_SHUFFLE} shuffles/target)")
    print(f"  -- {'OK: CI includes zero, the harness is not manufacturing a gain' if ok else 'BROKEN: a skill-free selector scores a gain'}")

    print("\nVERDICT")
    v = verdicts.get(best_arm, {})
    m20 = summary.get(best_arm, {}).get("20", {})
    if not ok:
        print("  INVALID: the shuffled control shows a gain, so the harness is wrong.")
    elif pc and pc["ci"][0] > 0:
        print(f"  The proxy saves folds. At a budget of 20 of ~105 candidates it beats "
              f"the shuffled")
        print(f"  control by {pc['mean']:+.2f} REU (CI [{pc['ci'][0]:+.2f},"
              f"{pc['ci'][1]:+.2f}], p={pc['wilcoxon_p']:.4f}, "
              f"{pc['n_improved']}/{pc['n_targets']} targets),")
        print(f"  reaching a dG that would otherwise take "
              f"{m20.get('multiplier_agg', 0):.1f}x the folds at random.")
        print(f"  Captured {100*headroom.get('20', {}).get('fraction_captured', 0):.0f}% "
              f"of the oracle's headroom at that budget -- real, and far from solved.")
    else:
        adv = pc.get("mean", float("nan"))
        aci = pc.get("ci", [float("nan")] * 2)
        print(f"  NOT ESTABLISHED: the proxy's advantage over the shuffled control is "
              f"{adv:+.2f} REU")
        print(f"  with CI [{aci[0]:+.2f},{aci[1]:+.2f}], which includes zero. Its raw "
              f"gain vs the analytic")
        print(f"  random baseline ({v.get('mean_gain', float('nan')):+.2f} REU) is NOT "
              f"sufficient evidence on its own.")
        print(f"  Do not wire this into the search on this evidence.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "dataset": str(args.dataset), "agg": args.agg,
            "feature_names": names, "budgets": list(BUDGETS),
            "n_rows": len(rows), "n_targets": len(targets),
            "summary": summary, "headroom": headroom, "paired": verdicts,
            "proxy_vs_control": pc, "n_shuffles": N_SHUFFLE,
            "per_target": results, "control_ok": ok,
        }, indent=1, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
