#!/usr/bin/env python3
"""Does the proxy survive a change of PREDICTION protocol? The strongest held-out test.

Everything measured so far is leave-one-target-out inside ONE run directory,
`llm_ladder`. `llm_ladder_pot` ran the same targets and the same scoring version
(`v3-cis-amide-muco-windows`) with **Boltz-2 potentials ON**, which changes the predicted
structure and therefore every dG. It is a different measurement instrument on the same
quantity -- so it is a protocol-level held-out set, and a proxy that transfers across it is
far better evidenced than one validated within a single protocol.

All 10 `llm_ladder_pot` targets are among the 20 trained ones, so the test is run
**leave-one-target-out anyway**: to score target t, the ridge is fit on the other 19
`llm_ladder` targets and asked to rank t's `llm_ladder_pot` candidates. Both the target and
the protocol are then unseen. Fitting on all 20 and testing on the 10 would leak the
target and is not done.

Reported: within-target Spearman, the fold-budget triage at pool-appropriate budgets, and
the protocol shift itself (how far the two instruments' labels differ on the same target),
because a proxy that transfers is only interesting if the thing it transfers to is actually
different.

Energies are Rosetta ref2015 REU, NOT measured binding free energies. Lower is better.

Usage:
    python script_utils/cpseq_meanscore_transfer.py \
        --train-dataset .../dataset_v3.json \
        --train-features .../mpnn_feats_bb_v3.json .../mpnn_feats_ar_v3.json \
        --test-dataset  .../dataset_pot_v3.json \
        --test-features .../mpnn_feats_bb_pot_v3.json .../mpnn_feats_ar_pot_v3.json \
        --json .../transfer_v3.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_meanscore_baseline import AA, CHARGE, KD  # noqa: E402
from cpseq_meanscore_scorer import (  # noqa: E402
    aggregate_label,
    extra_matrix,
    load_extra_features,
)
from cpseq_meanscore_triage import (  # noqa: E402
    best_random_curve,
    compute_multiplier,
)

ALPHAS = np.logspace(-2, 4, 25)
BUDGETS = (3, 5, 10)
N_SHUFFLE = 200


def comp_delta(r) -> list[float]:
    s, o = r["sequence"], r["original_sequence"]
    d = [(s.count(a) - o.count(a)) / len(s) for a in AA]
    return d + [
        st.mean(KD.get(c, 0.0) for c in s) - st.mean(KD.get(c, 0.0) for c in o),
        sum(CHARGE.get(c, 0.0) for c in s) - sum(CHARGE.get(c, 0.0) for c in o),
        float(r.get("n_mutations") or 0),
    ]


def prep(path: Path, feat_paths, agg: str):
    payload = json.loads(Path(path).read_text())
    rows = [r for r in payload["rows"] if r.get("arm") != "reference"]
    rows = [r for r in rows if r.get("original_sequence")
            and len(r["sequence"]) == len(r["original_sequence"])
            and any(a != b for a, b in zip(r["sequence"], r["original_sequence"]))]
    names: list[str] = []
    if feat_paths:
        merged, names, dropped, have = load_extra_features(
            [Path(p) for p in feat_paths], rows)
        rows = [r for r, h in zip(rows, have) if h]
    parents = [r["parent_id"] for r in rows]
    y = np.array([aggregate_label(r["fold_dG"], agg) for r in rows])
    Xc = np.array([comp_delta(r) for r in rows], dtype=float)
    X = Xc
    if names:
        X = np.hstack([Xc, extra_matrix(merged, names, rows, parents)])
    return rows, np.asarray(parents), X, y, names


def boot_ci(vals, n=5000, seed=0):
    """One bootstrap sample, both percentiles from it -- drawing two independent sets
    for the two ends would give a wider, incoherent interval."""
    rng = np.random.RandomState(seed)
    v = np.asarray(vals, dtype=float)
    draws = [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dataset", type=Path, required=True)
    ap.add_argument("--train-features", nargs="*", default=[])
    ap.add_argument("--test-dataset", type=Path, required=True)
    ap.add_argument("--test-features", nargs="*", default=[])
    ap.add_argument("--agg", default="trimmed20")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    _, pa_tr, X_tr, y_tr, n_tr = prep(args.train_dataset, args.train_features, args.agg)
    _, pa_te, X_te, y_te, n_te = prep(args.test_dataset, args.test_features, args.agg)
    if n_tr != n_te:
        raise SystemExit(f"feature sets differ:\n  train {n_tr}\n  test  {n_te}\n"
                         f"The same columns in the same order are required.")

    te_targets = sorted(set(pa_te.tolist()))
    print(f"train   {args.train_dataset.name}: {len(y_tr)} rows / "
          f"{len(set(pa_tr.tolist()))} targets")
    print(f"test    {args.test_dataset.name}: {len(y_te)} rows / "
          f"{len(te_targets)} targets")
    print(f"features {n_tr if n_tr else 'sequence only'}")
    overlap = sorted(set(pa_te.tolist()) & set(pa_tr.tolist()))
    print(f"targets in BOTH runs: {len(overlap)} -- each is held out of its own fit\n")

    # ---- the protocol shift: are the two instruments actually different? ------------
    print("PROTOCOL SHIFT (same target, same scoring version, potentials ON vs OFF)")
    print(f"  {'target':24s} {'n_off':>6s} {'n_on':>6s} {'med_off':>8s} {'med_on':>8s} "
          f"{'shift':>7s}")
    shift_rows = {}
    for t in te_targets:
        a, b = y_tr[pa_tr == t], y_te[pa_te == t]
        if not len(a) or not len(b):
            continue
        shift_rows[t] = {"median_train": float(np.median(a)),
                         "median_test": float(np.median(b)),
                         "shift": float(np.median(b) - np.median(a)),
                         "n_train": int(len(a)), "n_test": int(len(b))}
        print(f"  {t:24s} {len(a):6d} {len(b):6d} {np.median(a):8.2f} "
              f"{np.median(b):8.2f} {np.median(b) - np.median(a):+7.2f}")
    shifts = [v["shift"] for v in shift_rows.values()]
    if shifts:
        print(f"  median |shift| across targets: {np.median(np.abs(shifts)):.2f} REU "
              f"(within-target signal SD is 3.78 REU)\n")

    # ---- transfer: fit on train minus t, rank t's TEST candidates --------------------
    per_target, triage = {}, {}
    curves = {}
    for t in te_targets:
        tr = pa_tr != t
        te = pa_te == t
        if te.sum() < 4 or tr.sum() < 50:
            continue
        mu, sd = X_tr[tr].mean(0), X_tr[tr].std(0)
        sd[sd == 0] = 1.0
        ytr = y_tr[tr].copy()
        for p in set(pa_tr[tr].tolist()):
            m = pa_tr[tr] == p
            ytr[m] -= ytr[m].mean()
        model = RidgeCV(alphas=ALPHAS)
        model.fit((X_tr[tr] - mu) / sd, ytr)
        pred = model.predict((X_te[te] - mu) / sd)
        yv = y_te[te]
        if len(set(pred)) < 2:
            continue
        rho = spearmanr(yv, pred).statistic
        if rho != rho:
            continue
        per_target[t] = float(rho)
        curves[t] = best_random_curve(yv)
        order = np.argsort(pred, kind="stable")
        triage[t] = {}
        for m in BUDGETS:
            if m >= len(yv):
                continue
            ach = float(yv[order[:m]].min())
            triage[t][str(m)] = {
                "best_proxy": ach, "best_random": float(curves[t][m]),
                "gain_REU": float(curves[t][m]) - ach,
                "multiplier": compute_multiplier(curves[t], ach, m),
                "n": int(len(yv)),
            }

    rhos = list(per_target.values())
    lo, hi = boot_ci(rhos) if rhos else (float("nan"),) * 2
    print("TRANSFER: within-target rho on the UNSEEN protocol (target also held out)")
    for t, r in sorted(per_target.items(), key=lambda kv: -kv[1]):
        print(f"  {t:24s} {r:+.3f}")
    print(f"\n  mean {st.mean(rhos):+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  "
          f"over {len(rhos)} targets")
    print(f"  same-protocol LOTO reference : +0.253")
    print(f"  label ceiling                :  0.896")

    # ---- shuffled control on the test pool ------------------------------------------
    ctrl_rhos = []
    rs = np.random.RandomState(0)
    for t in per_target:
        yv = y_te[pa_te == t]
        vals = []
        for _ in range(N_SHUFFLE):
            r = spearmanr(yv, rs.permutation(yv)).statistic
            if r == r:
                vals.append(r)
        ctrl_rhos.append(float(np.mean(vals)))
    if ctrl_rhos:
        print(f"  shuffled control             : {st.mean(ctrl_rhos):+.3f} "
              f"({N_SHUFFLE} shuffles/target)")

    print("\nTRANSFER TRIAGE (fold-budget on the unseen protocol)")
    print(f"  {'M':>3s} {'pool':>5s} {'best':>8s} {'rand':>8s} {'gain':>7s} {'mult':>5s}")
    tri_summary = {}
    for m in BUDGETS:
        key = str(m)
        vals = [v[key] for v in triage.values() if key in v]
        if not vals:
            continue
        gm = float(np.mean([v["gain_REU"] for v in vals]))
        glo, ghi = boot_ci([v["gain_REU"] for v in vals])
        agg_ach = float(np.mean([v["best_proxy"] for v in vals]))
        mc = [curves[t] for t in triage if key in triage[t]]
        n_min = min(len(c) - 1 for c in mc)
        mean_curve = np.array([float(np.mean([c[mm] for c in mc]))
                               for mm in range(1, n_min + 1)])
        hit = np.nonzero(mean_curve <= agg_ach)[0]
        mult = (int(hit[0]) + 1) / m if len(hit) else n_min / m
        tri_summary[key] = {"gain_REU": gm, "gain_ci": [glo, ghi],
                            "multiplier_agg": mult,
                            "best_proxy": agg_ach,
                            "best_random": float(np.mean([v["best_random"] for v in vals])),
                            "mean_pool": float(np.mean([v["n"] for v in vals]))}
        print(f"  {m:3d} {np.mean([v['n'] for v in vals]):5.0f} {agg_ach:8.2f} "
              f"{np.mean([v['best_random'] for v in vals]):8.2f} {gm:+7.2f} {mult:5.2f}")

    print("\nVERDICT")
    if not rhos:
        print("  INCONCLUSIVE: no target had enough rows on both sides.")
        verdict = "inconclusive"
    elif lo > 0:
        verdict = "transfers"
        print(f"  TRANSFERS. Mean within-target rho {st.mean(rhos):+.3f} with CI "
              f"[{lo:+.3f},{hi:+.3f}] excluding zero,")
        print(f"  on a protocol the model never saw and targets held out of its own fit.")
        print(f"  That is {100*st.mean(rhos)/0.253:.0f}% of the same-protocol LOTO rho, so "
              f"the signal is")
        print(f"  {'largely protocol-independent' if st.mean(rhos) > 0.8*0.253 else 'degraded but present'}.")
    else:
        verdict = "does not transfer"
        print(f"  DOES NOT TRANSFER. Mean rho {st.mean(rhos):+.3f}, CI "
              f"[{lo:+.3f},{hi:+.3f}] includes zero.")
        print(f"  The proxy is tied to the prediction protocol it was trained on; do not")
        print(f"  use it to triage candidates that will be folded with other settings.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "train_dataset": str(args.train_dataset),
            "test_dataset": str(args.test_dataset),
            "feature_names": n_tr, "agg": args.agg,
            "protocol_shift": shift_rows,
            "per_target_rho": per_target,
            "mean_rho": st.mean(rhos) if rhos else None,
            "rho_ci": [lo, hi],
            "control_rho": st.mean(ctrl_rhos) if ctrl_rhos else None,
            "same_protocol_loto_rho": 0.253,
            "triage": tri_summary, "per_target_triage": triage,
            "verdict": verdict,
        }, indent=1, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
