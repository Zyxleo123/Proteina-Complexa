#!/usr/bin/env python3
"""Is the pretrained-feature gain real? A PAIRED per-target test, not a mean of means.

`cpseq_meanscore_scorer.py` reports that adding the ProteinMPNN features to a
composition ridge moves leave-one-target-out within-target rho from 0.205 to 0.240. That
is a difference of two averages over 20 targets, and this project has already been burned
once by exactly that shape of claim: in [[cpseq-mean-vs-median-aggregator]] a 2.2 REU
"win" turned out to live entirely in one arm of five and reversed on two others.

So the comparison is made the paired way. The SAME leave-one-target-out loop is run twice,
once with and once without the extra features, and the two within-target rho values are
kept **per target**. What is reported is the distribution of the per-target difference:
how many targets improve, by how much, a Wilcoxon signed-rank test, and a bootstrap
interval over targets. A gain that is real should show up as most targets moving the same
way, not as one target carrying the mean.

The ridge is deterministic (RidgeCV over a fixed alpha grid), so there is no seed noise
here and the only sampling noise is which targets exist -- which is what the bootstrap and
the sign count address.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_meanscore_feature_gain.py \
        --dataset /zfsauton/scratch/yixiz/cpseq/meanscore/dataset_v3.json \
        --extra-features /zfsauton/scratch/yixiz/cpseq/meanscore/mpnn_feats_bb_v3.json \
        --json /zfsauton/scratch/yixiz/cpseq/meanscore/feature_gain_v3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_meanscore_baseline import AA, CHARGE, KD, label_ceiling  # noqa: E402
from cpseq_meanscore_scorer import (  # noqa: E402
    aggregate_label,
    extra_matrix,
    load_extra_features,
)

ALPHAS = np.logspace(-2, 4, 25)


def comp_delta(r) -> list[float]:
    s, o = r["sequence"], r["original_sequence"]
    d = [(s.count(a) - o.count(a)) / len(s) for a in AA]
    return d + [
        st.mean(KD.get(c, 0.0) for c in s) - st.mean(KD.get(c, 0.0) for c in o),
        sum(CHARGE.get(c, 0.0) for c in s) - sum(CHARGE.get(c, 0.0) for c in o),
        float(r.get("n_mutations") or 0),
    ]


def loto_per_target(X, y, parents, targets, min_seq=4):
    """within-target rho for each held-out target, from one LOTO pass."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    parents = np.asarray(parents)
    out = {}
    for t in targets:
        te = parents == t
        tr = ~te
        if te.sum() < min_seq:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        ytr = y[tr].copy()
        for p in set(parents[tr]):
            m = parents[tr] == p
            ytr[m] -= ytr[m].mean()
        model = RidgeCV(alphas=ALPHAS)
        model.fit((X[tr] - mu) / sd, ytr)
        pred = model.predict((X[te] - mu) / sd)
        if len(set(pred)) < 2:
            continue
        rho = spearmanr(y[te], pred).statistic
        if rho == rho:
            out[t] = float(rho)
    return out


def boot_ci(vals, n=5000, seed=0):
    rng = np.random.RandomState(seed)
    draws = [float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--extra-features", nargs="+", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--agg", default="trimmed20")
    args = ap.parse_args()

    payload = json.loads(args.dataset.read_text())
    rows = [r for r in payload["rows"] if r.get("arm") != "reference"]
    rows = [r for r in rows if r.get("original_sequence")
            and len(r["sequence"]) == len(r["original_sequence"])
            and any(a != b for a, b in zip(r["sequence"], r["original_sequence"]))]

    merged, names, dropped, have = load_extra_features(args.extra_features, rows)
    rows = [r for r, h in zip(rows, have) if h]
    parents = [r["parent_id"] for r in rows]
    targets = sorted(set(parents))
    y = [aggregate_label(r["fold_dG"], args.agg) for r in rows]
    Xe = extra_matrix(merged, names, rows, parents)
    Xc = np.array([comp_delta(r) for r in rows], dtype=float)

    half, full, nceil = label_ceiling(rows, 4)
    ceiling = full**0.5

    print(f"dataset   {args.dataset}")
    print(f"rows      {len(rows)} over {len(targets)} targets")
    print(f"features  composition_delta ({Xc.shape[1]}) + extra ({len(names)}): {names}")
    if dropped:
        print(f"          dropped: {dropped}")
    print(f"ceiling   sqrt(reliability) = {ceiling:.3f}\n")

    arms = {
        "composition_delta": Xc,
        "extra_only": Xe,
        "composition_delta + extra": np.hstack([Xc, Xe]),
    }
    per_target = {k: loto_per_target(v, y, parents, targets) for k, v in arms.items()}

    print("MEAN within-target rho (leave-one-target-out)")
    for k, v in per_target.items():
        vals = list(v.values())
        lo, hi = boot_ci(vals)
        print(f"  {k:28s} {st.mean(vals):6.3f}  95% CI [{lo:.3f},{hi:.3f}]  "
              f"{100*st.mean(vals)/ceiling:5.1f}% of ceiling")

    base, aug = "composition_delta", "composition_delta + extra"
    common = sorted(set(per_target[base]) & set(per_target[aug]))
    a = np.array([per_target[base][t] for t in common])
    b = np.array([per_target[aug][t] for t in common])
    d = b - a

    print(f"\nPAIRED PER-TARGET DIFFERENCE  ({aug}) - ({base})")
    print(f"  {'target':24s} {'base':>7s} {'aug':>7s} {'delta':>7s}")
    for t, x, z in zip(common, a, b):
        print(f"  {t:24s} {x:7.3f} {z:7.3f} {z - x:+7.3f}")

    n_up = int((d > 0).sum())
    lo, hi = boot_ci(d)
    print(f"\n  targets improved : {n_up}/{len(d)}")
    print(f"  mean delta       : {d.mean():+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
    print(f"  median delta     : {np.median(d):+.3f}")
    try:
        stat, p = wilcoxon(d)
        print(f"  Wilcoxon signed-rank: W={stat:.1f}  p={p:.4f}")
    except Exception as exc:  # pragma: no cover
        p = float("nan")
        print(f"  Wilcoxon failed: {exc}")

    print("\nVERDICT")
    if lo > 0 and n_up >= 0.7 * len(d):
        verdict = "real: CI excludes 0 and most targets move the same way"
        print(f"  REAL. {n_up}/{len(d)} targets improve, mean {d.mean():+.3f}, and the")
        print(f"  bootstrap CI excludes zero. The gain is not one target carrying a mean.")
    elif lo > 0:
        verdict = f"CI excludes 0 but only {n_up}/{len(d)} targets improve -- check for one carrier"
        print(f"  MIXED. The CI excludes zero, but only {n_up}/{len(d)} targets improve,")
        print(f"  so inspect the per-target table for a single large carrier before")
        print(f"  quoting the mean.")
    else:
        verdict = "not established: bootstrap CI includes 0"
        print(f"  NOT ESTABLISHED. Mean {d.mean():+.3f} but the CI includes zero")
        print(f"  ({n_up}/{len(d)} targets improved). Do not quote this as a gain.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "dataset": str(args.dataset),
            "extra_features": [str(p) for p in args.extra_features],
            "feature_names": names,
            "agg": args.agg,
            "ceiling": ceiling,
            "n_rows": len(rows),
            "per_target": per_target,
            "paired": {"targets": common, "base": a.tolist(), "augmented": b.tolist(),
                       "delta": d.tolist(), "n_improved": n_up,
                       "mean_delta": float(d.mean()), "ci": [lo, hi],
                       "wilcoxon_p": float(p)},
            "verdict": verdict,
        }, indent=1, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
