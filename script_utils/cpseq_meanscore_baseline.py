#!/usr/bin/env python3
"""What can a mean-score predictor actually learn? Feature baselines, target-held-out.

Reads the table from `cpseq_meanscore_dataset.py` and asks one question: can a model
rank a target's candidate peptides by mean dG better than chance, on targets it never
saw? That is the only thing a scorer used to guide sequence optimisation needs to do.

Four rules, each of which makes the headline number smaller and honest:

1. **The metric is within-target Spearman, averaged over held-out targets.** A global
   correlation across the pooled table is mostly a readout of which target a sequence
   belongs to: between-target sd of the label is 44 REU against a within-target signal
   sd of 3.8. A model that predicts only the target mean scores well globally and is
   worthless for optimising a sequence against a fixed target.

2. **Whole targets are held out, and every fold of a sequence stays on one side.** The
   table is already one row per sequence, so folds cannot leak; the split is by
   `parent_id` so that family-level memorisation cannot pass for sequence understanding.

3. **Every score is reported against the label's own ceiling.** The label is a 10-seed
   estimate with within-target split-half rank reliability ~0.79, so a perfect predictor
   of the true latent mean scores ~0.79 on this metric, not 1.0. A raw rho is
   uninterpretable without it, so `ceiling` is computed here by the same per-target
   averaging, from the same folds, and printed in the same table.

4. **The unchanged reference is excluded by default.** It is the only row in a target
   with `n_mutations == 0`, so including it lets a model score by "is this the original"
   and inflate the within-target rho without ranking any design against another.
   `--include-reference` reports it the other way for comparison.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_meanscore_baseline.py \
        --dataset /data/user_data/yixiz/cpseq/meanscore/dataset_v3.json \
        --json /data/user_data/yixiz/cpseq/meanscore/baseline_v3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

AA = "ACDEFGHIKLMNPQRSTVWY"
# Kyte-Doolittle hydropathy, and formal charge at neutral pH. Two scalars that carry most
# of what a composition vector says about an interface, kept separate so a linear model
# can use them without 20 coefficients.
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
SEEDS_PER_HALF = 2  # stride for the interleaved split-half used by the ceiling


def seq_features(row: dict) -> dict[str, float]:
    """Sequence-only features. Nothing here needs a structure or a fold."""
    seq = row["sequence"]
    orig = row.get("original_sequence") or seq
    n = len(seq)
    f = {
        "length": float(n),
        "n_mutations": float(row.get("n_mutations") or 0),
        "mutation_fraction": float(row.get("mutation_fraction") or 0.0),
        "identity_to_original": float(row.get("identity_to_original") or 1.0),
        "hydropathy_mean": st.mean(KD.get(c, 0.0) for c in seq),
        "charge_net": sum(CHARGE.get(c, 0.0) for c in seq),
        "charge_abs": sum(abs(CHARGE.get(c, 0.0)) for c in seq),
    }
    for aa in AA:
        f[f"frac_{aa}"] = seq.count(aa) / n
    # The edit itself, as a composition delta against the parent peptide. A within-target
    # model can only move along these: everything else is shared by the target's rows.
    if len(orig) == n:
        for aa in AA:
            f[f"d_{aa}"] = (seq.count(aa) - orig.count(aa)) / n
        f["d_hydropathy"] = f["hydropathy_mean"] - st.mean(KD.get(c, 0.0) for c in orig)
        f["d_charge"] = f["charge_net"] - sum(CHARGE.get(c, 0.0) for c in orig)
    else:
        for aa in AA:
            f[f"d_{aa}"] = 0.0
        f["d_hydropathy"] = 0.0
        f["d_charge"] = 0.0
    return f


def fold_features(row: dict) -> dict[str, float]:
    """Post-fold readouts. NOT usable to guide generation -- they require the fold the
    scorer is meant to replace. Reported only as a reference point."""
    f = {}
    for key in ("mean_iptm", "mean_complex_plddt", "mean_complex_iplddt",
                "mean_refined_dSASA_int", "mean_refined_sc_value",
                "mean_refined_packstat", "mean_refined_delta_unsatHbonds",
                "mean_refined_hbonds_int", "mean_local_deformation_energy_proxy",
                "valid_rate"):
        v = row.get(key)
        f[key] = float(v) if isinstance(v, (int, float)) else 0.0
    return f


def within_target_rho(parents, y_true, y_pred, min_seq: int = 4):
    """Spearman inside each target, averaged over targets. Rule 1."""
    by = collections.defaultdict(list)
    for p, t, q in zip(parents, y_true, y_pred):
        by[p].append((t, q))
    rhos, used = [], 0
    for p, vals in by.items():
        if len(vals) < min_seq:
            continue
        t = [a for a, _ in vals]
        q = [b for _, b in vals]
        if len(set(q)) < 2 or len(set(t)) < 2:
            continue
        rho = spearmanr(t, q).statistic
        if rho == rho:  # not NaN
            rhos.append(rho)
            used += 1
    return (st.mean(rhos) if rhos else float("nan")), used


def label_ceiling(rows, min_seq: int = 4):
    """Split the seeds in half, label each half, and score one half against the other
    with the SAME per-target averaging the models are scored by. Rule 3."""
    parents, a_half, b_half = [], [], []
    for r in rows:
        seeds = r["seeds"]
        vals = r["fold_dG_clipped"]
        order = sorted(range(len(seeds)), key=lambda i: (seeds[i] is None, seeds[i]))
        a = [vals[i] for i in order[0::SEEDS_PER_HALF]]
        b = [vals[i] for i in order[1::SEEDS_PER_HALF]]
        if len(a) < 3 or len(b) < 3:
            continue
        parents.append(r["parent_id"])
        a_half.append(st.mean(a))
        b_half.append(st.mean(b))
    half, used = within_target_rho(parents, a_half, b_half, min_seq)
    # Spearman-Brown from a 5-seed half to the 10-seed label.
    full = 2 * half / (1 + half) if half == half else float("nan")
    return half, full, used


def run_cv(X, y, parents, model_fn, n_splits: int = 5, min_seq: int = 4):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(parents)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        model = model_fn()
        model.fit((X[tr] - mu) / sd, y[tr])
        pred[te] = model.predict((X[te] - mu) / sd)
    rho, used = within_target_rho(groups, y, pred, min_seq)
    glob = spearmanr(y, pred).statistic
    return rho, glob, used, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--include-reference", action="store_true")
    ap.add_argument("--min-seq", type=int, default=4, help="min sequences for a target to be scored")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    payload = json.loads(args.dataset.read_text())
    rows = payload["rows"]
    if not args.include_reference:
        rows = [r for r in rows if r.get("arm") != "reference"]

    y = [r["label"] for r in rows]
    parents = [r["parent_id"] for r in rows]
    seqf = [seq_features(r) for r in rows]
    foldf = [fold_features(r) for r in rows]
    seq_names = sorted(seqf[0])
    fold_names = sorted(foldf[0])

    print(f"dataset   {args.dataset}")
    print(f"label     {payload['label']}")
    print(f"rows      {len(rows)} sequences, {len(set(parents))} targets"
          f"{'' if args.include_reference else ', reference row excluded'}")
    print(f"label sd  {st.stdev(y):.2f} REU (pooled, dominated by between-target spread)")

    half, full, nceil = label_ceiling(rows, args.min_seq)
    print(f"\nceiling   within-target rho of a 5-seed label against the other 5 seeds: {half:.3f}"
          f"  ({nceil} targets)")
    print(f"          Spearman-Brown to the full 10-seed label:                      {full:.3f}")
    print("          A perfect predictor of the true latent mean scores about this, not 1.0.\n")

    blocks = {
        "sequence-only": (seq_names, seqf),
        "post-fold readouts (NOT usable for guidance)": (fold_names, foldf),
        "sequence + post-fold": (seq_names + fold_names,
                                 [dict(a, **b) for a, b in zip(seqf, foldf)]),
    }
    models = {
        "ridge": lambda: RidgeCV(alphas=np.logspace(-2, 4, 25)),
        "gbm": lambda: GradientBoostingRegressor(random_state=0, n_estimators=300,
                                                 max_depth=3, learning_rate=0.05,
                                                 subsample=0.8),
    }

    results = {}
    print(f"{'features':<44} {'model':<6} {'within-tgt rho':>14} {'frac of ceiling':>16} {'global rho':>11}")
    for bname, (names, feats) in blocks.items():
        X = [[f[n] for n in names] for f in feats]
        for mname, fn in models.items():
            rho, glob, used, _ = run_cv(X, y, parents, fn, args.folds, args.min_seq)
            frac = rho / full if full == full and full else float("nan")
            print(f"{bname:<44} {mname:<6} {rho:>14.3f} {frac:>16.2f} {glob:>11.3f}")
            results[f"{bname}|{mname}"] = {
                "within_target_rho": rho, "frac_of_ceiling": frac,
                "global_rho": glob, "targets_scored": used,
            }

    out = {
        "dataset": str(args.dataset),
        "label": payload["label"],
        "n_rows": len(rows),
        "n_targets": len(set(parents)),
        "include_reference": args.include_reference,
        "ceiling_half": half,
        "ceiling_full": full,
        "ceiling_targets": nceil,
        "results": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwrote     {args.json}")


if __name__ == "__main__":
    main()
