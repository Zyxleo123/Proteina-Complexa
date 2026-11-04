#!/usr/bin/env python3
"""Which statistic over a sequence's ~10 folds is the most reliable target score?

`cpseq_meanscore_dataset.py` has to commit to one aggregator before anything can be
trained on it, and that choice is not cosmetic: the unclipped mean over the same folds is
a readout of which sequence happened to draw a refinement blow-up. This script is the
measurement that settles it, and it is the one the dataset docstring cites.

The question is asked as reliability, not accuracy, because there is no ground truth to
be accurate against. A fold is a Boltz-2 prediction under one seed, scored by Rosetta;
the latent quantity a scorer wants is the sequence's expected interface energy, and the
only handle on it is whether independent halves of the folds agree about how sequences
rank.

How each aggregator is scored
-----------------------------
For one target, every sequence's folds are shuffled and split into two disjoint halves of
`k` folds. The aggregator is applied to each half, giving two independent estimates per
sequence, and those two vectors are rank-correlated **across sequences within the
target**. That within-target restriction is the whole point: a pooled correlation is
mostly a readout of which target a row belongs to (between-target label sd is ~44 REU
against a within-target signal sd of ~3.8), so it would rank every aggregator as
excellent. Many random splits are taken and the per-target median is kept; the headline
is the median over targets with a bootstrap interval over targets.

Two numbers come out of that, and they are not the same number
--------------------------------------------------------------
* **`r_half`** -- the raw agreement between two `k`-fold estimates. This is what the
  split gives directly.
* **`reliability`** -- `r_half` Spearman-Brown corrected to the full `2k` folds,
  `2r/(1+r)`. This is how well a `2k`-fold label agrees with an independent `2k`-fold
  label of the same sequences, i.e. how repeatable the target score is.
* **`ceiling`** -- `sqrt(reliability)`. This is the one to compare a trained model
  against. A model that predicted the latent truth *perfectly* would not score 1.0
  against our noisy label; it would score `sqrt(reliability)`, because the label carries
  error the model is right to not reproduce. Scoring a model against `reliability`
  instead understates the gap it has left to close.

  (The attenuation identity `r_observed = r_true * sqrt(rho)` is exact for Pearson
  correlation on the ranks. Applied to Spearman it is an approximation, which is why the
  Pearson-on-ranks form is reported next to it as `ceiling_pearson` -- they agree here.)

Reliability is also reported as a function of fold count, so "is 10 folds enough" is
answered rather than assumed: the same split is run at `k = 1..5`, giving the repeatability
of a 2-, 4-, 6-, 8- and 10-fold label, and the Spearman-Brown curve says what buying more
folds would return.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_meanscore_reliability.py \
        --dataset /zfsauton/scratch/yixiz/cpseq/meanscore/dataset_v3.json \
        --json    /zfsauton/scratch/yixiz/cpseq/meanscore/reliability_v3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

# Folds above this are refinement blow-ups, not weak binders: 0.22 % of folds come back at
# up to +19,587 REU and a handful pass the validity checks, so the validity gate does not
# remove them. Kept identical to `cpseq_meanscore_dataset.CLIP_AT` on purpose -- if these
# two ever disagree, the reliability measured here is not the label's.
CLIP_AT = 0.0


# --------------------------------------------------------------------------------------
# Candidate aggregators. Each maps a list of raw per-fold dG values to one target score.
# `valid` is passed alongside because one candidate conditions on it.
# --------------------------------------------------------------------------------------
def _clip(v):
    return [min(x, CLIP_AT) for x in v]


def _trimmed(v, frac):
    v = sorted(v)
    n = len(v)
    cut = int(n * frac)
    kept = v[cut : n - cut] if n - 2 * cut >= 1 else v
    return st.mean(kept)


def _winsor(v, frac):
    v = sorted(v)
    n = len(v)
    cut = int(n * frac)
    if n - 2 * cut < 1 or cut == 0:
        return st.mean(v)
    lo, hi = v[cut], v[n - cut - 1]
    return st.mean([min(max(x, lo), hi) for x in v])


def _hodges_lehmann(v):
    """Median of all pairwise averages. Robust location with near-mean efficiency."""
    pw = [(v[i] + v[j]) / 2 for i in range(len(v)) for j in range(i, len(v))]
    return st.median(pw)


def _valid_clipped_mean(v, valid):
    """Mean over folds that passed validity, clipped. Falls back to all folds.

    Conditioning the label on validity is tempting and is measured here rather than
    assumed, because validity flips across seeds for ~87 % of sequences -- so a
    valid-only label is partly a readout of the seed lottery that decided which folds
    survived, and it silently changes the effective fold count per sequence.
    """
    sel = [x for x, ok in zip(v, valid) if ok]
    return st.mean(_clip(sel if sel else v))


AGGREGATORS: dict[str, callable] = {
    "mean_raw": lambda v, k: st.mean(v),
    "mean_clip0": lambda v, k: st.mean(_clip(v)),
    "median": lambda v, k: st.median(v),
    "median_clip0": lambda v, k: st.median(_clip(v)),
    "trimmed10": lambda v, k: _trimmed(v, 0.10),
    "trimmed20": lambda v, k: _trimmed(v, 0.20),
    "winsor20": lambda v, k: _winsor(v, 0.20),
    "hodges_lehmann": lambda v, k: _hodges_lehmann(_clip(v)),
    "min": lambda v, k: min(v),
    "q25": lambda v, k: float(np.percentile(v, 25)),
    "valid_clip_mean": lambda v, k: _valid_clipped_mean(v, k),
}


def _spearman(a, b):
    if len({*a}) < 2 or len({*b}) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _pearson_on_ranks(a, b):
    """Pearson on the ranks -- the form the attenuation identity is exact for."""
    if len({*a}) < 2 or len({*b}) < 2:
        return float("nan")
    return float(pearsonr(rankdata(a), rankdata(b)).statistic)


def _pearson_values(a, b):
    """Pearson on the raw REU values. One blow-up fold wrecks this and not the ranks."""
    if len({*a}) < 2 or len({*b}) < 2:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def split_half_by_target(rows, agg_name, k, n_rep, rng):
    """Median split-half agreement per target for one aggregator at k folds per half."""
    fn = AGGREGATORS[agg_name]
    by_target = collections.defaultdict(list)
    for r in rows:
        if len(r["fold_dG"]) >= 2 * k:
            by_target[r["parent_id"]].append(r)

    out = {}
    for target, trows in by_target.items():
        if len(trows) < 5:
            continue
        reps_s, reps_p, reps_v = [], [], []
        for _ in range(n_rep):
            A, B = [], []
            for r in trows:
                idx = list(range(len(r["fold_dG"])))
                rng.shuffle(idx)
                ia, ib = idx[:k], idx[k : 2 * k]
                va = [r["fold_dG"][i] for i in ia]
                vb = [r["fold_dG"][i] for i in ib]
                ka = [r["fold_valid"][i] for i in ia]
                kb = [r["fold_valid"][i] for i in ib]
                A.append(fn(va, ka))
                B.append(fn(vb, kb))
            reps_s.append(_spearman(A, B))
            reps_p.append(_pearson_on_ranks(A, B))
            # Value-scale agreement. A squared-error or Gaussian-NLL loss trains on the
            # VALUES, so this -- not the rank version -- is the reliability that decides
            # whether the label needs clipping.
            reps_v.append(_pearson_values(A, B))
        reps_s = [x for x in reps_s if x == x]
        reps_p = [x for x in reps_p if x == x]
        reps_v = [x for x in reps_v if x == x]
        if not reps_s:
            continue
        out[target] = {
            "n_seq": len(trows),
            "r_half": st.median(reps_s),
            "r_half_pearson_ranks": st.median(reps_p) if reps_p else float("nan"),
            "r_half_values": st.median(reps_v) if reps_v else float("nan"),
        }
    return out


def _sb(r):
    """Spearman-Brown: half-length reliability r -> full-length reliability."""
    return 2 * r / (1 + r) if r > -1 else float("nan")


def _boot_ci(vals, rng, n=2000, lo=2.5, hi=97.5):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        draws.append(st.median(s))
    return (float(np.percentile(draws, lo)), float(np.percentile(draws, hi)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5, help="folds per half for the headline")
    ap.add_argument("--n-rep", type=int, default=51, help="random half-splits per target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--include-reference",
        action="store_true",
        help="keep the n_mutations==0 row; excluded by default so a model cannot score "
        "by spotting the original instead of ranking designs against each other",
    )
    args = ap.parse_args()

    payload = json.loads(args.dataset.read_text())
    rows = payload["rows"]
    n_all = len(rows)
    if not args.include_reference:
        rows = [r for r in rows if (r.get("n_mutations") or 0) > 0]
    rng = random.Random(args.seed)
    nprng = random.Random(args.seed + 1)

    print(f"dataset   {args.dataset}")
    print(f"label     {payload.get('label')}")
    print(f"rows      {len(rows)} of {n_all} "
          f"({'reference kept' if args.include_reference else 'reference excluded'})")
    print(f"headline  k={args.k} folds per half, {args.n_rep} splits per target\n")

    # ---- 1. aggregator sweep at the headline fold count -------------------------------
    sweep = {}
    for name in AGGREGATORS:
        per_t = split_half_by_target(rows, name, args.k, args.n_rep, rng)
        if not per_t:
            continue
        halves = [v["r_half"] for v in per_t.values()]
        rels = [_sb(v["r_half"]) for v in per_t.values()]
        med_rel = st.median(rels)
        ceil = med_rel**0.5 if med_rel > 0 else float("nan")
        ceil_p = (
            st.median([_sb(v["r_half_pearson_ranks"]) for v in per_t.values()]) ** 0.5
            if med_rel > 0
            else float("nan")
        )
        vals = [v["r_half_values"] for v in per_t.values() if v["r_half_values"] == v["r_half_values"]]
        rel_val = _sb(st.median(vals)) if vals else float("nan")
        sweep[name] = {
            "n_targets": len(per_t),
            "r_half_median": st.median(halves),
            "reliability_median": med_rel,
            "reliability_ci": _boot_ci(rels, nprng),
            "ceiling": ceil,
            "ceiling_pearson": ceil_p,
            "reliability_values": rel_val,
            "per_target": per_t,
        }

    order = sorted(sweep, key=lambda n: -sweep[n]["reliability_median"])
    print(f"AGGREGATOR SWEEP  (2k={2*args.k} folds, {sweep[order[0]]['n_targets']} targets)")
    print("  rank  = Spearman-Brown reliability of the RANKS (what a ranking scorer needs)")
    print("  value = same on the raw REU VALUES (what a squared-error/NLL loss trains on)")
    print(f"\n{'aggregator':18s} {'r_half':>7s} {'rank_rel':>9s} "
          f"{'95% CI':>16s} {'ceiling':>8s} {'value_rel':>10s}")
    for name in order:
        s = sweep[name]
        lo, hi = s["reliability_ci"]
        print(f"{name:18s} {s['r_half_median']:7.3f} {s['reliability_median']:9.3f} "
              f"  [{lo:5.3f},{hi:5.3f}] {s['ceiling']:8.3f} {s['reliability_values']:10.3f}")

    best = order[0]
    print(f"\nmost reliable: {best}")

    # ---- 2. reliability vs fold count, for the contenders -----------------------------
    contenders = order[: min(4, len(order))]
    curve = {}
    print(f"\nRELIABILITY vs FOLD COUNT  (Spearman-Brown to 2k folds)")
    hdr = "  ".join(f"2k={2*kk:<5d}" for kk in range(1, args.k + 1))
    print(f"{'aggregator':18s} {hdr}")
    for name in contenders:
        row = {}
        cells = []
        for kk in range(1, args.k + 1):
            per_t = split_half_by_target(rows, name, kk, args.n_rep, rng)
            if per_t:
                rel = st.median([_sb(v["r_half"]) for v in per_t.values()])
                row[2 * kk] = {"reliability": rel, "n_targets": len(per_t)}
                cells.append(f"{rel:7.3f}")
            else:
                cells.append("     --")
        curve[name] = row
        print(f"{name:18s} {'  '.join(cells)}")

    # Spearman-Brown extrapolation for the winner: what would more folds buy?
    extrap = {}
    if 2 in curve.get(best, {}) and args.k >= 1:
        r1 = curve[best][2]["reliability"]  # reliability of a 2-fold label
        # rho_n = n*rho_2/2 / (1 + (n/2 - 1)*rho_2)  -- SB with length ratio n/2
        for n_folds in (10, 20, 30, 50, 100):
            ratio = n_folds / 2
            extrap[n_folds] = ratio * r1 / (1 + (ratio - 1) * r1)
        print(f"\nSB EXTRAPOLATION for {best} (from the 2-fold point):")
        print("  " + "  ".join(f"n={n}: {v:.3f}" for n, v in extrap.items()))

    # ---- 3. label scatter in REU, for the winner --------------------------------------
    fn = AGGREGATORS[best]
    diffs = []
    for r in rows:
        if len(r["fold_dG"]) < 2 * args.k:
            continue
        idx = list(range(len(r["fold_dG"])))
        rng.shuffle(idx)
        ia, ib = idx[: args.k], idx[args.k : 2 * args.k]
        a = fn([r["fold_dG"][i] for i in ia], [r["fold_valid"][i] for i in ia])
        b = fn([r["fold_dG"][i] for i in ib], [r["fold_valid"][i] for i in ib])
        diffs.append(abs(a - b))
    # SD of a k-fold estimate from the paired differences, then scaled to 2k folds.
    sd_k = st.median(diffs) / 0.9539 / (2**0.5) if diffs else float("nan")
    sd_2k = sd_k / (2**0.5)
    sig = [
        st.stdev(v)
        for v in (
            [r["label_clipped_mean"] for r in rows if r["parent_id"] == t]
            for t in {r["parent_id"] for r in rows}
        )
        if len(v) >= 5
    ]
    print(f"\nLABEL SCATTER ({best})")
    print(f"  within-sequence SD of a {args.k}-fold estimate : {sd_k:5.2f} REU")
    print(f"  within-sequence SD of a {2*args.k}-fold label   : {sd_2k:5.2f} REU")
    print(f"  across-sequence within-target signal SD  : {st.median(sig):5.2f} REU")
    print(f"  signal-to-noise on the {2*args.k}-fold label  : "
          f"{st.median(sig)/sd_2k:5.2f}x")

    out = {
        "dataset": str(args.dataset),
        "label_source": payload.get("label"),
        "clip_at": CLIP_AT,
        "k_per_half": args.k,
        "n_rep": args.n_rep,
        "include_reference": args.include_reference,
        "n_rows": len(rows),
        "metric": "within-target split-half Spearman, Spearman-Brown corrected to 2k folds",
        "ceiling_definition": "sqrt(reliability): the score a PERFECT predictor of the "
        "latent mean would get against this noisy label. Compare trained models to this, "
        "not to 1.0 and not to reliability.",
        "recommended_aggregator": best,
        "sweep": {k: {kk: vv for kk, vv in v.items() if kk != "per_target"}
                  for k, v in sweep.items()},
        "per_target": {k: v["per_target"] for k, v in sweep.items()},
        "reliability_vs_folds": curve,
        "sb_extrapolation": extrap,
        "label_sd_k_folds_REU": sd_k,
        "label_sd_2k_folds_REU": sd_2k,
        "within_target_signal_sd_REU": st.median(sig) if sig else None,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))
    print(f"\nwrote     {args.json}")


if __name__ == "__main__":
    main()
