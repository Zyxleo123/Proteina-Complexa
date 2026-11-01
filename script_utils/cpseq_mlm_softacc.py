"""Score a cpseq_mlm prediction dump under substitution-tolerant metrics, and calibrate.

THE QUESTION THIS ANSWERS. The arm's docs argue that a hidden-position loss near 2.4 is
intrinsically high because many amino acids are interchangeable, so exact-match recovery
understates the model. That is a plausible story and it makes two testable predictions.
This script tests both, on CPU, from the dump.

PREDICTION 1 -- the errors should be CONSERVATIVE. If the model is losing exact matches to
substitution-equivalent residues (ILE where the crystal says LEU), then tolerating
conservative substitutions should lift its score a lot. Measured as

    lift(scorer) = soft_ge1(scorer) - exact(scorer)

and the number that matters is NOT the model's soft score, it is the lift DIFFERENTIAL
against the priors:

    lift(model) - lift(chem prior)

because a chemistry-tolerant metric flatters a chemistry-keyed lookup table too -- that is
close to what the lookup optimises. Reading the model's soft accuracy alone would restate
the metric's own tolerance as a property of the model. The pre-registered reading:

  * differential clearly > 0  -- the misses really are conservative; exact match
    understates the model and the ambiguity argument has support.
  * differential ~ 0          -- the model's errors are no more chemically sensible than a
    positional lookup's. Interchangeability then does not rescue the loss number, because
    the baseline enjoys it equally.
  * differential < 0          -- the model is worse than a lookup at being wrong in a
    defensible way, which is evidence against the story, not for it.

PREDICTION 2 -- the model should be CALIBRATED. If 2.4 nats is irreducible aleatoric
entropy, a converged model should know it: mean predictive entropy should sit near the NLL
and mean max-probability near the accuracy. A model that is CONFIDENTLY wrong (max-prob far
above accuracy) is not reporting an entropy floor, it is miscalibrated or underfit -- and
the ceiling claim does not follow. This is the sharper of the two tests, because it needs
no baseline to interpret.

WHAT IS NOT CLAIMED HERE. None of this measures whether a proposed sequence closes or
scores better dG. A substitution can be an exact-match error and a better peptide. These
metrics bound how much of the recovery gap is bookkeeping; they cannot convert recovery
into downstream value, and this project already has one proxy (retention) that correlated
with Rosetta dG at r2 = 0.004. The proposer test is the thing that settles that.

SIGNIFICANCE. Positions are CLUSTERED: one 16-mer contributes up to 16 correlated
positions, and peptides within a sequence cluster share a receptor family. Confidence
intervals here resample CLUSTERS with replacement, never positions -- an unclustered
binomial interval over pooled positions is optimistic by a large factor at this panel
size. Model-minus-baseline differences are bootstrapped PAIRED on the same resampled
clusters, because both scorers see identical positions.

Usage:
    python script_utils/cpseq_mlm_softacc.py \
        --dump .../preds_pocket_step60000.jsonl \
        --corpus-dir /zfsauton/scratch/yixiz/cpseq_mlm/corpus \
        --priors-cache .../priors_train.json \
        --out .../softacc_pocket_step60000.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "script_utils") not in sys.path:
    sys.path.insert(0, str(_REPO / "script_utils"))

from cpseq_mlm_data import (  # noqa: E402
    AA20,
    REGIMES,
    amino_acid_marginal,
    load_corpus,
    positional_chemistry_marginal,
    positional_marginal,
)

SCORERS = ("model", "chem_prior", "pos_prior", "marginal")
METRICS = ("exact", "soft_ge0", "soft_ge1", "graded")


def blosum62() -> dict[tuple[str, str], float]:
    """BLOSUM62 as a plain dict over the 20 canonical residues.

    Loaded through biopython's `substitution_matrices`, which is already how this repo
    gets BLOSUM62 (`src/proteinfoundation/cpseq/contexts.py` uses it for the receptor
    family split), rather than pasting a second copy of the matrix into the tree.
    """
    from Bio.Align import substitution_matrices

    m = substitution_matrices.load("BLOSUM62")
    return {(a, b): float(m[a, b]) for a in AA20 for b in AA20}


def graded_table(b62: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Partial credit in [0, 1], normalised PER TRUE RESIDUE.

    Normalising per true residue (diagonal -> 1, worst substitution for that residue -> 0)
    rather than over the whole matrix is what makes the average interpretable: raw BLOSUM
    diagonals run from 4 (most residues) to 11 (TRP), so a whole-matrix normalisation
    would score a correct TRP as worth three correct LEUs and the mean would then track
    which residues happened to be in the panel.
    """
    out: dict[tuple[str, str], float] = {}
    for true in AA20:
        col = [b62[(p, true)] for p in AA20]
        lo, hi = min(col), b62[(true, true)]
        span = hi - lo
        for pred in AA20:
            out[(pred, true)] = 1.0 if span <= 0 else (b62[(pred, true)] - lo) / span
    return out


def build_priors(corpus_dir: Path, train_glob: str, cache: Path | None) -> dict:
    """The three always-argmax priors, keyed for lookup, computed from TRAIN rows.

    Cached to JSON because the reduction is meant to be re-runnable without re-reading a
    2.4M-row corpus: every metric in this project has to be redrawable from disk, and a
    CPU job that has to load 386 MB before it can print a table is not that.
    """
    if cache and cache.exists():
        d = json.loads(cache.read_text())
        print(f"[priors] loaded cache {cache}")
        return {
            "marginal": d["marginal"],
            # JSON keys are strings; restore the tuple keys the lookups use.
            "positional": {tuple(map(int, k.split(","))): v for k, v in d["positional"].items()},
            "chem": {tuple(map(int, k.split(","))): v for k, v in d["chem"].items()},
            "n_train_rows": d["n_train_rows"],
        }
    files = sorted(corpus_dir.glob(train_glob))
    if not files:
        raise SystemExit(f"FATAL: no {train_glob} under {corpus_dir}")
    rows = load_corpus(files)
    print(f"[priors] {len(rows)} train rows from {len(files)} shard(s)")
    marg = amino_acid_marginal(rows)
    posm = positional_marginal(rows)
    chem = positional_chemistry_marginal(rows)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "marginal": marg,
            "positional": {",".join(map(str, k)): v for k, v in posm.items()},
            "chem": {",".join(map(str, k)): v for k, v in chem.items()},
            "n_train_rows": len(rows),
        }))
        print(f"[priors] cached -> {cache}")
    return {"marginal": marg, "positional": posm, "chem": chem, "n_train_rows": len(rows)}


def prior_predictions(rec: dict, priors: dict) -> dict[str, str]:
    """Each baseline's argmax residue at one hidden position.

    The chem prior backs off (pos, L, cyc) -> (pos, L) -> global exactly as
    `cpseq_mlm_data.baseline_accuracy` does. Without the identical backoff this scorer
    would differ from the one the manifest reports and the comparison would be against a
    baseline that was never the bar.
    """
    marg = priors["marginal"]
    best_global = max(marg, key=marg.get) if marg else "L"
    pos, L, cyc = rec["pos"], rec["L"], rec["cyc_type"]

    d_pos = priors["positional"].get((pos, L))
    pred_pos = max(d_pos, key=d_pos.get) if d_pos else best_global

    d_chem = priors["chem"].get((pos, L, cyc)) or d_pos
    pred_chem = max(d_chem, key=d_chem.get) if d_chem else best_global

    return {"marginal": best_global, "pos_prior": pred_pos, "chem_prior": pred_chem}


def score_position(pred: str, true: str, b62, graded) -> dict[str, float]:
    s = b62[(pred, true)]
    return {
        "exact": float(pred == true),
        "soft_ge0": float(s >= 0),
        "soft_ge1": float(s >= 1),
        "graded": graded[(pred, true)],
    }


def entropy_nats(logprobs: list[float]) -> float:
    return -sum(math.exp(lp) * lp for lp in logprobs)


def aggregate(units: list[dict], keys: list[str]) -> dict[str, float]:
    """Mean of each key over the positions in the given cluster units."""
    tot: dict[str, float] = defaultdict(float)
    n = 0
    for u in units:
        for p in u["positions"]:
            n += 1
            for k in keys:
                tot[k] += p[k]
    return {k: tot[k] / n for k in keys} if n else {k: float("nan") for k in keys}


def clustered_ci(
    units: list[dict], keys: list[str], *, n_boot: int, seed: int, alpha: float = 0.05
) -> dict[str, tuple[float, float]]:
    """Percentile CI from resampling CLUSTER UNITS with replacement.

    One draw resamples units, then pools every position inside the drawn units, so a
    cluster contributing 40 correlated positions moves as one object. This is the whole
    point: on this panel the unclustered interval is narrow enough to call a 0.5-point
    difference significant, and it is not.
    """
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {k: [] for k in keys}
    n = len(units)
    for _ in range(n_boot):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        got = aggregate(sample, keys)
        for k in keys:
            draws[k].append(got[k])
    out = {}
    lo_i = int(alpha / 2 * n_boot)
    hi_i = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    for k in keys:
        vals = sorted(draws[k])
        out[k] = (vals[lo_i], vals[hi_i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--corpus-dir", default="/zfsauton/scratch/yixiz/cpseq_mlm/corpus")
    ap.add_argument("--train-glob", default="train_*.jsonl")
    ap.add_argument("--priors-cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b62 = blosum62()
    graded = graded_table(b62)
    priors = build_priors(
        Path(args.corpus_dir), args.train_glob,
        Path(args.priors_cache) if args.priors_cache else None,
    )

    # --- score every position under every scorer, keeping the cluster label
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    by_regime_cluster: dict[str, dict[str, list[dict]]] = {r: defaultdict(list) for r in REGIMES}
    n_rows = 0
    for line in Path(args.dump).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n_rows += 1
        true = rec["true"]
        preds = prior_predictions(rec, priors)
        preds["model"] = rec["pred"]
        flat: dict[str, float] = {}
        for scorer in SCORERS:
            for m, v in score_position(preds[scorer], true, b62, graded).items():
                flat[f"{scorer}.{m}"] = v
        # Calibration, from the dumped distribution. `nll` is the per-position loss the
        # ceiling claim is about; `maxprob` minus `model.exact` is the overconfidence gap.
        lps = rec["logprobs"]
        flat["nll"] = -lps[AA20.index(true)]
        flat["entropy"] = entropy_nats(lps)
        flat["maxprob"] = math.exp(max(lps))
        # Paired differentials, carried per position so the bootstrap pairs them for free.
        for scorer in ("chem_prior", "pos_prior", "marginal"):
            flat[f"d_exact.{scorer}"] = flat["model.exact"] - flat[f"{scorer}.exact"]
            flat[f"d_soft_ge1.{scorer}"] = flat["model.soft_ge1"] - flat[f"{scorer}.soft_ge1"]
            # The headline of this script: does tolerating substitutions help the MODEL
            # more than it helps the lookup?
            flat[f"d_lift.{scorer}"] = (
                (flat["model.soft_ge1"] - flat["model.exact"])
                - (flat[f"{scorer}.soft_ge1"] - flat[f"{scorer}.exact"])
            )
        cl = rec.get("cluster_id") or rec["example_id"]
        by_cluster[cl].append(flat)
        by_regime_cluster[rec["regime"]][cl].append(flat)

    keys = (
        [f"{s}.{m}" for s in SCORERS for m in METRICS]
        + ["nll", "entropy", "maxprob"]
        + [f"d_{t}.{s}" for t in ("exact", "soft_ge1", "lift")
           for s in ("chem_prior", "pos_prior", "marginal")]
    )
    units = [{"cluster": c, "positions": ps} for c, ps in by_cluster.items()]
    n_pos = sum(len(u["positions"]) for u in units)
    print(f"dump rows={n_rows} scored positions={n_pos} clusters={len(units)}")
    # The ratio below is the reason the clustered interval differs from the naive one; it
    # is printed so a reader can see how much correlation the CI has to absorb.
    print(f"mean positions per cluster = {n_pos / max(len(units), 1):.1f}")

    point = aggregate(units, keys)
    ci = clustered_ci(units, keys, n_boot=args.n_boot, seed=args.seed)

    per_regime = {}
    for regime, d in by_regime_cluster.items():
        if not d:
            continue
        u = [{"cluster": c, "positions": ps} for c, ps in d.items()]
        per_regime[regime] = {
            "n_positions": sum(len(x["positions"]) for x in u),
            "n_clusters": len(u),
            "point": aggregate(u, keys),
            "ci": {k: list(v) for k, v in
                   clustered_ci(u, keys, n_boot=args.n_boot, seed=args.seed).items()},
        }

    result = {
        "dump": str(Path(args.dump).resolve()),
        "n_positions": n_pos,
        "n_clusters": len(units),
        "n_train_rows_for_priors": priors["n_train_rows"],
        "n_boot": args.n_boot,
        "point": point,
        "ci95_clustered": {k: list(v) for k, v in ci.items()},
        "per_regime": per_regime,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    # --- report
    def fmt(k: str) -> str:
        lo, hi = ci[k]
        return f"{point[k]:+.4f}  [{lo:+.4f}, {hi:+.4f}]"

    print("\n=== exact vs substitution-tolerant, pooled (95% CI over clusters) ===")
    print(f"{'scorer':<12}" + "".join(f"{m:>12}" for m in METRICS) + f"{'lift':>12}")
    for s in SCORERS:
        row = "".join(f"{point[f'{s}.{m}']:>12.4f}" for m in METRICS)
        lift = point[f"{s}.soft_ge1"] - point[f"{s}.exact"]
        print(f"{s:<12}{row}{lift:>12.4f}")

    print("\n=== the headline: does tolerance help the model more than the lookup? ===")
    for s in ("chem_prior", "pos_prior", "marginal"):
        print(f"  model - {s:<11} exact      {fmt(f'd_exact.{s}')}")
        print(f"  model - {s:<11} soft_ge1   {fmt(f'd_soft_ge1.{s}')}")
        print(f"  model - {s:<11} LIFT DIFF  {fmt(f'd_lift.{s}')}")
    print("\n  Reading: a LIFT DIFF whose CI contains 0 means the model's errors are no")
    print("  more conservative than that baseline's, so substitution tolerance does not")
    print("  explain its exact-match gap.")

    print("\n=== calibration: is 2.4 nats an entropy floor or overconfidence? ===")
    print(f"  mean NLL (the loss)        {fmt('nll')}")
    print(f"  mean predictive entropy    {fmt('entropy')}")
    print(f"  mean max-probability       {fmt('maxprob')}")
    print(f"  exact accuracy             {fmt('model.exact')}")
    gap = point["maxprob"] - point["model.exact"]
    print(f"  overconfidence (maxprob - acc) = {gap:+.4f}")
    print("  Reading: entropy ~ NLL and maxprob ~ accuracy is what an irreducible-entropy")
    print("  floor looks like. maxprob >> accuracy is overconfidence, under which the")
    print("  loss is not evidence of a ceiling.")

    print("\n=== per regime (exact / soft_ge1 / lift diff vs chem prior) ===")
    for regime in REGIMES:
        if regime not in per_regime:
            continue
        p = per_regime[regime]["point"]
        lo, hi = per_regime[regime]["ci"]["d_lift.chem_prior"]
        print(f"  {regime:<8} n={per_regime[regime]['n_positions']:<5} "
              f"exact={p['model.exact']:.4f} soft={p['model.soft_ge1']:.4f} "
              f"liftdiff={p['d_lift.chem_prior']:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
