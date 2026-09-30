"""Build the FIXED evaluation manifest the five one-pass arms are compared on.

One file, built once, used by every arm. The alternative -- letting each arm walk the
front of the eval dataloader -- is reproducible only as long as nobody changes the parquet,
the batch size or the worker count, and it gives no control at all over which peptide
lengths and which receptor families end up in the comparison. A written manifest makes the
set an artefact that can be quoted, re-used at a later milestone, and diffed.

WHAT IT SELECTS. 64 CPSea complexes from the receptor families held out of training
(``cp2lp_eval.parquet``), one row per family, spread as evenly as the pool allows over
peptide length. Length matters for its own reason: the terminal gap a linear peptide can
reach scales with chain length, so a manifest that happened to be short-peptide-heavy would
make the terminal-separation column look worse than the generator is.

COVERAGE IS REPORTED, NOT ASSUMED. The summary records the per-length counts and the number
of distinct families, so a thin slice is visible in the manifest itself rather than
discovered while reading the comparison table. On the current eval split every row is
``head_tail``, so there is no chemistry axis to stratify over -- that is a property of this
metadata and is written into the summary rather than left implicit.

Deterministic: the pick is a sort by a seeded hash of the example id, not a shuffle, so the
same manifest comes out on every machine and stays stable when rows are added.

Run (CPU):
    .venv/bin/python script_utils/cp2lp_eval_manifest.py \\
        --eval-parquet $LPDATA/cp2lp_eval.parquet \\
        --out-parquet  $LPDATA/cp2lp_eval64.parquet \\
        --n-parents 64 --seeds 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-parquet", required=True, help="Held-out-family metadata to draw from.")
    p.add_argument("--out-parquet", required=True, help="Where to write the manifest metadata.")
    p.add_argument("--out-json", default=None, help="Summary JSON; defaults beside --out-parquet.")
    p.add_argument("--n-parents", type=int, default=64)
    p.add_argument(
        "--seeds",
        default="0,1,2,3,4,5,6,7",
        help="Noise seeds each parent is generated under. Recorded here so every arm uses "
        "the same ones; the generation stage reads them from the summary.",
    )
    p.add_argument("--source", default="cpsea", help="dataset_source to draw parents from.")
    p.add_argument("--family-column", default="cluster_id")
    p.add_argument("--seed", type=int, default=0, help="Salt for the deterministic ordering.")
    return p.parse_args()


def _rank(example_id: str, seed: int) -> int:
    """Deterministic order key. A hash, not a shuffle: stable across processes and reruns."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{example_id}".encode()).digest()[:8], "big")


def select_parents(
    df: pd.DataFrame,
    n_parents: int,
    family_column: str = "cluster_id",
    seed: int = 0,
) -> pd.DataFrame:
    """One row per family, spread over peptide length, deterministic.

    Round-robins over the length buckets rather than taking the top ``n`` of a single
    ordering: a pool whose length distribution is peaked at 13 residues would otherwise give
    a manifest peaked there too, and the by-length columns of the comparison would rest on
    one or two examples at the ends.
    """
    df = df.copy()
    df["_rank"] = [_rank(str(e), seed) for e in df["example_id"]]
    # One row per family first, so no receptor can appear twice and inflate its weight.
    df = df.sort_values("_rank").drop_duplicates(subset=[family_column], keep="first")

    buckets: dict[int, list] = defaultdict(list)
    for _, row in df.iterrows():
        buckets[int(row["peptide_length"])].append(row)
    lengths = sorted(buckets)
    picked = []
    i = 0
    while len(picked) < n_parents and any(buckets[L] for L in lengths):
        L = lengths[i % len(lengths)]
        if buckets[L]:
            picked.append(buckets[L].pop(0))
        i += 1
    return pd.DataFrame(picked).drop(columns=["_rank"], errors="ignore")


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]

    df = pd.read_parquet(args.eval_parquet)
    if "dataset_source" in df.columns and args.source:
        df = df[df["dataset_source"] == args.source]
    if df.empty:
        logger.error(f"No rows with dataset_source={args.source!r} in {args.eval_parquet}")
        return 1

    fam = args.family_column if args.family_column in df.columns else "example_id"
    if fam != args.family_column:
        logger.warning(f"No {args.family_column!r} column; falling back to one parent per example_id.")

    n_available_families = int(df[fam].nunique())
    # "Use all available parents if fewer than 64 are eligible" -- and say so, because a
    # manifest quietly smaller than asked for changes every per-parent statistic.
    want = min(args.n_parents, n_available_families)
    if want < args.n_parents:
        logger.warning(
            f"Only {n_available_families} eligible families; the manifest will hold {want} "
            f"parents rather than the requested {args.n_parents}."
        )
    picked = select_parents(df, want, family_column=fam, seed=args.seed)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    picked.to_parquet(out_parquet, index=False)

    by_length = dict(sorted(Counter(int(x) for x in picked["peptide_length"]).items()))
    chem = (
        dict(Counter(str(x) for x in picked["cyclization_type"]))
        if "cyclization_type" in picked.columns
        else {}
    )
    summary = {
        "manifest_parquet": str(out_parquet),
        "n_parents": int(len(picked)),
        "n_parents_requested": int(args.n_parents),
        "n_eligible_families": n_available_families,
        "family_column": fam,
        "n_families_in_manifest": int(picked[fam].nunique()),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "generations_per_arm": int(len(picked)) * len(seeds),
        "by_peptide_length": by_length,
        "by_cyclization_type": chem,
        "source_parquet": str(args.eval_parquet),
        "selection_seed": int(args.seed),
        "example_ids": [str(x) for x in picked["example_id"]],
    }
    out_json = Path(args.out_json) if args.out_json else out_parquet.with_suffix(".summary.json")
    out_json.write_text(json.dumps(summary, indent=2))

    logger.info(
        f"CP2LP eval manifest: {len(picked)} parents / {summary['n_families_in_manifest']} families, "
        f"{len(seeds)} seeds -> {summary['generations_per_arm']} generations per arm"
    )
    logger.info(f"  lengths: {by_length}")
    if chem:
        logger.info(f"  chemistry: {chem} (one type means there is no chemistry axis to stratify)")
    logger.info(f"  wrote {out_parquet} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
