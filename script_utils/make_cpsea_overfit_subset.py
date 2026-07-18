#!/usr/bin/env python3
"""Writes a tiny CPSea metadata parquet for the overfitting sanity test.

The question the overfit test answers: can the flow fit CPSea *at all*? If `bb_ca` and
`local_latents` cannot be driven near zero on a handful of complexes with augmentation off
and deterministic AE *mean* targets, the plateau is an implementation / target-construction /
normalization / optimization bug and no amount of loss rebalancing or architecture work will
help. If they fit easily but full training stays flat, the problem is distribution or
optimization at scale.

Sampling is stratified over cyclization_type and peptide_length and takes one example per
cluster_id, so the subset is not N near-duplicates of the same peptide. Prefers complexes
with peptide_length + receptor_length < 256 so CroppingTransform2 never spatial-crops the
receptor (belt-and-suspenders with crop_size: 4096 in the overfit config).

Usage
-----
    source env.sh
    # default: the FULL root, which is what `cpsea_peptide_train_finetune.sh full` trains on
    # (that launcher re-exports CPSEA_DATA_PATH=$CPSEA_FULL_DATA_PATH).
    python script_utils/make_cpsea_overfit_subset.py --n 64

Writes <root>/preprocessed/metadata/cpsea_train_overfit<N>.parquet, which
configs/example/training_cpsea_peptide_overfit.yaml resolves via ${oc.env:CPSEA_DATA_PATH}.

The full train parquet is 2.4M rows and does not fit in memory with its string columns, so
selection reads only the stratification columns and the full rows are re-read filtered.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SELECT_COLS = ["example_id", "cluster_id", "cyclization_type", "peptide_length", "receptor_length"]

# Prefer complexes that never hit CroppingTransform2's crop path (crop fires when n_res >= crop_size).
MAX_COMPLEX_SIZE = 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="number of complexes in the subset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--data-path",
        default=os.environ.get(
            "CPSEA_FULL_DATA_PATH", "/zfsauton/scratch/yixiz/CPSea/CPSea_full/CPSea"
        ),
        help="dataset root containing preprocessed/metadata/cpsea_train.parquet",
    )
    args = ap.parse_args()

    meta_dir = Path(args.data_path) / "preprocessed" / "metadata"
    src = meta_dir / "cpsea_train.parquet"
    df = pd.read_parquet(src, columns=SELECT_COLS)

    # One row per cluster so the subset is not N copies of the same peptide.
    df = df.groupby("cluster_id", group_keys=False).head(1)

    n_clusters = len(df)
    df["_complex_size"] = df.peptide_length + df.receptor_length
    small = df[df._complex_size < MAX_COMPLEX_SIZE]
    large = df[df._complex_size >= MAX_COMPLEX_SIZE]
    print(
        f"cluster pool: {n_clusters}  "
        f"(<{MAX_COMPLEX_SIZE} residues: {len(small)}, "
        f">={MAX_COMPLEX_SIZE}: {len(large)}; preferring small)"
    )

    # Prefer small complexes so receptor spatial crop never fires. Fall back to the full
    # cluster pool only if there are fewer than N small examples after stratification.
    prefer = small if len(small) >= args.n else df
    if len(small) < args.n:
        print(
            f"WARNING: only {len(small)} small complexes available; "
            f"falling back to full pool (may include >= {MAX_COMPLEX_SIZE})"
        )

    # Stratify over cyclization type, keeping the natural type proportions of the preferred pool.
    frac = prefer.cyclization_type.value_counts(normalize=True)
    parts = []
    for ctype, p in frac.items():
        k = max(1, round(p * args.n))
        pool = prefer[prefer.cyclization_type == ctype]
        # Spread over peptide_length rather than taking the head of one length bucket.
        pool = pool.sample(frac=1.0, random_state=args.seed).sort_values("peptide_length")
        step = max(1, len(pool) // k)
        parts.append(pool.iloc[::step].head(k))

    sel = pd.concat(parts).sample(frac=1.0, random_state=args.seed).head(args.n)

    # If stratification undershot (rounding), top up from remaining preferred pool.
    if len(sel) < args.n:
        leftover = prefer[~prefer.example_id.isin(sel.example_id)]
        need = args.n - len(sel)
        extra = leftover.sample(n=min(need, len(leftover)), random_state=args.seed)
        sel = pd.concat([sel, extra])
        print(f"topped up {len(extra)} examples to reach n={args.n}")

    # Re-read the chosen rows with every column, so the subset keeps the full schema the
    # dataloader expects (path, split, source_path, ...).
    keep = set(sel.example_id)
    full = pq.read_table(src, filters=[("example_id", "in", keep)]).to_pandas()
    sub = full.drop_duplicates("example_id").reset_index(drop=True)

    out = meta_dir / f"cpsea_train_overfit{args.n}.parquet"
    sub.to_parquet(out, index=False)

    print(f"wrote {out}  ({len(sub)} complexes, {sub.cluster_id.nunique()} clusters)")
    print("\ncyclization_type:")
    print(sub.cyclization_type.value_counts().to_string())
    print("\npeptide_length: min %d  median %d  max %d" % (
        sub.peptide_length.min(), sub.peptide_length.median(), sub.peptide_length.max()))
    tot = sub.peptide_length + sub.receptor_length
    n_large = int((tot >= MAX_COMPLEX_SIZE).sum())
    print(
        "complex size:   min %d  median %d  max %d  "
        "(>=%d filtered/kept as large: %d)" % (
            tot.min(), tot.median(), tot.max(), MAX_COMPLEX_SIZE, n_large)
    )
    print(
        f"filtered out of preferred pool: {len(large)} large clusters "
        f"(not used unless small pool < n)"
    )


if __name__ == "__main__":
    main()
