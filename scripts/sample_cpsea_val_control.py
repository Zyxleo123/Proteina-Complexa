"""Sample a small, deterministic CPSea-val control set for the de novo generation control.

Why this exists
---------------
`val_generation` reports ~0.85-0.96 ring closure on the CPSea val split, while de novo
generation on the LNR-staged targets reports ~0.45-0.52. The two differ on TWO axes at once
(target population, and native-vs-forced cyclization type), so neither number explains the
other. This writes the control that holds the pipeline fixed and changes only the targets:
the same sampler, the same script, the same closure metric -- run on CPSea val with each
example's NATIVE type.

CPSea val is 134k rows, so it must be subsampled. Sampling is seeded and the chosen
example_ids are written alongside the parquet, so the control is reproducible and can be
diffed if it is ever regenerated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_VAL = ("/zfsauton/scratch/yixiz/CPSea/CPSea_full/CPSea/preprocessed/metadata/"
               "cpsea_val.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-metadata", default=DEFAULT_VAL,
                    help="The val parquet the model was actually validated against.")
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--n", type=int, default=200, help="Examples to sample.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-peptide-length", type=int, default=16,
                    help="Match the training crop's binder_max_length; longer binders are "
                         "cropped by the loader and would not be a clean control.")
    ap.add_argument("--min-peptide-length", type=int, default=5)
    args = ap.parse_args()

    m = pd.read_parquet(args.val_metadata)
    n_all = len(m)
    m = m[(m["peptide_length"] >= args.min_peptide_length)
          & (m["peptide_length"] <= args.max_peptide_length)]
    print(f"{n_all} val rows -> {len(m)} within peptide length "
          f"[{args.min_peptide_length}, {args.max_peptide_length}]")

    # One example per cluster: CPSea val is heavily redundant, and sampling rows directly
    # would spend the whole control budget on a handful of clusters.
    if "cluster_id" in m.columns:
        before = len(m)
        m = m.groupby("cluster_id", group_keys=False).head(1)
        print(f"deduplicated by cluster_id: {before} -> {len(m)}")

    if len(m) < args.n:
        print(f"WARNING: only {len(m)} candidates, less than requested {args.n}")
    sample = m.sample(n=min(args.n, len(m)), random_state=args.seed).sort_values("example_id")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)
    ids = out.with_suffix(".ids.txt")
    ids.write_text("\n".join(sample["example_id"].astype(str)) + "\n")

    print(f"wrote {len(sample)} rows -> {out}")
    print(f"example_ids -> {ids}")
    print("\nmetadata cyclization_type mix (coarse label; the loader derives the real type "
          "from CONECT records):")
    print(sample["cyclization_type"].value_counts(dropna=False).to_string())
    print("\npeptide_length:", sample["peptide_length"].describe()[["min", "50%", "max"]].to_dict())


if __name__ == "__main__":
    main()
