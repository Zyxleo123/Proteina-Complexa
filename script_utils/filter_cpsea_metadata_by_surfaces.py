#!/usr/bin/env python3
"""Keep CPSea metadata rows that have a valid ``*.surface.npz`` cache.

Use while full-corpus surface extraction is still running so training can use
every complex that is already ready (instead of sample100's ~90).

  .venv/bin/python script_utils/filter_cpsea_metadata_by_surfaces.py \
      --metadata $CPSEA_FULL/preprocessed/metadata/cpsea_train.parquet \
      --surface-dir $PROTEINA_ZFS_PATH/surfaces/cpsea \
      --out $PROTEINA_ZFS_PATH/surfaces/cpsea/metadata/cpsea_train_surfready.parquet

If the official val split has no caches yet, carve a holdout from the filtered
train table:

  .venv/bin/python script_utils/filter_cpsea_metadata_by_surfaces.py \
      --metadata .../cpsea_train.parquet --surface-dir ... \
      --out .../cpsea_train_surfready.parquet \
      --val-out .../cpsea_val_surfready.parquet --val-frac 0.05
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _holdout_mask(ids: list[str], val_frac: float, seed: str = "surface_holdout") -> list[bool]:
    """Deterministic per-id Bernoulli holdout (True = validation)."""
    out = []
    for i in ids:
        h = hashlib.md5(f"{seed}:{i}".encode(), usedforsecurity=False).digest()
        u = int.from_bytes(h[:8], "little") / 2**64
        out.append(u < val_frac)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--surface-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--id-col", default="example_id")
    p.add_argument(
        "--val-out",
        type=Path,
        default=None,
        help="If set, write a holdout split here and write the complement to --out.",
    )
    p.add_argument("--val-frac", type=float, default=0.05)
    args = p.parse_args()

    caches = {
        f.name[: -len(".surface.npz")]
        for f in args.surface_dir.rglob("*.surface.npz")
        if f.is_file() and not f.name.endswith(".npz.tmp")
    }
    table = pq.read_table(args.metadata)
    if args.id_col not in table.column_names:
        raise SystemExit(f"column {args.id_col!r} not in {table.column_names}")
    ids = [str(x) for x in table.column(args.id_col).to_pylist()]
    keep = [i in caches for i in ids]
    filtered = table.filter(pa.array(keep))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.val_out is not None:
        fids = [str(x) for x in filtered.column(args.id_col).to_pylist()]
        is_val = _holdout_mask(fids, args.val_frac)
        val_table = filtered.filter(pa.array(is_val))
        train_table = filtered.filter(pa.array([not v for v in is_val]))
        args.val_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(train_table, args.out)
        pq.write_table(val_table, args.val_out)
        print(
            f"wrote {args.out}: train={train_table.num_rows}  {args.val_out}: val={val_table.num_rows} "
            f"(from {filtered.num_rows} cached / {table.num_rows} metadata; caches={len(caches)})"
        )
        return

    pq.write_table(filtered, args.out)
    print(
        f"wrote {args.out}: kept {filtered.num_rows}/{table.num_rows} "
        f"(caches_available={len(caches)})"
    )


if __name__ == "__main__":
    main()
