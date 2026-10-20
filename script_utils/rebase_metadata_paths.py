#!/usr/bin/env python3
"""Rewrite absolute structure paths in the CPSea split metadata after a cluster move.

StructureDataset._construct_path only joins base_dir onto a *relative* path, so the
absolute /zfsauton paths the parquets carry from rhea are used verbatim and every row
fails to open. Rewriting the prefix keeps the split identity (row order and membership
are untouched) while making the paths resolve here.

The originals are kept as <name>.parquet.rhea so the move stays reversible.

Usage:
  python script_utils/rebase_metadata_paths.py --old /zfsauton/scratch/yixiz --new /data/user_data/yixiz
  python script_utils/rebase_metadata_paths.py ... --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PATH_COLS = ("path", "source_path")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/user_data/yixiz/CPSea",
                   help="tree to scan for */metadata/*.parquet")
    p.add_argument("--old", required=True, help="path prefix to replace")
    p.add_argument("--new", required=True, help="replacement prefix")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    files = sorted(Path(a.root).glob("**/metadata/*.parquet"))
    if not files:
        print(f"no metadata parquets under {a.root}")
        return 1

    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        cols = [c for c in PATH_COLS if c in names]
        if not cols:
            print(f"{f.name}: no path columns, skipped")
            continue

        # One row group at a time: the train split is 2.4M rows, and loading it
        # whole gets the process OOM-killed inside a modest Slurm allocation.
        if a.dry_run:
            head = next(pf.iter_batches(batch_size=1024, columns=cols))
            hits = {c: int(pc.sum(pc.starts_with(head[c], pattern=a.old)).as_py() or 0)
                    for c in cols}
            print(f"{f.name}: {pf.metadata.num_rows} rows, first batch " +
                  ", ".join(f"{c}={n}/{head.num_rows}" for c, n in hits.items()))
            continue

        backup = f.with_suffix(".parquet.rhea")
        if not backup.exists():
            shutil.copy2(f, backup)

        tmp = f.with_suffix(".parquet.tmp")
        rewritten = 0
        writer = None
        try:
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg)
                for c in cols:
                    col = tbl[c].cast(pa.string())
                    rewritten += int(pc.sum(pc.starts_with(col, pattern=a.old)).as_py() or 0)
                    tbl = tbl.set_column(
                        tbl.schema.get_field_index(c), c,
                        pc.replace_substring(col, pattern=a.old, replacement=a.new),
                    )
                if writer is None:
                    writer = pq.ParquetWriter(tmp, tbl.schema)
                writer.write_table(tbl)
        finally:
            if writer is not None:
                writer.close()
        # swap only once the whole file is written, so an interrupted run never
        # leaves a half-rewritten split behind
        tmp.replace(f)
        print(f"{f.name}: {pf.metadata.num_rows} rows, {rewritten} paths rewritten")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
