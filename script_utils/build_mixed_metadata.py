#!/usr/bin/env python3
"""Concatenate CPSea and linear-peptide metadata into one parquet with a `dataset_source` column.

`StructureDataModule` reads exactly one metadata file per split, so mixing two datasets
means writing one file. This does that, tags every row with which dataset it came from,
and reports the resulting composition.

It does NOT decide the mixing ratio. Ratios are a *training* knob
(`dataset.datamodule.source_fractions`), applied at sampling time by a weighted sampler,
so changing the mix does not mean rebuilding a 2.4M-row file -- and so the ratio is
visible in the run config rather than baked into a data artifact nobody re-reads.

Memory. The CPSea `full` train metadata is ~133 MB of parquet and roughly 2.44M rows of
python strings; loading it with pandas is enough to OOM a login node. This streams row
groups with pyarrow and never materialises the whole frame.

Validation split. CPSea val alone is 134k rows against PepBench's 114, so a naive
concatenation produces a val loss in which the linear data is 0.08% of the signal --
i.e. a val curve that cannot see the thing the run is testing. `--val-cpsea-sample`
downsamples the CPSea side of the *val* split (deterministically, seeded) so both sources
are actually represented. The train split is never subsampled; that is the sampler's job.

Run (see ``scripts/build_mixed_metadata.sbatch``):

    python -m script_utils.build_mixed_metadata \\
        --cpsea-train  $ZFS/CPSea/CPSea_full/CPSea/preprocessed/metadata/cpsea_train.parquet \\
        --cpsea-val    $ZFS/CPSea/CPSea_full/CPSea/preprocessed/metadata/cpsea_val.parquet \\
        --lp-train     $ZFS/LPData/preprocessed/metadata/lp_train.parquet \\
        --lp-val       $ZFS/LPData/preprocessed/metadata/lp_val.parquet \\
        --out-dir      $ZFS/LPData/preprocessed/metadata_mixed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from loguru import logger

# Every row the mixed file carries must have these; the datamodule's `columns_to_load`
# and the transforms read them by name. `dataset_source` is added here.
REQUIRED_COLUMNS = [
    "example_id",
    "path",
    "binder_chain_id",
    "cluster_id",
    "peptide_length",
    "cyclization_type",
]
SOURCE_COLUMN = "dataset_source"


def _tagged_batches(path: Path, default_source: str | None, batch_size: int = 65536):
    """Yield record batches from `path`, projected to REQUIRED_COLUMNS + dataset_source.

    Projecting here rather than after concatenation is what keeps peak memory flat: the
    CPSea metadata carries `source_path`, `receptor_length` and `config_hash` columns
    training never reads, and they are the bulk of the file.

    Args:
        path: parquet file to read.
        default_source: value for `dataset_source` when the file has no such column
            (CPSea predates it). If the file *does* have one, it wins, and a mismatch
            with `default_source` is an error rather than a silent relabel.
    """
    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing} (has {sorted(present)})")
    has_source = SOURCE_COLUMN in present
    columns = REQUIRED_COLUMNS + ([SOURCE_COLUMN] if has_source else [])

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        if has_source:
            if default_source is not None:
                uniq = set(pc.unique(batch.column(SOURCE_COLUMN)).to_pylist())
                if uniq - {default_source}:
                    raise ValueError(
                        f"{path}: --*-source was given as {default_source!r} but the file already "
                        f"carries dataset_source values {sorted(uniq)}. Refusing to relabel."
                    )
            yield batch
        else:
            src = pa.array([default_source] * batch.num_rows, type=pa.string())
            yield pa.RecordBatch.from_arrays(list(batch.columns) + [src], names=list(batch.schema.names) + [SOURCE_COLUMN])


def _normalize(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    """Cast a batch to the writer's schema so sources with different int widths merge."""
    return pa.Table.from_batches([batch]).cast(schema).to_batches()[0]


def build_split(
    out_path: Path,
    sources: list[tuple[Path, str | None]],
    sample: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Write one mixed parquet. Returns per-source row counts actually written.

    Args:
        out_path: destination parquet.
        sources: [(parquet path, dataset_source label)], written in order. A label of
            None means the file already carries its own `dataset_source` column and it
            is authoritative -- that is the LP case, where one file holds both
            "pepbench" and "protfrag" rows and a single label would erase the
            distinction.
        sample: optional {label: max rows to keep}, applied per input FILE. Rows are
            taken on a fixed stride across the whole file rather than as a prefix: the
            CPSea metadata is ordered by cluster, so a prefix is a biased slice of
            clusters, not a sample. Only files with a non-None label can be subsampled.
        seed: stride offset, so two builds with different seeds pick different rows.
    """
    schema = pa.schema(
        [
            pa.field("example_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("binder_chain_id", pa.string()),
            pa.field("cluster_id", pa.string()),
            pa.field("peptide_length", pa.int64()),
            pa.field("cyclization_type", pa.string()),
            pa.field(SOURCE_COLUMN, pa.string()),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    writer = pq.ParquetWriter(out_path, schema)
    try:
        for path, label in sources:
            n_rows = pq.ParquetFile(path).metadata.num_rows
            keep = sample.get(label) if (sample and label is not None) else None
            stride = 1
            if keep is not None and 0 < keep < n_rows:
                stride = n_rows // keep
                logger.info(f"  {label}: subsampling {n_rows} -> ~{keep} rows (every {stride}th, offset {seed % stride})")
            row_idx = 0
            written_here = 0
            for batch in _tagged_batches(path, label):
                if stride > 1:
                    offset = seed % stride
                    idx = [i for i in range(batch.num_rows) if (row_idx + i - offset) % stride == 0]
                    row_idx += batch.num_rows
                    if not idx:
                        continue
                    batch = pa.Table.from_batches([batch]).take(pa.array(idx)).to_batches()[0]
                else:
                    row_idx += batch.num_rows
                batch = _normalize(batch, schema)
                writer.write_batch(batch)
                written_here += batch.num_rows
                # Count from the column, not from `label`: a file with its own
                # dataset_source may hold several sources in one parquet.
                for vc in pc.value_counts(batch.column(SOURCE_COLUMN)):
                    counts[vc["values"].as_py()] += vc["counts"].as_py()
            logger.info(f"  {path.name}: wrote {written_here} rows")
    finally:
        writer.close()
    return dict(counts)


def main():
    p = argparse.ArgumentParser(description="Build mixed CPSea + linear-peptide metadata.")
    p.add_argument("--cpsea-train", type=Path, required=True)
    p.add_argument("--cpsea-val", type=Path, required=True)
    p.add_argument("--lp-train", type=Path, required=True)
    p.add_argument("--lp-val", type=Path, default=None, help="omit if the LP preprocess produced no val split")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cpsea-source-name", default="cpsea")
    p.add_argument(
        "--val-cpsea-sample",
        type=int,
        default=2000,
        help="cap CPSea rows in the mixed VAL split so the LP side is visible in the val loss; "
        "0 disables (keeps all CPSea val rows)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building mixed TRAIN metadata (no subsampling; ratios are the sampler's job)")
    train_counts = build_split(
        out_dir / "mixed_train.parquet",
        [(args.cpsea_train, args.cpsea_source_name), (args.lp_train, None)],
        seed=args.seed,
    )

    logger.info("Building mixed VAL metadata")
    val_sources: list[tuple[Path, str]] = [(args.cpsea_val, args.cpsea_source_name)]
    if args.lp_val and args.lp_val.exists():
        val_sources.append((args.lp_val, None))
    else:
        logger.warning(f"No LP val parquet at {args.lp_val}; val split will be CPSea only.")
    val_counts = build_split(
        out_dir / "mixed_val.parquet",
        val_sources,
        sample={args.cpsea_source_name: args.val_cpsea_sample} if args.val_cpsea_sample else None,
        seed=args.seed,
    )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
        "seed": args.seed,
        "val_cpsea_sample": args.val_cpsea_sample,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "train_total": sum(train_counts.values()),
        "val_total": sum(val_counts.values()),
        "train_natural_fractions": {
            k: v / max(1, sum(train_counts.values())) for k, v in train_counts.items()
        },
    }
    (out_dir / "mix_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(json.dumps(manifest, indent=2))
    logger.info(
        "Natural train fractions above are what UNWEIGHTED sampling would give. Set "
        "dataset.datamodule.source_fractions in the training config to override them."
    )


if __name__ == "__main__":
    main()
