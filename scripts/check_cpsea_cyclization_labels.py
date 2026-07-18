#!/usr/bin/env python3
"""
Sanity-check CPSea cyclization-linkage labels across a preprocessed dataset.

For every sample in the given metadata parquet(s), re-derives the
binder-local cyclization label (`i`, `j`, `type`) purely from the actual
bonded-atom identity in the preprocessed PDB's `CONECT` records (see
`proteinfoundation.cyclization.parse_labels.infer_cyclization_label`), then
reports:
  - How many samples resolve to each ground-truth type (MAINCHAIN /
    DISULFIDE / ISOPEPTIDE) vs. how many remain unlabeled (and why).
  - A cross-tab against the coarse dataset metadata `cyclization_type`
    hint ("head_tail" / "disulfide" / "other"), which is never used to
    assign the label -- only to sanity-check it here.
  - How often the resolved type actually disagrees with what the coarse
    hint would suggest (`hint_mismatch`), which should normally be 0.

Examples
--------
Local 100-sample checkout:
    python scripts/check_cpsea_cyclization_labels.py \\
        CPSea_data/preprocessed_sample100/metadata/cpsea_train.parquet

Full preprocessed dataset (all splits), using $CPSEA_DATA_PATH:
    python scripts/check_cpsea_cyclization_labels.py --use-env-data-root

Only print the summary tables:
    python scripts/check_cpsea_cyclization_labels.py <metadata.parquet> --quiet
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from proteinfoundation.cyclization.constants import CYCLIZATION_TYPE_TO_NAME  # noqa: E402
from proteinfoundation.cyclization.parse_labels import infer_cyclization_label  # noqa: E402


def residue_pdb_idx_from_pdb(path: str, chain: str) -> list[int]:
    """Binder-local index -> original PDB resSeq, in file order (post dedup)."""
    seen: list[int] = []
    seen_set: set[int] = set()
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    mode = "rt" if str(path).endswith(".gz") else "r"
    with opener(path, mode) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[21] != chain:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq not in seen_set:
                seen_set.add(resseq)
                seen.append(resseq)
    return seen


def find_metadata_parquets(use_env_data_root: bool, explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    if not use_env_data_root:
        raise SystemExit("Provide metadata parquet path(s), or pass --use-env-data-root.")
    data_root = os.environ.get("CPSEA_DATA_PATH")
    if not data_root:
        raise SystemExit("CPSEA_DATA_PATH is not set (source env.sh first).")
    metadata_dir = Path(data_root) / "preprocessed" / "metadata"
    paths = sorted(metadata_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"No parquet files found under {metadata_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", nargs="*", help="Path(s) to CPSea preprocessed metadata parquet file(s).")
    parser.add_argument(
        "--use-env-data-root",
        action="store_true",
        help="Ignore positional args and scan every split under $CPSEA_DATA_PATH/preprocessed/metadata/.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only check the first N rows (per file).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-row mismatch/failure details.")
    args = parser.parse_args()

    import pandas as pd

    paths = find_metadata_parquets(args.use_env_data_root, args.metadata)

    resolved_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    hint_x_resolved: dict[str, Counter[str]] = {}
    mismatch_rows: list[str] = []
    total = 0

    for path in paths:
        df = pd.read_parquet(path)
        if args.limit is not None:
            df = df.head(args.limit)

        print(f"\n## {path} ({len(df)} rows)")
        for _, row in df.iterrows():
            total += 1
            binder_chain_id = row.get("binder_chain_id", "B") or "B"
            hint = row.get("cyclization_type", None)
            try:
                residue_pdb_idx = residue_pdb_idx_from_pdb(row["path"], binder_chain_id)
                label = infer_cyclization_label(
                    pdb_path=row["path"],
                    residue_pdb_idx=residue_pdb_idx,
                    binder_length_hint=row.get("peptide_length", None),
                    binder_chain_id=binder_chain_id,
                    cyclization_type_hint=hint,
                )
            except Exception as e:  # noqa: BLE001 - sanity script, keep going on bad rows
                label = {"has_cyclization": False, "reason": f"exception:{type(e).__name__}: {e}"}

            if label["has_cyclization"]:
                resolved = CYCLIZATION_TYPE_TO_NAME.get(label["type"], "UNKNOWN_TYPE")
            else:
                resolved = "NO_LABEL"
            resolved_counts[resolved] += 1
            reason_counts[label["reason"].split("(")[0]] += 1

            hint_key = str(hint) if hint is not None else "<none>"
            hint_x_resolved.setdefault(hint_key, Counter())[resolved] += 1

            if "hint_mismatch" in label["reason"]:
                mismatch_rows.append(f"{row.get('example_id', row['path'])}: hint={hint} -> {label}")

    if mismatch_rows and not args.quiet:
        print("\n## Hint / resolved-type mismatches (bond identity won; hint was NOT used to assign the label)")
        for line in mismatch_rows:
            print(line)

    print("\n## Resolved type counts")
    print("| type | count | fraction |")
    print("|---|---:|---:|")
    for type_name, count in resolved_counts.most_common():
        print(f"| {type_name} | {count} | {count / total:.1%} |")

    print("\n## `has_cyclization=False` reasons")
    for reason, count in reason_counts.most_common():
        if reason not in ("conect_disulfide", "conect_mainchain", "conect_isopeptide"):
            print(f"  {reason}: {count}")

    print("\n## Cross-tab: metadata `cyclization_type` hint vs. resolved (bond-based) type")
    print("| hint | " + " | ".join(sorted(resolved_counts.keys())) + " |")
    print("|---|" + "---:|" * len(resolved_counts))
    for hint_key in sorted(hint_x_resolved.keys()):
        counts = hint_x_resolved[hint_key]
        row_str = " | ".join(str(counts.get(t, 0)) for t in sorted(resolved_counts.keys()))
        print(f"| {hint_key} | {row_str} |")

    print(f"\nTotal rows checked: {total}")
    print(f"hint_mismatch (bond identity overrode a misleading/coarse hint): {len(mismatch_rows)}")


if __name__ == "__main__":
    main()
