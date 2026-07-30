#!/usr/bin/env python3
"""Score generated CA traces (or surface caches) against oracle peptide-surface caches.

    python script_utils/score_surface_agreement.py \
        --oracle-dir surfaces/cpsea_sample100 \
        --pred-pdb-dir generated_pdbs \
        --binder-chain B \
        --output surface_scores.csv

For each ``*.pdb`` whose stem matches an oracle ``*.surface.npz``, computes Chamfer,
normal consistency (proxy), and coverage of the oracle interface points vs binder CA.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.eval.surface_metrics import surface_agreement_metrics  # noqa: E402
from proteinfoundation.surface.peptide_surface import load_surface_cache  # noqa: E402


def read_ca(pdb: Path, chain: str) -> np.ndarray:
    xyz = []
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[21] == chain and line[12:16].strip() == "CA":
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-dir", type=Path, required=True)
    ap.add_argument("--pred-pdb-dir", type=Path, required=True)
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    for pdb in sorted(args.pred_pdb_dir.glob("*.pdb")):
        stem = pdb.stem
        cache = args.oracle_dir / f"{stem}.surface.npz"
        if not cache.exists():
            # Allow stem prefixes (e.g. design id vs complex id)
            matches = list(args.oracle_dir.glob(f"*{stem}*.surface.npz"))
            if not matches:
                continue
            cache = matches[0]
        oracle = load_surface_cache(cache)
        valid = oracle.sampled_valid_mask
        pred = read_ca(pdb, args.binder_chain)
        if pred.size == 0:
            continue
        m = surface_agreement_metrics(
            pred_xyz=pred,
            oracle_xyz=oracle.sampled_xyz[valid],
            oracle_normals=oracle.sampled_normals[valid],
        )
        rows.append({"example_id": stem, "oracle_cache": cache.name, **m})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("no matching pdb/oracle pairs found")
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
