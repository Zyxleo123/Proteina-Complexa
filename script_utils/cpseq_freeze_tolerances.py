"""Derive the tranche's geometry tolerances from the experimental controls, then freeze.

Run once, before scoring. Writes `manifest/geometry_tolerances.json`, which every
scoring job then reads read-only, so a mid-tranche threshold change cannot happen
silently.

Usage:
    python script_utils/cpseq_freeze_tolerances.py \
        --dataset-dir /data/user_data/yixiz/cpseq/tranche1 \
        --controls /data/user_data/yixiz/cpseq/controls/controls.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.geometry import closure_geometry  # noqa: E402
from proteinfoundation.cpseq.tolerances import derive_tolerances, freeze  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    controls = json.loads(args.controls.read_text())
    by_chem: dict[str, list[dict]] = defaultdict(list)
    for c in controls:
        geo = closure_geometry(c["structure_path"], c["peptide_chain"], c["chemistry"])
        geo["pdb_id"] = c["pdb_id"]
        by_chem[c["chemistry"]].append(geo)
        print(f"[tol] {c['pdb_id']} {c['chemistry']}: bond={geo.get('bond_length_A')}A "
              f"dev={geo.get('bond_length_dev_A')}A "
              f"angle_dev=({geo.get('angle_i_dev_deg')},{geo.get('angle_j_dev_deg')}) "
              f"torsion_dev={geo.get('dihedral_dev_deg')}")

    tol = derive_tolerances(dict(by_chem))
    out = args.out or (args.dataset_dir / "manifest" / "geometry_tolerances.json")
    freeze(tol, out)

    print(f"\n[tol] frozen -> {out}")
    for chem, t in tol.items():
        print(f"  {chem:11s} bond [{t.bond_length_min_A}, {t.bond_length_max_A}] A | "
              f"angle +/-{t.angle_tol_deg} deg | torsion +/-{t.torsion_tol_deg} deg | "
              f"n_controls={t.n_controls}")
        if t.n_controls == 0:
            print(f"               (no control of this chemistry -- fallback tolerances)")


if __name__ == "__main__":
    main()
