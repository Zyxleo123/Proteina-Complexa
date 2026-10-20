"""Did refinement repair the ring, or just move atoms? Closure geometry, raw vs refined.

Validity in this dataset is judged on the **raw** prediction, which is the right default:
it asks what the model produced. But a finite coordinate-restrained relax can nudge a
marginal ring across a threshold, and a reader needs to know whether a "valid" count
would change under refinement -- and, more importantly, whether refinement is doing
chemistry or just redistributing error.

So this measures the same closure geometry on both structures and reports the shift. It
deliberately does **not** rewrite the validity verdict: a ring that closes only after the
refiner pulled it shut is not the same result as one the predictor closed, and
overwriting coordinates is not a physical closure repair.

Writes `tables/refined_closure.csv`.

Usage:
    python script_utils/cpseq_refined_closure.py --dataset-dir /data/user_data/yixiz/cpseq/tranche1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import tolerances as tolmod  # noqa: E402
from proteinfoundation.cpseq.geometry import closure_geometry  # noqa: E402

BINDER_CHAIN = "B"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--tolerances", type=Path, default=None)
    args = ap.parse_args()

    tol_path = args.tolerances or (args.dataset_dir / "manifest" / "geometry_tolerances.json")
    tol_set = tolmod.load(tol_path)

    rows = []
    for path in sorted((args.dataset_dir / "scores").glob("rows_*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            ros = r.get("rosetta") or {}
            refined = ros.get("refined_pdb_path")
            raw_clo = (r.get("geometry") or {}).get("closure") or {}
            if not refined or not Path(refined).exists() or raw_clo.get("status") != "ok":
                continue
            chem = r["chemistry"]
            ref_clo = closure_geometry(refined, BINDER_CHAIN, chem)
            if ref_clo.get("status") != "ok":
                rows.append({
                    "task_id": r["task_id"], "predictor": r["predictor"], "chemistry": chem,
                    "sample_index": r["sample_index"], "refined_status": ref_clo.get("status"),
                })
                continue
            tol = tol_set[chem]

            def passes(c):
                d = c.get("bond_length_A")
                return bool(
                    d is not None
                    and tol.bond_length_min_A <= d <= tol.bond_length_max_A
                    and (c.get("angle_i_dev_deg") or 1e9) <= tol.angle_tol_deg
                    and (c.get("angle_j_dev_deg") or 1e9) <= tol.angle_tol_deg
                    and (c.get("dihedral_dev_deg") or 1e9) <= tol.torsion_tol_deg
                )

            rows.append({
                "task_id": r["task_id"],
                "candidate_id": r["candidate_id"],
                "predictor": r["predictor"],
                "chemistry": chem,
                "sample_index": r["sample_index"],
                "raw_valid_overall": bool(r.get("valid")),
                "refined_status": "ok",
                "raw_bond_A": raw_clo.get("bond_length_A"),
                "refined_bond_A": ref_clo.get("bond_length_A"),
                "raw_bond_dev_A": raw_clo.get("bond_length_dev_A"),
                "refined_bond_dev_A": ref_clo.get("bond_length_dev_A"),
                "raw_torsion_dev_deg": raw_clo.get("dihedral_dev_deg"),
                "refined_torsion_dev_deg": ref_clo.get("dihedral_dev_deg"),
                "raw_angle_i_dev_deg": raw_clo.get("angle_i_dev_deg"),
                "refined_angle_i_dev_deg": ref_clo.get("angle_i_dev_deg"),
                # Closure-only verdict on each, so the two are comparable. This is NOT
                # the dataset's validity field, which also covers clashes, chirality and
                # atom completeness and is judged on the raw structure.
                "raw_closure_passes": passes(raw_clo),
                "refined_closure_passes": passes(ref_clo),
            })

    df = pd.DataFrame(rows)
    out = args.dataset_dir / "tables" / "refined_closure.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[refined] {len(df)} refined structures measured -> {out}")

    ok = df[df.refined_status == "ok"] if "refined_status" in df else df
    if not len(ok):
        return
    for (pred, chem), g in ok.groupby(["predictor", "chemistry"]):
        rp, fp = g.raw_closure_passes, g.refined_closure_passes
        gained = int((~rp & fp).sum())
        lost = int((rp & ~fp).sum())
        print(f"\n  {pred} / {chem}  (n={len(g)})")
        print(f"    closure passes raw: {int(rp.sum()):3d}   refined: {int(fp.sum()):3d}"
              f"   gained by refinement: {gained}   lost: {lost}")
        for col in ("bond_dev_A", "torsion_dev_deg", "angle_i_dev_deg"):
            a = pd.to_numeric(g[f"raw_{col}"], errors="coerce").dropna()
            b = pd.to_numeric(g[f"refined_{col}"], errors="coerce").dropna()
            if len(a) and len(b):
                print(f"    {col:16s} median raw {a.median():7.3f} -> refined {b.median():7.3f}")


if __name__ == "__main__":
    main()
