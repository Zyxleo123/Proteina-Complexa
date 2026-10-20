"""Receptor-aligned peptide pose recovery on the experimental controls.

For each control, every predicted sample is compared to the deposited complex: superpose
on the **receptor**, then measure the peptide's CA RMSD in that frame.

Aligning on the receptor is the whole point. Superposing on the peptide instead would
minimise exactly the quantity being measured, and two completely different docking
solutions would report as being in agreement -- which is the one thing pose recovery
exists to detect.

Chain ids differ between the two files and are mapped explicitly: the deposited structure
uses its own author chains (e.g. receptor `B`, peptide `P` for 1SLD), while a prediction
uses this pipeline's convention (receptor `A`, peptide `B`). The mapping comes from
`manifest/targets.csv`, which recorded it when the target was prepared.

**These are controls, not a benchmark.** Every one predates the predictors' training data
and may sit inside it, so a small RMSD here says the pipeline is wired correctly end to
end -- it does not support a claim about prospective accuracy.

Usage:
    python script_utils/cpseq_control_pose_recovery.py \
        --dataset-dir /data/user_data/yixiz/cpseq/controls_tranche
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.contexts import parse_structure  # noqa: E402
from proteinfoundation.cpseq.geometry import read_chain  # noqa: E402

PRED_PEPTIDE_CHAIN = "B"


def ca_of(path: str | Path, chains: list[str], *, reference: bool = False) -> np.ndarray:
    """CA coordinates for `chains`, in residue order.

    `reference=True` parses with `contexts.parse_structure`, which is the SAME parser
    that built the target-only template, so the reference receptor and the predicted
    receptor correspond residue-for-residue.

    This matters: `geometry.read_chain` accepts HETATM records (filtered to standard
    residues) while `parse_structure` takes ATOM only, and on 5TU6 that one-record
    difference gave 297 reference residues against the template's 296 -- enough to make
    every comparison for that control fail as a length mismatch. Two parsers disagreeing
    by one residue is exactly the kind of thing that silently kills a comparison, so the
    reference side now uses the template's own convention.
    """
    if reference:
        parsed = parse_structure(Path(path))
        out = []
        for ch in chains:
            for _, atoms in parsed["atoms"].get(ch, {}).items():
                if "CA" in atoms:
                    out.append(np.asarray(atoms["CA"], dtype=np.float64))
        return np.asarray(out, dtype=np.float64)
    out = []
    for ch in chains:
        for _, atoms in read_chain(path, ch):
            if "CA" in atoms:
                out.append(atoms["CA"])
    return np.asarray(out, dtype=np.float64)


def receptor_aligned_rmsd(
    pred_path: str | Path,
    ref_path: str | Path,
    *,
    pred_receptor: list[str],
    pred_peptide: str,
    ref_receptor: list[str],
    ref_peptide: str,
) -> dict:
    """Peptide CA RMSD after superposing on the receptor. Chain ids may differ per file."""
    ra = ca_of(pred_path, pred_receptor)
    rb = ca_of(ref_path, ref_receptor, reference=True)
    pa = ca_of(pred_path, [pred_peptide])
    pb = ca_of(ref_path, [ref_peptide], reference=True)
    if ra.size == 0 or rb.size == 0:
        return {"status": "no_receptor_ca", "n_pred_rec": len(ra), "n_ref_rec": len(rb)}
    if ra.shape != rb.shape:
        # The template was built from this very crystal, so a mismatch means something
        # upstream changed the residue set -- reported, not silently truncated.
        return {"status": "receptor_length_mismatch", "n_pred_rec": len(ra), "n_ref_rec": len(rb)}
    if pa.shape != pb.shape or pa.size == 0:
        return {"status": "peptide_length_mismatch", "n_pred_pep": len(pa), "n_ref_pep": len(pb)}

    # Kabsch on the receptor only; the transform is then applied to the peptide.
    ca_, cb_ = ra.mean(0), rb.mean(0)
    u, _, vt = np.linalg.svd((ra - ca_).T @ (rb - cb_))
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    rb_moved = (rb - cb_) @ rot.T + ca_
    pb_moved = (pb - cb_) @ rot.T + ca_
    per_atom = np.linalg.norm(pa - pb_moved, axis=1)
    return {
        "status": "ok",
        "receptor_ca_rmsd_after_alignment_A": round(
            float(np.sqrt(((ra - rb_moved) ** 2).sum(1).mean())), 3
        ),
        "peptide_ca_rmsd_receptor_aligned_A": round(float(np.sqrt((per_atom ** 2).mean())), 3),
        "peptide_worst_ca_dev_A": round(float(per_atom.max()), 3),
        "n_receptor_ca": int(len(ra)),
        "n_peptide_ca": int(len(pa)),
        "alignment_frame": "receptor",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    targets = pd.read_csv(args.dataset_dir / "manifest" / "targets.csv").set_index("parent_id")
    rows = []
    for rec_path in sorted((args.dataset_dir / "predictions").rglob("record.json")):
        rec = json.loads(rec_path.read_text())
        if rec["parent_id"] not in targets.index:
            continue
        t = targets.loc[rec["parent_id"]]
        pred_receptor = sorted(set(json.loads(t["chain_map"]).values()))
        ref_receptor = list(json.loads(t["chain_map"]).keys())

        if rec.get("status") != "ok":
            rows.append({
                "parent_id": rec["parent_id"], "pdb_id": t["pdb_id"],
                "predictor": rec["predictor"], "chemistry": rec["chemistry"],
                "sample_index": -1, "status": rec.get("status"),
                "reason": rec.get("reason"),
            })
            continue

        for s in rec.get("samples", []):
            base = {
                "parent_id": rec["parent_id"], "pdb_id": t["pdb_id"],
                "predictor": rec["predictor"], "chemistry": rec["chemistry"],
                "sequence": rec["sequence"], "peptide_length": rec.get("peptide_length"),
                "sample_index": s.get("sample_index"),
                "prediction_status": s.get("status"),
                "sequence_matches": s.get("sequence_matches"),
            }
            if s.get("status") != "ok" or not s.get("pdb_path"):
                rows.append({**base, "status": "no_structure"})
                continue
            res = receptor_aligned_rmsd(
                s["pdb_path"], t["reference_complex_path"],
                pred_receptor=pred_receptor, pred_peptide=PRED_PEPTIDE_CHAIN,
                ref_receptor=ref_receptor, ref_peptide=t["reference_peptide_chain"],
            )
            rows.append({**base, **res})

    df = pd.DataFrame(rows)
    out = args.out or (args.dataset_dir / "tables" / "control_pose_recovery.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[pose] {len(df)} rows -> {out}")

    ok = df[df.get("status") == "ok"] if "status" in df else df
    if len(ok):
        print("\n=== receptor-aligned peptide CA RMSD (A) ===")
        for (pred, chem), g in ok.groupby(["predictor", "chemistry"]):
            v = pd.to_numeric(g["peptide_ca_rmsd_receptor_aligned_A"], errors="coerce").dropna()
            w = pd.to_numeric(g["peptide_worst_ca_dev_A"], errors="coerce").dropna()
            print(f"  {pred:12s} {chem:11s} n={len(v):3d} "
                  f"median={v.median():6.2f} best={v.min():6.2f} worst={v.max():7.2f} "
                  f"| worst-atom median={w.median():6.2f}")
        print("\n=== per control (best sample per predictor) ===")
        for (pdb, pred), g in ok.groupby(["pdb_id", "predictor"]):
            v = pd.to_numeric(g["peptide_ca_rmsd_receptor_aligned_A"], errors="coerce").dropna()
            if len(v):
                print(f"  {pdb} {pred:12s} best={v.min():6.2f} A  (n={len(v)} samples)")
    bad = df[df.get("status").isin(["receptor_length_mismatch", "peptide_length_mismatch"])] \
        if "status" in df else pd.DataFrame()
    if len(bad):
        print(f"\n[pose] {len(bad)} comparison(s) could not be made -- reported, not dropped:")
        print(bad[["pdb_id", "predictor", "status"]].drop_duplicates().to_string(index=False))


if __name__ == "__main__":
    main()
