#!/usr/bin/env python3
"""How often does AF3 invert an alpha carbon when the peptide is a SMILES ligand?

This is the measurement that decides whether the SMILES-as-ligand representation is
viable at all, and it is deliberately independent of Rosetta, of the PDB writer and of
the project's own chirality check: it reads AF3's mmCIF coordinates, puts them on the
molecule through the verified SMILES-order permutation, and asks RDKit to re-derive the
configuration from 3D.

Why it matters. The frozen gate carries `require_all_L: true`. A fold with a D-residue is
rejected on stereochemistry no matter how well it binds, so its dG is not a usable number.
A SMILES states every stereocentre explicitly, but AF3 tokenises a ligand per ATOM and
nothing in that representation enforces a chiral centre -- so whether the stated
configuration survives prediction is an empirical question, not a guarantee.

One fold of 5TU6 came back with 3 of 7 centres inverted (ASN2, LEU5, TYR6), agreed by two
independent implementations. That is a mechanism, not noise, but it is n=1. This runs it
over every fold on disk so the claim rests on a rate.

If the rate is high, the fix is not to repair the coordinates -- that would be editing the
structure to pass its own gate. It is to stop representing the peptide as a ligand: as a
polymer chain with `bondedAtomPairs` for the closure, AF3 builds residues from its own
L-templates (visible in any output CIF's `_chem_comp` block as `L-PEPTIDE LINKING`), so
the configuration is correct by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from proteinfoundation.cpseq.adapters.af3 import (  # noqa: E402
    AF3MappingError,
    map_ligand_atoms,
    read_model_cif,
)
from proteinfoundation.cpseq.smiles import cyclic_peptide  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def survey_fold(cif: Path, sequence: str) -> dict:
    peptide = cyclic_peptide(sequence)
    ligand = [a for a in read_model_cif(cif) if a.comp_id.startswith("LIG_")]
    try:
        map_ligand_atoms(ligand, peptide)
    except AF3MappingError as exc:
        return {"cif": str(cif), "status": "mapping_refused", "error": str(exc)}

    mol = Chem.Mol(peptide.mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for k, mol_index in enumerate(peptide.smiles_atom_order):
        atom = ligand[k]
        conf.SetAtomPosition(mol_index, Point3D(atom.x, atom.y, atom.z))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    # From 3D, not from the input SMILES: the question is what the COORDINATES say.
    Chem.AssignStereochemistryFrom3D(mol)
    centres = dict(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False))

    per_residue = []
    for record in peptide.atom_records:
        if record.atom_name != "CA" or record.residue_name == "GLY":
            continue
        tag = centres.get(record.index)
        expected = "R" if record.residue_name == "CYS" else "S"
        per_residue.append({
            "residue_number": record.residue_number,
            "residue_name": record.residue_name,
            "cip": tag,
            "expected": expected,
            "configuration": "L" if tag == expected else ("D" if tag else "unassigned"),
        })
    inverted = [r for r in per_residue if r["configuration"] != "L"]
    return {
        "cif": str(cif),
        "status": "ok",
        "sequence": sequence,
        "n_scorable_centres": len(per_residue),
        "n_inverted": len(inverted),
        "all_L": not inverted,
        "inverted_residues": [f"{r['residue_name']}{r['residue_number']}" for r in inverted],
        "per_residue": per_residue,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-dir", type=Path, required=True,
                    help="directory holding per-complex subdirs with folds.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    folds_jsonl = args.folds_dir / "folds.jsonl"
    if not folds_jsonl.exists():
        print(f"FATAL: no {folds_jsonl}")
        return 1
    jobs = [json.loads(line) for line in folds_jsonl.read_text().splitlines() if line.strip()]

    rows = []
    for job in jobs:
        if job.get("status") != "ok":
            print(f"[chir] {job['label']}: skipped ({job.get('status')})")
            continue
        run_dir = Path(job["run_dir"])
        # Per-fold files only. AF3 also writes a top-level copy of the top-ranked sample,
        # and counting it would double-weight one seed.
        cifs = sorted(p for p in run_dir.rglob("*_model.cif") if "seed-" in p.parent.name)
        for cif in cifs:
            row = survey_fold(cif, job["sequence"])
            row["label"] = job["label"]
            row["fold"] = cif.parent.name
            rows.append(row)
            if row["status"] == "ok":
                print(f"[chir] {job['label']:8s} {row['fold']:22s} "
                      f"{row['n_inverted']}/{row['n_scorable_centres']} inverted"
                      + (f"  {row['inverted_residues']}" if row["inverted_residues"] else "  all L"))
            else:
                print(f"[chir] {job['label']:8s} {row['fold']:22s} {row['status']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    ok = [r for r in rows if r["status"] == "ok"]
    print("\n=== per complex ===")
    print(f"{'complex':10s} {'folds':>6s} {'all-L folds':>12s} "
          f"{'centres':>8s} {'inverted':>9s} {'rate':>7s}")
    for label in sorted({r["label"] for r in ok}):
        sub = [r for r in ok if r["label"] == label]
        centres = sum(r["n_scorable_centres"] for r in sub)
        inv = sum(r["n_inverted"] for r in sub)
        all_l = sum(1 for r in sub if r["all_L"])
        print(f"{label:10s} {len(sub):6d} {f'{all_l}/{len(sub)}':>12s} "
              f"{centres:8d} {inv:9d} {inv / centres if centres else 0:7.3f}")

    if ok:
        centres = sum(r["n_scorable_centres"] for r in ok)
        inv = sum(r["n_inverted"] for r in ok)
        all_l = sum(1 for r in ok if r["all_L"])
        print(f"\n=== pooled over {len(ok)} folds ===")
        print(f"  folds that are entirely L      {all_l}/{len(ok)}")
        print(f"  alpha carbons inverted         {inv}/{centres} = {inv / centres:.3f}")
        worst = Counter()
        for r in ok:
            for name in r["inverted_residues"]:
                worst[name] += 1
        if worst:
            print(f"  most-inverted positions        {worst.most_common(8)}")
        print("\n  A fold that is not entirely L fails the frozen gate on `all_L`")
        print("  regardless of its energy, so these folds yield no usable dG.")
        if all_l == 0:
            print("\n  NO fold is entirely L. The SMILES-as-ligand representation does not")
            print("  preserve the stereochemistry it states, so this arm cannot produce a")
            print("  gate-passing structure. The peptide has to go back to being a POLYMER")
            print("  chain (closure via bondedAtomPairs, or HighFold3), where AF3 builds")
            print("  residues from its own L-templates.")
    print(f"\n[chir] rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
