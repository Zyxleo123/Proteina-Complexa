"""Regression checks for the two `rosetta_energy._declare_cyclic_bond` fixes.

Both defects were real and both are load-bearing for this arm:

1. **Isopeptide was never declared.** The old branch logged a warning and returned False, so
   `binder_rosetta_cyclic_topology_declared` was 0.0 and the reported dG described an OPEN
   topology -- a linear peptide sitting in a ring-shaped pose. Any isopeptide dG this repo
   recorded before the fix is describing the wrong molecule.

2. **Endpoints assumed the binder is numbered 1..L.** `pdb2pose(binder_chain, i_local + 1)`
   returns 0 on a staged crystal complex that preserves its source numbering, and the call
   raised. This is not hypothetical: it is the normal case for `CPSea_data/lnr_staged`.

Run against any staged complex:

    .venv/bin/python script_utils/test_rosetta_energy_isopeptide.py \
        --pdb CPSea_data/lnr_staged/pdbs/LNR_2qos_C_A.pdb --binder-chain B
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--binder-chain", default="B")
    args = ap.parse_args()

    from proteinfoundation.evaluation import rosetta_energy as RE

    RE._ensure_pyrosetta_init()
    import pyrosetta

    pose0 = pyrosetta.pose_from_pdb(args.pdb)
    chain = args.binder_chain
    pdb_info = pose0.pdb_info()
    binder = [s for s in range(1, pose0.size() + 1) if pdb_info.chain(s) == chain]
    L = len(binder)
    print(f"binder chain {chain}: {L} residues at pose {binder[0]}..{binder[-1]}, "
          f"PDB numbering {pdb_info.number(binder[0])}..{pdb_info.number(binder[-1])}")

    # ---- defect 2: endpoint mapping -------------------------------------------------
    print("\nendpoint mapping (defect 2)")
    check(RE._binder_seqpos(pose0, chain, 0) == binder[0],
          f"local 0 -> pose {RE._binder_seqpos(pose0, chain, 0)} (chain order)")
    check(RE._binder_seqpos(pose0, chain, L - 1) == binder[-1],
          f"local {L - 1} -> pose {RE._binder_seqpos(pose0, chain, L - 1)}")
    for bad in (-1, L):
        try:
            RE._binder_seqpos(pose0, chain, bad)
            check(False, f"local {bad} should be out of range")
        except ValueError:
            check(True, f"local {bad} rejected as out of range")
    try:
        RE._binder_seqpos(pose0, "!", 0)
        check(False, "a missing chain should raise")
    except ValueError:
        check(True, "a missing chain raises rather than returning 0")

    # Show whether the OLD lookup would have worked here, so the record says which case
    # this complex is rather than implying every complex was broken.
    old = pdb_info.pdb2pose(chain, 1)
    if old == binder[0]:
        print(f"  note  this complex IS numbered from 1, so the old pdb2pose lookup also "
              f"worked ({old}); the fix matters on the ones that are not")
    else:
        check(True, f"old pdb2pose(chain, 1) returned {old}, not {binder[0]} -- the old "
                    f"code would have {'raised' if old == 0 else 'mapped the WRONG residue'}")

    # ---- defect 1: isopeptide declaration -------------------------------------------
    print("\nring declaration, all three chemistries (defect 1)")
    sf = pyrosetta.create_score_function("ref2015")
    receptor = [s for s in range(1, pose0.size() + 1) if pdb_info.chain(s) != chain]

    plans = [("mainchain", None, None), ("disulfide", "CYS", "CYS"),
             ("isopeptide", "LYS", "ASP")]
    for cyc_type, aa_i, aa_j in plans:
        pose = pyrosetta.Pose(pose0)
        i_pos, j_pos = binder[0], binder[-1]
        if aa_i:
            for pos, name3 in ((i_pos, aa_i), (j_pos, aa_j)):
                mover = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                mover.set_target(pos)
                mover.set_res_name(name3)
                mover.apply(pose)
        declared = RE._declare_cyclic_bond(pose, chain, cyc_type, 0, L - 1)
        bonded = bool(pose.residue(i_pos).is_bonded(j_pos))
        score = float(sf(pose))
        moved = max((pose0.residue(r).xyz("CA") - pose.residue(r).xyz("CA")).norm()
                    for r in receptor)
        print(f"  {cyc_type:10s} declared={declared} bonded={bonded} "
              f"types={pose.residue(i_pos).name()}/{pose.residue(j_pos).name()} "
              f"score={score:.1f} receptor_moved={moved:.2e} A")
        check(declared, f"{cyc_type}: _declare_cyclic_bond returned True")
        check(bonded, f"{cyc_type}: pose reports an explicit covalent bond")
        check(score == score and abs(score) < 1e7, f"{cyc_type}: ref2015 finite ({score:.1f})")
        check(moved < 1e-6, f"{cyc_type}: receptor untouched ({moved:.2e} A)")

    # The isopeptide patch must rewrite the side chains, not merely name a bond between
    # unmodified ones: NZ with three protons plus a new bond is a five-bonded nitrogen.
    print("\nisopeptide side-chain chemistry")
    pose = pyrosetta.Pose(pose0)
    for pos, name3 in ((binder[0], "LYS"), (binder[-1], "ASP")):
        mover = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mover.set_target(pos)
        mover.set_res_name(name3)
        mover.apply(pose)
    RE._declare_cyclic_bond(pose, chain, "isopeptide", 0, L - 1)
    lys, asp = pose.residue(binder[0]), pose.residue(binder[-1])
    lys_atoms = {lys.atom_name(k).strip() for k in range(1, lys.natoms() + 1)}
    asp_atoms = {asp.atom_name(k).strip() for k in range(1, asp.natoms() + 1)}
    check("SidechainConjugation" in lys.name(), f"LYS carries the patch ({lys.name()})")
    check(not {"1HZ", "2HZ", "3HZ"} & lys_atoms and "HZ" in lys_atoms,
          "LYS NZ lost its ammonium protons and gained one amide proton")
    check(not {"OD1", "OD2"} & asp_atoms and "OD" in asp_atoms,
          "ASP carboxylate became a single amide carbonyl oxygen")
    check(lys.atom_type(lys.atom_index("NZ")).name() == "Nbb",
          f"NZ retyped as an amide nitrogen ({lys.atom_type(lys.atom_index('NZ')).name()})")

    # An isopeptide request at anchors that cannot form one must RAISE, not quietly score an
    # open ring -- which is what the old warn-and-continue path did.
    print("\nrefusal to score an undeclarable ring")
    pose = pyrosetta.Pose(pose0)
    for pos in (binder[0], binder[-1]):
        mover = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mover.set_target(pos)
        mover.set_res_name("ALA")
        mover.apply(pose)
    try:
        RE._declare_cyclic_bond(pose, chain, "isopeptide", 0, L - 1)
        check(False, "ALA/ALA anchors should not yield an isopeptide bond")
    except ValueError as exc:
        check(True, f"ALA/ALA isopeptide raises instead of warning ({str(exc)[:60]}...)")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
