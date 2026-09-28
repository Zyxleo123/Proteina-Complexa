"""Checks for `cp_topology`: pure bookkeeping offline, ring declaration against a real complex.

Run the offline half anywhere:

    .venv/bin/python script_utils/test_cp_topology.py

Add a staged complex to exercise the Rosetta half (needs PyRosetta):

    .venv/bin/python script_utils/test_cp_topology.py \
        --pdb CPSea_data/lnr_staged/pdbs/LNR_2qos_C_A.pdb --binder-chain B
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cp_topology as T  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


def test_isopeptide_atoms() -> None:
    print("isopeptide atom resolution")
    a = T.isopeptide_atoms("LYS", "ASP")
    check(a == {"atom_i": "NZ", "atom_j": "CG", "prev_j": "CB", "next_i": "CE"},
          f"LYS(i)/ASP(j) -> {a}")
    b = T.isopeptide_atoms("ASP", "LYS")
    check(b == {"atom_i": "CG", "atom_j": "NZ", "prev_j": "CE", "next_i": "CB"},
          f"ASP(i)/LYS(j) swaps the roles -> {b}")
    c = T.isopeptide_atoms("LYS", "GLU")
    check(c["atom_j"] == "CD" and c["prev_j"] == "CG", f"GLU acylates from CD -> {c}")
    for bad in (("LYS", "LYS"), ("ASP", "GLU"), ("ALA", "ASP")):
        try:
            T.isopeptide_atoms(*bad)
            check(False, f"{bad} should raise")
        except ValueError:
            check(True, f"{bad} rejected")


def test_anchor_costs() -> None:
    print("anchor identity cost / budget feasibility")
    mc = T.anchor_mutation_cost("LEU", "TYR", T.MAINCHAIN)
    check(len(mc) == 1 and mc[0]["n_anchor_mutations"] == 0,
          "mainchain costs 0 anchor mutations on any LP")
    ss = T.anchor_mutation_cost("LEU", "TYR", T.DISULFIDE)
    check(ss[0]["n_anchor_mutations"] == 2, "disulfide on LEU/TYR costs 2")
    check(not T.feasible_at_budget("LEU", "TYR", T.DISULFIDE, 0),
          "disulfide is chemically EMPTY at budget 0")
    check(not T.feasible_at_budget("LEU", "TYR", T.DISULFIDE, 1),
          "disulfide is chemically EMPTY at budget 1")
    check(T.feasible_at_budget("LEU", "TYR", T.DISULFIDE, 2),
          "disulfide becomes reachable at budget 2")
    check(T.feasible_at_budget("CYS", "TYR", T.DISULFIDE, 1),
          "an LP already ending in CYS needs only 1")
    check(T.min_anchor_cost("CYS", "CYS", T.DISULFIDE) == 0,
          "CYS/CYS termini make disulfide free")
    iso = T.anchor_mutation_cost("LYS", "TYR", T.ISOPEPTIDE)
    check(iso[0]["n_anchor_mutations"] == 1,
          f"LYS already at i -> cheapest isopeptide costs 1 ({iso[0]})")
    check(T.min_anchor_cost("LEU", "ASP", T.ISOPEPTIDE) == 1,
          "ASP already at j -> 1 (mutate i to LYS)")


def test_windows() -> None:
    print("released-residue bookkeeping")
    bp = list(range(174, 185))          # 11 residues, as LNR_2qos_C_A loads
    r = T.released_residues(bp, 3, 3)
    check(r["loop_order"] == [182, 183, 184, 174, 175, 176],
          f"loop runs C-side then across the bond into N-side -> {r['loop_order']}")
    check(r["anchor_lower"] == 181 and r["anchor_upper"] == 177,
          f"anchors flank the loop -> {r['anchor_lower']}, {r['anchor_upper']}")
    check(r["released_local"] == [0, 1, 2, 8, 9, 10], f"local indices -> {r['released_local']}")
    check(r["n_released"] == 6, "6 residues released")
    try:
        T.released_residues(bp, 5, 5)   # 10 released of 11 leaves no distinct anchors
        check(False, "k=m=5 on L=11 should raise")
    except ValueError:
        check(True, "a window that leaves no anchor is rejected")
    try:
        T.released_residues(bp, 0, 3)
        check(False, "k=0 should raise")
    except ValueError:
        check(True, "releasing 0 residues at a terminus is rejected")

    ladder = T.window_ladder(11)
    check(ladder[0] == (1, 1), f"ladder starts at the smallest window -> {ladder[:4]}")
    check(all(k + m <= 9 for k, m in ladder), "every rung leaves an anchor")
    check([sum(x) for x in ladder] == sorted(sum(x) for x in ladder),
          "ladder expands monotonically in released count")
    short = T.window_ladder(5)
    check(all(k + m <= 3 for k, m in short), f"L=5 caps at 3 released -> {short}")


def test_rosetta(pdb: str, binder_chain: str) -> None:
    print(f"\nRosetta ring declaration on {pdb}")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init

    _ensure_pyrosetta_init()
    import pyrosetta

    pose0 = pyrosetta.pose_from_pdb(pdb)
    bp = T.binder_positions(pose0, binder_chain)
    rp = T.receptor_positions(pose0, binder_chain)
    L = len(bp)
    print(f"  binder {L} residues at pose {bp[0]}..{bp[-1]}; receptor {len(rp)} residues")
    check(L > 0 and len(rp) > 0, "chains split")
    check(bp == list(range(bp[0], bp[0] + L)), "binder positions are contiguous in pose order")

    native_i, native_j = pose0.residue(bp[0]).name3(), pose0.residue(bp[-1]).name3()
    sf = pyrosetta.create_score_function("ref2015")

    for chem_name in ("mainchain", "disulfide", "isopeptide"):
        chem = T.chemistry(chem_name)
        pose = pyrosetta.Pose(pose0)
        i_pos, j_pos = bp[0], bp[-1]
        opt = T.anchor_mutation_cost(native_i, native_j, chem)[0]
        T.mutate_residue(pose, i_pos, opt["aa_i"])
        T.mutate_residue(pose, j_pos, opt["aa_j"])
        atoms = T.resolve_ring_atoms(chem, opt["aa_i"], opt["aa_j"])
        info = T.declare_ring_bond(pose, i_pos, j_pos, chem, atoms)
        geom = T.ring_bond_geometry(pose, i_pos, j_pos, chem, atoms)
        n_cst = T.add_ring_constraints(pose, i_pos, j_pos, chem, atoms)
        score = sf(pose)
        print(f"  {chem_name:10s} anchors={opt['aa_i']}/{opt['aa_j']} "
              f"cost={opt['n_anchor_mutations']} types={info['residue_types']} "
              f"bond={geom['bond_atoms']} len={geom['bond_length_A']} "
              f"ncst={n_cst} score={score:.1f}")
        # Declaring the ring changes variant types, which marks the PDBInfo obsolete. Anything
        # that then addresses residues by CHAIN breaks -- and because only SOME chemistries
        # trip it, an aggregate check passes while one whole chemistry silently produces
        # nothing. So the invariant is asserted per chemistry, right after declaration.
        info_valid = pose.pdb_info() is not None and not pose.pdb_info().obsolete()
        check(T.ensure_pdb_info(pose, pose0),
              f"{chem_name}: PDBInfo is valid or rebuildable after declaration "
              f"(was {'valid' if info_valid else 'OBSOLETE/absent'})")
        check(pose.pdb_info() is not None and not pose.pdb_info().obsolete()
              and all(pose.pdb_info().chain(r) == pose0.pdb_info().chain(r)
                      for r in (bp[0], bp[-1], rp[0])),
              f"{chem_name}: rebuilt PDBInfo preserves chain assignments")
        check(bool(info["declared"]), f"{chem_name}: declare_ring_bond returned declared")
        check(bool(geom["bond_is_declared"]),
              f"{chem_name}: pose reports an explicit covalent bond")
        check(n_cst >= 3, f"{chem_name}: ring constraints added ({n_cst})")
        check(score == score and abs(score) < 1e7,
              f"{chem_name}: ref2015 is finite ({score:.1f})")
        # the declaration must not move anything
        moved = max((pose0.residue(r).xyz("CA") - pose.residue(r).xyz("CA")).norm() for r in rp)
        check(moved < 1e-6, f"{chem_name}: receptor untouched by declaration (max {moved:.2e} A)")

    # The isopeptide-specific defect this module exists to fix: the patch must actually
    # rewrite the side chains, not merely declare a bond between unmodified ones.
    pose = pyrosetta.Pose(pose0)
    T.mutate_residue(pose, bp[0], "LYS")
    T.mutate_residue(pose, bp[-1], "ASP")
    T.declare_ring_bond(pose, bp[0], bp[-1], T.ISOPEPTIDE,
                        T.resolve_ring_atoms(T.ISOPEPTIDE, "LYS", "ASP"))
    lys_atoms = {pose.residue(bp[0]).atom_name(k).strip()
                 for k in range(1, pose.residue(bp[0]).natoms() + 1)}
    asp_atoms = {pose.residue(bp[-1]).atom_name(k).strip()
                 for k in range(1, pose.residue(bp[-1]).natoms() + 1)}
    check("SidechainConjugation" in pose.residue(bp[0]).name(),
          f"LYS carries the conjugation patch ({pose.residue(bp[0]).name()})")
    check(not {"1HZ", "2HZ", "3HZ"} & lys_atoms and "HZ" in lys_atoms,
          "LYS NZ lost its ammonium protons and gained one amide proton")
    check(not {"OD1", "OD2"} & asp_atoms and "OD" in asp_atoms,
          "ASP carboxylate became a single amide carbonyl oxygen")
    check(pose.residue(bp[0]).atom_type(pose.residue(bp[0]).atom_index("NZ")).name() == "Nbb",
          "NZ is retyped as a backbone-like amide nitrogen")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default=None, help="Staged complex for the Rosetta checks.")
    ap.add_argument("--binder-chain", default="B")
    args = ap.parse_args()

    test_isopeptide_atoms()
    test_anchor_costs()
    test_windows()
    if args.pdb:
        test_rosetta(args.pdb, args.binder_chain)
    else:
        print("\n(skipping Rosetta checks; pass --pdb to run them)")

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
