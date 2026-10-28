#!/usr/bin/env python3
"""Checks for the geometric buried-unsatisfied detector. No GPU, no structures on disk.

The rules this pins are the ones that decide whether the detector reports a covalent
bond, a donor-donor pair, or a peptide's own buried core as an interface defect. Each of
those would produce a number that looks like a measurement and is an artifact, and each
is a one-line mistake to make, so each gets a test with hand-placed coordinates where
the right answer is known by construction.

Burial is injected rather than built: enclosing an atom enough to drive its
Shrake-Rupley area under the threshold takes a shell of neighbours whose placement would
itself be the thing under test. So the partner geometry is tested against explicit
coordinates and the burial decision against assigned areas, separately.

Run:  python script_utils/test_cpseq_unsat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.unsat import (  # noqa: E402
    BURIED_SASA_MAX_A2,
    DESOLVATED_MIN_A2,
    HBOND_HEAVY_MAX_A,
    HBOND_HEAVY_MIN_A,
    PolarAtom,
    _count_partners,
    atom_roles,
    render_unsat,
)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))


def _atom(chain, resname, resnum, name, pos=None, idx=0, **kw) -> PolarAtom:
    is_d, is_a = atom_roles(resname, name)
    return PolarAtom(
        chain=chain,
        resname=resname,
        resnum=resnum,
        atom=name,
        pos=pos,
        is_donor=is_d,
        is_acceptor=is_a,
        is_backbone=name in ("N", "O", "OXT"),
        index=idx,
        **kw,
    )


def _partners(atoms, xyz, *, cyclic=None, lengths=None):
    counts, labels = _count_partners(
        atoms,
        np.asarray(xyz, dtype=np.float64),
        subset=None,
        cyclic_chain=cyclic,
        chain_length=lengths or {},
        record_labels=True,
    )
    return counts, labels


# --------------------------------------------------------------------------------------


def test_typing() -> None:
    print("\natom typing")
    check("backbone N donates", atom_roles("ALA", "N") == (True, False))
    check("proline's backbone N has no hydrogen to donate",
          atom_roles("PRO", "N") == (False, False))
    check("backbone O accepts", atom_roles("ALA", "O") == (False, True))
    check("the C-terminal OXT accepts", atom_roles("ALA", "OXT") == (False, True))
    check("Lys NZ donates only", atom_roles("LYS", "NZ") == (True, False))
    check("Asp OD1 accepts only", atom_roles("ASP", "OD1") == (False, True))
    check("Ser OG does both", atom_roles("SER", "OG") == (True, True))
    check("His ND1 does both", atom_roles("HIS", "ND1") == (True, True))
    check("Trp NE1 donates only", atom_roles("TRP", "NE1") == (True, False))
    check("Asn ND2 donates, OD1 accepts",
          atom_roles("ASN", "ND2") == (True, False)
          and atom_roles("ASN", "OD1") == (False, True))
    # Variants relax leaves behind must type as the residue they are a form of.
    check("HIE types as histidine", atom_roles("HIE", "ND1") == (True, True))
    check("ASH types as aspartate", atom_roles("ASH", "OD1") == (False, True))
    check("a carbon is neither", atom_roles("ALA", "CB") == (False, False))
    check("an unknown residue's side-chain nitrogen is untyped",
          atom_roles("XYZ", "NX1") == (False, False))


def test_complementarity() -> None:
    print("\ncomplementary roles")
    # A peptide lysine against a receptor aspartate: donor to acceptor, a real pair.
    atoms = [
        _atom("B", "LYS", 7, "NZ", pos=7, idx=0),
        _atom("A", "ASP", 112, "OD1", idx=1),
    ]
    counts, labels = _partners(atoms, [[0, 0, 0], [2.9, 0, 0]])
    check("donor and acceptor at 2.9 A pair up", counts == [1, 1], f"{counts}")
    check("the partner label carries the distance",
          "A:Asp112:OD1 2.90A" in labels[0], f"{labels[0]}")

    # Two lysines cannot satisfy each other, however close.
    atoms = [
        _atom("B", "LYS", 7, "NZ", pos=7, idx=0),
        _atom("A", "LYS", 44, "NZ", idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.9, 0, 0]])
    check("donor against donor is not a hydrogen bond", counts == [0, 0], f"{counts}")

    # Nor can two carbonyls.
    atoms = [
        _atom("B", "ALA", 7, "O", pos=7, idx=0),
        _atom("A", "ALA", 44, "O", idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [3.1, 0, 0]])
    check("acceptor against acceptor is not a hydrogen bond", counts == [0, 0], f"{counts}")

    # A hydroxyl is both, so it pairs with either.
    atoms = [
        _atom("B", "SER", 7, "OG", pos=7, idx=0),
        _atom("A", "SER", 44, "OG", idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.8, 0, 0]])
    check("two hydroxyls pair", counts == [1, 1], f"{counts}")


def test_distance_window() -> None:
    print("\ndistance window")
    pair = [
        _atom("B", "LYS", 7, "NZ", pos=7, idx=0),
        _atom("A", "ASP", 112, "OD1", idx=1),
    ]
    counts, _ = _partners(pair, [[0, 0, 0], [HBOND_HEAVY_MAX_A + 0.2, 0, 0]])
    check("beyond the cutoff, no pair", counts == [0, 0], f"{counts}")
    counts, _ = _partners(pair, [[0, 0, 0], [HBOND_HEAVY_MIN_A - 0.3, 0, 0]])
    check("closer than any hydrogen bond reaches, no pair", counts == [0, 0], f"{counts}")
    counts, _ = _partners(pair, [[0, 0, 0], [3.0, 0, 0]])
    check("inside the window, a pair", counts == [1, 1], f"{counts}")


def test_same_residue() -> None:
    print("\nwithin one residue")
    # Asn's own ND2 and OD1 are 2.25 A apart and cannot hydrogen bond each other.
    atoms = [
        _atom("B", "ASN", 7, "ND2", pos=7, idx=0),
        _atom("B", "ASN", 7, "OD1", pos=7, idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.25, 0, 0]])
    check("one residue's own donor and acceptor do not pair", counts == [0, 0], f"{counts}")


def test_backbone_adjacency() -> None:
    print("\ncovalent adjacency, not hydrogen bonds")
    # N(i) and O(i-1) both hang off C(i-1): ~2.25 A, and not a hydrogen bond.
    atoms = [
        _atom("B", "ALA", 6, "O", pos=6, idx=0),
        _atom("B", "LYS", 7, "N", pos=7, idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.25, 0, 0]], lengths={"B": 14})
    check("sequence-adjacent backbone N and O are excluded",
          counts == [0, 0], f"{counts}")

    # The same two atoms three residues apart IS the helical hydrogen bond, and must
    # survive -- an exclusion written on backbone atoms alone would delete it.
    atoms = [
        _atom("B", "ALA", 4, "O", pos=4, idx=0),
        _atom("B", "LYS", 8, "N", pos=8, idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.95, 0, 0]], lengths={"B": 14})
    check("an i,i+4 backbone hydrogen bond is kept", counts == [1, 1], f"{counts}")

    # A side chain on the next residue may legitimately bond the previous backbone.
    atoms = [
        _atom("B", "ALA", 6, "O", pos=6, idx=0),
        _atom("B", "SER", 7, "OG", pos=7, idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.8, 0, 0]], lengths={"B": 14})
    check("a next-residue side chain to backbone bond is kept",
          counts == [1, 1], f"{counts}")


def test_cyclic_closure() -> None:
    print("\nthe closing amide of a cycle")
    # Head-to-tail: residue 1's N and residue 14's O are bonded through the closing
    # amide, exactly as 7 and 6 are, and the PDB usually carries no CONECT to say so.
    #
    # Placed at 2.9 A, not at the 2.25 A an ideal amide holds, so that what is under test
    # is the cyclic exclusion and not the distance floor -- which already rejects the
    # ideal geometry on its own, and would make this check pass without the rule.
    atoms = [
        _atom("B", "ARG", 1, "N", pos=1, idx=0),
        _atom("B", "ASN", 14, "O", pos=14, idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.9, 0, 0]], cyclic="B", lengths={"B": 14})
    check("the 1<->L closing amide is excluded when cyclic",
          counts == [0, 0], f"{counts}")
    counts, _ = _partners(atoms, [[0, 0, 0], [2.9, 0, 0]], cyclic=None, lengths={"B": 14})
    check("and is counted when the chain is not declared cyclic",
          counts == [1, 1], f"{counts}")
    # A cyclic peptide's 1<->L SIDE-CHAIN contact is a real interaction, not the amide.
    atoms = [
        _atom("B", "SER", 1, "OG", pos=1, idx=0),
        _atom("B", "ASP", 14, "OD1", pos=14, idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.7, 0, 0]], cyclic="B", lengths={"B": 14})
    check("a 1<->L side-chain bond survives the cyclic exclusion",
          counts == [1, 1], f"{counts}")


def test_cross_chain_adjacency_is_not_excluded() -> None:
    print("\nadjacency is per chain")
    # Peptide residue 7 and receptor residue 8 share a number gap of 1 but no bond. An
    # exclusion that ignored the chain would silently delete real interface bonds.
    atoms = [
        _atom("B", "ALA", 7, "O", pos=7, idx=0),
        _atom("A", "LYS", 8, "N", idx=1),
    ]
    counts, _ = _partners(atoms, [[0, 0, 0], [2.9, 0, 0]], lengths={"A": 181, "B": 14})
    check("a numbering gap of 1 across chains is a real bond",
          counts == [1, 1], f"{counts}")


def test_burial_and_states() -> None:
    print("\nburied, unsatisfied, and whose fault")
    # Burial is a conjunction: little area left AND area lost on binding. The fixtures
    # carry both, and the two one-sided cases get their own checks below.
    buried = BURIED_SASA_MAX_A2 / 2
    exposed = BURIED_SASA_MAX_A2 + DESOLVATED_MIN_A2 + 20.0

    a = _atom("B", "LYS", 7, "NZ", pos=7, sasa_complex=buried, sasa_alone=exposed)
    check("desolvated, little area left, no partner: unsatisfied", a.unsat_complex)
    check("exposed when alone, so not unsatisfied there", not a.unsat_alone)
    check("and the interface caused it", a.unsat_complex and not a.unsat_alone)

    b = _atom("B", "LYS", 7, "NZ", pos=7, sasa_complex=buried, sasa_alone=buried)
    check("buried and partnerless in the peptide's own core is unsatisfied unbound",
          b.unsat_alone)
    check("but the interface did not cause it, so it is not newly unsatisfied",
          not (b.unsat_complex and not b.unsat_alone))

    c = _atom("B", "LYS", 7, "NZ", pos=7, sasa_complex=buried, sasa_alone=exposed)
    c.n_partners_complex = 1
    check("a buried atom with a partner is satisfied", not c.unsat_complex)

    d = _atom("B", "LYS", 7, "NZ", pos=7, sasa_complex=exposed, sasa_alone=exposed)
    check("an exposed atom is never unsatisfied, partner or not", not d.unsat_complex)

    e = _atom("B", "LYS", 7, "NZ", pos=7)
    check("an atom with no area computed is not reported as a defect",
          not e.unsat_complex and not e.unsat_alone)
    check("and its desolvation is None, not zero", e.dsasa is None)

    # The two one-sided failures, each of which the single-condition version got wrong.
    # Measured on lnr_2v8x: ARG13:NH1 loses 26 A^2 and keeps 24, so it still reaches
    # water; the helix backbone keeps 0 A^2 but lost nothing, having been enclosed all
    # along by the peptide's own fold.
    g = _atom("B", "ARG", 13, "NH1", pos=13, sasa_complex=24.1, sasa_alone=50.3)
    check("heavily desolvated but still hydrated is not a defect", not g.unsat_complex)
    check("though its desolvation is large", g.dsasa > 20.0, f"{g.dsasa}")
    h = _atom("B", "LEU", 9, "O", pos=9, sasa_complex=0.0, sasa_alone=0.0)
    check("enclosed by its own fold, losing nothing, is not an interface defect",
          not h.unsat_complex)

    # The thresholds are boundaries, and the documented sides of them are `<=` and `>=`.
    f = _atom("B", "LYS", 7, "NZ", pos=7,
              sasa_complex=BURIED_SASA_MAX_A2,
              sasa_alone=BURIED_SASA_MAX_A2 + DESOLVATED_MIN_A2)
    check("an atom exactly at both thresholds counts as buried", f.buried_complex)

    # What a substitution can and cannot repair.
    check("a peptide side-chain atom is fixable by substitution",
          _atom("B", "LYS", 7, "NZ", pos=7).fixable_by_substitution)
    check("a peptide backbone amide is not",
          not _atom("B", "LYS", 7, "N", pos=7).fixable_by_substitution)
    check("nor is a receptor atom",
          not _atom("A", "ASP", 112, "OD1").fixable_by_substitution)


def test_rendering() -> None:
    print("\nrendering")
    bad = render_unsat({"ok": False, "error": "pdb not found"})
    check("an unavailable report says so", "unavailable" in bad and "pdb not found" in bad)

    clean = render_unsat({
        "ok": True,
        "counts": {"peptide": {"unsat_complex": 0, "newly_unsat": 0},
                   "receptor": {"newly_unsat": 0}},
        "by_position": {},
    })
    check("a clean interface says every buried atom has a partner",
          "none:" in clean, clean)

    rep = {
        "ok": True,
        "counts": {"peptide": {"unsat_complex": 3, "newly_unsat": 2},
                   "receptor": {"newly_unsat": 1}},
        "by_position": {
            "7": [{"atom": "NZ", "backbone": False, "sasa_complex_A2": 0.1,
                   "newly_unsat": True,
                   "reachable": [{"atom": "A:Asp112:OD1", "dist_A": 4.9}]}],
            "9": [{"atom": "N", "backbone": True, "sasa_complex_A2": 0.0,
                   "newly_unsat": False, "reachable": []},
                  {"atom": "OG", "backbone": False, "sasa_complex_A2": 0.2,
                   "newly_unsat": True, "reachable": []}],
        },
    }
    txt = render_unsat(rep)
    check("the headline separates interface-caused from total",
          "peptide=3" in txt and "interface-caused 2" in txt, txt)
    check("a reachable partner is named with its distance",
          "A:Asp112:OD1 4.9A" in txt, txt)
    check("the side chain is listed before the backbone at the same position",
          txt.index("OG") < txt.index("N(bb)"), txt)
    check("a backbone atom is marked as backbone", "N(bb)" in txt, txt)
    check("an atom already unsatisfied unbound is marked, not blamed on the interface",
          "also unsat unbound" in txt, txt)

    trimmed = render_unsat(
        {"ok": True,
         "counts": {"peptide": {"unsat_complex": 5, "newly_unsat": 5},
                    "receptor": {"newly_unsat": 0}},
         "by_position": {str(i): [{"atom": "NZ", "backbone": False,
                                   "sasa_complex_A2": 0.1, "newly_unsat": True,
                                   "reachable": []}] for i in range(1, 9)}},
        max_positions=3,
    )
    check("a long list is trimmed and says how much was dropped",
          "+5 more positions" in trimmed, trimmed)


def main() -> int:
    print("geometric buried-unsatisfied detector checks")
    test_typing()
    test_complementarity()
    test_distance_window()
    test_same_residue()
    test_backbone_adjacency()
    test_cyclic_closure()
    test_cross_chain_adjacency_is_not_excluded()
    test_burial_and_states()
    test_rendering()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
