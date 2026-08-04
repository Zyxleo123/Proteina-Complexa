#!/usr/bin/env python3
"""Unit tests for `proteinfoundation.evaluation.rosetta_energy._declare_cyclic_bond`.

Uses real PyRosetta (available in this environment) to build small poses and
verify the closing bond is actually declared -- not a mock, since the point of
this fix is that specific, previously-unverified PyRosetta API calls
(`declare_chemical_bond` after stripping terminus variants, `add_disulfide_bond`)
behave the way the fix assumes.

Skips cleanly if PyRosetta is not importable.

Usage:
    python script_utils/test_rosetta_cyclic_bond.py
    pytest script_utils/test_rosetta_cyclic_bond.py -v
"""

from __future__ import annotations

import pytest

pyrosetta = pytest.importorskip("pyrosetta")

from proteinfoundation.evaluation.rosetta_energy import _declare_cyclic_bond  # noqa: E402

_INITIALIZED = False


def _ensure_init():
    global _INITIALIZED
    if not _INITIALIZED:
        pyrosetta.init("-mute all")
        _INITIALIZED = True


def _pose_with_pdb_info(sequence: str, chain: str = "B"):
    """A pose whose PDBInfo reports 1-based, per-chain-restarting resSeq numbering
    on `chain` -- matching `utils.pdb_utils.to_pdb`'s convention that
    `_declare_cyclic_bond` relies on."""
    _ensure_init()
    pose = pyrosetta.pose_from_sequence(sequence, "fa_standard")
    pdb_info = pyrosetta.rosetta.core.pose.PDBInfo(pose)
    for seqpos in range(1, pose.size() + 1):
        pdb_info.chain(seqpos, chain)
        pdb_info.number(seqpos, seqpos)
    pose.pdb_info(pdb_info)
    return pose


def _atoms_bonded(pose, seqpos_i, atom_i, seqpos_j, atom_j) -> bool:
    from pyrosetta.rosetta.core.id import AtomID

    return pose.conformation().atoms_are_bonded(
        AtomID(pose.residue(seqpos_i).atom_index(atom_i), seqpos_i),
        AtomID(pose.residue(seqpos_j).atom_index(atom_j), seqpos_j),
    )


def test_mainchain_bond_is_declared():
    pose = _pose_with_pdb_info("AAAAA")
    declared = _declare_cyclic_bond(pose, binder_chain="B", cyclization_type="mainchain", i_local=0, j_local=4)
    assert declared is True
    assert _atoms_bonded(pose, 1, "N", 5, "C")


def test_disulfide_bond_is_declared():
    pose = _pose_with_pdb_info("ACAAC")  # residues 2 and 5 are CYS
    declared = _declare_cyclic_bond(pose, binder_chain="B", cyclization_type="disulfide", i_local=1, j_local=4)
    assert declared is True
    assert _atoms_bonded(pose, 2, "SG", 5, "SG")
    assert "disulfide" in pose.residue(2).name()


def test_disulfide_bond_fails_loudly_when_not_cysteine():
    pose = _pose_with_pdb_info("AAAAA")
    with pytest.raises(ValueError):
        _declare_cyclic_bond(pose, binder_chain="B", cyclization_type="disulfide", i_local=0, j_local=4)


def test_isopeptide_returns_false_not_silently_scored():
    """Documented gap: isopeptide bond declaration is not implemented; the caller
    must be told so via a False return, not have the bond silently skipped."""
    pose = _pose_with_pdb_info("KAAAD")  # 1=LYS, 5=ASP
    declared = _declare_cyclic_bond(pose, binder_chain="B", cyclization_type="isopeptide", i_local=0, j_local=4)
    assert declared is False


def test_unknown_cyclization_type_raises():
    pose = _pose_with_pdb_info("AAAAA")
    with pytest.raises(ValueError):
        _declare_cyclic_bond(pose, binder_chain="B", cyclization_type="not_a_type", i_local=0, j_local=4)


def test_bad_endpoint_mapping_raises():
    pose = _pose_with_pdb_info("AAAAA")
    with pytest.raises(ValueError):
        _declare_cyclic_bond(pose, binder_chain="Z", cyclization_type="mainchain", i_local=0, j_local=4)


ALL_TESTS = [
    test_mainchain_bond_is_declared,
    test_disulfide_bond_is_declared,
    test_disulfide_bond_fails_loudly_when_not_cysteine,
    test_isopeptide_returns_false_not_silently_scored,
    test_unknown_cyclization_type_raises,
    test_bad_endpoint_mapping_raises,
]


if __name__ == "__main__":
    failures = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  OK {test_fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(test_fn.__name__)
            print(f"  FAIL {test_fn.__name__}: {e}")

    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
