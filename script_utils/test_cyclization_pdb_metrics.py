#!/usr/bin/env python3
"""Unit tests for `proteinfoundation.evaluation.cyclization_pdb_metrics`.

Regression coverage for three confirmed bugs in the design-time, PDB-based
closure scorer (`compute_cyclization_metrics_single` / `_infer_chemistry`):

  1. Glutamate/glutamine isopeptide partners must bond through their carbonyl
     CD, not CG (GLU/GLN carry both atoms; a first-found search over
     (CG, CD) always resolved to the wrong one).
  2. The Lys/acid isopeptide orientation must work with the acid at EITHER
     terminus, not only "Lys first".
  3. A Cys-Cys or Lys-acid terminal residue PAIR must not, by identity alone,
     override an actually-closed mainchain bond as "disulfide"/"isopeptide".

Pure CPU tests: writes tiny synthetic PDB files to a temp dir, no GPU/model.

Usage (as a standalone script):
    python script_utils/test_cyclization_pdb_metrics.py

Usage (via pytest, if installed):
    pytest script_utils/test_cyclization_pdb_metrics.py -v
"""

from __future__ import annotations

import math
import os
import tempfile

from proteinfoundation.evaluation.cyclization_pdb_metrics import compute_cyclization_metrics_single

BINDER_CHAIN = "B"


def _atom_line(serial: int, atom_name: str, resname: str, chain: str, resseq: int, xyz) -> str:
    """Builds one ATOM record with exactly the fixed-column layout `_read_binder_residues` parses."""
    buf = [" "] * 80
    buf[0:6] = list("ATOM  ")
    buf[6:11] = list(f"{serial:>5}")
    buf[12:16] = list(f"{atom_name:<4}")
    buf[17:20] = list(f"{resname:>3}")
    buf[21] = chain
    buf[22:26] = list(f"{resseq:>4}")
    x, y, z = xyz
    buf[30:38] = list(f"{x:8.3f}")
    buf[38:46] = list(f"{y:8.3f}")
    buf[46:54] = list(f"{z:8.3f}")
    return "".join(buf) + "\n"


def _write_pdb(residues: list[tuple[str, int, dict[str, tuple[float, float, float]]]]) -> str:
    """`residues`: list of (resname, resseq, {atom_name: (x, y, z)}), in chain order."""
    lines = []
    serial = 1
    for resname, resseq, atoms in residues:
        for atom_name, xyz in atoms.items():
            lines.append(_atom_line(serial, atom_name, resname, BINDER_CHAIN, resseq, xyz))
            serial += 1
    fd, path = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as f:
        f.writelines(lines)
    return path


# ---------------------------------------------------------------------------
# A. GLU/GLN isopeptide must select CD, not CG (both are present in the residue).
# ---------------------------------------------------------------------------
def test_isopeptide_glu_selects_cd_not_cg():
    path = _write_pdb(
        [
            ("LYS", 1, {"NZ": (0.0, 0.0, 0.0)}),
            # GLU carries BOTH CG (far, non-bonding) and CD (the real, close carbonyl).
            ("GLU", 2, {"CG": (10.0, 0.0, 0.0), "CD": (1.32, 0.0, 0.0)}),
        ]
    )
    try:
        out = compute_cyclization_metrics_single(path, BINDER_CHAIN, requested_type="isopeptide")
        assert math.isclose(out["binder_cyc_bond_dist_A"], 1.32, abs_tol=1e-3), out
        assert out["binder_cyc_bond_closed"] is True
        assert out["binder_cyc_type_satisfied"] is True
    finally:
        os.remove(path)


def test_isopeptide_gln_selects_cd_not_cg():
    path = _write_pdb(
        [
            ("LYS", 1, {"NZ": (0.0, 0.0, 0.0)}),
            ("GLN", 2, {"CG": (10.0, 0.0, 0.0), "CD": (1.32, 0.0, 0.0)}),
        ]
    )
    try:
        out = compute_cyclization_metrics_single(path, BINDER_CHAIN, requested_type="isopeptide")
        assert math.isclose(out["binder_cyc_bond_dist_A"], 1.32, abs_tol=1e-3), out
        assert out["binder_cyc_bond_closed"] is True
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# B. Reverse Lys/acid orientation (acid first, Lys last) must also be scored.
# ---------------------------------------------------------------------------
def test_isopeptide_reverse_orientation_acid_first():
    path = _write_pdb(
        [
            ("ASP", 1, {"CG": (0.0, 0.0, 0.0)}),
            ("LYS", 2, {"NZ": (1.32, 0.0, 0.0)}),
        ]
    )
    try:
        out = compute_cyclization_metrics_single(path, BINDER_CHAIN, requested_type="isopeptide")
        assert math.isclose(out["binder_cyc_bond_dist_A"], 1.32, abs_tol=1e-3), out
        assert out["binder_cyc_bond_closed"] is True
        assert out["binder_cyc_type_satisfied"] is True
    finally:
        os.remove(path)


def test_isopeptide_reverse_orientation_with_asn():
    path = _write_pdb(
        [
            ("ASN", 1, {"CG": (0.0, 0.0, 0.0)}),
            ("LYS", 2, {"NZ": (1.32, 0.0, 0.0)}),
        ]
    )
    try:
        out = compute_cyclization_metrics_single(path, BINDER_CHAIN, requested_type="isopeptide")
        assert math.isclose(out["binder_cyc_bond_dist_A"], 1.32, abs_tol=1e-3), out
        assert out["binder_cyc_bond_closed"] is True
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# C. Terminal residue identity must not override an actually-closed mainchain bond.
# ---------------------------------------------------------------------------
def test_cys_cys_termini_with_closed_mainchain_is_not_misreported_as_disulfide():
    path = _write_pdb(
        [
            ("CYS", 1, {"N": (0.0, 0.0, 0.0), "SG": (20.0, 0.0, 0.0)}),
            ("CYS", 2, {"C": (1.33, 0.0, 0.0), "SG": (20.0, 0.0, 3.0)}),
        ]
    )
    try:
        out = compute_cyclization_metrics_single(path, BINDER_CHAIN, requested_type=None)
        assert out["binder_cyc_type_observed"] == "mainchain", out
        assert math.isclose(out["binder_cyc_bond_dist_A"], 1.33, abs_tol=1e-3)
        assert out["binder_cyc_bond_closed"] is True
    finally:
        os.remove(path)


def test_lys_asp_termini_with_closed_mainchain_is_not_misreported_as_isopeptide():
    path = _write_pdb(
        [
            ("LYS", 1, {"N": (0.0, 0.0, 0.0), "NZ": (20.0, 0.0, 0.0)}),
            ("ASP", 2, {"C": (1.33, 0.0, 0.0), "CG": (20.0, 0.0, 3.0)}),
        ]
    )
    try:
        out = compute_cyclization_metrics_single(path, BINDER_CHAIN, requested_type=None)
        assert out["binder_cyc_type_observed"] == "mainchain", out
        assert math.isclose(out["binder_cyc_bond_dist_A"], 1.33, abs_tol=1e-3)
        assert out["binder_cyc_bond_closed"] is True
    finally:
        os.remove(path)


def test_cys_cys_termini_with_actual_disulfide_still_detected():
    """Sanity check: the fix must not break the ordinary, actually-bonded case."""
    path = _write_pdb(
        [
            ("CYS", 1, {"N": (0.0, 0.0, 0.0), "SG": (0.0, 0.0, 0.0)}),
            ("CYS", 2, {"C": (50.0, 0.0, 0.0), "SG": (2.05, 0.0, 0.0)}),
        ]
    )
    try:
        out = compute_cyclization_metrics_single(path, BINDER_CHAIN, requested_type=None)
        assert out["binder_cyc_type_observed"] == "disulfide", out
        assert math.isclose(out["binder_cyc_bond_dist_A"], 2.05, abs_tol=1e-3)
    finally:
        os.remove(path)


ALL_TESTS = [
    test_isopeptide_glu_selects_cd_not_cg,
    test_isopeptide_gln_selects_cd_not_cg,
    test_isopeptide_reverse_orientation_acid_first,
    test_isopeptide_reverse_orientation_with_asn,
    test_cys_cys_termini_with_closed_mainchain_is_not_misreported_as_disulfide,
    test_lys_asp_termini_with_closed_mainchain_is_not_misreported_as_isopeptide,
    test_cys_cys_termini_with_actual_disulfide_still_detected,
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
