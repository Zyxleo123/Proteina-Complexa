"""Write a cyclic peptide whose ring is DECLARED, and verify it by reading the file back.

Why this exists
---------------
A macrocycle written as plain ATOM records is, to every downstream consumer, a linear
peptide whose ends happen to be near each other. Rosetta will not see a covalent ring,
`cyclization.parse_labels` will not find one, and the acceptance check would be grading a
claim the file does not make. The earlier evaluation on this project never measured
cyclization at all, which is how ten runs came to score 0.0 for reasons that had nothing to
do with the model.

So the ring bond is written as a CONECT record, and `verify_ring` reads the file back and
confirms it: the record exists, it names the two atoms the requested chemistry calls for,
and the distance between them in the file is inside the acceptance window. Asserting that
from the tensors would prove nothing about what landed on disk.

Anchor atoms follow `per_sample_requested_bond_distance` exactly -- N(i)<->C(j) for
mainchain with i < j, SG<->SG for disulfide, LYS:NZ<->ASP/ASN:CG or GLU/GLN:CD for
isopeptide, including ASN/GLN as acceptors because `parse_labels` accepts them. A different
choice here would mean the file declares a bond the metric is not measuring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openfold.np import residue_constants

from proteinfoundation.cyclization.constants import (
    AA_ASN,
    AA_ASP,
    AA_CYS,
    AA_GLN,
    AA_GLU,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
)
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    DISULFIDE_SG_BOND_WINDOW_A,
    ISOPEPTIDE_N_C_BOND_WINDOW_A,
    MAINCHAIN_CN_BOND_WINDOW_A,
)

BOND_WINDOW_A = {
    MAINCHAIN: MAINCHAIN_CN_BOND_WINDOW_A,
    DISULFIDE: DISULFIDE_SG_BOND_WINDOW_A,
    ISOPEPTIDE: ISOPEPTIDE_N_C_BOND_WINDOW_A,
}
_ACID_ATOM = {AA_ASP: "CG", AA_ASN: "CG", AA_GLU: "CD", AA_GLN: "CD"}


class RingUndeclarable(RuntimeError):
    """The requested ring cannot be named on this sequence -- e.g. a disulfide asked for
    where an endpoint did not decode as cysteine. Raised rather than silently writing a
    ring-free PDB, because a missing CONECT is indistinguishable from a linear design."""


def ring_atom_names(cyc_type: int, aa_i: int, aa_j: int) -> tuple[str, str]:
    """(atom at endpoint i, atom at endpoint j) for the requested chemistry."""
    cyc_type = int(cyc_type)
    if cyc_type == MAINCHAIN:
        return "N", "C"
    if cyc_type == DISULFIDE:
        if not (aa_i == AA_CYS and aa_j == AA_CYS):
            raise RingUndeclarable(
                f"disulfide needs CYS at both endpoints; decoded {aa_i}/{aa_j}"
            )
        return "SG", "SG"
    if cyc_type == ISOPEPTIDE:
        if aa_i == AA_LYS and aa_j in _ACID_ATOM:
            return "NZ", _ACID_ATOM[aa_j]
        if aa_j == AA_LYS and aa_i in _ACID_ATOM:
            return _ACID_ATOM[aa_i], "NZ"
        raise RingUndeclarable(
            f"isopeptide needs LYS at one endpoint and ASP/ASN/GLU/GLN at the other; "
            f"decoded {aa_i}/{aa_j}"
        )
    raise RingUndeclarable(f"unknown cyclization type {cyc_type}")


def _atom_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]


def _find_serial(lines: list[str], resnum: int, atom_name: str, chain: str | None = None) -> int:
    """PDB serial of `atom_name` in residue `resnum`. Raises if absent or ambiguous."""
    hits = [
        int(ln[6:11])
        for ln in lines
        if int(ln[22:26]) == resnum
        and ln[12:16].strip() == atom_name
        and (chain is None or ln[21] == chain)
    ]
    if len(hits) != 1:
        raise RingUndeclarable(
            f"expected exactly one {atom_name} in residue {resnum}"
            f"{'' if chain is None else ' chain ' + chain}, found {len(hits)}"
        )
    return hits[0]


def _xyz(lines: list[str], serial: int) -> np.ndarray:
    for ln in lines:
        if int(ln[6:11]) == serial:
            return np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    raise RingUndeclarable(f"serial {serial} not found")


def append_conect(path: Path | str, serial_a: int, serial_b: int) -> None:
    """Add one CONECT pair before END, leaving everything else byte-identical.

    Both directions are written. Some parsers treat CONECT as directed and only register
    the bond from the record's first atom, so a single line can leave the ring half
    declared -- which reads downstream as a missing bond rather than as a format quirk.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    body = [ln for ln in lines if ln.strip() != "END"]
    body.append(f"CONECT{serial_a:5d}{serial_b:5d}")
    body.append(f"CONECT{serial_b:5d}{serial_a:5d}")
    body.append("END")
    path.write_text("\n".join(body) + "\n")


def declare_ring(path: Path | str, res_i: int, res_j: int, cyc_type: int,
                 aa_i: int, aa_j: int, chain: str | None = None) -> dict:
    """Append the ring CONECT to an already-written PDB. `res_i`/`res_j` are PDB resSeq."""
    path = Path(path)
    lines = _atom_lines(path.read_text())
    name_i, name_j = ring_atom_names(cyc_type, aa_i, aa_j)
    sa = _find_serial(lines, res_i, name_i, chain)
    sb = _find_serial(lines, res_j, name_j, chain)
    append_conect(path, sa, sb)
    return {"serial_i": sa, "serial_j": sb, "atom_i": name_i, "atom_j": name_j}


def verify_ring(path: Path | str, res_i: int, res_j: int, cyc_type: int,
                chain: str | None = None) -> dict:
    """Read the file back and check the ring is declared, named right, and in window.

    Returns a dict with `declared` (a CONECT links the two anchor atoms), `dist_A` (as
    measured in the FILE, not in the tensors) and `in_window`. Never raises for a missing
    ring -- an undeclared ring is a result, and the caller records it as a failed gate.
    """
    path = Path(path)
    text = path.read_text()
    lines = _atom_lines(text)
    out: dict = {"declared": False, "dist_A": None, "in_window": False, "reason": None}

    aa3_by_res = {}
    for ln in lines:
        if chain is None or ln[21] == chain:
            aa3_by_res[int(ln[22:26])] = ln[17:20].strip()
    try:
        aa_i = residue_constants.restype_order[residue_constants.restype_3to1[aa3_by_res[res_i]]]
        aa_j = residue_constants.restype_order[residue_constants.restype_3to1[aa3_by_res[res_j]]]
        name_i, name_j = ring_atom_names(cyc_type, aa_i, aa_j)
        sa = _find_serial(lines, res_i, name_i, chain)
        sb = _find_serial(lines, res_j, name_j, chain)
    except (KeyError, RingUndeclarable) as exc:
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out

    bonded = set()
    for ln in text.splitlines():
        if not ln.startswith("CONECT"):
            continue
        fields = [ln[i : i + 5] for i in range(6, min(len(ln), 31), 5)]
        nums = [int(f) for f in fields if f.strip()]
        for other in nums[1:]:
            bonded.add((nums[0], other))
    out["declared"] = (sa, sb) in bonded or (sb, sa) in bonded
    lo, hi = BOND_WINDOW_A[int(cyc_type)]
    d = float(np.linalg.norm(_xyz(lines, sa) - _xyz(lines, sb)))
    out["dist_A"] = d
    out["in_window"] = bool(lo <= d <= hi)
    out["atom_i"], out["atom_j"] = name_i, name_j
    if not out["declared"]:
        out["reason"] = "no CONECT links the two anchor atoms"
    return out


def peptide_serial_offset(complex_path: Path | str, binder_chain: str) -> int:
    """Serial of the binder chain's first atom in a written complex, minus 1.

    `complex_frame.write_complex` renumbers every serial sequentially, receptor first, so a
    peptide atom's complex serial is this offset plus its 1-based index among the peptide's
    own atom lines. Recomputing it from the file rather than from the receptor line count
    keeps this correct if the writer's TER/END handling ever changes.
    """
    lines = _atom_lines(Path(complex_path).read_text())
    for ln in lines:
        if ln[21] == binder_chain:
            return int(ln[6:11]) - 1
    raise RingUndeclarable(f"no atoms on binder chain {binder_chain!r} in {complex_path}")
