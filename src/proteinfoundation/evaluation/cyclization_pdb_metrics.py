"""Reference-free macrocycle closure metrics, measured on a generated complex PDB.

The design pipeline is otherwise cyclization-blind: ProteinMPNN redesigns the binder as a
linear chain and AF2 refolds it as one, so `pLDDT` / `i_pAE` / `scRMSD` are all computed on
a molecule that has no closing bond. None of them can see whether the ring the model was
asked for actually formed -- which, for a macrocycle, is the whole point of the molecule.

This module reads the closing bond straight off the GENERATED binder chain, which does
carry the ring. It is reference-free: there is no ground-truth binder to compare against in
a design run, so every metric here scores the structure on its own terms ("is this bond at
a chemically valid length"), never against a native.

The bond windows are imported from `eval.cyclic_reconstruction_metrics` rather than
restated, so the number reported for a design run means the same thing as the
`*_bond_success` logged during training. Two definitions of "closed" that drift apart is a
bug that would be nearly impossible to see in aggregate.

Chemistry, matching `cyclization.constants`:
  mainchain  -- head-to-tail amide: N of residue i (first) <-> C of residue j (last)
  disulfide  -- CYS SG                                     <-> CYS SG
  isopeptide -- LYS NZ (either i or j)                     <-> ASP/ASN CG or GLU/GLN CD
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from proteinfoundation.cyclization.constants import ISOPEPTIDE_ACID_CARBON_ATOM
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    DISULFIDE_SG_BOND_WINDOW_A,
    ISOPEPTIDE_N_C_BOND_WINDOW_A,
    MAINCHAIN_CN_BOND_WINDOW_A,
)

CYCLIZATION_METRIC_COLS = [
    "binder_cyc_type_observed",
    "binder_cyc_bond_dist_A",
    "binder_cyc_bond_closed",
    "binder_cyc_bond_fused",
    "binder_cyc_type_requested",
    "binder_cyc_type_satisfied",
]

# Below this, the two anchor atoms are not bonded but interpenetrating -- no covalent bond
# is anywhere near this short. Tracked separately from "not closed" because it is a
# distinct failure with a distinct cause (a closure term pulling with no lower bound), and
# because it is invisible to any one-sided proximity test.
FUSED_BOND_DIST_A = 1.0

_WINDOWS = {
    "mainchain": MAINCHAIN_CN_BOND_WINDOW_A,
    "disulfide": DISULFIDE_SG_BOND_WINDOW_A,
    "isopeptide": ISOPEPTIDE_N_C_BOND_WINDOW_A,
}

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _mainchain_anchor(atoms_i, atoms_j):
    """N of the first (i) residue <-> C of the last (j) residue -- the only real
    head-to-tail closing bond (the free N-terminus amine to the free C-terminus
    carbonyl). See `eval.cyclic_reconstruction_metrics.per_sample_requested_bond_distance`,
    which measures the identical pair on the tensor side.
    """
    return atoms_i.get("N"), atoms_j.get("C")


def _disulfide_anchor(res_i, res_j, atoms_i, atoms_j):
    """SG <-> SG, only if both residues are actually CYS."""
    if res_i == "CYS" and res_j == "CYS":
        return atoms_i.get("SG"), atoms_j.get("SG")
    return None, None


def _isopeptide_anchor(res_i, res_j, atoms_i, atoms_j):
    """LYS(NZ) <-> ASP/ASN(CG) or GLU/GLN(CD), in EITHER i/j order.

    The bonding carbon is picked by the acid/amide residue's own identity
    (`ISOPEPTIDE_ACID_CARBON_ATOM`), never by "whichever of CG/CD happens to be
    present" -- GLU/GLN carry BOTH a CG (an ordinary, non-bonding side-chain
    carbon) and a CD (the real carbonyl); a first-found search over (CG, CD)
    silently always resolves to the wrong atom (CG) for those two. Returns
    (None, None) if neither orientation matches (no LYS, or its partner isn't
    one of the accepted acid/amide residues).
    """
    if res_i == "LYS" and res_j in ISOPEPTIDE_ACID_CARBON_ATOM:
        return atoms_i.get("NZ"), atoms_j.get(ISOPEPTIDE_ACID_CARBON_ATOM[res_j])
    if res_j == "LYS" and res_i in ISOPEPTIDE_ACID_CARBON_ATOM:
        return atoms_i.get(ISOPEPTIDE_ACID_CARBON_ATOM[res_i]), atoms_j.get("NZ")
    return None, None


_ANCHOR_FUNCS = {
    "mainchain": lambda res_i, res_j, atoms_i, atoms_j: _mainchain_anchor(atoms_i, atoms_j),
    "disulfide": _disulfide_anchor,
    "isopeptide": _isopeptide_anchor,
}


def _read_binder_residues(pdb_path: str, binder_chain: str) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Parse the binder chain into ordered (resname, {atom_name: xyz}) records."""
    residues: dict[int, tuple[str, dict[str, np.ndarray]]] = {}
    order: list[int] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")) or line[21] != binder_chain:
                continue
            try:
                resi = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            if resi not in residues:
                residues[resi] = (line[17:20].strip(), {})
                order.append(resi)
            residues[resi][1][line[12:16].strip()] = xyz
    return [residues[i] for i in sorted(order)]


def _infer_chemistry(
    res_i: str,
    res_j: str,
    atoms_i: dict[str, np.ndarray],
    atoms_j: dict[str, np.ndarray],
) -> str | None:
    """The macrocycle chemistry implied by the binder's two terminal residues, if any.

    A design run has no cyclization label, so the chemistry is read off what the model
    actually built. This is an OBSERVATION, not a check that it obeyed a request --
    `binder_cyc_type_satisfied` is what compares it against what was asked for.

    Determined from each eligible chemistry's REAL anchor-atom distance, not from
    residue identity alone: two termini that happen to be Cys/Cys, or Lys/Asp, do not
    by themselves mean their side chains are bonded -- the actual ring closure might
    be an ordinary backbone amide with those side chains simply sitting near each
    other, unbonded. Preferring disulfide/isopeptide over mainchain purely on residue
    identity (the previous behavior) let exactly that coincidence override a real,
    closed mainchain bond. Reports whichever eligible chemistry's anchor atoms are
    actually within their own bonding window; if more than one are (a pathological,
    not physically expected, structure), the tightest one wins.
    """
    candidates: list[tuple[str, float]] = []
    for chem, anchor_fn in _ANCHOR_FUNCS.items():
        atom_i, atom_j = anchor_fn(res_i, res_j, atoms_i, atoms_j)
        if atom_i is None or atom_j is None:
            continue
        dist = float(np.linalg.norm(atom_i - atom_j))
        lo, hi = _WINDOWS[chem]
        if lo <= dist <= hi:
            candidates.append((chem, dist))

    if not candidates:
        return None
    return min(candidates, key=lambda cd: cd[1])[0]


def compute_cyclization_metrics_single(
    pdb_path: str,
    binder_chain: str,
    requested_type: str | None = None,
) -> dict[str, object]:
    """Closure metrics for the macrocycle on one generated complex.

    Args:
        pdb_path: Complex PDB (generated, pre-refolding -- a refold has no ring).
        binder_chain: Binder chain ID.
        requested_type: The cyclization type the model was conditioned on, if any.
            When given, `binder_cyc_type_satisfied` reports whether the model complied.

    Returns:
        Dict keyed by `CYCLIZATION_METRIC_COLS`. NaN where not determinable.
    """
    out: dict[str, object] = dict.fromkeys(CYCLIZATION_METRIC_COLS, np.nan)
    out["binder_cyc_type_requested"] = requested_type if requested_type else np.nan

    try:
        residues = _read_binder_residues(pdb_path, binder_chain)
    except OSError as e:
        logger.error(f"Cyclization metrics failed to read {pdb_path}: {e}")
        return out

    if len(residues) < 2:
        return out

    (name_i, atoms_i), (name_j, atoms_j) = residues[0], residues[-1]

    # Score the bond the model was ASKED for when we know it; a model that was told
    # "disulfide" and produced a closed isopeptide has not done what it was told, and
    # scoring its isopeptide would report that failure as a success. With no request,
    # fall back to whatever chemistry its terminal residues imply.
    observed = _infer_chemistry(name_i, name_j, atoms_i, atoms_j)
    chem = requested_type if requested_type in _WINDOWS else observed
    out["binder_cyc_type_observed"] = observed if observed else np.nan

    if chem not in _WINDOWS:
        if requested_type in _WINDOWS:
            out["binder_cyc_type_satisfied"] = False
        return out

    lo, hi = _WINDOWS[chem]
    atom_i, atom_j = _ANCHOR_FUNCS[chem](name_i, name_j, atoms_i, atoms_j)

    if atom_i is None or atom_j is None:
        if requested_type in _WINDOWS:
            out["binder_cyc_type_satisfied"] = False
        return out

    dist = float(np.linalg.norm(atom_i - atom_j))
    out["binder_cyc_bond_dist_A"] = round(dist, 3)
    out["binder_cyc_bond_closed"] = bool(lo <= dist <= hi)
    out["binder_cyc_bond_fused"] = bool(dist < FUSED_BOND_DIST_A)

    if requested_type in _WINDOWS:
        if requested_type == "mainchain":
            # Mainchain constrains no side chain, so no residue signature can witness it --
            # every peptide has an N-terminus and a C-terminus. A closed N-C bond IS the only
            # evidence the model built the ring it was asked for. (Checking residue identity
            # here would call a wide-open backbone with a stray Lys/Asp pair a success.)
            out["binder_cyc_type_satisfied"] = out["binder_cyc_bond_closed"]
        else:
            # Right chemistry AND actually bonded. Either alone is not the requested molecule:
            # two cysteines 6 A apart are not a disulfide.
            out["binder_cyc_type_satisfied"] = bool(observed == requested_type) and out["binder_cyc_bond_closed"]
    return out
