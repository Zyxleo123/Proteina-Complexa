"""The acceptance contract for an LP -> CP edit. One place, fixed thresholds, no GPU.

Every number the FlowEdit and SDEdit arms report comes through here, so the two arms are
graded by identical code rather than by two similar-looking implementations. The thresholds
are the operational ones and are deliberately NOT tunable per arm: `CONTRACT` is a module
constant, and the sweep knobs (mutation budgets 0/1/2/3, identity 90/80/70%) are computed
as SEPARATE reporting columns rather than by moving the primary bar.

What this module deliberately does NOT do
-----------------------------------------
It does not accept C-beta proximity as closure. A Cb-Cb window is satisfied by two ends
merely lying near each other, which is what a peptide's ends do anyway -- measured here, the
`cyc_cb_window_success` metric saturates and separates nothing. Closure is judged on the
ACTUAL bond atoms of the requested chemistry, plus the anchor residue identities that
chemistry requires. A ring whose anchors are PRO/PRO is not an isopeptide however close the
C-betas are.

It also never superposes the peptide. Both RMSDs are computed in the receptor frame, because
an independent fit would rotate the binder back onto its input and hide exactly the movement
inside the binding site that the edit has to be priced for.

Contact definition
------------------
A peptide-receptor contact is a RESIDUE PAIR with any heavy-atom distance <= 4.5 A. This is
not the CA-CA proxy used elsewhere in the repo (`sdedit_cyclize.CONTACT_CUTOFF_NM`, 10 A
CA-CA): a CA-CA cutoff both misses side-chain-mediated contacts and invents contacts between
residues whose side chains point away from each other. atom37 carries no hydrogens, so
"heavy atom" is just "present atom".
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from openfold.np import residue_constants as rc

# Fixed operational thresholds. See the module docstring on why these are constants.
CONTRACT = {
    "seq_identity_min": 0.80,
    "max_substitutions": 2,
    "contact_retention_min": 0.80,
    "contact_cutoff_A": 4.5,
    "ca_rmsd_max_A": 2.0,
    "interface_ca_rmsd_max_A": 1.0,
    "ddg_max_REU": 5.0,
    # A non-bonded heavy-atom pair closer than this is a severe clash. 2.0 A sits below any
    # real non-bonded heavy-atom contact (the shortest, an N-O hydrogen bond, is ~2.6 A) and
    # well above the ~1.33-2.05 A of the covalent bonds we deliberately exclude.
    "clash_dist_A": 2.0,
    "max_severe_clashes": 0,
}

# Reporting-only sweeps, kept separate from the primary bar.
MUTATION_BUDGETS = (0, 1, 2, 3)
IDENTITY_THRESHOLDS = (0.90, 0.80, 0.70)

CA_IDX = rc.atom_order["CA"]

# Bond atoms of each chemistry, plus the residue identities that can host it and the
# distance window the bond has to fall in. Windows are generous relative to the ideal bond
# (mainchain/isopeptide amide ~1.33 A, disulfide S-S ~2.05 A) but far tighter than any
# Cb-proximity test.
_ACCEPTOR_ATOMS = {"ASP": "CG", "GLU": "CD", "ASN": "CG", "GLN": "CD"}

CHEMISTRY = {
    "mainchain": {
        "atoms": ("N", "C"),
        "window_A": (1.0, 1.8),
        "hosts": None,  # any residue can close head-to-tail
    },
    "disulfide": {
        "atoms": ("SG", "SG"),
        "window_A": (1.6, 2.6),
        "hosts": ({"CYS"}, {"CYS"}),
    },
    "isopeptide": {
        "atoms": ("NZ", None),  # acceptor atom depends on the acceptor residue
        "window_A": (1.0, 1.8),
        "hosts": ({"LYS"}, set(_ACCEPTOR_ATOMS)),
    },
}


def _restype_atom_mask(aatype: np.ndarray) -> np.ndarray:
    """[L, 37] bool: which atom37 slots exist for each residue's type."""
    return rc.restype_atom37_mask[np.clip(aatype, 0, 20)].astype(bool)


def _as_np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def resname(aatype_i: int) -> str:
    """3-letter name of an aatype index, 'UNK' out of range."""
    i = int(aatype_i)
    if not 0 <= i < len(rc.restypes):
        return "UNK"
    return rc.restype_1to3[rc.restypes[i]]


def heavy_atom_contacts(
    pep_xyz_A,
    pep_aatype,
    tgt_xyz_A,
    tgt_aatype,
    cutoff_A: float = CONTRACT["contact_cutoff_A"],
) -> set[tuple[int, int]]:
    """Residue pairs (peptide_idx, receptor_idx) with any heavy-atom distance <= cutoff.

    Args:
        pep_xyz_A: [Lp, 37, 3] peptide atom37 coordinates in ANGSTROM.
        pep_aatype: [Lp] integer residue types (used to know which slots are real atoms).
        tgt_xyz_A: [Lt, 37, 3] receptor atom37 coordinates in Angstrom.
        tgt_aatype: [Lt] integer residue types.

    Coordinates must already be in a shared frame; nothing here aligns anything.
    """
    p_xyz, t_xyz = _as_np(pep_xyz_A), _as_np(tgt_xyz_A)
    p_m = _restype_atom_mask(_as_np(pep_aatype))
    t_m = _restype_atom_mask(_as_np(tgt_aatype))
    # Flatten to atom lists, carrying each atom's residue index.
    p_idx, p_slot = np.nonzero(p_m)
    t_idx, t_slot = np.nonzero(t_m)
    if len(p_idx) == 0 or len(t_idx) == 0:
        return set()
    pa = p_xyz[p_idx, p_slot]  # [Na, 3]
    ta = t_xyz[t_idx, t_slot]  # [Nb, 3]
    # Chunked to keep the pairwise block bounded; peptides are short but receptors are not.
    hits: set[tuple[int, int]] = set()
    c2 = float(cutoff_A) ** 2
    chunk = 4096
    for s in range(0, len(ta), chunk):
        tb, tb_idx = ta[s : s + chunk], t_idx[s : s + chunk]
        d2 = ((pa[:, None, :] - tb[None, :, :]) ** 2).sum(-1)
        ii, jj = np.nonzero(d2 <= c2)
        hits.update(zip(p_idx[ii].tolist(), tb_idx[jj].tolist()))
    return hits


def contact_retention(before: set, after: set) -> float:
    """Fraction of the ORIGINAL contacts still present. NaN when the input had none."""
    if not before:
        return float("nan")
    return len(before & after) / len(before)


def interface_residues(contacts: Iterable[tuple[int, int]]) -> set[int]:
    """Peptide residue indices participating in at least one contact."""
    return {int(i) for i, _ in contacts}


def ca_rmsd_no_align(a_xyz_A, b_xyz_A, subset: Iterable[int] | None = None) -> float:
    """CA RMSD in the SHARED frame -- no superposition. See the module docstring.

    Args:
        a_xyz_A, b_xyz_A: [L, 37, 3] in Angstrom.
        subset: peptide residue indices to restrict to (e.g. the original interface).
    """
    a, b = _as_np(a_xyz_A)[:, CA_IDX], _as_np(b_xyz_A)[:, CA_IDX]
    if subset is not None:
        idx = sorted(set(int(i) for i in subset))
        if not idx:
            return float("nan")
        a, b = a[idx], b[idx]
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def sequence_budget(
    aatype_before,
    aatype_after,
    protected: Iterable[int] = (),
) -> dict:
    """Identity, substitution count and protected-residue integrity.

    Substitutions are counted over the whole peptide and INCLUDE anchor mutations -- an
    anchor change is exactly the kind of edit a budget is meant to price, so exempting it
    would make the budget unfalsifiable.
    """
    a, b = _as_np(aatype_before).astype(int), _as_np(aatype_after).astype(int)
    if a.shape != b.shape:
        raise ValueError(
            f"sequence length changed ({a.shape} -> {b.shape}); the primary comparison "
            "holds peptide length fixed, so this is a bug in the caller, not a result"
        )
    diff = np.nonzero(a != b)[0]
    prot = sorted({int(i) for i in protected})
    prot_bad = [int(i) for i in prot if i < len(a) and a[i] != b[i]]
    return {
        "seq_len": int(len(a)),
        "seq_identity": float((a == b).mean()),
        "n_substitutions": int(len(diff)),
        "substituted_positions": diff.tolist(),
        "substitutions": [f"{resname(a[i])}{int(i)}{resname(b[i])}" for i in diff.tolist()],
        "protected_positions": prot,
        "protected_violations": prot_bad,
        "protected_ok": len(prot_bad) == 0,
    }


def ring_geometry(xyz_A, aatype, i: int, j: int, cyc_type: str) -> dict:
    """Measure the ACTUAL bond of the requested chemistry at the requested endpoints.

    Returns the bond distance, whether it is inside the chemistry's window, and whether the
    anchor residues can chemically host that bond. All three have to hold: a 1.3 A NZ-CG
    distance between residues that are not LYS and ASP/GLU/ASN/GLN is not an isopeptide.
    """
    if cyc_type not in CHEMISTRY:
        raise ValueError(f"unknown chemistry {cyc_type!r}; known: {sorted(CHEMISTRY)}")
    spec = CHEMISTRY[cyc_type]
    xyz, aa = _as_np(xyz_A), _as_np(aatype).astype(int)
    L = len(aa)
    out = {
        "cyc_type": cyc_type,
        "anchor_i": int(i),
        "anchor_j": int(j),
        "anchor_i_res": resname(aa[i]) if 0 <= i < L else "OOB",
        "anchor_j_res": resname(aa[j]) if 0 <= j < L else "OOB",
        "bond_atoms": None,
        "bond_dist_A": float("nan"),
        "bond_in_window": False,
        "anchors_can_host": False,
        "ring_closed": False,
    }
    if not (0 <= i < L and 0 <= j < L) or i == j:
        return out

    ri, rj = out["anchor_i_res"], out["anchor_j_res"]
    hosts = spec["hosts"]
    if hosts is None:
        out["anchors_can_host"] = True
    else:
        out["anchors_can_host"] = ri in hosts[0] and rj in hosts[1]

    atom_i, atom_j = spec["atoms"]
    if cyc_type == "isopeptide":
        atom_j = _ACCEPTOR_ATOMS.get(rj)
        if atom_j is None:
            return out  # acceptor residue cannot host; distance would be meaningless
    out["bond_atoms"] = f"{atom_i}-{atom_j}"

    slot_i, slot_j = rc.atom_order.get(atom_i), rc.atom_order.get(atom_j)
    if slot_i is None or slot_j is None:
        return out
    # The atom must actually exist on that residue type, or the coordinate is padding.
    m = _restype_atom_mask(aa)
    if not (m[i, slot_i] and m[j, slot_j]):
        return out

    d = float(np.linalg.norm(xyz[i, slot_i] - xyz[j, slot_j]))
    lo, hi = spec["window_A"]
    out["bond_dist_A"] = d
    out["bond_in_window"] = bool(lo <= d <= hi)
    out["ring_closed"] = bool(out["bond_in_window"] and out["anchors_can_host"])
    return out


def severe_clashes(
    pep_xyz_A,
    pep_aatype,
    tgt_xyz_A=None,
    tgt_aatype=None,
    ring: tuple[int, int] | None = None,
    dist_A: float = CONTRACT["clash_dist_A"],
) -> dict:
    """Count non-bonded heavy-atom pairs closer than `dist_A`.

    Excluded from the count: atoms in the same residue, atoms in sequence-adjacent residues
    (their backbone atoms are genuinely bonded), and the declared ring pair. Everything else
    at sub-2 A is a steric impossibility rather than a tight contact.
    """
    xyz = _as_np(pep_xyz_A)
    aa = _as_np(pep_aatype).astype(int)
    m = _restype_atom_mask(aa)
    idx, slot = np.nonzero(m)
    pa = xyz[idx, slot]
    d2 = ((pa[:, None, :] - pa[None, :, :]) ** 2).sum(-1)
    ri, rj = idx[:, None], idx[None, :]
    sep = np.abs(ri - rj)
    ignore = sep <= 1
    if ring is not None:
        a, b = int(ring[0]), int(ring[1])
        ignore |= ((ri == a) & (rj == b)) | ((ri == b) & (rj == a))
    bad = (d2 <= dist_A**2) & ~ignore
    bad = np.triu(bad, k=1)
    n_intra = int(bad.sum())
    free = d2.copy()
    free[ignore] = np.inf
    min_intra = float(np.sqrt(free.min())) if np.isfinite(free).any() else float("nan")

    out = {
        "clash_intra_peptide": n_intra,
        "min_intra_dist_A": min_intra,
        "clash_peptide_receptor": 0,
        "min_inter_dist_A": float("nan"),
    }
    if tgt_xyz_A is not None:
        t_xyz, t_aa = _as_np(tgt_xyz_A), _as_np(tgt_aatype).astype(int)
        t_m = _restype_atom_mask(t_aa)
        t_idx, t_slot = np.nonzero(t_m)
        ta = t_xyz[t_idx, t_slot]
        n_inter, best = 0, np.inf
        for s in range(0, len(ta), 4096):
            dd = ((pa[:, None, :] - ta[None, s : s + 4096, :]) ** 2).sum(-1)
            n_inter += int((dd <= dist_A**2).sum())
            best = min(best, float(dd.min()) if dd.size else np.inf)
        out["clash_peptide_receptor"] = n_inter
        out["min_inter_dist_A"] = float(np.sqrt(best)) if np.isfinite(best) else float("nan")
    out["severe_clashes"] = out["clash_intra_peptide"] + out["clash_peptide_receptor"]
    return out


def evaluate(
    *,
    before_xyz_A,
    before_aatype,
    after_xyz_A,
    after_aatype,
    tgt_xyz_A,
    tgt_aatype,
    cyc_type: str,
    anchor_i: int,
    anchor_j: int,
    protected: Iterable[int] = (),
    ddg_REU: float | None = None,
) -> dict:
    """Grade one edit against the contract. Returns every measured value plus the verdict.

    `ddg_REU` is the PAIRED Rosetta interface-score change (CP minus the original LP, same
    preparation and relaxation). Pass None when Rosetta has not been run: the geometry and
    sequence gates are then reported, `pass_ddg` is None, and `accepted` is False, because
    acceptance requires the relaxed paired score.
    """
    ring = ring_geometry(after_xyz_A, after_aatype, anchor_i, anchor_j, cyc_type)
    seq = sequence_budget(before_aatype, after_aatype, protected=protected)
    clash = severe_clashes(after_xyz_A, after_aatype, tgt_xyz_A, tgt_aatype,
                           ring=(anchor_i, anchor_j))

    c_before = heavy_atom_contacts(before_xyz_A, before_aatype, tgt_xyz_A, tgt_aatype)
    c_after = heavy_atom_contacts(after_xyz_A, after_aatype, tgt_xyz_A, tgt_aatype)
    iface = interface_residues(c_before)
    retention = contact_retention(c_before, c_after)

    rmsd_all = ca_rmsd_no_align(before_xyz_A, after_xyz_A)
    rmsd_iface = ca_rmsd_no_align(before_xyz_A, after_xyz_A, subset=iface)

    row = {
        **ring,
        **seq,
        **clash,
        "n_contacts_before": len(c_before),
        "n_contacts_after": len(c_after),
        "n_interface_residues": len(iface),
        "contact_retention": retention,
        "ca_rmsd_A": rmsd_all,
        "interface_ca_rmsd_A": rmsd_iface,
        "ddg_REU": ddg_REU,
    }

    # ---- gates ----------------------------------------------------------------
    row["pass_ring"] = bool(ring["ring_closed"])
    row["pass_clash"] = bool(clash["severe_clashes"] <= CONTRACT["max_severe_clashes"])
    row["pass_identity"] = bool(seq["seq_identity"] >= CONTRACT["seq_identity_min"])
    row["pass_substitutions"] = bool(seq["n_substitutions"] <= CONTRACT["max_substitutions"])
    row["pass_protected"] = bool(seq["protected_ok"])
    # NaN retention (an input with no contacts) must not pass by accident: NaN comparisons
    # are False, which is the behaviour we want, but make it explicit so it cannot be
    # "fixed" into truthiness later. See the recorded NaN-is-truthy bug in the SDEdit sweep.
    row["pass_contacts"] = bool(
        retention == retention and retention >= CONTRACT["contact_retention_min"]
    )
    row["pass_rmsd"] = bool(
        rmsd_all == rmsd_all and rmsd_all <= CONTRACT["ca_rmsd_max_A"]
    )
    row["pass_interface_rmsd"] = bool(
        rmsd_iface == rmsd_iface and rmsd_iface <= CONTRACT["interface_ca_rmsd_max_A"]
    )
    row["pass_ddg"] = (
        None if ddg_REU is None
        else bool(ddg_REU == ddg_REU and ddg_REU <= CONTRACT["ddg_max_REU"])
    )
    row["ddg_non_worsening"] = (
        None if ddg_REU is None else bool(ddg_REU == ddg_REU and ddg_REU <= 0.0)
    )

    geom_seq_gates = [
        row["pass_ring"], row["pass_clash"], row["pass_identity"],
        row["pass_substitutions"], row["pass_protected"], row["pass_contacts"],
        row["pass_rmsd"], row["pass_interface_rmsd"],
    ]
    row["pass_geometry_and_sequence"] = all(geom_seq_gates)
    row["accepted"] = bool(row["pass_geometry_and_sequence"] and row["pass_ddg"] is True)

    # ---- reporting-only sweeps, separate from the primary bar ------------------
    for k in MUTATION_BUDGETS:
        row[f"pass_budget_{k}"] = bool(
            seq["n_substitutions"] <= k and row["pass_ring"] and row["pass_clash"]
            and row["pass_contacts"] and row["pass_rmsd"] and row["pass_interface_rmsd"]
            and row["pass_protected"]
        )
    for thr in IDENTITY_THRESHOLDS:
        row[f"pass_identity_{int(thr * 100)}"] = bool(
            seq["seq_identity"] >= thr and row["pass_ring"] and row["pass_clash"]
            and row["pass_contacts"] and row["pass_rmsd"] and row["pass_interface_rmsd"]
            and row["pass_protected"]
        )
    return row
