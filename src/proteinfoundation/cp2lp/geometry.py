"""Geometry primitives shared by the CP->LP discriminator, losses and report.

Everything here works in NANOMETRES on atom37 tensors, matching the training path
(``autoencoder.decode`` returns ``coors_nm``). The export and the report convert to
angstrom at their own boundary; nothing in between does, so there is one unit convention
to keep straight rather than two.

THE TERMINAL BOND IS ABSENT. Every backbone term here treats the peptide as an open
chain: bonded pairs are ``(i, i+1)`` for ``i < L-1`` and the ``(L-1, 0)`` closure is
never included. Scoring a generated LP with the ring bond present is the single easiest
way to make an open peptide look broken, and it is what the CP-side code does by default.
"""

from __future__ import annotations

import torch
from openfold.np.residue_constants import atom_order, restype_num

N_IDX = atom_order["N"]
CA_IDX = atom_order["CA"]
C_IDX = atom_order["C"]
O_IDX = atom_order["O"]

#: Ideal peptide-bond C(i)-N(i+1) length in nm, and the tolerance used as a flat bottom.
IDEAL_PEPTIDE_BOND_NM = 0.1329
PEPTIDE_BOND_TOL_NM = 0.02

#: Ideal intra-residue backbone bond lengths in nm.
IDEAL_N_CA_NM = 0.1458
IDEAL_CA_C_NM = 0.1525

#: Coarse chemistry classes, used instead of 20-way identity for POCKET residues so the
#: discriminator has a harder time memorising receptor composition (see discriminator.py
#: on the receptor shortcut). Order: hydrophobic, polar, charged, special (G/P), unknown.
_CHEM_GROUPS = {
    "hydrophobic": ["A", "C", "I", "L", "M", "F", "V", "W"],
    "polar": ["N", "Q", "S", "T", "Y", "H"],
    "charged": ["D", "E", "K", "R"],
    "special": ["G", "P"],
}
N_CHEM_CLASSES = len(_CHEM_GROUPS) + 1


def _chem_class_table() -> torch.Tensor:
    """[21] long tensor mapping restype index -> coarse chemistry class."""
    from openfold.np.residue_constants import restype_order

    table = torch.full((restype_num + 1,), len(_CHEM_GROUPS), dtype=torch.long)
    for cls_idx, (_, letters) in enumerate(_CHEM_GROUPS.items()):
        for letter in letters:
            table[restype_order[letter]] = cls_idx
    return table


CHEM_CLASS_TABLE = _chem_class_table()


#: Floor inside every sqrt, so a norm evaluated at exactly zero has a finite derivative.
_NORM_EPS = 1e-12


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Euclidean norm with an explicit floor inside the sqrt.

    DEFENSIVE, not a fix for a known break. Measured on torch 2.7.0:
    `torch.linalg.vector_norm` returns a 0 subgradient at the origin rather than NaN, so
    it was *not* the source of the NaN that this arm hit -- `atan2(0, 0)` in
    :func:`backbone_dihedrals` was, and that one is fixed at the call site.

    It is kept because padded residues genuinely do hold exactly-zero coordinates (the
    decoder multiplies by the residue and atom masks), the 0-subgradient behaviour is a
    torch implementation detail rather than a documented guarantee, and the failure mode
    if it ever changes is silent and total -- see `backbone_dihedrals` for why masking
    afterwards does not contain a NaN.
    """
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + _NORM_EPS)


def residue_lengths(mask: torch.Tensor) -> torch.Tensor:
    """[b] residue count per example from a [b, n] boolean mask."""
    return mask.bool().sum(dim=-1)


def ca_coords(atom37: torch.Tensor) -> torch.Tensor:
    """[b, n, 3] CA coordinates from a [b, n, 37, 3] atom37 tensor."""
    return atom37[..., CA_IDX, :]


def terminal_gap_nm(atom37: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[b] CA(first)-CA(last) distance, the quantity the generator must open.

    Zero for single-residue examples, which have no separation to speak of.
    """
    ca = ca_coords(atom37)
    lengths = residue_lengths(mask)
    idx = torch.arange(ca.shape[0], device=ca.device)
    gap = safe_norm(ca[idx, (lengths - 1).clamp(min=0)] - ca[idx, 0], dim=-1)
    return torch.where(lengths > 1, gap, torch.zeros_like(gap))


def open_chain_bond_pair_mask(mask: torch.Tensor) -> torch.Tensor:
    """[b, n-1] mask of bonded ``(i, i+1)`` pairs for an OPEN chain.

    The ``(L-1, 0)`` closure is deliberately not represented: there is no index pair for
    it here at all, so no caller can accidentally include it.
    """
    m = mask.bool()
    return m[:, :-1] & m[:, 1:]


def peptide_bond_deviation_nm(atom37: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """C(i)-N(i+1) length deviation from ideal, for the open chain.

    Returns ``(deviation [b, n-1], pair_mask [b, n-1])`` where deviation is
    ``|d - ideal|``. Masked entries are zero.
    """
    c = atom37[:, :-1, C_IDX, :]
    n_next = atom37[:, 1:, N_IDX, :]
    pair_mask = open_chain_bond_pair_mask(mask)
    d = safe_norm(c - n_next, dim=-1)
    return (d - IDEAL_PEPTIDE_BOND_NM).abs() * pair_mask, pair_mask


def intra_residue_bond_deviation_nm(
    atom37: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """N-CA and CA-C length deviation from ideal, per residue. ``([b, n, 2], [b, n])``."""
    m = mask.bool()
    n_ca = safe_norm(atom37[..., CA_IDX, :] - atom37[..., N_IDX, :], dim=-1)
    ca_c = safe_norm(atom37[..., C_IDX, :] - atom37[..., CA_IDX, :], dim=-1)
    dev = torch.stack(
        [(n_ca - IDEAL_N_CA_NM).abs(), (ca_c - IDEAL_CA_C_NM).abs()], dim=-1
    )
    return dev * m[..., None], m


def pairwise_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[.., n, m] Euclidean distances between two coordinate sets ``[.., n, 3]``/``[.., m, 3]``.

    ``cdist`` is avoided: its backward is numerically unstable at exactly-coincident
    points, which decoded padding rows routinely are. The explicit form with an epsilon
    under the sqrt keeps the gradient finite there.
    """
    diff = a[..., :, None, :] - b[..., None, :, :]
    return torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)


def heavy_atom_distances(
    atom37_a: torch.Tensor,
    mask_a: torch.Tensor,
    atom37_b: torch.Tensor,
    mask_b: torch.Tensor,
) -> torch.Tensor:
    """[b, na, nb] minimum heavy-atom distance between every residue pair.

    Args:
        atom37_a/b: ``[b, n*, 37, 3]`` coordinates in nm.
        mask_a/b: ``[b, n*, 37]`` per-atom presence masks.

    Absent atoms are pushed to a large distance rather than dropped, so the ``min`` over
    atoms is well-defined for every residue pair including fully-empty padding rows.
    """
    b, na = atom37_a.shape[0], atom37_a.shape[1]
    nb = atom37_b.shape[1]
    flat_a = atom37_a.reshape(b, na * 37, 3)
    flat_b = atom37_b.reshape(b, nb * 37, 3)
    d = pairwise_distances(flat_a, flat_b)  # [b, na*37, nb*37]
    valid = mask_a.reshape(b, na * 37, 1).bool() & mask_b.reshape(b, 1, nb * 37).bool()
    d = torch.where(valid, d, torch.full_like(d, 1e4))
    d = d.reshape(b, na, 37, nb, 37)
    return d.amin(dim=(2, 4))  # [b, na, nb]


def soft_contact(dist: torch.Tensor, cutoff_nm: float = 0.8, sharpness_nm: float = 0.1) -> torch.Tensor:
    """Differentiable in-contact indicator, ``sigmoid((cutoff - d) / sharpness)``.

    A hard ``d < cutoff`` has zero gradient everywhere, which makes it useless as a loss
    on contact retention -- there is no signal telling a contact that drifted to 0.9 nm to
    come back. The sigmoid keeps a usable slope through the decision boundary.
    """
    return torch.sigmoid((cutoff_nm - dist) / sharpness_nm)


def radius_of_gyration_nm(atom37: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[b] CA radius of gyration. A coarse shape summary, real for open and closed alike."""
    ca = ca_coords(atom37)
    m = mask.bool().float()[..., None]
    n = m.sum(dim=1).clamp(min=1.0)
    centroid = (ca * m).sum(dim=1, keepdim=True) / n[:, None, :]
    sq = ((ca - centroid) ** 2).sum(dim=-1, keepdim=True) * m
    return torch.sqrt(sq.sum(dim=1).squeeze(-1) / n.squeeze(-1) + 1e-12)


def backbone_dihedrals(atom37: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[b, n, 4] ``(sin phi, cos phi, sin psi, cos psi)`` for the OPEN chain.

    Undefined dihedrals -- phi at the first residue, psi at the last, anything adjacent to
    a masked residue -- are zero in both components, which is distinguishable from every
    real angle (whose sin^2 + cos^2 is 1) and so is a usable "absent" encoding.
    """
    m = mask.bool()
    b, n = m.shape
    zeros = torch.zeros(b, n, device=atom37.device, dtype=atom37.dtype)

    def _dihedral(p0, p1, p2, p3):
        b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
        b1n = b1 / safe_norm(b1, dim=-1, keepdim=True).clamp(min=1e-8)
        v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
        w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
        x = (v * w).sum(-1)
        y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
        # `atan2` differentiates to x/(x^2+y^2) and -y/(x^2+y^2), i.e. 0/0 = NaN at the
        # origin -- which is exactly where padded residues land, since all their atoms are
        # zero and so v and w collapse to zero too. Substitute a harmless (0, 1) there
        # BEFORE the call. Sanitising afterwards is not equivalent: by then the NaN is in
        # the graph, and `torch.where` hands gradient to both branches, so multiplying by
        # a zero mask gives 0 * NaN = NaN.
        degenerate = (x * x + y * y) < 1e-12
        x = torch.where(degenerate, torch.ones_like(x), x)
        y = torch.where(degenerate, torch.zeros_like(y), y)
        return torch.atan2(y, x)

    nat, caat, cat = atom37[..., N_IDX, :], atom37[..., CA_IDX, :], atom37[..., C_IDX, :]

    # phi(i) = C(i-1) - N(i) - CA(i) - C(i); defined for i >= 1.
    phi = torch.zeros(b, n, device=atom37.device, dtype=atom37.dtype)
    phi_valid = torch.zeros(b, n, dtype=torch.bool, device=atom37.device)
    if n > 1:
        phi[:, 1:] = _dihedral(cat[:, :-1], nat[:, 1:], caat[:, 1:], cat[:, 1:])
        phi_valid[:, 1:] = m[:, :-1] & m[:, 1:]

    # psi(i) = N(i) - CA(i) - C(i) - N(i+1); defined for i <= L-2.
    psi = torch.zeros(b, n, device=atom37.device, dtype=atom37.dtype)
    psi_valid = torch.zeros(b, n, dtype=torch.bool, device=atom37.device)
    if n > 1:
        psi[:, :-1] = _dihedral(nat[:, :-1], caat[:, :-1], cat[:, :-1], nat[:, 1:])
        psi_valid[:, :-1] = m[:, :-1] & m[:, 1:]

    return torch.stack(
        [
            torch.where(phi_valid, torch.sin(phi), zeros),
            torch.where(phi_valid, torch.cos(phi), zeros),
            torch.where(psi_valid, torch.sin(psi), zeros),
            torch.where(psi_valid, torch.cos(psi), zeros),
        ],
        dim=-1,
    )
