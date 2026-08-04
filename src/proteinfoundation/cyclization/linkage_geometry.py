"""Angle, dihedral and chirality terms for the cyclic closing bond.

`bond_loss.py` supervises one number: the distance between the two anchor atoms. That is
necessary but not sufficient -- a bond can sit at exactly 1.33 A while pointing in a
chemically impossible direction, because distance alone constrains neither the angles at
either end nor the torsion about the bond. This module adds the rest of the linkage
geometry, per chemistry, so "closed" means a real bond rather than two atoms that happen
to be the right distance apart.

Each term is masked independently: a linkage contributes to the angle term only where the
atoms that define the angle are actually present in the decoded structure. Missing atoms
produce no gradient rather than a NaN, so an incomplete side chain cannot poison the batch.

Reference values
----------------
The constants below are standard small-molecule/protein geometry, and each carries its
source in a comment. They are DEFAULTS, not measurements of this dataset: the intended
workflow is to replace them with training-set statistics via
`script_utils/measure_linkage_geometry_stats.py`, which writes a JSON this module accepts.
Statistics must never be fit on the evaluation targets.

Units: angles in radians internally; distances in Angstrom to match `bond_loss`.
"""

import math

import torch

from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    AA_ASN,
    AA_ASP,
    AA_CYS,
    AA_LYS,
    C_IDX,
    CA_IDX,
    CB_IDX,
    CD_IDX,
    CE_IDX,
    CG_IDX,
    N_IDX,
    NZ_IDX,
    SG_IDX,
    _gather_atom,
    is_isopeptide_acid,
)
from proteinfoundation.cyclization.constants import DISULFIDE, ISOPEPTIDE, MAINCHAIN
from proteinfoundation.utils.angle_utils import bond_angles, signed_dihedral_angle

_EPS = 1e-8

# --- reference geometry, per linkage type -----------------------------------------
# angle_ref: (angle_at_i, angle_at_j) in radians
# dihedral_ref: target torsion in radians; `dihedral_symmetric` folds +/-x together.
LINKAGE_GEOMETRY_REFERENCE = {
    MAINCHAIN: {
        # Trans peptide bond: CA-C-N ~116 deg, C-N-CA ~122 deg, omega ~180 deg.
        # (Engh & Huber backbone parameters.)
        "angle_ref": (math.radians(116.2), math.radians(121.7)),
        "dihedral_ref": math.pi,
        "dihedral_symmetric": False,
    },
    DISULFIDE: {
        # CB-SG-SG ~104 deg; the CB-SG-SG-CB torsion sits near +/-90 deg, and BOTH
        # handednesses occur in real disulfides -- so this one is symmetric, and a loss
        # that pulled every disulfide to a single sign would be enforcing a chirality
        # that does not exist.
        "angle_ref": (math.radians(103.8), math.radians(103.8)),
        "dihedral_ref": math.radians(90.0),
        "dihedral_symmetric": True,
    },
    ISOPEPTIDE: {
        # Side-chain amide: CE-NZ-C ~120 deg at the lysine, NZ-C-CG ~117 deg at the acid.
        # Planar amide, so the torsion is ~180 deg like the backbone case.
        "angle_ref": (math.radians(120.0), math.radians(117.0)),
        "dihedral_ref": math.pi,
        "dihedral_symmetric": False,
    },
}


def _huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Elementwise Huber, quadratic near zero and linear beyond `delta`."""
    absx = x.abs()
    return torch.where(absx <= delta, 0.5 * x * x, delta * (absx - 0.5 * delta))


def circular_loss(pred: torch.Tensor, ref: torch.Tensor, symmetric: bool = False) -> torch.Tensor:
    """`1 - cos(pred - ref)`, in [0, 2]; zero when equal.

    Using the cosine rather than a raw difference is what makes this correct across the
    -pi/+pi seam: angles of -179 and +179 degrees are 2 degrees apart, and a plain
    subtraction would score them as 358.

    `symmetric=True` folds +x and -x together (via cos of the doubled angle offset), for
    torsions where both handednesses are chemically real -- notably the disulfide
    CB-SG-SG-CB torsion, which occurs at both +90 and -90 degrees.
    """
    if symmetric:
        # cos(2p - 2r) is invariant to p -> -p when r is the reference magnitude.
        return 1.0 - torch.cos(2.0 * pred - 2.0 * ref)
    return 1.0 - torch.cos(pred - ref)


def _mainchain_atoms(atom37, mask_bool, i, j):
    """(a, b, c, d, valid) for CA(j)-C(j) : N(i)-CA(i).

    `i`/`j` are the sorted first/last binder-local endpoint indices, so the real
    head-to-tail closing bond is C of the LAST residue (j, C-terminus) to N of the
    FIRST residue (i, N-terminus) -- the same orientation as every other in-chain
    peptide bond (residue k's C to residue k+1's N), just wrapping around the ring.
    This must match `per_sample_requested_bond_distance`'s N(i)-C(j) anchor and
    `evaluation.cyclization_pdb_metrics`'s N(first)-C(last) measurement; building
    CA(i)-C(i):N(j)-CA(j) instead would supervise a different, chemically
    backwards pair (both atoms already engaged in their own in-chain bonds).
    """
    ca_j, v1 = _gather_atom(atom37, mask_bool, j, CA_IDX)
    c_j, v2 = _gather_atom(atom37, mask_bool, j, C_IDX)
    n_i, v3 = _gather_atom(atom37, mask_bool, i, N_IDX)
    ca_i, v4 = _gather_atom(atom37, mask_bool, i, CA_IDX)
    return ca_j, c_j, n_i, ca_i, v1 & v2 & v3 & v4


def _disulfide_atoms(atom37, mask_bool, i, j, aa_i, aa_j):
    """(CB_i, SG_i, SG_j, CB_j, valid) -- only where both residues really are cysteine."""
    cb_i, v1 = _gather_atom(atom37, mask_bool, i, CB_IDX)
    sg_i, v2 = _gather_atom(atom37, mask_bool, i, SG_IDX)
    sg_j, v3 = _gather_atom(atom37, mask_bool, j, SG_IDX)
    cb_j, v4 = _gather_atom(atom37, mask_bool, j, CB_IDX)
    return cb_i, sg_i, sg_j, cb_j, v1 & v2 & v3 & v4 & (aa_i == AA_CYS) & (aa_j == AA_CYS)


def _isopeptide_atoms(atom37, mask_bool, i, j, aa_i, aa_j):
    """(CE_lys, NZ_lys, C_acid, X_acid, valid), orientation-resolved.

    The isopeptide bond is directional -- a lysine NZ donor bonded to an Asp/Asn (or
    Glu/Gln) carbonyl acceptor -- and the label does not promise which of `i`/`j` is the
    lysine. Both assignments are built and selected per sample, so the angles are measured
    on the real donor/acceptor rather than on whichever index happened to come first.
    """
    lys_is_i = aa_i == AA_LYS

    lys_idx = torch.where(lys_is_i, i, j)
    acid_idx = torch.where(lys_is_i, j, i)
    aa_acid = torch.where(lys_is_i, aa_j, aa_i)

    # Lysine side chain: CE-NZ is the real bonded pair that defines the angle at the
    # donor (CE IS present in the atom37 slot table -- index 19 -- so no CD proxy is
    # needed; CD sits one bond further out, on the far side of CE, and is not bonded
    # to NZ at all).
    ce_lys, v1 = _gather_atom(atom37, mask_bool, lys_idx, CE_IDX)
    nz_lys, v2 = _gather_atom(atom37, mask_bool, lys_idx, NZ_IDX)
    # Asp/Asn's carbonyl carbon is CG (their other real neighbor is CB); Glu/Gln's is
    # CD (their other real neighbor is CG, one bond further out than Asp/Asn's CB).
    # ASN/GLN are included alongside ASP/GLU because `cyclization.parse_labels`
    # accepts them as real isopeptide partners too (see `is_isopeptide_acid`) --
    # excluding them here would silently drop the angle/dihedral term for a sample
    # already labeled `has_cyclization=True`.
    is_cg_family = (aa_acid == AA_ASP) | (aa_acid == AA_ASN)
    acid_c = torch.where(is_cg_family, torch.full_like(acid_idx, CG_IDX), torch.full_like(acid_idx, CD_IDX))
    acid_x = torch.where(is_cg_family, torch.full_like(acid_idx, CB_IDX), torch.full_like(acid_idx, CG_IDX))
    c_acid, v3 = _gather_atom(atom37, mask_bool, acid_idx, acid_c)
    x_acid, v4 = _gather_atom(atom37, mask_bool, acid_idx, acid_x)

    is_lys_pair = ((aa_i == AA_LYS) & is_isopeptide_acid(aa_j)) | (
        (aa_j == AA_LYS) & is_isopeptide_acid(aa_i)
    )
    return ce_lys, nz_lys, c_acid, x_acid, v1 & v2 & v3 & v4 & is_lys_pair


def linkage_geometry_terms(
    pred_atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    seq_tokens: torch.Tensor,
    cyclization_metadata: dict[str, torch.Tensor],
    reference: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Per-sample angle and dihedral losses for the requested closing bond.

    Returns a dict of [B] tensors: `angle`, `dihedral`, and matching `*_valid` masks.
    Entries where the defining atoms are absent are zero AND masked out, so a caller that
    averages using the mask never sees a contribution from an undefined geometry.

    All four atoms per chemistry are gathered as (a, b, c, d) where b-c is the closing
    bond, so the two angles are a-b-c and b-c-d and the torsion is a-b-c-d.
    """
    reference = reference or LINKAGE_GEOMETRY_REFERENCE
    device = pred_atom37.device
    B = pred_atom37.shape[0]
    mask_bool = atom37_mask.bool()

    i = cyclization_metadata["i"].to(device=device).long()
    j = cyclization_metadata["j"].to(device=device).long()
    cyc_type = cyclization_metadata["type"].to(device=device).long()

    L = pred_atom37.shape[1]
    i = i.clamp(0, L - 1)
    j = j.clamp(0, L - 1)
    batch_idx = torch.arange(B, device=device)
    aa_i = seq_tokens[batch_idx, i]
    aa_j = seq_tokens[batch_idx, j]

    angle_loss = torch.zeros(B, device=device, dtype=pred_atom37.dtype)
    dihedral_loss = torch.zeros(B, device=device, dtype=pred_atom37.dtype)
    valid = torch.zeros(B, device=device, dtype=torch.bool)

    builders = {
        MAINCHAIN: lambda: _mainchain_atoms(pred_atom37, mask_bool, i, j),
        DISULFIDE: lambda: _disulfide_atoms(pred_atom37, mask_bool, i, j, aa_i, aa_j),
        ISOPEPTIDE: lambda: _isopeptide_atoms(pred_atom37, mask_bool, i, j, aa_i, aa_j),
    }

    for type_idx, builder in builders.items():
        selected = cyc_type == type_idx
        if not bool(selected.any()):
            continue
        a, b, c, d = builder()[:4]
        atoms_ok = builder()[4]
        ok = selected & atoms_ok

        ref = reference[type_idx]
        # CAREFUL: `bond_angles(x, y, z)` returns the angle at vertex **x** (it builds
        # y - x and z - x), NOT at the middle argument as the a-b-c naming suggests. The
        # angles wanted here are the ones at the two bonded atoms b and c, so b and c must
        # be passed FIRST. Getting this wrong supervises a well-defined but meaningless
        # quantity, which trains silently and looks fine.
        ang_i = bond_angles(b, a, c)  # angle a-b-c, at b
        ang_j = bond_angles(c, b, d)  # angle b-c-d, at c
        ref_i, ref_j = ref["angle_ref"]
        ang_term = _huber(ang_i - ref_i) + _huber(ang_j - ref_j)

        tor = signed_dihedral_angle(a, b, c, d)  # [B]
        dih_term = circular_loss(
            tor,
            torch.full_like(tor, float(ref["dihedral_ref"])),
            symmetric=bool(ref.get("dihedral_symmetric", False)),
        )

        angle_loss = torch.where(ok, ang_term, angle_loss)
        dihedral_loss = torch.where(ok, dih_term, dihedral_loss)
        valid = valid | ok

    # Never let a masked-out sample carry a NaN into the reduction.
    angle_loss = torch.nan_to_num(angle_loss, nan=0.0) * valid.float()
    dihedral_loss = torch.nan_to_num(dihedral_loss, nan=0.0) * valid.float()

    return {"angle": angle_loss, "dihedral": dihedral_loss, "valid": valid}


def linkage_geometry_loss(
    pred_atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    seq_tokens: torch.Tensor,
    cyclization_metadata: dict[str, torch.Tensor] | None,
    w_angle: float = 1.0,
    w_dihedral: float = 1.0,
    t_weight: torch.Tensor | None = None,
    reference: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted angle+dihedral loss, a safe differentiable zero when nothing is supervisable.

    Mirrors `cyclization_bond_loss`'s contract exactly -- same metadata dict, same
    `t_weight` ramp, same "return a zero that still participates in autograd" behaviour so
    DDP and mixed dataloaders do not choke.
    """
    empty = {
        "n_valid": 0.0,
        "angle": float("nan"),
        "dihedral": float("nan"),
        "mean_angle_err_deg": float("nan"),
    }
    if cyclization_metadata is None:
        return pred_atom37.sum() * 0.0, empty

    device = pred_atom37.device
    has_cyc = cyclization_metadata["has_cyclization"].to(device=device).bool()

    terms = linkage_geometry_terms(
        pred_atom37=pred_atom37,
        atom37_mask=atom37_mask,
        seq_tokens=seq_tokens,
        cyclization_metadata=cyclization_metadata,
        reference=reference,
    )
    valid = has_cyc & terms["valid"]
    if not bool(valid.any()):
        return pred_atom37.sum() * 0.0, empty

    weight = valid.float()
    if t_weight is not None:
        weight = weight * t_weight.to(device=device).float().clamp(0.0, 1.0)
    weight = weight.detach()
    denom = weight.sum().clamp(min=_EPS)

    per_sample = w_angle * terms["angle"] + w_dihedral * terms["dihedral"]
    loss = (per_sample * weight).sum() / denom

    with torch.no_grad():
        supervised = weight > 0
        if not bool(supervised.any()):
            return loss, empty
        metrics = {
            "n_valid": float(supervised.sum().item()),
            "angle": float(terms["angle"][supervised].mean().item()),
            "dihedral": float(terms["dihedral"][supervised].mean().item()),
        }
    return loss, metrics
