"""Centralized batched closure/validity scorer for CPSea generated endpoints.

This is the single place that turns a decoded Atom37 structure + its requested
cyclization label into a scalar reward for reward-weighted replay (see
`proteinfoundation.replay`). It does not re-derive any linkage-atom mapping or
acceptance threshold: the distance/window logic is `per_sample_requested_bond_distance`
and the angle/dihedral logic is `linkage_geometry_terms`, both already used by
the training-time bond/geometry losses and validation metrics in
`proteinfoundation.eval.cyclic_reconstruction_metrics` /
`proteinfoundation.cyclization.linkage_geometry`. Reusing them here means this
scorer cannot silently drift from the metric/loss the rest of the codebase
reports and optimizes against.

Chirality and clash checks have no prior batched-tensor implementation in this
repo (the closest precedent, `utils.biopython_utils.calculate_clash_score`, is
a per-structure PDB/KDTree implementation) and are added fresh here, kept
deliberately simple (CA-CA clash distance, N/CA/C/CB improper-volume chirality
sign) since Stage 2 of the replay plan only requires them as logged
diagnostics/optional gates, not as the primary reward signal.
"""

from __future__ import annotations

import torch

from proteinfoundation.cyclization.constants import DISULFIDE, ISOPEPTIDE, MAINCHAIN
from proteinfoundation.cyclization.linkage_geometry import linkage_geometry_terms
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    C_IDX,
    CA_IDX,
    CB_IDX,
    N_IDX,
    per_sample_requested_bond_distance,
)

_EPS = 1e-6

# Additional margin (Angstrom) beyond each type's hard acceptance window
# (`*_BOND_WINDOW_A` in `cyclic_reconstruction_metrics.py`) that still counts as
# "near" a valid closure, for the shaped reward and `near_success`. Set equal to
# each window's own half-width -- every CPSea window (`MAINCHAIN_CN_BOND_WINDOW_A`,
# `DISULFIDE_SG_BOND_WINDOW_A`, `ISOPEPTIDE_N_C_BOND_WINDOW_A`) happens to be 0.5 A
# wide -- so this is read off the existing evaluator thresholds, not independently
# fit.
DEFAULT_NEAR_WINDOW_MARGIN_A: dict[int, float] = {
    MAINCHAIN: 0.5,
    DISULFIDE: 0.5,
    ISOPEPTIDE: 0.5,
}

# CA-CA clash distance (Angstrom). `biopython_utils.calculate_clash_score` uses
# 2.4 A for an all-atom/CA clash check on relaxed structures; we use a
# slightly tighter default here since this checks CA-CA only (sidechain atoms
# can legitimately sit closer than that).
DEFAULT_CLASH_THRESHOLD_A = 2.0


def _chirality_valid_per_residue(
    atom37: torch.Tensor, atom37_mask: torch.Tensor
) -> torch.Tensor:
    """Per-residue L-amino-acid stereo check, `[..., L]` bool.

    Uses the sign of the `(N-CA) x (C-CA) . (CB-CA)` scalar triple product.
    `virtual_cb_from_backbone` (used elsewhere in this codebase to reconstruct
    CB for glycine) places its output CB at a strictly positive sign under
    this convention for any non-degenerate backbone frame (the cross-product
    term in its formula is orthogonal to both inputs of this triple product by
    construction, so the sign is frame-independent, not merely
    dataset-typical) -- so a real, correctly-chiral L-amino-acid CB must also
    be positive; a reflected/D-amino-acid CB is negative. Only defined where
    N/CA/C/CB are all present; residues without a resolvable CB (e.g. glycine,
    or a masked-out slot) have nothing to check and are treated as valid.
    """
    mask = atom37_mask.bool()
    n = atom37[..., N_IDX, :]
    ca = atom37[..., CA_IDX, :]
    c = atom37[..., C_IDX, :]
    cb = atom37[..., CB_IDX, :]
    has_cb = mask[..., CB_IDX]
    has_backbone = mask[..., N_IDX] & mask[..., CA_IDX] & mask[..., C_IDX]

    v1 = n - ca
    v2 = c - ca
    v3 = cb - ca
    triple = (torch.cross(v1, v2, dim=-1) * v3).sum(dim=-1)
    is_l = triple > 0.0

    checkable = has_backbone & has_cb
    return is_l | ~checkable


def _clash_count_per_sample(
    atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    binder_mask: torch.Tensor,
    threshold_A: float = DEFAULT_CLASH_THRESHOLD_A,
    nm_to_ang: float = 10.0,
) -> torch.Tensor:
    """Counts binder-internal CA-CA pairs closer than `threshold_A`, `[B]`.

    Excludes self-pairs and sequence-adjacent residues (a normal covalent
    CA-CA spacing is ~3.8 A but neighboring residues are not a clash by
    construction). Each clashing pair is counted once.
    """
    ca_mask = atom37_mask[..., CA_IDX].bool() & binder_mask.bool()
    ca = atom37[..., CA_IDX, :]
    n = ca.shape[1]
    device = ca.device

    dist_A = torch.cdist(ca, ca) * nm_to_ang  # [B, L, L]
    pair_mask = ca_mask[:, :, None] & ca_mask[:, None, :]

    idx = torch.arange(n, device=device)
    not_adjacent = (idx[:, None] - idx[None, :]).abs() > 1
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)

    clash = pair_mask & not_adjacent[None, :, :] & upper[None, :, :] & (dist_A < threshold_A)
    return clash.sum(dim=(-1, -2)).float()


def score_cyclization(
    atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    aatype: torch.Tensor,
    binder_mask: torch.Tensor,
    linkage_metadata: dict[str, torch.Tensor],
    *,
    near_window_margin_A: dict[int, float] | None = None,
    gate_angle: bool = False,
    gate_dihedral: bool = False,
    gate_chirality: bool = False,
    gate_clash: bool = False,
    clash_threshold_A: float = DEFAULT_CLASH_THRESHOLD_A,
    angle_success_thresh: float = 0.5,
    dihedral_success_thresh: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Batched cyclization closure/validity reward and diagnostics, all `[B]`.

    Args:
        atom37: [B, L, 37, 3], nm. The decoded generated (or native) structure.
        atom37_mask: [B, L, 37], bool/float atom-presence mask for that same
            structure.
        aatype: [B, L] integer residue ids (OpenFold `restype_order`
            convention), of that same structure.
        binder_mask: [B, L] bool, True for real (non-padding) binder residues.
            Used only by the clash/chirality diagnostics (the closure check
            itself is restricted to the two linkage endpoints).
        linkage_metadata: `{"i", "j", "type", "has_cyclization"}`, the REQUESTED
            cyclization -- same schema as
            `eval.cyclic_reconstruction_metrics.extract_cyclization_metadata`
            (binder-local endpoint indices, int type in
            `cyclization.constants` convention, and a bool usability flag).
            This is the exact metadata produced by CPSea preprocessing
            (`CyclizationLabelTransform` / `parse_labels.infer_cyclization_label`);
            no atom mapping is re-derived here.
        near_window_margin_A: Optional override of `DEFAULT_NEAR_WINDOW_MARGIN_A`.
        gate_angle, gate_dihedral, gate_chirality, gate_clash: If True, require
            that criterion to additionally pass for `success`, AND zero
            `reward` wherever it fails (so a `raw_bounded`-weighted replay
            buffer -- see `replay.weighting` -- cannot assign a high weight to
            a candidate an enabled gate has flagged as invalid; `success` and
            `reward` must agree on which candidates are acceptable). All
            default False: the initial reward is closure/topology-distance
            only, per the "score only cyclization/structural validity"
            requirement.
        clash_threshold_A: CA-CA distance below which a pair counts as a clash.
        angle_success_thresh, dihedral_success_thresh: Huber/circular-loss
            thresholds (see `linkage_geometry_terms`) below which the angle/
            dihedral term counts as "passing", only used when the
            corresponding gate is enabled.

    Returns:
        Dict of `[B]` tensors:
            `reward`: bounded shaped reward in `[0, 1]`.
            `success`: bond closes (inside its acceptance window), AND every
                enabled gate also passes.
            `near_success`: bond is outside the acceptance window but within
                `near_window_margin_A`, and not already a `success`.
            `distance_error`: Angstrom distance outside the acceptance window
                (0 if inside), 0 wherever the required atoms are missing.
            `angle_errors`, `dihedral_error`: Huber/circular-loss-shaped error
                terms from `linkage_geometry_terms` (0 and masked out wherever
                the defining atoms are absent -- see that function).
            `chirality_valid`: True iff every real (non-glycine, non-masked)
                binder residue's CB sits on the L-amino-acid side of its
                N-CA-C plane.
            `clash_count`: number of binder-internal non-adjacent CA-CA pairs
                closer than `clash_threshold_A`.

        Missing linkage atoms (or `has_cyclization=False`) yield `reward=0`
        and `success=False`, never NaN.
    """
    device = atom37.device
    B = atom37.shape[0]
    margins = near_window_margin_A or DEFAULT_NEAR_WINDOW_MARGIN_A

    i = linkage_metadata["i"].to(device=device).long()
    j = linkage_metadata["j"].to(device=device).long()
    cyc_type = linkage_metadata["type"].to(device=device).long()
    has_cyc = linkage_metadata.get(
        "has_cyclization", torch.ones_like(i, dtype=torch.bool)
    )
    has_cyc = has_cyc.to(device=device).bool()

    bond = per_sample_requested_bond_distance(
        pred_atom37=atom37,
        atom37_mask=atom37_mask,
        seq_tokens=aatype,
        i=i,
        j=j,
        cyc_type=cyc_type,
    )
    dist_A = bond["dist_A"]
    window_lo_A, window_hi_A = bond["window_lo_A"], bond["window_hi_A"]
    atoms_valid = bond["atoms_valid"]

    type_gate = has_cyc & atoms_valid  # [B] bool

    below = torch.clamp(window_lo_A - dist_A, min=0.0)
    above = torch.clamp(dist_A - window_hi_A, min=0.0)
    violation = below + above  # 0 strictly inside the acceptance window
    distance_error = violation * type_gate.float()

    margin = torch.zeros(B, device=device, dtype=dist_A.dtype)
    for t, m in margins.items():
        margin = torch.where(cyc_type == t, torch.full_like(margin, float(m)), margin)
    margin = margin.clamp(min=_EPS)

    shape = (1.0 - violation / margin).clamp(0.0, 1.0)
    reward = torch.nan_to_num(type_gate.float() * shape, nan=0.0)

    success = type_gate & (violation <= 0.0)
    near_success = type_gate & (violation <= margin) & ~success

    geom = linkage_geometry_terms(
        pred_atom37=atom37,
        atom37_mask=atom37_mask,
        seq_tokens=aatype,
        cyclization_metadata={"i": i, "j": j, "type": cyc_type},
    )
    angle_errors = geom["angle"]
    dihedral_error = geom["dihedral"]
    geom_valid = geom["valid"]

    if gate_angle:
        angle_ok = (~geom_valid) | (angle_errors <= angle_success_thresh)
        success = success & angle_ok
        reward = reward * angle_ok.float()
    if gate_dihedral:
        dihedral_ok = (~geom_valid) | (dihedral_error <= dihedral_success_thresh)
        success = success & dihedral_ok
        reward = reward * dihedral_ok.float()

    chir_per_residue = _chirality_valid_per_residue(atom37, atom37_mask)  # [B, L]
    binder_bool = binder_mask.bool()
    chirality_valid = (~binder_bool | chir_per_residue).all(dim=-1)
    if gate_chirality:
        success = success & chirality_valid
        reward = reward * chirality_valid.float()

    clash_count = _clash_count_per_sample(
        atom37, atom37_mask, binder_mask, threshold_A=clash_threshold_A
    )
    if gate_clash:
        clash_ok = clash_count == 0
        success = success & clash_ok
        reward = reward * clash_ok.float()

    return {
        "reward": reward,
        "success": success,
        "near_success": near_success,
        "distance_error": distance_error,
        "angle_errors": angle_errors,
        "dihedral_error": dihedral_error,
        "chirality_valid": chirality_valid,
        "clash_count": clash_count,
    }
