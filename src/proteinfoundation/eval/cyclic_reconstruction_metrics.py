"""Shared, model-agnostic reconstruction/cyclization metrics for Atom37 structures.

These utilities are used by:
  - `proteinfoundation.eval.ae_reconstruction_eval` (AE-only diagnostic, four-way
    decode diagnostic).
  - `script_utils/eval_ae_reconstruction.py` (standalone CLI comparing AE checkpoints).
  - `Proteina` validation logging (see `log_unified_validation_metrics`).

Every function is pure-tensor (no model dependency) and operates on Atom37
coordinates/masks so it is trivially unit-testable (see
`script_utils/test_cyclic_reconstruction_metrics.py`).

Units: input coordinates are assumed to be in **nanometers** (nm), matching the
rest of the codebase (see `proteinfoundation.utils.coors_utils`). Metrics whose
name ends in `_A` are converted to Angstrom (`* 10`) for interpretability;
`sum_xyz_l1_per_residue_nm` intentionally stays in nm (it is the historical
reconstruction-L1 metric, renamed but numerically unchanged).

Atom37 ordering: this project's `UNIFIED_ATOM37_ENCODING`
(`proteinfoundation.utils.constants`) is index-identical to
`openfold.np.residue_constants.atom_order` (N=0, CA=1, C=2, CB=3, O=4, ...,
SG=10, ..., NZ=35, OXT=36), so `get_atom_index` simply wraps the canonical
OpenFold mapping instead of re-hardcoding it.
"""

from __future__ import annotations

import torch
from openfold.np.residue_constants import atom_order

from proteinfoundation.cyclization.constants import (
    AA_ASP,
    AA_CYS,
    AA_GLU,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
)

NM_TO_ANG = 10.0
_EPS = 1e-6


def get_atom_index(atom_name: str) -> int:
    """Returns the canonical Atom37 slot index for `atom_name` (e.g. "CA" -> 1).

    Thin wrapper over `openfold.np.residue_constants.atom_order`, which is the
    single source of truth for Atom37 ordering used throughout this repo (see
    module docstring). Not hardcoded; see
    `script_utils/test_cyclic_reconstruction_metrics.py` for an assertion that
    this matches the expected indices for N, CA, C, CB, SG, NZ, CG, CD.
    """
    return atom_order[atom_name]


N_IDX = get_atom_index("N")
CA_IDX = get_atom_index("CA")
C_IDX = get_atom_index("C")
CB_IDX = get_atom_index("CB")
SG_IDX = get_atom_index("SG")
NZ_IDX = get_atom_index("NZ")
CG_IDX = get_atom_index("CG")
CD_IDX = get_atom_index("CD")
CE_IDX = get_atom_index("CE")


def virtual_cb_from_backbone(
    n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Idealized Cβ position from backbone N/CA/C (tetrahedral geometry).

    Standard formula for reconstructing a Cβ atom from the backbone frame when
    the real Cβ is absent (e.g. glycine); the same formula used by
    ProteinMPNN/trRosetta. Inputs/outputs share shape `[..., 3]`.
    """
    b1 = ca - n
    b2 = c - ca
    b3 = torch.cross(b1, b2, dim=-1)
    return -0.58273431 * b3 + 0.56802827 * b1 - 0.54067466 * b2 + ca


def get_cb_position(
    atom37: torch.Tensor, atom37_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the Cβ position for each residue, [..., 37, 3] -> [..., 3].

    Uses the real Cβ where present in `atom37_mask`; otherwise falls back to
    `virtual_cb_from_backbone` computed from that same structure's N/CA/C. This
    generalizes "use a virtual Cβ for glycine" to "use a virtual Cβ whenever
    Cβ is absent", which handles glycine automatically without needing a
    separate sequence lookup, and is safe for any other residue with a missing
    Cβ atom.

    Returns:
        (cb_pos [..., 3], cb_is_real [...] bool)
    """
    real_cb = atom37[..., CB_IDX, :]
    cb_is_real = atom37_mask[..., CB_IDX].bool()
    virtual_cb = virtual_cb_from_backbone(
        atom37[..., N_IDX, :], atom37[..., CA_IDX, :], atom37[..., C_IDX, :]
    )
    cb_pos = torch.where(cb_is_real[..., None], real_cb, virtual_cb)
    return cb_pos, cb_is_real


def _per_structure_coord_errors(
    pred_atom37: torch.Tensor,
    gt_atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    eps: float = _EPS,
) -> dict[str, torch.Tensor]:
    """Per-structure (un-reduced) coordinate error statistics, shape [B] each.

    Shared by `masked_atom37_coord_metrics` (batch-mean reduction) and
    `cyclic_geometry_metrics` (endpoint-restricted mask + NaN-aware reduction),
    so both use identical error math.
    """
    mask = atom37_mask.bool()
    diff = (pred_atom37 - gt_atom37) * mask[..., None]
    abs_err = diff.abs()
    sq_err = diff**2

    valid_residue_mask = mask.any(dim=-1)  # [B, L]
    valid_residue_count = valid_residue_mask.sum(dim=-1).float()  # [B]
    valid_atom_count = mask.sum(dim=(-1, -2)).float()  # [B], number of valid atom slots
    valid_coord_count = valid_atom_count * 3.0  # [B], number of valid xyz scalars

    per_residue_sum_abs = abs_err.sum(dim=(-1, -2))  # [B, L]
    sum_xyz_l1_nm = per_residue_sum_abs.sum(dim=-1) / valid_residue_count.clamp(min=eps)

    total_abs_err = abs_err.sum(dim=(-3, -2, -1))  # [B]
    coord_mae_A = total_abs_err / valid_coord_count.clamp(min=eps) * NM_TO_ANG

    total_sq_err = sq_err.sum(dim=(-3, -2, -1))  # [B]
    atom_rmse_A = torch.sqrt(total_sq_err / valid_coord_count.clamp(min=eps)) * NM_TO_ANG

    return {
        "sum_xyz_l1_per_residue_nm": sum_xyz_l1_nm,
        "coord_mae_A": coord_mae_A,
        "atom_rmse_A": atom_rmse_A,
        "valid_atom_count": valid_atom_count,
        "valid_residue_count": valid_residue_count,
    }


def masked_atom37_coord_metrics(
    pred_atom37: torch.Tensor,
    gt_atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Batch-averaged Atom37 coordinate reconstruction metrics.

    Args:
        pred_atom37: [B, L, 37, 3], nm.
        gt_atom37: [B, L, 37, 3], nm.
        atom37_mask: [B, L, 37], bool/float valid-coordinate mask.
        prefix: metric key prefix, e.g. "val/ae_recon".

    Returns:
        dict with keys `{prefix}/sum_xyz_l1_per_residue_nm`, `{prefix}/coord_mae_A`,
        `{prefix}/atom_rmse_A`, `{prefix}/valid_atom_count_mean`,
        `{prefix}/valid_residue_count_mean`, each a python float (batch mean of
        the corresponding per-structure statistic).
    """
    per_struct = _per_structure_coord_errors(pred_atom37, gt_atom37, atom37_mask)
    return {
        f"{prefix}/sum_xyz_l1_per_residue_nm": float(per_struct["sum_xyz_l1_per_residue_nm"].mean().item()),
        f"{prefix}/coord_mae_A": float(per_struct["coord_mae_A"].mean().item()),
        f"{prefix}/atom_rmse_A": float(per_struct["atom_rmse_A"].mean().item()),
        f"{prefix}/valid_atom_count_mean": float(per_struct["valid_atom_count"].mean().item()),
        f"{prefix}/valid_residue_count_mean": float(per_struct["valid_residue_count"].mean().item()),
    }


COORD_METRIC_SUFFIXES = [
    "sum_xyz_l1_per_residue_nm",
    "coord_mae_A",
    "atom_rmse_A",
    "valid_atom_count_mean",
    "valid_residue_count_mean",
]


def sequence_metrics(
    pred_seq_logits_or_tokens: torch.Tensor,
    gt_seq_tokens: torch.Tensor,
    residue_mask: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Sequence reconstruction metrics.

    Args:
        pred_seq_logits_or_tokens: [B, L, num_classes] float logits, or [B, L]
            integer tokens.
        gt_seq_tokens: [B, L] integer tokens.
        residue_mask: [B, L] bool/float valid-residue mask.
        prefix: metric key prefix.

    Returns:
        dict with `{prefix}/seq_acc` (residue-level accuracy, batch mean of
        per-structure accuracy) and `{prefix}/seq_exact_match` (fraction of
        structures where every valid residue is predicted correctly).
    """
    mask = residue_mask.bool()
    if torch.is_floating_point(pred_seq_logits_or_tokens) and pred_seq_logits_or_tokens.dim() == gt_seq_tokens.dim() + 1:
        pred_tokens = pred_seq_logits_or_tokens.argmax(dim=-1)
    else:
        pred_tokens = pred_seq_logits_or_tokens
    correct = (pred_tokens == gt_seq_tokens) & mask

    n_valid = mask.sum(dim=-1).float()  # [B]
    n_correct = correct.sum(dim=-1).float()  # [B]
    seq_acc_per_struct = n_correct / n_valid.clamp(min=_EPS)
    seq_acc = float(seq_acc_per_struct.mean().item())

    exact_match_per_struct = (correct.sum(dim=-1) == mask.sum(dim=-1)) & (mask.sum(dim=-1) > 0)
    seq_exact_match = float(exact_match_per_struct.float().mean().item())

    return {
        f"{prefix}/seq_acc": seq_acc,
        f"{prefix}/seq_exact_match": seq_exact_match,
    }


SEQ_METRIC_SUFFIXES = ["seq_acc", "seq_exact_match"]


def extract_cyclization_metadata(batch: dict) -> dict[str, torch.Tensor] | None:
    """Extracts CPSea cyclization endpoint/type labels from a batch, if present.

    Supports the CPSea `StructureDataModule` batch fields set by
    `CyclizationLabelTransform` (see `proteinfoundation.datasets.transforms`):
    `cyclization_i`, `cyclization_j` (binder-local endpoint indices, sorted
    i < j), `cyclization_type` (0=mainchain, 1=disulfide, 2=isopeptide, -1
    unknown), and `has_cyclization` (bool, usable label).

    Returns:
        `{"i", "j", "type", "has_cyclization"}`-style dict of tensors (all
        `[B]`), or `None` if the batch does not carry cyclization fields at all
        (e.g. a non-CPSea dataset) -- callers should treat `None` as "log NaN
        for every cyclization metric", not as an error.
    """
    required = ("cyclization_i", "cyclization_j", "cyclization_type")
    if not all(k in batch for k in required):
        return None

    i = batch["cyclization_i"].long()
    j = batch["cyclization_j"].long()
    cyc_type = batch["cyclization_type"].long()
    has_cyclization = batch.get("has_cyclization", None)
    if has_cyclization is None:
        has_cyclization = torch.ones_like(i, dtype=torch.bool)
    else:
        has_cyclization = has_cyclization.bool()

    return {
        "i": i,
        "j": j,
        "type": cyc_type,
        "has_cyclization": has_cyclization,
    }


# Closing-bond acceptance windows, Angstrom. A bond "closed" if the predicted distance
# between the two anchor atoms falls inside its window. Amide C-N is 1.33 A and a
# disulfide S-S is 2.05 A; the windows bracket those with room for prediction error while
# staying narrow enough to exclude an open ring.
#
# These are two-sided ON PURPOSE. The lower bound is not padding: sampled structures do
# fuse the anchor atoms (observed NZ-CG at 0.76 A, well inside any one-sided "close
# enough" test), and a ring welded through itself is as wrong as one left open. Any
# auxiliary bond loss should use the same two-sided shape, or it will optimise straight
# into the lower failure.
MAINCHAIN_CN_BOND_WINDOW_A = (1.1, 1.6)
DISULFIDE_SG_BOND_WINDOW_A = (1.8, 2.3)
ISOPEPTIDE_N_C_BOND_WINDOW_A = (1.1, 1.6)

CYCLIC_METRIC_SUFFIXES = [
    "cyc_cb_dist_gt_A",
    "cyc_cb_dist_pred_A",
    "cyc_cb_dist_abs_err_A",
    "cyc_cb_window_success",
    "cyc_endpoint_atom_rmsd_A",
    "cyc_endpoint_coord_mae_A",
    "mainchain_cn_dist_gt_A",
    "mainchain_cn_dist_pred_A",
    "mainchain_cn_dist_abs_err_A",
    "mainchain_cn_bond_success",
    "disulfide_sg_dist_gt_A",
    "disulfide_sg_dist_pred_A",
    "disulfide_sg_dist_abs_err_A",
    "disulfide_bond_success",
    "isopeptide_n_c_dist_gt_A",
    "isopeptide_n_c_dist_pred_A",
    "isopeptide_n_c_dist_abs_err_A",
    "isopeptide_bond_success",
    # How many examples backed each family's numbers this batch. Every metric above is
    # NaN when its family is absent from the batch, and a per-type family is often
    # absent: disulfide is ~10% of CPSea, so most batches contain none. These counts
    # are what make "disulfide_bond_success = 0.6" readable -- over 2 examples or 200 --
    # and they are what the trainer weights the epoch mean by (see `proteina.py`).
    "n_valid_cyc",
    "n_valid_mainchain",
    "n_valid_disulfide",
    "n_valid_isopeptide",
]

# The tally suffixes above, split out so callers that select a subset of metrics can pull the
# counts along with them. A success rate without its count is unreadable -- "disulfide_bond_success
# = 0.6" over 2 examples and over 200 are different claims -- and the per-type subsets here differ
# by construction (disulfide is only scorable where the structure carries two CYS), so the counts
# are what tell you whether two types are even comparable.
CYCLIC_COUNT_SUFFIXES = [
    "n_valid_cyc",
    "n_valid_mainchain",
    "n_valid_disulfide",
    "n_valid_isopeptide",
]

# Metric suffix -> the count suffix whose validity mask it shares. Used to weight the
# epoch aggregation so a batch with no examples of a family contributes nothing to that
# family's mean, instead of poisoning it with NaN.
CYCLIC_METRIC_COUNT_FOR_SUFFIX = {
    **{
        s: "n_valid_cyc"
        for s in (
            "cyc_cb_dist_gt_A",
            "cyc_cb_dist_pred_A",
            "cyc_cb_dist_abs_err_A",
            "cyc_cb_window_success",
            "cyc_endpoint_atom_rmsd_A",
            "cyc_endpoint_coord_mae_A",
        )
    },
    **{
        s: "n_valid_mainchain"
        for s in (
            "mainchain_cn_dist_gt_A",
            "mainchain_cn_dist_pred_A",
            "mainchain_cn_dist_abs_err_A",
            "mainchain_cn_bond_success",
        )
    },
    **{
        s: "n_valid_disulfide"
        for s in (
            "disulfide_sg_dist_gt_A",
            "disulfide_sg_dist_pred_A",
            "disulfide_sg_dist_abs_err_A",
            "disulfide_bond_success",
        )
    },
    **{
        s: "n_valid_isopeptide"
        for s in (
            "isopeptide_n_c_dist_gt_A",
            "isopeptide_n_c_dist_pred_A",
            "isopeptide_n_c_dist_abs_err_A",
            "isopeptide_bond_success",
        )
    },
}

RECON_METRIC_SUFFIXES = COORD_METRIC_SUFFIXES + SEQ_METRIC_SUFFIXES + CYCLIC_METRIC_SUFFIXES


def _gather_atom(
    atom37: torch.Tensor,
    atom37_mask_bool: torch.Tensor,
    res_idx: torch.Tensor,
    atom_idx: torch.Tensor | int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gathers a single atom position/validity per batch element.

    Args:
        atom37: [B, L, 37, 3].
        atom37_mask_bool: [B, L, 37] bool.
        res_idx: [B] long, residue index per batch element.
        atom_idx: [B] long, or a python int broadcast to all batch elements.

    Returns:
        (pos [B, 3], valid [B] bool)
    """
    batch_idx = torch.arange(atom37.shape[0], device=atom37.device)
    pos = atom37[batch_idx, res_idx, atom_idx]
    valid = atom37_mask_bool[batch_idx, res_idx, atom_idx]
    return pos, valid


def _reduce_valid_to_float(value: torch.Tensor, valid: torch.Tensor) -> float:
    """NaN-aware batch reduction: mean over `valid` entries, NaN elsewhere/if none valid."""
    value = value.float()
    valid = valid.bool()
    filled = torch.where(valid, value, torch.full_like(value, float("nan")))
    return float(torch.nanmean(filled).item())


def cyclic_geometry_metrics(
    pred_atom37: torch.Tensor,
    gt_atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    seq_tokens: torch.Tensor,
    cyclization_metadata: dict[str, torch.Tensor] | None,
    prefix: str,
) -> dict[str, float]:
    """Cyclization-specific geometry metrics; always returns the full key set.

    Args:
        pred_atom37: [B, L, 37, 3], nm.
        gt_atom37: [B, L, 37, 3], nm.
        atom37_mask: [B, L, 37] bool/float, ground-truth valid-coordinate mask
            (used for both pred and gt, since predicted structures are decoded
            at the same residue/atom slots).
        seq_tokens: [B, L] integer ground-truth residue type ids (OpenFold
            `restype_order` convention), used to identify cyclization chemistry
            (CYS/LYS/ASP/GLU) regardless of which structure (`pred`/`gt`) the
            coordinates come from.
        cyclization_metadata: output of `extract_cyclization_metadata`, or
            `None`.
        prefix: metric key prefix.

    Returns:
        dict with every key in `CYCLIC_METRIC_SUFFIXES` (prefixed), filled with
        NaN wherever not applicable (missing metadata, wrong cyclization type,
        unsupported chemistry, or missing atoms) -- never omitted.
    """
    nan_result = {f"{prefix}/{suffix}": float("nan") for suffix in CYCLIC_METRIC_SUFFIXES}
    # Counts are a tally, not a measurement: "no examples" is 0, never NaN. They are the
    # aggregation weights, so a NaN here would propagate straight back into the poisoned
    # epoch mean this is meant to prevent.
    nan_result.update({f"{prefix}/{s}": 0.0 for s in CYCLIC_COUNT_SUFFIXES})
    if cyclization_metadata is None:
        return nan_result

    device = gt_atom37.device
    B, L = gt_atom37.shape[0], gt_atom37.shape[1]

    i = cyclization_metadata["i"].to(device=device).long().clamp(0, max(L - 1, 0))
    j = cyclization_metadata["j"].to(device=device).long().clamp(0, max(L - 1, 0))
    cyc_type = cyclization_metadata["type"].to(device=device).long()
    has_cyc = cyclization_metadata["has_cyclization"].to(device=device).bool()

    mask_bool = atom37_mask.bool()
    batch_idx = torch.arange(B, device=device)

    aa_i = seq_tokens[batch_idx, i]
    aa_j = seq_tokens[batch_idx, j]

    out: dict[str, float] = {}

    # ---- Common: endpoint Cb-Cb distance ----
    gt_res_i, mask_res_i = gt_atom37[batch_idx, i], mask_bool[batch_idx, i]
    gt_res_j, mask_res_j = gt_atom37[batch_idx, j], mask_bool[batch_idx, j]
    pred_res_i = pred_atom37[batch_idx, i]
    pred_res_j = pred_atom37[batch_idx, j]

    cb_gt_i, _ = get_cb_position(gt_res_i, mask_res_i)
    cb_gt_j, _ = get_cb_position(gt_res_j, mask_res_j)
    # Predicted CB falls back to a virtual CB built from the *predicted*
    # backbone when the real CB slot is masked out in the (ground-truth-
    # derived) atom mask, e.g. glycine.
    cb_pred_i, _ = get_cb_position(pred_res_i, mask_res_i)
    cb_pred_j, _ = get_cb_position(pred_res_j, mask_res_j)

    cyc_cb_dist_gt = torch.linalg.norm(cb_gt_i - cb_gt_j, dim=-1) * NM_TO_ANG
    cyc_cb_dist_pred = torch.linalg.norm(cb_pred_i - cb_pred_j, dim=-1) * NM_TO_ANG
    cyc_cb_abs_err = (cyc_cb_dist_pred - cyc_cb_dist_gt).abs()
    cyc_cb_success = (cyc_cb_dist_pred >= 3.0) & (cyc_cb_dist_pred <= 8.0)

    valid_common = has_cyc
    out[f"{prefix}/cyc_cb_dist_gt_A"] = _reduce_valid_to_float(cyc_cb_dist_gt, valid_common)
    out[f"{prefix}/cyc_cb_dist_pred_A"] = _reduce_valid_to_float(cyc_cb_dist_pred, valid_common)
    out[f"{prefix}/cyc_cb_dist_abs_err_A"] = _reduce_valid_to_float(cyc_cb_abs_err, valid_common)
    out[f"{prefix}/cyc_cb_window_success"] = _reduce_valid_to_float(cyc_cb_success.float(), valid_common)

    # ---- Common: endpoint-only atom RMSD/MAE ----
    is_endpoint = torch.zeros(B, L, dtype=torch.bool, device=device)
    is_endpoint[batch_idx, i] = True
    is_endpoint[batch_idx, j] = True
    endpoint_atom_mask = mask_bool & is_endpoint[..., None]
    endpoint_errors = _per_structure_coord_errors(pred_atom37, gt_atom37, endpoint_atom_mask)
    out[f"{prefix}/cyc_endpoint_atom_rmsd_A"] = _reduce_valid_to_float(endpoint_errors["atom_rmse_A"], valid_common)
    out[f"{prefix}/cyc_endpoint_coord_mae_A"] = _reduce_valid_to_float(endpoint_errors["coord_mae_A"], valid_common)

    # ---- Mainchain (head-to-tail): C of lower endpoint <-> N of higher endpoint ----
    # `i < j` is guaranteed by the CPSea label convention, but we also log the
    # flipped orientation and take the min distance in case that convention
    # does not hold for some sample.
    c_i, c_i_valid = _gather_atom(gt_atom37, mask_bool, i, C_IDX)
    n_j, n_j_valid = _gather_atom(gt_atom37, mask_bool, j, N_IDX)
    n_i, n_i_valid = _gather_atom(gt_atom37, mask_bool, i, N_IDX)
    c_j, c_j_valid = _gather_atom(gt_atom37, mask_bool, j, C_IDX)
    dist_o1_gt = torch.linalg.norm(c_i - n_j, dim=-1) * NM_TO_ANG
    dist_o2_gt = torch.linalg.norm(n_i - c_j, dim=-1) * NM_TO_ANG
    mainchain_dist_gt = torch.minimum(dist_o1_gt, dist_o2_gt)

    c_i_p, _ = _gather_atom(pred_atom37, mask_bool, i, C_IDX)
    n_j_p, _ = _gather_atom(pred_atom37, mask_bool, j, N_IDX)
    n_i_p, _ = _gather_atom(pred_atom37, mask_bool, i, N_IDX)
    c_j_p, _ = _gather_atom(pred_atom37, mask_bool, j, C_IDX)
    dist_o1_pred = torch.linalg.norm(c_i_p - n_j_p, dim=-1) * NM_TO_ANG
    dist_o2_pred = torch.linalg.norm(n_i_p - c_j_p, dim=-1) * NM_TO_ANG
    mainchain_dist_pred = torch.minimum(dist_o1_pred, dist_o2_pred)

    mainchain_abs_err = (mainchain_dist_pred - mainchain_dist_gt).abs()
    mainchain_success = (mainchain_dist_pred >= MAINCHAIN_CN_BOND_WINDOW_A[0]) & (
        mainchain_dist_pred <= MAINCHAIN_CN_BOND_WINDOW_A[1]
    )
    mainchain_atoms_valid = c_i_valid & n_j_valid & n_i_valid & c_j_valid
    valid_mainchain = has_cyc & (cyc_type == MAINCHAIN) & mainchain_atoms_valid

    out[f"{prefix}/mainchain_cn_dist_gt_A"] = _reduce_valid_to_float(mainchain_dist_gt, valid_mainchain)
    out[f"{prefix}/mainchain_cn_dist_pred_A"] = _reduce_valid_to_float(mainchain_dist_pred, valid_mainchain)
    out[f"{prefix}/mainchain_cn_dist_abs_err_A"] = _reduce_valid_to_float(mainchain_abs_err, valid_mainchain)
    out[f"{prefix}/mainchain_cn_bond_success"] = _reduce_valid_to_float(mainchain_success.float(), valid_mainchain)

    # ---- Disulfide: SG-SG, both endpoints must be CYS ----
    sg_i_gt, sg_i_valid = _gather_atom(gt_atom37, mask_bool, i, SG_IDX)
    sg_j_gt, sg_j_valid = _gather_atom(gt_atom37, mask_bool, j, SG_IDX)
    sg_i_pred, _ = _gather_atom(pred_atom37, mask_bool, i, SG_IDX)
    sg_j_pred, _ = _gather_atom(pred_atom37, mask_bool, j, SG_IDX)
    disulfide_dist_gt = torch.linalg.norm(sg_i_gt - sg_j_gt, dim=-1) * NM_TO_ANG
    disulfide_dist_pred = torch.linalg.norm(sg_i_pred - sg_j_pred, dim=-1) * NM_TO_ANG
    disulfide_abs_err = (disulfide_dist_pred - disulfide_dist_gt).abs()
    disulfide_success = (disulfide_dist_pred >= DISULFIDE_SG_BOND_WINDOW_A[0]) & (
        disulfide_dist_pred <= DISULFIDE_SG_BOND_WINDOW_A[1]
    )

    is_cys_pair = (aa_i == AA_CYS) & (aa_j == AA_CYS)
    valid_disulfide = has_cyc & (cyc_type == DISULFIDE) & is_cys_pair & sg_i_valid & sg_j_valid

    out[f"{prefix}/disulfide_sg_dist_gt_A"] = _reduce_valid_to_float(disulfide_dist_gt, valid_disulfide)
    out[f"{prefix}/disulfide_sg_dist_pred_A"] = _reduce_valid_to_float(disulfide_dist_pred, valid_disulfide)
    out[f"{prefix}/disulfide_sg_dist_abs_err_A"] = _reduce_valid_to_float(disulfide_abs_err, valid_disulfide)
    out[f"{prefix}/disulfide_bond_success"] = _reduce_valid_to_float(disulfide_success.float(), valid_disulfide)

    # ---- Isopeptide/lactam: LYS(NZ) <-> ASP(CG)/GLU(CD) ----
    # Supports Lys-Asp and Lys-Glu only (per spec); ASN/GLN or other chemistry
    # is left as NaN (unlike the coarse classification-only validity mask in
    # `proteinfoundation.cyclization.mask`, which optionally allows ASN/GLN).
    i_is_lys = aa_i == AA_LYS
    j_is_lys = aa_j == AA_LYS
    i_is_acid = (aa_i == AA_ASP) | (aa_i == AA_GLU)
    j_is_acid = (aa_j == AA_ASP) | (aa_j == AA_GLU)
    case_i_lys = i_is_lys & j_is_acid
    case_j_lys = j_is_lys & i_is_acid
    valid_chem = case_i_lys | case_j_lys

    acid_atom_j_idx = torch.where(aa_j == AA_ASP, torch.full_like(j, CG_IDX), torch.full_like(j, CD_IDX))
    nz_i, nz_i_valid = _gather_atom(gt_atom37, mask_bool, i, torch.full_like(i, NZ_IDX))
    acid_j_gt, acid_j_valid = _gather_atom(gt_atom37, mask_bool, j, acid_atom_j_idx)
    dist_a_gt = torch.linalg.norm(nz_i - acid_j_gt, dim=-1) * NM_TO_ANG
    nz_i_pred, _ = _gather_atom(pred_atom37, mask_bool, i, torch.full_like(i, NZ_IDX))
    acid_j_pred, _ = _gather_atom(pred_atom37, mask_bool, j, acid_atom_j_idx)
    dist_a_pred = torch.linalg.norm(nz_i_pred - acid_j_pred, dim=-1) * NM_TO_ANG
    valid_a = nz_i_valid & acid_j_valid

    acid_atom_i_idx = torch.where(aa_i == AA_ASP, torch.full_like(i, CG_IDX), torch.full_like(i, CD_IDX))
    nz_j, nz_j_valid = _gather_atom(gt_atom37, mask_bool, j, torch.full_like(j, NZ_IDX))
    acid_i_gt, acid_i_valid = _gather_atom(gt_atom37, mask_bool, i, acid_atom_i_idx)
    dist_b_gt = torch.linalg.norm(nz_j - acid_i_gt, dim=-1) * NM_TO_ANG
    nz_j_pred, _ = _gather_atom(pred_atom37, mask_bool, j, torch.full_like(j, NZ_IDX))
    acid_i_pred, _ = _gather_atom(pred_atom37, mask_bool, i, acid_atom_i_idx)
    dist_b_pred = torch.linalg.norm(nz_j_pred - acid_i_pred, dim=-1) * NM_TO_ANG
    valid_b = nz_j_valid & acid_i_valid

    isopeptide_dist_gt = torch.where(case_i_lys, dist_a_gt, dist_b_gt)
    isopeptide_dist_pred = torch.where(case_i_lys, dist_a_pred, dist_b_pred)
    isopeptide_atoms_valid = torch.where(case_i_lys, valid_a, valid_b)
    isopeptide_abs_err = (isopeptide_dist_pred - isopeptide_dist_gt).abs()
    isopeptide_success = (isopeptide_dist_pred >= ISOPEPTIDE_N_C_BOND_WINDOW_A[0]) & (
        isopeptide_dist_pred <= ISOPEPTIDE_N_C_BOND_WINDOW_A[1]
    )

    valid_isopeptide = has_cyc & (cyc_type == ISOPEPTIDE) & valid_chem & isopeptide_atoms_valid

    out[f"{prefix}/isopeptide_n_c_dist_gt_A"] = _reduce_valid_to_float(isopeptide_dist_gt, valid_isopeptide)
    out[f"{prefix}/isopeptide_n_c_dist_pred_A"] = _reduce_valid_to_float(isopeptide_dist_pred, valid_isopeptide)
    out[f"{prefix}/isopeptide_n_c_dist_abs_err_A"] = _reduce_valid_to_float(isopeptide_abs_err, valid_isopeptide)
    out[f"{prefix}/isopeptide_bond_success"] = _reduce_valid_to_float(isopeptide_success.float(), valid_isopeptide)

    out[f"{prefix}/n_valid_cyc"] = float(valid_common.sum().item())
    out[f"{prefix}/n_valid_mainchain"] = float(valid_mainchain.sum().item())
    out[f"{prefix}/n_valid_disulfide"] = float(valid_disulfide.sum().item())
    out[f"{prefix}/n_valid_isopeptide"] = float(valid_isopeptide.sum().item())

    return out


def _select_by_type(
    cyc_type: torch.Tensor,
    mainchain_val: torch.Tensor,
    disulfide_val: torch.Tensor,
    isopeptide_val: torch.Tensor,
) -> torch.Tensor:
    """Per-sample pick among three type-indexed [B] tensors, keyed by `cyc_type`.

    Any type outside {MAINCHAIN, DISULFIDE, ISOPEPTIDE} falls through to the
    isopeptide branch; callers must gate such rows out separately (see
    `known_type` in `per_sample_requested_bond_distance`).
    """
    out = torch.where(cyc_type == MAINCHAIN, mainchain_val, isopeptide_val)
    return torch.where(cyc_type == DISULFIDE, disulfide_val, out)


def per_sample_requested_bond_distance(
    pred_atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    seq_tokens: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    cyc_type: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Closing-bond anchor distance (Angstrom) for each sample's own requested type.

    `cyclic_geometry_metrics` above computes every type's distance for every sample
    and reduces to per-type logging means. This computes the single distance the
    sample's *requested* linkage actually calls for, un-reduced and differentiable,
    which is what an auxiliary bond loss needs. It reuses this module's atom-slot
    constants, the identical anchor choices (mainchain N(i)<->C(j), the single
    head-to-tail orientation given `i`/`j` are the sorted first/last binder-local
    indices -- matching what `evaluation.cyclization_pdb_metrics` measures on the
    decoded PDB -- disulfide SG<->SG gated on a CYS pair, isopeptide LYS:NZ<->ASP:CG
    / GLU:CD), and the identical `*_BOND_WINDOW_A` acceptance windows -- so a loss
    built on this cannot drift away from the closure metric it is meant to move.

    `atom37_mask` and `seq_tokens` must describe the SAME structure whose coordinates
    are in `pred_atom37` (i.e. the decoded prediction, not the ground truth): an
    anchor atom counts as present only if that structure actually carries it, and the
    chemistry checks must agree with the coordinates they gate. `i`, `j` and
    `cyc_type`, by contrast, are the *requested* cyclization -- the ground-truth label
    at training time, termini plus the requested type at design time -- and are
    constant across the flow time `t`.

    Args:
        pred_atom37: [B, L, 37, 3], nm.
        atom37_mask: [B, L, 37], bool/float atom-presence mask.
        seq_tokens: [B, L], integer residue ids (OpenFold `restype_order`).
        i, j: [B] long, closing-bond endpoints (binder-local indices).
        cyc_type: [B] long, requested linkage type per sample.

    Returns:
        dict of [B] tensors: `dist_A` (differentiable w.r.t. `pred_atom37`),
        `window_lo_A`, `window_hi_A`, and `atoms_valid` (bool; `dist_A` is
        meaningless wherever this is False).
    """
    device = pred_atom37.device
    B, L = pred_atom37.shape[0], pred_atom37.shape[1]

    mask_bool = atom37_mask.bool()
    i = i.to(device=device).long().clamp(0, max(L - 1, 0))
    j = j.to(device=device).long().clamp(0, max(L - 1, 0))
    cyc_type = cyc_type.to(device=device).long()
    seq_tokens = seq_tokens.to(device=device).long()

    batch_idx = torch.arange(B, device=device)
    aa_i = seq_tokens[batch_idx, i]
    aa_j = seq_tokens[batch_idx, j]

    # ---- Mainchain (head-to-tail): N of the first (lower-index) endpoint <-> C of
    # the last (higher-index) endpoint. `i < j` is guaranteed by the CPSea label
    # convention (`parse_labels.py` sorts endpoints), so this is the single real
    # closing bond -- not a min over both orientations, which would let a
    # coincidentally-close but chemically meaningless C(i)-N(j) pair (both atoms
    # already engaged in in-chain peptide bonds) masquerade as ring closure. This
    # must match `evaluation.cyclization_pdb_metrics`, which measures the same
    # N(first)-C(last) pair on the decoded PDB.
    n_i, n_i_valid = _gather_atom(pred_atom37, mask_bool, i, N_IDX)
    c_j, c_j_valid = _gather_atom(pred_atom37, mask_bool, j, C_IDX)
    mainchain_dist = torch.linalg.norm(n_i - c_j, dim=-1) * NM_TO_ANG
    mainchain_valid = n_i_valid & c_j_valid

    # ---- Disulfide: SG-SG, both endpoints must be CYS ----
    sg_i, sg_i_valid = _gather_atom(pred_atom37, mask_bool, i, SG_IDX)
    sg_j, sg_j_valid = _gather_atom(pred_atom37, mask_bool, j, SG_IDX)
    disulfide_dist = torch.linalg.norm(sg_i - sg_j, dim=-1) * NM_TO_ANG
    disulfide_valid = (aa_i == AA_CYS) & (aa_j == AA_CYS) & sg_i_valid & sg_j_valid

    # ---- Isopeptide/lactam: LYS(NZ) <-> ASP(CG)/GLU(CD) ----
    case_i_lys = (aa_i == AA_LYS) & ((aa_j == AA_ASP) | (aa_j == AA_GLU))
    case_j_lys = (aa_j == AA_LYS) & ((aa_i == AA_ASP) | (aa_i == AA_GLU))

    acid_atom_j_idx = torch.where(aa_j == AA_ASP, torch.full_like(j, CG_IDX), torch.full_like(j, CD_IDX))
    nz_i, nz_i_valid = _gather_atom(pred_atom37, mask_bool, i, torch.full_like(i, NZ_IDX))
    acid_j, acid_j_valid = _gather_atom(pred_atom37, mask_bool, j, acid_atom_j_idx)
    dist_a = torch.linalg.norm(nz_i - acid_j, dim=-1) * NM_TO_ANG

    acid_atom_i_idx = torch.where(aa_i == AA_ASP, torch.full_like(i, CG_IDX), torch.full_like(i, CD_IDX))
    nz_j, nz_j_valid = _gather_atom(pred_atom37, mask_bool, j, torch.full_like(j, NZ_IDX))
    acid_i, acid_i_valid = _gather_atom(pred_atom37, mask_bool, i, acid_atom_i_idx)
    dist_b = torch.linalg.norm(nz_j - acid_i, dim=-1) * NM_TO_ANG

    isopeptide_dist = torch.where(case_i_lys, dist_a, dist_b)
    isopeptide_valid = (case_i_lys | case_j_lys) & torch.where(
        case_i_lys, nz_i_valid & acid_j_valid, nz_j_valid & acid_i_valid
    )

    # ---- Select the requested type's distance / window / validity ----
    dist_A = _select_by_type(cyc_type, mainchain_dist, disulfide_dist, isopeptide_dist)
    atoms_valid = _select_by_type(cyc_type, mainchain_valid, disulfide_valid, isopeptide_valid)

    def _window(idx: int) -> torch.Tensor:
        return _select_by_type(
            cyc_type,
            torch.full((B,), MAINCHAIN_CN_BOND_WINDOW_A[idx], device=device),
            torch.full((B,), DISULFIDE_SG_BOND_WINDOW_A[idx], device=device),
            torch.full((B,), ISOPEPTIDE_N_C_BOND_WINDOW_A[idx], device=device),
        )

    # NO_CYCLIZATION_INDEX (-1) and any other stray label select nothing meaningful.
    known_type = (cyc_type == MAINCHAIN) | (cyc_type == DISULFIDE) | (cyc_type == ISOPEPTIDE)

    return {
        "dist_A": dist_A,
        "window_lo_A": _window(0),
        "window_hi_A": _window(1),
        "atoms_valid": atoms_valid.bool() & known_type,
    }
