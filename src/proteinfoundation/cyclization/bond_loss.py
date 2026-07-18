"""Two-sided (flat-bottom) closing-bond distance loss for CPSea cyclic peptides.

The linkage head next door is a *classifier*: it says which `(i, j, type)` edge
closes the ring, and nothing at all about whether the two anchor atoms end up a
bond length apart. That split shows up directly in the measurements -- the head
sits near 100% while sampled ring closure sits near half -- because nothing in
the objective ever told the model the bond *length* matters. This module is that
missing term.

The penalty is deliberately **two-sided**: zero inside the type's acceptance
window, quadratic outside on *both* sides. A naive `|d - d_ideal|`, and
especially any one-sided "pull the ends together" term, optimises straight into
the opposite failure -- sampled designs have been observed with anchor atoms
*fused* at 0.76-0.87 A, physically impossible next to a real amide C-N of 1.33 A.
A ring welded through itself is exactly as wrong as one left open, and only a
flat-bottom restraint says so. The shape matches AF2's structural-violation loss.

The windows are not redefined here: they are the same `*_BOND_WINDOW_A` constants
the closure metric uses, reached through
`eval.cyclic_reconstruction_metrics.per_sample_requested_bond_distance`, so the
loss and the metric it is meant to move cannot drift apart.
"""

import torch

from proteinfoundation.eval.cyclic_reconstruction_metrics import per_sample_requested_bond_distance

_EPS = 1e-6


def flat_bottom_penalty(
    dist: torch.Tensor,
    window_lo: torch.Tensor,
    window_hi: torch.Tensor,
) -> torch.Tensor:
    """Zero inside `[window_lo, window_hi]`, quadratic outside it on both sides.

    Differentiable in `dist`; the gradient is zero (not merely small) anywhere
    inside the window, so a bond that already closes acceptably is left alone
    instead of being dragged toward an arbitrary point estimate.

    All three arguments broadcast against each other; units must agree (Angstrom
    throughout this module).
    """
    below = torch.clamp(window_lo - dist, min=0.0)
    above = torch.clamp(dist - window_hi, min=0.0)
    return below * below + above * above


def cyclization_bond_loss(
    pred_atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    seq_tokens: torch.Tensor,
    cyclization_metadata: dict[str, torch.Tensor] | None,
    t_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Flat-bottom closing-bond distance loss on a decoded predicted structure.

    Safe no-op (differentiable zero) when the batch carries no cyclization labels,
    so mixed/non-CPSea training never breaks.

    Args:
        pred_atom37: [B, L, 37, 3] nm, the decoded predicted structure. Must carry
            gradient back to the flow model for this loss to do anything.
        atom37_mask: [B, L, 37] atom-presence mask *of that same structure*.
        seq_tokens: [B, L] residue ids *of that same structure* (chemistry checks
            must agree with the coordinates they gate).
        cyclization_metadata: `{"i", "j", "type", "has_cyclization"}` of [B] tensors
            describing the REQUESTED cyclization (see `extract_cyclization_metadata`),
            or None.
        t_weight: optional [B] weight in [0, 1] down-weighting samples at low flow
            time, where the predicted clean structure is meaningless and its bond
            distance carries no signal. None means uniform weight.

    Returns:
        (loss, metrics): a differentiable scalar (weighted mean of the per-sample
        penalty over valid samples) and float diagnostics. `window_success` is the
        same closure criterion the eval reports, restricted to the samples this loss
        actually supervises (label-valid AND not zeroed by `t_weight`); `n_valid`
        counts exactly those and is the weight the caller should aggregate by.
        Diagnostics are NaN when nothing in the batch is supervisable.
    """
    empty_metrics = {"n_valid": 0.0, "mean_dist_A": float("nan"), "window_success": float("nan")}
    if cyclization_metadata is None:
        return pred_atom37.sum() * 0.0, empty_metrics

    device = pred_atom37.device
    has_cyc = cyclization_metadata["has_cyclization"].to(device=device).bool()

    bond = per_sample_requested_bond_distance(
        pred_atom37=pred_atom37,
        atom37_mask=atom37_mask,
        seq_tokens=seq_tokens,
        i=cyclization_metadata["i"],
        j=cyclization_metadata["j"],
        cyc_type=cyclization_metadata["type"],
    )
    dist = bond["dist_A"]  # [B], differentiable
    window_lo, window_hi = bond["window_lo_A"], bond["window_hi_A"]

    valid = has_cyc & bond["atoms_valid"]  # [B]
    if not bool(valid.any()):
        # No usable label this batch: an exact zero that still participates in
        # autograd, so DDP and mixed dataloaders don't choke on a detached loss.
        return pred_atom37.sum() * 0.0, empty_metrics

    penalty = flat_bottom_penalty(dist, window_lo, window_hi)  # [B]

    weight = valid.float()
    if t_weight is not None:
        weight = weight * t_weight.to(device=device).float().clamp(0.0, 1.0)
    weight = weight.detach()

    loss = (penalty * weight).sum() / weight.sum().clamp(min=_EPS)

    with torch.no_grad():
        # Report over the samples the loss actually supervises, NOT every labelled one.
        # A sample the t-ramp has zeroed out is a prediction made from noise; averaging its
        # closure in would make this metric mostly a readout of the sampled `t` distribution
        # rather than of the model. Batches with nothing supervisable report NaN, which the
        # caller logs at zero weight rather than folding into the epoch mean.
        supervised = weight > 0
        if not bool(supervised.any()):
            return loss, empty_metrics
        inside = (dist >= window_lo) & (dist <= window_hi)
        metrics = {
            "n_valid": float(supervised.sum().item()),
            "mean_dist_A": float(dist[supervised].mean().item()),
            "window_success": float(inside[supervised].float().mean().item()),
        }
    return loss, metrics
