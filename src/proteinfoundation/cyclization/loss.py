"""Loss and inference-decoding for the CPSea cyclization-linkage head.

There is exactly one true cyclization edge per cyclic peptide binder, so
training uses a single global softmax / cross-entropy over all valid
`(i, j, type)` candidates -- never per-pair binary cross-entropy. We rely on
`F.cross_entropy`'s built-in log-softmax rather than manually applying
softmax before the loss; softmax is only used for inference probabilities
and logging.
"""

import torch
import torch.nn.functional as F

from proteinfoundation.cyclization.constants import (
    CYCLIZATION_TYPE_TO_NAME,
    MAINCHAIN,
    NO_CYCLIZATION_INDEX,
    NUM_CYCLIZATION_TYPES,
)

_NEG_INF = -1e9


def cyclization_link_loss(
    link_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    gold_i: torch.Tensor,
    gold_j: torch.Tensor,
    gold_type: torch.Tensor,
    has_cyclization: torch.Tensor,
    pre_force_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Global softmax/CE loss over all valid `(i, j, type)` candidates.

    Args:
        link_logits: [B, L, L, 3] symmetrized typed pair logits.
        valid_mask: [B, L, L, 3] bool, candidates allowed to compete in the softmax.
            This is the mask actually used for the CE loss below, and -- when the
            caller built it with `force_gold_valid=True` (the default in
            `proteina.py`) -- the gold entry is unconditionally `True` in it by
            construction, regardless of the hand-authored chemistry rules in
            `cyclization.mask`.
        gold_i, gold_j: [B] gold cyclization endpoints (binder-local index, any order).
        gold_type: [B] gold linkage type.
        has_cyclization: [B] bool, whether this sample has a usable gold label.
        pre_force_valid_mask: Optional [B, L, L, 3] bool, the SAME mask built with
            `force_gold_valid=False` -- i.e. before the gold entry was unconditionally
            punched through. Used only for the `gold_valid_before_force_frac`
            diagnostic below. If omitted, that diagnostic falls back to reading
            `valid_mask` directly, which reports a meaningless flat 1.0 whenever the
            caller used `force_gold_valid=True` (checking a mask against an entry
            that mask was just forced to contain). Passing this in is what makes the
            diagnostic able to expose an invalid chemistry rule at all.

    Returns:
        (loss, metrics): scalar CE loss (differentiable) and a dict of
        python-float diagnostic metrics (top1 accuracies, mean gold prob, etc).
    """
    B, L, _, T = link_logits.shape
    metrics: dict[str, float] = {}

    valid_examples = has_cyclization.bool()
    n_valid = int(valid_examples.sum().item())
    metrics["n_valid_examples"] = float(n_valid)
    if n_valid == 0:
        # No usable labels in this batch: return an exact zero that still
        # participates in autograd (so DDP / mixed dataloaders don't choke on
        # a detached loss), but touches no real gradients otherwise.
        return link_logits.sum() * 0.0, metrics

    link_logits_v = link_logits[valid_examples]
    valid_mask_v = valid_mask[valid_examples]
    gold_i_v = gold_i[valid_examples].long()
    gold_j_v = gold_j[valid_examples].long()
    gold_type_v = gold_type[valid_examples].long()

    gi = torch.minimum(gold_i_v, gold_j_v)
    gj = torch.maximum(gold_i_v, gold_j_v)

    batch_idx = torch.arange(gi.shape[0], device=link_logits.device)
    # Read the pre-force mask when the caller gives us one, so this diagnostic can
    # actually differ from 1.0 -- reading `valid_mask` itself here is meaningless
    # whenever the caller used `force_gold_valid=True`, since that mask was just
    # unconditionally stamped True at exactly this (gi, gj, gold_type) entry.
    diagnostic_mask_v = (
        pre_force_valid_mask[valid_examples] if pre_force_valid_mask is not None else valid_mask_v
    )
    gold_was_valid_before_force = diagnostic_mask_v[batch_idx, gi, gj, gold_type_v]
    metrics["gold_valid_before_force_frac"] = gold_was_valid_before_force.float().mean().item()

    masked_logits = link_logits_v.masked_fill(~valid_mask_v, _NEG_INF)
    flat_logits = masked_logits.reshape(masked_logits.shape[0], -1)  # [n_valid, L*L*3]
    gold_index = (gi * L + gj) * NUM_CYCLIZATION_TYPES + gold_type_v

    # F.cross_entropy applies log-softmax internally -- do not pre-softmax.
    loss = F.cross_entropy(flat_logits, gold_index)

    with torch.no_grad():
        probs = flat_logits.softmax(dim=-1)
        pred_flat = flat_logits.argmax(dim=-1)
        pred_type = pred_flat % NUM_CYCLIZATION_TYPES
        pred_pair = pred_flat // NUM_CYCLIZATION_TYPES
        pred_i = pred_pair // L
        pred_j = pred_pair % L

        exact_correct = pred_flat == gold_index
        pair_correct = (pred_i == gi) & (pred_j == gj)
        type_correct = pred_type == gold_type_v

        metrics["top1_exact_acc"] = exact_correct.float().mean().item()
        metrics["top1_pair_acc"] = pair_correct.float().mean().item()
        metrics["top1_type_acc"] = type_correct.float().mean().item()
        metrics["mean_gold_prob"] = probs.gather(-1, gold_index[:, None]).squeeze(-1).mean().item()

        for t, name in CYCLIZATION_TYPE_TO_NAME.items():
            type_mask = gold_type_v == t
            n_t = int(type_mask.sum().item())
            metrics[f"n_gold_{name}"] = float(n_t)
            if n_t > 0:
                metrics[f"top1_exact_acc_{name}"] = exact_correct[type_mask].float().mean().item()

    return loss, metrics


def decode_cyclization_prediction(
    link_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    fallback: bool = True,
) -> dict[str, torch.Tensor]:
    """Decodes the most likely `(i, j, type)` cyclization edge per sample.

    A sample can have no valid candidate at all -- with the predicted-AA mask at
    inference time, e.g. a binder with no CYS/LYS/acid residues, or a requested
    type the generated sequence cannot support. `fallback` decides what happens
    then:

    - `fallback=True` (unconditional path): allow the terminal MAINCHAIN
      candidate `(0, L_b - 1, MAINCHAIN)` so inference never crashes.
    - `fallback=False` (type-conditioned path): emit a null edge (`-1, -1, -1`)
      with zero confidence. Falling back to MAINCHAIN here would silently return
      a *different* type than the one the caller asked for, i.e. it would claim
      to have honored a request it did not. Callers should read
      `cyclization_type_satisfied` (see `inference.py`) rather than assume the
      returned edge is always meaningful.

    Args:
        link_logits: [B, L, L, 3] symmetrized typed pair logits.
        valid_mask: [B, L, L, 3] bool.
        fallback: Whether to fall back to terminal MAINCHAIN for samples with an
            empty candidate set.

    Returns:
        Dict with `pred_cyclization_i`, `pred_cyclization_j` (both [B] long,
        binder-local indices, i < j), `pred_cyclization_type` ([B] long), and
        `pred_cyclization_confidence` ([B] float, softmax probability mass on
        the predicted candidate). Unsatisfiable rows are `-1` / `0.0` when
        `fallback=False`.
    """
    B, L, _, T = link_logits.shape

    has_any_valid = valid_mask.reshape(B, -1).any(dim=-1)
    if fallback and (~has_any_valid).any():
        valid_mask = valid_mask.clone()
        fallback_j = max(L - 1, 0)
        valid_mask[~has_any_valid, 0, fallback_j, MAINCHAIN] = True
        has_any_valid = valid_mask.reshape(B, -1).any(dim=-1)

    masked_logits = link_logits.masked_fill(~valid_mask, _NEG_INF)
    probs = masked_logits.reshape(B, -1).softmax(dim=-1)
    best = probs.argmax(dim=-1)
    confidence = probs.max(dim=-1).values

    pair_type = best % T
    pair_flat = best // T
    i = pair_flat // L
    j = pair_flat % L

    # An all-invalid row produces a uniform softmax over -1e9 logits, so its
    # argmax is meaningless -- null it out rather than reporting index 0.
    if (~has_any_valid).any():
        null = torch.full_like(i, NO_CYCLIZATION_INDEX)
        i = torch.where(has_any_valid, i, null)
        j = torch.where(has_any_valid, j, null)
        pair_type = torch.where(has_any_valid, pair_type, null)
        confidence = torch.where(has_any_valid, confidence, torch.zeros_like(confidence))

    return {
        "pred_cyclization_i": i,
        "pred_cyclization_j": j,
        "pred_cyclization_type": pair_type,
        "pred_cyclization_confidence": confidence,
    }
