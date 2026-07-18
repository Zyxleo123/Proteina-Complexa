"""Validity mask for candidate cyclization `(i, j, type)` triples.

This module only encodes coarse, hand-authored chemical *plausibility* rules
(e.g. "disulfides require two cysteines") used to restrict the softmax
candidate set. It never inspects or modifies 3D coordinates.
"""

import torch

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
    NUM_CYCLIZATION_TYPES,
    UNSPECIFIED,
)


def build_cyclization_validity_mask(
    aa: torch.Tensor,
    binder_mask: torch.Tensor,
    gold_i: torch.Tensor | None = None,
    gold_j: torch.Tensor | None = None,
    gold_type: torch.Tensor | None = None,
    force_gold_valid: bool = False,
    allow_asn_gln_isopeptide: bool = True,
    cond_type: torch.Tensor | None = None,
    terminal_only: bool = False,
) -> torch.Tensor:
    """Builds a bool validity mask over candidate `(i, j, type)` triples.

    Args:
        aa: [B, L] integer AA ids (ground-truth during training, predicted
            during inference).
        binder_mask: [B, L] bool, True for real (non-padding) binder residues.
        gold_i, gold_j: [B] gold cyclization endpoints (binder-local index),
            only required if `force_gold_valid=True`. Order does not matter;
            they are sorted internally.
        gold_type: [B] gold linkage type, only required if `force_gold_valid=True`.
        force_gold_valid: If True, forces the gold `(i, j, type)` entry to be
            valid regardless of the hand-coded chemistry rules below. This is
            important during training so an imperfect rule set never masks
            out the true label. Values are clamped into range, so this is
            always safe to call even for samples without a gold label (as
            long as the caller excludes those samples from the loss).
        allow_asn_gln_isopeptide: If True, also allow ASN/GLN as isopeptide
            partners (some CPSea isopeptide labels use amide side chains
            rather than pure acids).
        cond_type: [B] requested cyclization type, values in `0..UNSPECIFIED`.
            For rows with a real type, every other type slice is zeroed out, so
            the downstream global softmax over the flattened candidate space
            becomes exactly `p(i, j | type)` -- restricting the support of a
            softmax and renormalizing *is* conditioning, which is why the head
            itself needs no type input. Rows at `UNSPECIFIED` (and `cond_type=None`)
            keep the full 3-type candidate set, recovering the unconditional
            joint `p(i, j, type)`.
        terminal_only: Restrict *every* type to the terminal pair `(0, L_b - 1)`,
            which MAINCHAIN is already hard-coded to below.

            Measured on CPSea: **520/520 sampled binders cyclize between the
            termini -- for all three types** (incl. 55/55 disulfides). So the
            endpoints are not something the model discovers, they are a constant,
            and leaving DISULFIDE/ISOPEPTIDE free to roam every CYS-CYS / LYS-acid
            pair is not extra expressiveness, it is an escape hatch: at inference
            the head argmaxes with Ca coordinates in hand, so it hunts down
            whichever pair is *already* closest to a bond. In a compact 10-16mer
            some pair of cysteines nearly always is, which is why disulfide
            closure reads far better than mainchain despite being no more
            predictable in location -- the metric rewards finding two nearby
            cysteines, not closing the ring. MAINCHAIN has no such hatch, which is
            the whole of its apparent disadvantage.

            Off by default (old behavior). Training is unaffected either way:
            `force_gold_valid` punches the gold entry through afterwards, so even
            a hypothetical non-terminal label (<1% by the above bound) is never
            masked out of the loss.

    Returns:
        Bool tensor of shape [B, L, L, 3]. True means candidate (i, j, type)
        is allowed. Only the strict upper triangle (i < j) can be valid.
    """
    B, L = aa.shape
    device = aa.device
    binder_mask = binder_mask.bool()

    valid = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES, dtype=torch.bool, device=device)

    real_pair = binder_mask[:, :, None] & binder_mask[:, None, :]  # [B, L, L]
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)  # [L, L]
    real_pair = real_pair & upper[None, :, :]

    # The terminal pair (0, L_b - 1) per sample: the only pair CPSea ever cyclizes
    # (see `terminal_only`), and by definition the only one a head-to-tail bond can use.
    lengths = binder_mask.long().sum(dim=1)  # [B]
    terminal_pair = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    for b in range(B):
        length_b = int(lengths[b].item())
        if length_b >= 2:
            terminal_pair[b, 0, length_b - 1] = True

    # MAINCHAIN: only the terminal pair per sample is valid.
    valid[..., MAINCHAIN] = terminal_pair

    # DISULFIDE: both residues must be CYS.
    is_cys = aa == AA_CYS
    valid[..., DISULFIDE] = real_pair & is_cys[:, :, None] & is_cys[:, None, :]

    # ISOPEPTIDE: one LYS, one acid/amide side chain (ASP/GLU, optionally ASN/GLN).
    is_lys = aa == AA_LYS
    is_acid_or_amide = (aa == AA_ASP) | (aa == AA_GLU)
    if allow_asn_gln_isopeptide:
        is_acid_or_amide = is_acid_or_amide | (aa == AA_ASN) | (aa == AA_GLN)
    valid[..., ISOPEPTIDE] = real_pair & (
        (is_lys[:, :, None] & is_acid_or_amide[:, None, :]) | (is_acid_or_amide[:, :, None] & is_lys[:, None, :])
    )

    # Keep only real upper-triangle pairs for all types (also re-applies to MAINCHAIN,
    # which is a no-op unless binder_mask is inconsistent with `lengths`).
    valid = valid & real_pair[..., None]

    # Hold DISULFIDE/ISOPEPTIDE to the same terminal pair MAINCHAIN is already held to.
    # Applied before `force_gold_valid` so a non-terminal gold label is still punched
    # through and never masked out of the training loss.
    if terminal_only:
        valid = valid & terminal_pair[..., None]

    # Type conditioning: restrict the candidate set to the requested type slice.
    # Applied before `force_gold_valid` so that the gold entry is still punched
    # through afterwards -- during training `cond_type` equals `gold_type` on every
    # non-dropped row, so the two never contradict each other.
    if cond_type is not None:
        type_ids = torch.arange(NUM_CYCLIZATION_TYPES, device=device)  # [3]
        requested = cond_type[:, None] == type_ids[None, :]  # [B, 3]
        unconditional = (cond_type == UNSPECIFIED)[:, None]  # [B, 1]
        allowed_types = requested | unconditional  # [B, 3]
        valid = valid & allowed_types[:, None, None, :]

    if force_gold_valid:
        if gold_i is None or gold_j is None or gold_type is None:
            raise ValueError("force_gold_valid=True requires gold_i, gold_j, gold_type")
        # Clamp defensively: samples without a real gold label (e.g. has_cyclication=False)
        # may carry sentinel/placeholder values. Callers are expected to exclude such
        # samples from the loss; clamping here just prevents out-of-range/negative-index
        # surprises rather than being semantically meaningful for those rows.
        gi = torch.minimum(gold_i, gold_j).clamp(min=0, max=L - 1)
        gj = torch.maximum(gold_i, gold_j).clamp(min=0, max=L - 1)
        gt = gold_type.clamp(min=0, max=NUM_CYCLIZATION_TYPES - 1)
        batch_idx = torch.arange(B, device=device)
        valid[batch_idx, gi, gj, gt] = True

    return valid
