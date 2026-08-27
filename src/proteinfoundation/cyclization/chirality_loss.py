"""Differentiable per-residue L-amino-acid chirality penalty.

`cyclization.scoring._chirality_valid_per_residue` already checks this at
replay-collection time to GATE candidates (`--gate-chirality`), but a gate can
only reject what a checkpoint already produces -- it supplies no gradient that
would teach the flow model to stop producing D-configured residues in the
first place. This module supplies that gradient: a smooth penalty whose zero
level set is exactly that gate's decision boundary, so the two can never
silently drift apart (`chirality_triple_product` is the single shared
geometry computation both are built on).
"""

from __future__ import annotations

import torch

from proteinfoundation.eval.cyclic_reconstruction_metrics import C_IDX, CA_IDX, CB_IDX, N_IDX


def chirality_triple_product(
    atom37: torch.Tensor, atom37_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(triple, checkable)`, both `[..., L]`.

    `triple = (N-CA) x (C-CA) . (CB-CA)`; `triple > 0` iff the residue is a
    correctly-chiral L-amino-acid (see `cyclization.scoring
    ._chirality_valid_per_residue`'s docstring for the geometric argument --
    `virtual_cb_from_backbone`'s reconstructed CB is always strictly positive
    under this convention for a non-degenerate backbone frame). `checkable`
    is False for glycine or a masked-out slot, which have no CB to check.
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
    return triple, has_backbone & has_cb


def chirality_loss(
    pred_atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    binder_mask: torch.Tensor,
    t_weight: torch.Tensor | None = None,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hinge penalty on D-configured (or too-close-to-the-boundary) residues.

    `penalty_i = relu(margin - triple_i)`: zero once `triple_i >= margin`, so
    `margin=0.0` reproduces `_chirality_valid_per_residue`'s exact decision
    boundary as the zero-penalty region; a small positive margin pushes
    training past the boundary rather than just up to it. Mean-reduced over
    every checkable, in-binder-mask residue in the batch (not per-sample, so
    one bad residue in a long peptide is not diluted away by the rest).

    Safe differentiable zero (still participates in autograd, so DDP does not
    choke when a batch happens to have nothing checkable -- e.g. an
    all-glycine or fully masked-out slice) when nothing is checkable.

    Args:
        pred_atom37: `[B, L, 37, 3]`, nm or Angstrom -- the triple product's
            sign is scale-invariant, so either works.
        atom37_mask: `[B, L, 37]` bool.
        binder_mask: `[B, L]` bool, restricts the penalty to real (non-padding)
            binder residues.
        t_weight: Optional `[B]` in `[0, 1]`, same "predicted clean structure
            is meaningless at high noise" ramp `cyclization_bond_loss` uses --
            multiplies the per-residue weight before reduction.
        margin: See above.
    """
    triple, checkable = chirality_triple_product(pred_atom37, atom37_mask)
    weight = (checkable & binder_mask.bool()).float()
    if t_weight is not None:
        weight = weight * t_weight[:, None]

    n_valid = weight.sum()
    if float(n_valid.item()) <= 0.0:
        return pred_atom37.sum() * 0.0, {"n_valid": 0.0, "frac_d": float("nan")}

    penalty = torch.relu(margin - triple)
    loss = (penalty * weight).sum() / n_valid

    with torch.no_grad():
        is_d = (triple <= 0.0) & (checkable & binder_mask.bool())
        frac_d = float(is_d.float().sum().item()) / float(n_valid.item())

    return loss, {"n_valid": float(n_valid.item()), "frac_d": frac_d}
