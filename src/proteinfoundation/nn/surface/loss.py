"""Geometry-only auxiliary losses tying the predicted binder to a conditioned surface.

Both modes operate on the decoded binder virtual-Cβ point cloud vs. a fixed-size surface
patch, in nm and a shared frame. They differ only in what pose they are minimized at:

``chamfer`` (legacy)
    Symmetric soft-Chamfer, minimized when binder Cβ *coincide* with the surface points.
    Correct only when the surface is the binder's OWN molecular surface
    (``AttachPeptideSurfaceTransform``): the binder's Cβ should indeed lie on the binder's
    own surface. With a RECEPTOR pocket surface (``AttachReceptorSurfaceTransform``) the
    same math asks the binder to occupy the receptor's surface — i.e. to interpenetrate
    the receptor — so its minimum is physically unreachable and the loss rises as the model
    learns the correct adjacent-not-overlapping pose.

``contact``
    Normal-aware complementarity, minimized when the binder sits just OUTSIDE the receptor
    surface (positive signed offset along the outward-pointing normal) and lines the
    extracted interface patch. Two independent one-sided hinges:

    * **clash** — for each binder Cβ, the signed offset ``(cb − p*)·n̂*`` to its nearest
      surface point ``p*`` (normal ``n̂*``, pointing outward toward the peptide) must be
      ``≥ clash_margin``. Penalizes only binder atoms on the *receptor* side of the surface
      (buried); says nothing about binder atoms that legitimately face away, so it does not
      force every residue to contact.
    * **coverage** — each surface point must have some binder Cβ within ``contact_band``.
      Legitimate here because the receptor cache is ALREADY the interface patch (every
      cached point is within the extraction cutoff of a peptide heavy atom), so this is not
      a spurious "cover the whole receptor" demand — only the contacted patch is stored.

    Both hinges reach exactly 0 at a realistic binding pose, so unlike ``chamfer`` this loss
    can actually be driven down rather than fighting the flow-matching objective.

Normals are unit directions in the same rotated frame as the points (``GlobalRotationTransform``
rotates them; ``CoordsToNanometers``/``CenteringTransform`` leave directions untouched), so the
dot products below are frame-consistent with ``cb``.
"""

from __future__ import annotations

import torch

_BIG = 1.0e6


def virtual_cb_surface_chamfer(
    cb: torch.Tensor,
    cb_mask: torch.Tensor,
    surf: torch.Tensor,
    surf_mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric soft-Chamfer between binder Cβ and surface points. Returns ``[B]`` (nm).

    Minimized at coincidence. Legacy behavior; see module docstring for when it is correct.
    """
    dist = (cb[:, :, None, :] - surf[:, None, :, :]).norm(dim=-1)  # [B, N, M]
    dist = dist.masked_fill(~surf_mask[:, None, :], _BIG)
    dist = dist.masked_fill(~cb_mask[:, :, None], _BIG)
    d_bs = dist.min(dim=-1).values  # [B, N] binder -> nearest surface
    d_sb = dist.min(dim=1).values  # [B, M] surface -> nearest binder
    per_b = (d_bs * cb_mask).sum(-1) / cb_mask.sum(-1).clamp_min(1)
    per_s = (d_sb * surf_mask).sum(-1) / surf_mask.sum(-1).clamp_min(1)
    return 0.5 * (per_b + per_s)


def surface_contact_loss(
    cb: torch.Tensor,
    cb_mask: torch.Tensor,
    surf: torch.Tensor,
    surf_normals: torch.Tensor,
    surf_mask: torch.Tensor,
    *,
    contact_band: float = 0.5,
    clash_margin: float = 0.0,
    clash_weight: float = 1.0,
    cover_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normal-aware contact/complementarity loss. Returns ``(per, clash, cover)``, each ``[B]``.

    Args:
        cb: ``[B, N, 3]`` binder virtual-Cβ (nm).
        cb_mask: ``[B, N]`` bool, valid binder residues.
        surf: ``[B, M, 3]`` surface points (nm).
        surf_normals: ``[B, M, 3]`` unit outward normals (pointing away from the receptor,
            toward the peptide).
        surf_mask: ``[B, M]`` bool, valid (non-padded) surface points.
        contact_band: nm; a surface point with a binder Cβ within this radius contributes 0
            to coverage. Should be looser than the extraction cutoff because the cache's
            reference is the nearest peptide *heavy atom*, not specifically Cβ.
        clash_margin: nm; require signed normal offset ``≥ clash_margin``. 0 forbids
            interpenetration; a small positive value asks the binder to sit slightly proud
            of the surface.
        clash_weight, cover_weight: relative weights of the two hinges.

    Returns:
        per: ``clash_weight * clash + cover_weight * cover`` per batch element.
        clash, cover: the two hinge means, for logging.
    """
    diff = cb[:, :, None, :] - surf[:, None, :, :]  # [B, N, M, 3]
    dist = diff.norm(dim=-1)  # [B, N, M]
    pair_valid = cb_mask[:, :, None] & surf_mask[:, None, :]
    dist_masked = dist.masked_fill(~pair_valid, _BIG)

    surf_any = surf_mask.any(dim=1)  # [B] does this sample have any real surface point
    cb_any = cb_mask.any(dim=1)  # [B]

    # --- clash: signed offset of each binder Cβ to its nearest surface point ------------
    j = dist_masked.argmin(dim=-1)  # [B, N] nearest valid surface index (0 if none valid)
    j_exp = j[..., None].expand(-1, -1, 3)  # [B, N, 3]
    p_star = torch.gather(surf, 1, j_exp)  # [B, N, 3]
    n_star = torch.gather(surf_normals, 1, j_exp)  # [B, N, 3]
    signed = ((cb - p_star) * n_star).sum(-1)  # [B, N] >0 outside, <0 buried
    clash = (float(clash_margin) - signed).clamp_min(0.0)  # [B, N]
    valid_cb = cb_mask & surf_any[:, None]  # a Cβ can only clash if a surface exists
    clash_term = (clash * valid_cb).sum(-1) / valid_cb.sum(-1).clamp_min(1)  # [B]

    # --- coverage: each surface point wants a binder Cβ within contact_band -------------
    d_sb = dist_masked.min(dim=1).values  # [B, M] surface -> nearest binder
    cover = (d_sb - float(contact_band)).clamp_min(0.0)  # [B, M]
    valid_s = surf_mask & cb_any[:, None]
    cover_term = (cover * valid_s).sum(-1) / valid_s.sum(-1).clamp_min(1)  # [B]

    per = float(clash_weight) * clash_term + float(cover_weight) * cover_term
    return per, clash_term, cover_term
