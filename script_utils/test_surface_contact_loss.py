"""Unit tests for the surface auxiliary losses (proteinfoundation.nn.surface.loss).

The point these pin down: the legacy Chamfer is minimized when the binder COINCIDES with
the surface (right for the binder's own surface, wrong for the receptor's), while the new
contact loss is minimized when the binder sits just OUTSIDE the surface along the outward
normal and penalizes interpenetration. Run:

    python -m pytest script_utils/test_surface_contact_loss.py -q
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.surface.loss import (
    surface_contact_loss,
    virtual_cb_surface_chamfer,
)


def _flat_patch(n_side: int = 5, spacing: float = 0.2):
    """A flat MxM surface patch on the z=0 plane with +z outward normals (nm)."""
    g = torch.arange(n_side, dtype=torch.float32) * spacing
    gx, gy = torch.meshgrid(g, g, indexing="ij")
    surf = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros(n_side * n_side)], dim=-1)
    surf = surf[None]  # [1, M, 3]
    normals = torch.zeros_like(surf)
    normals[..., 2] = 1.0  # +z, "outside" is z>0
    mask = torch.ones(surf.shape[:2], dtype=torch.bool)
    return surf, normals, mask


def _binder_over_patch(surf, z_offset: float):
    """One Cβ over the centre of each surface point, lifted by z_offset along +z."""
    cb = surf.clone()
    cb[..., 2] = z_offset
    cb_mask = torch.ones(cb.shape[:2], dtype=torch.bool)
    return cb, cb_mask


def test_contact_loss_is_minimized_outside_not_at_coincidence():
    surf, normals, smask = _flat_patch()
    band = 0.5
    margin = 0.1

    def contact(z):
        cb, cbm = _binder_over_patch(surf, z)
        per, _, _ = surface_contact_loss(
            cb, cbm, surf, normals, smask,
            contact_band=band, clash_margin=margin,
        )
        return per.item()

    # Sitting proud of the surface (z == margin) satisfies both hinges -> ~0.
    assert contact(margin) < 1e-6
    # Coincidence (z == 0) violates the clash margin -> strictly worse.
    assert contact(0.0) > contact(margin) + 1e-6
    # Buried behind the surface (z < 0) is worse still (deeper clash).
    assert contact(-0.2) > contact(0.0) + 1e-6


def test_chamfer_prefers_coincidence_where_contact_does_not():
    surf, normals, smask = _flat_patch()

    def chamfer(z):
        cb, cbm = _binder_over_patch(surf, z)
        return virtual_cb_surface_chamfer(cb, cbm, surf, smask).item()

    # Chamfer's minimum is coincidence: z=0 beats a proud pose. This is exactly the
    # misspecification for a receptor surface -- it rewards interpenetration.
    assert chamfer(0.0) < chamfer(0.2)


def test_clash_penalizes_interpenetration_only():
    surf, normals, smask = _flat_patch()
    # Far above (still within the in-plane band): clash 0, coverage 0.
    cb, cbm = _binder_over_patch(surf, 0.15)
    _, clash_above, _ = surface_contact_loss(cb, cbm, surf, normals, smask, clash_margin=0.0)
    # Below the plane: clash > 0.
    cb, cbm = _binder_over_patch(surf, -0.15)
    _, clash_below, _ = surface_contact_loss(cb, cbm, surf, normals, smask, clash_margin=0.0)
    assert clash_above.item() < 1e-6
    assert clash_below.item() > 0.1  # ~0.15 nm of penetration


def test_coverage_hinges_off_within_band():
    surf, normals, smask = _flat_patch()
    # A single binder Cβ far from most surface points: coverage should be large.
    cb = torch.tensor([[[5.0, 5.0, 0.1]]])  # one point, way outside the patch
    cbm = torch.ones(cb.shape[:2], dtype=torch.bool)
    _, _, cover_far = surface_contact_loss(cb, cbm, surf, normals, smask, contact_band=0.5)
    assert cover_far.item() > 1.0
    # A dense binder cloud matching the patch: coverage ~0.
    cb, cbm = _binder_over_patch(surf, 0.1)
    _, _, cover_near = surface_contact_loss(cb, cbm, surf, normals, smask, contact_band=0.5)
    assert cover_near.item() < 1e-6


def test_padded_points_are_ignored():
    surf, normals, smask = _flat_patch()
    cb, cbm = _binder_over_patch(surf, 0.1)
    # Append a garbage padded surface point and binder Cβ, both masked out.
    surf2 = torch.cat([surf, torch.full((1, 1, 3), 99.0)], dim=1)
    normals2 = torch.cat([normals, torch.zeros((1, 1, 3))], dim=1)
    smask2 = torch.cat([smask, torch.zeros((1, 1), dtype=torch.bool)], dim=1)
    cb2 = torch.cat([cb, torch.full((1, 1, 3), -99.0)], dim=1)
    cbm2 = torch.cat([cbm, torch.zeros((1, 1), dtype=torch.bool)], dim=1)

    per_ref, _, _ = surface_contact_loss(cb, cbm, surf, normals, smask)
    per_pad, _, _ = surface_contact_loss(cb2, cbm2, surf2, normals2, smask2)
    assert torch.allclose(per_ref, per_pad, atol=1e-6)


def test_shuffled_surface_control_separates_fit_from_specificity():
    """Pins the assumption behind the `surface_specificity` metric logged in training.

    A structure shaped for its OWN patch must score worse against a mismatched one; a
    structure that ignores the patch scores the same against both. That gap is the only
    part of `surface_attraction` that conditioning can explain -- the raw loss falling
    proves nothing, which is exactly how the S1/S2 runs drove it 0.15 -> 0.04 nm with the
    gates shut at |g| ~ 1e-4.
    """
    surf_a, normals_a, smask = _flat_patch()
    # A second patch, translated far along +x: the SAME binder cannot cover both.
    surf_b = surf_a + torch.tensor([5.0, 0.0, 0.0])
    surf = torch.cat([surf_a, surf_b], dim=0)  # [2, M, 3]
    normals = torch.cat([normals_a, normals_a], dim=0)
    smask = torch.cat([smask, smask], dim=0)

    # Each binder sits over its own patch -> surface-specific.
    cb = torch.cat(
        [_binder_over_patch(surf_a, 0.1)[0], _binder_over_patch(surf_b, 0.1)[0]], dim=0
    )
    cbm = torch.ones(cb.shape[:2], dtype=torch.bool)

    per_true, _, _ = surface_contact_loss(cb, cbm, surf, normals, smask)
    rolled = torch.roll(surf, shifts=1, dims=0)
    per_shuf, _, _ = surface_contact_loss(
        cb, cbm, rolled, torch.roll(normals, shifts=1, dims=0), torch.roll(smask, shifts=1, dims=0)
    )
    specificity = per_shuf - per_true
    assert (specificity > 1.0).all(), specificity

    # A binder placed identically for both samples is NOT surface-specific: swapping the
    # patch costs it the same either way, so the metric collapses toward 0.
    cb_flat = torch.cat([_binder_over_patch(surf_a, 0.1)[0]] * 2, dim=0)
    per_true_f, _, _ = surface_contact_loss(cb_flat, cbm, surf, normals, smask)
    per_shuf_f, _, _ = surface_contact_loss(
        cb_flat, cbm, rolled, torch.roll(normals, shifts=1, dims=0),
        torch.roll(smask, shifts=1, dims=0),
    )
    assert (per_shuf_f - per_true_f).mean().abs() < 1e-5


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
