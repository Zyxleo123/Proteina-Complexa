"""Checks on the two backbone arms, written around the things that would make them UNFAIR.

A representation comparison is worthless if the two arms differ in anything but the
representation, and the differences that creep in are not exotic -- they are a movement budget
measured in different units, one arm allowed to translate the peptide and the other not, and a
head whose first output is not the source. Each has a test here.

The other half is the construction guarantee: Arm B must be unable to emit an invalid
backbone even when its head emits garbage, and Arm A must be ABLE to, or there is nothing to
compare.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.cp2lp.backbone import (  # noqa: E402
    ANGLE_SOURCE_SOURCE,
    ANGLE_SOURCE_TEMPLATE,
    BB_SLOT,
    TEMPLATE_C_N_NM,
    TEMPLATE_CA_C_NM,
    TEMPLATE_N_CA_NM,
    backbone_validity,
    bond_angle,
)
from proteinfoundation.cp2lp.bb_arms import (  # noqa: E402
    ARM_CARTESIAN,
    ARM_TORSION,
    apply_backbone_arm,
    clip_to_budget,
    init_reconstruction_error,
)

sys.path.insert(0, str(REPO / "script_utils"))
from test_cp2lp_backbone import synthetic_backbone  # noqa: E402

torch.manual_seed(0)


def _heads(b: int, n: int, scale: float = 1.0, seed: int = 0) -> dict[str, dict]:
    """A head output for each arm, same RNG, comparable magnitude."""
    g = torch.Generator().manual_seed(seed)
    return {
        ARM_CARTESIAN: {"delta_cartesian": scale * torch.randn(b, n, 4, 3, generator=g)},
        ARM_TORSION: {
            "delta_torsion": scale * torch.randn(b, n, 6, generator=g),
            "delta_root": scale * torch.randn(b, 6, generator=g),
        },
    }


# ------------------------------------------------------------- the identity at zero init


def test_both_arms_start_at_the_source():
    """Zero head output reproduces the source backbone -- the zero-init splice property.

    Arm B is exact only because `DEFAULT_ANGLE_SOURCE` carries the source's bond angles. Under
    the template it would be 1.18 A off, which is what `init_reconstruction_error` reports and
    what the next test pins.
    """
    bb, mask, _ = synthetic_backbone(b=4, n=12, seed=1)
    for arm in (ARM_CARTESIAN, ARM_TORSION):
        stats = init_reconstruction_error(arm, bb, mask)
        assert stats["init_bb_rmsd_nm"] < 1e-5, f"{arm} init error {stats['init_bb_rmsd_nm']:.2e} nm"
        assert stats["init_bb_max_nm"] < 1e-4, f"{arm} worst init atom {stats['init_bb_max_nm']:.2e} nm"


def test_template_angle_source_is_the_regime_that_has_a_floor():
    """The recorded init error distinguishes the two angle sources, so a config cannot lie.

    Built from a backbone with varied angles, which is the realistic case; on a backbone that
    is already exactly on the template manifold both modes would read zero and the diagnostic
    would be vacuous.
    """
    from test_cp2lp_backbone import _backbone_with_varied_angles

    bb, mask = _backbone_with_varied_angles(b=6, n=13, seed=31)
    src = init_reconstruction_error(ARM_TORSION, bb, mask, angle_source=ANGLE_SOURCE_SOURCE)
    tpl = init_reconstruction_error(ARM_TORSION, bb, mask, angle_source=ANGLE_SOURCE_TEMPLATE)
    assert src["init_bb_rmsd_nm"] < 1e-5
    assert tpl["init_bb_rmsd_nm"] > 1e-3, (
        f"template mode shows no floor ({tpl['init_bb_rmsd_nm']:.2e} nm) -- it is not actually "
        "replacing the source angles"
    )


# -------------------------------------------------------------- the construction guarantee


def test_arm_b_cannot_emit_an_invalid_backbone_even_from_garbage():
    """Arm B's guarantee under an absurd head: chain intact, template lengths, every time.

    A GAN generator early in training emits exactly this kind of garbage, so the guarantee has
    to hold here or it does not matter.
    """
    bb, mask, _ = synthetic_backbone(b=6, n=14, seed=2)
    g = torch.Generator().manual_seed(3)
    head = {
        "delta_torsion": 50.0 * torch.randn(6, 14, 6, generator=g),
        "delta_root": 50.0 * torch.randn(6, 6, generator=g),
    }
    out, _ = apply_backbone_arm(ARM_TORSION, bb, {"backbone": head}, mask, scale=1.0)
    assert torch.isfinite(out).all()
    v = backbone_validity(out, mask)
    assert v["bb_chain_intact"].min() == 1.0, "Arm B broke a chain"
    nat, caat, cat = (out[:, :, BB_SLOT[k]] for k in ("N", "CA", "C"))
    assert ((caat - nat).norm(dim=-1) - TEMPLATE_N_CA_NM).abs().max() < 1e-5
    assert ((cat - caat).norm(dim=-1) - TEMPLATE_CA_C_NM).abs().max() < 1e-5
    assert ((nat[:, 1:] - cat[:, :-1]).norm(dim=-1) - TEMPLATE_C_N_NM).abs().max() < 1e-5


def test_arm_a_can_emit_an_invalid_backbone():
    """The contrast. If Arm A could not break a chain there would be no experiment."""
    bb, mask, _ = synthetic_backbone(b=6, n=14, seed=4)
    head = _heads(6, 14, scale=0.05, seed=5)[ARM_CARTESIAN]
    out, _ = apply_backbone_arm(ARM_CARTESIAN, bb, {"backbone": head}, mask)
    v = backbone_validity(out, mask)
    assert v["bb_chain_intact"].max() == 0.0, (
        "a 0.05 nm free Cartesian residual left a chain intact -- the arms are not actually "
        "different, or the validity check is not measuring the chain"
    )


# ------------------------------------------------------------------ the shared budget


def test_the_movement_budget_is_the_same_constraint_in_both_arms():
    """A budget of 0.1 nm caps the worst atom at 0.1 nm in BOTH arms.

    The trap this exists for: capping Arm A's Cartesian residual and Arm B's torsion residual
    at the same NUMBER defines two unrelated constraints, because 0.1 nm and 0.1 rad are not
    the same budget and 0.1 rad at residue 2 of a 20-mer moves the far end by nanometres.
    """
    bb, mask, _ = synthetic_backbone(b=5, n=16, seed=6)
    cap = 0.1
    for arm, head in _heads(5, 16, scale=2.0, seed=7).items():
        out, diag = apply_backbone_arm(
            arm, bb, {"backbone": head}, mask, scale=1.0, max_shift_nm=cap
        )
        shift = (out - bb).norm(dim=-1)[mask.bool()]
        assert shift.max() <= cap + 1e-5, f"{arm} moved an atom {float(shift.max()):.4f} nm > {cap}"
        assert float(diag["bb_shift_clip_frac"]) > 0.0, f"{arm} budget never bound; test is vacuous"


def test_budget_clip_preserves_direction():
    """The clip shortens an edit, never rotates it. A radial projection, not a box clamp.

    A per-axis clamp would turn a diagonal edit into a different edit, which is a silent
    change of what the generator asked for.
    """
    bb, mask, _ = synthetic_backbone(b=2, n=8, seed=8)
    target = bb + torch.tensor([0.3, 0.4, 0.0])
    out, _ = clip_to_budget(target, bb, mask, 0.1)
    d = (out - bb)[mask.bool()]
    unit = d / d.norm(dim=-1, keepdim=True)
    want = torch.tensor([0.6, 0.8, 0.0])
    assert (unit - want).abs().max() < 1e-5, "clip changed the edit direction"
    assert (d.norm(dim=-1) - 0.1).abs().max() < 1e-6


def test_both_arms_can_move_the_whole_peptide():
    """Arm B's root transform exists so neither arm has a rigid-motion monopoly.

    Without it, "Arm A is better" could just mean "Arm A was allowed to translate".
    """
    bb, mask, _ = synthetic_backbone(b=3, n=10, seed=9)
    shift = torch.tensor([0.5, -0.3, 0.2])
    heads = {
        ARM_CARTESIAN: {"delta_cartesian": shift.expand(3, 10, 4, 3).clone()},
        ARM_TORSION: {
            "delta_torsion": torch.zeros(3, 10, 6),
            "delta_root": torch.cat([shift.expand(3, 3), torch.zeros(3, 3)], dim=-1),
        },
    }
    for arm, head in heads.items():
        out, _ = apply_backbone_arm(arm, bb, {"backbone": head}, mask, scale=1.0)
        moved = (out - bb)[mask.bool()]
        assert (moved - shift).abs().max() < 1e-5, f"{arm} did not translate as asked"


def test_arm_b_root_rotation_is_about_the_centroid():
    """A pure rotation must not also translate the peptide -- the pivot is its own centroid."""
    bb, mask, _ = synthetic_backbone(b=3, n=10, seed=10)
    head = {
        "delta_torsion": torch.zeros(3, 10, 6),
        "delta_root": torch.cat([torch.zeros(3, 3), torch.tensor([0.0, 0.0, 0.7]).expand(3, 3)], -1),
    }
    out, _ = apply_backbone_arm(ARM_TORSION, bb, {"backbone": head}, mask, scale=1.0)
    mf = mask.bool()[..., None, None].float()
    c_in = (bb * mf).sum(dim=(1, 2)) / mf.sum(dim=(1, 2))
    c_out = (out * mf).sum(dim=(1, 2)) / mf.sum(dim=(1, 2))
    assert (c_out - c_in).abs().max() < 1e-5, "rotation moved the centroid"
    assert (out - bb).abs().max() > 1e-3, "rotation did nothing; test is vacuous"


# --------------------------------------------------------------------- plumbing hygiene


def test_padding_is_respected_and_gradients_are_finite():
    """Both arms: no NaN forward or backward, nothing written to a padded residue."""
    from test_cp2lp_backbone import padded_backbone

    bb, mask = padded_backbone(b=3, n=10)
    for arm, head in _heads(3, 10, scale=1.0, seed=12).items():
        h = {k: v.clone().requires_grad_(True) for k, v in head.items()}
        out, diag = apply_backbone_arm(arm, bb, {"backbone": h}, mask, scale=1.0)
        assert torch.isfinite(out).all(), f"{arm} emitted a non-finite coordinate"
        assert out[~mask.bool()].abs().max() == 0.0, f"{arm} wrote into a padded residue"
        for k, t in diag.items():
            assert torch.isfinite(t).all(), f"{arm} diagnostic {k} is non-finite"
        out.sum().backward()
        for k, t in h.items():
            assert torch.isfinite(t.grad).all(), f"{arm} gradient through {k} is non-finite"
            assert t.grad.abs().max() > 0, f"{arm} head {k} received no gradient"


def test_missing_head_raises_rather_than_silently_doing_nothing():
    """A config/arm mismatch must be loud. A silent no-op looks like a refusing generator."""
    import pytest

    bb, mask, _ = synthetic_backbone(b=2, n=8, seed=13)
    with pytest.raises(KeyError, match="editor_backbone_mode"):
        apply_backbone_arm(ARM_CARTESIAN, bb, {}, mask)
    with pytest.raises(KeyError, match="delta_cartesian"):
        apply_backbone_arm(ARM_CARTESIAN, bb, {"backbone": {"delta_torsion": torch.zeros(2, 8, 6)}}, mask)
    with pytest.raises(KeyError, match="delta_torsion"):
        apply_backbone_arm(ARM_TORSION, bb, {"backbone": {"delta_cartesian": torch.zeros(2, 8, 4, 3)}}, mask)
    with pytest.raises(ValueError, match="unknown backbone arm"):
        apply_backbone_arm("ca_only", bb, {"backbone": {"delta_torsion": torch.zeros(2, 8, 6)}}, mask)


def test_omega_channel_is_indexed_by_bond_not_by_residue():
    """The last residue has no outgoing peptide bond, so its omega channel must be unused.

    An off-by-one here would rotate the wrong bond and the error would look like noise.
    """
    bb, mask, _ = synthetic_backbone(b=2, n=9, seed=14)
    base = torch.zeros(2, 9, 6)
    head_a = {"delta_torsion": base.clone(), "delta_root": torch.zeros(2, 6)}
    head_b = {"delta_torsion": base.clone(), "delta_root": torch.zeros(2, 6)}
    head_b["delta_torsion"][:, -1, 4] = 5.0  # last residue's omega channel
    out_a, _ = apply_backbone_arm(ARM_TORSION, bb, {"backbone": head_a}, mask, scale=1.0)
    out_b, _ = apply_backbone_arm(ARM_TORSION, bb, {"backbone": head_b}, mask, scale=1.0)
    assert (out_a - out_b).abs().max() < 1e-6, "the last residue's omega channel changed the chain"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
