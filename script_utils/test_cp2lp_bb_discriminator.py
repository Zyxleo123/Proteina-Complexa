"""Checks on the backbone-only discriminator, aimed at the shortcuts that would void the arm.

An adversarial arm fails silently when D separates real from generated on something the
generator does not control. This arm has three such somethings, and each gets a test that
tries to exploit it:

* SIDECHAINS. Real LPs arrive with them, generated backbones never have them. If D saw them
  it would separate the populations on "has a CB" and the generator's gradient would be noise.
* SEQUENCE. The arms copy the source CP's sequence exactly and cannot change it, while real
  LPs come from a different population (PepBench). A sequence feature is a shortcut the
  generator is *forbidden* to close.
* EXACT LOCAL BOND GEOMETRY. Arm B's bond lengths are the template to float precision; a
  crystal's are not. One column would separate them perfectly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from proteinfoundation.cp2lp.backbone import backbone_to_atom37  # noqa: E402
from proteinfoundation.cp2lp.discriminator import (  # noqa: E402
    PeptideInterfaceDiscriminator,
)
from test_cp2lp_backbone import synthetic_backbone  # noqa: E402

torch.manual_seed(0)


def _receptor(b: int, m: int = 40, seed: int = 0):
    """A receptor with its FULL heavy-atom complement, which is what the real data has."""
    g = torch.Generator().manual_seed(seed)
    t37 = torch.randn(b, m, 37, 3, generator=g) * 0.5
    tam = torch.zeros(b, m, 37, dtype=torch.bool)
    tam[:, :, :14] = True
    return t37, tam, torch.ones(b, m, dtype=torch.bool)


def _with_sidechains(atom37: torch.Tensor, atom_mask: torch.Tensor, seed: int = 3):
    """Add plausible sidechain atoms, as a real crystal LP would carry."""
    g = torch.Generator().manual_seed(seed)
    a = atom37.clone()
    am = atom_mask.clone()
    ca = atom37[:, :, 1, :]
    for slot in (3, 5, 6, 7, 8, 9, 10, 11):
        a[:, :, slot, :] = ca + 0.15 * torch.randn(ca.shape, generator=g)
        am[:, :, slot] = True
    return a * am[..., None], am


def _d(backbone_only: bool, **kw):
    return PeptideInterfaceDiscriminator(
        hidden_dim=64, nlayers=2, nheads=4, backbone_only=backbone_only, **kw
    )


def _feats(d, atom37, atom_mask, mask, seq, rec):
    t37, tam, tmask = rec
    return d.build_features(
        pep_atom37=atom37, pep_atom_mask=atom_mask, pep_mask=mask, pep_seq_probs=seq,
        target_atom37=t37, target_atom_mask=tam, target_mask=tmask,
    )


# ------------------------------------------------------------------- the sidechain tell


def test_sidechains_are_invisible_to_a_backbone_only_discriminator():
    """THE test. A peptide with sidechains and the same peptide without give IDENTICAL features.

    This is what makes the real and generated paths symmetric by construction rather than by
    everyone remembering to strip sidechains at every call site.
    """
    b, n = 3, 11
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=1)
    bare37, bare_mask = backbone_to_atom37(bb, mask)
    full37, full_mask = _with_sidechains(bare37, bare_mask)
    assert full_mask.sum() > bare_mask.sum(), "fixture added no sidechains; test is vacuous"

    seq = torch.softmax(torch.randn(b, n, 21), -1)
    rec = _receptor(b)
    d = _d(True)
    f_bare = _feats(d, bare37, bare_mask, mask, seq, rec)
    f_full = _feats(d, full37, full_mask, mask, seq, rec)
    for key in f_bare:
        if f_bare[key].dtype == torch.bool:
            assert torch.equal(f_bare[key], f_full[key]), f"{key} differs"
        else:
            err = (f_bare[key] - f_full[key]).abs().max()
            assert err < 1e-6, f"feature block {key} leaks sidechain information ({float(err):.2e})"


def test_a_full_atom_discriminator_does_see_sidechains():
    """The contrast. If the non-backbone D also ignored them the reduction would be a no-op."""
    b, n = 3, 11
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=2)
    bare37, bare_mask = backbone_to_atom37(bb, mask)
    full37, full_mask = _with_sidechains(bare37, bare_mask)
    seq = torch.softmax(torch.randn(b, n, 21), -1)
    rec = _receptor(b)
    d = _d(False)
    f_bare = _feats(d, bare37, bare_mask, mask, seq, rec)
    f_full = _feats(d, full37, full_mask, mask, seq, rec)
    assert (f_bare["iface"] - f_full["iface"]).abs().max() > 1e-4, (
        "the full-atom discriminator ignores sidechains too, so the backbone-only mode is "
        "not actually removing anything"
    )


# -------------------------------------------------------------------- the sequence tell


def test_sequence_cannot_reach_a_backbone_only_discriminator():
    """No `res` block is produced, and two different sequences give the same logit."""
    b, n = 4, 10
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=4)
    atom37, atom_mask = backbone_to_atom37(bb, mask)
    rec = _receptor(b)
    d = _d(True).eval()
    f1 = _feats(d, atom37, atom_mask, mask, torch.softmax(torch.randn(b, n, 21), -1), rec)
    f2 = _feats(d, atom37, atom_mask, mask, torch.softmax(torch.randn(b, n, 21), -1), rec)
    assert "res" not in f1, "a backbone-only D built a sequence feature block"
    with torch.no_grad():
        assert (d(f1) - d(f2)).abs().max() == 0.0, "the logit moved with the sequence"


def test_passing_a_sequence_block_in_is_loud():
    """A features/discriminator mismatch raises rather than quietly dropping the block.

    Silently ignoring it would make "D never sees the sequence" unverifiable from the outside.
    """
    b, n = 2, 8
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=5)
    atom37, atom_mask = backbone_to_atom37(bb, mask)
    rec = _receptor(b)
    d = _d(True)
    f = _feats(d, atom37, atom_mask, mask, torch.softmax(torch.randn(b, n, 21), -1), rec)
    f["res"] = torch.softmax(torch.randn(b, n, 21), -1)
    with pytest.raises(ValueError, match="backbone-only discriminator received"):
        d(f)


# ---------------------------------------------------------- the local bond-geometry tell


def test_intra_residue_bond_columns_are_gone():
    """The 2 bond-deviation columns are absent, so the template/crystal tell has no column.

    Checked on the WIDTH rather than the values: a zeroed column is still a column that a
    later edit could start filling, and the width is what `in_proj` is built against.
    """
    assert _d(True).n_geom == _d(False).n_geom - 2
    b, n = 2, 9
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=6)
    atom37, atom_mask = backbone_to_atom37(bb, mask)
    rec = _receptor(b)
    for bo in (True, False):
        d = _d(bo)
        f = _feats(d, atom37, atom_mask, mask, torch.softmax(torch.randn(b, n, 21), -1), rec)
        assert f["geom"].shape[-1] == d.n_geom
        # The terminal flags must stay the LAST two columns, because
        # `continuous_column_masks` excludes them by position.
        assert not d.continuous_column_masks()["geom"][-2:].any()
        assert d.continuous_column_masks()["geom"][:-2].all()


# ------------------------------------------------------------------------- plumbing


def test_backbone_only_discriminator_trains():
    """Forward, hinge loss, backward, finite gradients on every parameter that should get one."""
    from proteinfoundation.cp2lp.discriminator import hinge_d_loss

    b, n = 4, 10
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=7)
    atom37, atom_mask = backbone_to_atom37(bb, mask)
    rec = _receptor(b)
    d = _d(True)
    f = _feats(d, atom37, atom_mask, mask, torch.softmax(torch.randn(b, n, 21), -1), rec)
    loss = hinge_d_loss(d(f), d(f))
    loss.backward()
    # The SHORTCUT PROBE is deliberately excluded: it is trained by its own loss on
    # `probe_forward`, and its gradient is detached from the trunk so that it diagnoses the
    # features without shaping them. Receiving nothing from the main hinge is the correct
    # behaviour, and asserting otherwise would be asserting the probe is wired wrong.
    missing = [
        k for k, p in d.named_parameters()
        if p.requires_grad and p.grad is None and not k.startswith("shortcut_probe.")
    ]
    assert not missing, f"no gradient reached: {missing}"
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)


def test_the_shortcut_probe_trains_on_its_own_loss_and_not_the_trunk():
    """The probe must learn from `probe_forward`, and must not push gradient into the trunk.

    Both halves matter. A probe that gets no gradient at all is a dead diagnostic that will
    read "no shortcut" forever; a probe whose gradient reaches the trunk would SHAPE the
    features it is supposed to be measuring.
    """
    b, n = 4, 10
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=9)
    atom37, atom_mask = backbone_to_atom37(bb, mask)
    rec = _receptor(b)
    d = _d(True)
    f = _feats(d, atom37, atom_mask, mask, torch.softmax(torch.randn(b, n, 21), -1), rec)
    d.probe_forward(f).sum().backward()
    probe = [p for k, p in d.named_parameters() if k.startswith("shortcut_probe.")]
    assert probe, "no shortcut probe was built"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe)
    assert any(p.grad.abs().max() > 0 for p in probe), "the probe is a dead diagnostic"
    leaked = [
        k for k, p in d.named_parameters()
        if not k.startswith("shortcut_probe.") and p.grad is not None and p.grad.abs().max() > 0
    ]
    assert not leaked, f"the probe's gradient reached the trunk: {leaked}"


def test_generator_gradient_reaches_the_backbone_through_d():
    """The adversarial signal must actually arrive at the coordinates. Otherwise there is no GAN."""
    from proteinfoundation.cp2lp.discriminator import hinge_g_loss

    b, n = 3, 10
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=8)
    bb = bb.clone().requires_grad_(True)
    atom37, atom_mask = backbone_to_atom37(bb, mask)
    rec = _receptor(b)
    d = _d(True)
    f = _feats(d, atom37, atom_mask, mask, torch.softmax(torch.randn(b, n, 21), -1), rec)
    hinge_g_loss(d(f)).backward()
    assert bb.grad is not None and torch.isfinite(bb.grad).all()
    assert bb.grad.abs().max() > 0, "D gives the backbone no gradient at all"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
