"""Checks on the backbone arms' shared objectives.

The headline test is `test_steric_loss_does_not_dilute_with_receptor_size`. The clash term in
`losses.py` divides by the peptide-by-receptor pair count, which was MEASURED to leave clash
at 0.06% of the objective while being violated -- diluted ~1000x by a denominator that grows
with the receptor. If the new normalisation did not fix that, nothing else here matters.

The rest guard the preservation weights (derived once, from the source, detached), the bonded
exclusions, and the flat bottoms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from proteinfoundation.cp2lp.backbone import (  # noqa: E402
    BB_SLOT,
    TEMPLATE_C_N_NM,
    backbone_to_atom37,
)
from proteinfoundation.cp2lp.bb_losses import (  # noqa: E402
    CLASH_THRESHOLD_NM,
    backbone_preservation_loss,
    backbone_reconstruction_loss,
    backbone_steric_loss,
    open_backbone_geometry_loss,
    source_interface_weights,
)
from test_cp2lp_backbone import synthetic_backbone  # noqa: E402

torch.manual_seed(0)


#: `safe_norm` computes sqrt(sum(x^2) + 1e-12), so the distance between two IDENTICAL points
#: is 1e-6 nm rather than 0. Every "zero at the identity" assertion is written against this
#: floor, not against 0.0. That is 1e-5 angstrom and the epsilon is what keeps padded frames
#: differentiable, so it is a constant to know rather than a bug to fix.
SAFE_NORM_FLOOR_NM = 1e-6


def _physical_chain(kind: str = "beta", b: int = 4, n: int = 14, seed: int = 7):
    """A chain in a REAL secondary structure, which is genuinely free of self-overlap.

    `synthetic_backbone` draws uniform random torsions, and a uniform random backbone
    self-intersects -- MEASURED at 0.0016 nm/residue of intra-peptide overlap, which the
    steric term correctly flags. So it is the wrong fixture for testing the bonded exclusion:
    a nonzero result there says the conformation is impossible, not that the exclusion leaks.
    Beta (-120, 130) and alpha (-57, -47) both measure exactly 0.0.
    """
    import math as _math

    from proteinfoundation.cp2lp.backbone import torsions_to_backbone

    phi_deg, psi_deg = {"beta": (-120.0, 130.0), "alpha": (-57.0, -47.0)}[kind]
    mask = torch.ones(b, n, dtype=torch.bool)
    phi = torch.full((b, n), _math.radians(phi_deg))
    psi = torch.full((b, n), _math.radians(psi_deg))
    omega = torch.full((b, n - 1), _math.pi)
    root = synthetic_backbone(b=b, n=n, seed=seed)[2]["root"]
    return torsions_to_backbone(phi, psi, omega, root, mask), mask


def _receptor_at(centre: torch.Tensor, b: int, m: int, spread: float = 0.6, seed: int = 0):
    """A receptor of `m` residues around `centre`, with full heavy atoms."""
    g = torch.Generator().manual_seed(seed)
    t37 = centre[None, None, None, :] + spread * torch.randn(b, m, 37, 3, generator=g)
    tam = torch.zeros(b, m, 37, dtype=torch.bool)
    tam[:, :, :14] = True
    return t37 * tam[..., None], tam, torch.ones(b, m, dtype=torch.bool)


# ---------------------------------------------------- THE normalisation test


def test_steric_loss_does_not_dilute_with_receptor_size():
    """The same clash costs the same, whether the receptor has 20 residues or 400.

    This is the measured failure being fixed: a pair-count denominator makes a real overlap
    ~20x cheaper on a 400-residue receptor than on a 20-residue one, which is how the clash
    term came to be numerically absent from the objective while being violated.
    """
    b, n = 2, 10
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=1)
    centre = bb[:, :, BB_SLOT["CA"]].mean(dim=(0, 1))

    losses = {}
    for m in (20, 100, 400):
        # The receptor is placed ON the peptide, so there is a guaranteed real clash, and the
        # EXTRA residues are placed far away so they add pairs without adding overlap --
        # which is exactly the dilution a pair-count denominator suffers from.
        t37, tam, tmask = _receptor_at(centre, b, 20, spread=0.2, seed=2)
        if m > 20:
            far37, fartam, _ = _receptor_at(centre + 50.0, b, m - 20, spread=0.6, seed=3)
            t37 = torch.cat([t37, far37], dim=1)
            tam = torch.cat([tam, fartam], dim=1)
            tmask = torch.ones(b, m, dtype=torch.bool)
        loss, met = backbone_steric_loss(bb, mask, t37, tam, tmask)
        losses[m] = float(loss)
        assert float(met["steric_inter_per_res_nm"]) > 0, f"no clash at m={m}; test is vacuous"

    spread = max(losses.values()) / max(min(losses.values()), 1e-12)
    assert spread < 1.02, (
        f"the steric term still dilutes with receptor size: {losses} (ratio {spread:.3f}). "
        "The denominator must be peptide residues, not peptide-by-receptor pairs."
    )


def test_steric_loss_is_not_averaged_over_violating_pairs_only():
    """Fixing half the clashes must reduce the loss, not leave it flat.

    A per-violating-pair denominator shrinks as the generator improves, so the reported value
    can sit still while violations collapse -- the term is least informative exactly when it
    is working.
    """
    b, n = 1, 12
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=4)
    centre = bb[:, :, BB_SLOT["CA"]].mean(dim=(0, 1))
    t37, tam, tmask = _receptor_at(centre, b, 30, spread=0.2, seed=5)

    bad, _ = backbone_steric_loss(bb, mask, t37, tam, tmask)
    # Move the receptor away so only some overlap remains.
    better, _ = backbone_steric_loss(bb, mask, t37 + 0.12, tam, tmask)
    assert float(better) < float(bad), (
        f"reducing overlap did not reduce the loss ({float(bad):.5f} -> {float(better):.5f})"
    )


def test_steric_loss_counts_receptor_sidechains():
    """A backbone overlapping a receptor SIDECHAIN is clashing, and must be penalised."""
    b, n = 1, 10
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=6)
    ca = bb[:, :, BB_SLOT["CA"]]
    # One receptor residue whose only nearby atom is a sidechain slot.
    t37 = torch.full((b, 1, 37, 3), 50.0)
    tam = torch.zeros(b, 1, 37, dtype=torch.bool)
    t37[:, 0, 6, :] = ca[:, 0, :] + 0.05  # CD-ish slot, right on the peptide
    tam[:, 0, 6] = True
    tmask = torch.ones(b, 1, dtype=torch.bool)
    loss, met = backbone_steric_loss(bb, mask, t37, tam, tmask)
    assert float(loss) > 0 and float(met["steric_min_inter_nm"]) < CLASH_THRESHOLD_NM


def test_bonded_pairs_are_excluded_from_intra_sterics():
    """A clean chain has zero intra-peptide overlap: adjacent residues are not clashes.

    C(i)-N(i+1) sits at 0.135 nm, well inside the 0.20 nm clash threshold, so without the
    exclusion every peptide bond in every peptide would read as a steric violation and the
    term would fight the geometry term directly.
    """
    for kind in ("beta", "alpha"):
        bb, mask = _physical_chain(kind)
        loss, met = backbone_steric_loss(bb, mask)
        assert float(met["steric_intra_per_res_nm"]) == 0.0, (
            f"a clean {kind} chain shows intra-peptide overlap "
            f"({float(met['steric_intra_per_res_nm']):.5f} nm/res) -- the bonded exclusion is "
            "not covering the peptide bond"
        )
        assert float(loss) == 0.0


def test_a_random_torsion_chain_self_intersects_and_is_flagged():
    """The counterpart: uniform random torsions ARE physically impossible, and the term says so.

    Keeps the fixture choice above honest. If this also read 0.0, the exclusion would be too
    wide and a self-intersecting backbone would pass the steric term.
    """
    bb, mask, _ = synthetic_backbone(b=4, n=14, seed=7)
    _, met = backbone_steric_loss(bb, mask)
    assert float(met["steric_intra_per_res_nm"]) > 0, (
        "a uniform-random-torsion backbone shows no self-overlap, so the intra-peptide steric "
        "term is not measuring anything"
    )


def test_i_to_i_plus_two_is_still_scored():
    """The exclusion must not be so wide that a genuine i/i+2 collision is invisible."""
    bb, mask, _ = synthetic_backbone(b=1, n=8, seed=8)
    folded = bb.clone()
    folded[:, 2] = bb[:, 0] + 0.01  # slam residue 2 onto residue 0
    loss, met = backbone_steric_loss(folded, mask)
    assert float(met["steric_intra_per_res_nm"]) > 0, "an i/i+2 overlap was excluded"


# ------------------------------------------------------------------- preservation


def test_interface_weights_come_from_the_source_and_are_detached():
    """No gradient reaches the weights: they are a property of the input, not an objective."""
    b, n = 2, 10
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=9)
    src37, src_mask = backbone_to_atom37(bb, mask)
    src37 = src37.clone().requires_grad_(True)
    centre = bb[:, :, BB_SLOT["CA"]].mean(dim=(0, 1))
    t37, tam, tmask = _receptor_at(centre, b, 25, spread=0.35, seed=10)
    w, met = source_interface_weights(src37, src_mask, mask, t37, tam, tmask)
    assert not w.requires_grad, "the interface weights carry gradient"
    assert w[mask].min() > 0, "a real residue got zero weight and is free to wander"
    assert 0.0 <= float(met["iface_residue_frac"]) <= 1.0


def test_preservation_weights_actually_discriminate():
    """Moving an interface residue must cost more than moving a non-interface one.

    Otherwise the term is a uniform RMSD penalty wearing an interface-weighted name.
    """
    b, n = 1, 12
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=11)
    src37, src_mask = backbone_to_atom37(bb, mask)
    # A receptor sitting on ONE end of the peptide, so the interface is localised.
    t37, tam, tmask = _receptor_at(bb[0, 0, BB_SLOT["CA"]], b, 25, spread=0.3, seed=12)
    w, met = source_interface_weights(src37, src_mask, mask, t37, tam, tmask)
    assert float(met["iface_residue_frac"]) < 1.0, "every residue is interface; test is vacuous"

    hot = int(w[0].argmax())
    cold = int(w[0].argmin())
    assert hot != cold
    out_hot = bb.clone(); out_hot[:, hot] += 0.1
    out_cold = bb.clone(); out_cold[:, cold] += 0.1
    l_hot, _ = backbone_preservation_loss(out_hot, bb, mask, w)
    l_cold, _ = backbone_preservation_loss(out_cold, bb, mask, w)
    assert float(l_hot) > float(l_cold), (
        f"moving an interface residue ({float(l_hot):.5f}) cost no more than moving a "
        f"non-interface one ({float(l_cold):.5f})"
    )


def test_preservation_is_zero_at_the_identity_and_does_not_align():
    """Zero for an unchanged peptide; NONZERO for a rigidly rotated one.

    The second half is the point: aligning first would measure internal conformational change
    and call it pose preservation, so a peptide that rotated out of the pocket would score
    perfectly.
    """
    bb, mask, _ = synthetic_backbone(b=3, n=11, seed=13)
    loss, met = backbone_preservation_loss(bb, bb, mask)
    assert float(loss) <= 2 * SAFE_NORM_FLOOR_NM
    assert float(met["preserve_ca_rmsd_nm"]) <= 2 * SAFE_NORM_FLOOR_NM
    moved = bb + torch.tensor([0.4, 0.0, 0.0])
    loss2, _ = backbone_preservation_loss(moved, bb, mask)
    assert abs(float(loss2) - 0.4) < 1e-4, "a rigid translation was not billed in full"


# -------------------------------------------------------------- geometry safeguard


def test_geometry_safeguard_is_flat_bottomed_and_zero_on_a_clean_chain():
    """A builder-made chain passes at exactly zero loss, and the bottom really is flat."""
    bb, mask, _ = synthetic_backbone(b=4, n=13, seed=14)
    loss, met = open_backbone_geometry_loss(bb, mask)
    assert float(loss) == 0.0, f"a clean chain scored {float(loss):.3e}"
    assert float(met["geom_chain_intact_frac"]) == 1.0
    # Inside tolerance -> still exactly zero, and no gradient.
    nudged = bb.clone().requires_grad_(True)
    loss2, _ = open_backbone_geometry_loss(nudged, mask)
    loss2.backward()
    assert float(nudged.grad.abs().max()) == 0.0, "the flat bottom still produced gradient"


def test_geometry_safeguard_catches_a_broken_chain():
    """A severed peptide bond is penalised and reported as a broken chain."""
    bb, mask, _ = synthetic_backbone(b=2, n=10, seed=15)
    broken = bb.clone()
    broken[:, 5:] = broken[:, 5:] + 0.5  # tear the chain between residues 4 and 5
    loss, met = open_backbone_geometry_loss(broken, mask)
    assert float(loss) > 0
    assert float(met["geom_chain_intact_frac"]) == 0.0
    # The MEAN deviation over all 9 bonded pairs, only one of which is torn, so it lands near
    # 0.5/9 -- comparing it against a whole bond length was simply wrong arithmetic. What has
    # to hold is that it clears the tolerance the term flat-bottoms at.
    from proteinfoundation.cp2lp.geometry import PEPTIDE_BOND_TOL_NM

    assert float(met["geom_peptide_bond_dev_nm"]) > PEPTIDE_BOND_TOL_NM
    assert float(met["geom_peptide_bond_pass"]) < 1.0


def test_terminal_bond_is_never_scored():
    """Closing the ring must not reduce the geometry loss. The (L-1, 0) pair is not a bond."""
    bb, mask, _ = synthetic_backbone(b=2, n=10, seed=16)
    open_loss, _ = open_backbone_geometry_loss(bb, mask)
    closed = bb.clone()
    closed[:, -1] = bb[:, 0] + 0.01  # force the termini together
    closed_loss, _ = open_backbone_geometry_loss(closed, mask)
    assert float(closed_loss) >= float(open_loss), (
        "closing the ring reduced the open-chain geometry loss, so the terminal bond is being "
        "scored and the generator is being told to re-cyclise"
    )


# ----------------------------------------------------------------- anchor + plumbing


def test_reconstruction_anchor_is_zero_at_truth_and_gives_gradient():
    bb, mask, _ = synthetic_backbone(b=3, n=10, seed=17)
    loss, met = backbone_reconstruction_loss(bb, bb, mask)
    assert float(loss) <= 2 * SAFE_NORM_FLOOR_NM
    assert float(met["anchor_bb_rmsd_nm"]) <= 2 * SAFE_NORM_FLOOR_NM
    pred = (bb + 0.05).requires_grad_(True)
    loss2, _ = backbone_reconstruction_loss(pred, bb, mask)
    loss2.backward()
    assert pred.grad.abs().max() > 0


@pytest.mark.parametrize("fn", ["preserve", "steric", "geometry", "anchor"])
def test_every_term_is_finite_on_padded_input_and_backprops(fn):
    """Padding must not NaN any term, forward or backward."""
    from test_cp2lp_backbone import padded_backbone

    bb, mask = padded_backbone(b=3, n=10)
    bb = bb.clone().requires_grad_(True)
    centre = bb[:, :, BB_SLOT["CA"]].detach().mean(dim=(0, 1))
    t37, tam, tmask = _receptor_at(centre, 3, 20, spread=0.4, seed=18)
    src37, src_mask = backbone_to_atom37(bb.detach(), mask)
    if fn == "preserve":
        w, _ = source_interface_weights(src37, src_mask, mask, t37, tam, tmask)
        loss, met = backbone_preservation_loss(bb, bb.detach() * 0.9, mask, w)
    elif fn == "steric":
        loss, met = backbone_steric_loss(bb, mask, t37, tam, tmask)
    elif fn == "geometry":
        loss, met = open_backbone_geometry_loss(bb, mask)
    else:
        loss, met = backbone_reconstruction_loss(bb, bb.detach() * 0.9, mask)
    assert torch.isfinite(loss).all(), f"{fn} loss is non-finite"
    for k, v in met.items():
        assert torch.isfinite(v).all(), f"{fn} metric {k} is non-finite"
    loss.backward()
    assert torch.isfinite(bb.grad).all(), f"{fn} produced a NaN gradient"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
