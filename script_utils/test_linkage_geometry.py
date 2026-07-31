"""Tests for the linkage angle/dihedral loss and the typed cyclic pair features."""

import math

import pytest
import torch

from proteinfoundation.cyclization.constants import DISULFIDE, ISOPEPTIDE, MAINCHAIN, UNSPECIFIED
from proteinfoundation.cyclization.linkage_geometry import (
    LINKAGE_GEOMETRY_REFERENCE,
    circular_loss,
    linkage_geometry_loss,
    linkage_geometry_terms,
)
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    AA_ASP,
    AA_CYS,
    AA_LYS,
    C_IDX,
    CA_IDX,
    CB_IDX,
    CG_IDX,
    N_IDX,
    NZ_IDX,
    SG_IDX,
)
from proteinfoundation.nn.feature_factory.pair_feats import CyclizationGraphPositionalPairFeat
from proteinfoundation.utils.angle_utils import bond_angles, signed_dihedral_angle

# ---------------------------------------------------------------------------
# circular_loss
# ---------------------------------------------------------------------------


def test_circular_loss_is_zero_at_the_reference():
    a = torch.tensor([0.3, -2.0, 3.0])
    assert torch.allclose(circular_loss(a, a), torch.zeros(3), atol=1e-7)


def test_circular_loss_handles_the_pi_seam():
    """179 and -179 degrees are 2 degrees apart, not 358."""
    near = circular_loss(torch.tensor([math.radians(179.0)]), torch.tensor([math.radians(-179.0)]))
    far = circular_loss(torch.tensor([0.0]), torch.tensor([math.pi]))
    assert float(near) < 1e-3
    assert float(far) == pytest.approx(2.0)


def test_circular_loss_is_bounded():
    x = torch.linspace(-4 * math.pi, 4 * math.pi, 200)
    out = circular_loss(x, torch.zeros_like(x))
    assert float(out.min()) >= -1e-6
    assert float(out.max()) <= 2.0 + 1e-6


def test_symmetric_mode_folds_both_handednesses():
    """Real disulfides occur at both +90 and -90; penalising one would invent a chirality."""
    ref = torch.tensor([math.radians(90.0)])
    plus = circular_loss(torch.tensor([math.radians(90.0)]), ref, symmetric=True)
    minus = circular_loss(torch.tensor([math.radians(-90.0)]), ref, symmetric=True)
    assert float(plus) == pytest.approx(0.0, abs=1e-7)
    assert float(minus) == pytest.approx(0.0, abs=1e-7)
    # Asymmetric mode must still separate them.
    assert float(circular_loss(torch.tensor([math.radians(-90.0)]), ref, symmetric=False)) > 1.0


# ---------------------------------------------------------------------------
# Synthetic structures
# ---------------------------------------------------------------------------


def _place_four(angle_i: float, angle_j: float, dihedral: float, bond: float = 0.133):
    """Four points (a, b, c, d) with exactly the requested angles and torsion.

    b-c lies along +x; a is placed in the xy-plane; d is rotated about the b-c axis by
    the requested dihedral. Distances are nanometres, matching `pred_atom37`.
    """
    b = torch.tensor([0.0, 0.0, 0.0])
    c = torch.tensor([bond, 0.0, 0.0])
    a = b + bond * torch.tensor([math.cos(angle_i), math.sin(angle_i), 0.0])
    # Direction from c, at `angle_j` to the c->b direction, then rotated by the torsion.
    base = torch.tensor([math.cos(math.pi - angle_j), math.sin(math.pi - angle_j), 0.0])
    ct, st = math.cos(dihedral), math.sin(dihedral)
    rotated = torch.tensor([base[0], base[1] * ct, base[1] * st])
    d = c + bond * rotated
    return a, b, c, d


def _structure(link_type: int, perturb: torch.Tensor | None = None):
    """A [1, 2, 37, 3] structure whose closing bond has exactly the reference geometry."""
    ref = LINKAGE_GEOMETRY_REFERENCE[link_type]
    ai, aj = ref["angle_ref"]
    a, b, c, d = _place_four(ai, aj, ref["dihedral_ref"])
    if perturb is not None:
        a = a + perturb

    atom37 = torch.zeros(1, 2, 37, 3)
    mask = torch.zeros(1, 2, 37, dtype=torch.bool)
    seq = torch.zeros(1, 2, dtype=torch.long)

    if link_type == MAINCHAIN:
        # Closing bond is C of the LAST residue (j=1, C-terminus) to N of the FIRST
        # residue (i=0, N-terminus) -- see `_mainchain_atoms`. `a`/`b` sit on residue
        # j, `c`/`d` sit on residue i.
        slots = [(1, CA_IDX, a), (1, C_IDX, b), (0, N_IDX, c), (0, CA_IDX, d)]
    elif link_type == DISULFIDE:
        slots = [(0, CB_IDX, a), (0, SG_IDX, b), (1, SG_IDX, c), (1, CB_IDX, d)]
        seq[0, 0] = seq[0, 1] = AA_CYS
    else:
        # donor lysine at residue 0 (CE, NZ), acceptor Asp at residue 1 (CG, CB)
        from proteinfoundation.eval.cyclic_reconstruction_metrics import CE_IDX

        slots = [(0, CE_IDX, a), (0, NZ_IDX, b), (1, CG_IDX, c), (1, CB_IDX, d)]
        seq[0, 0] = AA_LYS
        seq[0, 1] = AA_ASP

    for res, atom, pos in slots:
        atom37[0, res, atom] = pos
        mask[0, res, atom] = True

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([1]),
        "type": torch.tensor([link_type]),
        "has_cyclization": torch.tensor([True]),
    }
    return atom37, mask, seq, meta


ALL_TYPES = [MAINCHAIN, DISULFIDE, ISOPEPTIDE]


@pytest.mark.parametrize("link_type", ALL_TYPES)
def test_exact_reference_geometry_gives_near_zero_loss(link_type):
    atom37, mask, seq, meta = _structure(link_type)
    terms = linkage_geometry_terms(atom37, mask, seq, meta)
    assert bool(terms["valid"][0]), "reference structure should be supervisable"
    assert float(terms["angle"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["dihedral"][0]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("link_type", ALL_TYPES)
def test_perturbing_a_bonding_atom_increases_the_loss(link_type):
    atom37, mask, seq, meta = _structure(link_type)
    base = linkage_geometry_terms(atom37, mask, seq, meta)
    bad_atom37, bad_mask, bad_seq, bad_meta = _structure(link_type, perturb=torch.tensor([0.0, 0.05, 0.05]))
    bad = linkage_geometry_terms(bad_atom37, bad_mask, bad_seq, bad_meta)
    assert float(bad["angle"][0]) > float(base["angle"][0])


@pytest.mark.parametrize("link_type", ALL_TYPES)
def test_loss_is_invariant_to_global_rotation_and_translation(link_type):
    atom37, mask, seq, meta = _structure(link_type)
    before = linkage_geometry_terms(atom37, mask, seq, meta)

    theta = 0.7
    rot = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = atom37 @ rot.T + torch.tensor([3.0, -1.0, 2.0])
    moved = moved * mask[..., None]  # keep absent atoms at the origin
    after = linkage_geometry_terms(moved, mask, seq, meta)

    assert float(after["angle"][0]) == pytest.approx(float(before["angle"][0]), abs=1e-5)
    assert float(after["dihedral"][0]) == pytest.approx(float(before["dihedral"][0]), abs=1e-5)


def test_missing_atoms_are_masked_without_nans():
    atom37, mask, seq, meta = _structure(MAINCHAIN)
    mask[0, 0, N_IDX] = False  # drop one defining atom (N sits on residue i=0)
    terms = linkage_geometry_terms(atom37, mask, seq, meta)
    assert not bool(terms["valid"][0])
    assert not torch.isnan(terms["angle"]).any()
    assert not torch.isnan(terms["dihedral"]).any()


def test_wrong_residue_chemistry_is_not_supervised():
    """A disulfide label on two non-cysteines must not be scored."""
    atom37, mask, seq, meta = _structure(DISULFIDE)
    seq[0, 0] = AA_LYS
    terms = linkage_geometry_terms(atom37, mask, seq, meta)
    assert not bool(terms["valid"][0])


def test_isopeptide_orientation_is_resolved_either_way():
    """The label does not promise which endpoint is the lysine."""
    atom37, mask, seq, meta = _structure(ISOPEPTIDE)
    swapped = {**meta, "i": torch.tensor([1]), "j": torch.tensor([0])}
    terms = linkage_geometry_terms(atom37, mask, seq, swapped)
    assert bool(terms["valid"][0]), "donor/acceptor should be found regardless of index order"


def test_gradients_reach_the_coordinates():
    atom37, mask, seq, meta = _structure(MAINCHAIN, perturb=torch.tensor([0.0, 0.03, 0.02]))
    atom37 = atom37.clone().requires_grad_(True)
    loss, _ = linkage_geometry_loss(atom37, mask, seq, meta)
    loss.backward()
    assert atom37.grad is not None
    assert float(atom37.grad.abs().sum()) > 0


def test_disabled_metadata_is_a_differentiable_zero():
    """Mixed/non-CPSea batches must not break DDP with a detached loss."""
    atom37 = torch.zeros(1, 2, 37, 3, requires_grad=True)
    loss, metrics = linkage_geometry_loss(atom37, torch.zeros(1, 2, 37, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.long), None)
    assert float(loss) == 0.0
    assert loss.requires_grad
    assert metrics["n_valid"] == 0.0


def test_t_weight_zero_removes_supervision():
    atom37, mask, seq, meta = _structure(MAINCHAIN, perturb=torch.tensor([0.0, 0.05, 0.0]))
    _, metrics = linkage_geometry_loss(atom37, mask, seq, meta, t_weight=torch.tensor([0.0]))
    assert metrics["n_valid"] == 0.0


# ---------------------------------------------------------------------------
# Typed cyclic pair features
# ---------------------------------------------------------------------------


def _pair_batch(types, n=6):
    b = len(types)
    return {
        "mask": torch.ones(b, n, dtype=torch.bool),
        "coords_nm": torch.zeros(b, n, 37, 3),
        "cyclization_type_cond": torch.tensor(types),
        "cyclization_i": torch.zeros(b, dtype=torch.long),
        "cyclization_j": torch.full((b,), n - 1, dtype=torch.long),
    }


def test_all_flags_off_reproduces_the_legacy_feature_exactly():
    """An existing checkpoint must be bit-identical when nothing new is enabled."""
    batch = _pair_batch([MAINCHAIN, DISULFIDE])
    legacy = CyclizationGraphPositionalPairFeat(ring_sep_dim=32)
    extended = CyclizationGraphPositionalPairFeat(
        ring_sep_dim=32, typed_edge=False, link_direction=False, cyclic_offset_dim=0
    )
    assert legacy.dim == extended.dim == 33
    assert torch.equal(legacy(batch), extended(batch))


def test_new_channels_are_appended_not_inserted():
    """Append-only is what lets a run warm-start from a checkpoint without them."""
    batch = _pair_batch([MAINCHAIN, DISULFIDE])
    legacy = CyclizationGraphPositionalPairFeat(ring_sep_dim=32)
    extended = CyclizationGraphPositionalPairFeat(
        ring_sep_dim=32, typed_edge=True, link_direction=True, cyclic_offset_dim=9
    )
    assert extended.dim == 33 + 3 + 2 + 9
    assert torch.equal(extended(batch)[..., :33], legacy(batch))


def test_typed_edge_marks_the_right_chemistry_on_the_closing_edge_only():
    batch = _pair_batch([DISULFIDE, MAINCHAIN, ISOPEPTIDE])
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=32, typed_edge=True)
    out = feat(batch)
    typed = out[..., 33:36]
    for row, want in enumerate([DISULFIDE, MAINCHAIN, ISOPEPTIDE]):
        onehot = typed[row, 0, 5]
        assert int(onehot.argmax()) == want
        assert float(onehot.sum()) == pytest.approx(1.0)
        # NONE is the zero vector everywhere else.
        assert float(typed[row, 1, 3].sum()) == 0.0


def test_unspecified_type_emits_no_typed_channel():
    """The all-zero null for classifier-free guidance must survive."""
    batch = _pair_batch([UNSPECIFIED])
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=32, typed_edge=True)
    assert float(feat(batch).abs().sum()) == 0.0


def test_link_direction_distinguishes_the_two_orientations():
    batch = _pair_batch([ISOPEPTIDE])
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=32, link_direction=True)
    out = feat(batch)
    assert out[0, 0, 5, 33:35].tolist() == [1.0, 0.0]
    assert out[0, 5, 0, 33:35].tolist() == [0.0, 1.0]


def test_cyclic_offset_is_mainchain_only():
    """A signed backbone wrap on a side-chain linkage would assert a bond that isn't there."""
    batch = _pair_batch([MAINCHAIN, DISULFIDE, ISOPEPTIDE])
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=32, cyclic_offset_dim=9)
    out = feat(batch)[..., 33:]
    assert float(out[0].abs().sum()) > 0.0
    assert float(out[1].abs().sum()) == 0.0
    assert float(out[2].abs().sum()) == 0.0


def test_widened_pair_repr_warm_starts_identically_from_a_narrower_checkpoint():
    """The whole point of appending: a v4 run must be able to warm-start into this.

    The new channels widen the pair-repr concat (33 -> 47), so the projection gains input
    columns. `_splice_tensor_to_model_shape` copies the checkpoint's leading columns and
    zero-fills the rest, so on identical inputs the widened layer must reproduce the old
    one exactly -- the run STARTS as v4 and learns the typed signal from there.
    """
    from proteinfoundation.train import _splice_tensor_to_model_shape

    torch.manual_seed(0)
    ckpt_w = torch.randn(64, 33)
    model_w = torch.randn(64, 47)  # freshly initialised, about to be overwritten
    spliced = _splice_tensor_to_model_shape(model_w, ckpt_w)

    assert torch.equal(spliced[:, :33], ckpt_w), "pre-existing columns must be preserved"
    assert float(spliced[:, 33:].abs().sum()) == 0.0, "new columns must start at exactly zero"

    # Same input, padded with the new channels' values -> identical output.
    legacy_in = torch.randn(8, 33)
    widened_in = torch.cat([legacy_in, torch.randn(8, 14)], dim=-1)
    assert torch.allclose(widened_in @ spliced.T, legacy_in @ ckpt_w.T, atol=1e-6)


def test_pair_repr_builder_is_not_excluded_from_the_splice():
    """`skip_prefixes` must not cover the module that consumes feats_pair_repr."""
    import inspect

    from proteinfoundation.train import _splice_pretrained_weights

    defaults = inspect.signature(_splice_pretrained_weights).parameters["skip_prefixes"].default
    assert not any("pair_repr_builder" in p for p in defaults), (
        "pair_repr_builder is skipped, so the widened layer would be randomly initialised "
        "instead of warm-started"
    )


def test_scalar_topology_features_are_unaffected_by_coordinates():
    """These are topology, not geometry: moving the structure must not change them."""
    batch = _pair_batch([MAINCHAIN])
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=32, typed_edge=True, cyclic_offset_dim=9)
    before = feat(batch)
    batch["coords_nm"] = torch.randn_like(batch["coords_nm"]) * 5.0
    assert torch.equal(feat(batch), before)
