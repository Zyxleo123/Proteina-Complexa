"""Unit tests for the one-pass CP -> LP editor and its three add-ons.

CPU only, no checkpoints, no dataloader: this covers the pure pieces and the config
invariants. The wiring that needs a real batch and a GPU -- one trunk call per generated
batch, frame sensitivity against the base velocity head, train/export agreement, one real
G/D update -- lives in ``script_utils/cp2lp_onepass_smoke.py``, because those checks are
only meaningful against the actual trunk and the actual data.

Every test here exists because the failure it covers is SILENT. Nothing in this file
catches a crash; each one catches a run that trains, logs falling losses, and answers a
different question than the one that was asked.

Run:
    .venv/bin/python -m pytest script_utils/test_cp2lp_onepass.py -q
"""

from __future__ import annotations

import os
import pathlib

import pytest
import torch

from proteinfoundation.cp2lp.addons import (
    ADDONS,
    AddonRandomStreams,
    FeatureNoiseController,
    FeatureScaleCache,
    LocalRetentionNCE,
    approx_r1_penalty,
    feature_widths,
    validate_addon,
)
from proteinfoundation.cp2lp.conditioning import SRC_CP_KEY, SRC_CP_PRESENT_KEY
from proteinfoundation.cp2lp.discriminator import PeptideInterfaceDiscriminator
from proteinfoundation.cp2lp.editor import (
    DEFAULT_EDITOR_T,
    EDITOR_NOISE_KEY,
    apply_residuals,
    assert_inputs_unmutated,
    build_editor_batch,
    count_trunk_calls,
    init_noise_adapter_,
    make_noise_generator,
    sample_editor_noise,
    snapshot_inputs,
)
from proteinfoundation.cp2lp.export import displacement_from_source_nm, terminal_bond_nm
from proteinfoundation.cyclization.constants import LINEAR, MAINCHAIN

REPO = pathlib.Path(__file__).resolve().parents[1]
MODES = ["bb_ca", "local_latents"]


def _cp_batch(b: int = 2, n: int = 6, d: int = 8) -> dict:
    """A minimal CPSea-shaped batch, post-``add_clean_samples``."""
    mask = torch.ones(b, n, dtype=torch.bool)
    return {
        "mask": mask,
        "x_1": {"bb_ca": torch.randn(b, n, 3), "local_latents": torch.randn(b, n, d)},
        "x_0": {"bb_ca": torch.randn(b, n, 3), "local_latents": torch.randn(b, n, d)},
        "x_t": {"bb_ca": torch.randn(b, n, 3), "local_latents": torch.randn(b, n, d)},
        "t": {"bb_ca": torch.rand(b), "local_latents": torch.rand(b)},
        "x_sc": {"bb_ca": torch.randn(b, n, 3), "local_latents": torch.randn(b, n, d)},
        "residue_type": torch.randint(0, 20, (b, n)),
        "x_target": torch.randn(b, 5, 37, 3),
        "target_mask": torch.ones(b, 5, 37, dtype=torch.bool),
        "seq_target_mask": torch.ones(b, 5, dtype=torch.bool),
        "seq_target": torch.randint(0, 20, (b, 5)),
        "cyclization_type_cond": torch.full((b,), MAINCHAIN, dtype=torch.long),
        "cyclization_i": torch.zeros(b, dtype=torch.long),
        "cyclization_j": torch.full((b,), n - 1, dtype=torch.long),
        "has_cyclization": torch.ones(b, dtype=torch.bool),
    }


# ============================================================== the editor batch


def test_editor_batch_feeds_the_source_as_state_and_keeps_it_as_condition():
    """The source CP has to arrive through the PRETRAINED input channels.

    Routing it only through `x_src_cp` would leave `x_t` holding whatever the interpolant
    last wrote -- so the trunk's main coordinate path would be reading noise while the
    edit was supposed to be of the source. Both roles are required, and they are not
    interchangeable.
    """
    batch = _cp_batch()
    gen = build_editor_batch(batch, MODES, t_fixed=0.5, noise=None)
    for m in MODES:
        assert torch.equal(gen["x_t"][m], batch[SRC_CP_KEY][m]), f"x_t is not the source CP for {m}"
        assert torch.equal(gen[SRC_CP_KEY][m], batch[SRC_CP_KEY][m])
    assert float(gen[SRC_CP_PRESENT_KEY].min()) == 1.0


def test_editor_batch_drops_every_interpolant_tensor():
    """A stale `x_1` is the dangerous one: it is the encoded SOURCE CP, and any loss that
    read it as "the ground-truth linear peptide" would be supervising the editor to return
    the ring it was asked to open."""
    gen = build_editor_batch(_cp_batch(), MODES, t_fixed=0.5, noise=None)
    for key in ("x_1", "x_0", "x_recycle"):
        assert key not in gen, f"{key} survived into the editor batch"


def test_editor_batch_pins_t_and_zeroes_self_conditioning():
    """`t` is an ARCHITECTURE INPUT here, identical on every row and every mode: nothing
    integrates. And there is no evolving state to self-condition on, so `x_sc` is
    explicitly zero rather than inherited from the interpolant."""
    gen = build_editor_batch(_cp_batch(), MODES, t_fixed=DEFAULT_EDITOR_T, noise=None)
    for m in MODES:
        assert torch.allclose(gen["t"][m], torch.full_like(gen["t"][m], DEFAULT_EDITOR_T))
        assert float(gen["x_sc"][m].abs().max()) == 0.0


def test_editor_batch_requests_linear_and_clears_the_ring_labels():
    """LINEAR is a request for NO ring and is distinct from UNSPECIFIED (the CFG null).
    Inherited endpoint labels would be read downstream as a bond to supervise."""
    gen = build_editor_batch(_cp_batch(), MODES, t_fixed=0.5, noise=None)
    assert torch.all(gen["cyclization_type_cond"] == LINEAR)
    for stale in ("cyclization_i", "cyclization_j", "has_cyclization", "cyclization_type"):
        assert stale not in gen, f"{stale} survived a LINEAR request"


def test_editor_noise_is_masked_and_shape_checked():
    batch = _cp_batch(b=2, n=6)
    batch["mask"][1, 4:] = False
    z = sample_editor_noise(2, 6, 4, torch.device("cpu"))
    gen = build_editor_batch(batch, MODES, t_fixed=0.5, noise=z)
    assert float(gen[EDITOR_NOISE_KEY][1, 4:].abs().max()) == 0.0
    with pytest.raises(ValueError):
        build_editor_batch(_cp_batch(b=2, n=6), MODES, t_fixed=0.5, noise=torch.randn(3, 6, 4))


def test_noise_generator_is_reproducible_and_stream_separated():
    """Two arms are comparable only if the same seed gives the same noise. And the editor's
    stream must not be the addons' stream, or an addon's extra draws would shift the noise
    the OTHER arms never see -- making the addon change the data, not just the objective."""
    dev = torch.device("cpu")
    a = sample_editor_noise(2, 5, 4, dev, generator=make_noise_generator(7, dev))
    b = sample_editor_noise(2, 5, 4, dev, generator=make_noise_generator(7, dev))
    assert torch.equal(a, b)
    c = sample_editor_noise(2, 5, 4, dev, generator=make_noise_generator(8, dev))
    assert not torch.equal(a, c)
    s = AddonRandomStreams(7, dev)
    assert s.cpu.initial_seed() != make_noise_generator(7, dev).initial_seed()


# =================================================================== the residual


class _FakeTrunk(torch.nn.Module):
    """A stand-in with the same output contract, so the residual algebra is testable on CPU."""

    def __init__(self, d: int = 8, emit_delta: bool = True, scale: float = 1.0):
        super().__init__()
        self.d, self.emit_delta, self.scale = d, emit_delta, scale
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        b, n = batch["mask"].shape
        out = {"bb_ca": {"v": torch.zeros(b, n, 3)}, "local_latents": {"v": torch.zeros(b, n, self.d)}}
        if self.emit_delta:
            out["bb_ca"]["delta"] = torch.full((b, n, 3), self.scale)
            out["local_latents"]["delta"] = torch.full((b, n, self.d), self.scale)
        return out


def test_apply_residuals_adds_and_masks():
    batch = _cp_batch(b=2, n=5)
    batch["mask"][1, 3:] = False
    gen = build_editor_batch(batch, MODES, t_fixed=0.5, noise=None)
    nn_out = _FakeTrunk(scale=0.25)(gen)
    edited, deltas = apply_residuals(gen["x_t"], nn_out, gen["mask"], MODES)
    assert torch.allclose(edited["bb_ca"][0], gen["x_t"]["bb_ca"][0] + 0.25)
    # Padding untouched: the residual is zeroed there, so the state is exactly the source.
    assert torch.equal(edited["bb_ca"][1, 3:], gen["x_t"]["bb_ca"][1, 3:])
    assert float(deltas["bb_ca"][1, 3:].abs().max()) == 0.0


def test_apply_residuals_refuses_to_read_the_velocity_head():
    """The pretrained head emits a flow VELOCITY. Consuming it as an edit displacement
    would start the editor at `x_src + v(x_src, t)` -- an arbitrary jump, not the identity
    edit the zero-initialised residual heads give. Separate keys, and a hard failure."""
    gen = build_editor_batch(_cp_batch(), MODES, t_fixed=0.5, noise=None)
    nn_out = _FakeTrunk(emit_delta=False)(gen)
    assert "v" in nn_out["bb_ca"] and "delta" not in nn_out["bb_ca"]
    with pytest.raises(KeyError, match="enable_editor_residual"):
        apply_residuals(gen["x_t"], nn_out, gen["mask"], MODES)


def test_zero_residual_is_the_identity_edit():
    """The property zero-init buys: step zero returns the source CP's own state."""
    gen = build_editor_batch(_cp_batch(), MODES, t_fixed=0.5, noise=None)
    nn_out = _FakeTrunk(scale=0.0)(gen)
    edited, _ = apply_residuals(gen["x_t"], nn_out, gen["mask"], MODES)
    for m in MODES:
        assert torch.equal(edited[m], gen["x_t"][m])


def test_count_trunk_calls_counts_forwards_and_detaches_cleanly():
    """The one-pass claim is exactly this number, and a forward hook is the only check a
    hidden recycle or CFG branch cannot pass."""
    trunk = _FakeTrunk()
    gen = build_editor_batch(_cp_batch(), MODES, t_fixed=0.5, noise=None)
    with count_trunk_calls(trunk) as calls:
        trunk(gen)
        assert calls[0] == 1
        trunk(gen)
        assert calls[0] == 2
    trunk(gen)  # hook removed
    assert calls[0] == 2


def test_input_mutation_guard_notices_an_in_place_edit():
    """An in-place edit of the source CP or the receptor would make contact retention
    compare a sample against itself and read as a perfect score."""
    gen = build_editor_batch(_cp_batch(), MODES, t_fixed=0.5, noise=None)
    snap = snapshot_inputs(gen)
    assert assert_inputs_unmutated(gen, snap) == []
    gen["x_target"] = gen["x_target"] + 1.0
    assert "x_target" in assert_inputs_unmutated(gen, snap)


# ============================================================== the noise adapter


class _FakeFeature(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim


class _FakeNoiseFeature(_FakeFeature):
    pass


_FakeNoiseFeature.__name__ = "EditorNoiseSeqFeat"


class _FakeFactory(torch.nn.Module):
    def __init__(self, dims, noise_at):
        super().__init__()
        self.feat_creators = torch.nn.ModuleList(
            [(_FakeNoiseFeature(d) if i == noise_at else _FakeFeature(d)) for i, d in enumerate(dims)]
        )
        self.linear_out = torch.nn.Linear(sum(dims), 16, bias=False)
        torch.nn.init.zeros_(self.linear_out.weight)


def test_noise_adapter_writes_exactly_the_noise_columns():
    """Off-by-one here is silent and total: the adapter would randomise a PRETRAINED
    feature's columns and leave the noise at zero, so the editor would be deterministic
    AND damaged."""
    dims = [3, 8, 2, 4]
    f = _FakeFactory(dims, noise_at=3)
    span = init_noise_adapter_(f, std=0.05)
    assert span == (13, 4)
    w = f.linear_out.weight
    assert float(w[:, :13].abs().max()) == 0.0, "pretrained columns were disturbed"
    assert float(w[:, 13:].abs().max()) > 0.0, "noise columns were not initialised"


def test_noise_adapter_warns_and_returns_none_without_the_feature():
    """The silent version of this is "the editor is deterministic and nothing said so"."""
    assert init_noise_adapter_(_FakeFactory([3, 8], noise_at=-1)) is None


# ================================================================== arm exclusivity


def test_addon_switch_is_exclusive():
    for a in ADDONS:
        assert validate_addon(a, adversarial_enabled=True) == a
    assert validate_addon("NONE", adversarial_enabled=False) == "none"
    for bad in ("both", "ar1+noise", "nce,ar1", "r1"):
        with pytest.raises(ValueError):
            validate_addon(bad, adversarial_enabled=True)


def test_an_addon_without_an_adversary_is_rejected():
    """All three addons modify the adversarial objective. Configured with the adversary off
    they would be inert, the arm would silently BE the contact-only control, and the table
    would carry a row that measured nothing."""
    for a in ("ar1", "noise", "nce"):
        with pytest.raises(ValueError, match="adversarial.enabled=true"):
            validate_addon(a, adversarial_enabled=False)


# ================================================================== feature scales


def _disc(noise_level_dim: int = 0) -> PeptideInterfaceDiscriminator:
    return PeptideInterfaceDiscriminator(
        hidden_dim=32, nlayers=1, nheads=2, k_pocket=4, noise_level_dim=noise_level_dim
    )


def _feats(D, b: int = 4, n: int = 7, req: bool = False) -> dict:
    return {
        "res": torch.softmax(torch.randn(b, n, 21), -1),
        "geom": torch.randn(b, n, D.n_geom, requires_grad=req),
        "iface": torch.rand(b, n, D.n_iface, requires_grad=req),
        "glob": torch.randn(b, D.n_global, requires_grad=req),
        "mask": torch.ones(b, n, dtype=torch.bool),
    }


def test_continuous_column_masks_exclude_the_categoricals():
    """Ownership of the layout is in the discriminator because it builds the features; a
    copy of these offsets anywhere else goes stale the first time one is appended."""
    D = _disc()
    cm = D.continuous_column_masks()
    # The two terminal flags at the end of `geom`.
    assert cm["geom"][-2:].tolist() == [False, False]
    assert cm["geom"][:-2].all()
    # The one-hot pocket-identity block inside `iface`.
    assert int(cm["iface"].sum()) == D.k_pocket + 2
    assert cm["glob"].all()
    assert "res" not in cm and "mask" not in cm


def test_feature_scales_freeze_and_floor():
    """A scale that keeps moving makes both addons' effective amplitude drift for reasons
    unrelated to the controller. And a constant channel has std exactly 0, which turns the
    first standardisation into inf and then NaN."""
    D = _disc()
    sc = FeatureScaleCache(feature_widths(D), min_examples=8, std_floor=1e-3)
    assert not sc.ready
    sc.update(_feats(D, b=4))
    assert not sc.ready
    sc.update(_feats(D, b=4))
    assert sc.ready
    before = sc.stats("geom", torch.device("cpu"), torch.float32)[1].clone()
    sc.update(_feats(D, b=4))  # frozen: a no-op
    assert torch.equal(sc.stats("geom", torch.device("cpu"), torch.float32)[1], before)

    const = FeatureScaleCache({"glob": 3}, min_examples=1, std_floor=0.25)
    const.update({"glob": torch.ones(4, 3), "mask": torch.ones(4, 2, dtype=torch.bool)})
    _, std = const.stats("glob", torch.device("cpu"), torch.float32)
    assert torch.allclose(std, torch.full((3,), 0.25)), "the std floor did not apply"


# ==================================================================== arm A: aR1


def test_ar1_perturbs_only_continuous_channels_and_reaches_d():
    D = _disc()
    f = _feats(D)
    sc = FeatureScaleCache(feature_widths(D), min_examples=1)
    sc.update(f)
    pen, met = approx_r1_penalty(D, f, sc, sigma_rel=0.05)
    assert float(pen) >= 0.0
    assert int(met["ar1_n_perturbed_cols"]) == (D.n_geom - 2) + (D.k_pocket + 2) + D.n_global
    pen.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in D.parameters())


def test_ar1_is_zero_at_zero_sigma_and_restores_train_mode():
    """Zero sigma is the algebraic check that the two forwards differ ONLY by the
    perturbation -- if it is nonzero, dropout is live and the penalty is measuring the
    sampling variance of D's own regularisation instead of its input sensitivity."""
    D = _disc()
    D.train()
    f = _feats(D)
    sc = FeatureScaleCache(feature_widths(D), min_examples=1)
    sc.update(f)
    pen, _ = approx_r1_penalty(D, f, sc, sigma_rel=0.0)
    assert float(pen) == pytest.approx(0.0, abs=1e-10)
    assert D.training, "approx_r1_penalty left the discriminator in eval mode"


def test_ar1_no_ops_before_the_scales_are_ready():
    D = _disc()
    sc = FeatureScaleCache(feature_widths(D), min_examples=1000)
    pen, met = approx_r1_penalty(D, _feats(D), sc, sigma_rel=0.01)
    assert float(pen) == 0.0 and "ar1_skipped_scales_not_ready" in met


# ================================================= arm B: adaptive feature noise


def test_corruption_leaves_categorical_channels_bit_identical():
    """Not "almost equal": round-tripping an untouched column through
    `(v - mean)/std * std + mean` leaves ~1e-8 of float error on a one-hot, which would
    make "categorical features are unchanged" a claim no test could assert."""
    D = _disc(noise_level_dim=8)
    f = _feats(D)
    sc = FeatureScaleCache(feature_widths(D), min_examples=1)
    sc.update(f)
    ctl = FeatureNoiseController()
    ctl.p.fill_(1.0)
    st = AddonRandomStreams(0, torch.device("cpu"))
    lv = torch.full((f["mask"].shape[0],), 0.2)
    out = ctl.corrupt(f, sc, lv, D.continuous_column_masks(), generator=st.dev)
    cm = D.continuous_column_masks()
    assert torch.equal(out["res"], f["res"])
    assert torch.equal(out["iface"][..., ~cm["iface"]], f["iface"][..., ~cm["iface"]])
    assert torch.equal(out["geom"][..., -2:], f["geom"][..., -2:])
    assert (out["geom"][..., :-2] - f["geom"][..., :-2]).abs().sum() > 0


def test_corruption_preserves_gradient_to_the_generator():
    """G is scored through the same transformation D is trained under, so the corruption
    has to pass gradient. Detaching here would silence the adversarial term entirely."""
    D = _disc(noise_level_dim=8)
    f = _feats(D, req=True)
    sc = FeatureScaleCache(feature_widths(D), min_examples=1)
    sc.update({k: (v.detach() if torch.is_tensor(v) else v) for k, v in f.items()})
    ctl = FeatureNoiseController()
    ctl.p.fill_(1.0)
    st = AddonRandomStreams(0, torch.device("cpu"))
    out = ctl.corrupt(f, sc, torch.full((4,), 0.2), D.continuous_column_masks(), generator=st.dev)
    D(out, noise_level=torch.full((4,), 0.2)).sum().backward()
    assert f["geom"].grad is not None and f["geom"].grad.abs().sum() > 0


def test_level_pool_is_half_clean_and_bounded():
    """Half the batch is shown clean, as in the reference's `t_epl`; and the amplitude
    never exceeds `max_std`, so "raise the noise until D stops objecting" has a ceiling."""
    ctl = FeatureNoiseController(max_std=0.2, n_levels=32, pool_size=64)
    ctl.p.fill_(1.0)
    st = AddonRandomStreams(1, torch.device("cpu"))
    draws = torch.cat([ctl.sample_levels(64, torch.device("cpu"), cpu_generator=st.cpu) for _ in range(20)])
    frac_clean = float((draws == 0).float().mean())
    assert 0.35 < frac_clean < 0.65, f"clean fraction {frac_clean} is not about half"
    assert float(draws.max()) <= 0.2 + 1e-6


def test_controller_follows_the_real_logit_sign():
    """The reference's loop: confident on (corrupted) reals -> raise the noise; losing on
    them -> lower it. Bang-bang on the SIGN of the error, and clipped to [0, 1]."""
    ctl = FeatureNoiseController(target_sign=0.6, adapt_rate_examples=8.0, adapt_every_n_steps=1)
    for step in range(8):
        ctl.observe(torch.full((4,), 3.0))
        ctl.maybe_adapt(step)
    assert float(ctl.p) == pytest.approx(1.0), "p did not rise while D was winning on reals"
    for step in range(8, 40):
        ctl.observe(torch.full((4,), -3.0))
        ctl.maybe_adapt(step)
    assert float(ctl.p) == pytest.approx(0.0), "p did not fall while D was losing on reals"


def test_noise_level_is_required_exactly_when_the_head_has_it():
    """A silently zero-filled level would tell D every example is clean, which is the one
    thing that makes the corruption uninformative rather than merely hard."""
    D = _disc(noise_level_dim=8)
    with pytest.raises(ValueError, match="noise_level"):
        D(_feats(D))
    plain = _disc(noise_level_dim=0)
    with pytest.raises(ValueError, match="noise_level_dim=0"):
        plain(_feats(plain), noise_level=torch.zeros(4))


def test_noise_level_columns_are_appended_last():
    """So `_splice_pretrained_weights` can warm-start this arm's D from the common fork:
    the head's leading columns are copied and only the new ones are zero-padded."""
    plain, noisy = _disc(0), _disc(16)
    assert noisy.head[1].in_features == plain.head[1].in_features + 16


# ============================================ arm C: local contrastive retention


def _chain(b: int = 2, n: int = 10, jitter: float = 0.0) -> torch.Tensor:
    x = torch.randn(b, n, 37, 3) * 0.2
    for i in range(n):
        x[:, i, :, 0] += i * 0.38
    return x + jitter * torch.randn_like(x)


def test_nce_is_monotone_in_how_far_the_edit_moved():
    """The term's job. Its FLOOR is not zero -- a random projection leaves descriptors
    nearly collinear -- so only the ordering is meaningful, which is what this asserts."""
    torch.manual_seed(0)
    nce = LocalRetentionNCE(k_neighbors=4, max_positions=6)
    src = _chain()
    kw = dict(
        src_atom_mask=torch.ones(2, 10, 37, dtype=torch.bool),
        pep_mask=torch.ones(2, 10, dtype=torch.bool),
        target_atom37=torch.randn(2, 12, 37, 3) * 0.5,
        target_atom_mask=torch.ones(2, 12, 37, dtype=torch.bool),
        target_mask=torch.ones(2, 12, dtype=torch.bool),
    )
    losses = [
        float(nce(gen_atom37=src + amp * torch.randn_like(src), src_atom37=src, **kw)[0])
        for amp in (0.0, 0.25, 0.75)
    ]
    assert losses[0] < losses[1] < losses[2], f"not monotone in the perturbation: {losses}"


def test_nce_never_scores_the_termini():
    """Opening the ring IS the task, so asking the termini to keep their local environment
    is asking the edit not to happen."""
    nce = LocalRetentionNCE(k_neighbors=4, max_positions=99, exclude_terminal=2)
    src = _chain(b=1, n=9)
    chosen, _ = nce._select(
        src,
        torch.ones(1, 9, 37, dtype=torch.bool),
        torch.ones(1, 9, dtype=torch.bool),
        torch.randn(1, 12, 37, 3) * 0.5,
        torch.ones(1, 12, 37, dtype=torch.bool),
        torch.ones(1, 12, dtype=torch.bool),
    )
    assert not bool(chosen[0, :2].any()), "a leading terminal residue was scored"
    assert not bool(chosen[0, -2:].any()), "a trailing terminal residue was scored"


def test_nce_skips_an_example_with_too_few_positions_and_says_so():
    """Fewer than two positions makes the softmax over one class a constant. Silently
    returning zero would look like a satisfied term."""
    nce = LocalRetentionNCE(k_neighbors=4, max_positions=6, exclude_terminal=2)
    src = _chain(b=1, n=4)  # 4 residues, 4 excluded -> nothing scorable
    loss, met = nce(
        gen_atom37=src,
        src_atom37=src,
        src_atom_mask=torch.ones(1, 4, 37, dtype=torch.bool),
        pep_mask=torch.ones(1, 4, dtype=torch.bool),
        target_atom37=torch.randn(1, 12, 37, 3),
        target_atom_mask=torch.ones(1, 12, 37, dtype=torch.bool),
        target_mask=torch.ones(1, 12, dtype=torch.bool),
    )
    assert float(loss) == 0.0
    assert int(met["nce_n_scored"]) == 0 and int(met["nce_n_skipped_examples"]) == 1


def test_nce_gradient_never_reaches_the_source():
    """The source is a fixed target. A gradient path into it would let the term be
    satisfied by moving the SOURCE towards the generated peptide."""
    torch.manual_seed(0)
    nce = LocalRetentionNCE(k_neighbors=4, max_positions=6)
    src = _chain().requires_grad_(True)
    gen = (src.detach() + 0.1 * torch.randn_like(src)).requires_grad_(True)
    loss, _ = nce(
        gen_atom37=gen,
        src_atom37=src,
        src_atom_mask=torch.ones(2, 10, 37, dtype=torch.bool),
        pep_mask=torch.ones(2, 10, dtype=torch.bool),
        target_atom37=torch.randn(2, 12, 37, 3) * 0.5,
        target_atom_mask=torch.ones(2, 12, 37, dtype=torch.bool),
        target_mask=torch.ones(2, 12, dtype=torch.bool),
    )
    loss.backward()
    assert gen.grad is not None and gen.grad.abs().sum() > 0
    assert src.grad is None or float(src.grad.abs().sum()) == 0.0


def test_nce_projection_belongs_to_the_generator_side():
    """It is a trainable head, so it must have parameters to train; and it is this arm's
    alone, which is why the other four build no `nce` module at all."""
    assert sum(p.numel() for p in LocalRetentionNCE().parameters()) > 0
    assert sum(p.numel() for p in LocalRetentionNCE(use_projection=False).parameters()) == 0


# ============================================================ export-side measures


def test_terminal_bond_records_a_distance_not_a_verdict():
    """A boolean discards the number, and the number is what says whether a rejected
    sample missed the gate by 0.01 nm or by 2 nm."""
    x = torch.zeros(1, 4, 37, 3)
    from proteinfoundation.cp2lp.geometry import C_IDX, N_IDX

    x[0, 3, C_IDX] = torch.tensor([1.0, 0.0, 0.0])
    x[0, 0, N_IDX] = torch.tensor([0.0, 0.0, 0.0])
    assert float(terminal_bond_nm(x, torch.ones(1, 4, dtype=torch.bool))) == pytest.approx(1.0, abs=1e-4)


def test_displacement_is_measured_without_superposition():
    """The receptor is fixed and shared, so the peptides are already in one frame and the
    difference IS the pose change. Aligning first would report only the conformational
    change and hide a sample that slid off its binding mode."""
    src = _chain(b=1, n=8)
    shifted = src + torch.tensor([0.5, 0.0, 0.0])  # a pure rigid translation
    all_rmsd, iface = displacement_from_source_nm(
        gen_atom37=shifted,
        src_atom37=src,
        src_atom_mask=torch.ones(1, 8, 37, dtype=torch.bool),
        pep_mask=torch.ones(1, 8, dtype=torch.bool),
        target_atom37=src[:, :4] + torch.tensor([0.0, 0.3, 0.0]),
        target_atom_mask=torch.ones(1, 4, 37, dtype=torch.bool),
        target_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    assert float(all_rmsd) == pytest.approx(0.5, abs=1e-3), "a rigid shift was superposed away"
    assert torch.isfinite(iface).all() or torch.isnan(iface).all()


def test_interface_displacement_is_nan_not_zero_without_an_interface():
    """A zero would average in as "the interface did not move", which is the opposite of
    "there was no interface"."""
    src = _chain(b=1, n=8)
    _, iface = displacement_from_source_nm(
        gen_atom37=src,
        src_atom37=src,
        src_atom_mask=torch.ones(1, 8, 37, dtype=torch.bool),
        pep_mask=torch.ones(1, 8, dtype=torch.bool),
        target_atom37=src[:, :2] + 100.0,  # far away: no contacts at all
        target_atom_mask=torch.ones(1, 2, 37, dtype=torch.bool),
        target_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert bool(torch.isnan(iface).all())


# =================================================================== the configs


ARM_CONFIGS = {
    "onepass_contactonly": "example/training_cp2lp_onepass_contactonly",
    "onepass_gan": "example/training_cp2lp_onepass_gan",
    "onepass_gan_ar1": "example/training_cp2lp_onepass_gan_ar1",
    "onepass_gan_noise": "example/training_cp2lp_onepass_gan_noise",
    "onepass_gan_nce": "example/training_cp2lp_onepass_gan_nce",
}


def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out.update(_flat(v, key)) if isinstance(v, dict) else out.update({key: v})
    return out


def _resolved(names):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        return {n: _flat(OmegaConf.to_container(compose(config_name=c), resolve=False)) for n, c in names.items()}


#: Derived from `run_name` by interpolation, so a difference here is not an independent
#: change. Listed explicitly rather than pattern-matched: a key that merely CONTAINS the run
#: name could still be a real divergence.
RUN_NAME_DERIVED = {"val_generation.rosetta_dump_dir"}


def test_each_arm_differs_from_the_reference_in_exactly_one_knob():
    """The comparison is only interpretable if every arm IS single-variable against
    `onepass_gan`. Config drift is silent and cheap: one tuned loss weight in one arm and
    that arm stops isolating its addon. Resolved trees, so an override anywhere in the
    defaults chain is caught."""
    cfgs = _resolved(ARM_CONFIGS)
    ref = cfgs["onepass_gan"]
    expected = {
        "onepass_contactonly": "cp2lp.adversarial.enabled",
        "onepass_gan_ar1": "cp2lp.addon",
        "onepass_gan_noise": "cp2lp.addon",
        "onepass_gan_nce": "cp2lp.addon",
    }
    for arm, knob in expected.items():
        got = cfgs[arm]
        diffs = {
            k
            for k in set(ref) | set(got)
            if ref.get(k, "<absent>") != got.get(k, "<absent>")
        } - {"run_name"} - RUN_NAME_DERIVED
        assert diffs == {knob}, f"{arm} drifted from onepass_gan: {sorted(diffs)}"


def test_every_arm_is_a_one_pass_editor_with_the_flow_term_off():
    """`real_lp_flow_weight` must be 0 in this mode: it supervises a velocity field the
    residual heads are not connected to. And nothing replaces it -- no identity loss, no
    distillation. The module raises if this drifts, so this test is the cheap version."""
    for arm, cfg in _resolved(ARM_CONFIGS).items():
        assert cfg["cp2lp.mode"] == "onepass", arm
        assert cfg["cp2lp.real_lp_flow_weight"] == 0.0, arm
        assert cfg["nn.enable_editor_residual"] is True, arm
        assert cfg["cp2lp.onepass.noise_channels"] == cfg["nn.editor_noise_channels"], arm


def test_the_noise_feature_is_appended_last_in_every_arm():
    """Each factory concatenates in list order and projects with ONE Linear, and the splice
    copies the overlapping LEADING columns. Appending preserves every pretrained feature's
    weights; inserting re-binds trained weights to the wrong inputs."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        for arm, name in ARM_CONFIGS.items():
            cfg = compose(config_name=name)
            seq = list(cfg.nn.feats_seq)
            assert seq[-1] == "editor_noise_seq", f"{arm}: noise feature is not last: {seq}"
            # The source-CP block the init was trained with keeps its position.
            assert seq[-4:-1] == ["x_src_cp_bb_ca", "x_src_cp_local_latents", "src_cp_present"], arm


def test_all_five_arms_share_the_fork_and_the_ae():
    """The whole design is "one checkpoint, five arms, one line apart". Five arms on
    different forks would differ by a warmup, which is a live alternative explanation for
    any addon result."""
    cfgs = _resolved(ARM_CONFIGS)
    forks = {c["pretrain_ckpt_path"] for c in cfgs.values()}
    aes = {c["autoencoder_ckpt_path"] for c in cfgs.values()}
    seeds = {c["seed"] for c in cfgs.values()}
    budgets = {c["cp2lp.max_g_updates"] for c in cfgs.values()}
    assert len(forks) == 1, f"the arms start from different checkpoints: {forks}"
    assert len(aes) == 1, f"the arms use different autoencoders: {aes}"
    assert len(seeds) == 1 and len(budgets) == 1
    assert "fork" in next(iter(forks)), f"the arms do not start from the fork: {forks}"


def test_the_warmup_starts_from_the_lpmix_init_not_from_its_own_fork():
    """It is what PRODUCES the fork, so it cannot start from it -- and the pairing is not a
    free variable: `local_latents` is defined by the AE, and the lpmix run is the only CPSea
    flow checkpoint trained against the shared LP/CP AE."""
    cfgs = _resolved({"warmup": "example/training_cp2lp_onepass_warmup", **ARM_CONFIGS})
    assert "cp2lp_init.ckpt" in cfgs["warmup"]["pretrain_ckpt_path"]
    assert cfgs["warmup"]["cp2lp.adversarial.enabled"] is False
    assert cfgs["warmup"]["cp2lp.max_g_updates"] == 300
    assert cfgs["warmup"]["autoencoder_ckpt_path"] == cfgs["onepass_gan"]["autoencoder_ckpt_path"]


def test_arm_budgets_account_for_the_forks_inherited_update_count():
    """`g_updates` is a PERSISTENT buffer, so the arms load the fork's 300 with the weights.
    A budget of 10000 would therefore give 9700 post-warmup updates, and the milestones
    would land 300 short of where the comparison says they are."""
    cfgs = _resolved({"warmup": "example/training_cp2lp_onepass_warmup", **ARM_CONFIGS})
    warm = cfgs["warmup"]["cp2lp.max_g_updates"]
    for arm in ARM_CONFIGS:
        assert cfgs[arm]["cp2lp.max_g_updates"] == warm + 10000, arm
        assert list(cfgs[arm]["cp2lp.eval_checkpoint_g_updates"]) == [warm + 1000, warm + 5000, warm + 10000], arm


def test_preamble_run_names_track_the_onepass_configs():
    """`cp2lp_train.sbatch` passes `++run_name=${RUN_NAME}`, a Hydra override that BEATS the
    yaml. A stale default silently points a new config's run at another arm's directory and
    overwrites the checkpoints and wandb history of the arm being compared against."""
    preamble = (REPO / "scripts" / "_cp2lp_preamble.sh").read_text()
    cfgs = _resolved({"onepass_warmup": "example/training_cp2lp_onepass_warmup", **ARM_CONFIGS})
    for arm, cfg in cfgs.items():
        expected = f"{arm})"
        assert any(
            line.strip().startswith(expected) and cfg["run_name"] in line
            for line in preamble.splitlines()
        ), f"preamble RUN_NAME for {arm} does not match the config's {cfg['run_name']!r}"


def test_the_module_gates_the_onepass_warmup_on_generator_updates():
    """MEASURED (lightning 2.5.6, manual optimization): `global_step` counts G+D optimizer
    steps, so an adversarial arm advances it twice as fast as the control. A warmup or a
    budget expressed in it means a different amount of generator training in each arm.

    A source-level check: exercising it needs a Trainer, two optimizers and a GPU batch.
    """
    import re

    src = (REPO / "src" / "proteinfoundation" / "cp2lp" / "module.py").read_text()
    required = {
        "a persistent G-update counter": r'register_buffer\(\s*"g_updates"',
        "incremented on the G step": r"self\.g_updates\s*\+=\s*1",
        # Accepts the modes-tuple form too: the BACKBONE arms also step two optimizers, so
        # they must count G updates for exactly the reason one-pass does. What has to hold is
        # that `progress` is g_updates for those modes -- not the precise spelling.
        "warmup gated on it in the two-optimizer modes": (
            r'progress\s*=\s*\(?\s*int\(self\.g_updates\)\s*if\s*self\.cp2lp_mode\s*'
            r'(?:==\s*"onepass"|in\s*\(\s*"onepass",\s*"backbone"\s*\))'
        ),
        "budget gated on it": r"self\.g_updates\)\s*>=\s*self\.max_g_updates",
    }
    for what, pattern in required.items():
        assert re.search(pattern, src), f"module.py no longer has {what} ({pattern})"


def test_milestone_checkpoints_cannot_be_mistaken_for_the_resume_checkpoint():
    """The retained G-update milestones live in the same directory as `last.ckpt`.

    `fetch_last_ckpt` scans that directory, and an arm that resumed from `g010300.ckpt`
    would silently restart from a milestone instead of from where it left off -- or, worse,
    from an `-EMA` file, which would resume optimisation from an average of the last few
    hundred updates rather than from the trajectory. Verified rather than assumed, because
    the naming is mine and the scanner is not.
    """
    from proteinfoundation.utils.fetch_last_ckpt import get_version_number

    def considered(name: str) -> bool:
        return "last" in name and name.endswith(".ckpt") and get_version_number(name) is not None

    for name in ("last.ckpt", "last-v1.ckpt", "last-v12.ckpt"):
        assert considered(name), f"{name} should be a resume candidate"
    for name in (
        "g000300.ckpt",
        "g001300.ckpt",
        "g005300-EMA.ckpt",
        "g010300.ckpt",
        "g010300-EMA.ckpt",
        "last-EMA.ckpt",
        "cp2lp_onepass_fork.ckpt",
    ):
        assert not considered(name), f"{name} must NOT be a resume candidate"


def test_the_milestone_writer_names_files_by_generator_updates():
    """Source-level: the filename has to carry the number the comparison is indexed by, or
    two arms' milestones are not comparable. Exercising it needs a Trainer and a GPU."""
    import re

    src = (REPO / "src" / "proteinfoundation" / "cp2lp" / "module.py").read_text()
    assert re.search(r'f"g\{n:06d\}\.ckpt"', src), "the raw milestone is not named by g_updates"
    assert re.search(r'f"g\{n:06d\}-EMA\.ckpt"', src), "the EMA milestone is not named by g_updates"
    # And it must go through the EMA callback's own swap, or the -EMA file holds raw weights.
    assert "save_ema_model" in src, "the EMA milestone is not written through the EMA swap"


def test_the_fork_pin_job_does_not_override_the_milestone_preference():
    """`pin_cp2lp_fork.sh` prefers the exact G-update milestone ONLY when given no argument.

    MEASURED (pin job 69416): the sbatch hardcoded `last.ckpt` and passed it as $1, which
    the script treats as an explicit override -- so the milestone-preference logic was dead
    and the fork pinned at g=250 instead of g=300. The fix was in the helper; the bug was in
    the caller. This test guards the caller, which is the half that was missed.
    """
    sbatch = (REPO / "scripts" / "cp2lp_fork_pin.sbatch").read_text()
    invocations = [ln.strip() for ln in sbatch.splitlines() if "pin_cp2lp_fork.sh" in ln and not ln.strip().startswith("#")]
    assert invocations, "the fork-pin job no longer calls pin_cp2lp_fork.sh"
    for line in invocations:
        after = line.split("pin_cp2lp_fork.sh", 1)[1].strip()
        assert after == "", (
            f"the fork-pin job passes an explicit source argument ({after!r}), which "
            "overrides the milestone preference in pin_cp2lp_fork.sh and silently pins a "
            "checkpoint short of max_g_updates."
        )


def test_the_pin_script_prefers_a_milestone_and_survives_having_none():
    """Both halves of the helper: the glob must exclude the -EMA twin (the arms continue
    training, so they need raw weights), and a missing glob must not abort under `set -e`
    before the 'run the warmup first' message -- a failing command substitution inside an
    assignment does exactly that."""
    src = (REPO / "scripts" / "pin_cp2lp_fork.sh").read_text()
    assert "g[0-9][0-9][0-9][0-9][0-9][0-9].ckpt" in src, "the milestone glob is gone"
    assert "-EMA" not in src.split("SRC=\"$(ls")[1].split(")\"")[0], "the glob would match the -EMA twin"
    glob_line = next(ln for ln in src.splitlines() if "ls -1" in ln and "g[0-9]" in ln)
    assert "|| true" in glob_line, "the glob substitution would abort the script under set -e"


def test_the_bond_tolerance_has_one_definition():
    """The export gate's tolerance is calibrated against 536 native LPs, so it must not
    exist as two independent literals.

    `geometry.PEPTIDE_BOND_TOL_NM` previously sat unreferenced while `losses.py` carried its
    own `0.02` -- a named constant that looks authoritative and controls nothing. Anyone
    editing it would believe they had moved the gate and would not have.
    """
    import inspect

    from proteinfoundation.cp2lp.geometry import PEPTIDE_BOND_TOL_NM
    from proteinfoundation.cp2lp.losses import open_chain_geometry_loss

    default = inspect.signature(open_chain_geometry_loss).parameters["bond_tol_nm"].default
    assert default == PEPTIDE_BOND_TOL_NM, (
        f"the gate's tolerance is duplicated: losses default {default} vs "
        f"geometry.PEPTIDE_BOND_TOL_NM {PEPTIDE_BOND_TOL_NM}"
    )
    src = (REPO / "src" / "proteinfoundation" / "cp2lp" / "losses.py").read_text()
    assert "bond_tol_nm: float = PEPTIDE_BOND_TOL_NM" in src, "the default is not the shared constant"
