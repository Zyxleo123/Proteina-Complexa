#!/usr/bin/env python3
"""Unit tests for unrolled clean prediction and gradient-carrying (DRaFT-K) rollouts.

Covers the two fixes for the gap between what the cyclization losses optimise and
what the sampler emits:

  * `proteinfoundation.training.unrolled_prediction` -- penalise the endpoint of a
    short ODE unroll instead of the one-step conditional mean (the Jensen gap).
  * `ProductSpaceFlowMatcher.full_simulation_draft` / `partial_simulation(grad_enabled=)`
    -- integrate the DEPLOYED sampler from t=0 and keep gradients through the last
    K steps (exposure bias).

Pure-tensor tests: no GPU, no dataset, no trained model. The "network" is a toy
`nn.Module` so gradient flow can be asserted exactly.

Usage (as a standalone script):
    python script_utils/test_unrolled_and_rollout.py

Usage (via pytest, if installed):
    pytest script_utils/test_unrolled_and_rollout.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from proteinfoundation.cyclization.bond_loss import flat_bottom_penalty
from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from proteinfoundation.training.unrolled_prediction import (
    unroll_alpha,
    unroll_time,
    unrolled_clean_prediction,
)

torch.manual_seed(0)

B, N, D_CA, D_LAT = 3, 5, 3, 4
MODES = ("bb_ca", "local_latents")


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class ToyNet(nn.Module):
    """Predicts `x_1` as an affine map of `x_t`, and records how it was called.

    `x_1 = w * x_t + b` with a single shared scalar per mode, which is enough to make
    the whole unroll a differentiable function of `w` while keeping the reference
    computation something a test can write out by hand.
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.25))
        self.calls: list[dict] = []

    def forward(self, batch):
        self.calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "t": {m: batch["t"][m].clone() for m in MODES},
                "has_x_sc": "x_sc" in batch,
            }
        )
        return {m: {"x_1": self.w * batch["x_t"][m] + self.b} for m in MODES}


def make_fm() -> ProductSpaceFlowMatcher:
    cfg = SimpleNamespace(
        product_flowmatcher={
            "bb_ca": {"zero_com_noise": False, "guidance_enabled": True, "dim": D_CA},
            "local_latents": {"zero_com_noise": False, "guidance_enabled": True, "dim": D_LAT},
        }
    )
    return ProductSpaceFlowMatcher(cfg)


def make_batch(t_value=None):
    mask = torch.ones(B, N, dtype=torch.bool)
    t = {
        "bb_ca": torch.rand(B) * 0.8 if t_value is None else torch.full((B,), float(t_value)),
        "local_latents": torch.rand(B) * 0.8 if t_value is None else torch.full((B,), float(t_value)),
    }
    return {
        "mask": mask,
        "x_t": {"bb_ca": torch.randn(B, N, D_CA), "local_latents": torch.randn(B, N, D_LAT)},
        "t": t,
    }


def call_nn_of(net):
    def call_nn(batch, n_recycle=0):
        return net(batch)

    return call_nn


# --------------------------------------------------------------------------------------
# Schedule arithmetic
# --------------------------------------------------------------------------------------


def test_unroll_time_starts_at_t_and_never_reaches_one():
    t = torch.tensor([0.0, 0.4, 0.95])
    k = 4
    assert torch.allclose(unroll_time(t, 0, k), t)
    last = unroll_time(t, k - 1, k)
    assert torch.all(last < 1.0), "evaluating at t=1 would divide by zero in v = (x_1-x_t)/(1-t)"
    # Uniform in the remaining interval: the gap is (1-t)/k at every step.
    for step in range(k - 1):
        gap = unroll_time(t, step + 1, k) - unroll_time(t, step, k)
        assert torch.allclose(gap, (1.0 - t) / k)


def test_unroll_alpha_is_one_on_the_final_step():
    for k in (1, 2, 5, 17):
        assert unroll_alpha(k - 1, k) == 1.0, "the last Euler step must land exactly on x_hat_1"
        assert unroll_alpha(0, k) == 1.0 / k


def test_unroll_alpha_matches_the_euler_ratio_it_replaces():
    """`1/(k-step)` is a closed form for `dt / (1 - t_step)`; check it against the raw ratio."""
    t = torch.tensor([0.1, 0.6])
    k = 5
    for step in range(k):
        t_step = unroll_time(t, step, k)
        dt = (1.0 - t) / k
        raw = dt / (1.0 - t_step)
        assert torch.allclose(raw, torch.full_like(raw, unroll_alpha(step, k)), atol=1e-6)


# --------------------------------------------------------------------------------------
# unrolled_clean_prediction
# --------------------------------------------------------------------------------------


def test_k1_reproduces_the_one_step_prediction_exactly():
    fm, net, batch = make_fm(), ToyNet(), make_batch()
    one_step = fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=net(batch))
    unrolled = unrolled_clean_prediction(fm, call_nn_of(net), batch, k_steps=1, grad_steps=1)
    for m in MODES:
        assert torch.allclose(one_step[m], unrolled[m], atol=1e-6)


def test_unroll_matches_an_explicit_euler_reference():
    fm, net, batch = make_fm(), ToyNet(), make_batch()
    k = 4
    got = unrolled_clean_prediction(fm, call_nn_of(net), batch, k_steps=k, grad_steps=k)

    # Reference: the same recursion written out, no library help.
    x = {m: batch["x_t"][m].clone() for m in MODES}
    for step in range(k):
        alpha = 1.0 / (k - step)
        for m in MODES:
            x_1 = net.w * x[m] + net.b
            x[m] = x[m] + (x_1 - x[m]) * alpha
    for m in MODES:
        assert torch.allclose(got[m], x[m], atol=1e-6)


def test_oracle_network_makes_the_unroll_a_fixed_point():
    """A network that already knows x_1 must return it, for every k -- no drift from unrolling."""
    fm, batch = make_fm(), make_batch()
    target = {"bb_ca": torch.randn(B, N, D_CA), "local_latents": torch.randn(B, N, D_LAT)}

    def call_nn(b, n_recycle=0):
        return {m: {"x_1": target[m].clone()} for m in MODES}

    for k in (1, 2, 8):
        got = unrolled_clean_prediction(fm, call_nn, batch, k_steps=k, grad_steps=1)
        for m in MODES:
            assert torch.allclose(got[m], target[m], atol=1e-6), f"k={k}"


def test_unroll_does_not_mutate_the_caller_batch():
    """The flow loss, chirality loss and replay mixer all read `batch` after this runs."""
    fm, net, batch = make_fm(), ToyNet(), make_batch()
    x_t_before = {m: batch["x_t"][m].clone() for m in MODES}
    t_before = {m: batch["t"][m].clone() for m in MODES}
    keys_before = set(batch)

    unrolled_clean_prediction(fm, call_nn_of(net), batch, k_steps=4, grad_steps=1)

    assert set(batch) == keys_before, "unroll leaked keys into the training batch"
    for m in MODES:
        assert torch.equal(batch["x_t"][m], x_t_before[m])
        assert torch.equal(batch["t"][m], t_before[m])


def test_grad_steps_limits_the_backprop_window():
    fm, net, batch = make_fm(), ToyNet(), make_batch()
    k, g = 5, 2
    out = unrolled_clean_prediction(fm, call_nn_of(net), batch, k_steps=k, grad_steps=g)
    assert len(net.calls) == k
    grad_on = [c["grad_enabled"] for c in net.calls]
    assert grad_on == [False] * (k - g) + [True] * g, grad_on
    assert out["bb_ca"].requires_grad
    out["bb_ca"].sum().backward()
    assert net.w.grad is not None and torch.isfinite(net.w.grad)


def test_full_grad_window_gives_a_larger_gradient_than_a_truncated_one():
    """Sanity that truncation is actually truncating something, not a no-op."""
    fm, batch = make_fm(), make_batch(t_value=0.2)
    grads = {}
    for g in (1, 4):
        net = ToyNet()
        out = unrolled_clean_prediction(fm, call_nn_of(net), batch, k_steps=4, grad_steps=g)
        out["bb_ca"].sum().backward()
        grads[g] = net.w.grad.abs().item()
    assert grads[4] > grads[1], grads


def test_no_grad_context_is_never_overridden():
    fm, net, batch = make_fm(), ToyNet(), make_batch()
    with torch.no_grad():
        out = unrolled_clean_prediction(fm, call_nn_of(net), batch, k_steps=3, grad_steps=3)
    assert not out["bb_ca"].requires_grad
    assert all(not c["grad_enabled"] for c in net.calls)


def test_nn_out_0_reuse_saves_a_forward_and_changes_nothing():
    fm, batch = make_fm(), make_batch()
    k = 4

    net_a = ToyNet()
    out_a = unrolled_clean_prediction(fm, call_nn_of(net_a), batch, k_steps=k, grad_steps=1)

    net_b = ToyNet()
    nn_out_0 = net_b(batch)
    n_calls_after_seed = len(net_b.calls)
    out_b = unrolled_clean_prediction(
        fm, call_nn_of(net_b), batch, k_steps=k, grad_steps=1, nn_out_0=nn_out_0
    )
    assert len(net_b.calls) - n_calls_after_seed == k - 1, "step 0 should have been reused"
    for m in MODES:
        assert torch.allclose(out_a[m], out_b[m], atol=1e-6)


def test_self_cond_feeds_the_previous_clean_prediction():
    fm, net, batch = make_fm(), ToyNet(), make_batch()
    unrolled_clean_prediction(fm, call_nn_of(net), batch, k_steps=3, grad_steps=1, self_cond=True)
    assert [c["has_x_sc"] for c in net.calls] == [False, True, True]

    net2 = ToyNet()
    unrolled_clean_prediction(fm, call_nn_of(net2), batch, k_steps=3, grad_steps=1, self_cond=False)
    assert not any(c["has_x_sc"] for c in net2.calls)


def test_unroll_advances_time_toward_one():
    fm, net, batch = make_fm(), ToyNet(), make_batch()
    k = 4
    unrolled_clean_prediction(fm, call_nn_of(net), batch, k_steps=k, grad_steps=1)
    ts = [c["t"]["bb_ca"] for c in net.calls]
    assert torch.allclose(ts[0], batch["t"]["bb_ca"])
    for a, b in zip(ts, ts[1:]):
        assert torch.all(b > a)


# --------------------------------------------------------------------------------------
# The premise: why the one-step mean is the wrong object for a closure constraint
# --------------------------------------------------------------------------------------


def test_flat_bottom_penalty_of_the_mean_understates_the_mean_penalty():
    """The Jensen gap this whole module exists to close, on a two-mode toy posterior.

    Two equally likely ring conformations, one with the anchors 0.6 A too close and one
    0.6 A too far. Their MEAN distance lands dead centre in the acceptance window, so a
    loss read off the conditional mean reports a perfect zero -- while every actual
    sample is outside the window and would be scored as a failed closure.
    """
    lo, hi = torch.tensor(1.20), torch.tensor(1.45)
    centre = (lo + hi) / 2
    samples = torch.stack([centre - 0.6, centre + 0.6])

    penalty_of_mean = flat_bottom_penalty(samples.mean(), lo, hi)
    mean_of_penalty = flat_bottom_penalty(samples, lo, hi).mean()

    assert penalty_of_mean.item() == 0.0
    assert mean_of_penalty.item() > 0.2
    inside = (samples >= lo) & (samples <= hi)
    assert not inside.any(), "neither sample closes, yet the mean-based loss is zero"


# --------------------------------------------------------------------------------------
# full_simulation_draft
# --------------------------------------------------------------------------------------


SAMPLING_ARGS = {
    m: {
        "schedule": {"mode": "uniform", "p": 1.0},
        "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
        "simulation_step_params": {
            "sampling_mode": "vf",
            "sc_scale_noise": 0.0,
            "sc_scale_score": 1.0,
            "t_lim_ode": 0.98,
            "t_lim_ode_below": 0.02,
            "tsr_k": 1.0,
            "tsr_sigma": 1.0,
            "center_every_step": False,
        },
    }
    for m in MODES
}


def predict_for_sampling_of(net):
    def predict_for_sampling(batch, mode="full", n_recycle=0):
        return net(batch)

    return predict_for_sampling


def run_draft(net, nsteps, grad_steps):
    fm = make_fm()
    mask = torch.ones(B, N, dtype=torch.bool)
    return fm.full_simulation_draft(
        batch={"mask": mask},
        predict_for_sampling=predict_for_sampling_of(net),
        nsteps=nsteps,
        nsamples=B,
        n=N,
        self_cond=False,
        sampling_model_args=SAMPLING_ARGS,
        device=torch.device("cpu"),
        grad_steps=grad_steps,
    )


def test_draft_rollout_reproduces_full_simulation_exactly():
    """Same schedule, same noise, same steps -- the graph is the ONLY difference.

    If this ever fails, the rollout is training a sampler that design does not run,
    which is the one thing this path must not do.
    """
    fm = make_fm()
    mask = torch.ones(B, N, dtype=torch.bool)
    net = ToyNet()

    torch.manual_seed(1234)
    reference = fm.full_simulation(
        batch={"mask": mask},
        predict_for_sampling=predict_for_sampling_of(net),
        nsteps=6,
        nsamples=B,
        n=N,
        self_cond=False,
        sampling_model_args=SAMPLING_ARGS,
        device=torch.device("cpu"),
    )

    torch.manual_seed(1234)
    drafted = run_draft(net, nsteps=6, grad_steps=2)

    for m in MODES:
        assert torch.allclose(reference[m], drafted[m].detach(), atol=1e-6), m


def test_draft_rollout_grad_window():
    net = ToyNet()
    nsteps, grad_steps = 6, 2
    out = run_draft(net, nsteps=nsteps, grad_steps=grad_steps)
    grad_on = [c["grad_enabled"] for c in net.calls]
    assert len(grad_on) == nsteps
    assert grad_on == [False] * (nsteps - grad_steps) + [True] * grad_steps, grad_on
    assert out["bb_ca"].requires_grad
    out["bb_ca"].sum().backward()
    assert net.w.grad is not None and torch.isfinite(net.w.grad)


def test_draft_rollout_grad_steps_is_clamped():
    net = ToyNet()
    out = run_draft(net, nsteps=3, grad_steps=99)
    assert all(c["grad_enabled"] for c in net.calls)
    assert out["bb_ca"].requires_grad


def test_partial_simulation_still_defaults_to_no_grad():
    """Every inference caller (beam search, FK steering, MCTS) relies on this default."""
    fm = make_fm()
    mask = torch.ones(B, N, dtype=torch.bool)
    net = ToyNet()
    ts, gt = fm.sample_schedule(nsteps=4, sampling_model_args=SAMPLING_ARGS)
    x = fm.sample_noise(N, shape=(B,), device=torch.device("cpu"), mask=mask)
    x_out, _ = fm.partial_simulation(
        batch={"mask": mask},
        x=x,
        x_1_pred=None,
        mask=mask,
        predict_for_sampling=predict_for_sampling_of(net),
        start_step=0,
        end_step=4,
        self_cond=False,
        ts=ts,
        gt=gt,
        simulation_step_params={m: SAMPLING_ARGS[m]["simulation_step_params"] for m in MODES},
        device=torch.device("cpu"),
    )
    assert not x_out["bb_ca"].requires_grad
    assert all(not c["grad_enabled"] for c in net.calls)


# --------------------------------------------------------------------------------------
# Proteina.compute_rollout_finetune_loss wiring
# --------------------------------------------------------------------------------------


def _disulfide_batch(n=4, gap_A=6.0):
    """A [1, n] batch whose decoded SG-SG distance is `gap_A` -- i.e. an OPEN ring."""
    from proteinfoundation.cyclization.constants import AA_CYS, DISULFIDE
    from proteinfoundation.eval.cyclic_reconstruction_metrics import SG_IDX

    coords = torch.zeros(1, n, 37, 3)
    atom_mask = torch.zeros(1, n, 37, dtype=torch.bool)
    for r in range(n):
        atom_mask[0, r, SG_IDX] = True
    coords[0, n - 1, SG_IDX, 0] = gap_A / 10.0  # nm
    aa = torch.full((1, n), AA_CYS, dtype=torch.long)
    batch = {
        "mask": torch.ones(1, n, dtype=torch.bool),
        "cyclization_i": torch.tensor([0]),
        "cyclization_j": torch.tensor([n - 1]),
        "cyclization_type": torch.tensor([DISULFIDE]),
        "has_cyclization": torch.tensor([True]),
        # Interpolation state that must NOT reach the sampler.
        "x_1": {"bb_ca": torch.randn(1, n, 3)},
        "x_t": {"bb_ca": torch.randn(1, n, 3)},
        "t": {"bb_ca": torch.rand(1)},
        "x_0": {"bb_ca": torch.randn(1, n, 3)},
        "x_sc": {"bb_ca": torch.randn(1, n, 3)},
    }
    return batch, coords, atom_mask, aa


def _rollout_stub(n=4, global_step=0, every_n=1, with_design=True):
    """Minimal duck-typed `Proteina` for `compute_rollout_finetune_loss`."""
    from proteinfoundation.cyclization.constants import AA_CYS

    seen = {}

    def fake_draft(**kw):
        seen["batch_keys"] = set(kw["batch"])
        seen["nsteps"] = kw["nsteps"]
        seen["grad_steps"] = kw["grad_steps"]
        seen["nsamples"] = kw["nsamples"]
        seen["ag_ratio"] = kw["ag_ratio"]
        return {"bb_ca": torch.zeros(kw["nsamples"], n, 3), "local_latents": torch.zeros(kw["nsamples"], n, 8)}

    design = SimpleNamespace(
        args=SimpleNamespace(nsteps=400, self_cond=True, get=lambda k, d=None: {"guidance_w": 1.0}.get(k, d)),
        model=SAMPLING_ARGS,
        get=lambda k, d=None: {"n_recycle": 0}.get(k, d),
    )
    cfg = {"enabled": True, "nsteps": 12, "design_sampling": design if with_design else None}
    stub = SimpleNamespace(
        global_step=global_step,
        rollout_ft_cfg=SimpleNamespace(get=lambda k, d=None: cfg.get(k, d)),
        rollout_ft_every_n=every_n,
        rollout_ft_grad_steps=1,
        rollout_ft_max_samples=2,
        rollout_ft_geometry=False,
        _rollout_ft_sampler_warned=True,
        val_gen_cfg=None,
        fm=SimpleNamespace(full_simulation_draft=fake_draft),
        predict_for_sampling=lambda batch, mode="full", n_recycle=0: None,
        apply_cyclization_type_conditioning=lambda batch, bs: batch,
    )
    stub.seen = seen
    stub.logged = {}
    stub.log = lambda k, v, **kw: stub.logged.__setitem__(k, float(v))
    return stub


def _bind_rollout(stub, coords, atom_mask, aa):
    stub.autoencoder = SimpleNamespace(
        decode=lambda z_latent, ca_coors_nm, mask: {
            "coors_nm": coords[: z_latent.shape[0]],
            "atom_mask": atom_mask[: z_latent.shape[0]],
            "residue_type": aa[: z_latent.shape[0]],
        }
    )
    stub._rollout_ft_sampler = lambda: stub.rollout_ft_cfg.get("design_sampling")
    return stub


def test_rollout_loss_is_positive_for_an_open_sampled_ring():
    from proteinfoundation.proteina import Proteina

    batch, coords, atom_mask, aa = _disulfide_batch()
    stub = _bind_rollout(_rollout_stub(), coords, atom_mask, aa)
    loss = Proteina.compute_rollout_finetune_loss(stub, batch, log_prefix="train", bs=1)
    assert float(loss) > 0.0
    assert stub.logged["train/loss_rollout_finetune"] > 0.0
    # Measured on GENERATED structures, so this key is the honest one.
    assert stub.logged["train/rollout_window_success"] == 0.0
    assert stub.logged["train/rollout_n_valid"] == 1.0


def test_rollout_loss_never_hands_the_sampler_the_training_interpolant():
    """x_1/x_0/x_t/t/x_sc are ground-truth-derived; leaking them would be teacher forcing."""
    from proteinfoundation.proteina import Proteina

    batch, coords, atom_mask, aa = _disulfide_batch()
    stub = _bind_rollout(_rollout_stub(), coords, atom_mask, aa)
    Proteina.compute_rollout_finetune_loss(stub, batch, log_prefix="train", bs=1)
    leaked = {"x_1", "x_0", "x_t", "t", "x_sc", "x_recycle"} & stub.seen["batch_keys"]
    assert not leaked, leaked
    # Conditioning the sampler IS allowed to see must survive.
    assert {"cyclization_i", "cyclization_j", "cyclization_type", "mask"} <= stub.seen["batch_keys"]


def test_rollout_loss_respects_cadence_and_is_a_differentiable_zero_when_skipped():
    from proteinfoundation.proteina import Proteina

    batch, coords, atom_mask, aa = _disulfide_batch()
    stub = _bind_rollout(_rollout_stub(global_step=3, every_n=16), coords, atom_mask, aa)
    loss = Proteina.compute_rollout_finetune_loss(stub, batch, log_prefix="train", bs=1)
    assert float(loss) == 0.0
    assert "nsteps" not in stub.seen, "sampler must not run on an off-cadence step"
    assert stub.logged == {}


def test_rollout_loss_uses_its_own_nsteps_and_never_autoguides():
    from proteinfoundation.proteina import Proteina

    batch, coords, atom_mask, aa = _disulfide_batch()
    stub = _bind_rollout(_rollout_stub(), coords, atom_mask, aa)
    Proteina.compute_rollout_finetune_loss(stub, batch, log_prefix="train", bs=1)
    assert stub.seen["nsteps"] == 12, "rollout_finetune.nsteps must win over the design sampler's"
    assert stub.seen["grad_steps"] == 1
    assert stub.seen["ag_ratio"] == 0.0, "autoguidance needs a second ckpt training cannot load"


def test_rollout_loss_caps_the_rollout_batch():
    from proteinfoundation.proteina import Proteina

    n = 4
    batch, coords, atom_mask, aa = _disulfide_batch(n=n)
    bs = 5
    batch = {
        k: (torch.cat([v] * bs, dim=0) if torch.is_tensor(v) and v.shape[0] == 1 else v) for k, v in batch.items()
    }
    coords, atom_mask, aa = (torch.cat([x] * bs, dim=0) for x in (coords, atom_mask, aa))
    stub = _bind_rollout(_rollout_stub(n=n), coords, atom_mask, aa)
    Proteina.compute_rollout_finetune_loss(stub, batch, log_prefix="train", bs=bs)
    assert stub.seen["nsamples"] == 2, "max_samples must cap the rollout, not the flow loss"
    assert stub.logged["train/rollout_n_valid"] == 2.0


def test_rollout_loss_is_a_noop_without_cyclization_labels():
    from proteinfoundation.proteina import Proteina

    batch, coords, atom_mask, aa = _disulfide_batch()
    for k in ("cyclization_i", "cyclization_j", "cyclization_type"):
        batch.pop(k)
    stub = _bind_rollout(_rollout_stub(), coords, atom_mask, aa)
    loss = Proteina.compute_rollout_finetune_loss(stub, batch, log_prefix="train", bs=1)
    assert float(loss) == 0.0
    assert "nsteps" not in stub.seen


def test_rollout_sampler_resolution_refuses_to_guess():
    """No `design_sampling` anywhere must raise, not silently fall back to the monomer sampler."""
    from proteinfoundation.proteina import Proteina

    stub = _rollout_stub(with_design=False)
    stub._rollout_ft_sampler_warned = False
    try:
        Proteina._rollout_ft_sampler(stub)
    except ValueError as e:
        assert "design_sampling" in str(e)
    else:
        raise AssertionError("expected a ValueError rather than a guessed sampler")


def test_rollout_sampler_falls_back_to_val_generation():
    from proteinfoundation.proteina import Proteina

    stub = _rollout_stub(with_design=False)
    stub._rollout_ft_sampler_warned = False
    sentinel = object()
    stub.val_gen_cfg = SimpleNamespace(get=lambda k, d=None: sentinel if k == "design_sampling" else d)
    assert Proteina._rollout_ft_sampler(stub) is sentinel


# --------------------------------------------------------------------------------------
# _subset_batch (the unroll's memory knob)
# --------------------------------------------------------------------------------------


def test_subset_batch_slices_only_batch_leading_tensors():
    from proteinfoundation.proteina import _subset_batch

    bs, n = 8, 5
    batch = {
        "mask": torch.ones(bs, n, dtype=torch.bool),
        "x_t": {"bb_ca": torch.randn(bs, n, 3), "local_latents": torch.randn(bs, n, 4)},
        "t": {"bb_ca": torch.rand(bs)},
        "cyclization_i": torch.arange(bs),
        # Must pass through untouched: leading dim is not the batch size.
        "window_lo": torch.zeros(3),
        "some_flag": True,
        "name": "not-a-tensor",
    }
    out = _subset_batch(batch, n_keep=3, bs=bs)
    assert out["mask"].shape == (3, n)
    assert out["x_t"]["bb_ca"].shape == (3, n, 3)
    assert out["x_t"]["local_latents"].shape == (3, n, 4)
    assert out["t"]["bb_ca"].shape == (3,)
    assert torch.equal(out["cyclization_i"], torch.arange(3))
    assert out["window_lo"].shape == (3,) and torch.equal(out["window_lo"], batch["window_lo"])
    assert out["some_flag"] is True and out["name"] == "not-a-tensor"
    # Truncation, not sampling: reproducible across ranks and reruns.
    assert torch.equal(out["x_t"]["bb_ca"], batch["x_t"]["bb_ca"][:3])
    # Original untouched.
    assert batch["mask"].shape == (bs, n)


def test_subset_batch_is_identity_when_keeping_everything():
    from proteinfoundation.proteina import _subset_batch

    bs = 4
    batch = {"mask": torch.ones(bs, 3, dtype=torch.bool), "t": {"bb_ca": torch.rand(bs)}}
    out = _subset_batch(batch, n_keep=bs, bs=bs)
    assert torch.equal(out["mask"], batch["mask"])
    assert torch.equal(out["t"]["bb_ca"], batch["t"]["bb_ca"])


def test_unroll_on_a_subset_matches_unrolling_the_subset_alone():
    """Capping `max_samples` must change only WHICH examples are supervised, not how."""
    from proteinfoundation.proteina import _subset_batch

    fm, batch = make_fm(), make_batch()
    n_keep = 2
    sub = _subset_batch(batch, n_keep=n_keep, bs=B)

    net_a = ToyNet()
    from_subset = unrolled_clean_prediction(fm, call_nn_of(net_a), sub, k_steps=3, grad_steps=1)
    net_b = ToyNet()
    from_full = unrolled_clean_prediction(fm, call_nn_of(net_b), batch, k_steps=3, grad_steps=1)

    for m in MODES:
        assert from_subset[m].shape[0] == n_keep
        assert torch.allclose(from_subset[m], from_full[m][:n_keep], atol=1e-6), m


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _main()
