#!/usr/bin/env python3
"""Unit tests for `proteinfoundation.training.flow_loss.flow_loss_from_clean_target`.

Pure CPU tests: builds a real (but tiny) `ProductSpaceFlowMatcher` + a toy
linear "nn" instead of the full `Proteina` model/dataset, so the equivalence
claims are checked against the actual flow-matching primitives
(`sample_t`/`sample_noise`/`interpolate`/`compute_loss`), not a mock of them.

Usage (as a standalone script):
    python script_utils/test_flow_loss_shared_fn.py

Usage (via pytest, if installed):
    pytest script_utils/test_flow_loss_shared_fn.py -v
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from proteinfoundation.training.flow_loss import flow_loss_from_clean_target

BB_CA_DIM = 3
LATENT_DIM = 2


def _make_cfg_exp():
    return OmegaConf.create(
        {
            "product_flowmatcher": {
                "bb_ca": {"zero_com_noise": False, "guidance_enabled": False, "dim": BB_CA_DIM},
                "local_latents": {"zero_com_noise": False, "guidance_enabled": False, "dim": LATENT_DIM},
            },
            "loss": {
                "t_distribution": {
                    "bb_ca": {"name": "uniform", "p2": 1.0, "loss_t_clamp": 1.0},
                    "local_latents": {"name": "uniform", "p2": 1.0, "loss_t_clamp": 1.0},
                    "shared_groups": None,
                },
                "aux_losses": {},
            },
        }
    )


class _TinyNN(torch.nn.Module):
    """Pointwise "flow head": v = MLP(x_t, t) per data mode, per residue.

    `t` is concatenated in explicitly (rather than using a bare linear map of
    `x_t` alone) because the ideal target `v = x1 - x0` is only a linear
    function of `x_t` *given* `t`; a model that cannot see `t` has irreducible
    error even at convergence, which would make the overfit test's loss curve
    meaningless.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        dims = {"bb_ca": BB_CA_DIM, "local_latents": LATENT_DIM}
        self.mlps = torch.nn.ModuleDict(
            {
                dm: torch.nn.Sequential(
                    torch.nn.Linear(d + 1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, d)
                )
                for dm, d in dims.items()
            }
        )

    def forward(self, batch):
        out = {}
        for dm, mlp in self.mlps.items():
            x_t = batch["x_t"][dm]
            t = batch["t"][dm]
            t_b = t[:, None, None].expand(x_t.shape[0], x_t.shape[1], 1)
            out[dm] = {"v": mlp(torch.cat([x_t, t_b], dim=-1))}
        return out


class _FakeProteina:
    """Minimal stand-in for `Proteina`: exposes `.fm` and `.call_nn`, nothing else."""

    def __init__(self, fm, nn_module):
        self.fm = fm
        self.nn = nn_module

    def call_nn(self, batch, n_recycle=0):
        nn_out = self.nn(batch)
        for _ in range(n_recycle):
            nn_out = self.nn(batch)
        return nn_out


def _build(batch_size=2, n=5):
    torch.manual_seed(7)
    fm = ProductSpaceFlowMatcher(_make_cfg_exp())
    nn_module = _TinyNN()
    proteina = _FakeProteina(fm, nn_module)
    x1_ca = torch.randn(batch_size, n, BB_CA_DIM)
    z1 = torch.randn(batch_size, n, LATENT_DIM)
    mask = torch.ones(batch_size, n, dtype=torch.bool)
    return proteina, fm, nn_module, x1_ca, z1, mask


def _manual_reference_loss(fm, nn_module, clean_x1_ca, clean_z1, mask, n_recycle=0):
    """Replicates `corrupt_batch` + `call_nn` + `compute_loss` inline, for direct comparison."""
    x_1 = {"bb_ca": clean_x1_ca, "local_latents": clean_z1}
    x_1 = fm._apply_mask(x=x_1, mask=mask)
    batch_shape = tuple(mask.shape[:-1])
    n = mask.shape[-1]
    device = clean_x1_ca.device

    t = fm.sample_t(shape=batch_shape, device=device)
    x_0 = fm.sample_noise(n=n, shape=batch_shape, mask=mask, device=device)
    x_t = fm.interpolate(x_0=x_0, x_1=x_1, t=t, mask=mask)

    batch = {"x_0": x_0, "x_1": x_1, "x_t": x_t, "t": t, "mask": mask}
    nn_out = nn_module(batch)
    for _ in range(n_recycle):
        nn_out = nn_module(batch)
    return fm.compute_loss(batch=batch, nn_out=nn_out)


# ---------------------------------------------------------------------------
# A. weight=1 exactly matches the ordinary flow loss (same seed => same t/noise).
# ---------------------------------------------------------------------------
def test_weight_one_matches_reference_loss_exactly():
    proteina, fm, nn_module, x1_ca, z1, mask = _build()

    torch.manual_seed(123)
    losses_ref = _manual_reference_loss(fm, nn_module, x1_ca, z1, mask)

    torch.manual_seed(123)
    losses_fn, _ = flow_loss_from_clean_target(
        proteina, x1_ca, z1, condition={}, mask=mask, sample_weight=torch.ones(x1_ca.shape[0])
    )

    assert set(losses_ref.keys()) == set(losses_fn.keys())
    for k in losses_ref:
        assert torch.allclose(losses_ref[k], losses_fn[k], atol=1e-6), k


def test_weight_none_matches_weight_one():
    proteina, fm, nn_module, x1_ca, z1, mask = _build()

    torch.manual_seed(42)
    losses_none, _ = flow_loss_from_clean_target(proteina, x1_ca, z1, condition={}, mask=mask, sample_weight=None)

    torch.manual_seed(42)
    losses_ones, _ = flow_loss_from_clean_target(
        proteina, x1_ca, z1, condition={}, mask=mask, sample_weight=torch.ones(x1_ca.shape[0])
    )

    for k in losses_none:
        assert torch.allclose(losses_none[k], losses_ones[k], atol=1e-6), k


# ---------------------------------------------------------------------------
# B. weight=0 -> exactly zero total loss and exactly zero parameter gradient.
# ---------------------------------------------------------------------------
def test_weight_zero_gives_zero_loss_and_zero_grad():
    proteina, fm, nn_module, x1_ca, z1, mask = _build()
    nn_module.zero_grad()

    torch.manual_seed(0)
    losses, _ = flow_loss_from_clean_target(
        proteina, x1_ca, z1, condition={}, mask=mask, sample_weight=torch.zeros(x1_ca.shape[0])
    )
    total = sum(torch.mean(v) for k, v in losses.items() if "_justlog" not in k)
    assert total.item() == 0.0

    total.backward()
    for p in nn_module.parameters():
        assert p.grad is not None
        assert torch.all(p.grad == 0.0)


# ---------------------------------------------------------------------------
# C. No gradient path back into a mocked "reward"/upstream tensor.
# ---------------------------------------------------------------------------
def test_no_grad_into_detached_upstream_tensor():
    proteina, fm, nn_module, x1_ca, z1, mask = _build()
    nn_module.zero_grad()

    some_reward = torch.randn(x1_ca.shape[0], requires_grad=True)
    x1_ca_from_reward = some_reward.detach()[:, None, None] * 2.0 * torch.ones_like(x1_ca)
    assert not x1_ca_from_reward.requires_grad

    torch.manual_seed(1)
    losses, _ = flow_loss_from_clean_target(
        proteina, x1_ca_from_reward, z1, condition={}, mask=mask, sample_weight=None, from_replay=True
    )
    total = sum(torch.mean(v) for k, v in losses.items() if "_justlog" not in k)
    total.backward()
    assert some_reward.grad is None


def test_from_replay_rejects_tensor_requiring_grad():
    proteina, fm, nn_module, x1_ca, z1, mask = _build()
    x1_ca_live = x1_ca.clone().requires_grad_(True)
    try:
        flow_loss_from_clean_target(
            proteina, x1_ca_live, z1, condition={}, mask=mask, sample_weight=None, from_replay=True
        )
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# D. Variable-length masking is padding-invariant (single example, extra pad).
# ---------------------------------------------------------------------------
def test_padding_invariant_single_example():
    """Same t/noise, only the amount of (masked-out) padding differs.

    Note: this deliberately does NOT rely on reseeding + drawing a larger
    `torch.randn` shape producing a shared prefix with a smaller one -- that
    is not guaranteed by PyTorch's RNG (block/vectorized fill algorithms can
    consume the stream differently depending on total element count, and
    empirically do here). Instead, `sample_t`/`sample_noise` are monkeypatched
    to return one fixed set of values, zero-padded for the "long" call, so the
    only thing that differs between the two calls is `mask`/padding itself --
    which is exactly the invariance this test is about.
    """
    torch.manual_seed(7)
    fm = ProductSpaceFlowMatcher(_make_cfg_exp())
    nn_module = _TinyNN()
    proteina = _FakeProteina(fm, nn_module)

    real_len = 5
    pad = 3
    x1_ca_short = torch.randn(1, real_len, BB_CA_DIM)
    z1_short = torch.randn(1, real_len, LATENT_DIM)
    mask_short = torch.ones(1, real_len, dtype=torch.bool)

    x1_ca_long = torch.nn.functional.pad(x1_ca_short, (0, 0, 0, pad))
    z1_long = torch.nn.functional.pad(z1_short, (0, 0, 0, pad))
    mask_long = torch.nn.functional.pad(mask_short, (0, pad), value=False)

    t_fixed = {"bb_ca": torch.tensor([0.4]), "local_latents": torch.tensor([0.7])}
    noise_short = {
        "bb_ca": torch.randn(1, real_len, BB_CA_DIM),
        "local_latents": torch.randn(1, real_len, LATENT_DIM),
    }
    noise_long = {
        "bb_ca": torch.nn.functional.pad(noise_short["bb_ca"], (0, 0, 0, pad)),
        "local_latents": torch.nn.functional.pad(noise_short["local_latents"], (0, 0, 0, pad)),
    }

    def _make_patches(noise_val):
        def _sample_t(shape, device):
            return {k: v.to(device) for k, v in t_fixed.items()}

        def _sample_noise(n, shape=tuple(), device=None, mask=None):
            return {k: v.to(device) for k, v in noise_val.items()}

        return _sample_t, _sample_noise

    orig_sample_t, orig_sample_noise = fm.sample_t, fm.sample_noise
    try:
        fm.sample_t, fm.sample_noise = _make_patches(noise_short)
        losses_short, _ = flow_loss_from_clean_target(proteina, x1_ca_short, z1_short, {}, mask_short)

        fm.sample_t, fm.sample_noise = _make_patches(noise_long)
        losses_long, _ = flow_loss_from_clean_target(proteina, x1_ca_long, z1_long, {}, mask_long)
    finally:
        fm.sample_t, fm.sample_noise = orig_sample_t, orig_sample_noise

    for k in losses_short:
        assert torch.allclose(losses_short[k], losses_long[k], atol=1e-5), (k, losses_short[k], losses_long[k])


# ---------------------------------------------------------------------------
# E. Tiny overfit test: replay loss on one fixed endpoint decreases with training.
# ---------------------------------------------------------------------------
def test_tiny_overfit_reduces_loss():
    import statistics

    torch.manual_seed(3)
    # A tighter max-t / `loss_t_clamp` bounds the 1/(1-t)^2 reweight's variance
    # (t is resampled fresh every step, per the "fresh noisy states"
    # requirement), so the per-step loss is a less noisy readout of actual
    # training progress -- this is a single fixed endpoint, so even a "tiny
    # overfit" run is really an online regression over many fresh (x_0, t)
    # draws for that one x_1, not literal memorization, hence the modest (not
    # near-zero) but consistent decrease asserted below.
    cfg = _make_cfg_exp()
    cfg.loss.t_distribution.bb_ca.p2 = 0.5
    cfg.loss.t_distribution.bb_ca.loss_t_clamp = 0.5
    cfg.loss.t_distribution.local_latents.p2 = 0.5
    cfg.loss.t_distribution.local_latents.loss_t_clamp = 0.5
    fm = ProductSpaceFlowMatcher(cfg)
    nn_module = _TinyNN()
    proteina = _FakeProteina(fm, nn_module)
    opt = torch.optim.Adam(nn_module.parameters(), lr=0.01)

    x1_ca = torch.randn(1, 6, BB_CA_DIM)
    z1 = torch.randn(1, 6, LATENT_DIM)
    mask = torch.ones(1, 6, dtype=torch.bool)

    losses_over_time = []
    for _ in range(2000):
        opt.zero_grad()
        losses, _ = flow_loss_from_clean_target(proteina, x1_ca, z1, {}, mask)
        total = sum(torch.mean(v) for k, v in losses.items() if "_justlog" not in k)
        total.backward()
        torch.nn.utils.clip_grad_norm_(nn_module.parameters(), 1.0)
        opt.step()
        losses_over_time.append(float(total.item()))

    early = statistics.median(losses_over_time[:200])
    late = statistics.median(losses_over_time[-200:])
    assert late < early * 0.97, f"early={early}, late={late}"


ALL_TESTS = [
    test_weight_one_matches_reference_loss_exactly,
    test_weight_none_matches_weight_one,
    test_weight_zero_gives_zero_loss_and_zero_grad,
    test_no_grad_into_detached_upstream_tensor,
    test_from_replay_rejects_tensor_requiring_grad,
    test_padding_invariant_single_example,
    test_tiny_overfit_reduces_loss,
]


if __name__ == "__main__":
    failures = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  OK {test_fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(test_fn.__name__)
            print(f"  FAIL {test_fn.__name__}: {e}")

    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
