"""Unit tests for the FlowEdit sampler. No GPU, no checkpoint, no network.

These run against the REAL `ProductSpaceFlowMatcher` -- so the real `interpolate`,
`sample_noise` and `simulation_step`, which is where a sign error or a masking mistake
would actually hide. Only the network boundary is stubbed, which is the one thing a unit
test should stub.

The load-bearing test is `test_identical_conditioning_is_a_no_op`: if the source and target
conditionings agree, the velocity difference is identically zero and FlowEdit must return
the input UNCHANGED, to the last bit. That property is what distinguishes FlowEdit from
SDEdit, and if it ever breaks, every retention number this pipeline reports is inflated by
an amount nothing downstream can detect.

Run:  .venv/bin/python script_utils/test_flowedit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.editing.flowedit import (  # noqa: E402
    flowedit_simulate,
    ode_step_params,
)
from proteinfoundation.flow_matching.product_space_flow_matcher import (  # noqa: E402
    ProductSpaceFlowMatcher,
)

BS, N, D_LAT = 2, 6, 8
MODES = ("bb_ca", "local_latents")

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(name)


def build_fm() -> ProductSpaceFlowMatcher:
    """A real product-space flow matcher over the two tracks this model uses.

    `zero_com_noise=False` on both, matching `training_cpsea_lpmix_from_v4cfg.yaml`.
    """
    cfg = OmegaConf.create(
        {
            "product_flowmatcher": {
                "bb_ca": {"dim": 3, "guidance_enabled": True, "zero_com_noise": False},
                "local_latents": {"dim": D_LAT, "guidance_enabled": True, "zero_com_noise": False},
            }
        }
    )
    return ProductSpaceFlowMatcher(cfg)


def make_mask() -> torch.Tensor:
    """A mask with real padding, so a leak into padded residues shows up."""
    mask = torch.ones(BS, N, dtype=torch.bool)
    mask[1, -2:] = False
    return mask


def make_x(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "bb_ca": torch.randn(BS, N, 3) * mask[..., None],
        "local_latents": torch.randn(BS, N, D_LAT) * mask[..., None],
    }


def ascending_ts(nsteps: int, t_start: dict[str, float]) -> dict[str, torch.Tensor]:
    """Uniform grids from each track's start up to 1 -- the shape `build_edit_schedule` makes."""
    return {
        dm: torch.linspace(float(t_start[dm]), 1.0, nsteps + 1) for dm in MODES
    }


def step_params() -> dict:
    base = {
        dm: {
            "simulation_step_params": {
                "sampling_mode": "sc",  # deliberately NOT vf, to prove the forcing works
                "sc_scale_noise": 0.35,
                "sc_scale_score": 1.0,
                "t_lim_ode": 0.98,
                "t_lim_ode_below": 0.02,
                "center_every_step": True,  # also deliberately on
            }
        }
        for dm in MODES
    }
    return ode_step_params(MODES, base)


def stub_network(fm, velocity_fn):
    """Install a fake velocity field on the flow matcher.

    `velocity_fn(batch, x_t, t) -> dict[mode, tensor]`. We patch the two methods
    `_velocity` calls, so everything downstream of the network -- `simulation_step`,
    masking, the `dt` bookkeeping -- stays the real implementation.
    """

    def get_clean_pred_n_guided_vector(batch, predict_for_sampling, guidance_w, ag_ratio):
        v = velocity_fn(batch, batch["x_t"], batch["t"])
        return {dm: {"v": v[dm], "v_guided": v[dm]} for dm in fm.data_modes}

    def nn_out_to_clean_sample_prediction(batch, nn_out):
        return {dm: nn_out[dm]["v"] for dm in fm.data_modes}

    fm.get_clean_pred_n_guided_vector = get_clean_pred_n_guided_vector
    fm.nn_out_to_clean_sample_prediction = nn_out_to_clean_sample_prediction


def run(fm, x_src, mask, ts, nsteps, velocity_fn, **kw):
    stub_network(fm, velocity_fn)
    return flowedit_simulate(
        fm,
        batch_src={"cyclization_type_cond": torch.full((BS,), 4)},
        batch_tar={"cyclization_type_cond": torch.full((BS,), kw.pop("tar_type", 0))},
        x_src=x_src,
        mask=mask,
        ts=ts,
        step_params=step_params(),
        predict_for_sampling=lambda *a, **k: None,
        nsteps=nsteps,
        **kw,
    )


# --------------------------------------------------------------------------- tests


def test_identical_conditioning_is_a_no_op():
    """V^delta == 0 => the output IS the input. The property FlowEdit exists for."""
    fm, mask = build_fm(), make_mask()
    x_src = make_x(mask)
    ts = ascending_ts(20, {"bb_ca": 0.2, "local_latents": 0.4})

    # A velocity that depends on the state and the time but NOT on the conditioning. The
    # source and target branches then produce the same field at their respective points,
    # and only the coupled-noise construction decides whether the difference cancels.
    def v_fn(batch, x_t, t):
        return {dm: 3.0 * x_t[dm] + 0.5 for dm in MODES}

    out = run(fm, x_src, mask, ts, 20, v_fn, n_avg=2, tar_type=4)
    for dm in MODES:
        delta = (out["x"][dm] - x_src[dm]).abs().max().item()
        check(f"no-op edit leaves {dm} untouched", delta < 1e-5, f"max|dx| = {delta:.2e}")


def test_constant_delta_integrates_analytically():
    """A constant velocity difference c must move the state by exactly c * (1 - t_start).

    This is the sign-and-convention test. Getting `dt` backwards, or integrating downward
    in `t`, changes this result's sign; getting the interval wrong changes its magnitude.
    """
    fm, mask = build_fm(), make_mask()
    x_src = make_x(mask)
    t_start = {"bb_ca": 0.25, "local_latents": 0.6}
    nsteps = 40
    ts = ascending_ts(nsteps, t_start)

    # v depends ONLY on the conditioning, so the difference is a known constant.
    def v_fn(batch, x_t, t):
        is_tar = int(batch["cyclization_type_cond"][0]) == 0
        val = 2.0 if is_tar else 0.5  # difference = 1.5
        return {dm: torch.full_like(x_t[dm], val) for dm in MODES}

    out = run(fm, x_src, mask, ts, nsteps, v_fn, n_avg=1, tar_type=0)
    for dm in MODES:
        expected = 1.5 * (1.0 - t_start[dm])
        got = (out["x"][dm] - x_src[dm])[mask].mean().item()
        check(f"constant delta integrates exactly on {dm}",
              abs(got - expected) < 1e-4, f"got {got:.6f}, expected {expected:.6f}")


def test_frozen_track_never_moves():
    """t_start = 1 gives dt == 0 for that track: the frozen-sequence corner."""
    fm, mask = build_fm(), make_mask()
    x_src = make_x(mask)
    nsteps = 10
    ts = ascending_ts(nsteps, {"bb_ca": 0.3, "local_latents": 1.0})

    def v_fn(batch, x_t, t):
        is_tar = int(batch["cyclization_type_cond"][0]) == 0
        return {dm: torch.full_like(x_t[dm], 5.0 if is_tar else 0.0) for dm in MODES}

    out = run(fm, x_src, mask, ts, nsteps, v_fn, n_avg=1, tar_type=0)
    frozen = (out["x"]["local_latents"] - x_src["local_latents"]).abs().max().item()
    moved = (out["x"]["bb_ca"] - x_src["bb_ca"]).abs().max().item()
    check("frozen latent track is bit-identical", frozen == 0.0, f"max|dx| = {frozen:.2e}")
    check("the unfrozen track did move", moved > 1e-3, f"max|dx| = {moved:.3f}")


def test_coupled_noise_displacement_is_shared():
    """x_tar_t - Z^FE must equal x_src_t - X^src at every step, for every draw.

    That identity IS the coupling. If the two branches were noised independently, the
    difference of velocities would compare two unrelated perturbations and the edit would
    be a re-sampling with extra steps.
    """
    fm, mask = build_fm(), make_mask()
    x_src = make_x(mask)
    nsteps = 8
    ts = ascending_ts(nsteps, {"bb_ca": 0.2, "local_latents": 0.2})

    seen: list[dict[str, torch.Tensor]] = []

    def v_fn(batch, x_t, t):
        seen.append({dm: x_t[dm].clone() for dm in MODES})
        is_tar = int(batch["cyclization_type_cond"][0]) == 0
        return {dm: torch.full_like(x_t[dm], 1.0 if is_tar else 0.0) for dm in MODES}

    run(fm, x_src, mask, ts, nsteps, v_fn, n_avg=1, tar_type=0)
    # calls alternate source, target, source, target, ...
    ok, worst = True, 0.0
    for k in range(0, len(seen) - 1, 2):
        src_state, tar_state = seen[k], seen[k + 1]
        for dm in MODES:
            src_disp = src_state[dm] - x_src[dm]
            # Z^FE at that step is unknown here, but the DIFFERENCE of the two
            # displacements must be constant across residues within a sample, because both
            # got the same additive noise displacement.
            resid = (tar_state[dm] - src_disp) - x_src[dm]
            spread = (resid[mask] - resid[mask].mean()).abs().max().item()
            worst = max(worst, spread)
    # A tighter, exact statement for the first step, where Z^FE == X^src:
    for dm in MODES:
        d = (seen[1][dm] - seen[0][dm]).abs().max().item()
        ok = ok and d < 1e-6
    check("first step: target state equals source state when Z==X_src", ok)
    # `resid` above reduces to Z^FE - X^src. Under the constant velocity difference used
    # here that displacement is uniform across residues, so its spread must be ~0. The
    # bound is tight on purpose: a loose one would pass no matter what the coupling did.
    check("coupled displacement is uniform under a constant delta", worst < 1e-4,
          f"spread {worst:.2e}")


def test_edit_end_step_switches_to_plain_target():
    """edit_end_step=0 must reduce to ordinary sampling of the target field from the input."""
    fm, mask = build_fm(), make_mask()
    x_src = make_x(mask)
    nsteps = 12
    ts = ascending_ts(nsteps, {"bb_ca": 0.5, "local_latents": 0.5})
    calls = {"n": 0}

    def v_fn(batch, x_t, t):
        calls["n"] += 1
        return {dm: torch.full_like(x_t[dm], 1.0) for dm in MODES}

    out = run(fm, x_src, mask, ts, nsteps, v_fn, n_avg=3, edit_end_step=0, tar_type=0)
    # One forward per step, not 2 * n_avg: the delta branch never ran.
    check("edit_end_step=0 costs one forward per step", calls["n"] == nsteps,
          f"{calls['n']} calls for {nsteps} steps")
    expected = 1.0 * (1.0 - 0.5)
    got = (out["x"]["bb_ca"] - x_src["bb_ca"])[mask].mean().item()
    check("the tail integrates the plain target field", abs(got - expected) < 1e-4,
          f"got {got:.6f}, expected {expected:.6f}")


def test_forced_ode_params():
    """The score modes and centering must be off, whatever the config said."""
    p = step_params()
    for dm in MODES:
        check(f"{dm}: sampling_mode forced to vf", p[dm]["sampling_mode"] == "vf",
              p[dm]["sampling_mode"])
        check(f"{dm}: center_every_step forced off", p[dm]["center_every_step"] is False)


def test_padding_is_not_written():
    """Masked residues must stay exactly zero -- `simulation_step` applies the mask."""
    fm, mask = build_fm(), make_mask()
    x_src = make_x(mask)
    ts = ascending_ts(10, {"bb_ca": 0.2, "local_latents": 0.2})

    def v_fn(batch, x_t, t):
        is_tar = int(batch["cyclization_type_cond"][0]) == 0
        return {dm: torch.full_like(x_t[dm], 7.0 if is_tar else 0.0) for dm in MODES}

    out = run(fm, x_src, mask, ts, 10, v_fn, n_avg=1, tar_type=0)
    pad = ~mask
    worst = max(out["x"][dm][pad].abs().max().item() for dm in MODES)
    check("padded residues remain zero", worst == 0.0, f"max|x_pad| = {worst:.2e}")


def test_reproducible_under_a_generator():
    """Same generator seed => same edit, which resumable search depends on."""
    fm, mask = build_fm(), make_mask()
    x_src = make_x(mask)
    ts = ascending_ts(10, {"bb_ca": 0.3, "local_latents": 0.3})

    def v_fn(batch, x_t, t):
        is_tar = int(batch["cyclization_type_cond"][0]) == 0
        # state-dependent, so the noise draw actually influences the result
        return {dm: (2.0 if is_tar else 1.0) * x_t[dm] for dm in MODES}

    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(1234)
        outs.append(run(fm, x_src, mask, ts, 10, v_fn, n_avg=2, generator=g, tar_type=0))
    worst = max((outs[0]["x"][dm] - outs[1]["x"][dm]).abs().max().item() for dm in MODES)
    check("same seed reproduces the edit", worst == 0.0, f"max|dx| = {worst:.2e}")

    g2 = torch.Generator().manual_seed(999)
    other = run(fm, x_src, mask, ts, 10, v_fn, n_avg=2, generator=g2, tar_type=0)
    diff = max((outs[0]["x"][dm] - other["x"][dm]).abs().max().item() for dm in MODES)
    check("a different seed gives a different edit", diff > 1e-6, f"max|dx| = {diff:.2e}")


def main() -> int:
    tests = [
        ("identical conditioning is a no-op", test_identical_conditioning_is_a_no_op),
        ("constant delta integrates analytically", test_constant_delta_integrates_analytically),
        ("frozen track never moves", test_frozen_track_never_moves),
        ("coupled noise displacement", test_coupled_noise_displacement_is_shared),
        ("edit_end_step switches to plain target", test_edit_end_step_switches_to_plain_target),
        ("forced ODE params", test_forced_ode_params),
        ("padding is not written", test_padding_is_not_written),
        ("reproducible under a generator", test_reproducible_under_a_generator),
    ]
    for name, fn in tests:
        print(f"\n{name}")
        fn()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print(f"all {sum(1 for _ in tests)} test groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
