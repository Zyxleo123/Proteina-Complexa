"""FlowEdit for the Proteina-Complexa product-space flow: linear peptide -> cyclic peptide.

Method: Kulikov et al., "FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained
Flow Models" (arXiv:2412.08629). This module is the faithful port, plus the two
adaptations this model forces. Nothing here is a heuristic dressed as the paper; where a
choice is ours rather than theirs it says so.

Why FlowEdit rather than SDEdit or DDIB
---------------------------------------
SDEdit noises the input and re-samples, so the input survives only as whatever the noise
did not destroy: the measured consequence here is that closure only arrives once the
sequence is allowed to move, and at `t_lat = 1` (frozen sequence) closure is 1/59. DDIB
needs an inversion, and an inversion is only as good as its round trip.

FlowEdit needs neither. It never starts from noise and never inverts. It integrates the
DIFFERENCE between two conditional velocity fields, so the trajectory begins at the input
and moves only insofar as the two conditionings disagree. Where the cyclic and linear
fields agree, the state does not move at all -- which is exactly the behaviour a
pose-and-sequence-preserving edit wants.

The time convention, which is flipped relative to the paper
----------------------------------------------------------
The paper runs `t = 1` (noise) down to `t = 0` (clean). This codebase runs the opposite:
`interpolate` is `x_t = (1-t) * x_0 + t * x_1` with `x_0` noise and `x_1` data, so `t = 0`
is noise and `t = 1` is clean, and the sampler integrates UPWARD with `x += v * dt`.

Writing the paper's `t` as `1 - s` for our `s`:

    paper   Z_t^src = (1 - t) X^src + t * n        ->   ours  (1 - s) * n + s * X^src
                                                        which is exactly
                                                        interpolate(x_0=n, x_1=X^src, t=s)

    paper   Z^FE <- Z^FE - dt * V^delta,  dt < 0   ->   ours  Z^FE <- Z^FE + ds * V^delta

So the interpolant is form-identical (we call the model's own `interpolate`, so masking and
stochastic centering are the model's, not a re-derivation), and the update is the ordinary
forward Euler step with the velocity replaced by the velocity DIFFERENCE. We integrate `s`
upward from `t_start` to 1, the same direction as ordinary sampling.

The two tracks, which the paper does not have
---------------------------------------------
The flow is a product space over `bb_ca` (CA coordinates) and `local_latents` (the
autoencoder's per-residue sequence+sidechain code). `t` is sampled INDEPENDENTLY per track
during training, so a per-track start time is in-distribution rather than a hack.

That gives the editing interval a per-track form, and it is the axis that matters here:
`t_start[local_latents] = 1` freezes the sequence (its `dt` is identically 0, so the track
never leaves the clean source), while `t_start[bb_ca]` controls how much the backbone is
allowed to travel. This is the direct analogue of the paper's `n_max` edit-strength knob,
one per track. `edit_end_step` is the analogue of their `n_min`: after it, the delta is
dropped and the ordinary TARGET field finishes the trajectory.

Why the integrator is forced to pure `vf`
-----------------------------------------
`simulation_step`'s score-based modes (`sc`, `vf_ss`, `vf_tsr`, ...) convert the velocity to
a score with `vf_to_score(x_t, v, t)`, an identity that holds only when `v` is the true
velocity field AT `x_t`. A velocity DIFFERENCE is not a velocity field at `Z^FE`, so that
conversion has no meaning and the injected noise would not be the paper's algorithm. We
therefore force `sampling_mode="vf"` and `gt=0` on the delta branch. `ddib_invert.py` forces
the same thing for the same class of reason.

`center_every_step` is forced off: the peptide is scored in the RECEPTOR frame, and
re-centering the binder each step would move it out of that frame -- the same reason
`ddib_invert.py` disables it.
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import Tensor


def ode_step_params(data_modes, sampling_model_args: dict, *, center: bool = False) -> dict:
    """The config's step params with the integrator forced to the deterministic ODE.

    Deep-copied because the config node is shared with whatever else holds this config.
    See the module docstring for why a velocity difference cannot go through the
    score-based branches.
    """
    params = copy.deepcopy(
        {dm: dict(sampling_model_args[dm]["simulation_step_params"]) for dm in data_modes}
    )
    for dm in params:
        params[dm]["sampling_mode"] = "vf"
        params[dm]["center_every_step"] = bool(center)
    return params


def _synth_nn_out(v: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """Wrap a velocity dict so `fm.simulation_step` consumes it.

    `simulation_step` reads `nn_out[dm]["v_guided"]` when the track has
    `guidance_enabled`, else `nn_out[dm]["v"]`. Both are set to the same tensor: the
    guidance combination has already happened (or been declined) by the time a velocity
    reaches here, so there is no second place for a weight to apply.
    """
    return {dm: {"v": val, "v_guided": val} for dm, val in v.items()}


def _velocity(
    fm,
    batch: dict,
    x_t: dict[str, Tensor],
    t: dict[str, Tensor],
    mask: Tensor,
    predict_for_sampling: Callable,
    guidance_w: float,
    x_sc: dict[str, Tensor] | None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """One conditional velocity evaluation. Returns (velocity, clean-sample prediction).

    `batch` carries the conditioning that distinguishes source from target -- in this
    application a single key, `cyclization_type_cond`. It is mutated in place for the
    forward and the caller owns it, matching `_simulation_loop`'s own style.
    """
    batch["x_t"] = x_t
    batch["t"] = t
    batch["mask"] = mask
    if x_sc is not None:
        # Detached, matching training's `handle_self_cond` and `_simulation_loop`.
        batch["x_sc"] = {dm: val.detach() for dm, val in x_sc.items()}

    nn_out = fm.get_clean_pred_n_guided_vector(
        batch=batch,
        predict_for_sampling=predict_for_sampling,
        guidance_w=guidance_w,
        ag_ratio=0.0,
    )
    x_1_pred = fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
    # Mirror `simulation_step`'s own choice of tensor, per track.
    v = {}
    for dm in fm.data_modes:
        fm_dm = fm.base_flow_matchers[dm]
        key = "v_guided" if getattr(fm_dm, "guidance_enabled", False) else "v"
        if key not in nn_out[dm]:
            raise KeyError(
                f"track {dm!r}: expected {key!r} in nn_out, got {sorted(nn_out[dm])}. "
                "FlowEdit needs a velocity parameterization; a score-only track is not "
                "supported."
            )
        v[dm] = nn_out[dm][key]
    return v, x_1_pred


def flowedit_simulate(
    fm,
    *,
    batch_src: dict,
    batch_tar: dict,
    x_src: dict[str, Tensor],
    mask: Tensor,
    ts: dict[str, Tensor],
    step_params: dict[str, dict],
    predict_for_sampling: Callable,
    nsteps: int,
    edit_end_step: int | None = None,
    n_avg: int = 1,
    guidance_w: float = 1.0,
    self_cond: bool = False,
    generator: torch.Generator | None = None,
    record_every: int = 0,
) -> dict:
    """Integrate the FlowEdit ODE from the clean source to an edited sample.

    Args:
        fm: the `ProductSpaceFlowMatcher`.
        batch_src: conditioning for the SOURCE field (here: `cyclization_type_cond=LINEAR`).
        batch_tar: conditioning for the TARGET field (here: the requested chemistry).
            Must be identical to `batch_src` in every other respect -- same receptor, same
            hotspots, same mask -- or the difference stops being a difference of
            conditionings and becomes a difference of inputs.
        x_src: the clean source sample in product space (`x_1` from `add_clean_samples`).
        mask: [b, n] residue mask.
        ts: per-track time grids, length `nsteps + 1`, ascending toward 1. A track whose
            grid is constant is frozen: its `dt` is 0 at every step.
        step_params: per-track `simulation_step_params`, already forced to `vf` by
            `ode_step_params`.
        predict_for_sampling: the network, as `partial(model.predict_for_sampling, ...)`.
        nsteps: number of Euler steps.
        edit_end_step: steps at or after this index drop the delta and integrate the plain
            TARGET field (the paper's `n_min`). `None` means edit the whole interval.
        n_avg: noise draws averaged into each delta (the paper's `n_avg`). Each draw costs
            two network forwards.
        guidance_w: classifier-free guidance weight applied within each branch. 1.0 is
            faithful FlowEdit: the paper's velocities are the plain conditional ones.
        self_cond: feed each branch its own previous clean-sample prediction. Off is the
            paper's setting; on matches how this model was trained. Under `n_avg > 1` the
            source branch's `x_sc` is the average over that step's draws, since there is no
            single source state.
        generator: RNG for the coupled noise draws, for reproducibility across resumes.
        record_every: if > 0, also return the trajectory every N steps (diagnostics only).

    Returns:
        dict with `x` (the edited product-space sample), `x_1_pred` (the target branch's
        last clean-sample prediction) and `trajectory` (possibly empty).
    """
    if nsteps <= 0:
        raise ValueError(f"nsteps must be positive, got {nsteps}")
    if n_avg <= 0:
        raise ValueError(f"n_avg must be positive, got {n_avg}")
    for dm in fm.data_modes:
        if dm not in ts:
            raise KeyError(f"no time grid for track {dm!r}")
        if len(ts[dm]) < nsteps + 1:
            raise ValueError(
                f"track {dm!r}: time grid has {len(ts[dm])} entries, need nsteps+1 = {nsteps + 1}"
            )
    if edit_end_step is None:
        edit_end_step = nsteps
    edit_end_step = max(0, min(int(edit_end_step), nsteps))

    bs, n = mask.shape
    device = mask.device
    # Z^FE starts AT the clean source -- the defining property of FlowEdit. No noise is
    # added to the state itself, ever; noise enters only inside the delta construction.
    z = {dm: val.clone() for dm, val in x_src.items()}
    x_sc_src: dict[str, Tensor] | None = None
    x_sc_tar: dict[str, Tensor] | None = None
    x_1_pred_tar: dict[str, Tensor] | None = None
    trajectory: list[dict] = []

    zero_gt = {dm: 0.0 for dm in fm.data_modes}

    for step in range(nsteps):
        t = {dm: ts[dm][step] * torch.ones(bs, device=device) for dm in fm.data_modes}
        dt = {dm: ts[dm][step + 1] - ts[dm][step] for dm in fm.data_modes}

        if step < edit_end_step:
            # ---- FlowEdit delta step -------------------------------------------------
            v_delta = {dm: torch.zeros_like(z[dm]) for dm in fm.data_modes}
            acc_src: dict[str, Tensor] = {}
            acc_tar: dict[str, Tensor] = {}
            for _ in range(n_avg):
                # Coupled noise: ONE draw builds the source state, and the SAME
                # displacement is applied to Z^FE to build the target state. That coupling
                # is what makes the difference of velocities a difference along the same
                # perturbation rather than two unrelated samples.
                x_0 = _sample_noise_like(fm, n=n, bs=bs, mask=mask, device=device,
                                         generator=generator)
                x_src_t = fm.interpolate(x_0=x_0, x_1=x_src, t=t, mask=mask)
                x_tar_t = {dm: z[dm] + (x_src_t[dm] - x_src[dm]) for dm in fm.data_modes}

                v_src, p_src = _velocity(fm, batch_src, x_src_t, t, mask,
                                         predict_for_sampling, guidance_w, x_sc_src)
                v_tar, p_tar = _velocity(fm, batch_tar, x_tar_t, t, mask,
                                         predict_for_sampling, guidance_w, x_sc_tar)
                for dm in fm.data_modes:
                    v_delta[dm] = v_delta[dm] + (v_tar[dm] - v_src[dm]) / n_avg
                    if self_cond:
                        acc_src[dm] = acc_src.get(dm, 0) + p_src[dm] / n_avg
                        acc_tar[dm] = acc_tar.get(dm, 0) + p_tar[dm] / n_avg
                x_1_pred_tar = p_tar

            if self_cond:
                x_sc_src, x_sc_tar = dict(acc_src), dict(acc_tar)

            z = fm.simulation_step(
                x_t=z, nn_out=_synth_nn_out(v_delta), t=t, dt=dt, gt=zero_gt,
                mask=mask, simulation_step_params=step_params,
            )
        else:
            # ---- tail: the ordinary TARGET field, no delta ---------------------------
            # The paper's `n_min`: past it the edit is settled and the target model alone
            # finishes denoising. `Z^FE` is a plain state here, so this is a normal step.
            v_tar, x_1_pred_tar = _velocity(fm, batch_tar, z, t, mask,
                                            predict_for_sampling, guidance_w, x_sc_tar)
            if self_cond:
                x_sc_tar = x_1_pred_tar
            z = fm.simulation_step(
                x_t=z, nn_out=_synth_nn_out(v_tar), t=t, dt=dt, gt=zero_gt,
                mask=mask, simulation_step_params=step_params,
            )

        if record_every and (step % record_every == 0 or step == nsteps - 1):
            trajectory.append(
                {"step": step,
                 "t": {dm: float(ts[dm][step]) for dm in fm.data_modes},
                 "x": {dm: val.detach().clone() for dm, val in z.items()}}
            )

    return {"x": z, "x_1_pred": x_1_pred_tar, "trajectory": trajectory}


def _sample_noise_like(fm, *, n: int, bs: int, mask: Tensor, device, generator):
    """A product-space noise draw, honouring each track's own reference distribution.

    `fm.sample_noise` does not take a generator, so when one is supplied we scope it with
    `torch.random.fork_rng` and seed the global RNG from it. That keeps per-track
    behaviour (zero-CoM forcing, stochastic centering) the model's own while still making
    the draw reproducible.
    """
    if generator is None:
        return fm.sample_noise(n=n, shape=(bs,), mask=mask, device=device)
    seed = int(torch.randint(0, 2**62, (1,), generator=generator).item())
    devices = [device] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return fm.sample_noise(n=n, shape=(bs,), mask=mask, device=device)
