"""Differentiable ODE rollout through the CPSea product-space sampler (D-Flow's inner loop).

What this is, and how it differs from `sdedit_guidance.py`
---------------------------------------------------------
`sdedit_guidance.py` is *per-step* guidance: at each Euler step it builds a small graph,
reads dL/dx_t (or dL/dx1_hat), nudges the state, and throws the graph away. Nothing
connects step k to step k+1, so the correction is myopic -- it cannot trade a worse step
now for a better final structure.

This module implements the other thing: ONE graph from the source state all the way to the
final decoded structure, so a loss defined on the *output* can be differentiated back to
the *input noise*. Model weights stay frozen; the free variable is the source state. That
is D-Flow (Ben-Hamu et al., ICML 2024, https://proceedings.mlr.press/v235/ben-hamu24a.html)
applied to this flow.

Three things had to be true for that to work here, and only one of them was already:

1. `partial_simulation` takes `grad_enabled` (added for DRaFT-K), so the loop itself does
   not have to be rewritten to admit a graph. But its own docstring warns to keep the
   window short, because it keeps every step's activations alive. At the production
   nsteps=400 that is ~400 network evaluations of activations at once, which does not fit.
   So this module runs its own loop and wraps each step in `torch.utils.checkpoint`:
   memory becomes one network forward plus nsteps tiny state tensors, at 2x forward cost.

2. The production sampler is an SDE, not an ODE. `configs/pipeline/model_sampling.yaml`
   sets `sampling_mode: sc` for BOTH tracks, whose branch draws a fresh
   `eps = torch.randn(...)` every step. With fresh noise the map source -> sample is not a
   function, and the gradient would be through one arbitrary realization that changes on
   every optimizer step.

   Two deterministic options are exposed, and they are not the same experiment:

       vf    the pure probability-flow ODE, `delta_x = v * dt`. No noise term at all.
             This is what D-Flow assumes and the default here.
       sc    the production SDE with its noise sequence FROZEN. Determinism comes from
             reseeding the global RNG to `noise_seed` at the start of every rollout: the
             `randn` draws depend only on step index and shape, both fixed, so the same
             seed reproduces the same eps sequence and the map is again a function of the
             source state. Gradients are then correct with eps a constant.

   `sc` is the sampler every earlier CPSea/SDEdit number on this repo was produced with,
   so it is the honest comparison arm; `vf` is the one the method is designed for. Which
   one is better here is measured, not assumed -- see `--sampler-mode` in the driver.

   NOTE the schedule is always built from the config's OWN sampling_mode (so
   `get_schedule_tsr_safe` vs `get_schedule` is chosen exactly as production does). Only
   the integrator is swapped. Changing the schedule too would confound "different
   integrator" with "different time grid".

3. The decoded output has to be differentiable in both tracks, and it is:
   `autoencoder.decode` is a plain forward pass returning `seq_logits` (differentiable)
   and `coors_nm` (differentiable in `z_latent` AND `ca_coors_nm`).

The latent de-normalization trap
--------------------------------
`format_sample_local_latents` un-normalizes the latent before decoding
(`z_ae = z_flow * std + mean`) whenever the autoencoder carries `latent_norm_stats`.
Anything that calls `ae.decode` directly on a flow-space latent and skips that step
decodes a wrongly scaled vector. `decode_final` below is the single place this is handled,
and every consumer goes through it. The pinned checkpoint
(`cpsea_bondunroll_pin20260828`) has `latent_normalization: null`, so the two paths agree
*today* -- which is exactly why this needs to be in code rather than in a comment.
"""

from __future__ import annotations

import contextlib
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from proteinfoundation.flow_matching.product_space_flow_matcher import (
    get_gt,
    get_schedule,
    get_schedule_tsr_safe,
)

# `vf_to_score` asserts t < 1 strictly, so a frozen track cannot start at exactly 1.
# Same constant and same reason as scripts/sdedit_cyclize.py.
T_START_MAX = 1.0 - 1e-3

SAMPLER_MODES = ("vf", "sc")


def build_schedule(fm, nsteps: int, sampling_model_args, t_start: dict[str, float]):
    """Per-track (ts, gt) compressed into [t_start[mode], 1].

    Identical construction to `sdedit_cyclize.build_edit_schedule` -- deliberately, so a
    D-Flow rollout at t_start={0,0} walks the same time grid the production sampler does
    and an SDEdit-initialized rollout walks the same grid that script's arms do. Kept here
    rather than imported so this module has no dependency on a sibling script, and checked
    for agreement in `script_utils/test_dflow.py`.
    """
    ts, gt = {}, {}
    for dm in fm.data_modes:
        args_dm = sampling_model_args[dm]
        cfg_mode = args_dm["simulation_step_params"]["sampling_mode"]
        schedule_func = get_schedule_tsr_safe if cfg_mode == "vf_tsr" else get_schedule
        base = schedule_func(
            mode=args_dm["schedule"]["mode"], nsteps=int(nsteps), p1=args_dm["schedule"]["p"]
        )
        t0 = min(float(t_start[dm]), T_START_MAX)
        ts[dm] = t0 + (1.0 - t0) * base
        gt[dm] = get_gt(
            t=ts[dm][:-1],
            mode=args_dm["gt"]["mode"],
            param=args_dm["gt"]["p"],
            clamp_val=args_dm["gt"]["clamp_val"],
        )
    return ts, gt


def resolve_sim_params(fm, sampling_model_args, sampler_mode: str) -> dict:
    """`simulation_step_params` per track, with the integrator swapped to `sampler_mode`.

    Everything else in the block (score/noise scales, t_lim_ode, centering) is carried
    over untouched. In `vf` mode the scale knobs are simply unread by that branch, and
    `gt` never enters -- which is what makes the rollout deterministic with no reseeding.
    """
    if sampler_mode not in SAMPLER_MODES:
        raise ValueError(f"sampler_mode must be one of {SAMPLER_MODES}, got {sampler_mode!r}")
    out = {}
    for dm in fm.data_modes:
        p = dict(sampling_model_args[dm]["simulation_step_params"])
        p["sampling_mode"] = sampler_mode
        out[dm] = p
    return out


def decode_final(autoencoder, x: dict[str, torch.Tensor], mask: torch.Tensor) -> dict:
    """Decode a clean product-space sample to atom37 + sequence logits, differentiably.

    The ONLY decode path in this workflow. Applies the same latent un-normalization
    `format_sample_local_latents` does, so a checkpoint that turns `latent_normalization`
    on does not silently start feeding the decoder a mis-scaled latent here.

    Returns the decoder dict plus `coors_nm_raw`: the decoder's atom37 prediction BEFORE
    the per-atom presence mask is applied. See `dflow_losses.py` for what that is and is
    not good for -- slots for atoms the argmax residue does not have are unsupervised
    during AE training, so `coors_nm_raw` is not a licence to read an arbitrary atom.
    """
    z = x["local_latents"]
    stats = getattr(autoencoder, "latent_norm_stats", None)
    if stats is not None:
        z = z * stats["std"].to(z.device, z.dtype) + stats["mean"].to(z.device, z.dtype)
    # The raw decoder, not the `decode` wrapper: the wrapper multiplies coordinates by the
    # argmax-derived atom mask, and the pre-mask tensor is needed too.
    out = autoencoder.decoder(
        {"z_latent": z, "ca_coors_nm": x["bb_ca"], "residue_mask": mask, "mask": mask}
    )
    rmask = out["residue_mask"]
    atom_mask = out["atom_mask"]
    return {
        "coors_nm": out["coors_nm"] * rmask[..., None, None] * atom_mask[..., None],
        "coors_nm_raw": out["coors_nm"],
        "seq_logits": out["seq_logits"] * rmask[..., None],
        "residue_type": out["aatype_max"] * rmask,
        "residue_mask": rmask,
        "atom_mask": atom_mask,
    }


class DiffRollout:
    """One differentiable source -> final-state map, reusable across optimizer steps.

    Construct once per (example, chemistry, schedule) and call it with a source state.
    Everything expensive that does not depend on the source state -- schedule, masked
    conditioning batch, integrator params -- is resolved here.

    Args:
        fm: the product-space flow matcher (`model.fm`).
        predict_for_sampling: `partial(model.predict_for_sampling, n_recycle=...)`.
        batch: conditioning batch. `x_t`/`t`/`x_sc` are stripped; everything else
            (receptor, masks, `cyclization_type_cond`) is fixed conditioning.
        mask: [1, n] residue mask. B=1 only -- restarts are separate rollouts, not a
            batch dimension, because each carries its own source state and the per-restart
            memory of a retained graph is the binding constraint.
        nsteps: integration steps. Short for optimization, fine for re-evaluation.
        t_start: per-track start time. {0, 0} is full source-noise D-Flow; anything
            higher is the SDEdit-initialized variant, which the driver labels separately.
        sampler_mode: "vf" (probability-flow ODE) or "sc" (production SDE, frozen noise).
        noise_seed: RNG seed reapplied at the start of every rollout. Only read in "sc"
            mode, where it is what makes the map deterministic.
        self_cond: as the production sampler. `x_sc` is detached every step, matching
            both `_simulation_loop` and training, which keeps graph size linear in nsteps
            rather than quadratic.
        grad_checkpoint: recompute each step's activations in the backward pass.
    """

    def __init__(
        self,
        *,
        fm,
        predict_for_sampling,
        batch: dict,
        mask: torch.Tensor,
        nsteps: int,
        t_start: dict[str, float],
        sampling_model_args,
        guidance_w: float = 1.0,
        sampler_mode: str = "vf",
        noise_seed: int = 0,
        self_cond: bool = True,
        grad_checkpoint: bool = True,
    ):
        if mask.shape[0] != 1:
            raise ValueError(f"DiffRollout is B=1 only; got batch size {mask.shape[0]}")
        self.fm = fm
        self.predict = predict_for_sampling
        self.mask = mask
        self.nsteps = int(nsteps)
        self.sampler_mode = sampler_mode
        self.noise_seed = int(noise_seed)
        self.self_cond = bool(self_cond)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.guidance_w = float(guidance_w)
        self.data_modes = list(fm.data_modes)

        missing = set(self.data_modes) - set(t_start)
        if missing:
            raise ValueError(f"no start time supplied for data mode(s) {sorted(missing)}")
        self.t_start = {dm: min(float(t_start[dm]), T_START_MAX) for dm in self.data_modes}

        self.batch = {k: v for k, v in batch.items() if k not in ("x_t", "t", "x_sc")}
        self.batch["mask"] = mask
        ts, gt = build_schedule(fm, self.nsteps, sampling_model_args, self.t_start)
        self.ts = {k: v.to(mask.device) for k, v in ts.items()}
        self.gt = {k: v.to(mask.device) for k, v in gt.items()}
        self.sim_params = resolve_sim_params(fm, sampling_model_args, sampler_mode)

    # ------------------------------------------------------------------ one Euler step
    def _step(self, step: int, flat: tuple[torch.Tensor, ...], use_sc: bool):
        """Network + clean-sample prediction + Euler step, as flat tensors in and out.

        Flat because `torch.utils.checkpoint` passes tensors positionally; the dict
        packing/unpacking lives here so the checkpointed callable is a pure function of
        tensors plus this closure's constants.

        Call order mirrors `_simulation_loop` exactly -- clean-sample prediction BEFORE
        the Euler step, because `nn_out_add_clean_sample_prediction` mutates `nn_out` in
        place and `simulation_step` then reads the same dict.
        """
        n = len(self.data_modes)
        x = dict(zip(self.data_modes, flat[:n]))
        b = dict(self.batch)
        t = {dm: self.ts[dm][step] * torch.ones(1, device=self.mask.device) for dm in self.data_modes}
        dt = {dm: self.ts[dm][step + 1] - self.ts[dm][step] for dm in self.data_modes}
        gt_step = {dm: self.gt[dm][step] for dm in self.data_modes}
        b["x_t"] = x
        b["t"] = t
        b["mask"] = self.mask
        if use_sc:
            b["x_sc"] = dict(zip(self.data_modes, flat[n : 2 * n]))

        nn_out = self.fm.get_clean_pred_n_guided_vector(
            batch=b,
            predict_for_sampling=self.predict,
            guidance_w=self.guidance_w,
            ag_ratio=0.0,
        )
        x1 = self.fm.nn_out_to_clean_sample_prediction(batch=b, nn_out=nn_out)
        x_next = self.fm.simulation_step(
            x_t=x,
            nn_out=nn_out,
            t=t,
            dt=dt,
            gt=gt_step,
            mask=self.mask,
            simulation_step_params=self.sim_params,
        )
        return tuple(x_next[dm] for dm in self.data_modes) + tuple(x1[dm] for dm in self.data_modes)

    # ------------------------------------------------------------------ the rollout
    def __call__(self, x_source: dict[str, torch.Tensor], *, grad: bool = True):
        """Integrate from `x_source` to t=1. Returns (x_final, x1_pred_last).

        `x_source` is the state at `t_start`, i.e. the free variable being optimized. With
        `grad=False` this is an ordinary sampler call and allocates no graph.
        """
        for dm in self.data_modes:
            if dm not in x_source:
                raise ValueError(f"x_source is missing data mode {dm!r}")

        # Frozen noise: reseeding here is what makes "sc" a deterministic function of
        # `x_source`. In "vf" mode nothing draws, so this is a harmless no-op that keeps
        # the two modes on one code path.
        if self.sampler_mode == "sc":
            torch.manual_seed(self.noise_seed)
            if self.mask.device.type == "cuda":
                torch.cuda.manual_seed_all(self.noise_seed)

        n = len(self.data_modes)
        x = tuple(x_source[dm] for dm in self.data_modes)
        x_sc = tuple(torch.zeros_like(v) for v in x)
        x1 = None

        # `grad and torch.is_grad_enabled()`: an enclosing no_grad always wins, the same
        # contract `partial_simulation` uses, so a caller inside `torch.no_grad()` cannot
        # accidentally allocate a 400-step graph.
        ctx = torch.set_grad_enabled(grad and torch.is_grad_enabled())
        with ctx:
            for step in range(self.nsteps):
                use_sc = bool(step > 0 and self.self_cond and x1 is not None)
                flat_in = x + (x_sc if use_sc else tuple())
                fn = partial(self._step, step)
                if self.grad_checkpoint and torch.is_grad_enabled():
                    # use_reentrant=False: required for a callable that closes over
                    # grad-requiring tensors and for outputs that do not all require
                    # grad. preserve_rng_state stays on (the default) so the recomputed
                    # forward in "sc" mode draws the same eps the first pass did --
                    # without it the backward would differentiate a different SDE.
                    out = checkpoint(fn, flat_in, use_sc, use_reentrant=False)
                else:
                    out = fn(flat_in, use_sc)
                x, x1 = out[:n], out[n:]
                # Detached, matching `_simulation_loop` and training's `handle_self_cond`.
                # Without it the graph also chains through x_sc and memory grows with the
                # square of the window instead of linearly.
                x_sc = tuple(v.detach() for v in x1)

        return dict(zip(self.data_modes, x)), dict(zip(self.data_modes, x1))

    # ------------------------------------------------------------------ source states
    def init_source(self, *, x_1: dict[str, torch.Tensor] | None = None, seed: int | None = None):
        """A fresh source state at `t_start`, built the way the existing scripts build it.

        At t_start = 0 this is `fm.sample_noise` (pure source noise: full D-Flow). Above
        it, it is `fm.interpolate(x_0, x_1, t_start)` -- the SDEdit construction, which
        needs the encoded input `x_1`. Returned detached, ready to `requires_grad_(True)`.

        No `training=` kwarg on `sample_noise`: it reads `self.training`, which `model.eval()`
        has already set False, and that flag gates the stochastic-centering branch. Passing
        it explicitly would diverge from what the production sampler does.
        """
        if seed is not None:
            torch.manual_seed(int(seed))
            if self.mask.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            x_0 = self.fm.sample_noise(
                n=self.mask.shape[1], shape=(1,), mask=self.mask, device=self.mask.device
            )
            if all(v == 0.0 for v in self.t_start.values()):
                src = x_0
            else:
                if x_1 is None:
                    raise ValueError(
                        "an SDEdit-initialized rollout (t_start > 0) needs the encoded input "
                        "`x_1`; only the t_start=0 corner can be built from noise alone."
                    )
                t_vec = {
                    dm: torch.full((1,), self.t_start[dm], device=self.mask.device)
                    for dm in self.data_modes
                }
                src = self.fm.interpolate(x_0=x_0, x_1=x_1, t=t_vec, mask=self.mask)
        return {dm: src[dm].detach().clone() for dm in self.data_modes}


@contextlib.contextmanager
def frozen_model(model):
    """Eval mode, no parameter gradients. D-Flow optimizes the input, never the weights."""
    was_training = model.training
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.eval()
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield model
    finally:
        for p, r in flags:
            p.requires_grad_(r)
        model.train(was_training)
