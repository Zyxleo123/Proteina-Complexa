"""Multi-step (unrolled) clean-sample prediction for structure-space auxiliary losses.

Why this exists
---------------
Every auxiliary loss that reads a *decoded structure* during training -- the
cyclization closing-bond loss, the linkage-geometry loss, the chirality loss --
is applied to the network's **one-step** clean prediction
``x_hat_1(x_t, t) = E[x_1 | x_t]``. That is the conditional *mean*, and the
quantities those losses measure (a bond distance, a bond angle, a chirality
sign) are nonlinear functionals of coordinates, so

    d( E[x_1 | x_t] )  !=  E[ d(x_1) | x_t ].

The gap is not a technicality for a *closure* constraint. Averaging over the
posterior of ring conformations pulls the two anchor atoms toward each other, so
the mean structure can sit comfortably inside the acceptance window while the
samples the ODE actually emits do not. Training loss near zero, sampled closure
near half, and no contradiction between them.

This module replaces that one-step mean with the endpoint of a short ODE unroll:
K Euler steps from ``(x_t, t)`` to ``t = 1``. The object being penalised then
moves from "the posterior mean" toward "a sample of the kind the sampler
produces", which is what the loss was always meant to constrain. K=1 reproduces
the one-step behaviour exactly (see ``unrolled_clean_prediction``), so this is a
strict generalisation of what the callers did before.

What this is NOT
----------------
This is a plain ODE (``sampling_mode: vf``) Euler unroll: no score scaling, no
noise injection, no per-step centering. The deployed design sampler uses
``sampling_mode: sc`` (see ``configs/pipeline/model_sampling.yaml``), which this
deliberately does not reproduce -- ``simulation_step`` asserts a single shared
``t`` across the batch, and training samples a *different* ``t`` per example.
Reproducing the exact sampler needs a rollout from ``t=0`` on a shared schedule,
which is what the ``rollout_finetune`` path does instead (at much higher cost).
Read this module as the cheap variance-aware correction and that one as the
faithful one.

Cost
----
K steps means K network evaluations. Step 0 is free when the caller passes the
``nn_out`` it already computed for the flow loss (``nn_out_0``), so the marginal
cost is K-1 forwards, of which only the last ``grad_steps`` retain activations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def unroll_time(t: Tensor, step: int, k_steps: int) -> Tensor:
    """Flow time at unroll `step`, spacing the REMAINING interval `[t, 1]` uniformly.

    ``t`` is per-sample (training draws a different flow time for every example,
    and a different one per data mode), so the grid is per-sample too -- a sample
    starting at t=0.9 takes small steps and one starting at t=0.2 takes large
    ones, both landing on t=1 after `k_steps`.

    Args:
        t: [B] starting flow time.
        step: 0-based step index, in `[0, k_steps)`.
        k_steps: total number of Euler steps.

    Returns:
        [B] flow time at which the network is evaluated for this step. Never
        reaches 1.0 (the last evaluation is at `1 - (1-t)/k_steps`), so the
        `1/(1-t)` in the velocity conversion never blows up.
    """
    return t + (1.0 - t) * (step / k_steps)


def unroll_alpha(step: int, k_steps: int) -> float:
    """Euler step size expressed as the fraction of the way to the clean prediction.

    For the linear interpolant `x_t = (1-t) x_0 + t x_1` the velocity is
    `v = (x_1 - x_t) / (1 - t)`, so an Euler step of `dt` gives

        x <- x + (x_hat_1 - x) * dt / (1 - t).

    On the uniform remaining-time grid of `unroll_time`, `dt = (1-t)/k_steps` and
    `1 - t_step = (1-t)(1 - step/k_steps)`, so the whole ratio collapses to a
    constant `1 / (k_steps - step)` -- independent of `t`, and with no division
    by `1 - t` anywhere. At the final step it equals 1, i.e. the unroll lands
    exactly on the network's last clean prediction, which is the standard
    "last Euler step is the clean prediction" behaviour of the sampler.
    """
    return 1.0 / float(k_steps - step)


def unrolled_clean_prediction(
    fm,
    call_nn: Callable,
    batch: dict,
    k_steps: int,
    grad_steps: int = 1,
    self_cond: bool = False,
    n_recycle: int = 0,
    nn_out_0: dict | None = None,
) -> dict[str, Tensor]:
    """Runs `k_steps` Euler steps from `(batch["x_t"], batch["t"])` to `t=1`.

    Does **not** mutate `batch`: every step runs the network on a shallow copy
    with its own `x_t`/`t`/`x_sc`. This matters because the caller's `batch` and
    `nn_out` are still live -- the flow loss, the chirality loss and the replay
    mixer all read them after this returns, and stamping the unroll's final
    `x_t` over the training interpolant would silently corrupt every one of them.

    Args:
        fm: the `ProductSpaceFlowMatcher`.
        call_nn: `Proteina.call_nn`, i.e. `(batch, n_recycle) -> nn_out`.
        batch: corrupted training batch, carrying `x_t`, `t`, `mask`.
        k_steps: number of Euler steps. `k_steps <= 1` returns the one-step clean
            prediction, which is exactly the pre-unroll behaviour.
        grad_steps: how many of the FINAL steps keep autograd. Earlier steps run
            under `no_grad`, so the returned tensor carries gradient only through
            the last `grad_steps` network evaluations -- this is DRaFT-K's
            truncated backprop, and it is what keeps memory flat as `k_steps`
            grows. Clamped into `[1, k_steps]`.
        self_cond: feed each step's clean prediction as the next step's `x_sc`,
            the way the sampler does. Only turn this on for a model trained with
            self-conditioning; the value is always detached, as in training.
        n_recycle: passed through to `call_nn`.
        nn_out_0: the network output already computed at `(x_t, t)` for the flow
            loss. Reusing it makes step 0 free. It is detached when step 0 falls
            outside the gradient window, which is numerically identical to having
            recomputed it under `no_grad`.

    Returns:
        `{data_mode: [B, n, d]}` clean-sample prediction at the end of the unroll.
    """
    data_modes = list(fm.data_modes)
    mask = batch["mask"]
    x = {dm: batch["x_t"][dm] for dm in data_modes}
    t0 = {dm: batch["t"][dm] for dm in data_modes}

    k_steps = int(k_steps)
    if k_steps <= 1:
        nn_out = call_nn(dict(batch), n_recycle=n_recycle) if nn_out_0 is None else nn_out_0
        return fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)

    grad_steps = max(1, min(int(grad_steps), k_steps))
    first_grad_step = k_steps - grad_steps
    outer_grad = torch.is_grad_enabled()

    x_sc = None
    x_1_pred = None
    for step in range(k_steps):
        # `outer_grad and ...` so an enclosing `no_grad` (validation, diagnostics)
        # is never overridden into building a graph.
        step_grad = outer_grad and (step >= first_grad_step)
        with torch.set_grad_enabled(step_grad):
            t_step = {dm: unroll_time(t0[dm], step, k_steps) for dm in data_modes}
            sub_batch = dict(batch)
            sub_batch["x_t"] = x
            sub_batch["t"] = t_step
            sub_batch["mask"] = mask
            if self_cond and x_sc is not None:
                sub_batch["x_sc"] = x_sc
            elif not self_cond:
                sub_batch.pop("x_sc", None)

            if step == 0 and nn_out_0 is not None:
                # Same inputs, same weights -> same values. Detaching when this step is
                # outside the gradient window reproduces a no_grad recompute exactly.
                nn_out = nn_out_0 if step_grad else _detach_nn_out(nn_out_0)
            else:
                nn_out = call_nn(sub_batch, n_recycle=n_recycle)

            x_1_pred = fm.nn_out_to_clean_sample_prediction(batch=sub_batch, nn_out=nn_out)
            alpha = unroll_alpha(step, k_steps)
            x = fm._apply_mask({dm: x[dm] + (x_1_pred[dm] - x[dm]) * alpha for dm in data_modes}, mask)
            if self_cond:
                x_sc = {dm: x_1_pred[dm].detach() for dm in data_modes}

    # The final step used alpha=1, so `x` IS the last clean prediction (masked).
    # Return `x` rather than `x_1_pred` so the result is the masked endpoint the
    # sampler would have emitted.
    return x


def _detach_nn_out(nn_out: dict) -> dict:
    """Detaches a nested `{data_mode: {key: Tensor}}` network output, one level deep."""
    out = {}
    for data_mode, preds in nn_out.items():
        if isinstance(preds, dict):
            out[data_mode] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in preds.items()}
        elif torch.is_tensor(preds):
            out[data_mode] = preds.detach()
        else:
            out[data_mode] = preds
    return out
