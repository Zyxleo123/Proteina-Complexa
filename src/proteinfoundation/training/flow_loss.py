"""Shared clean-target -> flow-matching-loss path.

`Proteina.training_step` builds its loss in four steps: `corrupt_batch`
(fresh `t`, fresh source noise, interpolation -- all pure functions of a
clean `x_1` dict and a residue mask), `call_nn` (the forward pass), and
`fm.compute_loss` (the per-example Cα + local-latent flow losses). Every one
of those steps already operates on whatever `x_1`/`mask` it is given; nothing
inside `ProductSpaceFlowMatcher` needs to change to also accept a *generated*
clean target (a replay-buffer endpoint) instead of one derived from a real
CPSea batch via `add_clean_samples`.

This module extracts exactly that chain into one function so replay training
(Stage 7) can call the *same* code real CPSea batches call, rather than
reimplementing interpolation/time-sampling/masking. The only thing this adds
on top of the existing pieces is per-example `sample_weight` support, applied
to every loss key `compute_loss` returns *except* the `_justlog`-suffixed
diagnostic-only keys -- mirroring exactly which keys `Proteina.training_step`
itself sums into the trained loss (`"_justlog" not in k`), so weighting here
composes correctly with the caller's own `torch.mean(...)` reduction.

Deliberately NOT changed by this module: `Proteina.training_step`'s own
real-batch code path (still calls `add_clean_samples` + `corrupt_batch`
directly, unmodified) -- this function is additive, used by the replay path
introduced in Stage 7, and is provably equivalent to `training_step`'s
existing math (see `script_utils/test_flow_loss_shared_fn.py`) rather than
replacing it outright, since the production training path cannot be
regression-tested here without a GPU/full dataset.
"""

from __future__ import annotations

import torch


def flow_loss_from_clean_target(
    proteina,
    clean_x1_ca: torch.Tensor,
    clean_z1: torch.Tensor,
    condition: dict,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    n_recycle: int = 0,
    from_replay: bool = False,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Computes the ordinary CPSea flow-matching loss for a given clean target.

    Args:
        proteina: Object exposing `.fm` (a `ProductSpaceFlowMatcher`-like
            instance with `sample_t`/`sample_noise`/`interpolate`/
            `compute_loss`) and `.call_nn(batch, n_recycle)`. In production
            this is the `Proteina` LightningModule itself.
        clean_x1_ca: [B, N, 3], nm. Clean Cα target (real dataset coordinates,
            or a stored replay-buffer `x1_ca`).
        clean_z1: [B, N, D], clean local-latent target (real AE encoding, or a
            stored replay-buffer `z1`).
        condition: Everything else `call_nn` needs on the batch dict (e.g.
            `cyclization_type_cond`, hotspot/target conditioning, `cath_code`)
            -- merged into the constructed batch before the forward pass.
            Must NOT contain `x_0`/`x_1`/`x_t`/`t`/`mask` keys (those are set
            here).
        mask: [B, N] bool, valid-residue mask.
        sample_weight: Optional [B] per-example weight, multiplied into every
            trainable (non-`_justlog`) loss key before the caller's own
            `torch.mean(...)` reduction. `None` is equivalent to all-ones.
        n_recycle: Forwarded to `call_nn` (0 by default, matching a plain,
            non-recycled forward pass).
        from_replay: If True, defensively asserts `clean_x1_ca`/`clean_z1` do
            not require grad (a replay endpoint must already be fully
            detached at collection time -- see
            `proteinfoundation.replay.buffer` -- so this should never fire in
            practice; it exists to catch an accidental live/undetached tensor
            being routed through the replay path).

    Returns:
        `(losses, batch)`: `losses` is the same `{key: [B]-tensor}` dict shape
        `Proteina.fm.compute_loss` returns (with `sample_weight` applied,
        `_justlog` keys untouched), ready for
        `sum(torch.mean(losses[k]) for k in losses if "_justlog" not in k)`
        exactly as `Proteina.training_step` does today. `batch` is the
        constructed batch dict, returned for callers that want to log/inspect
        `t`, `x_0`, `x_t`, etc.
    """
    if from_replay:
        if clean_x1_ca.requires_grad or clean_z1.requires_grad:
            raise RuntimeError(
                "flow_loss_from_clean_target(from_replay=True) received a "
                "clean target that still requires grad. Replay endpoints "
                "must be fully detached at collection time (see "
                "proteinfoundation.replay.buffer); refusing to silently "
                "backprop through sampling/decoding/reward."
            )

    fm = proteina.fm
    x_1 = {"bb_ca": clean_x1_ca, "local_latents": clean_z1}
    x_1 = fm._apply_mask(x=x_1, mask=mask)

    batch_shape = tuple(mask.shape[:-1])
    n = mask.shape[-1]
    device = clean_x1_ca.device

    t = fm.sample_t(shape=batch_shape, device=device)
    x_0 = fm.sample_noise(n=n, shape=batch_shape, mask=mask, device=device)
    x_t = fm.interpolate(x_0=x_0, x_1=x_1, t=t, mask=mask)

    batch = dict(condition)
    batch["x_0"] = x_0
    batch["x_1"] = x_1
    batch["x_t"] = x_t
    batch["t"] = t
    batch["mask"] = mask

    nn_out = proteina.call_nn(batch, n_recycle=n_recycle)
    losses = fm.compute_loss(batch=batch, nn_out=nn_out)

    if sample_weight is not None:
        weight = sample_weight.to(device=device)
        losses = {
            key: (value * weight if "_justlog" not in key else value)
            for key, value in losses.items()
        }

    return losses, batch
