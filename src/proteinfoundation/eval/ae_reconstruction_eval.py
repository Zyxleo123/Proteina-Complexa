"""AE-only reconstruction diagnostic (Part 2) and four-way decode diagnostic (Part 3).

These functions never modify the autoencoder, the flow matcher, or the batch's
existing `x_1`/`x_t`/`local_latents_encoder_output` entries -- they are purely
read-only diagnostics layered on top of quantities that are already computed
during a normal forward pass (`AutoEncoder.encode`/`decoder`,
`ProductSpaceFlowMatcher.nn_out_to_clean_sample_prediction`).

See `proteinfoundation.eval.cyclic_reconstruction_metrics` for the underlying
metric math, and `proteinfoundation.logging.metric_schema` for the full set of
keys these functions are expected to (eventually) populate.
"""

from __future__ import annotations

from typing import Any

import torch

from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    cyclic_geometry_metrics,
    extract_cyclization_metadata,
    masked_atom37_coord_metrics,
    sequence_metrics,
)

CA_IDX = 1  # Atom37 slot for CA (see cyclic_reconstruction_metrics.get_atom_index("CA"))
_EPS = 1e-6


def _resolve_binder_mask(batch: dict) -> torch.Tensor:
    """Same fallback chain as `AutoEncoder.encode`/`training_step`: `mask_dict` (PDB-monomer
    datamodule) -> `mask` (already set, e.g. by `corrupt_batch`) -> CA-presence from `coord_mask`
    (CPSea `StructureDataModule`, which has neither `mask_dict` nor a pre-existing `mask` before
    `add_clean_samples`/`corrupt_batch` runs)."""
    if "mask_dict" in batch:
        return batch["mask_dict"]["coords"][..., 0, 0].bool()
    if "mask" in batch:
        return batch["mask"].bool()
    return batch["coord_mask"][..., CA_IDX].bool()


def decode_and_score(
    autoencoder: Any,
    x_ca_nm: torch.Tensor,
    z_latent: torch.Tensor,
    gt_atom37: torch.Tensor,
    gt_atom37_mask: torch.Tensor,
    gt_seq_tokens: torch.Tensor,
    residue_mask: torch.Tensor,
    cyclization_metadata: dict[str, torch.Tensor] | None,
    prefix: str,
) -> dict[str, float]:
    """Decodes `(x_ca_nm, z_latent)` and scores against ground truth.

    Calls the raw `autoencoder.decoder(...)` (not the `AutoEncoder.decode()` wrapper) so the
    returned `coors_nm` is masked only by `residue_mask` -- not additionally zeroed at atom slots
    where the *predicted* sequence disagrees with ground truth, which would corrupt comparisons
    against the ground-truth atom mask. This mirrors the same raw-decoder-call pattern already
    used by `AutoEncoder.compute_ae_reg_losses`.

    Args:
        autoencoder: `AutoEncoder` instance (any checkpoint: PC pretrained, CPSea-finetuned, or
            the current run's).
        x_ca_nm: [B, L, 3] Ca coordinates to condition the decoder on, nm.
        z_latent: [B, L, d] latent to decode, in the AE's own (unnormalized) latent space.
        gt_atom37: [B, L, 37, 3] ground-truth Atom37 coordinates, nm.
        gt_atom37_mask: [B, L, 37] ground-truth valid-atom mask.
        gt_seq_tokens: [B, L] ground-truth residue type ids.
        residue_mask: [B, L] valid-residue mask.
        cyclization_metadata: output of `extract_cyclization_metadata`, or `None`.
        prefix: metric key prefix, e.g. "val/decode_gtca_gtz".

    Returns:
        Flat dict merging `masked_atom37_coord_metrics`, `sequence_metrics`, and
        `cyclic_geometry_metrics`, all under `prefix`.
    """
    mask = residue_mask.bool()
    input_decoder = {
        "z_latent": z_latent,
        "ca_coors_nm": x_ca_nm * mask[..., None],
        "residue_mask": mask,
        "mask": mask,
    }
    output_dec = autoencoder.decoder(input_decoder)
    pred_atom37 = output_dec["coors_nm"]
    pred_seq_logits = output_dec["seq_logits"]

    metrics: dict[str, float] = {}
    metrics.update(masked_atom37_coord_metrics(pred_atom37, gt_atom37, gt_atom37_mask, prefix=prefix))
    metrics.update(sequence_metrics(pred_seq_logits, gt_seq_tokens, mask, prefix=prefix))
    metrics.update(
        cyclic_geometry_metrics(
            pred_atom37=pred_atom37,
            gt_atom37=gt_atom37,
            atom37_mask=gt_atom37_mask,
            seq_tokens=gt_seq_tokens,
            cyclization_metadata=cyclization_metadata,
            prefix=prefix,
        )
    )
    return metrics


def run_ae_reconstruction_eval(
    autoencoder: Any,
    batch: dict,
    prefix: str = "val/ae_recon",
    sample_posterior: bool = False,
) -> dict[str, float]:
    """AE-only reconstruction diagnostic (Part 2): `D(x_ca_gt, E(x_gt))` vs ground truth.

    Works for any AE checkpoint (PC pretrained, CPSea-finetuned, or the current run's own),
    since it only needs `autoencoder.encode`/`autoencoder.decoder` and a CPSea-style batch dict
    (`coords_nm`, `coord_mask`, `residue_type`, optionally `cyclization_i/j/type`).

    Args:
        autoencoder: `AutoEncoder` instance.
        batch: data batch with `coords_nm` [B, L, 37, 3] (nm), `coord_mask` [B, L, 37],
            `residue_type` [B, L], and optionally CPSea cyclization fields.
        prefix: metric key prefix.
        sample_posterior: if True, decode from a reparameterized posterior sample
            (`z_latent`); if False (default, matches `eval.reconstruction.sample_posterior`),
            decode from the deterministic posterior mean.

    Returns:
        Flat dict of reconstruction metrics (see `decode_and_score`).
    """
    mask = _resolve_binder_mask(batch)
    batch = dict(batch)
    batch["mask"] = mask  # AutoEncoder.encode reads/derives this; keep consistent for callers.

    x_ca_gt = batch["coords_nm"][..., CA_IDX, :] * mask[..., None]

    enc = autoencoder.encode(batch)
    z = enc["z_latent"] if sample_posterior else enc["mean"]

    cyclization_metadata = extract_cyclization_metadata(batch)

    return decode_and_score(
        autoencoder=autoencoder,
        x_ca_nm=x_ca_gt,
        z_latent=z,
        gt_atom37=batch["coords_nm"],
        gt_atom37_mask=batch["coord_mask"],
        gt_seq_tokens=batch["residue_type"],
        residue_mask=mask,
        cyclization_metadata=cyclization_metadata,
        prefix=prefix,
    )


def _unnormalize_latent(autoencoder: Any, z: torch.Tensor) -> torch.Tensor:
    """Inverts the channel-wise `latent_normalization` applied in `get_clean_sample`
    (`z_ae = z_flow * std + mean`), identical to the inverse already used in
    `format_sample_local_latents` (`utils/sample_utils.py`). No-op if normalization is disabled."""
    latent_norm_stats = getattr(autoencoder, "latent_norm_stats", None)
    if latent_norm_stats is None:
        return z
    norm_mean = latent_norm_stats["mean"].to(device=z.device, dtype=z.dtype)
    norm_std = latent_norm_stats["std"].to(device=z.device, dtype=z.dtype)
    return z * norm_std + norm_mean


def _masked_mean_abs_and_sq(err: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Per-structure-then-batch-mean L1/L2 statistics of `err` (last dim is the feature dim),
    masked over the leading [B, L] residue dims. Returns (mean_abs, sqrt_mean_sq)."""
    mask = mask.bool()
    err = err * mask[..., None]
    feat_dim = err.shape[-1]
    n_valid = mask.sum(dim=-1).float() * feat_dim  # [B], number of valid scalar entries
    abs_err = err.abs().sum(dim=(-2, -1)) / n_valid.clamp(min=_EPS)  # [B]
    sq_err = (err**2).sum(dim=(-2, -1)) / n_valid.clamp(min=_EPS)  # [B]
    return float(abs_err.mean().item()), float(torch.sqrt(sq_err).mean().item())


def run_four_way_decode_eval(
    autoencoder: Any,
    fm: Any,
    batch: dict,
    nn_out: dict,
) -> dict[str, float]:
    """Four-way decode diagnostic (Part 3): isolates decoder/AE, Ca-branch, latent-branch,
    and interaction failure modes by decoding all four combinations of
    `{x_ca_gt, x_ca_pred} x {z_gt, z_pred}`.

    Must be called after `add_clean_samples` has populated
    `batch["local_latents_encoder_output"]` and `batch["x_1"]`. `x_ca_pred`/`z_pred` are derived
    from `nn_out` using exactly the same conversion as the flow loss
    (`fm.nn_out_to_clean_sample_prediction`, i.e. `x_1 = x_t + (1 - t) * v` under the hood) -- no
    new prediction math is introduced.

    Args:
        autoencoder: `AutoEncoder` instance (the run's own, frozen or joint-trained).
        fm: `ProductSpaceFlowMatcher` instance (`Proteina.fm`).
        batch: batch after `add_clean_samples` (and, for `x_t`/`t`, after `corrupt_batch`).
        nn_out: output of `Proteina.call_nn(batch, ...)`.

    Returns:
        Flat dict merging the four `val/decode_*` prefixes (each with the full
        `RECON_METRIC_SUFFIXES` key set) plus `val/flow_clean/*` direct clean-state metrics.
    """
    mask = batch["mask"].bool()
    gt_atom37 = batch["coords_nm"]
    gt_atom37_mask = batch["coord_mask"]
    gt_seq_tokens = batch["residue_type"]
    cyclization_metadata = extract_cyclization_metadata(batch)

    z_gt = batch["local_latents_encoder_output"]["mean"]
    x_ca_gt = batch["x_1"]["bb_ca"]

    x_1_pred = fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
    x_ca_pred = x_1_pred["bb_ca"]
    z_pred = _unnormalize_latent(autoencoder, x_1_pred["local_latents"])

    combos = {
        "val/decode_gtca_gtz": (x_ca_gt, z_gt),
        "val/decode_predca_gtz": (x_ca_pred, z_gt),
        "val/decode_gtca_predz": (x_ca_gt, z_pred),
        "val/decode_predca_predz": (x_ca_pred, z_pred),
    }

    metrics: dict[str, float] = {}
    for prefix, (x_ca, z) in combos.items():
        metrics.update(
            decode_and_score(
                autoencoder=autoencoder,
                x_ca_nm=x_ca,
                z_latent=z,
                gt_atom37=gt_atom37,
                gt_atom37_mask=gt_atom37_mask,
                gt_seq_tokens=gt_seq_tokens,
                residue_mask=mask,
                cyclization_metadata=cyclization_metadata,
                prefix=prefix,
            )
        )

    ca_mae_A, ca_rmse_A = _masked_mean_abs_and_sq((x_ca_pred - x_ca_gt) * 10.0, mask)
    z_l1, z_rmse = _masked_mean_abs_and_sq(z_pred - z_gt, mask)
    metrics["val/flow_clean/ca_coord_mae_A"] = ca_mae_A
    metrics["val/flow_clean/ca_rmse_A"] = ca_rmse_A
    metrics["val/flow_clean/z_mse"] = z_rmse**2
    metrics["val/flow_clean/z_l1"] = z_l1

    return metrics
