"""Inference-time cyclization prediction, using the predicted-AA validity mask.

Kept separate from `proteina.py` so the search/generation call sites only
need a couple of no-op-safe helper calls.
"""

import torch

from proteinfoundation.cyclization.loss import decode_cyclization_prediction
from proteinfoundation.cyclization.mask import build_cyclization_validity_mask


@torch.no_grad()
def predict_cyclization_from_clean(
    proteina,
    ca: torch.Tensor,
    z_latent: torch.Tensor,
    mask: torch.Tensor,
    cond_type: torch.Tensor | None = None,
) -> dict:
    """Predicts the cyclization edge from a predicted-clean generated sample.

    Builds the validity mask from the *predicted* AA sequence (never ground truth).

    Args:
        proteina: `Proteina` instance with a trained `cyclization_head` and `autoencoder`.
        ca: [B, L, 3] predicted clean CA coordinates (nm).
        z_latent: [B, L, latent_dim] predicted clean local latents.
        mask: [B, L] bool binder mask.
        cond_type: [B] requested cyclization type (`UNSPECIFIED` for "any"), or
            None for the unconditional path.

    Returns:
        Dict with `pred_cyclization_i/j/type/confidence` tensors, each [B], plus
        `cyclization_type_satisfied` ([B] bool): whether the generated sequence
        actually admits the requested type. A False row means the request could
        not be honored -- its predicted edge is null (-1).
    """
    decoded = proteina.autoencoder.decode(z_latent=z_latent, ca_coors_nm=ca, mask=mask)
    aa_probs = decoded["seq_logits"].softmax(dim=-1)
    pred_aa = decoded["residue_type"]

    single_for_link = torch.cat([z_latent, aa_probs], dim=-1)
    link_logits = proteina.cyclization_head(single_for_link, ca)

    valid_mask = build_cyclization_validity_mask(
        aa=pred_aa,
        binder_mask=mask,
        force_gold_valid=False,
        allow_asn_gln_isopeptide=getattr(proteina, "cyclization_allow_asn_gln_isopeptide", True),
        cond_type=cond_type,
        terminal_only=getattr(proteina, "cyclization_terminal_only", False),
    )

    # Whether the requested type is achievable at all under the generated sequence.
    # Must be read off the mask *before* decoding, since the unconditional path's
    # MAINCHAIN fallback would otherwise paper over an empty candidate set.
    satisfied = valid_mask.reshape(valid_mask.shape[0], -1).any(dim=-1)

    conditioned = cond_type is not None
    pred = decode_cyclization_prediction(link_logits, valid_mask, fallback=not conditioned)
    pred["cyclization_type_satisfied"] = satisfied
    return pred


@torch.no_grad()
def attach_cyclization_prediction(proteina, gen_samples: dict, batch: dict, sample_prots: dict) -> dict:
    """Attaches predicted cyclization metadata to a `sample_prots` dict, in place.

    No-op if cyclization is disabled, or if the generation does not include
    the `local_latents` data mode (cyclization needs decoded AA probabilities
    from the autoencoder).

    The requested type is read from `batch["cyclization_type_cond"]`, which
    `Proteina.generate` stamps in from `cyclization.target_type` (and which a CPSea
    batch may already carry per-sample). Reading it from the batch rather than
    re-deriving it from the config guarantees the head is conditioned on exactly
    the same type the denoiser saw.
    """
    if not getattr(proteina, "cyclization_enabled", False) or getattr(proteina, "cyclization_head", None) is None:
        return sample_prots
    if "local_latents" not in gen_samples or "bb_ca" not in gen_samples:
        return sample_prots

    cond_type = batch.get("cyclization_type_cond") if proteina.cyclization_type_conditioning else None
    if cond_type is not None:
        cond_type = cond_type.long().to(gen_samples["bb_ca"].device)

    pred = predict_cyclization_from_clean(
        proteina,
        ca=gen_samples["bb_ca"],
        z_latent=gen_samples["local_latents"],
        mask=batch["mask"],
        cond_type=cond_type,
    )
    if cond_type is not None:
        pred["cyclization_type_requested"] = cond_type
    sample_prots.update(pred)
    return sample_prots
