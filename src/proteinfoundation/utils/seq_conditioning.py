"""Sequence conditioning: reveal (part of) the binder sequence and generate the rest.

The flow model emits two data modes -- ``bb_ca`` (Ca trace) and ``local_latents`` (the
per-residue autoencoder latent, which decodes to Atom37 *and* to a residue type). Nothing
in that product space is "the sequence": the sequence is a *readout* of the latents. So
"give me the peptide sequence, generate the structure" is not a masking of the generated
variables -- it is a **conditioning feature**: the requested residue types are fed to the
network as an input (``OptionalResidueTypeSeqFeat``, feature key ``optional_res_type_seq_feat``)
and the model must produce latents whose decoded structure is consistent with them.

``batch["residue_type"]`` is BINDER-ONLY here (the receptor's sequence travels separately as
``seq_target``; see ``FilterTargetResiduesTransform``), so revealing it reveals exactly the
peptide and nothing about the receptor.

The conditioning flag ``batch["use_residue_type_feature"]`` accepts four forms, in
increasing resolution. All four are honoured by every consumer that routes through
``resolve_sequence_conditioning_mask``:

    False / absent      nothing revealed (the historical default)
    True                the whole binder sequence revealed, for every example
    bool tensor [b]     per-example all-or-nothing
    bool tensor [b, n]  per-residue -- a partial sequence, "generate the rest"

The per-residue form is what makes partial specification possible, and it is also what
makes the full-sequence case *learnable*: a model trained only on all-or-nothing has no
gradient telling it what an individually-specified residue means.

WHY AN UNKNOWN POSITION IS ALL-ZEROS AND NOT A 21ST CHANNEL. A revealed position is a
one-hot over 20 types, so the all-zero vector is already unambiguously "not specified" --
no extra channel is needed to distinguish them. That matters practically: keeping the
feature at dim 20 leaves the trunk's input projection shape-compatible with existing
checkpoints, so a sequence-conditioned arm can warm-start from an unconditioned one
instead of training from scratch.
"""

from __future__ import annotations

import torch


def resolve_sequence_conditioning_mask(
    flag,
    b: int,
    n: int,
    device: torch.device,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Normalise ``use_residue_type_feature`` into a ``[b, n]`` bool mask.

    Args:
        flag: the raw value of ``batch["use_residue_type_feature"]`` (see module docstring).
        b, n: batch size and number of binder residues.
        device: device to place the mask on.
        pad_mask: optional ``[b, n]`` residue mask; padding is forced un-revealed so a
            padded row can never be counted as "conditioned" by a downstream metric.

    Returns:
        ``[b, n]`` bool tensor, or ``None`` when nothing is revealed at all. ``None`` (rather
        than an all-False mask) lets callers skip the work entirely and, in the feature, skip
        the ``residue_type`` lookup -- which need not exist when conditioning is off.
    """
    if flag is None:
        return None

    if torch.is_tensor(flag):
        mask = flag.to(device=device)
        mask = mask.bool() if mask.dtype != torch.bool else mask
        if mask.dim() == 0:
            mask = mask.view(1, 1).expand(b, n)
        elif mask.dim() == 1:
            if mask.shape[0] != b:
                raise ValueError(
                    f"use_residue_type_feature has shape {tuple(flag.shape)}; a 1-D flag must be "
                    f"per-example, i.e. length b={b}."
                )
            mask = mask[:, None].expand(b, n)
        elif mask.dim() == 2:
            if mask.shape != (b, n):
                raise ValueError(
                    f"use_residue_type_feature has shape {tuple(flag.shape)}, expected ({b}, {n})."
                )
        else:
            raise ValueError(f"use_residue_type_feature must be 0/1/2-D, got {mask.dim()}-D.")
        mask = mask.contiguous()
    else:
        if not bool(flag):
            return None
        mask = torch.ones(b, n, dtype=torch.bool, device=device)

    if pad_mask is not None:
        mask = mask & pad_mask.to(device=device).bool()

    if not bool(mask.any()):
        return None
    return mask


def sequence_conditioning_fraction(flag, b: int, n: int, device: torch.device) -> float:
    """Fraction of ``b * n`` slots revealed -- for logging only. 0.0 when nothing is."""
    mask = resolve_sequence_conditioning_mask(flag, b, n, device)
    if mask is None:
        return 0.0
    return float(mask.float().mean().item())


def sample_sequence_conditioning_mask(
    pad_mask: torch.Tensor,
    p: float,
    keep_frac_min: float,
    keep_frac_max: float,
    p_full: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a training-time ``[b, n]`` reveal mask.

    Three-level draw, per example:

    1. with probability ``p`` this example is conditioned at all (otherwise all-False --
       this is the classifier-free-style dropout that keeps the *unconditional* generator
       intact, so one checkpoint serves both "design me a peptide" and "fold this peptide");
    2. of the conditioned ones, a fraction ``p_full`` get the ENTIRE sequence. This is not
       decoration: full specification is the query the user actually types, and if it only
       ever arises as the measure-zero endpoint of a continuous keep-fraction it is
       effectively never trained;
    3. the rest get a keep fraction drawn uniformly from ``[keep_frac_min, keep_frac_max]``,
       then an independent per-residue Bernoulli at that rate. Drawing the *rate* per example
       rather than using one global rate makes the model see whole rows that are 20% specified
       and whole rows that are 90% specified, instead of every row landing near the mean.

    Padding is excluded, and an example whose draw reveals nothing is left all-False (which
    is simply the unconditional case, so it needs no special handling).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if not (0.0 <= p_full <= 1.0):
        raise ValueError(f"p_full must be in [0, 1], got {p_full}")
    if not (0.0 <= keep_frac_min <= keep_frac_max <= 1.0):
        raise ValueError(
            f"require 0 <= keep_frac_min <= keep_frac_max <= 1, got {keep_frac_min}, {keep_frac_max}"
        )

    pad_mask = pad_mask.bool()
    b, n = pad_mask.shape
    device = pad_mask.device

    def _rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    conditioned = _rand(b) < p  # [b]
    full = _rand(b) < p_full  # [b]
    keep_frac = keep_frac_min + (keep_frac_max - keep_frac_min) * _rand(b)  # [b]
    keep_frac = torch.where(full, torch.ones_like(keep_frac), keep_frac)  # [b]

    per_res = _rand(b, n) < keep_frac[:, None]  # [b, n]
    return per_res & conditioned[:, None] & pad_mask
