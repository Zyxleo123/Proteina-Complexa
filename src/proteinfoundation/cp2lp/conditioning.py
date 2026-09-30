"""The persistent source-CP condition, and the open-terminal topology request.

Two things have to be kept apart in the generator batch and the naming is deliberate:

``x_t`` / ``x_sc``
    the *evolving* LP state. It starts at noise and is integrated to t=1 by the sampler.

``x_src_cp``
    the *source* cyclic peptide, encoded once and frozen. It never changes across the
    trajectory and carries no gradient into the encoder. It is the condition, not the
    state -- the thing the generator is editing *away from*.

Reusing ``x_sc``'s tensor layout (``{"bb_ca": [b,n,3], "local_latents": [b,n,d]}``) is
what lets the existing ``XscBBCASeqFeat`` / ``XscLocalLatentsSeqFeat`` /
``XscBBCAPairwiseDistancesPairFeat`` classes serve as the source-CP feature extractors
under a different ``mode_key``, so the network gains this conditioning without a new
feature implementation. See ``feature_factory.get_creator``.

WHY THE CONDITION IS DETACHED. The AE is frozen in every arm here, but ``x_1`` as built
by ``add_clean_samples`` is a live encoder output. Feeding it in attached would let the
adversarial gradient flow into the encoder through the *condition* path even when
``freeze_autoencoder`` is honoured elsewhere, and would silently couple the condition to
the reconstruction objective. Detaching makes "the source CP is a fixed input" true by
construction rather than by config.

WHY A PRESENCE FLAG AND NOT JUST ZEROS. Generator training interleaves two batch kinds:
CP-conditioned (adversarial) and real-LP (plain flow matching, source condition absent).
Zeros are a legal value for a centred coordinate, so "absent" and "at the origin" are not
distinguishable from the tensor alone. ``src_cp_present`` is a [b] flag the network reads
as an explicit feature, which is also what makes the absent case a *trained* mode rather
than an out-of-distribution one.
"""

from __future__ import annotations

import torch
from loguru import logger

from proteinfoundation.cyclization.constants import LINEAR

#: Batch key holding the frozen encoded source CP, laid out like ``x_sc``.
SRC_CP_KEY = "x_src_cp"

#: Batch key holding the [b] float presence flag for ``SRC_CP_KEY``.
SRC_CP_PRESENT_KEY = "src_cp_present"


def attach_source_cp_condition(
    batch: dict,
    data_modes: list[str] | None = None,
    from_key: str = "x_1",
) -> dict:
    """Freezes the encoded source CP into ``batch[SRC_CP_KEY]``.

    Called on a CPSea batch *after* ``add_clean_samples`` has populated ``x_1`` with the
    AE-encoded clean structure, and *before* the sampler overwrites the state keys. The
    CP and the LP it generates share a residue count and a sequence, so the condition
    needs no alignment to the evolving state -- index ``i`` is the same residue in both.

    Args:
        batch: a batch carrying ``from_key`` as ``{data_mode: Tensor}``.
        data_modes: modes to copy across. Defaults to every mode present in ``from_key``.
        from_key: where the encoded clean structure lives. ``x_1`` in training.

    Returns:
        The same dict, with ``SRC_CP_KEY`` and ``SRC_CP_PRESENT_KEY`` set.
    """
    if from_key not in batch:
        raise KeyError(
            f"attach_source_cp_condition needs `{from_key}` in the batch -- call "
            "`add_clean_samples` first so the source CP has been encoded."
        )
    modes = list(batch[from_key].keys()) if data_modes is None else list(data_modes)
    missing = [m for m in modes if m not in batch[from_key]]
    if missing:
        raise KeyError(f"`{from_key}` is missing data modes {missing}; has {list(batch[from_key])}.")

    batch[SRC_CP_KEY] = {m: batch[from_key][m].detach() for m in modes}
    ref = batch[SRC_CP_KEY][modes[0]]
    batch[SRC_CP_PRESENT_KEY] = torch.ones(ref.shape[0], device=ref.device, dtype=ref.dtype)
    return batch


def drop_source_cp_condition(batch: dict, data_modes: list[str], n: int | None = None) -> dict:
    """Marks the source-CP condition absent, with correctly-shaped zeros.

    This is the real-LP branch: the shared flow network gets ordinary flow-matching
    supervision on bound linear peptides with nothing to edit from. Zeros rather than a
    missing key so the feature stack sees a consistent tensor shape across both batch
    kinds -- the *flag* is what carries "absent", see the module docstring.

    Args:
        batch: batch to modify in place. Must carry ``mask`` ([b, n]).
        data_modes: modes to zero-fill. ``local_latents`` is sized from the batch's own
            ``x_1`` when available, since the latent width is an AE property.
        n: residue count override; defaults to ``batch["mask"].shape[1]``.
    """
    mask = batch["mask"]
    b = mask.shape[0]
    n = mask.shape[1] if n is None else n
    device = mask.device
    ref_dtype = batch["x_1"][data_modes[0]].dtype if "x_1" in batch else torch.float32

    zeros = {}
    for m in data_modes:
        if m == "bb_ca":
            dim = 3
        elif "x_1" in batch and m in batch["x_1"]:
            dim = batch["x_1"][m].shape[-1]
        else:
            raise KeyError(
                f"drop_source_cp_condition cannot size data mode '{m}' -- pass a batch "
                "carrying `x_1`, or extend this function with an explicit width."
            )
        zeros[m] = torch.zeros(b, n, dim, device=device, dtype=ref_dtype)

    batch[SRC_CP_KEY] = zeros
    batch[SRC_CP_PRESENT_KEY] = torch.zeros(b, device=device, dtype=ref_dtype)
    return batch


def request_linear_topology(batch: dict, bs: int, device: torch.device | None = None) -> dict:
    """Sets the evolving state's topology request to LINEAR.

    LINEAR is an explicit request for *no ring*, distinct from UNSPECIFIED ("model's
    choice"), which is the classifier-free-guidance null. That distinction is load-bearing
    twice over here:

    * ``is_ring_request`` is False for LINEAR, so ``cyclization_ring_pe`` stays zero and
      the network is never handed a cycle graph asserting a bond the LP does not have.
    * the ring bond loss and the cyclization head skip LINEAR rows, so nothing in the
      training objective pulls the generated termini back together.

    Note this labels the *output*. The source CP's own topology is not overwritten -- it
    reaches the network through ``x_src_cp`` geometry, where closed termini are visible
    as a short CA-CA distance, rather than as a competing ring request on the same row.
    """
    device = batch["mask"].device if device is None else device
    batch["cyclization_type_cond"] = torch.full((bs,), LINEAR, dtype=torch.long, device=device)
    # An LP has no ring, so any endpoint labels inherited from the CP row would be read
    # by downstream consumers as a bond to supervise. Clear them rather than leaving
    # stale CP endpoints attached to a linear request.
    for stale in ("cyclization_i", "cyclization_j", "has_cyclization", "cyclization_type"):
        batch.pop(stale, None)
    return batch


def source_cp_terminal_gap_nm(batch: dict) -> torch.Tensor:
    """CA(0)-CA(L-1) distance of the source CP, in nm, per example. [b]

    Diagnostic only: this is the quantity the generator is supposed to *open*, so it is
    the natural x-axis for "did we actually move the termini apart" plots and the
    per-example baseline the export records alongside the generated gap.
    """
    src = batch[SRC_CP_KEY]["bb_ca"]  # [b, n, 3]
    mask = batch["mask"].bool()
    lengths = mask.sum(dim=-1)  # [b]
    b = src.shape[0]
    idx = torch.arange(b, device=src.device)
    first = src[idx, 0]
    last = src[idx, (lengths - 1).clamp(min=0)]
    gap = torch.sqrt(((last - first) ** 2).sum(dim=-1) + 1e-12)
    return torch.where(lengths > 1, gap, torch.zeros_like(gap))


def warn_if_condition_unused(feats_seq: list[str], feats_pair_repr: list[str]) -> None:
    """Loud check that the config actually wires the source-CP condition into the network.

    Building ``x_src_cp`` and never requesting a feature that reads it is a silent
    failure: training runs, losses fall, and the generator is an unconditional LP model
    that ignores the CP it was asked to edit. This is the exact shape of the
    type-conditioning bug that cost a full eval cycle, so it gets a warning rather than a
    comment.
    """
    wired = [f for f in list(feats_seq) + list(feats_pair_repr) if f.startswith("x_src_cp")]
    if not wired:
        logger.warning(
            "CP2LP: the source-CP condition is built but NO feature reads it -- add "
            "`x_src_cp_bb_ca` / `x_src_cp_local_latents` to nn.feats_seq and "
            "`x_src_cp_bb_ca_pair_dists` to nn.feats_pair_repr, or the generator is "
            "unconditional and the CP argument does nothing."
        )
    else:
        logger.info(f"CP2LP: source-CP condition wired through {wired}")
