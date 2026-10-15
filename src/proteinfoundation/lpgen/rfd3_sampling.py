"""Per-row sampling weights for the RFD3 LP dataset.

Without a `weights` (or `sampler`) sibling of `dataset` in the dataset config, foundry
falls back to uniform sampling with replacement -- loudly, but the warning is one line in a
long log:

    No weights or sampler provided for dataset: None, using uniform weights with replacement.

Uniform over rows is *not* what this milestone wants. The LP pool is 92% ProtFrag by row
count, while the configured mixture is 24% PepBench / 76% ProtFrag -- the ratio the project
already chose (``training_cpsea_lpmix_from_v4cfg.yaml``, renormalised over the LP-only
pool), reasoned ~sqrt-proportionally to pool size. The manifest therefore carries a
per-row ``sample_weight`` that integrates to that mixture, and this reads it.

``calculate_weights_for_pdb_dataset_df``, the weights function the PDB datasets use, cannot
serve here: it derives weights from PDB-specific cluster and composition columns
(``alphas`` over protein/nucleic/ligand counts, a ``cluster`` column) that our manifest does
not have, and it would in any case recompute a mixture we have already decided.

foundry calls this as ``instantiate(cfg.weights, dataset_df=dataset.data)``, where
``dataset.data`` is the frame *after* the dataset's own ``filters`` have been applied -- so
the rows seen here are exactly the rows that will be sampled, and the weights line up by
position.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def weights_from_column(
    dataset_df,
    column: str = "sample_weight",
    *,
    allow_zero: bool = False,
) -> torch.Tensor:
    """Per-row sampling weights taken from a manifest column.

    Args:
        dataset_df: the dataset's post-filter DataFrame, supplied by foundry.
        column: the column holding the weights.
        allow_zero: permit rows with zero weight. Off by default, because a zero-weight row
            is silently never sampled -- which is indistinguishable from the row not being
            in the dataset at all, and is exactly how a filter mismatch would hide. The
            manifest's weights are nonzero only on ``split == 'train'`` rows, so a zero here
            means the dataset's filter and the weight column disagree about the split.

    Returns:
        A float tensor, one weight per row, in row order.
    """
    if column not in dataset_df.columns:
        raise KeyError(
            f"weights column {column!r} is not in the dataset frame; available: "
            f"{sorted(dataset_df.columns)}. Note the column must also be listed in the "
            f"dataset's `columns_to_load`, which is applied before this runs."
        )

    w = torch.as_tensor(dataset_df[column].to_numpy(), dtype=torch.float64)

    if not torch.isfinite(w).all():
        n = int((~torch.isfinite(w)).sum())
        raise ValueError(f"{n} non-finite value(s) in {column!r}")
    if (w < 0).any():
        n = int((w < 0).sum())
        raise ValueError(f"{n} negative value(s) in {column!r}")

    n_zero = int((w == 0).sum())
    if n_zero and not allow_zero:
        raise ValueError(
            f"{n_zero} of {len(w)} rows have zero {column!r}. A zero-weight row is never "
            f"sampled, which looks identical to it being absent. The manifest zeroes this "
            f"column outside the train split, so this usually means the dataset's filter "
            f"does not match the split the weights were computed for. Pass "
            f"allow_zero=true only if that is intended."
        )

    total = float(w.sum())
    if total <= 0:
        raise ValueError(f"{column!r} sums to {total}; nothing could be sampled")

    # WeightedRandomSampler normalises internally, so the absolute scale is irrelevant; the
    # realised mixture is what matters and it is reported here so the log records the
    # intended ratio rather than leaving it to be inferred from a config file.
    if "dataset_source" in dataset_df.columns:
        shares = (
            dataset_df.assign(_w=w.numpy()).groupby("dataset_source")._w.sum() / total
        )
        logger.info(
            "LP sampling weights from %r over %d rows; realised mixture: %s",
            column, len(w), {k: round(float(v), 4) for k, v in shares.items()},
        )
    else:
        logger.info("LP sampling weights from %r over %d rows", column, len(w))

    return w.to(torch.float32)
