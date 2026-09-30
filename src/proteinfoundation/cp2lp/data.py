"""Batch plumbing for CP -> LP: role splitting, length matching, receptor-family splits.

Three separate concerns live here.

ROLE SPLITTING. One mixed dataloader feeds both branches. CPSea rows are generator
inputs; PepBench/ProtFrag rows are real LP examples for the discriminator and for the
plain flow-matching term. The role is read off ``cyclization_type_cond``, which
``CyclizationLabelTransform`` has already set -- a ring request means cyclic, LINEAR means
linear -- so no new column has to be plumbed through the collate. It must be read BEFORE
``apply_cyclization_type_conditioning`` runs its CFG dropout, which rewrites some rows to
UNSPECIFIED; :func:`split_roles` is therefore called at the top of the training step.

LENGTH MATCHING. The discriminator must not be able to separate real from generated on
peptide length. Within a single batch of 16 at a 75/25 mix there are only ~4 real rows,
so matching inside the batch would usually fail. A per-length reservoir of previously
seen real examples fixes that: features are deterministic given a frozen AE, so a cached
feature dict is exactly what a freshly computed one would be, and drawing a same-length
real from the cache costs nothing.

RECEPTOR-FAMILY SPLITS. Train and eval must not share receptor families, or the "held-out"
evaluation is measuring memorisation. Families come from ``cluster_id`` where the metadata
has one. The split is a deterministic hash of the family id, not a shuffle, so it is
reproducible without storing a list and stable when the metadata gains rows.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict, deque

import torch
from loguru import logger

from proteinfoundation.cyclization.constants import is_ring_request


def slice_batch(batch: dict, index: torch.Tensor, bs: int) -> dict:
    """Selects examples from a collated batch along the batch axis.

    Args:
        batch: collated batch. Nested dicts of tensors (``x_1``, ``mask_dict``, ...) are
            sliced recursively; anything whose leading dimension is not ``bs`` is passed
            through untouched, which is the right behaviour for scalars, config objects
            and per-batch metadata.
        index: long or bool tensor selecting examples.
        bs: the batch size to match against, so a coincidentally-``bs``-shaped inner axis
            of a non-batch tensor is not sliced by accident.
    """

    def _slice(v):
        if torch.is_tensor(v):
            return v[index] if v.dim() >= 1 and v.shape[0] == bs else v
        if isinstance(v, dict):
            return {k: _slice(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)) and len(v) == bs:
            keep = index.nonzero(as_tuple=True)[0].tolist() if index.dtype == torch.bool else index.tolist()
            out = [v[i] for i in keep]
            return type(v)(out) if isinstance(v, tuple) else out
        return v

    return {k: _slice(v) for k, v in batch.items()}


def split_roles(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cyclic_index, linear_index)`` boolean masks over the batch axis.

    Read from the *pre-dropout* ``cyclization_type_cond``. A row with no cyclization
    label at all counts as linear: that is the conservative direction, since treating a
    genuinely cyclic row as a real LP example would poison the discriminator's real pool,
    while treating a linear row as a generator input merely wastes a rollout.
    """
    bs = batch["mask"].shape[0]
    device = batch["mask"].device
    if "cyclization_type_cond" not in batch:
        logger.warning("split_roles: no `cyclization_type_cond` in batch -- treating every row as linear.")
        return (
            torch.zeros(bs, dtype=torch.bool, device=device),
            torch.ones(bs, dtype=torch.bool, device=device),
        )
    cond = batch["cyclization_type_cond"].to(device)
    cyclic = is_ring_request(cond)
    return cyclic, ~cyclic


class RealLPFeatureReservoir:
    """Per-length ring buffers of discriminator features for real bound LPs.

    Stores feature dicts, not raw structures: a frozen AE makes the decode deterministic,
    so a cached feature dict equals a recomputed one, and the per-example cost drops from
    ~100 KB of receptor coordinates to a few KB of local descriptors.

    Entries are held on CPU and moved to the device on draw. That keeps a large reservoir
    off the GPU, where the rollout already needs the headroom.
    """

    def __init__(self, capacity_per_length: int = 64, seed: int = 0):
        self.capacity = int(capacity_per_length)
        self.buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.capacity))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return sum(len(v) for v in self.buffers.values())

    def lengths(self) -> list[int]:
        return sorted(k for k, v in self.buffers.items() if len(v) > 0)

    def push_batch(self, feats: dict[str, torch.Tensor], lengths: torch.Tensor) -> None:
        """Adds each example of a feature batch to the buffer for its peptide length.

        Per-residue tensors are TRIMMED to the example's own length before storing. They
        arrive padded to whatever width their batch happened to have, so two peptides of
        the same length collected from different batches would otherwise be stored at
        different widths and fail to stack on draw -- while the buffer key claims they are
        interchangeable. Trimming makes the key true: every entry under length L has
        exactly L residue rows.

        Per-example tensors ("glob") are stored whole. Trimming those by length would be
        silent corruption, which is why the residue keys are named explicitly rather than
        inferred from a shape.
        """
        from proteinfoundation.cp2lp.discriminator import RESIDUE_FEATURE_KEYS

        b = lengths.shape[0]
        for i in range(b):
            L = int(lengths[i])
            entry = {}
            for k, v in feats.items():
                t = v[i].detach().to("cpu")
                entry[k] = t[:L] if k in RESIDUE_FEATURE_KEYS else t
            self.buffers[L].append(entry)

    def draw_matched(
        self, lengths: torch.Tensor, device: torch.device, max_length_slack: int = 0
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor]:
        """Draws one real example per requested length.

        Args:
            lengths: ``[b]`` peptide lengths to match.
            device: where to put the assembled batch.
            max_length_slack: if no exact-length real exists, accept one within this many
                residues. 0 means exact matching only. Any slack is a crack the
                discriminator can read length through, so it is opt-in and logged by the
                caller through the returned mask.

        Returns:
            ``(feats, matched)`` where ``matched`` is a ``[b]`` bool mask of which
            requests were satisfied, and ``feats`` is the collated batch of the satisfied
            ones (``None`` when nothing matched).
        """
        b = lengths.shape[0]
        matched = torch.zeros(b, dtype=torch.bool)
        picked: list[dict[str, torch.Tensor]] = []
        available = self.lengths()
        for i in range(b):
            want = int(lengths[i])
            pool = self.buffers.get(want)
            if not pool and max_length_slack > 0:
                near = [ln for ln in available if abs(ln - want) <= max_length_slack and self.buffers[ln]]
                if near:
                    pool = self.buffers[min(near, key=lambda ln: (abs(ln - want), ln))]
            if pool:
                picked.append(self.rng.choice(list(pool)))
                matched[i] = True
        if not picked:
            return None, matched
        # Exact-length matching (the default) makes every entry the same width, so a plain
        # stack works. Slack lets entries of different lengths into one draw, which needs
        # padding back to a common width.
        widths = {int(p["mask"].shape[0]) for p in picked}
        feats = pad_feature_dicts(picked) if len(widths) > 1 else {
            k: torch.stack([p[k] for p in picked]) for k in picked[0]
        }
        return {k: v.to(device) for k, v in feats.items()}, matched


def pad_feature_dicts(entries: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pads per-example feature dicts to a common residue count.

    Only used when a draw spans lengths (``max_length_slack > 0``); exact-length draws need
    no padding, which is the normal path. Only PER-RESIDUE keys are padded -- padding
    ``glob`` would append zeros to a fixed-width per-example vector and change its meaning.
    """
    from proteinfoundation.cp2lp.discriminator import RESIDUE_FEATURE_KEYS

    n = max(e["mask"].shape[0] for e in entries)
    out: dict[str, list[torch.Tensor]] = defaultdict(list)
    for e in entries:
        pad = n - e["mask"].shape[0]
        for k, v in e.items():
            if pad > 0 and k in RESIDUE_FEATURE_KEYS:
                shape = list(v.shape)
                shape[0] = pad
                v = torch.cat([v, torch.zeros(*shape, dtype=v.dtype)], dim=0)
            out[k].append(v)
    return {k: torch.stack(v) for k, v in out.items()}


# ---------------------------------------------------------------- family splitting


def family_of(row: dict, family_column: str = "cluster_id") -> str:
    """Receptor-family key for a metadata row, with a documented fallback chain."""
    for col in (family_column, "cluster_id", "receptor_cluster", "pdb_id"):
        v = row.get(col)
        if v is not None and str(v) != "" and str(v).lower() != "nan":
            return str(v)
    # Last resort: the leading token of the example id, which for both CPSea and PepBench
    # is the source PDB accession. Coarser than a sequence cluster, never finer.
    return str(row.get("example_id", "")).split("_")[0]


def in_eval_split(family: str, eval_frac: float, seed: int = 0) -> bool:
    """Deterministic family-level assignment. No shuffle, no stored id list.

    A hash rather than a random draw so the answer for a given family is the same in
    every process, on every rerun, and stays the same when new rows are added to the
    metadata -- which a shuffle-based split does not.
    """
    h = hashlib.sha256(f"{seed}:{family}".encode()).digest()
    return (int.from_bytes(h[:8], "big") / 2**64) < eval_frac


def split_metadata_by_family(
    df,
    eval_frac: float = 0.15,
    seed: int = 0,
    family_column: str = "cluster_id",
):
    """Splits a metadata frame into ``(train_df, eval_df)`` with disjoint receptor families.

    Operates on a pandas frame so it can be used both by a preprocessing script and by a
    quick interactive check. Returns views, not copies of the underlying arrays.
    """
    families = df.apply(lambda r: family_of(r.to_dict(), family_column), axis=1)
    is_eval = families.map(lambda f: in_eval_split(f, eval_frac, seed))
    train_df, eval_df = df[~is_eval], df[is_eval]
    n_fam_train = families[~is_eval].nunique()
    n_fam_eval = families[is_eval].nunique()
    overlap = set(families[~is_eval]) & set(families[is_eval])
    if overlap:
        raise AssertionError(f"family split leaked {len(overlap)} families across train/eval")
    logger.info(
        f"CP2LP family split: train {len(train_df)} rows / {n_fam_train} families, "
        f"eval {len(eval_df)} rows / {n_fam_eval} families (eval_frac={eval_frac}, seed={seed})"
    )
    return train_df, eval_df
