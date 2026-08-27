"""Disk-backed FIFO replay buffer of generated terminal flow-space endpoints.

Stores exactly the fields Stage 3 of the reward-weighted-replay plan calls
for: `target_or_dataset_id`, `linkage_type`, `linkage_sites`, `peptide_length`,
`x1_ca`, `z1`, `binder_mask`, `raw_reward`, `reward_components`,
`collector_checkpoint`, `reward_version`, `random_seed`, plus a `cluster_id`
(the condition-granularity decision recorded in the implementation plan: near-
duplicate receptors are grouped by cluster rather than exact receptor id, so
balanced sampling/dominance diagnostics aren't fooled by redundant structures).

Also stores `receptor_conditions`: a `target_or_dataset_id -> {feature_name:
tensor}` side table of the receptor-conditioning features (every batch key
whose name contains "target", e.g. `x_target`/`seq_target`/`target_mask`/
`seq_target_mask`/`target_hotspot_mask`) that CPSea's `enable_target: true`
config needs at training time. This is a side table rather than a per-entry
field so the K candidates collected for one receptor share a single stored
copy instead of duplicating (typically much larger) receptor tensors K times.
Entries only carry the join key (`target_or_dataset_id`); `ReplayMixer`
resolves it back to the actual tensors at draw time.

Design tradeoff, stated explicitly: entries are held as an in-memory FIFO
deque (not lazily memory-mapped per-shard), with `save`/`load` doing whole
sharded-file snapshots. At the default `max_size=10_000` and typical CPSea
peptide lengths/latent dims this is a few hundred MB at most -- simple and
exactly testable. Revisit with true lazy paging only if `max_size` grows by
an order of magnitude or more.
"""

from __future__ import annotations

import collections
import json
import os
import random
import statistics
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch

REQUIRED_ENTRY_FIELDS = (
    "target_or_dataset_id",
    "cluster_id",
    "linkage_type",
    "linkage_sites",
    "peptide_length",
    "x1_ca",
    "z1",
    "binder_mask",
    "raw_reward",
    "reward_components",
    "collector_checkpoint",
    "reward_version",
    "random_seed",
)

DEFAULT_LENGTH_BIN_EDGES = (0, 8, 12, 16, 20, 10_000)


def _detach_cpu(value):
    """Recursively detaches tensors and moves them to CPU before storage.

    Entries live in an in-memory deque across training steps (see module
    docstring), so anything still attached to the live computation graph or
    parked on the GPU would silently retain autograd history and device
    memory for as long as it sits in the buffer -- collectors must not have
    to remember to do this themselves.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _detach_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cast = [_detach_cpu(v) for v in value]
        return type(value)(cast) if isinstance(value, tuple) else cast
    return value


def length_bin(peptide_length: int, edges=DEFAULT_LENGTH_BIN_EDGES) -> int:
    """Returns the index of the half-open bin `[edges[k], edges[k+1])` containing `peptide_length`."""
    for k in range(len(edges) - 1):
        if edges[k] <= peptide_length < edges[k + 1]:
            return k
    return len(edges) - 2


@dataclass
class ReplayBufferStats:
    n_entries: int
    by_linkage_type: dict
    by_length_bin: dict
    by_cluster_id: dict
    reward_mean: float
    reward_min: float
    reward_max: float
    checkpoint_counts: dict
    reward_version: str | None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class ReplayBuffer:
    """FIFO, disk-backed replay buffer of terminal generated endpoints.

    Args:
        max_size: Maximum number of entries retained; oldest entries are
            evicted first once exceeded (append-time FIFO eviction).
        shard_size: Number of entries per on-disk shard file at `save()` time.
        length_bin_edges: Bin edges (see `length_bin`) used by
            `sample_balanced`'s `"length_bin"` stratification field and by
            `stats()`.
    """

    def __init__(
        self,
        max_size: int = 10_000,
        shard_size: int = 256,
        length_bin_edges=DEFAULT_LENGTH_BIN_EDGES,
    ):
        self.max_size = max_size
        self.shard_size = shard_size
        self.length_bin_edges = tuple(length_bin_edges)
        self._entries: collections.deque[dict] = collections.deque()
        self.reward_version: str | None = None
        # target_or_dataset_id -> {feature_name: CPU tensor}, see module docstring.
        self.receptor_conditions: dict[str, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Receptor conditioning side table
    # ------------------------------------------------------------------
    def add_receptor_conditions(self, conditions_by_id: dict[str, dict]) -> None:
        """Stores (or overwrites) the receptor-conditioning tensors for each id.

        Called once per receptor at collection time (not once per candidate),
        which is what keeps this a side table rather than a per-entry field.
        """
        for target_id, cond in conditions_by_id.items():
            self.receptor_conditions[target_id] = {k: _detach_cpu(v) for k, v in cond.items()}

    def _prune_unreferenced_receptor_conditions(self) -> None:
        referenced = {e["target_or_dataset_id"] for e in self._entries}
        for stale_id in [k for k in self.receptor_conditions if k not in referenced]:
            del self.receptor_conditions[stale_id]

    # ------------------------------------------------------------------
    # Append / eviction
    # ------------------------------------------------------------------
    def _validate_entry(self, entry: dict) -> dict:
        missing = [f for f in REQUIRED_ENTRY_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"replay entry missing required field(s): {missing}")
        if self.reward_version is None:
            self.reward_version = entry["reward_version"]
        elif entry["reward_version"] != self.reward_version:
            raise ValueError(
                "reward_version mismatch: buffer was built with "
                f"'{self.reward_version}' but entry carries "
                f"'{entry['reward_version']}'. Scores from different scorer "
                "versions cannot be mixed in one buffer -- start a fresh "
                "buffer (or rescore) instead."
            )
        return {k: _detach_cpu(v) for k, v in entry.items()}

    def append(self, entries: list[dict]) -> None:
        """Appends entries (each a dict with `REQUIRED_ENTRY_FIELDS`), evicting oldest-first at `max_size`."""
        for entry in entries:
            self._entries.append(self._validate_entry(entry))
        while len(self._entries) > self.max_size:
            self._entries.popleft()
        self._prune_unreferenced_receptor_conditions()

    # ------------------------------------------------------------------
    # Stratified/balanced sampling
    # ------------------------------------------------------------------
    def _stratum_key(self, entry: dict, by: tuple[str, ...]):
        parts = []
        for f in by:
            if f == "linkage_type":
                parts.append(("linkage_type", int(entry["linkage_type"])))
            elif f == "length_bin":
                parts.append(
                    ("length_bin", length_bin(int(entry["peptide_length"]), self.length_bin_edges))
                )
            elif f in ("cluster_id", "target"):
                parts.append((f, entry["cluster_id"]))
            else:
                raise ValueError(f"unknown stratification field: {f!r}")
        return tuple(parts)

    def sample_balanced(
        self,
        batch_size: int,
        by: tuple[str, ...] = ("linkage_type", "length_bin"),
        rng: random.Random | None = None,
    ) -> list[dict]:
        """Draws `batch_size` entries with (approximately) equal probability per stratum.

        Each draw first picks a stratum uniformly at random among those
        present in the buffer, then an entry uniformly within that stratum.
        This is what keeps a rare linkage type or length bin from being
        starved by a common one dominating raw entry counts (contrast with
        i.i.d. sampling over entries, which would reproduce the buffer's own
        imbalance).
        """
        if len(self._entries) == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        rng = rng or random

        strata: dict[tuple, list[int]] = collections.defaultdict(list)
        for idx, entry in enumerate(self._entries):
            strata[self._stratum_key(entry, by)].append(idx)
        stratum_keys = list(strata.keys())

        sampled = []
        for _ in range(batch_size):
            key = rng.choice(stratum_keys)
            idx = rng.choice(strata[key])
            sampled.append(self._entries[idx])
        return sampled

    def sample_grouped(
        self,
        batch_size: int,
        group_size: int,
        by: tuple[str, ...] = ("linkage_type", "length_bin"),
        rng: random.Random | None = None,
    ) -> list[dict]:
        """Draws entries as intact `target_or_dataset_id` groups, for group-relative weighting.

        `sample_balanced` draws each entry independently, so a 16-entry batch
        almost always lands one candidate per `(cluster_id, linkage_type)`
        group -- zero reward variance, zero weight under
        `geocycler_group_relative_weights`. This instead samples whole
        candidate groups (all K candidates generated for one receptor share
        one `target_or_dataset_id`, the exact identity `_collate` uses for
        `group_ids`), so within-group reward variance is actually available
        for the group-relative weighting to use.

        Groups with fewer than 2 members are excluded (a singleton group has
        zero possible reward variance by construction, so including it would
        just reproduce the bug this method exists to fix). Each selected
        group contributes up to `group_size` of its members (all of them, if
        it has fewer); stratum selection uses one representative entry per
        group, which is safe because every candidate in a group shares the
        same native cyclization label and generation length (see
        `scripts/collect_cpsea_replay_rollouts.py`'s per-receptor repeat).
        """
        if len(self._entries) == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        rng = rng or random

        groups: dict[str, list[int]] = collections.defaultdict(list)
        for idx, entry in enumerate(self._entries):
            groups[entry["target_or_dataset_id"]].append(idx)
        eligible = {gid: idxs for gid, idxs in groups.items() if len(idxs) >= 2}
        if not eligible:
            raise ValueError(
                "no replay group has >= 2 candidates -- cannot draw group-relative "
                "batches. This buffer has only singleton target_or_dataset_id "
                "groups (check the collector wrote K > 1 candidates per receptor)."
            )

        strata: dict[tuple, list[str]] = collections.defaultdict(list)
        for gid, idxs in eligible.items():
            strata[self._stratum_key(self._entries[idxs[0]], by)].append(gid)
        stratum_keys = list(strata.keys())

        sampled: list[dict] = []
        while len(sampled) < batch_size:
            key = rng.choice(stratum_keys)
            gid = rng.choice(strata[key])
            idxs = eligible[gid]
            take = idxs if len(idxs) <= group_size else rng.sample(idxs, group_size)
            sampled.extend(self._entries[idx] for idx in take)
        return sampled[:batch_size]

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> ReplayBufferStats:
        entries = list(self._entries)
        rewards = [float(e["raw_reward"]) for e in entries]
        return ReplayBufferStats(
            n_entries=len(entries),
            by_linkage_type=dict(collections.Counter(int(e["linkage_type"]) for e in entries)),
            by_length_bin=dict(
                collections.Counter(
                    length_bin(int(e["peptide_length"]), self.length_bin_edges) for e in entries
                )
            ),
            by_cluster_id=dict(collections.Counter(e["cluster_id"] for e in entries)),
            reward_mean=statistics.fmean(rewards) if rewards else float("nan"),
            reward_min=min(rewards) if rewards else float("nan"),
            reward_max=max(rewards) if rewards else float("nan"),
            checkpoint_counts=dict(collections.Counter(e["collector_checkpoint"] for e in entries)),
            reward_version=self.reward_version,
        )

    # ------------------------------------------------------------------
    # Save / load (sharded, lossless round trip)
    # ------------------------------------------------------------------
    def save(self, dir: str | Path) -> None:
        """Writes a new snapshot with `manifest.json` as the sole atomic commit point.

        Each call writes its shards under a filename unique to this save --
        `shard_<generation>_<index>.pt`, never reusing a previous save's shard
        filenames -- so every shard byte the new manifest could possibly reference
        is fully written to disk *before* that manifest exists at all. Publishing
        is then a single `os.replace` of `manifest.json` (atomic on a POSIX
        same-filesystem rename): a reader (or a crash) can only ever see the
        complete old manifest pointing at the complete old shards, or the complete
        new manifest pointing at the complete new shards -- never a manifest whose
        shard list is partly old and partly new content under reused filenames,
        which a naive per-file-rename promotion (this function's previous
        implementation) could produce if interrupted between renaming a shard and
        renaming the manifest.

        Stale shards from the previous generation are deleted only after the
        manifest swap, as best-effort cleanup: `load()` never looks at a file the
        current manifest doesn't name, so a leftover orphan from an interrupted
        cleanup is inert, not a correctness issue.
        """
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)

        previous_manifest_path = dir / "manifest.json"
        previous_shard_files: set[str] = set()
        previous_receptor_conditions_file: str | None = None
        if previous_manifest_path.exists():
            with open(previous_manifest_path) as f:
                previous_manifest = json.load(f)
            previous_shard_files = set(previous_manifest.get("shard_files", []))
            previous_receptor_conditions_file = previous_manifest.get("receptor_conditions_file")

        generation = uuid.uuid4().hex[:12]
        entries = list(self._entries)
        shard_files = []
        for start in range(0, len(entries), self.shard_size):
            shard = entries[start : start + self.shard_size]
            shard_name = f"shard_{generation}_{start // self.shard_size:06d}.pt"
            torch.save(shard, dir / shard_name)
            shard_files.append(shard_name)

        receptor_conditions_file = f"receptor_conditions_{generation}.pt"
        torch.save(self.receptor_conditions, dir / receptor_conditions_file)

        manifest = {
            "max_size": self.max_size,
            "shard_size": self.shard_size,
            "length_bin_edges": list(self.length_bin_edges),
            "reward_version": self.reward_version,
            "n_entries": len(entries),
            "shard_files": shard_files,
            "receptor_conditions_file": receptor_conditions_file,
        }
        tmp_manifest = dir / f".tmp-{os.getpid()}-{generation}-manifest.json"
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_manifest, previous_manifest_path)

        for stale_shard in previous_shard_files - set(shard_files):
            try:
                (dir / stale_shard).unlink()
            except OSError:
                pass
        if previous_receptor_conditions_file and previous_receptor_conditions_file != receptor_conditions_file:
            try:
                (dir / previous_receptor_conditions_file).unlink()
            except OSError:
                pass

    @classmethod
    def load(cls, dir: str | Path, expected_reward_version: str | None = None) -> "ReplayBuffer":
        """Loads a buffer saved by `save()`, e.g. to recover an interrupted run.

        If `expected_reward_version` is given, raises if it disagrees with the
        manifest's recorded version -- this is the "reward-version validation"
        the plan asks for at the load boundary (append-time validation is
        handled separately by `append`/`_validate_entry`).
        """
        dir = Path(dir)
        with open(dir / "manifest.json") as f:
            manifest = json.load(f)

        if (
            expected_reward_version is not None
            and manifest["reward_version"] is not None
            and manifest["reward_version"] != expected_reward_version
        ):
            raise ValueError(
                f"reward_version mismatch on load: buffer on disk was scored "
                f"with '{manifest['reward_version']}', caller expected "
                f"'{expected_reward_version}'."
            )

        buffer = cls(
            max_size=manifest["max_size"],
            shard_size=manifest["shard_size"],
            length_bin_edges=tuple(manifest["length_bin_edges"]),
        )
        buffer.reward_version = manifest["reward_version"]
        for shard_name in manifest["shard_files"]:
            shard = torch.load(dir / shard_name, weights_only=False)
            buffer._entries.extend(shard)

        receptor_conditions_file = manifest.get("receptor_conditions_file")
        if receptor_conditions_file:
            buffer.receptor_conditions = torch.load(dir / receptor_conditions_file, weights_only=False)
        return buffer
