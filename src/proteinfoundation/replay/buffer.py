"""Disk-backed FIFO replay buffer of generated terminal flow-space endpoints.

Stores exactly the fields Stage 3 of the reward-weighted-replay plan calls
for: `target_or_dataset_id`, `linkage_type`, `linkage_sites`, `peptide_length`,
`x1_ca`, `z1`, `binder_mask`, `raw_reward`, `reward_components`,
`collector_checkpoint`, `reward_version`, `random_seed`, plus a `cluster_id`
(the condition-granularity decision recorded in the implementation plan: near-
duplicate receptors are grouped by cluster rather than exact receptor id, so
balanced sampling/dominance diagnostics aren't fooled by redundant structures).

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
import random
import statistics
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

    def __len__(self) -> int:
        return len(self._entries)

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
        return entry

    def append(self, entries: list[dict]) -> None:
        """Appends entries (each a dict with `REQUIRED_ENTRY_FIELDS`), evicting oldest-first at `max_size`."""
        for entry in entries:
            self._entries.append(self._validate_entry(entry))
        while len(self._entries) > self.max_size:
            self._entries.popleft()

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
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        for old_shard in dir.glob("shard_*.pt"):
            old_shard.unlink()

        entries = list(self._entries)
        shard_files = []
        for start in range(0, len(entries), self.shard_size):
            shard = entries[start : start + self.shard_size]
            shard_name = f"shard_{start // self.shard_size:06d}.pt"
            torch.save(shard, dir / shard_name)
            shard_files.append(shard_name)

        manifest = {
            "max_size": self.max_size,
            "shard_size": self.shard_size,
            "length_bin_edges": list(self.length_bin_edges),
            "reward_version": self.reward_version,
            "n_entries": len(entries),
            "shard_files": shard_files,
        }
        with open(dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

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
        return buffer
