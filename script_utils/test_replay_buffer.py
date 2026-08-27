#!/usr/bin/env python3
"""Unit tests for `proteinfoundation.replay.buffer.ReplayBuffer`.

Pure CPU tests: no GPU, no dataset, no model required.

Usage (as a standalone script):
    python script_utils/test_replay_buffer.py

Usage (via pytest, if installed):
    pytest script_utils/test_replay_buffer.py -v
"""

from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from proteinfoundation.replay.buffer import ReplayBuffer, length_bin


def _make_entry(
    idx: int,
    linkage_type: int = 0,
    peptide_length: int = 10,
    cluster_id: str = "clusterA",
    reward_version: str = "v1",
    checkpoint: str = "ckpt_step_1000",
    seed: int = 0,
    latent_dim: int = 4,
) -> dict:
    L = peptide_length
    return {
        "target_or_dataset_id": f"cpsea_target_{idx}",
        "cluster_id": cluster_id,
        "linkage_type": linkage_type,
        "linkage_sites": (0, L - 1),
        "peptide_length": L,
        "x1_ca": torch.randn(L, 3, dtype=torch.float16),
        "z1": torch.randn(L, latent_dim, dtype=torch.float16),
        "binder_mask": torch.ones(L, dtype=torch.bool),
        "raw_reward": 0.5 + 0.01 * idx,
        "reward_components": {"distance_error": 0.1, "success": True},
        "collector_checkpoint": checkpoint,
        "reward_version": reward_version,
        "random_seed": seed,
    }


# ---------------------------------------------------------------------------
# A. FIFO eviction
# ---------------------------------------------------------------------------
def test_fifo_eviction_drops_oldest_first():
    buf = ReplayBuffer(max_size=5)
    entries = [_make_entry(i) for i in range(8)]
    buf.append(entries)
    assert len(buf) == 5
    remaining_ids = [e["target_or_dataset_id"] for e in buf._entries]
    assert remaining_ids == [f"cpsea_target_{i}" for i in range(3, 8)]


def test_fifo_eviction_incremental_appends():
    buf = ReplayBuffer(max_size=3)
    for i in range(6):
        buf.append([_make_entry(i)])
    assert len(buf) == 3
    remaining_ids = [e["target_or_dataset_id"] for e in buf._entries]
    assert remaining_ids == ["cpsea_target_3", "cpsea_target_4", "cpsea_target_5"]


# ---------------------------------------------------------------------------
# B. Save/load round trip is lossless
# ---------------------------------------------------------------------------
def test_save_load_round_trip_lossless():
    tmpdir = tempfile.mkdtemp()
    try:
        buf = ReplayBuffer(max_size=100, shard_size=4)
        entries = [_make_entry(i, peptide_length=8 + i % 5) for i in range(11)]
        buf.append(entries)

        buf.save(tmpdir)
        loaded = ReplayBuffer.load(tmpdir)

        assert len(loaded) == len(buf)
        assert loaded.reward_version == buf.reward_version
        for orig, rt in zip(buf._entries, loaded._entries):
            assert orig["target_or_dataset_id"] == rt["target_or_dataset_id"]
            assert orig["cluster_id"] == rt["cluster_id"]
            assert orig["linkage_type"] == rt["linkage_type"]
            assert orig["linkage_sites"] == rt["linkage_sites"]
            assert orig["peptide_length"] == rt["peptide_length"]
            assert torch.equal(orig["x1_ca"], rt["x1_ca"])
            assert torch.equal(orig["z1"], rt["z1"])
            assert torch.equal(orig["binder_mask"], rt["binder_mask"])
            assert orig["raw_reward"] == rt["raw_reward"]
            assert orig["reward_components"] == rt["reward_components"]
            assert orig["collector_checkpoint"] == rt["collector_checkpoint"]
            assert orig["reward_version"] == rt["reward_version"]
            assert orig["random_seed"] == rt["random_seed"]
    finally:
        shutil.rmtree(tmpdir)


def test_save_load_recovers_from_interrupted_run():
    """Loading a saved buffer mid-run should let a new buffer resume appending."""
    tmpdir = tempfile.mkdtemp()
    try:
        buf = ReplayBuffer(max_size=100, shard_size=4)
        buf.append([_make_entry(i) for i in range(5)])
        buf.save(tmpdir)

        recovered = ReplayBuffer.load(tmpdir)
        recovered.append([_make_entry(i) for i in range(5, 9)])
        assert len(recovered) == 9
    finally:
        shutil.rmtree(tmpdir)


def test_save_interrupted_before_manifest_swap_leaves_old_snapshot_intact():
    """Regression: manifest.json is the sole atomic commit point. A crash before
    its `os.replace` must leave the previous snapshot exactly as it was --
    including its shard files, which a naive per-file-rename promotion could
    otherwise leave partially overwritten under reused filenames."""
    tmpdir = tempfile.mkdtemp()
    try:
        buf = ReplayBuffer(max_size=100, shard_size=4)
        buf.append([_make_entry(i) for i in range(5)])
        buf.save(tmpdir)
        old_shard_names = sorted(p.name for p in Path(tmpdir).glob("shard_*.pt"))
        assert old_shard_names, "expected at least one shard file after the first save"

        buf2 = ReplayBuffer(max_size=100, shard_size=4)
        buf2.append([_make_entry(i) for i in range(20, 28)])  # different count/content

        with patch(
            "proteinfoundation.replay.buffer.os.replace",
            side_effect=OSError("simulated crash before manifest swap"),
        ):
            try:
                buf2.save(tmpdir)
                assert False, "expected the simulated crash to propagate"
            except OSError:
                pass

        # Old shard files must be untouched (never overwritten in place -- the new
        # generation used fresh filenames) and the old manifest must still resolve
        # to exactly the old snapshot.
        for name in old_shard_names:
            assert (Path(tmpdir) / name).exists()
        recovered = ReplayBuffer.load(tmpdir)
        assert len(recovered) == 5
        assert [e["target_or_dataset_id"] for e in recovered._entries] == [
            f"cpsea_target_{i}" for i in range(5)
        ]
    finally:
        shutil.rmtree(tmpdir)


def test_load_reward_version_mismatch_raises():
    tmpdir = tempfile.mkdtemp()
    try:
        buf = ReplayBuffer(max_size=10)
        buf.append([_make_entry(0, reward_version="v1")])
        buf.save(tmpdir)
        try:
            ReplayBuffer.load(tmpdir, expected_reward_version="v2")
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmpdir)


def test_append_reward_version_mismatch_raises():
    buf = ReplayBuffer(max_size=10)
    buf.append([_make_entry(0, reward_version="v1")])
    try:
        buf.append([_make_entry(1, reward_version="v2")])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_append_missing_field_raises():
    buf = ReplayBuffer(max_size=10)
    bad_entry = _make_entry(0)
    del bad_entry["z1"]
    try:
        buf.append([bad_entry])
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# C. Balanced sampling does not starve rare strata
# ---------------------------------------------------------------------------
def test_balanced_sampling_does_not_starve_rare_linkage_type():
    buf = ReplayBuffer(max_size=1000)
    entries = [_make_entry(i, linkage_type=0) for i in range(90)]
    entries += [_make_entry(i, linkage_type=1) for i in range(90, 95)]  # rare type
    entries += [_make_entry(i, linkage_type=2) for i in range(95, 100)]
    buf.append(entries)

    rng = random.Random(0)
    counts = {0: 0, 1: 0, 2: 0}
    n_draws = 3000
    sampled = buf.sample_balanced(n_draws, by=("linkage_type",), rng=rng)
    for e in sampled:
        counts[e["linkage_type"]] += 1

    assert all(c > 0 for c in counts.values()), counts
    # Roughly uniform across the 3 strata (each ~1000 of 3000 draws), not
    # proportional to the 90/5/5 raw entry counts.
    for c in counts.values():
        assert n_draws / 3 * 0.5 < c < n_draws / 3 * 1.5, counts


def test_balanced_sampling_length_bin_stratification():
    buf = ReplayBuffer(max_size=1000)
    entries = [_make_entry(i, peptide_length=6) for i in range(50)]
    entries += [_make_entry(i, peptide_length=25) for i in range(50, 52)]  # rare, long
    buf.append(entries)

    rng = random.Random(1)
    sampled = buf.sample_balanced(1000, by=("length_bin",), rng=rng)
    bins_seen = {length_bin(e["peptide_length"]) for e in sampled}
    assert len(bins_seen) == 2


# ---------------------------------------------------------------------------
# D. Padding/masks survive round trips for variable-length entries collated together
# ---------------------------------------------------------------------------
def test_variable_length_entries_survive_round_trip_when_collated():
    tmpdir = tempfile.mkdtemp()
    try:
        buf = ReplayBuffer(max_size=100, shard_size=2)
        lengths = [6, 9, 14]
        entries = [_make_entry(i, peptide_length=lengths[i]) for i in range(len(lengths))]
        buf.append(entries)
        buf.save(tmpdir)
        loaded = ReplayBuffer.load(tmpdir)

        max_len = max(lengths)
        for orig_len, entry in zip(lengths, loaded._entries):
            x1_ca = entry["x1_ca"]
            mask = entry["binder_mask"]
            assert x1_ca.shape[0] == orig_len
            assert mask.shape[0] == orig_len
            assert mask.all()
            # Simulate downstream collation padding and check it round-trips cleanly.
            pad = max_len - orig_len
            if pad > 0:
                padded = torch.nn.functional.pad(x1_ca, (0, 0, 0, pad))
                padded_mask = torch.nn.functional.pad(mask, (0, pad), value=False)
                assert padded.shape[0] == max_len
                assert padded_mask[orig_len:].sum() == 0
                assert padded_mask[:orig_len].all()
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# E. Receptor-conditioning side table
# ---------------------------------------------------------------------------
def test_receptor_conditions_survive_save_load_round_trip():
    tmpdir = tempfile.mkdtemp()
    try:
        buf = ReplayBuffer(max_size=100, shard_size=4)
        entries = [_make_entry(i) for i in range(3)]
        buf.append(entries)
        buf.add_receptor_conditions(
            {
                f"cpsea_target_{i}": {
                    "x_target": torch.randn(5, 37, 3),
                    "target_mask": torch.ones(5, 37, dtype=torch.bool),
                }
                for i in range(3)
            }
        )
        buf.save(tmpdir)
        loaded = ReplayBuffer.load(tmpdir)

        assert set(loaded.receptor_conditions.keys()) == {f"cpsea_target_{i}" for i in range(3)}
        for i in range(3):
            orig = buf.receptor_conditions[f"cpsea_target_{i}"]
            rt = loaded.receptor_conditions[f"cpsea_target_{i}"]
            assert torch.equal(orig["x_target"], rt["x_target"])
            assert torch.equal(orig["target_mask"], rt["target_mask"])
    finally:
        shutil.rmtree(tmpdir)


def test_receptor_conditions_pruned_on_eviction():
    """A receptor condition with no referencing entry left in the buffer must be dropped,
    so the side table does not grow unboundedly as FIFO eviction churns through entries."""
    buf = ReplayBuffer(max_size=3)
    buf.append([_make_entry(i) for i in range(3)])
    buf.add_receptor_conditions({f"cpsea_target_{i}": {"x_target": torch.randn(2, 37, 3)} for i in range(3)})
    assert set(buf.receptor_conditions.keys()) == {f"cpsea_target_{i}" for i in range(3)}

    # Evicts cpsea_target_0 (oldest-first FIFO at max_size=3).
    buf.append([_make_entry(3)])
    buf.add_receptor_conditions({"cpsea_target_3": {"x_target": torch.randn(2, 37, 3)}})
    assert "cpsea_target_0" not in buf.receptor_conditions
    assert set(buf.receptor_conditions.keys()) == {f"cpsea_target_{i}" for i in range(1, 4)}


# ---------------------------------------------------------------------------
# F. Group-preserving sampling for geocycler_group_relative weighting
# ---------------------------------------------------------------------------
def _make_grouped_entries(n_groups: int, group_size: int, linkage_type: int = 0) -> list[dict]:
    """`n_groups` groups of `group_size` entries each, all sharing one `target_or_dataset_id` per group
    (as the collector writes for the K candidates generated from one receptor)."""
    entries = []
    for g in range(n_groups):
        for m in range(group_size):
            e = _make_entry(g * group_size + m, linkage_type=linkage_type)
            e["target_or_dataset_id"] = f"receptor_{g}"
            entries.append(e)
    return entries


def test_sample_grouped_keeps_groups_intact():
    buf = ReplayBuffer(max_size=1000)
    buf.append(_make_grouped_entries(n_groups=5, group_size=4))

    rng = random.Random(0)
    sampled = buf.sample_grouped(batch_size=100, group_size=4, by=("linkage_type",), rng=rng)
    assert len(sampled) == 100

    from collections import Counter

    counts = Counter(e["target_or_dataset_id"] for e in sampled)
    # Every drawn receptor should contribute more than one candidate in a 100-entry
    # draw across only 5 groups -- i.i.d. `sample_balanced` would instead scatter
    # draws near-uniformly across all 20 entries, mostly landing singletons.
    assert all(c >= 2 for c in counts.values()), counts


def test_sample_grouped_excludes_singleton_groups():
    buf = ReplayBuffer(max_size=1000)
    entries = _make_grouped_entries(n_groups=3, group_size=1)  # every group has exactly 1 member
    buf.append(entries)
    try:
        buf.sample_grouped(batch_size=10, group_size=4)
        assert False, "expected ValueError: no group has >= 2 candidates"
    except ValueError:
        pass


def test_sample_grouped_caps_at_group_size():
    buf = ReplayBuffer(max_size=1000)
    buf.append(_make_grouped_entries(n_groups=1, group_size=10))
    rng = random.Random(0)
    sampled = buf.sample_grouped(batch_size=3, group_size=3, rng=rng)
    assert len(sampled) == 3


ALL_TESTS = [
    test_fifo_eviction_drops_oldest_first,
    test_fifo_eviction_incremental_appends,
    test_receptor_conditions_survive_save_load_round_trip,
    test_receptor_conditions_pruned_on_eviction,
    test_sample_grouped_keeps_groups_intact,
    test_sample_grouped_excludes_singleton_groups,
    test_sample_grouped_caps_at_group_size,
    test_save_load_round_trip_lossless,
    test_save_load_recovers_from_interrupted_run,
    test_save_interrupted_before_manifest_swap_leaves_old_snapshot_intact,
    test_load_reward_version_mismatch_raises,
    test_append_reward_version_mismatch_raises,
    test_append_missing_field_raises,
    test_balanced_sampling_does_not_starve_rare_linkage_type,
    test_balanced_sampling_length_bin_stratification,
    test_variable_length_entries_survive_round_trip_when_collated,
]


if __name__ == "__main__":
    failures = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  OK {test_fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(test_fn.__name__)
            print(f"  FAIL {test_fn.__name__}: {e}")

    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
