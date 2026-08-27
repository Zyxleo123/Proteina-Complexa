#!/usr/bin/env python3
"""Unit tests for `proteinfoundation.training.replay_mixer.ReplayMixer`.

Pure CPU tests: a tiny real `ProductSpaceFlowMatcher` + toy NN (same pattern as
`test_flow_loss_shared_fn.py`), no full `Proteina`/dataset/GPU.

Usage:
    python script_utils/test_replay_mixer.py
    pytest script_utils/test_replay_mixer.py -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_flow_loss_shared_fn import BB_CA_DIM, LATENT_DIM, _FakeProteina, _make_cfg_exp, _TinyNN  # noqa: E402

from proteinfoundation.cyclization.constants import DISULFIDE, ISOPEPTIDE, MAINCHAIN
from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from proteinfoundation.replay.buffer import ReplayBuffer
from proteinfoundation.training.replay_mixer import ReplayMixer


def _make_entry(idx, linkage_type, peptide_length, reward, success, near_success, cluster_id="cA"):
    L = peptide_length
    return {
        "target_or_dataset_id": f"target_{idx}",
        "cluster_id": cluster_id,
        "linkage_type": linkage_type,
        "linkage_sites": (0, L - 1),
        "peptide_length": L,
        "x1_ca": torch.randn(L, BB_CA_DIM),
        "z1": torch.randn(L, LATENT_DIM),
        "binder_mask": torch.ones(L, dtype=torch.bool),
        "raw_reward": reward,
        "reward_components": {"success": success, "near_success": near_success},
        "collector_checkpoint": "ckpt_0",
        "reward_version": "v1",
        "random_seed": idx,
    }


def _make_receptor_condition(n_target=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x_target": torch.randn(n_target, 37, 3, generator=g),
        "target_mask": torch.ones(n_target, 37, dtype=torch.bool),
        "seq_target_mask": torch.ones(n_target, dtype=torch.bool),
    }


def _build_mixer_and_proteina(tmpdir, weighting_mode="success_only", with_receptor_conditions=True):
    torch.manual_seed(0)
    fm = ProductSpaceFlowMatcher(_make_cfg_exp())
    nn_module = _TinyNN()
    proteina = _FakeProteina(fm, nn_module)

    buf = ReplayBuffer(max_size=100)
    entries = [
        _make_entry(0, MAINCHAIN, 6, reward=1.0, success=True, near_success=False),
        _make_entry(1, DISULFIDE, 8, reward=0.0, success=False, near_success=False),
        _make_entry(2, ISOPEPTIDE, 10, reward=0.5, success=False, near_success=True),
    ]
    buf.append(entries)
    if with_receptor_conditions:
        buf.add_receptor_conditions(
            {f"target_{i}": _make_receptor_condition(n_target=3 + i, seed=i) for i in range(3)}
        )
    buf.save(tmpdir)

    mixer = ReplayMixer(buffer_dir=tmpdir, reward_version="v1", weighting_mode=weighting_mode)
    return mixer, proteina, nn_module


def test_empty_buffer_returns_none():
    tmpdir = tempfile.mkdtemp()
    try:
        mixer = ReplayMixer(buffer_dir=tmpdir, reward_version="v1")
        result = mixer.sample_replay_loss(proteina=None, batch_size=4, device=torch.device("cpu"))
        assert result is None
    finally:
        shutil.rmtree(tmpdir)


def test_zero_batch_size_returns_none():
    tmpdir = tempfile.mkdtemp()
    try:
        mixer, proteina, _ = _build_mixer_and_proteina(tmpdir)
        result = mixer.sample_replay_loss(proteina, batch_size=0, device=torch.device("cpu"))
        assert result is None
    finally:
        shutil.rmtree(tmpdir)


def test_sample_replay_loss_produces_finite_weighted_loss():
    tmpdir = tempfile.mkdtemp()
    try:
        mixer, proteina, _ = _build_mixer_and_proteina(tmpdir, weighting_mode="success_only")
        result = mixer.sample_replay_loss(proteina, batch_size=6, device=torch.device("cpu"))
        assert result is not None
        losses, diagnostics = result
        total = sum(torch.mean(v) for k, v in losses.items() if "_justlog" not in k)
        assert torch.isfinite(total)
        assert diagnostics["n_sampled"] == 6.0
    finally:
        shutil.rmtree(tmpdir)


def test_replay_loss_backprops_into_nn_params_not_stored_endpoint():
    tmpdir = tempfile.mkdtemp()
    try:
        mixer, proteina, nn_module = _build_mixer_and_proteina(tmpdir)
        nn_module.zero_grad()
        losses, _ = mixer.sample_replay_loss(proteina, batch_size=4, device=torch.device("cpu"))
        total = sum(torch.mean(v) for k, v in losses.items() if "_justlog" not in k)
        total.backward()
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in nn_module.parameters())
    finally:
        shutil.rmtree(tmpdir)


def test_all_zero_success_weights_give_success_only_zero_loss():
    """weighting_mode='success_only' with an all-failure sample should down-weight
    every example to exactly zero, since none succeeded or near-succeeded."""
    tmpdir = tempfile.mkdtemp()
    try:
        torch.manual_seed(0)
        fm = ProductSpaceFlowMatcher(_make_cfg_exp())
        nn_module = _TinyNN()
        proteina = _FakeProteina(fm, nn_module)

        buf = ReplayBuffer(max_size=10)
        buf.append(
            [_make_entry(i, MAINCHAIN, 6, reward=0.0, success=False, near_success=False) for i in range(3)]
        )
        buf.add_receptor_conditions(
            {f"target_{i}": _make_receptor_condition(n_target=4, seed=i) for i in range(3)}
        )
        buf.save(tmpdir)
        mixer = ReplayMixer(buffer_dir=tmpdir, reward_version="v1", weighting_mode="success_only")

        losses, _ = mixer.sample_replay_loss(proteina, batch_size=3, device=torch.device("cpu"))
        total = sum(torch.mean(v) for k, v in losses.items() if "_justlog" not in k)
        assert total.item() == 0.0
    finally:
        shutil.rmtree(tmpdir)


def test_geocycler_mode_gets_nonzero_weight_variance_via_grouped_sampling():
    """Regression for the sample_balanced grouping bug: with i.i.d. sampling over
    individual entries, a batch drawn from many small receptor groups almost always
    lands one candidate per group, so every `geocycler_group_relative` weight
    collapses to 0 (zero within-group reward std). `sample_replay_loss` must use
    `sample_grouped` for this mode so within-group variance is actually present."""
    tmpdir = tempfile.mkdtemp()
    try:
        torch.manual_seed(0)
        fm = ProductSpaceFlowMatcher(_make_cfg_exp())
        nn_module = _TinyNN()
        proteina = _FakeProteina(fm, nn_module)

        n_groups, group_size = 10, 6
        entries = []
        receptor_conditions = {}
        for g in range(n_groups):
            receptor_id = f"receptor_{g}"
            receptor_conditions[receptor_id] = _make_receptor_condition(n_target=4, seed=g)
            for m in range(group_size):
                e = _make_entry(g * group_size + m, MAINCHAIN, 6, reward=float(m), success=False, near_success=False)
                e["target_or_dataset_id"] = receptor_id
                entries.append(e)

        buf = ReplayBuffer(max_size=1000)
        buf.append(entries)
        buf.add_receptor_conditions(receptor_conditions)
        buf.save(tmpdir)

        mixer = ReplayMixer(
            buffer_dir=tmpdir, reward_version="v1", weighting_mode="geocycler_group_relative", group_size=group_size
        )
        _, diagnostics = mixer.sample_replay_loss(proteina, batch_size=60, device=torch.device("cpu"))
        assert diagnostics["weight_mean"] > 0.0, "all-zero weights: group-relative signal was not available"
    finally:
        shutil.rmtree(tmpdir)


def test_reward_version_mismatch_raises_on_load():
    tmpdir = tempfile.mkdtemp()
    try:
        buf = ReplayBuffer(max_size=10)
        buf.append([_make_entry(0, MAINCHAIN, 6, reward=1.0, success=True, near_success=False)])
        buf.reward_version = "v1"
        buf.save(tmpdir)
        try:
            ReplayMixer(buffer_dir=tmpdir, reward_version="v2")
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmpdir)


def test_collate_pads_and_threads_receptor_condition():
    """Regression for the receptor-dropping bug: `_collate`'s `condition` must carry
    each entry's actual `x_target`/`target_mask`, not leave them out entirely."""
    tmpdir = tempfile.mkdtemp()
    try:
        mixer, _, _ = _build_mixer_and_proteina(tmpdir)
        entries = list(mixer.buffer._entries)
        _, _, _, condition, *_ = mixer._collate(entries, torch.device("cpu"))

        assert "x_target" in condition and "target_mask" in condition
        max_n_target = max(
            mixer.buffer.receptor_conditions[e["target_or_dataset_id"]]["x_target"].shape[0] for e in entries
        )
        assert condition["x_target"].shape == (len(entries), max_n_target, 37, 3)
        for k, e in enumerate(entries):
            stored = mixer.buffer.receptor_conditions[e["target_or_dataset_id"]]["x_target"]
            n = stored.shape[0]
            assert torch.allclose(condition["x_target"][k, :n], stored, atol=1e-6)
            if n < max_n_target:
                assert condition["target_mask"][k, n:].sum() == 0  # padding reads as "not a target position"
    finally:
        shutil.rmtree(tmpdir)


def test_missing_receptor_condition_raises_keyerror():
    """A buffer collected before receptor-condition storage (or missing entries for
    other reasons) must fail loudly, not silently zero out the receptor again."""
    tmpdir = tempfile.mkdtemp()
    try:
        mixer, proteina, _ = _build_mixer_and_proteina(tmpdir, with_receptor_conditions=False)
        try:
            mixer.sample_replay_loss(proteina, batch_size=4, device=torch.device("cpu"))
            assert False, "expected KeyError"
        except KeyError:
            pass
    finally:
        shutil.rmtree(tmpdir)


ALL_TESTS = [
    test_empty_buffer_returns_none,
    test_zero_batch_size_returns_none,
    test_sample_replay_loss_produces_finite_weighted_loss,
    test_replay_loss_backprops_into_nn_params_not_stored_endpoint,
    test_all_zero_success_weights_give_success_only_zero_loss,
    test_geocycler_mode_gets_nonzero_weight_variance_via_grouped_sampling,
    test_reward_version_mismatch_raises_on_load,
    test_collate_pads_and_threads_receptor_condition,
    test_missing_receptor_condition_raises_keyerror,
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
