#!/usr/bin/env python3
"""Unit tests for the pure helper functions in `scripts/collect_cpsea_replay_rollouts.py`.

Only the CPU-testable pieces (`_to_str_list`, `_repeat_batch`) are covered here;
`collect()` itself needs a real checkpoint/dataset/GPU and cannot be unit tested.

Usage:
    python script_utils/test_collect_replay_rollouts_helpers.py
    pytest script_utils/test_collect_replay_rollouts_helpers.py -v
"""

from __future__ import annotations

import importlib.util
import pathlib

import torch

_SPEC = importlib.util.spec_from_file_location(
    "collect_cpsea_replay_rollouts",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "collect_cpsea_replay_rollouts.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_to_str_list = _MODULE._to_str_list
_repeat_batch = _MODULE._repeat_batch


def test_to_str_list_from_tensor():
    out = _to_str_list(torch.tensor([1, 2, 3]), batch_size=3)
    assert out == ["1", "2", "3"]


def test_to_str_list_from_list():
    out = _to_str_list(["a", "b"], batch_size=2)
    assert out == ["a", "b"]


def test_to_str_list_from_scalar_broadcasts():
    out = _to_str_list("shared", batch_size=3)
    assert out == ["shared", "shared", "shared"]


def test_repeat_batch_tensor_repeat_interleave():
    batch = {"mask": torch.tensor([[True, True], [True, False]])}
    out = _repeat_batch(batch, k=3)
    assert out["mask"].shape == (6, 2)
    # receptor 0's 3 copies come first, then receptor 1's -- repeat_interleave, not tile.
    assert torch.equal(out["mask"][:3], batch["mask"][0:1].expand(3, -1))
    assert torch.equal(out["mask"][3:], batch["mask"][1:2].expand(3, -1))


def test_repeat_batch_list_column():
    batch = {"example_id": ["a", "b"]}
    out = _repeat_batch(batch, k=2)
    assert out["example_id"] == ["a", "a", "b", "b"]


def test_repeat_batch_scalar_passthrough():
    batch = {"config_name": "some_config"}
    out = _repeat_batch(batch, k=4)
    assert out["config_name"] == "some_config"


ALL_TESTS = [
    test_to_str_list_from_tensor,
    test_to_str_list_from_list,
    test_to_str_list_from_scalar_broadcasts,
    test_repeat_batch_tensor_repeat_interleave,
    test_repeat_batch_list_column,
    test_repeat_batch_scalar_passthrough,
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
