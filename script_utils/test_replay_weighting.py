#!/usr/bin/env python3
"""Unit tests for `proteinfoundation.replay.weighting`.

Pure CPU tests: no GPU, no dataset, no model required.

Usage (as a standalone script):
    python script_utils/test_replay_weighting.py

Usage (via pytest, if installed):
    pytest script_utils/test_replay_weighting.py -v
"""

from __future__ import annotations

import math

import torch

from proteinfoundation.replay.weighting import (
    compute_sample_weights,
    effective_sample_size,
    geocycler_group_relative_weights,
    raw_bounded_weights,
    reward_weight_correlation,
    success_only_weights,
    weight_histogram,
)


# ---------------------------------------------------------------------------
# A. GeoCycler group-relative weighting
# ---------------------------------------------------------------------------
def test_geocycler_low_std_group_is_skipped_to_zero():
    rewards = torch.tensor([0.50, 0.51, 0.49, 0.50])
    group_ids = ["g0", "g0", "g0", "g0"]
    weights = geocycler_group_relative_weights(rewards, group_ids, reward_std_threshold=0.05)
    assert torch.all(weights == 0.0), weights


def test_geocycler_high_std_group_gets_shaped_weights():
    rewards = torch.tensor([0.0, 0.5, 1.0])
    group_ids = ["g0", "g0", "g0"]
    weights = geocycler_group_relative_weights(rewards, group_ids, reward_std_threshold=0.01)
    # Highest reward -> highest weight, lowest reward -> lowest weight.
    assert weights[2] > weights[1] > weights[0]
    assert torch.all((weights >= 0.0) & (weights <= 1.0))


def test_geocycler_groups_are_independent():
    # Group A has spread (should get shaped weights); group B is degenerate (skipped).
    rewards = torch.tensor([0.0, 1.0, 0.5, 0.5])
    group_ids = ["A", "A", "B", "B"]
    weights = geocycler_group_relative_weights(rewards, group_ids, reward_std_threshold=0.05)
    assert weights[0] < weights[1]
    assert weights[2] == 0.0 and weights[3] == 0.0


# ---------------------------------------------------------------------------
# B. Success-only weighting
# ---------------------------------------------------------------------------
def test_success_only_mapping():
    success = torch.tensor([True, False, False])
    near_success = torch.tensor([False, True, False])
    weights = success_only_weights(success, near_success, near_weight=0.3)
    expected = [1.0, 0.3, 0.0]
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(weights.tolist(), expected)), weights


def test_success_only_success_takes_precedence_over_near():
    # A malformed input marking the same entry both success and near_success
    # should still resolve to the success weight, not the near weight.
    success = torch.tensor([True])
    near_success = torch.tensor([True])
    weights = success_only_weights(success, near_success, near_weight=0.3)
    assert weights.tolist() == [1.0]


# ---------------------------------------------------------------------------
# C. Raw bounded weighting
# ---------------------------------------------------------------------------
def test_raw_bounded_is_identity_on_bounded_input():
    rewards = torch.tensor([0.0, 0.25, 0.75, 1.0])
    weights = raw_bounded_weights(rewards)
    assert torch.equal(weights, rewards)


def test_raw_bounded_clamps_out_of_range_input():
    rewards = torch.tensor([-0.1, 1.5])
    weights = raw_bounded_weights(rewards)
    assert weights.tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# D. Dispatch
# ---------------------------------------------------------------------------
def test_compute_sample_weights_dispatches_by_mode():
    rewards = torch.tensor([0.0, 1.0])
    group_ids = ["g", "g"]
    success = torch.tensor([True, False])
    near_success = torch.tensor([False, True])

    w1 = compute_sample_weights("geocycler_group_relative", rewards=rewards, group_ids=group_ids)
    w2 = compute_sample_weights("success_only", success=success, near_success=near_success)
    w3 = compute_sample_weights("raw_bounded", rewards=rewards)

    assert w1.shape == rewards.shape
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(w2.tolist(), [1.0, 0.3]))
    assert w3.tolist() == [0.0, 1.0]


def test_compute_sample_weights_unknown_mode_raises():
    try:
        compute_sample_weights("not_a_mode", rewards=torch.tensor([0.5]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compute_sample_weights_missing_args_raise():
    try:
        compute_sample_weights("success_only")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# E. Effective sample size
# ---------------------------------------------------------------------------
def test_ess_uniform_weights_equals_n():
    weights = torch.ones(10) * 0.7
    assert math.isclose(effective_sample_size(weights), 10.0, rel_tol=1e-4)


def test_ess_one_dominant_weight_is_one():
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert math.isclose(effective_sample_size(weights), 1.0, abs_tol=1e-6)


def test_ess_all_zero_is_zero():
    weights = torch.zeros(5)
    assert effective_sample_size(weights) == 0.0


# ---------------------------------------------------------------------------
# F. Weight histogram / reward-weight correlation
# ---------------------------------------------------------------------------
def test_weight_histogram_shape_and_total_count():
    weights = torch.tensor([0.05, 0.15, 0.95, 1.0])
    hist = weight_histogram(weights, bins=10)
    assert len(hist["bin_edges"]) == 11
    assert len(hist["counts"]) == 10
    assert sum(hist["counts"]) == 4


def test_reward_weight_correlation_perfect_positive():
    rewards = torch.tensor([0.0, 0.5, 1.0])
    weights = torch.tensor([0.0, 0.5, 1.0])
    corr = reward_weight_correlation(rewards, weights)
    assert math.isclose(corr, 1.0, abs_tol=1e-5)


def test_reward_weight_correlation_constant_input_is_nan():
    rewards = torch.tensor([0.5, 0.5, 0.5])
    weights = torch.tensor([1.0, 0.3, 0.0])
    corr = reward_weight_correlation(rewards, weights)
    assert corr != corr  # NaN


ALL_TESTS = [
    test_geocycler_low_std_group_is_skipped_to_zero,
    test_geocycler_high_std_group_gets_shaped_weights,
    test_geocycler_groups_are_independent,
    test_success_only_mapping,
    test_success_only_success_takes_precedence_over_near,
    test_raw_bounded_is_identity_on_bounded_input,
    test_raw_bounded_clamps_out_of_range_input,
    test_compute_sample_weights_dispatches_by_mode,
    test_compute_sample_weights_unknown_mode_raises,
    test_compute_sample_weights_missing_args_raise,
    test_ess_uniform_weights_equals_n,
    test_ess_one_dominant_weight_is_one,
    test_ess_all_zero_is_zero,
    test_weight_histogram_shape_and_total_count,
    test_reward_weight_correlation_perfect_positive,
    test_reward_weight_correlation_constant_input_is_nan,
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
