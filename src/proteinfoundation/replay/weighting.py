"""Turns stored `raw_reward` (+ `success`/`near_success`) into per-example training weights.

Three modes (Stage 6 of the reward-weighted-replay plan), all pure functions
of `[N]` tensors -- no side effects, no randomness:

  - `"geocycler_group_relative"`: GeoCycler-style within-condition-group
    z-scored preference, mapped to `[0, 1]`.
  - `"success_only"`: a hard 1 / `near_weight` / 0 ladder from the Stage-2
    `success`/`near_success` flags.
  - `"raw_bounded"`: identity on the already-`[0, 1]`-bounded reward from
    `score_cyclization`.

No explicit negative loss is implemented anywhere here (an all-zero-or-positive
weight only ever down-weights or excludes an example from the ordinary flow
loss; it never flips its sign) -- this is a deliberate Stage-6 non-goal.
"""

from __future__ import annotations

import torch

DEFAULT_REWARD_STD_THRESHOLD = 0.05
DEFAULT_NEAR_WEIGHT = 0.3


def _group_index(group_ids: list) -> torch.Tensor:
    """Maps arbitrary hashable group ids to a dense `[N]` long tensor of group indices."""
    unique = {}
    idx = torch.empty(len(group_ids), dtype=torch.long)
    for i, g in enumerate(group_ids):
        if g not in unique:
            unique[g] = len(unique)
        idx[i] = unique[g]
    return idx, len(unique)


def geocycler_group_relative_weights(
    rewards: torch.Tensor,
    group_ids: list,
    reward_std_threshold: float = DEFAULT_REWARD_STD_THRESHOLD,
    eps: float = 1e-6,
) -> torch.Tensor:
    """GeoCycler-style group-relative advantage, mapped to `[0, 1]`.

    For each group of `K` candidates sharing a `group_id` (e.g. K samples
    drawn for the same condition):

        a_i = R_i - mean(R)
        r_i = 0.5 + 0.5 * clip(a_i / (std(R) + eps), -1, 1)

    Groups whose reward standard deviation is below `reward_std_threshold`
    get weight 0 for every member (skipped entirely), rather than falling
    through to `r_i ~= 0.5` for everyone -- which would silently train on
    every candidate with no preference signal (uniform self-imitation).
    """
    rewards = rewards.float()
    group_idx, n_groups = _group_index(group_ids)
    weights = torch.zeros_like(rewards)

    for g in range(n_groups):
        members = group_idx == g
        r = rewards[members]
        std = r.std(unbiased=False) if r.numel() > 1 else torch.zeros(())
        if float(std) < reward_std_threshold:
            weights[members] = 0.0
            continue
        a = r - r.mean()
        weights[members] = 0.5 + 0.5 * torch.clamp(a / (std + eps), -1.0, 1.0)

    return weights


def success_only_weights(
    success: torch.Tensor,
    near_success: torch.Tensor,
    near_weight: float = DEFAULT_NEAR_WEIGHT,
) -> torch.Tensor:
    """weight = 1 (success) / `near_weight` (near_success, not success) / 0 (failure)."""
    success = success.bool()
    near_success = near_success.bool() & ~success
    weights = torch.zeros(success.shape, dtype=torch.float32)
    weights = torch.where(success, torch.ones_like(weights), weights)
    weights = torch.where(near_success, torch.full_like(weights, float(near_weight)), weights)
    return weights


def raw_bounded_weights(rewards: torch.Tensor) -> torch.Tensor:
    """Identity: the `score_cyclization` reward is already bounded in `[0, 1]`."""
    return rewards.float().clamp(0.0, 1.0)


def compute_sample_weights(
    mode: str,
    *,
    rewards: torch.Tensor | None = None,
    group_ids: list | None = None,
    success: torch.Tensor | None = None,
    near_success: torch.Tensor | None = None,
    reward_std_threshold: float = DEFAULT_REWARD_STD_THRESHOLD,
    near_weight: float = DEFAULT_NEAR_WEIGHT,
) -> torch.Tensor:
    """Dispatches to one of the three weighting modes. See module docstring."""
    if mode == "geocycler_group_relative":
        if rewards is None or group_ids is None:
            raise ValueError("geocycler_group_relative requires `rewards` and `group_ids`")
        return geocycler_group_relative_weights(rewards, group_ids, reward_std_threshold=reward_std_threshold)
    elif mode == "success_only":
        if success is None or near_success is None:
            raise ValueError("success_only requires `success` and `near_success`")
        return success_only_weights(success, near_success, near_weight=near_weight)
    elif mode == "raw_bounded":
        if rewards is None:
            raise ValueError("raw_bounded requires `rewards`")
        return raw_bounded_weights(rewards)
    else:
        raise ValueError(f"unknown weighting mode: {mode!r}")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def effective_sample_size(weights: torch.Tensor) -> float:
    """`ESS = (sum w)^2 / sum(w^2)`; equals `N` when weights are uniform, 1 when one dominates."""
    weights = weights.float()
    denom = float((weights**2).sum().item())
    if denom <= 0.0:
        return 0.0
    return float(weights.sum().item()) ** 2 / denom


def weight_histogram(weights: torch.Tensor, bins: int = 10) -> dict[str, list[float]]:
    """Fixed-range `[0, 1]` histogram of weights, as `{"bin_edges": [...], "counts": [...]}`."""
    weights = weights.float().clamp(0.0, 1.0)
    counts = torch.histc(weights, bins=bins, min=0.0, max=1.0)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def reward_weight_correlation(rewards: torch.Tensor, weights: torch.Tensor) -> float:
    """Pearson correlation between raw reward and assigned weight; NaN if either is constant."""
    rewards = rewards.float()
    weights = weights.float()
    if rewards.numel() < 2 or float(rewards.std()) == 0.0 or float(weights.std()) == 0.0:
        return float("nan")
    stacked = torch.stack([rewards, weights])
    corr = torch.corrcoef(stacked)[0, 1]
    return float(corr.item())
