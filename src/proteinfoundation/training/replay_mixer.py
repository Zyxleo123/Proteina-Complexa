"""Mixes reward-weighted replay endpoints into ordinary CPSea training steps.

Additive only: the real-CPSea-batch path in `Proteina.training_step` is
unmodified. When `replay_buffer.enabled` is set, `Proteina.training_step` also
draws a balanced sample from an on-disk `ReplayBuffer` via this class and adds
`lambda_replay * replay_loss` to the real-batch loss -- the two terms share the
exact same `flow_loss_from_clean_target` code path (Stage 5), so the only new
code here is turning a stored `ReplayBuffer` entry into that function's
`(clean_x1_ca, clean_z1, condition, mask, sample_weight)` arguments.

`condition` carries the cyclization-conditioning fields a stored replay entry
has (`cyclization_i/j/type/type_cond`, `has_cyclization`) plus every
receptor-conditioning feature (`x_target`, `seq_target`, `target_mask`,
`seq_target_mask`, `target_hotspot_mask`, ...) CPSea's shipped
`enable_target: true` config needs -- resolved per entry from
`ReplayBuffer.receptor_conditions` via each entry's `target_or_dataset_id`
join key. Receptor tensors are stored once per receptor (not once per
candidate) in that side table, see `proteinfoundation.replay.buffer`, and
padded/stacked here to the batch's max target length.
"""

from __future__ import annotations

import torch

from proteinfoundation.cyclization.constants import NO_CYCLIZATION_INDEX
from proteinfoundation.replay.buffer import ReplayBuffer
from proteinfoundation.replay.weighting import compute_sample_weights
from proteinfoundation.training.flow_loss import flow_loss_from_clean_target


def _pad_stack_target_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Zero-pads variable-length `[n_target, ...]` tensors to a common `n_target` and stacks them.

    Padding rows read as "not a valid target position" downstream: boolean
    mask tensors (`target_mask`/`seq_target_mask`/`target_hotspot_mask`) zero-
    pad to `False`, which is exactly the semantics `ExtractTargetCoordinatesTransform`
    already relies on elsewhere in the codebase.
    """
    max_len = max(t.shape[0] for t in tensors)
    trailing_shape = tensors[0].shape[1:]
    out = torch.zeros((len(tensors), max_len) + trailing_shape, dtype=tensors[0].dtype)
    for k, t in enumerate(tensors):
        out[k, : t.shape[0]] = t
    return out


class ReplayMixer:
    """Owns a `ReplayBuffer` and turns a balanced draw from it into a replay loss."""

    def __init__(
        self,
        buffer_dir: str,
        reward_version: str,
        weighting_mode: str = "success_only",
        reward_std_threshold: float = 0.05,
        near_weight: float = 0.3,
        max_size: int = 10_000,
        stratify_by: tuple[str, ...] = ("linkage_type", "length_bin"),
        group_size: int = 8,
    ):
        try:
            self.buffer = ReplayBuffer.load(buffer_dir, expected_reward_version=reward_version)
        except FileNotFoundError:
            self.buffer = ReplayBuffer(max_size=max_size)
            self.buffer.reward_version = reward_version
        self.weighting_mode = weighting_mode
        self.reward_std_threshold = reward_std_threshold
        self.near_weight = near_weight
        self.stratify_by = stratify_by
        # Only consulted by `sample_replay_loss` for "geocycler_group_relative":
        # max candidates drawn per selected group (see `ReplayBuffer.sample_grouped`).
        self.group_size = group_size

    def _collate(self, entries: list[dict], device: torch.device):
        max_len = max(int(e["peptide_length"]) for e in entries)
        latent_dim = entries[0]["z1"].shape[-1]
        b = len(entries)

        x1_ca = torch.zeros(b, max_len, 3)
        z1 = torch.zeros(b, max_len, latent_dim)
        mask = torch.zeros(b, max_len, dtype=torch.bool)
        cyc_i = torch.full((b,), NO_CYCLIZATION_INDEX, dtype=torch.long)
        cyc_j = torch.full((b,), NO_CYCLIZATION_INDEX, dtype=torch.long)
        cyc_type = torch.full((b,), NO_CYCLIZATION_INDEX, dtype=torch.long)
        cyc_type_cond = torch.full((b,), NO_CYCLIZATION_INDEX, dtype=torch.long)
        has_cyc = torch.zeros(b, dtype=torch.bool)
        rewards = torch.zeros(b)
        success = torch.zeros(b, dtype=torch.bool)
        near_success = torch.zeros(b, dtype=torch.bool)
        group_ids: list[str] = []

        for k, entry in enumerate(entries):
            length = int(entry["peptide_length"])
            x1_ca[k, :length] = entry["x1_ca"].float()
            z1[k, :length] = entry["z1"].float()
            mask[k, :length] = entry["binder_mask"][:length].bool()

            i_local, j_local = entry["linkage_sites"]
            cyc_i[k] = int(i_local)
            cyc_j[k] = int(j_local)
            cyc_type[k] = int(entry["linkage_type"])
            cyc_type_cond[k] = int(entry["linkage_type"])
            has_cyc[k] = True

            rewards[k] = float(entry["raw_reward"])
            components = entry["reward_components"]
            success[k] = bool(components.get("success", False))
            near_success[k] = bool(components.get("near_success", False))
            # Exact receptor identity, not the coarser cluster_id: group-relative
            # weighting needs candidates that are actually K samples of the SAME
            # condition, and only entries sharing one target_or_dataset_id are
            # (see `ReplayBuffer.sample_grouped`, which this pairs with).
            group_ids.append(str(entry["target_or_dataset_id"]))

        condition = {
            "cyclization_i": cyc_i.to(device),
            "cyclization_j": cyc_j.to(device),
            "cyclization_type": cyc_type.to(device),
            "cyclization_type_cond": cyc_type_cond.to(device),
            "has_cyclization": has_cyc.to(device),
        }
        condition.update(self._resolve_receptor_condition(entries, device))
        return (
            x1_ca.to(device),
            z1.to(device),
            mask.to(device),
            condition,
            rewards.to(device),
            success.to(device),
            near_success.to(device),
            group_ids,
        )

    def _resolve_receptor_condition(self, entries: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
        """Looks up each entry's stored receptor tensors and batches them.

        This is what keeps replay from silently training "noise + cyclization
        type -> peptide" with the receptor zeroed out: `x_target` etc. are not
        stored per replay entry (see `ReplayBuffer.receptor_conditions`'s
        docstring for why), so they must be resolved here via each entry's
        `target_or_dataset_id` before this batch reaches `call_nn`.
        """
        receptor_conditions = self.buffer.receptor_conditions
        missing = sorted({e["target_or_dataset_id"] for e in entries if e["target_or_dataset_id"] not in receptor_conditions})
        if missing:
            raise KeyError(
                f"No stored receptor condition for target_or_dataset_id {missing!r}. "
                "This replay buffer predates receptor-condition storage (or was "
                "collected by a mismatched collector) -- recollect with the "
                "current scripts/collect_cpsea_replay_rollouts.py."
            )

        per_entry = [receptor_conditions[e["target_or_dataset_id"]] for e in entries]
        target_keys = set()
        for cond in per_entry:
            target_keys.update(cond.keys())

        resolved: dict[str, torch.Tensor] = {}
        for key in target_keys:
            tensors = [cond[key] for cond in per_entry if key in cond]
            if len(tensors) != len(per_entry):
                raise KeyError(
                    f"receptor-condition key {key!r} is present for some but not "
                    "all replay entries in this draw -- buffer entries have "
                    "inconsistent target-feature sets (mixed collector runs?)."
                )
            resolved[key] = _pad_stack_target_tensors(tensors).to(device)
        return resolved

    def sample_replay_loss(
        self,
        proteina,
        batch_size: int,
        device: torch.device,
        n_recycle: int = 0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]] | None:
        """Draws a balanced replay batch and computes its (weighted) flow loss.

        Returns `None` if the buffer has nothing to sample yet (e.g. before the
        first offline collection pass has populated/saved it) -- callers should
        treat that as "no replay contribution this step", not an error.
        """
        if len(self.buffer) == 0 or batch_size <= 0:
            return None

        if self.weighting_mode == "geocycler_group_relative":
            # i.i.d. `sample_balanced` almost never lands >1 candidate from the same
            # receptor in one draw, so every group is a singleton and every weight
            # collapses to 0 (see `geocycler_group_relative_weights`'s std-threshold
            # skip) -- `sample_grouped` draws intact per-receptor candidate groups
            # instead, so within-group reward variance actually exists to weight on.
            entries = self.buffer.sample_grouped(batch_size, group_size=self.group_size, by=self.stratify_by)
        else:
            entries = self.buffer.sample_balanced(batch_size, by=self.stratify_by)
        x1_ca, z1, mask, condition, rewards, success, near_success, group_ids = self._collate(entries, device)

        weights = compute_sample_weights(
            self.weighting_mode,
            rewards=rewards,
            group_ids=group_ids,
            success=success,
            near_success=near_success,
            reward_std_threshold=self.reward_std_threshold,
            near_weight=self.near_weight,
        )

        losses, _ = flow_loss_from_clean_target(
            proteina,
            x1_ca.detach(),
            z1.detach(),
            condition=condition,
            mask=mask,
            sample_weight=weights,
            n_recycle=n_recycle,
            from_replay=True,
        )
        diagnostics = {
            "n_sampled": float(len(entries)),
            "reward_mean": float(rewards.mean().item()),
            "weight_mean": float(weights.mean().item()),
        }
        return losses, diagnostics
