"""Mixes reward-weighted replay endpoints into ordinary CPSea training steps.

Additive only: the real-CPSea-batch path in `Proteina.training_step` is
unmodified. When `replay_buffer.enabled` is set, `Proteina.training_step` also
draws a balanced sample from an on-disk `ReplayBuffer` via this class and adds
`lambda_replay * replay_loss` to the real-batch loss -- the two terms share the
exact same `flow_loss_from_clean_target` code path (Stage 5), so the only new
code here is turning a stored `ReplayBuffer` entry into that function's
`(clean_x1_ca, clean_z1, condition, mask, sample_weight)` arguments.

Scope note: `condition` here carries only the cyclization-conditioning fields a
stored replay entry has (`cyclization_i/j/type/type_cond`, `has_cyclization`).
CPSea's shipped training config (`training_cpsea_peptide_cyc_typecond.yaml`) is
a receptor-free cyclic-peptide setup, so this is sufficient for it; a config
that also conditions on a receptor/hotspot would need those fields threaded
through the replay buffer and this collation step too (they are not currently
stored -- see `proteinfoundation.replay.buffer`).
"""

from __future__ import annotations

import torch

from proteinfoundation.cyclization.constants import NO_CYCLIZATION_INDEX
from proteinfoundation.replay.buffer import ReplayBuffer
from proteinfoundation.replay.weighting import compute_sample_weights
from proteinfoundation.training.flow_loss import flow_loss_from_clean_target


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
            group_ids.append(f"{entry['cluster_id']}:{int(entry['linkage_type'])}")

        condition = {
            "cyclization_i": cyc_i.to(device),
            "cyclization_j": cyc_j.to(device),
            "cyclization_type": cyc_type.to(device),
            "cyclization_type_cond": cyc_type_cond.to(device),
            "has_cyclization": has_cyc.to(device),
        }
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
