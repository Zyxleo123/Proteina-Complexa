"""Lightweight typed pairwise linkage head for CPSea cyclization prediction.

This is a classifier only. It does not enforce chemical geometry and never
modifies coordinates; it only reads a predicted-clean per-residue
representation and CA coordinates and outputs typed pair logits.
"""

import torch
import torch.nn as nn

from proteinfoundation.cyclization.constants import NUM_CYCLIZATION_TYPES


class CyclizationLinkHead(nn.Module):
    """Predicts symmetrized typed pair logits for the cyclization edge.

    Args:
        single_dim: Feature dimension of the per-residue input representation.
        hidden_dim: Hidden dimension used throughout the head.
        num_types: Number of linkage types (default 3, see `constants.py`).
    """

    def __init__(self, single_dim: int, hidden_dim: int = 256, num_types: int = NUM_CYCLIZATION_TYPES):
        super().__init__()
        self.num_types = num_types
        self.single_proj = nn.Sequential(
            nn.Linear(single_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, single: torch.Tensor, ca: torch.Tensor) -> torch.Tensor:
        """
        Args:
            single: [B, L, D] binder per-residue features from the predicted
                clean state (see caller for what goes into D).
            ca: [B, L, 3] predicted clean C-alpha coordinates.

        Returns:
            Symmetrized typed pair logits, shape [B, L, L, num_types].
        """
        h = self.single_proj(single)  # [B, L, H]

        hi = h[:, :, None, :].expand(-1, -1, h.shape[1], -1)
        hj = h[:, None, :, :].expand(-1, h.shape[1], -1, -1)

        diff = torch.abs(hi - hj)
        prod = hi * hj

        dist = torch.linalg.norm(ca[:, :, None, :] - ca[:, None, :, :], dim=-1, keepdim=True)

        pair = torch.cat([hi, hj, diff, prod, dist], dim=-1)
        logits = self.pair_mlp(pair)

        # Symmetrize so link_logits[i, j] == link_logits[j, i]; there is no
        # intrinsic ordering to a cyclization edge.
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return logits
