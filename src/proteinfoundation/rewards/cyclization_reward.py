"""Ring-closure reward model for macrocyclic binder design.

The other in-search rewards (AF2 `i_pae`, TMol, bioinformatics) are all blind to the one
thing a macrocycle is *for*: whether the ring closed. AF2 in particular refolds the binder
as a linear chain, so its interface confidence is computed on a molecule with no closing
bond (see `evaluation.cyclization_pdb_metrics`). This reward reads the closing bond straight
off the generated binder chain -- the direct signal, not a correlate of it.

It is pure geometry (one interatomic distance), so it is cheap enough to run inside the
beam-search lookahead (~256x/target), unlike a Rosetta or folding call. Reward is
`-flat_bottom_penalty(dist)`: exactly zero when the anchor atoms sit inside the same
acceptance window used by the closure metric and the training bond loss, and quadratically
negative as the ring opens *or* fuses. The two-sidedness matters -- a one-sided "pull the
ends together" reward optimises straight into the interpenetrating-ring failure that shows
up as sub-1A anchor distances.

Drift-free by construction: distance/anchor selection comes from
`compute_cyclization_metrics_single` and the windows from `eval.cyclic_reconstruction_metrics`,
the same definitions the metric and the bond loss use. If "closed" ever gets redefined, this
reward moves with it.
"""

import logging
from typing import Any

import numpy as np
import torch

from proteinfoundation.cyclization.bond_loss import flat_bottom_penalty
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    DISULFIDE_SG_BOND_WINDOW_A,
    ISOPEPTIDE_N_C_BOND_WINDOW_A,
    MAINCHAIN_CN_BOND_WINDOW_A,
)
from proteinfoundation.evaluation.cyclization_pdb_metrics import compute_cyclization_metrics_single
from proteinfoundation.rewards.base_reward import BaseRewardModel, standardize_reward

logger = logging.getLogger(__name__)

_WINDOW_BY_TYPE = {
    "mainchain": MAINCHAIN_CN_BOND_WINDOW_A,
    "disulfide": DISULFIDE_SG_BOND_WINDOW_A,
    "isopeptide": ISOPEPTIDE_N_C_BOND_WINDOW_A,
}


class CyclizationClosureRewardModel(BaseRewardModel):
    """Reward the closing bond of a macrocycle sitting at a chemically valid length.

    Reward = ``-weight * flat_bottom_penalty(dist, window)`` for the requested linkage
    type, so it is 0 for a closed ring and negative for an open or fused one. Intended for
    the search reward (steering); it scores the *generated* structure (which carries the
    ring), never a refold, so leave ``structure_source`` at None.
    """

    IS_FOLDING_MODEL = False
    SUPPORTS_GRAD = False
    SUPPORTS_SAVE_PDB = False

    def __init__(
        self,
        cyclization_type: str | None,
        weight: float = 1.0,
        miss_dist_A: float = 4.0,
        structure_source: str | None = None,
    ) -> None:
        """Initialize the closure reward model.

        Args:
            cyclization_type: The linkage the model was asked for ("mainchain", "disulfide",
                "isopeptide"). None (or any type not in the window table) makes this reward a
                differentiable no-op -- safe to leave wired into a non-cyclic run.
            weight: Scales the penalty. The composite model also applies its own weight; this
                is here so the reward can be tuned without touching the composite weights.
            miss_dist_A: Distance substituted when the requested anchor atoms are absent (e.g.
                an "isopeptide" request whose terminal residues are not Lys/Asp, so there is
                no NZ-CG pair to measure). Treated as an open ring at this distance rather than
                silently scoring 0, so steering still penalises a sample that did not even build
                the right chemistry. Must sit outside the acceptance window to carry signal.
            structure_source: Folding-model key whose structure to score. None = generated
                structure (the only one that has the ring); do not change for this reward.
        """
        super().__init__()
        self.cyclization_type = cyclization_type if cyclization_type in _WINDOW_BY_TYPE else None
        self.weight = float(weight)
        self.miss_dist_A = float(miss_dist_A)
        self.structure_source = structure_source
        if cyclization_type is not None and self.cyclization_type is None:
            logger.warning(
                f"CyclizationClosureRewardModel got unknown cyclization_type={cyclization_type!r}; "
                f"reward will be a no-op. Expected one of {sorted(_WINDOW_BY_TYPE)}."
            )

    def score(
        self,
        pdb_path: str,
        requires_grad: bool = False,
        binder_chain: str | None = None,
        target_chain: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Closure reward for one generated complex.

        Args:
            pdb_path: Generated complex PDB (must be the generated structure, not a refold).
            requires_grad: Unsupported; raises if True.
            binder_chain: Binder chain ID (required unless this reward is a no-op).
            target_chain: Unused (interface compatibility).
            **kwargs: Ignored.

        Returns:
            Standard reward dict. Components: ``penalty`` (flat-bottom, Angstrom^2),
            ``bond_dist_A``, ``closed`` (1.0/0.0). ``total_reward = -weight * penalty``.
        """
        self._check_capabilities(requires_grad=requires_grad)

        # No-op for non-cyclic / unknown-type runs: differentiable zero, no PDB read.
        if self.cyclization_type is None:
            return standardize_reward(reward={}, total_reward=0.0)

        if binder_chain is None:
            raise ValueError("binder_chain is required for CyclizationClosureRewardModel")

        lo, hi = _WINDOW_BY_TYPE[self.cyclization_type]
        metrics = compute_cyclization_metrics_single(pdb_path, binder_chain, self.cyclization_type)

        dist = metrics.get("binder_cyc_bond_dist_A")
        closed = metrics.get("binder_cyc_bond_closed")
        # dist is NaN when the requested anchor atoms are absent -> treat as an open ring at
        # miss_dist_A so the sample is still penalised for not building the right linkage.
        anchors_missing = dist is None or (isinstance(dist, float) and np.isnan(dist))
        dist_eff = self.miss_dist_A if anchors_missing else float(dist)
        closed_val = 0.0 if (anchors_missing or not bool(closed)) else 1.0

        penalty = flat_bottom_penalty(
            torch.tensor(dist_eff, dtype=torch.float32),
            torch.tensor(lo, dtype=torch.float32),
            torch.tensor(hi, dtype=torch.float32),
        )
        total_reward = -self.weight * penalty

        return standardize_reward(
            reward={
                "penalty": penalty,
                "bond_dist_A": torch.tensor(dist_eff, dtype=torch.float32),
                "closed": torch.tensor(closed_val, dtype=torch.float32),
            },
            total_reward=total_reward,
            cyc_closed=closed_val,
            cyc_bond_dist_A=dist_eff,
        )
