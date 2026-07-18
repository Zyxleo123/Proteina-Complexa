"""Unit tests for CyclizationClosureRewardModel.

Run: python -m pytest script_utils/test_cyclization_reward.py -q
The closure distance itself is measured by compute_cyclization_metrics_single (tested against
the training bond loss elsewhere); here we mock it to control the distance and pin the reward's
mapping distance -> reward, plus the no-op and missing-anchor edges.
"""

import numpy as np
import pytest
import torch

from proteinfoundation.rewards import cyclization_reward as cr
from proteinfoundation.rewards.cyclization_reward import CyclizationClosureRewardModel

MAIN_LO, MAIN_HI = cr._WINDOW_BY_TYPE["mainchain"]


def _patch_dist(monkeypatch, dist, closed=None):
    """Force compute_cyclization_metrics_single to report a given distance."""
    if closed is None:
        closed = (dist is not None) and (not np.isnan(dist)) and (MAIN_LO <= dist <= MAIN_HI)

    def fake(pdb_path, binder_chain, requested_type):
        return {"binder_cyc_bond_dist_A": dist, "binder_cyc_bond_closed": closed}

    monkeypatch.setattr(cr, "compute_cyclization_metrics_single", fake)


def test_closed_ring_scores_zero(monkeypatch):
    _patch_dist(monkeypatch, dist=(MAIN_LO + MAIN_HI) / 2)
    m = CyclizationClosureRewardModel(cyclization_type="mainchain", weight=3.0)
    out = m.score(pdb_path="x.pdb", binder_chain="B")
    assert out["total_reward"].item() == pytest.approx(0.0)
    assert out["cyc_closed"] == 1.0


def test_open_ring_is_penalised(monkeypatch):
    d = MAIN_HI + 0.5
    _patch_dist(monkeypatch, dist=d)
    m = CyclizationClosureRewardModel(cyclization_type="mainchain", weight=1.0)
    out = m.score(pdb_path="x.pdb", binder_chain="B")
    assert out["total_reward"].item() == pytest.approx(-((d - MAIN_HI) ** 2))
    assert out["cyc_closed"] == 0.0


def test_fused_ring_is_penalised(monkeypatch):
    # Below the window: the two-sided shape must punish this, not reward proximity.
    d = MAIN_LO - 0.3
    _patch_dist(monkeypatch, dist=d)
    m = CyclizationClosureRewardModel(cyclization_type="mainchain", weight=1.0)
    out = m.score(pdb_path="x.pdb", binder_chain="B")
    assert out["total_reward"].item() == pytest.approx(-((MAIN_LO - d) ** 2))
    assert out["cyc_closed"] == 0.0


def test_weight_scales_penalty(monkeypatch):
    d = MAIN_HI + 0.5
    _patch_dist(monkeypatch, dist=d)
    r1 = CyclizationClosureRewardModel("mainchain", weight=1.0).score(pdb_path="x", binder_chain="B")
    r5 = CyclizationClosureRewardModel("mainchain", weight=5.0).score(pdb_path="x", binder_chain="B")
    assert r5["total_reward"].item() == pytest.approx(5.0 * r1["total_reward"].item())


def test_missing_anchors_use_miss_distance(monkeypatch):
    # dist NaN -> treated as an open ring at miss_dist_A, not silently scored 0.
    _patch_dist(monkeypatch, dist=float("nan"), closed=False)
    miss = 4.0
    m = CyclizationClosureRewardModel(cyclization_type="mainchain", weight=1.0, miss_dist_A=miss)
    out = m.score(pdb_path="x.pdb", binder_chain="B")
    assert out["total_reward"].item() == pytest.approx(-((miss - MAIN_HI) ** 2))
    assert out["cyc_closed"] == 0.0
    assert out["cyc_bond_dist_A"] == pytest.approx(miss)


def test_none_type_is_noop_without_reading_pdb(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no-op reward must not read the PDB")

    monkeypatch.setattr(cr, "compute_cyclization_metrics_single", boom)
    m = CyclizationClosureRewardModel(cyclization_type=None)
    out = m.score(pdb_path="x.pdb", binder_chain=None)  # binder_chain not required for no-op
    assert out["total_reward"].item() == pytest.approx(0.0)
    assert out["reward"] == {}


def test_unknown_type_is_noop(monkeypatch):
    m = CyclizationClosureRewardModel(cyclization_type="lactam")  # not a known linkage
    assert m.cyclization_type is None
    out = m.score(pdb_path="x.pdb", binder_chain="B")
    assert out["total_reward"].item() == pytest.approx(0.0)


def test_binder_chain_required_when_active():
    m = CyclizationClosureRewardModel(cyclization_type="mainchain")
    with pytest.raises(ValueError, match="binder_chain"):
        m.score(pdb_path="x.pdb", binder_chain=None)


def test_grad_not_supported():
    m = CyclizationClosureRewardModel(cyclization_type="mainchain")
    with pytest.raises(ValueError):
        m.score(pdb_path="x.pdb", binder_chain="B", requires_grad=True)


def test_reward_output_conforms_to_schema(monkeypatch):
    _patch_dist(monkeypatch, dist=MAIN_HI + 0.5)
    m = CyclizationClosureRewardModel(cyclization_type="mainchain")
    out = m.score(pdb_path="x.pdb", binder_chain="B")
    assert {"reward", "grad", "total_reward"} <= set(out)
    assert isinstance(out["reward"], dict) and isinstance(out["grad"], dict)
    assert isinstance(out["total_reward"], torch.Tensor)
