"""Tests for the topology-agnostic feature profile (pre-build check 3a).

The contract under test is narrow and specific: `profile()` must return the same key set,
in the same order, with finite values, for a cyclic macrocycle, a synthetic open state and
a real linear binder.  A feature that silently becomes NaN on one topology would decide the
adversarial comparison by its own absence, so "runs on all three" is the deliverable, not a
smoke test.

The real-data cases are skipped rather than failed when the staged sets are not mounted, so
this file is still useful on a machine without the cluster filesystem.  The synthetic cases
always run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from script_utils.peptide_profile import (
    FEATURES,
    HELDOUT_FEATURES,
    STAGING_FEATURES,
    TUNING_FEATURES,
    contact_order,
    end_to_end,
    load_complex,
    profile,
    receptor_segments,
    residue_frame,
    rg_scaled,
    staging_profile,
    terminal_frame_6dof,
)

REPO = Path(__file__).resolve().parents[1]
ZFS = Path("/zfsauton/scratch/yixiz")

LNR_PDB = REPO / "CPSea_data/lnr_pocket14/pdbs/LNR_1bjr_E_I.pdb"
CPSEA_PDB = (ZFS / "CPSea/CPSea_full/CPSea/preprocessed/processed/train"
                   "/AF-A0A009E759-F1_0_89_101_relaxed_relaxed.pdb")
PEPBENCH_PDB = (ZFS / "LPData/preprocessed/processed/train/pepbench__A_K_pdb4wjv.pdb")


# --------------------------------------------------------------------------- pure geometry
def test_feature_partition_is_a_partition():
    """The tuning / held-out split must cover FEATURES exactly once, or the harness lies."""
    assert set(TUNING_FEATURES) | set(HELDOUT_FEATURES) == set(FEATURES)
    assert not (set(TUNING_FEATURES) & set(HELDOUT_FEATURES))
    assert not (set(FEATURES) & set(STAGING_FEATURES))


def test_residue_frame_is_orthonormal_right_handed():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, ca, c = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        R = residue_frame(n, ca, c)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def _straight_backbone(L: int, rise: float = 3.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A crude but valid N/CA/C trace along +x.  Enough to exercise the frame algebra."""
    CA = np.stack([np.arange(L) * rise, np.zeros(L), np.zeros(L)], axis=1).astype(float)
    N = CA + np.array([-0.9, 1.1, 0.0])
    C = CA + np.array([1.0, 0.9, 0.0])
    return N, CA, C


def test_terminal_frame_is_invariant_to_rigid_motion():
    """The six DOF must not move when the whole complex is rotated and translated.

    This is the property that lets two loaders which centre the receptor differently still
    agree -- and the CPSea loader is known to redraw its frame per process, so a feature
    that failed here would drift between runs for no physical reason.
    """
    N, CA, C = _straight_backbone(8)
    before = terminal_frame_6dof(N, CA, C)

    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    t = rng.normal(size=3) * 25.0
    after = terminal_frame_6dof(N @ Q.T + t, CA @ Q.T + t, C @ Q.T + t)

    assert np.allclose(before, after, atol=1e-8)


def test_terminal_frame_identity_rotation_is_zero_vector():
    """Two identically-oriented termini give a zero axis-angle, not a NaN at theta = 0."""
    N, CA, C = _straight_backbone(6)
    tf = terminal_frame_6dof(N, CA, C)
    assert all(math.isfinite(v) for v in tf)
    assert np.allclose(tf[3:], 0.0, atol=1e-7)


def test_end_to_end_normalises_by_links_not_residues():
    _, CA, _ = _straight_backbone(5, rise=4.0)
    raw, per = end_to_end(CA)
    assert np.isclose(raw, 16.0)
    assert np.isclose(per, 4.0)          # 4 links, not 5 residues


def test_rg_scaled_is_size_normalised():
    """Two straight chains of different length must land closer together after scaling."""
    _, ca_short, _ = _straight_backbone(6)
    _, ca_long, _ = _straight_backbone(18)

    def raw_rg(x):
        c = x - x.mean(axis=0, keepdims=True)
        return float(np.sqrt((c ** 2).sum(axis=1).mean()))

    raw_ratio = raw_rg(ca_long) / raw_rg(ca_short)
    scaled_ratio = rg_scaled(ca_long) / rg_scaled(ca_short)
    assert scaled_ratio < raw_ratio


def test_contact_order_zero_on_extended_chain():
    """A straight chain makes no |i-j| >= 3 contacts, which is 0.0 and not a missing value."""
    _, CA, _ = _straight_backbone(12)
    assert contact_order([c.reshape(1, 3) for c in CA]) == 0.0


def test_contact_order_positive_when_ends_touch():
    _, CA, _ = _straight_backbone(12)
    CA[-1] = CA[0] + np.array([1.0, 0.0, 0.0])      # fold the last residue onto the first
    assert contact_order([c.reshape(1, 3) for c in CA]) > 0.0


def test_receptor_segments_counts_numbering_gaps():
    """The pocket crop deletes residues but keeps numbering, so gaps ARE the segmentation."""
    assert receptor_segments([1, 2, 3, 4, 5]) == (1, 5, 0)
    assert receptor_segments([1, 2, 3, 40, 41, 90]) == (3, 3, 3)
    assert receptor_segments([]) == (0, 0, 0)


# ------------------------------------------------------------------------- real topologies
def _require(path: Path):
    if not path.exists():
        pytest.skip(f"staged set not mounted: {path}")
    cx = load_complex(path)
    if cx is None:
        pytest.skip(f"unreadable staged complex: {path}")
    return cx


def _assert_well_formed(row):
    assert list(row.keys()) == list(FEATURES), "feature order is part of the contract"
    bad = [k for k, v in row.items() if not math.isfinite(v)]
    assert not bad, f"non-finite features: {bad}"


@pytest.mark.parametrize("path,label", [
    (LNR_PDB, "linear (LNR)"),
    (CPSEA_PDB, "cyclic (CPSea)"),
    (PEPBENCH_PDB, "linear (PepBench)"),
])
def test_profile_runs_on_real_topology(path, label):
    _assert_well_formed(profile(_require(path)))


def test_profile_runs_on_synthetic_open_state():
    """The third topology: a ring pulled apart at one point.

    No corruption pipeline exists yet and this test does not build one -- it displaces the
    C-terminal residue of a real cyclic peptide far enough that the ring is unambiguously
    open, purely to confirm the profile stays defined when it is.
    """
    cx = _require(CPSEA_PDB)
    shift = np.array([12.0, 0.0, 0.0])
    cx.CA[-1] += shift
    cx.N[-1] += shift
    cx.C[-1] += shift
    cx.pep_heavy[-1] = cx.pep_heavy[-1] + shift

    # Move the same residue in the mdtraj copies, so the SASA / torsion / H-bond features
    # see the open state too rather than silently reporting the closed one.
    last = cx.pep_res_index[-1]
    for traj, res_index in ((cx.traj, last), (cx.pep_traj, cx.pep_traj.n_residues - 1)):
        idx = [a.index for a in traj.topology.residue(res_index).atoms]
        traj.xyz[0, idx, :] += (shift / 10.0).astype(traj.xyz.dtype)   # mdtraj is nm

    row = profile(cx)
    _assert_well_formed(row)
    assert row["e2e_ca_A"] > 10.0, "the displaced terminus should read as an open ring"


def test_staging_profile_is_disjoint_from_the_feature_vector():
    cx = _require(LNR_PDB)
    assert not (set(profile(cx)) & set(staging_profile(cx)))
    assert list(staging_profile(cx).keys()) == list(STAGING_FEATURES)


def test_pepbench_staging_is_separable_from_cpsea():
    """The confound, pinned as a test so it cannot be forgotten.

    PepBench receptors are whole chains and CPSea receptors are pocket crops, so the
    staging vector alone separates the sets trivially.  Any adversarial AUC computed
    without controlling for this is measuring the staging, not the conformation.
    """
    a = staging_profile(_require(CPSEA_PDB))
    b = staging_profile(_require(PEPBENCH_PDB))
    assert b["receptor_n_segments"] < a["receptor_n_segments"]
    assert b["receptor_max_run"] > a["receptor_max_run"]
