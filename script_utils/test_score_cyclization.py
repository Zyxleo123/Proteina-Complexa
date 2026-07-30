#!/usr/bin/env python3
"""Unit tests for `proteinfoundation.cyclization.scoring.score_cyclization`.

Pure-tensor tests: no GPU, no dataset, no model required.

Usage (as a standalone script):
    python script_utils/test_score_cyclization.py

Usage (via pytest, if installed):
    pytest script_utils/test_score_cyclization.py -v
"""

from __future__ import annotations

import math

import torch

from proteinfoundation.cyclization.constants import (
    AA_ASP,
    AA_CYS,
    AA_GLU,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
)
from proteinfoundation.cyclization.scoring import score_cyclization
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    C_IDX,
    CA_IDX,
    CB_IDX,
    CD_IDX,
    CG_IDX,
    N_IDX,
    NZ_IDX,
    SG_IDX,
    virtual_cb_from_backbone,
)

AA_OTHER = 0  # never CYS/LYS/ASP/GLU


def _zeros_atom37(batch_size: int, length: int):
    coords = torch.zeros(batch_size, length, 37, 3)
    mask = torch.zeros(batch_size, length, 37, dtype=torch.bool)
    return coords, mask


def _uniform_aa(batch_size: int, length: int, fill: int = AA_OTHER) -> torch.Tensor:
    return torch.full((batch_size, length), fill, dtype=torch.long)


def _linkage_meta(i: int, j: int, cyc_type: int, has_cyclization: bool = True):
    return {
        "i": torch.tensor([i]),
        "j": torch.tensor([j]),
        "type": torch.tensor([cyc_type]),
        "has_cyclization": torch.tensor([has_cyclization]),
    }


# ---------------------------------------------------------------------------
# Fixture builders: one per {linkage type} x {endpoint orientation}.
# ---------------------------------------------------------------------------
def _mainchain_fixture(dist_nm: float, i_is_first: bool = True):
    """2-residue binder closing head-to-tail: N(i) <-> C(j)."""
    atom37, mask = _zeros_atom37(1, 2)
    aatype = _uniform_aa(1, 2)
    binder_mask = torch.ones(1, 2, dtype=torch.bool)

    lo, hi = (0, 1) if i_is_first else (1, 0)
    # `per_sample_requested_bond_distance` takes the min over both N/C
    # orientations, so it requires N *and* C present at *both* residues (see
    # `mainchain_valid = c_i_valid & n_j_valid & n_i_valid & c_j_valid`).
    mask[0, :, N_IDX] = True
    mask[0, :, C_IDX] = True
    mask[0, :, CA_IDX] = True
    atom37[0, lo, N_IDX] = torch.tensor([0.0, 0.0, 0.0])
    atom37[0, hi, C_IDX] = torch.tensor([0.0, 0.0, dist_nm])
    # The "other" (unrelated) N/C pair: placed far apart so it never becomes
    # the min-distance orientation and corrupts the intended bond distance.
    atom37[0, hi, N_IDX] = torch.tensor([50.0, 0.0, 0.0])
    atom37[0, lo, C_IDX] = torch.tensor([50.0, 1.0, 0.0])

    meta = _linkage_meta(0, 1, MAINCHAIN)
    return atom37, mask, aatype, binder_mask, meta


def _disulfide_fixture(dist_nm: float, i_is_first: bool = True):
    atom37, mask = _zeros_atom37(1, 2)
    aatype = _uniform_aa(1, 2, fill=AA_CYS)
    binder_mask = torch.ones(1, 2, dtype=torch.bool)

    lo, hi = (0, 1) if i_is_first else (1, 0)
    mask[0, :, SG_IDX] = True
    atom37[0, lo, SG_IDX] = torch.tensor([0.0, 0.0, 0.0])
    atom37[0, hi, SG_IDX] = torch.tensor([0.0, 0.0, dist_nm])

    meta = _linkage_meta(0, 1, DISULFIDE)
    return atom37, mask, aatype, binder_mask, meta


def _isopeptide_fixture(dist_nm: float, lys_is_first: bool = True, acid: str = "ASP"):
    atom37, mask = _zeros_atom37(1, 2)
    aatype = _uniform_aa(1, 2)
    binder_mask = torch.ones(1, 2, dtype=torch.bool)

    acid_aa = AA_ASP if acid == "ASP" else AA_GLU
    acid_idx = CG_IDX if acid == "ASP" else CD_IDX

    lys_pos, acid_pos = (0, 1) if lys_is_first else (1, 0)
    aatype[0, lys_pos] = AA_LYS
    aatype[0, acid_pos] = acid_aa

    mask[0, lys_pos, NZ_IDX] = True
    mask[0, acid_pos, acid_idx] = True
    atom37[0, lys_pos, NZ_IDX] = torch.tensor([0.0, 0.0, 0.0])
    atom37[0, acid_pos, acid_idx] = torch.tensor([0.0, 0.0, dist_nm])

    meta = _linkage_meta(0, 1, ISOPEPTIDE)
    return atom37, mask, aatype, binder_mask, meta


FIXTURE_MATRIX = {
    "mainchain_i_first": lambda: _mainchain_fixture(0.133, i_is_first=True),
    "mainchain_j_first": lambda: _mainchain_fixture(0.133, i_is_first=False),
    "disulfide_i_first": lambda: _disulfide_fixture(0.205, i_is_first=True),
    "disulfide_j_first": lambda: _disulfide_fixture(0.205, i_is_first=False),
    "isopeptide_lys_first": lambda: _isopeptide_fixture(0.132, lys_is_first=True),
    "isopeptide_lys_second": lambda: _isopeptide_fixture(0.132, lys_is_first=False),
}


# ---------------------------------------------------------------------------
# A. Every linkage type + orientation: native/near-native distance scores high.
# ---------------------------------------------------------------------------
def test_every_fixture_native_distance_scores_high_and_succeeds():
    for name, build in FIXTURE_MATRIX.items():
        atom37, mask, aatype, binder_mask, meta = build()
        out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
        assert out["reward"].item() > 0.99, f"{name}: reward={out['reward'].item()}"
        assert out["success"].item() is True, f"{name}: success={out['success'].item()}"
        assert out["distance_error"].item() == 0.0, name


# ---------------------------------------------------------------------------
# B. Monotonic degradation as the linked atom moves away.
# ---------------------------------------------------------------------------
def test_reward_monotonically_decreases_with_displacement():
    dists_nm = [0.205, 0.23, 0.28, 0.35, 0.50, 1.0]  # disulfide ideal -> far
    rewards = []
    for d in dists_nm:
        atom37, mask, aatype, binder_mask, meta = _disulfide_fixture(d)
        out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
        rewards.append(out["reward"].item())
    for a, b in zip(rewards, rewards[1:]):
        assert a >= b - 1e-6, f"reward not monotonically non-increasing: {rewards}"
    assert rewards[0] > rewards[-1]


def test_reward_monotonically_decreases_moving_below_window_too():
    # Disulfide window is [1.8, 2.3] A; sweep from ideal down through "fused".
    dists_nm = [0.205, 0.18, 0.15, 0.10, 0.05, 0.01]
    rewards = []
    for d in dists_nm:
        atom37, mask, aatype, binder_mask, meta = _disulfide_fixture(d)
        out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
        rewards.append(out["reward"].item())
    for a, b in zip(rewards, rewards[1:]):
        assert a >= b - 1e-6, f"reward not monotonically non-increasing: {rewards}"


# ---------------------------------------------------------------------------
# C. Missing required atoms -> zero reward, no NaN.
# ---------------------------------------------------------------------------
def test_missing_atoms_zero_reward_no_nan():
    atom37, mask, aatype, binder_mask, meta = _disulfide_fixture(0.205)
    mask[0, 1, SG_IDX] = False  # drop the partner SG entirely
    out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
    assert out["reward"].item() == 0.0
    assert out["success"].item() is False
    assert out["distance_error"].item() == 0.0
    for k, v in out.items():
        assert torch.isnan(v.float()).sum().item() == 0, f"NaN in {k}"


def test_has_cyclization_false_zero_reward_no_nan():
    atom37, mask, aatype, binder_mask, meta = _mainchain_fixture(0.133)
    meta["has_cyclization"] = torch.tensor([False])
    out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
    assert out["reward"].item() == 0.0
    assert out["success"].item() is False
    for k, v in out.items():
        assert torch.isnan(v.float()).sum().item() == 0, f"NaN in {k}"


def test_wrong_chemistry_zero_reward_no_nan():
    # Disulfide type requested, but neither residue is CYS.
    atom37, mask, aatype, binder_mask, meta = _disulfide_fixture(0.205)
    aatype[:, :] = AA_LYS
    out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
    assert out["reward"].item() == 0.0
    for k, v in out.items():
        assert torch.isnan(v.float()).sum().item() == 0, f"NaN in {k}"


# ---------------------------------------------------------------------------
# D. SE(3) invariance.
# ---------------------------------------------------------------------------
def _random_rotation():
    a = torch.randn(3, 3)
    q, r = torch.linalg.qr(a)
    d = torch.diag(torch.sign(torch.diag(r)))
    q = q @ d
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_se3_invariance():
    torch.manual_seed(0)
    atom37, mask, aatype, binder_mask, meta = _isopeptide_fixture(0.132)
    out_before = score_cyclization(atom37, mask, aatype, binder_mask, meta)

    rot = _random_rotation()
    trans = torch.tensor([1.5, -2.0, 0.7])
    atom37_rt = torch.einsum("ij,blaj->blai", rot, atom37) + trans

    out_after = score_cyclization(atom37_rt, mask, aatype, binder_mask, meta)

    for k in ("reward", "distance_error", "angle_errors", "dihedral_error", "clash_count"):
        assert math.isclose(
            out_before[k].item(), out_after[k].item(), abs_tol=1e-4
        ), f"{k}: {out_before[k].item()} vs {out_after[k].item()}"
    assert out_before["success"].item() == out_after["success"].item()
    assert out_before["chirality_valid"].item() == out_after["chirality_valid"].item()


# ---------------------------------------------------------------------------
# E. Batched call == looped single-example calls.
# ---------------------------------------------------------------------------
def test_batched_matches_single_example_loop():
    builds = [
        _mainchain_fixture(0.133),
        _disulfide_fixture(0.205),
        _isopeptide_fixture(0.132),
        _disulfide_fixture(0.50),  # a failing one, mixed in
    ]
    L = max(b[0].shape[1] for b in builds)

    def _pad(t, target_len, pad_value=0):
        if t.shape[1] == target_len:
            return t
        pad_shape = list(t.shape)
        pad_shape[1] = target_len - t.shape[1]
        pad = torch.full(pad_shape, pad_value, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    atom37 = torch.cat([_pad(b[0], L) for b in builds], dim=0)
    mask = torch.cat([_pad(b[1], L, pad_value=False) for b in builds], dim=0)
    aatype = torch.cat([_pad(b[2], L) for b in builds], dim=0)
    binder_mask = torch.cat([_pad(b[3], L, pad_value=False) for b in builds], dim=0)
    meta = {
        "i": torch.cat([b[4]["i"] for b in builds]),
        "j": torch.cat([b[4]["j"] for b in builds]),
        "type": torch.cat([b[4]["type"] for b in builds]),
        "has_cyclization": torch.cat([b[4]["has_cyclization"] for b in builds]),
    }

    out_batched = score_cyclization(atom37, mask, aatype, binder_mask, meta)

    for idx, build in enumerate(builds):
        single_meta = {k: v[idx : idx + 1] for k, v in meta.items()}
        out_single = score_cyclization(
            atom37[idx : idx + 1],
            mask[idx : idx + 1],
            aatype[idx : idx + 1],
            binder_mask[idx : idx + 1],
            single_meta,
        )
        for k in out_batched:
            assert math.isclose(
                float(out_batched[k][idx]), float(out_single[k][0]), abs_tol=1e-5
            ), f"mismatch at {idx}, key {k}: {out_batched[k][idx]} vs {out_single[k][0]}"


# ---------------------------------------------------------------------------
# F. Chirality.
# ---------------------------------------------------------------------------
def test_chirality_valid_for_standard_l_backbone():
    atom37, mask, aatype, binder_mask, meta = _mainchain_fixture(0.133)
    n = torch.tensor([0.0, 0.0, 0.0])
    ca = torch.tensor([1.46, 0.0, 0.0])
    c = torch.tensor([2.0, 1.4, 0.0])
    cb = virtual_cb_from_backbone(n, ca, c)

    atom37[0, 0, N_IDX] = n
    atom37[0, 0, CA_IDX] = ca
    atom37[0, 0, C_IDX] = c
    atom37[0, 0, CB_IDX] = cb
    mask[0, 0, N_IDX] = True
    mask[0, 0, CA_IDX] = True
    mask[0, 0, C_IDX] = True
    mask[0, 0, CB_IDX] = True

    out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
    assert out["chirality_valid"].item() is True


def test_chirality_invalid_when_cb_reflected():
    atom37, mask, aatype, binder_mask, meta = _mainchain_fixture(0.133)
    n = torch.tensor([0.0, 0.0, 0.0])
    ca = torch.tensor([1.46, 0.0, 0.0])
    c = torch.tensor([2.0, 1.4, 0.0])
    cb = virtual_cb_from_backbone(n, ca, c)
    cb_reflected = 2 * ca - cb  # reflect CB through CA along the same axis -> flips the stereocenter

    atom37[0, 0, N_IDX] = n
    atom37[0, 0, CA_IDX] = ca
    atom37[0, 0, C_IDX] = c
    atom37[0, 0, CB_IDX] = cb_reflected
    mask[0, 0, N_IDX] = True
    mask[0, 0, CA_IDX] = True
    mask[0, 0, C_IDX] = True
    mask[0, 0, CB_IDX] = True

    out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
    assert out["chirality_valid"].item() is False


# ---------------------------------------------------------------------------
# G. Clash count.
# ---------------------------------------------------------------------------
def test_clash_count_detects_non_adjacent_close_residues():
    atom37, mask = _zeros_atom37(1, 4)
    aatype = _uniform_aa(1, 4)
    binder_mask = torch.ones(1, 4, dtype=torch.bool)
    mask[0, :, CA_IDX] = True
    for k in range(4):
        atom37[0, k, CA_IDX] = torch.tensor([float(k) * 0.38, 0.0, 0.0])  # ~3.8 A spacing, no clash
    # Now push residue 3 on top of residue 0 (non-adjacent: |3-0|>1).
    atom37[0, 3, CA_IDX] = torch.tensor([0.01, 0.0, 0.0])  # 0.1 A away -> clash

    meta = _linkage_meta(0, 1, MAINCHAIN, has_cyclization=False)
    out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
    assert out["clash_count"].item() == 1.0


def test_clash_count_zero_when_well_separated():
    atom37, mask = _zeros_atom37(1, 4)
    aatype = _uniform_aa(1, 4)
    binder_mask = torch.ones(1, 4, dtype=torch.bool)
    mask[0, :, CA_IDX] = True
    for k in range(4):
        atom37[0, k, CA_IDX] = torch.tensor([float(k) * 0.38, 0.0, 0.0])

    meta = _linkage_meta(0, 1, MAINCHAIN, has_cyclization=False)
    out = score_cyclization(atom37, mask, aatype, binder_mask, meta)
    assert out["clash_count"].item() == 0.0


ALL_TESTS = [
    test_every_fixture_native_distance_scores_high_and_succeeds,
    test_reward_monotonically_decreases_with_displacement,
    test_reward_monotonically_decreases_moving_below_window_too,
    test_missing_atoms_zero_reward_no_nan,
    test_has_cyclization_false_zero_reward_no_nan,
    test_wrong_chemistry_zero_reward_no_nan,
    test_se3_invariance,
    test_batched_matches_single_example_loop,
    test_chirality_valid_for_standard_l_backbone,
    test_chirality_invalid_when_cb_reflected,
    test_clash_count_detects_non_adjacent_close_residues,
    test_clash_count_zero_when_well_separated,
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
