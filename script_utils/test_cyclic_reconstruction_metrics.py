#!/usr/bin/env python3
"""Unit tests for the shared Atom37 reconstruction / cyclization geometry metrics.

Pure-tensor tests: no GPU, no dataset, no model required. Covers
`proteinfoundation.eval.cyclic_reconstruction_metrics`.

Usage (as a standalone script):
    python script_utils/test_cyclic_reconstruction_metrics.py

Usage (via pytest, if installed):
    pytest script_utils/test_cyclic_reconstruction_metrics.py -v
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
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    CYCLIC_METRIC_SUFFIXES,
    RECON_METRIC_SUFFIXES,
    cyclic_geometry_metrics,
    extract_cyclization_metadata,
    get_atom_index,
    masked_atom37_coord_metrics,
    sequence_metrics,
)

AA_OTHER = 0  # never CYS/LYS/ASP/GLU
N_IDX, CA_IDX, C_IDX, CB_IDX = 0, 1, 2, 3
SG_IDX, NZ_IDX, CG_IDX, CD_IDX = 10, 35, 5, 11


def _zeros_atom37(batch_size: int, length: int):
    coords = torch.zeros(batch_size, length, 37, 3)
    mask = torch.zeros(batch_size, length, 37, dtype=torch.bool)
    return coords, mask


def _uniform_aa(batch_size: int, length: int, fill: int = AA_OTHER) -> torch.Tensor:
    return torch.full((batch_size, length), fill, dtype=torch.long)


# ---------------------------------------------------------------------------
# A. get_atom_index
# ---------------------------------------------------------------------------
def test_get_atom_index_matches_canonical_atom37_order():
    assert get_atom_index("N") == 0
    assert get_atom_index("CA") == 1
    assert get_atom_index("C") == 2
    assert get_atom_index("CB") == 3
    assert get_atom_index("SG") == 10
    assert get_atom_index("NZ") == 35
    assert get_atom_index("CG") == 5
    assert get_atom_index("CD") == 11


# ---------------------------------------------------------------------------
# B. masked_atom37_coord_metrics
# ---------------------------------------------------------------------------
def test_masked_coord_metrics_known_offsets():
    gt, mask = _zeros_atom37(1, 2)
    pred = gt.clone()

    # residue 0: N and CA valid.
    mask[0, 0, N_IDX] = True
    mask[0, 0, CA_IDX] = True
    gt[0, 0, N_IDX] = torch.tensor([0.0, 0.0, 0.0])
    pred[0, 0, N_IDX] = torch.tensor([0.1, 0.0, 0.0])
    gt[0, 0, CA_IDX] = torch.tensor([1.0, 0.0, 0.0])
    pred[0, 0, CA_IDX] = torch.tensor([1.0, 0.2, 0.0])

    # residue 1: only CA valid.
    mask[0, 1, CA_IDX] = True
    gt[0, 1, CA_IDX] = torch.tensor([2.0, 0.0, 0.0])
    pred[0, 1, CA_IDX] = torch.tensor([2.0, 0.0, 0.3])

    out = masked_atom37_coord_metrics(pred, gt, mask, prefix="test")

    # residue0 abs-sum = 0.1 + 0.2 = 0.3 (nm); residue1 abs-sum = 0.3 (nm).
    # sum_xyz_l1_per_residue_nm = mean over structures of (sum over residues / n_valid_res)
    #                            = (0.3 + 0.3) / 2 = 0.3
    assert math.isclose(out["test/sum_xyz_l1_per_residue_nm"], 0.3, abs_tol=1e-5)

    # coord_mae_A = total_abs_err / n_valid_coords * 10; n_valid_coords = 3 atoms * 3 = 9.
    total_abs_err = 0.1 + 0.2 + 0.3
    expected_mae_A = total_abs_err / 9.0 * 10.0
    assert math.isclose(out["test/coord_mae_A"], expected_mae_A, abs_tol=1e-5)

    # atom_rmse_A = sqrt(total_sq_err / n_valid_coords) * 10
    total_sq_err = 0.1**2 + 0.2**2 + 0.3**2
    expected_rmse_A = math.sqrt(total_sq_err / 9.0) * 10.0
    assert math.isclose(out["test/atom_rmse_A"], expected_rmse_A, abs_tol=1e-5)

    assert math.isclose(out["test/valid_atom_count_mean"], 3.0, abs_tol=1e-6)
    assert math.isclose(out["test/valid_residue_count_mean"], 2.0, abs_tol=1e-6)


def test_masked_coord_metrics_zero_error_is_zero():
    gt, mask = _zeros_atom37(2, 3)
    mask[:, :, CA_IDX] = True
    gt[:, :, CA_IDX] = torch.randn(2, 3, 3)
    pred = gt.clone()

    out = masked_atom37_coord_metrics(pred, gt, mask, prefix="test")
    assert out["test/sum_xyz_l1_per_residue_nm"] == 0.0
    assert out["test/coord_mae_A"] == 0.0
    assert out["test/atom_rmse_A"] == 0.0


# ---------------------------------------------------------------------------
# C. sequence_metrics
# ---------------------------------------------------------------------------
def test_sequence_metrics_partial_mask_and_exact_match():
    pred_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    gt_tokens = torch.tensor([[1, 2, 9, 4], [5, 6, 7, 8]])
    residue_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    out = sequence_metrics(pred_tokens, gt_tokens, residue_mask, prefix="seq")

    # sample 0: 2/3 correct (position 2 wrong, position 3 masked out); not exact match.
    # sample 1: 4/4 correct; exact match.
    expected_acc = (2.0 / 3.0 + 1.0) / 2.0
    assert math.isclose(out["seq/seq_acc"], expected_acc, abs_tol=1e-6)
    assert math.isclose(out["seq/seq_exact_match"], 0.5, abs_tol=1e-6)


def test_sequence_metrics_logits_argmax():
    gt_tokens = torch.tensor([[0, 1]])
    residue_mask = torch.tensor([[True, True]])
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 0] = 10.0  # argmax -> 0, correct
    logits[0, 1, 2] = 10.0  # argmax -> 2, wrong (gt=1)

    out = sequence_metrics(logits, gt_tokens, residue_mask, prefix="seq")
    assert math.isclose(out["seq/seq_acc"], 0.5, abs_tol=1e-6)
    assert out["seq/seq_exact_match"] == 0.0


def test_sequence_metrics_exact_match_requires_all_valid_residues():
    gt_tokens = torch.tensor([[1, 1, 1]])
    pred_tokens = torch.tensor([[1, 1, 9]])
    # Only the first two residues are valid; the (wrong) third is padding.
    residue_mask = torch.tensor([[True, True, False]])
    out = sequence_metrics(pred_tokens, gt_tokens, residue_mask, prefix="seq")
    assert out["seq/seq_exact_match"] == 1.0
    assert out["seq/seq_acc"] == 1.0


# ---------------------------------------------------------------------------
# D. extract_cyclization_metadata
# ---------------------------------------------------------------------------
def test_extract_cyclization_metadata_none_when_absent():
    assert extract_cyclization_metadata({}) is None
    assert extract_cyclization_metadata({"coords_nm": torch.zeros(1, 2, 37, 3)}) is None


def test_extract_cyclization_metadata_defaults_has_cyclization_true():
    batch = {
        "cyclization_i": torch.tensor([0]),
        "cyclization_j": torch.tensor([2]),
        "cyclization_type": torch.tensor([MAINCHAIN]),
    }
    meta = extract_cyclization_metadata(batch)
    assert meta is not None
    assert meta["has_cyclization"].item() is True


# ---------------------------------------------------------------------------
# E. cyclic_geometry_metrics: missing metadata -> full NaN key set (not missing keys)
# ---------------------------------------------------------------------------
def test_cyclic_geometry_metrics_none_metadata_returns_all_nan_keys():
    gt, mask = _zeros_atom37(1, 3)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 3)

    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, None, prefix="cyc")
    expected_keys = {f"cyc/{s}" for s in CYCLIC_METRIC_SUFFIXES}
    assert set(out.keys()) == expected_keys

    measurements = {k: v for k, v in out.items() if not k.startswith("cyc/n_valid_")}
    assert all(v != v for v in measurements.values()), "every measurement should be NaN when metadata is None"


def test_cyclic_geometry_metrics_counts_are_zero_never_nan():
    """The `n_valid_*` counts are aggregation weights, so a NaN there would reintroduce the
    NaN-poisoned epoch mean they exist to prevent. "No examples" must be 0, not NaN."""
    gt, mask = _zeros_atom37(1, 3)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 3)

    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, None, prefix="cyc")
    counts = {k: v for k, v in out.items() if k.startswith("cyc/n_valid_")}
    assert len(counts) == 4
    assert all(v == 0.0 for v in counts.values()), f"counts must be 0, not NaN: {counts}"


def test_recon_metric_suffixes_count():
    # 5 coord + 2 seq + 22 cyclic = 29 (single source of truth used by the W&B schema).
    # 22 cyclic = 18 measurements + 4 `n_valid_*` aggregation-weight counts.
    assert len(RECON_METRIC_SUFFIXES) == 29
    assert len(CYCLIC_METRIC_SUFFIXES) == 22


# ---------------------------------------------------------------------------
# F. Disulfide geometry
# ---------------------------------------------------------------------------
def _disulfide_batch(pred_sg_dist_nm: float, gt_sg_dist_nm: float = 0.204):
    gt, mask = _zeros_atom37(1, 2)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 2)
    seq_tokens[0, 0] = AA_CYS
    seq_tokens[0, 1] = AA_CYS

    mask[0, :, SG_IDX] = True
    gt[0, 0, SG_IDX] = torch.tensor([0.0, 0.0, 0.0])
    gt[0, 1, SG_IDX] = torch.tensor([0.0, 0.0, gt_sg_dist_nm])
    pred[0, 0, SG_IDX] = torch.tensor([0.0, 0.0, 0.0])
    pred[0, 1, SG_IDX] = torch.tensor([0.0, 0.0, pred_sg_dist_nm])

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([1]),
        "type": torch.tensor([DISULFIDE]),
        "has_cyclization": torch.tensor([True]),
    }
    return pred, gt, mask, seq_tokens, meta


def test_disulfide_bond_success_inside_window():
    pred, gt, mask, seq_tokens, meta = _disulfide_batch(pred_sg_dist_nm=0.20)  # 2.0 A
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert math.isclose(out["cyc/disulfide_sg_dist_pred_A"], 2.0, abs_tol=1e-4)
    assert out["cyc/disulfide_bond_success"] == 1.0


def test_disulfide_bond_success_outside_window():
    pred, gt, mask, seq_tokens, meta = _disulfide_batch(pred_sg_dist_nm=0.50)  # 5.0 A
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert math.isclose(out["cyc/disulfide_sg_dist_pred_A"], 5.0, abs_tol=1e-4)
    assert out["cyc/disulfide_bond_success"] == 0.0


def test_disulfide_nan_when_endpoints_not_cysteine():
    pred, gt, mask, seq_tokens, meta = _disulfide_batch(pred_sg_dist_nm=0.20)
    seq_tokens[0, 1] = AA_LYS  # not a CYS-CYS pair anymore
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert out["cyc/disulfide_sg_dist_pred_A"] != out["cyc/disulfide_sg_dist_pred_A"]  # NaN
    assert out["cyc/disulfide_bond_success"] != out["cyc/disulfide_bond_success"]  # NaN


# ---------------------------------------------------------------------------
# G. Isopeptide geometry (Lys-Asp and Lys-Glu, both orderings)
# ---------------------------------------------------------------------------
def test_isopeptide_lys_asp_uses_cg_and_succeeds():
    gt, mask = _zeros_atom37(1, 2)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 2)
    seq_tokens[0, 0] = AA_LYS
    seq_tokens[0, 1] = AA_ASP

    mask[0, 0, NZ_IDX] = True
    mask[0, 1, CG_IDX] = True
    gt[0, 0, NZ_IDX] = torch.tensor([0.0, 0.0, 0.0])
    gt[0, 1, CG_IDX] = torch.tensor([0.0, 0.0, 0.132])
    pred[0, 0, NZ_IDX] = torch.tensor([0.0, 0.0, 0.0])
    pred[0, 1, CG_IDX] = torch.tensor([0.0, 0.0, 0.132])  # 1.32 A, inside [1.1, 1.6]

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([1]),
        "type": torch.tensor([ISOPEPTIDE]),
        "has_cyclization": torch.tensor([True]),
    }
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert math.isclose(out["cyc/isopeptide_n_c_dist_pred_A"], 1.32, abs_tol=1e-4)
    assert out["cyc/isopeptide_bond_success"] == 1.0


def test_isopeptide_lys_glu_uses_cd():
    gt, mask = _zeros_atom37(1, 2)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 2)
    seq_tokens[0, 0] = AA_LYS
    seq_tokens[0, 1] = AA_GLU

    mask[0, 0, NZ_IDX] = True
    mask[0, 1, CD_IDX] = True
    gt[0, 0, NZ_IDX] = torch.tensor([0.0, 0.0, 0.0])
    gt[0, 1, CD_IDX] = torch.tensor([0.0, 0.0, 0.132])
    pred[0, 0, NZ_IDX] = torch.tensor([0.0, 0.0, 0.0])
    pred[0, 1, CD_IDX] = torch.tensor([0.0, 0.0, 0.132])

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([1]),
        "type": torch.tensor([ISOPEPTIDE]),
        "has_cyclization": torch.tensor([True]),
    }
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert math.isclose(out["cyc/isopeptide_n_c_dist_pred_A"], 1.32, abs_tol=1e-4)
    assert out["cyc/isopeptide_bond_success"] == 1.0


def test_isopeptide_reversed_order_lys_at_j():
    # LYS at the higher-index endpoint (j), acid at the lower-index endpoint (i).
    gt, mask = _zeros_atom37(1, 2)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 2)
    seq_tokens[0, 0] = AA_ASP
    seq_tokens[0, 1] = AA_LYS

    mask[0, 0, CG_IDX] = True
    mask[0, 1, NZ_IDX] = True
    gt[0, 0, CG_IDX] = torch.tensor([0.0, 0.0, 0.0])
    gt[0, 1, NZ_IDX] = torch.tensor([0.0, 0.0, 0.132])
    pred[0, 0, CG_IDX] = torch.tensor([0.0, 0.0, 0.0])
    pred[0, 1, NZ_IDX] = torch.tensor([0.0, 0.0, 0.132])

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([1]),
        "type": torch.tensor([ISOPEPTIDE]),
        "has_cyclization": torch.tensor([True]),
    }
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert math.isclose(out["cyc/isopeptide_n_c_dist_pred_A"], 1.32, abs_tol=1e-4)
    assert out["cyc/isopeptide_bond_success"] == 1.0


def test_isopeptide_unsupported_chemistry_is_nan():
    gt, mask = _zeros_atom37(1, 2)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 2)  # neither LYS nor ASP/GLU
    mask[0, :, NZ_IDX] = True

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([1]),
        "type": torch.tensor([ISOPEPTIDE]),
        "has_cyclization": torch.tensor([True]),
    }
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert out["cyc/isopeptide_n_c_dist_pred_A"] != out["cyc/isopeptide_n_c_dist_pred_A"]


# ---------------------------------------------------------------------------
# H. Mainchain geometry with orientation ambiguity (min-distance logic)
# ---------------------------------------------------------------------------
def test_mainchain_min_distance_picks_the_real_bond():
    # Head-to-tail closure: N of the N-terminus (i) bonds to C of the C-terminus (j).
    # The naive "C_i - N_j" orientation is the *unrelated*, far-apart pair here.
    gt, mask = _zeros_atom37(1, 2)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 2)

    mask[0, 0, N_IDX] = True
    mask[0, 0, C_IDX] = True
    mask[0, 1, N_IDX] = True
    mask[0, 1, C_IDX] = True

    # Real bond: N(i) <-> C(j), ~1.3 A.
    gt[0, 0, N_IDX] = torch.tensor([0.0, 0.0, 0.0])
    gt[0, 1, C_IDX] = torch.tensor([0.0, 0.0, 0.13])
    pred[0, 0, N_IDX] = torch.tensor([0.0, 0.0, 0.0])
    pred[0, 1, C_IDX] = torch.tensor([0.0, 0.0, 0.13])

    # Unrelated pair: C(i) <-> N(j), far apart (not a bond).
    gt[0, 0, C_IDX] = torch.tensor([5.0, 0.0, 0.0])
    gt[0, 1, N_IDX] = torch.tensor([5.0, 0.0, 5.0])
    pred[0, 0, C_IDX] = torch.tensor([5.0, 0.0, 0.0])
    pred[0, 1, N_IDX] = torch.tensor([5.0, 0.0, 5.0])

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([1]),
        "type": torch.tensor([MAINCHAIN]),
        "has_cyclization": torch.tensor([True]),
    }
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert math.isclose(out["cyc/mainchain_cn_dist_pred_A"], 1.3, abs_tol=1e-3)
    assert out["cyc/mainchain_cn_bond_success"] == 1.0


def test_mainchain_nan_for_non_mainchain_type():
    gt, mask = _zeros_atom37(1, 2)
    pred = gt.clone()
    seq_tokens = _uniform_aa(1, 2)
    mask[0, :, N_IDX] = True
    mask[0, :, C_IDX] = True

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([1]),
        "type": torch.tensor([DISULFIDE]),  # not mainchain
        "has_cyclization": torch.tensor([True]),
    }
    out = cyclic_geometry_metrics(pred, gt, mask, seq_tokens, meta, prefix="cyc")
    assert out["cyc/mainchain_cn_dist_pred_A"] != out["cyc/mainchain_cn_dist_pred_A"]


ALL_TESTS = [
    test_get_atom_index_matches_canonical_atom37_order,
    test_masked_coord_metrics_known_offsets,
    test_masked_coord_metrics_zero_error_is_zero,
    test_sequence_metrics_partial_mask_and_exact_match,
    test_sequence_metrics_logits_argmax,
    test_sequence_metrics_exact_match_requires_all_valid_residues,
    test_extract_cyclization_metadata_none_when_absent,
    test_extract_cyclization_metadata_defaults_has_cyclization_true,
    test_cyclic_geometry_metrics_none_metadata_returns_all_nan_keys,
    test_recon_metric_suffixes_count,
    test_disulfide_bond_success_inside_window,
    test_disulfide_bond_success_outside_window,
    test_disulfide_nan_when_endpoints_not_cysteine,
    test_isopeptide_lys_asp_uses_cg_and_succeeds,
    test_isopeptide_lys_glu_uses_cd,
    test_isopeptide_reversed_order_lys_at_j,
    test_isopeptide_unsupported_chemistry_is_nan,
    test_mainchain_min_distance_picks_the_real_bond,
    test_mainchain_nan_for_non_mainchain_type,
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
