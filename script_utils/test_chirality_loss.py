#!/usr/bin/env python3
"""Unit tests for `proteinfoundation.cyclization.chirality_loss`.

Pure-tensor tests: no GPU, no dataset, no model required.

Usage (as a standalone script):
    python script_utils/test_chirality_loss.py

Usage (via pytest, if installed):
    pytest script_utils/test_chirality_loss.py -v
"""

from __future__ import annotations

import math

import torch

from proteinfoundation.cyclization.chirality_loss import chirality_loss, chirality_triple_product
from proteinfoundation.eval.cyclic_reconstruction_metrics import CA_IDX, CB_IDX, C_IDX, N_IDX, virtual_cb_from_backbone


def _zeros_atom37(batch_size: int, length: int):
    coords = torch.zeros(batch_size, length, 37, 3)
    mask = torch.zeros(batch_size, length, 37, dtype=torch.bool)
    return coords, mask


def _place_residue(atom37, mask, b, r, n, ca, c, cb):
    atom37[b, r, N_IDX] = n
    atom37[b, r, CA_IDX] = ca
    atom37[b, r, C_IDX] = c
    atom37[b, r, CB_IDX] = cb
    mask[b, r, N_IDX] = True
    mask[b, r, CA_IDX] = True
    mask[b, r, C_IDX] = True
    mask[b, r, CB_IDX] = True


def _l_and_d_frames():
    n = torch.tensor([0.0, 0.0, 0.0])
    ca = torch.tensor([1.46, 0.0, 0.0])
    c = torch.tensor([2.0, 1.4, 0.0])
    cb_l = virtual_cb_from_backbone(n, ca, c)
    cb_d = 2 * ca - cb_l  # reflect through CA -> flips the stereocenter, exactly test_score_cyclization's fixture
    return n, ca, c, cb_l, cb_d


# ---------------------------------------------------------------------------
# A. chirality_triple_product matches the boolean gate's sign convention.
# ---------------------------------------------------------------------------
def test_triple_product_positive_for_l_negative_for_d():
    n, ca, c, cb_l, cb_d = _l_and_d_frames()
    atom37, mask = _zeros_atom37(1, 2)
    _place_residue(atom37, mask, 0, 0, n, ca, c, cb_l)
    _place_residue(atom37, mask, 0, 1, n, ca, c, cb_d)

    triple, checkable = chirality_triple_product(atom37, mask)
    assert bool(checkable[0, 0]) and bool(checkable[0, 1])
    assert float(triple[0, 0]) > 0.0
    assert float(triple[0, 1]) < 0.0


def test_not_checkable_when_cb_missing():
    atom37, mask = _zeros_atom37(1, 1)
    n, ca, c, cb_l, _ = _l_and_d_frames()
    _place_residue(atom37, mask, 0, 0, n, ca, c, cb_l)
    mask[0, 0, CB_IDX] = False  # e.g. glycine
    _, checkable = chirality_triple_product(atom37, mask)
    assert not bool(checkable[0, 0])


# ---------------------------------------------------------------------------
# B. chirality_loss: zero for all-L, positive+differentiable for D residues.
# ---------------------------------------------------------------------------
def test_loss_is_zero_for_all_l_backbone():
    n, ca, c, cb_l, _ = _l_and_d_frames()
    atom37, mask = _zeros_atom37(1, 3)
    for r in range(3):
        _place_residue(atom37, mask, 0, r, n, ca, c, cb_l)
    binder_mask = torch.ones(1, 3, dtype=torch.bool)

    loss, metrics = chirality_loss(atom37, mask, binder_mask)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-6)
    assert metrics["n_valid"] == 3.0
    assert metrics["frac_d"] == 0.0


def test_loss_is_positive_when_some_residues_are_d():
    n, ca, c, cb_l, cb_d = _l_and_d_frames()
    atom37, mask = _zeros_atom37(1, 3)
    _place_residue(atom37, mask, 0, 0, n, ca, c, cb_l)
    _place_residue(atom37, mask, 0, 1, n, ca, c, cb_d)
    _place_residue(atom37, mask, 0, 2, n, ca, c, cb_l)
    binder_mask = torch.ones(1, 3, dtype=torch.bool)

    loss, metrics = chirality_loss(atom37, mask, binder_mask)
    assert float(loss) > 0.0
    assert math.isclose(metrics["frac_d"], 1.0 / 3.0, rel_tol=1e-6)


def test_loss_gradient_reaches_coordinates():
    """The whole point of this module: unlike the boolean gate, this must backprop."""
    n, ca, c, cb_l, cb_d = _l_and_d_frames()
    atom37, mask = _zeros_atom37(1, 1)
    _place_residue(atom37, mask, 0, 0, n, ca, c, cb_d)
    atom37 = atom37.clone().requires_grad_(True)
    binder_mask = torch.ones(1, 1, dtype=torch.bool)

    loss, _ = chirality_loss(atom37, mask, binder_mask)
    loss.backward()
    assert atom37.grad is not None
    assert torch.any(atom37.grad[0, 0, CB_IDX] != 0.0)


def test_loss_respects_binder_mask():
    """A D-configured residue outside binder_mask (e.g. padding) must not contribute."""
    n, ca, c, cb_l, cb_d = _l_and_d_frames()
    atom37, mask = _zeros_atom37(1, 2)
    _place_residue(atom37, mask, 0, 0, n, ca, c, cb_l)
    _place_residue(atom37, mask, 0, 1, n, ca, c, cb_d)
    binder_mask = torch.tensor([[True, False]])

    loss, metrics = chirality_loss(atom37, mask, binder_mask)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-6)
    assert metrics["n_valid"] == 1.0


def test_loss_zero_and_finite_when_nothing_checkable():
    atom37, mask = _zeros_atom37(1, 2)  # no atoms placed -> nothing checkable
    binder_mask = torch.ones(1, 2, dtype=torch.bool)
    loss, metrics = chirality_loss(atom37, mask, binder_mask)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-8)
    assert metrics["n_valid"] == 0.0
    assert math.isnan(metrics["frac_d"])


def test_t_weight_zeroes_out_low_t_samples():
    n, ca, c, cb_l, cb_d = _l_and_d_frames()
    atom37, mask = _zeros_atom37(2, 1)
    _place_residue(atom37, mask, 0, 0, n, ca, c, cb_d)
    _place_residue(atom37, mask, 1, 0, n, ca, c, cb_d)
    binder_mask = torch.ones(2, 1, dtype=torch.bool)

    t_weight = torch.tensor([0.0, 1.0])
    loss, metrics = chirality_loss(atom37, mask, binder_mask, t_weight=t_weight)
    # Sample 0 is fully gated out of both the numerator AND the weighted denominator
    # (n_valid), so the mean-reduced loss should equal computing over sample 1 alone,
    # not merely be diluted by an uncounted zero-weight sample.
    loss_sample1_only, _ = chirality_loss(atom37[1:], mask[1:], binder_mask[1:])
    assert math.isclose(float(loss), float(loss_sample1_only), rel_tol=1e-5)
    assert metrics["n_valid"] == 1.0


def test_margin_shrinks_zero_penalty_region():
    """A small positive margin must penalize a borderline-but-technically-L residue
    that margin=0 (the exact gate boundary) would score as zero loss."""
    n, ca, c, cb_l, _ = _l_and_d_frames()
    atom37, mask = _zeros_atom37(1, 1)
    _place_residue(atom37, mask, 0, 0, n, ca, c, cb_l)
    binder_mask = torch.ones(1, 1, dtype=torch.bool)
    triple, _ = chirality_triple_product(atom37, mask)

    loss_zero_margin, _ = chirality_loss(atom37, mask, binder_mask, margin=0.0)
    loss_with_margin, _ = chirality_loss(atom37, mask, binder_mask, margin=float(triple[0, 0]) + 1.0)
    assert math.isclose(float(loss_zero_margin), 0.0, abs_tol=1e-6)
    assert float(loss_with_margin) > 0.0


ALL_TESTS = [
    test_triple_product_positive_for_l_negative_for_d,
    test_not_checkable_when_cb_missing,
    test_loss_is_zero_for_all_l_backbone,
    test_loss_is_positive_when_some_residues_are_d,
    test_loss_gradient_reaches_coordinates,
    test_loss_respects_binder_mask,
    test_loss_zero_and_finite_when_nothing_checkable,
    test_t_weight_zeroes_out_low_t_samples,
    test_margin_shrinks_zero_penalty_region,
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
