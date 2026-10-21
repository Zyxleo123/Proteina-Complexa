#!/usr/bin/env python3
"""Does `cyc_cb_*` survive a model that designs its own sequence?

`cyclic_geometry_metrics` takes ONE `atom37_mask` and applies it to both the predicted and
the ground-truth coordinates. Its docstring states the assumption plainly: "used for both
pred and gt, since predicted structures are decoded at the same residue/atom slots", and
`seq_tokens` is documented as "ground-truth residue type ids".

That assumption holds for a RECONSTRUCTION evaluation, where the predicted sequence is the
ground-truth sequence. It does not hold for GENERATION, where the model decodes its own
residue types: the mask then says "Cb exists here" because the *ground truth* residue has
one, while the *predicted* residue at that position may be a glycine whose Cb slot was
never filled. `get_cb_position` trusts the mask, reads the unpopulated slot, and returns
whatever is in it.

This was not hypothetical. On 12 unconditional `cpsea_bondunroll_from_v4cfg` samples the
logged `val_gen/cyc/cyc_cb_window_success` was **0.0** while a direct measurement of the
terminal Cb-Cb distance on the very same written PDBs put 9/12 inside the 3-8 A window.
The logged mean distance was 6.069 A with 0 % inside 3-8 A -- a mean inside the window with
nothing in it, which is the signature of a bimodal mix of near-zero and far-out values
rather than of a real distribution.

The tests below isolate the mechanism, so a fix can be checked against them:

1. `test_metric_is_correct_when_sequences_agree` -- the reconstruction case works.
2. `test_unfilled_pred_cb_slot_corrupts_the_metric` -- zeroing only the PREDICTED Cb slot,
   exactly as a decoded glycine at a non-glycine ground-truth position would leave it,
   moves the reported distance off the true value and flips the window verdict. The
   ground-truth distance is unchanged, so the corruption is attributable to the shared
   mask alone.
3. `test_mainchain_cn_is_unaffected` -- N and C are always decoded, so the N-C criterion
   does not share this failure. That is why N-C remains usable where Cb does not.

Usage:
    python script_utils/test_cyclic_cb_metric_mask.py
    pytest script_utils/test_cyclic_cb_metric_mask.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.eval.cyclic_reconstruction_metrics import (  # noqa: E402
    CA_IDX,
    CB_IDX,
    C_IDX,
    N_IDX,
    cyclic_geometry_metrics,
    virtual_cb_from_backbone,
)

NM = 0.1  # 1 A in nm
MAINCHAIN = 0


def _residue_frame() -> dict[int, torch.Tensor]:
    """A single non-degenerate residue: N, CA, C at a realistic N-CA-C angle, plus Cb.

    The backbone must NOT be colinear. `virtual_cb_from_backbone` builds its frame from
    `cross(CA - N, C - CA)`, which is the zero vector for a straight backbone -- so a
    colinear test residue makes the virtual Cb meaningless and cannot distinguish a
    working fallback from a broken one. (This bit the first version of this test.)

    Cb is placed AT the virtual position, which is what a real residue looks like: the
    idealized and actual Cb coincide to within a few hundredths of an Angstrom, so the
    fallback is expected to reproduce the real Cb rather than merely be "some point".
    """
    import math

    n = torch.tensor([0.0, 0.0, 0.0])
    ca = torch.tensor([1.458 * NM, 0.0, 0.0])
    ang = math.radians(111.0)  # N-CA-C
    c = ca + torch.tensor([1.525 * NM * math.cos(ang), 1.525 * NM * math.sin(ang), 0.0])
    cb = virtual_cb_from_backbone(n, ca, c)
    return {N_IDX: n, CA_IDX: ca, C_IDX: c, CB_IDX: cb}


def _build(length: int = 5, cb_gap_A: float = 5.5, nc_gap_A: float | None = None):
    """One peptide whose terminal Cb-Cb distance is set by construction.

    Every residue is the same rigid frame, translated. Because the frame is rigid, the
    terminal Cb-Cb distance equals the translation between the two end residues, so
    `cb_gap_A` is exact. `nc_gap_A`, when given, overrides C of the last residue to set
    the head-to-tail N-C distance directly (that path never touches Cb).
    """
    frame = _residue_frame()
    atom37 = torch.zeros(1, length, 37, 3)
    mask = torch.zeros(1, length, 37, dtype=torch.bool)

    for r in range(length):
        # The last residue is translated to put its Cb exactly `cb_gap_A` from residue 0's.
        shift = torch.tensor([cb_gap_A * NM, 0.0, 0.0]) if r == length - 1 else torch.tensor(
            [r * 3.8 * NM, 0.0, 0.0]
        )
        for idx, pos in frame.items():
            atom37[0, r, idx] = pos + shift
        mask[0, r, [N_IDX, CA_IDX, C_IDX, CB_IDX]] = True

    if nc_gap_A is not None:
        atom37[0, length - 1, C_IDX] = atom37[0, 0, N_IDX] + torch.tensor(
            [nc_gap_A * NM, 0.0, 0.0]
        )

    meta = {
        "i": torch.tensor([0]),
        "j": torch.tensor([length - 1]),
        "type": torch.tensor([MAINCHAIN]),
        "has_cyclization": torch.tensor([True]),
    }
    # Ground-truth residue types: anything non-glycine, so the mask claims a real Cb.
    seq = torch.full((1, length), 4, dtype=torch.long)
    return atom37, mask, seq, meta


def test_metric_is_correct_when_sequences_agree() -> None:
    """The reconstruction case: pred == gt, so the shared mask is valid."""
    gt, mask, seq, meta = _build(cb_gap_A=5.5)
    pred = gt.clone()
    out = cyclic_geometry_metrics(pred, gt, mask, seq, meta, prefix="t")
    assert out["t/cyc_cb_dist_pred_A"] == pytest.approx(5.5, abs=0.05), out["t/cyc_cb_dist_pred_A"]
    assert out["t/cyc_cb_window_success"] == pytest.approx(1.0)


def test_unfilled_pred_cb_slot_falls_back_to_virtual_cb() -> None:
    """Zero ONLY the predicted Cb slots -- what a decoded glycine leaves behind.

    Before the fix in `get_cb_position`, the mask's claim that a Cb exists was trusted,
    the empty slot was read, and the terminal distance came back 0.000 A with window
    success 0.0 against a true separation of 5.5 A. The fix requires a slot to be both
    masked-in AND populated, falling back to the virtual Cb from that structure's own
    backbone -- which is always decoded.

    The ground truth is untouched here, so anything that moves is attributable to the
    shared mask alone.
    """
    gt, mask, seq, meta = _build(cb_gap_A=5.5)
    pred = gt.clone()
    pred[0, 0, CB_IDX] = 0.0
    pred[0, -1, CB_IDX] = 0.0

    out = cyclic_geometry_metrics(pred, gt, mask, seq, meta, prefix="t")

    assert out["t/cyc_cb_dist_gt_A"] == pytest.approx(5.5, abs=0.05), (
        f"ground truth moved: {out['t/cyc_cb_dist_gt_A']}"
    )
    reported = out["t/cyc_cb_dist_pred_A"]
    assert reported == pytest.approx(5.5, abs=0.15), (
        f"predicted terminal Cb-Cb reported as {reported:.3f} A; the virtual-Cb fallback "
        f"should recover ~5.5 A from the predicted backbone"
    )
    assert out["t/cyc_cb_window_success"] == pytest.approx(1.0), (
        "a correctly-placed ring was still failed by the window"
    )


def test_a_genuinely_open_ring_is_still_failed() -> None:
    """The fallback must not rescue a ring that is actually open.

    A guard that made everything pass would be worse than the bug it replaced, so the
    same construction at a 14 A separation must still fail the 3-8 A window.
    """
    gt, mask, seq, meta = _build(cb_gap_A=14.0)
    pred = gt.clone()
    pred[0, 0, CB_IDX] = 0.0
    pred[0, -1, CB_IDX] = 0.0
    out = cyclic_geometry_metrics(pred, gt, mask, seq, meta, prefix="t")
    assert out["t/cyc_cb_dist_pred_A"] == pytest.approx(14.0, abs=0.2), (
        out["t/cyc_cb_dist_pred_A"]
    )
    assert out["t/cyc_cb_window_success"] == pytest.approx(0.0)


def test_mainchain_cn_is_unaffected() -> None:
    """N and C are always decoded, so the N-C criterion does not share the failure."""
    gt, mask, seq, meta = _build(nc_gap_A=1.33)
    pred = gt.clone()
    pred[0, 0, CB_IDX] = 0.0
    pred[0, -1, CB_IDX] = 0.0
    out = cyclic_geometry_metrics(pred, gt, mask, seq, meta, prefix="t")
    assert out["t/mainchain_cn_dist_pred_A"] == pytest.approx(1.33, abs=0.05), (
        out["t/mainchain_cn_dist_pred_A"]
    )
    assert out["t/mainchain_cn_bond_success"] == pytest.approx(1.0)


ALL_TESTS = [
    test_metric_is_correct_when_sequences_agree,
    test_unfilled_pred_cb_slot_falls_back_to_virtual_cb,
    test_a_genuinely_open_ring_is_still_failed,
    test_mainchain_cn_is_unaffected,
]


if __name__ == "__main__":
    print(f"running {len(ALL_TESTS)} cyc_cb mask checks\n")
    failures = []
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(fn.__name__)
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
