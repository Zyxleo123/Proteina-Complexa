#!/usr/bin/env python3
"""Unit tests for the CPSea cyclization-linkage prediction head.

Pure-tensor tests: no GPU, no dataset, no full model required. Covers the
validity mask, the global CE loss, and inference decoding described in
`proteinfoundation.cyclization`.

Usage (as a standalone script):
    python script_utils/test_cyclization.py

Usage (via pytest, if installed):
    pytest script_utils/test_cyclization.py -v
"""

from __future__ import annotations

import torch

from proteinfoundation.cyclization.constants import (
    AA_ASN,
    AA_ASP,
    AA_CYS,
    AA_GLN,
    AA_GLU,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
    NO_CYCLIZATION_INDEX,
    NUM_CYCLIZATION_TYPES,
    UNSPECIFIED,
)
from proteinfoundation.cyclization.loss import cyclization_link_loss, decode_cyclization_prediction
from proteinfoundation.cyclization.mask import build_cyclization_validity_mask

AA_OTHER = 0  # ALA, never CYS/LYS/ASP/GLU/ASN/GLN


def _uniform_aa(batch_size: int, length: int, fill: int = AA_OTHER) -> torch.Tensor:
    return torch.full((batch_size, length), fill, dtype=torch.long)


def _full_mask(batch_size: int, length: int) -> torch.Tensor:
    return torch.ones(batch_size, length, dtype=torch.bool)


# ---------------------------------------------------------------------------
# A. Validity mask shape and symmetry behavior
# ---------------------------------------------------------------------------
def test_mask_shape_and_triangle():
    B, L = 2, 6
    aa = _uniform_aa(B, L)
    binder_mask = _full_mask(B, L)

    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask)

    assert valid.shape == (B, L, L, NUM_CYCLIZATION_TYPES), valid.shape
    assert valid.dtype == torch.bool

    # Diagonal always invalid, for every type.
    diag = valid[:, torch.arange(L), torch.arange(L), :]
    assert not diag.any(), "diagonal should never be valid"

    # Lower triangle (i > j) always invalid.
    lower = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)
    assert not valid[:, lower, :].any(), "lower triangle should never be valid"

    # Only the strict upper triangle can ever be valid.
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    any_valid_type = valid.any(dim=-1)  # [B, L, L]
    assert torch.equal(any_valid_type, any_valid_type & upper[None, :, :])


# ---------------------------------------------------------------------------
# B. Mainchain mask
# ---------------------------------------------------------------------------
def test_mainchain_mask_only_termini():
    B, L = 1, 7
    aa = _uniform_aa(B, L)
    binder_mask = _full_mask(B, L)

    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask)
    mainchain = valid[..., MAINCHAIN]  # [B, L, L]

    assert mainchain[0, 0, L - 1].item() is True
    mainchain_clone = mainchain.clone()
    mainchain_clone[0, 0, L - 1] = False
    assert not mainchain_clone.any(), "no other (i, j) pair should be valid for MAINCHAIN"


def test_mainchain_mask_respects_binder_length_per_sample():
    # Two samples in the batch with different real binder lengths (padding at the end).
    B, L = 2, 8
    aa = _uniform_aa(B, L)
    binder_mask = torch.zeros(B, L, dtype=torch.bool)
    binder_mask[0, :5] = True  # length 5 -> terminal pair (0, 4)
    binder_mask[1, :8] = True  # length 8 -> terminal pair (0, 7)

    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask)
    mainchain = valid[..., MAINCHAIN]

    assert mainchain[0, 0, 4].item() is True
    assert mainchain[1, 0, 7].item() is True
    assert mainchain[0].sum().item() == 1
    assert mainchain[1].sum().item() == 1


# ---------------------------------------------------------------------------
# C. Disulfide mask
# ---------------------------------------------------------------------------
def test_disulfide_mask_cys_cys_only():
    B, L = 1, 5
    aa = _uniform_aa(B, L)
    aa[0, 1] = AA_CYS
    aa[0, 3] = AA_CYS
    binder_mask = _full_mask(B, L)

    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask)
    disulfide = valid[..., DISULFIDE]

    assert disulfide[0, 1, 3].item() is True
    disulfide_clone = disulfide.clone()
    disulfide_clone[0, 1, 3] = False
    assert not disulfide_clone.any(), "only the CYS-CYS pair should be valid for DISULFIDE"


def test_disulfide_mask_rejects_non_cys():
    B, L = 1, 4
    aa = _uniform_aa(B, L)  # no CYS at all
    binder_mask = _full_mask(B, L)

    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask)
    assert not valid[..., DISULFIDE].any()


# ---------------------------------------------------------------------------
# D. Isopeptide mask
# ---------------------------------------------------------------------------
def test_isopeptide_mask_lys_asp_and_lys_glu():
    B, L = 1, 6
    aa = _uniform_aa(B, L)
    aa[0, 0] = AA_LYS
    aa[0, 2] = AA_ASP
    aa[0, 4] = AA_GLU
    binder_mask = _full_mask(B, L)

    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask, allow_asn_gln_isopeptide=False)
    isopeptide = valid[..., ISOPEPTIDE]

    assert isopeptide[0, 0, 2].item() is True  # LYS-ASP
    assert isopeptide[0, 0, 4].item() is True  # LYS-GLU
    # Unrelated pair: (2, 4) is ASP-GLU, not LYS-anything.
    assert isopeptide[0, 2, 4].item() is False


def test_isopeptide_mask_asn_gln_flag():
    B, L = 1, 4
    aa = _uniform_aa(B, L)
    aa[0, 0] = AA_LYS
    aa[0, 2] = AA_ASN
    aa[0, 3] = AA_GLN
    binder_mask = _full_mask(B, L)

    valid_off = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask, allow_asn_gln_isopeptide=False)
    assert not valid_off[..., ISOPEPTIDE].any()

    valid_on = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask, allow_asn_gln_isopeptide=True)
    assert valid_on[0, 0, 2, ISOPEPTIDE].item() is True
    assert valid_on[0, 0, 3, ISOPEPTIDE].item() is True


# ---------------------------------------------------------------------------
# E. Gold force-valid
# ---------------------------------------------------------------------------
def test_force_gold_valid_overrides_hand_coded_chemistry():
    B, L = 1, 5
    aa = _uniform_aa(B, L)  # no CYS/LYS/acid anywhere: gold below violates every rule
    binder_mask = _full_mask(B, L)

    gold_i = torch.tensor([1])
    gold_j = torch.tensor([3])
    gold_type = torch.tensor([DISULFIDE])

    valid_without_force = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, gold_i=gold_i, gold_j=gold_j, gold_type=gold_type, force_gold_valid=False
    )
    assert not valid_without_force[0, 1, 3, DISULFIDE].item()

    valid_with_force = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, gold_i=gold_i, gold_j=gold_j, gold_type=gold_type, force_gold_valid=True
    )
    assert valid_with_force[0, 1, 3, DISULFIDE].item() is True


def test_force_gold_valid_sorts_reversed_gold_pair():
    B, L = 1, 5
    aa = _uniform_aa(B, L)
    binder_mask = _full_mask(B, L)

    # Gold given as (j, i) with j > i: function should sort internally so (i, j) with i<j is set.
    gold_i = torch.tensor([3])
    gold_j = torch.tensor([1])
    gold_type = torch.tensor([ISOPEPTIDE])

    valid = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, gold_i=gold_i, gold_j=gold_j, gold_type=gold_type, force_gold_valid=True
    )
    assert valid[0, 1, 3, ISOPEPTIDE].item() is True
    assert not valid[0, 3, 1, ISOPEPTIDE].item()


# ---------------------------------------------------------------------------
# F. Loss
# ---------------------------------------------------------------------------
def test_loss_high_gold_logit_gives_low_loss():
    B, L = 2, 5
    link_logits = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES)
    valid_mask = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES, dtype=torch.bool)

    gold_i = torch.tensor([0, 1])
    gold_j = torch.tensor([4, 3])
    gold_type = torch.tensor([MAINCHAIN, DISULFIDE])
    has_cyclization = torch.tensor([True, True])

    for b in range(B):
        valid_mask[b, gold_i[b], gold_j[b], gold_type[b]] = True
        valid_mask[b, 0, 1, MAINCHAIN] = True  # a distractor candidate
        link_logits[b, gold_i[b], gold_j[b], gold_type[b]] = 20.0

    loss, metrics = cyclization_link_loss(
        link_logits=link_logits, valid_mask=valid_mask, gold_i=gold_i, gold_j=gold_j, gold_type=gold_type,
        has_cyclization=has_cyclization,
    )
    assert loss.item() < 1e-3, f"expected near-zero loss, got {loss.item()}"
    assert metrics["top1_exact_acc"] == 1.0
    assert metrics["top1_pair_acc"] == 1.0
    assert metrics["top1_type_acc"] == 1.0
    assert metrics["mean_gold_prob"] > 0.99


def test_loss_invalid_candidates_do_not_win():
    B, L = 1, 4
    link_logits = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES)
    valid_mask = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES, dtype=torch.bool)

    gold_i = torch.tensor([0])
    gold_j = torch.tensor([1])
    gold_type = torch.tensor([MAINCHAIN])
    has_cyclization = torch.tensor([True])

    valid_mask[0, 0, 1, MAINCHAIN] = True
    # Put an enormous logit on an INVALID candidate; masking should still win.
    link_logits[0, 2, 3, DISULFIDE] = 1e6
    link_logits[0, 0, 1, MAINCHAIN] = 1.0

    loss, metrics = cyclization_link_loss(
        link_logits=link_logits, valid_mask=valid_mask, gold_i=gold_i, gold_j=gold_j, gold_type=gold_type,
        has_cyclization=has_cyclization,
    )
    assert metrics["top1_exact_acc"] == 1.0
    assert torch.isfinite(loss)


def test_loss_ignores_samples_without_cyclization():
    B, L = 2, 4
    link_logits = torch.randn(B, L, L, NUM_CYCLIZATION_TYPES)
    valid_mask = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES, dtype=torch.bool)
    valid_mask[:, 0, 1, MAINCHAIN] = True

    gold_i = torch.tensor([0, 0])
    gold_j = torch.tensor([1, 1])
    gold_type = torch.tensor([MAINCHAIN, MAINCHAIN])
    has_cyclization = torch.tensor([True, False])

    # Corrupt the excluded sample's logits with NaN: if the implementation
    # accidentally included it, the loss would become NaN.
    link_logits[1] = float("nan")

    loss, metrics = cyclization_link_loss(
        link_logits=link_logits, valid_mask=valid_mask, gold_i=gold_i, gold_j=gold_j, gold_type=gold_type,
        has_cyclization=has_cyclization,
    )
    assert torch.isfinite(loss), "excluded (has_cyclization=False) sample must not contaminate the loss"
    assert metrics["n_valid_examples"] == 1.0


def test_loss_all_missing_batch_returns_zero_without_crashing():
    B, L = 3, 4
    link_logits = torch.randn(B, L, L, NUM_CYCLIZATION_TYPES, requires_grad=True)
    valid_mask = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES, dtype=torch.bool)
    gold_i = torch.zeros(B, dtype=torch.long)
    gold_j = torch.zeros(B, dtype=torch.long)
    gold_type = torch.zeros(B, dtype=torch.long)
    has_cyclization = torch.zeros(B, dtype=torch.bool)

    loss, metrics = cyclization_link_loss(
        link_logits=link_logits, valid_mask=valid_mask, gold_i=gold_i, gold_j=gold_j, gold_type=gold_type,
        has_cyclization=has_cyclization,
    )
    assert loss.item() == 0.0
    assert metrics["n_valid_examples"] == 0.0
    # Must stay part of the autograd graph (differentiable zero), not a bare python float.
    loss.backward()


# ---------------------------------------------------------------------------
# G. Inference decode
# ---------------------------------------------------------------------------
def test_decode_flat_index_roundtrip():
    B, L = 2, 6
    link_logits = torch.full((B, L, L, NUM_CYCLIZATION_TYPES), -100.0)
    valid_mask = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES, dtype=torch.bool)

    target = {0: (2, 5, DISULFIDE), 1: (0, 3, ISOPEPTIDE)}
    for b, (i, j, t) in target.items():
        valid_mask[b, i, j, t] = True
        link_logits[b, i, j, t] = 50.0

    pred = decode_cyclization_prediction(link_logits, valid_mask)
    for b, (i, j, t) in target.items():
        assert pred["pred_cyclization_i"][b].item() == i
        assert pred["pred_cyclization_j"][b].item() == j
        assert pred["pred_cyclization_type"][b].item() == t
        assert pred["pred_cyclization_confidence"][b].item() > 0.99


def test_decode_falls_back_to_terminal_mainchain_when_no_valid_candidate():
    B, L = 1, 5
    link_logits = torch.randn(B, L, L, NUM_CYCLIZATION_TYPES)
    valid_mask = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES, dtype=torch.bool)  # nothing valid

    pred = decode_cyclization_prediction(link_logits, valid_mask)
    assert pred["pred_cyclization_i"][0].item() == 0
    assert pred["pred_cyclization_j"][0].item() == L - 1
    assert pred["pred_cyclization_type"][0].item() == MAINCHAIN


# ---------------------------------------------------------------------------
# H. Type conditioning (cond_type)
# ---------------------------------------------------------------------------
def test_cond_type_none_and_unspecified_reproduce_unconditional_mask():
    # Regression guard: the conditional code path must not perturb existing behavior.
    B, L = 2, 6
    aa = _uniform_aa(B, L)
    aa[0, 1] = AA_CYS
    aa[0, 3] = AA_CYS
    aa[1, 0] = AA_LYS
    aa[1, 4] = AA_GLU
    binder_mask = _full_mask(B, L)

    baseline = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask)
    explicit_none = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask, cond_type=None)
    unspecified = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, cond_type=torch.full((B,), UNSPECIFIED, dtype=torch.long)
    )

    assert torch.equal(baseline, explicit_none)
    assert torch.equal(baseline, unspecified), "UNSPECIFIED must recover the unconditional candidate set"


def test_cond_type_restricts_to_requested_type_slice():
    B, L = 1, 6
    aa = _uniform_aa(B, L)
    aa[0, 1] = AA_CYS
    aa[0, 3] = AA_CYS  # a valid disulfide candidate exists
    binder_mask = _full_mask(B, L)

    unconditional = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask)
    # Unconditionally, the terminal MAINCHAIN candidate is also valid.
    assert unconditional[0, 0, L - 1, MAINCHAIN].item() is True

    valid = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, cond_type=torch.tensor([DISULFIDE])
    )
    assert not valid[..., MAINCHAIN].any(), "MAINCHAIN slice must be empty when DISULFIDE is requested"
    assert not valid[..., ISOPEPTIDE].any()
    # The disulfide candidate survives, unchanged.
    assert valid[0, 1, 3, DISULFIDE].item() is True
    assert torch.equal(valid[..., DISULFIDE], unconditional[..., DISULFIDE])


def test_cond_type_is_per_row_in_a_mixed_batch():
    # A dropped row (UNSPECIFIED) and a conditioned row must coexist in one batch --
    # this is exactly what cyclization_type_dropout_rate produces every step.
    B, L = 2, 5
    aa = _uniform_aa(B, L)
    aa[:, 1] = AA_CYS
    aa[:, 3] = AA_CYS
    binder_mask = _full_mask(B, L)

    valid = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, cond_type=torch.tensor([DISULFIDE, UNSPECIFIED])
    )
    # Row 0 is conditioned: only DISULFIDE survives.
    assert not valid[0, ..., MAINCHAIN].any()
    assert valid[0, 1, 3, DISULFIDE].item() is True
    # Row 1 is unconditional: the terminal MAINCHAIN candidate is still available.
    assert valid[1, 0, L - 1, MAINCHAIN].item() is True


def test_cond_type_unsatisfiable_request_yields_empty_mask():
    # Disulfide requested, but the sequence has no CYS at all.
    B, L = 1, 5
    aa = _uniform_aa(B, L)
    binder_mask = _full_mask(B, L)

    valid = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, cond_type=torch.tensor([DISULFIDE])
    )
    assert not valid.any(), "an unsupportable request must leave no candidate, not silently fall back"


def test_force_gold_valid_still_applies_under_conditioning():
    # cond_type narrowing runs BEFORE force_gold_valid, so an imperfect chemistry rule
    # still cannot mask out the true label on a conditioned row.
    B, L = 1, 5
    aa = _uniform_aa(B, L)  # no CYS: the gold disulfide below violates the rules
    binder_mask = _full_mask(B, L)

    valid = build_cyclization_validity_mask(
        aa=aa,
        binder_mask=binder_mask,
        gold_i=torch.tensor([1]),
        gold_j=torch.tensor([3]),
        gold_type=torch.tensor([DISULFIDE]),
        force_gold_valid=True,
        cond_type=torch.tensor([DISULFIDE]),
    )
    assert valid[0, 1, 3, DISULFIDE].item() is True
    assert not valid[..., MAINCHAIN].any(), "conditioning must still suppress the other type slices"


def test_conditional_loss_equals_p_ij_given_type():
    # The whole design rests on this: narrowing the mask to one type slice turns the
    # existing global (i, j, type) softmax into exactly p(i, j | type). Verify against
    # a hand-computed conditional cross-entropy.
    import math

    B, L = 1, 4
    torch.manual_seed(0)
    link_logits = torch.randn(B, L, L, NUM_CYCLIZATION_TYPES)

    aa = _uniform_aa(B, L)
    aa[0, 0] = AA_CYS
    aa[0, 1] = AA_CYS
    aa[0, 2] = AA_CYS  # three CYS => three competing disulfide candidates
    binder_mask = _full_mask(B, L)

    gold_i, gold_j = torch.tensor([0]), torch.tensor([1])
    gold_type = torch.tensor([DISULFIDE])

    valid_mask = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, cond_type=gold_type, allow_asn_gln_isopeptide=False
    )
    loss, _ = cyclization_link_loss(
        link_logits=link_logits,
        valid_mask=valid_mask,
        gold_i=gold_i,
        gold_j=gold_j,
        gold_type=gold_type,
        has_cyclization=torch.tensor([True]),
    )

    # Hand-computed: softmax over ONLY the valid disulfide candidates (the CYS-CYS pairs
    # in the strict upper triangle), i.e. the conditional distribution over (i, j).
    candidates = [(0, 1), (0, 2), (1, 2)]
    logits = [link_logits[0, i, j, DISULFIDE].item() for i, j in candidates]
    m = max(logits)
    log_z = m + math.log(sum(math.exp(v - m) for v in logits))
    expected = log_z - link_logits[0, 0, 1, DISULFIDE].item()

    assert abs(loss.item() - expected) < 1e-4, f"got {loss.item()}, expected p(i,j|type) CE {expected}"


def test_decode_without_fallback_nulls_unsatisfiable_row():
    # Under type conditioning the MAINCHAIN fallback must NOT fire: returning
    # (0, L-1, MAINCHAIN) when a disulfide was requested would silently report a
    # different type than the caller asked for.
    B, L = 2, 5
    link_logits = torch.randn(B, L, L, NUM_CYCLIZATION_TYPES)
    valid_mask = torch.zeros(B, L, L, NUM_CYCLIZATION_TYPES, dtype=torch.bool)
    valid_mask[0, 1, 3, DISULFIDE] = True  # row 0 satisfiable, row 1 has nothing

    pred = decode_cyclization_prediction(link_logits, valid_mask, fallback=False)

    assert pred["pred_cyclization_i"][0].item() == 1
    assert pred["pred_cyclization_j"][0].item() == 3
    assert pred["pred_cyclization_type"][0].item() == DISULFIDE

    assert pred["pred_cyclization_i"][1].item() == NO_CYCLIZATION_INDEX
    assert pred["pred_cyclization_j"][1].item() == NO_CYCLIZATION_INDEX
    assert pred["pred_cyclization_type"][1].item() == NO_CYCLIZATION_INDEX
    assert pred["pred_cyclization_confidence"][1].item() == 0.0


# ---------------------------------------------------------------------------
# Bonus: CONECT-based label parsing (fallback path, see parse_labels.py)
# ---------------------------------------------------------------------------
def _write_pdb(tmp_path, atom_lines, conect_lines):
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".pdb", dir=tmp_path)
    with os.fdopen(fd, "w") as f:
        for line in atom_lines:
            f.write(line + "\n")
        for line in conect_lines:
            f.write(line + "\n")
        f.write("END\n")
    return path


def _atom_line(serial, name, resname, chain, resseq, x=0.0, y=0.0, z=0.0):
    # Fixed-width PDB ATOM record; columns must line up with the slicing used
    # by `parse_labels.read_atom_conect_records` (serial=[6:11], name=[12:16],
    # resname=[17:20], chain=[21], resseq=[22:26]).
    return f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"


def test_parse_labels_disulfide_from_conect():
    import tempfile

    from proteinfoundation.cyclization.parse_labels import infer_cyclization_label

    with tempfile.TemporaryDirectory() as tmp_dir:
        atom_lines = [
            _atom_line(1, "N", "CYS", "B", 1),
            _atom_line(2, "SG", "CYS", "B", 1),
            _atom_line(3, "N", "ALA", "B", 2),
            _atom_line(4, "N", "CYS", "B", 3),
            _atom_line(5, "SG", "CYS", "B", 3),
        ]
        conect_lines = ["CONECT    2    5", "CONECT    5    2"]
        path = _write_pdb(tmp_dir, atom_lines, conect_lines)

        label = infer_cyclization_label(
            pdb_path=path,
            residue_pdb_idx=[1, 2, 3],
            binder_length_hint=3,
            binder_chain_id="B",
            cyclization_type_hint="other",
        )
        assert label["has_cyclization"] is True
        assert label["type"] == DISULFIDE
        assert (label["i"], label["j"]) == (0, 2)


def test_parse_labels_isopeptide_from_conect():
    import tempfile

    from proteinfoundation.cyclization.parse_labels import infer_cyclization_label

    with tempfile.TemporaryDirectory() as tmp_dir:
        atom_lines = [
            _atom_line(1, "NZ", "LYS", "B", 1),
            _atom_line(2, "N", "ALA", "B", 2),
            _atom_line(3, "OD1", "ASP", "B", 3),
        ]
        conect_lines = ["CONECT    1    3", "CONECT    3    1"]
        path = _write_pdb(tmp_dir, atom_lines, conect_lines)

        label = infer_cyclization_label(
            pdb_path=path,
            residue_pdb_idx=[1, 2, 3],
            binder_length_hint=3,
            binder_chain_id="B",
            cyclization_type_hint="other",
        )
        assert label["has_cyclization"] is True
        assert label["type"] == ISOPEPTIDE
        assert (label["i"], label["j"]) == (0, 2)


def test_parse_labels_hint_alone_without_conect_returns_no_label():
    # A metadata hint by itself (no PDB / CONECT to confirm the actual bond)
    # must never be enough to assign a type -- we never guess from position
    # or metadata alone.
    from proteinfoundation.cyclization.parse_labels import infer_cyclization_label

    label = infer_cyclization_label(
        pdb_path=None,
        residue_pdb_idx=[1, 2, 3, 4],
        binder_length_hint=4,
        binder_chain_id="B",
        cyclization_type_hint="head_tail",
    )
    assert label["has_cyclization"] is False


def test_parse_labels_mainchain_from_conect_bond():
    import tempfile

    from proteinfoundation.cyclization.parse_labels import infer_cyclization_label

    with tempfile.TemporaryDirectory() as tmp_dir:
        atom_lines = [
            _atom_line(1, "N", "ALA", "B", 1),
            _atom_line(2, "C", "ALA", "B", 1),
            _atom_line(3, "N", "GLY", "B", 2),
            _atom_line(4, "N", "SER", "B", 3),
            _atom_line(5, "C", "SER", "B", 3),
        ]
        # Terminal closure bond: C of residue 3 (last) <-> N of residue 1 (first).
        conect_lines = ["CONECT    5    1", "CONECT    1    5"]
        path = _write_pdb(tmp_dir, atom_lines, conect_lines)

        label = infer_cyclization_label(
            pdb_path=path,
            residue_pdb_idx=[1, 2, 3],
            binder_length_hint=3,
            binder_chain_id="B",
            cyclization_type_hint=None,
        )
        assert label["has_cyclization"] is True
        assert label["type"] == MAINCHAIN
        assert (label["i"], label["j"]) == (0, 2)


def test_parse_labels_type_never_inferred_from_hint_or_position_alone():
    # Even if metadata claims "head_tail", the actual observed bond (a
    # disulfide here, deliberately NOT at the termini) must win: type is
    # determined by bond-atom identity, never by the hint or by position.
    import tempfile

    from proteinfoundation.cyclization.parse_labels import infer_cyclization_label

    with tempfile.TemporaryDirectory() as tmp_dir:
        atom_lines = [
            _atom_line(1, "N", "ALA", "B", 1),
            _atom_line(2, "SG", "CYS", "B", 2),
            _atom_line(3, "N", "ALA", "B", 3),
            _atom_line(4, "SG", "CYS", "B", 4),
            _atom_line(5, "N", "ALA", "B", 5),
        ]
        conect_lines = ["CONECT    2    4", "CONECT    4    2"]
        path = _write_pdb(tmp_dir, atom_lines, conect_lines)

        label = infer_cyclization_label(
            pdb_path=path,
            residue_pdb_idx=[1, 2, 3, 4, 5],
            binder_length_hint=5,
            binder_chain_id="B",
            cyclization_type_hint="head_tail",  # deliberately misleading
        )
        assert label["has_cyclization"] is True
        assert label["type"] == DISULFIDE  # bond identity wins, not the hint
        assert (label["i"], label["j"]) == (1, 3)


def test_parse_labels_isopeptide_via_side_chain_carbon():
    # Real CPSea CONECT records bond LYS:NZ to the side-chain carbonyl carbon
    # (CG for ASP, not the oxygens) -- must still resolve to ISOPEPTIDE.
    import tempfile

    from proteinfoundation.cyclization.parse_labels import infer_cyclization_label

    with tempfile.TemporaryDirectory() as tmp_dir:
        atom_lines = [
            _atom_line(1, "NZ", "LYS", "B", 1),
            _atom_line(2, "N", "ALA", "B", 2),
            _atom_line(3, "CG", "ASP", "B", 3),
        ]
        conect_lines = ["CONECT    1    3", "CONECT    3    1"]
        path = _write_pdb(tmp_dir, atom_lines, conect_lines)

        label = infer_cyclization_label(
            pdb_path=path,
            residue_pdb_idx=[1, 2, 3],
            binder_length_hint=3,
            binder_chain_id="B",
            cyclization_type_hint="other",
        )
        assert label["has_cyclization"] is True
        assert label["type"] == ISOPEPTIDE
        assert (label["i"], label["j"]) == (0, 2)


def test_parse_labels_no_conect_records_returns_no_label():
    import tempfile

    from proteinfoundation.cyclization.parse_labels import infer_cyclization_label

    with tempfile.TemporaryDirectory() as tmp_dir:
        atom_lines = [_atom_line(1, "N", "ALA", "B", 1), _atom_line(2, "N", "ALA", "B", 2)]
        path = _write_pdb(tmp_dir, atom_lines, [])

        label = infer_cyclization_label(
            pdb_path=path,
            residue_pdb_idx=[1, 2],
            binder_length_hint=2,
            binder_chain_id="B",
            cyclization_type_hint="other",
        )
        assert label["has_cyclization"] is False


ALL_TESTS = [
    test_mask_shape_and_triangle,
    test_mainchain_mask_only_termini,
    test_mainchain_mask_respects_binder_length_per_sample,
    test_disulfide_mask_cys_cys_only,
    test_disulfide_mask_rejects_non_cys,
    test_isopeptide_mask_lys_asp_and_lys_glu,
    test_isopeptide_mask_asn_gln_flag,
    test_force_gold_valid_overrides_hand_coded_chemistry,
    test_force_gold_valid_sorts_reversed_gold_pair,
    test_loss_high_gold_logit_gives_low_loss,
    test_loss_invalid_candidates_do_not_win,
    test_loss_ignores_samples_without_cyclization,
    test_loss_all_missing_batch_returns_zero_without_crashing,
    test_decode_flat_index_roundtrip,
    test_decode_falls_back_to_terminal_mainchain_when_no_valid_candidate,
    test_cond_type_none_and_unspecified_reproduce_unconditional_mask,
    test_cond_type_restricts_to_requested_type_slice,
    test_cond_type_is_per_row_in_a_mixed_batch,
    test_cond_type_unsatisfiable_request_yields_empty_mask,
    test_force_gold_valid_still_applies_under_conditioning,
    test_conditional_loss_equals_p_ij_given_type,
    test_decode_without_fallback_nulls_unsatisfiable_row,
    test_parse_labels_disulfide_from_conect,
    test_parse_labels_isopeptide_from_conect,
    test_parse_labels_hint_alone_without_conect_returns_no_label,
    test_parse_labels_mainchain_from_conect_bond,
    test_parse_labels_type_never_inferred_from_hint_or_position_alone,
    test_parse_labels_isopeptide_via_side_chain_carbon,
    test_parse_labels_no_conect_records_returns_no_label,
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
