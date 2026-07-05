#!/usr/bin/env python3
"""Non-GPU validation checks for segment-aware CPSea feature construction.

Validates the behavior described in docs/README_DATA_SEQ_BREAK_ISSUE.md:
  - target REAL_BREAKs (and NUMBERING_ONLY jumps) create new segment_id values
  - pairwise seq_sep is all-zero across a target segment boundary
  - pairwise seq_sep is all-zero for binder-target pairs
  - backbone torsions crossing a target break are invalid/null (all-zero one-hot)
  - a clean binder chain has B_REAL_BREAK = 0 and B_NUMBERING_ONLY = 0 (and the
    break detector does flag a broken one, i.e. the check isn't vacuous)
  - ordinary binder-binder seq_sep is unchanged from before (no effective_chain_id
    dependence when there is only one segment)

Usage:
    source env.sh
    python script_utils/test_segment_aware_features.py
"""

from __future__ import annotations

import torch

from proteinfoundation.datasets.segment_utils import (
    BREAK_CLEAN,
    BREAK_NEW_CHAIN,
    BREAK_NONE,
    BREAK_NUMBERING_ONLY,
    BREAK_REAL_BREAK,
    compute_segment_info,
    summarize_breaks,
)
from proteinfoundation.nn.feature_factory.pair_feats import (
    ChainIdxPairFeat,
    CrossSequenceRelativeSequenceSeparationPairFeat,
    SequenceSeparationPairFeat,
)
from proteinfoundation.nn.feature_factory.seq_feats import BackboneTorsionAnglesSeqFeat
from proteinfoundation.datasets.transforms import (
    Data,
    ExtractTargetCoordinatesTransform,
    SegmentAwareResidueFeaturesTransform,
)

N_IDX, CA_IDX, C_IDX = 0, 1, 2


def _checks(name):
    print(f"\n>>> {name}")


def _ok(msg):
    print(f"  OK {msg}")


def make_synthetic_structure():
    """Builds a synthetic 9-residue structure (mirrors the docstring example):

    Chain A (target), 6 residues, 3 continuous segments:
        res 0,1,2: segment 0 (clean, pdb idx 1,2,3)
        -- REAL_BREAK (C-N distance ~100 A, consecutive numbering 3->4) --
        res 3,4:   segment 1 (clean, pdb idx 4,5)
        -- NUMBERING_ONLY (normal C-N bond, but pdb idx jumps 5->10) --
        res 5:     segment 2 (pdb idx 10)
    Chain B (binder), 3 residues, contiguous (pdb idx 1,2,3).

    Returns a dict with chains, residue_pdb_idx, coords, coord_mask (unbatched, [n, ...]).
    """
    n = 9
    chains = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    residue_pdb_idx = torch.tensor([1, 2, 3, 4, 5, 10, 1, 2, 3], dtype=torch.long)

    coords = torch.zeros(n, 37, 3)
    coord_mask = torch.zeros(n, 37, dtype=torch.bool)
    coord_mask[:, [N_IDX, CA_IDX, C_IDX]] = True

    x = 0.0
    # jump in x right before a residue to simulate a physical (REAL_BREAK) discontinuity;
    # small step keeps a normal ~1 A "peptide bond" from the previous residue's C.
    x_jump_before = {3: 100.0}
    for i in range(n):
        if i in x_jump_before:
            x += x_jump_before[i]
        coords[i, N_IDX] = torch.tensor([x, 0.0, 0.0])
        coords[i, CA_IDX] = torch.tensor([x + 1.0, 0.0, 0.0])
        coords[i, C_IDX] = torch.tensor([x + 2.0, 0.0, 0.0])
        x += 3.0  # next residue's N is 1 A away from this residue's C

    return {
        "chains": chains,
        "residue_pdb_idx": residue_pdb_idx,
        "coords": coords,
        "coord_mask": coord_mask,
    }


def test_segment_detection():
    _checks("compute_segment_info: target REAL_BREAK/NUMBERING_ONLY create new segments")
    s = make_synthetic_structure()
    info = compute_segment_info(s["chains"], s["residue_pdb_idx"], s["coords"], s["coord_mask"])

    expected_break_type = [
        BREAK_NONE,
        BREAK_CLEAN,
        BREAK_CLEAN,
        BREAK_REAL_BREAK,
        BREAK_CLEAN,
        BREAK_NUMBERING_ONLY,
        BREAK_NEW_CHAIN,
        BREAK_CLEAN,
        BREAK_CLEAN,
    ]
    assert info.break_type == expected_break_type, f"break_type mismatch: {info.break_type}"
    _ok("break_type classification matches expected REAL_BREAK/NUMBERING_ONLY/CLEAN/NEW_CHAIN pattern")

    expected_segment_id = torch.tensor([0, 0, 0, 1, 1, 2, 0, 0, 0])
    assert torch.equal(info.segment_id, expected_segment_id), info.segment_id
    _ok("segment_id increments at each break, resets at chain change")

    expected_pos_in_segment = torch.tensor([0, 1, 2, 0, 1, 0, 0, 1, 2])
    assert torch.equal(info.pos_in_segment, expected_pos_in_segment), info.pos_in_segment
    _ok("pos_in_segment resets to 0 at each new segment")

    # effective_chain_id must be constant within a segment and different across segments/chains.
    expected_groups = [{0, 1, 2}, {3, 4}, {5}, {6, 7, 8}]
    eff = info.effective_chain_id.tolist()
    for group in expected_groups:
        vals = {eff[i] for i in group}
        assert len(vals) == 1, f"effective_chain_id not constant within group {group}: {eff}"
    distinct_per_group = {next(iter({eff[i] for i in group})) for group in expected_groups}
    assert len(distinct_per_group) == len(expected_groups), f"effective_chain_id not unique per group: {eff}"
    _ok("effective_chain_id is constant within each (chain, segment) group and unique across groups")

    # Target chain (0) counts: 1 REAL_BREAK, 1 NUMBERING_ONLY.
    target_counts = summarize_breaks(info.break_type, s["chains"], chain_id_of_interest=0)
    assert target_counts.get(BREAK_REAL_BREAK, 0) == 1, target_counts
    assert target_counts.get(BREAK_NUMBERING_ONLY, 0) == 1, target_counts
    _ok("summarize_breaks reports REAL_BREAK=1, NUMBERING_ONLY=1 for the target chain")

    # Binder chain (1) must be clean: B_REAL_BREAK = 0, B_NUMBERING_ONLY = 0.
    binder_counts = summarize_breaks(info.break_type, s["chains"], chain_id_of_interest=1)
    assert binder_counts.get(BREAK_REAL_BREAK, 0) == 0, binder_counts
    assert binder_counts.get(BREAK_NUMBERING_ONLY, 0) == 0, binder_counts
    _ok("summarize_breaks reports B_REAL_BREAK=0, B_NUMBERING_ONLY=0 for the (clean) binder chain")

    return info, s


def test_missing_atom_is_real_break():
    _checks("compute_segment_info: missing C/N atom is classified as REAL_BREAK")
    chains = torch.tensor([0, 0, 0], dtype=torch.long)
    residue_pdb_idx = torch.tensor([1, 2, 3], dtype=torch.long)
    coords = torch.zeros(3, 37, 3)
    coord_mask = torch.zeros(3, 37, dtype=torch.bool)
    coord_mask[:, [N_IDX, CA_IDX, C_IDX]] = True
    coord_mask[0, C_IDX] = False  # residue 0's C atom is missing
    for i in range(3):
        coords[i, N_IDX] = torch.tensor([3.0 * i, 0.0, 0.0])
        coords[i, CA_IDX] = torch.tensor([3.0 * i + 1.0, 0.0, 0.0])
        coords[i, C_IDX] = torch.tensor([3.0 * i + 2.0, 0.0, 0.0])

    info = compute_segment_info(chains, residue_pdb_idx, coords, coord_mask)
    assert info.break_type[1] == BREAK_REAL_BREAK, info.break_type
    assert info.segment_id.tolist() == [0, 1, 1], info.segment_id
    _ok("a missing C (or N) atom triggers REAL_BREAK and a new segment, even with consecutive numbering")


def test_seq_sep_null_across_target_break_and_for_binder_target():
    _checks("SequenceSeparationPairFeat: null across target break, unchanged within segment/binder")
    s = make_synthetic_structure()
    info = compute_segment_info(s["chains"], s["residue_pdb_idx"], s["coords"], s["coord_mask"])

    batch = {
        "residue_pdb_idx": s["residue_pdb_idx"].unsqueeze(0),  # [1, 9]
        "effective_chain_id": info.effective_chain_id.unsqueeze(0),  # [1, 9]
    }
    feat_fn = SequenceSeparationPairFeat(seq_sep_dim=21)
    feat = feat_fn(batch)  # [1, 9, 9, 21]

    def onehot_is_null(i, j):
        return torch.all(feat[0, i, j] == 0)

    # Within-segment target pairs (e.g. residues 0,1,2) must be non-null.
    assert not onehot_is_null(0, 1), "same-segment target pair unexpectedly null"
    assert not onehot_is_null(1, 2), "same-segment target pair unexpectedly null"
    _ok("target-target seq_sep is non-null (a real one-hot bin) within a continuous segment")

    # Across the REAL_BREAK (residues 2 <-> 3) and NUMBERING_ONLY (residues 4 <-> 5) boundaries: null.
    assert onehot_is_null(2, 3), "seq_sep across REAL_BREAK should be null"
    assert onehot_is_null(4, 5), "seq_sep across NUMBERING_ONLY break should be null"
    _ok("target-target seq_sep is all-zero (null) across both a REAL_BREAK and a NUMBERING_ONLY boundary")

    # Binder-binder (residues 6,7,8) unaffected: normal contiguous seq_sep.
    assert not onehot_is_null(6, 7), "binder-binder seq_sep unexpectedly null"
    _ok("binder-binder seq_sep is unaffected (non-null, single segment)")

    # Binder-target (any binder residue vs any target residue): must be null.
    assert onehot_is_null(0, 6), "binder-target seq_sep should be null"
    assert onehot_is_null(5, 8), "binder-target seq_sep should be null"
    _ok("binder-target seq_sep is all-zero (null)")


def test_chain_idx_pair_feat_segment_aware():
    _checks("ChainIdxPairFeat: segment-aware same/different chain-like object")
    s = make_synthetic_structure()
    info = compute_segment_info(s["chains"], s["residue_pdb_idx"], s["coords"], s["coord_mask"])
    batch = {"effective_chain_id": info.effective_chain_id.unsqueeze(0)}
    feat = ChainIdxPairFeat()(batch)  # [1, 9, 9, 1], 0 = same chain-like object, 1 = different

    assert feat[0, 0, 1, 0].item() == 0.0, "target-target within same segment should be 'same chain'"
    assert feat[0, 2, 3, 0].item() == 1.0, "target-target across REAL_BREAK should be 'different chain'"
    assert feat[0, 4, 5, 0].item() == 1.0, "target-target across NUMBERING_ONLY break should be 'different chain'"
    assert feat[0, 6, 7, 0].item() == 0.0, "binder-binder should be 'same chain'"
    assert feat[0, 0, 6, 0].item() == 1.0, "binder-target should be 'different chain'"
    _ok("chain_idx_pair correctly reflects same-segment/different-segment/different-chain in all 4 cases")


def test_ordinary_binder_binder_seq_sep_unchanged():
    _checks("SequenceSeparationPairFeat: ordinary binder-binder seq_sep unchanged from before")
    binder_pdb_idx = torch.tensor([[1, 2, 3]], dtype=torch.long)  # [1, 3], single contiguous segment
    feat_fn = SequenceSeparationPairFeat(seq_sep_dim=21)

    feat_before = feat_fn({"residue_pdb_idx": binder_pdb_idx})  # no effective_chain_id at all (pre-fix datasets)
    feat_after = feat_fn(
        {
            "residue_pdb_idx": binder_pdb_idx,
            "effective_chain_id": torch.zeros_like(binder_pdb_idx),  # single segment -> mask is all-ones
        }
    )
    assert torch.equal(feat_before, feat_after), "adding effective_chain_id changed single-segment seq_sep"
    _ok("presence of effective_chain_id does not alter seq_sep when there is only one segment")


def test_cross_sequence_target_seq_sep_segment_aware():
    _checks("CrossSequenceRelativeSequenceSeparationPairFeat: target-target segment-aware, binder-target null")
    target_pdb_idx = torch.tensor([[1, 2, 3, 8, 9]], dtype=torch.long)  # segment break between idx 2 and 3
    target_eff_chain = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long)
    dummy_seq = torch.zeros_like(target_pdb_idx)

    feat_fn = CrossSequenceRelativeSequenceSeparationPairFeat(
        seq1_key="seq_target",
        seq2_key="seq_target",
        idx1_key="target_pdb_idx",
        idx2_key="target_pdb_idx",
        effective_chain1_key="target_effective_chain_id",
        effective_chain2_key="target_effective_chain_id",
        seq_sep_dim=21,
    )
    batch = {
        "seq_target": dummy_seq,
        "target_pdb_idx": target_pdb_idx,
        "target_effective_chain_id": target_eff_chain,
        "coords": torch.zeros(1, 5, 37, 3),
    }
    feat = feat_fn(batch)  # [1, 5, 5, 21]
    assert torch.any(feat[0, 0, 1] != 0), "within-segment target-target seq_sep unexpectedly null"
    assert torch.all(feat[0, 2, 3] == 0), "cross-segment target-target seq_sep should be null"
    _ok("target-target cross-sequence seq_sep is segment-aware (null across the break)")

    # Binder-target block (different idx keys) is always null, independent of segment info.
    binder_target_feat_fn = CrossSequenceRelativeSequenceSeparationPairFeat(
        seq1_key="residue_type",
        seq2_key="seq_target",
        idx1_key="residue_pdb_idx",
        idx2_key="target_pdb_idx",
        seq_sep_dim=21,
    )
    binder_batch = {
        "residue_type": torch.zeros(1, 3, dtype=torch.long),
        "seq_target": dummy_seq,
        "residue_pdb_idx": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "target_pdb_idx": target_pdb_idx,
        "coords": torch.zeros(1, 3, 37, 3),
    }
    binder_target_feat = binder_target_feat_fn(binder_batch)
    assert torch.all(binder_target_feat == 0), "binder-target cross seq_sep should always be null"
    _ok("binder-target cross-sequence seq_sep is all-zero (null)")


def test_backbone_torsion_invalid_across_break():
    _checks("BackboneTorsionAnglesSeqFeat: invalid (all-zero) one-hot across a target break, valid within segment")
    s = make_synthetic_structure()
    info = compute_segment_info(s["chains"], s["residue_pdb_idx"], s["coords"], s["coord_mask"])

    batch = {
        "coords": s["coords"].unsqueeze(0),  # [1, 9, 37, 3]
        "effective_chain_id": info.effective_chain_id.unsqueeze(0),
        "pos_in_segment": info.pos_in_segment.unsqueeze(0),
    }
    feat_fn = BackboneTorsionAnglesSeqFeat()
    feat = feat_fn(batch)  # [1, 9, 63] (3 angles * 21 bins)
    feat = feat.reshape(1, 9, 3, 21)

    def is_all_zero(i):
        return torch.all(feat[0, i] == 0)

    # Residue 2 sits right before the REAL_BREAK (needs residue 3, invalid); residue 4 sits
    # right before the NUMBERING_ONLY break (needs residue 5, invalid). Both must be all-zero.
    assert is_all_zero(2), "torsion at residue just before REAL_BREAK should be null"
    assert is_all_zero(4), "torsion at residue just before NUMBERING_ONLY break should be null"
    _ok("backbone torsions spanning a target break are null (all-zero one-hot, not bin 0)")

    # Residue 0 (well within the first clean segment) must be valid: exactly one bin set per angle.
    counts = feat[0, 0].sum(dim=-1)  # [3], one-hot sum per angle
    assert torch.equal(counts, torch.ones(3)), f"expected a single set bin per angle, got sums {counts}"
    _ok("backbone torsions within a continuous segment remain valid one-hot vectors")

    # Last residue overall always has no "next" residue -> always null.
    assert is_all_zero(8), "last residue overall should always be null (no next residue)"
    _ok("last residue overall is null (no next-residue pair exists)")


def make_synthetic_data_object(binder_chain_id="B"):
    """Wraps `make_synthetic_structure` into an actual `Data` object (chain 0 -> 'A', chain 1 -> 'B'),
    the way it looks right after `atomarray_to_atom37`, i.e. before any atom37_transforms run."""
    s = make_synthetic_structure()
    n = s["chains"].shape[0]
    chain_id = ["A"] * 6 + ["B"] * 3
    return Data(
        chains=s["chains"],
        chain_id=chain_id,
        residue_pdb_idx=s["residue_pdb_idx"],
        coords=s["coords"],
        coords_nm=s["coords"] / 10.0,
        coord_mask=s["coord_mask"],
        residue_type=torch.zeros(n, dtype=torch.long),
        binder_chain_id=binder_chain_id,
        id="synthetic_test_sample",
    )


def test_segment_aware_transform_end_to_end():
    _checks("SegmentAwareResidueFeaturesTransform: sets fields on a real Data object")
    graph = make_synthetic_data_object()
    graph = SegmentAwareResidueFeaturesTransform(cn_break_cutoff=2.0)(graph)

    assert hasattr(graph, "segment_id") and hasattr(graph, "pos_in_segment") and hasattr(graph, "effective_chain_id")
    expected_segment_id = torch.tensor([0, 0, 0, 1, 1, 2, 0, 0, 0])
    assert torch.equal(graph.segment_id, expected_segment_id), graph.segment_id
    assert graph.effective_chain_id[6] == graph.effective_chain_id[7] == graph.effective_chain_id[8]
    assert graph.effective_chain_id[0] != graph.effective_chain_id[6]
    _ok("transform populates segment_id/pos_in_segment/effective_chain_id matching compute_segment_info")


def test_segment_aware_transform_flags_broken_binder():
    _checks("SegmentAwareResidueFeaturesTransform: detects it when the binder chain itself has a break")
    graph = make_synthetic_data_object()
    # Corrupt the binder chain's numbering (residue 8's pdb idx jumps 2 -> 50): a NUMBERING_ONLY
    # break inside the binder, which the transform should log a warning about (point 7/9).
    graph.residue_pdb_idx = graph.residue_pdb_idx.clone()
    graph.residue_pdb_idx[8] = 50
    SegmentAwareResidueFeaturesTransform(cn_break_cutoff=2.0)(graph)
    # The underlying detection must be non-vacuous: the binder chain now splits into two segments.
    assert graph.segment_id[7].item() != graph.segment_id[8].item(), "broken binder should split into two segments"
    _ok("a broken binder chain is detected (segment_id splits it, triggering the warning check)")


def test_extract_target_coordinates_propagates_segment_fields():
    _checks("ExtractTargetCoordinatesTransform: propagates target_effective_chain_id / target_pos_in_segment")
    graph = make_synthetic_data_object()
    graph = SegmentAwareResidueFeaturesTransform(cn_break_cutoff=2.0)(graph)
    # residues 0-5 (chain A) are the target; residues 6-8 (chain B) are the binder.
    graph.target_mask = torch.zeros(9, 37, dtype=torch.bool)
    graph.target_mask[:6, [N_IDX, CA_IDX, C_IDX]] = True

    graph = ExtractTargetCoordinatesTransform(compact_mode=True)(graph)

    assert hasattr(graph, "target_effective_chain_id"), "target_effective_chain_id missing after extraction"
    assert hasattr(graph, "target_pos_in_segment"), "target_pos_in_segment missing after extraction"
    assert graph.target_effective_chain_id.shape[0] == 6, graph.target_effective_chain_id.shape
    # Must exactly match the corresponding slice of the full-sequence fields.
    assert torch.equal(graph.target_effective_chain_id, graph.effective_chain_id[:6])
    assert torch.equal(graph.target_pos_in_segment, graph.pos_in_segment[:6])
    _ok("target_effective_chain_id/target_pos_in_segment correctly sliced out for target residues")


def test_segment_fix_level_tiers():
    _checks("segment_fix_level tiers: mild=chain only, aggressive=+seq_sep, full=+torsions")
    s = make_synthetic_structure()
    info = compute_segment_info(s["chains"], s["residue_pdb_idx"], s["coords"], s["coord_mask"])
    base = {
        "residue_pdb_idx": s["residue_pdb_idx"].unsqueeze(0),
        "effective_chain_id": info.effective_chain_id.unsqueeze(0),
        "coords": s["coords"].unsqueeze(0),
        "pos_in_segment": info.pos_in_segment.unsqueeze(0),
    }

    def seq_null_at_break(level):
        batch = {**base, "segment_fix_level": [level]}
        feat = SequenceSeparationPairFeat(seq_sep_dim=21)(batch)
        return torch.all(feat[0, 2, 3] == 0).item()

    def torsion_null_at_break(level):
        batch = {**base, "segment_fix_level": [level]}
        feat = BackboneTorsionAnglesSeqFeat()(batch).reshape(1, 9, 3, 21)
        return torch.all(feat[0, 2] == 0).item()

    assert not seq_null_at_break("mild"), "mild should not null seq_sep at breaks"
    assert seq_null_at_break("aggressive"), "aggressive should null seq_sep at breaks"
    assert seq_null_at_break("full"), "full should null seq_sep at breaks"
    _ok("seq_sep nulling only at aggressive/full tiers")

    assert not torsion_null_at_break("mild"), "mild should not null torsions at breaks"
    assert not torsion_null_at_break("aggressive"), "aggressive should not null torsions at breaks"
    assert torsion_null_at_break("full"), "full should null torsions at breaks"
    _ok("backbone torsion nulling only at full tier")


def main() -> int:
    test_segment_detection()
    test_missing_atom_is_real_break()
    test_seq_sep_null_across_target_break_and_for_binder_target()
    test_chain_idx_pair_feat_segment_aware()
    test_ordinary_binder_binder_seq_sep_unchanged()
    test_cross_sequence_target_seq_sep_segment_aware()
    test_backbone_torsion_invalid_across_break()
    test_segment_aware_transform_end_to_end()
    test_segment_aware_transform_flags_broken_binder()
    test_extract_target_coordinates_propagates_segment_fields()
    test_segment_fix_level_tiers()
    print("\nAll segment-aware feature validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
