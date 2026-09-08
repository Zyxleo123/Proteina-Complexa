#!/usr/bin/env python3
"""Tests for linear-peptide (LP) data mixing: the LINEAR topology token and the sampler.

Run: PYTHONPATH=. .venv/bin/python script_utils/test_lp_mixing.py

Two things are being protected here.

1. **LINEAR must not behave like a ring request anywhere.** The whole point of adding a
   fifth conditioning index instead of reusing UNSPECIFIED is that linear rows get their
   own token; the failure mode is a gate somewhere still written as `!= UNSPECIFIED`,
   which would hand a linear peptide a cycle graph asserting a bond it does not have.
   That failure is silent -- the run trains, the loss looks normal.

2. **The weighted sampler must actually produce the requested mix.** A 45k pool beside
   2.44M rows is 1.8% of an unweighted epoch. If the weighting silently no-ops (a missing
   column, a sampler Lightning replaced), the run looks exactly like the control it is
   supposed to differ from.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from proteinfoundation.cyclization.constants import (
    DISULFIDE,
    ISOPEPTIDE,
    LINEAR,
    MAINCHAIN,
    NUM_CYCLIZATION_COND_TYPES,
    NUM_CYCLIZATION_TYPES,
    UNSPECIFIED,
    is_ring_request,
)
from proteinfoundation.datasets.structure_data import StructureDataModule
from proteinfoundation.datasets.transforms import CyclizationLabelTransform, Data

# --------------------------------------------------------------------------------------
# The LINEAR token
# --------------------------------------------------------------------------------------


def test_linear_is_a_distinct_index_outside_the_head_output_space():
    assert LINEAR != UNSPECIFIED, "LINEAR must not collapse onto the CFG null"
    assert LINEAR >= NUM_CYCLIZATION_TYPES, "LINEAR must sit outside the head's 3-way output space"
    assert NUM_CYCLIZATION_COND_TYPES == LINEAR + 1, "conditioning embedding must cover LINEAR"


def test_is_ring_request_excludes_both_non_ring_tokens():
    t = torch.tensor([MAINCHAIN, DISULFIDE, ISOPEPTIDE, UNSPECIFIED, LINEAR])
    expected = torch.tensor([True, True, True, False, False])
    assert torch.equal(is_ring_request(t), expected)


def test_ring_pe_is_all_zero_for_linear_rows():
    """A LINEAR row must get the same all-zero cycle graph a dropped (UNSPECIFIED) row gets."""
    from proteinfoundation.nn.feature_factory.pair_feats import CyclizationGraphPositionalPairFeat

    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=8, typed_edge=True, link_direction=True)
    n = 6

    def batch(cond: int) -> dict:
        return {
            "x_t": {"bb_ca": torch.zeros(1, n, 3)},
            "mask": torch.ones(1, n, dtype=torch.bool),
            "cyclization_type_cond": torch.tensor([cond], dtype=torch.long),
            "cyclization_i": torch.tensor([0], dtype=torch.long),
            "cyclization_j": torch.tensor([n - 1], dtype=torch.long),
        }

    linear_out = feat(batch(LINEAR))
    assert torch.all(linear_out == 0), "LINEAR must produce the all-zero pair feature"
    assert torch.equal(linear_out, feat(batch(UNSPECIFIED))), "LINEAR and UNSPECIFIED must both be inactive"
    assert not torch.all(feat(batch(MAINCHAIN)) == 0), "a real ring request must still be active"


def test_validity_mask_does_not_starve_a_linear_row():
    """LINEAR must leave the candidate set unrestricted, not empty.

    A LINEAR row carries has_cyclization=False and is dropped from the linkage loss, but
    an all-False validity mask would still be a fully-masked softmax for anything that
    reads it. It must behave like UNSPECIFIED here.
    """
    from proteinfoundation.cyclization import build_cyclization_validity_mask
    from proteinfoundation.cyclization.constants import AA_CYS

    aa = torch.full((2, 4), AA_CYS, dtype=torch.long)
    binder_mask = torch.ones(2, 4, dtype=torch.bool)
    mask = build_cyclization_validity_mask(
        aa=aa, binder_mask=binder_mask, cond_type=torch.tensor([LINEAR, UNSPECIFIED])
    )
    assert mask[0].any(), "LINEAR must not produce an all-False validity mask"
    assert torch.equal(mask[0], mask[1]), "LINEAR must match UNSPECIFIED's candidate set"


def test_label_transform_maps_linear_metadata_to_the_linear_token():
    """A row whose metadata says "linear" is labeled LINEAR without touching the PDB.

    The `file_path` below does not exist: reaching the CONECT parser at all would raise
    or warn, so this also pins the short-circuit that keeps 45k linear rows from paying
    a file read per epoch.
    """
    transform = CyclizationLabelTransform()
    graph = Data()
    graph.binder_chain_id = "B"
    graph.cyclization_type = "linear"
    graph.peptide_length = 8
    graph.file_path = "/nonexistent/must/not/be/read.pdb"
    graph.residue_pdb_idx = torch.arange(8)

    out = transform(graph)
    assert out.cyclization_type_cond == LINEAR
    assert out.has_cyclization is False
    assert out.cyclization_i == -1 and out.cyclization_j == -1


def test_label_transform_still_gives_unspecified_to_merely_unlabeled_rows():
    """"linear" and "unknown to us" must stay distinguishable."""
    transform = CyclizationLabelTransform()
    graph = Data()
    graph.binder_chain_id = "B"
    graph.cyclization_type = "other"
    graph.peptide_length = 8
    graph.file_path = None
    graph.residue_pdb_idx = torch.arange(8)

    out = transform(graph)
    assert out.cyclization_type_cond == UNSPECIFIED
    assert out.has_cyclization is False


def test_preprocessor_linear_name_is_one_the_transform_recognizes():
    """The producer and the consumer must agree on the string, or conditioning silently dies."""
    from script_utils.preprocess_lp import LINEAR_TYPE_NAME

    assert LINEAR_TYPE_NAME in CyclizationLabelTransform.LINEAR_TYPE_NAMES


def test_none_and_null_still_mean_unspecified_at_generation_time():
    """Pre-existing configs say "none" meaning "model's choice"; that must not become LINEAR."""
    from proteinfoundation.datasets.gen_dataset import _parse_cyclization_type_request

    assert _parse_cyclization_type_request("none") == UNSPECIFIED
    assert _parse_cyclization_type_request("null") == UNSPECIFIED
    assert _parse_cyclization_type_request("unspecified") == UNSPECIFIED
    assert _parse_cyclization_type_request("linear") == LINEAR
    assert _parse_cyclization_type_request("mainchain") == MAINCHAIN
    assert _parse_cyclization_type_request(None) is None


def test_type_dropout_still_produces_the_cfg_null_from_a_linear_row():
    from proteinfoundation.utils.training_handlers import handle_cyclization_type_dropout

    batch = {"cyclization_type_cond": torch.full((256,), LINEAR, dtype=torch.long)}
    out = handle_cyclization_type_dropout(batch, dropout_rate=1.0)
    assert torch.all(out["cyclization_type_cond"] == UNSPECIFIED)


# --------------------------------------------------------------------------------------
# The weighted sampler
# --------------------------------------------------------------------------------------


def _mixed_frame(counts: dict[str, int]) -> pd.DataFrame:
    rows = [{"example_id": f"{s}_{i}", "dataset_source": s} for s, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows)


def _dm(**kw) -> StructureDataModule:
    return StructureDataModule(metadata_file="unused.parquet", **kw)


def test_sample_weights_give_each_source_its_requested_share():
    """The realised draw fractions must match the request, not the natural proportions.

    Sizes here are the real ratio in miniature: a large source and a small one at ~50:1,
    asked to come out 3:1.
    """
    counts = {"cpsea": 50_000, "pepbench": 700, "protfrag": 300}
    df = _mixed_frame(counts)
    dm = _dm(source_fractions={"cpsea": 0.75, "pepbench": 0.175, "protfrag": 0.075})

    weights = dm._compute_sample_weights(df, dm.source_fractions)
    assert weights is not None and len(weights) == len(df)

    probs = weights / weights.sum()
    got = df.groupby("dataset_source", observed=True).apply(
        lambda g: probs[g.index].sum(), include_groups=False
    )
    for source, want in dm.source_fractions.items():
        assert abs(float(got[source]) - want) < 1e-9, f"{source}: wanted {want}, got {float(got[source])}"

    # And the LP side really is oversampled -- the point of the exercise.
    natural_lp = (counts["pepbench"] + counts["protfrag"]) / sum(counts.values())
    assert natural_lp < 0.03 and float(got["pepbench"] + got["protfrag"]) > 0.24


def test_sampler_draws_match_the_requested_fractions_empirically():
    """End-to-end through WeightedRandomSampler, not just the weight arithmetic."""
    from torch.utils.data import WeightedRandomSampler

    counts = {"cpsea": 20_000, "lp": 400}
    df = _mixed_frame(counts)
    dm = _dm(source_fractions={"cpsea": 0.75, "lp": 0.25})
    weights = dm._compute_sample_weights(df, dm.source_fractions)

    torch.manual_seed(0)
    drawn = list(WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), 40_000, replacement=True))
    lp_frac = float(np.mean([df["dataset_source"].iloc[i] == "lp" for i in drawn]))
    assert abs(lp_frac - 0.25) < 0.01, f"empirical LP fraction {lp_frac:.4f} is not ~0.25"


def test_missing_source_column_is_an_error_not_a_silent_uniform_run():
    df = pd.DataFrame({"example_id": ["a", "b"]})
    dm = _dm(source_fractions={"cpsea": 1.0})
    try:
        dm._compute_sample_weights(df, dm.source_fractions)
    except ValueError as e:
        assert "dataset_source" in str(e)
    else:
        raise AssertionError("a missing dataset_source column must raise, not fall back to uniform")


def test_source_present_in_data_but_absent_from_config_is_an_error():
    """An unweighted source would get probability zero -- data silently dropped."""
    df = _mixed_frame({"cpsea": 10, "protfrag": 10})
    dm = _dm(source_fractions={"cpsea": 1.0})
    try:
        dm._compute_sample_weights(df, dm.source_fractions)
    except ValueError as e:
        assert "protfrag" in str(e)
    else:
        raise AssertionError("a source with no configured fraction must raise")


def test_no_fractions_means_no_weighting():
    df = _mixed_frame({"cpsea": 10})
    assert _dm()._compute_sample_weights(df, None) is None


def test_val_source_fractions_is_refused():
    """Silently ignoring it would report a val loss over a mix the config does not describe."""
    try:
        _dm(val_source_fractions={"cpsea": 0.5, "lp": 0.5})
    except ValueError as e:
        assert "val_source_fractions" in str(e)
    else:
        raise AssertionError("val_source_fractions must raise at construction, before any file is read")


# --------------------------------------------------------------------------------------
# The mixed metadata builder
# --------------------------------------------------------------------------------------


def test_builder_tags_sources_and_preserves_per_file_source_columns():
    from script_utils.build_mixed_metadata import SOURCE_COLUMN, build_split

    def write(path: Path, n: int, source: str | None):
        df = pd.DataFrame(
            {
                "example_id": [f"{source or 'x'}_{i}" for i in range(n)],
                "path": ["/tmp/x.pdb"] * n,
                "binder_chain_id": ["B"] * n,
                "cluster_id": ["c"] * n,
                "peptide_length": [10] * n,
                "cyclization_type": ["linear" if source else "head_tail"] * n,
            }
        )
        if source:
            df[SOURCE_COLUMN] = source
        df.to_parquet(path, index=False)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write(td / "cpsea.parquet", 1000, None)  # no source column: gets labeled
        lp = pd.DataFrame(
            {
                "example_id": [f"lp_{i}" for i in range(60)],
                "path": ["/tmp/y.pdb"] * 60,
                "binder_chain_id": ["B"] * 60,
                "cluster_id": ["c"] * 60,
                "peptide_length": [8] * 60,
                "cyclization_type": ["linear"] * 60,
                SOURCE_COLUMN: ["pepbench"] * 40 + ["protfrag"] * 20,
            }
        )
        lp.to_parquet(td / "lp.parquet", index=False)

        counts = build_split(td / "out.parquet", [(td / "cpsea.parquet", "cpsea"), (td / "lp.parquet", None)])
        # A single label for the LP file would have erased the pepbench/protfrag split.
        assert counts == {"cpsea": 1000, "pepbench": 40, "protfrag": 20}, counts

        out = pd.read_parquet(td / "out.parquet")
        assert len(out) == 1060
        assert set(out.columns) >= {SOURCE_COLUMN, "cyclization_type"}
        assert set(out.loc[out[SOURCE_COLUMN] != "cpsea", "cyclization_type"]) == {"linear"}

        # Subsampling spreads across the file rather than taking a prefix.
        sub = build_split(td / "sub.parquet", [(td / "cpsea.parquet", "cpsea")], sample={"cpsea": 100})
        assert 90 <= sub["cpsea"] <= 110, sub
        ids = pd.read_parquet(td / "sub.parquet")["example_id"].tolist()
        assert int(ids[-1].split("_")[1]) > 800, "subsample must reach the end of the file, not just its head"


ALL_TESTS = [
    test_linear_is_a_distinct_index_outside_the_head_output_space,
    test_is_ring_request_excludes_both_non_ring_tokens,
    test_ring_pe_is_all_zero_for_linear_rows,
    test_validity_mask_does_not_starve_a_linear_row,
    test_label_transform_maps_linear_metadata_to_the_linear_token,
    test_label_transform_still_gives_unspecified_to_merely_unlabeled_rows,
    test_preprocessor_linear_name_is_one_the_transform_recognizes,
    test_none_and_null_still_mean_unspecified_at_generation_time,
    test_type_dropout_still_produces_the_cfg_null_from_a_linear_row,
    test_sample_weights_give_each_source_its_requested_share,
    test_sampler_draws_match_the_requested_fractions_empirically,
    test_missing_source_column_is_an_error_not_a_silent_uniform_run,
    test_source_present_in_data_but_absent_from_config_is_an_error,
    test_no_fractions_means_no_weighting,
    test_val_source_fractions_is_refused,
    test_builder_tags_sources_and_preserves_per_file_source_columns,
]


if __name__ == "__main__":
    failures = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  OK {test_fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(test_fn.__name__)
            print(f"  FAIL {test_fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        sys.exit(1)
