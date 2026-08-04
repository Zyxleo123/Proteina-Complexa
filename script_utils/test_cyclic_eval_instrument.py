"""Tests for the cyclic-peptide evaluation instrument fix.

Covers the two pieces that make refold metrics describe the molecule that was
actually generated: the AF2 cyclic sequence offset
(`proteinfoundation.evaluation.cyclic_offset`) and macrocycle-aware success
accounting (`proteinfoundation.result_analysis.cyclic_success`).
"""

import numpy as np
import pandas as pd
import pytest

from script_utils.derive_cyclic_thresholds import derive_thresholds

from proteinfoundation.evaluation.cyclic_offset import (
    apply_cyclic_offset,
    build_complex_cyclic_offset,
    cyclic_offset_block,
    cyclic_offset_supported,
)
from proteinfoundation.result_analysis.cyclic_success import (
    CYC_CLOSED_COL,
    CYC_DIST_COL,
    CYC_FUSED_COL,
    CYC_TYPE_REQ_COL,
    CYC_TYPE_SAT_COL,
    compute_cyclic_success_breakdown,
    cyclization_validity_mask,
    filter_by_cyclization_validity,
)

# ---------------------------------------------------------------------------
# cyclic_offset_block
# ---------------------------------------------------------------------------


def test_termini_become_adjacent():
    """The whole point: residue 0 and residue L-1 read as one bond apart."""
    off = cyclic_offset_block(12)
    assert off[0, 11] == 1
    assert off[11, 0] == -1


def test_diagonal_is_zero_and_matrix_antisymmetric():
    off = cyclic_offset_block(9)
    assert np.all(np.diag(off) == 0)
    np.testing.assert_array_equal(off, -off.T)


def test_no_offset_exceeds_half_the_ring():
    for length in (5, 8, 12, 13, 16):
        off = cyclic_offset_block(length)
        assert np.abs(off).max() <= length // 2


def test_short_range_offsets_are_unchanged():
    """Nearby residues must keep their ordinary linear separation."""
    off = cyclic_offset_block(16)
    for i in range(16):
        for j in range(16):
            linear = i - j
            if abs(linear) <= 16 // 2 - 1:
                assert off[i, j] == linear


def test_antipodal_tie_is_deterministic_and_linear():
    """Even L has exactly antipodal pairs; the tie must resolve the same way every call."""
    off = cyclic_offset_block(12)
    # off[i, j] == i - j, so the linear value at the antipode is -6; the strict '>'
    # keeps it rather than flipping to the equally-short +6.
    assert off[0, 6] == -6
    assert off[6, 0] == 6
    np.testing.assert_array_equal(off, cyclic_offset_block(12))


def test_odd_length_has_no_ties():
    off = cyclic_offset_block(11)
    assert np.abs(off).max() == 5


def test_degenerate_lengths():
    assert cyclic_offset_block(0).shape == (0, 0)
    assert cyclic_offset_block(1).shape == (1, 1)
    with pytest.raises(ValueError):
        cyclic_offset_block(-1)


# ---------------------------------------------------------------------------
# build_complex_cyclic_offset
# ---------------------------------------------------------------------------


def test_target_block_is_untouched():
    """No cyclization relates receptor residues; their offsets must stay linear."""
    total, binder = 30, 12
    off = build_complex_cyclic_offset(total, binder)
    n_target = total - binder
    idx = np.arange(n_target)
    np.testing.assert_array_equal(off[:n_target, :n_target], idx[:, None] - idx[None, :])


def test_target_binder_cross_block_is_untouched():
    total, binder = 30, 12
    off = build_complex_cyclic_offset(total, binder)
    n_target = total - binder
    idx = np.arange(total)
    linear = idx[:, None] - idx[None, :]
    np.testing.assert_array_equal(off[:n_target, n_target:], linear[:n_target, n_target:])
    np.testing.assert_array_equal(off[n_target:, :n_target], linear[n_target:, :n_target])


def test_binder_block_is_the_ring():
    total, binder = 30, 12
    off = build_complex_cyclic_offset(total, binder)
    np.testing.assert_array_equal(off[-binder:, -binder:], cyclic_offset_block(binder))


def test_binder_longer_than_complex_raises():
    with pytest.raises(ValueError):
        build_complex_cyclic_offset(10, 12)


def test_chain_break_gap_is_preserved_not_erased():
    """The bug this guards: rebuilding from `arange(total_len)` silently turns a real
    target/binder chain-break gap (e.g. -50) into an ordinary linear separation (-1),
    which tells AF2 the two chains are backbone neighbours."""
    target_len, binder_len = 20, 12
    # Mirrors ColabDesign's prep_pdb/prep_inputs: binder residue_index starts at
    # target's last index + 50, not at target_len.
    residue_index = np.concatenate(
        [np.arange(target_len), residue_index_gap := np.arange(binder_len) + (target_len - 1) + 50]
    )
    off = build_complex_cyclic_offset(residue_index, binder_len)
    # Target-binder cross block must reflect the real ~50-residue gap, not ~1.
    assert off[0, target_len] == pytest.approx(-(residue_index_gap[0] - 0))
    assert abs(off[0, target_len]) > 40
    # Binder-binder block is still the wrapped ring, unaffected by the gap.
    np.testing.assert_array_equal(off[-binder_len:, -binder_len:], cyclic_offset_block(binder_len))


def test_apply_cyclic_offset_preserves_real_chain_break():
    target_len, binder_len = 20, 12
    residue_index = np.concatenate([np.arange(target_len), np.arange(binder_len) + (target_len - 1) + 50])

    class _RealisticFakeAFModel:
        def __init__(self, residue_index):
            self._inputs = {"residue_index": residue_index}

    model = _RealisticFakeAFModel(residue_index)
    assert apply_cyclic_offset(model, binder_len=binder_len, linkage_type="mainchain") is True
    cross_offset = model._inputs["offset"][0, target_len]
    assert abs(cross_offset) > 40, "chain-break gap was erased by the cyclic-offset patch"


# ---------------------------------------------------------------------------
# apply_cyclic_offset -- linkage gating
# ---------------------------------------------------------------------------


class _FakeAFModel:
    def __init__(self, total_len):
        self._inputs = {"residue_index": np.arange(total_len)}


def test_supported_types():
    assert cyclic_offset_supported("mainchain")
    # Side-chain crosslinks do NOT make the backbone termini neighbours.
    assert not cyclic_offset_supported("disulfide")
    assert not cyclic_offset_supported("isopeptide")
    assert not cyclic_offset_supported(None)


def test_applies_for_mainchain():
    model = _FakeAFModel(30)
    assert apply_cyclic_offset(model, binder_len=12, linkage_type="mainchain") is True
    assert model._inputs["offset"][-12:, -12:][0, 11] == 1


@pytest.mark.parametrize("linkage", ["disulfide", "isopeptide", None])
def test_skipped_for_sidechain_linkages(linkage):
    """Wrapping the backbone for a side-chain ring would assert something false."""
    model = _FakeAFModel(30)
    assert apply_cyclic_offset(model, binder_len=12, linkage_type=linkage) is False
    assert "offset" not in model._inputs


def test_skipped_when_inputs_absent():
    class Unprepared:
        _inputs = None

    assert apply_cyclic_offset(Unprepared(), binder_len=12, linkage_type="mainchain") is False


def test_skipped_when_binder_too_short_or_too_long():
    assert apply_cyclic_offset(_FakeAFModel(30), binder_len=2, linkage_type="mainchain") is False
    assert apply_cyclic_offset(_FakeAFModel(10), binder_len=12, linkage_type="mainchain") is False


# ---------------------------------------------------------------------------
# Success accounting
# ---------------------------------------------------------------------------


def _df(**cols):
    return pd.DataFrame(cols)


def test_open_ring_is_rejected():
    df = _df(**{CYC_CLOSED_COL: [True, False, True]})
    assert list(cyclization_validity_mask(df)) == [True, False, True]


def test_fused_ring_is_rejected_even_though_closed():
    """A fused ring passes any proximity test; it needs its own veto."""
    df = _df(**{CYC_CLOSED_COL: [True, True], CYC_FUSED_COL: [False, True]})
    assert list(cyclization_validity_mask(df)) == [True, False]


def test_undeterminable_closure_counts_as_failure():
    """NaN means we could not measure the ring, which is not evidence it closed."""
    df = _df(**{CYC_CLOSED_COL: [True, np.nan]})
    assert list(cyclization_validity_mask(df)) == [True, False]


def test_string_booleans_from_csv_roundtrip():
    """CSV turns booleans into strings; the gate must not silently pass everything."""
    df = _df(**{CYC_CLOSED_COL: ["True", "False", "true"]})
    assert list(cyclization_validity_mask(df)) == [True, False, True]


def test_missing_closure_column_fails_closed():
    """A gate that evaporates when its column is absent is the bug being fixed."""
    df = _df(other=[1, 2, 3])
    assert list(cyclization_validity_mask(df)) == [False, False, False]


def test_type_satisfaction_gate_is_opt_in():
    df = _df(**{CYC_CLOSED_COL: [True, True], CYC_TYPE_SAT_COL: [True, False]})
    assert list(cyclization_validity_mask(df)) == [True, True]
    assert list(cyclization_validity_mask(df, require_type_satisfied=True)) == [True, False]


def test_filter_returns_only_valid_rows():
    df = _df(**{CYC_CLOSED_COL: [True, False, True], "id": [0, 1, 2]})
    assert list(filter_by_cyclization_validity(df)["id"]) == [0, 2]


def test_raw_and_joint_success_are_separate():
    """A metric used to filter must never be reported as the headline result."""
    df = _df(**{CYC_CLOSED_COL: [True, True, False, False]})
    quality = pd.Series([True, False, True, False])
    out = compute_cyclic_success_breakdown(df, quality_mask=quality)
    assert out["raw_ring_closed_rate"] == 0.5  # generation quality, unfiltered
    assert out["quality_pass_rate"] == 0.5  # independent binder quality
    assert out["joint_success_rate"] == 0.25  # only row 0 satisfies both


def test_joint_success_is_nan_without_quality_mask():
    """Absent an independent quality verdict, joint success is undefined, not 1.0."""
    df = _df(**{CYC_CLOSED_COL: [True, True]})
    out = compute_cyclic_success_breakdown(df)
    assert out["quality_available"] is False
    assert np.isnan(out["joint_success_rate"])
    assert out["raw_chemistry_valid_rate"] == 1.0


def test_macro_average_does_not_let_one_target_dominate():
    """9 designs on target A, 1 on target B: macro must not follow the micro rate."""
    df = _df(
        **{
            CYC_CLOSED_COL: [True] * 9 + [False],
            "target_name": ["A"] * 9 + ["B"],
        }
    )
    out = compute_cyclic_success_breakdown(df)
    assert out["raw_chemistry_valid_rate"] == 0.9  # micro
    assert out["macro_chemistry_valid_rate"] == 0.5  # macro: mean(1.0, 0.0)
    assert out["n_targets"] == 2


def test_target_coverage_is_reported():
    df = _df(**{CYC_CLOSED_COL: [False, True, False], "target_name": ["A", "A", "B"]})
    out = compute_cyclic_success_breakdown(df)
    assert out["target_coverage"] == 0.5  # A has one, B has none
    assert out["per_target"]["A"]["any_success"] is True
    assert out["per_target"]["B"]["any_success"] is False


def test_per_linkage_type_breakdown():
    """Pooling hides which chemistry is actually broken."""
    df = _df(
        **{
            CYC_CLOSED_COL: [True, True, False, False],
            CYC_TYPE_REQ_COL: ["mainchain", "mainchain", "disulfide", "disulfide"],
        }
    )
    out = compute_cyclic_success_breakdown(df)
    assert out["by_linkage_type"]["mainchain"]["chemistry_valid_rate"] == 1.0
    assert out["by_linkage_type"]["disulfide"]["chemistry_valid_rate"] == 0.0


def test_bond_distance_summary_ignores_missing():
    df = _df(**{CYC_CLOSED_COL: [True, True, True], CYC_DIST_COL: [1.3, 1.5, np.nan]})
    out = compute_cyclic_success_breakdown(df)
    assert out["bond_dist_A_n"] == 2
    assert out["bond_dist_A_median"] == pytest.approx(1.4)


def test_empty_dataframe_does_not_crash():
    out = compute_cyclic_success_breakdown(_df(**{CYC_CLOSED_COL: []}))
    assert out["n_designs"] == 0
    assert np.isnan(out["raw_ring_closed_rate"])


# ---------------------------------------------------------------------------
# Config plumbing
#
# The offset flag has to survive four hops:
#   cfg.metric -> compute_binder_metrics -> run_binder_eval -> run_af_eval -> AF2
# A break anywhere makes the ON runs silently identical to the OFF runs, which does
# not error -- it produces a full, plausible, meaningless calibration. That happened
# once already (the kwarg existed on run_af_eval but nothing passed it), so each hop
# is asserted rather than assumed.
# ---------------------------------------------------------------------------


def test_run_af_eval_accepts_cyclic_kwargs():
    import inspect

    from proteinfoundation.utils.colabdesign_utils import run_af_eval

    params = inspect.signature(run_af_eval).parameters
    assert "cyclic_offset" in params
    assert "cyclization_type" in params
    # Default must be off, or existing evaluations silently change meaning.
    assert params["cyclic_offset"].default is False


def test_run_binder_eval_accepts_and_forwards_cyclic_kwargs():
    import inspect

    from proteinfoundation.metrics.binder_metrics import run_binder_eval

    params = inspect.signature(run_binder_eval).parameters
    assert "cyclic_offset" in params
    assert params["cyclic_offset"].default is False

    # The kwarg existing is not the same as it being forwarded; assert the colabdesign
    # eval_kwargs literal actually carries it.
    source = inspect.getsource(run_binder_eval)
    assert '"cyclic_offset": cyclic_offset' in source
    assert '"cyclization_type": cyclization_type' in source


def test_binder_eval_reads_cyclic_offset_from_config():
    import inspect

    from proteinfoundation.evaluation import binder_eval

    source = inspect.getsource(binder_eval.compute_binder_metrics)
    assert 'cfg_metric.get("cyclic_offset"' in source
    assert 'cfg_metric.get("cyclization_type"' in source


def test_launcher_uses_the_config_key_the_code_reads():
    """The override prefix must be `metric.`, matching `eval_config.metric`."""
    import pathlib

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_cpsea_native_calibration.sh"
    text = script.read_text()
    assert "++metric.cyclic_offset=true" in text
    assert "++metric.cyclization_type=" in text
    # `evaluation.` is not a real config block here; hydra's ++ would accept it silently.
    assert "++evaluation.cyclic_offset" not in text


def test_launcher_resolves_repo_root_under_sbatch():
    """`$0` is a spool copy under sbatch, so dirname-based cd lands outside the repo."""
    import pathlib

    scripts_dir = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    for name in ("run_cpsea_native_calibration.sh", "run_cpsea_cyclization_audit.sh"):
        text = (scripts_dir / name).read_text()
        assert "SLURM_SUBMIT_DIR" in text, f"{name} would cd outside the repo under sbatch"
        assert 'cd "$(dirname "$0")/.."' not in text, f"{name} still uses the broken pattern"


def test_launcher_uses_plusplus_prefix_on_every_override():
    """Bare `key=value` reaches hydra but NOT the CLI's own run_name parse.

    cli_runner re-appends `++run_name=<its own parse>` last, so a bare `run_name=` is
    outranked by the config default and all runs share one output directory. The other
    overrides still apply, so the right targets get evaluated into the wrong place --
    a zero exit code and a full results tree containing only the last run.
    """
    import pathlib
    import re

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_cpsea_native_calibration.sh"
    text = script.read_text()

    for key in ("run_name", "dataset.task_name", "sample_storage_path"):
        assert f"++{key}=" in text, f"{key} override missing the '++' prefix"
        # A bare occurrence at the start of a continuation line is the broken form.
        assert not re.search(rf"^\s+{re.escape(key)}=", text, re.MULTILINE), (
            f"{key} still passed as a bare override"
        )


def test_launcher_verifies_output_landed():
    """Exit 0 is not proof the results reached the directory derive_ will read."""
    import pathlib

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_cpsea_native_calibration.sh"
    text = script.read_text()
    assert "compgen -G" in text
    assert "EXPECTED_DIR" in text


def test_expected_dir_matches_derive_script_pattern():
    """The launcher's run_name and the derive script's glob must agree."""
    import pathlib

    from script_utils.derive_cyclic_thresholds import main as _  # noqa: F401

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_cpsea_native_calibration.sh"
    text = script.read_text()
    # Launcher builds native_calib_<target>_cyc{off,on}; derive defaults to *_cycon.
    assert 'RUN_NAME="native_calib_${TARGET}_cyc${MODE}"' in text
    import fnmatch

    assert fnmatch.fnmatch("native_calib_cpsea_1E2T_cycon", "native_calib_*_cycon")
    assert not fnmatch.fnmatch("native_calib_cpsea_1E2T_cycoff", "native_calib_*_cycon")


def test_launcher_fails_loudly_on_partial_sweep():
    """A swallowed failure looks exactly like a successful run that produced no data."""
    import pathlib

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_cpsea_native_calibration.sh"
    text = script.read_text()
    assert "FAILURES+=" in text
    assert "exit 1" in text


# ---------------------------------------------------------------------------
# Requested-cyclization-type resolution
#
# When this returns None, `binder_cyc_type_satisfied` is never computed and closure is
# scored against whatever chemistry the model happened to build -- which cannot tell
# "did what it was told" from "did something else that also closed". It returned None on
# every run for months because it read `cfg.dataloader...`, a key the evaluation config
# does not have, under a blanket suppress.
# ---------------------------------------------------------------------------


def _resolver():
    from proteinfoundation.evaluate import _resolve_requested_cyclization_type

    return _resolve_requested_cyclization_type


def test_explicit_metric_override_wins():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"dataset": {"task_name": "cpsea_1GYT", "target_dict_cfg": {"cpsea_1GYT": {"cyclization_type": "isopeptide"}}}})
    metric = OmegaConf.create({"cyclization_type": "disulfide"})
    assert _resolver()(cfg, metric) == "disulfide"


def test_falls_back_to_targets_dict():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"dataset": {"task_name": "cpsea_1E2T", "target_dict_cfg": {"cpsea_1E2T": {"cyclization_type": "mainchain"}}}})
    assert _resolver()(cfg, OmegaConf.create({})) == "mainchain"


def test_generation_config_path_still_works():
    """The original source must keep working when the full pipeline config is in scope."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"dataloader": {"dataset": {"conditional_features": [{"cyclization_type": "isopeptide"}]}}})
    assert _resolver()(cfg, OmegaConf.create({})) == "isopeptide"


def test_non_cyclic_target_resolves_to_none():
    """A normal binder target must not acquire a spurious linkage."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"dataset": {"task_name": "02_PDL1", "target_dict_cfg": {"02_PDL1": {}}}})
    assert _resolver()(cfg, OmegaConf.create({})) is None


def test_missing_everything_returns_none_without_raising():
    from omegaconf import OmegaConf

    assert _resolver()(OmegaConf.create({}), OmegaConf.create({})) is None


def test_resolves_against_the_real_configs():
    """Integration: the shipped configs must actually resolve, not just synthetic dicts.

    The bug this guards was invisible to any unit test using a hand-built config -- it
    only appeared against the real evaluation config, which has no `dataloader` key.
    """
    import os

    from hydra import compose, initialize_config_dir

    configs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    expected = {
        "cpsea_1E2T": "mainchain",
        "cpsea_1J7K": "mainchain",
        "cpsea_1GYT": "isopeptide",
        "cpsea_1M46": "isopeptide",
        "cpsea_1MU2": "isopeptide",
    }
    resolve = _resolver()
    with initialize_config_dir(config_dir=os.path.abspath(configs), version_base="1.3"):
        for task, want in expected.items():
            cfg = compose(
                config_name="evaluate_cpsea_native_calibration",
                overrides=[f"++dataset.task_name={task}"],
            )
            assert resolve(cfg, cfg.metric) == want, f"{task} resolved wrongly"


# ---------------------------------------------------------------------------
# CPSea evaluation target set
# ---------------------------------------------------------------------------


def _eval_targets():
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "targets" / "cpsea_eval_targets.yaml"
    if not path.exists():
        pytest.skip("cpsea_eval_targets.yaml not generated")
    return yaml.safe_load(path.read_text())["target_dict_cfg"]


def test_eval_target_set_is_balanced_across_chemistries():
    """Unbalanced strata are what made disulfide unstudiable in the recorded runs (n=21)."""
    from collections import Counter

    counts = Counter(v["cyclization_type"] for v in _eval_targets().values())
    assert set(counts) == {"mainchain", "disulfide", "isopeptide"}
    assert len(set(counts.values())) == 1, f"strata not balanced: {counts}"


def test_eval_targets_are_cluster_unique():
    """One target per sequence cluster: homologues are not independent observations."""
    import csv
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "calibration_native" / "cpsea_eval_targets.csv"
    if not path.exists():
        pytest.skip("target manifest not generated")
    with open(path) as handle:
        clusters = [row["cluster_id"] for row in csv.DictReader(handle)]
    assert len(clusters) == len(set(clusters)), "duplicate cluster in evaluation set"


def test_every_eval_target_has_a_native_dg_bar():
    """The quality gate is anchored on the native; a target without one cannot be scored."""
    missing = [k for k, v in _eval_targets().items() if v.get("native_rosetta_dG") is None]
    assert not missing, f"{len(missing)} targets lack a native dG bar"


def test_eval_target_structures_exist():
    import os

    missing = [v["target_path"] for v in _eval_targets().values() if not os.path.exists(v["target_path"])]
    assert not missing, f"{len(missing)} target PDBs missing"


def test_eval_targets_merged_into_targets_dict_without_clobbering():
    """The append must be purely additive -- the original targets still resolve."""
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "targets" / "targets_dict.yaml"
    merged = yaml.safe_load(path.read_text())["target_dict_cfg"]
    for original in ("01_PD1", "02_PDL1", "cpsea_1E2T", "cpsea_1MU2"):
        assert original in merged, f"{original} lost from targets dict"
    assert sum(1 for k in merged if k.startswith("cpsea_eval_")) == len(_eval_targets())


# ---------------------------------------------------------------------------
# Threshold derivation from native calibration
# ---------------------------------------------------------------------------


def _native_df():
    return pd.DataFrame(
        {
            "self_complex_pLDDT": [0.82, 0.75, 0.79],
            "self_complex_i_pAE": [0.31, 0.40, 0.35],
            "self_binder_scRMSD_ca": [0.4, 0.9, 0.6],
        }
    )


def test_every_native_passes_derived_thresholds():
    """The defining property: a bar no real cyclic peptide clears is a broken bar."""
    thresholds, _ = derive_thresholds(_native_df())
    df = _native_df()
    assert (df["self_complex_pLDDT"] >= thresholds["pLDDT"]["threshold"]).all()
    assert (df["self_complex_i_pAE"] * 31.0 <= thresholds["i_pAE"]["threshold"]).all()
    assert (df["self_binder_scRMSD_ca"] < thresholds["scRMSD_ca"]["threshold"]).all()


def test_thresholds_anchor_on_worst_native_not_median():
    _, report = derive_thresholds(_native_df())
    assert report["pLDDT"]["worst_native"] == 0.75  # the minimum, for a >= metric
    assert report["i_pAE"]["worst_native"] == 0.40  # the maximum, for a <= metric


def test_derived_thresholds_are_reachable_unlike_stock_ones():
    """Stock pLDDT>=0.9 exceeds every native; the derived bar must not."""
    thresholds, _ = derive_thresholds(_native_df())
    assert thresholds["pLDDT"]["threshold"] < 0.9


def test_missing_columns_are_skipped_not_guessed():
    thresholds, report = derive_thresholds(pd.DataFrame({"self_complex_pLDDT": [0.8]}))
    assert "pLDDT" in thresholds
    assert "i_pAE" not in thresholds and "i_pAE" not in report


def test_all_nan_metric_is_skipped():
    df = _native_df()
    df["self_complex_pLDDT"] = np.nan
    thresholds, _ = derive_thresholds(df)
    assert "pLDDT" not in thresholds
    assert "i_pAE" in thresholds


def test_margin_widens_the_bar_in_the_permissive_direction():
    tight, _ = derive_thresholds(_native_df(), margin=0.0)
    loose, _ = derive_thresholds(_native_df(), margin=0.20)
    assert loose["pLDDT"]["threshold"] < tight["pLDDT"]["threshold"]  # lower bar
    assert loose["i_pAE"]["threshold"] > tight["i_pAE"]["threshold"]  # higher ceiling
