"""Unit tests for Rosetta interface energy helpers (no PyRosetta required)."""

from proteinfoundation.evaluation.rosetta_energy import (
    ROSETTA_INTERFACE_SCORE_KEYS,
    ROSETTA_METRIC_COLS,
    _nan_rosetta_metrics,
    is_pyrosetta_available,
)


def test_metric_column_names_align_with_score_keys():
    assert ROSETTA_METRIC_COLS == [f"binder_rosetta_{key}" for key in ROSETTA_INTERFACE_SCORE_KEYS]


def test_nan_rosetta_metrics_has_all_columns():
    metrics = _nan_rosetta_metrics()
    assert set(metrics.keys()) == set(ROSETTA_METRIC_COLS)
    assert all(str(v) == "nan" for v in metrics.values())


def test_is_pyrosetta_available_is_bool():
    assert isinstance(is_pyrosetta_available(), bool)
