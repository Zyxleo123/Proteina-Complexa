"""Unit tests for the closure-gate + Rosetta reselection.

Run: python -m pytest script_utils/test_reselect_by_closure_rosetta.py -q
"""

import pandas as pd
import pytest

from proteinfoundation.result_analysis.reselect_by_closure_rosetta import (
    CLOSED_COL,
    ROSETTA_COL,
    reselect,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["id", CLOSED_COL, ROSETTA_COL])


def test_gate_drops_open_and_ranks_by_lowest_dg():
    df = _df([
        ["a", True, -30.0],
        ["b", False, -99.0],   # best dG but OPEN -> must be dropped
        ["c", True, -35.0],    # best among closed
        ["d", True, -20.0],
    ])
    top = reselect(df, n_best=2)
    assert list(top["id"]) == ["c", "a"]          # ranked ascending dG, open 'b' excluded
    assert list(top["reselect_rank"]) == [1, 2]


def test_open_ring_never_wins_even_with_best_dg():
    df = _df([["open_great", False, -100.0], ["closed_ok", True, -10.0]])
    top = reselect(df, n_best=5)
    assert list(top["id"]) == ["closed_ok"]


def test_starvation_returns_fewer_than_n_best():
    # Mirrors 1J7K: only a few rings close, so a hard gate yields < n_best.
    df = _df([["a", True, -25.0], ["b", False, -40.0], ["c", False, -41.0]])
    top = reselect(df, n_best=5)
    assert len(top) == 1 and top.iloc[0]["id"] == "a"


def test_no_gate_ranks_everything_by_dg():
    df = _df([["a", True, -30.0], ["b", False, -99.0]])
    top = reselect(df, n_best=5, gate_closed=False)
    assert list(top["id"]) == ["b", "a"]           # open 'b' allowed, wins on dG


def test_missing_rosetta_rows_are_dropped():
    df = _df([["a", True, -30.0], ["b", True, float("nan")]])
    top = reselect(df, n_best=5)
    assert list(top["id"]) == ["a"]


def test_nan_closed_treated_as_open():
    df = _df([["a", True, -30.0], ["b", float("nan"), -50.0]])
    top = reselect(df, n_best=5)
    assert list(top["id"]) == ["a"]


def test_missing_ranking_column_raises():
    df = pd.DataFrame({"id": ["a"], CLOSED_COL: [True]})
    with pytest.raises(KeyError):
        reselect(df, n_best=1)
