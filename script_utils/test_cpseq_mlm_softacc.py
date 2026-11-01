"""Unit tests for the cpseq_mlm substitution-tolerance and ambiguity probes.

Everything here is hand-computable. The point is that the two probes are instruments for
settling a disagreement about whether the arm is near its ceiling, so their arithmetic has
to be checkable without running either of them on real data -- including the DIRECTION of
each estimator's bias, which is what the reported numbers will be read against.

Run:
    .venv/bin/python -m pytest script_utils/test_cpseq_mlm_softacc.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cpseq_mlm_ambiguity_probe import analyse, pocket_digest, row_key, stratum_of
from cpseq_mlm_data import AA20
from cpseq_mlm_softacc import (
    blosum62,
    clustered_ci,
    graded_table,
    prior_predictions,
    score_position,
)

# ---------------------------------------------------------------- substitution scoring


def test_blosum62_has_the_expected_shape():
    b = blosum62()
    assert len(b) == 400
    # A handful of canonical entries, so a wrong matrix (BLOSUM45/80) is caught here
    # rather than showing up as a mysteriously different lift.
    assert b[("I", "L")] == 2
    assert b[("K", "E")] == 1
    assert b[("A", "W")] == -3
    assert b[("C", "C")] == 9
    assert b[("W", "W")] == 11
    # Symmetry is relied on: the metric is scored (pred, true) but a substitution matrix
    # is symmetric, so a transposed lookup would silently still "work".
    for a in AA20:
        for c in AA20:
            assert b[(a, c)] == b[(c, a)]


def test_graded_table_is_normalised_per_true_residue():
    g = graded_table(blosum62())
    for true in AA20:
        assert g[(true, true)] == 1.0, true
        col = [g[(p, true)] for p in AA20]
        assert min(col) == 0.0, true
        assert max(col) == 1.0, true


def test_score_position_separates_conservative_from_unrelated():
    b, g = blosum62(), graded_table(blosum62())
    exact = score_position("L", "L", b, g)
    assert exact == {"exact": 1.0, "soft_ge0": 1.0, "soft_ge1": 1.0, "graded": 1.0}

    # ILE for LEU is the textbook conservative miss: not exact, but tolerated.
    conservative = score_position("I", "L", b, g)
    assert conservative["exact"] == 0.0
    assert conservative["soft_ge1"] == 1.0
    assert 0.0 < conservative["graded"] < 1.0

    # ALA for TRP is not a defensible substitution and must score zero on both gates.
    unrelated = score_position("A", "W", b, g)
    assert unrelated["exact"] == 0.0
    assert unrelated["soft_ge0"] == 0.0
    assert unrelated["soft_ge1"] == 0.0


def test_soft_ge0_is_weaker_than_soft_ge1():
    """A BLOSUM score of exactly 0 passes the >=0 gate and fails the >=1 gate.

    Both thresholds are reported because the choice is not obvious, and a reader should be
    able to see whether a conclusion depends on it.
    """
    b, g = blosum62(), graded_table(blosum62())
    zeros = [(p, t) for p in AA20 for t in AA20 if b[(p, t)] == 0]
    assert zeros, "BLOSUM62 has zero-score pairs; the two gates would be identical without"
    p, t = zeros[0]
    s = score_position(p, t, b, g)
    assert s["soft_ge0"] == 1.0
    assert s["soft_ge1"] == 0.0


# ---------------------------------------------------------------- baseline reconstruction


def _priors():
    return {
        "marginal": {a: (0.5 if a == "D" else 0.5 / 19) for a in AA20},
        "positional": {(0, 8): {"C": 0.7, "A": 0.3}},
        "chem": {(0, 8, 1): {"K": 0.9, "A": 0.1}},
        "n_train_rows": 1,
    }


def test_prior_predictions_uses_the_most_specific_cell_available():
    rec = {"pos": 0, "L": 8, "cyc_type": 1}
    got = prior_predictions(rec, _priors())
    assert got["marginal"] == "D"
    assert got["pos_prior"] == "C"
    assert got["chem_prior"] == "K"


def test_chem_prior_backs_off_to_positional_then_global():
    p = _priors()
    # cyc_type 3 has no chem cell -> must fall back to (pos, L), not score a free miss.
    assert prior_predictions({"pos": 0, "L": 8, "cyc_type": 3}, p)["chem_prior"] == "C"
    # position 5 has neither cell -> global argmax.
    got = prior_predictions({"pos": 5, "L": 8, "cyc_type": 3}, p)
    assert got["chem_prior"] == "D"
    assert got["pos_prior"] == "D"


# ---------------------------------------------------------------- clustered bootstrap


def _units(n_clusters: int, per: int, value: float):
    return [
        {"cluster": f"c{i}", "positions": [{"x": value} for _ in range(per)]}
        for i in range(n_clusters)
    ]


def test_clustered_ci_is_deterministic_for_a_seed():
    units = _units(20, 3, 0.0)
    for i, u in enumerate(units):
        for p in u["positions"]:
            p["x"] = float(i % 2)
    a = clustered_ci(units, ["x"], n_boot=200, seed=7)
    b = clustered_ci(units, ["x"], n_boot=200, seed=7)
    assert a == b
    c = clustered_ci(units, ["x"], n_boot=200, seed=8)
    assert c != a, "a different seed must actually resample differently"


def test_clustered_ci_narrows_with_more_clusters():
    def width(n):
        units = _units(n, 3, 0.0)
        for i, u in enumerate(units):
            for p in u["positions"]:
                p["x"] = float(i % 2)
        lo, hi = clustered_ci(units, ["x"], n_boot=400, seed=1)["x"]
        return hi - lo

    assert width(100) < width(10)


def test_clustered_ci_is_wider_than_ignoring_clusters():
    """The reason the probe resamples clusters at all.

    Each cluster here is internally perfectly correlated, so the effective sample size is
    the cluster count, not the position count. Resampling positions would report an
    interval sqrt(per) times too narrow.
    """
    per = 25
    units = _units(16, per, 0.0)
    for i, u in enumerate(units):
        for p in u["positions"]:
            p["x"] = float(i % 2)
    lo, hi = clustered_ci(units, ["x"], n_boot=600, seed=3)["x"]
    clustered_width = hi - lo

    flat = [{"cluster": f"p{j}", "positions": [p]}
            for u in units for j, p in enumerate(u["positions"])]
    lo2, hi2 = clustered_ci(flat, ["x"], n_boot=600, seed=3)["x"]
    assert clustered_width > (hi2 - lo2) * 2


# ---------------------------------------------------------------- ambiguity estimators


def test_analyse_matches_hand_computed_group():
    """One group, counts {A:3, C:1}.

    LOO: each of the 3 A's sees {A:2, C:1} and predicts A (3 hits); the single C sees
    {A:3, C:0} and predicts A (0 hits). So 3/4.
    H = -(0.75 ln 0.75 + 0.25 ln 0.25) = 0.562335 nats; Miller-Madow adds (2-1)/(2*4).
    """
    res = analyse({("g",): {"A": 3, "C": 1}})
    p = res["pooled"]
    assert p["n_groups"] == 1
    assert p["n_positions"] == 4
    assert p["loo_argmax_acc"] == 0.75
    assert math.isclose(p["cond_entropy_nats"], 0.5623351446188083, rel_tol=1e-9)
    assert math.isclose(p["cond_entropy_nats_mm"], 0.5623351446188083 + 1 / 8, rel_tol=1e-9)
    assert math.isclose(p["cond_perplexity_mm"], math.exp(p["cond_entropy_nats_mm"]))
    # m = 4 lands in the 3-4 stratum and nowhere else.
    assert res["by_stratum"]["3-4"]["n_positions"] == 4
    assert res["by_stratum"]["2-2"]["n_positions"] == 0


def test_loo_scores_zero_on_a_genuinely_split_group():
    """A 50/50 group of size 2 gets LOO accuracy 0, not 0.5.

    This is the estimator's bias, not a bug: with one sample held out the remaining
    observation always points at the other residue. It is why small strata are reported
    separately and why LOO is described as a LOWER bound on the ceiling.
    """
    res = analyse({("g",): {"A": 1, "C": 1}})
    p = res["pooled"]
    assert p["loo_argmax_acc"] == 0.0
    assert math.isclose(p["cond_entropy_nats"], math.log(2), rel_tol=1e-9)
    assert math.isclose(p["cond_entropy_nats_mm"], math.log(2) + 0.25, rel_tol=1e-9)


def test_singletons_are_excluded_not_counted_as_certain():
    """A group of one must not contribute a 100% ceiling from a single observation."""
    res = analyse({("a",): {"A": 1}, ("b",): {"C": 1}, ("c",): {"A": 3, "C": 1}})
    assert res["n_singleton_groups"] == 2
    assert res["n_singleton_positions"] == 2
    assert res["pooled"]["n_positions"] == 4
    assert res["pooled"]["loo_argmax_acc"] == 0.75


def test_a_deterministic_group_has_zero_entropy_and_full_loo():
    res = analyse({("g",): {"A": 10}})
    p = res["pooled"]
    assert p["loo_argmax_acc"] == 1.0
    assert p["cond_entropy_nats"] == 0.0
    # K = 1 observed outcome, so Miller-Madow adds nothing and the perplexity is exactly 1.
    assert p["cond_entropy_nats_mm"] == 0.0
    assert math.isclose(p["cond_perplexity_mm"], 1.0)


def test_stratum_boundaries_are_inclusive_and_disjoint():
    assert stratum_of(1) is None
    assert stratum_of(2) == (2, 2)
    assert stratum_of(3) == (3, 4)
    assert stratum_of(4) == (3, 4)
    assert stratum_of(5) == (5, 9)
    assert stratum_of(10) == (10, 29)
    assert stratum_of(10**6) == (100, 10**9)


# ---------------------------------------------------------------- grouping keys


def test_pocket_digest_is_stable_and_discriminating():
    a = pocket_digest("ACDEFG")
    assert a == pocket_digest("ACDEFG"), "must not depend on process hash salt"
    assert a != pocket_digest("ACDEFH")
    assert len(a) == 32


def test_row_key_strictness_ordering():
    d = {"pep_seq": "ACDEFGHI", "pocket_seq": "MMMM", "cyc_type": 1, "cluster_id": "cl7"}
    assert row_key("chem_prior", d) == (8, 1)
    assert row_key("model", d) == (pocket_digest("MMMM"), 8, 1)
    assert row_key("cluster", d) == ("cl7", 8, 1)
    # A row with no cluster id cannot enter the cluster grouping, and must be dropped from
    # it rather than silently grouped under None with every other such row.
    assert row_key("cluster", {**d, "cluster_id": None}) is None


def test_two_pockets_in_one_cluster_share_a_cluster_key_but_not_a_model_key():
    """The bound direction the probe relies on.

    The cluster grouping merges contexts the model can distinguish, so it over-groups and
    its ceiling is a lower bound on the model's.
    """
    base = {"pep_seq": "ACDEFGHI", "cyc_type": 1, "cluster_id": "cl7"}
    a = {**base, "pocket_seq": "MMMM"}
    b = {**base, "pocket_seq": "WWWW"}
    assert row_key("cluster", a) == row_key("cluster", b)
    assert row_key("model", a) != row_key("model", b)
