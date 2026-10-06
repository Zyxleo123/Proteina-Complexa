"""Checks on the functions that decide the reported numbers.

`is_valid`, `select_representative` and `joint_success` are what turn a pile of packed poses
into "arm X succeeded N times". They are pure functions over dicts, they are three of the
easiest things in this codebase to get quietly wrong, and until now none of them had a test.

The failures these aim at are not crashes. They are a selection rule that silently prefers
the candidate that edited least, a gate that passes a peptide whose sequence changed, a
denominator that drops failed attempts, and a bin boundary that puts a near-identity in the
same cell as a real edit.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from proteinfoundation.cp2lp.packing import (  # noqa: E402
    PackResult,
    SELECTION_ORDER,
    is_valid,
    select_representative,
)


def _met(**kw) -> dict:
    """A metrics dict that PASSES every gate, so each test perturbs exactly one thing."""
    base = {
        "linear_topology": True,
        "sequence_exact": True,
        "clash_receptor_count": 0,
        "clash_self_count": 0,
        "contact_retention": 0.8,
        "dG_separated": -10.0,
        "valid": True,
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------------- is_valid


def test_the_passing_fixture_actually_passes():
    """Otherwise every test below is vacuously green."""
    ok, why = is_valid(_met())
    assert ok and why == ""


@pytest.mark.parametrize(
    "perturb,reason",
    [
        ({"linear_topology": False}, "cyclic_topology"),
        ({"sequence_exact": False}, "sequence_mismatch"),
        ({"clash_receptor_count": 1}, "receptor_clash"),
        ({"clash_self_count": 1}, "self_clash"),
    ],
)
def test_each_gate_rejects_and_names_its_reason(perturb, reason):
    """A rejection must carry its cause into the manifest, not just a False.

    Without the reason, "0 valid candidates" is one number for four completely different
    problems: a re-formed disulfide, a mutated sequence, a backbone through the receptor and a
    self-overlapping peptide each call for different work.
    """
    ok, why = is_valid(_met(**perturb))
    assert not ok
    assert why == reason


def test_a_cyclic_candidate_is_rejected_even_when_everything_else_is_perfect():
    """The whole project exists to produce LINEAR peptides.

    Rosetta will re-bond two nearby cysteines on its own, and the source peptides are
    macrocycles whose sequences are full of them -- so this gate is the last thing standing
    between "a cycle scored beautifully" and a reported success.
    """
    ok, why = is_valid(_met(linear_topology=False, contact_retention=1.0, dG_separated=-50.0))
    assert not ok and why == "cyclic_topology"


def test_sequence_gate_can_be_waived_only_explicitly():
    """With no source sequence to compare against the gate must not silently fail everything."""
    assert is_valid(_met(sequence_exact=False), require_sequence=False)[0]
    assert not is_valid(_met(sequence_exact=False), require_sequence=True)[0]


# --------------------------------------------------------------- select_representative


def _res(seed: int, ok: bool = True, **met) -> PackResult:
    return PackResult(seed, ok, None if ok else "boom", None, _met(**met) if ok else {})


def test_selection_follows_the_predeclared_order():
    """Valid first, then retention, then dG as a TIE-BREAK only.

    The order is data in `SELECTION_ORDER` precisely so it cannot drift into "whichever metric
    separated the arms", and this pins that the code implements the data.
    """
    assert SELECTION_ORDER[0] == "valid"
    assert SELECTION_ORDER[1] == "contact_retention"
    assert SELECTION_ORDER[2] == "interface_energy"

    # A better dG must NOT beat a better retention: retention outranks it.
    results = [
        _res(0, contact_retention=0.9, dG_separated=100.0),
        _res(1, contact_retention=0.5, dG_separated=-999.0),
    ]
    sel, basis = select_representative(results)
    assert sel == 0, "a better dG outranked a better retention"
    assert basis == "valid"

    # Equal retention -> dG breaks the tie, lower being better.
    results = [
        _res(0, contact_retention=0.7, dG_separated=10.0),
        _res(1, contact_retention=0.7, dG_separated=-10.0),
    ]
    assert select_representative(results)[0] == 1


def test_validity_outranks_both_other_criteria():
    """An invalid candidate with a perfect retention and dG must lose to a valid worse one."""
    results = [
        _res(0, linear_topology=False, valid=False, contact_retention=1.0, dG_separated=-999.0),
        _res(1, contact_retention=0.3, dG_separated=50.0),
    ]
    sel, basis = select_representative(results)
    assert sel == 1 and basis == "valid"


def test_no_valid_candidate_still_returns_one_and_says_so():
    """A failure must stay in the manifest as a row, not vanish.

    Dropping it would make the denominators differ between arms, and a comparison whose
    denominators differ per arm is not a comparison.
    """
    results = [
        _res(0, clash_receptor_count=5, valid=False, contact_retention=0.4),
        _res(1, clash_receptor_count=2, valid=False, contact_retention=0.6),
    ]
    sel, basis = select_representative(results)
    assert basis == "no_valid_candidate"
    assert sel == 1, "among invalid candidates the better retention should still be chosen"


def test_all_packing_failed_is_distinguishable_from_no_valid_candidate():
    """Three crashed packings and three clashing ones are different problems."""
    sel, basis = select_representative([_res(0, ok=False), _res(1, ok=False)])
    assert basis == "all_packing_failed"
    sel2, basis2 = select_representative([_res(0, clash_self_count=3, valid=False)])
    assert basis2 == "no_valid_candidate"


def test_nan_metrics_sort_last_rather_than_first():
    """A scorer that returned NaN must not thereby look like the best candidate.

    NaN compares False against everything, so a naive sort can float it to the top -- the same
    class of bug as the abstained-SDEdit rows, where NaN being truthy scored a 100%-abstention
    cell as 100% success.
    """
    results = [
        _res(0, contact_retention=float("nan"), dG_separated=float("nan")),
        _res(1, contact_retention=0.2, dG_separated=5.0),
    ]
    assert select_representative(results)[0] == 1


def test_empty_results_does_not_crash():
    sel, basis = select_representative([])
    assert sel == -1 and basis == "no_results"


# ------------------------------------------------------------------- joint_success


def _report():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cp2lp_bb_report", REPO / "script_utils" / "cp2lp_bb_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(**kw) -> dict:
    base = {"terminal_gap_nm": 0.9, "bb_chain_intact": 1.0}
    base.update(kw)
    return base


def test_joint_success_passes_a_good_attempt():
    rpt = _report()
    ok, why = rpt.joint_success(_row(), _met(), 0.45)
    assert ok and why == ""


@pytest.mark.parametrize(
    "row_kw,sel_kw,reason",
    [
        ({"terminal_gap_nm": 0.2}, {}, "not_open"),
        ({"bb_chain_intact": 0.0}, {}, "raw_backbone_broken"),
        ({}, {"sequence_exact": False}, "sequence_mismatch"),
        ({}, {"clash_receptor_count": 3}, "clash"),
        ({}, {"clash_self_count": 1}, "clash"),
        ({}, {"contact_retention": 0.1}, "contacts_lost"),
    ],
)
def test_each_joint_condition_is_necessary(row_kw, sel_kw, reason):
    """All five conditions are necessary; failing any one is not a usable triplet."""
    rpt = _report()
    ok, why = rpt.joint_success(_row(**row_kw), _met(**sel_kw), 0.45)
    assert not ok and why == reason


def test_a_failed_packing_is_a_joint_failure_not_a_skip():
    rpt = _report()
    ok, why = rpt.joint_success(_row(), None, 0.45)
    assert not ok and why == "packing_failed"


def test_raw_backbone_validity_is_judged_before_packing_not_after():
    """The chain-intact test reads the GENERATOR's raw output, not the packed pose.

    If it read the packed structure, a packer that happened to tidy a bond would mask a broken
    backbone -- and Arm A's entire measured weakness is broken backbones.
    """
    rpt = _report()
    ok, why = rpt.joint_success(_row(bb_chain_intact=0.0), _met(), 0.45)
    assert not ok and why == "raw_backbone_broken"


def test_nan_retention_does_not_silently_pass_the_contact_gate():
    """A NaN retention means "not measured", which must not be read as "retained".

    DELIBERATE REVERSAL of this test's original assertion, 2026-10-05. It used to assert
    `ok and why == ""` -- that a NaN SKIPS the gate rather than failing it -- on the reasoning
    that the skip would stay visible because `contacts_lost` would never fire for the row.

    THE 2026-10-04 RELEASE EVALUATION DISPROVED THAT REASONING. `contact_retention` was absent
    for every attempt (nothing passes `source_contacts` into `score_packed`), so the gate was
    skipped 64 times out of 64, every one of those rows returned `ok=True`, and they went
    straight into a reported "25% joint success". The absence of `contacts_lost` from the
    failure table is NOT self-announcing: it reads exactly like "retention was fine", which is
    how it was first read. A skipped gate that still returns a pass is a silent pass.

    So: not measured is not a failure of the contact gate, it is a failure to EVALUATE the
    gate, and it raises. Same family as the recorded NaN-truthiness bug, opposite direction.
    """
    rpt = _report()
    with pytest.raises(ValueError, match="NaN"):
        rpt.joint_success(_row(), _met(contact_retention=float("nan")), 0.45)


def test_joint_success_raises_rather_than_defaulting_a_missing_gate():
    """The absent-field version of the same bug: `sequence_exact` defaulted through
    `.get(..., True)`, so a gate nobody could evaluate was reported as satisfied."""
    rpt = _report()
    for field in ("sequence_exact", "contact_retention"):
        met = _met()
        del met[field]
        with pytest.raises(KeyError, match=field):
            rpt.joint_success(_row(), met, 0.45)


# ---------------------------------------------------------------------------- binning


def test_the_lowest_edit_bin_isolates_a_near_identity():
    """0.05 nm is the 0.5 A scale the previous export gate was measured to select for.

    If a near-identity shared a bin with a real edit, the edit-binned table could not do the
    one job it exists for.
    """
    rpt = _report()
    assert rpt._bin(0.004, rpt.EDIT_BINS) == "[0,0.05)"
    assert rpt._bin(0.05, rpt.EDIT_BINS) != "[0,0.05)"
    assert rpt._bin(0.2, rpt.EDIT_BINS) == "[0.15,0.3)"
    assert rpt._bin(9.0, rpt.EDIT_BINS) == ">=0.6"


def test_bins_are_contiguous_and_cover_everything_nonnegative():
    """No value may fall through into "nan" and silently leave the table."""
    rpt = _report()
    for bins in (rpt.EDIT_BINS, rpt.GAP_BINS):
        assert bins[0][0] == 0.0
        assert bins[-1][1] == math.inf
        for (lo1, hi1), (lo2, _) in zip(bins, bins[1:]):
            assert hi1 == lo2, f"gap between {hi1} and {lo2}"
        for v in (0.0, 0.049, 0.05, 0.5, 1.0, 50.0):
            assert rpt._bin(v, bins) != "nan", f"{v} fell outside every bin"


def test_agg_ignores_nan_and_reports_its_own_n():
    """An aggregate over a column with NaNs must say how many values it actually used."""
    rpt = _report()
    out = rpt._agg([1.0, 2.0, float("nan"), 3.0])
    assert out["n"] == 3 and out["median"] == 2.0
    assert rpt._agg([float("nan")])["n"] == 0
    assert rpt._agg([])["n"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))


# --------------------------------------------------- the real-LP openness reference


def test_open_threshold_is_far_below_real_openness():
    """The report's openness narrative rests on an ORDERING, so assert the ordering.

    `min_open_gap_nm` = 0.45 is a floor for "not still a ring", and the report used to treat
    clearing it as being open. It is the 1.1st percentile of real bound-LP openness. If
    someone later raises the threshold toward the real median, the report's "the first two
    bins are below the 2nd percentile of real" line stops being true and has to be rewritten
    with it -- this test is what makes that a failure instead of a stale sentence.
    """
    from cp2lp_bb_report import REAL_LP_GAP_PCTL_NM

    from proteinfoundation.cp2lp.losses import MIN_OPEN_TERMINAL_GAP_NM

    assert MIN_OPEN_TERMINAL_GAP_NM < REAL_LP_GAP_PCTL_NM[5], (
        "the open threshold is no longer below the 5th percentile of real openness; the "
        "report's reference text and the release configs' headers describe a different regime"
    )
    assert REAL_LP_GAP_PCTL_NM[50] > 4.0 * MIN_OPEN_TERMINAL_GAP_NM, (
        "real openness is no longer several times the threshold -- re-derive the gap bins"
    )
    ps = sorted(REAL_LP_GAP_PCTL_NM)
    vals = [REAL_LP_GAP_PCTL_NM[p] for p in ps]
    assert vals == sorted(vals), "percentiles are not monotonic, so one of them is wrong"


def test_real_lp_gap_reference_matches_the_measured_rows():
    """The hardcoded reference must still be what the reference DATA says.

    It is a copied-in measurement, which is the kind of constant that drifts from its source
    silently -- the reference set can be regenerated with a different AE or a different
    receptor split and nothing would flag the stale numbers. Recomputes from the rows when
    they are reachable and skips when they are not, so this is a guard on a workstation and
    not an obstacle in CI.
    """
    import glob
    import json

    import numpy as np

    from cp2lp_bb_report import REAL_LP_GAP_PCTL_NM

    pattern = (
        "/zfsauton/scratch/yixiz/Proteina-Complexa/training_runs/store/cp2lp_triplets/"
        "real_reference/seed0/*.jsonl"
    )
    files = glob.glob(pattern)
    if not files:
        pytest.skip("real_reference rows not reachable from here")

    gaps = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            # A corrupt row must FAIL rather than be skipped: concurrent appends have
            # NUL-corrupted shared output files in this project before, and silently
            # dropping the bad lines is how a reference quietly loses half its population.
            row = json.loads(line)
            gaps.append(float(row["scores"]["terminal_gap_nm"]))

    assert len(gaps) > 400, f"reference population shrank to {len(gaps)}; re-derive the constant"
    for p, want in REAL_LP_GAP_PCTL_NM.items():
        got = float(np.percentile(gaps, p))
        assert abs(got - want) < 0.01, (
            f"p{p} of the real gap is {got:.3f} nm, constant says {want:.3f} -- the reference "
            "set changed under the hardcoded number"
        )


# --------------------------------------------- the gates must be measurable, not defaulted


def _row_sel(**over):
    """A row/sel pair that passes every gate, so each test perturbs exactly one thing."""
    row = {"terminal_gap_nm": 1.8, "bb_chain_intact": 1.0}
    sel = {
        "clash_receptor_count": 0, "clash_self_count": 0,
        "sequence_exact": True, "contact_retention": 0.9,
    }
    row.update(over.pop("row", {})); sel.update(over.pop("sel", {}))
    return row, sel


def test_joint_success_passes_a_fully_measured_row():
    from cp2lp_bb_report import joint_success

    row, sel = _row_sel()
    ok, why = joint_success(row, sel, 0.45)
    assert ok and why == "", (ok, why)


@pytest.mark.parametrize("field", ["sequence_exact", "contact_retention"])
def test_joint_success_raises_when_a_gate_cannot_be_evaluated(field):
    """THE BUG THIS EXISTS FOR. `sequence_exact` defaulted through `.get(..., True)` and
    `contact_retention` through a NaN that made `ret == ret` false, so both gates passed
    silently. Neither field was ever produced -- nothing passes `source_contacts` /
    `source_sequence` into `score_packed` -- so the 2026-10-04 release evaluation reported
    "25% joint success" against three live gates while documenting five. An unmeasured
    condition is not a satisfied one.
    """
    from cp2lp_bb_report import joint_success

    row, sel = _row_sel()
    del sel[field]
    with pytest.raises(KeyError, match=field):
        joint_success(row, sel, 0.45)


@pytest.mark.parametrize("field", ["terminal_gap_nm", "bb_chain_intact"])
def test_joint_success_raises_on_missing_row_fields_too(field):
    from cp2lp_bb_report import joint_success

    row, sel = _row_sel()
    del row[field]
    with pytest.raises(KeyError, match=field):
        joint_success(row, sel, 0.45)


def test_joint_success_refuses_nan_retention():
    """A NaN means the source had no contacts to retain, i.e. the source complex was supplied
    but is not a bound complex. Scoring that as a pass is how the gate died the first time."""
    from cp2lp_bb_report import joint_success

    row, sel = _row_sel()
    sel["contact_retention"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        joint_success(row, sel, 0.45)


def test_joint_success_still_fails_each_real_condition():
    """The loud-on-missing change must not have turned any live gate into a pass."""
    from cp2lp_bb_report import joint_success

    for over, expect in (
        ({"row": {"terminal_gap_nm": 0.1}}, "not_open"),
        ({"row": {"bb_chain_intact": 0.0}}, "raw_backbone_broken"),
        ({"sel": {"sequence_exact": False}}, "sequence_mismatch"),
        ({"sel": {"clash_receptor_count": 3}}, "clash"),
        ({"sel": {"contact_retention": 0.1}}, "contacts_lost"),
    ):
        row, sel = _row_sel(**over)
        ok, why = joint_success(row, sel, 0.45)
        assert not ok and why == expect, (over, ok, why)
    # and packing failure short-circuits before any field is required
    assert joint_success({}, None, 0.45) == (False, "packing_failed")
