#!/usr/bin/env python3
"""Checks for the head-to-head arm comparison. No GPU, no folds on disk.

This script produces the number the whole experiment exists to produce, so its logic is
worth testing on hand-built folds where the right answer is known by construction. The
cases below are the ones where a plausible implementation gives a confidently wrong
answer rather than an error.

Run:  python script_utils/test_cpseq_arm_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpseq_arm_compare import arm_pick, arm_validity, compare  # noqa: E402
from cpseq_ladder_report import Sequence  # noqa: E402

PASS, FAIL = 0, 0
DEPTH = 10
HALF = DEPTH // 2


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))


def seq(energies: list[float | None], draw_index: int = 0) -> Sequence:
    """A Sequence from one energy per fold rank; None means that fold failed the gate."""
    rows = []
    for rank, e in enumerate(energies):
        rows.append({
            "seed_rank": rank,
            "draw_index": draw_index,
            "valid": e is not None,
            "best_valid_dG_separated": e,
        })
    return Sequence(rows)


def test_crossfit_splits_folds() -> None:
    print("\n[cross-fit uses disjoint folds to choose and to measure]")
    # Draw 0 looks best on the selection half and is terrible on the measurement half.
    # Draw 1 is mediocre on selection and good on measurement. Best-of-B picks by all
    # folds; cross-fit must pick on the first half ONLY, then report the second half.
    d0 = seq([-50.0] * HALF + [-1.0] * HALF, draw_index=0)
    d1 = seq([-10.0] * HALF + [-40.0] * HALF, draw_index=1)
    draws = {0: d0, 1: d1}

    e, status = arm_pick(draws, 2, DEPTH, k_needed=HALF, crossfit=True)
    check("chooses on the selection half, scores on the measurement half",
          status == "ok" and e == -1.0,
          f"got {e} ({status}); -50 selection winner must score its -1 measurement half")

    f, _ = arm_pick(draws, 2, DEPTH, k_needed=HALF, crossfit=False)
    check("best-of-B instead takes the best over all folds",
          f == min(d0.median_energy(DEPTH), d1.median_energy(DEPTH)), f"got {f}")
    check("the two differ here, which is the selection bias being removed", e != f)


def test_unmeasurable_is_not_a_win_or_a_loss() -> None:
    print("\n[a candidate chosen then unmeasurable is excluded, not scored]")
    # Passes every selection fold, fails every measurement fold.
    d = {0: seq([-99.0] * HALF + [None] * HALF, draw_index=0)}
    e, status = arm_pick(d, 1, DEPTH, k_needed=1, crossfit=True)
    check("returns no energy", e is None, f"got {e}")
    check("and says why", status == "chosen candidate passed no measurement fold", status)

    print("\n[an arm with no passing selection fold is reported, not scored as 0]")
    d2 = {0: seq([None] * DEPTH, draw_index=0)}
    e2, st2 = arm_pick(d2, 1, DEPTH, k_needed=1, crossfit=True)
    check("no energy", e2 is None)
    check("distinct reason from the unmeasurable case",
          st2 == "no draw passed a selection-half fold", st2)

    print("\n[a short prefix is refused rather than silently compared at a lower B]")
    d3 = {0: seq([-5.0] * DEPTH, draw_index=0)}
    e3, st3 = arm_pick(d3, 4, DEPTH, k_needed=1, crossfit=True)
    check("B=4 against 1 present draw yields nothing", e3 is None)
    check("and names the shortfall", "1/4 draws present" in st3, st3)


def test_validity_is_separate_from_energy() -> None:
    print("\n[validity]")
    # Passes 3 of 10 folds: below a 5/10 reliability bar, but it does pass sometimes.
    d = {0: seq([-20.0, -20.0, -20.0] + [None] * 7, draw_index=0)}
    v = arm_validity(d, 1, DEPTH, k_needed=5)
    check("not reliable at 5/10", v["has_reliable"] is False)
    check("but recorded as passing at least once", v["has_any_pass"] is True)
    check("per-fold pass rate is 0.3", abs(v["fold_pass_rate"] - 0.3) < 1e-9,
          f"{v['fold_pass_rate']}")


def _state(per_arm_draws: dict[str, dict[int, Sequence]]) -> dict[str, dict]:
    return {
        pid: {"reference": None, "draws": draws, "source": None, "corpus": "LNR"}
        for pid, draws in per_arm_draws.items()
    }


def test_compare_pairs_within_complex() -> None:
    print("\n[compare: sign test counts complexes, and B is matched across arms]")
    # Three complexes. Arm B wins two and loses one on the measurement half.
    a = {}
    b = {}
    for pid, (ea, eb) in {
        "c1": (-10.0, -20.0),   # B better
        "c2": (-10.0, -20.0),   # B better
        "c3": (-10.0, -5.0),    # B worse
    }.items():
        a[pid] = {0: seq([-30.0] * HALF + [ea] * HALF, 0)}
        b[pid] = {0: seq([-30.0] * HALF + [eb] * HALF, 0)}
    rows = compare(_state(a), _state(b), [1], DEPTH, reliability=0.5)
    check("one row for the single rung", len(rows) == 1, f"{len(rows)}")
    r = rows[0]
    check("all three complexes paired", r["n_energy_pairs"] == 3, f"{r}")
    check("2 wins / 1 loss for arm B",
          (r["crossfit_wins_b"], r["crossfit_losses_b"]) == (2, 1), f"{r}")
    check("median delta is negative (arm B better)",
          r["crossfit_median_delta"] == -10.0, f"{r['crossfit_median_delta']}")
    check("sign p for 2/1 is 1.0", abs(r["crossfit_sign_p"] - 1.0) < 1e-9,
          f"{r['crossfit_sign_p']}")

    print("\n[a rung one arm cannot reach is dropped, not compared unmatched]")
    # Arm A has 4 draws, arm B only 1. B=4 must not be reported.
    a2 = {"c1": {i: seq([-10.0] * DEPTH, i) for i in range(4)}}
    b2 = {"c1": {0: seq([-99.0] * DEPTH, 0)}}
    rows2 = compare(_state(a2), _state(b2), [1, 4], DEPTH, reliability=0.5)
    reported = [r["B"] for r in rows2]
    check("B=1 reported, B=4 dropped", reported == [1], f"{reported}")

    print("\n[a complex where only one arm is measurable is counted, not dropped quietly]")
    a3 = {"c1": {0: seq([-30.0] * HALF + [-10.0] * HALF, 0)}}
    b3 = {"c1": {0: seq([-30.0] * HALF + [None] * HALF, 0)}}
    rows3 = compare(_state(a3), _state(b3), [1], DEPTH, reliability=0.5)
    r3 = rows3[0]
    check("no energy pair formed", r3["n_energy_pairs"] == 0)
    check("and it is recorded as arm-A-only",
          r3["measurable_arm_a_only"] == 1 and r3["measurable_arm_b_only"] == 0, f"{r3}")


def main() -> int:
    print("arm comparison checks")
    test_crossfit_splits_folds()
    test_unmeasurable_is_not_a_win_or_a_loss()
    test_validity_is_separate_from_energy()
    test_compare_pairs_within_complex()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
