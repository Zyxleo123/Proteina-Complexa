#!/usr/bin/env python3
"""Checks for the ranking-rule comparison. No folds, no journal, no GPU.

What would go wrong silently:

1. **A rule scored on the folds it was computed from.** Then it confirms itself and every
   rule looks good. The halves are pinned here.
2. **A disqualified candidate treated as a bad candidate.** A rule that refuses to rank
   something must not thereby make it lose a target where everything was refused -- that
   turns an abstention into a pick.
3. **The shrinkage pulling the wrong way.** A candidate resting on one fold must move
   toward the target's average MORE than one resting on many, and if the candidates are
   indistinguishable on the selection half it must pull all the way.

Run:  .venv/bin/python script_utils/test_cpseq_rank_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpseq_rank_rules import (  # noqa: E402
    RULES,
    build_targets,
    evaluate,
    half_stats,
    score,
    shrinkage_context,
    split_half,
)

PASS = 0
FAIL = 0


def check(cond: bool, msg: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def fold(rank: int, *, valid: bool = True, dg: float | None = None,
         area: float | None = None, seq: str = "AAAA", arm: str = "llm_search") -> dict:
    return {
        "sequence": seq, "seed_rank": rank, "arm": arm, "stage": "ladder",
        "valid": valid,
        "best_valid_dG_separated": dg if valid else None,
        "refined_dG_separated_per_1000A2": area if valid else None,
    }


def cand(sel_energies, mea_energies, *, sel_n=None, mea_n=None) -> dict:
    """A candidate stated directly as its two halves."""
    sel = [fold(i, valid=e is not None, dg=e) for i, e in enumerate(sel_energies)]
    mea = [fold(100 + i, valid=e is not None, dg=e) for i, e in enumerate(mea_energies)]
    return {"sel": half_stats(sel), "mea": half_stats(mea)}


def test_halves() -> None:
    print("split_half / half_stats")
    check(split_half([{"seed_rank": r} for r in range(4)]) == (4, 2),
          "four folds split two and two")
    check(split_half([{"seed_rank": r} for r in range(10)]) == (10, 5),
          "ten folds split five and five")
    s = half_stats([fold(0, dg=-10.0), fold(1, valid=False), fold(2, dg=-20.0)])
    check(s["n"] == 3 and s["n_valid"] == 2, "failed folds are counted but not averaged")
    check(s["mean"] == -15.0 and s["median"] == -15.0,
          "the mean and median use only the passing folds")
    check(half_stats([fold(0, valid=False)])["mean"] is None,
          "no passing fold gives no energy rather than a zero")


def test_rules_direction() -> None:
    print("rule scores, lower is better")
    good = cand([-30.0, -30.0], [-30.0])
    bad = cand([-5.0, -5.0], [-5.0])
    ctx = {"grand": -17.5, "within": 0.0, "between": 100.0, "worst": -5.0}
    for rule in ("mean_valid", "median_valid", "min_ratio_half", "pass_then_energy",
                 "penalized_mean", "shrunk_mean"):
        a, b = score(rule, good, ctx), score(rule, bad, ctx)
        check(a is not None and b is not None and a < b,
              f"`{rule}` prefers the stronger candidate")


def test_abstention_is_not_a_bad_score() -> None:
    print("min_ratio_half")
    thin = cand([None, None, -40.0], [-12.0])     # one fold of three passed
    steady = cand([-20.0, -20.0, -20.0], [-19.0])
    ctx = {"grand": -26.7, "within": 0.0, "between": 50.0, "worst": -20.0}
    check(score("min_ratio_half", thin, ctx) is None,
          "a candidate passing one fold of three is refused, not scored")
    check(score("mean_valid", thin, ctx) == -40.0,
          "today's rule ranks that candidate on its single lucky fold")
    check(score("mean_valid", thin, ctx) < score("mean_valid", steady, ctx),
          "and therefore prefers it to the steady one -- the flaw being tested")
    check(score("min_ratio_half", steady, ctx) == -20.0,
          "the steady candidate is still ranked")

    # The boundary, pinned because it decides whether this rule can do anything at the
    # depth the pilots ran at. The threshold is "at least half", so with two selection
    # folds a candidate passing ONE of them is admitted -- meaning at depth 4 this rule
    # cannot exclude a single lucky fold. Only depth 6 or more gives it teeth.
    half = cand([None, -40.0], [-12.0])
    check(score("min_ratio_half", half, ctx) == -40.0,
          "one fold of two is exactly half, so it is admitted, not refused")

    # And a refusal must not win a target by default.
    t = [{"run": "r", "target": "t", "depth": 6, "mid": 3,
          "cands": {"THIN": thin, "STDY": steady}, "ctx": ctx}]
    rep = evaluate(t)
    picks = rep["picks"]["min_ratio_half"]
    check(len(picks) == 1 and picks[0]["sequence"] == "STDY",
          "with one candidate refused, the rule picks the other one")
    check(rep["rules"]["min_ratio_half"]["abstained_on"] == 0,
          "that is a pick, not an abstention")


def test_penalized_mean_charges_failures() -> None:
    print("penalized_mean")
    lucky = cand([None, -40.0], [-12.0])
    ctx = {"grand": -30.0, "within": 0.0, "between": 50.0, "worst": -5.0}
    # One pass at -40 and one failure charged at -5 averages to -22.5.
    check(score("penalized_mean", lucky, ctx) == -22.5,
          "a failed fold is charged the worst energy on the target, not dropped")
    check(score("penalized_mean", lucky, ctx) > score("mean_valid", lucky, ctx),
          "so failing can no longer improve a candidate's score")


def test_shrinkage() -> None:
    print("shrunk_mean")
    thin = cand([-40.0], [-12.0])
    thick = cand([-40.0, -40.0, -40.0, -40.0], [-38.0])
    ctx = {"grand": -20.0, "within": 16.0, "between": 4.0, "worst": -5.0}
    s_thin, s_thick = score("shrunk_mean", thin, ctx), score("shrunk_mean", thick, ctx)
    check(s_thin > s_thick,
          "one fold of evidence is pulled toward the target average more than four")
    check(-40.0 < s_thick < -20.0 and -40.0 < s_thin < -20.0,
          "both land between the candidate's own mean and the target average")
    flat = {"grand": -20.0, "within": 16.0, "between": 0.0, "worst": -5.0}
    check(score("shrunk_mean", thin, flat) == -20.0,
          "candidates indistinguishable on the selection half all shrink to the average")

    # The context itself, estimated from data: identical candidates must give no
    # between-candidate variance, so the rule cannot invent a difference.
    same = {f"S{i}": cand([-20.0, -20.0], [-20.0]) for i in range(5)}
    c = shrinkage_context(same)
    check(c["within"] == 0.0 and c["between"] == 0.0,
          "five identical candidates have neither kind of variance")
    spread = {f"S{i}": cand([-10.0 * i, -10.0 * i], [-10.0 * i]) for i in range(1, 6)}
    c2 = shrinkage_context(spread)
    check(c2["between"] is not None and c2["between"] > 0,
          "genuinely different candidates do have between-candidate variance")


def test_build_targets_requires_both_halves() -> None:
    print("build_targets")
    rows = (
        [fold(r, dg=-10.0, seq="AAAA") for r in range(4)]
        + [fold(r, dg=-20.0, seq="CCCC") for r in range(4)]
        + [fold(r, dg=-30.0, seq="GGGG") for r in range(2)]   # selection half only
        + [fold(r, dg=-1.0, seq="REFR", arm="reference") for r in range(4)]
    )
    ts = build_targets([(Path("/run"), {"t1": rows})], 2)
    check(len(ts) == 1, "one target was built")
    cands = ts[0]["cands"]
    check(set(cands) == {"AAAA", "CCCC"},
          "a candidate folded in only one half cannot be evaluated and is dropped")
    check("REFR" not in cands,
          "the unmutated baseline is not a candidate the search may pick")
    check(ts[0]["ctx"]["worst"] == -10.0,
          "the penalty level is the worst selection-half energy on that target")
    check(build_targets([(Path("/run"), {"t1": rows})], 5) == [],
          "a target with too few candidates offers no ranking decision")


def test_held_out_is_held_out() -> None:
    print("evaluate")
    # Candidate A looks best on the selection half and is worst on the measurement half.
    a = cand([-50.0, -50.0], [-1.0])
    b = cand([-20.0, -20.0], [-19.0])
    c = cand([-10.0, -10.0], [-11.0])
    ctx = {"grand": -26.7, "within": 0.0, "between": 100.0, "worst": -10.0}
    t = [{"run": "r", "target": "t", "depth": 4, "mid": 2,
          "cands": {"AAAA": a, "BBBB": b, "CCCC": c}, "ctx": ctx}]
    rep = evaluate(t)
    pick = rep["picks"]["mean_valid"][0]
    check(pick["sequence"] == "AAAA", "the rule picks what the selection half says")
    check(pick["held_out"] == -1.0,
          "but is scored on the measurement half, where that pick is worst")
    check(pick["best_possible"] == -19.0 and pick["regret"] == 18.0,
          "regret is the gap to the best candidate available")
    check(rep["rules"]["mean_valid"]["picked_the_true_best"] == 0,
          "and it did not pick the true best")
    check(rep["rules"]["mean_valid"]["rho"] is not None,
          "the rank correlation against held-out truth is reported")


def test_unusable_pick_is_recorded() -> None:
    print("picked_unusable")
    doomed = cand([-99.0, -99.0], [None])   # nothing passes in the held-out half
    ok = cand([-20.0, -20.0], [-19.0])
    ctx = {"grand": -59.5, "within": 0.0, "between": 100.0, "worst": -20.0}
    t = [{"run": "r", "target": "t", "depth": 4, "mid": 2,
          "cands": {"DOOM": doomed, "GOOD": ok}, "ctx": ctx}]
    rep = evaluate(t)
    check(rep["rules"]["mean_valid"]["picked_unusable"] == 1,
          "picking a candidate with no passing held-out fold is counted")
    check(rep["rules"]["mean_valid"]["n_targets_ranked"] == 0,
          "and it contributes no energy, rather than a missing value averaged in")


def main() -> int:
    test_halves()
    test_rules_direction()
    test_abstention_is_not_a_bad_score()
    test_penalized_mean_charges_failures()
    test_shrinkage()
    test_build_targets_requires_both_halves()
    test_held_out_is_held_out()
    test_unusable_pick_is_recorded()
    print(f"\n{PASS} passed, {FAIL} failed  ({len(RULES)} rules)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
