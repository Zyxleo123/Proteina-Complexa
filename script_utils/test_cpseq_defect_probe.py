#!/usr/bin/env python3
"""Checks for the defect probe. No GPU, no structures, no journal on disk.

The three mistakes that would produce a plausible-looking and wrong answer, each pinned
here:

1. **Off-by-one between the summary and the sequence.** Interface summaries number
   positions from 1, sequences are indexed from 0. Getting this wrong shifts every flag
   by one residue and the probe would still report a tidy table.
2. **Scoring a flag on the folds it came from.** Defects are read from the selection half
   and outcomes from the measurement half; if either half leaked, a flag would confirm
   itself.
3. **Keeping the retired gate's verdict.** A re-score appends rather than replaces
   (§14), so the reader has to keep the NEWEST row per fold and filter to one protocol.

Run:  .venv/bin/python script_utils/test_cpseq_defect_probe.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpseq_defect_probe import (  # noqa: E402
    _sign_test,
    classify_position,
    flags_from_summaries,
    load_rows,
    outcome,
    representative_row,
    score_units,
    split_halves,
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


def pos(**kw) -> dict:
    """A summary position with defect-free defaults, so each test states only what it
    is about."""
    base = {
        "pos": 1,
        "residue": "L",
        "min_dist_A": 3.5,
        "n_contacts": 10,
        "n_clashes": 0,
        "worst_clash_A": None,
        "buried_fraction": 0.6,
        "ss_approx": "H",
        "partners": [],
        "interactions": [],
        "clash_partners": [],
    }
    base.update(kw)
    return base


def test_classify() -> None:
    print("classify_position")
    check(classify_position(pos()) == set(), "a buried hydrophobic with contacts is clean")
    check("clash" in classify_position(pos(n_clashes=1)), "an overlap is a clash")
    check(
        "empty" in classify_position(
            pos(buried_fraction=0.0, min_dist_A=6.1, n_contacts=0)
        ),
        "no burial, far, no contacts is an empty position",
    )
    # Each of the three conditions alone must not flag: a residue can read 0.00 burial
    # while sitting 3.5 A off the receptor and making contacts, and that is a shallow
    # contact to leave alone, not a wasted position.
    check(
        "empty" not in classify_position(
            pos(buried_fraction=0.0, min_dist_A=3.5, n_contacts=9)
        ),
        "zero burial with contacts nearby is not an empty position",
    )
    check(
        "empty" not in classify_position(
            pos(buried_fraction=0.0, min_dist_A=6.1, n_contacts=3)
        ),
        "far but still contacting is not an empty position",
    )
    check(
        "unsat_polar" in classify_position(pos(residue="N", buried_fraction=0.8)),
        "a buried asparagine with no polar partner is unsatisfied",
    )
    check(
        "unsat_polar" not in classify_position(
            pos(residue="N", buried_fraction=0.8,
                interactions=[{"type": "hbond", "partner": "A:Glu9"}])
        ),
        "an h-bond satisfies it",
    )
    check(
        "unsat_polar" not in classify_position(pos(residue="N", buried_fraction=0.2)),
        "an exposed polar pays no desolvation cost, so it is not a defect",
    )
    check(
        "unsat_polar" not in classify_position(pos(residue="L", buried_fraction=0.9)),
        "a buried leucine is not an unsatisfied polar",
    )
    lab = classify_position(pos(residue="K", buried_fraction=0.9))
    check("buried_charged" in lab and "unsat_polar" in lab,
          "a buried unpaired lysine is both charged and unsatisfied")
    check(
        "buried_charged" not in classify_position(
            pos(residue="K", buried_fraction=0.9,
                interactions=[{"type": "saltbridge", "partner": "A:Asp112"}])
        ),
        "a salt bridge satisfies a buried charge",
    )


def test_recurrence_levels() -> None:
    print("flags_from_summaries")
    # Position 4 (1-based) clashes in one fold of two; position 9 clashes in both.
    f1 = {"ok": True, "positions": [pos(pos=4, n_clashes=2), pos(pos=9, n_clashes=1)]}
    f2 = {"ok": True, "positions": [pos(pos=4), pos(pos=9, n_clashes=1)]}
    fl = flags_from_summaries([f1, f2], representative=f1)
    check(set(fl["any"]) == {3, 8}, "`any` flags a defect seen in one fold of two")
    check(set(fl["all"]) == {8}, "`all` keeps only the defect seen in every fold")
    check(set(fl["single"]) == {3, 8}, "`single` reads the representative fold alone")
    check(fl["_n_selection_folds"] == 2, "the fold count is carried for the report")
    # The off-by-one: summary position 4 must become sequence index 3.
    check(3 in fl["any"] and 4 not in fl["any"],
          "summary pos 4 maps to sequence index 3")
    fl2 = flags_from_summaries([f1, f2], representative=f2)
    check(set(fl2["single"]) == {8},
          "a different representative fold gives a different single-fold flag set")
    check(flags_from_summaries([], representative=None)["any"] == {},
          "no usable summary flags nothing")
    check(
        flags_from_summaries([{"ok": False, "error": "x"}], representative=None)["any"]
        == {},
        "a failed summary is not evidence of a clean interface",
    )


def test_halves_and_outcome() -> None:
    print("split_halves / outcome")
    rows = [{"seed_rank": r} for r in range(4)]
    check(split_halves(rows) == (4, 2), "depth 4 splits at rank 2")
    rows = [{"seed_rank": r} for r in range(10)]
    check(split_halves(rows) == (10, 5), "depth 10 splits at rank 5")

    folds = [
        {"seed_rank": 0, "valid": True, "best_valid_dG_separated": -100.0},
        {"seed_rank": 1, "valid": True, "best_valid_dG_separated": -100.0},
        {"seed_rank": 2, "valid": True, "best_valid_dG_separated": -10.0},
        {"seed_rank": 3, "valid": False, "best_valid_dG_separated": None},
    ]
    m = outcome(folds, min_rank=2, max_rank=None)
    check(m["dG"] == -10.0,
          "the measurement half ignores the selection half entirely")
    check(m["n_folds"] == 2 and m["n_valid"] == 1 and m["pass_frac"] == 0.5,
          "pass fraction counts every measurement fold, valid or not")
    s = outcome(folds, min_rank=0, max_rank=2)
    check(s["dG"] == -100.0, "the selection half is addressable on its own")
    check(outcome(folds, min_rank=9, max_rank=None)["dG"] is None,
          "no folds in range yields no energy rather than a zero")


def test_representative_row() -> None:
    print("representative_row")
    rows = [
        {"valid": True, "best_valid_dG_separated": -5.0, "raw_pdb_path": "a"},
        {"valid": True, "best_valid_dG_separated": -9.0, "raw_pdb_path": "b"},
        {"valid": False, "min_violation": 0.1, "raw_pdb_path": "c"},
    ]
    check(representative_row(rows)["raw_pdb_path"] == "b",
          "the best-energy valid fold is what the live prompt would have shown")
    check(
        representative_row([
            {"valid": False, "min_violation": 3.0, "raw_pdb_path": "x"},
            {"valid": False, "min_violation": 0.5, "raw_pdb_path": "y"},
        ])["raw_pdb_path"] == "y",
        "with nothing valid, the smallest violation stands in",
    )
    check(
        representative_row([{"valid": True, "technical_failure": True,
                             "raw_pdb_path": "z"}]) is None,
        "a crashed fold has no structure to summarise",
    )


def test_newest_row_wins() -> None:
    print("load_rows")
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        d = run / "state" / "lnr_x"
        d.mkdir(parents=True)
        old = {"sequence": "AAA", "seed_rank": 0, "arm": "llm_search",
               "stage": "ladder", "valid": False,
               "scoring_version": "v5-junction-restrained-relax"}
        new = dict(old, valid=True)
        v4 = dict(old, scoring_version="v4-postrelax-validity", valid=True)
        crash = {"sequence": "BBB", "seed_rank": 0, "arm": "llm_search",
                 "stage": "ladder", "technical_failure": True,
                 "scoring_version": "v5-junction-restrained-relax"}
        (d / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(r) for r in (v4, old, new, crash)) + "\n"
        )
        by_parent, census = load_rows(run, "v5-junction-restrained-relax")
        rows = by_parent["lnr_x"]
        check(len(rows) == 1 and rows[0]["valid"] is True,
              "the newest row per fold wins, and the v4 re-score is filtered out")
        check(census["v4-postrelax-validity"] == 1,
              "the census still reports the protocol mix it filtered")
        check(all(not r.get("technical_failure") for r in rows),
              "crashed folds are dropped")


def test_hit_attribution() -> None:
    print("score_units")
    base, cand = "AAAACAAAA", "AAAAKAAAA"  # one edit, at index 4
    unit = {
        "run": "r", "run_dir": "/r", "parent_id": "p", "mode": "elite",
        "base_sequence": base, "candidate_sequence": cand,
        "edited_vs_base": [4], "n_edits_vs_base": 1,
        "depth": 4, "measurement_min_rank": 2,
        "selection_hashes": ["h1"], "representative_hash": "h1",
        "base_outcome": {"dG": -10.0, "dG_norm": -1.0, "pass_frac": 0.5,
                         "n_folds": 2, "n_valid": 1},
        "cand_outcome": {"dG": -14.0, "dG_norm": -1.4, "pass_frac": 1.0,
                         "n_folds": 2, "n_valid": 2},
    }
    # The flag sits at summary position 5, which is sequence index 4 -- the edited one.
    summaries = {("/r", "p", "h1"): {
        "ok": True,
        "positions": [pos(pos=5, residue="C", buried_fraction=0.9, n_clashes=1)],
    }}
    rec = score_units([unit], summaries)[0]
    check(rec["levels"]["any"]["hit"] is True,
          "an edit on the flagged position registers as a hit")
    check(rec["levels"]["any"]["labels_hit"] == ["buried_charged", "clash", "unsat_polar"]
          or "clash" in rec["levels"]["any"]["labels_hit"],
          "the defect classes hit are recorded")
    check(rec["d_dG"] == -4.0, "a negative delta means the edit improved on its base")
    check(rec["d_pass"] == 0.5, "pass fraction delta is carried alongside")
    check(rec["levels"]["any"]["flagged_share"] == round(1 / 9, 3),
          "the flagged share is reported, so a vacuous flag set is visible")

    off = dict(unit, candidate_sequence="AAAAACAAA", edited_vs_base=[4, 5],
               n_edits_vs_base=2)
    rec2 = score_units([off], summaries)[0]
    check(rec2["levels"]["any"]["hit_share"] == 0.5,
          "with two edits and one on a flag, the share is a half")

    elsewhere = dict(unit, candidate_sequence="KAAACAAAA", edited_vs_base=[0])
    rec3 = score_units([elsewhere], summaries)[0]
    check(rec3["levels"]["any"]["hit"] is False,
          "an edit away from every flag is a miss")


def test_sign_test() -> None:
    print("_sign_test")
    s = _sign_test([-1.0, -2.0, -3.0, -4.0, -5.0])
    check(s["n"] == 5 and s["n_better"] == 5 and s["p"] is not None and s["p"] < 0.07,
          "five improvements out of five is a small two-sided p")
    s = _sign_test([-1.0, 1.0, -1.0, 1.0])
    check(s["p"] == 1.0, "an even split is p = 1")
    s = _sign_test([0.0, 0.0])
    check(s["n"] == 0 and s["p"] is None,
          "ties are dropped rather than counted as agreement")


def main() -> int:
    test_classify()
    test_recurrence_levels()
    test_halves_and_outcome()
    test_representative_row()
    test_newest_row_wins()
    test_hit_attribution()
    test_sign_test()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
