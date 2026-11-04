#!/usr/bin/env python3
"""Checks for the selection-transfer decomposition. No GPU, no folds on disk.

This script separates "the arm re-picked under the new setting" (A) from "the old
setting's winner, measured under the new one" (B), and that distinction is what decides
whether a measured change is attributed to selection or to the baseline. So the cases
below are built so the right answer is known by construction, including the ones where a
plausible implementation silently returns the OTHER quantity.

Run:  python script_utils/test_cpseq_selection_transfer.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GATE = "v3-cis-amide-muco-windows"
DEPTH = 10

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))


spec = importlib.util.spec_from_file_location(
    "tr", HERE / "cpseq_selection_transfer.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def rows_for(pid, seq, arm, draw_index, energies):
    """One row per fold rank; None energy means that fold failed the gate."""
    out = []
    for rank, e in enumerate(energies):
        out.append({
            "parent_id": pid, "stage": "ladder", "arm": arm, "sequence": seq,
            "draw_index": draw_index, "seed_rank": rank, "scoring_version": GATE,
            "valid": e is not None,
            "refined_dG_separated": e, "best_valid_dG_separated": e,
        })
    return out


def write_run(root: Path, pid: str, ref: list, draws: dict[int, list]):
    d = root / "state" / pid
    d.mkdir(parents=True, exist_ok=True)
    rows = rows_for(pid, "REFSEQ", "reference", None, ref)
    for di, es in draws.items():
        rows += rows_for(pid, f"DRAW{di}", "llm_search", di, es)
    (d / "outcomes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))


def run(off: Path, on: Path):
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "out.json"
        r = subprocess.run(
            [sys.executable, str(HERE / "cpseq_selection_transfer.py"),
             "--off-dir", str(off), "--on-dir", str(on),
             "--arm", "llm_search", "--scoring-version", GATE, "--json", str(j)],
            capture_output=True, text=True, cwd=str(REPO))
        doc = json.loads(j.read_text()) if j.exists() else None
        return r, doc


def test_same_pick_makes_A_equal_B() -> None:
    print("\n[when both runs pick the same draw, A and B must be identical]")
    # Draw 1 is best on every fold in both runs, so selection cannot differ.
    with tempfile.TemporaryDirectory() as td:
        off, on = Path(td) / "off", Path(td) / "on"
        write_run(off, "c1", [-20.0] * DEPTH,
                  {0: [-21.0] * DEPTH, 1: [-30.0] * DEPTH})
        write_run(on, "c1", [-25.0] * DEPTH,
                  {0: [-26.0] * DEPTH, 1: [-35.0] * DEPTH})
        r, doc = run(off, on)
        check("runs", r.returncode == 0, r.stdout + r.stderr)
        row = doc["per_complex"][0]
        check("A == B exactly", row["reselected_on"] == row["transferred_on"],
              f"{row['reselected_on']} vs {row['transferred_on']}")
        check("same_pick reported true", row["same_pick"] is True)
        # ON: picked draw 1 at -35, reference -25 => -10
        check("A is candidate minus THIS run's reference",
              abs(row["reselected_on"] - (-10.0)) < 1e-9, str(row["reselected_on"]))


def test_divergent_pick_separates_A_from_B() -> None:
    print("\n[when the runs pick differently, B must follow the OFF winner]")
    # OFF prefers draw 0; ON prefers draw 1. Under ON, draw 0 is mediocre.
    # A (re-picked) must read draw 1's ON energy; B must read draw 0's ON energy.
    with tempfile.TemporaryDirectory() as td:
        off, on = Path(td) / "off", Path(td) / "on"
        write_run(off, "c1", [-20.0] * DEPTH,
                  {0: [-40.0] * DEPTH, 1: [-22.0] * DEPTH})
        write_run(on, "c1", [-20.0] * DEPTH,
                  {0: [-25.0] * DEPTH, 1: [-50.0] * DEPTH})
        r, doc = run(off, on)
        row = doc["per_complex"][0]
        check("same_pick reported false", row["same_pick"] is False)
        check("A follows the ON winner (draw 1): -50 - -20 = -30",
              abs(row["reselected_on"] - (-30.0)) < 1e-9, str(row["reselected_on"]))
        check("B follows the OFF winner (draw 0): -25 - -20 = -5",
              abs(row["transferred_on"] - (-5.0)) < 1e-9, str(row["transferred_on"]))
        check("A is better than B here, as constructed",
              row["reselected_on"] < row["transferred_on"])


def test_reference_move_is_reported_separately() -> None:
    print("\n[a pure baseline shift leaves the candidate's energy alone]")
    # The candidate is identical in both runs; only the reference improves. The deltas
    # must move by exactly the reference's change and nothing else -- this is the case
    # that distinguishes "the arm got worse" from "the baseline got better", which is
    # the whole reason this script exists.
    with tempfile.TemporaryDirectory() as td:
        off, on = Path(td) / "off", Path(td) / "on"
        write_run(off, "c1", [-20.0] * DEPTH, {0: [-30.0] * DEPTH})
        write_run(on, "c1", [-24.0] * DEPTH, {0: [-30.0] * DEPTH})
        r, doc = run(off, on)
        row = doc["per_complex"][0]
        check("reference energy change is -4.00",
              abs((row["ref_median_on"] - row["ref_median_off"]) - (-4.0)) < 1e-9,
              f"{row['ref_median_off']} -> {row['ref_median_on']}")
        # ON delta is -30 - -24 = -6; OFF delta was -30 - -20 = -10. The margin shrank
        # by 4, entirely from the reference.
        check("A reflects the new reference: -6.00",
              abs(row["reselected_on"] - (-6.0)) < 1e-9, str(row["reselected_on"]))


def test_min_ref_valid_and_depth_are_enforced() -> None:
    print("\n[complexes without a usable baseline in BOTH runs are dropped]")
    with tempfile.TemporaryDirectory() as td:
        off, on = Path(td) / "off", Path(td) / "on"
        # thin: reference passes twice under ON, below the min of 3
        write_run(off, "thin", [-20.0] * DEPTH, {0: [-30.0] * DEPTH})
        write_run(on, "thin", [-20.0] * 2 + [None] * 8, {0: [-30.0] * DEPTH})
        # shallow: fewer than DEPTH folds on the ON reference
        write_run(off, "shallow", [-20.0] * DEPTH, {0: [-30.0] * DEPTH})
        write_run(on, "shallow", [-20.0] * 6, {0: [-30.0] * DEPTH})
        # good: survives
        write_run(off, "good", [-20.0] * DEPTH, {0: [-30.0] * DEPTH})
        write_run(on, "good", [-20.0] * DEPTH, {0: [-30.0] * DEPTH})
        r, doc = run(off, on)
        got = {x["parent_id"] for x in doc["per_complex"]}
        check("thin baseline dropped", "thin" not in got, str(got))
        check("shallow reference dropped", "shallow" not in got, str(got))
        check("the usable complex is kept", got == {"good"}, str(got))


def test_uses_the_reports_own_median_rule() -> None:
    print("\n[the energy rule is the report's: median over PASSING folds only]")
    # A failed fold must not drag the median. Ranks 5..9 are the measurement half:
    # two passing folds at -30 and -40, plus three failures -> median -35, not a
    # value computed over five entries.
    with tempfile.TemporaryDirectory() as td:
        off, on = Path(td) / "off", Path(td) / "on"
        write_run(off, "c1", [-20.0] * DEPTH, {0: [-30.0] * DEPTH})
        write_run(on, "c1", [-20.0] * DEPTH,
                  {0: [-1.0] * 5 + [-30.0, -40.0, None, None, None]})
        r, doc = run(off, on)
        row = doc["per_complex"][0]
        # direction 1 measures on [5,10): median(-30,-40) = -35, ref -20 => -15
        # direction 2 measures on [0,5):  median(-1 x5)  = -1,  ref -20 => +19
        # averaged: (-15 + 19) / 2 = +2
        check("median ignores failed folds and both directions average",
              abs(row["reselected_on"] - 2.0) < 1e-9, str(row["reselected_on"]))


def main() -> int:
    test_same_pick_makes_A_equal_B()
    test_divergent_pick_separates_A_from_B()
    test_reference_move_is_reported_separately()
    test_min_ref_valid_and_depth_are_enforced()
    test_uses_the_reports_own_median_rule()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
