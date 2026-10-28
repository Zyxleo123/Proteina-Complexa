#!/usr/bin/env python3
"""Checks for the prompt ablation. No GPU, no model, no prompts on disk.

The failure this file exists to prevent: an ablation that removes nothing. A variant that
silently leaves the prompt intact produces a clean null result for that component, and
nothing downstream can tell it apart from a component the model ignores. So every
transform is tested for having changed what it claimed to change, and for leaving the
rest alone.

Also pinned: the table row format. `strip_overlap_mentions` and `scramble_tables` both
re-render rows from the parsed fields, so a change to `render_complex`'s f-string would
make them emit a subtly different table -- the regex and these tests are the tripwire.

Run:  .venv/bin/python script_utils/test_cpseq_prompt_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_prompt_ablation import (  # noqa: E402
    ROW_RE,
    VARIANTS,
    apply_variant,
    blocks_of,
    label_of,
    parse_facts,
    position_class,
    read_prompt_file,
    scramble_tables,
    score_completion,
    strip_overlap_mentions,
    true_table,
)
from proteinfoundation.cpseq.interactions import render_complex  # noqa: E402

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


def summary(seq: str) -> dict:
    """A summary whose rendered table is what the real renderer would produce."""
    positions = []
    for i, aa in enumerate(seq, start=1):
        positions.append({
            "pos": i,
            "residue": aa,
            "min_dist_A": 3.1 if i % 2 else 6.0,
            "n_contacts": 12 if i % 2 else 0,
            "n_clashes": 1 if i == 2 else 0,
            "worst_clash_A": 2.19 if i == 2 else None,
            "buried_fraction": 0.8 if i % 2 else 0.0,
            "ss_approx": "H" if i % 2 else "?",
            "partners": ["A:Val38"] if i % 2 else [],
            "interactions": (
                [{"type": "hbond", "partner": "A:His6", "dist_A": 3.2}] if i % 2 else []
            ),
            "clash_partners": ["A:Pro7"] if i == 2 else [],
        })
    return {
        "ok": True,
        "peptide_sequence_observed": seq,
        "peptide_sequence_expected": seq,
        "peptide_chain": "B",
        "buried_sasa_A2": 423.5,
        "n_interface_contacts": 58,
        "n_interface_clashes": 1,
        "sequence_matches": True,
        "plip": {"ok": True},
        "receptor_pocket": [],
        "positions": positions,
    }


ELITE = "GTPRRLLYCQR"
ORIGINAL = "GTPRRLLYCQA"


def prompt_text() -> str:
    """A prompt with the same block structure the production builder emits."""
    table = render_complex(summary(ELITE), title=ELITE)
    ref = render_complex(summary(ORIGINAL), title="reference")
    return "\n\n".join([
        f"TARGET: pdb3x23\nORIGINAL LINEAR PEPTIDE: {ORIGINAL}  (length 11)\n"
        f"TOPOLOGY: head-to-tail cyclic -- residue 11 is bonded to residue 1, so both "
        f"ends are constrained and there are no free termini.\n"
        f"SUBSTITUTION BUDGET: at most 5 positions may differ from the original "
        f"(that is floor(L/2); it is a ceiling, not a target).\n"
        f"ALLOWED RESIDUES FOR SUBSTITUTIONS: ADEFGHIKLMNPQRSTVWY",
        "HOW CANDIDATES ARE JUDGED\nA candidate is folded in complex with the target, "
        "then screened by a fixed geometry gate.\n"
        "Peptide-receptor overlap is by far the most common reason for rejection, so "
        "the 'ovl' column below is the single most important thing to act on.",
        "INTERFACE OF THE UNMUTATED REFERENCE (the cyclic peptide with the original "
        "sequence; this is the baseline every candidate is compared against)\n"
        "Columns: pos=position, aa=residue, ss=approximate secondary structure, "
        "burial=fraction of side-chain surface buried by the receptor (1.0 = fully "
        "enclosed), dmin=closest approach to the receptor in angstroms, ctc=receptor "
        "atoms within 4.5 A, ovl=steric overlaps (any is fatal).\n" + ref,
        "CANDIDATES ALREADY SCORED (3 total, 2 passed the gate; showing 3)\n"
        "Passed, best energy first:\n  GTPRRLLYCQR  (1 sub: A11R)  PASSED gate, mean "
        "energy -35.0 REU (lower is better) [2/2 fold(s) valid]",
        f"DO NOT PROPOSE ANY OF THESE ALREADY-SCORED SEQUENCES:\n  {ELITE}\n  "
        f"GTPRRLLYCQK",
        "The energies above are means over the valid folds among the first 2 of 4 "
        "structure-prediction seeds.",
        f"SEQUENCE TO IMPROVE -- edit THIS one, not the original\n  {ELITE}\n"
        f"It differs from the original at 1 position(s): A11R\n"
        f"It passed the gate on 2 of 2 fold(s), mean interface energy -35.0 REU.",
        f"INTERFACE OF THE SEQUENCE TO IMPROVE {ELITE}\n" + table,
        "TASK\nPropose exactly 4 new sequences of length 11, each built by editing "
        f"{ELITE}.\nEnd with exactly 4 lines of the form:\nPROPOSE: <SEQUENCE> | <reason>",
    ])


def test_row_regex_matches_the_real_renderer() -> None:
    print("row format")
    rendered = render_complex(summary(ELITE), title=ELITE)
    rows = [l for l in rendered.splitlines() if ROW_RE.match(l)]
    check(len(rows) == len(ELITE),
          f"the regex matches every one of {len(ELITE)} rendered rows")
    m = ROW_RE.match(rows[1])
    check(m.group("aa") == ELITE[1] and int(m.group("pos")) == 2,
          "position and residue are read off the right columns")
    check(m.group("ovl").strip() == "1" and "OVERLAP" in m.group("detail"),
          "a clashing row carries both the ovl count and the OVERLAP annotation")


def test_labels_and_facts() -> None:
    print("block labels and facts")
    user = prompt_text()
    labels = [label_of(b) for b in blocks_of(user)]
    for want in ("target", "judging", "reference_table", "history_lines",
                 "do_not_repeat", "noise_note", "elite_block", "elite_table", "task"):
        check(want in labels, f"block `{want}` is recognised")
    check("other" not in labels, "every block in a production-shaped prompt is named")

    f = parse_facts(user)
    check(f["ok"] and f["original"] == ORIGINAL, "the original is read back")
    check(f["base"] == ELITE and f["is_offspring_prompt"],
          "the base is the elite, not the original")
    check("C" not in f["alphabet"] and "A" in f["alphabet"], "the alphabet is read back")
    check(f["already"] == {ELITE, "GTPRRLLYCQK"},
          "the do-not-repeat list becomes the `already` set")
    check(f["n_requested"] == 4 and f["stated_budget"] == 5,
          "the ask and the budget are read back")

    t = true_table(user, ELITE)
    check(t is not None and len(t) == len(ELITE),
          "the elite's own table is the structural truth")
    check(t[1]["ovl"] == 1.0 and t[0]["ctc"] == 12.0,
          "the table's numbers survive parsing")
    check(true_table(user, "WRONGSEQXYZ") is None,
          "a table that does not belong to the base sequence is refused")


def test_position_class() -> None:
    print("position_class")
    check(position_class({"ovl": 1, "ctc": 9, "burial": 0.8}) == "clash",
          "an overlap outranks everything else")
    check(position_class({"ovl": 0, "ctc": 0, "burial": 0.0}) == "empty",
          "no contacts is an empty position")
    check(position_class({"ovl": 0, "ctc": 12, "burial": 0.8}) == "anchor",
          "buried and contacting is an anchor the prompt says to keep")
    check(position_class({"ovl": 0, "ctc": 4, "burial": 0.1}) == "shallow",
          "contacting but not buried is shallow")


def test_every_variant_changes_something() -> None:
    print("variants change what they claim")
    user = prompt_text()
    for name in VARIANTS:
        got = apply_variant(user, name, scramble_seed=7)
        if name == "full":
            check(got["user"] == user, "`full` is the prompt unchanged")
            continue
        if name in {"no_scored_tables", "no_rejected_table"}:
            # Not present in this fixture; the harness skips such cells rather than
            # counting them as a no-op ablation, which is what `applicable` is for.
            check(not got["applicable"],
                  f"`{name}` reports itself inapplicable when the block is absent")
            continue
        check(got["applicable"] and got["user"] != user,
              f"`{name}` changed the prompt ({got['n_changed']} edit(s))")


def test_removals_are_surgical() -> None:
    print("removals leave the rest alone")
    user = prompt_text()
    got = apply_variant(user, "no_elite_table")
    check("INTERFACE OF THE SEQUENCE TO IMPROVE" not in got["user"],
          "the elite's table is gone")
    check("SEQUENCE TO IMPROVE -- edit THIS one" in got["user"],
          "the instruction to edit the elite stays, so the search still means the same")
    check("INTERFACE OF THE UNMUTATED REFERENCE" in got["user"],
          "the reference table is untouched")

    got = apply_variant(user, "no_any_table")
    check(ROW_RE.search(got["user"]) is None, "no table rows survive `no_any_table`")
    check("PASSED gate, mean energy -35.0 REU" in got["user"],
          "the verdicts and energies survive, so only the tables were removed")

    got = apply_variant(user, "no_history_lines")
    check("CANDIDATES ALREADY SCORED" not in got["user"], "the history lines are gone")
    check("DO NOT PROPOSE" in got["user"],
          "the do-not-repeat list is a separate block and stays")


def test_overlap_stripping_is_total() -> None:
    print("no_overlap_info")
    user = prompt_text()
    got = apply_variant(user, "no_overlap_info")
    u = got["user"]
    check("OVERLAP" not in u, "the OVERLAP annotation is gone from the detail column")
    check("ovl" not in u, "no mention of `ovl` survives, column or legend")
    check("overlaps=" not in u, "the per-table overlap total is gone")
    check("most important thing to act on" not in u,
          "the sentence calling overlap the most important signal is gone")
    check("burial" in u and "dmin" in u and "ctc" in u,
          "every other column is still there")
    rows = [l for l in u.splitlines() if l.startswith("    1 ") or l.startswith("    2 ")]
    check(all("  0.80" in r or "  0.00" in r for r in rows),
          "the surviving columns keep their values and alignment")


def test_scramble_preserves_shape() -> None:
    print("scramble_tables")
    user = prompt_text()
    blocks = blocks_of(user)
    out, n = scramble_tables(blocks, seed=3)
    before = "\n\n".join(blocks)
    after = "\n\n".join(out)
    check(n >= len(ELITE), "every row of every table was rewritten")
    check(after != before, "the text changed")
    check(len(after.splitlines()) == len(before.splitlines()),
          "no line was added or removed")
    rows_b = [ROW_RE.match(l) for l in before.splitlines() if ROW_RE.match(l)]
    rows_a = [ROW_RE.match(l) for l in after.splitlines() if ROW_RE.match(l)]
    check([m.group("pos") for m in rows_a] == [m.group("pos") for m in rows_b],
          "positions stay in place")
    check([m.group("aa") for m in rows_a] == [m.group("aa") for m in rows_b],
          "residues stay with their positions, so the table still describes the peptide")
    check(sorted(m.group("burial") for m in rows_a)
          == sorted(m.group("burial") for m in rows_b),
          "the burial values are permuted, not altered")
    check(sum(1 for m in rows_a if "OVERLAP" in m.group("detail"))
          == sum(1 for m in rows_b if "OVERLAP" in m.group("detail")),
          "the overlap is still reported, at a different position")
    moved = sum(
        1 for a, b in zip(rows_a, rows_b) if a.group("burial") != b.group("burial")
    )
    check(moved > 0, "at least one row's measurements moved")
    # The point of the control: the prompt is the same length, so a difference against
    # `full` cannot be attributed to a shorter prompt.
    check(abs(len(after) - len(before)) <= 2,
          "the scrambled prompt is the same size as the control")


def test_scoring_counts_placement() -> None:
    print("score_completion")
    user = prompt_text()
    facts = parse_facts(user)
    table = true_table(user, ELITE)
    # Position 2 (index 1) clashes; even indices (0-based) are anchors; odd are empty.
    text = (
        "Reasoning about the interface.\n"
        f"PROPOSE: {ELITE[:1]}W{ELITE[2:]} | fill position 2\n"
        f"PROPOSE: {ELITE[:3]}W{ELITE[4:]} | change an empty position\n"
        f"PROPOSE: {ELITE} | this one is already scored\n"
        "PROPOSE: GTPRRLLYCQRK | too long\n"
    )
    s = score_completion(text, facts, table)
    check(s["n_parsed"] == 3 and s["n_malformed"] == 1,
          "the over-length proposal is malformed, not merely rejected")
    check(s["n_accepted"] == 2, "the already-scored sequence is screened out")
    check(s["rejections"].get("already_evaluated") == 1,
          "the rejection reason is the production one")
    check(s["edit_placement"].get("clash") == 1,
          "an edit on the clashing position is counted as such")
    check(s["edit_placement"].get("empty") == 1,
          "an edit on a position with no contacts is counted as empty")
    check(s["mean_edits"] == 1.0 and s["distinct_positions"] == 2,
          "edits are counted against the base sequence, not the original")


def test_read_prompt_file(tmp_path: Path) -> None:
    print("read_prompt_file")
    p = tmp_path / "x_r1_elite0_a0.prompt.txt"
    p.write_text("=== SYSTEM ===\nyou are expert\n\n=== USER ===\n" + prompt_text() + "\n")
    system, user = read_prompt_file(p)
    check(system == "you are expert", "the system turn is recovered")
    check(user.startswith("TARGET:") and user.rstrip().endswith("<reason>"),
          "the user turn is recovered whole")


def main() -> int:
    import tempfile

    test_row_regex_matches_the_real_renderer()
    test_labels_and_facts()
    test_position_class()
    test_every_variant_changes_something()
    test_removals_are_surgical()
    test_overlap_stripping_is_total()
    test_scramble_preserves_shape()
    test_scoring_counts_placement()
    with tempfile.TemporaryDirectory() as td:
        test_read_prompt_file(Path(td))
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
