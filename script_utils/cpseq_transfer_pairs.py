#!/usr/bin/env python3
"""Emit the (searched, native) sequence pairs that a transfer test folds.

A search picks a sequence because the predictor that scored it liked the pose it produced.
Whether that sequence is good, or merely suits that predictor, is answered by folding the
SAME sequence somewhere else -- another predictor, or the sequence-conditioned generator --
and asking whether the advantage survives. Both consumers read the file this writes:

    script_utils/cpseq_highfold3_designs.py  --seq-source   (folds both labels per parent)
    script_utils/eval_val_gen_testset.py     --seq-source   (one label per run; use --label)

Two rules it does not let you break.

**The pick is made on the selection half of the folds only.** `select_candidate` is
imported from `cpseq_interface_compare`, which is itself deliberately the same rule as
`cpseq_arm_compare.arm_pick`'s cross-fit branch. A pick made on all folds would carry the
selection bias into the transfer test, and a shrunken advantage elsewhere would then be
regression to the mean wearing the costume of a predictor disagreement.

**The native sequence is emitted as its own labelled row.** The comparison is paired per
complex, and the pair has to be folded by the same instrument under the same settings. A
transfer test against the previous run's reference numbers would be comparing across
predictors AND across runs at once.

`--energy-key dG_separated_per_1000A2` selects on dG per buried area, which is the
objective whose held-out advantage actually survived on this cohort (12/0, p=0.001;
README_CPSEQ_LLM_SEARCH §13.3) -- raw dG's did not (-0.06 REU, 6/6, p=1.00). Selecting on
an objective with no held-out effect gives a transfer test with nothing to transfer.

Usage:
    python script_utils/cpseq_transfer_pairs.py \\
        --out-dir /data/user_data/yixiz/cpseq/llm_ladder \\
        --depth 10 --energy-key dG_separated_per_1000A2 \\
        --jsonl /data/user_data/yixiz/cpseq/transfer/selected_vs_native.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_interface_compare import select_candidate  # noqa: E402
from cpseq_ladder_report import DRAW_ARM, build, load, warn_if_mixed  # noqa: E402


def sequence_string(seq) -> str | None:
    """The peptide this Sequence's folds are all folds of."""
    for row in seq.by_rank.values():
        s = row.get("sequence")
        if s:
            return str(s)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="A search run directory holding scores/.")
    ap.add_argument("--depth", type=int, default=10,
                    help="Folds per sequence to read. The pick uses the first depth//2.")
    ap.add_argument("--energy-key", default="dG_separated_per_1000A2",
                    help="Objective the pick is made on. Default is dG per 1000 A^2, the one "
                         "whose held-out advantage survived; pass dG_separated for raw dG.")
    ap.add_argument("--draw-arm", default=DRAW_ARM,
                    help=f"Which arm's draws to pick from (default {DRAW_ARM!r}).")
    ap.add_argument("--scoring-version", default=None,
                    help="Restrict folds to one scoring version. Strongly recommended: a mixed "
                         "run compares a candidate under one gate to a reference under another.")
    ap.add_argument("--parents", default="",
                    help="Comma-separated subset of parent_ids (default: all with a pick).")
    ap.add_argument("--searched-label", default="searched")
    ap.add_argument("--native-label", default="native")
    ap.add_argument("--jsonl", required=True, help="Output path for the sequence-source file.")
    ap.add_argument("--json", default=None, help="Optional provenance summary.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    per_parent = load(out_dir)
    if not per_parent:
        print(f"[transfer-pairs] no score rows under {out_dir}")
        return 1
    mixed = warn_if_mixed(per_parent, args.scoring_version)
    state = build(
        per_parent,
        draw_arm=args.draw_arm,
        energy_key=args.energy_key,
        scoring_version=args.scoring_version,
    )

    half = args.depth // 2
    if half < 1:
        print(f"[transfer-pairs] depth {args.depth} cannot hold out a selection half")
        return 1

    wanted = {p.strip() for p in args.parents.split(",") if p.strip()}
    rows: list[dict] = []
    skipped: list[dict] = []

    for pid in sorted(state):
        if wanted and pid not in wanted:
            continue
        ref = state[pid].get("reference")
        draws = state[pid].get("draws") or {}
        if ref is None:
            skipped.append({"parent_id": pid, "reason": "no reference sequence in this run"})
            continue
        native = sequence_string(ref)
        if not native:
            skipped.append({"parent_id": pid, "reason": "reference rows carry no sequence"})
            continue
        if not draws:
            skipped.append({"parent_id": pid, "reason": f"no draws in arm {args.draw_arm!r}"})
            continue
        pick = select_candidate(draws, half)
        if pick is None:
            skipped.append({"parent_id": pid,
                            "reason": f"no draw has a median {args.energy_key} over folds [0,{half})"})
            continue
        searched = sequence_string(pick)
        if not searched:
            skipped.append({"parent_id": pid, "reason": "picked draw carries no sequence"})
            continue
        if len(searched) != len(native):
            skipped.append({"parent_id": pid,
                            "reason": f"picked {searched} is {len(searched)} aa, native {native} is "
                                      f"{len(native)}; not a substitution-only candidate"})
            continue

        n_sub = sum(a != b for a, b in zip(searched, native))
        sel_energy = pick.median_energy_in(0, half)
        common = {
            "parent_id": pid,
            "native_sequence": native,
            "source_run": str(out_dir),
            "selected_on": f"{args.energy_key} median over folds [0,{half})",
        }
        rows.append({**common, "sequence": searched, "label": args.searched_label,
                     "arm": args.draw_arm, "draw_index": pick.draw_index,
                     "selection_energy": sel_energy, "n_substitutions": n_sub})
        rows.append({**common, "sequence": native, "label": args.native_label,
                     "arm": "reference", "n_substitutions": 0})

    if not rows:
        print("[transfer-pairs] no complex produced a pair; nothing written")
        for s in skipped:
            print(f"  {s['parent_id']}: {s['reason']}")
        return 1

    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    n_pairs = len(rows) // 2
    identical = [r["parent_id"] for r in rows
                 if r["label"] == args.searched_label and r["n_substitutions"] == 0]
    subs = [r["n_substitutions"] for r in rows if r["label"] == args.searched_label]
    print(f"[transfer-pairs] {n_pairs} pairs -> {jsonl}")
    print(f"[transfer-pairs] selected on {args.energy_key}, folds [0,{half}) of {args.depth}, "
          f"arm {args.draw_arm!r}")
    print(f"[transfer-pairs] substitutions per pick: min {min(subs)}, median "
          f"{sorted(subs)[len(subs)//2]}, max {max(subs)}")
    if identical:
        print(f"[transfer-pairs] NOTE {len(identical)} pick(s) ARE the native sequence "
              f"({', '.join(identical)}); those pairs measure predictor noise, not a design, "
              "and are worth keeping for exactly that reason")
    if skipped:
        print(f"[transfer-pairs] {len(skipped)} complex(es) contributed no pair:")
        for s in skipped:
            print(f"  {s['parent_id']}: {s['reason']}")
    if mixed and args.scoring_version is None:
        print("[transfer-pairs] WARNING this run mixes scoring versions and none was pinned; "
              "re-run with --scoring-version before trusting the picks")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "out_dir": str(out_dir),
            "depth": args.depth,
            "select_folds": [0, half],
            "energy_key": args.energy_key,
            "draw_arm": args.draw_arm,
            "scoring_version": args.scoring_version,
            "mixed_scoring_versions": mixed,
            "n_pairs": n_pairs,
            "jsonl": str(jsonl),
            "picks_identical_to_native": identical,
            "skipped": skipped,
            "rows": rows,
        }, indent=1) + "\n")
        print(f"[transfer-pairs] provenance -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
