#!/usr/bin/env python3
"""Rebuild an arm's proposal files from a run's candidate journal.

Why this exists
---------------
`cpseq_search.py ladder --llm-round N` folds the sequences in
`<out-dir>/llm_proposals/round<N>/<parent>.jsonl`. Those files are what makes a re-fold
use the SAME draws as the original run rather than newly proposed ones. When the only
copy of a finished run is its published journal bundle -- which packs `state/` and
nothing else, because every reported number redraws from the journals and no structure
is needed -- the proposal files are absent and the arm cannot be re-folded until they
are reconstructed.

They are reconstructible exactly, because `Journal` records every candidate it ever
folded along with the provenance the proposer gave it: `draw_index`, `round_index`,
`proposal_seed`, `ancestor_sequence` and the note.

The ordering contract is the load-bearing part
----------------------------------------------
`load_llm_proposals` does NOT read `draw_index` from the file. It concatenates rounds in
order and assigns each proposal the index it gets from that concatenation, because a
ladder rung is a PREFIX of the draws and draw 9 must mean the same sequence in a
two-round run as in a five-round one. So this script writes each round's rows sorted by
their journaled `draw_index`, and verifies before writing that round 1 occupies draw
slots 0..k-1, round 2 the next block, and so on. If that does not hold, the
reconstruction would silently renumber the draws and every rung would measure a
different set than the original -- so the script refuses instead.

Usage:
    python script_utils/cpseq_proposals_from_journal.py \
        --journal-dir <bundle>/state --dest <run-dir> --arm llm_search
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# The two proposal trees and where each arm's files belong. `llm_search` is arm D and
# reads from `llm_proposals/round<N>/`; the mask/fill arms read from
# `mask_proposals/<arm>/round<N>/`.
LLM_ARM = "llm_search"


def dest_dir_for(dest: Path, arm: str, rnd: int) -> Path:
    if arm == LLM_ARM:
        return dest / "llm_proposals" / f"round{rnd}"
    return dest / "mask_proposals" / arm / f"round{rnd}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal-dir", required=True,
                    help="the run's state/ directory (holds <parent>/candidates.jsonl)")
    ap.add_argument("--dest", required=True,
                    help="the run directory to write the proposal tree into")
    ap.add_argument("--arm", default=LLM_ARM)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jdir, dest = Path(args.journal_dir), Path(args.dest)
    if not jdir.is_dir():
        print(f"[proposals] FATAL: no journal directory at {jdir}")
        return 1

    # parent -> round -> list of rows
    per: dict[str, dict[int, list[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    seen_arms: collections.Counter = collections.Counter()
    for f in sorted(jdir.glob("*/candidates.jsonl")):
        pid = f.parent.name
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen_arms[r.get("arm")] += 1
            if r.get("arm") != args.arm:
                continue
            di, rnd = r.get("draw_index"), r.get("round_index")
            if di is None or di < 0 or rnd is None or rnd < 0:
                continue
            if not r.get("sequence"):
                continue
            per[pid][int(rnd)].append(r)

    if not per:
        print(f"[proposals] FATAL: no rows for arm {args.arm!r} in {jdir}")
        print(f"            arms present: {dict(seen_arms)}")
        return 1

    # Verify the prefix contract BEFORE writing anything.
    bad: list[str] = []
    for pid, by_round in sorted(per.items()):
        lo = 0
        for rnd in sorted(by_round):
            idx = sorted(int(r["draw_index"]) for r in by_round[rnd])
            if len(set(idx)) != len(idx):
                bad.append(f"{pid} round {rnd}: duplicate draw_index")
            if idx != list(range(lo, lo + len(idx))):
                bad.append(
                    f"{pid} round {rnd}: draw slots {idx[0]}..{idx[-1]} "
                    f"(expected {lo}..{lo + len(idx) - 1})")
            lo += len(idx)
    if bad:
        print("[proposals] REFUSING: the journaled draw slots are not a round-ordered")
        print("    prefix, so rebuilding the files would renumber the draws and every")
        print("    rung would measure a different set than the run being reproduced.")
        for b in bad[:20]:
            print(f"      {b}")
        return 1

    rounds = sorted({r for by in per.values() for r in by})
    total = sum(len(v) for by in per.values() for v in by.values())
    print(f"[proposals] arm={args.arm}  parents={len(per)}  rounds={rounds}  "
          f"draws={total}")
    for pid, by_round in sorted(per.items()):
        counts = {r: len(by_round[r]) for r in sorted(by_round)}
        print(f"    {pid:<26} {sum(counts.values()):>3} draws  per round {counts}")

    if args.dry_run:
        print("[proposals] --dry-run: nothing written")
        return 0

    written = 0
    for pid, by_round in sorted(per.items()):
        for rnd in sorted(by_round):
            d = dest_dir_for(dest, args.arm, rnd)
            d.mkdir(parents=True, exist_ok=True)
            rows = sorted(by_round[rnd], key=lambda r: int(r["draw_index"]))
            out = []
            for r in rows:
                # Exactly the fields `load_llm_proposals` and `Proposal` read. The
                # journaled `draw_index` is carried as a comment field only -- the loader
                # derives the slot from file order, and writing it as `_draw_index` here
                # would be ignored anyway, so the ORDER above is what preserves identity.
                out.append({
                    "sequence": r["sequence"],
                    "llm_round": int(rnd),
                    "round_index": int(rnd),
                    "proposal_seed": int(r.get("proposal_seed") or 0),
                    "ancestor_sequence": r.get("ancestor_sequence") or None,
                    "note": str(r.get("note") or "")[:200],
                    "accepted": True,
                    "journaled_draw_index": int(r["draw_index"]),
                    "rebuilt_from": "candidates.jsonl",
                })
            (d / f"{pid}.jsonl").write_text(
                "".join(json.dumps(o) + "\n" for o in out))
            written += 1
    print(f"[proposals] wrote {written} file(s) under {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
