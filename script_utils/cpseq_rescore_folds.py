#!/usr/bin/env python3
"""Re-score folds already on disk under the CURRENT scoring protocol. No GPU.

Why this exists as its own tool
-------------------------------
The journal dedupes on `(sequence, seed, stage, arm)` and that key carries no scoring
version, so re-running `ladder` after a protocol change reports every fold as already
done and rescores nothing. The two-cache split in `evaluate.py` is what makes a rescore
cheap -- predictions are keyed by `input_hash` and scores by `(scoring_version,
input_hash)`, so a new protocol version costs CPU only -- but nothing walked the journal
and spent that CPU. This does.

It re-evaluates with `force_score=True`, which re-reads the retained raw structure, runs
the relax and the gate again, and writes a fresh score file under the new version's
directory. The prediction is a cache hit and no GPU is touched. Rows under the old
version stay on disk, so the two protocols can be compared rather than one overwriting
the other.

The journal is APPENDED to, not rewritten: the new rows carry the new
`scoring_version`, and a reader that wants one protocol filters on it. Rewriting history
in place would make a run's own log disagree with the numbers it reported earlier.

Usage:
    python script_utils/cpseq_rescore_folds.py --out-dir $ZFS/cpseq/es_pilot_lnr
    python script_utils/cpseq_rescore_folds.py --out-dir <run> --parents lnr_6seo --limit 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.evaluate import (  # noqa: E402
    SCORING_PROTOCOL_VERSION,
    Protocol,
    Store,
    evaluate,
    load_target_contexts,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_search import Journal, outcome_row  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parents", default="", help="comma separated; default all")
    ap.add_argument("--stage", default="", help="only this stage; default all")
    ap.add_argument("--limit", type=int, default=0, help="stop after N folds")
    ap.add_argument(
        "--max-draw-index", type=int, default=-1,
        help="only draws at or below this stream position, plus the reference. A rung-B "
        "comparison reads the first B draws of each arm, so re-scoring draw 400 to "
        "answer a question about B=8 is wasted CPU. -1 keeps every draw.",
    )
    ap.add_argument(
        "--arms", default="",
        help="comma separated arms to re-score; default all. The reference is always "
        "kept, because a comparison whose baseline is at a different protocol than the "
        "arm under test is the bug this tool exists to repair.",
    )
    ap.add_argument("--chemistry", default="mainchain")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cohort = json.loads((out_dir / "cohort.json").read_text())
    parents = [p.strip() for p in args.parents.split(",") if p.strip()] or cohort["cohort"]

    sources = cohort.get("parent_sources") or {}
    contexts = {}
    for pid in parents:
        d = sources.get(pid) or cohort.get("dataset_dir")
        contexts |= load_target_contexts(Path(d), [pid])

    protocol = Protocol(chemistry=args.chemistry)
    # The cohort's own frozen gate, exactly as `ladder` loads it. A rescore that picked
    # up a different tolerances file would change the verdict for a reason that has
    # nothing to do with the scoring protocol under test.
    store = Store(
        root=out_dir,
        tolerances_path=Path(cohort["dataset_dir"]) / "manifest"
        / "geometry_tolerances.json",
        retention="full",
    )
    print(f"[rescore] {out_dir}")
    print(f"[rescore] target protocol: {SCORING_PROTOCOL_VERSION}")

    done = 0
    t0 = time.time()
    for pid in parents:
        if pid not in contexts:
            print(f"[rescore] {pid}: no target context -- skipped")
            continue
        journal = Journal(out_dir, pid)
        # One row per (sequence, seed, stage, arm) that is not already at the current
        # version. Technical failures are skipped: there is no structure to re-score,
        # and re-running one here would only re-record the crash without the GPU that
        # caused it.
        todo, seen = [], set()
        for o in journal.outcomes:
            if o.get("technical_failure") or not o.get("input_hash"):
                continue
            if o.get("scoring_version") == SCORING_PROTOCOL_VERSION:
                continue
            if args.stage and o.get("stage") != args.stage:
                continue
            is_ref = o.get("arm") == "reference"
            if args.arms and not is_ref:
                if o.get("arm") not in [a.strip() for a in args.arms.split(",")]:
                    continue
            if args.max_draw_index >= 0 and not is_ref:
                di = o.get("draw_index")
                # A draw with no stream position cannot be placed in a rung, so it is
                # not something a rung-limited rescore needs.
                if di is None or int(di) > args.max_draw_index:
                    continue
            key = (o.get("sequence"), o.get("prediction_seed"), o.get("stage"), o.get("arm"))
            if key in seen:
                continue
            seen.add(key)
            todo.append(o)

        print(f"  [{pid}] {len(todo)} fold(s) to re-score")
        for o in todo:
            if args.limit and done >= args.limit:
                print(f"[rescore] limit {args.limit} reached")
                break
            if args.dry_run:
                print(f"    [dry] {o['sequence']} seed={o['prediction_seed']} "
                      f"({o.get('scoring_version')} -> {SCORING_PROTOCOL_VERSION})")
                done += 1
                continue
            rec = evaluate(
                original_sequence=contexts[pid].reference_sequence,
                candidate_sequence=o["sequence"],
                target=contexts[pid],
                prediction_seed=int(o["prediction_seed"]),
                protocol=protocol,
                store=store,
                force_score=True,
            )
            row = outcome_row(
                rec=rec, proposal=None, parent_id=pid,
                arm=o.get("arm", "unknown"), stage=o.get("stage", "ladder"),
                # `proposal=None` above means `outcome_row` writes no `draw_index` or
                # `round_index`, and those are what place a fold in the draw stream. A
                # row without them is invisible to a prefix ladder, so the re-score that
                # was just paid for drops the whole draw rather than replacing it. They
                # are copied from the row being re-scored: the provenance of a fold does
                # not change when the scorer does.
                extra={"seed_rank": o.get("seed_rank"),
                       "draw_index": o.get("draw_index"),
                       "round_index": o.get("round_index"),
                       "proposal_kind": o.get("proposal_kind"),
                       "ancestor_sequence": o.get("ancestor_sequence"),
                       "rescored_from": o.get("scoring_version")},
            )
            journal.add_outcome(row)
            done += 1
            pre = "valid" if row.get("prerelax_valid") else "invalid"
            print(f"    {o['sequence']} seed={o['prediction_seed']} -> "
                  f"{'VALID' if row['valid'] else 'invalid'} (pre-relax {pre})  "
                  f"dG={row.get('best_valid_dG_separated')}  "
                  f"cache={rec['cache'].get('prediction')}")
        if args.limit and done >= args.limit:
            break

    print(f"[rescore] {done} fold(s) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
