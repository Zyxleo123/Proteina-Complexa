#!/usr/bin/env python3
"""Write one mask set per parent, to be shared by several fillers.

Why this exists
---------------
`cpseq_mask_mpnn_fill.py` in standalone mode draws its own masks, seeded per
(parent, arm, round). That is right for comparing an LLM's position choice against a
matched random control -- the control's masks are generated to match the LLM's sizes.

It is wrong for a sweep over the FILLER, because each arm would then fill a different set
of masks and the comparison would mix "which positions were opened" with "what was put in
them". The filler is the only thing under test here, so every arm must be handed the same
masks. This script writes them once, in the format the filler's `--masks-dir` reads, so
the sweep arms differ in temperature and in nothing else.

The mask SIZES come from the project's own distribution (`sampled_mask_sizes`), which
spreads sizes from 1 up to the budget rather than piling them at the ceiling -- a
full-budget mask is a bigger extrapolation for weights trained on complexes.

Output, per parent: `<dest>/<parent_id>_r<round>.json` with `parent_id`,
`original_sequence` and `masks` as comma-joined 0-based position strings.

Usage:
    python script_utils/cpseq_make_shared_masks.py \
        --out-dir <run dir with reference folds> --dest <masks dir> \
        --n-masks 12 --round 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_mask_mpnn_fill import (  # noqa: E402
    journal_parents,
    matched_random_masks,
    sampled_mask_sizes,
)
from proteinfoundation.cpseq.search import budget_for, proposal_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="the run directory whose journal names the parents and holds "
                         "their reference folds")
    ap.add_argument("--dest", type=Path, required=True,
                    help="where to write the mask records")
    ap.add_argument("--n-masks", type=int, default=12)
    ap.add_argument("--round", type=int, default=1, dest="round_index")
    ap.add_argument("--parents", default="",
                    help="comma-separated; default every parent with a reference fold")
    ap.add_argument("--seed-arm", default="shared_masks",
                    help="the arm label the mask seed is derived from. Fixed and "
                         "recorded so the same masks can be regenerated exactly; it is "
                         "deliberately NOT one of the real arm names, because the point "
                         "is that no arm owns these masks.")
    args = ap.parse_args()

    want = {p.strip() for p in args.parents.split(",") if p.strip()} or None
    parents = journal_parents(args.out_dir, want)
    if not parents:
        print(f"FATAL: no parent in {args.out_dir} has a reference fold on disk",
              file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    index = []
    for pid, original in parents:
        L = len(original)
        budget = budget_for(L)
        seed = int(proposal_seed(pid, args.seed_arm, args.round_index, 0) % 10**6)
        sizes = sampled_mask_sizes(args.n_masks, budget, seed)
        if not sizes:
            print(f"[masks] {pid}: L={L} gives budget {budget}; no mask fits, excluded")
            continue
        masks = matched_random_masks(sizes, L, seed=seed)
        rec = {
            "parent_id": pid,
            "original_sequence": original,
            "round_index": args.round_index,
            "masks": [",".join(str(p) for p in m) for m in masks],
            "mask_sizes": [len(m) for m in masks],
            "length": L,
            "substitution_budget": budget,
            "seed": seed,
            "note": (
                "Shared masks for a filler sweep. Every arm fed these fills the SAME "
                "positions, so a difference between arms is attributable to the filler "
                "and to nothing else."
            ),
        }
        (args.dest / f"{pid}_r{args.round_index}.json").write_text(
            json.dumps(rec, indent=2))
        index.append({"parent_id": pid, "n_masks": len(masks),
                      "sizes": sorted(rec["mask_sizes"]), "length": L,
                      "budget": budget})
        print(f"[masks] {pid:<24} L={L:<3} budget={budget:<2} "
              f"{len(masks)} masks, sizes {sorted(rec['mask_sizes'])}")

    (args.dest / "index.json").write_text(json.dumps(
        {"round": args.round_index, "n_masks_requested": args.n_masks,
         "seed_arm": args.seed_arm, "parents": index}, indent=2))
    print(f"\n[masks] {len(index)} parent(s) -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
