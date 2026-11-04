#!/usr/bin/env python3
"""Arm E, stage 1: ask the LLM which positions to open, for a whole cohort.

Masks only. The filler is a separate stage and a separate model
(`cpseq_mask_mpnn_fill.py`), which is the point of the split -- see
`README_CPSEQ_MASK_ARM.md`.

Runs against an existing run directory and reuses what is already there: the per-parent
journals, the cached reference interface summaries, and the same cross-fit guard arm D
obeys. Writing into the run that already holds `random_control` and `llm_search` is
deliberate -- the reference folds are keyed by `input_hash`, so every arm in that directory
is compared against the SAME measurements of the unchanged peptide rather than against two
independent foldings of it.

Usage:
  python script_utils/cpseq_llm_mask_round.py --out-dir $DATA/cpseq/llm_ladder --round 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from cpseq_llm_propose import (  # noqa: E402
    STAGE_LADDER,
    cached_interface,
    load_journal,
)

from proteinfoundation.cpseq.llm_propose import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    LlamaProposer,
    DEFAULT_ELITE_MIN_DISTANCE,
    DEFAULT_N_ELITES,
    history_from_outcomes,
    propose_mask_round,
    select_elites,
)
from proteinfoundation.cpseq.search import default_alphabet  # noqa: E402


def reference_row(rows: list[dict], original: str) -> dict | None:
    for r in rows:
        if (
            r.get("arm") == "reference"
            and not r.get("technical_failure")
            and r.get("raw_pdb_path")
            and r.get("sequence") == original
        ):
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--round", type=int, default=1, dest="round_index")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--n-masks", type=int, default=8)
    ap.add_argument("--n-elites", type=int, default=DEFAULT_N_ELITES,
                    help="how many elites this round breeds from, sharing --n-masks "
                         "between them. 0 disables evolution and proposes from the "
                         "native sequence, which is what rounds before this flag did. "
                         "The default is live even in generation 1, where it is a no-op: "
                         "the reference is the only scored sequence, so it is the only "
                         "elite and the round starts from the original either way.")
    ap.add_argument("--elite-min-distance", type=int, default=DEFAULT_ELITE_MIN_DISTANCE,
                    help="Hamming veto between elites, so the elite set cannot collapse "
                         "into several neighbours of one sequence and spend a whole "
                         "generation re-exploring one basin.")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--fold-depth", type=int, default=10,
                    help="only folds 0..depth/2 are shown; see --feedback-folds in arm D")
    ap.add_argument("--feedback-folds", default="select", choices=("select", "all"))
    ap.add_argument("--parents", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cohort_path = args.out_dir / "cohort.json"
    if not cohort_path.exists():
        print(f"FATAL: no cohort.json in {args.out_dir}", file=sys.stderr)
        return 1
    cohort = json.loads(cohort_path.read_text())["cohort"]
    if args.parents:
        want = set(args.parents.split(","))
        cohort = [p for p in cohort if p in want]
    print(f"[mask] {len(cohort)} parent(s) from {cohort_path}")

    # The cross-fit guard, identical to arm D's: an adaptive proposer that sees the folds
    # its proposals are later scored on makes the held-out column not held out.
    max_fold_rank = None
    if args.feedback_folds == "select" and args.fold_depth:
        max_fold_rank = args.fold_depth // 2 - 1

    dest = args.out_dir / "masks" / f"round{args.round_index}"
    dest.mkdir(parents=True, exist_ok=True)
    prompts = args.out_dir / "masks" / f"round{args.round_index}" / "prompts"

    alpha = default_alphabet(allow_cys=False)
    llm = LlamaProposer(args.model, device=args.device, temperature=args.temperature)
    t0 = time.time()
    llm.load()
    print(f"[mask] llama loaded in {time.time() - t0:.1f}s", flush=True)

    index = []
    for pid in cohort:
        rows = load_journal(args.out_dir, pid)
        if not rows:
            print(f"[mask] {pid}: empty journal; skipped")
            continue
        original = next(
            (r.get("original_sequence") for r in rows if r.get("original_sequence")), None
        )
        if not original:
            print(f"[mask] {pid}: no original_sequence in journal; skipped")
            continue
        budget = len(original) // 2

        ref = reference_row(rows, original)
        ref_summary = (
            cached_interface(args.out_dir, pid, ref, sequence=original) if ref else None
        )
        history = history_from_outcomes(
            rows, stage=STAGE_LADDER, max_fold_rank=max_fold_rank
        )

        # The elites this generation breeds from. `--n-elites 0` restores the
        # non-evolutionary behaviour exactly: one call, starting from the native
        # sequence, which is what every round before this one did.
        #
        # Generation 1 needs no special case. The only scored sequence in the journal is
        # the reference, so `select_elites` returns the original and `start_from` is the
        # original -- byte-identical to the non-evolutionary path. The search only
        # starts moving once a round has scored offspring to rank.
        elites = (select_elites(history, n_elites=args.n_elites, 
                                min_distance=args.elite_min_distance)
                  if args.n_elites > 0 else [])
        if not elites:
            elites = [None]

        shares = [args.n_masks // len(elites)] * len(elites)
        for i in range(args.n_masks - sum(shares)):
            shares[i] += 1

        masks, record, ancestors = [], None, []
        for elite, share in zip(elites, shares):
            if share < 1:
                continue
            got, rec = propose_mask_round(
                proposer=llm,
                original=original,
                budget=budget,
                round_index=args.round_index,
                parent_id=pid,
                reference_summary=ref_summary,
                history=history,
                alphabet=alpha,
                pdb_id=(ref or {}).get("pdb_id"),
                max_attempts=args.max_attempts,
                prompt_dir=prompts,
                n_requested=share,
                parent_candidate=elite,
            )
            masks.extend(got)
            ancestors.extend([rec["start_from"]] * len(got))
            if record is None:
                record = rec
                record["per_elite"] = []
            else:
                # Merge rather than overwrite: the malformed and rejected counts are
                # how a round reports that the model is struggling, and keeping only
                # the last elite's would hide most of it.
                record["malformed"].extend(rec["malformed"])
                record["rejections"].extend(rec["rejections"])
            record["per_elite"].append({
                "start_from": rec["start_from"],
                "parent_candidate": rec.get("parent_candidate"),
                "parent_energy": rec.get("parent_energy"),
                "headroom": rec.get("headroom"),
                "requested": share, "n_masks": len(got),
            })

        if record is None:
            print(f"[mask] {pid}: --n-masks={args.n_masks} left every elite a zero "
                  f"share; nothing proposed")
            continue

        record["masks"] = [",".join(str(i) for i in m.mask) for m in masks]
        record["mask_ancestors"] = ancestors
        record["n_elites"] = len([e for e in elites if e is not None])
        record["reference_interface_ok"] = bool(ref_summary and ref_summary.get("ok"))
        record["max_fold_rank_shown"] = max_fold_rank
        record["n_history"] = len(history)
        (dest / f"{pid}_r{args.round_index}.json").write_text(json.dumps(record, indent=2))

        sizes = [len(m.mask) for m in masks]
        print(f"[mask] {pid:24s} L={len(original):2d} budget={budget:2d}  "
              f"{len(masks)}/{args.n_masks} masks {sizes}  "
              f"{record['n_elites']} elite(s)  "
              f"{len(record['malformed'])} malformed, {len(record['rejections'])} rejected"
              f"{'  iface=MISSING' if not record['reference_interface_ok'] else ''}",
              flush=True)
        index.append({
            "parent_id": pid, "n_masks": len(masks), "mask_sizes": sizes,
            "budget": budget, "n_malformed": len(record["malformed"]),
            "n_rejected": len(record["rejections"]),
            "reference_interface_ok": record["reference_interface_ok"],
            "note": record.get("note", ""),
        })

    (dest / "index.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "round": args.round_index, "llm": llm.config,
        "n_requested": args.n_masks, "feedback_folds": args.feedback_folds,
        "max_fold_rank_shown": max_fold_rank,
        "parents": index,
    }, indent=2))
    total = sum(p["n_masks"] for p in index)
    short = [p["parent_id"] for p in index if p["n_masks"] < args.n_masks]
    print(f"\n[mask] {total} masks over {len(index)} parents -> {dest}")
    if short:
        print(f"[mask] short rounds (NOT padded): {short}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
