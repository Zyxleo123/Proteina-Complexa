#!/usr/bin/env python3
"""Bounded cyclic-peptide sequence search: two core arms, matched budgets, fresh-seed confirmation.

Given a parent's original linear-peptide sequence and its target, look for a head-to-tail
cyclic sequence of the same length, within `floor(L/2)` substitutions, whose predicted
complex passes the frozen geometry gate with a better interface energy than the SAME
sequence space's unmutated cyclic reference.

Why this is one job and not the usual two stages
------------------------------------------------
`cpseq_predict.py` and `cpseq_score.py` are deliberately separate so a rescore never
re-predicts. A search cannot be split that way: round n+1's proposals are a function of
round n's scores, so prediction and scoring have to alternate inside one process. That
makes this a GPU job that also runs PyRosetta, and it is the one place in this project
where the two resource kinds share a job. Everything else is unchanged -- the same
adapter, the same evaluator, the same frozen tolerances, and the same caches, so a
structure this search predicts is reusable by the batch pipeline and vice versa.

Modes
-----
  select-cohort  stratify the existing tranche and record the 6 development parents
  screen         arms A and B (and C when enabled), resumable per evaluation
  confirm        best 2 per arm + the reference, re-evaluated at 3 fresh seeds
  report         per-parent tables and the pilot summary

What makes a result a result here
---------------------------------
Selection is a hierarchy, not a tuned weighted sum: the sequence budget and the ring
chemistry are enforced first, then the frozen geometry/interface gate separates accepted
candidates from failures, and only accepted candidates are ranked by corrected interface
energy. A structure with an attractive energy and an open or clashing ring is never
promoted; its continuous violation is used for exploration only.

Usage:
    python script_utils/cpseq_search.py select-cohort --dataset-dir $ZFS/cpseq/tranche1 \
        --out-dir $ZFS/cpseq/search_pilot --n-parents 6
    python script_utils/cpseq_search.py screen --out-dir $ZFS/cpseq/search_pilot \
        --shard-index 0 --shard-count 1
    python script_utils/cpseq_search.py confirm --out-dir $ZFS/cpseq/search_pilot
    python script_utils/cpseq_search.py report  --out-dir $ZFS/cpseq/search_pilot
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.retention import TIERS as RETENTION_TIERS  # noqa: E402
from proteinfoundation.cpseq.evaluate import (  # noqa: E402
    SCORING_PROTOCOL_VERSION,
    prefetch_predictions,
    Protocol,
    Store,
    TargetContext,
    evaluate,
    load_target_contexts,
)
from proteinfoundation.cpseq.search import (  # noqa: E402
    ARM_GUIDED,
    ARM_LLM,
    ARM_MPNN,
    ARM_RANDOM,
    KIND_LLM,
    KIND_MPNN,
    KIND_REFERENCE,
    Archive,
    ArchiveEntry,
    Proposal,
    budget_for,
    default_alphabet,
    guided_round,
    match_counts,
    standalone_random_arm,
    standalone_random_stream,
    proposal_seed,
    screen_slots_remaining,
    select_spread_singles,
)

# Pilot budget. 16 + 8 + 8 = 32 screened sequences per parent per arm.
SCAN_BUDGET = 16
ROUND_BUDGET = 8
N_ROUNDS = 2
# The per-arm screening budget, enforced against what is already on disk so that a
# requeued task cannot buy extra slots for one arm (see `_evaluate`).
_SCREEN_BUDGET = SCAN_BUDGET + ROUND_BUDGET * N_ROUNDS
ARM_BUDGETS = {ARM_GUIDED: _SCREEN_BUDGET, ARM_RANDOM: _SCREEN_BUDGET,
               ARM_MPNN: _SCREEN_BUDGET}
SCREEN_SEED = 42
CONFIRM_SEEDS = (101, 202, 303)
FINALISTS_PER_ARM = 2

# --------------------------------------------------------------------------------------
# Seed ladder
# --------------------------------------------------------------------------------------
# The `screen` -> `confirm` split cannot answer the budget question, and the reason is a
# selection effect rather than a shortage of data. `screen` folds each proposal ONCE, so
# the geometry gate is applied to a single draw of the predictor's noise; `confirm` then
# re-folds only the rows that passed it, ranked by that same single fold's energy. A
# candidate that fell on an unlucky seed can never be rescued, and among the survivors
# the one that got the luckiest energy is the one that buys a confirmation slot. Both
# choices are made before anything is measured. Sec. 7 of the pilot README already
# retracted one arm ordering as a best-of-seeds artifact, which is this effect in the
# output rather than a separate bug.
#
# The ladder run collapses the two stages: NOTHING is promoted and nothing is filtered.
# Every draw is folded at every seed in `LADDER_SEEDS`, so a draw's outcome is a rate
# over seeds -- how often it clears the frozen gate -- instead of a coin flip.
# Append-only. A cohort records the prefix it used, and a deeper run re-reads that
# prefix, so the folds already paid for are reused rather than re-predicted -- which only
# holds while the first N entries never change. Never reorder or insert.
#
# Extended to 30 on 2026-10-24. Depth 10 leaves a 5-fold selection half holding 1.91
# PASSING folds on average (measured over 1,122 candidates, pass rate ~1/3), so the
# candidate is chosen on a median carrying +-2.4 REU against quality differences of 1-4
# REU: selection is mostly noise. Depth 30 puts ~6.7 passing folds in a 20-fold
# selection half, for +-1.4 REU.
LADDER_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808,
                909, 111, 222, 333, 444, 555, 666, 777,
                1201, 1302, 1403, 1504, 1605, 1706, 1807, 1908,
                2009, 2110, 2211, 2312, 2413, 2514)

# Seed depths the REPORT truncates to. Every folded draw carries all `seed_depth` folds,
# so these cost nothing to produce: "does the conclusion change with fold count" is a
# truncation of one run rather than three runs that also differ in which structures they
# happened to draw. Depth 1 is in the list because it is what every superseded run saw,
# and the gap between it and depth 10 is the thing being measured.
LADDER_SEED_TIERS = (1, 4, 8, 16)

# The ladder itself: how many random draws per complex a run is credited with. These are
# NESTED -- the first 8 draws of the 64 are exactly what a B=8 run would have produced,
# because draw k is an independent uniform draw whose identity does not depend on how
# many draws are taken (`standalone_random_stream`). So every rung is read by truncating
# one run, and no rung costs its own folds.
#
# This is what the superseded ladder could NOT do. There, arm A matched the guided arm's
# substitution counts, whose round 0 was a single-substitution scan: 1,102 of 1,102 of
# the first draws were 1-mutant, so truncating to a small B confounded the budget with
# the number of substitutions, and the ladder had to be built by subsampling instead.
LADDER_BUDGETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
LADDER_DRAWS = 64
STAGE_LADDER = "ladder"

# Consecutive predictor failures that end a task. A healthy run has none at all -- 12,097
# folds produced 0 outside the one unhealthy node -- so a run of five is a dead GPU or a
# revoked allocation, not bad luck on a sequence. Low enough to stop within minutes,
# above 1 so a single genuine timeout does not end an otherwise good task.
MAX_CONSECUTIVE_FAILURES = 5


def gate_sha256(path: Path) -> str:
    """Content hash of the acceptance thresholds, ignoring their provenance keys.

    Over the THRESHOLDS, not the file: a copied tolerance file records where it came from
    and why, so two corpora can hold the same numbers in byte-different files. Hashing
    raw bytes would refuse a legitimate pairing, and hashing nothing at all is how the
    mismatch below went unnoticed.

    Why this exists as a shared function rather than a line in each caller: on
    2026-10-25 the `seed_ladder` run was found to have scored every fold against
    `tranche1`'s pre-correction thresholds (trans-only planarity 8.44 deg, bond window
    1.1-1.6 A) while labelling every row `v3-cis-amide-muco-windows`. The label is a
    module constant and the thresholds are a per-corpus file, so the two could disagree
    silently -- and did, for 12,000 folds. The cohort records this hash; `cmd_ladder`
    re-checks it before folding anything.
    """
    import hashlib

    doc = json.loads(Path(path).read_text())
    thresholds = {k: v for k, v in doc.items()
                  if k not in ("copied_from", "copy_rationale", "frozen_on")}
    return hashlib.sha256(json.dumps(thresholds, sort_keys=True).encode()).hexdigest()


def ladder_plan(
    sequences: list[str], seeds: tuple[int, ...], budgets: tuple[int, ...]
) -> list[tuple[int, int, str, int]]:
    """Evaluation order as (budget_rung, seed_rank, sequence, seed).

    `sequences[0]` is the unchanged peptide; `sequences[1:]` are the draws in stream
    order. The ordering is the one piece of this run that had to be designed rather than
    inherited, because it decides what a half-finished night is worth.

    Budget-rung major: all folds of draws 1..B are paid for before any fold of draw B+1.
    A run cut off anywhere therefore holds a COMPLETE ladder up to the last finished
    rung -- every draw in it folded at every seed -- and the only thing an interruption
    costs is how far along the draw axis the ladder reaches. That is the honest failure
    mode for a ladder: it gets shorter, never less trustworthy.

    The alternative order, fold-rank major over all draws at once, was rejected after
    being written: it holds all 64 draws at an equal but shallow fold count, so an
    interrupted run reproduces the single-fold filter this whole re-run exists to
    correct, at every rung at once. Depth is what makes a rung mean anything, so depth
    is what the order protects.

    Within a rung, fold-rank major over the block keeps the draws balanced, so even a
    part-finished rung has its draws at equal depth rather than some at ten folds and
    the rest at none. The unchanged peptide sits in the first block and is therefore
    folded at full depth before any rung is complete: it is the comparison every rung is
    read against, so no rung can exist at a depth it has not reached.

    The plan is a pure function of its arguments. It reads neither the archive nor the
    journal, so a requeued task reproduces it exactly and `journal.has` skips what was
    already paid for, instead of planning fresh work on top of it (the overspend
    recorded in README_CPSEQ_SEARCH_PILOT sec. 5, from the proposer's side).
    """
    draws = sequences[1:]
    out: list[tuple[int, int, str, int]] = []
    prev = 0
    for rung in budgets:
        block = draws[prev:rung]
        if prev == 0:
            # The unchanged peptide, at full depth, before the first rung closes.
            block = sequences[:1] + block
        for rank in range(len(seeds)):
            for seq in block:
                out.append((rung, rank, seq, seeds[rank]))
        prev = rung
        if prev >= len(draws):
            break
    return out


# --------------------------------------------------------------------------------------
# Cohort selection
# --------------------------------------------------------------------------------------


def native_reference_stats(dataset_dir: Path) -> dict[str, dict]:
    """Per-parent native-sequence cyclic outcome, from the existing tranche.

    Raw-geometry validity is used for stratification because it is a measurement of the
    predicted coordinates and is therefore independent of the scoring protocol -- the
    closure-constraint fix changed refined energies and left it untouched. So the cohort
    can be chosen and recorded without waiting for the corrected rescore, while every
    ENERGY comparison in the pilot uses only `SCORING_PROTOCOL_VERSION` rows.
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "n_valid": 0, "dsasa": [], "length": None, "sequence": None}
    )
    roots = [dataset_dir / "scores", dataset_dir / "scores_unconstrained_relax"]
    seen: set[tuple] = set()
    for root in roots:
        for f in sorted(root.glob("rows_boltz2_*.jsonl")) if root.exists() else []:
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001, S112
                    continue
                if r.get("chemistry") != "mainchain" or not r.get("is_reference"):
                    continue
                idx = r.get("sample_index", -1)
                if idx is None or idx < 0:
                    continue
                key = (r.get("parent_id"), r.get("input_hash"), idx)
                if key in seen:
                    continue
                seen.add(key)
                s = stats[r["parent_id"]]
                s["n"] += 1
                s["n_valid"] += bool(r.get("valid"))
                s["length"] = r.get("peptide_length")
                s["sequence"] = r.get("sequence_requested")
                d = (r.get("rosetta") or {}).get("raw", {}).get("dSASA_int")
                if d is not None:
                    s["dsasa"].append(d)
    out = {}
    for pid, s in stats.items():
        out[pid] = {
            "parent_id": pid,
            "peptide_length": s["length"],
            "reference_sequence": s["sequence"],
            "n_native_samples": s["n"],
            "n_native_raw_valid": s["n_valid"],
            "native_raw_validity": round(s["n_valid"] / s["n"], 4) if s["n"] else None,
            "median_raw_dSASA_int": round(st.median(s["dsasa"]), 1) if s["dsasa"] else None,
            "substitution_budget": budget_for(s["length"]) if s["length"] else None,
        }
    return out


def select_cohort(stats: dict[str, dict], n: int) -> tuple[list[str], str]:
    """Stratify by native cyclic validity, then spread peptide length.

    The rule, applied in this order and recorded so the cohort is auditable:

    1. Bucket parents by native raw validity into strong (>= 0.5), weak (0 < v < 0.5) and
       zero (0.0). Selecting only successes would make the pilot's yield meaningless --
       a search that only ever starts from sequences that already work cannot show
       whether it rescues one that does not.
    2. Round-robin across the buckets, and within a bucket take the parent whose peptide
       length is furthest from the lengths already chosen. Length drives both the search
       space size and the substitution ceiling, so a cohort of one length would conflate
       them with the target.
    3. Ties break on parent_id, so the selection is deterministic.
    """
    def bucket(v: float | None) -> str:
        if v is None:
            return "unknown"
        if v >= 0.5:
            return "strong"
        if v > 0.0:
            return "weak"
        return "zero"

    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in stats.values():
        buckets[bucket(s["native_raw_validity"])].append(s)
    for b in buckets.values():
        b.sort(key=lambda s: s["parent_id"])

    order = [b for b in ("strong", "weak", "zero", "unknown") if buckets.get(b)]
    chosen: list[dict] = []
    bi = 0
    while len(chosen) < n and any(buckets[b] for b in order):
        b = order[bi % len(order)]
        bi += 1
        pool = buckets[b]
        if not pool:
            continue
        if not chosen:
            pick = pool[0]
        else:
            taken = [c["peptide_length"] for c in chosen if c["peptide_length"]]
            pick = max(
                pool,
                key=lambda s: (
                    min((abs((s["peptide_length"] or 0) - t) for t in taken), default=0),
                    -ord(s["parent_id"][-1]),
                ),
            )
        pool.remove(pick)
        chosen.append(pick)

    rationale = (
        f"stratified by native raw cyclic validity into "
        f"{ {b: len(v) for b, v in buckets.items()} }, then round-robin across buckets "
        f"taking the most length-distant parent within each; deterministic on parent_id"
    )
    return [c["parent_id"] for c in chosen], rationale


def cmd_select_cohort(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = native_reference_stats(Path(args.dataset_dir))

    if args.all_targets:
        # A corpus prepared from scratch has no scored native-reference rows, so the
        # validity stratification cannot run. That is not a problem to work around: the
        # stratification exists to pick a representative SUBSET, and here every prepared
        # target is taken, so there is nothing to select. The reference is still
        # evaluated per parent during `screen`, which is where the comparison comes from.
        tcsv = Path(args.dataset_dir) / "manifest" / "targets.csv"
        if not tcsv.exists():
            print(f"[cohort] --all-targets needs {tcsv}; run cpseq_prepare_targets.py first")
            return 1
        cohort = sorted(pd.read_csv(tcsv)["parent_id"].astype(str))
        rationale = (
            f"all {len(cohort)} prepared targets from {tcsv}; no stratification, because "
            "the whole prepared set is taken rather than a subset of it"
        )
        if not stats:
            print(f"[cohort] no scored native-reference rows in {args.dataset_dir} "
                  "(expected for a fresh corpus); taking all prepared targets")
    elif not stats:
        print("[cohort] no native reference rows found -- has the tranche been scored? "
              "For a fresh corpus with no prior predictions, pass --all-targets.")
        return 1
    elif args.parents:
        cohort = [p.strip() for p in args.parents.split(",") if p.strip()]
        rationale = "explicitly supplied with --parents"
        missing = [p for p in cohort if p not in stats]
        if missing:
            print(f"[cohort] unknown parent(s): {missing}")
            return 1
    else:
        cohort, rationale = select_cohort(stats, args.n_parents)

    record = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_dir": str(args.dataset_dir),
        "n_parents": len(cohort),
        "cohort": cohort,
        "selection_rationale": rationale,
        "scoring_protocol_version_for_energies": SCORING_PROTOCOL_VERSION,
        "stratification_metric": "native raw-geometry cyclic validity (protocol-independent)",
        # The full table, including parents NOT chosen, so the choice is inspectable.
        "all_parents": sorted(stats.values(), key=lambda s: s["parent_id"]),
        "stratification_applied": not args.all_targets,
        "budget": {
            "scan": SCAN_BUDGET,
            "per_round": ROUND_BUDGET,
            "n_rounds": N_ROUNDS,
            "screened_per_parent_per_arm": SCAN_BUDGET + ROUND_BUDGET * N_ROUNDS,
            "screen_seed": SCREEN_SEED,
            "confirm_seeds": list(CONFIRM_SEEDS),
            "finalists_per_arm": FINALISTS_PER_ARM,
        },
        "alphabet": "".join(default_alphabet(allow_cys=args.allow_cys)),
        "allow_cys": args.allow_cys,
    }
    path = out_dir / "cohort.json"
    path.write_text(json.dumps(record, indent=2))

    print(f"[cohort] {len(cohort)} parents recorded -> {path}")
    # A fresh corpus has no `stats`, so length and sequence come from the target
    # manifest and the validity columns read "-" rather than being invented.
    fallback = {}
    tcsv = Path(args.dataset_dir) / "manifest" / "targets.csv"
    if tcsv.exists():
        for r in pd.read_csv(tcsv).to_dict("records"):
            fallback[str(r["parent_id"])] = r
    hdr = f"{'parent':22s} {'L':>3s} {'budget':>6s} {'nat_valid':>10s} {'dSASA':>8s}  sequence"
    print(hdr)
    for pid in cohort:
        s = stats.get(pid)
        if s is None:
            t = fallback.get(pid, {})
            seq = str(t.get("reference_sequence", ""))
            length = int(t.get("peptide_length", len(seq)) or len(seq))
            print(f"{pid:22s} {length:3d} {budget_for(length):6d} "
                  f"{'-':>10s} {'-':>8s}  {seq}")
            continue
        print(
            f"{pid:22s} {s['peptide_length']:3d} {s['substitution_budget']:6d} "
            f"{str(s['native_raw_validity']):>10s} {str(s['median_raw_dSASA_int']):>8s}  "
            f"{s['reference_sequence']}"
        )
    return 0


# --------------------------------------------------------------------------------------
# Resumable state
# --------------------------------------------------------------------------------------


class Journal:
    """Append-only candidate and outcome records, and the resume index built from them.

    Resume is driven by the OUTCOME file, not by a separate progress marker: a marker can
    disagree with what was actually written, and a search that re-evaluates a sequence it
    has already paid for would both waste budget and double-count it in the yield.

    The key is (sequence, prediction_seed, stage, ARM). The arm belongs in it because the
    two arms legitimately propose the same sequence -- on a 5-mer with 92 single mutants,
    arm B takes 16 of them and a uniform arm A lands on several of the same. Excluding
    arm B's picks from arm A would make the control "uniform except where the guided arm
    agreed", biasing it away from exactly the sequences under test. So the collision is
    allowed, each arm records its own outcome row, and the prediction/score caches make
    the second one free -- a cache hit filling an evaluation slot, which is what the
    budget rules permit. `shared_evaluation_with_arms` records the link.
    """

    def __init__(self, out_dir: Path, parent_id: str):
        self.dir = out_dir / "state" / parent_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = self.dir / "candidates.jsonl"
        self.outcomes_path = self.dir / "outcomes.jsonl"
        self.done: set[tuple[str, int, str, str]] = set()
        self.outcomes: list[dict] = []
        if self.outcomes_path.exists():
            for line in self.outcomes_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001, S112
                    continue
                self.outcomes.append(r)
                # A technical failure does NOT satisfy the resume check. The predictor
                # crashing or timing out is an evaluation that did not happen -- which
                # is why `outcome_row` keeps it in its own field and every report drops
                # it from the denominators -- so treating it as paid for would retire
                # the fold permanently on the first bad GPU. Measured 2026-10-23:
                # `pepbench_C_D_pdb4wjq` lost its GPU mid-task ("CUDA unknown error")
                # and recorded 72 failed folds; under the old rule the complex then read
                # as 650/650 rows with only 578 real folds, and no requeue would ever
                # have gone back for them. Retried rows are appended, and the readers
                # skip the failed ones, so the successful retry is what gets used.
                if not r.get("technical_failure"):
                    self.done.add(self._key(r))
        self._seen_candidates: set[str] = set()
        if self.candidates_path.exists():
            for line in self.candidates_path.read_text().splitlines():
                try:
                    self._seen_candidates.add(json.loads(line)["sequence"])
                except Exception:  # noqa: BLE001, S112
                    continue

    @staticmethod
    def _key(row: dict) -> tuple[str, int, str, str]:
        return (
            row["sequence"],
            row["prediction_seed"],
            row.get("stage", "screen"),
            row.get("arm", ""),
        )

    def has(self, sequence: str, seed: int, stage: str, arm: str) -> bool:
        return (sequence, seed, stage, arm) in self.done

    def arms_with_input_hash(self, ihash: str | None, arm: str) -> list[str]:
        """Other arms that already paid for this exact prediction."""
        if not ihash:
            return []
        return sorted(
            {
                o.get("arm")
                for o in self.outcomes
                if o.get("input_hash") == ihash and o.get("arm") not in (arm, None)
            }
        )

    def sequences_for_arm(self, arm: str) -> set[str]:
        return {o["sequence"] for o in self.outcomes if o.get("arm") == arm}

    def add_candidate(self, row: dict) -> None:
        if row["sequence"] in self._seen_candidates:
            return
        self._seen_candidates.add(row["sequence"])
        with open(self.candidates_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def add_outcome(self, row: dict) -> None:
        self.outcomes.append(row)
        self.done.add(self._key(row))
        with open(self.outcomes_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def screened(self, arm: str | None = None) -> list[dict]:
        return [
            o
            for o in self.outcomes
            if o.get("stage") == "screen" and (arm is None or o.get("arm") == arm)
        ]


def outcome_row(
    *,
    rec: dict,
    proposal: Proposal | None,
    parent_id: str,
    arm: str,
    stage: str,
    extra: dict | None = None,
) -> dict:
    """Flatten one `evaluate` record into a per-seed outcome row.

    Structural failure and technical failure are kept in different fields on purpose: a
    crashed job is not evidence that a sequence does not bind, and pooling the two would
    put predictor timeouts into a nonbinding denominator.
    """
    best = None
    samples = rec.get("samples", [])
    if samples:
        # The representative sample: the valid one with the best energy, else the one
        # with the smallest violation, so the row always describes something.
        valid = [s for s in samples if s.get("valid")]
        pool = valid or samples
        best = min(
            pool,
            key=lambda s: (
                s.get("refined_dG_separated")
                if s.get("valid") and s.get("refined_dG_separated") is not None
                else (s.get("violation") if s.get("violation") is not None else 1e9)
            ),
        )

    row = {
        "parent_id": parent_id,
        "pdb_id": rec.get("pdb_id"),
        "family_id": rec.get("family_id"),
        "split": rec.get("split"),
        "stage": stage,
        "arm": arm,
        "sequence": rec["sequence"],
        "original_sequence": rec.get("original_sequence"),
        "peptide_length": rec.get("peptide_length"),
        "n_mutations": rec.get("n_mutations"),
        "mutation_fraction": rec.get("mutation_fraction"),
        "mutated_positions": rec.get("mutated_positions"),
        "substitution_budget": rec.get("substitution_budget"),
        "identity_to_original": rec.get("identity_to_original"),
        "prediction_seed": rec.get("prediction_seed"),
        "chemistry": rec.get("chemistry"),
        "status": rec.get("status"),
        "technical_failure": bool(rec.get("technical_failure", False)),
        "failure_reasons": rec.get("reasons"),
        "valid": bool(rec.get("valid", False)),
        "n_samples": rec.get("n_samples"),
        "n_valid_samples": rec.get("n_valid"),
        "best_valid_dG_separated": rec.get("best_valid_dG_separated"),
        "min_violation": rec.get("min_violation"),
        "input_hash": rec.get("input_hash"),
        "scoring_version": rec.get("scoring_version"),
        "cache_prediction": (rec.get("cache") or {}).get("prediction"),
        "cache_score": (rec.get("cache") or {}).get("score"),
        "wall_time_s": rec.get("wall_time_s"),
        "prediction_runtime_s": rec.get("prediction_runtime_s"),
    }
    if proposal is not None:
        row |= {
            "round_index": proposal.round_index,
            "proposal_kind": proposal.kind,
            "ancestor_sequence": proposal.ancestor_sequence,
            "edited_positions": ",".join(str(p) for p in proposal.edited_positions),
            "proposal_seed": proposal.seed,
        }
    if best is not None:
        cl = best.get("closure") or {}
        row |= {
            "sample_index": best.get("sample_index"),
            "scoring_status": best.get("scoring_status"),
            "validity_reasons": best.get("validity_reasons"),
            "violation": best.get("violation"),
            "sequence_matches": best.get("sequence_matches"),
            "closure_status": cl.get("status"),
            "closure_bond_A": cl.get("bond_length_A"),
            "closure_bond_dev_A": cl.get("bond_length_dev_A"),
            "closure_angle_i_dev_deg": cl.get("angle_i_dev_deg"),
            "closure_angle_j_dev_deg": cl.get("angle_j_dev_deg"),
            "closure_dihedral_dev_deg": cl.get("dihedral_dev_deg"),
            "n_intra_clashes": (best.get("clashes") or {}).get("n_intra_clashes"),
            "n_inter_clashes": (best.get("clashes") or {}).get("n_inter_clashes"),
            "raw_pdb_path": best.get("raw_pdb_path"),
            "refined_pdb_path": best.get("refined_pdb_path"),
            "deformation_status": best.get("deformation_status"),
            "strain_available": best.get("strain_available"),
            "local_deformation_energy_proxy": best.get("local_deformation_energy_proxy"),
            "refine_binder_ca_rmsd_A": best.get("refine_binder_ca_rmsd_A"),
        }
        for k in ("dG_separated", "dSASA_int", "dG_separated_per_1000A2",
                  "delta_unsatHbonds", "hbonds_int", "sc_value", "packstat"):
            row[f"raw_{k}"] = best.get(f"raw_{k}")
            row[f"refined_{k}"] = best.get(f"refined_{k}")
        row["confidence"] = best.get("confidence")
    if extra:
        row |= extra
    return row


def entry_from_outcome(o: dict) -> ArchiveEntry:
    return ArchiveEntry(
        sequence=o["sequence"],
        objective=o.get("best_valid_dG_separated"),
        violation=o.get("min_violation"),
        valid=bool(o.get("valid")),
        round_index=o.get("round_index", 0),
        kind=o.get("proposal_kind", ""),
        ancestor_sequence=o.get("ancestor_sequence"),
    )


# --------------------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------------------


def run_parent(
    *,
    parent_id: str,
    target: TargetContext,
    out_dir: Path,
    protocol: Protocol,
    store: Store,
    arms: list[str],
    dry_run: bool = False,
    limit: int = 0,
    proposals: list[dict] | None = None,
    random_policy: str = "match_guided",
) -> dict:
    journal = Journal(out_dir, parent_id)
    original = target.reference_sequence
    alpha = default_alphabet(allow_cys=protocol.allow_cys)
    summary = {"parent_id": parent_id, "original_sequence": original,
               "peptide_length": len(original), "budget": budget_for(len(original)),
               "evaluations": 0, "cache_hits": 0, "random_policy": random_policy,
               "notes": []}

    def _evaluate(seq: str, proposal: Proposal | None, arm: str, stage: str, seed: int):
        if journal.has(seq, seed, stage, arm):
            return next(
                (o for o in journal.outcomes
                 if o["sequence"] == seq and o["prediction_seed"] == seed
                 and o.get("stage") == stage and o.get("arm") == arm),
                None,
            )
        if dry_run:
            print(f"    [dry] {arm} {stage} seed={seed} {seq}")
            return None
        # HARD per-arm budget, counted from the JOURNAL rather than from this process's
        # own tally. Without this a requeued task overspends: the proposers were being
        # handed the journal's sequences as an `exclude` set, so on resume they avoided
        # reproducing their own earlier proposals and drew fresh ones on top of the work
        # already paid for. Measured: `lnr_5njx` reached 36 guided evaluations against a
        # budget of 32 after one preemption, while parents that never requeued sat at
        # exactly 32. The resume never DUPLICATED an evaluation -- it bought extra ones,
        # which is the same budget violation seen from the other side, and it would have
        # handed one arm more slots than the arm it is compared against.
        if stage == "screen" and arm in ARM_BUDGETS:
            remaining = screen_slots_remaining(
                journal.outcomes, arm, ARM_BUDGETS[arm]
            )
            if remaining <= 0:
                key = f"_budget_announced_{arm}"
                if not summary.get(key):
                    summary[key] = True
                    summary["notes"].append(
                        f"{arm}: budget {ARM_BUDGETS[arm]} already spent on disk; "
                        f"further proposals not evaluated"
                    )
                return None

        # Preflight cap. Counts only NEW evaluations, so a resumed run is not throttled
        # by work it already has on disk.
        if limit and summary["evaluations"] >= limit:
            if not summary.get("_limit_announced"):
                summary["_limit_announced"] = True
                summary["notes"].append(
                    f"preflight --limit {limit} reached; remaining proposals not evaluated"
                )
                print(f"    [limit] {limit} new evaluation(s) reached; stopping this parent")
            return None
        rec = evaluate(
            original_sequence=original,
            candidate_sequence=seq,
            target=target,
            prediction_seed=seed,
            protocol=protocol,
            store=store,
        )
        row = outcome_row(
            rec=rec, proposal=proposal, parent_id=parent_id, arm=arm, stage=stage,
            extra={
                "shared_evaluation_with_arms": journal.arms_with_input_hash(
                    rec.get("input_hash"), arm
                )
            },
        )
        journal.add_outcome(row)
        summary["evaluations"] += 1
        if row.get("cache_prediction") == "hit":
            summary["cache_hits"] += 1
        flag = "VALID" if row["valid"] else row["status"]
        print(
            f"    {arm:14s} r{row.get('round_index', '-')} n_mut={row['n_mutations']} "
            f"{seq} -> {flag} dG={row.get('best_valid_dG_separated')} "
            f"viol={row.get('min_violation')} {row.get('wall_time_s')}s"
        )
        return row

    # --- the unmutated cyclic reference, shared by every arm ------------------------
    # This is a CP predicted from the original LP sequence. It is NOT an LP energy
    # baseline, and the pilot's primary comparison is mutant CP vs this.
    ref_prop = Proposal(
        sequence=original, arm="reference", round_index=-1, kind=KIND_REFERENCE,
        ancestor_sequence=None, original_sequence=original, edited_positions=(),
        seed=proposal_seed(parent_id, "reference"),
    )
    journal.add_candidate(
        {"parent_id": parent_id, **ref_prop.to_row(), "generator": None, "edit_mask": None}
    )
    print(f"  [{parent_id}] reference {original} (L={len(original)}, "
          f"budget={budget_for(len(original))})")
    _evaluate(original, ref_prop, "reference", "screen", SCREEN_SEED)

    # --- Arm B: single-substitution scan, then two guided rounds --------------------
    guided_counts_by_round: dict[int, list[int]] = {}
    if ARM_GUIDED in arms:
        archive = Archive(capacity=6, min_distance=1, keep_infeasible=3)
        for o in journal.screened():
            if o.get("arm") in (ARM_GUIDED, "reference"):
                archive.add(entry_from_outcome(o))

        scan = select_spread_singles(
            original, SCAN_BUDGET, seed=proposal_seed(parent_id, "scan"), alphabet=alpha
        )
        if len(scan) < SCAN_BUDGET:
            summary["notes"].append(
                f"single-substitution scan yielded {len(scan)}/{SCAN_BUDGET}: the "
                f"neighbourhood has only {len(scan)} distinct single mutants"
            )
        print(f"  [{parent_id}] arm B round 0: {len(scan)} single substitutions")
        for p in scan:
            journal.add_candidate(
                {"parent_id": parent_id, **p.to_row(), "generator": None, "edit_mask": None}
            )
            row = _evaluate(p.sequence, p, ARM_GUIDED, "screen", SCREEN_SEED)
            if row:
                archive.add(entry_from_outcome(row))
        guided_counts_by_round[0] = [p.n_mutations for p in scan]

        for rnd in range(1, N_ROUNDS + 1):
            # Only this arm's own sequences and the original are excluded, so the plan is
            # a deterministic function of (parent, round, archive) and a resume
            # regenerates it. Excluding the whole journal made the proposers skip their
            # own earlier proposals and overspend the budget on requeue.
            seen = {
                o["sequence"] for o in journal.screened(ARM_GUIDED)
            } | {original}
            props, note = guided_round(
                original=original, archive=archive, round_index=rnd,
                budget=ROUND_BUDGET, seed_base=parent_id, alphabet=alpha, exclude=seen,
            )
            if note:
                summary["notes"].append(note)
            print(f"  [{parent_id}] arm B round {rnd}: {len(props)} proposals"
                  f"{' -- ' + note if note else ''}")
            for p in props:
                journal.add_candidate(
                    {"parent_id": parent_id, **p.to_row(), "generator": None, "edit_mask": None}
                )
                row = _evaluate(p.sequence, p, ARM_GUIDED, "screen", SCREEN_SEED)
                if row:
                    archive.add(entry_from_outcome(row))
            guided_counts_by_round[rnd] = [p.n_mutations for p in props]
            # State is persisted after every round by the append-only journal; this
            # line records the archive itself so a resumed run can be inspected.
            (journal.dir / f"archive_round{rnd}.json").write_text(
                json.dumps(
                    [
                        {"sequence": e.sequence, "objective": e.objective,
                         "violation": e.violation, "valid": e.valid}
                        for e in archive.working_set()
                    ],
                    indent=2,
                )
            )

    # --- Arm C: ProteinMPNN redesigns of generator CP backbones ---------------------
    # The sequences arrive pre-built from `cpseq_mpnn_propose.py` (the generator and MPNN
    # are a separate GPU stage), and are evaluated here through the IDENTICAL interface as
    # the other arms. That is the whole point: whatever the proposal mechanism, the
    # measurement is the same fresh Boltz-2 refold and the same frozen gate, so the arms
    # differ only in where their sequences came from.
    if ARM_MPNN in arms:
        mine = [p for p in (proposals or []) if p.get("parent_id") == parent_id]
        if not mine:
            summary["notes"].append(
                "arm C: no admitted-backbone proposals for this parent (see "
                "backbones/admission_*.jsonl for why each backbone was rejected)"
            )
        else:
            print(f"  [{parent_id}] arm C: {len(mine)} MPNN proposal(s)")
            for pr in mine:
                seq = pr["sequence"]
                mask = tuple(pr.get("edit_mask") or ())
                p = Proposal(
                    sequence=seq,
                    arm=ARM_MPNN,
                    round_index=0,
                    kind=KIND_MPNN,
                    # The backbone, not a sequence, is this proposal's ancestor; naming a
                    # sequence ancestor would imply an edit path that does not exist.
                    ancestor_sequence=None,
                    original_sequence=original,
                    edited_positions=tuple(
                        i for i, (a, b) in enumerate(zip(original, seq)) if a != b
                    ),
                    seed=int(pr.get("backbone_sample_index") or 0),
                )
                journal.add_candidate({
                    "parent_id": parent_id, **p.to_row(),
                    "generator": pr.get("generator"),
                    "edit_mask": ",".join(str(i) for i in mask),
                })
                row = _evaluate(seq, p, ARM_MPNN, "screen", SCREEN_SEED)
                if row is not None:
                    # Proposal-side provenance, recorded on the outcome so a per-generator
                    # comparison is possible without rejoining another file.
                    row.setdefault("generator", pr.get("generator"))
                    row.setdefault("generator_ckpt_step", pr.get("generator_ckpt_step"))
                    row.setdefault("edit_mask", ",".join(str(i) for i in mask))
                    row.setdefault("mask_size", pr.get("mask_size"))
                    row.setdefault("mpnn_score", pr.get("mpnn_score"))
                    row.setdefault("mpnn_global_score", pr.get("mpnn_global_score"))
                    row.setdefault("backbone_pdb_used_path", pr.get("backbone_pdb_used_path"))
                    row.setdefault("mpnn_is_cyclic_aware", pr.get("mpnn_is_cyclic_aware"))

    # --- Arm A: random controls ------------------------------------------------------
    if ARM_RANDOM in arms:
        if not guided_counts_by_round:
            # No guided counts to match. Two possibilities, and they are different
            # situations rather than one fallback: a matched pilot that lost its guided
            # arm has a real problem and should say so, while a deliberate random-only
            # run has simply declared that it does not want the match. `--random-policy
            # standalone` is that declaration, and it is opt-in so that a matched run can
            # never silently become an unmatched one.
            if random_policy != "standalone":
                summary["notes"].append(
                    "arm A skipped: no guided counts to match (run arm B first, or pass "
                    "--random-policy standalone for a self-contained random arm)"
                )
            else:
                cap = ARM_BUDGETS.get(ARM_RANDOM, _SCREEN_BUDGET)
                remaining = screen_slots_remaining(journal.outcomes, ARM_RANDOM, cap)
                # The full `cap`-length plan is generated every time, deliberately, and
                # `exclude` is NOT seeded from the journal. The plan is then a pure
                # function of (parent, cap), so a requeued task reproduces the same
                # slot-to-count assignment and `journal.has` skips what was already paid
                # for. Handing it the journal's sequences is precisely the overspend bug
                # documented in `_evaluate` below, seen from the proposer's side.
                props, note = standalone_random_arm(
                    original=original, n_slots=cap, seed_base=parent_id,
                    alphabet=alpha,
                )
                if note:
                    summary["notes"].append(f"arm A standalone: {note}")
                print(f"  [{parent_id}] arm A standalone: {len(props)} draws "
                      f"(counts {sorted(p.n_mutations for p in props)}), "
                      f"{remaining} slots remaining of {cap}")
                for p in props:
                    journal.add_candidate(
                        {"parent_id": parent_id, **p.to_row(),
                         "generator": None, "edit_mask": None}
                    )
                    _evaluate(p.sequence, p, ARM_RANDOM, "screen", SCREEN_SEED)
        else:
            # Arm A excludes only the original and its OWN prior draws, so that its 32
            # slots hold 32 distinct sequences without duplicate evaluations. It does
            # NOT exclude arm B's sequences -- see `Journal` for why that would bias the
            # control. A collision reuses the cached prediction and score.
            seen = journal.sequences_for_arm(ARM_RANDOM) | {original}
            for rnd, counts in sorted(guided_counts_by_round.items()):
                props, note = match_counts(
                    original=original, counts=counts, round_index=rnd,
                    seed_base=parent_id, exclude=seen, alphabet=alpha,
                )
                if note:
                    summary["notes"].append(f"arm A round {rnd}: {note}")
                print(f"  [{parent_id}] arm A round {rnd}: {len(props)} matched controls"
                      f" (counts {sorted(counts)})")
                for p in props:
                    seen.add(p.sequence)
                    journal.add_candidate(
                        {"parent_id": parent_id, **p.to_row(),
                         "generator": None, "edit_mask": None}
                    )
                    _evaluate(p.sequence, p, ARM_RANDOM, "screen", SCREEN_SEED)

    return summary


def cmd_screen(args) -> int:
    out_dir = Path(args.out_dir)
    cohort_path = out_dir / "cohort.json"
    if not cohort_path.exists():
        print(f"[screen] no cohort at {cohort_path} -- run select-cohort first")
        return 1
    cohort = json.loads(cohort_path.read_text())
    parents = cohort["cohort"][args.shard_index :: args.shard_count]
    if args.parents:
        parents = [p.strip() for p in args.parents.split(",") if p.strip()]

    protocol = Protocol(
        chemistry=args.chemistry,
        diffusion_samples=args.samples,
        boltz_python=args.boltz_python,
        boltz_cache=args.boltz_cache,
        allow_cys=bool(cohort.get("allow_cys", False)),
    )
    store = Store(
        root=out_dir,
        tolerances_path=Path(cohort["dataset_dir"]) / "manifest" / "geometry_tolerances.json",
        retention=args.retention,
    )
    contexts = load_target_contexts(Path(cohort["dataset_dir"]), parents)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # Arm C's sequences are produced by a separate GPU stage (generator + ProteinMPNN);
    # this stage only evaluates them, so several generators' proposal files can be
    # supplied at once and each outcome carries the generator that produced it.
    proposals: list[dict] = []
    for pattern in [q.strip() for q in (args.proposals or "").split(",") if q.strip()]:
        for path in sorted(glob.glob(pattern)):
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    proposals.append(json.loads(line))
    if proposals:
        print(f"[screen] {len(proposals)} MPNN proposal(s) loaded for arm C")

    print(f"[screen] {len(parents)} parent(s) shard {args.shard_index}/{args.shard_count}")
    print(f"[screen] arms={arms} protocol={protocol.to_dict()}")

    summaries = []
    for pid in parents:
        if pid not in contexts:
            print(f"[screen] {pid}: no target context -- skipped")
            continue
        s = run_parent(
            parent_id=pid, target=contexts[pid], out_dir=out_dir, protocol=protocol,
            store=store, arms=arms, dry_run=args.dry_run, limit=args.limit,
            random_policy=args.random_policy,
            proposals=proposals,
        )
        summaries.append(s)
        print(f"  [{pid}] {s['evaluations']} new evaluation(s), "
              f"{s['cache_hits']} prediction cache hit(s)")
        for n in s["notes"]:
            print(f"  [{pid}] note: {n}")

    (out_dir / f"screen_summary_{args.shard_index}.json").write_text(
        json.dumps(
            [{k: v for k, v in s.items() if not k.startswith("_")} for s in summaries],
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------------------


def cmd_confirm(args) -> int:
    out_dir = Path(args.out_dir)
    cohort = json.loads((out_dir / "cohort.json").read_text())
    parents = cohort["cohort"][args.shard_index :: args.shard_count]

    protocol = Protocol(
        chemistry=args.chemistry, diffusion_samples=args.samples,
        boltz_python=args.boltz_python, boltz_cache=args.boltz_cache,
        allow_cys=bool(cohort.get("allow_cys", False)),
    )
    store = Store(
        root=out_dir,
        tolerances_path=Path(cohort["dataset_dir"]) / "manifest" / "geometry_tolerances.json",
        retention=args.retention,
    )
    contexts = load_target_contexts(Path(cohort["dataset_dir"]), parents)

    # Seed set is overridable because 3 seeds cannot size the effect: the measured
    # seed-to-seed spread on this cohort is 4.3 REU median and 21.97 REU max, which is
    # larger than most of the arm differences being compared. The journal dedupes on
    # (sequence, seed, stage, arm), so passing a superset of an earlier run re-uses the
    # seeds already paid for and evaluates only the new ones.
    seeds = (tuple(int(x) for x in args.seeds.replace(",", " ").split())
             if getattr(args, "seeds", "") else CONFIRM_SEEDS)

    for pid in parents:
        if pid not in contexts:
            continue
        target = contexts[pid]
        journal = Journal(out_dir, pid)
        original = target.reference_sequence
        screened = journal.screened()
        if not screened:
            print(f"[confirm] {pid}: nothing screened yet")
            continue

        # Best FINALISTS_PER_ARM per arm, ranked by the hierarchy: accepted candidates
        # by energy first, and nothing invalid is promoted.
        finalists: dict[str, dict] = {}
        for arm in sorted({o["arm"] for o in screened} - {"reference"}):
            rows = [o for o in screened if o["arm"] == arm and o["valid"]
                    and o.get("best_valid_dG_separated") is not None]
            rows.sort(key=lambda o: o["best_valid_dG_separated"])
            for o in rows[:FINALISTS_PER_ARM]:
                # Deduplicated across arms: one sequence gets one confirmation record,
                # shared, rather than being predicted twice under two labels.
                finalists.setdefault(o["sequence"], o)
            if not rows:
                print(f"[confirm] {pid}: arm {arm} has no valid screened candidate")
        # The unmutated reference is always confirmed -- it is the comparison.
        finalists.setdefault(original, {"arm": "reference", "proposal_kind": KIND_REFERENCE})

        print(f"[confirm] {pid}: {len(finalists)} sequence(s) x {len(seeds)} seeds")
        for seq, src in finalists.items():
            arm = src.get("arm", "unknown")
            for seed in seeds:
                if journal.has(seq, seed, "confirm", arm):
                    continue
                if args.dry_run:
                    print(f"    [dry] confirm seed={seed} {seq}")
                    continue
                rec = evaluate(
                    original_sequence=original, candidate_sequence=seq, target=target,
                    prediction_seed=seed, protocol=protocol, store=store,
                )
                row = outcome_row(
                    rec=rec, proposal=None, parent_id=pid,
                    arm=arm, stage="confirm",
                    extra={
                        "screened_arm": src.get("arm"),
                        "proposal_kind": src.get("proposal_kind"),
                        "screen_best_valid_dG_separated": src.get("best_valid_dG_separated"),
                    },
                )
                journal.add_outcome(row)
                print(f"    confirm seed={seed} {seq} -> "
                      f"{'VALID' if row['valid'] else row['status']} "
                      f"dG={row.get('best_valid_dG_separated')}")
    return 0


# --------------------------------------------------------------------------------------
# Seed-ladder cohort and run
# --------------------------------------------------------------------------------------


def _parse_corpus_spec(spec: str) -> dict:
    """One `--corpus` argument: `key=value` pairs, comma separated.

    Keys: `dataset_dir` (required), `n` (parents to take), `label` (what the corpus is
    called in the tables).

    `parents` is an explicit comma-separated list, for a PAIRED comparison: the same
    complexes evaluated under a different inference setting, where sampling a fresh set
    would confound the setting with the complexes. It is not a way to inherit a target
    list filtered by a previous run's outcomes -- that remains unsupported, because
    taking the targets some earlier run happened to get through is how a one-fold result
    would leak into a run meant to replace it. The distinction is whether the list was
    chosen by what was measured: naming complexes up front is auditable (the list is
    written into the cohort), selecting them by their results is not.

    The other leak route -- sharing a prediction cache -- is closed by giving each run
    its own output directory, which is also required here because the journal key is
    (sequence, seed, stage, arm) and carries no inference settings: two settings in one
    directory would see each other's folds as already paid for.
    """
    out: dict = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"[ladder-cohort] malformed --corpus fragment {part!r}; "
                             "expected key=value")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    if "dataset_dir" not in out:
        raise SystemExit(f"[ladder-cohort] --corpus {spec!r} has no dataset_dir=")
    unknown = set(out) - {"dataset_dir", "n", "label", "parents"}
    if unknown:
        raise SystemExit(
            f"[ladder-cohort] --corpus {spec!r} has unsupported key(s) {sorted(unknown)}. "
            "Selecting targets from a previous run's output is not supported here; see "
            "the docstring for why."
        )
    # An explicit `parents` list is itself the request, so it must not inherit the
    # default count -- otherwise naming 2 complexes reports "asked for 10, taking 2",
    # which reads as a shortfall when nothing was short.
    if "n" in out or not out.get("parents"):
        out["n"] = int(out.get("n", 10))
    return out


def cmd_ladder_cohort(args) -> int:
    """Cohort for the seed ladder: several corpora, each sampled by a recorded rule.

    Why this is not `select-cohort --all-targets`: the ladder asks whether the budget
    answer generalises, which needs more than one corpus in one run, and `select-cohort`
    writes a single `dataset_dir` that every later stage reads. Here each parent carries
    its own, so one run spans LNR and PepBench without the two being pooled into a single
    unlabelled set.

    The sample is uniform without replacement from the eligible pool, seeded on the
    corpus label so it is reproducible and recorded. Uniform is the right rule for a
    ladder: coverage-at-B is a corpus-level expectation, and stratifying on observed
    validity -- the rule `select_cohort` uses to pick a development cohort -- would
    choose targets by the outcome the ladder is trying to estimate.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpora = [_parse_corpus_spec(s) for s in args.corpus]
    if not corpora:
        print("[ladder-cohort] at least one --corpus is required")
        return 1

    # The frozen gate must be ONE gate. The tolerances come from four experimental
    # crystal macrocycle controls rather than from any corpus's own data, so two corpora
    # whose tolerance files differ are not comparable and no table built from them would
    # mean anything. Checked by content hash, not by trusting that the copy was made.
    tol_hashes: dict[str, str] = {}
    for c in corpora:
        tp = Path(c["dataset_dir"]) / "manifest" / "geometry_tolerances.json"
        if not tp.exists():
            print(f"[ladder-cohort] missing frozen tolerances: {tp}")
            return 1
        tol_hashes[str(tp)] = gate_sha256(tp)
    if len(set(tol_hashes.values())) != 1:
        print("[ladder-cohort] the corpora do not share one frozen gate; refusing to "
              "build a cohort whose rows would not be comparable:")
        for p, h in tol_hashes.items():
            print(f"    {h[:16]}  {p}")
        return 1

    import random as _random

    cohort: list[str] = []
    parent_sources: dict[str, str] = {}
    corpus_records: list[dict] = []
    for c in corpora:
        ddir = Path(c["dataset_dir"])
        label = c.get("label") or ddir.name
        tcsv = ddir / "manifest" / "targets.csv"
        if not tcsv.exists():
            print(f"[ladder-cohort] {label}: no {tcsv}")
            return 1
        prepared = sorted(pd.read_csv(tcsv)["parent_id"].astype(str))
        pool = prepared
        rule = ("every prepared target; no filtering by what any earlier run produced, "
                "so this evaluation stands alone")
        if c.get("parents"):
            named = [p.strip() for p in str(c["parents"]).split("|") if p.strip()]
            missing = [p for p in named if p not in prepared]
            if missing:
                print(f"[ladder-cohort] {label}: not prepared in this corpus: {missing}")
                return 1
            pool = named
            rule = (f"{len(named)} complexes named explicitly, for a paired comparison "
                    f"against another run on the same complexes")
        if not pool:
            print(f"[ladder-cohort] {label}: empty pool")
            return 1
        # No n= with an explicit list means "all of them"; the list IS the request.
        requested = int(c["n"]) if c.get("n") else len(pool)
        n = min(requested, len(pool))
        if n < requested:
            print(f"[ladder-cohort] {label}: pool holds {len(pool)}, "
                  f"asked for {requested}; taking all {n}")
        if c.get("parents") and n == len(pool):
            # Named and taken whole: no sampling, so the order is the list's own.
            picked = sorted(pool)
        else:
            rng = _random.Random(proposal_seed(label, "ladder_cohort", requested))
            picked = sorted(rng.sample(pool, n))
        cohort += picked
        for p in picked:
            parent_sources[p] = str(ddir)
        corpus_records.append({
            "label": label,
            "dataset_dir": str(ddir),
            "n_prepared": len(prepared),
            "n_pool": len(pool),
            "pool_rule": rule,
            "n_selected": n,
            "selection_rule": ("named explicitly" if c.get("parents") and n == len(pool)
                               else "uniform without replacement, seeded on the corpus label"),
            "selection_seed": (None if c.get("parents") and n == len(pool)
                               else proposal_seed(label, "ladder_cohort", requested)),
            "parents": picked,
        })
        print(f"[ladder-cohort] {label}: {n} of {len(pool)} eligible "
              f"({len(prepared)} prepared)")

    record = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": "seed_ladder",
        "cohort": cohort,
        "n_parents": len(cohort),
        "parent_sources": parent_sources,
        "corpora": corpus_records,
        # Kept so that any tool written against a `select-cohort` record still finds a
        # dataset_dir; every ladder stage reads `parent_sources` instead.
        "dataset_dir": corpus_records[0]["dataset_dir"],
        "geometry_tolerances_sha256": sorted(set(tol_hashes.values()))[0],
        "n_draws": args.draws,
        "seeds": list(LADDER_SEEDS[: args.seed_depth]),
        # Fold counts the REPORT truncates to; the requested depth is always the last of
        # them. They cost nothing, because every folded draw carries all of them -- this
        # is one run read at several depths, not several runs.
        "seed_tiers": sorted(
            {t for t in LADDER_SEED_TIERS if t < args.seed_depth} | {args.seed_depth}
        ),
        # The ladder's rungs. Nested prefixes of one draw stream, so B=8 is the first 8
        # draws of the 64 rather than a separate run, and the evaluation order finishes
        # a rung at full fold depth before opening the next.
        "budget_rungs": [b for b in LADDER_BUDGETS if b <= args.draws] + (
            [args.draws] if args.draws not in LADDER_BUDGETS else []
        ),
        # The arm this cohort's draws come from BY DEFAULT. A cohort fixes the corpus,
        # the parents, the seed set and the rungs -- all of which arm D reuses unchanged
        # -- so an LLM run is the same cohort folded with a different proposal source,
        # and `ladder --llm-round` records the arm it actually used on its own summary.
        "arm": ARM_RANDOM,
        "random_policy": "standalone",
        "stage": STAGE_LADDER,
        # Boltz-2's physical-plausibility steering. Recorded on the cohort because it
        # changes what the structures ARE, not just how they are scored: a run with it on
        # is not comparable fold-for-fold with one without, and the journal key carries no
        # inference settings, so the two must live in separate output directories.
        "use_potentials": bool(args.use_potentials),
        "scoring_protocol_version_for_energies": SCORING_PROTOCOL_VERSION,
        "allow_cys": args.allow_cys,
        "alphabet": "".join(default_alphabet(allow_cys=args.allow_cys)),
        "design_note": (
            "Every draw is folded at every seed; nothing is promoted and nothing is "
            "filtered on a single fold. A draw's outcome is its validity RATE over "
            "seeds, so the gate is a measurement rather than a filter applied to one "
            "draw of the predictor's noise."
        ),
    }
    path = out_dir / "cohort.json"
    path.write_text(json.dumps(record, indent=2))
    evals = len(cohort) * (args.draws + 1) * args.seed_depth
    print(f"[ladder-cohort] {len(cohort)} complexes -> {path}")
    print(f"[ladder-cohort] plan: {args.draws} draws + the unchanged peptide, "
          f"{args.seed_depth} folds each")
    print(f"[ladder-cohort] ladder rungs B={record['budget_rungs']} "
          f"(nested prefixes of one draw stream; no rung costs its own folds)")
    print(f"[ladder-cohort] report also truncates the folds to {record['seed_tiers']}")
    print(f"[ladder-cohort] {evals} folds at full depth "
          f"(~{evals * 75 / 3600:.0f} GPU-h at the measured 75 s mean per fold)")
    # Cost per rung, so the stopping decision is informed rather than guessed at.
    prev = 0
    for b in record["budget_rungs"]:
        cum = len(cohort) * (b + 1) * args.seed_depth
        print(f"    B={b:>3}  cumulative {cum:>6} folds  ~{cum * 75 / 3600:>5.0f} GPU-h")
        prev = b
    return 0


def load_llm_proposals(
    out_dir: Path, parent_id: str, *, through_round: int
) -> tuple[list[dict], list[str]]:
    """Arm D's draws for one parent, in a globally stable order.

    Rounds are concatenated in order and each proposal keeps the index it gets from that
    concatenation, so round 1's batch is draws 0..k-1, round 2's is k..2k-1, and so on.
    That ordering is what lets the ladder report treat this arm exactly like the random
    stream: its rungs are PREFIXES of the draws, so draw 9 must mean the same sequence in
    a two-round run and in a five-round run. Assigning indices per round, or sorting by
    sequence, would renumber earlier draws as later rounds arrived and silently change
    what every finished rung had measured.
    """
    rows: list[dict] = []
    notes: list[str] = []
    for rnd in range(1, through_round + 1):
        path = out_dir / "llm_proposals" / f"round{rnd}" / f"{parent_id}.jsonl"
        if not path.exists():
            notes.append(f"round {rnd}: no proposal file for this parent")
            continue
        n_before = len(rows)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            if r.get("sequence"):
                rows.append(r)
        notes.append(f"round {rnd}: {len(rows) - n_before} proposal(s)")
    # Deduplicate while preserving first appearance. A sequence proposed twice across
    # rounds is one draw, not two: the second would be a cache hit filling a slot, which
    # would overstate how many distinct candidates this arm actually bought.
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        if r["sequence"] in seen:
            notes.append(f"dropped duplicate proposal {r['sequence']} (already a draw)")
            continue
        seen.add(r["sequence"])
        unique.append(r)
    for i, r in enumerate(unique):
        r["_draw_index"] = i
    return unique, notes


def run_parent_ladder(
    *,
    parent_id: str,
    target: TargetContext,
    out_dir: Path,
    protocol: Protocol,
    store: Store,
    n_draws: int,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    dry_run: bool = False,
    limit: int = 0,
    llm_proposals: list[dict] | None = None,
    # Group the sequences that share a seed into one boltz invocation. A prefetch into
    # the same cache `evaluate` reads, so it changes no result -- only who paid for the
    # prediction. See `prefetch_predictions` for the measured 44 s + 26 s/structure.
    batch_predict: bool = True,
    # Overlap the GPU phase with the CPU phase. Off by default: it changes no result,
    # but it changes the shape of a running task, and the corpus run should adopt it
    # deliberately rather than by upgrade. See the block below for what it does.
    prefetch_ahead: bool = False,
    # Stop after the predictions, before any relax. Splits one GPU job that is 75 % idle
    # into a GPU job that only predicts and a CPU job that only scores -- measured 14.3 s
    # of GPU prediction against 43.7 s of CPU relax per fold, so three quarters of the
    # wall time of a "GPU" task was Rosetta with the card sitting idle. The second pass
    # is the SAME command without this flag: every prediction is then a cache hit, so it
    # needs no GPU and can run under `preempt_cpu_qos`. Changes no result -- the caches
    # are keyed identically either way -- only who pays for what.
    predict_only: bool = False,
) -> dict:
    """One parent's draws, each folded at every seed, in prefix-safe order."""
    journal = Journal(out_dir, parent_id)
    original = target.reference_sequence
    alpha = default_alphabet(allow_cys=protocol.allow_cys)
    summary = {
        "parent_id": parent_id,
        "original_sequence": original,
        "peptide_length": len(original),
        "budget": budget_for(len(original)),
        "n_draws_planned": n_draws,
        "evaluations": 0,
        "cache_hits": 0,
        "notes": [],
    }

    # Where the draws come from. This is the ONLY thing arm D changes: everything below
    # -- the prefix-safe plan, the fold depth, the gate, the evaluator, the journal -- is
    # shared, so a difference between the arms is attributable to the proposer and to
    # nothing else.
    draw_arm = ARM_LLM if llm_proposals is not None else ARM_RANDOM
    if llm_proposals is not None:
        props = []
        for r in llm_proposals:
            seq = r["sequence"]
            try:
                props.append(Proposal(
                    sequence=seq,
                    arm=ARM_LLM,
                    round_index=int(r.get("llm_round") or r.get("round_index") or 0),
                    kind=KIND_LLM,
                    # The elite the model was asked to edit, as the propose stage
                    # recorded it. Carried through rather than dropped: the ladder is
                    # where the offspring is measured, so this is the row that has to
                    # state whether an improvement descended from the elite that was
                    # supposed to produce it. It is None for a generation that had no
                    # elite to breed from -- the first one, or a parent with nothing
                    # scored yet -- and naming an ancestor there would assert an edit
                    # path that does not exist.
                    ancestor_sequence=r.get("ancestor_sequence") or None,
                    original_sequence=original,
                    edited_positions=tuple(
                        i for i, (a, b) in enumerate(zip(original, seq)) if a != b
                    ),
                    seed=int(r.get("proposal_seed") or 0),
                    note=str(r.get("note") or "")[:200],
                    slot=int(r["_draw_index"]),
                ))
            except (ValueError, KeyError) as exc:
                # `Proposal` re-enforces length and budget, so a proposal file that was
                # edited by hand, or written against a different parent, raises here.
                # Skipping it with a note beats taking a GPU job down: the rest of this
                # parent's draws are legitimate and the ladder is a prefix, so the run
                # stays usable up to the gap.
                summary["notes"].append(
                    f"rejected proposal {seq!r} from the round file: {exc}"
                )
                print(f"    [skip] {seq}: {exc}")
        summary["llm_rounds_loaded"] = sorted(
            {p.round_index for p in props}
        )
    else:
        # The STREAM planner, not `standalone_random_arm`: that one draws its whole count
        # vector in a single rng call sized by `n_slots`, so a 48-draw plan and a 96-draw
        # plan share nothing, and extending this run would mean discarding it and starting
        # over. Here draw k is the same sequence at any n_draws, so "use a lot" can be
        # answered by appending draws once the ladder shows where it flattens.
        props, note = standalone_random_stream(
            original=original, n_draws=n_draws, seed_base=parent_id, alphabet=alpha
        )
        if note:
            summary["notes"].append(f"draws: {note}")
    summary["arm"] = draw_arm
    summary["n_draws_realised"] = len(props)
    if llm_proposals is not None and len(props) < n_draws:
        # Said plainly, because it changes what a rung means for this parent: the ladder
        # can only reach B = the number of draws that exist, and a report comparing this
        # parent at B=16 against a random arm at B=16 would be comparing different
        # numbers of candidates.
        summary["notes"].append(
            f"arm D has {len(props)} proposal(s) against the cohort's {n_draws} draws, "
            f"so its ladder cannot reach beyond B={len(props)}. The shortfall is not "
            f"padded with random draws; run another llm-propose round to extend it."
        )

    ref_prop = Proposal(
        sequence=original, arm="reference", round_index=-1, kind=KIND_REFERENCE,
        ancestor_sequence=None, original_sequence=original, edited_positions=(),
        seed=proposal_seed(parent_id, "reference"),
    )
    # The reference is draw -1 and is evaluated first in every tier: it is the comparison
    # every draw is read against, so a run cut off mid-tier must never hold draws at a
    # depth the reference has not reached.
    plan_seqs = [original] + [p.sequence for p in props]
    by_seq: dict[str, tuple[Proposal, str, int]] = {original: (ref_prop, "reference", -1)}
    for i, p in enumerate(props):
        # The draw's index is its SLOT in the stream, not its position in this list.
        # They differ whenever a slot could not be filled with a sequence not already
        # drawn, and using the list position would renumber every later draw -- so the
        # same draw would carry different indices in a 48-draw and a 96-draw run.
        by_seq.setdefault(p.sequence, (p, draw_arm, p.slot if p.slot is not None else i))
    for seq, (p, _arm, idx) in by_seq.items():
        journal.add_candidate({
            "parent_id": parent_id, **p.to_row(),
            "generator": None, "edit_mask": None, "draw_index": idx,
        })

    plan = ladder_plan(plan_seqs, seeds, budgets)
    print(f"  [{parent_id}] {len(props)} draws + reference, L={len(original)}, "
          f"budget={budget_for(len(original))}, {len(plan)} evaluations planned")

    # Predict a whole seed's worth of sequences in one boltz invocation.
    #
    # Measured on an A6000 with these settings: 44 s per invocation plus 26 s per
    # structure, so one structure per invocation spends most of its time on process
    # spawn, the torch import and the 2.2 GB checkpoint read. Grouping the sequences
    # that share a seed takes the per-fold prediction cost from ~70 s to ~31 s.
    #
    # It is a prefetch into the same cache `evaluate` reads, so the loop below is
    # unchanged and anything the batch missed is simply predicted there instead. Grouping
    # is by SEED because the seed is a boltz command-line argument: sequences sharing a
    # seed can share an invocation, sequences sharing a sequence cannot.
    # Set only when the prefetch runs on its own thread; the scoring loop gates on it.
    prefetch_ready: dict[int, threading.Event] | None = None
    if not dry_run and batch_predict:
        by_seed: dict[int, list[str]] = defaultdict(list)
        for _rung, _rank, s_, sd_ in plan:
            arm_ = by_seq[s_][1]
            if not journal.has(s_, sd_, STAGE_LADDER, arm_):
                by_seed[sd_].append(s_)
        def _prefetch_seed(sd_: int) -> None:
            group = by_seed.get(sd_) or []
            if len(group) < 2:
                return          # nothing to amortise over
            t0 = time.time()
            try:
                res = prefetch_predictions(
                    candidate_sequences=group, target=target, prediction_seed=sd_,
                    protocol=protocol, store=store,
                )
            except Exception as e:  # noqa: BLE001
                # Batching is an optimisation and must never be able to end a task. Any
                # failure here leaves the cache as it was, so the loop below predicts
                # each fold itself, exactly as every run before 2026-10-25 did.
                res = {"status": f"batch_error: {type(e).__name__}: {e}"[:200],
                       "requested": len(group), "produced": 0}
                print(f"  [{parent_id}] batch seed={sd_} failed ({res['status']}); "
                      f"falling back to one prediction per fold")
            summary.setdefault("batches", []).append(
                {"seed": sd_, **{k: v for k, v in res.items() if k != "status"},
                 "status": res.get("status")}
            )
            print(f"  [{parent_id}] batch seed={sd_}: {res.get('produced')}"
                  f"/{res.get('requested')} structure(s) in one invocation, "
                  f"{time.time() - t0:.0f}s")
            if res.get("produced", 0) < res.get("requested", 0):
                tail = (res.get("stderr_tail") or "").strip().splitlines()
                for line in tail[-4:]:
                    print(f"      boltz: {line[:200]}")

        if not prefetch_ahead:
            for sd_ in seeds:
                _prefetch_seed(sd_)
        else:
            # Overlap the two phases instead of running them end to end.
            #
            # As written above this loop, a task folds EVERY seed first and only then
            # scores, so the GPU is idle for the whole scoring phase and the CPU for the
            # whole folding phase. Measured on the live corpus run: 51.5 % boltz duty
            # cycle, 157 s/fold end to end. Folding is a subprocess and scoring is
            # single-threaded PyRosetta in this one, so the two genuinely run at once;
            # the thread below folds seed k+1 while the main loop relaxes seed k.
            #
            # Nothing about a result changes. The thread only fills the same prediction
            # cache the sequential loop filled, in the same order, with the same inputs;
            # the scoring loop still reads that cache through `evaluate`. What changes is
            # who is waiting. `evaluate.GPU_LOCK` keeps the thread and any fallback
            # prediction off the GPU at the same time.
            #
            # The main loop gates on the seed it is about to score, so it can never read
            # a half-filled cache -- a miss there is a correctness-neutral fallback, but
            # it would also be an unbatched prediction, which is the cost we are cutting.
            prefetch_ready = {sd_: threading.Event() for sd_ in seeds}

            def _prefetch_worker() -> None:
                try:
                    for sd_ in seeds:
                        try:
                            _prefetch_seed(sd_)
                        finally:
                            # Set before moving on, so the scorer is released seed by
                            # seed rather than at the end.
                            prefetch_ready[sd_].set()
                finally:
                    # A thread that died owes every remaining seed an answer. Without
                    # this the scoring loop waits on an event nobody will ever set and
                    # the task hangs until Slurm kills it -- turning an optimisation
                    # into a silent stall, which is worse than not having it.
                    for ev in prefetch_ready.values():
                        ev.set()

            prefetch_thread = threading.Thread(
                target=_prefetch_worker, name=f"prefetch-{parent_id}", daemon=True
            )
            prefetch_thread.start()
            print(f"  [{parent_id}] prefetch running ahead of scoring "
                  f"({len(seeds)} seed group(s))")

    if predict_only:
        # The predictions are on disk and the candidate rows are in the journal; no
        # outcome row is written, because nothing has been scored and a row claiming
        # otherwise would be a fold that never happened.
        n_pending = sum(
            1 for _r, _k, s_, sd_ in plan
            if not journal.has(s_, sd_, STAGE_LADDER, by_seq[s_][1])
        )
        summary["predict_only"] = True
        summary["folds_left_to_score"] = n_pending
        print(f"  [{parent_id}] predict-only: predictions cached, {n_pending} fold(s) "
              f"left for the scoring pass (re-run without --predict-only)")
        return summary

    done_rung = 0
    consecutive_failures = 0
    for rung, rank, seq, seed in plan:
        proposal, arm, draw_index = by_seq[seq]
        if rung != done_rung:
            done_rung = rung
            print(f"  [{parent_id}] --- ladder rung B={rung} "
                  f"(draws up to {rung}, {len(seeds)} folds each) ---")
        if journal.has(seq, seed, STAGE_LADDER, arm):
            continue
        if dry_run:
            print(f"    [dry] draw={draw_index} seed={seed} (rank {rank}) {seq}")
            continue
        if limit and summary["evaluations"] >= limit:
            if not summary.get("_limit_announced"):
                summary["_limit_announced"] = True
                summary["notes"].append(
                    f"preflight --limit {limit} reached; plan not completed"
                )
                print(f"    [limit] {limit} new evaluation(s) reached; stopping")
            break
        # Wait for this seed's folds, if the prefetch is still working on them. The
        # timeout is a backstop against a wedged prefetch, not a schedule: on expiry the
        # loop proceeds and `evaluate` predicts the fold itself, which is slower but
        # correct. Time parked here is recorded -- it is exactly the GPU-bound remainder
        # that the overlap did not manage to hide, so it is the number that says whether
        # more scoring workers would help.
        if prefetch_ready is not None:
            ev = prefetch_ready.get(seed)
            if ev is not None and not ev.is_set():
                t_wait = time.time()
                if not ev.wait(timeout=protocol.timeout_s):
                    print(f"    [prefetch] seed={seed} not ready after "
                          f"{protocol.timeout_s}s; predicting this fold directly")
                summary["prefetch_wait_s"] = round(
                    summary.get("prefetch_wait_s", 0.0) + time.time() - t_wait, 1
                )

        rec = evaluate(
            original_sequence=original, candidate_sequence=seq, target=target,
            prediction_seed=seed, protocol=protocol, store=store,
        )
        row = outcome_row(
            rec=rec, proposal=proposal, parent_id=parent_id, arm=arm,
            stage=STAGE_LADDER,
            extra={
                # The report truncates BOTH axes to a prefix, so each row has to say
                # where it sits on each: which draw of the stream it is, and which fold
                # of that draw. Recovering either by looking a seed up in a constant
                # would silently mis-order any row written under a different seed list.
                "draw_index": draw_index,
                "seed_rank": rank,
                # The rung this fold was paid for under. Not a property of the fold --
                # the same fold belongs to every rung above it too -- but it records
                # the order the run bought them in, which is what makes a cut-off run
                # legible.
                "budget_rung": rung,
                "n_draws_planned": n_draws,
                "shared_evaluation_with_arms": journal.arms_with_input_hash(
                    rec.get("input_hash"), arm
                ),
            },
        )
        journal.add_outcome(row)
        summary["evaluations"] += 1
        if row.get("cache_prediction") == "hit":
            summary["cache_hits"] += 1
        print(
            f"    draw={draw_index:>3} seed={seed} r{rank:<2} n_mut={row['n_mutations']} "
            f"{seq} -> {'VALID' if row['valid'] else row['status']} "
            f"dG={row.get('best_valid_dG_separated')} "
            f"viol={row.get('min_violation')} {row.get('wall_time_s')}s"
        )

        # Stop on a run of technical failures instead of marching through the plan.
        #
        # When the GPU goes away -- which on a preemptible queue it can, mid-task -- every
        # remaining fold fails in a couple of seconds. Measured 2026-10-23: one task kept
        # going and wrote 72 consecutive failures, so the complex read as 650/650 rows
        # with 578 real folds. The work is resumable, so stopping costs nothing and a
        # requeue picks up where this left off; continuing costs the rest of the plan and
        # hides a dead node behind a full-looking progress line.
        if row.get("technical_failure"):
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                summary["notes"].append(
                    f"stopped after {consecutive_failures} consecutive technical "
                    f"failures (last: {row.get('status')}); the remaining plan was not "
                    f"attempted. Nothing is lost -- these folds are not marked done, so "
                    f"a resubmission retries them."
                )
                print(f"  [{parent_id}] ABORT: {consecutive_failures} consecutive "
                      f"technical failures -- the node or its GPU is unhealthy. "
                      f"Resubmit this shard; finished folds are skipped.")
                break
        else:
            consecutive_failures = 0

    # How far this parent actually got, per depth, so a partial night is legible from the
    # log without rerunning the report.
    have: dict[str, set[int]] = defaultdict(set)
    for o in journal.outcomes:
        if o.get("stage") == STAGE_LADDER and not o.get("technical_failure"):
            have[o["sequence"]].add(o.get("seed_rank", -1))
    depth = len(seeds)
    complete = [s for s in plan_seqs[1:] if len(have.get(s, set())) >= depth]
    # The rung reached: the largest B whose first B draws are ALL folded at full depth.
    # Counting complete draws anywhere in the stream would overstate it -- a ladder rung
    # is a PREFIX of the draws, so a gap at draw 3 caps the ladder at B=2 however many
    # later draws finished.
    reached = 0
    for i, sq in enumerate(plan_seqs[1:], start=1):
        if len(have.get(sq, set())) < depth:
            break
        reached = i
    summary["n_draws_complete_at_full_depth"] = len(complete)
    summary["ladder_rung_reached"] = max(
        [b for b in budgets if b <= reached], default=0
    )
    summary["reference_complete_at_full_depth"] = len(have.get(original, set())) >= depth
    return summary


def cmd_ladder(args) -> int:
    out_dir = Path(args.out_dir)
    cohort_path = out_dir / "cohort.json"
    if not cohort_path.exists():
        print(f"[ladder] no cohort at {cohort_path} -- run ladder-cohort first")
        return 1
    cohort = json.loads(cohort_path.read_text())
    if cohort.get("kind") != "seed_ladder":
        print(f"[ladder] {cohort_path} is not a seed-ladder cohort")
        return 1

    parents = cohort["cohort"][args.shard_index :: args.shard_count]
    if args.parents:
        parents = [p.strip() for p in args.parents.split(",") if p.strip()]

    seeds = tuple(cohort["seeds"])
    budgets = tuple(cohort["budget_rungs"])
    n_draws = int(cohort["n_draws"])

    # The thresholds this shard will apply must be the ones the cohort was built against.
    #
    # They are a per-corpus FILE while the version label written on every row is a module
    # constant, so the two can disagree without anything failing. On 2026-10-25 that was
    # found to have happened: `seed_ladder` scored 12,000 folds against `tranche1`'s
    # pre-correction thresholds (trans-only planarity 8.44 deg, bond window 1.1-1.6 A)
    # while every row claimed `v3-cis-amide-muco-windows`. Nothing crashed, nothing
    # looked wrong, and the absolute pass rate it produced was not the one its label
    # described. Checking the recorded hash is cheap and catches exactly that, including
    # the case where someone updates a corpus's thresholds while a run is in flight.
    recorded = cohort.get("geometry_tolerances_sha256")
    tol_path = Path(cohort["dataset_dir"]) / "manifest" / "geometry_tolerances.json"
    if recorded:
        if not tol_path.exists():
            print(f"[ladder] thresholds missing: {tol_path}")
            return 1
        actual = gate_sha256(tol_path)
        if actual != recorded:
            print(f"[ladder] REFUSING: the acceptance thresholds have changed since this "
                  f"cohort was built.")
            print(f"    cohort recorded {recorded[:16]}")
            print(f"    {tol_path} is now {actual[:16]}")
            print(f"    Folding now would mix two gates under one version label. Either "
                  f"restore the recorded thresholds, or build a new cohort (and a new "
                  f"--out-dir) against the new ones and re-score what exists.")
            return 1
    gate = recorded or gate_sha256(tol_path)

    protocol = Protocol(
        chemistry=args.chemistry,
        diffusion_samples=args.samples,
        boltz_python=args.boltz_python,
        boltz_cache=args.boltz_cache,
        allow_cys=bool(cohort.get("allow_cys", False)),
        # From the cohort, not a flag: a shard that disagreed with its own cohort about
        # the inference settings would write folds that are not comparable to the rest
        # and nothing downstream would notice.
        use_potentials=bool(cohort.get("use_potentials", False)),
    )
    sources = cohort["parent_sources"]
    store = Store(
        root=out_dir,
        tolerances_path=Path(cohort["dataset_dir"]) / "manifest"
        / "geometry_tolerances.json",
        retention=args.retention,
    )

    # Contexts come from each parent's own corpus. Grouped so a corpus manifest is read
    # once rather than once per parent.
    contexts: dict[str, TargetContext] = {}
    by_dir: dict[str, list[str]] = defaultdict(list)
    for p in parents:
        if p not in sources:
            print(f"[ladder] {p}: not in the cohort's parent_sources -- skipped")
            continue
        by_dir[sources[p]].append(p)
    for ddir, pids in by_dir.items():
        contexts |= load_target_contexts(Path(ddir), pids)

    llm_round = int(getattr(args, "llm_round", 0) or 0)
    arm_label = ARM_LLM if llm_round else ARM_RANDOM
    print(f"[ladder] {len(parents)} parent(s) shard {args.shard_index}/{args.shard_count}")
    print(f"[ladder] arm={arm_label}, {n_draws} draws x {len(seeds)} folds, "
          f"ladder rungs B={list(budgets)}")
    if llm_round:
        print(f"[ladder] folding LLM proposals from rounds 1..{llm_round} under "
              f"{out_dir / 'llm_proposals'}; the random draw stream is not used")
    print(f"[ladder] protocol={protocol.to_dict()} retention={args.retention}")

    summaries = []
    for pid in parents:
        if pid not in contexts:
            print(f"[ladder] {pid}: no target context -- skipped")
            continue
        llm_props = None
        if llm_round:
            llm_props, load_notes = load_llm_proposals(
                out_dir, pid, through_round=llm_round
            )
            for n in load_notes:
                print(f"  [{pid}] proposals: {n}")
            if not llm_props:
                # Not folded as a random arm instead: silently substituting the control's
                # draws for a missing LLM plan would write rows labelled arm D that the
                # model never proposed.
                print(f"  [{pid}] no LLM proposals for rounds 1..{llm_round} -- "
                      f"skipped. Run cpseq_llm_propose.py for this round first.")
                continue
        s = run_parent_ladder(
            parent_id=pid, target=contexts[pid], out_dir=out_dir, protocol=protocol,
            store=store, n_draws=n_draws, seeds=seeds, budgets=budgets,
            dry_run=args.dry_run, limit=args.limit,
            batch_predict=not args.no_batch_predict,
            prefetch_ahead=args.prefetch_ahead, llm_proposals=llm_props,
            predict_only=getattr(args, "predict_only", False),
        )
        summaries.append(s)
        if s.get("predict_only"):
            # A predict-only pass has scored nothing, so the ladder-depth counters do
            # not exist yet. Reporting them as 0 would read as a run that folded and
            # found nothing, which is the opposite of what happened.
            print(f"  [{pid}] predictions ready, "
                  f"{s.get('folds_left_to_score', 0)} fold(s) await the scoring pass")
        else:
            print(f"  [{pid}] {s['evaluations']} new fold(s), "
                  f"{s['cache_hits']} cache hit(s), "
                  f"{s['n_draws_complete_at_full_depth']} draw(s) at full depth, "
                  f"ladder reaches B={s['ladder_rung_reached']}")
        for n in s["notes"]:
            print(f"  [{pid}] note: {n}")

    (out_dir / f"ladder_summary_{args.shard_index}.json").write_text(
        json.dumps([{k: v for k, v in s.items() if not k.startswith("_")}
                    for s in summaries], indent=2)
    )
    return 0


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def cmd_report(args) -> int:
    out_dir = Path(args.out_dir)
    cohort = json.loads((out_dir / "cohort.json").read_text())
    rows: list[dict] = []
    cands: list[dict] = []
    for pid in cohort["cohort"]:
        j = Journal(out_dir, pid)
        rows += j.outcomes
        if j.candidates_path.exists():
            for line in j.candidates_path.read_text().splitlines():
                try:
                    cands.append(json.loads(line))
                except Exception:  # noqa: BLE001, S112
                    continue
    if not rows:
        print("[report] no outcomes yet")
        return 1

    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Everything on disk, before any filtering: the full journal stays available as a
    # table even when the report below reads one protocol.
    df.to_csv(tables / "outcomes.csv", index=False)

    # One scoring protocol per report. A rescore appends rows rather than rewriting
    # history, so after a protocol change the same fold is on disk twice under two
    # versions -- and the versions disagree about validity, which is the whole reason a
    # version was bumped. Pooling them would count rescored folds twice and average a
    # validity label across two different scorers. Rows predating the field carry no
    # version and are kept.
    if "scoring_version" in df.columns:
        want = args.scoring_version or SCORING_PROTOCOL_VERSION
        present = sorted(v for v in df.scoring_version.dropna().unique())
        if want == "all":
            print(f"[report] pooling scoring versions {present} as asked -- a fold "
                  f"re-scored under more than one is counted once per version")
        else:
            keep = df.scoring_version.isna() | (df.scoring_version == want)
            if len(present) > 1 or (present and present[0] != want):
                print(f"[report] scoring versions on disk: {present}")
                print(f"[report] reporting {want} only: {int(keep.sum())} of {len(df)} "
                      f"row(s). Pass --scoring-version all to pool, or name another.")
            df = df[keep]
            if df.empty:
                print(f"[report] no rows at scoring version {want}")
                return 1
    if cands:
        pd.DataFrame(cands).to_csv(tables / "candidates.csv", index=False)

    screen = df[df.stage == "screen"]

    # Enforce the per-arm budget at REPORT time as well as at evaluation time. A task
    # requeued before the budget fix bought extra slots for its arm, and those rows are
    # on disk; comparing a 36-evaluation arm against a 32-evaluation arm would make the
    # comparison unmatched. The first N in journal order are kept -- journal order is
    # proposal order, so this keeps the plan's own prefix rather than the best rows --
    # and the surplus is counted and reported, never silently dropped.
    keep_idx: list = []
    surplus: dict[tuple[str, str], int] = {}
    for (pid, arm), g in screen.groupby(["parent_id", "arm"]):
        cap = ARM_BUDGETS.get(arm)
        if cap is None or len(g) <= cap:
            keep_idx += list(g.index)
            continue
        keep_idx += list(g.index[:cap])
        surplus[(pid, arm)] = len(g) - cap
    screen = screen.loc[sorted(keep_idx)]
    if surplus:
        print("\n=== budget enforcement (surplus evaluations excluded from the "
              "matched comparison, retained on disk) ===")
        for (pid, arm), n in sorted(surplus.items()):
            print(f"  {pid:12s} {arm:16s} +{n} over budget {ARM_BUDGETS[arm]} "
                  f"(requeue overspend)")

    per_parent = []
    for pid, g in screen.groupby("parent_id"):
        ref = g[g.arm == "reference"]
        ref_dg = ref.best_valid_dG_separated.dropna()
        ref_best = ref_dg.min() if len(ref_dg) else None
        for arm, ga in g[g.arm != "reference"].groupby("arm"):
            # Technical failures leave the yield denominator, since a crashed job is not
            # a nonbinding label; they are reported in their own column instead.
            tech = int(ga.technical_failure.sum())
            evaluable = ga[~ga.technical_failure.astype(bool)]
            valid = evaluable[evaluable.valid.astype(bool)]
            dgs = valid.best_valid_dG_separated.dropna()
            per_parent.append({
                "parent_id": pid,
                "arm": arm,
                "n_screened": len(ga),
                "n_technical_failures": tech,
                "n_evaluable": len(evaluable),
                "n_valid": len(valid),
                "valid_yield": round(len(valid) / len(evaluable), 4) if len(evaluable) else None,
                "best_valid_dG": round(float(dgs.min()), 3) if len(dgs) else None,
                # The stated objective: the average over valid folds. Reported beside
                # the best and the median because a best-of-N column is a record of
                # which fold got lucky, and the three disagreeing is itself the signal.
                "mean_valid_dG": round(float(dgs.mean()), 3) if len(dgs) else None,
                "median_valid_dG": round(float(dgs.median()), 3) if len(dgs) else None,
                "reference_best_dG": round(float(ref_best), 3) if ref_best is not None else None,
                "delta_vs_reference": (
                    round(float(dgs.min() - ref_best), 3)
                    if len(dgs) and ref_best is not None else None
                ),
                "median_n_mutations": float(ga.n_mutations.median()),
                "gpu_wall_time_s": round(float(ga.wall_time_s.fillna(0).sum()), 1),
                "n_cache_hits": int((ga.cache_prediction == "hit").sum()),
            })
    # A ladder run has no `screen` stage at all -- its folds are stage `ladder`, read by
    # `cpseq_ladder_report.py` -- so this table is legitimately empty there. Sorting an
    # empty frame by a column it never got raises KeyError, which read as a broken run
    # rather than as an absent stage.
    if per_parent:
        pp = pd.DataFrame(per_parent).sort_values(["parent_id", "arm"])
        pp.to_csv(tables / "per_parent_arm.csv", index=False)
        print("\n=== per parent x arm (screening) ===")
        print(pp.to_string(index=False))
    else:
        stages = sorted(df.stage.dropna().unique()) if "stage" in df.columns else []
        print(f"\n=== per parent x arm (screening) ===\n  no `screen` rows; "
              f"stages present: {stages}. For a ladder run use cpseq_ladder_report.py.")

    conf = df[df.stage == "confirm"]
    if len(conf):
        crows = []
        for (pid, seq), g in conf.groupby(["parent_id", "sequence"]):
            # Invalid seeds stay in the denominator: never report only the best seed.
            dgs = g[g.valid.astype(bool)].best_valid_dG_separated.dropna()
            crows.append({
                "parent_id": pid,
                "sequence": seq,
                "arm": g.screened_arm.dropna().iloc[0] if g.screened_arm.notna().any() else g.arm.iloc[0],
                "n_seeds": len(g),
                "n_valid_seeds": int(g.valid.sum()),
                "validity_frequency": round(float(g.valid.mean()), 4),
                "best_valid_dG": round(float(dgs.min()), 3) if len(dgs) else None,
                # What the plan reports for a selected sequence: the average dG among
                # its valid folds, read next to `n_valid_seeds`/`n_seeds`. The pair is
                # the claim -- an average over 1 of 6 valid folds is not the same result
                # as the same average over 6 of 6, and no minimum ratio is imposed, so
                # the denominator has to travel with the number.
                "mean_valid_dG": round(float(dgs.mean()), 3) if len(dgs) else None,
                "median_valid_dG": round(float(dgs.median()), 3) if len(dgs) else None,
                "n_mutations": int(g.n_mutations.iloc[0]) if g.n_mutations.notna().any() else None,
            })
        cdf = pd.DataFrame(crows).sort_values(["parent_id", "best_valid_dG"])
        cdf.to_csv(tables / "confirmation.csv", index=False)
        print("\n=== confirmation (fresh seeds) ===")
        print(cdf.to_string(index=False))

    total = {
        "n_outcomes": len(df),
        "n_screen": int((df.stage == "screen").sum()),
        "n_screen_in_budget": int(len(screen)),
        "budget_surplus_excluded": {f"{p}|{a}": n for (p, a), n in surplus.items()},
        "arm_screening_budget": ARM_BUDGETS,
        "n_confirm": int((df.stage == "confirm").sum()),
        "n_technical_failures": int(df.technical_failure.sum()),
        "n_prediction_cache_hits": int((df.cache_prediction == "hit").sum()),
        "wall_time_hours": round(float(df.wall_time_s.fillna(0).sum()) / 3600, 3),
        "scoring_version": SCORING_PROTOCOL_VERSION,
    }
    (out_dir / "tables" / "totals.json").write_text(json.dumps(total, indent=2))
    print(f"\n=== totals ===\n{json.dumps(total, indent=2)}")
    print(f"\n[report] tables -> {tables}")
    return 0


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("select-cohort")
    c.add_argument("--dataset-dir", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--n-parents", type=int, default=6)
    c.add_argument("--parents", default="", help="comma separated, overrides selection")
    c.add_argument("--all-targets", action="store_true",
                   help="take every parent in manifest/targets.csv and skip the "
                        "validity stratification. For a corpus with no prior "
                        "scored tranche (PepBench), where the whole prepared set "
                        "is the cohort.")
    c.add_argument("--allow-cys", action="store_true")
    c.set_defaults(func=cmd_select_cohort)

    for name, fn in (("screen", cmd_screen), ("confirm", cmd_confirm)):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", required=True)
        s.add_argument("--shard-index", type=int, default=0)
        s.add_argument("--shard-count", type=int, default=1)
        s.add_argument("--parents", default="")
        s.add_argument("--chemistry", default="mainchain")
        s.add_argument("--samples", type=int, default=1)
        s.add_argument("--arms", default=f"{ARM_GUIDED},{ARM_RANDOM}")
        s.add_argument("--retention", default="full", choices=RETENTION_TIERS,
                       help="what to keep per prediction once scored. full (default) "
                            "retains Boltz's whole working directory, ~1.31 MB/candidate "
                            "measured. rescorable drops inputs and already-summarised "
                            "matrices and gzips the raw complex (~50 KB, 26x), keeping "
                            "rescore-without-re-predict. deliverable keeps only the "
                            "canonical-frame peptide and the score row (~5 KB, 263x), at "
                            "the cost of needing a re-prediction if the scoring protocol "
                            "changes.")
        s.add_argument("--random-policy", default="match_guided",
                       choices=("match_guided", "standalone"),
                       help="match_guided (default) keeps arm A defined by arm B's "
                            "realised substitution counts, which is what makes the two "
                            "arms comparable and what the pilot measured. standalone "
                            "draws counts from a frozen distribution instead, so a "
                            "random-only run needs no guided arm; see "
                            "search.standalone_counts for the evidence that this costs "
                            "nothing measurable.")
        s.add_argument("--boltz-python",
                       default="/data/user_data/yixiz/envs/boltz2/.venv/bin/python")
        s.add_argument("--boltz-cache", default="/data/user_data/yixiz/boltz_cache")
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--proposals", default="",
                       help="comma-separated jsonl paths/globs from cpseq_mpnn_propose.py "
                            "(arm C). Required for arms containing backbone_mpnn.")
        s.add_argument("--limit", type=int, default=0,
                       help="cap NEW evaluations per parent; 0 = no cap (preflight uses 2-3)")
        if name == "confirm":
            s.add_argument("--seeds", default="",
                           help="comma-separated prediction seeds, overriding CONFIRM_SEEDS. "
                                "The journal dedupes on (sequence, seed, stage, arm), so a "
                                "superset of an earlier run only pays for the new seeds.")
        s.set_defaults(func=fn)

    lc = sub.add_parser(
        "ladder-cohort",
        help="cohort for the seed ladder: several corpora in one run, each sampled by "
             "a recorded rule, each complex keeping the corpus it was prepared in",
    )
    lc.add_argument("--out-dir", required=True)
    lc.add_argument("--corpus", action="append", default=[], required=True,
                    help="repeatable. `dataset_dir=<path>,n=<complexes>,label=<name>`, "
                         "e.g. dataset_dir=$ZFS/cpseq/tranche1,n=10,label=LNR")
    lc.add_argument("--draws", type=int, default=LADDER_DRAWS,
                    help=f"random draws per complex (default {LADDER_DRAWS}). Draw k is "
                         f"the same sequence whatever this is set to, so a finished run "
                         f"extends by re-running with a larger value instead of being "
                         f"replaced by one.")
    lc.add_argument("--seed-depth", type=int, default=len(LADDER_SEEDS),
                    help=f"how many of the {len(LADDER_SEEDS)} seeds every draw is "
                         f"folded at. This is the number that decides whether a "
                         f"per-candidate result is a measurement or a coin flip.")
    lc.add_argument("--allow-cys", action="store_true")
    lc.add_argument("--use-potentials", action="store_true",
                    help="enable Boltz-2's physical-plausibility steering for every fold "
                         "of this cohort. Off everywhere else in this project, which is "
                         "why peptide-receptor overlap is the largest single rejector "
                         "(README_CPSEQ_PEPBENCH_SCALEUP sec. 11). It changes the "
                         "structures themselves, so such a run needs its own --out-dir: "
                         "the journal key carries no inference settings and would treat "
                         "the other setting's folds as already paid for.")
    lc.set_defaults(func=cmd_ladder_cohort)

    lr = sub.add_parser(
        "ladder",
        help="fold every draw, and the unmodified peptide, at every seed in the "
             "cohort's seed set, shallowest tier first",
    )
    lr.add_argument("--out-dir", required=True)
    lr.add_argument("--shard-index", type=int, default=0)
    lr.add_argument("--shard-count", type=int, default=1)
    lr.add_argument("--parents", default="")
    lr.add_argument("--chemistry", default="mainchain")
    lr.add_argument("--samples", type=int, default=1)
    lr.add_argument("--retention", default="rescorable", choices=RETENTION_TIERS,
                    help="what to keep per fold. Defaults to rescorable rather than "
                         "full: ten folds per draw is ten times the structures, and "
                         "rescorable (~50 KB each, measured) still allows a re-score "
                         "without re-folding, which is the property worth keeping.")
    lr.add_argument("--boltz-python",
                    default="/data/user_data/yixiz/envs/boltz2/.venv/bin/python")
    lr.add_argument("--boltz-cache", default="/data/user_data/yixiz/boltz_cache")
    lr.add_argument("--dry-run", action="store_true")
    lr.add_argument("--limit", type=int, default=0,
                    help="cap NEW folds per complex; 0 = no cap (preflight uses 2-3)")
    lr.add_argument("--no-batch-predict", action="store_true",
                    help="predict one structure per boltz invocation, as every run "
                         "before 2026-10-25 did. Batching the sequences that share a "
                         "seed is a prefetch into the same cache and changes no result, "
                         "but it is the difference between ~70 s and ~31 s of prediction "
                         "per fold; this flag exists to isolate it if a batch misbehaves.")
    lr.add_argument("--predict-only", action="store_true",
                    help="run the Boltz-2 predictions and stop, writing no outcome rows. "
                         "Splits this command into a GPU pass and a CPU pass: measured "
                         "14.3 s of prediction against 43.7 s of relax per fold, so "
                         "three quarters of a 'GPU' task's wall time was Rosetta with "
                         "the card idle. Score the same plan afterwards by re-running "
                         "the IDENTICAL command without this flag -- every prediction is "
                         "then a cache hit, so the scoring pass needs no GPU and belongs "
                         "under preempt_cpu_qos. Changes no result, because both passes "
                         "key the same caches the same way. Prefer this over "
                         "--prefetch-ahead, which only overlaps the two phases inside "
                         "one GPU job rather than letting them be scheduled apart.")
    lr.add_argument("--prefetch-ahead", action="store_true",
                    help="fold the next seed on a background thread while this one is "
                         "being scored. A task currently folds every seed and only then "
                         "scores, so the GPU idles through the whole scoring phase: "
                         "measured 51.5%% boltz duty cycle and 157 s/fold on the live "
                         "corpus run. Folding is a subprocess and scoring is "
                         "single-threaded PyRosetta, so they overlap for real. Fills the "
                         "same cache in the same order with the same inputs, so no "
                         "result changes -- only who is waiting. Expect the task to "
                         "become CPU-bound, at which point more scoring workers, not "
                         "faster folding, is the next lever.")
    lr.add_argument("--llm-round", type=int, default=0,
                    help="fold arm D instead of the random stream: use the LLM "
                         "proposals from rounds 1..N written by cpseq_llm_propose.py "
                         "under <out-dir>/llm_proposals/. Rounds are concatenated in "
                         "order, so round 1's batch is the ladder's first draws and a "
                         "later round extends the prefix rather than renumbering it. "
                         "Everything else -- fold depth, gate, evaluator, report -- is "
                         "unchanged, which is what makes the two arms comparable.")
    lr.set_defaults(func=cmd_ladder)

    r = sub.add_parser("report")
    r.add_argument("--scoring-version", default="",
                   help="which scoring protocol to report. Default is the current one "
                        f"({SCORING_PROTOCOL_VERSION}); `all` pools every version on "
                        "disk, which double-counts any fold that was re-scored.")
    r.add_argument("--out-dir", required=True)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
