#!/usr/bin/env python3
"""Arm D's proposal stage: an LLM reads the interfaces and proposes the next sequences.

This is one round of proposals for every parent in a cohort, written to disk. It does not
fold or score anything. The fold stage (`cpseq_search.py ladder --llm-proposals ...`)
reads what this writes and evaluates it through the IDENTICAL path as every other arm --
the same Boltz-2 adapter, the same frozen geometry gate, the same fold depth, the same
cross-fitted report. Only the origin of the sequences differs, which is the whole point.

Why a separate stage rather than a call inside the fold loop
-----------------------------------------------------------
Llama-3.1-8B in bf16 holds ~17 GB of GPU memory (measured on an A6000: 20 GB of other
work plus the model reached 36.9 GB of 48 GB). Boltz-2 needs the GPU too, and the fold
loop runs it as a subprocess, so a resident LLM would be competing with the thing it
exists to serve. Keeping them in separate jobs means one model load serves every parent
in a round and the GPU is free again before any folding starts. It is also the pattern
arm C already uses (`cpseq_mpnn_propose.py`), for the same reason.

The alternating shape of a run
------------------------------
    llm-propose --round 1   ->  ladder --llm-round 1   ->  llm-propose --round 2  -> ...

Each propose stage sees the folds the previous ladder stage paid for. Rounds are what
make the proposer adaptive; the random control it is compared against is non-adaptive by
construction, which is the difference being measured.

The cross-fit guard
-------------------
`--feedback-folds select` (the default) shows the model only folds `0 .. depth/2`, the
half the report is allowed to select on. The other half measures, and the proposer never
sees it. This is not a conservatism: without it an adaptive proposer chooses what to
propose using the folds its proposals are later scored on, and the held-out column stops
being held out. `--feedback-folds all` exists so the size of that leak can be measured
rather than assumed, and it is never the setting a reported number comes from.

Usage:
    python script_utils/cpseq_llm_propose.py --out-dir $ZFS/cpseq/llm_ladder --round 1
    python script_utils/cpseq_llm_propose.py --out-dir $ZFS/cpseq/llm_ladder --round 2 \
        --batch-size 8 --interfaces-best 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.evaluate import (  # noqa: E402
    TargetContext,
    load_target_contexts,
)
from proteinfoundation.cpseq.interactions import summarize_complex  # noqa: E402
from proteinfoundation.cpseq.llm_propose import (  # noqa: E402
    DEFAULT_ELITE_MIN_DISTANCE,
    DEFAULT_HISTORY_BEST,
    DEFAULT_HISTORY_WORST,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL_PATH,
    DEFAULT_N_ELITES,
    DEFAULT_OFFSPRING_PER_ELITE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LlamaProposer,
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FORMATS,
    history_from_outcomes,
    shuffle_outcome_associations,
    propose_generation,
    round_diversity,
    select_elites,
    write_record,
)
from proteinfoundation.cpseq.search import (  # noqa: E402
    ARM_LLM,
    budget_for,
    default_alphabet,
)

STAGE_LADDER = "ladder"


def proposals_dir(out_dir: Path, round_index: int) -> Path:
    return out_dir / "llm_proposals" / f"round{round_index}"


def load_journal(out_dir: Path, parent_id: str) -> list[dict]:
    p = out_dir / "state" / parent_id / "outcomes.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001, S112
            continue
    return rows


def hash_parent(parent_id: str) -> int:
    """A stable per-target offset for the shuffled-history control.

    `hash()` is randomised per process, so it would give a different shuffle on every
    re-run of the same command -- which for a control arm means the arm is not
    reproducible and its result cannot be checked.
    """
    import hashlib

    return int(hashlib.sha1(parent_id.encode()).hexdigest()[:8], 16) % 100000


def cached_interface(
    out_dir: Path,
    parent_id: str,
    row: dict,
    *,
    sequence: str,
    force: bool = False,
) -> dict | None:
    """One complex's interface summary, computed once and kept.

    Keyed on `input_hash`, the same identity the prediction cache uses, so the summary
    belongs to exactly one structure. Cached for two reasons: round 2 should not recompute
    round 1's summaries, and `retention.py` will eventually delete the structure this was
    derived from -- after which the cached summary is the only surviving description of
    that interface.
    """
    ihash = row.get("input_hash")
    if not ihash:
        return None
    dest = out_dir / "interfaces" / parent_id / f"{ihash}.json"
    if dest.exists() and not force:
        try:
            return json.loads(dest.read_text())
        except Exception:  # noqa: BLE001
            pass
    # The RAW complex, not the refined one: the gate judges raw, and relax repairs the
    # overlaps the proposer most needs to see. See `interactions.summarize_complex`.
    raw = row.get("raw_pdb_path")
    if not raw:
        return None
    s = summarize_complex(raw, peptide_sequence=sequence, structure_stage="raw")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(s))
    return s


def best_row_for(rows: list[dict], sequence: str) -> dict | None:
    """The row whose structure best represents this sequence.

    A valid fold with the best energy if there is one, else the smallest violation. The
    representative fold must not be a technical failure: a crashed fold has no structure
    to summarise and its absence says nothing about the sequence.
    """
    mine = [
        r for r in rows
        if r.get("sequence") == sequence
        and not r.get("technical_failure")
        and r.get("raw_pdb_path")
    ]
    if not mine:
        return None
    valid = [r for r in mine if r.get("valid")]
    if valid:
        with_e = [r for r in valid if r.get("best_valid_dG_separated") is not None]
        if with_e:
            return min(with_e, key=lambda r: r["best_valid_dG_separated"])
        return valid[0]
    with_v = [r for r in mine if r.get("min_violation") is not None]
    if with_v:
        return min(with_v, key=lambda r: r["min_violation"])
    return mine[0]


def pocket_note_for(target: TargetContext) -> str:
    """The conditioning pocket, named for the prompt.

    This is the pocket the predictor was conditioned on, taken from the corpus manifest
    rather than measured from a structure, so it is the same for every candidate and is
    available even when no fold has succeeded yet.
    """
    contacts = getattr(target, "pocket_contacts", None) or []
    if not contacts:
        return ""
    items = []
    for c in contacts[:24]:
        try:
            ch, idx = c[0], c[1]
            items.append(f"{ch}:{idx}")
        except Exception:  # noqa: BLE001, S112
            continue
    if not items:
        return ""
    return (
        "The structure predictor was conditioned on this pocket (receptor "
        "chain:residue): " + ", ".join(items)
    )


def run_parent(
    *,
    shard_index: int,
    parent_id: str,
    target: TargetContext,
    out_dir: Path,
    proposer: LlamaProposer,
    round_index: int,
    batch_size: int,
    n_elites: int,
    offspring_per_elite: int,
    elite_min_distance: int,
    max_fold_rank: int | None,
    interfaces_best: int,
    interfaces_worst: int,
    history_best: int,
    history_worst: int,
    max_attempts: int,
    feedback_note: str,
    dry_run: bool,
    shuffle_history_seed: int | None = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> dict:
    original = target.reference_sequence
    budget = budget_for(len(original))
    rows = load_journal(out_dir, parent_id)
    alpha = default_alphabet(allow_cys=False)

    history = history_from_outcomes(
        rows, stage=STAGE_LADDER, max_fold_rank=max_fold_rank
    )
    # Every sequence the run has evaluated, under ANY arm, plus the original. Blocking
    # across arms is deliberate: a sequence the random control already folded has its
    # measurement half on disk, and an LLM proposal of the same string would be scored
    # from that cache rather than measured afresh.
    already = {r.get("sequence") for r in rows if r.get("sequence")} | {original}

    # The reference interface: the baseline the model is asked to beat.
    ref_row = next(
        (
            r for r in rows
            if r.get("arm") == "reference" and not r.get("technical_failure")
            and r.get("raw_pdb_path")
        ),
        None,
    )
    ref_summary = (
        cached_interface(out_dir, parent_id, ref_row, sequence=original)
        if ref_row else None
    )

    # Full interface tables for the best few scored candidates, and for one rejected
    # candidate -- the rejected one is often the more instructive, since it shows the
    # overlap that caused the rejection.
    passed = sorted(
        [h for h in history if h.valid and h.energy is not None], key=lambda h: h.energy
    )
    failed = sorted(
        [h for h in history if not h.valid],
        key=lambda h: (h.violation if h.violation is not None else 1e9),
    )
    # The elites this generation breeds from, chosen on the SELECTION half only -- the
    # same cross-fit guard the prompt history obeys, for the same reason: elites picked
    # on the measurement folds would make the held-out column a function of the folds it
    # is supposed to hold out.
    elites = select_elites(
        history, n_elites=n_elites, min_distance=elite_min_distance
    )

    # Interface tables for the candidates the model will actually be shown in full: the
    # elites first, since each one is the sequence a prompt asks to improve, then the
    # best and worst of the rest. A rejected candidate's table is often the more
    # instructive, since it shows the overlap that caused the rejection.
    elite_seqs = {e.sequence for e in elites}
    show = list(elites)
    show += [h for h in passed[:interfaces_best] if h.sequence not in elite_seqs]
    show += [h for h in failed[:interfaces_worst] if h.sequence not in elite_seqs]
    n_iface = 0
    for h in show:
        row = best_row_for(rows, h.sequence)
        if row is None:
            continue
        h.interaction_summary = cached_interface(
            out_dir, parent_id, row, sequence=h.sequence
        )
        if h.interaction_summary and h.interaction_summary.get("ok"):
            n_iface += 1

    info = {
        "parent_id": parent_id,
        "round_index": round_index,
        "original_sequence": original,
        "peptide_length": len(original),
        "budget": budget,
        "n_requested": batch_size,
        "n_history": len(history),
        "n_history_passed": len(passed),
        "n_interfaces_shown": n_iface,
        "reference_interface": bool(ref_summary and ref_summary.get("ok")),
        "n_already_evaluated": len(already),
        "max_fold_rank_shown": max_fold_rank,
        "n_elites": len(elites),
        "offspring_per_elite": offspring_per_elite,
        "elites": [
            {
                "rank": r,
                "sequence": e.sequence,
                "n_mutations": e.n_mutations,
                "valid": e.valid,
                "mean_valid_dG_separated": e.energy,
                "n_valid_folds": e.n_valid_folds,
                "n_folds": e.n_folds,
            }
            for r, e in enumerate(elites)
        ],
    }
    if elites:
        print(f"  [{parent_id}] {len(elites)} elite(s) x {offspring_per_elite} "
              f"offspring:")
        for r, e in enumerate(elites):
            dg = f"{e.energy:+.1f}" if e.energy is not None else "n/a"
            print(f"      [{r}] {e.sequence}  {e.n_mutations} sub  "
                  f"mean dG {dg} REU over {e.n_valid_folds}/{e.n_folds} valid fold(s)"
                  f"{'' if e.valid else '  (INFEASIBLE -- carried for its geometry)'}")
    else:
        print(f"  [{parent_id}] no elite yet, so this generation seeds from the "
              f"original")
    if dry_run:
        print(f"  [{parent_id}] [dry] would request {batch_size}; "
              f"history={len(history)} ({len(passed)} passed), interfaces={n_iface}, "
              f"ref_iface={info['reference_interface']}")
        info["dry_run"] = True
        return info

    # The shuffled-history control. Elites are chosen from the REAL history above, and
    # each elite's own standing stays real, so this arm breeds from the same parents as
    # the live one and differs only in the sequence-to-outcome mapping the prompt shows.
    # Shuffling before elite selection would make it a different search rather than a
    # different prompt, and the comparison would no longer isolate the history.
    prompt_history = history
    if shuffle_history_seed is not None:
        prompt_history = shuffle_outcome_associations(
            history, seed=shuffle_history_seed + hash_parent(parent_id)
        )
        print(f"  [{parent_id}] CONTROL ARM: history outcomes shuffled "
              f"({len(prompt_history)} candidate(s)); this arm's proposals are NOT a "
              f"result, they are the null for whether the history is read")

    props, record = propose_generation(
        proposer=proposer,
        original=original,
        budget=budget,
        round_index=round_index,
        parent_id=parent_id,
        already=already,
        elites=elites,
        offspring_per_elite=offspring_per_elite,
        # The generation's total budget, spent whether or not `n_elites` elites exist.
        # Keeping it equal to the random arm's draws per round is what makes the two
        # arms comparable; letting it shrink with the elite count silently ran this
        # arm at 75% of its control.
        generation_size=batch_size,
        reference_summary=ref_summary,
        history=prompt_history,
        alphabet=alpha,
        pdb_id=target.pdb_id,
        pocket_note=pocket_note_for(target),
        feedback_note=feedback_note,
        max_attempts=max_attempts,
        prompt_dir=proposals_dir(out_dir, round_index) / "prompts",
        output_format=output_format,
    )
    diversity = round_diversity([p.sequence for p in props], original)
    record |= {
        "substitution_budget": budget,
        "n_interfaces_shown": n_iface,
        "reference_interface": info["reference_interface"],
        "max_fold_rank_shown": max_fold_rank,
        "diversity": diversity,
    }

    dest = proposals_dir(out_dir, round_index)
    dest.mkdir(parents=True, exist_ok=True)
    # The proposal rows the fold stage consumes. Overwritten rather than appended, so a
    # re-run of this round replaces its plan instead of growing it -- the fold stage
    # dedupes on the journal anyway, but a doubled file would misstate the round's size.
    with open(dest / f"{parent_id}.jsonl", "w") as fh:
        for p in props:
            fh.write(json.dumps({
                "parent_id": parent_id,
                "arm": ARM_LLM,
                **p.to_row(),
                "llm_model_id": proposer.config["model_id"],
                "llm_round": round_index,
            }) + "\n")
    # Per-shard, because the propose stage is sharded across nodes and concurrent
    # appends to one file over NFS are not atomic -- a torn line would corrupt the
    # audit trail this arm depends on being reproducible from.
    write_record(dest / f"records_shard{shard_index}.jsonl", record)

    # Summed over the generation's per-elite rounds, because that is where the audit
    # trail now lives. A generation that spent all its attempts on one elite and none on
    # the others has to be visible as a per-elite shortfall, not as one aggregate.
    sub = record.get("rounds") or []
    n_rej = sum(len(r.get("rejections") or []) for r in sub)
    n_mal = sum(len(r.get("malformed") or []) for r in sub)
    n_att = sum(len(r.get("attempts") or []) for r in sub)
    info |= {
        "n_requested": record.get("n_requested", batch_size),
        "n_accepted": len(props),
        "n_rejected": n_rej,
        "n_malformed": n_mal,
        "n_attempts": n_att,
        "note": record.get("note", ""),
        "per_elite_accepted": [r.get("n_accepted", 0) for r in sub],
        "sequences": [p.sequence for p in props],
        "ancestors": [p.ancestor_sequence for p in props],
        "substitution_counts": sorted(p.n_mutations for p in props),
        "diversity": diversity,
    }
    print(f"  [{parent_id}] {len(props)}/{record.get('n_requested', batch_size)} "
          f"proposals "
          f"(counts {info['substitution_counts']}), "
          f"{n_rej} rejected, {n_mal} malformed, {n_att} attempt(s), "
          f"per-elite {info['per_elite_accepted']}, "
          f"history={len(history)} ({len(passed)} passed), interfaces={n_iface}")
    # Printed on its own line rather than buried in the JSON: a round that converged on
    # one neighbourhood is the main way this arm can look busy while searching very
    # little, and it should be visible from the log without opening a file.
    print(f"  [{parent_id}] diversity: {diversity['distinct_edited_positions']}"
          f"/{diversity['peptide_length']} positions touched, pairwise Hamming median "
          f"{diversity['pairwise_hamming_median']} "
          f"(min {diversity['pairwise_hamming_min']}, "
          f"max {diversity['pairwise_hamming_max']})")
    if record.get("note"):
        print(f"  [{parent_id}] note: {record['note']}")
    for r in sub:
        if r.get("note"):
            print(f"  [{parent_id}] elite {r.get('elite_rank')}: {r['note']}")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--round", type=int, required=True, dest="round_index",
                    help="which proposal round this is. Round 1 is the first LLM round; "
                         "it sees whatever the ladder has already folded, which for a "
                         "fresh run is the unmutated reference only.")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH,
                    help="local path to Llama-3.1-8B-Instruct. Never downloaded here.")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="sequences requested per parent per generation. Default 0 "
                         "derives it as --elites x --offspring-per-elite. Set it "
                         "explicitly only to cap a generation; keep it equal to the "
                         "random arm's draws per round, or the comparison is between a "
                         "proposer and a budget rather than between two proposers.")
    ap.add_argument("--elites", type=int, default=DEFAULT_N_ELITES, dest="n_elites",
                    help="how many sequences this generation breeds from, ranked on the "
                         "mean post-relax interface dG of their valid folds over the "
                         "selection half only. Several rather than one because the "
                         "per-generation budget is far too small to be confident the "
                         "single best candidate sits in the right basin. Slots left "
                         "over when fewer than this many have passed the gate are "
                         "filled with the smallest geometry violations, labelled as "
                         "infeasible in the prompt.")
    ap.add_argument("--offspring-per-elite", type=int,
                    default=DEFAULT_OFFSPRING_PER_ELITE,
                    help="sequences requested per elite. Each elite gets its own prompt "
                         "and its own share of the generation, so one elite cannot "
                         "monopolise it, and every proposal records the elite it was "
                         "asked to edit.")
    ap.add_argument("--elite-min-distance", type=int,
                    default=DEFAULT_ELITE_MIN_DISTANCE,
                    help="minimum Hamming distance between elites. Below 2 an elite set "
                         "can be several neighbours of one sequence, which is one "
                         "hypothesis wearing N hats.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--parents", default="")
    ap.add_argument("--feedback-folds", default="select", choices=("select", "all"),
                    help="which folds of the already-scored candidates the model may "
                         "see. select (default) shows folds 0..depth/2 only, the half "
                         "the report selects on, so the measurement half never informs "
                         "a proposal and the held-out column stays held out. all shows "
                         "every fold and LEAKS the measurement half -- for measuring "
                         "the size of that leak, never for a reported number.")
    ap.add_argument("--fold-depth", type=int, default=0,
                    help="fold depth, for the cross-fit guard. Default 0 reads it from "
                         "the cohort's seed list.")
    ap.add_argument("--interfaces-best", type=int, default=2,
                    help="how many passing candidates get a full per-position interface "
                         "table in the prompt")
    ap.add_argument("--interfaces-worst", type=int, default=1,
                    help="how many rejected candidates get one. A rejected interface "
                         "shows the overlap that caused the rejection, which is the "
                         "most common failure and the most actionable.")
    ap.add_argument("--history-best", type=int, default=DEFAULT_HISTORY_BEST)
    ap.add_argument("--history-worst", type=int, default=DEFAULT_HISTORY_WORST)
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--proposal-format", default=DEFAULT_OUTPUT_FORMAT,
                    choices=OUTPUT_FORMATS, dest="output_format",
                    help="how the model is asked to answer. `edits` (the default) asks "
                         "for a substitution list and builds the sequence in software; "
                         "`sequence` asks it to write the sequence out, which is what "
                         "every run before 2026-10-28 did. Measured on the pilot "
                         "prompts, `edits` returns 0.717 of the proposals requested "
                         "against 0.528 (p=0.0052) and eliminates wrong-length answers "
                         "and repeats of already-scored sequences entirely. Do not "
                         "continue an existing run under the other value: the proposals "
                         "are not drawn from the same process, so a mixed run cannot be "
                         "read as one arm. Use a fresh --out-dir.")
    ap.add_argument("--shuffle-history", type=int, default=None,
                    dest="shuffle_history_seed",
                    help="CONTROL ARM, not a feature. Shuffle which outcome belongs to "
                         "which already-scored sequence in the prompt, keeping the "
                         "candidate set, the outcome set and the elites real. Fold what "
                         "this proposes against an identically-budgeted arm without the "
                         "flag: if real pairings produce better candidates, the scored "
                         "history is informing the proposals. Without that comparison a "
                         "prompt change cannot be shown to do anything, because the "
                         "proposer's own sampling spread swamps proposal-level "
                         "differences. Needs its own --out-dir.")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the prompts' inputs and report what would be asked, "
                         "without loading the model")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cohort_path = out_dir / "cohort.json"
    if not cohort_path.exists():
        print(f"[llm-propose] no cohort at {cohort_path}")
        return 1
    cohort = json.loads(cohort_path.read_text())

    parents = cohort["cohort"][args.shard_index :: args.shard_count]
    if args.parents:
        parents = [p.strip() for p in args.parents.split(",") if p.strip()]

    depth = args.fold_depth or len(cohort.get("seeds") or []) or 1
    max_fold_rank = (depth // 2) if args.feedback_folds == "select" else None
    if args.feedback_folds == "select" and max_fold_rank < 1:
        # At depth 1 there is no half to hold out. Rather than silently showing the only
        # fold -- which is the leak -- say so, and let the caller decide.
        print(f"[llm-propose] fold depth {depth} has no held-out half; "
              f"--feedback-folds select cannot be honoured. Re-run with a deeper "
              f"cohort, or pass --feedback-folds all and treat the result as leaked.")
        return 1

    batch_size = args.batch_size or (args.n_elites * args.offspring_per_elite)

    feedback_note = (
        f"The energies above are means over the valid folds among the first "
        f"{max_fold_rank} of {depth} structure-prediction seeds, and the fold counts "
        f"say how many of those folds passed the gate. Seed-to-seed spread on this "
        f"cohort is 4.3 REU median and up to 22 REU, so treat a difference of a few REU "
        f"between two candidates as noise and look for the larger, consistent effects."
        if max_fold_rank
        else f"The energies above are means over the valid folds among all {depth} "
             f"prediction seeds."
    )

    # Contexts come from each parent's own corpus, as in `cpseq_search.py ladder`.
    sources = cohort.get("parent_sources") or {}
    contexts: dict[str, TargetContext] = {}
    if sources:
        by_dir: dict[str, list[str]] = defaultdict(list)
        for p in parents:
            if p in sources:
                by_dir[sources[p]].append(p)
        for ddir, pids in by_dir.items():
            contexts |= load_target_contexts(Path(ddir), pids)
    else:
        contexts = load_target_contexts(Path(cohort["dataset_dir"]), parents)

    print(f"[llm-propose] round {args.round_index}, {len(parents)} parent(s), "
          f"shard {args.shard_index}/{args.shard_count}")
    print(f"[llm-propose] output format: {args.output_format}")
    print(f"[llm-propose] {args.n_elites} elite(s) x {args.offspring_per_elite} "
          f"offspring = {batch_size}/parent, fold depth {depth}, "
          f"feedback-folds={args.feedback_folds}"
          + (f" (showing ranks 0..{max_fold_rank - 1})" if max_fold_rank else " (LEAKED)"))

    proposer = LlamaProposer(
        args.model, temperature=args.temperature, top_p=args.top_p
    )
    if not args.dry_run:
        print(f"[llm-propose] loading {args.model}")
        proposer.load()
        print(f"[llm-propose] model ready in {proposer.load_seconds:.1f}s")

    t0 = time.time()
    infos = []
    for pid in parents:
        if pid not in contexts:
            print(f"[llm-propose] {pid}: no target context -- skipped")
            continue
        infos.append(run_parent(
            shard_index=args.shard_index,
            parent_id=pid, target=contexts[pid], out_dir=out_dir, proposer=proposer,
            round_index=args.round_index, batch_size=batch_size,
            n_elites=args.n_elites,
            offspring_per_elite=args.offspring_per_elite,
            elite_min_distance=args.elite_min_distance,
            max_fold_rank=max_fold_rank,
            interfaces_best=args.interfaces_best,
            interfaces_worst=args.interfaces_worst,
            history_best=args.history_best, history_worst=args.history_worst,
            max_attempts=args.max_attempts, feedback_note=feedback_note,
            dry_run=args.dry_run,
            shuffle_history_seed=args.shuffle_history_seed,
            output_format=args.output_format,
        ))

    asked = sum(i.get("n_requested", 0) for i in infos)
    got = sum(i.get("n_accepted", 0) for i in infos)
    summary = {
        "round_index": args.round_index,
        "arm": ARM_LLM,
        "llm": proposer.config,
        "feedback_folds": args.feedback_folds,
        "output_format": args.output_format,
        "shuffle_history_seed": args.shuffle_history_seed,
        "is_history_control_arm": args.shuffle_history_seed is not None,
        "fold_depth": depth,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_fold_rank_shown": max_fold_rank,
        "batch_size": batch_size,
        "n_elites": args.n_elites,
        "offspring_per_elite": args.offspring_per_elite,
        "elite_min_distance": args.elite_min_distance,
        "elite_objective": "mean post-relax dG_separated over valid folds",
        "n_parents": len(infos),
        "n_requested_total": asked,
        "n_accepted_total": got,
        "wall_time_s": round(time.time() - t0, 1),
        "parents": infos,
    }
    if not args.dry_run:
        # Likewise per-shard. `cpseq_es_report.py` and the scale-up report glob
        # `summary*.json`, so a sharded round reads back as one round.
        name = ("summary.json" if args.shard_count == 1
                else f"summary_shard{args.shard_index}.json")
        (proposals_dir(out_dir, args.round_index) / name).write_text(
            json.dumps(summary, indent=2)
        )
    if args.dry_run:
        print(f"[llm-propose] dry run: {len(infos)} parent(s) ready, {asked} proposal(s) "
              f"would be requested in {summary['wall_time_s']}s. No model was loaded.")
        return 0
    print(f"[llm-propose] {got}/{asked} proposals across {len(infos)} parent(s) "
          f"in {summary['wall_time_s']}s")
    if got < asked:
        print(f"[llm-propose] {asked - got} slot(s) unfilled. These are NOT padded with "
              f"random draws; the affected parents' round budget is the number "
              f"proposed. Per-parent notes are above and in records.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
