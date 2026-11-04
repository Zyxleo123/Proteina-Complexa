#!/usr/bin/env python3
"""Arm E/F/G: fill a chosen mask with ProteinMPNN, on the reference backbone.

The second half of "the LLM says WHERE, a learned model says WHAT". It takes mask sets --
chosen by the LLM (arm E), drawn uniformly (arm F), or either (arm G, random residues) --
and fills them with ProteinMPNN, which is already a structure-conditioned unmasking model
and needs no training to do partial fills: that is what `fixed_positions` IS.

WHY MPNN AND NOT THE TWO FILLERS TRIED FIRST. `README_CPSEQ_MASK_ARM.md` §4 and §6 measure
both. The CPSea unmasking LM is near-flat on these targets (per-position perplexity 16.3
against 20.0 for uniform), and the sequence-conditioned co-generator fills a mask at
free_recovery 0.044-0.058 against a 0.127 positional prior -- below even the flat marginal.
MPNN is the bar this project's own notes set for the whole line of work, and unlike the
other two it reads backbone GEOMETRY.

THE BACKBONE IS THE REFERENCE FOLD, NOT A GENERATOR SAMPLE. `cpseq_mpnn_propose.py` (arm C)
redesigns several generator conformations per parent, which asks a different question --
does a new conformation want a better sequence. Here the conformation is held FIXED at the
Boltz-2 fold of the original peptide, the same structure arm D's interface tables were
computed from, so the only thing differing between arm E and arm F is which positions were
opened. A second varying factor would make the mask effect unattributable.

Threading matters and is not cosmetic: `fixed_positions` pins residues to their identity IN
THE INPUT PDB, so the original LP sequence is written onto the backbone first. Otherwise
the unmasked positions would be pinned to whatever the input PDB happened to carry. MPNN
reads backbone atoms only and uses identity solely for fixed positions, so this is the
supported operation and changes no geometry. Reused from arm C rather than reimplemented.

Nothing here is folded. The fresh Boltz-2 refold through the frozen gate is the measurement.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from cpseq_mpnn_propose import (  # noqa: E402
    PEPTIDE_CHAIN,
    run_mpnn,
    thread_sequence_onto_backbone,
)

from proteinfoundation.cpseq.llm_propose import screen_candidates  # noqa: E402
from proteinfoundation.cpseq.search import (  # noqa: E402
    budget_for,
    default_alphabet,
    proposal_seed,
)

ARM_LLM_MASK_MPNN = "llm_mask_mpnn_fill"
ARM_RAND_MASK_MPNN = "random_mask_mpnn_fill"
ARM_LLM_MASK_RAND = "llm_mask_random_fill"
KIND = "mask_fill"


def resolve_fill(fill: str, arm: str) -> str:
    """Which filler runs for `arm`: `"mpnn"` or `"random"`.

    `by-name` is the original dispatch and stays the default, so the three pilot arms
    behave exactly as they did: only `llm_mask_random_fill` fills randomly. An explicit
    `mpnn`/`random` overrides it, which is what lets a sweep arm carry a label naming its
    own setting instead of borrowing another arm's name to inherit its filler -- a
    borrowed name is permanent, because the label is what lands in the journal.
    """
    if fill in ("mpnn", "random"):
        return fill
    if fill != "by-name":
        raise ValueError(f"unknown fill mode {fill!r}")
    return "random" if arm == ARM_LLM_MASK_RAND else "mpnn"


def ancestor_backbone(
    out_dir: Path, parent_id: str, sequence: str, work: Path, *, tag: str = "ancestor",
) -> Path | None:
    """The run's own fold of `sequence` for this parent, decompressed if need be.

    The evolutionary generalisation of `reference_backbone`. An elite's mask should be
    read off the ELITE's structure, not off the native one: the whole premise of
    breeding from it is that its backbone is a different -- and better -- place in the
    space, and threading its sequence onto the native fold would discard exactly the
    information that makes it an elite.

    Any arm's fold is acceptable here, unlike the reference lookup, which pins
    `arm == "reference"`. An elite is whichever sequence scored best; the arm that
    happened to propose it is not part of its identity.

    Returns None when the sequence has no fold on disk, which the caller must treat as
    "fall back to the reference backbone" rather than as an error -- a retained run
    gzips or drops raw complexes, so a perfectly good elite can simply have no structure
    left to read.
    """
    journal = out_dir / "state" / parent_id / "outcomes.jsonl"
    if not journal.exists():
        return None
    for line in journal.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("technical_failure") or row.get("sequence") != sequence:
            continue
        raw = row.get("raw_pdb_path")
        if not raw:
            continue
        src, gz = Path(raw), Path(str(raw) + ".gz")
        dst = work / f"{parent_id}_{tag}.pdb"
        if src.exists():
            shutil.copyfile(src, dst)
            return dst
        if gz.exists():
            with gzip.open(gz, "rb") as fh, open(dst, "wb") as out:
                shutil.copyfileobj(fh, out)
            return dst
    return None


def reference_backbone(out_dir: Path, parent_id: str, original: str, work: Path) -> Path | None:
    """The run's own reference fold for this parent, decompressed if need be.

    Found by reading the journal for an `arm == "reference"` row whose sequence is the
    original -- the same attribution rule the rest of the pipeline uses. Retention gzips
    raw complexes at `rescorable`, so most of these are `.pdb.gz`.
    """
    journal = out_dir / "state" / parent_id / "outcomes.jsonl"
    if not journal.exists():
        return None
    for line in journal.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("arm") != "reference" or row.get("technical_failure"):
            continue
        if row.get("sequence") != original:
            continue
        raw = row.get("raw_pdb_path")
        if not raw:
            continue
        src, gz = Path(raw), Path(str(raw) + ".gz")
        dst = work / f"{parent_id}_reference.pdb"
        if src.exists():
            shutil.copyfile(src, dst)
            return dst
        if gz.exists():
            with gzip.open(gz, "rb") as fh, open(dst, "wb") as out:
                shutil.copyfileobj(fh, out)
            return dst
    return None


def matched_random_masks(
    sizes: list[int], length: int, seed: int,
    compare_to: set[tuple[int, ...]] | None = None, compare_label: str = "supplied",
) -> list[tuple[int, ...]]:
    """Uniform masks at the SAME sizes the supplied mask set used.

    `compare_label` names what the overlap is being counted against, and defaults to the
    neutral word rather than "LLM". The supplied masks are NOT always an LLM's: the
    filler sweep feeds a random shared set, and a log that calls those "LLM masks" is a
    provenance error that outlives the run -- the log is what a later reader has.

    Matched on size because `search.match_counts` exists for exactly this reason: a
    difference between the two runs must not be an artefact of one of them mutating
    harder. A size with no distinct mask left is skipped and the run is reported short,
    never padded.

    THE LLM'S OWN MASKS ARE NOT EXCLUDED, and that is a correction. Excluding them makes
    this draw deliberately anti-correlated with the thing it is the comparison for, which
    is a bias and not a control. It also starves short peptides: `lnr_1elr` is a 5-mer
    with budget 2, so only C(5,1)+C(5,2)=15 masks exist, the LLM had taken 12, and
    excluding those left this draw ONE mask against the LLM's six. Drawing independently
    is the honest comparison; overlap is a legitimate outcome of two uniform draws over a
    small space, and `compare_to` only counts it so the report can state it.
    """
    rng = random.Random(seed)
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    short: list[int] = []
    for k in sizes:
        k = max(1, min(k, length))
        for _ in range(500):
            m = tuple(sorted(rng.sample(range(length), k)))
            if m not in seen:
                seen.add(m)
                out.append(m)
                break
        else:
            short.append(k)
    if short:
        print(f"    note: no distinct mask left at size(s) {sorted(set(short))} for "
              f"length {length}; short by {len(short)}, NOT padded", flush=True)
    if compare_to:
        overlap = len([m for m in out if m in compare_to])
        if overlap:
            print(f"    note: {overlap}/{len(out)} random mask(s) coincide with a "
                  f"{compare_label} mask (expected on a short peptide; not excluded)",
                  flush=True)
    return out


def journal_parents(out_dir: Path, want: set[str] | None) -> list[tuple[str, str]]:
    """`(parent_id, original_sequence)` for parents whose reference fold is on disk.

    Standalone mode's source of truth, read from the JOURNALS rather than from
    `cohort.json`, in the cohort's recorded order. A parent appears only once its
    reference has actually been folded, which is also exactly the condition for a
    backbone to exist -- the backbone IS the reference fold. A parent that is in the
    cohort but not yet folded is counted and reported, never attempted, because the
    alternative is an arm that silently covers a subset of the corpus.
    """
    cohort = json.loads((out_dir / "cohort.json").read_text())
    order = [e["parent_id"] if isinstance(e, dict) else e for e in cohort["cohort"]]
    out: list[tuple[str, str]] = []
    unfolded: list[str] = []
    for pid in order:
        if want is not None and pid not in want:
            continue
        jf = out_dir / "state" / pid / "outcomes.jsonl"
        original = None
        if jf.exists():
            for line in jf.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001, S112
                    continue
                if r.get("arm") == "reference" and r.get("original_sequence"):
                    original = r["original_sequence"]
                    break
        if original:
            out.append((pid, original))
        else:
            unfolded.append(pid)
    if unfolded:
        print(f"[fill] {len(unfolded)} parent(s) have no reference fold yet and are "
              f"EXCLUDED, not attempted: {', '.join(unfolded[:6])}"
              + (" ..." if len(unfolded) > 6 else ""))
    return out


# The mask-size distribution the pilot actually ran under, from its 239 LLM-chosen masks
# over 20 parents (README_CPSEQ_MPNN_FILL_SETTINGS.md sec. 1.1). Recorded as counts, not
# as a fitted shape: the measured -7.82 REU belongs to THIS distribution, and a tidier
# one (uniform, say) would be a different setting whose result does not carry over.
PILOT_MASK_SIZE_COUNTS = {1: 63, 2: 98, 3: 58, 4: 19, 5: 1}


def sampled_mask_sizes(n_masks: int, budget: int, seed: int) -> list[int]:
    """`n_masks` sizes drawn from the pilot's distribution, truncated at `budget`.

    This replaces taking the sizes from the LLM's masks for the same parent. That
    coupling was right for the pilot -- matching sizes is what stopped a difference
    between the two arms being an artefact of one mutating harder -- but it leaves the
    winning arm depending on an LLM call it does not otherwise need, which at corpus
    scale is thousands of calls to obtain integers.

    Truncation is by DISCARDING sizes above the budget and renormalising, not by
    clamping them down to it. Clamping would pile every oversized draw onto exactly
    `budget` and invent a spike the pilot never had; a 5-mer has budget 2, so under
    clamping 24 % of its masks would be size 2 that were drawn as 3, 4 or 5.

    A parent whose budget is 0 (L < 2) gets no masks and is reported, not silently given
    a size-1 mask it has no budget to spend.
    """
    if budget < 1:
        return []
    pop = [k for k in sorted(PILOT_MASK_SIZE_COUNTS) if k <= budget]
    if not pop:
        return []
    w = [PILOT_MASK_SIZE_COUNTS[k] for k in pop]
    rng = random.Random(seed)
    return rng.choices(pop, weights=w, k=n_masks)


def random_fill(original: str, mask: tuple[int, ...], alphabet, n: int, seed: int):
    """Random residues into the same mask -- the filler control.

    Draws replacement residues from the shared alphabet and never touches an unedited
    position, which is exactly what `search._draw_residue` does for the random arm, so a
    native cysteine outside the mask survives here as it does there.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chars = list(original)
        for p in mask:
            chars[p] = rng.choice(alphabet)
        out.append("".join(chars))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="the run directory; its reference folds and journals are read")
    ap.add_argument("--n-masks", type=int, default=12,
                    help="standalone mode only: masks per parent, sizes drawn from the "
                         "pilot's distribution truncated at the parent's budget. 12 is "
                         "what the pilot ran.")
    ap.add_argument("--masks-dir", type=Path, default=None,
                    help="per-parent mask records from cpseq_llm_mask_round.py")
    ap.add_argument("--round", type=int, default=1, dest="round_index")
    ap.add_argument("--arms", default=f"{ARM_LLM_MASK_MPNN},{ARM_RAND_MASK_MPNN}",
                    help="comma-separated subset of the three arms. A name outside the "
                         "three is allowed and is used as the arm LABEL -- needed for a "
                         "sweep, where several arms differ only in a filler setting and "
                         "each needs its own label in the journal. Such a name requires "
                         "--masks-dir (it has no matched-control mask rule of its own) "
                         "and --fill (its fill type cannot be inferred from the name).")
    ap.add_argument("--fill", choices=("mpnn", "random", "by-name"), default="by-name",
                    help="which filler to use, overriding the name-based dispatch. "
                         "`by-name` (the default) preserves the original behaviour "
                         "exactly: random fill for the one arm whose name says so, MPNN "
                         "for the others. Naming it explicitly is what lets a sweep arm "
                         "carry an honest label instead of borrowing another arm's name "
                         "to get its filler.")
    # ONE sequence per mask, and the diversity comes from the masks. Arm C draws two per
    # mask because it varies the BACKBONE as well; here the backbone is fixed on purpose,
    # and at T=0.2 MPNN is near-deterministic -- measured on `lnr_2v8x`, both draws were
    # byte-identical for three of three masks, so the second draw bought nothing and was
    # rejected as `duplicate within this round`.
    ap.add_argument("--seqs-per-mask", type=int, default=1)
    # Arm C's default. Kept rather than raised: a higher temperature would buy diversity
    # by making MPNN less sure, which is the opposite of what this arm is testing. The
    # mask is where diversity is supposed to come from.
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--model-name", default="v_48_020")
    ap.add_argument("--omit-aas", default="XC")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--parents", default="", help="comma-separated; default all in masks-dir")
    ap.add_argument("--run-tag", default="",
                    help="suffix for this run's scratch directory under mask_fill/. The "
                         "per-arm proposal files are already arm-disjoint, but the "
                         "run-level proposals.jsonl and summary.json are NOT: two runs "
                         "over the same round would overwrite each other's. Concurrent "
                         "runs (a sweep where arms differ only in a filler setting) must "
                         "each pass a distinct tag. Default empty keeps the original "
                         "single-run layout.")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    alpha = default_alphabet(allow_cys=False)
    round_dir = (args.out_dir / "mask_fill"
                 / f"round{args.round_index}{'__' + args.run_tag if args.run_tag else ''}")
    work = round_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    want = set(p for p in args.parents.split(",") if p) or None

    # Two sources of parents and mask sizes, and the choice is which arms are runnable.
    # With --masks-dir the LLM's masks are read and the random arm's sizes are MATCHED to
    # them, which is the pilot's paired design. Without it, only the random-mask arm can
    # run -- there are no LLM masks to fill or to match -- and its sizes are drawn from
    # the pilot's distribution instead. See `sampled_mask_sizes`.
    standalone = args.masks_dir is None
    jobs: list[tuple[str, str, list[tuple[int, ...]] | None, list[str] | None]] = []
    if standalone:
        needs_llm = [a for a in arms if a != ARM_RAND_MASK_MPNN]
        if needs_llm:
            print(f"FATAL: {', '.join(needs_llm)} need --masks-dir; only "
                  f"{ARM_RAND_MASK_MPNN} runs standalone", file=sys.stderr)
            return 1
        jobs = [(pid, original, None, None) for pid, original in
                journal_parents(args.out_dir, want)]
        if not jobs:
            print(f"FATAL: no parent in {args.out_dir} has a reference fold on disk",
                  file=sys.stderr)
            return 1
        print(f"[fill] standalone: {len(jobs)} parent(s) with a reference fold, "
              f"{args.n_masks} mask(s) each, sizes ~ {PILOT_MASK_SIZE_COUNTS}")
    else:
        mask_files = sorted(args.masks_dir.glob("*_r%d.json" % args.round_index))
        if want:
            mask_files = [f for f in mask_files
                          if json.loads(f.read_text())["parent_id"] in want]
        if not mask_files:
            print(f"FATAL: no mask records in {args.masks_dir}", file=sys.stderr)
            return 1
        for mf in mask_files:
            rec = json.loads(mf.read_text())
            masks = [tuple(int(p) for p in m.split(",") if p != "")
                     for m in rec.get("masks", [])]
            if not masks:
                print(f"[fill] {rec['parent_id']}: no masks in {mf.name}; skipped")
                continue
            # `mask_ancestors` is the evolutionary part, and it is OPTIONAL: a round
            # that bred from elites writes one ancestor per mask, a round that did not
            # writes none and every mask is read against the original. Defaulting to
            # the original is what keeps every pre-ES mask file readable unchanged.
            anc = rec.get("mask_ancestors") or []
            if anc and len(anc) != len(masks):
                print(f"[fill] {rec['parent_id']}: {len(anc)} ancestors for "
                      f"{len(masks)} masks in {mf.name}; REFUSING rather than guessing "
                      f"which mask belongs to which elite")
                continue
            ancestors = list(anc) or [rec["original_sequence"]] * len(masks)
            jobs.append((rec["parent_id"], rec["original_sequence"], masks, ancestors))

    rows: list[dict] = []
    summary: list[dict] = []
    # `given_masks`: the masks supplied via --masks-dir, whoever chose them. Named for
    # their ROLE and not for a presumed author -- an LLM round writes them, and so does
    # `cpseq_make_shared_masks.py` for a filler sweep, where they are uniform random.
    for pid, original, given_masks, mask_ancestors in jobs:
        L, budget = len(original), budget_for(len(original))

        bb = reference_backbone(args.out_dir, pid, original, work)
        if bb is None:
            # Not skipped silently: a parent missing its reference backbone is missing
            # from the arm, and an arm quietly covering 18 of 20 complexes is not the
            # comparison it claims to be.
            print(f"[fill] {pid}: NO REFERENCE BACKBONE on disk -- excluded from this arm")
            summary.append({"parent_id": pid, "excluded": "no_reference_backbone"})
            continue
        # One threaded structure per DISTINCT ancestor. A non-evolutionary round has
        # exactly one (the original) and this is the old behaviour unchanged; an
        # evolutionary round has one per elite, each read off that elite's OWN fold when
        # the run still has it. Falling back to the reference backbone is explicit and
        # reported, because an elite silently masked on the native structure would be a
        # different experiment wearing this one's name.
        wanted = sorted(set(mask_ancestors or [original]) | {original})
        threaded_for: dict[str, Path] = {}
        backbone_for: dict[str, str] = {}
        for i, anc in enumerate(wanted):
            if anc == original:
                anc_bb, src = bb, "reference"
            else:
                anc_bb = ancestor_backbone(args.out_dir, pid, anc, work, tag=f"anc{i}")
                src = "own fold" if anc_bb is not None else "reference (no own fold)"
                if anc_bb is None:
                    anc_bb = bb
            dst = work / f"{pid}_threaded_{i}.pdb"
            thread_sequence_onto_backbone(
                anc_bb, dst, peptide_chain=PEPTIDE_CHAIN, original=anc
            )
            threaded_for[anc] = dst
            backbone_for[anc] = anc_bb.name
            if anc != original:
                print(f"[fill] {pid}: elite {anc} masked on its {src}")
        threaded = threaded_for[original]

        # `hash()` is salted per process for str, so it cannot appear in a seed that has
        # to reproduce: the pilot's own control masks are not regenerable because of it.
        # `proposal_seed` is the project's stable per-(parent, arm, round) seed.
        mask_seed = int(proposal_seed(pid, ARM_RAND_MASK_MPNN, args.round_index, 0)
                        % 10**6)
        if given_masks is None:
            sizes = sampled_mask_sizes(args.n_masks, budget, seed=mask_seed)
            if not sizes:
                print(f"[fill] {pid}: L={L} gives budget {budget}; no mask fits, excluded")
                summary.append({"parent_id": pid, "excluded": "budget_too_small"})
                continue
            ctrl_masks = matched_random_masks(sizes, L, seed=mask_seed)
            print(f"\n=== {pid}  {original}  L={L} budget={budget}  "
                  f"{len(ctrl_masks)} sampled masks {sorted(sizes)}", flush=True)
        else:
            sizes = [len(m) for m in given_masks]
            ctrl_masks = matched_random_masks(
                sizes, L, seed=mask_seed, compare_to=set(given_masks),
                compare_label="supplied",
            )
            print(f"\n=== {pid}  {original}  L={L} budget={budget}  "
                  f"{len(given_masks)} supplied masks {sizes} "
                  f"(from {args.masks_dir.name})", flush=True)

        for arm in arms:
            masks = given_masks if arm != ARM_RAND_MASK_MPNN else ctrl_masks
            if masks is None:
                # Only ARM_RAND_MASK_MPNN has a mask rule of its own in standalone mode.
                # Any other arm reads them from --masks-dir, so reaching here means a
                # custom arm was named without one, and filling nothing silently would
                # look like an arm that proposed badly rather than one never fed.
                print(f"[fill] {pid} {arm}: no masks available -- a custom arm name "
                      f"needs --masks-dir; skipped")
                continue
            use_random = resolve_fill(args.fill, arm) == "random"
            produced: list[tuple[str, str]] = []
            for k, mask in enumerate(masks):
                # The mask's ancestor, positionally. The random-mask control reuses the
                # SAME ancestor as the mask it is matched to: it controls for position
                # choice, so letting it breed from a different sequence would confound
                # the one axis it exists to isolate.
                anc = (mask_ancestors[k] if mask_ancestors and k < len(mask_ancestors)
                       else original)
                anc_pdb = threaded_for.get(anc, threaded)
                if use_random:
                    seqs = random_fill(
                        anc, mask, alpha, args.seqs_per_mask,
                        seed=int(proposal_seed(pid, arm, args.round_index, k) % 10**6),
                    )
                    got = [(s, "random fill") for s in seqs]
                else:
                    fixed = [i for i in range(L) if i not in mask]
                    designs = run_mpnn(
                        pdb_path=anc_pdb,
                        fixed_positions=fixed,
                        out_dir=work / f"{pid}_{arm}_m{'-'.join(map(str, mask))}",
                        n_seq=args.seqs_per_mask,
                        temperature=args.temperature,
                        seed=int(proposal_seed(pid, arm, args.round_index, mask) % 100000),
                        model_name=args.model_name,
                        omit_aas=args.omit_aas,
                    )
                    got = []
                    for d in designs:
                        seq = d.get("sequence")
                        if d.get("status") != "ok" or not seq:
                            print(f"[fill] {pid} {arm} mask={mask}: {d.get('status')} "
                                  f"{str(d.get('reason'))[:100]}")
                            continue
                        if len(seq) != L:
                            print(f"[fill] {pid} {arm} mask={mask}: length {len(seq)} "
                                  f"!= {L}; dropped")
                            continue
                        got.append((seq, f"mpnn score {d.get('score')}"))
                for seq, note in got:
                    produced.append((seq, note))
                    rows.append({
                        "parent_id": pid, "arm": arm, "round_index": args.round_index,
                        "proposal_kind": KIND, "original": original, "sequence": seq,
                        "mask": ",".join(map(str, mask)), "mask_size": len(mask),
                        "edited_positions": ",".join(
                            str(i) for i in range(L) if seq[i] != original[i]
                        ),
                        "n_mutations": sum(1 for i in range(L) if seq[i] != original[i]),
                        "noop": seq == original,
                        # The backbone actually used, which for an elite bred on its
                        # own fold is NOT the reference. Recording the reference here
                        # regardless would claim every offspring was masked on the
                        # native structure.
                        "backbone": backbone_for.get(anc, bb.name), "note": note,
                        # Ancestry, recorded and never inferred: without it a later
                        # reader cannot tell an elite's offspring from a fresh draw,
                        # and "did the improvement descend from the elite that was
                        # supposed to produce it" becomes unanswerable.
                        "ancestor_sequence": (None if anc == original else anc),
                        "n_mutations_from_ancestor": sum(
                            1 for i in range(L) if seq[i] != anc[i]
                        ),
                        # Per ROW, not only in the run summary. A filler sweep puts
                        # several arms in one directory, and a row that cannot say which
                        # filler and which temperature produced it is unattributable once
                        # the summary for its run has been overwritten by the next one.
                        "fill": "random" if use_random else "mpnn",
                        "mpnn_temperature": (None if use_random else args.temperature),
                        "mpnn_model": (None if use_random else args.model_name),
                        "masks_shared_from": (
                            str(args.masks_dir) if args.masks_dir else None),
                    })
            # The shared legality screen, against the ORIGINAL. A legal mask can still
            # produce an illegal sequence, and that rejection belongs to the filler.
            # `already` carries the ancestors too. A fill that reproduces the elite it
            # was bred from is not a new candidate -- it is a wasted draw dressed as
            # one -- and counting it would let a generation "produce" its own parents.
            kept, rejected = screen_candidates(
                produced, original=original, alphabet=alpha,
                already={original} | set(mask_ancestors or ()),
            )
            # FIRST occurrence only. `screen_candidates` already dropped the repeats, but
            # matching its output back by sequence STRING would re-accept every copy --
            # two masks can land on the same sequence, and marking both accepted would
            # put a duplicate in the arm's draw list. The ladder dedups again on load, so
            # this is belt and braces, but the counts reported here would otherwise
            # overstate how many distinct candidates the arm actually bought.
            keep = {s for s, _ in kept}
            used: set[str] = set()
            for r in rows:
                if r["parent_id"] != pid or r["arm"] != arm:
                    continue
                seq = r["sequence"]
                r["accepted"] = seq in keep and seq not in used
                if r["accepted"]:
                    used.add(seq)
            n_noop = sum(1 for s, _ in produced if s == original)
            print(f"  {arm:24s} {len(produced):3d} fills -> {len(kept):3d} accepted, "
                  f"{n_noop} no-op, {len(rejected)} rejected", flush=True)
            summary.append({
                "parent_id": pid, "arm": arm, "n_masks": len(masks),
                "mask_sizes": [len(m) for m in masks], "n_fills": len(produced),
                "n_accepted": len(kept), "n_noop": n_noop,
                "rejection_reasons": sorted({r["reason"] for r in rejected}),
            })

    dest = round_dir
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "proposals.jsonl"
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # Also per parent and per arm, which is the layout the ladder's `load_mask_proposals`
    # reads. Written in production order and NOT sorted: a rung is a prefix of the draws,
    # so reordering here would silently change what every finished rung had measured.
    n_files = 0
    for arm in arms:
        for pid in sorted({r["parent_id"] for r in rows if r["arm"] == arm}):
            keep = [
                r for r in rows
                if r["arm"] == arm and r["parent_id"] == pid and r.get("accepted")
            ]
            if not keep:
                continue
            pdir = args.out_dir / "mask_proposals" / arm / f"round{args.round_index}"
            pdir.mkdir(parents=True, exist_ok=True)
            with (pdir / f"{pid}.jsonl").open("w") as fh:
                for i, r in enumerate(keep):
                    fh.write(json.dumps({**r, "mask_round": args.round_index,
                                         "round_index": args.round_index,
                                         "draw_slot": i}) + "\n")
            n_files += 1
    print(f"wrote {n_files} per-parent proposal file(s) under "
          f"{args.out_dir / 'mask_proposals'}")
    (dest / "meta.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arms": arms, "seqs_per_mask": args.seqs_per_mask,
        "mpnn_model": args.model_name, "mpnn_temperature": args.temperature,
        "omit_aas": args.omit_aas, "seed": args.seed,
        "backbone": "reference Boltz-2 fold of the original peptide, held fixed",
        "summary": summary, "folded": False,
        "note": "Proposals only. Nothing here has been through the Boltz-2 gate.",
    }, indent=2))
    print(f"\nwrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
