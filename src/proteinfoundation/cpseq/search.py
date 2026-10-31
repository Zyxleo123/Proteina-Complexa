"""Proposal arms for cyclic-peptide sequence search, and the budget they share.

The task: given an original linear-peptide sequence and its target, find a head-to-tail
cyclic sequence of the SAME length whose predicted complex scores well. Up to
`floor(L/2)` substitutions are permitted.

Three things this module keeps honest, because each was a way to produce a comparison
that looks like a result and is not:

1. **The budget is measured against the ORIGINAL sequence, never against the ancestor
   a proposal was edited from.** Two rounds of "only two edits each" would otherwise
   walk arbitrarily far from the original while every individual step looked legal.
   `within_budget()` is the only gate, and it takes the original.

2. **The ceiling is not a target.** `floor(L/2)` is the most that is allowed, not the
   number to aim for, so no proposer draws its mutation count from the top of the range
   and nothing penalises a candidate for being close to the original. Arm A matches Arm
   B's realised counts (see `match_counts`) precisely so that a difference between the
   arms cannot be an artefact of one arm mutating harder than the other.

3. **Exhausted space is reported, not padded.** When the permitted neighbourhood cannot
   supply the requested number of UNIQUE sequences, the proposer returns what exists and
   says so. Re-evaluating one sequence twice and calling it two candidates would inflate
   a yield denominator with duplicates.

Reversion to the original residue is permitted and is not a mutation: it spends budget
back. That is what makes a local edit able to undo an earlier bad guess.

Nothing here predicts or scores. These are string operations over an alphabet, so they
are cheap, deterministic given a seed, and testable without a GPU.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from proteinfoundation.cpseq.candidates import AA20, DEFAULT_EXCLUDED_FROM_RANDOM

# Arm names, as they appear in every record.
ARM_RANDOM = "random_control"
ARM_GUIDED = "guided_search"
ARM_MPNN = "backbone_mpnn"
# Arm D: the same search, with an instruction-tuned text model choosing the positions and
# residues that `propose_edit` draws from an rng. See `llm_propose.py`.
ARM_LLM = "llm_search"
# Arm E: the proposal split in two. The LLM names positions only and the CPSea unmasking
# model fills them (`mlm_propose.py`). ARM_MLM_MASK is its control and differs in exactly
# one thing -- the positions are drawn uniformly instead of chosen -- so the pair measures
# the LLM's choice of WHERE and nothing else. Both fill with the same model, the same
# temperature and the same number of draws; without that the comparison measures the
# filler, not the chooser.
ARM_LLM_MASK = "llm_mask_mlm_fill"
ARM_MLM_MASK = "random_mask_mlm_fill"

# Proposal kinds, recorded per candidate so a result can be attributed to the mechanism
# that produced it rather than to the arm as a whole.
KIND_REFERENCE = "unmutated_reference"
KIND_SINGLE_SCAN = "single_substitution_scan"
KIND_LOCAL = "local_edit"
KIND_COORDINATED = "coordinated_edit"
KIND_RESTART = "random_restart"
KIND_RANDOM = "random_matched_count"
# Distinguished from KIND_RANDOM on the row itself: a matched draw inherited its count
# from the guided arm, a standalone draw did not, and pooling the two would lose the one
# fact that says whether a row is comparable to the matched pilot.
KIND_RANDOM_STANDALONE = "random_standalone_count"
KIND_MPNN = "mpnn_redesign"
KIND_LLM = "llm_proposed"
# A sequence written by the unmasking model into a mask somebody else chose. The row
# records which positions were opened separately from which ones actually changed: the
# model is free to re-emit the native residue, and a mask of three that produced one
# substitution is a different event from a mask of one.
KIND_MLM_FILL = "mlm_fill"
# A sequence written into a mask somebody else chose, by whichever filler the arm names
# (the arm says which; the kind says how the proposal was made). Separate from KIND_LLM
# because under KIND_LLM the model that picked the positions also picked the residues,
# and pooling the two would lose the one fact that distinguishes arms D and E.
KIND_MASK_FILL = "mask_fill"


def default_alphabet(allow_cys: bool = False) -> tuple[str, ...]:
    """Canonical residues available to substitution.

    CYS is excluded by default, inherited from `candidates.py` and for the same reason: a
    free cysteine introduced at random can pair with another and close a ring nobody
    asked for, which would make the molecule differ from the `mainchain` chemistry its
    own label names. The exclusion costs one residue of alphabet and is recorded in the
    run config, so the enumeration sizes below are 18 alternatives per position rather
    than the 19 a full canonical alphabet would give.
    """
    if allow_cys:
        return AA20
    return tuple(a for a in AA20 if a not in set(DEFAULT_EXCLUDED_FROM_RANDOM))


def budget_for(length: int) -> int:
    """`floor(L/2)` -- the substitution ceiling, which also guarantees >= 50 % identity."""
    return length // 2


def mutated_positions(original: str, sequence: str) -> tuple[int, ...]:
    """0-based positions where `sequence` differs from `original`."""
    if len(original) != len(sequence):
        raise ValueError(
            f"substitutions only, but length changed {len(original)} -> {len(sequence)}"
        )
    return tuple(i for i, (a, b) in enumerate(zip(original, sequence)) if a != b)


def n_mutations(original: str, sequence: str) -> int:
    return len(mutated_positions(original, sequence))


def within_budget(original: str, sequence: str) -> bool:
    """The one gate. Always measured against the original, never against an ancestor."""
    return n_mutations(original, sequence) <= budget_for(len(original))


def screen_slots_remaining(outcomes: Iterable[dict], arm: str, cap: int) -> int:
    """Screening slots left for `arm`, counted from the journal already on disk.

    The resume guarantee cannot come from deterministic proposals alone. Round 0 (the
    single-substitution scan) is a pure function of the parent, so a resume reproduces it
    and skips it. Rounds 1+ are a function of the ARCHIVE, and a resumed task rebuilds
    that archive from a journal that has GROWN, so it legitimately plans different
    sequences and pays for them on top of the work already done -- no evaluation is
    duplicated, yet the arm ends up with more slots than the arm it is compared against.

    Measured on `lnr_5njx` across two preemptions: 32 -> 36 -> 45 guided evaluations
    against a budget of 32, while parents that never requeued sat at exactly 32. Counting
    spent slots from the journal is therefore the only level that actually caps an arm,
    which is why it is a function here rather than a line inside the driver's loop.
    """
    spent = sum(
        1 for o in outcomes
        if o.get("arm") == arm and o.get("stage") == "screen"
    )
    return max(0, cap - spent)


def proposal_seed(*parts: object) -> int:
    """Stable 63-bit seed from the proposal's identity, not from a global counter.

    Same convention as `candidates._seed_for`: a re-run of one round reproduces its own
    proposals without depending on how many draws happened before it.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1


@dataclass
class Proposal:
    """One proposed sequence, with the provenance needed to attribute an outcome."""

    sequence: str
    arm: str
    round_index: int
    kind: str
    # The sequence this was edited from (None for the reference and for draws made
    # around the original). Kept as the string so ancestry survives without an id map.
    ancestor_sequence: str | None
    original_sequence: str
    edited_positions: tuple[int, ...]
    seed: int
    note: str = ""
    # Position in the extendable random stream, when the proposal came from one. It is
    # the draw's identity across runs: slot 7 is the same sequence in a 8-draw run and
    # in an 800-draw one, so a report can say which draws a number was computed over.
    slot: int | None = None

    def __post_init__(self) -> None:
        if len(self.sequence) != len(self.original_sequence):
            raise ValueError(
                f"{self.kind}: length changed "
                f"{len(self.original_sequence)} -> {len(self.sequence)}"
            )
        # The budget is enforced here as well as at the call site, so a proposer that
        # forgets cannot emit an over-budget sequence.
        if not within_budget(self.original_sequence, self.sequence):
            raise ValueError(
                f"{self.kind}: {n_mutations(self.original_sequence, self.sequence)} "
                f"mutations exceeds budget {budget_for(len(self.original_sequence))}"
            )

    @property
    def n_mutations(self) -> int:
        return n_mutations(self.original_sequence, self.sequence)

    @property
    def mutation_fraction(self) -> float:
        return self.n_mutations / len(self.original_sequence)

    def to_row(self) -> dict:
        return {
            "sequence": self.sequence,
            "arm": self.arm,
            "round_index": self.round_index,
            "proposal_kind": self.kind,
            "ancestor_sequence": self.ancestor_sequence,
            "edited_positions": ",".join(str(p) for p in self.edited_positions),
            "n_mutations": self.n_mutations,
            "mutation_fraction": round(self.mutation_fraction, 6),
            "proposal_seed": self.seed,
            "note": self.note,
            "draw_slot": self.slot,
        }


# --------------------------------------------------------------------------------------
# Arm B, round 0: the single-substitution scan
# --------------------------------------------------------------------------------------


def enumerate_single_substitutions(
    original: str, alphabet: tuple[str, ...] | None = None
) -> list[tuple[int, str, str]]:
    """Every single substitution as (position, residue, sequence).

    Enumeration is over STRINGS and costs nothing; it is not a prediction budget. A
    12-mer over the 18-residue alphabet gives 216 of these, and the pilot evaluates 16.
    `exhaustive_single_scan` in the driver is the option that spends real compute on all
    of them.
    """
    alpha = alphabet or default_alphabet()
    out: list[tuple[int, str, str]] = []
    for i, native in enumerate(original):
        for aa in alpha:
            if aa == native:
                continue
            out.append((i, aa, original[:i] + aa + original[i + 1 :]))
    return out


def select_spread_singles(
    original: str,
    k: int,
    *,
    seed: int,
    alphabet: tuple[str, ...] | None = None,
) -> list[Proposal]:
    """Choose `k` single substitutions spread across positions and residue choices.

    Round-robin over positions in a shuffled order, taking one residue per position
    before any position is revisited. With k=16 on a 7-mer every position is probed
    twice; on a 16-mer sixteen distinct positions are probed once each. A uniform draw
    over all L x 18 pairs would instead leave several positions untouched while probing
    others three times, and position coverage is what the first round is for.
    """
    alpha = alphabet or default_alphabet()
    rng = np.random.default_rng(seed)

    # Per position, the alternatives in a shuffled order.
    per_pos: dict[int, list[str]] = {}
    for i, native in enumerate(original):
        choices = [a for a in alpha if a != native]
        rng.shuffle(choices)
        per_pos[i] = choices

    positions = list(range(len(original)))
    rng.shuffle(positions)

    proposals: list[Proposal] = []
    seen: set[str] = {original}
    # Cycle through positions until k are collected or the space runs dry.
    while len(proposals) < k:
        progressed = False
        for i in positions:
            if len(proposals) >= k:
                break
            if not per_pos[i]:
                continue
            aa = per_pos[i].pop()
            seq = original[:i] + aa + original[i + 1 :]
            if seq in seen:
                continue
            seen.add(seq)
            progressed = True
            proposals.append(
                Proposal(
                    sequence=seq,
                    arm=ARM_GUIDED,
                    round_index=0,
                    kind=KIND_SINGLE_SCAN,
                    ancestor_sequence=original,
                    original_sequence=original,
                    edited_positions=(i,),
                    seed=proposal_seed(original, "single", i, aa),
                )
            )
        if not progressed:
            # Space exhausted: every single substitution has been taken. Report what
            # exists rather than repeating one.
            break
    return proposals


# --------------------------------------------------------------------------------------
# Edit proposers
# --------------------------------------------------------------------------------------


def _draw_residue(rng, native: str, current: str, alphabet: tuple[str, ...]) -> str:
    """A residue different from what is already at the position.

    Excluding `current` is what makes an edit an edit: drawing the residue already
    present would spend a proposal on a no-op and let an unchanged sequence be counted
    as a new candidate.
    """
    choices = [a for a in alphabet if a != current]
    if not choices:
        return current
    return str(rng.choice(choices))


def propose_edit(
    *,
    ancestor: str,
    original: str,
    n_positions: int,
    arm: str,
    round_index: int,
    kind: str,
    seed: int,
    alphabet: tuple[str, ...] | None = None,
    allow_reversion: bool = True,
    max_attempts: int = 64,
) -> Proposal | None:
    """Edit `n_positions` of `ancestor`, staying within the budget against `original`.

    Reversion to the original residue is offered explicitly when the ancestor is already
    at the ceiling: at that point the only legal edits are ones that spend budget back,
    and a proposer that only ever substitutes would return nothing but failures there.
    """
    alpha = alphabet or default_alphabet()
    rng = np.random.default_rng(seed)
    budget = budget_for(len(original))
    L = len(original)
    n_positions = max(1, min(n_positions, L))

    for _ in range(max_attempts):
        positions = rng.choice(L, size=n_positions, replace=False)
        seq = list(ancestor)
        for p in positions:
            p = int(p)
            native = original[p]
            at_ceiling = n_mutations(original, "".join(seq)) >= budget
            diverged = seq[p] != native
            # Revert when budget is tight and this position is one of the spent ones;
            # otherwise substitute. The coin keeps reversion available even when there
            # is room, so the search can undo a bad early guess.
            revert = allow_reversion and diverged and (at_ceiling or rng.random() < 0.35)
            seq[p] = native if revert else _draw_residue(rng, native, seq[p], alpha)
        cand = "".join(seq)
        if cand == ancestor:
            continue
        if not within_budget(original, cand):
            continue
        return Proposal(
            sequence=cand,
            arm=arm,
            round_index=round_index,
            kind=kind,
            ancestor_sequence=ancestor,
            original_sequence=original,
            edited_positions=mutated_positions(ancestor, cand),
            seed=seed,
        )
    return None


def propose_restart(
    *,
    original: str,
    round_index: int,
    seed: int,
    max_count: int | None = None,
    alphabet: tuple[str, ...] | None = None,
) -> Proposal | None:
    """A fresh draw around the original, at a count drawn from the legal range.

    The count is uniform over 1..budget rather than fixed at the ceiling, so restarts
    sample the permitted space instead of only its far edge.
    """
    alpha = alphabet or default_alphabet()
    rng = np.random.default_rng(seed)
    budget = budget_for(len(original))
    hi = budget if max_count is None else min(max_count, budget)
    if hi < 1:
        return None
    count = int(rng.integers(1, hi + 1))
    seq = random_at_count(original, count, seed=seed, alphabet=alpha)
    if seq is None or seq == original:
        return None
    return Proposal(
        sequence=seq,
        arm=ARM_GUIDED,
        round_index=round_index,
        kind=KIND_RESTART,
        ancestor_sequence=None,
        original_sequence=original,
        edited_positions=mutated_positions(original, seq),
        seed=seed,
    )


# --------------------------------------------------------------------------------------
# Arm A: random control at a matched mutation count
# --------------------------------------------------------------------------------------


def random_at_count(
    original: str,
    count: int,
    *,
    seed: int,
    alphabet: tuple[str, ...] | None = None,
) -> str | None:
    """A sequence with EXACTLY `count` substitutions, drawn uniformly.

    Positions uniform without replacement, residues uniform over the alphabet minus the
    native one, so the realised count is exactly `count` and cannot be deflated by a
    draw that happens to match the native residue.
    """
    alpha = alphabet or default_alphabet()
    if count < 0 or count > budget_for(len(original)):
        return None
    if count == 0:
        return original
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(original), size=count, replace=False)
    seq = list(original)
    for p in positions:
        p = int(p)
        choices = [a for a in alpha if a != original[p]]
        if not choices:
            return None
        seq[p] = str(rng.choice(choices))
    return "".join(seq)


def match_counts(
    *,
    original: str,
    counts: list[int],
    round_index: int,
    seed_base: object,
    exclude: set[str] | None = None,
    alphabet: tuple[str, ...] | None = None,
    attempts_per: int = 48,
) -> tuple[list[Proposal], str]:
    """Random controls whose mutation counts match `counts` one for one.

    This is the whole point of Arm A. Comparing a guided arm that mostly makes single
    edits against a random arm drawing at the 50 % ceiling would measure mutation load,
    not proposal quality. So the control is generated only AFTER the guided round has
    declared its realised counts, and reproduces that multiset.

    Returns the proposals and a coverage note naming any count that could not be filled
    with a unique sequence.
    """
    alpha = alphabet or default_alphabet()
    seen = set(exclude or set())
    seen.add(original)
    out: list[Proposal] = []
    unfilled: list[int] = []

    for n, count in enumerate(counts):
        got = None
        for attempt in range(attempts_per):
            s = proposal_seed(seed_base, round_index, n, count, attempt)
            cand = random_at_count(original, count, seed=s, alphabet=alpha)
            if cand is None or cand in seen:
                continue
            got = (cand, s)
            break
        if got is None:
            unfilled.append(count)
            continue
        cand, s = got
        seen.add(cand)
        out.append(
            Proposal(
                sequence=cand,
                arm=ARM_RANDOM,
                round_index=round_index,
                kind=KIND_RANDOM,
                ancestor_sequence=None,
                original_sequence=original,
                edited_positions=mutated_positions(original, cand),
                seed=s,
            )
        )

    note = ""
    if unfilled:
        note = (
            f"{len(unfilled)} of {len(counts)} matched draws could not be filled with a "
            f"unique sequence at counts {sorted(unfilled)}; arm coverage is "
            f"{len(out)}/{len(counts)}"
        )
    return out, note


# --------------------------------------------------------------------------------------
# The archive
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Arm A standalone: a random arm that owes nothing to the guided arm
# --------------------------------------------------------------------------------------

# The measured marginal substitution-count distribution of the realised random arm over
# the PepBench valid split: 2,041 draws across 69 parents, from
# `cpseq_budget_ladder.py --arm random_control`.
#
# Reproducing this marginal rather than inventing one is deliberate. It is what makes the
# budget ladder measured on the matched run transferable to a standalone run: the ladder's
# coverage-vs-B curve was observed under these counts, so a scale-up that draws from the
# same distribution inherits those numbers instead of needing its own calibration.
STANDALONE_COUNT_WEIGHTS: dict[int, float] = {
    1: 0.592, 2: 0.195, 3: 0.121, 4: 0.059, 5: 0.017, 6: 0.010, 7: 0.005, 8: 0.001,
}


def standalone_counts(
    *,
    original: str,
    n_slots: int,
    seed_base: object,
    weights: dict[int, float] | None = None,
) -> list[int]:
    """Substitution counts for a random arm drawn WITHOUT the guided arm's realised set.

    `match_counts` exists so that a guided arm making mostly single edits is not compared
    against a control drawing at the 50 % ceiling -- that would measure mutation load
    rather than proposal quality. It buys that at the cost of making arm A undefined
    until arm B has run, which is fine for a matched pilot and fatal for a random-only
    scale-up.

    This replaces the dependency with a frozen distribution, which the data says costs
    nothing measurable. Over the PepBench valid split the gate pass rate does not depend
    detectably on substitution count across the well-sampled range 1-4 (381/382/429/312
    per mille on 1,319/411/266/128 draws; chi-square 6.52 on 3 df, p = 0.089), and
    neither does interface energy among valid draws (Spearman rho = -0.020, p = 0.58 on
    658 valid rows). So the count policy is a free parameter in the region where there is
    enough data to judge it, and the comparison `match_counts` was protecting is not
    sensitive to which of these distributions is used.

    Counts above the parent's `floor(L/2)` ceiling are dropped and the remainder
    renormalised, so a 5-mer (ceiling 2) draws only from {1, 2}.

    The plan is a pure function of `(seed_base, n_slots)` and reads neither the archive
    nor the journal, which also removes the resume-overspend failure mode recorded in
    README_CPSEQ_SEARCH_PILOT sec. 5: a requeued standalone arm reproduces exactly the
    same slot-to-count assignment, so `journal.has` skips the work already paid for
    instead of planning fresh draws on top of it.
    """
    ceiling = budget_for(len(original))
    w = {k: v for k, v in (weights or STANDALONE_COUNT_WEIGHTS).items()
         if 1 <= k <= ceiling and v > 0}
    if not w or n_slots <= 0:
        return []
    ks = sorted(w)
    total = sum(w[k] for k in ks)
    probs = [w[k] / total for k in ks]
    rng = np.random.default_rng(proposal_seed(seed_base, "standalone_counts", n_slots))
    return [int(x) for x in rng.choice(ks, size=n_slots, p=probs)]


def standalone_random_arm(
    *,
    original: str,
    n_slots: int,
    seed_base: object,
    exclude: set[str] | None = None,
    alphabet: tuple[str, ...] | None = None,
    weights: dict[int, float] | None = None,
    attempts_per: int = 48,
) -> tuple[list[Proposal], str]:
    """`n_slots` distinct uniform substitution draws, with no guided input at all.

    Round index is 0 for every proposal: a standalone random arm has no rounds, because
    nothing about draw k+1 depends on the outcome of draw k. Recording a fake round
    structure would imply feedback that does not exist.
    """
    alpha = alphabet or default_alphabet()
    counts = standalone_counts(
        original=original, n_slots=n_slots, seed_base=seed_base, weights=weights
    )
    seen = set(exclude or set())
    seen.add(original)
    out: list[Proposal] = []
    unfilled: list[int] = []

    for slot, count in enumerate(counts):
        got = None
        for attempt in range(attempts_per):
            s = proposal_seed(seed_base, "standalone", slot, count, attempt)
            cand = random_at_count(original, count, seed=s, alphabet=alpha)
            if cand is None or cand in seen:
                continue
            got = (cand, s)
            break
        if got is None:
            unfilled.append(count)
            continue
        cand, s = got
        seen.add(cand)
        out.append(
            Proposal(
                sequence=cand,
                arm=ARM_RANDOM,
                round_index=0,
                kind=KIND_RANDOM_STANDALONE,
                ancestor_sequence=None,
                original_sequence=original,
                edited_positions=mutated_positions(original, cand),
                seed=s,
            )
        )

    note = ""
    if unfilled:
        note = (
            f"{len(unfilled)} of {len(counts)} standalone draws could not be filled with "
            f"a unique sequence at counts {sorted(unfilled)}; arm coverage is "
            f"{len(out)}/{len(counts)}. On a short peptide the distinct-sequence space "
            f"at these counts is small, so this is exhaustion, not an error."
        )
    return out, note


KIND_RANDOM_STREAM = "random_stream_draw"


def stream_count(
    *,
    original: str,
    slot: int,
    seed_base: object,
    weights: dict[int, float] | None = None,
) -> int | None:
    """How many substitutions draw number `slot` makes.

    `standalone_counts` draws the whole vector in ONE rng call sized by `n_slots`, so
    the plan for 32 draws and the plan for 64 draws share nothing -- not even their
    first draw. That is fine when the number of draws is decided in advance and fatal
    when it is not: extending a finished run from 32 to 64 would throw away the 32
    already paid for.

    Here the count is a pure function of `(seed_base, slot)`, so draw 7 is the same
    sequence whether the run asks for 8 draws or 800. A run can therefore be extended by
    appending slots, and any prefix of the stream is still an independent sample from
    the same count distribution -- which is what lets a prefix be read as a smaller run
    rather than as a biased piece of a larger one.
    """
    ceiling = budget_for(len(original))
    w = {k: v for k, v in (weights or STANDALONE_COUNT_WEIGHTS).items()
         if 1 <= k <= ceiling and v > 0}
    if not w:
        return None
    ks = sorted(w)
    total = sum(w[k] for k in ks)
    probs = [w[k] / total for k in ks]
    rng = np.random.default_rng(proposal_seed(seed_base, "stream_count", slot))
    return int(rng.choice(ks, p=probs))


def standalone_random_stream(
    *,
    original: str,
    n_draws: int,
    seed_base: object,
    start: int = 0,
    alphabet: tuple[str, ...] | None = None,
    weights: dict[int, float] | None = None,
    attempts_per: int = 48,
) -> tuple[list[Proposal], str]:
    """Draws `start` .. `start + n_draws` of the extendable random stream.

    Slots before `start` are re-planned (string operations only, no evaluation) purely
    to rebuild the set of sequences already drawn, so that a later slot never repeats an
    earlier one. Without that, two slots landing on the same sequence would be deduped
    by the journal and the run would quietly hold fewer distinct draws than it believes.
    Re-planning is what keeps the stream BOTH prefix-stable and duplicate-free.
    """
    alpha = alphabet or default_alphabet()
    seen = {original}
    out: list[Proposal] = []
    unfilled: list[int] = []

    for slot in range(start + n_draws):
        count = stream_count(
            original=original, slot=slot, seed_base=seed_base, weights=weights
        )
        if count is None:
            break
        got = None
        for attempt in range(attempts_per):
            s = proposal_seed(seed_base, "stream", slot, count, attempt)
            cand = random_at_count(original, count, seed=s, alphabet=alpha)
            if cand is None or cand in seen:
                continue
            got = (cand, s)
            break
        if got is None:
            if slot >= start:
                unfilled.append(count)
            continue
        cand, s = got
        seen.add(cand)
        if slot < start:
            continue
        out.append(
            Proposal(
                sequence=cand,
                arm=ARM_RANDOM,
                round_index=0,
                kind=KIND_RANDOM_STREAM,
                ancestor_sequence=None,
                original_sequence=original,
                edited_positions=mutated_positions(original, cand),
                seed=s,
                # The slot is the draw's identity in the stream, and it is what makes a
                # resumed or extended run line up with the one before it.
                slot=slot,
            )
        )

    note = ""
    if unfilled:
        note = (
            f"{len(unfilled)} of {n_draws} stream draws could not be filled with a "
            f"unique sequence at counts {sorted(unfilled)}; the stream yielded "
            f"{len(out)}/{n_draws}. On a short peptide the distinct-sequence space at "
            f"these counts is small, so this is exhaustion, not an error."
        )
    return out, note


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


@dataclass
class ArchiveEntry:
    sequence: str
    # Rank key, lower is better: the corrected interface energy for accepted candidates.
    objective: float | None
    # Continuous violation magnitude, lower is better; used only while nothing is valid.
    violation: float | None
    valid: bool
    round_index: int
    kind: str
    ancestor_sequence: str | None
    extra: dict = field(default_factory=dict)


class Archive:
    """The working set: several promising, sequence-diverse candidates.

    Two reasons it is not just "the best one so far":

    - A single winner collapses the search onto one basin, and the pilot's budget is far
      too small to be confident that basin is the right one.
    - Before anything is valid there is no energy to rank by. Keeping the smallest
      geometry violations lets the search move toward feasibility instead of stalling,
      which is `keep_infeasible` below.

    Diversity is enforced by Hamming distance between kept sequences, so six entries
    cannot be six neighbours of one sequence.
    """

    def __init__(self, capacity: int = 6, min_distance: int = 1, keep_infeasible: int = 3):
        self.capacity = capacity
        self.min_distance = min_distance
        self.keep_infeasible = keep_infeasible
        self.entries: list[ArchiveEntry] = []

    def add(self, entry: ArchiveEntry) -> None:
        self.entries = [e for e in self.entries if e.sequence != entry.sequence]
        self.entries.append(entry)

    def _select(self, pool: list[ArchiveEntry], k: int, key) -> list[ArchiveEntry]:
        """Greedy best-first with a diversity veto."""
        ranked = sorted([e for e in pool if key(e) is not None], key=key)
        chosen: list[ArchiveEntry] = []
        for e in ranked:
            if len(chosen) >= k:
                break
            if all(hamming(e.sequence, c.sequence) >= self.min_distance for c in chosen):
                chosen.append(e)
        # The veto is a real constraint, not a preference: a caller asking for a minimum
        # separation gets it, and a starved selection returns FEWER parents rather than
        # being padded with near-duplicates. Padding would quietly reinstate the
        # single-winner behaviour the archive exists to avoid -- six entries all one
        # substitution apart is one candidate wearing six hats.
        #
        # The one exception is an empty result: the next round needs something to edit
        # from, so when nothing clears the veto the single best entry is returned.
        if not chosen and ranked:
            chosen = [ranked[0]]
        return chosen

    def working_set(self) -> list[ArchiveEntry]:
        """The parents the next round edits from.

        Valid candidates ranked by objective come first. When fewer than the capacity
        are valid, the gap is filled with the smallest geometry violations -- explicitly
        labelled, because an infeasible parent is a stepping stone, not a result.
        """
        valid = [e for e in self.entries if e.valid]
        chosen = self._select(valid, self.capacity, key=lambda e: e.objective)
        if len(chosen) < self.capacity:
            room = min(self.keep_infeasible, self.capacity - len(chosen))
            infeasible = [e for e in self.entries if not e.valid]
            chosen += self._select(infeasible, room, key=lambda e: e.violation)
        return chosen

    def best_valid(self, k: int = 2) -> list[ArchiveEntry]:
        valid = [e for e in self.entries if e.valid]
        return self._select(valid, k, key=lambda e: e.objective)


def guided_round(
    *,
    original: str,
    archive: Archive,
    round_index: int,
    budget: int,
    seed_base: object,
    alphabet: tuple[str, ...] | None = None,
    exclude: set[str] | None = None,
) -> tuple[list[Proposal], str]:
    """One round of Arm B after the scan: local edits, one coordinated edit, restarts.

    The mix is ~5 local (1-2 positions), 1 coordinated (3-4 positions), 2 restarts at a
    budget of 8. Multi-position edits are not a luxury: two substitutions can help
    together while each is neutral or harmful alone, and a strictly single-position
    search cannot reach those at all.

    Edit sizes are clipped to the peptide's own budget, so a 5-mer (ceiling 2) never
    receives a 4-position "coordinated" proposal it could not legally hold.
    """
    alpha = alphabet or default_alphabet()
    seen = set(exclude or set())
    seen.add(original)
    parents = archive.working_set()
    cap = budget_for(len(original))

    if not parents:
        # The archive can be empty for a reason that is not "nothing good was found":
        # if the reference prediction itself failed technically there is no entry to
        # rank, and a round that proposed nothing would stall the parent for the rest of
        # the run. The task says to start from the original sequence, so that is the
        # fallback -- labelled infeasible, since nothing has been shown about it.
        parents = [
            ArchiveEntry(
                sequence=original,
                objective=None,
                violation=None,
                valid=False,
                round_index=-1,
                kind="original_fallback",
                ancestor_sequence=None,
            )
        ]

    n_restart = min(2, budget)
    n_coord = 1 if budget - n_restart >= 1 and cap >= 3 else 0
    n_local = budget - n_restart - n_coord

    out: list[Proposal] = []

    def _take(p: Proposal | None) -> bool:
        if p is None or p.sequence in seen:
            return False
        seen.add(p.sequence)
        out.append(p)
        return True

    # Local edits, cycling through the archive so one parent cannot monopolise a round.
    attempt = 0
    while sum(1 for p in out if p.kind == KIND_LOCAL) < n_local and attempt < n_local * 24:
        if not parents:
            break
        parent = parents[attempt % len(parents)]
        n_pos = 1 if (attempt % 2 == 0 or cap < 2) else 2
        _take(
            propose_edit(
                ancestor=parent.sequence,
                original=original,
                n_positions=min(n_pos, cap),
                arm=ARM_GUIDED,
                round_index=round_index,
                kind=KIND_LOCAL,
                seed=proposal_seed(seed_base, round_index, "local", attempt),
                alphabet=alpha,
            )
        )
        attempt += 1

    # One coordinated edit over 3-4 positions, from the best parent available.
    attempt = 0
    while n_coord and sum(1 for p in out if p.kind == KIND_COORDINATED) < n_coord and attempt < 24:
        if not parents:
            break
        parent = parents[0]
        n_pos = min(3 + (attempt % 2), cap)
        _take(
            propose_edit(
                ancestor=parent.sequence,
                original=original,
                n_positions=n_pos,
                arm=ARM_GUIDED,
                round_index=round_index,
                kind=KIND_COORDINATED,
                seed=proposal_seed(seed_base, round_index, "coord", attempt),
                alphabet=alpha,
            )
        )
        attempt += 1

    # Restarts around the original.
    attempt = 0
    while sum(1 for p in out if p.kind == KIND_RESTART) < n_restart and attempt < 24 * max(1, n_restart):
        _take(
            propose_restart(
                original=original,
                round_index=round_index,
                seed=proposal_seed(seed_base, round_index, "restart", attempt),
                alphabet=alpha,
            )
        )
        attempt += 1

    # Backfill with local edits if a category ran dry, so the round still spends its
    # budget; if the whole neighbourhood is exhausted, report the shortfall instead.
    attempt = 0
    while len(out) < budget and attempt < budget * 32:
        if not parents:
            break
        parent = parents[attempt % len(parents)]
        _take(
            propose_edit(
                ancestor=parent.sequence,
                original=original,
                n_positions=min(1 + (attempt % 2), cap),
                arm=ARM_GUIDED,
                round_index=round_index,
                kind=KIND_LOCAL,
                seed=proposal_seed(seed_base, round_index, "backfill", attempt),
                alphabet=alpha,
            )
        )
        attempt += 1

    note = ""
    if len(out) < budget:
        note = (
            f"round {round_index} proposed {len(out)}/{budget} unique sequences; the "
            f"permitted neighbourhood is exhausted for this parent"
        )
    return out, note
