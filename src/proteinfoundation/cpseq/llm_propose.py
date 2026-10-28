"""An instruction-tuned text model as the proposal mechanism, in place of random draws.

What this replaces, and what it must not change
-----------------------------------------------
`search.py`'s proposers pick positions and residues from an rng. This module picks them
with `meta-llama/Llama-3.1-8B-Instruct`, shown a description of the complex and the
candidates already scored. **Nothing else about the experiment moves**: the same
substitution budget measured against the same original sequence, the same alphabet, the
same fold depth, the same frozen geometry gate, the same evaluator, and the same
cross-fitted selection. An arm that changed the proposer *and* the measurement could not
attribute a difference to either.

Four rules this module enforces, each because the obvious implementation would have
produced a comparison that looks like a result and is not:

1. **A model's output is a suggestion, never a candidate.** Every proposed sequence is
   checked for length, alphabet, budget against the ORIGINAL, and novelty, and a failure
   is counted by reason rather than quietly dropped. `rejections` on the returned record
   is the audit trail. A model that mostly emits illegal strings and a model that mostly
   emits good ones must not look alike in the log.

2. **A short round is reported, never padded.** When the model cannot supply the
   requested number of legal, novel sequences within its attempt limit, the round returns
   what it has and says so. Backfilling the gap with random draws would make the LLM arm
   a silent hybrid of itself and the control it is being compared against -- and the
   fewer sequences it proposed, the more of the control it would contain.

3. **The proposer never sees the measurement folds.** This is what keeps the cross-fitted
   comparison meaningful. `README_CPSEQ_PEPBENCH_SCALEUP.md` §10 reports that the whole
   -4.66 REU best-of-8 advantage vanished under cross-fitting, so the honest number is
   the held-out one: candidates are chosen on folds `0..depth/2` and measured on
   `depth/2..depth`. An adaptive proposer shown the measurement half would be choosing
   what to propose using the folds its proposals are later scored on, which reinstates
   exactly the selection bias cross-fitting exists to remove. `feedback_folds="select"`
   is therefore the default, and `"all"` exists only to make the leak measurable.

4. **Re-proposing an evaluated sequence is refused, not cached.** The evaluator caches on
   `input_hash`, so a duplicate sequence would silently reuse the folds already on disk --
   including its measurement half. A model that re-emitted a sequence it had seen score
   well would be laundering a measured number into a "new" proposal. Duplicates are
   rejected as `already_evaluated`.

The model is read from a local path and never downloaded inside a job.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import statistics as st
import time
from dataclasses import dataclass, field
from pathlib import Path

from proteinfoundation.cpseq.candidates import AA20
from proteinfoundation.cpseq.evaluate import SCORING_PROTOCOL_VERSION
from proteinfoundation.cpseq.interactions import render_complex, render_pocket
from proteinfoundation.cpseq.search import (
    ARM_LLM,
    KIND_LLM,
    Archive,
    ArchiveEntry,
    Proposal,
    budget_for,
    default_alphabet,
    n_mutations,
    proposal_seed,
    within_budget,
)

# The cluster's shared copy. Checked for existence before use; nothing here downloads.
DEFAULT_MODEL_PATH = "/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Decoding. Sampling rather than greedy because one round needs a BATCH of distinct
# sequences, and greedy decoding returns the same string every time. Recorded on the
# record so a run's proposals can be reproduced.
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 700
# Attempts per round. Each attempt is a fresh sampling pass at a different seed; the
# round stops as soon as it has enough legal novel sequences.
DEFAULT_MAX_ATTEMPTS = 4

# How much candidate history goes in the prompt. The scored history is ranked and
# truncated rather than dumped: an 8B model given 200 rows attends to none of them.
DEFAULT_HISTORY_BEST = 8
DEFAULT_HISTORY_WORST = 4
DEFAULT_HISTORY_INTERFACES = 2

# The population. Four elites x four offspring is 16 candidates per generation, which
# matches the random arm's draws per round so the two arms spend the same number of folds
# per generation -- the comparison is between proposers, not between budgets.
DEFAULT_N_ELITES = 4
DEFAULT_OFFSPRING_PER_ELITE = 4
# The Hamming veto between elites. One substitution apart is a different sequence but not
# a different hypothesis, so two elites must differ in at least two positions.
DEFAULT_ELITE_MIN_DISTANCE = 2
# Slots an elite set may spend on candidates that have not passed the gate, when fewer
# than `n_elites` have. An infeasible elite is a stepping stone toward feasibility, not a
# result, and it is labelled as such in the prompt.
DEFAULT_ELITE_INFEASIBLE = 2


# --------------------------------------------------------------------------------------
# What the model is told
# --------------------------------------------------------------------------------------


@dataclass
class CandidateRecord:
    """One already-scored candidate, as the prompt sees it."""

    sequence: str
    n_mutations: int
    valid: bool
    # The ranking key, and what the prompt shows: the MEAN post-relax interface dG over
    # this sequence's valid folds. The mean rather than the minimum because the minimum
    # is a record of which fold got lucky -- the measured seed-to-seed spread on this
    # cohort is 4.3 REU median, larger than most differences being chased -- and the mean
    # rather than the median because the mean among valid folds is the stated objective
    # the confirmation stage reports, so selection and reporting rank by the same number.
    energy: float | None = None
    # Kept alongside for comparison, not for ranking.
    energy_median: float | None = None
    # Spread of this candidate's own valid folds, max minus min. Shown per candidate
    # because one global sentence about a 4.3 REU median spread does not say which of
    # the candidates in front of the model is the uncertain one: a -38 REU mean with a
    # 0.4 REU spread and a -38 REU mean with an 18 REU spread are different claims, and
    # the prompt was presenting them identically.
    energy_spread: float | None = None
    violation: float | None = None
    reasons: list[str] = field(default_factory=list)
    # The per-position interface table, for the few candidates worth showing in full.
    interaction_summary: dict | None = None
    n_folds: int = 0
    # Valid folds out of folds attempted. Shown because a -40 REU mean over 1 of 6 folds
    # and a -40 REU mean over 6 of 6 are not the same candidate, and no minimum validity
    # ratio is imposed -- so the ratio has to be visible rather than enforced.
    n_valid_folds: int = 0

    def one_line(self, original: str, *, show_substitutions: bool = True) -> str:
        """`show_substitutions` spells out the differences as `D1Y`.

        Switched off under the edit-list output format. That notation is anchored on the
        ORIGINAL while the model is being asked to name residues in the sequence it is
        editing, and a prompt carrying eighteen such tokens was measured producing the
        dominant failure of that format: 26 of 58 mis-cited substitutions named the
        residue the original holds at that position, against roughly 5% expected by
        chance. The count stays, because how far a candidate sits from the original is
        real information; only the notation goes, and the sequence itself is printed on
        the same line anyway.
        """
        subs = ",".join(
            f"{original[i]}{i + 1}{self.sequence[i]}"
            for i in range(min(len(original), len(self.sequence)))
            if original[i] != self.sequence[i]
        ) or "(none)"
        if self.valid and self.energy is not None:
            spread = (
                f", spread {self.energy_spread:.1f}"
                if self.energy_spread is not None and self.n_valid_folds > 1
                else ""
            )
            verdict = (
                f"PASSED gate, mean energy {self.energy:+.1f} REU{spread} "
                f"(lower is better)"
            )
        elif self.valid:
            verdict = "PASSED gate, energy unavailable"
        else:
            why = "; ".join(self.reasons[:2]) if self.reasons else "geometry gate"
            v = f", violation {self.violation:.2f}" if self.violation is not None else ""
            verdict = f"REJECTED{v} -- {why}"
        folds = (
            f" [{self.n_valid_folds}/{self.n_folds} fold(s) valid]"
            if self.n_folds else ""
        )
        detail = f"{self.n_mutations} sub: {subs}" if show_substitutions else (
            f"{self.n_mutations} sub"
        )
        return f"  {self.sequence}  ({detail})  {verdict}{folds}"


SYSTEM_PROMPT = (
    "You are an expert in peptide chemistry and structure-based design, proposing "
    "sequence substitutions for a head-to-tail cyclic peptide bound to a protein "
    "target.\n\n"
    "You will be given the original linear peptide sequence, a hard substitution "
    "budget, a per-residue description of the interface from PLIP and Biopython, and "
    "every candidate already built and scored.\n\n"
    "Reason about the interface, then propose new sequences. Useful moves include: "
    "filling a pocket that a small residue leaves empty; removing a side chain that "
    "overlaps the receptor; adding a hydrogen-bond or salt-bridge partner opposite a "
    "complementary receptor residue; replacing a buried charged residue with a "
    "hydrophobic one; and keeping residues that already make good contacts.\n\n"
    "Hard rules you must obey:\n"
    "- Substitutions only. The sequence length must never change.\n"
    "- Use only the 20 standard amino acids, as single uppercase letters.\n"
    "- Never exceed the stated substitution budget, counted against the ORIGINAL "
    "sequence (not against any candidate you were shown).\n"
    "- Never repeat a sequence that has already been scored.\n"
    "- The budget is a ceiling, not a target. A one- or two-substitution proposal is "
    "entirely acceptable and often better than a large change.\n\n"
    "Output format. First a short block of reasoning, at most 120 words. Then the "
    "proposals, one per line, each as the complete sequence followed by a brief reason:\n"
    "PROPOSE: <SEQUENCE> | <reason>\n"
    "Emit exactly the number of PROPOSE lines requested and nothing after the last one."
)


def single_substitution_pairs(
    history: list[CandidateRecord], *, limit: int = 8
) -> list[tuple[CandidateRecord, CandidateRecord, int, str, str]]:
    """Scored candidates that differ from each other at exactly one position.

    These are the only entries in the history from which the effect of a single
    substitution can be read off. Everywhere else, two candidates differ at several
    positions and their energy difference cannot be attributed to any one of them -- so
    presenting such a difference as evidence that a particular substitution helped is a
    mistake the prompt should not invite, and naming the clean pairs is the cheapest way
    to not invite it.

    Only pairs where both sides have an energy are returned, ordered by the size of the
    difference so the informative ones come first. The pair is reported, never the
    conclusion: a 2 REU difference on one-fold means is noise and the prompt says so
    beside it.
    """
    scored = [h for h in history if h.energy is not None and len(h.sequence) > 0]
    out: list[tuple[CandidateRecord, CandidateRecord, int, str, str]] = []
    for i in range(len(scored)):
        for j in range(i + 1, len(scored)):
            a, b = scored[i], scored[j]
            if len(a.sequence) != len(b.sequence):
                continue
            diff = [
                k for k, (x, y) in enumerate(zip(a.sequence, b.sequence)) if x != y
            ]
            if len(diff) != 1:
                continue
            k = diff[0]
            # Order each pair better-first, so the arrow always points at the change
            # that improved things.
            lo, hi = (a, b) if a.energy <= b.energy else (b, a)
            out.append((lo, hi, k + 1, hi.sequence[k], lo.sequence[k]))
    out.sort(key=lambda t: -(t[1].energy - t[0].energy))
    return out[:limit]


def render_single_substitution_pairs(
    history: list[CandidateRecord], *, limit: int = 8
) -> str:
    pairs = single_substitution_pairs(history, limit=limit)
    if not pairs:
        return ""
    lines = [
        "SINGLE-SUBSTITUTION COMPARISONS (pairs of already-scored candidates that "
        "differ at exactly ONE position -- the only places where one substitution's "
        "effect can be read directly)"
    ]
    for better, worse, pos, frm, to in pairs:
        gain = worse.energy - better.energy
        spread = max(
            better.energy_spread or 0.0, worse.energy_spread or 0.0
        )
        caveat = (
            "  [within the spread of these candidates' own folds -- treat as no "
            "difference]" if gain <= spread else ""
        )
        lines.append(
            f"  position {pos}: {frm} -> {to} improved by {gain:.1f} REU "
            f"({worse.sequence} {worse.energy:+.1f} -> {better.sequence} "
            f"{better.energy:+.1f}){caveat}"
        )
    lines.append(
        "Every other pair of candidates differs at several positions at once, so a "
        "difference between them does NOT show that any single substitution helped."
    )
    return "\n".join(lines)


def shuffle_outcome_associations(
    history: list[CandidateRecord], *, seed: int
) -> list[CandidateRecord]:
    """The same candidates, each wearing another one's outcome. A control, not a feature.

    This exists to answer a question the fold-free prompt measurements cannot: does the
    scored history carry information the proposals use, or is its presence enough? Run an
    arm on this and fold what it proposes -- if real pairings beat shuffled ones, the
    history is doing work even though individual proposals rarely repeat.

    Only the outcome fields move. Each sequence keeps its own structure summary, because
    that is a fact about the sequence rather than an outcome, and because a summary
    moved to another sequence would show a table whose peptide line disagrees with the
    candidate it is attached to -- a tell the model could detect without understanding
    anything. The remaining tell is that a candidate labelled as having passed may show a
    table with an overlap in it; that is inherent to shuffling associations and is the
    reason this is a control arm rather than a prompt variant anyone should ship.
    """
    import random as _random

    if len(history) < 2:
        return list(history)
    fields = (
        "valid", "energy", "energy_median", "energy_spread", "violation",
        "reasons", "n_folds", "n_valid_folds",
    )
    outcomes = [
        {
            f: (list(getattr(h, f)) if f == "reasons" else getattr(h, f))
            for f in fields
        }
        for h in history
    ]
    order = list(range(len(history)))
    rng = _random.Random(seed)
    for _ in range(64):
        rng.shuffle(order)
        if all(o != i for i, o in enumerate(order)):
            break
    out: list[CandidateRecord] = []
    for i, h in enumerate(history):
        donor = outcomes[order[i]]
        out.append(
            CandidateRecord(
                sequence=h.sequence,
                n_mutations=h.n_mutations,
                interaction_summary=h.interaction_summary,
                **{f: donor[f] for f in fields},
            )
        )
    return out


# Measured on the ES pilot prompts, twelve complexes x three samples, same seeds both
# ways: asking for an edit list rather than a whole sequence raised usable proposals from
# 0.528 to 0.717 of those requested (+0.189 paired, better on 23 of 30 cells, p=0.0052).
# It removes two failure classes outright -- 63 wrong-length answers and 21 repeats of
# sequences the prompt had just listed as already scored, both to zero -- in slightly
# fewer output tokens. OUTPUT_FORMAT_EDITS is therefore the default. The old format is
# kept because every run before 2026-10-28 used it, and a run must not be continued under
# the other one: the proposals are not drawn from the same process.
OUTPUT_FORMAT_EDITS = "edits"
OUTPUT_FORMAT_SEQUENCE = "sequence"
OUTPUT_FORMATS = (OUTPUT_FORMAT_EDITS, OUTPUT_FORMAT_SEQUENCE)
DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMAT_EDITS

SYSTEM_PROMPT_EDITS = (
    SYSTEM_PROMPT.split("Output format.")[0]
    + "Output format. First a short block of reasoning, at most 120 words. Then the "
    "proposals, one per line. Each proposal is a LIST OF SUBSTITUTIONS, not a "
    "sequence:\n"
    "EDIT: <residue currently there><position><new residue>, ... | <brief reason>\n"
    "For example `EDIT: K7R, L9M | add a salt bridge at 7 and fill the pocket at 9`.\n"
    "Positions are numbered from 1 along the sequence you were told to improve, which "
    "is listed for you position by position. Take the residue you name from THAT list. "
    "Do not take it from the original sequence: the original is used only to count the "
    "budget, and a residue copied from it will not match, which discards the "
    "proposal.\n"
    "Do not write out the full sequence. It is built from your edits automatically, so "
    "you never need to count letters.\n"
    "Emit exactly the number of EDIT lines requested and nothing after the last one."
)


def system_prompt_for(output_format: str) -> str:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"unknown output format {output_format!r}")
    return (
        SYSTEM_PROMPT_EDITS
        if output_format == OUTPUT_FORMAT_EDITS else SYSTEM_PROMPT
    )


def render_numbered_sequence(seq: str) -> str:
    """The sequence being edited, laid out so the position-to-residue map is given.

    Residue before number, matching the order an EDIT line asks for, so the model reads
    `Y1` and writes `Y1F` rather than transposing a `1Y` it was shown.
    """
    cells = "   ".join(f"{aa}{i + 1}" for i, aa in enumerate(seq))
    return (
        f"THE SEQUENCE YOU ARE EDITING, POSITION BY POSITION\n  {cells}\n"
        f"Every residue you name in an EDIT line must come from THIS list."
    )


def build_user_prompt(
    *,
    original: str,
    budget: int,
    n_requested: int,
    reference_summary: dict | None = None,
    history: list[CandidateRecord] | None = None,
    alphabet: tuple[str, ...] | None = None,
    pdb_id: str | None = None,
    pocket_note: str | None = None,
    history_best: int = DEFAULT_HISTORY_BEST,
    history_worst: int = DEFAULT_HISTORY_WORST,
    history_interfaces: int = DEFAULT_HISTORY_INTERFACES,
    feedback_note: str | None = None,
    parent_candidate: CandidateRecord | None = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> str:
    """The user turn. Pure string assembly, so it is testable without a model.

    `parent_candidate` makes this an offspring prompt: the model is told to edit ONE
    named elite rather than the original, which is what gives the arm an ancestry the
    record can state. The substitution budget is still counted against the ORIGINAL --
    an offspring prompt changes where the search starts, never how far it may go, or two
    generations of "three edits each" would walk off the budget the arms share.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"unknown output format {output_format!r}")
    edits_format = output_format == OUTPUT_FORMAT_EDITS
    # Under the edit-list format the prompt must not model original-anchored `D1Y`
    # notation anywhere, because that is the notation the model is asked to emit against
    # a DIFFERENT sequence. See `CandidateRecord.one_line`.
    show_subs = not edits_format
    alpha = alphabet or default_alphabet()
    excluded = sorted(set(AA20) - set(alpha))
    L = len(original)
    parts: list[str] = []

    # The exclusion is stated as a rule about SUBSTITUTIONS, and it names the positions
    # where the original already holds an excluded residue. Saying only "C is not
    # allowed" about a sequence that contains a C is a contradiction the model cannot
    # satisfy: it either disobeys the rule or deletes a residue nobody asked it to
    # touch. Observed on `lnr_2v8x` (original RIIYDRKFLMECRN) before this was fixed.
    native_excluded = sorted(
        {f"{original[i]}{i + 1}" for i in range(L) if original[i] not in set(alpha)}
    )
    if excluded:
        rule = (
            f"ALLOWED RESIDUES FOR SUBSTITUTIONS: {''.join(alpha)}\n"
            f"You must not INTRODUCE {' or '.join(excluded)} at any position "
            f"(a free cysteine could form an unintended disulfide and change the "
            f"molecule's chemistry)."
        )
        if native_excluded:
            rule += (
                f" The original already has {', '.join(native_excluded)}; KEEP "
                f"{'that residue' if len(native_excluded) == 1 else 'those residues'} "
                f"as it is unless you deliberately substitute it for an allowed one. "
                f"Keeping it is not a substitution and costs no budget."
            )
    else:
        rule = f"ALLOWED RESIDUES FOR SUBSTITUTIONS: {''.join(alpha)}"

    parts.append(
        f"TARGET: {pdb_id or 'unknown'}\n"
        f"ORIGINAL LINEAR PEPTIDE: {original}  (length {L})\n"
        f"TOPOLOGY: head-to-tail cyclic -- residue {L} is bonded to residue 1, so both "
        f"ends are constrained and there are no free termini.\n"
        f"SUBSTITUTION BUDGET: at most {budget} positions may differ from the original "
        f"(that is floor(L/2); it is a ceiling, not a target).\n"
        + rule
    )

    parts.append(
        "HOW CANDIDATES ARE JUDGED\n"
        "A candidate is folded in complex with the target, then screened by a fixed "
        "geometry gate: the ring must close, the peptide must not overlap itself, and "
        "the peptide must not overlap the receptor. Only candidates that pass are "
        "ranked, by computed interface energy in Rosetta REU, where LOWER IS BETTER.\n"
        "Peptide-receptor overlap is by far the most common reason for rejection, so "
        "the 'ovl' column below is the single most important thing to act on."
    )

    if reference_summary is not None:
        parts.append(
            "INTERFACE OF THE UNMUTATED REFERENCE (the cyclic peptide with the original "
            "sequence; this is the baseline every candidate is compared against)\n"
            "Columns: pos=position, aa=residue, ss=approximate secondary structure, "
            "burial=fraction of side-chain surface buried by the receptor (1.0 = fully "
            "enclosed), dmin=closest approach to the receptor in angstroms, "
            "ctc=receptor atoms within 4.5 A, ovl=steric overlaps (any is fatal).\n"
            + render_complex(reference_summary, title="reference")
        )
        pocket = render_pocket(reference_summary)
        if pocket:
            parts.append(pocket)
    if pocket_note:
        parts.append(pocket_note)

    hist = list(history or [])
    if hist:
        passed = sorted(
            [h for h in hist if h.valid and h.energy is not None],
            key=lambda h: h.energy,
        )
        others = [h for h in hist if not (h.valid and h.energy is not None)]
        others.sort(key=lambda h: (h.violation if h.violation is not None else 1e9))
        shown = passed[:history_best] + others[:history_worst]
        block = [
            f"CANDIDATES ALREADY SCORED ({len(hist)} total, "
            f"{len(passed)} passed the gate; showing {len(shown)})"
        ]
        if passed:
            block.append("Passed, best energy first:")
            block += [h.one_line(original, show_substitutions=show_subs)
                      for h in passed[:history_best]]
        if others[:history_worst]:
            block.append("Rejected, smallest geometry violation first:")
            block += [h.one_line(original, show_substitutions=show_subs)
                      for h in others[:history_worst]]
        parts.append("\n".join(block))

        # Full interface tables for the few best candidates. This is where the model can
        # see what a *better* interface than the reference actually looks like.
        with_iface = [h for h in passed if h.interaction_summary is not None]
        for h in with_iface[:history_interfaces]:
            parts.append(
                f"INTERFACE OF SCORED CANDIDATE {h.sequence} "
                f"(energy {h.energy:+.1f} REU)\n"
                + render_complex(h.interaction_summary, title=h.sequence)
            )
        # A rejected candidate's interface is often more instructive than a good one's,
        # because it shows the overlap that caused the rejection.
        bad_iface = [
            h for h in others if h.interaction_summary is not None and not h.valid
        ]
        for h in bad_iface[:1]:
            parts.append(
                f"INTERFACE OF A REJECTED CANDIDATE {h.sequence} "
                f"(this one failed the gate -- see the ovl column)\n"
                + render_complex(h.interaction_summary, title=h.sequence)
            )
        # Placed after the tables and before the do-not-repeat list: it is the one block
        # that states a relationship between a change and an outcome rather than listing
        # a state, which is what the history is for in the first place.
        pair_block = render_single_substitution_pairs(hist)
        if pair_block:
            parts.append(pair_block)
        parts.append(
            "DO NOT PROPOSE ANY OF THESE ALREADY-SCORED SEQUENCES:\n"
            + "\n".join(f"  {h.sequence}" for h in hist)
        )
    else:
        parts.append(
            "CANDIDATES ALREADY SCORED: none. This is the first round, so propose a "
            "spread of distinct hypotheses rather than variations on one idea."
        )

    if feedback_note:
        parts.append(feedback_note)

    if parent_candidate is not None:
        p = parent_candidate
        psubs = ",".join(
            f"{original[i]}{i + 1}{p.sequence[i]}"
            for i in range(min(L, len(p.sequence)))
            if original[i] != p.sequence[i]
        ) or "(none -- it is the original)"
        if p.energy is not None:
            standing = (
                f"It passed the gate on {p.n_valid_folds} of {p.n_folds} fold(s), "
                f"mean interface energy {p.energy:+.1f} REU."
            )
        else:
            why = "; ".join(p.reasons[:2]) if p.reasons else "the geometry gate"
            standing = (
                f"It has not passed the gate yet ({why}); it is being carried forward "
                f"because its geometry is the closest to feasible so far. Your edits "
                f"should FIX that, not add to it."
            )
        if edits_format:
            worded = "; ".join(
                f"position {i + 1} holds {p.sequence[i]} here, where the original "
                f"holds {original[i]}"
                for i in range(min(L, len(p.sequence)))
                if original[i] != p.sequence[i]
            ) or "nowhere -- it is the original"
            psubs = worded + "."
        parts.append(
            f"SEQUENCE TO IMPROVE -- edit THIS one, not the original\n"
            f"  {p.sequence}\n"
            f"It differs from the original at {p.n_mutations} position(s): {psubs}\n"
            f"{standing}\n"
            f"Propose variants OF THIS SEQUENCE: copy it, then change a small number of "
            f"positions (one to three is usually right). Keep the substitutions that "
            f"earned its standing unless you have a specific reason to revert one. "
            f"The budget of {budget} is still counted against the ORIGINAL "
            f"{original}, so it caps the TOTAL number of differences from the original, "
            f"your edits and the ones it already carries together."
        )
        if p.interaction_summary is not None:
            parts.append(
                f"INTERFACE OF THE SEQUENCE TO IMPROVE {p.sequence}\n"
                + render_complex(p.interaction_summary, title=p.sequence)
            )

    start_from = parent_candidate.sequence if parent_candidate is not None else original
    if edits_format:
        parts.append(
            f"TASK\nPropose exactly {n_requested} distinct sets of substitutions to "
            f"{start_from}, the sequence named above as the one to improve.\n"
            + render_numbered_sequence(start_from) + "\n"
            f"State each proposal as substitutions only -- do not write out a "
            f"sequence:\n"
            f"EDIT: <residue currently there><position><new residue>, ... | "
            f"<brief reason>\n"
            f"Vary the number of substitutions across your proposals; one to three is "
            f"usually right, and include at least one proposal with only one or two. "
            f"Each proposal must differ from the others and from every sequence already "
            f"scored.\n"
            f"Budget: the finished sequence may differ from the original at no more "
            f"than {budget} positions in total, counting the differences {start_from} "
            f"already carries. That count is checked in software -- state the edits you "
            f"intend and it will be enforced, so do not work it out from the original.\n"
            f"End with exactly {n_requested} lines of the form:\n"
            f"EDIT: <residue currently there><position><new residue>, ... | "
            f"<brief reason>"
        )
        return "\n\n".join(parts)
    parts.append(
        f"TASK\nPropose exactly {n_requested} new sequences of length {L}, each built by "
        f"editing {start_from}, each within "
        f"{budget} substitutions of the original {original}, each distinct from the "
        f"other proposals and from every sequence already scored. Vary the number of "
        f"substitutions across your proposals; include at least one proposal with only "
        f"one or two substitutions.\n"
        f"Every sequence must be EXACTLY {L} letters long. Build each one by copying "
        f"{start_from} and changing only the positions you intend to change, then count "
        f"the letters before you write the line -- a sequence of any other length is "
        f"discarded.\n"
        f"End with exactly {n_requested} lines of the form:\n"
        f"PROPOSE: <SEQUENCE> | <reason>"
    )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Reading the model's answer
# --------------------------------------------------------------------------------------

# `PROPOSE:` is what the prompt asks for, but the model also answers as a numbered list
# of `9. Proposal: <SEQ> | <reason>`, and "PROPOSAL" does not contain "PROPOSE", so the
# strict pattern matched nothing at all. Measured on `lnr_6seo` round 1: attempt 0 emitted
# ten well-formed proposals in that shape, parsed 0 of them, and a 13 s generation was
# discarded -- one of that complex's three attempts, for a difference in punctuation.
#
# Accepting the variant is parsing, not repair: the label is explicit and the sequence is
# written out in full. What is NOT accepted is a bare sequence sitting in prose, because
# the reasoning text names sequences it is arguing against and harvesting those would
# fold candidates the model did not propose.
_LABEL_RE = re.compile(r"\b(?:PROPOSE|PROPOSALS?|PROPOSED)\s*:", re.IGNORECASE)
_PROPOSE_RE = re.compile(
    r"^[ \t]*(?:\d{1,2}[.)]\s*)?(?:PROPOSE|PROPOSALS?|PROPOSED)\s*:\s*"
    r"([A-Za-z]{2,})[ \t]*(?:\|\s*(.*))?$",
    re.MULTILINE | re.IGNORECASE,
)
# Explicit substitution notation, e.g. "L3K" or "Leu3Lys" is not accepted; only the
# compact one-letter form, because the three-letter form is ambiguous to tokenise and the
# prompt does not ask for it.
_SUB_RE = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,3})([ACDEFGHIKLMNPQRSTVWY])\b")


def apply_substitutions(original: str, subs: list[tuple[str, int, str]]) -> str | None:
    """Apply 1-based (from, pos, to) substitutions; None if any `from` disagrees."""
    out = list(original)
    for frm, pos, to in subs:
        i = pos - 1
        if not (0 <= i < len(out)) or out[i] != frm:
            return None
        out[i] = to
    return "".join(out)


def parse_completion(
    text: str, *, original: str, allow_substitution_notation: bool = True
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Pull candidate sequences out of a completion.

    Returns `(candidates, malformed)` where each candidate is `(sequence, reason)`.
    Nothing is validated here beyond shape -- `screen_candidates` owns the legality
    rules, so that parsing and validation can be tested separately.
    """
    L = len(original)
    found: list[tuple[str, str]] = []
    malformed: list[dict] = []

    for m in _PROPOSE_RE.finditer(text):
        raw, reason = m.group(1).strip().upper(), (m.group(2) or "").strip()
        if len(raw) == L and set(raw) <= set(AA20):
            found.append((raw, reason))
        else:
            malformed.append(
                {
                    "text": m.group(0)[:120],
                    "reason": (
                        f"length {len(raw)} != {L}"
                        if len(raw) != L
                        else f"non-standard letters: {''.join(sorted(set(raw) - set(AA20)))}"
                    ),
                }
            )

    # Fall back to substitution notation only for PROPOSE lines that held no usable
    # sequence. A model that answers in the requested format is never reinterpreted.
    if allow_substitution_notation and not found:
        for line in text.splitlines():
            # The same label set the whole-sequence pattern accepts, so a numbered
            # `9. Proposal: D5A,K7R` is not silently ineligible for the fallback too.
            if not _LABEL_RE.search(line):
                continue
            subs = [
                (a, int(p), b) for a, p, b in _SUB_RE.findall(line) if 1 <= int(p) <= L
            ]
            if not subs:
                continue
            seq = apply_substitutions(original, subs)
            if seq is not None:
                found.append((seq, line.strip()[:160]))
            else:
                malformed.append(
                    {"text": line.strip()[:120], "reason": "substitution cites wrong source residue"}
                )
    return found, malformed


_EDIT_LINE_RE = re.compile(
    r"^[ \t]*(?:\d{1,2}[.)]\s*)?EDITS?\s*:\s*(?P<body>[^|\n]*?)\s*"
    r"(?:\|\s*(?P<reason>.*))?$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_edit_completion(
    text: str, *, base: str
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Read EDIT lines and build the sequences they describe.

    Same return shape as `parse_completion`, so both output formats feed the identical
    screen and the legality rules stay in one place.

    A line whose named residue disagrees with `base` is malformed rather than applied.
    That check is the reason the format is safe to use: a disagreement means the model has
    lost track of which sequence it is editing, and silently applying the edit would
    change a position it did not mean to touch. Measured on the pilot prompts, 45% of
    those disagreements named the residue the ORIGINAL holds at that position, which is
    why the prompt no longer shows original-anchored notation anywhere.
    """
    found: list[tuple[str, str]] = []
    malformed: list[dict] = []
    for m in _EDIT_LINE_RE.finditer(text):
        body, reason = m.group("body") or "", (m.group("reason") or "").strip()
        subs = [(a, int(pos), b) for a, pos, b in _SUB_RE.findall(body.upper())]
        if not subs:
            malformed.append(
                {"text": m.group(0)[:120], "reason": "no substitution parsed"}
            )
            continue
        if any(not (1 <= pos <= len(base)) for _a, pos, _b in subs):
            malformed.append(
                {"text": m.group(0)[:120], "reason": "position outside the sequence"}
            )
            continue
        seq = apply_substitutions(base, subs)
        if seq is None:
            malformed.append(
                {"text": m.group(0)[:120],
                 "reason": "substitution cites wrong source residue"}
            )
            continue
        found.append((seq, reason))
    return found, malformed


def screen_candidates(
    candidates: list[tuple[str, str]],
    *,
    original: str,
    alphabet: tuple[str, ...],
    already: set[str],
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Apply every legality rule, keeping a reason for each rejection.

    `already` must contain every sequence the run has evaluated plus the original and
    anything accepted earlier in this round. Mutating it is the caller's job.
    """
    cap = budget_for(len(original))
    alpha = set(alphabet)
    accepted: list[tuple[str, str]] = []
    rejected: list[dict] = []
    seen_here: set[str] = set()

    for seq, reason in candidates:
        # The alphabet constrains what a substitution may INTRODUCE, not what the
        # original sequence already contains. `search._draw_residue` draws replacement
        # residues from the alphabet and never touches an unedited position, so the
        # random arm preserves a native cysteine in every one of its draws -- the
        # constant is named DEFAULT_EXCLUDED_FROM_RANDOM for exactly that reason.
        #
        # Measured the hard way on `lnr_2v8x`, whose original is RIIYDRKFLMECRN with a
        # native C at position 12: testing `set(seq) <= alpha` rejected all 24 of the
        # model's proposals across 4 attempts, every one of them for keeping a residue
        # the original already had and the control arm keeps too. That would have made
        # arm D propose nothing at all for any parent carrying a cysteine, and the log
        # would have read as a model that cannot follow instructions.
        introduced = {
            seq[i] for i in range(min(len(seq), len(original))) if seq[i] != original[i]
        }
        if len(seq) != len(original):
            why = f"length {len(seq)} != {len(original)}"
        elif not set(seq) <= set(AA20):
            why = f"non-standard letters: {''.join(sorted(set(seq) - set(AA20)))}"
        elif not introduced <= alpha:
            why = (
                f"substitution introduces excluded residue(s): "
                f"{''.join(sorted(introduced - alpha))}"
            )
        elif seq == original:
            why = "identical to the original (that is the reference, already evaluated)"
        elif seq in already:
            why = "already_evaluated"
        elif seq in seen_here:
            why = "duplicate within this round"
        elif not within_budget(original, seq):
            why = f"{n_mutations(original, seq)} substitutions exceeds budget {cap}"
        else:
            seen_here.add(seq)
            accepted.append((seq, reason))
            continue
        rejected.append({"sequence": seq, "reason": why, "model_reason": reason[:160]})
    return accepted, rejected


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------


class LlamaProposer:
    """A local Llama-3.1-8B-Instruct, loaded once and reused across parents.

    Loading costs ~16 GB of GPU memory and a NAS read of the same size, which is why the
    propose stage is a job of its own rather than something the fold loop does between
    evaluations: one load serves every parent in a round, and the GPU is free again
    before any folding starts.
    """

    def __init__(
        self,
        model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> None:
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"model path does not exist: {self.model_path}. This stage never "
                f"downloads weights; point --model at a local copy."
            )
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._tok = None
        self._model = None
        self.load_seconds: float | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype),
            device_map=self.device,
        )
        self._model.eval()
        self.load_seconds = time.time() - t0

    @property
    def config(self) -> dict:
        """Everything needed to reproduce a completion, recorded on every round."""
        return {
            "model_id": MODEL_ID,
            "model_path": self.model_path,
            "dtype": self.dtype,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "load_seconds": (
                None if self.load_seconds is None else round(self.load_seconds, 1)
            ),
        }

    def complete(self, system: str, user: str, *, seed: int) -> dict:
        """One sampling pass. The seed is set immediately before generation."""
        import torch

        self.load()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        enc = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers >= 5 returns a BatchEncoding here, earlier versions a bare
        # tensor. Both are accepted so this module does not pin the library version.
        if hasattr(enc, "keys"):
            inputs = {k: v.to(self._model.device) for k, v in enc.items()}
        else:
            inputs = {"input_ids": enc.to(self._model.device)}
        n_prompt = int(inputs["input_ids"].shape[-1])
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self._tok.pad_token_id,
            )
        text = self._tok.decode(out[0][n_prompt:], skip_special_tokens=True)
        return {
            "text": text,
            "seed": seed,
            "prompt_tokens": n_prompt,
            "completion_tokens": int(out.shape[-1]) - n_prompt,
            "seconds": round(time.time() - t0, 1),
        }


# --------------------------------------------------------------------------------------
# One round
# --------------------------------------------------------------------------------------


def propose_round(
    *,
    proposer: LlamaProposer,
    original: str,
    budget: int,
    round_index: int,
    parent_id: str,
    already: set[str],
    reference_summary: dict | None = None,
    history: list[CandidateRecord] | None = None,
    alphabet: tuple[str, ...] | None = None,
    pdb_id: str | None = None,
    pocket_note: str | None = None,
    feedback_note: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    prompt_dir: Path | None = None,
    n_requested: int | None = None,
    parent_candidate: CandidateRecord | None = None,
    prompt_tag: str = "llm",
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> tuple[list[Proposal], dict]:
    """Ask for `budget` new sequences; return the legal, novel ones and a full audit.

    The returned record is written to the run directory by the caller. It carries the
    prompt, every completion, and every rejection, because an LLM arm whose prompts were
    not kept is not reproducible and therefore is not a result.
    """
    alpha = alphabet or default_alphabet()
    blocked = set(already) | {original}
    want = budget if n_requested is None else max(1, int(n_requested))
    system = system_prompt_for(output_format)
    # What the edits are stated against: the same sequence the TASK block names.
    edit_base = (
        parent_candidate.sequence if parent_candidate is not None else original
    )
    accepted: list[tuple[str, str]] = []
    attempts: list[dict] = []
    rejections: list[dict] = []
    malformed: list[dict] = []

    record: dict = {
        "parent_id": parent_id,
        "round_index": round_index,
        "arm": ARM_LLM,
        "original_sequence": original,
        "budget": budget,
        "n_requested": want,
        "output_format": output_format,
        "elite_sequence": (
            parent_candidate.sequence if parent_candidate is not None else None
        ),
        "elite_energy": (
            parent_candidate.energy if parent_candidate is not None else None
        ),
        "llm": proposer.config,
        "feedback_candidates_shown": len(history or []),
        "attempts": attempts,
        "rejections": rejections,
        "malformed": malformed,
        "note": "",
    }

    for attempt in range(max_attempts):
        need = want - len(accepted)
        if need <= 0:
            break
        # Asking for a little more than is needed costs nothing and absorbs the
        # duplicates and over-budget strings an 8B model reliably produces.
        ask = min(need + 2, want + 4)
        user = build_user_prompt(
            original=original,
            budget=budget,
            n_requested=ask,
            reference_summary=reference_summary,
            history=history,
            alphabet=alpha,
            pdb_id=pdb_id,
            pocket_note=pocket_note,
            feedback_note=feedback_note,
            parent_candidate=parent_candidate,
            output_format=output_format,
        )
        if attempt:
            # A retry says what was wrong with the last pass. Without this the model
            # repeats the same illegal answer and the attempt limit is spent for nothing.
            tail = rejections[-6:]
            user += (
                "\n\nYOUR PREVIOUS ANSWER WAS PARTLY UNUSABLE. Rejected proposals and "
                "why:\n"
                + "\n".join(f"  {r['sequence']}: {r['reason']}" for r in tail)
                + (
                    f"\nPropose {ask} DIFFERENT sets of substitutions that obey every "
                    f"rule."
                    if output_format == OUTPUT_FORMAT_EDITS
                    else f"\nPropose {ask} DIFFERENT sequences that obey every rule."
                )
            )
        seed = proposal_seed(parent_id, round_index, prompt_tag, attempt)
        comp = proposer.complete(system, user, seed=seed)

        if output_format == OUTPUT_FORMAT_EDITS:
            cands, bad = parse_edit_completion(comp["text"], base=edit_base)
        else:
            cands, bad = parse_completion(comp["text"], original=original)
        malformed.extend(bad)
        ok, rej = screen_candidates(
            cands, original=original, alphabet=alpha, already=blocked
        )
        for seq, reason in ok:
            if len(accepted) >= want:
                break
            blocked.add(seq)
            accepted.append((seq, reason))
        rejections.extend(rej)
        attempts.append(
            {
                "attempt": attempt,
                "seed": seed,
                "n_parsed": len(cands),
                "n_malformed": len(bad),
                "n_accepted": len(ok),
                "n_rejected": len(rej),
                "prompt_tokens": comp["prompt_tokens"],
                "completion_tokens": comp["completion_tokens"],
                "seconds": comp["seconds"],
            }
        )
        if prompt_dir is not None:
            prompt_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{parent_id}_r{round_index}_{prompt_tag}_a{attempt}"
            (prompt_dir / f"{stem}.prompt.txt").write_text(
                f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{user}\n"
            )
            (prompt_dir / f"{stem}.completion.txt").write_text(comp["text"])

    proposals: list[Proposal] = []
    for k, (seq, reason) in enumerate(accepted):
        proposals.append(
            Proposal(
                sequence=seq,
                arm=ARM_LLM,
                round_index=round_index,
                kind=KIND_LLM,
                # The elite this was asked to edit, when there was one. Without a named
                # elite the model saw only the scored history, and naming an ancestor
                # would assert an edit path that does not exist -- the same choice
                # `search.py` makes for the MPNN arm, for the same reason.
                ancestor_sequence=(
                    parent_candidate.sequence if parent_candidate is not None else None
                ),
                original_sequence=original,
                edited_positions=tuple(
                    i for i, (a, b) in enumerate(zip(original, seq)) if a != b
                ),
                seed=proposal_seed(parent_id, round_index, prompt_tag, "pick", k),
                note=reason[:200],
            )
        )

    record["n_accepted"] = len(proposals)
    record["sequences"] = [p.sequence for p in proposals]
    if len(proposals) < want:
        record["note"] = (
            f"round {round_index} produced {len(proposals)}/{want} legal novel "
            f"sequences after {len(attempts)} attempt(s); the shortfall is NOT padded "
            f"with random draws, so this arm's budget for this round is "
            f"{len(proposals)}, not {want}"
        )
    return proposals, record


# --------------------------------------------------------------------------------------
# One generation: several elites, a few offspring each
# --------------------------------------------------------------------------------------


def select_elites(
    history: list[CandidateRecord],
    *,
    n_elites: int = DEFAULT_N_ELITES,
    min_distance: int = DEFAULT_ELITE_MIN_DISTANCE,
    keep_infeasible: int = DEFAULT_ELITE_INFEASIBLE,
) -> list[CandidateRecord]:
    """The sequences this generation breeds from.

    Delegates to `search.Archive` rather than re-implementing the ranking, so arm B and
    arm D keep elites by the same rule: valid candidates by mean post-relax interface dG
    first, the remaining slots filled by the smallest geometry violations, and a Hamming
    veto so the elite set cannot be several neighbours of one sequence. A starved
    selection returns FEWER elites, never near-duplicates.

    Nothing here requires a minimum validity ratio. A sequence valid on one fold out of
    six is ranked on the mean of that one fold and competes on it; the ratio is carried
    on the record and shown in the prompt, which is what the plan asks for.
    """
    archive = Archive(
        capacity=max(1, int(n_elites)),
        min_distance=max(1, int(min_distance)),
        keep_infeasible=max(0, int(keep_infeasible)),
    )
    by_seq = {h.sequence: h for h in history}
    for h in history:
        archive.add(
            ArchiveEntry(
                sequence=h.sequence,
                objective=h.energy if h.valid else None,
                violation=h.violation,
                valid=bool(h.valid and h.energy is not None),
                round_index=-1,
                kind=KIND_LLM,
                ancestor_sequence=None,
            )
        )
    return [by_seq[e.sequence] for e in archive.working_set() if e.sequence in by_seq]


def propose_generation(
    *,
    proposer: LlamaProposer,
    original: str,
    budget: int,
    round_index: int,
    parent_id: str,
    already: set[str],
    elites: list[CandidateRecord],
    offspring_per_elite: int,
    generation_size: int | None = None,
    reference_summary: dict | None = None,
    history: list[CandidateRecord] | None = None,
    alphabet: tuple[str, ...] | None = None,
    pdb_id: str | None = None,
    pocket_note: str | None = None,
    feedback_note: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    prompt_dir: Path | None = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> tuple[list[Proposal], dict]:
    """One generation of the evolutionary loop: offspring of each elite, in turn.

    One prompt per elite rather than one prompt asking for everything. Two reasons:

    - **Ancestry is recorded, not inferred.** Each proposal carries the elite it was
      asked to edit, so `cpseq_arm_compare.py` can ask whether an improvement descended
      from the elite that was supposed to produce it, or appeared somewhere else. A
      single prompt shown six elites yields sequences no one can attribute.
    - **One elite cannot monopolise the generation.** The budget is spent per elite, so
      a generation explores every basin it is holding instead of the one the model finds
      most interesting. This is the same reason `search.py`'s `guided_round` cycles its
      parents.

    `generation_size` is the generation's TOTAL budget, and it is spent whether or not
    there are `n_elites` elites to spend it on. Without it the generation was
    `len(elites) * offspring_per_elite`, which starved exactly the generations that
    needed breadth most: measured on this pilot's generation 1, five of six complexes
    had a single elite -- nothing else had been folded yet -- so they proposed 3
    candidates against an intended 12, and the arm finished the run on 27 candidates
    against the random control's 36. That was not the model failing to supply legal
    sequences: generations 2 and 3 filled 12 of 12 on all six complexes. It was this
    function asking for less.

    The budget is divided as evenly as possible over the elites that exist, with the
    remainder going to the better-ranked ones, so one elite gets the whole generation and
    four get a quarter each.

    `already` accumulates across elites within the generation, so two elites cannot
    produce the same offspring and have it counted twice.

    With no elites -- the first generation, or a parent where nothing has been scored
    yet -- this falls back to one history-conditioned round, which is what the arm did
    before it had a population to breed from.
    """
    blocked = set(already)
    all_props: list[Proposal] = []
    per_elite: list[dict] = []
    total = int(generation_size) if generation_size else (
        offspring_per_elite * max(1, len(elites))
    )

    if not elites:
        props, rec = propose_round(
            proposer=proposer, original=original, budget=budget,
            round_index=round_index, parent_id=parent_id, already=blocked,
            reference_summary=reference_summary, history=history, alphabet=alphabet,
            pdb_id=pdb_id, pocket_note=pocket_note, feedback_note=feedback_note,
            max_attempts=max_attempts, prompt_dir=prompt_dir,
            n_requested=total, prompt_tag="seed", output_format=output_format,
        )
        rec["elite_rank"] = None
        return props, {
            "parent_id": parent_id,
            "round_index": round_index,
            "arm": ARM_LLM,
            "original_sequence": original,
            "budget": budget,
            "n_elites": 0,
            "offspring_per_elite": offspring_per_elite,
            "n_requested": total,
            "n_accepted": len(props),
            "elites": [],
            "rounds": [rec],
            "note": (
                "no elite available, so this generation is one history-conditioned "
                "round rather than offspring of a population"
            ),
        }

    # Even shares, remainder to the better-ranked elites.
    base, extra = divmod(total, len(elites))
    shares = [base + (1 if r < extra else 0) for r in range(len(elites))]
    for rank, (elite, share) in enumerate(zip(elites, shares)):
        if share <= 0:
            continue
        props, rec = propose_round(
            proposer=proposer, original=original, budget=budget,
            round_index=round_index, parent_id=parent_id, already=blocked,
            reference_summary=reference_summary, history=history, alphabet=alphabet,
            pdb_id=pdb_id, pocket_note=pocket_note, feedback_note=feedback_note,
            max_attempts=max_attempts, prompt_dir=prompt_dir,
            n_requested=share, parent_candidate=elite,
            prompt_tag=f"elite{rank}", output_format=output_format,
        )
        rec["elite_rank"] = rank
        per_elite.append(rec)
        all_props.extend(props)
        blocked.update(p.sequence for p in props)

    asked = total
    gen: dict = {
        "parent_id": parent_id,
        "round_index": round_index,
        "arm": ARM_LLM,
        "original_sequence": original,
        "budget": budget,
        "output_format": output_format,
        "n_elites": len(elites),
        "offspring_per_elite": offspring_per_elite,
        "n_requested": asked,
        "n_accepted": len(all_props),
        "elites": [
            {
                "rank": r,
                "sequence": e.sequence,
                "n_mutations": e.n_mutations,
                "valid": e.valid,
                "mean_valid_dG_separated": e.energy,
                "median_valid_dG_separated": e.energy_median,
                "n_valid_folds": e.n_valid_folds,
                "n_folds": e.n_folds,
                "violation": e.violation,
            }
            for r, e in enumerate(elites)
        ],
        "rounds": per_elite,
        "note": "",
    }
    if len(all_props) < asked:
        gen["note"] = (
            f"generation {round_index} produced {len(all_props)}/{asked} legal novel "
            f"sequences across {len(elites)} elite(s); the shortfall is NOT padded, so "
            f"this generation's size is {len(all_props)}. Per-elite detail is in "
            f"`rounds`."
        )
    return all_props, gen


def history_from_outcomes(
    outcomes: list[dict],
    *,
    stage: str,
    max_fold_rank: int | None = None,
    interaction_for: set[str] | None = None,
    scoring_version: str | None = SCORING_PROTOCOL_VERSION,
) -> list[CandidateRecord]:
    """Fold the journal's rows into one record per sequence, for the prompt.

    `max_fold_rank` is the cross-fit guard: with `depth//2` passed in, only the folds
    that selection is allowed to use reach the prompt, and the measurement half never
    informs a proposal. Rows with `technical_failure` are dropped -- a crashed fold is
    not evidence about a sequence, and telling the model a sequence "failed" because a
    GPU died would teach it to avoid a perfectly good one.

    `scoring_version` keeps one protocol's rows together. A rescore APPENDS rows rather
    than rewriting history, so after a protocol change the journal holds the same fold
    twice under two versions -- and v4 called 1 of 24 folds valid where v5 calls 24 of
    24. Pooling those would count every rescored fold twice and average a validity label
    across two different scorers. Rows written before the field existed carry no version
    and are kept, since there is only one protocol they can belong to. Pass None to pool
    deliberately.
    """
    by_seq: dict[str, dict] = {}
    for o in outcomes:
        if o.get("stage") != stage or o.get("technical_failure"):
            continue
        if o.get("arm") == "reference":
            continue
        sv = o.get("scoring_version")
        if scoring_version is not None and sv is not None and sv != scoring_version:
            continue
        rank = o.get("seed_rank")
        if max_fold_rank is not None and rank is not None and rank >= max_fold_rank:
            continue
        seq = o.get("sequence")
        if not seq:
            continue
        e = by_seq.setdefault(
            seq,
            {
                "n_mutations": o.get("n_mutations") or 0,
                "energies": [],
                "violations": [],
                "valid": False,
                "reasons": [],
                "n_folds": 0,
                "n_valid_folds": 0,
            },
        )
        e["n_folds"] += 1
        if o.get("valid"):
            e["valid"] = True
            e["n_valid_folds"] += 1
            if o.get("best_valid_dG_separated") is not None:
                e["energies"].append(float(o["best_valid_dG_separated"]))
        else:
            for r in o.get("validity_reasons") or o.get("failure_reasons") or []:
                if r not in e["reasons"]:
                    e["reasons"].append(str(r))
        if o.get("min_violation") is not None:
            e["violations"].append(float(o["min_violation"]))

    out: list[CandidateRecord] = []
    for seq, e in by_seq.items():
        # An average over folds, never the best. The project measured 4.3 REU median and
        # 21.97 REU max seed-to-seed spread on this cohort, so a per-sequence minimum is
        # mostly a record of which fold got lucky -- and handing the model the luckiest
        # fold of each sequence would teach it to chase noise. The mean is the objective
        # the confirmation stage reports; the median is carried alongside it.
        out.append(
            CandidateRecord(
                sequence=seq,
                n_mutations=e["n_mutations"],
                valid=e["valid"],
                energy=st.fmean(e["energies"]) if e["energies"] else None,
                energy_median=st.median(e["energies"]) if e["energies"] else None,
                energy_spread=(
                    max(e["energies"]) - min(e["energies"])
                    if len(e["energies"]) > 1 else None
                ),
                violation=min(e["violations"]) if e["violations"] else None,
                reasons=e["reasons"],
                n_folds=e["n_folds"],
                n_valid_folds=e["n_valid_folds"],
            )
        )
    return out


def round_diversity(sequences: list[str], original: str) -> dict:
    """How much of the sequence space one round actually covers.

    Recorded on every round because a batch of 8 near-identical proposals is one
    candidate wearing eight hats, and the ladder would credit it with eight. The random
    control cannot concentrate this way, so without this number an arm that converged
    prematurely and an arm that searched broadly look identical in the budget column.

    Measured on round 1 of the probe cohort (8 proposals per complex, at depth 10):

    | complex    | arm            | median pairwise Hamming | distinct edited positions |
    |------------|----------------|-------------------------|---------------------------|
    | lnr_2v8x   | llm_search     | 2.0                     | 4 of 14                   |
    | lnr_2v8x   | random_control | 4.0                     | 9 of 14                   |
    | lnr_6seo   | llm_search     | 2.0                     | 5 of 16                   |
    | lnr_6seo   | random_control | 2.0                     | 10 of 16                  |

    So the model concentrates its budget on roughly a third of the positions. That is
    the intended behaviour -- it is spending the budget where the interface says it
    matters -- but it also means best-of-B over this arm samples less independent
    variation than best-of-B over the control, and if the chosen positions are wrong the
    arm has no mechanism for finding that out. Which of those two readings holds is an
    empirical question, and it cannot be asked at all unless the number is on the record.
    """
    uniq = list(dict.fromkeys(sequences))
    positions: set[int] = set()
    for s in uniq:
        positions |= {
            i for i in range(min(len(s), len(original))) if s[i] != original[i]
        }
    dists = [
        sum(1 for x, y in zip(a, b) if x != y)
        for a, b in itertools.combinations(uniq, 2)
    ]
    return {
        "n_unique": len(uniq),
        "distinct_edited_positions": len(positions),
        "peptide_length": len(original),
        "position_coverage": (
            round(len(positions) / len(original), 3) if original else None
        ),
        "pairwise_hamming_median": st.median(dists) if dists else None,
        "pairwise_hamming_min": min(dists) if dists else None,
        "pairwise_hamming_max": max(dists) if dists else None,
    }


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
