#!/usr/bin/env python3
"""Checks for arm D's proposal path. No GPU and no model: parsing and legality only.

The model call is the one part not covered here, deliberately -- it is a network of
weights whose output varies, and a test that asserted what it returns would be testing
the weights. What IS tested is everything that decides whether a returned string becomes
a candidate, because that is where a bug silently changes the experiment rather than
breaking it.

Run:  python script_utils/test_cpseq_llm_propose.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.interactions import (  # noqa: E402
    render_complex,
    _ss_from_phi_psi,
)
from proteinfoundation.cpseq.llm_propose import (  # noqa: E402
    CandidateRecord,
    apply_substitutions,
    build_user_prompt,
    history_from_outcomes,
    parse_completion,
    parse_edit_completion,
    propose_generation,
    render_single_substitution_pairs,
    round_diversity,
    screen_candidates,
    select_elites,
    shuffle_outcome_associations,
    single_substitution_pairs,
)
from proteinfoundation.cpseq.search import default_alphabet  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))


ALPHA = default_alphabet()
# A real parent from the pilot cohort. Its native cysteine at position 12 is the whole
# point of several checks below.
ORIG_CYS = "RIIYDRKFLMECRN"


def test_parsing() -> None:
    print("\n[parse_completion]")
    txt = (
        "Reasoning about the interface here.\n\n"
        "PROPOSE: RIIYDRKFLMECRA | extend the contact at 14\n"
        "PROPOSE: riiydrkflmecrw | lowercase should still parse\n"
        "PROPOSE: RIIYDRKFLMECR | too short\n"
        "PROPOSE: RIIYDRKFLMECRNN | too long\n"
        "PROPOSE: RIIYDRKFLMECRB | non-standard letter\n"
    )
    found, bad = parse_completion(txt, original=ORIG_CYS)
    seqs = [s for s, _ in found]
    check("accepts a well-formed line", "RIIYDRKFLMECRA" in seqs)
    check("uppercases a lowercase answer", "RIIYDRKFLMECRW" in seqs)
    check("rejects short/long/non-standard", len(bad) == 3, f"{bad}")
    check("keeps the model's stated reason", found[0][1].startswith("extend"))

    print("\n[substitution notation]")
    # Only used when no PROPOSE line held a usable whole sequence.
    sub_only = "PROPOSE: D5A, K7R | open the pocket\n"
    found2, bad2 = parse_completion(sub_only, original=ORIG_CYS)
    check("applies A5/R7 style substitutions", found2 and found2[0][0] == "RIIYARRFLMECRN",
          f"{found2}")
    wrong = "PROPOSE: W5A | cites a residue the original does not have\n"
    _, bad3 = parse_completion(wrong, original=ORIG_CYS)
    check("refuses a substitution citing the wrong source residue", len(bad3) == 1, f"{bad3}")
    check("apply_substitutions returns None on a mismatch",
          apply_substitutions(ORIG_CYS, [("W", 5, "A")]) is None)
    # A model that answers in the requested format is never reinterpreted.
    mixed = "PROPOSE: RIIYDRKFLMECRA | good\nPROPOSE: D5A | also mentioned\n"
    found4, _ = parse_completion(mixed, original=ORIG_CYS)
    check("whole sequences win over substitution notation",
          [s for s, _ in found4] == ["RIIYDRKFLMECRA"], f"{found4}")

    print("\n[numbered 'Proposal:' list is parsed, prose is not]")
    # Observed on lnr_6seo round 1 attempt 0: the model answered as a numbered list and
    # the strict PROPOSE-only pattern parsed 0 of ten well-formed proposals, discarding
    # a 13 s generation over punctuation. "PROPOSAL" does not contain "PROPOSE".
    numbered = (
        "1. Proposal: RIIYDRKFLMECRA | extend the contact at 14\n"
        "2. Proposal: RIIYDRKFLMECRW | fill the pocket\n"
    )
    found5, _ = parse_completion(numbered, original=ORIG_CYS)
    check("numbered 'N. Proposal:' lines are parsed",
          [s for s, _ in found5] == ["RIIYDRKFLMECRA", "RIIYDRKFLMECRW"], f"{found5}")

    # The reasoning text names sequences it is arguing AGAINST; harvesting those would
    # fold candidates the model never proposed.
    prose = "I considered RIIYDRKFLMECRW but rejected it as too buried.\n"
    found6, _ = parse_completion(prose, original=ORIG_CYS)
    check("an unlabelled sequence in prose is never harvested", found6 == [], f"{found6}")


def test_native_cysteine() -> None:
    """The regression that mattered: a native C must not be treated as illegal.

    `search._draw_residue` draws only REPLACEMENT residues from the alphabet, so every
    draw of the random control preserves the original's cysteine. Applying the alphabet
    to the whole sequence rejected all 24 of the model's proposals for `lnr_2v8x` and
    would have made arm D propose nothing for any cysteine-bearing parent.
    """
    print("\n[native cysteine is kept, not introduced]")
    keep = [("RIIYDRKFLMECRA", "keeps the native C12")]
    ok, rej = screen_candidates(keep, original=ORIG_CYS, alphabet=ALPHA, already=set())
    check("a proposal keeping the native C is accepted", len(ok) == 1, f"{rej}")

    introduce = [("RIIYDRKFLMECRC", "introduces a new C at 14")]
    ok2, rej2 = screen_candidates(
        introduce, original=ORIG_CYS, alphabet=ALPHA, already=set()
    )
    check("a proposal introducing a new C is rejected", len(ok2) == 0 and len(rej2) == 1)
    check("and the reason names the introduction",
          "introduces excluded" in rej2[0]["reason"], f"{rej2}")

    # And with no native excluded residue, nothing changes.
    plain = "AAAAAAAAAA"
    ok3, _ = screen_candidates(
        [("AAAACAAAAA", "introduce C")], original=plain, alphabet=ALPHA, already=set()
    )
    check("introducing C is still rejected on a C-free original", len(ok3) == 0)

    print("\n[prompt states the cysteine rule without contradicting itself]")
    p = build_user_prompt(
        original=ORIG_CYS, budget=7, n_requested=4, alphabet=ALPHA, pdb_id="2v8x"
    )
    check("prompt tells the model to keep the native C12", "C12" in p, p[:300])
    check("prompt forbids introducing C", "must not INTRODUCE" in p)
    check("prompt says keeping it costs no budget", "costs no budget" in p)


def test_legality() -> None:
    print("\n[screen_candidates]")
    cap = len(ORIG_CYS) // 2  # 7
    # Exactly one substitution past the cap, introducing nothing excluded: the native
    # C at index 11 is left alone, so the budget rule is the only thing that can fire.
    over = "A" * (cap + 1) + ORIG_CYS[cap + 1 :]
    ok, rej = screen_candidates(
        [(over, "far too many edits")], original=ORIG_CYS, alphabet=ALPHA, already=set()
    )
    check("over-budget proposal rejected", len(ok) == 0 and "exceeds budget" in rej[0]["reason"],
          f"{rej}")

    ok2, rej2 = screen_candidates(
        [(ORIG_CYS, "the original itself")], original=ORIG_CYS, alphabet=ALPHA, already=set()
    )
    check("the original is refused (it is the reference)", len(ok2) == 0, f"{rej2}")

    ok3, rej3 = screen_candidates(
        [("RIIYDRKFLMECRA", "x")], original=ORIG_CYS, alphabet=ALPHA,
        already={"RIIYDRKFLMECRA"},
    )
    check("an already-evaluated sequence is refused, not cached",
          len(ok3) == 0 and rej3[0]["reason"] == "already_evaluated", f"{rej3}")

    dupes = [("RIIYDRKFLMECRA", "a"), ("RIIYDRKFLMECRA", "b")]
    ok4, rej4 = screen_candidates(
        dupes, original=ORIG_CYS, alphabet=ALPHA, already=set()
    )
    check("a duplicate inside one round counts once",
          len(ok4) == 1 and len(rej4) == 1, f"{ok4} {rej4}")
    at_cap = "A" * cap + ORIG_CYS[cap:]
    ok5, rej5 = screen_candidates(
        [(at_cap, "at cap")], original=ORIG_CYS, alphabet=ALPHA, already=set()
    )
    check("budget boundary accepted at exactly the cap", len(ok5) == 1, f"{rej5}")


def test_history_and_crossfit() -> None:
    print("\n[history_from_outcomes]")
    rows = [
        # Two folds of one sequence in the selection half, one in the measurement half
        # with a very different energy. Only the selection half may reach the prompt.
        {"stage": "ladder", "arm": "llm_search", "sequence": "AAAC", "seed_rank": 0,
         "valid": True, "best_valid_dG_separated": -10.0, "n_mutations": 1},
        {"stage": "ladder", "arm": "llm_search", "sequence": "AAAC", "seed_rank": 1,
         "valid": True, "best_valid_dG_separated": -12.0, "n_mutations": 1},
        {"stage": "ladder", "arm": "llm_search", "sequence": "AAAC", "seed_rank": 5,
         "valid": True, "best_valid_dG_separated": -99.0, "n_mutations": 1},
        # A crashed fold says nothing about a sequence.
        {"stage": "ladder", "arm": "llm_search", "sequence": "AADC", "seed_rank": 0,
         "valid": False, "technical_failure": True, "n_mutations": 1},
        # The reference is the baseline, not a candidate.
        {"stage": "ladder", "arm": "reference", "sequence": "AAAA", "seed_rank": 0,
         "valid": True, "best_valid_dG_separated": -5.0, "n_mutations": 0},
        # Another stage must not leak in.
        {"stage": "screen", "arm": "llm_search", "sequence": "AAEC", "seed_rank": 0,
         "valid": True, "best_valid_dG_separated": -50.0, "n_mutations": 1},
    ]
    h_all = history_from_outcomes(rows, stage="ladder")
    seqs_all = {r.sequence for r in h_all}
    check("reference excluded from the candidate history", "AAAA" not in seqs_all)
    check("other stages excluded", "AAEC" not in seqs_all)
    check("a technically-failed fold yields no record", "AADC" not in seqs_all,
          f"{seqs_all}")

    rec_all = next(r for r in h_all if r.sequence == "AAAC")
    check("ranked on the mean over folds, not the best",
          abs(rec_all.energy - (-10.0 - 12.0 - 99.0) / 3) < 1e-9, f"{rec_all.energy}")
    check("the median is carried alongside the mean",
          rec_all.energy_median == -12.0, f"{rec_all.energy_median}")
    check("valid folds counted out of folds attempted",
          rec_all.n_valid_folds == 3 and rec_all.n_folds == 3,
          f"{rec_all.n_valid_folds}/{rec_all.n_folds}")

    h_sel = history_from_outcomes(rows, stage="ladder", max_fold_rank=3)
    rec_sel = next(r for r in h_sel if r.sequence == "AAAC")
    check("cross-fit guard hides the measurement half",
          rec_sel.energy == -11.0 and rec_sel.n_folds == 2,
          f"energy={rec_sel.energy} n_folds={rec_sel.n_folds}")

    print("\n[prompt carries the history and the do-not-repeat list]")
    hist = [
        CandidateRecord(sequence="RIIYDRKFLMECRA", n_mutations=1, valid=True,
                        energy=-30.0, n_folds=5),
        CandidateRecord(sequence="RIIYDRKFLMECRW", n_mutations=1, valid=False,
                        violation=4.0, reasons=["4 peptide-receptor clash(es)"],
                        n_folds=5),
    ]
    p = build_user_prompt(
        original=ORIG_CYS, budget=7, n_requested=4, history=hist, alphabet=ALPHA,
        output_format="sequence",
    )
    check("passed candidate shown with its energy", "-30.0 REU" in p)
    check("rejected candidate shown with its reason", "clash(es)" in p)
    check("every scored sequence listed as off-limits",
          p.count("RIIYDRKFLMECRA") >= 2 and "DO NOT PROPOSE" in p)
    check("substitutions spelled out for the model", "N14A" in p, "expected N14A in the line")


def test_elites() -> None:
    print("\n[elite selection]")
    # Four valid candidates, two of which are one substitution apart, plus one that
    # never passed the gate.
    hist = [
        CandidateRecord(sequence="AAAAAAAA", n_mutations=1, valid=True, energy=-30.0,
                        n_folds=4, n_valid_folds=4),
        CandidateRecord(sequence="AAAAAAAC", n_mutations=2, valid=True, energy=-29.0,
                        n_folds=4, n_valid_folds=1),
        CandidateRecord(sequence="DDDDAAAA", n_mutations=4, valid=True, energy=-20.0,
                        n_folds=4, n_valid_folds=2),
        CandidateRecord(sequence="EEEEEEEE", n_mutations=8, valid=False,
                        violation=0.5, reasons=["1 peptide-receptor clash"], n_folds=4),
    ]
    el = select_elites(hist, n_elites=3, min_distance=2)
    seqs = [e.sequence for e in el]
    check("best mean dG is the first elite", seqs[0] == "AAAAAAAA", f"{seqs}")
    check("a one-substitution neighbour of an elite is vetoed",
          "AAAAAAAC" not in seqs, f"{seqs}")
    check("a distinct, worse candidate is still kept", "DDDDAAAA" in seqs, f"{seqs}")
    check("an infeasible candidate fills a leftover slot", "EEEEEEEE" in seqs, f"{seqs}")

    # No minimum validity ratio: 1 valid fold out of 4 competes on its mean.
    thin = [
        CandidateRecord(sequence="AAAAAAAC", n_mutations=2, valid=True, energy=-50.0,
                        n_folds=4, n_valid_folds=1),
        CandidateRecord(sequence="DDDDAAAA", n_mutations=4, valid=True, energy=-20.0,
                        n_folds=4, n_valid_folds=4),
    ]
    el2 = select_elites(thin, n_elites=2, min_distance=2)
    check("a 1-of-4-valid candidate is not excluded by its ratio",
          el2[0].sequence == "AAAAAAAC", f"{[e.sequence for e in el2]}")

    print("\n[offspring prompt]")
    elite = CandidateRecord(sequence="RIIYDRKFLMECRA", n_mutations=1, valid=True,
                            energy=-30.0, n_folds=6, n_valid_folds=4)
    p = build_user_prompt(
        original=ORIG_CYS, budget=7, n_requested=4, alphabet=ALPHA,
        parent_candidate=elite, output_format="sequence",
    )
    check("the elite is named as the sequence to edit",
          "SEQUENCE TO IMPROVE" in p and "RIIYDRKFLMECRA" in p)
    check("its standing is stated with the valid-fold count",
          "4 of 6 fold(s)" in p and "-30.0 REU" in p, "")
    check("the budget is still counted against the original",
          "counted against the ORIGINAL" in p and ORIG_CYS in p)
    check("the model is told to build from the elite, not the original",
          f"copying {elite.sequence}" in p, "")

    infeasible = CandidateRecord(sequence="RIIYDRKFLMECRW", n_mutations=1, valid=False,
                                 violation=4.0, reasons=["4 peptide-receptor clash(es)"],
                                 n_folds=6)
    p2 = build_user_prompt(
        original=ORIG_CYS, budget=7, n_requested=4, alphabet=ALPHA,
        parent_candidate=infeasible,
    )
    check("an infeasible elite is labelled as one, with what to fix",
          "has not passed the gate" in p2 and "clash(es)" in p2)

    # Without a parent the prompt must not grow an offspring block.
    p3 = build_user_prompt(original=ORIG_CYS, budget=7, n_requested=4, alphabet=ALPHA)
    check("no parent means no offspring block", "SEQUENCE TO IMPROVE" not in p3)


def test_generation() -> None:
    print("\n[one generation, several elites]")

    class StubProposer:
        """Returns a fixed answer per elite, so only the plumbing is under test."""

        config = {"model_id": "stub", "temperature": 0.0, "top_p": 1.0}

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, system: str, user: str, *, seed: int) -> dict:
            self.prompts.append(user)
            # Edit whichever sequence the prompt asked to improve.
            start = ORIG_CYS
            for line in user.splitlines():
                s = line.strip()
                if len(s) == len(ORIG_CYS) and s.isalpha() and s.isupper():
                    if "SEQUENCE TO IMPROVE" in user.split(s)[0][-200:]:
                        start = s
                        break
            out = []
            for k, aa in enumerate("AGLV"):
                seq = start[:-1] + aa if start[-1] != aa else start[:-1] + "T"
                out.append(f"PROPOSE: {seq} | variant {k}")
            return {
                "text": "reasoning\n" + "\n".join(out),
                "prompt_tokens": 10, "completion_tokens": 10, "seconds": 0.1,
            }

    elites = [
        CandidateRecord(sequence=ORIG_CYS[:-1] + "A", n_mutations=1, valid=True,
                        energy=-30.0, n_folds=4, n_valid_folds=4),
        CandidateRecord(sequence="D" + ORIG_CYS[1:-1] + "D", n_mutations=2, valid=True,
                        energy=-25.0, n_folds=4, n_valid_folds=3),
    ]
    stub = StubProposer()
    props, gen = propose_generation(
        proposer=stub, original=ORIG_CYS, budget=7, round_index=2,
        parent_id="t1", already={ORIG_CYS}, elites=elites, offspring_per_elite=2,
        alphabet=ALPHA, max_attempts=2,
    )
    check("one prompt per elite, at least", len(stub.prompts) >= 2, f"{len(stub.prompts)}")
    check("the generation records both elites", gen["n_elites"] == 2)
    check("every offspring names the elite it was asked to edit",
          all(p.ancestor_sequence in {e.sequence for e in elites} for p in props),
          f"{[p.ancestor_sequence for p in props]}")
    check("offspring are distinct across elites",
          len({p.sequence for p in props}) == len(props),
          f"{[p.sequence for p in props]}")
    check("per-elite audit trail is kept", len(gen["rounds"]) == 2)
    check("the elite objective is recorded on the generation",
          gen["elites"][0]["mean_valid_dG_separated"] == -30.0)

    # No elites: the generation falls back to one history-conditioned round and says so.
    stub2 = StubProposer()
    props2, gen2 = propose_generation(
        proposer=stub2, original=ORIG_CYS, budget=7, round_index=1,
        parent_id="t1", already={ORIG_CYS}, elites=[], offspring_per_elite=3,
        alphabet=ALPHA, max_attempts=2,
    )
    check("with no elite the generation seeds instead of breeding",
          gen2["n_elites"] == 0 and "no elite" in gen2["note"], f"{gen2['note']}")
    check("a seeding generation claims no ancestry",
          all(p.ancestor_sequence is None for p in props2),
          f"{[p.ancestor_sequence for p in props2]}")


def test_rendering() -> None:
    print("\n[interaction rendering]")
    bad = {"ok": False, "error": "pdb not found"}
    check("an unavailable summary renders as unavailable, not as empty",
          "unavailable" in render_complex(bad))

    s = {
        "ok": True,
        "peptide_sequence_observed": "AC",
        "peptide_chain": "B",
        "buried_sasa_A2": 100.0,
        "n_interface_contacts": 5,
        "n_interface_clashes": 2,
        "plip": {"ok": False, "error": "boom", "n_interactions": 0},
        "positions": [
            {"pos": 1, "residue": "A", "ss_approx": "H", "buried_fraction": 0.5,
             "min_dist_A": 1.0, "n_contacts": 5, "n_clashes": 2, "worst_clash_A": 1.0,
             "clash_partners": ["A:His2"], "partners": ["A:His2"], "interactions": []},
            {"pos": 2, "residue": "C", "ss_approx": "-", "buried_fraction": 0.0,
             "min_dist_A": 9.0, "n_contacts": 0, "n_clashes": 0,
             "clash_partners": [], "partners": [], "interactions": []},
        ],
    }
    out = render_complex(s)
    check("a PLIP failure is announced in the prompt text",
          "interaction detection failed" in out, out)
    check("an overlap is flagged on its position", "OVERLAP 1.0A" in out, out)
    check("a non-contacting position says so", "no receptor contact" in out, out)

    print("\n[secondary structure from dihedrals]")
    import numpy as np
    check("helical phi/psi -> H",
          _ss_from_phi_psi(np.radians(-60), np.radians(-45)) == "H")
    check("sheet phi/psi -> E",
          _ss_from_phi_psi(np.radians(-120), np.radians(130)) == "E")
    check("missing dihedral -> ?", _ss_from_phi_psi(None, None) == "?")


def test_diversity() -> None:
    """A converged round and a spread one must not produce the same number."""
    print("\n[round_diversity]")
    orig = "AAAAAAAAAA"
    # Eight proposals all editing position 1: one candidate wearing eight hats.
    narrow = [f"{c}AAAAAAAAA" for c in "CDEFGHIK"]
    d = round_diversity(narrow, orig)
    check("a converged round touches one position",
          d["distinct_edited_positions"] == 1, f"{d}")
    check("and its pairwise distance is 1", d["pairwise_hamming_median"] == 1, f"{d}")

    # Eight proposals each editing a different position.
    spread = [orig[:i] + "C" + orig[i + 1 :] for i in range(8)]
    d2 = round_diversity(spread, orig)
    check("a spread round touches eight positions",
          d2["distinct_edited_positions"] == 8, f"{d2}")
    check("and its pairwise distance is larger",
          d2["pairwise_hamming_median"] > d["pairwise_hamming_median"], f"{d2}")
    check("coverage is a fraction of the peptide length",
          d2["position_coverage"] == 0.8, f"{d2}")

    # Duplicates must not inflate the count.
    d3 = round_diversity(["CAAAAAAAAA", "CAAAAAAAAA"], orig)
    check("duplicates collapse to one unique sequence", d3["n_unique"] == 1, f"{d3}")
    check("a single sequence has no pairwise distance",
          d3["pairwise_hamming_median"] is None, f"{d3}")




def test_per_candidate_spread() -> None:
    """A candidate's own fold-to-fold spread reaches the prompt.

    One global sentence saying the cohort's median spread is 4.3 REU does not tell the
    model which of the candidates in front of it is the uncertain one. Two candidates
    with the same mean and spreads of 0.4 and 18 REU were being shown identically.
    """
    print("per-candidate spread")
    rows = [
        {"stage": "ladder", "arm": "llm_search", "sequence": "AAAC", "seed_rank": 0,
         "valid": True, "best_valid_dG_separated": -30.0, "n_mutations": 1},
        {"stage": "ladder", "arm": "llm_search", "sequence": "AAAC", "seed_rank": 1,
         "valid": True, "best_valid_dG_separated": -48.0, "n_mutations": 1},
        {"stage": "ladder", "arm": "llm_search", "sequence": "AADC", "seed_rank": 0,
         "valid": True, "best_valid_dG_separated": -39.0, "n_mutations": 1},
        {"stage": "ladder", "arm": "llm_search", "sequence": "AADC", "seed_rank": 1,
         "valid": True, "best_valid_dG_separated": -39.0, "n_mutations": 1},
    ]
    hist = history_from_outcomes(rows, stage="ladder", scoring_version=None)
    by = {h.sequence: h for h in hist}
    check("the wide candidate carries its spread", by["AAAC"].energy_spread == 18.0,
          f"{by['AAAC'].energy_spread}")
    check("the identical-fold candidate has zero spread",
          by["AADC"].energy_spread == 0.0, f"{by['AADC'].energy_spread}")
    check("both have the same mean, which is the point",
          by["AAAC"].energy == by["AADC"].energy == -39.0)
    line = by["AAAC"].one_line("AAAA")
    check("the spread is in the prompt line", "spread 18.0" in line, line)
    check("a single-fold candidate shows no spread",
          "spread" not in CandidateRecord(
              sequence="AAAC", n_mutations=1, valid=True, energy=-30.0,
              energy_spread=None, n_folds=1, n_valid_folds=1,
          ).one_line("AAAA"))


def test_single_substitution_pairs() -> None:
    """Only one-position differences support a claim about one substitution."""
    print("single-substitution comparisons")
    a = CandidateRecord(sequence="AAAA", n_mutations=0, valid=True, energy=-20.0,
                        energy_spread=1.0, n_folds=2, n_valid_folds=2)
    b = CandidateRecord(sequence="AAAD", n_mutations=1, valid=True, energy=-35.0,
                        energy_spread=1.0, n_folds=2, n_valid_folds=2)
    c = CandidateRecord(sequence="CCAD", n_mutations=3, valid=True, energy=-50.0,
                        energy_spread=1.0, n_folds=2, n_valid_folds=2)
    noisy = CandidateRecord(sequence="AAAW", n_mutations=1, valid=True, energy=-22.0,
                            energy_spread=14.0, n_folds=2, n_valid_folds=2)
    pairs = single_substitution_pairs([a, b, c])
    seqs = {(lo.sequence, hi.sequence) for lo, hi, *_ in pairs}
    check("the one-position pair is found", ("AAAD", "AAAA") in seqs, f"{seqs}")
    check("the three-position pair is not", ("CCAD", "AAAA") not in seqs, f"{seqs}")
    lo, hi, pos, frm, to = pairs[0]
    check("the position is 1-based", pos == 4, f"{pos}")
    check("the arrow points from worse to better", (frm, to) == ("A", "D"),
          f"{frm}->{to}")
    check("the better candidate is listed first", lo.energy < hi.energy)

    text = render_single_substitution_pairs([a, b, c])
    check("the block names the position and the change",
          "position 4: A -> D improved by 15.0 REU" in text, text)
    check("the block warns that other pairs prove nothing",
          "does NOT show that any single substitution helped" in text)

    # A difference inside the candidates' own fold spread must be labelled as noise.
    text2 = render_single_substitution_pairs([a, noisy])
    check("a difference within the fold spread is flagged as no difference",
          "treat as no difference" in text2, text2)
    check("a difference larger than the spread is not flagged",
          "treat as no difference" not in render_single_substitution_pairs([a, b]))
    check("no pairs gives no block", render_single_substitution_pairs([a]) == "")


def test_shuffled_history_control() -> None:
    """The control arm for whether the history carries information at all."""
    print("shuffled history control")
    hist = [
        CandidateRecord(sequence=f"AAA{aa}", n_mutations=1, valid=True,
                        energy=float(-10 * i), energy_spread=1.0,
                        n_folds=2, n_valid_folds=2)
        for i, aa in enumerate("CDEFG", start=1)
    ]
    for h in hist:
        h.interaction_summary = {"ok": True, "tag": h.sequence}
    out = shuffle_outcome_associations(hist, seed=7)
    check("the candidate set is unchanged",
          [h.sequence for h in out] == [h.sequence for h in hist])
    check("the set of outcomes is unchanged",
          sorted(h.energy for h in out) == sorted(h.energy for h in hist))
    check("no candidate keeps its own outcome",
          all(a.energy != b.energy for a, b in zip(out, hist)),
          f"{[(h.sequence, h.energy) for h in out]}")
    check("each sequence keeps its own structure summary",
          all(h.interaction_summary["tag"] == h.sequence for h in out))
    check("it is reproducible for a given seed",
          [h.energy for h in shuffle_outcome_associations(hist, seed=7)]
          == [h.energy for h in out])
    check("a different seed gives a different pairing",
          [h.energy for h in shuffle_outcome_associations(hist, seed=8)]
          != [h.energy for h in out])
    check("one candidate cannot be shuffled and is returned as is",
          [h.energy for h in shuffle_outcome_associations(hist[:1], seed=7)]
          == [hist[0].energy])


def test_prompt_carries_the_new_blocks() -> None:
    print("prompt assembly")
    a = CandidateRecord(sequence=ORIG_CYS, n_mutations=0, valid=True, energy=-20.0,
                        energy_spread=2.0, n_folds=2, n_valid_folds=2)
    b = CandidateRecord(sequence="RIIYDRKFLMECRW", n_mutations=1, valid=True,
                        energy=-35.0, energy_spread=1.0, n_folds=2, n_valid_folds=2)
    user = build_user_prompt(
        original=ORIG_CYS, budget=7, n_requested=4, history=[a, b], alphabet=ALPHA,
    )
    check("the single-substitution block is in the prompt",
          "SINGLE-SUBSTITUTION COMPARISONS" in user)
    check("it names the change", "position 14: N -> W" in user, "")
    check("per-candidate spread is in the prompt", "spread 2.0" in user)
    check("the do-not-repeat list still follows it",
          user.index("SINGLE-SUBSTITUTION") < user.index("DO NOT PROPOSE"))





def test_edit_output_format() -> None:
    """The default output format: a substitution list, built into a sequence in code.

    Adopted on measurement -- 0.717 of requested proposals usable against 0.528,
    p=0.0052 paired over 30 cells -- so what is pinned here is the two things that made
    it work. The position-to-residue map is given rather than counted, and NO
    original-anchored `D1Y` notation appears anywhere, because 45% of the format's
    mis-cited substitutions were naming the residue the original holds.
    """
    print("edit output format")
    elite = CandidateRecord(
        sequence="RIIYDRKFLMECRA", n_mutations=1, valid=True, energy=-30.0,
        energy_spread=2.0, n_folds=6, n_valid_folds=4,
    )
    hist = [
        elite,
        CandidateRecord(sequence="RIIYDRKFLMECRW", n_mutations=1, valid=True,
                        energy=-25.0, energy_spread=1.0, n_folds=6, n_valid_folds=6),
    ]
    p = build_user_prompt(
        original=ORIG_CYS, budget=7, n_requested=4, alphabet=ALPHA, history=hist,
        parent_candidate=elite,
    )
    check("the edit format is the default", "EDIT: " in p and "PROPOSE:" not in p)
    check("the position-to-residue map is given",
          "POSITION BY POSITION" in p and "R1   I2   I3" in p, p[:0])
    check("the map uses the elite, not the original",
          "A14" in p and "N14" not in p.split("POSITION BY POSITION")[1][:200])
    check("no original-anchored substitution notation anywhere",
          not re.search(r"\b[A-Z]\d{1,3}[A-Z]\b", p),
          re.findall(r"\b[A-Z]\d{1,3}[A-Z]\b", p)[:5].__str__())
    check("the difference from the original is stated in words",
          "where the original holds" in p)
    check("history keeps its substitution counts", "(1 sub)" in p)
    check("the budget is still stated against the original",
          "differ from the original at no more than 7 positions" in p)
    check("the model is told not to write a sequence",
          "do not write out a" in p)

    # The sequence format must be unchanged for the runs that used it.
    q = build_user_prompt(
        original=ORIG_CYS, budget=7, n_requested=4, alphabet=ALPHA, history=hist,
        parent_candidate=elite, output_format="sequence",
    )
    check("the old format still asks for whole sequences",
          "PROPOSE: <SEQUENCE>" in q and "EDIT: " not in q)
    check("and still spells out substitutions", "N14A" in q)

    try:
        build_user_prompt(original=ORIG_CYS, budget=7, n_requested=4,
                          output_format="whatever")
    except ValueError:
        check("an unknown format is refused", True)
    else:
        check("an unknown format is refused", False, "no ValueError raised")


def test_edit_parsing() -> None:
    """Edit lines become sequences, and a mis-named residue is caught."""
    print("edit parsing")
    base = "RIIYDRKFLMECRA"
    text = (
        "Reasoning about the interface.\n"
        "EDIT: D5E, K7R | add charge\n"
        "2) EDIT: A14W | fill the pocket\n"
        "EDIT: N14W | N is not at 14 -- that is the ORIGINAL's residue\n"
        "EDIT: D5E, D99E | position off the end\n"
        "EDIT: nothing here | no substitution\n"
    )
    found, bad = parse_edit_completion(text, base=base)
    seqs = [s for s, _ in found]
    check("a two-substitution line is built", "RIIYERRFLMECRA" in seqs, f"{seqs}")
    check("a one-substitution line is built", "RIIYDRKFLMECRW" in seqs, f"{seqs}")
    check("every built sequence has the base's length",
          all(len(s) == len(base) for s in seqs))
    reasons = [b["reason"] for b in bad]
    check("naming the original's residue is caught",
          "substitution cites wrong source residue" in reasons, f"{reasons}")
    check("a position off the end is caught",
          "position outside the sequence" in reasons, f"{reasons}")
    check("a line with no substitution is caught",
          "no substitution parsed" in reasons, f"{reasons}")
    check("the reason text is carried as provenance",
          any("add charge" == r for _s, r in found))
    check("nothing is parsed from an answer in the old format",
          parse_edit_completion("PROPOSE: RIIYDRKFLMECRW | x", base=base)[0] == [])



def main() -> int:
    print("arm D (LLM proposal) checks")
    test_parsing()
    test_native_cysteine()
    test_legality()
    test_history_and_crossfit()
    test_elites()
    test_generation()
    test_rendering()
    test_diversity()
    test_per_candidate_spread()
    test_single_substitution_pairs()
    test_shuffled_history_control()
    test_prompt_carries_the_new_blocks()
    test_edit_output_format()
    test_edit_parsing()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
