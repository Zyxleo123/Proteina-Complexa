#!/usr/bin/env python3
"""Focused checks for the cyclic-peptide sequence search (`cpseq.search`, `cpseq.evaluate`).

These are the properties whose failure would produce a result that looks valid and is
not, so each one is checked against an adversarial case rather than a happy path:

- **The budget is measured against the ORIGINAL sequence.** The archive is deliberately
  seeded with sequences already AT the ceiling, because that is the only state in which a
  "two more edits" proposer can walk out of bounds while every individual step looks
  legal.
- **Reversion spends budget back** and is available precisely when the ceiling is hit.
- **Arm A reproduces Arm B's mutation-count multiset**, since a mismatch would make the
  arms differ by mutation load rather than by proposal quality.
- **An exhausted neighbourhood is reported, not padded** with duplicate evaluations.
- **The admissibility check rejects before compute is spent.**

No GPU, no PyRosetta, no network: these are string operations over an alphabet.

Usage:
    python script_utils/test_cpseq_search.py
    pytest script_utils/test_cpseq_search.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.evaluate import check_sequence, violation_magnitude  # noqa: E402
from proteinfoundation.cpseq.search import (  # noqa: E402
    ARM_GUIDED,
    ARM_RANDOM,
    KIND_LOCAL,
    Archive,
    ArchiveEntry,
    budget_for,
    default_alphabet,
    enumerate_single_substitutions,
    guided_round,
    hamming,
    match_counts,
    n_mutations,
    propose_edit,
    proposal_seed,
    random_at_count,
    screen_slots_remaining,
    select_spread_singles,
    standalone_random_arm,
    standalone_random_stream,
    within_budget,
)
from proteinfoundation.cpseq.tolerances import ChemistryTolerance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_search import LADDER_BUDGETS, ladder_plan  # noqa: E402

ORIGINAL_12 = "GEPETGEWTYDD"[:12]
ORIGINAL_7 = "INPYLYP"
ORIGINAL_5 = "CEEVC"


def test_budget_is_floor_half() -> None:
    for L, want in ((5, 2), (7, 3), (12, 6), (13, 6), (16, 8)):
        got = budget_for(L)
        assert got == want, f"L={L}: budget {got} != {want}"
    # floor(L/2) is exactly what keeps identity at or above 50 %.
    for L in range(5, 17):
        assert (L - budget_for(L)) / L >= 0.5


def test_cys_excluded_by_default() -> None:
    alpha = default_alphabet()
    assert "C" not in alpha, "CYS must be excluded from random draws by default"
    assert len(alpha) == 19
    assert "C" in default_alphabet(allow_cys=True)


def test_random_at_count_is_exact() -> None:
    """Exactly `count` substitutions -- never fewer because a draw matched the native."""
    for count in range(0, budget_for(len(ORIGINAL_12)) + 1):
        for trial in range(25):
            seq = random_at_count(
                ORIGINAL_12, count, seed=proposal_seed("exact", count, trial)
            )
            assert seq is not None
            assert len(seq) == len(ORIGINAL_12)
            got = n_mutations(ORIGINAL_12, seq)
            assert got == count, f"asked {count}, got {got} ({seq})"


def test_random_at_count_refuses_over_budget() -> None:
    over = budget_for(len(ORIGINAL_12)) + 1
    assert random_at_count(ORIGINAL_12, over, seed=1) is None


def test_single_scan_enumeration_size() -> None:
    """L x 18 over the 19-residue alphabet minus the native residue at each position."""
    singles = enumerate_single_substitutions(ORIGINAL_7)
    # Every position contributes len(alphabet) - 1 when its native residue is in the
    # alphabet, len(alphabet) when it is not (e.g. a native CYS, which is excluded).
    alpha = set(default_alphabet())
    expected = sum(len(alpha) - (1 if aa in alpha else 0) for aa in ORIGINAL_7)
    assert len(singles) == expected, f"{len(singles)} != {expected}"
    # Each is exactly one substitution away.
    for pos, aa, seq in singles:
        assert n_mutations(ORIGINAL_7, seq) == 1
        assert seq[pos] == aa


def test_single_scan_spreads_across_positions() -> None:
    """16 probes on a 7-mer must cover all 7 positions before repeating any."""
    props = select_spread_singles(ORIGINAL_7, 16, seed=7)
    assert len(props) == 16
    positions = [p.edited_positions[0] for p in props]
    assert set(positions) == set(range(len(ORIGINAL_7))), (
        f"positions covered {sorted(set(positions))}, expected all 7"
    )
    # No duplicate sequences.
    assert len({p.sequence for p in props}) == 16
    # The first 7 probes are 7 distinct positions -- round-robin, not uniform draw.
    assert len(set(positions[:7])) == 7


def test_single_scan_reports_exhaustion_rather_than_padding() -> None:
    """Asking for more singles than exist returns what exists, with no duplicates."""
    total = len(enumerate_single_substitutions(ORIGINAL_5))
    props = select_spread_singles(ORIGINAL_5, total + 25, seed=3)
    assert len(props) == total, f"{len(props)} != {total} available"
    assert len({p.sequence for p in props}) == len(props)


def test_edits_respect_the_original_budget_not_the_ancestor() -> None:
    """The adversarial case: edit from an ancestor already AT the ceiling.

    A proposer that measured its budget against the ancestor would happily emit a
    sequence at ceiling+2 here, and every individual step would look like "two edits".
    """
    original = ORIGINAL_12
    budget = budget_for(len(original))
    at_ceiling = random_at_count(original, budget, seed=99)
    assert n_mutations(original, at_ceiling) == budget

    emitted = 0
    for trial in range(300):
        p = propose_edit(
            ancestor=at_ceiling,
            original=original,
            n_positions=2,
            arm=ARM_GUIDED,
            round_index=1,
            kind=KIND_LOCAL,
            seed=proposal_seed("ceiling", trial),
        )
        if p is None:
            continue
        emitted += 1
        assert within_budget(original, p.sequence), (
            f"over budget: {n_mutations(original, p.sequence)} > {budget} ({p.sequence})"
        )
    assert emitted > 0, "no proposal was emitted from an at-ceiling ancestor at all"


def test_reversion_is_available_at_the_ceiling() -> None:
    """At the ceiling the only legal edits spend budget back, so some must do that."""
    original = ORIGINAL_12
    budget = budget_for(len(original))
    at_ceiling = random_at_count(original, budget, seed=5)
    counts = []
    for trial in range(200):
        p = propose_edit(
            ancestor=at_ceiling,
            original=original,
            n_positions=1,
            arm=ARM_GUIDED,
            round_index=1,
            kind=KIND_LOCAL,
            seed=proposal_seed("revert", trial),
        )
        if p is not None:
            counts.append(p.n_mutations)
    assert counts, "nothing emitted"
    assert min(counts) < budget, (
        "no proposal ever reverted: at the ceiling every edit stayed at the ceiling, "
        "so the search cannot undo an earlier bad guess"
    )


def test_guided_round_stays_in_budget_from_an_at_ceiling_archive() -> None:
    original = ORIGINAL_12
    budget = budget_for(len(original))
    archive = Archive(capacity=6)
    for k in range(6):
        seq = random_at_count(original, budget, seed=1000 + k)
        archive.add(
            ArchiveEntry(
                sequence=seq,
                objective=-20.0 + k,
                violation=0.0,
                valid=True,
                round_index=0,
                kind=KIND_LOCAL,
                ancestor_sequence=original,
            )
        )
    props, note = guided_round(
        original=original,
        archive=archive,
        round_index=1,
        budget=8,
        seed_base="t",
    )
    assert props, f"round produced nothing (note: {note})"
    for p in props:
        assert within_budget(original, p.sequence), f"{p.sequence} over budget"
        assert len(p.sequence) == len(original)
    # No duplicates within the round.
    assert len({p.sequence for p in props}) == len(props)


def test_guided_round_mixes_edit_sizes() -> None:
    """Multi-position proposals must actually appear: two edits can help together."""
    original = ORIGINAL_12
    archive = Archive(capacity=6)
    archive.add(
        ArchiveEntry(
            sequence=original,
            objective=-25.0,
            violation=0.0,
            valid=True,
            round_index=0,
            kind=KIND_LOCAL,
            ancestor_sequence=None,
        )
    )
    props, _ = guided_round(
        original=original, archive=archive, round_index=1, budget=8, seed_base="mix"
    )
    kinds = {p.kind for p in props}
    assert "random_restart" in kinds, f"no restarts proposed: {kinds}"
    sizes = {len(p.edited_positions) for p in props}
    assert max(sizes) >= 2, f"every proposal edited one position only: {sizes}"


def test_short_peptide_gets_no_illegal_coordinated_edit() -> None:
    """A 5-mer has a ceiling of 2, so a 3-4 position 'coordinated' edit is impossible."""
    original = ORIGINAL_5
    archive = Archive(capacity=4)
    archive.add(
        ArchiveEntry(
            sequence=original, objective=-10.0, violation=0.0, valid=True,
            round_index=0, kind=KIND_LOCAL, ancestor_sequence=None,
        )
    )
    props, _ = guided_round(
        original=original, archive=archive, round_index=1, budget=8, seed_base="short"
    )
    for p in props:
        assert n_mutations(original, p.sequence) <= 2, f"{p.sequence} exceeds ceiling 2"


def test_arm_a_matches_arm_b_counts_exactly() -> None:
    """Arm A's realised mutation-count multiset equals the counts it was handed."""
    original = ORIGINAL_12
    guided_counts = [1, 1, 1, 2, 2, 3, 4, 6]
    props, note = match_counts(
        original=original,
        counts=guided_counts,
        round_index=0,
        seed_base="match",
    )
    assert not note, f"unexpected coverage shortfall: {note}"
    assert len(props) == len(guided_counts)
    got = sorted(p.n_mutations for p in props)
    assert got == sorted(guided_counts), f"{got} != {sorted(guided_counts)}"
    assert len({p.sequence for p in props}) == len(props)


def test_arm_a_reports_coverage_when_space_is_too_small() -> None:
    """Exhausted space is reported, not padded with duplicate evaluations.

    Count 1 on a 5-mer, not count 2: with two substitutions there are ~3,240 distinct
    sequences and 500 is comfortably satisfiable, so the request has to be genuinely
    bigger than the neighbourhood for this to test anything. Single substitutions on a
    5-mer number 92.
    """
    props, note = match_counts(
        original=ORIGINAL_5,
        counts=[1] * 500,
        round_index=0,
        seed_base="tiny",
    )
    assert len({p.sequence for p in props}) == len(props), "duplicates emitted"
    assert len(props) < 500
    assert "coverage" in note, f"shortfall not reported: {note!r}"


def test_archive_keeps_diverse_and_falls_back_to_violation() -> None:
    archive = Archive(capacity=3, min_distance=2, keep_infeasible=2)
    # Three near-identical valid entries plus one distant one.
    base = ORIGINAL_12
    archive.add(ArchiveEntry(base, -30.0, 0.0, True, 0, KIND_LOCAL, None))
    near = base[:1] + ("A" if base[1] != "A" else "V") + base[2:]
    archive.add(ArchiveEntry(near, -29.0, 0.0, True, 0, KIND_LOCAL, None))
    far = random_at_count(base, 5, seed=42)
    archive.add(ArchiveEntry(far, -28.0, 0.0, True, 0, KIND_LOCAL, None))
    chosen = archive.working_set()
    assert len(chosen) <= 3
    # The veto is honoured even though that returns 2 parents instead of 3: `near` is
    # one substitution from `base` and must be dropped rather than padding the set.
    seqs = [c.sequence for c in chosen]
    assert near not in seqs, "a near-duplicate was padded into the working set"
    for i, a in enumerate(seqs):
        for b in seqs[i + 1 :]:
            assert hamming(a, b) >= 2, f"{a} and {b} too close"

    # With nothing valid, the smallest violations are kept instead.
    arch2 = Archive(capacity=4, min_distance=1, keep_infeasible=3)
    for k, viol in enumerate([5.0, 0.5, 2.0, 9.0]):
        arch2.add(
            ArchiveEntry(
                random_at_count(base, 2, seed=200 + k), None, viol, False, 0, KIND_LOCAL, None
            )
        )
    ws = arch2.working_set()
    assert ws, "no infeasible fallback"
    assert all(not e.valid for e in ws)
    assert ws[0].violation == 0.5, f"fallback not ranked by violation: {[e.violation for e in ws]}"


def test_check_sequence_rejects_before_compute() -> None:
    original = ORIGINAL_12
    ok, reasons, _ = check_sequence(original, original)
    assert ok, reasons

    # Length change.
    ok, reasons, _ = check_sequence(original, original[:-1])
    assert not ok and "length" in reasons[0]

    # Unsupported residue: CYS is out by default, in when allowed.
    with_cys = "C" + original[1:]
    ok, reasons, _ = check_sequence(original, with_cys)
    assert not ok and any("unsupported" in r for r in reasons)
    ok, _, _ = check_sequence(original, with_cys, allow_cys=True)
    assert ok

    # Over budget.
    over = random_at_count(original, budget_for(len(original)), seed=7)
    one_more = [c for c in over]
    for i, (a, b) in enumerate(zip(original, over)):
        if a == b:
            one_more[i] = "A" if a != "A" else "V"
            break
    ok, reasons, info = check_sequence(original, "".join(one_more))
    assert not ok and any("ceiling" in r for r in reasons)
    assert info["n_mutations"] == budget_for(len(original)) + 1


def _seeded_archive(original: str, n: int = 3) -> Archive:
    archive = Archive(capacity=6, min_distance=1, keep_infeasible=3)
    archive.add(ArchiveEntry(original, -25.0, 0.0, True, 0, KIND_LOCAL, None))
    for k in range(n):
        archive.add(
            ArchiveEntry(
                random_at_count(original, 1, seed=500 + k), -24.0 + k, 0.0, True,
                0, KIND_LOCAL, original,
            )
        )
    return archive


def test_guided_round_plan_is_deterministic() -> None:
    """Same parent, same round, same archive -> the same proposals.

    This is what lets a preempted task resume onto its own plan: it regenerates the
    identical round and the journal then skips the evaluations already paid for.
    """
    original = ORIGINAL_12
    a = guided_round(
        original=original, archive=_seeded_archive(original), round_index=1,
        budget=8, seed_base="p1",
    )[0]
    b = guided_round(
        original=original, archive=_seeded_archive(original), round_index=1,
        budget=8, seed_base="p1",
    )[0]
    assert [p.sequence for p in a] == [p.sequence for p in b], (
        "the round is not reproducible, so a resumed task cannot land on its own plan"
    )


def test_excluding_prior_proposals_makes_a_resume_overspend() -> None:
    """Why the journal must NOT be handed to the proposers as `exclude`.

    Measured in the real run before this was fixed: `lnr_5njx` was preempted and came
    back with 36 guided evaluations against a budget of 32, while parents that never
    requeued sat at exactly 32. The resume duplicated nothing -- it BOUGHT EXTRA, which
    hands one arm more slots than the arm it is compared against.

    The mechanism is below: excluding a round's own earlier output makes it propose
    entirely different sequences instead of reproducing them.
    """
    original = ORIGINAL_12
    first = guided_round(
        original=original, archive=_seeded_archive(original), round_index=1,
        budget=8, seed_base="p2",
    )[0]
    assert first

    # Replay the same round while excluding what it produced last time -- i.e. what a
    # journal-derived exclude set does on resume.
    replay = guided_round(
        original=original, archive=_seeded_archive(original), round_index=1,
        budget=8, seed_base="p2", exclude={p.sequence for p in first},
    )[0]

    overlap = {p.sequence for p in first} & {p.sequence for p in replay}
    assert not overlap, "exclude was not honoured; this test no longer tests anything"
    assert replay, "nothing proposed on replay"
    # Every replayed sequence is NEW work on top of the budget already spent.
    assert len(first) + len(replay) > 8, (
        f"{len(first)} + {len(replay)} should exceed the round budget of 8 -- that "
        "overspend is exactly the bug"
    )


def test_a_grown_archive_cannot_buy_extra_slots_on_resume() -> None:
    """The level that actually caps an arm, as opposed to the two that do not.

    Deterministic proposals are not enough. Round 0 is a pure function of the parent, so
    a resume reproduces and skips it, but rounds 1+ read the ARCHIVE -- and a resumed
    task rebuilds that archive from a journal that has grown, so it plans *different*
    sequences and pays for them on top of the work already done. Nothing is duplicated,
    and the arm still ends up with more slots than the arm it is compared against.

    Observed live on `lnr_5njx` across two preemptions: 32 -> 36 -> 45 guided
    evaluations against a budget of 32, by a process that had imported the code before
    the cap existed. These are those numbers.
    """
    cap = 32
    at_budget = [
        {"arm": ARM_GUIDED, "stage": "screen", "sequence": f"s{i}"} for i in range(cap)
    ]
    assert screen_slots_remaining(at_budget, ARM_GUIDED, cap) == 0, (
        "an arm at exactly its budget must have no slots left, whatever the archive "
        "now proposes"
    )

    # The real overspent journals: the cap must not go negative and must not reopen.
    for spent in (36, 45):
        journal = [
            {"arm": ARM_GUIDED, "stage": "screen", "sequence": f"s{i}"}
            for i in range(spent)
        ]
        assert screen_slots_remaining(journal, ARM_GUIDED, cap) == 0, (
            f"{spent} spent against cap {cap} must clamp to 0, not reopen the arm"
        )

    # The cap is per arm: one arm's overspend must not throttle the other, or the
    # matched comparison would be broken from the other side.
    mixed = at_budget + [
        {"arm": ARM_RANDOM, "stage": "screen", "sequence": f"r{i}"} for i in range(5)
    ]
    assert screen_slots_remaining(mixed, ARM_RANDOM, cap) == cap - 5

    # Confirmation rows and references share the journal and must not consume screening
    # slots -- the confirmation stage re-evaluates finalists at three fresh seeds, which
    # would otherwise look like 3x the budget.
    with_confirm = at_budget[:10] + [
        {"arm": ARM_GUIDED, "stage": "confirm", "sequence": "s0", "prediction_seed": s}
        for s in (101, 202, 303)
    ] + [{"arm": "reference", "stage": "screen", "sequence": ORIGINAL_12}]
    assert screen_slots_remaining(with_confirm, ARM_GUIDED, cap) == cap - 10


def test_native_cysteine_is_admissible_where_the_original_has_it() -> None:
    """A real cohort parent, `lnr_2v8x`, carries a native CYS.

    CYS is excluded from substitution draws so no unrequested crosslink partner appears,
    but a flat alphabet test would reject this parent's own reference sequence and every
    candidate derived from it. The rule is positional.
    """
    original = "RIIYDRKFLMECRN"
    assert "C" in original and "C" not in default_alphabet()

    # The reference itself must pass.
    ok, reasons, info = check_sequence(original, original)
    assert ok, f"the parent's own sequence was rejected: {reasons}"
    assert info["n_mutations"] == 0

    # Keeping the native CYS while mutating elsewhere is fine.
    kept = original[:1] + "A" + original[2:]
    ok, reasons, _ = check_sequence(original, kept)
    assert ok, reasons

    # Substituting the native CYS away is a legal mutation.
    ci = original.index("C")
    away = original[:ci] + "A" + original[ci + 1 :]
    ok, reasons, info = check_sequence(original, away)
    assert ok, reasons
    assert info["n_mutations"] == 1

    # Introducing a CYS anywhere else is still refused.
    intro = original[:1] + "C" + original[2:]
    ok, reasons, _ = check_sequence(original, intro)
    assert not ok and any("introduced" in r for r in reasons), reasons


def test_proposers_never_introduce_cysteine() -> None:
    """Every proposer draws from the alphabet, so a new CYS cannot appear."""
    original = "RIIYDRKFLMECRN"
    ci = original.index("C")
    seqs = [p.sequence for p in select_spread_singles(original, 20, seed=11)]
    for count in (1, 3, 5):
        for trial in range(20):
            s = random_at_count(original, count, seed=proposal_seed("cys", count, trial))
            if s:
                seqs.append(s)
    assert seqs
    for s in seqs:
        for i, c in enumerate(s):
            if c == "C":
                assert i == ci, f"CYS introduced at {i} (native is {ci}): {s}"


def test_violation_is_none_when_the_ring_is_unmeasurable() -> None:
    """An unmeasurable closure must not score as a small violation."""
    tol = ChemistryTolerance(
        chemistry="mainchain",
        bond_length_min_A=1.1, bond_length_max_A=1.6, bond_length_source="t",
        angle_tol_deg=7.54, angle_tol_source="t",
        torsion_tol_deg=8.44, torsion_tol_source="t",
        n_controls=4,
    )
    assert violation_magnitude({"closure": {"status": "atoms_missing"}}, tol) is None

    # A passing ring scores 0; a failing torsion scores > 0.
    good = {
        "closure": {
            "status": "ok", "bond_length_A": 1.33,
            "angle_i_dev_deg": 1.0, "angle_j_dev_deg": 1.0, "dihedral_dev_deg": 3.0,
        },
        "clashes": {"n_intra_clashes": 0, "n_inter_clashes": 0},
    }
    assert violation_magnitude(good, tol) == 0.0
    bad = dict(good)
    bad["closure"] = dict(good["closure"], dihedral_dev_deg=48.5)
    v = violation_magnitude(bad, tol)
    assert v is not None and v > 0, v


def test_stream_draw_k_is_the_same_sequence_at_any_draw_count() -> None:
    """A 96-draw plan must CONTAIN the 48-draw plan, draw for draw.

    This is what makes "we do not know how many draws we need" answerable by appending
    rather than by restarting. `standalone_random_arm` fails this check by construction
    -- it sizes one rng call by the number of slots -- which is why the ladder uses the
    stream instead.
    """
    original = "RIIYDRKFLMECRN".replace("C", "A")
    short, _ = standalone_random_stream(original=original, n_draws=48, seed_base="p1")
    long, _ = standalone_random_stream(original=original, n_draws=96, seed_base="p1")
    assert [p.sequence for p in long[:48]] == [p.sequence for p in short]
    assert [p.slot for p in short] == list(range(48))

    # And the same draws can be appended without re-folding the first 48.
    tail, _ = standalone_random_stream(
        original=original, n_draws=48, start=48, seed_base="p1"
    )
    assert [p.sequence for p in tail] == [p.sequence for p in long[48:]]
    assert [p.slot for p in tail] == list(range(48, 96))

    # The old planner is kept for the runs already on disk, and must NOT be used here:
    # this records why, so the two are never swapped back by accident.
    a32, _ = standalone_random_arm(original=original, n_slots=32, seed_base="p1")
    a64, _ = standalone_random_arm(original=original, n_slots=64, seed_base="p1")
    assert [p.sequence for p in a64[:32]] != [p.sequence for p in a32]


def test_stream_draws_are_distinct_and_in_budget() -> None:
    original = "MEEVD"  # ceiling floor(5/2) = 2, so the space is small on purpose
    props, note = standalone_random_stream(original=original, n_draws=40, seed_base="x")
    seqs = [p.sequence for p in props]
    assert len(seqs) == len(set(seqs)), "a draw was repeated"
    assert original not in seqs, "the unmodified peptide was drawn as a candidate"
    for p in props:
        assert p.n_mutations <= budget_for(len(original))
        assert p.n_mutations >= 1
    # Exhaustion is reported, never padded with duplicates.
    if len(props) < 40:
        assert note and "exhaustion" in note


def test_a_crashed_fold_is_retried_not_retired() -> None:
    """A predictor crash must not satisfy the resume check.

    The failure this guards against was observed: one complex lost its GPU mid-task and
    recorded 72 rows with `technical_failure: true`. Keyed the same way as a real fold,
    those rows made the complex read as complete while 72 of its folds had never
    happened, and no requeue would have gone back for them.
    """
    import json as _json
    import tempfile

    from cpseq_search import Journal

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        j = Journal(out, "cplx")
        j.add_outcome({"sequence": "AAAA", "prediction_seed": 101, "stage": "ladder",
                       "arm": "random_control", "technical_failure": False,
                       "valid": True})
        j.add_outcome({"sequence": "AAAC", "prediction_seed": 101, "stage": "ladder",
                       "arm": "random_control", "technical_failure": True,
                       "status": "prediction_failed"})
        # Re-open, as a requeued task does.
        j2 = Journal(out, "cplx")
        assert j2.has("AAAA", 101, "ladder", "random_control"), "a real fold was re-run"
        assert not j2.has("AAAC", 101, "ladder", "random_control"), (
            "a crashed fold counted as done; it would never be retried"
        )

        # The retry appends; both rows are on disk and the successful one is usable.
        j2.add_outcome({"sequence": "AAAC", "prediction_seed": 101, "stage": "ladder",
                        "arm": "random_control", "technical_failure": False,
                        "valid": True})
        rows = [_json.loads(x) for x in
                (out / "state" / "cplx" / "outcomes.jsonl").read_text().splitlines() if x]
        aac = [r for r in rows if r["sequence"] == "AAAC"]
        assert len(aac) == 2 and sum(not r["technical_failure"] for r in aac) == 1
        assert Journal(out, "cplx").has("AAAC", 101, "ladder", "random_control")


def test_a_cut_off_ladder_is_shorter_never_shallower() -> None:
    """Stopping anywhere must leave a COMPLETE ladder up to some rung.

    The failure this guards against is the one the whole re-run exists to correct: a
    table whose rungs rest on draws that were folded a different number of times, so
    that what looks like a budget effect is a fold-count effect. Running short of time
    is allowed to cost rungs; it is not allowed to cost depth.
    """
    reference = "REF"
    draws = [f"D{i}" for i in range(16)]
    seeds = tuple(range(10))
    rungs = (1, 2, 4, 8, 16)
    plan = ladder_plan([reference] + draws, seeds, rungs)

    # Every (sequence, seed) pair exactly once: nothing folded twice, nothing skipped.
    assert len(plan) == len({(s, sd) for _b, _r, s, sd in plan}) == 17 * 10

    # Cut the plan at 200 arbitrary points. At each, the draws that are complete at full
    # depth must be a PREFIX of the stream, and the unchanged peptide must be complete
    # before the first rung closes.
    for cut in range(0, len(plan), max(1, len(plan) // 200)):
        folds: dict[str, set[int]] = {}
        for _b, rank, seq, _sd in plan[:cut]:
            folds.setdefault(seq, set()).add(rank)
        complete = [d for d in draws if len(folds.get(d, set())) == len(seeds)]
        assert complete == draws[: len(complete)], (cut, complete)
        if complete:
            assert len(folds.get(reference, set())) == len(seeds), (
                f"cut {cut}: draws complete while the peptide they are compared "
                f"against is not"
            )

    # And the rung label on each row is the rung that row belongs to.
    for rung, _rank, seq, _sd in plan:
        if seq == reference:
            assert rung == rungs[0]
        else:
            assert draws.index(seq) < rung


def test_every_ladder_rung_is_a_prefix_of_one_draw_stream() -> None:
    """B=8 must be the first 8 draws of the 64, not a separately planned set.

    This is what lets the whole ladder be read off one run. If the rungs were planned
    independently they would also differ in which sequences they happened to contain,
    and a difference between rungs could not be attributed to the budget.
    """
    original = "SEKFKLLFQSY"
    full, _ = standalone_random_stream(original=original, n_draws=64, seed_base="pid")
    for b in LADDER_BUDGETS:
        if b > 64:
            continue
        rung, _ = standalone_random_stream(original=original, n_draws=b, seed_base="pid")
        assert [p.sequence for p in rung] == [p.sequence for p in full[:b]], b


ALL_TESTS = [
    test_a_crashed_fold_is_retried_not_retired,
    test_stream_draw_k_is_the_same_sequence_at_any_draw_count,
    test_stream_draws_are_distinct_and_in_budget,
    test_a_cut_off_ladder_is_shorter_never_shallower,
    test_every_ladder_rung_is_a_prefix_of_one_draw_stream,
    test_budget_is_floor_half,
    test_cys_excluded_by_default,
    test_random_at_count_is_exact,
    test_random_at_count_refuses_over_budget,
    test_single_scan_enumeration_size,
    test_single_scan_spreads_across_positions,
    test_single_scan_reports_exhaustion_rather_than_padding,
    test_edits_respect_the_original_budget_not_the_ancestor,
    test_reversion_is_available_at_the_ceiling,
    test_guided_round_stays_in_budget_from_an_at_ceiling_archive,
    test_guided_round_mixes_edit_sizes,
    test_short_peptide_gets_no_illegal_coordinated_edit,
    test_arm_a_matches_arm_b_counts_exactly,
    test_arm_a_reports_coverage_when_space_is_too_small,
    test_archive_keeps_diverse_and_falls_back_to_violation,
    test_check_sequence_rejects_before_compute,
    test_guided_round_plan_is_deterministic,
    test_excluding_prior_proposals_makes_a_resume_overspend,
    test_a_grown_archive_cannot_buy_extra_slots_on_resume,
    test_native_cysteine_is_admissible_where_the_original_has_it,
    test_proposers_never_introduce_cysteine,
    test_violation_is_none_when_the_ring_is_unmeasurable,
]


if __name__ == "__main__":
    print(f"running {len(ALL_TESTS)} search checks\n")
    failures = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  OK   {test_fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(test_fn.__name__)
            print(f"  FAIL {test_fn.__name__}: {e}")

    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
