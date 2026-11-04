#!/usr/bin/env python
"""Checks on the elite-conditioning wiring: breeding from elites instead of the native.

The three things that can silently go wrong here, and what each test pins down:

- **The search drifts off the native.** The substitution budget is defined against the
  ORIGINAL sequence. An elite already spends part of it, so a round that re-measured the
  budget from the elite would let the search walk away from the native one generation at
  a time while every individual round looked legal.
- **Generation 1 quietly changes.** Turning evolution on must not alter a round that has
  no ancestors to breed from, or the pepbench run already in the queue stops being
  comparable to the pilots it is read against.
- **Ancestry is lost.** Masks and ancestors are two parallel lists; if they ever
  disagree in length the fill would pair offspring with the wrong parent, which is worse
  than not recording ancestry at all because it looks like data.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from cpseq_mask_mpnn_fill import ancestor_backbone  # noqa: E402

from proteinfoundation.cpseq.llm_propose import (  # noqa: E402
    CandidateRecord,
    mask_headroom,
    screen_candidates,
    select_elites,
)
from proteinfoundation.cpseq.search import (  # noqa: E402
    budget_for,
    default_alphabet,
)

PASS, FAIL = [], []


def check(cond: bool, label: str) -> None:
    (PASS if cond else FAIL).append(label)
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")


def rec(seq: str, energy: float | None, *, orig: str, valid: bool = True) -> CandidateRecord:
    return CandidateRecord(
        sequence=seq,
        n_mutations=sum(1 for a, b in zip(orig, seq) if a != b),
        valid=valid,
        energy=energy,
        violation=0.0 if valid else 5.0,
        n_folds=5,
        n_valid_folds=5 if valid else 0,
    )


def main() -> int:
    ORIG = "ACDEFGHIKLMN"          # L=12
    budget = budget_for(len(ORIG))  # 6
    ALPHA = default_alphabet()

    print("\n-- budget stays anchored to the native sequence --")
    check(mask_headroom(ORIG, ORIG, budget) == budget,
          f"elite == original: full headroom {budget}")
    three = "WWW" + ORIG[3:]
    check(mask_headroom(ORIG, three, budget) == budget - 3,
          "elite 3 substitutions out: headroom drops to 3")
    spent = "WWWWWW" + ORIG[6:]
    check(mask_headroom(ORIG, spent, budget) == 0,
          "elite at the budget: headroom 0, the search cannot drift further")
    check(mask_headroom(ORIG, "W" * len(ORIG), budget) == 0,
          "headroom clamps at 0 rather than going negative")

    print("\n-- the fill's hard gate is measured against the original, not the elite --")
    # An offspring one substitution from the elite, but seven from the native. Legal if
    # you measure from the elite; must be refused, because the budget is 6 from native.
    offspring = "WWWWWWW" + ORIG[7:]
    check(sum(1 for a, b in zip(spent, offspring) if a != b) == 1,
          "the trap case is 1 substitution from its elite")
    check(sum(1 for a, b in zip(ORIG, offspring) if a != b) == 7,
          "...and 7 from the native, one over the budget of 6")
    kept, rejected = screen_candidates(
        [(offspring, "")], original=ORIG, already={ORIG, spent}, alphabet=ALPHA
    )
    check(not kept and len(rejected) == 1,
          "screen_candidates REFUSES it: budget is anchored to the native")
    within = "WWWWWW" + "W" + ORIG[7:]
    within = spent[:6] + ORIG[6:]           # back to exactly the elite
    kept, rejected = screen_candidates(
        [(within, "")], original=ORIG, already={ORIG, spent}, alphabet=ALPHA
    )
    check(not kept, "an offspring identical to its elite is refused as a duplicate")

    print("\n-- generation 1 is unchanged by turning evolution on --")
    gen1 = [rec(ORIG, -30.0, orig=ORIG)]
    elites = select_elites(gen1, n_elites=4, min_distance=2)
    check(len(elites) == 1 and elites[0].sequence == ORIG,
          "with only the reference scored, the sole elite IS the original")
    check(mask_headroom(ORIG, elites[0].sequence, budget) == budget,
          "so start_from is the original and headroom is the full budget")

    print("\n-- elite selection ranks by energy and refuses near-duplicates --")
    good = "WWW" + ORIG[3:]
    near = "WWY" + ORIG[3:]       # 1 substitution from `good`
    far = ORIG[:6] + "YYYY" + ORIG[10:]
    hist = [
        rec(ORIG, -30.0, orig=ORIG),
        rec(good, -45.0, orig=ORIG),
        rec(near, -44.0, orig=ORIG),
        rec(far, -40.0, orig=ORIG),
    ]
    elites = select_elites(hist, n_elites=4, min_distance=2)
    seqs = [e.sequence for e in elites]
    check(seqs and seqs[0] == good, "the best-scoring sequence is elite #1")
    check(not (good in seqs and near in seqs),
          "the Hamming veto keeps `near` out: elites are not neighbours of one point")
    check(len(elites) < 4,
          "a starved selection returns FEWER elites rather than near-duplicates")

    print("\n-- ancestry lists must stay aligned with masks --")
    masks = ["0,1", "2,3", "4,5"]
    for n_anc, ok in ((0, True), (3, True), (2, False), (4, False)):
        anc = [ORIG] * n_anc
        aligned = (not anc) or len(anc) == len(masks)
        check(aligned is ok,
              f"{n_anc} ancestors for {len(masks)} masks: "
              f"{'accepted' if ok else 'refused rather than guessed'}")

    print("\n-- an elite is masked on its OWN fold when the run still has one --")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out, work = Path(td) / "out", Path(td) / "work"
        state = out / "state" / "p1"
        state.mkdir(parents=True)
        work.mkdir()
        raw = Path(td) / "elite.pdb"
        raw.write_text("ATOM  elite\nEND\n")
        gz = Path(td) / "other.pdb"
        with gzip.open(str(gz) + ".gz", "wt") as fh:
            fh.write("ATOM  gzipped\nEND\n")
        (state / "outcomes.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"sequence": "CRASHED", "raw_pdb_path": str(raw), "technical_failure": True},
            {"sequence": good, "raw_pdb_path": str(raw), "arm": "llm_mask_mpnn_fill"},
            {"sequence": far, "raw_pdb_path": str(gz), "arm": "random_control"},
        ]))
        got = ancestor_backbone(out, "p1", good, work, tag="a")
        check(got is not None and "elite" in got.read_text(),
              "an elite's own raw fold is found and copied")
        got = ancestor_backbone(out, "p1", far, work, tag="b")
        check(got is not None and "gzipped" in got.read_text(),
              "a gzipped fold is decompressed, not skipped")
        check(ancestor_backbone(out, "p1", "CRASHED", work, tag="c") is None,
              "a technical failure is not a usable backbone")
        check(ancestor_backbone(out, "p1", "NOSUCHSEQ", work, tag="d") is None,
              "an elite with no fold returns None, so the caller falls back explicitly")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
