# Plan — does folding with potentials ON hold the arm's advantage, before we scale

Status: **intended, not yet run.** Written 2026-11-01. This is a decision record, not a
result. Read it with `README_CPSEQ_MPNN_FILL_SETTINGS.md` §0.2–§0.3, which hold the
numbers it refers to.

## The decision, in plain terms

Before spending ~90 GPU-h running a proposer arm on the corpus, flip **one** setting —
folding potentials, off → on — on the **same 10-complex cohort that produced the pilot
numbers**, and check that the arm's advantage over the native peptide does not collapse.

Cheap first, expensive second. If potentials alone tank the number, scaling teaches us
nothing and we have paid almost nothing to find that out. If the number holds, we scale
with a setting we trust.

## What is held fixed, and the one thing that changes

"Do not change anything that produced the −5.23" means: **same sequences, same complexes,
same seeds, same gate, same estimator.** The draws (the proposed mutant sequences) are
already on disk and must be reused verbatim, not re-proposed — the proposals are
reproducible from `proposal_seed`, but reusing the files on disk removes all doubt.

| held fixed | value |
|---|---|
| complexes | the 10 with a v3 baseline in `llm_ladder` (6 PepBench, 4 LNR) |
| sequences (draws) | the exact ones already journaled, reused from `mask_proposals/` and the candidate journals |
| seeds / draw count | unchanged — same 10 seeds, same number of draws per target |
| gate | `v3-cis-amide-muco-windows` |
| estimator | `--pilot` = `--draws all --directions 2 --min-ref-valid 3 --energy-key refined_dG_separated` |
| **CHANGES → folding potentials** | **off → on** |

Only the predictor configuration moves. Because ΔdG is paired against the reference, the
reference peptide must be re-folded with potentials on **too** — a ΔdG that mixes a
potentials-on arm against a potentials-off reference is not a comparison.

## Which arm

Run **`llm_search`** as the lead, and `random_mask_mpnn_fill` alongside it.

Under the pinned estimator (`README_CPSEQ_MPNN_FILL_SETTINGS.md` §0.2) `llm_search` reads
**−7.23 REU, 9/1** against `random_mask_mpnn_fill`'s **−5.23, 10/0**. §0.1 calls the two a
tie (0.24 SE, 5/5 sign split), so the pilot does not separate them — but `llm_search` has
one decisive advantage for *this* check: it is the only arm with **no v5 rows** (3,150 v3
/ 0 v5), so its v3 cohort is whole, where the mask arms are gate-split and currently reach
full draw depth on only a handful of complexes. If we are re-folding anyway, lead with the
arm whose baseline is clean. Running both keeps the §0.1 tie honest rather than quietly
promoting one.

## The prediction, and what each outcome means

Potentials were measured this session to be **nearly energy-neutral**: paired on 24
targets shared between the two corpus runs, the reference's median dG moved **+0.10 REU**
(ON − OFF) while the pass rate rose 0.44 → 0.67, and the 4-complex probe in
`old_docs/README_CPSEQ_PEPBENCH_SCALEUP.md` §"Steering potentials were off" found interface
energies unchanged (−18.2/−25.4/−69.3/−29.0 vs −18.9/−24.8/−68.1/−29.0) with
peptide–receptor clashes eliminated (299 → 0).

So the **prediction is that the arm number barely moves.** That +0.10 was measured on
reference peptides and random draws, not on the arm's selected winners, which is exactly
why this check is worth running rather than asserting.

| outcome | reading |
|---|---|
| arm stays within ~1 REU of its pilot value | potentials are safe; scale with them on, the clash failure mode gone for free |
| arm drops toward the random-control level | the pilot advantage was partly an artefact of unsteered folding; re-examine before scaling |
| arm improves | the clash-elimination recovers complexes the OFF run lost; even better, but check it is not just more complexes now clearing the baseline |

Watch `no_base` and `unmeasurable`: potentials should **raise** the number of complexes
with a usable baseline (the native peptide folds valid more often), so the comparison may
cover more than 10 of the 20. That is a gain, but it changes the complex set, so report
the fixed-10 subset and the enlarged set separately.

## How it would be run (for whoever implements it)

Potentials is a **property of the cohort**, baked into `cohort.json` at creation
(`search.py:1863` reads `cohort.get("use_potentials")`); there is no per-run sbatch flag.
So the check needs a **sibling cohort** — same 10 parents, same seeds, same draws, same
proposals — created with `--use-potentials`, folded into its own directory
(e.g. `/data/user_data/yixiz/cpseq/llm_ladder_pot`). The proposals on disk under
`llm_ladder/mask_proposals/` and the `llm_search` journals are copied in so the sequences
are identical by construction, not merely reproducible.

Then, per arm:

    bash scripts/cpseq_pilot_table.sh            # but pointed at the potentials-on dir
    # i.e. OUT_DIR=/data/user_data/yixiz/cpseq/llm_ladder_pot ARMS="llm_search random_mask_mpnn_fill" ...

and compare each arm's `--pilot` row to its `llm_ladder` (potentials-off) row, same 10
complexes.

Open implementation question to settle before running: whether to reuse the existing
`llm_ladder` reference folds (no — they are potentials-off) or re-fold the reference under
potentials (yes). The reference re-fold is the one unavoidable new cost even for the
sequences we already have.

## Cost

Small, which is the whole point. Per arm, 10 complexes × (~8–16 draws + 1 reference) × 10
folds ≈ 1,000–1,700 folds. At ~100 s/fold with potentials on (1.48× the OFF rate, but
1.17× cheaper per *usable* fold) that is roughly **25–35 GPU-h per arm**, against ~90 for
the corpus arm run. The reference re-fold is shared across arms in one directory.

## If it holds

Then — and only then — run the arm on the corpus, potentials on, against the 391
complexes that already hold a paid-for v3 reference (which would themselves need a
potentials-on reference, or the corpus reference re-folded once). That is the run that
turns "the arm beats random draws" from a 10-complex claim into a corpus-scale one. The
quantity to defend there is the **arm-minus-random-control gap**, paired per complex, not
the absolute −5.23: §0.3 showed the absolute number is inflated by a favourable 10-complex
sample and by draw count, both of which cancel in the paired gap.

## What this plan deliberately does NOT do

- It does not re-propose any sequence. Same draws, reused.
- It does not change the gate, the estimator, the seeds, or the draw count.
- It does not touch `llm_ladder` itself; the potentials-on folds go to a sibling directory
  so the −5.23 run stays exactly as it is on disk and in the published bundle.
