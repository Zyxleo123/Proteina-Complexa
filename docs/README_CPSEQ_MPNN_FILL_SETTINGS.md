# Locked settings — random mask, ProteinMPNN fill

> ## STANDARD — read this first, with `README_CPSEQ_MASK_ARM.md`
>
> **These two documents are the current record for the mask/fill arms.** Together they
> supersede any earlier arm description for this work. `README_CPSEQ_MASK_ARM.md` is how
> the arm works and what the proposer probes measured; this document is the settings and
> the arm choice. Everything else named below is a dependency, not an alternative:
> `old_docs/README_CPSEQ_LLM_SEARCH.md` for `llm_search` (whose §14.5 withdrawals still stand),
> `old_docs/README_CPSEQ_PEPBENCH_SCALEUP.md` for the corpus runs.
>
> **Where the arms live.** All four mask/fill arms are in
> `/data/user_data/yixiz/cpseq/llm_ladder`, on the 20-complex LNR+PepBench cohort — **not**
> in `pepbench_corpus` or `pepbench_corpus_pot`, which hold `random_control` and
> `reference` only and are a different question (random draw vs native peptide, corpus
> scale). Looking for the arms in the corpus directories finds nothing and is the
> confusion this banner exists to prevent.
>
> | arm | rows | what it is |
> |---|---|---|
> | `random_mask_mpnn_fill` | 4,931 | **primary** (§0) |
> | `llm_mask_mpnn_fill` | 4,563 | LLM picks positions, MPNN fills |
> | `llm_mask_random_fill` | 4,837 | LLM picks positions, random fills |
> | `llm_search` | 3,150 | arm D, retained co-leading (§0.1) |
> | `random_control` | 11,998 | arm A baseline |
> | `reference` | 400 | the unchanged peptide, 200 v3 + 200 v4 |
>
> **Report with the repo's own scripts, naming the gate and the arm** — the protocol is
> theirs, not the caller's:
> ```
> python script_utils/cpseq_selected_candidate_report.py --out-dir <dir> \
>     --arm <arm> --scoring-version v3-cis-amide-muco-windows     # best vs peptide
> python script_utils/cpseq_arm_compare.py --out-dir <dir> \
>     --arm-a random_control --arm-b <arm> --scoring-version v3-cis-amide-muco-windows
> ```
> Both refuse to print a comparison from a gate-mixed directory, and `llm_ladder` is
> gate-mixed (v3 22,642 / v5 6,150 / v4 1,087).
>
> **Use v3.** Verified 2026-11-01 — see §5.2, whose open question is now answered and the
> answer is not the one it expected.
>
> **Published bundles** (journals only; every number redraws from them, no structure
> needed, and each replays with no repo and no `/data`):
> `b2:proteina/cpseq/results/llm_ladder_pilot_table.tar.zst` (5.4 MB, the §0.2 arm table)
> and `b2:proteina/cpseq/results/pepbench_corpus_pot_v3.tar.zst` (12 MB, §0.3's 391
> complexes). Built by `scripts/cpseq_result_bundle.sh`, which measures each bundle's
> caveats from the journals it packs rather than copying another run's.
>
> **−7.82 is superseded by −5.23 (§0.2).** It was a pilot number on 10 of 20 complexes
> with the winner chosen from every draw the arm made, and that estimator is now
> `cpseq_selected_candidate_report.py --pilot`. Re-measured on the same gate and cohort
> the primary arm reads **−5.23 REU, 10/0, p = 0.0020**; `llm_search` reproduces its
> −7.23 exactly. The default (no `--pilot`) rungs table answers a different question —
> per-budget B, not all draws — and its numbers are not comparable to either. §0.2,
> §5.2 and §5.3 bound all of them.

The configuration the scale-up runs. Fixed 2026-10-30 after the pilot; every value below
is the one the pilot result was measured under, or is flagged as a change that still
needs a decision (§5).

**Scope.** §1–§4 specify the primary arm, `random_mask_mpnn_fill`. `llm_search` is
retained as a co-leading arm because the pilot does not separate the two (§0.1); its own
settings live in `old_docs/README_CPSEQ_LLM_SEARCH.md` and are unchanged by this document. §2, §3
and the one-writer-per-complex rule apply to **both** arms, since they are what makes any
two arms comparable.

**Why this arm.** Measured one method at a time, per complex, cross-fitted — select the
best candidate on half the folds, report its energy from the other half:

| | arm | complexes | median held-out ΔdG | improves | mean draws |
|---|---|---|---|---|---|
| **PRIMARY** | **`random_mask_mpnn_fill`** | 10 | **−7.82 REU** | **10/10** | 7.7 |
| **RETAINED** | **`llm_search`** | 10 | **−7.23 REU** | **9/10** | 13.1 |
| | `random_control` | 10 | −5.27 | 9/10 | 42.0 |
| | `llm_mask_mpnn_fill` | 10 | −3.49 | 9/10 | 7.5 |
| | `llm_mask_random_fill` | 7 | −3.34 | 6/7 | 7.7 |

The primary arm improves on every complex with a usable baseline, from a fifth as many draws as
the random control. The paired contrast that matters:
`random_mask_mpnn_fill` −7.82 against `llm_mask_mpnn_fill` −3.49 at near-identical draw
counts with the filler held constant, so **random position choice beat the LLM's position
choice**. The second contrast, −7.82 against `llm_mask_random_fill` −3.34, says the fill
model is doing the work.

Read these numbers only with §0.1, §0.2, §5.1 and §5.2 in hand.

## 0.2 Re-measured 2026-11-01 on the same estimator. The arm ordering changed.

§3's estimator is now implemented, as `cpseq_selected_candidate_report.py --pilot`
(`--draws all --directions 2 --min-ref-valid 3 --energy-key refined_dG_separated`), so
§0's table is redrawable for the first time. Same gate (v3), same cohort, same rule:

| arm | §0 ΔdG | **re-measured** | §0 improves | **w/l now** | sign p | §0 draws | draws now |
|---|---|---|---|---|---|---|---|
| `llm_search` | −7.23 | **−7.23** | 9/10 | **9/1** | 0.022 | 13.1 | 15.7 |
| `llm_mask_random_fill` | −3.34 | **−5.99** | 6/7 | **8/1** | 0.039 | 7.7 | 11.7 |
| `random_mask_mpnn_fill` | **−7.82** | **−5.23** | 10/10 | **10/0** | **0.0020** | 7.7 | 10.7 |
| `random_control` | −5.27 | −3.65 | 9/10 | 8/1 | 0.039 | 42.0 | 59.2 |
| `llm_mask_mpnn_fill` | −3.49 | −2.16 | 9/10 | 7/2 | 0.180 | 7.5 | 9.8 |

**`llm_search` reproduces to the digit**, which is what certifies the implementation: it
is the one arm with no v5 rows (3,150 v3 / 0 v5). Every arm that moved is an arm whose
folds were later split across gates — `random_mask_mpnn_fill` 2,204 v3 / 1,840 v5,
`llm_mask_mpnn_fill` 2,613 / 1,950, `llm_mask_random_fill` 2,477 / 2,360 — so the v3 draws it keeps
today is a ~55 % subset of what the pilot measured, selected by which folds a later
rescore happened to touch. That is the leading explanation for the movement and it is
fixable: see §5.2's v5→v3 recommendation.

What survives and what does not:

- **Survives.** `random_mask_mpnn_fill` is still the only arm negative on every complex
  with a baseline (10/0, p = 0.0020, the lowest of the five), still from a fifth as many draws
  as the control. The §0 contrast against `llm_mask_mpnn_fill` at matched filler also
  survives, −5.23 against −2.16, so **random position choice still beats the LLM's**.
- **Does not survive.** §0's second contrast, "the fill model is doing the work"
  (−7.82 against `llm_mask_random_fill` −3.34), **reverses**: −5.23 against −5.99. On
  today's v3 rows the random fill is marginally better than the MPNN fill, and §0's
  −3.34 was measured on 7 complexes against the others' 10.
- **Changes sign.** §0.1 put `random_mask_mpnn_fill` 0.59 REU ahead of `llm_search` and
  called it a tie at 0.24 SE. It is now 2.0 REU behind. The tie verdict stands — this is
  the same quantity moving within its spread — but the primary-arm choice now rests
  entirely on 10/0 against 9/1 and on proposal cost, with no energy margin at all.

**`no_base` is 10 for all five arms**, confirming §5.3 exactly: the same ten complexes
have a reference with fewer than 3 valid folds, so every row above is 10 of 20.

Redraw: `cpseq_selected_candidate_report.py --out-dir /data/user_data/yixiz/cpseq/llm_ladder
--scoring-version v3-cis-amide-muco-windows --arm <arm> --pilot`, or
`bash scripts/cpseq_pilot_table.sh` for all five.

## 0.3 What exists at scale, and which axis moves a number

**There is a corpus-scale result, and it is not an arm result.** `pepbench_corpus_pot`,
v3, same search and same report, under this section's own estimator:

| | complexes | ΔdG | w/l | sign p | d(valid) |
|---|---|---|---|---|---|
| `random_control` vs peptide, `--pilot` | **391** | **−1.43 REU** | 250/141 | **3.9e-8** | +0.000 (p = 0.76) |

Best-of-8 random substitution beats the unchanged native peptide at 39x the arm table's
complex count, with no validity cost. **No proposer has ever been run on those 391
complexes**, which is the gap: §0.2's −5.23 and this −1.43 are different cohorts at
different draw counts and are not comparable. 391 of them already have a paid-for v3
reference at full depth, so the baseline for an arm run at scale is already bought.

**Why the same run reads −2.21 or −1.43.** One axis changed at a time, `random_control`,
v3, so a number seen elsewhere can be placed:

| estimator | n | ΔdG | w/l | unmeas | no_base |
|---|---|---|---|---|---|
| default, B=8 | 400 | −2.21 | 249/124 | 6 | 21 |
| `--draws all` | 418 | −2.24 | 257/127 | 13 | 21 |
| `--min-ref-valid 3` | 374 | −2.24 | 246/122 | 4 | 29 |
| `--directions 2` | 400 | **−1.60** | 249/**136** | **0** | 15 |
| full `--pilot` | 391 | −1.43 | 250/141 | 0 | 27 |

- **`--draws all` is a no-op here.** The corpus arm made exactly 8 draws per target, so
  B=8 *is* all of them. This is the axis that dominates on `llm_ladder`, where the arms
  made 10–59 draws per target.
- **`--min-ref-valid 3` is a no-op on the estimate**, dropping 26 complexes and moving
  nothing. The ~29 % of corpus targets with a thin baseline are not carrying the effect.
  A reassuring null, and not the one expected.
- **`--directions 2` is the whole difference.** `unmeas` goes 6 → 0 and w/l goes
  249/124 → 249/**136**: twelve losses appear and no wins. The single direction drops a
  complex whose pick passed no fold in ranks 5–9; running the reverse gives it a second
  chance at a measurement, and the rescued complexes are almost all losses. Dropping them
  was quietly optimistic, so the two-direction number is smaller because it stops hiding
  them — not because anything was re-folded.
- **`--energy-key` is a no-op, ruled out.** `best_valid_dG_separated` and
  `refined_dG_separated` are equal on all 14,602 v3 rows across both run directories where
  both exist; they differ only in being null on an invalid fold, which `median_energy_in`
  already drops. They would diverge at `diffusion_samples > 1`; every run here is at 1.

So three axes, not four, and on the corpus only one of the three bites.

## 0.1 `llm_search` is RETAINED, not eliminated

The pilot does not separate the two leading arms. Paired per complex, on the same 10
complexes, held-out:

| | value |
|---|---|
| mean difference | **−0.43 REU** (random_mask better) |
| sd / SE of the difference | 5.67 / 1.79 |
| in standard errors | **0.24 SE** |
| median difference | +0.05 REU |
| sign split | **5/10 random_mask, 5/10 `llm_search`** |

A dead-even sign split and a difference a quarter of one standard error. The medians
−7.82 and −7.23 differ by less than the per-complex spread, and the two arms win on
disjoint complexes: `llm_search` is far better on `pepbench_C_A_pdb2jz3` (−17.98 vs
−8.28) and `random_mask_mpnn_fill` is far better on `pepbench_A_B_pdb2phk` (−9.96 vs
+1.41), which is also the one complex `llm_search` fails.

So the choice of primary rests on **10/10 against 9/10 and a cheaper proposal stage**,
not on a measured advantage. Two consequences:

- `llm_search` keeps its settings frozen (`old_docs/README_CPSEQ_LLM_SEARCH.md`) and stays
  runnable. Do not delete its proposals or journals.
- Its larger draw count (13.1 against 7.7) means it carries *more* selection inflation than
  the primary arm and still matched it. At equal draw counts it may be ahead. §3's
  draw-count finding — held-out value improving up to N≈20 — applies to both, so a
  scale-up at matched, larger draw counts is what would actually separate them.

**What would settle it:** both arms at a matched ~20 draws per complex, on the
complexes that have a baseline, scored under one protocol. That is the smallest
experiment that distinguishes them; anything less re-reads a tie.

---

## 1. Proposal stage

### 1.1 Mask — which positions are opened

| setting | value |
|---|---|
| positions | uniform without replacement over `range(L)` |
| masks per parent per round | 12 |
| substitution budget | `floor(L/2)`, which also guarantees ≥50 % identity to the original |
| mask size | drawn from the pilot's empirical distribution — see §5.1 |
| duplicate masks | rejected within a round; a size with no distinct mask left is **skipped and reported short, never padded** |
| overlap with another arm's masks | allowed and counted, never excluded |

Mask sizes in the pilot, 239 masks over 20 parents: size 1 26 %, size 2 41 %, size 3
24 %, size 4 8 %, size 5 0.4 %. Mean 2.15, median 2, max 5.

`matched_random_masks` draws with `random.Random(seed)`, so a round is reproducible from
`seed` and the size list alone.

### 1.2 Fill — which residues go in

| setting | value |
|---|---|
| model | ProteinMPNN `v_48_020`, vanilla weights |
| checkout | `community_models/ProteinMPNN` |
| temperature | 0.2 |
| `omit_aas` | `XC` |
| seed | 1234 |
| sequences per mask | 1 |
| designed chain | `B` (the peptide) via `--pdb_path_chains B`; every receptor chain is fixed structural context by construction |
| unmasked positions | pinned with `fixed_positions` |

**Backbone: the run's own reference Boltz-2 fold of the original peptide, held fixed.**
Found by reading the journal for an `arm == "reference"` row matching the original
sequence. Not a generator sample — holding the conformation fixed is what makes the mask
the only thing that varies between arms.

**Threading is required, not cosmetic.** `fixed_positions` pins residues to their
identity *in the input PDB*, so the original sequence is written onto the backbone first.
Without it the unmasked positions would be pinned to whatever the input PDB happened to
carry.

### 1.3 Legality

| rule | value |
|---|---|
| alphabet | `ADEFGHIKLMNPQRSTVWY` — 19 residues, no cysteine |
| `allow_cys` | `false`; the filler can never put a cysteine back |
| length | substitutions only; a length change is an error, not a candidate |
| identical to the original | rejected — that is the reference, already measured |
| duplicate sequence within a round | rejected |
| re-proposing an evaluated sequence | refused, not served from cache |

Opening a native cysteine position is the intended use, not a violation: it is how the
search replaces one. Cysteine removal was the single largest effect in the pilot
(median −6.04 REU over 12 draws against −1.30 for draws on cysteine-free originals), so
it is worth tracking separately in any report.

---

## 2. Measurement stage

Identical across every candidate and the reference. Nothing here may differ between a
design and its baseline.

| setting | value |
|---|---|
| predictor | Boltz-2, revision `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` |
| chemistry | `mainchain` — head-to-tail amide, predicted with `cyclic: true` |
| diffusion samples | 1 |
| recycling steps | 3 |
| sampling steps | 200 |
| potentials | off |
| seeds | `101, 202, 303, 404, 505, 606, 707, 808, 909, 111` |
| fold depth | all 10 seeds, **for every draw and for the reference** |
| retention | `rescorable` (~50 KB/fold; keeps the gzipped raw complex so a protocol change costs CPU only) |
| scoring protocol | `v4-postrelax-validity` — see §5.2 |
| reported energy | `refined_dG_separated`, Rosetta ref2015 REU |

**Ten folds is not negotiable.** Boltz-2 returns a different structure per seed, sd
3.67 REU on the computed energy, and a pass/fail check on a single fold is close to a
coin flip at the observed pass rate. Four earlier results were withdrawn because the
apparent effect was extra folds the designs got and the reference did not.

**Evaluation order:** every draw is folded at seed rank *k* before any draw is folded at
rank *k+1*, so an interrupted run is the whole experiment at fewer folds rather than a
few draws at full depth and the rest at none.

**One array task per complex.** The journal is per complex and dedupes on
`(sequence, seed, stage, arm)` — two processes appending to one file interleave and
corrupt rows that cost GPU time to replace. Parallelism comes from the array, never from
splitting a complex, and **never from running two arms concurrently on the same
`OUT_DIR`**.

---

## 3. How the result is read

| step | rule |
|---|---|
| unit | one candidate's median `refined_dG_separated` over its valid folds |
| comparison | difference against **its own parent's** reference, folded the same 10 times |
| baseline admissible | reference needs ≥3 valid folds; below that the complex has no baseline and is reported as such, not dropped silently |
| best candidate | **cross-fitted** — select on seed ranks 0–4, report from 5–9, then the reverse, and average |
| population median | **not** cross-fitted; nothing is selected, so all 10 folds are the best estimate |
| arms | one at a time; never pool arms before selecting |

**Why cross-fitting.** The best of *N* draws is the maximum of *N* noisy values, so
ranking and reporting on the same folds bakes in the noise that won the selection.
Measured on the pilot: naive best per complex median −11.60 and negative in 10/10,
cross-fitted −5.97 and negative in 8/10 when arms are pooled. Selection inflation grows
with the number of draws searched — subsampling the random control's draws gave a selected-to-held-out gap of
1.77 REU at N=2 rising to 6.48 at N=40.

**Why one arm at a time.** Pooling lets the weakest arm's lucky draw win a complex's
selection half and then fail on the held-out half. `lnr_6seo` is the worked example: it
reads +3.59 pooled, where the winner came from `llm_mask_random_fill`, but −7.37 for
`random_mask_mpnn_fill` alone and −6.98 for `llm_search` alone. Pooling turned two
successes into one failure.

**Draw count is worth spending on.** Held-out value improved with draws searched up to
N≈20 and flattened after, while the pilot ran at a mean of 7.7 draws per complex. More
draws per complex is the cheapest available improvement.

---

## 4. Reproducing the pilot number

```
# proposal
python script_utils/cpseq_mask_mpnn_fill.py \
  --out-dir   /data/user_data/yixiz/cpseq/llm_ladder \
  --masks-dir /data/user_data/yixiz/cpseq/llm_ladder/masks/round1 \
  --round 1 --arms random_mask_mpnn_fill \
  --model-name v_48_020 --temperature 0.2 --omit-aas XC --seed 1234 --seqs-per-mask 1

# measurement, one task per complex
sbatch --array=0-19%20 scripts/cpseq_ladder.sbatch \
  OUT_DIR=/data/user_data/yixiz/cpseq/llm_ladder \
  SHARD_COUNT=20 MASK_ARM=random_mask_mpnn_fill MASK_ROUND=1
```

Artefacts: proposals in `mask_proposals/random_mask_mpnn_fill/round1/`, journals in
`state/<parent>/outcomes.jsonl`, scores in `scores/<scoring_version>/<input_hash>.json`.
Every figure is redrawable from those without refolding.

---

## 5. Open before scale-up

### 5.1 Mask sizes are inherited from the LLM — must be replaced

`matched_random_masks(sizes, ...)` takes its size list from **the LLM's masks for that
parent**, by design: the pilot matched sizes so a difference between the two arms could
not be an artefact of one mutating harder. That makes the winning arm depend on an LLM
call it otherwise does not need, which defeats the point of choosing it.

Decide one of:

1. **Sample sizes from the pilot's empirical distribution** (1:26 %, 2:41 %, 3:24 %,
   4:8 %, 5:0.4 %), truncated at `floor(L/2)`. Closest to what was measured; recommended.
2. Uniform on `1..min(4, floor(L/2))`. Simpler, but a different setting from the measured
   one, so −7.82 would not carry over.
3. Keep the LLM solely as a size generator. Cheap per round but keeps a dependency with
   no remaining justification.

Until this is settled the arm is not independently runnable, and a scale-up that quietly
reuses `masks/round1` is reusing the LLM's choices.

### 5.2 −7.82 is a v3 number; v4 is now the default

The pilot was scored under `v3-cis-amide-muco-windows`, which judges validity on the raw
prediction and reports an energy from the relaxed structure — a label from one structure
paired with a number from another. `v4-postrelax-validity` judges the structure the
energy came from and is now the default.

So §0's table must be re-read at v4 before it is quoted as a v4 result. The cheap first
step is running: job `10994185` rescores the 20 references only (200 folds, CPU) to test
whether post-relax validity gives the ten complexes that currently have **no usable
baseline** one for the first time. If it does, the evaluable set roughly doubles and the
whole table needs the retest. If it does not, there is no case for rescoring the ~21,000
remaining folds.

#### ANSWERED 2026-11-01: it does not. Stay on v3, and do not rescore.

Job `10994185`'s own output cannot answer it — **195 of its 200 reference rows have
`validity_reasons: null` and every geometry field null**, i.e. `valid: false` because they
were never scored, not because they failed. Those rows are indistinguishable from
rejections to `n_pass()`, which counts `r["valid"]`, so any v4 reading of `llm_ladder`
silently treats 195 unscored folds as failures. The 5 that did score are 0/5 valid.

The question is answerable from `pepbench_corpus`, where the v4 rescore did complete.
Reference folds, per gate, targets at full depth 10:

| gate | targets | ≥3 valid reference folds | zero valid |
|---|---|---|---|
| `v3-cis-amide-muco-windows` | 70 | **50 (71%)** | 7 |
| `v4-postrelax-validity` | 201 | **17 (8.5%)** | 113 |

Per-fold reference pass rate across the three runs, scored rows only — the gate choice
dominates every effect any arm could have:

| gate | `pepbench_corpus` | `pepbench_corpus_pot` | `llm_ladder` |
|---|---|---|---|
| v3 | 43.8 % | 70.8 % | 31 % |
| v4 | 7.3 % | 6.1 % | 0/5 scored |
| v5 | — | 97.1 % | — |

v4 does not double the evaluable set; it removes it. §5.3's problem — a complex with no
baseline is not a negative result — gets about eight times worse under v4, because the
native peptide itself almost never passes. v5 fails the opposite way (97–98 %, the gate
rather than the designs). **v3 is the only one of the three where both the reference and
the draws pass often enough to be compared**, which is why the banner says v3.

Two consequences. The ~21,000 remaining folds should not be rescored, per this section's
own stopping rule. And the v5 rows already written are not a bonus: they split
`llm_mask_mpnn_fill` 2,613 v3 / 1,950 v5 and `random_mask_mpnn_fill` 2,204 / 1,840, so
under a single gate those arms reach only B=4–8 on 4–11 complexes of a plan that went to
B=64. Rescoring **v5 back to v3** on the mask arms is the one rescore with a case for it.

Redraw: the tables above come from `state/*/outcomes.jsonl` in each run directory, keyed
on `arm`, `scoring_version`, `valid`, and whether `closure_status` is null.

### 5.3 Half the complexes are unmeasurable

Ten of 20 have references with fewer than three valid folds; four produced zero. Those
complexes have no baseline to beat rather than a negative result, and "improves 10/10"
means 10 of the 10 that can be evaluated. §5.2 is the first attempt to fix this.

### 5.4 ProteinMPNN does not know the peptide is cyclic

Vanilla ProteinMPNN derives topology from `residue_idx[i] - residue_idx[j]`, clipped to
±32, with no wrap, so for a head-to-tail 14-mer positions 0 and 13 read as 13 apart when
they are covalently bonded. It is proposing for a cut ring, on PDB linear-protein weights.
`src/proteinfoundation/evaluation/cyclic_offset.py` already implements the fix for the
same reason on the refolding side and is restricted to `mainchain`, which is this
search's chemistry; it is wired into the AF2 reward and not into this proposer. Applying
it needs the peptide chain's index range wrapped and an in-process MPNN call, because the
CLI cannot pass an offset matrix.

Not blocking — the arm wins as it stands — but it means the measured −7.82 is a floor for
this method rather than its ceiling.
