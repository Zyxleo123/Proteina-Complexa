# Arm D — an LLM as the proposal mechanism

Replacing the random substitution draws in the cyclic-peptide search with
`meta-llama/Llama-3.1-8B-Instruct`, prompted with PLIP/Biopython descriptions of the
complexes rather than raw sequence or coordinates.

Started 2026-10-24; results through 2026-10-26. **Everything here was verified by
execution against the folds on disk.** §7 is the verdict on dG and validity, §8–§10 are
the measured follow-ups, §11 is the measurement caveats that change how they read, and
§12 is what remains open. §13 (added 2026-10-26) quantifies §10's area-for-polars trade
and tests the one-field change that reverses it, which resolves §12's fourth item. §14
(added 2026-10-27) is a data-integrity finding about runs that straddle the 2026-10-22
gate recalibration — it leaves §1-§13 intact and says why. Bare §-references point within
this file; references to the scale-up doc name it explicitly.

---

## 1. What changes, and what must not

The goal is unchanged: find the head-to-tail cyclic sequence, within `floor(L/2)`
substitutions of the original linear peptide, whose predicted complex scores best — and
read the answer off the **cross-fitted** column, because
`README_CPSEQ_PEPBENCH_SCALEUP.md` §10 showed the whole −4.66 REU best-of-8 advantage
was selection bias and vanished to −0.06 REU under cross-fitting.

Exactly one thing is different: where the candidate sequences come from.

| | arm A (control) | arm D (this) |
|---|---|---|
| proposer | `rng.choice` over the alphabet | Llama-3.1-8B-Instruct |
| sees | nothing | interface tables + scored history |
| adaptive | no | yes, in rounds |
| budget | `floor(L/2)`, vs the original | identical |
| alphabet | identical | identical |
| fold depth | identical | identical |
| geometry gate | identical, frozen | identical, frozen |
| evaluator | `evaluate.py` | identical |
| report | `cpseq_ladder_report.py` | identical, `--arm llm_search` |

Everything in the lower half of that table is shared code on a shared path, which is the
only reason a difference between the arms can be attributed to the proposer.

### New files

| file | what it is |
|---|---|
| `cpseq/interactions.py` | PLIP + Biopython interface summary, per peptide position |
| `cpseq/llm_propose.py` | prompt construction, parsing, legality screening, the model |
| `script_utils/cpseq_llm_propose.py` | the proposal stage (one round, all complexes) |
| `script_utils/test_cpseq_llm_propose.py` | 36 checks, no GPU and no model (all passing) |
| `scripts/cpseq_llm_propose.sbatch` | the propose job |

Modified: `cpseq/search.py` (+`ARM_LLM`, `+KIND_LLM`), `script_utils/cpseq_search.py`
(`ladder --llm-round`), `script_utils/cpseq_ladder_report.py` (`--arm`).

---

## 2. What the model is shown

Not the receptor sequence and not coordinates. A per-position table of what the peptide
is *doing*, which is both far smaller and far more actionable. For `lnr_2v8x`
(`RIIYDRKFLMECRN`, a 14-mer on a 181-residue receptor), the whole prompt is **~4,000
tokens**:

```
  pos aa  ss  burial  dmin  ctc  ovl  interactions / nearby receptor
    1 E   ?   0.00   6.0    0    0  no receptor contact
    2 I   H   0.30   3.7    9    0  hydrophobic:A:His2
    4 Y   E   0.80   3.1   13    0  hydrophobic:A:Val38, hbond:A:His6
    7 K   H   0.47   2.7    6    0  hbond:A:Glu109, saltbridge:A:Asp112
    9 L   H   1.00   3.5   14    0  hydrophobic:A:Val38, Leu104, Ile107 (+1)
   11 E   H   0.00   6.1    0    0  no receptor contact
```

A proposer can act on this directly: position 9 is a fully buried hydrophobic anchor to
keep, position 7 makes a salt bridge to keep, and positions 1, 5, 8, 11, 12, 14 touch
nothing and are where the budget should go.

The prompt also carries the original sequence, the budget, the cyclic topology, the
conditioning pocket, every already-scored candidate with its verdict, full interface
tables for the best two and one *rejected* candidate, and an explicit do-not-repeat list.

### The `ovl` column is the point

`README_CPSEQ_PEPBENCH_SCALEUP.md` §11 measured that **peptide–receptor overlap is the
sole cause of 3,457 of 11,953 rejected folds** — removing it would raise the pass rate by
0.289, more than every other cause combined. It is the one failure a sequence proposer
can actually do something about, and a random proposer cannot see it at all.

### Raw structures, not relaxed ones

The gate judges the **raw** predicted complex: `evaluate.py` calls `validity_report` on
the Boltz output and uses the refined copy for energies only. Relax is allowed to move
atoms to relieve strain, so it repairs the very defect the proposer needs to see.
Measured on `lnr_6seo`/`VPLRARNLPPFFFKEP`:

| structure | `overlaps` | position 6 min distance |
|---|---|---|
| raw (what the gate judged) | **4** | **0.79 Å** — the rejection reason |
| refined (post-relax) | 0 | 2.7 Å |

Summarising the refined complex would have told the model a rejected candidate's
interface was clean. `summarize_complex` therefore defaults to `structure_stage="raw"`,
and its overlap count is in **atom pairs** so that it equals the `n_inter_clashes` the
same candidate carries in the journal (3 distinct atoms vs the gate's 4 pairs, before
this was fixed).

Summaries read `.pdb.gz` transparently, because the ladder runs at `rescorable`
retention, which gzips the raw complex — 12,008 of them in the existing run. Each
summary costs **1.2–2.6 s** and is cached under `<out-dir>/interfaces/<parent>/<input_hash>.json`,
so later rounds re-read rather than recompute, and the summary outlives the structure
that `retention.py` will eventually delete.

---

## 3. The four rules that keep arm D comparable

Each exists because the obvious implementation produces a comparison that looks like a
result and is not.

1. **A model's output is a suggestion, never a candidate.** Length, alphabet, budget
   against the *original*, and novelty are all re-checked, and each rejection is recorded
   by reason in `records.jsonl`. A model that mostly emits illegal strings and one that
   mostly emits good ones must not look alike in the log.

2. **A short round is reported, never padded.** If the model cannot supply the requested
   number of legal novel sequences within its attempt limit, the round returns what it
   has and says so. Backfilling with random draws would make arm D a silent hybrid of
   itself and the control — and the worse it performed, the more of the control it would
   contain.

3. **The proposer never sees the measurement folds.** `--feedback-folds select` (the
   default) shows only folds `0..depth/2`, the half the report is allowed to select on.
   Without this, an adaptive proposer chooses what to propose using the folds its
   proposals are later scored on, and the held-out column stops being held out.
   `--feedback-folds all` exists only to measure the size of that leak.

4. **Re-proposing an evaluated sequence is refused, not cached.** The evaluator caches on
   `input_hash`, so a duplicate would silently reuse folds already on disk *including
   their measurement half*. A model re-emitting a sequence it saw score well would be
   laundering a measured number into a "new" proposal. Blocked across all arms.

Two further honesty details: the history shows each sequence's **median** over folds, not
its best (seed spread is 4.3 REU median and up to 22 REU, so a per-sequence minimum
mostly records which fold got lucky); and technically-failed folds are dropped entirely,
because telling the model a sequence "failed" when a GPU died would teach it to avoid a
perfectly good one.

---

## 4. The bug that would have silently emptied the arm

First real run on `lnr_2v8x`: **0 of 24 proposals accepted across 4 attempts.** Every one
rejected as `excluded residue(s): C`.

The original sequence is `RIIYDRKFLMEC RN` — it *has* a cysteine, at position 12. The
alphabet excludes C, and the validator was testing `set(sequence) <= alphabet`, so every
proposal that correctly **kept** the native residue was rejected.

This was wrong, and the project had already settled the point: `search._draw_residue`
draws only *replacement* residues from the alphabet and never touches an unedited
position, so the random control preserves a native cysteine in every one of its draws.
The constant is even named `DEFAULT_EXCLUDED_FROM_RANDOM`, and
`test_cpseq_search.py::test_native_cysteine_is_admissible_where_the_original_has_it`
asserts exactly this.

The rule is now: **the alphabet constrains what a substitution introduces, not what the
original contains.** The prompt was also contradicting itself — it forbade C to a model
reading a sequence containing C — and now names the position and says keeping it costs no
budget.

Worth recording because of how it would have failed: not with an error, but as an arm
that proposed nothing for every cysteine-bearing complex, in a log that read as a model
unable to follow instructions. A regression test covers both directions.

### What the model actually does, once it is allowed to

Unprompted, on the reference interface, it located the overlap and the empty positions:

> *"The residue at position 4 (Y) is particularly notable, as it is in close proximity to
> the receptor residue A:Pro7, resulting in an overlap of 2.19 Å. Additionally, several
> residues are not making good contacts with the receptor, such as positions 5 (D), 7 (K),
> 8 (F), 11 (E), and 12 (C)."*

Two cautions from the same output. It emitted two 15-mers among nine proposals (caught as
malformed; the prompt now insists on counting letters), and its stated *reasons* do not
always match the edit it made — `RIIYSRKFLMECRN` is a change at position 5 described as
fixing position 4. The reason string is recorded as provenance and is **never** trusted
or parsed into an edit.

---

## 5. Running it

Arm D alternates with the fold stage; rounds are what make it adaptive.

```bash
# Round 1: propose (GPU, ~17 GiB, one model load serves every complex)
sbatch scripts/cpseq_llm_propose.sbatch OUT_DIR=$ZFS/cpseq/llm_ladder ROUND=1

# Round 1: fold, exactly as the random arm is folded
sbatch --array=0-19%20 scripts/cpseq_ladder.sbatch \
    OUT_DIR=$ZFS/cpseq/llm_ladder SHARD_COUNT=20 LLM_ROUND=1

# Round 2 sees round 1's selection-half folds, and so on
sbatch scripts/cpseq_llm_propose.sbatch OUT_DIR=$ZFS/cpseq/llm_ladder ROUND=2
sbatch --array=0-19%20 scripts/cpseq_ladder.sbatch \
    OUT_DIR=$ZFS/cpseq/llm_ladder SHARD_COUNT=20 LLM_ROUND=2

# Report each arm against the SAME shared reference folds
python script_utils/cpseq_ladder_report.py --out-dir $ZFS/cpseq/llm_ladder --arm llm_search
python script_utils/cpseq_ladder_report.py --out-dir $ZFS/cpseq/llm_ladder --arm random_control
```

**Both arms belong in one run directory.** The reference is written under its own arm and
keyed by `input_hash`, so the second arm's reference folds are free *and identical* — the
arms are compared against the same measurements of the unchanged peptide rather than
against two independent foldings of it. `--arm` writes
`ladder_report_llm_search.json` beside `ladder_report.json`.

Rounds concatenate into one prefix-stable draw stream: round 1's batch is draws
`0..k-1`, round 2's is `k..2k-1`. That is what lets the ladder's rungs — which are
prefixes — mean the same thing in a two-round and a five-round run.

### Why the LLM is a separate job

Llama-3.1-8B in bf16 holds **16.5 GiB** of GPU memory (measured: an A6000 at 20 GiB of
other work reached 36.9 of 47.4 GiB with the model loaded). Boltz-2 needs the GPU too, so
a resident LLM would compete with the thing it exists to serve. Separate jobs mean one
load (**11.9 s warm, ~95 s cold off the NAS**) serves every complex in a round and the
GPU is free before folding starts. This is the pattern arm C already uses.

The GPU type is pinned to A6000: `preempt` includes 11 GiB cards on which the model
cannot load at all, and a CUDA allocation failure can kill the process with no traceback.

**Do not run the propose stage inside a shared interactive allocation.** Measured
2026-10-24: a VS Code session's cgroup sat at 48.5 of 51.5 GiB with other work in it, and
the load died with `std::bad_alloc`.

### Dependencies added

`plip==3.0.1` and `accelerate==1.15.0`. PLIP pulled `openbabel` 3.1.0 → 3.2.1; nothing in
`cpseq/` uses openbabel, and the only consumer is
`datasets/atomworks_ligand_transforms.py`, a separate pipeline that was not exercised.
`HF_HUB_OFFLINE=1` is set in the job: the weights are a local directory, the hub repo is
gated, and a 401 would otherwise read as a missing file.

---

## 6. Round 1, measured

Job `10920855`, 3 complexes of the probe cohort, batch 8, `--feedback-folds select`
(folds 0–4 of 10), on an L40S. **24 of 24 proposals accepted.** Model load 165.8 s cold;
generation 110.2 s for all three complexes.

| complex | accepted | attempts | rejected | malformed | substitution counts |
|---|---|---|---|---|---|
| `lnr_2v8x` (L=14, budget 7) | 8/8 | 4 | 5 | 8 | 1,1,1,1,1,1,2,2 |
| `lnr_1elr` (L=5, budget 2) | 8/8 | 3 | 9 | 4 | 1,1,2,2,2,2,2,2 |
| `lnr_6seo` (L=16, budget 8) | 8/8 | 3 | 1 | 4 | 1,1,1,1,1,1,2,2 |

**Yield is about two usable proposals per sampling attempt.** Filling 8 slots took 3–4
attempts, so a round costs ~35–60 s of generation per complex plus the one-off load.

Two guards fired for real, which is the point of recording them:

- **Dropped letters are a real yield cost.** Every one of `lnr_6seo`'s 4 malformed
  proposals was `length 15 != 16`, and **all 16** malformed proposals across the three
  complexes were an off-by-one length — every single one, no other malformation
  occurred. The prompt already tells the model to copy the
  original and count its letters; that was in place for this run and did not fully fix
  it. The clean fix is to have longer peptides answer in **substitution notation**
  (`D5A,K7R`), which is immune to miscounting — the parser already accepts it as a
  fallback. Not done yet; nothing is ever repaired into a sequence the model did not
  write.
- **A whole generation was discarded over punctuation, and that is now fixed.**
  `lnr_6seo`'s first attempt parsed **0 of 10** proposals — not from length errors, but
  because the model answered as a numbered list, `9. Proposal: <SEQ> | <reason>`, and
  "PROPOSAL" does not contain "PROPOSE". 13 s of generation and one of that complex's
  three attempts were thrown away for a difference in label. `_PROPOSE_RE` now accepts
  an optional list index and the `PROPOSAL`/`PROPOSED` variants; re-parsing that exact
  saved completion recovers **6 legal proposals where it previously found none**. This
  is parsing, not repair: the label is explicit and the sequence written out in full. A
  bare sequence sitting in prose is still never harvested, because the reasoning text
  names sequences it is arguing *against* — there is a regression test for both halves.
- **The duplicate guard caught a real duplicate.** `VPLRARNLPPSFFTWP` was rejected as
  `already_evaluated`. Had it been accepted, the evaluator would have served it from
  cache on `input_hash` — including its measurement-half folds — and a sequence the
  model had seen score well would have been laundered into a "new" proposal (§3 rule 4).

Provenance is complete per round: `records.jsonl` carries the model path, dtype,
temperature, top-p, per-attempt seeds and timings, every rejection with its reason, and
`max_fold_rank_shown: 5` confirming the cross-fit guard was active. The exact prompt and
raw completion of all 10 attempts are saved under `prompts/`. Interface summaries cached
to 592 KB for 12 complexes.

### Substitution counts are not a confound

The model proposes almost only 1–2 substitutions. That initially looked like a problem,
since `search.match_counts` exists precisely so a difference between arms cannot be an
artefact of one mutating harder. It is not one here — the random stream is *also*
dominated by single substitutions:

| complex | budget | random arm, all draws | arm D, 8 proposals |
|---|---|---|---|
| `lnr_2v8x` | 7 | median 1, range 1–6 (37/64 singles) | median 1, range 1–2 |
| `lnr_1elr` | 2 | median 1, range 1–2 | median 2, range 1–2 |
| `lnr_6seo` | 8 | median 1, range 1–4 (30/48 singles) | median 1, range 1–2 |

Same median on both larger complexes. The LLM's tail is shorter (stops at 2 where random
reaches 5–6), which is a property of the proposer under comparison rather than a
mismatch in how hard the two arms are allowed to mutate.

### Diversity is the real risk, and it is now recorded

`round_diversity` is computed for every round and printed in the log:

| complex | arm | median pairwise Hamming | distinct edited positions |
|---|---|---|---|
| `lnr_2v8x` | `llm_search` | 2.0 | **4 of 14** |
| `lnr_2v8x` | `random_control` | 4.0 | **9 of 14** |
| `lnr_6seo` | `llm_search` | 2.0 | **5 of 16** |
| `lnr_6seo` | `random_control` | 2.0 | **10 of 16** |
| `lnr_1elr` | `llm_search` | 3.0 | 4 of 5 |
| `lnr_1elr` | `random_control` | 2.0 | 5 of 5 |

**The model concentrates its budget on roughly a third of the positions.** Both readings
are live: it is spending the budget where the interface says it matters, which is the
point — or it has converged prematurely, in which case best-of-B over this arm samples
less independent variation than best-of-B over the control, and if its chosen positions
are wrong it has no mechanism for discovering that. The arm's ladder should therefore be
read for its *shape*, not just its endpoint: premature convergence looks like a ladder
that flattens earlier than the control's.

### The model reads the interface; its stated reasons are unreliable

It does use the history adaptively — one proposal is explicitly *"Combining the good
interactions from RIKYDRKELMETRN and RIIYDR…"* — and it cites real receptor residues
(`Val38`, `His198`, `Glu187`, `Gln46`) drawn from the PLIP tables.

But the *rationales* frequently do not match the edit, often by one position:
`RIWYDRKFLMECRN` is described as "replacing position 1 with W" when the edit is at
position 3; `RIIYDRHFLMECRN` as "position 8" for an edit at 7; and one reads "Replace Glu
at position 3 with Glu". The prompt is consistently 1-based, so this is model sloppiness,
not an indexing mismatch.

This costs nothing because the reason string is **never parsed into an edit** — the
validated sequence is what gets folded, and the note is provenance only. It does mean
these notes must not later be mined as a record of design intent.

---

## 7. The answer: does the LLM proposer beat random draws?

**Not established.** 3,150 arm-D folds over 20 complexes at depth 10, zero technical
failures, compared head to head against the random arm's existing 11,854 folds on the
**same complexes and the same shared reference folds** (`cpseq_arm_compare.py`).

```
          |  validity: reliable cand.  |    dG cross-fitted (D vs A)    |     dG best-of-B
  B  cplx |   armA   armD  passA  passD |    dREU     w/l  sign p   n |    dREU     w/l     p
  1    20 |  0.350  0.350  0.335  0.355 |   -1.78     6/4   0.754  10 |   +0.70     1/2  1.000
  2    20 |  0.700  0.450  0.335  0.375 |   -2.91    10/4   0.180  14 |   -0.26     5/4  1.000
  4    20 |  0.700  0.800  0.325  0.386 |   -0.90    12/5   0.143  17 |   -1.61    10/2  0.039
  8    20 |  0.800  0.850  0.367  0.383 |   -3.30    11/5   0.210  16 |   -1.01    10/5  0.302
 12    20 |  0.850  0.950  0.361  0.400 |   -0.72    12/5   0.143  17 |   -2.28    12/5  0.143
 16    17 |  0.824  0.941  0.360  0.379 |   -1.03    10/3   0.092  13 |   -1.94    11/3  0.057
```

**The cross-fitted delta favours the LLM at all six rungs and never once reverses**
(−1.78, −2.91, −0.90, −3.30, −0.72, −1.03 REU), with wins exceeding losses at every rung.
The best-powered single test is **B=16: −1.03 REU, 10 wins / 3 losses, p = 0.092**. Nothing
reaches significance.

Those six rungs must **not** be pooled into 61 wins / 26 losses. They are nested prefixes
of the same draws on the same complexes, so pooling them is the pseudo-replication that
turned this project's pilot p = 2.6e-6 into 0.031 once the complex was the unit.

### The marginal rates in that table are misleading, and the paired tests are what count

Coverage looks like a +10 to +12 point win for arm D at B=12 and B=16. It is not:

| B | reliable, A | reliable, D | D-only | A-only | exact p | pass-rate median Δ | w/l | p |
|---|---|---|---|---|---|---|---|---|
| 1 | 7 | 7 | 4 | 4 | 1.000 | +0.1000 | 11/8 | 0.648 |
| 2 | 14 | 9 | 0 | **5** | 0.062 | +0.0750 | 11/7 | 0.481 |
| 4 | 14 | 16 | 4 | 2 | 0.688 | +0.0500 | 13/5 | 0.096 |
| 8 | 16 | 17 | 2 | 1 | 1.000 | +0.0063 | 10/9 | 1.000 |
| 12 | 17 | 19 | 2 | 0 | 0.500 | +0.0292 | 15/5 | **0.041** |
| 16 | 14 | 16 | 2 | 0 | 0.500 | +0.0250 | 10/5 | 0.302 |

**Coverage is not significant at any rung, and at B=2 it runs the other way** (5 complexes
where only the random arm found a reliable candidate, p = 0.062). The apparent +10 points
at B=12 is **two discordant complexes**.

Per-fold pass rate is positive at all six rungs and reaches a nominal **p = 0.041 at
B=12** — but its immediate neighbours are p = 1.000 (B=8) and p = 0.302 (B=16). Adjacent
nested rungs disagreeing that violently is the signature of noise, and it is one nominal
hit among six correlated tests. It is not a result.

### The selection-bias trap fired twice more

At **B=4**, best-of-B reads −1.61 REU, 10/2, **p = 0.039** while its cross-fitted twin is
−0.90, 12/5, p = 0.143. At **B=16**, best-of-B is p = 0.057 against the cross-fit's 0.092.
Reporting the best-of-B column alone would have produced a false positive here, exactly as
the scale-up doc's §10 found for the random arm. Note also that at B=8 the two columns *invert* (cross-fit
−3.30 vs best-of-B −1.01), so the bias is not reliably one-directional.

### Yield became a real constraint at B=16

Round 2 accepted **155/160**: `lnr_6efk` and `pepbench_A_B_pdb4bvg` got 6 of 8 and
`pepbench_C_A_pdb2jz3` got 7, after 4 attempts each. The shortfall was **not padded with
random draws** (§3 rule 2), so those three complexes top out at 14–15 draws and are
excluded from B=16 — which is why B=16 rests on 17 complexes and 13 energy pairs. An arm
that cannot fill its budget reaches a lower rung than its control, and that is a cost of
the method, not an accident of this run.

### What would actually settle it

Not more draws. The cross-fitted deltas do not trend with B (they scatter around −2 REU
with no slope), which matches the scale-up doc's §11 finding that selection saturates near B≈8. At depth 10
each side of the cross-fit rests on 5 folds against a 4.3 REU median seed spread, so the
measurement is noise-limited, not candidate-limited. **Depth 20 at B=8** is the experiment
that would resolve a ~2 REU effect. It costs ~50 GPU-h rather than ~12, because extra
draws are cheap (the random arm is already folded to B=64) while extra depth must be paid
on both arms and the reference.

---

## 8. Where the reference sits: the (validity, dG) Pareto front

"Does the arm beat the reference" turned out to be the wrong question for most complexes,
because **the reference is usually already on the front.** Cross-fitted — a candidate
counts as dominating only if it beats the reference on *both* axes on folds 5–9, having
been identified on folds 0–4:

| B | random dominates ref | llm dominates ref |
|---|---|---|
| 2 | 0.05 | **0.20** |
| 4 | 0.15 | **0.25** |
| 8 | 0.10 | **0.30** |
| 12 | 0.10 | **0.30** |
| 16 | 0.20 | 0.24 |
| 24 | 0.35 | — |
| 32 | 0.30 | — |

**In ~70–76 % of complexes nothing in the draws beats the reference on both axes at once.**
Where it is dominated, the LLM arm does it 2–3× more often than random at matched budget
(0.30 vs 0.10 at B=8); random needs B=24–32 to reach the rate the LLM hits at B=8.

The two one-sided gaps, which is where the arms separate most:

**dG available at no validity cost** (negative = gain):

| B | random | llm |
|---|---|---|
| 4 | +0.46 (4/7) | −5.91 (6/3) |
| 8 | **+0.35** (4/8) | **−7.88** (7/2, p=0.180) |
| 12 | +1.84 (3/8) | −6.09 (9/3, p=0.146) |
| 32 | −3.16 (10/3, p=0.092) | — |

At B=8 random offers **nothing** — matching the reference's validity costs 0.35 REU — while
the LLM offers **−7.88 REU for free.** Largest LLM-favouring effect measured, still p=0.180.

**Validity available at no dG cost:** ~+0.20 for both arms (random 8/4 p=0.388, llm 10/3
p=0.092 at B=8). No separation.

> **Correction to an earlier draft of this file.** An interim note claimed the most-valid
> draw beats the reference "19/0, p<0.001". True, but that was neither cross-fitted nor
> constrained to hold dG. Once both are imposed the available validity gain collapses to
> +0.20 at 8/4, p=0.388. The headline was selection bias plus an unconstrained trade.

**The practical consequence:** the 24–35 % of complexes where the reference *is* dominated
is where design effort pays, and they are identifiable from folds already on disk.
Restricting a proposer comparison to that subset buys far more power than more draws.

---

## 9. What causes invalidity, and what search can and cannot fix

**62.2 % of folds are invalid** (9,458 of 15,204). Causes, with the overlapping categories
made disjoint per fold:

| | share of failures |
|---|---|
| clash only | **67.9 %** |
| both clash and closure | 19.0 % |
| closure only | 13.1 % |
| → any clash | **86.9 %** |
| → any closure | 32.1 % |

By sole cause: peptide–receptor overlap 45.6 %, intra-peptide overlap 13.4 %, closure
torsion 10.9 %, closure angles 0.6 %, closing bond length 0.0 %. Counterfactuals on the
0.378 overall pass rate: **eliminate clash → 0.918** (+0.541); eliminate cyclization →
0.578 (+0.200). This reproduces §11 of the scale-up doc on a different cohort.

### Cyclization is not sequence-limited, so search cannot fix it

Reference closure pass rate over the 20 complexes: **mean 0.750, median 0.800; 7 of 20
close in every fold and none fail in all folds.** The ring already closes three-quarters of
the time before any search runs.

Cross-fitted, candidate chosen on folds 0–4, closure rate measured on folds 5–9:

| selection rule | arm | B=8 | B=16 | mean cand / ref |
|---|---|---|---|---|
| by energy | random | +0.000, 9/4, p=0.267 | +0.000, 6/6, p=1.000 | 0.884 / 0.779 |
| by energy | llm | +0.000, 6/7, p=1.000 | +0.000, 7/7, p=1.000 | 0.790 / 0.750 |
| **by closure** | random | +0.000, 7/4, p=0.549 | +0.000, 7/4, p=0.549 | 0.850 / 0.750 |
| **by closure** | llm | +0.000, 9/4, p=0.267 | +0.000, 7/4, p=0.549 | **0.890** / 0.750 |

Mean closure rate rises to 0.85–0.89, but **the paired median is exactly +0.000 with 6–9
ties out of 20** and nothing reaches p<0.05 — *even when selecting explicitly on closure.*
The modal complex closes fine on both sides, so there is nothing to win.

**Closure is predictor-limited, not sequence-limited.** Which residues are chosen barely
moves whether Boltz-2 emits a closeable ring; that is a property of the backbone it
generates. The ~13 % of failures that are closure-only are not addressable by either arm.
The levers aimed at closure sit outside search: the already-enabled
`v2-closure-constrained` relax constraint, and `--use_potentials` (see §11).

---

## 10. What selecting on dG actually buys

The cross-fitted dG winner at B=8, selected on folds 0–4, every metric below read off
folds 5–9 for both it and the reference. These are out-of-sample readings of a sequence
picked for its energy: what else you get, or give up.

**random_control** (18 complexes)

| metric | cand | ref | median Δ | w/l | p |
|---|---|---|---|---|---|
| pair ipTM | 0.911 | 0.881 | +0.012 | 8/4 | 0.388 |
| interface pLDDT | 0.930 | 0.913 | +0.006 | 9/3 | 0.146 |
| buried unsat polars, refined ↓ | 4.50 | 4.50 | +1.00 | 4/7 | 0.549 |
| interface H-bonds | 4.00 | 3.00 | +1.25 | 8/4 | 0.388 |
| shape complementarity | 0.665 | 0.710 | −0.001 | 6/6 | 1.000 |
| buried interface area Å² | 1040 | 988 | +32 | 8/4 | 0.388 |

**llm_search** (17 complexes)

| metric | cand | ref | median Δ | w/l | p |
|---|---|---|---|---|---|
| pair ipTM | 0.921 | 0.883 | +0.018 | **9/2** | **0.065** |
| interface pLDDT | 0.906 | 0.922 | −0.000 | 5/6 | 1.000 |
| buried unsat polars, refined ↓ | 4.00 | 3.00 | +1.00 | 3/7 | 0.344 |
| buried unsat polars, raw ↓ | 9.00 | 9.00 | **+2.00** | 1/6 | 0.125 |
| interface H-bonds | 3.50 | 3.00 | +0.000 | 5/3 | 0.727 |
| shape complementarity | 0.723 | 0.701 | +0.020 | 6/5 | 1.000 |
| **buried interface area Å²** | **1075** | **945** | **+119** | **10/1** | **0.012** |

**dG selection buys buried area and pays for it in unsatisfied polars, in both arms.** The
LLM winner buries +119 Å² more than the reference (10/1, p=0.012 — the strongest single
result in this study) with ipTM up +0.018 (9/2, p=0.065), while buried unsat polars move
the **wrong way in both arms** (+1.0 refined for random at 4/7; +1.0 refined and +2.0 raw
for the LLM at 1 win / 6 losses).

That is a mechanism, not noise: `dG_separated` rewards burial, and burying more surface
pulls polar atoms out of solvent without necessarily giving them partners. Shape
complementarity is flat in both arms (p=1.000), consistent with the extra area being
larger rather than better packed. **The sequence selected on dG is more buried, more
confidently placed, and worse hydrated at the interface than the reference.**

Direction is consistent across both arms, which makes it more credible than the arm-vs-arm
dG comparison; 17–18 complexes still means only the area result is individually convincing.

---

## 11. Measurement details that change how the numbers read

Each of these was checked against the data, not the code's capability.

- **dG excludes invalid folds, and there is only one filter, not two.** A sequence's dG is
  the median of `best_valid_dG_separated` over its *passing* folds. **Every fold in this
  run produced exactly one structure** (`n_samples = 1` for all 15,204 folds;
  `diffusion_samples` defaults to 4 in the adapter but the driver's `--samples` and the
  sbatch both set 1), so the within-fold "best valid sample" is a `min()` over a list of
  length 0 or 1. A sequence passing 2/10 folds reports a dG resting on two
  single-structure measurements with nothing averaging the diffusion noise inside each.
  5,746 of 15,204 folds passed (37.8 %).
- **Validity is measured before refinement; dG after.** `validity_report` runs on the raw
  Boltz output and only the energies come from the relaxed copy. On
  `lnr_6seo`/`VPLRARNLPPFFFKEP` the raw complex overlaps His198 at **0.79 Å** — the
  rejection reason — and after relax the same position sits at 2.7 Å with no overlap. A
  sequence is judged on geometry it is never credited for fixing.
- **`--use_potentials` was OFF for both arms, by default rather than by decision.**
  Boltz-2's inference-time steering potentials (sterics, bond geometry, chirality) are the
  feature aimed squarely at the clash failures of §9. The key is **absent** from both
  cohorts (they predate the field) and resolves through
  `bool(cohort.get("use_potentials", False))`. Behaviour is correct and identical across
  arms — which is what the comparison needs — but the provenance is weaker than it looks.
- **Interface PAE is not recoverable from this run.** `--write_full_pae` *is* passed, so
  the matrices were written, but `rescorable` retention deletes the pae/pde npz (29 % of
  each prediction dir). **0 of 3,150 survive.** Interface PAE needs a re-prediction at
  `full` retention. Pair ipTM is unaffected: all 20 complexes have exactly one receptor
  chain, so Boltz's global `iptm` *is* the peptide↔receptor pair value, and it is stored
  per fold in the journal.
- **Buried unsat polars are populated but their exact definition is unverified.**
  `refined_delta_unsatHbonds` has values on all 15,204 scored rows (refined 0–9, raw 4–13,
  i.e. interface-scale rather than whole-protein), so something is computing them — but
  `set_compute_interface_delta_hbond_unsat(True)` is **not called anywhere in
  `scorer.py`**. Confirm these are the interface *delta* before leaning on §10's unsat rows.
- **The substitution budget is enforced, verified empirically.** Across all **1,567
  distinct (arm, sequence) pairs folded: 0 budget violations, 0 length changes.** The
  budget is `floor(L/2)`, which implies ≥50 % identity — exactly 50 % only for even
  lengths (L=5 → cap 2 → 60 % identity; L=15 → cap 7 → 53.3 %). `within_budget` is
  measured against the **original, never an ancestor**, enforced at three independent
  sites (the random proposers, `Proposal.__post_init__` as a crash-not-corrupt backstop,
  and `screen_candidates` for LLM output) plus a fourth guard when the fold stage loads a
  proposal file. The "against the original" rule is load-bearing for arm D specifically:
  round 2 edits from round 1's candidates, and measuring against the ancestor would let
  two rounds of "two edits each" walk arbitrarily far while every step looked legal.
  Observed maxima stay under the cap (LLM max 6 against a cap of 7), and both arms are
  dominated by single substitutions — but the random stream reaches the ceiling on 9 of 20
  complexes where the LLM uses a third of it on the long peptides.

---

## 12. What is not yet known

The verdict on dG and validity is §7; the Pareto position is §8. What remains open:

Ranked by expected value per GPU-hour, which has changed now that §9 is measured.

1. **Turn on `--use_potentials`.** Clash is 86.9 % of all failures and eliminating it
   alone would take the pass rate from 0.378 to 0.918 — larger than any proposer effect
   and larger than depth 20. Potentials target exactly that, on exactly the raw structure
   the gate judges, and they were off here only by default. Needs its own `--out-dir`
   (it changes the structures, and the journal key carries no inference settings).
2. **Restrict the proposer comparison to the complexes where the reference is
   dominated** (§8: 24–35 %). Elsewhere there is nothing to win, so those complexes are
   diluting every sign test in §7. Identifiable from folds already on disk, costs nothing.
3. **Depth 20 at B=8** if a dG answer is still wanted. Cross-fitting leaves 5 folds a side
   against a 4.3 REU median seed spread, so §7 is noise-limited, not candidate-limited —
   the deltas scatter around −2 REU with no slope in B. ~50 GPU-h, because depth is paid
   on both arms *and* the reference, whereas draws were cheap (arm A was already at B=64).
4. **Fix the selection rule before buying more folds.** §10 shows dG selection trading
   unsatisfied polars for buried area in both arms, and the earlier validity work showed a
   much more valid sequence usually sits in the same draws. A rule that ranks on pass
   count and breaks ties on energy is testable on existing folds for free.
   — *Partly answered in §13:* the trade is quantified (buried area vs the objective,
   rho −0.464 within parent) and dividing dG by buried area reverses it, moving every
   interface-quality metric the right way on held-out folds. Still open is the
   pass-count-first rule, and whether a search *run* under the normalised objective beats
   one merely re-ranked by it.

Still genuinely unknown:

- **Whether adaptivity survives the cross-fit guard.** With `--feedback-folds select` at
  depth 10 the model reasons from 5 folds per candidate, where the seed spread is 4.3 REU
  median. It may be reading noise. Running both `select` and `all` sizes that leak.
- **Whether round 2 added anything over round 1.** B=16 (−1.03, 10/3, p=0.092) is not
  distinguishable from B=8 (−3.30, 11/5, p=0.210), so there is no evidence either way
  that later rounds propose better rather than merely more.
- **Whether the narrow search costs anything.** §6 measured the LLM touching 4–5 of 14–16
  positions against the control's 9–10, and §11 shows it using a third of the budget where
  random reaches the ceiling. It matched or beat random anyway, so this may be efficiency
  rather than premature convergence — unresolved.
- **Interface PAE**, which cannot be recovered from this run at all (§11).

Resolved since the first draft, recorded so they are not re-asked: **yield** became a real
constraint at B=16 (155/160, three complexes capped at 14–15 draws, §7); **cyclization**
is not improvable by search even when selected on directly (§9); and the **reference is
usually already on the Pareto front** (§8), which reframes most of the above.

---

## 13. The §10 trade, quantified — and the one-field fix that reverses it

§10 established the trade qualitatively: dG selection buys buried area and pays in
unsatisfied polars, in both arms. §12's fourth item called for fixing the selection rule
before buying more folds. This is that measurement, and it was prompted by a different
proposal — feeding the interface's unsatisfied atoms back to the model as text, so it
could repair them. **That proposal does not work, and the reason it does not work is also
the fix.**

### 13.1 The defect is mostly not substitutable

`cpseq/unsat.py` reimplements buried-unsat geometrically, per atom, because Rosetta's
`delta_unsatHbonds` arrives as one integer and a proposer cannot act on an integer. It
agrees with the Rosetta count at **median within-parent Spearman 0.557** (IQR 0.50–0.75,
all 20 parents, none negative) at burial ≤0.5 Å² with desolvation ≥1.0 Å². Pooled across
parents that reads 0.714, so roughly a fifth of the pooled figure was target identity;
the within-parent number is the honest one.

Burial needs **two** conditions, and getting this wrong is the module's main hazard. With
absolute enclosure alone, the only buried polar atoms on a helical peptide are backbone N
and O already satisfied by the peptide's own i,i+4 network — on `lnr_2v8x`/`RIIYDRKFLMECRN`
all 16 of them — so interface-caused unsat came out 0 against Rosetta's 5. With
desolvation alone it errs the other way: that fold's `ARG13:NH1` loses 26 Å² and keeps
24 Å², so it still reaches water and is no defect at all.

Across 480 folds (24 per parent, both arms, raw structures off disk), where the flagged
atoms sit:

| | per fold | share of all flagged |
|---|---|---|
| peptide side chain — a substitution can fix this | 0.26 | **11.0 %** |
| peptide backbone amide | 0.90 | 37.9 % |
| receptor side | 1.21 | 51.1 % |

103/480 folds (21 %) have even one fixable defect; 25/171 valid folds (15 %). **Half the
count is on the receptor, which the peptide's sequence cannot change.**

### 13.2 Under the current objective, repairing it lowers your score

Within parent, on valid folds, against `best_valid_dG_separated` (negative is better, so
positive rho means more unsat goes with worse energy):

| | vs dG (the objective) | vs dG per 1000 Å² |
|---|---|---|
| buried area | **−0.464** (15/17 parents neg.) | +0.255 (3/17 neg.) |
| Rosetta refined buried-unsat | +0.204 (5/17 neg.) | **+0.493** (1/17 neg.) |
| geometric *fixable* unsat | **−0.289** (7/9 neg.) | +0.106 (4/9 neg.) |

The objective is dominated by buried area, and the fixable defect carries the **wrong
sign**: candidates with more repairable unsatisfied atoms score *better*, because both
come from burying more surface. So a model shown these atoms and asked to repair them,
while still ranked on dG, is being asked to lower its own score. **No textual
representation of the interface fixes that, which is why the prompt was never the lever.**

Dividing by buried area — `refined_dG_separated_per_1000A2`, already on every row —
reverses both correlations. Relax also erases much of what the proposer would be shown:
raw against refined buried-unsat is only rho 0.336 within parent.

### 13.3 Re-ranking on dG per area, same folds, same cross-fit

`cpseq_interface_compare.py --energy-key` re-runs §10's comparison with the objective
swapped and everything else held fixed. Selected on folds 0–4, measured on 5–9, candidate
and reference on the same folds, all complete draws, no reliability bar.

| metric | dG (default) | dG per 1000 Å² |
|---|---|---|
| pair ipTM | +0.035, 11/2, **p=0.022** | +0.021, 8/4, p=0.388 |
| interface pLDDT | +0.013, 8/5, p=0.581 | +0.004, 6/6, p=1.000 |
| buried unsat, refined ↓ | +1.00, 6/7, p=1.000 | **−0.50**, 6/5, p=1.000 |
| buried unsat, raw ↓ | +3.00, 3/9, p=0.146 | **+1.00**, 5/7, p=0.774 |
| interface H-bonds | +0.50, 8/3, p=0.227 | **+0.75**, 8/2, p=0.109 |
| shape complementarity | −0.006, 6/7, p=1.000 | **+0.032**, 7/4, p=0.549 |
| buried area Å² | +106.5, 12/1, **p=0.003** | +41.0, 10/2, p=0.039 |
| dG per 1000 Å² ↓ | −0.297, 9/4, p=0.267 | **−0.318, 12/0, p=0.001** |

n = 13 and 12 paired complexes. These settings are not §10's (which used B=8 and a
reliability bar, 17 complexes), so the left column is a re-run rather than a restatement;
the two columns are apples-to-apples with each other, which is what the comparison needs.

**Every interface-quality metric moves the right way, and area-chasing halves.** Refined
unsat flips from +1.00 to −0.50, raw from +3.00 to +1.00, shape complementarity from
−0.006 to +0.032, H-bonds from +0.50 to +0.75 at 8/2. Buried area's advantage falls from
+106 Å² to +41. The selection target itself reaches **12/0, p=0.001 on folds the
selection never saw** — the strongest cross-fitted result in this study. ipTM's
significance is spent in exchange (p=0.022 → 0.388), which is consistent: ipTM rewards
contact, and contact is what stopped being optimised.

**What this does and does not establish.** No individual quality metric reaches
significance at n=12 — unsat is 6/5, shape complementarity 7/4. This is a consistent
directional shift plus one strong result on the selection metric, not a demonstrated
improvement in interface quality. Three further limits: it re-ranks existing folds rather
than running a search under the new objective, so the proposer never got to adapt to it;
a row's `refined_*` fields come from the one representative sample, still chosen by
un-normalised dG (`cpseq_search.py:571`); and even with the objective fixed, §13.1 caps
what a sequence proposal can repair at ~11 % of the defect.

**A run under the normalised objective needs its own control.** Changing what "best" means
breaks comparability with arm A as measured in §7 — the re-ranking above is free, a run is
not. Worth noting that the arm-vs-arm lean survives the swap: at B=16 the cross-fitted
comparison goes 10/4 (p=0.180) normalised against 10/3 (p=0.092) on the default, so the
LLM proposer's edge over random draws is not an artifact of an objective that rewarded
size.

### 13.4 Files

`cpseq/unsat.py` (detector and renderer), `script_utils/test_cpseq_unsat.py` (54 checks,
no GPU), `script_utils/cpseq_unsat_probe.py` (agreement, ceiling, and the objective
relation; `--from-jsonl` re-reports without touching a structure),
`script_utils/cpseq_interface_compare.py` (§10's table with a switchable objective).
`ENERGY_KEY` threaded through `cpseq_ladder_report.py` and `cpseq_arm_compare.py`,
verified a no-op on the default path against the existing `arm_compare.json`.

Nothing here touches the prompt or `interactions.py`. The detector is built and measured;
it is **not** wired into what the model is shown, because §13.2 says that would be the
wrong intervention until the objective changes.

---

## 14. The gate moved under the runs, and the journal kept both verdicts

Found 2026-10-27 while checking the ES pilot. **§13 and everything above it are
unaffected** — `llm_ladder` is uniformly `v3-cis-amide-muco-windows`, and the fixed reader
reproduces the published `arm_compare.json` byte for byte. What follows is about the runs
that straddle a gate change, and about a reader bug that made the straddling invisible.

### 14.1 What happened

`tolerances.py` was recalibrated on 2026-10-22: the trans-only 8.44° closure-torsion
window was replaced by a 30° cis-or-trans standard, after the old one was measured
admitting 27 of 150 native CPSea head-to-tail macrocycles while cis-or-trans at 30°
admits 117. That change is right and is argued in the module itself. Its side effect is
that **pass rates are not comparable across versions**: v3 ≈ 0.37–0.81, v4 ≈ 0.06–0.07,
v5 ≈ 0.98–1.00 on the same kind of folds.

Two mechanisms then combined to hide the mix:

1. `Journal.has` keys on `(sequence, prediction_seed, stage, arm)` and carries **no
   scoring version**, so re-scoring after the change *appends* a row rather than replacing
   one. `cpseq_rescore_folds.py` documents this as deliberate — history is not rewritten —
   and it is the right call.
2. `Sequence.by_rank` used `setdefault`, keeping the **first** row per rank. So the
   readers took the *retired* gate's verdict and discarded the re-score that had been paid
   for.

`cpseq_search.py report` already guarded against exactly this (one protocol per report,
`--scoring-version`, an explicit `all` to pool). The newer analysis scripts —
`cpseq_ladder_report.py`, `cpseq_arm_compare.py` — never inherited that guard.

### 14.2 What it cost

| run | rows | dup keys | disagree | gates |
|---|---|---|---|---|
| `llm_ladder` | 15,348 | 72 | **0** | v3 only |
| `es_pilot_pep` | 768 | 0 | 0 | v5 only |
| `es_pilot_lnr` | 816 | 24 | **24** | v4 + v5 |
| `pepbench_corpus_pot` | 45,854 | 20 | **20** | v3 + v4 + v5 |

In `es_pilot_lnr` the reference's four folds exist twice — **1/4 valid under v4, 4/4 under
v5** — while its LLM arm is entirely v5. First-wins therefore read the baseline under a
harsher, retired gate than the arm under test, and read the random control's draw 0 at a
**0.000** pass rate when the current gate says 1.000. Correcting it flips the B=1 sign
from "no measurable pair" to a **loss** for the LLM arm (+5.68 REU, 1/2).

`pepbench_corpus_pot` spans three gates, but the damage is narrower than the version
count suggests: **656 of 670 complexes are internally single-gate** and remain readable as
paired comparisons, which is the unit this project insists on. Only **14 complexes** mix
gates within themselves (12 span v3+v5, 2 span v3+v4), totalling **602 non-v5 folds**.
Corpus-level *pooled* rates across all 670 are still distorted, because a v4 complex at
0.07 and a v5 complex at 0.99 differ by gate and not by chemistry.

### 14.3 The fix

- `Sequence` keeps the **newest** row per rank, not the first, and records the
  `scoring_version`s it saw.
- `--scoring-version` on `cpseq_arm_compare.py` and `cpseq_interface_compare.py`, plus
  `warn_if_mixed`, which prints the census and says plainly when a directory spans gates.
- The default is **no filter plus a warning**, deliberately unlike
  `cpseq_search.py report`'s default of the current version. Defaulting to current here
  would silently empty every v3 report — `llm_ladder` has no v5 rows at all — and a
  comparison that quietly drops its own data is worse than one that refuses to be read.

Verified: 167 checks pass across the five `test_cpseq_*` files, and `llm_ladder`'s
cross-fitted columns, coverage and pass rates are identical to the published report.

### 14.4 Fold depth, and why it governs everything below

**Fold depth** is how many times each candidate sequence is folded, with a different
random seed each time. Boltz-2 is stochastic: the same sequence folded twice gives two
structures and two energies. On this cohort the seed-to-seed spread is a 4.3 REU median
and up to 22 REU -- larger than most effects being chased -- so one fold is a filter and
never a measurement.

Depth is then split in half. The first half **chooses** which candidate is best; the
second half **measures** the one that was chosen. That split is the cross-fit, and it
exists because choosing and measuring on the same folds reports the winner's luck as its
energy (§7: a -4.66 REU advantage became -0.06 under cross-fitting).

So depth 10 means 5 folds to choose on and 5 to measure on. **Depth 4 means 2 and 2.**
§12's third item already recorded depth 10 as noise-limited at 5 a side. The ES pilot ran
at 2.

### 14.5 The pilot's dG numbers are an artifact of depth 4. They are withdrawn.

They looked striking: cross-fitted deltas of -8 to -11.5 REU against the random control,
3 to 10x §7's -0.72 to -3.30 at depth 10, on both corpora. None of it survives.

**Nothing physical differs between the runs.** Median refined dG per fold, no selection:

| arm | llm_ladder v3 depth 10 | es_pilot v5 depth 4 |
|---|---|---|
| reference `lnr_5njx` | -24.57 | -25.68 |
| random control | -25.52 | -24.82 |
| LLM arm | -25.81 | -26.18 |
| reference `lnr_6seo` | -28.41 | -27.68 |
| random control | -24.60 | -23.97 |
| LLM arm | -25.43 | -25.77 |

All within ~1 REU, same substitution budgets. Three candidate explanations were tested
and all three are dead:

- **Not the validity filter.** For the control, folds the gate failed differ in refined dG
  from folds it passed by -0.40 and +0.28 REU. Validity and energy are uncorrelated there.
  (They are correlated for the LLM arm, +3.59 and +2.26 -- the arm that did not collapse.)
- **Not the relax protocol.** The reference is the SAME sequence in both runs and sits at
  -24.57 against -25.68. A protocol that moved energies would have moved it.
- **Not the gate recalibration.** Per-fold energies above are unchanged across v3 and v5.
  A gate changes which folds count, not what they are worth.

**It is the selection step failing.** "Selection gain" is the arm's own per-fold median
minus its cross-fitted value: positive means picking the best of 8 beat picking at random,
which is the only reason to have a selection step at all.

| complex | run | control | LLM | delta |
|---|---|---|---|---|
| `lnr_5njx` | llm_ladder, depth 10 | +4.75 | +1.21 | +3.54 |
| `lnr_6seo` | llm_ladder, depth 10 | +4.41 | +7.10 | -2.69 |
| `lnr_5njx` | es_pilot, **depth 4** | **-3.68** | +6.63 | **-10.31** |
| `lnr_6seo` | es_pilot, **depth 4** | **-5.06** | +3.59 | **-8.65** |
| `lnr_5njx` | llm_ladder, **depth 4** | **+11.04** | +0.83 | **+10.22** |

The control's selection gain is **negative** in the pilot: choosing the best of 8 control
draws gave -21.15 against a per-fold median of -24.82, i.e. worse than drawing at random.
A selection step that loses to random is not measuring sequence quality. The -10 is the
gap between the LLM arm's selection luck and the control's anti-luck, and neither term is
about the sequences.

The last row settles it. **`llm_ladder` truncated to depth 4 gives +10.22** -- the LLM arm
losing by ten -- on the same folds, same gate, same sequences that give +3.54 at depth 10.
Running the whole arm comparison at both depths shows the same reversal:

| B | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| llm_ladder **depth 10** dREU | -1.78 | -2.91 | -0.90 | -3.30 | -0.72 | -1.03 |
| w/l | 6/4 | 10/4 | 12/5 | 11/5 | 12/5 | 10/3 |
| llm_ladder **depth 4** dREU | -5.13 | -2.20 | -3.56 | **+1.93** | **+4.82** | **+5.72** |
| w/l | 2/0 | 6/2 | 7/2 | 3/3 | 2/6 | **0/5** |

At depth 4 the selection gain spans -5.06 to +11.04; at depth 10, +1.21 to +7.10. A -10
and a +10 are both available from this statistic at depth 4 depending which data it is
fed, so it carries no information about the proposer.

**Withdrawn accordingly:** the -8 to -11.5 REU pilot advantage; the per-generation
ordering that assigned the best candidate to round 3 in 4 of 6 complexes; the claim that
later generations made the held-out pick worse in 3 of 6; and the raw-vs-normalised
5/0-to-3/2 comparison. All are functions of 2-fold medians. The pilot's dG is
**uninformative, not zero** -- it cannot distinguish -10 from +10, so it says nothing
either way about whether evolutionary search helps.

### 14.6 What the pilot does establish

**The machinery works.** Three generations loaded, references complete at depth, 0.997
-1.000 arm pass rates, requeue-safe, caches behaving, and the gate-mix bug of §14.1-14.3
found and fixed through it.

**Interface PAE is recoverable here**, which §11 and §12 recorded as impossible for
`llm_ladder` (`interface_pae_status` is "ok" on all 1,608 pilot folds). It is now a column
in `cpseq_interface_compare.py`, absent where it does not exist.

**The interface metrics are not best-of-B selections** -- they are per-fold medians of one
chosen candidate against the reference on the same held-out folds -- so they do not carry
the §14.5 artifact. They are still only 3 complexes, so p floors at 0.250.

| metric | es_pilot_pep | es_pilot_lnr |
|---|---|---|
| pair ipTM | +0.046, 3/0 | -0.031, **0/3** |
| interface pLDDT | **-0.022**, 1/2 | **-0.017**, 1/2 |
| interface PAE v | -1.110, 3/0 | +0.885, **0/3** |
| binder PAE v | -0.248, 3/0 | +0.425, 0/3 |
| buried unsat, refined v | +0.500, 1/2 | +0.500, 0/2 |
| interface H-bonds | +2.000, 3/0 | -0.500, 1/2 |
| shape complementarity | +0.022, 2/1 | +0.083, 2/1 |
| buried area A^2 | **+364.7**, 3/0 | +12.7, 2/1 |

Two readings survive. **The corpora disagree on every confidence metric**: PepBench
candidates are more confidently placed than the reference, LNR candidates unanimously
less (0/3 on ipTM and on both PAE columns). And **interface pLDDT is worse than the
reference in both corpora** -- the selected candidate is placed less confidently than the
peptide it replaces, in 4 of 6 complexes. The +364.7 Å² of buried area in PepBench is
§13's objective bias at full strength, but what it bought cannot be priced here, because
pricing it means the dG that §14.5 withdraws.

**Validity carries no information.** Both arms pass 1.000, coverage 1.000/1.000, against
0.20-0.42 in `llm_ladder`. Every fold re-scored in §14.3 was pre-relax invalid and valid
only after the junction-restrained relax, so this is the v5 gate's behaviour and not a
property of the designs. §8's (validity, dG) Pareto question cannot be asked of this pilot.

### 14.7 Open

- **`pepbench_corpus_pot`**: re-score the 602 non-v5 folds in the 14 mixed complexes and
  all 670 become readable as pairs; re-score all 40,326 non-v5 folds and pooled corpus
  rates become valid too. The narrow fix is cheap and has no downside. Held pending a
  decision on the wide one.
- **Put the version key where it belongs.** The readers now cope, but the real fix is for
  `Journal.has` to include `scoring_version`, so a rescore is a new row by construction
  and no reader has to guess which verdict is current.
- **Hash the effective gate, not just the file.** `cpseq_search.py:1755` refuses to fold
  when `geometry_tolerances.json` changes hash, and that guard still passes: the file is
  untouched since 2026-10-19 and still reads `torsion_tol_deg: 8.44`. But
  `derive_tolerances` now builds the bar from code constants in `tolerances.py`
  (`ADOPTED_TORSION_TOL_DEG = 30.0`), so the guard protects a file that no longer decides
  anything. The comment above that guard describes this exact failure happening once
  before. The adopted constants belong in the cohort's recorded hash, and `tolerances.py`
  is still uncommitted.
- **Depth before anything else.** §14.5 cost four CPU-hours of re-scoring aimed at three
  wrong explanations before depth was tested, and depth was free to test (`--depth 4` on
  existing folds). Any future result whose effect size jumps several-fold against §7
  should have its depth checked first. A pilot is not cheaper for running at depth 4; it
  is uninformative, which is more expensive.
