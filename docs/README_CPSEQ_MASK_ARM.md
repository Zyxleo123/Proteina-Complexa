# Arm E — the LLM names the positions, a learned model fills them

Splitting the proposal in two. The LLM reads the same PLIP/Biopython interface tables arm D
reads and answers with **positions only**; a model trained on complexes chooses the residues.

Started 2026-10-29. Everything below was verified by execution unless it says otherwise.
§4 is a negative result that redirected the arm, and §5 is the open question it redirected to.

---

## 1. Why split it

Arm D (`README_CPSEQ_LLM_SEARCH.md`) asks one model to do two jobs. The record says it is
good at one of them and bad at the other:

- **Good at reading the interface.** §6 records it locating the overlap and the empty
  positions unprompted, and citing real receptor residues from the PLIP tables.
- **Bad at writing sequences.** All 16 malformed round-1 proposals were an off-by-one
  length — every single one, no other malformation occurred. And §4 records its stated
  reasons routinely not matching its own edits.
- **Concentrated.** §6's diversity table: 4 of 14 distinct edited positions on `lnr_2v8x`,
  5 of 16 on `lnr_6seo`, against a random control's 9 and 10. Both readings were live —
  spending the budget where it matters, or premature convergence.

A position list cannot be miscounted, and the residue can be chosen by something fitted for
it. The budget is also easier to spend: freed from picking residues, the model opens more
positions (§3).

## 2. What was built

| file | what it is |
|---|---|
| `cpseq/llm_propose.py` | `OUTPUT_FORMAT_MASK`, `SYSTEM_PROMPT_MASK`, `parse_mask_completion`, `screen_masks`, `mask_headroom`, `MaskProposal`, `propose_mask_round` |
| `cpseq/mlm_propose.py` | `MlmProposer.fill` — fills masks somebody else chose |
| `script_utils/cpseq_mask_pilot.py` | the round-1 driver: LLM masks, fill, matched control |
| `script_utils/cpseq_build_pilot_pockets.py` | pockets for the pilot targets, via the training path |
| `utils/seq_conditioning.py` | `choose_reveal_mask` — explicit reveal mask overrides `keep_frac` |

Tests: `test_cpseq_llm_propose.py` 140 checks (22 new), `test_seq_cond_reveal_mask.py` 14.

Four rules carried over from arm D unchanged, because the arms must stay comparable: a
model's output is a suggestion and never a candidate; a short round is reported, never
padded; the proposer never sees the measurement folds; re-proposing an evaluated sequence
is refused, not cached.

Three decisions specific to this format:

- **A residue letter in a MASK line is malformed, not parsed.** Reading the number out of
  `K7` would hide a model that has not understood the division of labour, in the one place
  the log should show it.
- **History is rendered as bare positions**, `3 sub at 2, 7, 11`, not as `D5A`. No prompt
  line should teach the model to write a residue letter into its answer.
- **The cysteine rule is restated as the filler's constraint.** Stated as a prohibition it
  reads as "do not touch position 12", which is backwards: opening that position is how the
  search replaces a native cysteine, and the filler cannot put one back.

**Two screens run, and they charge different models.** `screen_masks` bounds the worst case
before any weight loads; `screen_candidates` judges what the fill actually produced, against
the original and with the shared alphabet and novelty rules. A legal mask can yield an
illegal sequence. On an elite the budget is not a cap on mask size — `mask_headroom` is the
budget less what the elite already spends, so re-opening a position it already changed is
free.

## 3. Round 1, measured

`cpsea_mlm_pocket` step 60000 as the filler, 3 pilot complexes, 8 masks each, 2 fills per
mask, `/data/user_data/yixiz/cpseq/mask_pilot/`.

**24 of 24 legal masks, 0 malformed**, against arm D's 16 malformed on the same complexes.

Diversity, against `README_CPSEQ_LLM_SEARCH.md` §6's table for the same parents:

| complex | distinct positions, arm D | arm E | median pairwise Hamming, D → E |
|---|---|---|---|
| `lnr_2v8x` | 4 of 14 | **12 of 14** | 2.0 → 5.0 |
| `lnr_6seo` | 5 of 16 | **16 of 16** | 2.0 → 4.0 |
| `lnr_1elr` | 4 of 5 | **5 of 5** | 3.0 → 3.0 |

Mask sizes rose with the division of labour: `lnr_2v8x` drew [3,2,2,2,2,4,4,3] where arm D
drew [1,1,1,1,1,1,2,2]. Part of the Hamming gain is therefore mechanical — bigger masks
change more positions — but the position coverage is not.

**Caveats.** The stated reasons remain unreliable (`lnr_2v8x` mask [1,2] is justified as
"ovl overlaps" where the reference has zero clashes); they are provenance and are never
parsed. The `lnr_1elr` control is undersized at 2 masks of 8: with L=5 there are only 5
masks of size 1 and the LLM had taken most of them, and the generator broke out of the loop
on first exhaustion instead of skipping that size. Fixed; the run on disk predates the fix.

**Nothing here has been folded.** These are proposals.

## 4. Why the CPSea unmasking model is the wrong filler for these targets

Measured before folding anything, and it is the reason this arm is not yet a result.

| | the 3 pilot complexes | CPSea val, same code |
|---|---|---|
| mean per-position perplexity | **16.3** (uniform = 20.0) | 12.0 |
| top-1 probability | mostly 0.08–0.13 | — |
| argmax = native residue | **2/35** | 0.231 |

**This is not an inference bug.** `cpseq_mlm_val_check.py` runs the identical path —
same `EsmMlmEncoder`, same `MlmProposer`, same logit restriction — on the model's own
validation shard and reproduces its reported accuracy: top-1 0.231 single-position, 0.232
budget-regime, against the manifest's 0.204 positional prior and the 0.268 the migration
doc reports.

**It is not a format mismatch either.** The pilot targets sit inside the training
distribution on every structural axis: lengths 5/14/16 (CPSea 5–16, median 13), pockets
46/37/48 residues (median 37, range 13–104), 17/8/6 fragments (median 10, range 2–31). The
pockets were rebuilt through `extract_pocket` and reproduce the target manifest's own pocket
sizes exactly (46/37/48), which is an independent check on the contact rule.

**The pocket channel is near-inert here.** Swapping pockets between the three targets at
matched seeds changes the sampled fill in 8/40, 10/112 and 7/128 draws. Native-peptide
pseudo-perplexity under its own pocket versus the best swapped one: `lnr_1elr` −16.2 (a real
effect), `lnr_2v8x` +0.31, `lnr_6seo` +0.11 (none).

The likely reason is the peptides, not the pockets: LNR originals are *linear* binder
sequences from crystal structures, and the model was fitted on cyclic peptides.

Redraw: `cpseq_mlm_val_check.py`, `cpseq_mlm_fill_probe.py`, `cpseq_mlm_pocket_probe.py`;
results under `/data/user_data/yixiz/cpseq_mlm/probes/`.

## 5. The co-generator as the filler, and the one number nobody has

`cpsea_seqcond_v2/checkpoints/last-v3-EMA.ckpt` (step 3000) is the project's own
sequence-conditioned flow model, and it is **in-domain for these targets** — the dataset doc
records it run on this same LNR tranche, docking reliably at `dSASA_int` median 1022 Å² with
a real interface in 100 % of samples, against Boltz-2's 933 Å² / 99.5 %.

Partial reveal is a first-class trained feature, not an improvisation. Its saved
`exp_config` carries `sequence_conditioning: {enabled: true, p: 0.5, p_full: 0.5,
keep_frac_min: 0.25, keep_frac_max: 1.0}`, so it saw rows 25–100 % specified with a
per-residue Bernoulli. `use_residue_type_feature` accepts a `[b, n]` bool documented as
"a partial sequence, generate the rest", and an unspecified position is the all-zero vector
in a dim-20 feature.

**Its known defect is in the half this arm discards.** It scored 0/48 on validity, entirely
from closure torsion (median deviation 48.5°, 14.6 % in window), inherited from training
data in which 41 % of native CPSea head-to-tail closures are cis amides. The proposal is the
decoded `residue_type`; the structure is dropped and the sequence refolded by Boltz-2
(76.2 % in-window) through the same frozen gate as every other arm.

**The missing number.** `sequence_recovery_metrics` already separates `cond_recovery`
(revealed — compliance) from `free_recovery` (hidden — actual prediction). Every seq-cond
run ever done used `keep_frac: 1.0`, where the free population is empty:

```
val_gen/seq/cond_recovery 0.9954   cond_exact_frac 0.9531
val_gen/seq/free_recovery    nan   n_free_res          0.0
```

So the strong compliance number is the **copy test**, which this project's notes flag twice
as a trap. What the model writes into a mask has never been measured.

The bar it must clear, priors fitted on CPSea and evaluated on this tranche
(`cpseq_seqcond_free_baseline.py`, `/data/user_data/yixiz/cpseq/seqcond/free_baseline.json`):

| prior | top-1 |
|---|---|
| positional, P(residue \| position, length) | **0.1273** |
| marginal (always `D`) | 0.0545 |

Both are below CPSea's own priors (0.204 / 0.120), which is itself consistent with §4: these
linear-binder sequences are harder for a CPSea-fitted table.

`free_recovery` scores agreement with the **native** residue, so it rewards reproducing the
original rather than improving on it. That is the right first instrument — it separates a
live channel from a dead one — but it is not a design-quality metric, and the Boltz-2 gate
remains the arbiter of that.

## 6. Measured: the co-generator does not fill a mask informatively

Jobs 10984088 / 10984089, both `COMPLETED`, 12 parents x 4 repeats, 400 ODE steps,
`/data/user_data/yixiz/cpseq/seqcond/freerec_keep0{50,75}/closure_rows_*.jsonl`. The first
non-NaN `free_recovery` this project has.

| keep_frac | cond_recovery | cond_exact_frac | **free_recovery** | n revealed / hidden |
|---|---|---|---|---|
| 0.50 | 0.959 | 0.851 | **0.0576** | 197 / 243 |
| 0.75 | 0.964 | 0.771 | **0.0444** | 305 / 135 |
| positional prior | — | — | *0.1273* | — |
| marginal prior (always `D`) | — | — | *0.0545* | — |

`n_cond + n_free` is 440 at both settings = 110 positions x 4 repeats, the same 110 the
baseline was computed over, so the comparison is position-for-position.

**The channel is wired; the prediction is not there.** `cond_recovery` at 0.96 says the
model faithfully reproduces what it IS shown, which also confirms the partial reveal took
effect. At hidden positions it lands at less than half the positional prior, and at
`keep_frac` 0.75 below even the flat marginal.

**It is not leaning on revealed context.** That was the alternative the second keep fraction
was run to test, and it fails in the wrong direction: MORE context (0.75) gave WORSE
hidden-position recovery, not better.

**Strength of the claim.** The four repeats are repeated draws over the same 110 positions,
so the effective sample is between 110*keep and 440*keep. Against the positional prior the
gap is roughly 1.5-3 SE depending on how the repeats are counted -- suggestive rather than
significant on its own. What makes it actionable is that both keep fractions agree, the
direction is the same as §4's independent finding on a different model, and the value sits
at the flat marginal.

**The standing caveat.** `free_recovery` scores agreement with the NATIVE residue, and a
proposer should eventually disagree with the native on purpose. Scoring below the marginal
does not prove the fills are bad, only that they carry no position-specific information
about the native. Separating "confidently different" from "uninformative" needs the entropy
of what it emits, which is the measurement that settled §4 and has not been made here.

**A mechanism worth recording.** This model co-generates sequence and structure, and the
sequence is a readout of the latents through the frozen AE decoder. Its structures fail
closure 0/48. A sequence decoded from a geometrically wrong backbone has no particular
reason to be a good sequence -- which would mean the structure defect and the sequence half
are not separable after all, and "discard the structure" does not rescue the sequence.

**What is still open, in order of cost.** `p_full: 0.5` means half this checkpoint's
conditioned training was full-reveal, and it stopped at step 3000; the config and warm start
for a partial-reveal-heavier run both exist. That is the cheap fix, and unlike §4's model
this one is at least in-domain for these targets.
