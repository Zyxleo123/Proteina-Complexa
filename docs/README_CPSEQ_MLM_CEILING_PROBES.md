# Is the cpseq_mlm plateau the task, or the model?

Three probes built to settle a disagreement about the unmasking proposer, not to defend a
position in it. The arm's migration doc argues that hidden-position recovery near 26.7% and
a loss near 2.36 nats are close to intrinsic, because many amino acids are interchangeable.
That argument was challenged on three grounds, all of which hold up:

1. **Interchangeability is plausible but does not establish a numerical limit.** "Residues
   substitute for each other" does not pin the best achievable loss at 2.36 any more than
   at 1.6.
2. **The plateau is entangled with the learning-rate schedule.** The trainer decays LR to
   zero, so flattening at the end cannot distinguish convergence from steps becoming too
   small — and the run covered ~0.76 dataset-equivalents, which is less than one epoch.
3. **Recovering the recorded sequence is not the same as finding a good one.** The
   objective contains no energy supervision, so a substitution can be a recovery error and
   a better peptide. This project already has a cautionary case: retention correlated with
   Rosetta dG at r² = 0.004.

A fourth point was a straight correction, now accepted: the migration doc's "23 standard
deviations from zero" was computed from spread across checkpoints on one fixed validation
panel. Those are correlated measurements of the same panel, so that is an estimate of
optimization noise, not of significance across unseen targets. The clustered intervals in
these probes are the replacement, and they resample **clusters**, never positions.

These probes address (1). They do **not** address (3) — nothing here measures closure or
dG, and the proposer test is what settles that. (2) needs a short constant-LR continuation,
which is a training job, not a probe.

---

## 1. What was missing from disk

`cpseq_train_mlm.evaluate` accumulates hit counts and discards the predictions. Every
number the arm has reported is therefore an exact-match accuracy, and nothing else could be
asked of a finished run without another GPU pass. The first probe fixes that once:

| Stage | Script | Resource | What it writes |
|---|---|---|---|
| dump | `script_utils/cpseq_mlm_dump_preds.py` | GPU, one forward pass | per-position 20-way log-probs on the fixed panel |
| reduce | `script_utils/cpseq_mlm_softacc.py` | CPU | substitution-tolerant scores, calibration, clustered CIs |
| ceiling | `script_utils/cpseq_mlm_ambiguity_probe.py` | CPU, no model | the task's irreducible accuracy and entropy from the data |

Launch: `scripts/submit_cpseq_mlm_probes.sh`. The dump→reduce edge is `afterok`; the
ceiling probe is a **sibling** with no dependency, because it needs no checkpoint and
making it wait behind a GPU job would only delay it.

The dump rebuilds the panel from the run's **own manifest** (seed 42, panel_size 800, the
recorded mask schedule) and encodes through the same path, including the `too_long` filter.
It then gates on reproducing the run's recorded `val/acc/all` to within 5e-4. That gate is
the point: a differently sharded corpus changes row order, which changes the panel, which
would make every paired number quietly wrong while still printing a plausible accuracy.

## 2. The substitution-tolerance test, and why the naive version is worthless

Score each position four ways: exact match, BLOSUM62 ≥ 0, BLOSUM62 ≥ 1, and a graded score
normalised per true residue (diagonal → 1, worst substitution → 0; normalising over the
whole matrix instead would make a correct TRP worth three correct LEUs).

The tempting number — "the model's soft accuracy" — means nothing on its own, because a
chemistry-tolerant metric also flatters a chemistry-keyed lookup table; that is close to
what the lookup optimises. The quantity reported instead is the **lift differential**:

```
lift(scorer)  = soft_ge1(scorer) - exact(scorer)
differential  = lift(model) - lift(chem prior)
```

Pre-registered reading, fixed before the run:

- **differential clearly > 0** — the misses really are conservative. Exact match
  understates the model and the interchangeability argument has support.
- **differential ≈ 0** — the model's errors are no more chemically sensible than a
  positional lookup's. Interchangeability then does not rescue the loss, because the
  baseline enjoys it equally.
- **differential < 0** — evidence against the story, not for it.

**The instrument was validated before being pointed at the model.** Two synthetic dumps
were built at matched exact accuracy, one whose misses are the best BLOSUM substitution and
one whose misses are uniform:

| synthetic model | exact | soft_ge1 | model − chem (exact) | **LIFT DIFF** |
|---|---:|---:|---:|---:|
| conservative errors | 0.2334 | 0.8810 | +0.0602 | **+0.5653 [+0.5450, +0.5846]** |
| random errors | 0.2385 | 0.3178 | +0.0572 | **+0.0026 [−0.0139, +0.0191]** |

The two are indistinguishable on the metric the arm currently reports — same exact
accuracy, same margin over the chem prior — and the lift differential separates them
completely. That is the discriminating power the question needs.

## 3. The calibration test, which needs no baseline

The sharper half, and it follows from the dumped distributions for free. If 2.36 nats is
irreducible aleatoric entropy, a converged model should *know* it: mean predictive entropy
should sit near the NLL, and mean max-probability near the accuracy. A model that is
**confidently wrong** — max-probability far above accuracy — is not reporting an entropy
floor; it is miscalibrated or underfit, and the ceiling claim does not follow. Unlike the
lift differential this needs no baseline to interpret, which is why it is the better test
of the plateau claim.

## 4. The data-side ceiling, and the confound that limits it

A predictor's ceiling is a property of its **input set**, and both the inputs and the data
are on disk. Group corpus positions by the context a predictor can see, then measure how
much the recorded residue still varies:

| grouping | context | what its ceiling means |
|---|---|---|
| `chem_prior` | (pos, L, cyc) | the chemistry-aware lookup's own ceiling |
| `model` | (pocket, L, cyc, pos) | **the model's** ceiling — the number the claim needs |
| `cluster` | (cluster_id, L, cyc, pos) | over-groups, so a **lower bound** on the model's |

Two estimators per grouping, biased in opposite directions and both reported: leave-one-out
argmax accuracy (understates, because it predicts from m−1 samples) and conditional entropy
with the Miller-Madow correction (plug-in entropy overstates determinism on small groups,
i.e. it flatters the ceiling claim). Reported by group-size stratum, because both biases
shrink with m — if the ceiling rises with group size, the pooled figure is an underestimate.

**A trap found while building this, which limits what the probe can claim.** A group of
identical rows is not evidence of a determined task. The pocket here is extracted by radius
around the peptide, so two rows sharing a pocket string to the character are near-duplicate
complexes, not two different peptides against one target. On the smoke corpus the `model`
grouping found 6 shared contexts out of 5994 and reported LOO 0.80 / H 0.19 — "nearly
deterministic" — purely from duplicate entries. Groups offering only one distinct peptide
are therefore dropped before estimating, and the drop count is reported, because that count
*is* the measurement of how much apparent determinism was redundancy.

The consequence is a limitation to state plainly rather than work around: **because the
model's own input is a function of the thing being predicted, the data cannot hand us a
clean Bayes ceiling for the model's input set.** Exact-pocket duplicates are rare, and the
ones that exist are duplicates. The `chem_prior` grouping is clean and fully covered; the
`cluster` grouping is a bound whose direction is known but which redundancy inflates. So
§3's calibration columns, not this probe, carry the weight on the plateau question.

---

## 5. Results

Run 2026-10-29, jobs 90401 (dump, gpu25 a6000), 90402 (softacc), 90403 (ambiguity, lov1),
on `cpseq_mlm_pocket` / `hf_step60000`. Every number below is a reduction over files in
`/zfsauton/scratch/yixiz/cpseq_mlm/probes` and is redrawable without a GPU.

The dump reproduced the run's own `val/acc/all` to six decimals (0.267606) and the shuffled
control to 0.226193, so the panel is provably the one the training curve scored.

### 5.1 Substitution tolerance: the interchangeability argument does not hold

4757 positions, 617 clusters, 7.7 positions per cluster. CIs resample clusters.

| scorer | exact | soft ≥0 | soft ≥1 | graded | lift |
|---|---:|---:|---:|---:|---:|
| **model** | 0.2676 | 0.4574 | 0.3521 | 0.4429 | 0.0845 |
| chem prior | 0.2281 | 0.3822 | 0.2958 | 0.3899 | 0.0677 |
| pos prior | 0.2041 | 0.3544 | 0.2781 | 0.3610 | 0.0740 |
| marginal ("D") | 0.1200 | 0.2964 | 0.2146 | 0.3094 | **0.0946** |

| differential | value, 95% CI over clusters |
|---|---|
| model − chem prior, exact | +0.0395 [+0.0264, +0.0547] |
| model − chem prior, soft ≥1 | +0.0563 [+0.0388, +0.0746] |
| model − chem prior, **LIFT DIFF** | **+0.0168 [+0.0081, +0.0258]** |
| model − marginal, **LIFT DIFF** | **−0.0101 [−0.0211, +0.0009]** |

Substitution tolerance helps the model more than the fair baseline, significantly — and by
**1.7 points**, against the ~56 points the validated instrument registers on a model whose
misses are actually conservative (§2). Measured against a constant-residue predictor the
differential is *negative*: the model's errors are no more chemically defensible than always
guessing "D", whose lift (0.0946) is the highest of any scorer here.

So exact-match recovery does understate the model, by a real but small margin. The
interchangeability mechanism the migration doc proposed is **not** what makes the loss 2.36.
The lift differential is positive in every regime and largest on `anchor` (+0.0206).

### 5.2 Calibration: the loss is not inflated by overconfidence

| | correct pocket | shuffled pocket |
|---|---:|---:|
| mean NLL (the loss) | 2.3567 [2.3052, 2.4035] | 2.5352 [2.4928, 2.5783] |
| mean predictive entropy | 2.3511 [2.3035, 2.3933] | 2.3890 [2.3495, 2.4238] |
| mean max-probability | 0.2700 [0.2565, 0.2854] | 0.2579 [0.2468, 0.2706] |
| exact accuracy | 0.2676 [0.2516, 0.2860] | 0.2262 [0.2129, 0.2398] |
| **overconfidence** (maxprob − acc) | **+0.0024** | **+0.0317** |

Correctly conditioned, the model is essentially perfectly calibrated: entropy within 0.006
nats of its NLL, confidence within 0.24 points of its accuracy. The pre-registered
refutation — "confidently wrong, so the loss is not a floor" — **did not occur**.

Read the asymmetry precisely. Overconfidence would have refuted the ceiling claim; its
absence does not establish it. The Bayes-optimal predictor and a calibrated *underfit*
predictor both satisfy entropy ≈ NLL — they differ in the **level**, not the calibration. So
this closes one route to "there is headroom" and leaves the level question open.

**The shuffled arm is the more interesting half.** With a wrong pocket the model lands on
the chemistry-aware lookup table almost exactly (0.2262 vs 0.2281) and its lift differential
collapses to +0.0040 [−0.0043, +0.0119] — CI containing zero. And its overconfidence jumps
13×: NLL rises 0.18 nats while its self-reported entropy barely moves. **The model cannot
tell that it has been handed the wrong target.** For a proposer inside a search loop that is
a named failure mode, not a curiosity: its confidence is not a usable signal for detecting
bad conditioning.

### 5.3 The data-side ceiling: not obtainable, for a structural reason

2,578,602 rows, 32,595,510 positions.

| grouping | usable contexts | coverage | LOO acc (largest supported stratum) | H (MM) |
|---|---:|---:|---:|---:|
| `chem_prior` | 36 / 36 | **1.000** | 0.2036 (n=32.6M) | **2.5571** |
| `model` | 507 / 2232 | **0.000** | 0.8637 (n=675) | 0.1359 |
| `cluster` | 253,731 / 258,050 | 0.703 | 0.5445 (n=5.2M) | 1.4391 |

The estimator validates: `chem_prior` LOO accuracy 0.2036 against the manifest's in-sample
`positional_argmax_acc` of 0.2041.

**The headline from this table is a measurement of the conditioning channel, not a ceiling.**
The (pos, L, cyc) context carries H = 2.5571 nats. The model achieves 2.3567. So everything
the model extracts from the pocket sequence is worth **0.20 nats, about 8% of the entropy a
position-and-chemistry lookup faces** — and §5.2 shows that removing the correct pocket drops
it onto that lookup. That 0.20 nats, equivalently ~4 points of exact accuracy, is the entire
prize the target-conditioning apparatus is currently winning.

Both candidate ceilings fail, each for a stated reason:

- **`model` grouping: coverage 0.000.** 2232 of 2,576,306 pocket contexts are shared at all,
  and 1725 of those offer only one distinct peptide — pure duplicates, which the filter
  drops. The cause is structural: the pocket is extracted by radius *around the peptide*, so
  an identical pocket string nearly implies a near-identical complex. A predictor's Bayes
  ceiling is not estimable from data when its input is a function of its target.
- **`cluster` grouping: unusable as a ceiling.** It reaches 0.5445 / 1.44 nats on 5.2M
  positions, which looks like enormous headroom — but **train and val clusters are 100%
  disjoint** (0 of 30,872 val clusters appear in train; measured directly). So that figure is
  a *within-family memorization* ceiling for a predictor that has already seen the cluster,
  and nothing the model could reach on the val panel. Quoting it as headroom would be wrong.

### 5.4 Verdict on the disagreement

Neither side's stronger claim survives.

- The docs' "loss ≈ 2.4 is near-intrinsic" is **not established**, and its stated mechanism
  (interchangeability) is **refuted**: tolerance is worth 1.7 points, not tens.
- The counter-claim that a bad metric or an unconverged model is hiding large headroom is
  **also not supported**: exact match understates by 1.7 points and the model is calibrated
  to within 0.24 points of its own accuracy.
- The level of 2.3567 nats remains **undetermined**, and these probes establish that it
  cannot be determined from this corpus for this input set.

What they do establish, and what should drive the next decision: **pocket sequence is worth
0.20 nats.** The conditioning channel the arm is built around contributes 8%, and with it
removed the model is a chemistry-keyed lookup table. That is a measured argument for changing
the *inputs* — target geometry — over further training on the same ones, and it is an
argument that did not depend on settling the ceiling question at all.

## 6. What these probes do not settle

- **Downstream value.** No number here is closure or dG. A 4-point recovery edge over a
  lookup may enrich good candidates or may not; the matched-budget proposer test against
  chemistry-aware sampling and random mutation is the only thing that answers it.
- **Optimization headroom.** Needs a short continuation at a meaningful LR, compared
  against the annealed baseline at *matched total tokens* — resuming an annealed checkpoint
  usually spikes loss then recovers, and a probe too short to distinguish "spike and return"
  from "genuine further descent" can read positive or negative on the transient alone.
- **The deployed sampler.** Recovery is measured here with all hidden positions masked at
  once; deployment fills them one per forward (the arm's own prior-art table cites Brooks et
  al. for exactly that reason). Those are different prediction problems. The fix is cheap
  and falls out of the proposer test, which runs the sequential sampler anyway: instrument
  per-position recovery in that run and the deployment-setting number comes free.
