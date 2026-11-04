# The cpseq dG proxy scorer: what it is, what it is worth, and how to re-run it

A learned proxy that predicts Rosetta interface dG from `(sequence, target)` so the cpseq
search can triage candidates without paying the **~238 s/candidate** Boltz-2 fold +
Rosetta refinement.

Everything below is measured. Energies are Rosetta **ref2015 REU** and are **not**
measured binding free energies. Lower dG is better throughout.

---

## 0. Data provenance: every result here is **potentials OFF**

| | source run | Boltz-2 potentials | rows / targets | `min_folds` |
|---|---|---|---|---|
| **training and all headline results** | `cpseq/llm_ladder` | **OFF** | 2031 / 20 (22,283 folds) | 8 |
| transfer *test set only* (§5, last subsection) | `cpseq/llm_ladder_pot` | **ON** | 167 / 10 (1,774 folds) | 5 |

Both use scoring version `v3-cis-amide-muco-windows`, stage `ladder`, energy field
`refined_dG_separated`, clipped at 0.0 REU. Checkable at any time: each dataset JSON
carries its own `source_out_dir`, and `learning_curve_v3.json`, `triage_v3.json` and
`verdict.json` all record which dataset they read.

**So: §1-§5 (except the transfer subsection), §6, §9 and §10 are all potentials-OFF
numbers.** `llm_ladder_pot` appears in exactly one place, as the *test* side of the
transfer experiment.

**Why the pin exists.** Potentials-ON changes the predicted structure and therefore every
dG, so the two runs are different measurement instruments on the same quantity. Pooling
the run directories would pool instruments. `MEANSCORE_SOURCE` in
`scripts/_cpseq_meanscore_preamble.sh` is deliberately ONE run directory, and
`llm_ladder_pot` is a transfer set, never training data.

**What is therefore NOT established.** No result here says the proxy behaves the same way
when trained on potentials-ON labels. In particular the §10 saturation finding is a
potentials-OFF statement: the pot run holds only ~16 sequences/target, far below the
130-row half-saturation point, so a potentials-ON learning curve is not measurable from
what is on disk. The transfer test is also **one-directional** - it trains OFF and tests
ON, never the reverse.

---

## 1. The headline

| question | answer |
|---|---|
| Does a proxy work at all? | Yes, weakly. Within-target Spearman **0.257** against a ceiling of 0.896. |
| What is the best predictor? | A **ridge** on composition-delta + ProteinMPNN features. Not a neural net. |
| Does it save folds? | Yes: **1.8x** at a budget of 20 of ~105 candidates (+0.98 REU vs a shuffled control, p=0.030). |
| Should it be wired in? | Yes, but only for **aggressive** triage (top 5-20 %). By 50 % the advantage is gone. |

## 2. The label: which statistic over the ~10 seed folds

The dataset's 31k outcome rows are only **2,443 unique `(target, sequence)` pairs** with
~10 Boltz-2 prediction seeds each. Per-fold supervision caps R^2 at **0.49**, because
seed-noise SD (4.41 REU) is about equal to the sequence-signal SD (4.33 REU). Aggregating
the folds is therefore not a convenience, it is the whole game.

`script_utils/cpseq_meanscore_reliability.py` sweeps 11 aggregators, scoring each by
within-target split-half Spearman with a Spearman-Brown correction:

| aggregator | reliability | 95 % CI |
|---|---|---|
| `hodges_lehmann` | **0.795** | [0.766, 0.817] |
| `trimmed20` | 0.790 | [0.762, 0.820] |
| `mean_clip0` | 0.785 | [0.774, 0.815] |
| `median` | 0.756 | [0.742, 0.791] |
| `min` (best fold) | 0.680 | [0.614, 0.709] |
| `valid_clip_mean` | **0.648** | [0.605, 0.697] |

Four rules fall out of this, and they are the part of this document most worth keeping:

1. **Never condition the label on `valid`.** It is the *worst* aggregator, because
   validity flips across seeds for **87.3 %** of sequences. Filtering on it throws away
   the signal and keeps the noise.
2. **Clip at 0 REU.** It barely changes *ranks* (`mean_raw` scores 0.778) but it is
   essential for *values*: a single +19,587 REU fold destroys any squared-error loss.
   An earlier recorded "-0.02" for the unclipped mean was this effect, on the value scale.
3. **The ceiling is `sqrt(reliability)` = 0.896, not 0.803.** Reliability is the agreement
   of two independent 10-fold labels; a *perfect* model scores its square root against one
   noisy label. Comparing a model to 0.803 over-credits it.
4. **More folds is the cheapest way to raise the ceiling**: 10 -> 0.79, **20 -> 0.90,
   50 -> 0.96**.

The pipeline trains under `trimmed20` and `mean_clip0`, the two the sweep cannot separate,
rather than picking one by assertion.

## 3. The metric: within-target, always

Between-target label SD is **12.3 REU**; within-target it is **3.76 REU**. A pooled
correlation is therefore mostly a target-identity readout. Every number here is
**within-target Spearman averaged over targets**, with whole targets held out.

How much this matters, measured: mean-pooled ESM-2 embeddings score **0.731 pooled** and
**0.141 within-target**. ProteinMPNN and the pocket MLM correlate **+0.875 pooled** and
**-0.008 within-target**. A pooled metric would have called all three excellent.

## 4. What the model may not see

- **No target-id embedding** - undefined for a held-out target. The target enters only
  through its reference peptide.
- **No post-fold readouts.** `iptm`, `plddt`, `dSASA_int`, `sc_value` and friends reach
  within-target rho **0.781** (97 % of ceiling) - but they require the fold the proxy
  exists to avoid. They are reported only as a diagnostic, and they are the reason we know
  **sequence -> structure is the entire difficulty**: once a structure exists, the dG is
  essentially determined.

## 5. Results

*All potentials **OFF** (`llm_ladder`), except the transfer subsection at the end, which
tests on potentials-ON. See §0.*

### Predictors, leave-one-target-out

| predictor | rho | % of 0.896 |
|---|---|---|
| **composition_delta + ProteinMPNN (bb + AR) ridge** | **0.257** | **28.7 %** |
| best network (`mean_clip0`/`rank`, + MPNN) | 0.232 | 25.9 % |
| composition_delta ridge | 0.205 | 22.9 % |
| `n_mutations` + identity | 0.023 | 2.6 % |

The 22.6k-parameter additive mutation-effect network **loses to a ridge at every one of
the six (aggregator x loss) configurations**. It is not a capacity problem; see §4.

### Pretrained weights: only structure-aware ones help

| weights | result |
|---|---|
| **ProteinMPNN** (backbone marginals + autoregressive) | **+0.048 paired, CI [+0.013, +0.084], p = 0.022** |
| PepMLM-650M fine-tuned on CPSea (pocket-conditioned) | **NULL** (+0.026) |
| ESM-2 650M, frozen, every pooling/PCA variant tried | **NULL** - all worse than composition alone |

Single-feature within-target rho (|rho| is the signal; a log-likelihood is
higher-is-better while dG is lower-is-better, so negative agrees):
`mpnn_ar_ll` **-0.149**, `mpnn_bb_ll` -0.130, `mlm_marg_ll` +0.026.

**Cost is not a consideration.** The MPNN features take **0.46 s/candidate** (933 s for
2031, CPU) against ~238 s for a fold. The "expensive" autoregressive feature is **0.19 %
of a fold**.

### Does it save folds? (`cpseq_meanscore_triage.py`)

Leave-one-target-out, rank a held-out target's ~105 candidates, "fold" the top M:

| M | best proxy | best random | gain | multiplier | % of oracle headroom |
|---|---|---|---|---|---|
| 5 | -34.58 | -33.75 | +0.83 | 1.6x | 17 % |
| 10 | -36.13 | -35.21 | +0.93 | 1.7x | 24 % |
| **20** | **-37.44** | **-36.50** | **+0.94** | **1.8x** | **33 %** |
| 30 | -37.90 | -37.18 | +0.72 | 1.6x | 31 % |
| 50 | -38.09 | -37.96 | +0.13 | 1.1x | 8 % |

Decisive test, proxy vs the shuffled control paired per target at M=20:
**+0.98 REU, CI [+0.18, +1.78], p = 0.0296, 15/20 targets.**

Two findings here matter more than the headline:

- **rho does not predict fold savings.** The MPNN features are an unambiguous rho win
  (+0.048, p=0.022) and worth almost nothing on the budget metric (+0.81 -> +0.94 at
  M=20; *worse* at M=50). Spearman rho is dominated by the bulk of the ranking;
  best-of-M depends only on the extreme tail. **Never promote a feature into the search
  on a rho gain alone.**
- **`n_mutations` is actively harmful**: -0.91 REU, multiplier **0.65x**, enrichment 0.74.
  Ranking by fewest substitutions is *worse than random*. "Closer to native is better" is
  false here, so anything that implicitly prefers low-mutation candidates costs folds.

### Does it survive a change of prediction protocol? (`cpseq_meanscore_transfer.py`)

**The only result in this document that touches potentials-ON data.** Trained on
`llm_ladder` (potentials **OFF**), tested on `llm_ladder_pot` (potentials **ON**). All 10
pot targets are among the trained 20, so each is **held out of its own fit** - both the
target and the protocol are unseen.

```
mean within-target rho  +0.171   95 % CI [+0.019, +0.288]   9/10 targets positive
same-protocol LOTO      +0.253   -> transfer retains 68 %
shuffled control        -0.014   (200 shuffles/target)
```

**Weak positive evidence, not a clean win.** Two qualifications that must travel with the
number: the protocol shift is **small** (median |shift| in the per-target median label is
0.84 REU, ~22 % of the 3.78 REU within-target signal SD), so this shows robustness to a
*mild* instrument change; and it is **underpowered** - the `>=5`-fold filter leaves only
157 rows over 10 targets (~16 each), per-target rho SE ~0.26, and the CI lower bound of
+0.019 is marginal. The transfer triage is null because M is a large fraction of a
16-candidate pool.

It is also **one-directional**: OFF-trained, ON-tested, never the reverse. That is the
useful direction if the search will run with potentials on - but every label the proxy was
fit to is potentials-OFF.

Conclusion: do not re-pin the proxy per protocol on this evidence, and do not claim
protocol independence either. The cheap way to settle it is **more scored sequences in the
pot run** - the one place extra folding buys real information rather than the +0.008 that
§10 shows more OFF data is worth.

## 6. Hard limits, stated so they are not rediscovered

- **Scope.** Every training candidate is a **1-7 substitution variant of one reference
  peptide per target** (median identity 0.83, lengths 5-16). The model interpolates inside
  a small mutational ball around a reference. That is the search-triage use case; it is
  **not** a general peptide-dG predictor.
- **The structural context is LINEAR.** All 20 native complexes have a median
  N(1)-C(L) gap of **21.8 A** (a closing amide is 1.33 A); **0 of 20** carry a closed ring.
  The candidates are all head-to-tail cyclic. So the MPNN features score each candidate
  against a backbone open by ~22 A in exactly the dimension the task is about. They work
  anyway, but this caps every fixed-native-backbone feature.
- **Do not build a threaded-Rosetta-dG feature on these structures.** It was considered and
  rejected: a force field is far more sensitive to the real geometry than a cyclic-blind
  GNN, declaring the ring would restrain a bond 22 A from closing (which forces the ring
  shut and then certifies the result), and the reason post-fold readouts work is that each
  candidate has *its own* structure - which no fixed backbone can provide.
- **Collinearity in the saved model.** Per-target centring makes each `*_delta` MPNN column
  **identical** to its raw column, so the ridge splits one effect across two coefficients:
  read `mpnn_bb_ll` and `mpnn_bb_ll_delta` as one feature at their sum. Reported by
  `cpseq_meanscore_fit_final.py` rather than silently deduplicated, because dropping a
  duplicated column changes the effective ridge penalty and would move every number above.
- **Only 20 of 92 targets have >= 5 unique sequences**, which is what limits the power of
  every paired test here.

## 7. Statistical practice this pipeline enforces

- **A null control in every harness.** The triage harness runs a shuffled selector, an
  oracle and `n_mutations` alongside the real arms, and refuses to interpret the table if
  the shuffled arm shows a gain. It currently reads **-0.04 REU, CI [-0.10, +0.02]**, with
  multiplier 1.00 and enrichment 1.00 at every budget.
- **The control needs many draws.** One shuffle per target is unbiased but so noisy
  (+/-1 REU over 20 targets, it read +0.36) that it cannot rule anything out. 200
  shuffles/target is what makes the +0.98 interpretable.
- **Never average a nonlinear transform.** The "compute multiplier" is a steep nonlinear
  function of the achieved dG, so its mean is biased upward: a *skill-free* selector
  scored **1.79x** by mean and 1.00x by median. Aggregate first and transform once, or use
  the median. This was caught only by the null control.
- **Oracles are split-half.** An oracle that selects on the same noisy label it is graded
  against is picking the luckiest seed, not the best peptide. Select on half the folds,
  score on the other.
- **Paired, per-target tests** (Wilcoxon + bootstrap over targets), never a difference of
  two averages - a mean-of-means "win" in this project has already turned out to live
  entirely in one arm of five.
- **`E[min of m draws without replacement]` has a closed form**: for ascending `y`,
  `P(y_i is the min) = C(n-i, m-1)/C(n, m)`. Validated against 20k-draw Monte Carlo;
  `m=1` gives the mean and `m=n` the min exactly. Monte Carlo made the self-test
  unrunnable; the closed form is instant and removes the harness's own noise.

## 8. How to re-run

All stages are CPU and read saved JSON, so every verdict is redrawable without refolding
or retraining. Configuration lives in `scripts/_cpseq_meanscore_preamble.sh` and arrives
as a **positional env file**, never via `sbatch --export`.

```bash
# the main chain: dataset + reliability -> scorer sweep -> report
sbatch scripts/submit_cpseq_meanscore.sh          # or the stages individually

# pretrained features
sbatch scripts/cpseq_meanscore_mpnn.sbatch   <envfile>
sbatch scripts/cpseq_meanscore_mlm.sbatch    <envfile>
sbatch scripts/cpseq_meanscore_esm.sbatch    <envfile>

# verdicts and artifacts
sbatch scripts/cpseq_meanscore_report.sbatch    <envfile>
sbatch scripts/cpseq_meanscore_triage.sbatch    <envfile>   # the deployment number
sbatch scripts/cpseq_meanscore_fitfinal.sbatch  <envfile>   # writes proxy_model_v3.json

# protocol-transfer arm (llm_ladder_pot, Boltz potentials ON)
sbatch scripts/cpseq_meanscore_potfeats.sbatch  <envfile>
sbatch --dependency=afterok:<id> scripts/cpseq_meanscore_transfer.sbatch <envfile>
```

Check the harness before the result:
`python script_utils/cpseq_meanscore_triage.py --self-test` must print
`all controls pass`.

**Source pin: `llm_ladder` only (potentials OFF).** `llm_ladder_pot` runs Boltz-2 with
potentials **on**, which changes the structure and therefore every dG. It is a *transfer*
set, not training data. Pooling run directories pools measurement instruments. Anything
re-run with `MEANSCORE_SOURCE` repointed is no longer comparable to the numbers in this
document - change the output paths too, or it will silently overwrite them.

## 9. The shipping artifact

`proxy_model_v3.json` holds the ridge: `mean`, `scale`, `coef`, `intercept`, the feature
names, and the scope and application notes. It is fitted on **potentials-OFF** labels
(`llm_ladder`, all 20 targets) - see §0 for what that does and does not license. Its output is a **per-target centred** dG, so
it ranks candidates **within one target** and is not comparable across targets.

Honest expectations, not the in-sample 0.310: **LOTO 0.257** for a target never scored,
**0.287** for a target whose journal already holds scored sequences (which is the setting
a search actually runs in).

## 10. Is it data-hungry? Measured, and it decides where to spend

`cpseq_meanscore_learning_curve.py`, 24 nested permutations per held-out target, on
`dataset_v3.json` - i.e. **potentials OFF** throughout (§0). Targets
and rows are confounded in the obvious curve (5 targets is ~500 rows, 19 is ~1900) and
they are different things to go buy, so three arms are run:

| arm | at 5 targets / 350 rows | at 19 targets / all rows | span |
|---|---|---|---|
| **A. targets**, all rows (diversity *and* volume) | 0.192 | 0.253 | **+0.061** |
| **B. diversity only**, total rows fixed at 350 | 0.183 | 0.198 | **+0.015** |
| **C. volume only**, all 19 targets | 0.198 | 0.253 | **+0.055** |

Arm A, paired per held-out target, K=5 -> 19: **+0.061, CI [+0.041, +0.083], p = 0.0001,
17/20 targets up.** The curve really rises - and arms B and C show the rise is about **4x
more volume than diversity**.

Full arm A curve: K=2 **0.150**, 3 0.177, 5 0.192, 10 0.220, 15 0.241, 19 **0.253**.

**Both axes are saturated, and they converge on the same wall.**

| axis | saturating fit | where we are | what more buys |
|---|---|---|---|
| **targets** `a*K/(K+b)` | a = **0.266**, b = 1.63 | K=19 -> 0.253 = **95 %** of a | K=40 -> 0.256, **K=92 -> 0.261 (+0.008)** |
| **rows** `a*R/(R+b)` | a = **0.265**, b = **130 rows** | 2011 rows -> 0.249 = **94 %** of a | **2x -> 0.257, 4x -> 0.261, 10x -> 0.263** |

Residuals <= 0.009 (targets) and <= 0.007 (rows). `b = 130 rows` is the number that
matters: the row curve half-saturates at 130 rows and is 94 % of the way up by 2000, so
the +0.056 measured from 350 -> 1995 rows is **the tail of the bend, not a slope to
extrapolate**. Doubling every target's scored sequences buys +0.008.

Even the label-noise lever is capped: 0.253 against a label of reliability 0.79 implies a
disattenuated true rho of 0.253/0.896 = **0.282**, so *perfect* labels would reach ~0.28.

**This feature set therefore tops out at rho ~0.27-0.28 whatever data is added**, which is
what the post-fold 0.781 in §4 already implied: the limit is the information in the
features, not the amount of it.

**Scope of the saturation claim:** potentials-OFF data only. It is not known whether a
potentials-ON proxy saturates in the same place, and it cannot be measured from the
current pot run (~16 sequences/target against a 130-row half-saturation point).

Two methodological notes that must travel with the numbers:

- **The centring confound was measured, not assumed.** At K=19 on a 350-row budget there
  are only 18 rows/target, so the per-target label offset is estimated from 18 noisy
  values (SE ~0.89 REU vs ~0.45 at 70). Arm B is therefore run twice, the second time with
  offsets from *all* of each target's rows. **Centring cost: +0.001** - not a confound.
- **Do not read this curve off few draws.** A 1-draw smoke test showed arm B *falling*
  (0.208 -> 0.179); that was draw noise and the 24-draw run reverses the sign.
- `volume@350` and `rows_matched@K=19` are the **same configuration** (19 targets x 18
  rows) via two independent code paths and agree to 3 decimals. The script asserts this.

The composition-only ridge runs the same shape one level lower (0.119 -> 0.205, asymptote
0.217), so the MPNN gain is a level shift, not a data-quantity effect.

## 11. What would actually move this

**There is only one real lever, and it is not data.** §10 fits both data axes and both
saturate at rho ~0.265; even noiseless labels cap at ~0.28.

1. **A candidate-specific structure cheaper than Boltz-2.** The only thing with real
   headroom. §4 shows the dG is essentially determined once a structure exists (0.781),
   §6 shows no fixed native backbone can substitute, and §10 shows no quantity of data
   closes the gap.
2. **More folds per sequence** - a *different* mechanism from more rows: it raises the
   label ceiling (10 -> 20 folds, 0.79 -> 0.90) rather than adding training data. Still
   bounded by the 0.282 disattenuated figure, so worth at most ~+0.03.
3. **More targets - for measurement, not accuracy.** Every paired test here is n=20
   targets and its CI width is target-limited, so more targets buys narrower error bars.
   It does **not** buy a better model (+0.008 projected at K=92).

Two earlier versions of this document recommended more targets, then more sequences per
target. Both were wrong on accuracy, and the second was written before the row axis had
been fitted. **A span measured across a range is not a slope to extrapolate** - fit the
curve before anyone spends fold budget on more data.

Not worth retrying: ESM-2 embeddings, the pocket-conditioned MLM, a bigger network, and
threaded-Rosetta on the native backbones. All four are measured or argued dead above.
