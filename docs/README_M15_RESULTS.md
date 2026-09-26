# Milestone 1.5 -- results

Run `m15_20260922_184436`, 22 September 2026. Complete: 60 LNR + 600 PepBench ceilings,
4,000 calibration bridges, 20,000 spatial pairs, 19/20 timing complexes.

**Superseded in part by run `m15_20260925_190541` (26 September 2026), which adds the
terminal-pinned bridged arm. Read section 4b before quoting any bridged number from
sections 3, 4 or "Read this first" -- those maximise over anchor pairs CPSea cannot
reach.**

Method, caveats and how to re-run: `README_M15_CEILING_AUDIT.md`. This file is the
findings and what they oblige.

Report artifact: `evaluation_results/m15_20260922_184436/M15_CEILING_REPORT.md`
Flat per-complex rows: `ceiling_rows.csv`, `achieved_over_ceiling.csv`

---

## Read this first

The audit produces a **bracket, not a number.** `ceiling_refined` is a FLOOR (it writes off
every released residue's contacts); `ceiling_permissive` is the upper bound (it saturates).
Where they are close, the chemistry is answered. Where they are far apart, it is not.

| chemistry | bracket | achieved | verdict |
|---|---|---|---|
| isopeptide | [1.000, 1.000] | 0.505 | **answered** -- ~50% of what is available |
| disulfide | [0.981, 1.000] | 0.478 | **answered** -- ~48% of what is available |
| head-to-tail | [0.586, 1.000] | 0.500 | **open** -- the bracket does not constrain |

`achieved` is the median contact retention over sampled edits that produced the requested
chemistry **and** actually closed the ring (53,595 rows -> 36,265 right type -> 17,721
closed).

Two labels the numbers above need. The head-to-tail lower bound **0.586 is the LNR slice**
(n = 60); PepBench is 0.584 and is 600 of the 660 rows, so the pooled median is essentially
PepBench's. And `ceiling_refined` is a **ratio with peptide length in its denominator**, which
is why section 3's gap and length conclusions were withdrawn -- quote `k` (released residues),
not the ratio, for anything about gap or length.

---

## 1. The question the milestone was set to answer

> "If a 13-mer at 30 A can retain at most ~0.5 of its contacts under head-to-tail closure,
> then the 0.37-0.42 retention the prior line was treated as a failure is near ceiling."

**For bridged chemistries: no, it is not near ceiling.** Isopeptide and disulfide ceilings
are 1.000 under both bounds -- a bridge can be placed without moving the peptide at all,
because some (i, j) pair is already at bond distance. Measured retention is ~0.5. That is a
real deficit of roughly half, not a geometric limit.

**For head-to-tail: undetermined, and it cannot be settled with this instrument.** The
bracket is [0.586, 1.000] -- lower bound the **LNR slice**, n = 60; PepBench is 0.584 -- against
an achieved 0.500. The true ceiling could be 0.6 (in which case 0.5 is near-optimal) or 1.0
(in which case half the interface is being thrown away). Nothing in this run distinguishes
those.

The bracket width is also flat across every gap bin past 10 A (0.39-0.47), so this is not
undetermined in *some* band and settled elsewhere -- it is undetermined everywhere past 10 A.

## 2. What that obliges

**Route the 20-45 A coverage band to bridged targets.** This is the change to section 3's
coverage target that Milestone 1.5 was gating. It rests on section 1 -- the bridged ceiling
is exactly 1.000 with a measured ~50% deficit -- and not on section 3, whose gap and length
conclusions are withdrawn (see below).

The reason is *not* that the head-to-tail ceiling is least constrained in that band. The
bracket is `[floor, 1.000]` and its width is 0.41 / 0.42 / 0.47 / 0.39 / 0.40 / 0.44 across
the 10-15, 15-20, 20-25, 25-30, 30-45 and >45 A bins -- flat to within noise. Head-to-tail
is **equally unconstrained everywhere past 10 A**, so no band is special. Bridged is simply
the only chemistry where the deficit is measurable at all, which is what makes it the place
to spend effort. Optimising head-to-tail retention anywhere past 10 A means optimising
against an unknown scale.

**Keep the 20-45 A coverage target. Do not relax it.** The plan's section 5 has closure at
60% in the smallest gap bin against 4% in the largest. Nothing geometric explains that
collapse: `feasible_any` is **1.000 in every gap bin and for every chemistry** -- closure is
geometrically possible for all 660 complexes, including both >45 A cases. What rises with
gap is the retention *cost* of closing (section 3), not the *possibility*. So the 60% -> 4%
fall is a sampler limitation, not a wall, and relaxing the coverage target would be
conceding a solvable problem.

**If bridge span replaces terminal gap as the discriminator, the coverage target changes
axis -- and the number to match is not in this run.** Section 4 measures isopeptide
hostability at 71.7% (LNR) and 64.7% (PepBench), while CPSea natives are 100% hostable by
construction. So a corruption that produces realistic open states has to **reduce
hostability to roughly two thirds**, not move the terminal-gap median from 14 to 20 A. That
is a gentler requirement than the gap-based one, but it is also unmeasured: nothing here
reports hostability on synthetic open states. **Measure it on the corrupted states before
the pilot**, and treat ~2/3 isopeptide hostability as the acceptance criterion.

**Stop quoting mainchain `frac_of_ceiling`.** 86.4% of mainchain complexes have a sampled
attempt above `ceiling_refined` (max 2.74x). That is the floor being a floor, not the
sampler beating physics.

**Decide whether to close the head-to-tail bracket at all.** It needs conformational
sampling under the closure constraint -- generate closure-satisfying conformers, measure
retention directly -- not a tighter analytic bound. Milestone-2-scale work, so it is a
scope call.

## 3. Ceiling against gap and length (head-to-tail, floor)

> **Both conclusions previously drawn in this section are withdrawn.** "Past 10 A the gap
> stops mattering" and "short peptides are the constrained case" were artifacts of the
> ceiling being a ratio with peptide length in its denominator. The corrected reading is
> below and it points the other way on both counts.

### Why the ratio cannot carry either claim

`ceiling_strict` is `len(contacts whose peptide residue is in [a, b]) / len(all contacts)`.
With contacts spread roughly evenly along the chain that is `held_window / L`: **peptide
length sits in the denominator.** Two failures follow.

**The length trend is arithmetic.** The ceiling climbs 0.48 -> 0.79 across the length bins
while the number of residues that must actually be released *also* climbs, 4 -> 6:

| peptide length | n | gap_median | ceiling (ratio) | **`k_direct`** |
|---|---|---|---|---|
| 5-8 | 215 | 16.8 | 0.48 | **4.0** |
| 9-11 | 271 | 24.3 | 0.58 | **5.0** |
| 12-14 | 120 | 23.5 | 0.72 | **5.0** |
| 15+ | 54 | 26.2 | 0.79 | **6.0** |

**A longer peptide needs MORE residues released, yet shows a HIGHER retention ratio.** The
two move in opposite directions, which is the artifact stated as plainly as it can be: the
denominator grows faster than the cost does. So "long peptides retain more" cannot be read as
"closure is easier for them" -- it is the same-or-worse absolute cost divided by a bigger
number.

That table still confounds gap (its `gap_median` column climbs too), so the decisive evidence
is the stratified one. **Holding gap fixed in the 20-25 A bin** -- the largest, n = 224 -- and
walking across length:

| at gap 20-25 A | 5-8 | 9-11 | 12-14 | 15+ |
|---|---|---|---|---|
| `k_direct` | 5.0 | 5.0 | 5.0 | **5.5** |
| ceiling (ratio) | 0.48 | 0.52 | 0.75 | **0.90** |

**The cost is flat at constant gap while the ratio nearly doubles.** That is the whole
confound in four columns, and it is the form of the claim that survives: length is not what
makes closure expensive, it is what makes the ratio look generous.

**The gap trend is masked by a confound.** Gap and length move together in this dataset
(Spearman 0.47 overall, 0.55 past 10 A). The report's own `len_median` column (measured, not
inferred) shows length climbing monotonically with gap:

| terminal gap | n | ceiling (floor) | measured `len_median` | max held window | implied released `k` |
|---|---|---|---|---|---|
| <5 A | 4 | 0.97 | 10.5 | 9.5 | 0.3 |
| 5-10 | 35 | 0.85 | 11.0 | 10.0 | 1.7 |
| 10-15 | 83 | 0.59 | **6.0** | 4.0 | 2.5 |
| 15-20 | 162 | 0.58 | **8.0** | 5.0 | 3.4 |
| 20-25 | 224 | 0.53 | **9.0** | 6.0 | 4.2 |
| 25-30 | 93 | 0.61 | **11.0** | 7.0 | 4.3 |
| 30-45 | 57 | 0.60 | **13.0** | 8.0 | 5.2 |
| >45 | 2 | 0.56 | **15.0** | 8.5 | 6.6 |

Longer peptides sit at larger gaps, and length lifts the ratio mechanically. So past 10 A a
*rising* length effect cancels a *falling* gap effect and the ceiling column looks flat.
Taking the denominator out -- these are now **per-complex medians of `k_direct =
L - refined_window_len`**, measured, with no uniform-contact assumption:

| terminal gap | n | ceiling (ratio) | **`k_direct`** |
|---|---|---|---|
| <5 A | 4 | 0.97 | **2.0** |
| 5-10 | 35 | 0.85 | **3.0** |
| 10-15 | 83 | 0.59 | **3.0** |
| 15-20 | 162 | 0.58 | **4.0** |
| 20-25 | 224 | 0.53 | **5.0** |
| 25-30 | 93 | 0.61 | **6.0** |
| 30-45 | 57 | 0.60 | **7.0** |
| >45 | 2 | 0.56 | **8.5** |

**`k` rises monotonically across all eight gap bins, 2.0 -> 8.5, with no flattening
anywhere.** From the 15-20 A bin to 30-45 A the ceiling moves 0.58 -> 0.60 (1.03x,
apparently flat) while the released-residue cost moves 4 -> 7 (1.75x). The gap keeps
mattering; it was hidden by a growing denominator.

### What this changes

- **The gap does not stop mattering at 10 A.** It is the dominant driver of closure cost
  (`beta_gap` +0.87 on `k`, r2 = 0.76), rising monotonically over all eight bins and costing
  4 -> 7 released residues between the 15-20 and 30-45 A bins.
- **Long peptides are not the easy case.** Marginally the cost *rises* with length (4 -> 6);
  at constant gap it is flat (5.0 -> 5.5). Either way only the fraction it represents falls,
  and length is a ~4:1 weaker predictor of cost than gap.
- **The brief's original intuition was closer to right than section 3's rebuttal of it.**
  The ceiling does keep tightening through 20-45 A once measured in residues.
- **This does not weaken the routing decision in section 2** -- that rests on section 1's
  brackets and the bridged deficit, not on this section. It does remove the stated reason
  "the ceiling is least constrained in that band", which was never supportable.

### The joint regression

Standardised coefficients, so `beta_gap` and `beta_len` are comparable (n = 660):

| response | model | beta_gap | beta_len | r2 |
|---|---|---|---|---|
| ceiling (ratio) | gap only | **-0.101** | - | 0.010 |
| ceiling (ratio) | length only | - | +0.593 | 0.352 |
| ceiling (ratio) | gap + length | **-0.478** | +0.814 | 0.532 |
| **`k` released** | gap only | **+0.874** | - | 0.764 |
| **`k` released** | length only | - | +0.565 | 0.319 |
| **`k` released** | gap + length | **+0.779** | +0.204 | 0.796 |

Three things fall out.

1. **The ratio shows essentially no gap effect (-0.101, r2 = 0.010) while `k` shows a
   dominant one (+0.874, r2 = 0.764).** Restricted to gaps past 10 A the ratio's gap
   coefficient is **+0.042** -- flat and, if anything, the wrong sign. That is precisely the
   reading section 3 originally published.
2. **Controlling for length recovers the gap effect even in the ratio**, -0.101 -> -0.478, a
   4.7x change from adding one covariate. A marginal coefficient that moves that much was
   reading the other variable.
3. **On `k`, gap beats length roughly 4:1** (+0.779 vs +0.204, and +0.700 vs +0.276 past
   10 A). Length is a real but secondary effect on closure cost; it is the *primary* effect
   on the ratio (+0.814) only because it is the denominator.

**VIF = 1.3**, so these joint coefficients are stable and readable as effect sizes -- the
collinearity here is moderate, not severe. (An earlier draft of this section warned they
would be unstable; that warning was written before the measurement and does not apply.)

### One caveat that did not go away

`k_implied - k_direct` has a median of **-0.91 residues**: the uniform-contact algebra
*understates* the true released cost by about one residue, so contacts are not uniformly
distributed along the chain. The trend directions above are unaffected -- they are all
measured on `k_direct` -- but any hand calculation of the form `L * (1 - ceiling)` should be
expected to run about a residue light.

Full tables, including gap stratified within every length bin: section 2b of
`M15_CEILING_REPORT.md`.

## 4. Bridge-span profile (section 3 of the reply)

The terminal N-C gap measures the free tails, not the ring. Per peptide, over every (i, j)
with |i - j| >= 3:

| set | n | N-C gap | median bridge span | best span | disulfide hostable | isopeptide hostable |
|---|---|---|---|---|---|---|
| LNR | 60 | 18.2 A | 11.6 A | 5.2 A | 46.7% | 71.7% |
| PepBench | 600 | 21.7 A | 12.5 A | 6.8 A | 30.0% | 64.7% |

"Hostable" = the peptide already carries an (i, j) pair inside the measured bond window.
**Two thirds of real linear binders can host an isopeptide bridge as they stand**, against
a third for disulfide. This is the feature that says whether a given linear peptide is
cyclizable and where -- the terminal gap does not.

Caveat: hostable is geometric. It says nothing about whether residues i and j are a LYS/ASP
or CYS/CYS pair. `*_anchor_native_compatible` in `ceiling_rows.csv` carries that, and prior
work already established isopeptide failure is ~50% anchor identity.

## 4b. The bridged advantage, on anchors the model can actually reach

*Run `m15_20260925_190541`, 25-26 September 2026 (660 ceilings, all 12 shards).* This
section supersedes every bridged number above it.

Sections 3 and 4 report the bridged ceiling as a maximisation over all `(i, j)` pairs with
`|i-j| >= 3`. CPSea conditions on the chain **termini**, so `(0, L-1)` is the only anchor
pair it can place a bridge between; the free scan is an upper bound it cannot reach. The
re-run measures both (`<chem>_term_*` = pinned).

**The free scan's 1.000 was an artifact, and its mechanism is now known.** In `k` --
residues released to close, which has no length in the denominator and is therefore the
quantity these claims must be made on:

| chemistry | k median | k mean | k = 0 | refined ratio |
|---|---|---|---|---|
| mainchain | 5.0 | 4.77 | 0.2% | 0.585 |
| **disulfide_term** | **4.0** | 3.78 | 3.3% | 0.697 |
| **isopeptide_term** | **3.0** | 3.16 | 7.9% | 0.755 |
| disulfide (free) | 2.0 | 1.34 | 42.6% | 1.000 |
| isopeptide (free) | 0.0 | 0.02 | **99.2%** | 1.000 |

In 99.2% of complexes *some* interior pair is already at bond distance, so the free scan
reports "nothing needs to move." That is a statement about peptide geometry, and every one
of those pairs is unreachable.

**The section-4.4 conclusion survives, and is now sized.** Paired per complex against
head-to-tail -- the median of per-complex differences, not the difference of medians:

| chemistry | median dk | cheaper | equal | worse |
|---|---|---|---|---|
| disulfide_term | **-1.0** | 79.8% | 19.1% | 1.1% |
| isopeptide_term | **-2.0** | 95.6% | 3.9% | 0.5% |

A near-unanimous paired sign is much stronger evidence than a median comparison. But the
magnitude is **1-2 released residues**, not free closure: "bridged beats head-to-tail" is
confirmed on a reachable basis while the size of the win drops by roughly an order of
magnitude.

### Pinning the anchors re-opens the bridged ceiling question

Every `permissive_median` in this run is 1.000 -- for all five chemistries. So the bracket
is `[floor, 1.0]` everywhere and **only the floor discriminates**; the numbers above are
floor comparisons, not ceiling claims. The reachable bridged brackets ([0.70, 1.00] and
[0.76, 1.00]) are as wide as head-to-tail's [0.585, 1.00], so by this file's own
"where they are close, the chemistry is answered" rule, bridged closure is **no longer
answered**. The previous run's "bridged is answered (1.000 both ends)" held only because
the free scan's floor happened to saturate on unreachable pairs.

Known gap: the report's `k` tables still cover `mainchain` only, so the `k` figures above
were computed from `ceiling_rows.csv` rather than read out of `M15_CEILING_REPORT.md`.
Wiring `released_cost` over `REACHABLE_CHEMISTRIES` would make them reproducible from the
report itself.

## 5. Bridge windows, measured not asserted

From 4,000 CONECT-resolved CPSea linkages, [p1, p99] + 0.5 A:

| type | CA-CA | CB-CB | n |
|---|---|---|---|
| disulfide | 3.51 - 7.32 A | 2.79 - 5.10 A | 430 |
| isopeptide | 4.21 - 10.28 A | 3.79 - 7.88 A | 2,284 |
| mainchain | 2.48 - 4.43 A | 2.25 - 6.50 A | 1,286 |

## 6. Deliverable E -- oversupplied, not deferred

On CPSea_full (2,444,209 rows):

| | |
|---|---|
| domains | 1,552,353 |
| multi-segment domains | **557,613** |
| within-domain segment pairs | 1,440,794 |
| sequence-disjoint | 592,206 |
| disjoint and >=10 residues apart | 565,109 |
| survive the spatial gate | **98.8%** [98.6, 99.0] |
| projected competitor pairs | **~558,000** |

The CPSea_PDB figure was 2,395 multi-segment domains. CPSea_full has **233x** more.

The gate is real but almost never binds. The median pair shares **45 receptor residues**
and still has **zero** shared contacts, at 28.4 A interface-centroid separation and 19.2 A
minimum peptide-peptide distance. Within-domain segment pairs genuinely bind different
sites; only 1.2% fail on Jaccard and 0.02% on separation.

One honest limit: 15.1% of pairs share fewer than 3 receptor residues, so there is no
common frame and they pass on a Jaccard that is zero by construction. Those are pairs whose
receptor crops do not overlap at all -- i.e. the two peptides sit on different parts of the
protein -- so passing them is correct, not vacuous.

**The question flips from "is the arm viable" to "how do we subsample 558,000 pairs".**

## 7. Replica economics -- the brief is inverted, but it is cheap

19 complexes x 10 replicas, one core, reference protocol:

| | |
|---|---|
| context setup | **1.7 s** |
| marginal per replica | **30.0 s** |
| setup / marginal | **0.074x** (median of per-complex ratios) |
| per decoy at 8 / 30 / 100 | 30.2 / 30.1 / 30.1 s |

**Read that ratio as labelled.** 0.074 is `median(setup_i / marginal_i)` over the 19
complexes (`m15_replica_timing.py:134` computes the per-complex ratio, `:156` takes its
median). Dividing the two medians in the rows above gives 1.7 / 30.0 = **0.057**, which is a
different and equally valid statistic. Both are correct; they are not the same number, so
neither is a typo to be "corrected" into the other. Every conclusion here holds under both.

The run was **not** truncated: all 20 complexes were attempted, 19 returned and one failed
outright (see below). `partial: true` in `replica_timing.summary.json` only means
`n_complexes < n_complexes_requested`, which conflates "killed by the wall clock" with "a
complex raised". The report's hedge on this point is resolved -- it was a failure, not a kill.

The `ExampleContext` docstring's "~63 s of a ~65 s replica is setup" does **not** reproduce
on CPSea. Untested explanation: it was measured on full-chain LNR receptors, while CPSea's
are pocket-cropped to ~106 residues.

Consequences: setup amortises to nothing by 8 decoys, so "go wide on replicas because
context is expensive" has no basis -- but at 30 s a replica, 30 decoys per complex is only
~15 CPU-minutes, so going wide is affordable anyway. The recommendation survives; its
stated reason does not.

**Sizing warning.** The same measurement gave 81.5 s at n=2 and 42.2 s on the login node.
The *ratio* was stable across all three; the *absolute cost* was wrong by 1.4-2.7x. Read a
ratio off a small sample if you must, never a budget.

### The OpenMM NaN, and the policy before scaling

One complex dies inside OpenMM with `Particle coordinate is NaN`. The full id is
**`AF-A0A537RZW2-F1_0_130_140_relaxed_relaxed`** -- note the doubled `_relaxed` suffix, which
is the first thing the reproducer should check, since a twice-relaxed input may already carry
a degenerate geometry. This is unrelated to the timing numbers, but the settling step runs
the same minimizer on every replica of every decoy, so at decoy-build scale it stops being
one lost complex.

**It is contained, not silent.** `m15_replica_timing.py:198-204` catches per complex and
writes a `status: "failed"` row carrying the exception, which is why the failure is legible
at all and why the other 19 still produced a summary. The decoy build must keep that
property and tighten it:

**Two different minimizer phases run in the timing job, and they are not equally suspect.**
`time_one` first does 10 replicas of *hold* (r0 pinned at the native separation, which is the
decoy-settling force), and then -- because `time_ladder: true` in `m15.yaml` -- walks a
6-rung *ladder* that drags the termini from the native separation down to **1.33 A**. `main()`'s
`try/except` wraps both, so "the NaN came from the timing run" does not say which.

First result (job 62142, still running at the time of writing): **the hold phase is clean** --
target repeat 0 passed all 10 replicas with a finite-coordinate check after every minimize.
That points at the ladder, which is the far more aggressive deformation and the only one that
can crush a backbone that cannot reach. If that holds up, the consequence is specific:

- the **decoy build settles** (hold), so it would **not** be affected;
- the **closure projection** (`soft_closure_project.project_once`, which walks the ladder) is
  the path at risk.

- **Reproducer**: `scripts/m15_nan_repro.sbatch` isolates this one complex, runs the hold
  replicas *and* the ladder with a finite-coordinate check after every minimize, and reports
  the first rung at which a NaN appears, whether the input coordinates were already
  non-finite, and whether the failure is ladder-only. It runs clean controls alongside to
  separate "this complex" from "this code", and repeats the target to test determinism. Run it
  before scaling.
- **Skip-or-fix policy**: a NaN is a *skip at replica granularity*, never a job failure and
  never a silent drop. A replica that goes non-finite is discarded with a recorded reason; a
  complex whose replicas all go non-finite is written as a `failed` row and excluded from
  medians. Both counts must appear in the run summary, because a build that quietly loses 5%
  of its complexes to NaN reports a biased decoy distribution and nothing flags it.
  **Scope the policy to the phase the reproducer implicates** -- gating the decoy build on a
  ladder-only failure would cost complexes for a reason that does not apply to it.
- The guard belongs next to the minimize call, not in an outer `except`: catching
  `OpenMMException` after the fact cannot say which rung diverged. OpenMM also does not always
  raise -- it can return positions that are already NaN and fail on the *next* call -- so the
  check is an explicit `isfinite` on the returned coordinates, not a caught exception.

## 8. What this run does not establish

- **The head-to-tail ceiling.** See above. The instrument brackets it too loosely.
- **`ceiling_refined` independently.** The refinement and the validator are the same
  torsion solver, so the refined number is self-consistent, not independently confirmed.
  The validator's 0/113 agreement is measuring the *analytic* boundary, which is already
  known to over-license.
- **Anything about sterics.** Every bound here ignores excluded volume, Ramachandran and
  side-chain packing. Real ceilings are lower than `permissive` by an unmeasured amount.
- **Section 5's confounds.** Pocket-cropped PepBench and the staging-only control classifier
  are recorded but unbuilt -- Milestone 4, as instructed.
- **Length stratification is no longer on this list.** It was scheduled for Milestone 4, but
  section 3's conclusions turned out to depend on it, so it moved up and is built (section
  2b of the report). Deferring a stratification that a published conclusion rests on is how
  the confound got into section 3 in the first place.
- **Hostability on synthetic open states.** Section 2 sets ~2/3 isopeptide hostability as the
  acceptance criterion for the corruption. This run measures hostability on *real* linear
  binders only; nothing yet measures it on corrupted CPSea natives, and the pilot should not
  start until it does.

## 9. Re-running the report

Everything in section 2b reads files already on disk, so the corrected tables need no
recomputation of the ceiling scan -- only the report stage, which is CPU-only and takes
minutes. Pointing `--run-id` at the existing run regenerates `M15_CEILING_REPORT.md` and
`ceiling_rows.csv` in place:

```bash
bash scripts/submit_m15_ceiling_audit.sh --submit report --run-id m15_20260922_184436
```

The NaN reproducer is a separate CPU job, independent of the report:

```bash
bash scripts/submit_m15_ceiling_audit.sh --submit nan-repro --run-id m15_20260922_184436
```
