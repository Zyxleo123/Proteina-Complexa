# Milestone 1.5 -- results

Run `m15_20260922_184436`, 22 September 2026. Complete: 60 LNR + 600 PepBench ceilings,
4,000 calibration bridges, 20,000 spatial pairs, 19/20 timing complexes.

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

---

## 1. The question the milestone was set to answer

> "If a 13-mer at 30 A can retain at most ~0.5 of its contacts under head-to-tail closure,
> then the 0.37-0.42 retention the prior line was treated as a failure is near ceiling."

**For bridged chemistries: no, it is not near ceiling.** Isopeptide and disulfide ceilings
are 1.000 under both bounds -- a bridge can be placed without moving the peptide at all,
because some (i, j) pair is already at bond distance. Measured retention is ~0.5. That is a
real deficit of roughly half, not a geometric limit.

**For head-to-tail: undetermined, and it cannot be settled with this instrument.** The
bracket is [0.586, 1.000] against an achieved 0.500. The true ceiling could be 0.6 (in
which case 0.5 is near-optimal) or 1.0 (in which case half the interface is being thrown
away). Nothing in this run distinguishes those.

## 2. What that obliges

**Route the 20-45 A coverage band to bridged targets.** This is the change to section 3's
coverage target that Milestone 1.5 was gating. In that band the head-to-tail ceiling is
least constrained and the bridged ceiling is exactly 1.000 with a measured, quantified
deficit. Optimising head-to-tail retention there means optimising against an unknown scale.

**Stop quoting mainchain `frac_of_ceiling`.** 86.4% of mainchain complexes have a sampled
attempt above `ceiling_refined` (max 2.74x). That is the floor being a floor, not the
sampler beating physics.

**Decide whether to close the head-to-tail bracket at all.** It needs conformational
sampling under the closure constraint -- generate closure-satisfying conformers, measure
retention directly -- not a tighter analytic bound. Milestone-2-scale work, so it is a
scope call.

## 3. Ceiling against gap and length (head-to-tail, floor)

Both read off `ceiling_refined`, so both are floors.

| terminal gap | n | ceiling (floor) | max held window |
|---|---|---|---|
| <5 A | 4 | 0.97 | 9.5 |
| 5-10 | 35 | 0.85 | 10.0 |
| 10-15 | 83 | 0.59 | 4.0 |
| 15-20 | 162 | 0.58 | 5.0 |
| 20-25 | 224 | 0.53 | 6.0 |
| 25-30 | 93 | 0.61 | 7.0 |
| 30-45 | 57 | 0.60 | 8.0 |
| >45 | 2 | 0.56 | 8.5 |

The floor drops sharply to ~10 A and then **flattens**. The brief's intuition that the
ceiling keeps falling through 20-45 A is not what the data shows -- past 10 A the gap stops
mattering much.

| peptide length | n | ceiling (floor) |
|---|---|---|
| 5-8 | 215 | 0.48 |
| 9-11 | 271 | 0.58 |
| 12-14 | 120 | 0.72 |
| 15+ | 54 | 0.79 |

Length matters more than gap: a longer peptide has residues to spare for closure. Short
peptides are the constrained case, not long ones.

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
| setup / marginal | **0.074x** |
| per decoy at 8 / 30 / 100 | 30.2 / 30.1 / 30.1 s |

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

One complex dies inside OpenMM with `Particle coordinate is NaN`
(`AF-A0A537RZW2-F1_0_130_140`) -- unrelated to the timing, but it will bite the real decoy
build, which runs the same minimizer.

## 8. What this run does not establish

- **The head-to-tail ceiling.** See above. The instrument brackets it too loosely.
- **`ceiling_refined` independently.** The refinement and the validator are the same
  torsion solver, so the refined number is self-consistent, not independently confirmed.
  The validator's 0/113 agreement is measuring the *analytic* boundary, which is already
  known to over-license.
- **Anything about sterics.** Every bound here ignores excluded volume, Ramachandran and
  side-chain packing. Real ceilings are lower than `permissive` by an unmeasured amount.
- **Section 5's confounds.** Pocket-cropped PepBench, the staging-only control classifier
  and length stratification are recorded but unbuilt -- Milestone 4, as instructed.
