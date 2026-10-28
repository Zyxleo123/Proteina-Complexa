# Scaling the random-substitution search to PepBench — interim record

Started 2026-10-22. **Everything here was verified by execution.** Where a number comes
from an incomplete run it says so. Section 3 records four results that were reported and
then withdrawn, with the measurement that withdrew each one.

The question this run exists to answer: **does the LNR finding generalise?** On six LNR
targets, a peptide found by random substitution passed the structural validity checks
about twice as often as the unchanged cyclized peptide. PepBench is the test of whether
that holds on hundreds of targets.

**It does not answer that question yet, and the reason is in §3.** What it has produced
is a filter pass over 851 targets, and one fair comparison that comes out a tie.

---

## 0. Words used here

The code and the saved data use short names; this document spells them out, and so should
anything written from it.

| name in code/data | what it means |
|---|---|
| `stage="screen"` | fold each proposed peptide **once** with Boltz-2 and discard the ones whose structure fails the validity checks. A filter. |
| `stage="confirm"` | fold the survivors **many times** (the LNR pilot used 16) with different random seeds. The measurement. |
| `arm` | one method being compared. `random_control` = random substitutions, `guided_search` = score-guided edits, `backbone_mpnn` = generator backbone redesigned by ProteinMPNN. |
| `parent` | one receptor protein together with its original peptide. Called **target** below. |
| budget *B* | how many candidate peptides are tried per target. |
| `reference` | the original peptide, cyclized head-to-tail, unchanged. The baseline to beat. |
| seed | the random number that makes Boltz-2 return a different structure for the same sequence. |
| validity checks | bond length of the closing bond, planarity of that bond, closure angles, no severe atom overlaps, exact requested sequence, all-L stereochemistry. |
| dG, REU | Rosetta's **computed** interaction energy. Not a measured binding affinity, and nothing here converts it into one. |

---

## 1. What was run

| | |
|---|---|
| corpus | PepBench `train.txt`, 4,157 entries |
| sampled | **1,000**, seeded random (`train_subset1000_seed0.txt`, seed 0) |
| prepared | **851** (149 rejected: peptide outside the 5–16 residue window, or a missing chain) |
| method | random substitutions only, self-contained (§5) |
| budget | **8 candidates per target**, plus 1 fold of the unchanged reference |
| folds per candidate | **1** — this is the filter pass, and §3 is about what that costs |
| validity criteria | `v3-cis-amide-muco-windows` (§4) |
| storage | `rescorable` tier (§6) |

The sample is random rather than the first 1,000 lines because `train.txt` is grouped by
PDB entry — it opens with two chains of `4wjv` — so a head slice would have been
structurally clustered.

**Interim state at 2026-10-22 19:59: 250 of 851 targets have data, 6,481 folds, 1.1 GB,
zero technical failures.** The run is still going; every number below is from the
subset that had finished at the time of writing.

---

## 2. Choosing the budget, and what that choice does and does not rest on

The budget ladder was computed from folds already on disk for the PepBench *valid* split
(1,984 random candidates over 62 targets, no new compute). Because a random candidate is
drawn independently of every other, taking *B* of a target's finished candidates without
replacement is exactly what a budget-*B* run would have produced, so the statistics follow
from hypergeometric order statistics rather than a simulation
(`script_utils/cpseq_budget_ladder.py`).

| budget *B* | targets with ≥1 passing design | E[best dG] |
|---|---|---|
| 1 | 0.385 | −28.66 |
| 2 | 0.603 | −29.45 |
| 4 | 0.809 | −30.93 |
| **8** | **0.929** | −33.21 |
| 16 | 0.969 | −35.77 |
| 32 | 0.984 | −37.95 |

**B = 8 was chosen, and that choice stands.** It reaches 94 % of the achievable ceiling
(0.984) for a quarter of the cost of 32, and the marginal return per 1,000 further folds
collapses four-fold between B=8 and B=12. That knee is set by the per-fold pass rate and
the concavity of `1−(1−p)^B`; it does not depend on anything this document later
withdraws.

Prefix truncation was deliberately not used: the first 16 candidates of the realised
schedule are all single substitutions (1,102 of 1,102 measured), so a prefix below B=16
would have confounded budget with the number of substitutions.

### The energy column of that table is noise, and was withdrawn

Folding the same sequence with different seeds moves its computed energy by **sd 3.67
REU** (median over 25 sequences with ≥4 passing seeds, from the LNR 16-fold data).
Simulating a *single fixed sequence quality* with draws = quality + N(0, 3.67) and taking
the minimum of *B*:

| B | ladder Δ from B=1 | noise alone Δ from B=1 |
|---|---|---|
| 2 | −0.79 | −2.08 |
| 4 | −2.27 | −3.78 |
| 8 | −4.55 | −5.25 |
| 12 | −6.06 | −6.01 |
| 16 | −7.11 | −6.50 |
| 32 | −9.29 | −7.62 |

Noise alone reproduces the whole column. A minimum over more noisy folds is lower whether
or not the sequences differ, so that column carries no information about sequence quality
and must not be read as "energy improves with budget."

---

## 3. Results, and the four things withdrawn

Interim, 198 finished targets at the time these were computed.

### The one fair comparison is a tie

| | designs | unchanged reference |
|---|---|---|
| **per-fold pass rate** | **690/1,584 = 0.436** | **89/198 = 0.449** |

Both sides are a single Boltz-2 fold of one sequence, so the fold counts match and this is
an apples-to-apples comparison. **A random single or double substitution passes the
validity checks about as often as the native cyclized peptide.** Fisher p = 0.50.

This is the headline of the run so far, and it is a null result.

### Withdrawn 1 — "designs beat the reference on energy"

Best of 8 candidates against 1 fold of the reference: median **−3.56 REU** in favour of
the designs, 61/85 targets, sign test p = 6.0e-5, Wilcoxon p = 2.2e-7. All of which
measures the extra folds, not the sequences.

Matching the fold count — **one** design fold against **one** reference fold, repeated
using each of the 8 candidate slots in turn:

| | |
|---|---|
| pairs with both passing | 371 |
| design better | **158/371 = 0.426** |
| median difference | **+1.02 REU** (positive = design **worse**) |
| direction across the 8 slots | 8/8 point the same way |

**At matched fold counts the sign flips: the designs are slightly worse.** The −3.56 REU
advantage was entirely the effect of taking a minimum over 8 noisy folds.

### Withdrawn 2 — "94 % of targets get a valid design" as evidence the search works

186/198 = 0.939 of targets had at least one of their 8 candidates pass. That number is
real but it is a property of the **budget**, not of the search:

| | per-fold pass rate | fraction of targets with ≥1 pass in 8 folds, if folds were independent |
|---|---|---|
| designs | 0.436 | 0.990 |
| reference | 0.449 | **0.992** |

Eight folds at a 44 % pass rate almost always yield one pass whatever the sequences are.
The reference, given the same eight folds, would reach about 0.99 — **above** the
designs' observed 0.939. The observed value sits below the independence prediction
because some targets are intrinsically hard (8 of 198 produced nothing from any
candidate).

An earlier version of this comparison claimed the 101-vs-4 asymmetry in "only the design
passed" versus "only the reference passed" was too large to be a fold-count effect. That
claim was arithmetically wrong and is retracted.

### Withdrawn 3 — best-of-8 energy as a description of output

Per-target best of 8: median **−32.03 REU** (quartiles −39.33 / −24.49, best −69.55).
This is a minimum over 8 folds each carrying sd 3.67 REU of noise, so it is biased low by
roughly 5 REU on the §2 simulation. It describes what the filter emits, not what the
designs are worth.

### Withdrawn 4 — the per-target win rate

"Best design beats its own reference in 71 % of targets" is best-of-8 against best-of-1
and is not evidence of anything.

### Why none of this is a bug

The code computes what it claims to. Folding each candidate once is a deliberate choice,
stated in `evaluate.Protocol`: *"One structure per screening prediction, by design: the
pilot buys breadth over samples, and the confirmation stage is where repeats are spent."*
The filter is working as designed. The error was in reporting filter output as a result —
the same error the LNR pilot had already made and corrected when it retired `best_valid_dG`
as "a biased statistic."

### What it takes to answer the original question

Fold the chosen design **and** the unchanged reference the same number of times, and
report pass frequency and median energy over those folds. `stage="confirm"` already does
exactly this and always includes the reference. At 16 folds per sequence, 2 sequences per
target, 851 targets ≈ 11,900 folds ≈ 15 h at the concurrency available. It must be chained
after the filter pass, not run alongside it: two processes writing one target's journal
would corrupt it.

Sizing: the standard error on a pass frequency is `√(p(1−p)/N)`, which at p≈0.5 is **0.18
at 8 folds** — too coarse to resolve a 0.2 difference. At 16 folds the standard error on
the median energy is ≈1.4 REU, enough for a 3–4 REU effect. 16 is the right number, and
is what the pilot used.

---

## 4. The validity criteria changed, and the previous ones were miscalibrated

`SCORING_PROTOCOL_VERSION` moved from `v2-closure-constrained` to
`v3-cis-amide-muco-windows`. Structures are untouched; only the labels change, which is
what the separate structure and score caches exist for.

### An amide bond has two planar arrangements and the old criteria allowed one

The previous planarity tolerance was ±8.44° **from trans only**, derived as "worst of 4
experimental controls + 5 % margin." The derivation procedure was sound; the control set
was not representative. All four crystal controls happen to be trans (closing torsions
177.0, 173.0, 172.0, 179.2°), so that procedure could only ever produce a trans-only
window.

Measured on 150 cluster-unique native CPSea head-to-tail macrocycles — the corpus these
models are trained on:

| | old: trans within 8.44° | new: cis **or** trans within 30° |
|---|---|---|
| native CPSea passing | **27/150 = 0.180** | **117/150 = 0.780** |
| crystal controls passing | 4/4 mainchain | **6/6** |

**75 of those 150 natives sit nearer cis than trans**, median deviation from trans 90.3°.
The old criteria rejected 82 % of the training corpus's own closures. `chemistry.AtomBond`
now declares which arrangements each chemistry permits and the deviation is measured
against the nearest, with the match recorded per structure.

A bug was introduced and caught in the same session: the 30° planarity rule was initially
applied to disulfides, which are not planar — their CB–SG–SG–CB torsion sits near ±90°
with a broad spread. It failed `1SLD`, a deposited crystal structure, at 52.8°. Planarity
now applies only to amide chemistries; disulfides keep the control-derived tolerance.

### Closure-bond windows

Adopted MuCO's published windows: amide **1.23–1.43 Å**, disulfide **1.95–2.15 Å**. These
are tighter than the previous 1.1–1.6 Å. On native CPSea they cost nothing (150/150 pass
either way). On the controls they reject one: **7K2F at 1.554 Å**, which is retained as a
known out-of-window reference — a genuine deposited structure with an unusually long
closing bond, and the same control MuCO's own ±0.10 Å window rejects.

### The acceptance audit

`script_utils/cpseq_gate_audit.py` applies the criteria to the deposited crystal
macrocycles and to native CPSea, and requires every failure to be explained rather than
absorbed into a tolerance. Two failures, both named:

- **7K2F**, above.
- **33/150 native CPSea exceed 30° planarity** (deviations 30.1–86.0°, median 36.4).
  Classified as **corpus coordinate defects, not unusual chemistry**: CPSea builds and
  minimises its closing bond computationally and never enforced amide planarity, whereas
  all four deposited crystals are planar within 8.04°. Admitting them would require ~90°,
  i.e. no planarity requirement at all. **The bar was not moved for them.**

### Still conservative, so v3 pass rates are a lower bound

**Ordered wrongly here; see §11 for the measurement.** On 11,953 scored folds the
closure-angle threshold below is the sole cause of rejection for 43 of them (+0.003 pass
rate if fixed), while the atom-overlap criterion is the sole cause for 3,457. The overlap
criterion is the one to fix first, and the native-CPSea rejection counts quoted below are
not a guide to how much yield a criterion costs on designed structures.

- **Closure angles** remain worst-control-derived at 7.54°, which rejects **66/150**
  native CPSea. The adopted requirements call for five reference standard deviations
  instead, which needs a σ library — `BACKBONE_REFERENCE` holds means only.
- **Atom overlap is the one criterion with no calibration provenance at all.** Every other
  criterion carries a source string; `MAX_INTER_CLASHES = 0` is a bare module constant.
  Applied to the four mainchain crystal controls that calibrated the others it **rejects
  2 of 4**: `1SFI` on a native internal disulfide (`CYS2.SG–CYS10.SG`, 2.05 Å — the
  overlap test has no concept of a crosslink) and `7K2F` on a hydrogen bond
  (`GLU6.OE2–receptor.OG`, 2.178 Å — the 2.2 Å cutoff's own rationale is carbon van der
  Waals radii, but it is applied element-blind).

  Classifying the offending atom pairs in this corpus: **57.7 % are polar O/N–O/N**
  (hydrogen-bond candidates), 38.3 % involve carbon (genuine interpenetration), 4.0 % are
  disulfides. **54.2 % of overlap-only rejections contain no carbon contact at all.**
  Replacing this with MolProbity/Probe contact analysis is blocked: `cctbx-base` installs
  and imports, but `reduce2` needs the CCP4 monomer library (6.6 GB) *and* `rdkit`.

Both of these only loosen when fixed, so v3 numbers cannot fall.

---

## 5. The random-substitution method no longer needs the guided one

`search.match_counts` generated the random candidates only *after* the guided method
declared how many substitutions its own candidates had, and reproduced that multiset. That
is the right design for a matched comparison and fatal for a random-only run: passing
`--arms random_control` alone skipped the method entirely.

`search.standalone_counts` replaces the dependency with a frozen distribution — the
measured marginal of the realised random candidates (59.2 % single, 19.5 % double, 12.1 %
triple, 5.9 % quadruple, remainder ≤2 %), truncated to each target's `floor(L/2)` ceiling
and renormalised. Reproducing the measured marginal rather than inventing one is what
lets the §2 budget ladder transfer.

The data says the choice costs nothing measurable: over this corpus the pass rate does not
depend detectably on the number of substitutions across the well-sampled range 1–4
(381/382/429/312 per mille on 1,319/411/266/128 candidates; χ² = 6.52 on 3 df, **p =
0.089**), and neither does energy among passing candidates (Spearman ρ = −0.020, **p =
0.58**, n = 658).

It also removes a failure mode. The plan is a pure function of `(target, budget)` and
reads neither the archive nor the journal, so a requeued task reproduces the same
assignment and skips work already paid for. The old guided-dependent path grew its budget
on each requeue — `lnr_5njx` went 32 → 36 → 45.

What the method does, exactly: choose *count* positions uniformly without replacement, and
at each position choose a residue uniformly from 19 canonical types **minus the native one
at that position**, so the realised count is exactly *count* and cannot be deflated by
redrawing the native residue (verified over 400 candidates: zero occurrences). Cysteine is
excluded because a free cysteine introduced at random can pair with another and close a
ring the molecule's own label does not claim. There is **no** position weighting and **no**
residue weighting — interface positions are as likely as solvent-exposed ones, and
tryptophan is as likely as alanine. That is what makes it a fair baseline.

---

## 6. Storage was a 263× overhead, not a property of the procedure

The essential output per candidate — peptide coordinates plus the score row — measures
**5.0 KB**. The pipeline stored **1,310 KB**. Measured over 150 directories:

| component | MB/dir | share | what it is |
|---|---|---|---|
| processed MSA | 0.423 | 32.3 % | an **input**, per-receptor, identical for every candidate |
| pae/pde arrays | 0.380 | 29.0 % | matrices whose summary numbers are already on the score row |
| refined complex PDB | 0.298 | 22.8 % | 95.6 % receptor |
| raw complex PDB | 0.135 | 10.3 % | 95.6 % receptor |
| template | 0.039 | 3.0 % | an **input**, per-receptor |
| the result | 0.017 | 1.3 % | — |

A refined 16-mer complex on a 316-residue receptor is 480 KB, of which the peptide — the
only part that differs between a target's candidates — is 19.2 KB. The receptor looked
variable (12.8 Å median pairwise backbone RMSD) but is **0.90 Å after superposition**:
Boltz-2 emits each complex in its own arbitrary global frame, so the bulk of every file is
one receptor re-stored in a new pose.

`cpseq/retention.py` adds three tiers, measured on a real directory (2,186 KB):

| tier | result | can re-score without re-folding? |
|---|---|---|
| `full` (default) | 2,186 KB | yes |
| **`rescorable`** (used here) | **51.6 KB (2.4 %)** | **yes** — raw complex gzipped, decompressed on demand |
| `deliverable` | 4.4 KB (0.2 %) | no |

Every tier also writes the deliverable the old tree never held in usable form: each
target's peptides expressed in **one** receptor frame (`designs/<target>/receptor.pdb.gz`
once, plus a ~5 KB array per candidate), so a target's designs are directly comparable
without re-superimposing every complex. The receptor fit is recorded per candidate rather
than assumed (observed 0.42–0.47 Å).

Default is `full` and the design records are off by default, because a Slurm array with
queued tasks picks up an edit its running tasks never see, and anything unconditional here
would have made the corpus half-and-half mid-run.

**Measured effect: 1.1 GB for 6,481 folds, against ~8.5 GB unpruned.**

---

## 7. Operational failures, each of which cost real time

**A background process does not survive Slurm teardown.** The launcher that was to wait
for target preparation and then submit the work was started with `setsid nohup`, which
detaches it from the terminal but **not from the job's cgroup**. The interactive
allocation ended, the cgroup was torn down, and the launcher was killed mid-wait.
Preparation had finished correctly at 01:43; nothing was alive to act on it, and **~11 h
were lost**. Work that must outlive a session has to be a Slurm job or a
`--dependency=afterok` chain, not a shell process.

**The queue limits are 24 concurrent GPUs and 100 submitted jobs**
(`preempt_qos`: `MaxJobsPU=24`, `MaxSubmitJobsPU=100`). A 120-task array was rejected
outright with `QOSMaxSubmitJobPerUserLimit`. Every throughput estimate made before
checking this was wrong by up to 5×.

**The tail, not the mean, sets the time limit.** Measured over 4,942 folds: median 63 s
per fold, mean 75.5 s, **max 2,048 s**; per target median 10.0 min, mean 12.4, p90 19.9,
**max 130.9 min**. A 9-target task averages ~112 min, so a 4 h limit looks generous — but
one 131-minute target exhausts it alone, and **16 tasks hit the wall**. Worse, each task's
target list is fixed (`cohort[i::n_shards]`), so targets a timed-out task never reached
are orphaned: **58 of them**. Resubmitting recovers them at no loss, because the journal
records every finished fold and a resubmitted task skips completed targets and resumes a
partial one where it stopped. The limit is now 8 h.

Slurm will not let a user *raise* `TimeLimit` on already-queued tasks, so the fix was
cancel and resubmit rather than `scontrol update`.

---

## 8. Commands

```bash
cd /home/yixiz/Proteina-Complexa && source env.sh
ZFS=/data/user_data/yixiz

# 1. prepare targets (templates + rate-limited, checkpointed MSA fetches)
python script_utils/cpseq_prepare_targets.py \
  --out-dir $ZFS/cpseq/pepbench_train \
  --lnr-root $ZFS/LPData/PepBench/train_valid --list-file train_subset1000_seed0.txt \
  --source PepBench --msa-delay-s 1.0 \
  --boltz-python $ZFS/envs/boltz2/.venv/bin/python

# 2. freeze the validity criteria, then audit them on the native references
python script_utils/cpseq_gate_audit.py --n-cpsea 150 \
  --out $ZFS/cpseq/pepbench_train/manifest/gate_audit_v3.json

# 3. every prepared target is the cohort; nothing to stratify against
python script_utils/cpseq_search.py select-cohort \
  --dataset-dir $ZFS/cpseq/pepbench_train \
  --out-dir $ZFS/cpseq/pepbench_train_search --all-targets

# 4. filter pass -- one fold per candidate. Preflight ONE target as a job first:
#    processes adopted into an interactive allocation's extern step get a memory
#    ceiling below Boltz's ~11 GiB working set, and the failure is a bare SIGKILL.
sbatch --job-name=pbt_preflight --time=1:00:00 scripts/cpseq_search.sbatch \
  MODE=screen OUT_DIR=$ZFS/cpseq/pepbench_train_search PARENTS=<one> LIMIT=3 \
  SHARD_COUNT=1 RETENTION=rescorable ARMS=random_control RANDOM_POLICY=standalone

sbatch --dependency=afterok:<preflight> --array=0-79%16 --time=8:00:00 \
  scripts/cpseq_search.sbatch MODE=screen \
  OUT_DIR=$ZFS/cpseq/pepbench_train_search SHARD_COUNT=80 \
  RETENTION=rescorable ARMS=random_control RANDOM_POLICY=standalone

# 5. interim report -- safe to run against a live run
python script_utils/cpseq_scaleup_report.py \
  --out-dir $ZFS/cpseq/pepbench_train_search --budget 8

# 6. NOT YET RUN -- the measurement. Folds design and reference the same 16 times.
sbatch --dependency=afterok:<filter_array> --array=0-79%16 --time=8:00:00 \
  scripts/cpseq_search.sbatch MODE=confirm \
  OUT_DIR=$ZFS/cpseq/pepbench_train_search SHARD_COUNT=80 RETENTION=rescorable \
  SEEDS=101,202,303,404,505,606,707,808,909,111,222,333,444,555,666,777

# the budget ladder, from folds already on disk (no new compute)
python script_utils/cpseq_budget_ladder.py --out-dir $ZFS/cpseq/pepbench_search
```

---

## 9. The LNR result this run is testing, restated honestly

Recomputed from the 480 LNR confirmation folds (6 targets, 30 sequences, 16 folds each,
under the old `v2` criteria):

| target | reference | best design | Δ median | wins on both? |
|---|---|---|---|---|
| lnr_1elr | −17.94 @ 2/16 | −21.83 @ 12/16 | −3.89 | yes |
| lnr_2v8x | −22.17 @ 2/16 | −31.15 @ 12/16 | −8.98 | yes |
| lnr_5njx | −29.03 @ 6/16 | −30.67 @ 6/16 | −1.64 | **no** (pass rate ties) |
| lnr_5y59 | −24.76 @ 4/16 | −32.08 @ 16/16 | −7.32 | yes |
| lnr_6a30 | −25.27 @ 9/16 | −29.01 @ 13/16 | −3.74 | yes |
| lnr_6seo | −25.13 @ 3/16 | −30.90 @ 7/16 | −5.77 | yes |

**5 of 6 targets beat the reference on median energy and pass frequency simultaneously.**
The pilot README says "every parent improves on both axes"; on a strict greater-than test
for pass frequency it is 5/6, not 6/6, because `lnr_5njx` ties at 6/16.

Pooled: random substitutions 92/160 = 0.575 pass rate at median −28.24 REU; reference
26/96 = 0.271 at −25.14.

**The reported Fisher p = 2.6e-6 is pseudo-replicated.** Those 160 folds come from only
10 sequences across 6 targets, and the 96 reference folds from 6 sequences; 16 folds of one
sequence are repeated measures, not independent trials. Taking the target as the unit of
analysis, the pass-rate differences are +0.625, +0.625, 0.000, +0.750, +0.250, +0.250 —
**5 wins, 1 tie, 0 losses, Wilcoxon one-sided p = 0.031**, sign test p = 0.109. With n = 6
the smallest p Wilcoxon can produce is 0.031, so the result sits exactly at the floor of
what six targets can detect.

So the prior going into PepBench is **a large, consistently-signed advantage in pass
frequency, measured on six targets, significant only at the limit of that sample size** —
which is the reason to scale it, and the reason it has to be scaled with folds repeated on
both sides.

---

## 10. The re-run: a ladder from few draws to 64, every draw folded ten times

Launched 2026-10-22, in `$ZFS/cpseq/seed_ladder`. This is §3's "what it takes to answer
the original question", built as a ladder so that the draw budget is a measured axis
rather than a number chosen in advance.

**What changes.** Nothing is promoted and nothing is filtered. Every draw is folded at
the same ten seeds, and so is the unchanged cyclized peptide, so every comparison in the
report has the same fold count on both sides. A draw's result is its pass rate over ten
folds and its median energy over the folds that passed — never one fold, and never a
minimum over folds, which is what produced all four withdrawn results in §3.

**The size.** 20 complexes — 10 LNR and 10 PepBench — and 64 draws each. The cohort is
deliberately small because the fold count is what was missing, not the target count: 20
complexes × 65 sequences × 10 folds is 13,000 folds, about the same spend as 851
complexes at one fold, and it buys an error bar on every number instead of none.

| | |
|---|---|
| complexes | 10 LNR (of 12 prepared) + 10 PepBench valid (of 80) |
| selection | uniform without replacement, seeded on the corpus label and recorded |
| draws per complex | 64, plus the unchanged peptide |
| folds per sequence | 10 (seeds 101…111, the first ten of the pilot's sixteen) |
| folds in total | 13,000 ≈ 271 GPU-h at the measured 75 s/fold |
| storage | ~650 MB at `rescorable` |

**Why the ladder is free.** The rungs B = 1, 2, 4, 8, 12, 16, 24, 32, 48, 64 are nested
prefixes of one draw stream: draw *k* is the same sequence whatever the total is, so the
first 8 draws of the 64 are exactly what a B=8 run would have produced. Every rung is
read by truncating one run, and no rung costs its own folds. The superseded ladder could
not do this — it count-matched the guided arm, whose round 0 was a single-substitution
scan, so 1,102 of 1,102 first draws were 1-mutant and a short prefix confounded the
budget with the number of substitutions. §2 had to subsample around that; this one does
not.

**Why the folds are paid for in rung order.** All ten folds of draws 1…B are finished
before any fold of draw B+1. A run cut off anywhere therefore holds a *complete* ladder
up to the last finished rung — every draw in it folded ten times — and an interruption
costs rungs, never depth. The alternative order, folding every draw once before any draw
twice, was written first and rejected: it holds all 64 draws at an equal but shallow
fold count, which reproduces the single-fold filter this re-run exists to correct, at
every rung at once. Depth is what makes a rung mean anything, so depth is what the order
protects. Both orders are checked in `test_cpseq_search.py`.

**What the report refuses to do.** `cpseq_ladder_report.py` reports coverage at a
reliability bar (a draw counts when it clears the checks in ≥ 5 of 10 folds) *and*
beside it the old one-fold definition, so the gap between them is visible rather than
argued. Energies are medians over passing folds. The fold count is printed on both sides
of every comparison.

### Commands

```bash
ZFS=/data/user_data/yixiz
# 1. the cohort: two corpora in one run, each complex keeping the corpus it came from.
#    Refuses to build if the two corpora's validity criteria differ.
python script_utils/cpseq_search.py ladder-cohort \
  --out-dir $ZFS/cpseq/seed_ladder \
  --corpus "dataset_dir=$ZFS/cpseq/tranche1,n=10,label=LNR" \
  --corpus "dataset_dir=$ZFS/cpseq/pepbench_valid,n=10,label=PepBench" \
  --draws 64 --seed-depth 10

# 2. preflight ONE complex as a job -- never in an interactive allocation
sbatch --job-name=ladder_pre --time=1:00:00 scripts/cpseq_ladder.sbatch \
  OUT_DIR=$ZFS/cpseq/seed_ladder PARENTS=lnr_1elr LIMIT=3 SHARD_COUNT=1

# 3. the run: one array task per complex. Resubmitting the same line resumes;
#    finished folds are skipped on the journal key (sequence, seed, stage, arm).
sbatch --array=0-19%20 --job-name=ladder scripts/cpseq_ladder.sbatch \
  OUT_DIR=$ZFS/cpseq/seed_ladder SHARD_COUNT=20

# 4. the ladder, from folds on disk; safe to run mid-flight, no new compute
python script_utils/cpseq_ladder_report.py --out-dir $ZFS/cpseq/seed_ladder --by-corpus
# the same run read at fewer folds, which is what every superseded run saw:
python script_utils/cpseq_ladder_report.py --out-dir $ZFS/cpseq/seed_ladder --depth 1
```

**To extend past 64 draws**, re-run step 3 with a cohort rebuilt at a larger `--draws`.
The first 64 draws are unchanged and are not re-folded.

### Interim, 2026-10-23: 11,262 of 13,000 folds

All 20 complexes have the unchanged peptide folded 10 times; candidates complete from
the start of the stream: min 32, median 56, max 64. Rungs up to **B=32 hold all 20
complexes**. 72 folds were lost to a GPU fault on one complex and are being re-run; see
"A crashed fold was being retired, not retried" below.

**More candidates buy luckier folds, not better peptides.** This is the clearest thing
the ladder has produced, and it only shows up because both columns exist:

| B | cplx | best of B | w/l | sign p | cross-fitted | w/l | sign p |
|---|---|---|---|---|---|---|---|
| 1 | 20 | −1.71 | 4/2 | 0.688 | +0.19 | 3/4 | 1.000 |
| 8 | 20 | −4.66 | 12/1 | 0.003 | −0.06 | 6/6 | 1.000 |
| 16 | 20 | −4.54 | 13/1 | 0.002 | −3.86 | 7/3 | 0.344 |
| 24 | 20 | −5.00 | 13/2 | 0.007 | −3.88 | 9/2 | 0.065 |
| 32 | 20 | −4.99 | 13/3 | 0.021 | −3.41 | 9/2 | 0.065 |
| 48 | 14 | −8.62 | 12/0 | 0.0004 | −1.20 | 6/3 | 0.508 |
| 64 | 10 | **−11.40** | **9/0** | **0.004** | −2.07 | 4/2 | 0.688 |

The left column improves monotonically with the number of candidates and its p-values get
*more* impressive as it goes — 9 complexes to 0 at B=64. It is the maximum of B noisy
estimates, so it must grow like that whether or not any candidate is better than another.
The right column, which chooses on five folds and measures on the other five, **does not
move with B at all**: it sits between −2 and −4 REU from B=16 upward and never reaches
significance.

So "best of 64 beats the unchanged peptide by 11.4 REU, 9 complexes to 0, p = 0.004" is
available from this data, would have been the headline, and is an artifact. The budget
question §2 asked — how many candidates to pay for — has the answer **beyond roughly 16,
more candidates add nothing measurable**, because everything the ladder gains above that
rung is selection noise.

**The pass rate is a tie, and that now rests on ten folds a side.** At every rung where
all 20 complexes are present, the per-complex paired comparison is flat: B=8 gives 12
wins / 8 losses (median +0.125, sign p = 0.50), B=16 gives 12/8 (+0.100, p = 0.50).
Pooled per-fold rates are 0.362 for candidates against 0.310 for the unchanged peptide.
So §3's null result survives the fold-count fix — it was not an artifact of folding once.

**A fifth withdrawal, caught by design rather than after the fact.** Both columns above
fold every sequence ten times and compare per-complex medians, so neither is the
fold-count error of §3. The difference is **selection**: the left column picks the best of
B candidates and reports that same candidate's energy, so a candidate wins partly because
its noise was favourable and that noise is then the number reported. The right column
picks on folds 1–5 and measures on folds 6–10, with the peptide also measured on folds
6–10, so selection and measurement share no fold. Standard sample splitting; the bias it
removes is the winner's curse.

**At B=8 the entire −4.66 REU advantage disappears under cross-fitting** (−0.06 REU, 6
wins / 6 losses). Matching the fold count per sequence is necessary and *not sufficient*:
a maximum over B noisy estimates is biased low even when every estimate has the same
number of folds behind it. **Any future "best of B" number in this project needs the
cross-fitted column beside it**, and the pilot's and scale-up's remaining best-of-N
figures should be assumed to carry this bias until re-measured this way.

### A crashed fold was being retired, not retried

`pepbench_C_D_pdb4wjq` lost its GPU partway through its last rung ("CUDA unknown error —
setting the available devices to be zero") and recorded **72 folds with
`technical_failure: true`**, draws 48–63 at fold ranks 5–9, each failing in ~24 s.

The failure itself is a transient node fault: 72 of 11,262 folds, one task, one node, and
nothing else in the run has failed. What it exposed is a real flaw in the resume logic.
`Journal` keyed every recorded row as paid for, so a crashed fold satisfied the resume
check and **no requeue would ever have gone back for it**. The complex read as 650/650
rows with 578 real folds — complete on the progress line, quietly missing 11 % of its
folds in the data.

Fixed: a row with `technical_failure: true` no longer enters the resume index, so a
requeue re-runs it; the retry is appended and every reader already skips the failed row.
`test_cpseq_search.py` reproduces the exact case. This affects `screen` and `confirm`
too, which share the journal — both had zero technical failures on their runs, so no
earlier result changes.

**What is left open.** The best-powered honest estimates are B=24 and B=32, both on all 20
complexes: **−3.88 and −3.41 REU, 9 wins / 2 losses, p = 0.065 each.** Suggestive, not
established, and the direction is consistent across LNR and PepBench. Resolving it needs
**more folds per sequence, not more candidates** — cross-fitting spends half the folds on
selection, so at depth 10 each side rests on 5 folds and the sign test has little power.
Depth 20 would give 10 a side, and the ladder already says the candidate axis is spent
above B≈16, so the next compute belongs on the fold axis.

Two further cautions on the tables above:

- **Rungs above B=32 are still a speed-biased subset** (14 and 10 complexes). The
  complexes that get there first are the small ones — `lnr_1elr` is a 5-mer — because the
  run finishes rungs in order and small complexes fold faster. Read B≤32 only until the
  run completes.
- **Only 13–14 of 20 complexes contribute to the energy comparison at all**, because a
  complex needs a candidate clearing the reliability bar *and* a measurable peptide median.
  The remaining 6–7 are not failures of the search; they are complexes where nothing on
  either side passed often enough to compare, and reporting the energy over the subset
  without saying so would describe a filtered population.

---

## 11. Costing a corpus-building run over the whole corpus

The goal here is different from §10's: ship one cyclic peptide per target, for every
target, rather than estimate the method's average advantage. A random sample *is* the
estimate, so for §10's question more targets past a couple of hundred buy little. For a
corpus every target is a deliverable, so the per-target cost is what decides feasibility.

**The old estimate was wrong by 10–20x because it assumed one fold per candidate.**
§11's table said the full train split cost ~2,900 GPU-h; that was 32 candidates at ONE
fold each, which §3 showed is a filter and not a measurement. Folded properly:

| design | folds/target | 851 targets | 4,157 targets |
|---|---|---|---|
| B=32, D=1 (the old plan) | 33 | 585 h | 2,858 h |
| B=16, D=10 | 170 | 3,014 h | 14,723 h |
| B=16, D=20 | 340 | 6,028 h | 29,445 h |

At the 24-GPU QOS ceiling (576 GPU-h/day, assuming no preemption and nothing else of
yours running), the full split at a trustworthy depth is **25–51 days**. Storage is no
longer the constraint: at the measured 228 KB/fold the full split is 154–307 GB against
701 GB free.

### The selection rule, measured rather than assumed

Replaying the 20 complexes' folds under different rules, each scored only on folds that
took no part in choosing (300 fold-order replicates per complex):

| rule | folds/target | ships | dG median | vs best-of-16 | sign p | 4,157 targets |
|---|---|---|---|---|---|---|
| first acceptable | 22 | 0.91 | −27.25 | +2.95 | 0.003 | 1,892 h |
| best of 4 | 40 | 0.84 | −29.41 | +1.95 | 0.008 | 3,464 h |
| best of 8 | 80 | 0.86 | −29.96 | +0.00 | 1.000 | 6,928 h |
| best of 16 | 160 | 0.82 | −31.38 | — | — | 13,857 h |
| best of 32 | 320 | 0.79 | −30.75 | +0.00 | 1.000 | 27,713 h |
| **adaptive 16→4** | **76** | **0.89** | −28.99 | −0.08 | 1.000 | **6,560 h** |

**Selection helps, and it saturates at about 8 candidates.** Best-of-8 is
indistinguishable from best-of-16 (9/8, p = 1.00) at half the folds, and best-of-32 adds
nothing over best-of-16. This does not contradict §10's flat cross-fitted ladder; it
sharpens it. §10 compared best-of-B against the *unchanged peptide* and found no gain
from B=8 to B=64. Here the comparison is between selection rules, and going from
essentially no selection to B≈8 is worth **~3 REU** (first acceptable loses by +2.95, 17
complexes to 3, p = 0.003). Both statements hold: selection is worth paying for once, and
worth nothing after that.

**Adaptive allocation buys best-of-16 quality for 76 folds instead of 160.** Probe 16
candidates at 3 folds each, keep the 4 that pass most, fold those to full depth, rank on
one half and score on the other. Uniform ranking spends most of its budget folding
candidates ten times that failed their first fold.

**The cheap rules ship more targets** (0.89–0.91 against 0.82 for uniform best-of-16),
because a candidate selected for having passed is more likely to be measurable at all.
For a corpus, coverage is itself a deliverable.

### What to fix before committing weeks of GPU

**The pass rate is the cost driver, so it was worth measuring what actually rejects.**
Two thirds of every fold is discarded, so ~3 folds are needed just to learn whether a
candidate passes. Across the 11,953 scored folds of the §10 run, by sole cause:

| failure category | appears | sole cause | pass rate if removed entirely |
|---|---|---|---|
| **peptide–receptor overlap** | 5,277 | **3,457** | **+0.289** |
| intra-peptide overlap | 2,260 | 999 | +0.084 |
| closure torsion | 2,216 | 771 | +0.065 |
| closure angle_i | 545 | 38 | +0.003 |
| closure angle_j | 292 | 5 | +0.000 |
| closing bond length | 147 | 0 | 0.000 |

**This reverses the priority §4 states.** That section lists the closure-angle threshold
first among the remaining conservative criteria, on the strength of it rejecting 66/150
native CPSea. In this run it is the sole cause of rejection for **43 folds**: fixing it is
worth **+0.003**. The atom-overlap criterion — §4's second item, and the one with no
calibration provenance at all — is the sole cause for 3,457.

**But most overlap rejections are real overlaps, so the recoverable yield is modest.**
Closest-contact distribution of those 3,457 folds, median **1.94 Å**:

| closest contact | folds | pass rate if admitted |
|---|---|---|
| 2.10–2.20 Å (hydrogen-bond plausible) | 741 | +0.062 |
| 2.00–2.10 Å (hydrogen-bond plausible) | 625 | +0.052 |
| 1.80–2.00 Å (short) | 1,003 | +0.084 |
| **< 1.80 Å (severe)** | **1,082** | +0.091 |

An element-aware cutoff — 2.0 Å for polar (N/O–N/O) and sulfur pairs, 2.2 Å unchanged
where carbon is involved — is therefore worth **at most +0.115** (0.372 → 0.486), and
less once the contacts must actually be carbon-free. It is still worth doing, because the
criterion is uncalibrated and rejects 2 of the 4 crystal controls that calibrated every
other criterion, and because it costs **CPU only**: the structures are on disk and the
two-cache split means a re-score never re-predicts. But it buys roughly 15–20 % off the
corpus run, not half.

**The remaining 60 % is not a threshold problem either.** 2,085 folds — at least 17 % of
all folds — place the peptide into the receptor with a heavy-atom contact under 2.0 Å,
and 1,082 under 1.8 Å. No choice of cutoff admits those. Two caveats on reading that as a
model failure: this run has `use_potentials: false`, so Boltz-2's own physical-plausibility
steering was never applied (a paired comparison is running in
`$ZFS/cpseq/potentials_probe`), and the figure counts only folds rejected *solely* on
overlap, so it is a lower bound rather than the full rate.

### Relaxation resolves 98 % of those overlaps, which is more interesting than it sounds

The clash check reads the RAW prediction, before refinement, by design. Re-relaxing a
stratified sample of 160 overlap-only rejections straight from the retained raw complexes
(`cpseq_refined_clash_probe.py`; no prediction re-run):

| raw closest contact | n | overlap gone after relax | refined closest | receptor deformed | peptide vs receptor |
|---|---|---|---|---|---|
| 2.10–2.20 Å | 40 | **40/40** | 2.68 Å | 0.64 Å | 0.99 Å |
| 2.00–2.10 Å | 40 | 39/40 | 2.64 Å | 0.72 Å | 0.99 Å |
| 1.80–2.00 Å | 40 | 38/40 | 2.65 Å | 0.66 Å | 1.00 Å |
| below 1.80 Å | 40 | **40/40** | 2.69 Å | 0.68 Å | 1.04 Å |

Receptor figures are internal deformation with rigid-body drift removed — a plain
before/after RMSD read 1.2–3.5 Å and was mostly the complex translating, which is
bookkeeping, not the relax rebuilding the target.

**Applying the check to the refined structure instead would raise the pass rate from
0.372 to ~0.66**, recovering ~29 % of folds against ~11 % for the element-aware threshold
fix, and needing no recalibration at all.

**The uniformity is the catch, and it argues against simply moving the gate.** A 1.64 Å
interpenetration and a 2.15 Å near-miss both end at ~2.67 Å, and the worst band resolves
40/40. That is the minimiser satisfying its repulsive term, not evidence the poses were
sound. A check that passes 98 % of its inputs is not a check, and the raw contact is
informative *because* relaxation erases it — which is the reason `scorer` reads raw
coordinates.

Recommended split, which the raw/refined design already supports: **decide shippability on
the refined structure, keep the raw contact as a recorded diagnostic of what the predictor
produced.** The cost is the ~1.0 Å of peptide movement: on a 5–16mer that is a real
fraction of its own size, so a shipped pose is partly Rosetta's.

**Not yet measured, and it decides the question:** whether the 157 rescued structures are
energetically comparable to folds that passed cleanly, or merely clash-free. If their
refined interface energies match, moving the gate is right; if they are systematically
worse, the raw overlap was carrying information the relax hid.

**The adaptive rule's knobs were tuned on these 20 complexes** (16→4, probe 3, accept on
≥1 pass), so its 76-fold figure is optimistic. Freeze the rule and validate it on targets
it has not seen before trusting that number at corpus scale.

---

## 12. The corpus run, and where its wall time actually goes

Launched 2026-10-25 in `$ZFS/cpseq/pepbench_corpus_pot`: **4,912 targets** — the whole
PepBench corpus (`all.txt`, 6,105 entries, 80.5 % inside the 5–16 residue window) — at
8 candidates plus the unchanged peptide, each folded 10 times. **442,080 folds.**

Every candidate sequence is kept, not only the selected one, and every fold keeps its dG
and its validity verdict, so the run is a dataset rather than a search that discards its
own evidence.

| | |
|---|---|
| storage, measured | **48 KB/fold** (37 structures gzipped, 4 score row, 4 peptide coords, 3 journal) |
| corpus total | **20.2 GB** of 701 GB free |
| `full` retention instead | 493 GB — does not fit, which is what the tier is for |
| retention chosen | `rescorable`: keeps the gzipped raw complex, so a scoring change costs CPU instead of re-folding |

### Steering potentials were off, and that was most of the clash problem

Paired comparison on 4 complexes, the same sequences and the same 10 seeds, one gate
(`$ZFS/cpseq/potentials_probe`, 680 folds, 0 technical failures):

| | potentials ON | OFF |
|---|---|---|
| per-fold pass rate | **0.681** | 0.393 |
| peptide–receptor overlap rejections | **0** | 299 |
| intra-peptide overlap | 121 | 96 |
| closure torsion | 114 | 124 |

Enabling Boltz-2's own steering **eliminates peptide–receptor overlap entirely** and
leaves interface energies unchanged (per-complex medians −18.2/−25.4/−69.3/−29.0 against
−18.9/−24.8/−68.1/−29.0). The +0.288 pass-rate gain matches the +0.289 that §11 computed
for deleting the overlap criterion outright, which is the consistency check: with
potentials on there is nothing left for it to reject.

So the 44 % overlap rate was a configuration choice, not a model property, and the
raw-versus-refined gate question is largely moot. Potentials cost 1.48× more per fold
(101.5 s against 68.5 s) but yield 1.73× more passing folds, so they are **1.17× cheaper
per usable fold**. Every run in this project before this one had them off.

### 44 s of every 70 s prediction was process startup

Measured on an A6000 with these settings, folding a directory of inputs in one
invocation:

| inputs per invocation | wall | marginal |
|---|---|---|
| 1 | 70 s | — |
| 4 | 145 s | 25 s |
| 8 | 253 s | 27 s |

**44 s fixed + 26 s per structure.** The fixed part — process spawn, the torch import,
the 2.2 GB checkpoint read, the model build on the GPU — was being paid once per FOLD,
because every fold was its own `boltz predict` subprocess. Grouping the nine sequences
that share a seed takes prediction from ~70 s to ~31 s per fold; with scoring unchanged
at 27.5 s the fold goes from 101.5 s to ~62.5 s, and the corpus run from ~21.6 to
~13.3 days. Nine is already near the asymptote (28 s/structure at 18).

Implemented as a **prefetch into the same cache `evaluate` reads** (`prefetch_predictions`),
not as a new prediction path. A candidate the batch did not produce has no record written,
which is a cache miss, so `evaluate` predicts it individually exactly as before; the call
is wrapped so a batch failure cannot end a task, and `--no-batch-predict` restores the
old behaviour. There is no path on which batching changes a result — only who paid for it.

Grouping is by SEED because the seed is a boltz command-line argument: sequences sharing
a seed can share an invocation, folds of one sequence cannot.

**Verified 2026-10-25: 9 of 9 structures in one invocation, 214 s** — 23.8 s per
structure against ~74 s one at a time.

With prediction already paid for, the scoring loop records folds at **13-17 s** each
rather than the 27.5 s inferred from unbatched rows, so a target costs
`10 x 214 s + 90 x 15 s = 3,490 s` — **38.8 s per fold, and the corpus run ~8.3 days**
against 21.6 unbatched.

**A throughput reading taken over 25 minutes said 375 s/fold, and was an artifact.** The
prefetch folds every seed for a target and only then scores any of them, so a task spends
its first ~36 minutes predicting and writing no rows at all. Any measurement window
shorter than one target's full cycle lands inside one phase or the other and measures the
phase, not the run. Fold-rate windows here have to span a whole target.

That phase structure is itself the next inefficiency: the GPU idles through the scoring
phase and the CPU through the folding phase, which is what `--prefetch-ahead` exists to
overlap.

Two bugs stood in the way, and neither announced itself:

- **The output directory must not sit inside the input directory.** Boltz is pointed at a
  *directory* here and scans it for inputs, so a nested `out/` is scanned as input and
  the run yields nothing. 0 of 6 structures, no error text. The batch record now carries
  the predictor's stderr tail so the next such failure names itself.
- **Peak memory scales with batch size.** Boltz preprocesses a directory's inputs in
  parallel ("Processing 9 inputs with 9 threads"), each thread holding its own working
  set, so a batch dies with `SIGKILL` (rc -9) where a single input never would.
  `--preprocessing-threads 1` fixes it and costs a few seconds, because what the batch
  amortises is the model load, not the preprocessing.

**Neither could be diagnosed from an interactive allocation.** The node-hold job's memory
ceiling kills boltz whether batched or not — the trap sec. 11 already records — so every
measurement here came from a submitted job against a one-target cohort in its own output
directory, which is also what keeps a verification run from becoming a second writer on a
complex the live array owns.

The live array was requeued onto the fixed code once verified; 614 folds already on disk
were untouched, since the journal and the prediction cache make a restart resume rather
than repeat. Before the requeue its tasks were logging 76 batches at 0/9 and folding at
130-150 s; after it, 9/9 and ~51 s.

---

## 13. Status

- [x] 851 targets prepared from a seeded random sample of the PepBench train split
- [x] Budget chosen from folds already on disk; **B = 8**, and that choice survives §3
- [x] Random-substitution method made self-contained; 33 checks passing
- [x] Storage reduced 42× with re-scoring preserved; measured 1.1 GB for 6,481 folds
- [x] Validity criteria corrected to admit cis amides, audited against deposited crystals
      and native CPSea with every failure classified
- [x] **Filter pass — superseded at 245 of 851 targets**, 6,382 folds, zero technical
      failures. Its array left the queue on 2026-10-22 and was not resubmitted: §3 is
      why one fold per candidate cannot answer the question, so the remaining 606 targets
      would have bought more of a statistic already known not to work. The 6,382 rows are
      kept as the single-fold baseline the re-run is measured against.
- [ ] **The measurement — running (§10).** 20 complexes, 64 draws each, design and
      unchanged peptide folded 10 times each, as a ladder over the draw budget. Until it
      reaches a rung, the generalisation question is open and the only fair number from
      this run is the per-fold pass rate, which is a tie (0.436 vs 0.449).
- [ ] Atom-overlap criterion: uncalibrated, rejects 2 of 4 deposited crystal controls.
      Blocked on MolProbity/Probe (`reduce2` needs the CCP4 monomer library and `rdkit`).
- [ ] Closure angles: still worst-control-derived at 7.54°, rejecting 66/150 native
      CPSea. Needs a reference σ library for the five-standard-deviation test.
- [ ] Re-score the existing sequence-conditioned and generator-backbone folds under v3.
      Their "0 of 48 passing" was measured under the trans-only criteria now known to
      reject 82 % of native CPSea, so it is not safe to read as a property of the model.
      Structures are cached, so this costs CPU only.

Not done, deliberately: no scorer training, no classifier guidance, no reinforcement
learning. The open question is whether the evaluator supplies reproducible distinctions
between plausible sequences, and §3 is why that is not yet answered.
