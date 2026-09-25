# LP → CP pre-build check suite

Four model-free checks that run against data that already exists. Nothing here builds the
pose-decoy dataset, and nothing here trains or evaluates an LP→CP model.

These are **measurements**. A check that comes back bad is a result: the reports state it and
stop, and none of them proposes a remedy — the remedies are design decisions made outside
this suite.

**Status: full run complete, 24 September 2026.** Results live in
`evaluation_results/precheck_20260923_214714/`. 4798 of 4800 poses, 300 complexes, 60 LNR
targets, 1200 + 1200 + 60 feature profiles.

---

## 1. Verdicts

| task | question | verdict |
|---|---|---|
| 0 pin | is the best known pocket14 SDEdit config still runnable? | **clean** — commit, config and files recorded and verified |
| 1 decoy filter | can `contact_retention` replace Rosetta interface ΔdG? | **NO** — r² 0.004 in the operating range |
| 2 LNR feasibility | does a ring-closing solution preserving the interface exist? | **yes for all 60**, but see the terminal-anchor correction |
| 3 profile / adversary | can synthetic open states be told from real linears? | apparatus **works** (control AUC 0.998); one **unresolved confound** at 0.900 |

**Three of the four came back against the design.** Read section 6 before planning a build.

---

## 2. Corrections this run forced

Recorded here because each one invalidates something previously written down.

### 2.1 The bridged `k_min = 0` result is unreachable by this model

Task 2 reported median `k_min = 0` for disulfide and isopeptide — the ring closes with the
peptide not moving at all. That number scans **every** (i, j) pair with |i−j| ≥ 3 and takes the
cheapest.

**CPSea is exclusively terminal-bonded.** Measured 2026-09-24 on 60 sampled complexes: 60/60
bond at exactly (0, L−1). Bond lengths split 1.33–1.36 Å (peptide / isopeptide) and 2.03 Å
(disulfide), so the *chemistry* varies but the *endpoints never do*. This confirms
`cpsea-head-to-tail-verified` (27k raw structures, all 2.71M preserved). The design pipeline
independently **derives endpoints as first/last valid residue**, so the inference path is
terminal-only too.

So the model cannot express an interior bridge, in data or at sampling time. Pinning anchors
to the termini — the only thing it can express — collapses the result:

| chemistry | calibrated CA–CA window | LNR targets inside at k = 0 |
|---|---|---|
| disulfide | 3.5–7.3 Å | **5%** (3/60) |
| isopeptide | 4.2–10.3 Å | **15%** (9/60) |
| head-to-tail | N–C 1.08–1.58 Å | 0% |

Against a median LNR terminal CA gap of 16.3 Å. The three validity-control targets sit at
42.8 / 24.5 / 13.4 Å — outside every window — so they remain hard under terminal bridging too.

**Consequences.** Terminal gap *is* the right discriminator for this model, which contradicts
`README_POSE_DECOY_INVENTORY.md` §4.4 ("the chain-terminal gap is the wrong discriminator for
bridged linkages"). Milestone 1.5's routing of the 20–45 Å coverage band to bridged chemistries
rests on §4.4, and M1.5's own bridged ceilings scan interior pairs, so they carry the same
optimism. **Open item:** re-run Task 2's bridged arm with anchors pinned to (0, L−1).

What survives: terminal bridging still beats head-to-tail, just far less dramatically —
it relaxes "ends within ~1.3 Å" to "ends within ~7–10 Å", i.e. a softer constraint on the same
axis, not a different axis.

### 2.2 "Short peptides are at chance" — WITHDRAWN

An earlier read reported held-out AUC 0.406 in the ≤8-residue bin and called discriminability
length-dependent. That was an artifact of the classifier bug in §7.2. At full n the strata are
uniform: 0.991 / 0.997 / 0.993 / 0.998 across the four length bins. **Do not cite the 0.406.**

### 2.3 The staging confound is reduced but NOT removed

Reported as 0.870 from a 240-row sample; **0.900 at full scale** (n = 2400), against 0.999
before the uniform crop. Read every held-out AUC against 0.900, not against 0.5.

---

## 3. Layout

| piece | file |
|---|---|
| config (inherits `m15.yaml`) | `configs/pose_decoy/precheck.yaml` |
| config loader with `inherit:` | `script_utils/precheck_config.py` |
| shared sbatch preamble | `scripts/_precheck_preamble.sh` |
| submitter (calls `sbatch` only) | `scripts/submit_precheck.sh` |

| task | modules | jobs |
|---|---|---|
| 0 | `precheck_pin_baseline.py` | `precheck_pin.sbatch` |
| 1 | `pose_decoys.py`, `precheck_decoy_{sample,minimize,score,report}.py` | `precheck_decoy_{sample,minimize,score,report}.sbatch` |
| 2 | `precheck_lnr_feasibility.py`, `precheck_lnr_report.py` | `precheck_feasibility{,_report}.sbatch` |
| 3 | `peptide_profile.py`, `precheck_profile.py`, `precheck_adversary.py`, `precheck_task3_report.py` | `precheck_profile.sbatch`, `precheck_adversary.sbatch` |

Tasks 1–3 are independent and submit as siblings so they queue concurrently. Stages within a
task chain on `--dependency=afterok`; reports hang off `afterany`, so a slow shard hitting its
wall clock costs sample size rather than the report.

`geometry` is **inherited** from `m15.yaml` rather than restated, so the contact cutoff, the
pseudo-angle bound and the torsion budget are the same numbers by construction. A retention
computed against a different cutoff is a different quantity and must not be divided into one
that was not.

---

## 4. Running it

```bash
# smoke first -- capped budgets and limits, NOT a result
bash scripts/submit_precheck.sh --all --smoke --partition general

# the real thing
bash scripts/submit_precheck.sh --all --partition general

# resubmit task 1's minimize -> score -> report without redoing stage A
bash scripts/submit_precheck.sh --resume-task1 --partition general --run-id <existing run>
```

Everything is **CPU-only**. The single GPU job is Task 0's confirmation run, which the pin job
*prints* rather than submits — GPU work here goes through `sbatch` after checking `myfree`.

Measured costs: profile ~0.5 s/complex; feasibility ~60 s/complex (mainchain alone ~70 s);
minimize ~6 s/pose at 16-way parallelism (~45 min/shard); Rosetta ~36 s for the first call
then ~2 s/structure. Whole suite: a few hours wall clock.

---

## 5. Task detail

### Task 0 — pinned baseline

Records commit, config and the exact launch command, and proves every file it names exists
(60 targets, 2.94 GB checkpoint).

| key | value | why |
|---|---|---|
| staging | `CPSea_data/lnr_pocket14/…` | receptor segmentation alone is worth 22–34 points of closure |
| flow ckpt | `cpsea_bondunroll_pin20260828/last-EMA.ckpt` | pinned **with its AE**; a flow ckpt scored against an AE it wasn't trained on is incomparable |
| `t_ca` / `t_lat` | 0.8 / 0.4 | `t_lat = 1.0` is the frozen-sequence corner and closed 1/59 at every `t_ca` |

The confirmation run is one seed at one grid point. It shows the pin **executes**; it is not a
closure rate.

### Task 1 — decoy filter calibration: the answer is no

Four stages: sample → minimize (`.venv_openmm`) → score (`.venv`) → report.

**Operating range** (≤2 Å, ≤12°, retention ≥0.7; n = 1829): Spearman **−0.106**, Pearson
−0.061, r² **0.004**, residual SD **19.44 REU** against a dG SD of **19.47 REU**. Fit the best
line and you remove 0.03 of 19.47 — retention explains essentially none of the binding-energy
variation where a filter would run.

Median ΔdG moves only **−29.6 → −33.6 REU** across the *entire* retention span, about 4 REU
against a within-band spread of 19.5. Sorting by retention sorts by ~0.2 SD of ΔdG.

Globally Spearman is −0.392 (p ≈ 1e-13) while Pearson is −0.034 and the dG SD balloons to 99.8
REU: the monotone signal is entirely extreme-clash outliers. Nobody needed a filter to know a
5 Å displacement is bad.

**Retention tracks geometry, not energy** — Jaccard 0.443 → 0.090 and CA-RMSD 2.09 → 0.53 Å
across retention quantiles. Those are all the same geometric quantity.

**Second result, arguably more useful — a 2 Å amplitude floor:**

| initial translation | collapsed back within 1 Å of native |
|---|---|
| ≤1 Å | **86%** |
| 1–2 Å | 19% |
| >2 Å | **0%** |

A pose that collapses is not a decoy — it carries the native answer and reinforces the
pocket→canonical-peptide shortcut. Overall collapse 26%.

**And the two results collide.** The filter's operating band is ≤2 Å; the decoy floor is >2 Å.
Disjoint. The regime needing fine judgement doesn't yield usable decoys, and the regime that
does may not need a subtle filter. That reframes the filter problem rather than solving it.

**Why direction is decomposed, not pooled.** Translations are drawn in the interface frame and
logged as normal (out of the receptor) and two tangential components, with direction modes
cycling `normal+`, `normal−`, `tangent`, `isotropic`. At ≈4.3 Å, `normal+` retained **0.098**
of contacts while `normal−` retained **0.878**. A scalar amplitude averages those into a
mixture and a regression on the mixture fits an artifact.

**The ring restraint is a HOLD, not a PULL.** The decoy starts closed and must stay closed:
CONECT records do not survive into `createSystem`, so a force field that never saw the ring
bond would open the macrocycle. The bonded pair is detected geometrically in stage A (in
`.venv`) and carried in the manifest, because the CONECT parser sits behind the torch stack
that `.venv_openmm` lacks.

**FastRelax is off** — relaxing before scoring would undo the perturbation being calibrated.

Caveats: rigid-body only, so a torsion-perturbing generator visits poses this never saw; the
receptor is frozen so induced fit is absent by construction; 6 of 16 strata cells were
underfilled (one had 1 candidate against a quota of 18) and topped up from outside the strata,
so the 300 is the right size but not fully balanced; 2 poses lost to OpenMM NaN, non-randomly
at the extreme-clash end.

### Task 2 — LNR closure feasibility

Reuses the M1.5 solver (`kinematic_ceiling.py`) unchanged. `k` = residues released from their
input conformation; the held complement is a contiguous window at its native bound position.
`k_min` = the smallest feasible `k`.

**All 60 targets feasible under some (k, chemistry).** Median `k_min`: mainchain **4** of L=10
(anchor retention 0.682), disulfide and isopeptide **0** (retention 1.000) — *but read §2.1,
the bridged figure is unreachable*.

Mainchain frontier, fraction of targets with a solution:

| tolerance | k≤2 | k≤4 | k≤5 | k≤7 |
|---|---|---|---|---|
| any | 0.07 | 0.55 | 0.78 | 0.98 |
| ≥0.70 anchors | 0.05 | 0.27 | 0.40 | 0.60 |

**Quote `k`, not the retention ratio.** Retention has peptide length in its denominator, so any
trend against length or gap is arithmetic — the trap `README_M15_CEILING_AUDIT.md` §2 records.

**Two optima per `k`**, on all-contact and on anchor retention separately, because they
disagree: on `LNR_1bjr` mainchain at k = 4 the all-contact optimum retains **zero** anchors.
Anchors are the top-3 peptide residues by buried surface area.

**Analytic and torsion results bracket.** Analytic is a true upper bound (what it rejects is
genuinely infeasible); torsion is the confirmed lower bound. On `LNR_1bjr` analytic licenses
k = 1 where the backbone needs k = 3. Torsion feasibility is monotone in `k`, so the threshold
is binary-searched and levels below it are marked `torsion_inferred` rather than re-proved —
proving a level infeasible is the expensive operation.

**Validity control PASSES.** The three targets SDEdit never closed across 20 grid points come
back as the expensive ones for mainchain:

| target | terminal CA gap | L | `k_min` mainchain | anchor retention |
|---|---|---|---|---|
| `1jrr` | 42.8 Å | 14 | **8** | 0.617 |
| `4x3h` | 24.5 Å | 9 | **5** | 0.560 |
| `3cvl` | 13.4 Å | 5 | 3 | **0.132** |

Two independent instruments agreeing means the rest of the table can be trusted. `3cvl` — the
one that later *did* close in the paired soft-closure control — shows why: it closes, but
anchor retention collapses to 0.132. It bought closure by destroying the interface. Only
`1jrr` and `4x3h` remain genuine never-closed hold-outs; the config records both lists.

The released set is the two flanks of a contiguous window; releasing an interior loop while
pinning both termini is not in the sweep. Harmless for head-to-tail (pinning both termini fixes
the atoms the bond must join) but a real restriction for a bridged chemistry.

### Task 3 — profile, reference distributions, adversary, OOD

`peptide_profile.profile()` maps a complex to 15 numbers across the 8 named feature groups and
returns the same layout for cyclic, open and linear peptides. Three rules make that true, each
costing fidelity on purpose: torsions are read off the **linear** chain, never around a ring
bond; no CONECT record is read; nothing is normalised by a quantity only one topology has.
Staging descriptors live in a **separate** vector — they are the confound, not the signal.

**The three confounds.** Every set is cropped with the same 14 Å operation at profile time
rather than staging one to match another (PepBench 378 → 150 residues, 1 → 17 segments,
max run 378 → 10; LNR unchanged, confirming idempotence). A control classifier on staging
features alone runs every time. Length is stratified, not pooled.

**Positive control** — cpsea vs pepbench, n = 1200/1200, grouped on `cluster_id`: AUC **0.998**
(tuning 0.993, held-out 0.997), threshold 0.90, **PASS**. Importances are conformational
(`e2e_ca_A` 0.032, `terminal_exposure` 0.013, `e2e_ca_per_link` 0.012), not staging. The control
**gates** the rest — an AUC of 0.5 elsewhere means nothing until the harness has shown it can
detect a difference it should. Folds split on `cluster_id`, not rows: these sets are
family-redundant and a row split inflates every AUC toward 1.0 for an irrelevant reason.

**Staging control 0.900** post-crop (0.999 pre-crop) — see §2.3.

Length strata (held-out features): 0.991 / 0.997 / 0.993 / 0.998 — uniform. See §2.2.

**`lnr` vs `pepbench` is pure staging**: control 0.729 ≥ full-feature 0.702. Do **not** treat
them as interchangeable real-linear references.

**OOD** — Gaussian KDE on the standardized vector, fitted on the CPSea side only (fitting on
the union would blunt the signal). Median percentile vs the fit set: cpsea 0.500, lnr **0.000**,
pepbench **0.000**. Both linear sets sit essentially entirely below CPSea's 1st percentile.

**Not built: 3d, the learned-representation adversary.** Specified as a second stage after
3a–3c work; they now work, and 3d remains open. No synthetic open states exist yet, so the
adversary has only run against real sets — which is what this stage was specified to do.

---

## 6. What still has to exist before a dataset build

From `README_POSE_DECOY_INVENTORY.md` §3.1, still absent as of this run:

1. **Ring opening** — nothing in the repo cuts a ring bond. `pose_decoys.detect_ring_bond`
   *finds* one; it does not open it.
2. **The corruption module** — only `perturb_backbone_torsions` exists, 1 of 7 knobs.
3. **Backbone-geometry and "relaxed back toward target" filters.**
4. **A sharded columnar writer plus manifest** for this schema.
5. **A structural cluster partition against LNR** — no `foldseek` or `mmseqs` on PATH; the
   existing holdout is receptor-sequence only, and that should not be presented as a
   structural partition.
6. **The dataset-level identifiability audit**, and **3d**.

And four decisions the measurements do not settle:

1. What replaces `contact_retention` as the acceptance filter, given r² 0.004.
2. What amplitude band decoys are drawn from, given the 2 Å floor.
3. Whether the §3 coverage target changes now that the bridged routing is invalid for this model.
4. Whether interior-bridge training data is worth acquiring — the unrestricted scan says it
   would unlock a great deal (median `k_min` 0 vs 4), and it is the one result arguing to change
   the *data* rather than the *method*.

---

## 7. Traps

### 7.1 A walltime over 8 h never schedules on `general`

A 24 h request for the minimize stage sat `PENDING` for **15.5 hours** while every job at 8 h
or less ran. The identical job ran fine at 2 h in the smoke, which is the control that pins the
cause. Real need is ~45 min/shard. **Do not raise walltimes "to be safe"** — the job silently
never runs rather than erroring.

### 7.2 `HistGradientBoostingClassifier` is degenerate on small folds

`min_samples_leaf` defaults to 20, so with 32 training rows no split is legal, every tree is a
single leaf, and the result is AUC **exactly 0.500** with all-zero permutation importances —
which reads as "the sets are indistinguishable". It produced the withdrawn finding in §2.2.
Now scaled to the fold size, with a guard that reports a constant-prediction fit as unusable
rather than as 0.5. The positive-control gate caught it, which is what that gate is for.

### 7.3 Other

- `select_rows` must not shuffle all 2.44M `cpsea_train` rows (minutes and GB *per shard*) —
  draw cluster ids down first, then pick one row per cluster by seeded hash.
- `mdtraj`'s `top.select()` returns atom **indices**, not `Atom` objects.
- `hostname` is not on PATH on every compute node; the banner uses `SLURMD_NODENAME`.
- **Every aggregator refuses corrupt rows rather than skipping them.** Concurrent appends
  NUL-corrupt rows on this filesystem (88 of 144 lost, measured previously), and a skip turns
  lost data into a smaller-but-plausible number. Each job writes its own output file.
- Never `sbatch --export`: any explicit export list sets `SLURM_GET_USER_ENV=1`, slurmd fails to
  rebuild the login environment, and the job is requeued and **held**. The submitter writes one
  env file and passes its **path as a positional argument**.

---

## 8. Outputs

```
evaluation_results/precheck_20260923_214714/
  pin/PRECHECK_TASK0_PIN.md          task0_pin.json
  PRECHECK_TASK1_REPORT.md           decoy_pose_table.csv, precheck_task1_summary.json
  PRECHECK_TASK2_REPORT.md           frontier_by_target.csv, sweep_{all,anchor}_contacts.csv
  PRECHECK_TASK3_REPORT.md           reference_distributions.json, profile_rows.parquet,
                                     adversary_results.json, ood_scores.csv
  decoys/selection_strata.json       which strata cells were underfilled
```

`reference_distributions.json` is the serialized artifact the §5 corruption sampler will draw
target feature values from. It carries `reference_only: ["lnr"]`: LNR is development data that
has already shaped this repository's preprocessing and sampling decisions, so it is profiled
and reported but never used to tune anything.
