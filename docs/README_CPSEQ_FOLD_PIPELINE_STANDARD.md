# Standard practice: how a cpseq fold chain is built

Two rules, both about not paying for the same thing twice. They are defaults, not
options — a new chain gets them unless there is a stated reason not to.

1. **Predict on the GPU, score on the CPU.** Never one job that does both. *In force.*
2. **One trunk per sequence, not one per seed.** The seed axis is diffusion sampling and
   belongs inside a single invocation. *Wired and off by default — see §2.* The three
   blockers are cleared and the payoff is measured — **2.34× at n=9** (job 95978, with a
   0.01% positional control). Opt-in per cohort only because no cohort has been folded
   end to end this way yet.

Measured basis for both: `llm_ladder`, 22,642 v3 folds, **327 job-hours**, split 150 h of
Boltz-2 prediction against 177 h of Rosetta scoring. Per-fold medians 37.8 s predict /
29.6 s score. Those rows were written on babel, so treat the hours as a ratio, not as an
autonlab cost.

---

## 1. Predict on the GPU, score on the CPU

**The shape.** `prepare → predict (GPU array) → score (CPU array) → report`, chained with
`--dependency=afterok`. `scripts/submit_cpseq_potentials_check.sh` is the reference
implementation and submits exactly this.

**Why.** Scoring is 54% of a fold's wall time and needs no card. Run as one job, the two
phases hold an a6000 through a PyRosetta relax; run as two, the scoring half goes to `cpu`
(88-core nodes, `lov[1-2,5-9]`) which is not contended against training. Every `general`
a6000 node was fully allocated for the whole of 2026-11-03 while `cpu` had four idle
nodes — the split buys queue position as much as compute.

**Why it is safe.** `evaluate` keys predictions by `input_hash` and scores by
`(scoring_version, input_hash)`. The plan run as two passes fills the same two caches as
the plan run as one. Only who pays changes.

**How.**

```bash
# GPU pass: predictions only, no outcome rows written
sbatch ... scripts/cpseq_potentials_fold.sbatch "$ENV_FILE" PREDICT_ONLY=1
# CPU pass: same plan, every prediction now a cache hit
sbatch ... scripts/cpseq_potentials_score.sbatch "$ENV_FILE"
```

### 1a. Per-complex scoring parallelism

**`--score-workers N` is how a scoring job uses its cores.** Not `--shard-count`.

The shard axis in `cpseq_search.py` is the **parent list**
(`cohort["cohort"][shard_index::shard_count]`), so an array that is already one complex
per task — which the potentials and fill-sweep arrays are, by design — gets *nothing*
from sharding, and its Rosetta phase runs strictly serially however many CPUs it was
given. `cpseq_ladder_score.sbatch`'s `WORKERS=3` sub-shard trick works only because that
array holds many complexes per task.

**The constraint that sets the design:** the journal is per complex, and concurrent
`O_APPEND` to one file NUL-corrupts rows on this cluster. So the **workers score and the
parent writes** — records come back over a pipe, the journal keeps a single writer, and
the prefix-safe plan order is unchanged.

Two properties worth knowing:

- Only folds whose prediction is **already on disk** are handed to workers.
  `evaluate.GPU_LOCK` is a lock within one process and cannot serialise N of them, so a
  pool that met a missing prediction would put several boltz invocations on one card.
  Uncached folds get a sentinel and the parent predicts them itself, serially. This is
  what makes `--score-workers` a *scoring-pass* tool: pair it with `PREDICT_ONLY=1`.
- **The allocation is the cap, not the pool.** One core and ~1.5 GB of PyRosetta per
  worker on a 422-residue pose. Set `SCORE_WORKERS` as high as `--cpus-per-task` and
  `--mem` allow; raise both together to go past 16.

### Measured scaling, and why it is not a plateau

Job 95812, 10 folds, serial baseline 793.3 s. Optimal makespan for a task set is
`max(longest task, total / workers)` — no pool can beat it:

| w | total CPU | longest fold | optimal | measured | vs optimal |
|---|---|---|---|---|---|
| 2 | 727.0 s | 169.5 s | 363.5 s | 385.1 s | 94.4% |
| 4 | 733.8 s | 174.1 s | 183.4 s | 235.8 s | 77.8% |
| 8 | 732.9 s | 172.5 s | 172.5 s | 172.7 s | **99.9%** |
| 16 | 728.8 s | 172.0 s | 172.0 s | 172.3 s | **99.8%** |

The flat wall between w=8 and w=16 is **not** a limit of the pool — it is within 0.2% of
the theoretical optimum at both. The wall is pinned by **one 172 s fold** while the other
nine total ~560 s, so past w≈4 there is nothing left to overlap the long one with. Ten
items cannot measure further; this says nothing about w=16 in production.

**There is no contention term at all.** Total CPU work is 727–734 s at every worker
count, and *below* serial's 791 s because the page cache is warm. Per-fold median is
~78 s in every pool arm against 95.2 s serial. Sixteen PyRosetta processes on an 88-core
node do not slow each other down.

So on a real complex — ~1,040 folds, ~73 s each, ~21 h of serial work — the tail is
irrelevant and the pool stays `total/workers`-bound until roughly
`total/longest = 76,000/180 ≈ 420` workers. **Scale workers with the allocation.**

A caution for whoever re-measures: a benchmark whose item count is close to its worker
count measures its own task distribution, not the pool. Use `N` at least 8× the largest
worker count.

### 1b. Optional: skip the relax on folds the gate already failed

`--skip-invalid-relax`. Under a **prerelax-gated** protocol — v3 is one; the module
default `v4-postrelax-validity` is not — the verdict is read off the structure the
predictor produced, so a fold that fails it is already decided and its refined energy can
never reach the gate or `best_valid_dG_separated`.

Measured directly (job 95812, 10 folds on `llm_ladder_pot` structures):

```
score_structure full:        95.2 s median
score_structure skip_relax:   6.39 s median      -> the relax is 93.3% of scoring
validity_report:              0.15 s             -> the gate is ~1/600 of the relax
```

So the saving is `invalid_fraction × 93.3%`. At the journal-wide **60.9%** invalid rate
that is **~57% of all scoring time**, which agrees with the 58.2% derived independently
from the `llm_ladder` journals.

Do not use the 18.7% the probe printed for its own sample: it drew `rows[:take]` per
parent, i.e. the earliest journal rows, which are disproportionately the *reference*
peptide — native, and so more often valid. Its sample came out 2/10 invalid against 60.9%
across the full journals. That is a sampling artefact of the probe, not a property of the
corpus.

The correctness checks in that run all passed, which is what makes the lever admissible:

```
prerelax verdict mismatches between arms: 0      (must be 0)
skipped rows that KEPT the raw analysis:  10/10  (raw dG is not relax-dependent)
skipped rows carrying a refined energy:    0/10  (must be 0)
```

**Off by default, and it must stay opt-in.** A skipped row carries no refined energy, no
deformation proxy and no prerelax-to-postrelax gap, so it cannot be rescored to v4/v5. The
score record marks itself `relax_skipped` and a later pass without the flag treats it as a
**miss** — otherwise the fold would stay unrelaxed forever while every run reported a
cache hit, which is the shape of the crashed-prediction caching bug that once cost 72
folds. The flag is also inert under a postrelax protocol by construction: the stage is
checked, not just the flag.

---

## 2. One trunk per sequence, not one per seed

**The waste.** The trunk — template → MSA → pairformer, `recycling_steps+1` rounds — is
**deterministic at inference**: `--subsample_msa` is never passed, so the `torch.randperm`
in `trunkv2.py` never runs, and compiled modules revert to `_orig_mod` outside training.
The protocol runs `diffusion_samples=1`, and `ladder_plan` emits the full cartesian
draws × seeds grid. So a candidate folded at 16 seeds computes a **bit-identical trunk 16
times**, and the seeds only ever perturbed diffusion noise.

**Measured on an A6000** (job 95853, warm arms only, single input, potentials ON):

| ds | wall | per structure |
|---|---|---|
| 1 | 45 s | 45 s |
| 10 (mps=5, 6 chunks) | 148 s | 14.8 s |
| 10 (mps=15, 2 chunks) | 141 s | 14.1 s |
| 10 (mps=30, 1 chunk) | 139 s | 13.9 s |

Solving `T = F + K + ds·s` on this single input: **`s` = 10.4 s per sample, `F + K` =
34.6 s** — 77% of a single-structure fold is startup-plus-trunk. (Batched, `s` falls to
8.0 s and `F`/`K` separate; see "The production saving" below, which supersedes this pair.)
Set `max_parallel_samples` high enough for one chunk; granularity is worth only 6% at
ds=10, so it is a correctness choice more than a speed one.

**What this does NOT license.** Production already batches many sequences per invocation
(`prefetch_predictions`), which amortises `F` but **not** `K` — every input still needs
its own trunk. So the saving from collapsing seeds is `(depth−1) × K` per input plus `F`
once, not `(depth−1) × (F+K)`. This was the open question and phase 3 answered it: `K` is
11.5 s of the 34.66 s, so the batched speedup is **2.2×**, not the 3.2× the unbatched
arithmetic suggests. Do not quote the unbatched 720 s → 216 s figure as a production
number.

### The three blockers, and how each was cleared

All three are closed as of 2026-11-03. What each needed, and what it became:

**1. Equivalence — measured, and it passes.** Job 95853 phase 2: ten folds the current
way (ten invocations, seeds 42–51, ds=1, unpatched) against ten the proposed way (one
invocation, seed 42, ds=10, patched), compared as confidence *distributions*.

| metric | old mean (sd) | new mean (sd) | Mann–Whitney p |
|---|---|---|---|
| `confidence_score` | 0.9519 (0.0073) | 0.9491 (0.0106) | 0.597 |
| `iptm` | 0.9247 (0.0142) | 0.9115 (0.0289) | 0.496 |
| `ptm` | 0.9785 (0.0015) | 0.9776 (0.0025) | 0.650 |
| `complex_plddt` | 0.9587 (0.0075) | 0.9585 (0.0083) | 0.880 |

No mean shift on any of the four. The decisive part is the **spread**, because the failure
mode this check exists to catch is the collapse understating fold noise — and the spread
*widened* on all four (sd ratios 1.1–2.0), which is the opposite of that. Two caveats to
carry: it is n=10 per arm, so it would not detect a small shift; and it was run at
**ds=1 against ds=10**, which is the production depth but says nothing about ds=40.

**2. The chunk count — shipped, hashed, and verified against the live install.**
`diffusionv2.py:383` computes `multiplicity % max_parallel_samples + 1` where it means
`ceil(multiplicity / max_parallel_samples)`. The probe's `PYTHONPATH` shadow is now
`adapters/boltz_chunkfix.py`: `ensure_patched_boltz()` builds a verified copy (idempotent,
atomic `os.replace` so concurrent array tasks cannot interleave, ~0.7 s to reuse), and the
installed package is still never written to — checked, it still hashes to
`e68643da577decdd` and still contains the bug.

It is **hashed into the prediction key** (`chunk_fix: "chunkfix1"`), not treated as a free
optimisation. The patch cannot change *what* is computed — noise and augmentation are
drawn for the full multiplicity outside the chunk loop and the denoiser has no cross-sample
term — but a different batch size is not bitwise identical, and §3 is the rule that such a
thing must key apart. `max_parallel_samples` is hashed for the same reason.

At ds=1 the bug costs only ~10% (warm unpatched ~50 s vs warm patched 45 s), so it is an
**enabler, not a lever**: `ds` cannot be raised at all without it. Do not read job 95853's
A1 (227 s) as the unpatched cost — A1 was the cold first arm, and cold start is ~3.8× here.

**3. Fold identity — shipped, and it needed more than a key change.** Journal rows key on
`(sequence, seed, stage, arm)`, so a collapsed row at `seeds[0]` would key identically to
the legacy single fold at `seeds[0]`: resume would call it done and the run would write
**one** fold where the plan asked for `depth` — a rung reported at full depth holding one
draw of the noise. The key now carries a fifth component, the fold's slot inside its
prediction, absent on a per-seed row. Legacy resume is therefore bit-for-bit unchanged, and
the two regimes simply never satisfy each other.

The part that was *understated* as "a report-axis change": the ladder's unit of evidence is
an **independent fold row**. `cpseq_ladder_report.py` keys folds by rank, counts `n_pass`
out of `n_folds`, applies a `k`-of-`n` reliability rule and holds out half the folds. A
collapsed prediction aggregates to a single `valid` boolean and one best energy, so writing
it as one row would reduce a depth-10 rung to depth 1 and turn the reliability rate back
into the coin flip §1 of the ladder design exists to replace. So the collapse happens at the
**prediction** and is undone at the **row**: `evaluate.single_fold_views()` gives each
sample its own verdict, energy and violation, and the row's `seed_rank` is filled with the
sample index. Every existing reader — `by_rank`, the `k`-of-`n` rule, the held-out half, the
completeness count — then works with **no change at all**, because all any of them asks of
that field is "which fold of this draw".

### The production saving: 2.34× at n=9, measured

Phase 3 was re-run on its own as **job 95978** (A6000, `preempt`/gpu1, phases 1–2 skipped)
and finished completely. This is the measurement that was missing, in production shape —
`A6` is today's batch, `A7` is the proposal, both 9 inputs, same input set, warm:

| arm | n | ds | wall | folds | per fold |
|---|---|---|---|---|---|
| `A6` today's batch | 9 | 1 | 199.0 s | 9 | 22.12 s |
| `A7` the proposal | 9 | 10 | 850.5 s | 90 | **9.45 s** |
| `A6b` positional control | 9 | 1 | 199.1 s | 9 | 22.12 s |

Depth 10 today is ten `A6` invocations: **1990 s → 850 s, a 2.34× speedup**, saving 1140 s
per 9-input group. The `A6b` control reproduced `A6` to **0.01%**, so that ratio is the
proposal and not drift — which is the check that matters here, because a cold first arm
once faked a 5× result on this probe.

`P2`/`P4` (2 and 4 preprocessing threads) came back at 199.1 s and 198.4 s against `A6`'s
199.0 s: **serialised preprocessing costs nothing measurable**, so the
`--preprocessing-threads 1` pin that keeps a batch inside memory is free. `K3` (kernels)
failed as expected — see "Not available" above.

**`F` and `K` are now separated**, which is what phase 3 existed for:

```
F = 23.2 s  per INVOCATION   spawn, torch import, the 2.2 GB checkpoint
K = 11.5 s  per INPUT        the trunk: template, MSA, pairformer, recycling+1
s =  8.0 s  per SAMPLE       one diffusion draw
```

`F + K` = 34.66 s against phase 1's independently measured 34.6 s — **0.2% apart**, across
two runs on different days. That cross-check is what makes the split trustworthy rather
than a fit to one run. The old fused `34.6 / 10.4` pair is superseded; note `s` is
**8.0 s batched** against 10.4 s single-input, so the samples get cheaper in a batch too.

`K` is what the collapse saves, `(depth−1)` times per input. `F` is saved only once per
invocation, so **the speedup falls as the batch grows** and the doc's "quote 15 × K"
instinct was right:

| n | today | collapsed | speedup |
|---|---|---|---|
| 9 | 1990 s | 851 s | 2.34× |
| 17 | 3553 s | 1586 s | 2.24× |
| 25 | 5116 s | 2321 s | 2.20× |
| 65 | 12931 s | 5998 s | 2.16× |
| →∞ | | | 2.13× |

So **~2.1–2.3× is the production number**, against the 3.2× unbatched upper bound. Quote
2.2×, not 3.2×, and never the unbatched 720 s → 216 s figure.

### Why it is still opt-in

The saving is now measured and the blockers are closed, so what remains is exposure, not
evidence: no cohort has yet been folded end to end this way, and the two items below are
untested. A first collapsed cohort that reproduces a known uncollapsed one's pass rates is
what would make this the default.

**Unmeasured: depth above 10, and peak GPU memory.** The collapse does not widen the input
batch — the prefetch already grouped every sequence sharing a seed — it deepens the
*per-input* denoiser multiplicity, from `3 × 1` to `3 × depth` resident at once. The chunk
fix is what makes that a number someone chose (`max_parallel_samples`) instead of a
remainder, and `A7` ran fine at `3 × 10` in 32 GB with `mps=15`, but nothing above depth 10
has run. Pick `--max-parallel-samples` deliberately and watch the first task.

**One trap that is handled.** Each input costs `K + depth·s` instead of `K + s` — 91.9 s
against 19.5 s at depth 10 — so a 65-input group that fitted the 7200 s timeout at ds=1
becomes ~6000 s, and a timeout returns *nothing*: the batch is lost and the scoring loop
then predicts each of its folds singly, throwing away the whole saving.
`prefetch_batch_cap()` splits a collapsed group into invocations sized to half the timeout
(38 inputs at depth 10, on the measured cost above). The split costs one `F` per extra
invocation — 23.2 s against a ~3500 s sub-batch, under 1% — and bounds a timeout's damage
to one sub-batch instead of the lot.

### How to turn it on

It is a **cohort** setting, not a shard flag — the same rule as `use_potentials`, and it
matters more here: the two regimes key apart in the journal, so a shard disagreeing with its
cohort would not collide loudly, it would quietly fold the same complex twice under two keys
and report both. A collapsed run therefore needs its own `--out-dir`.

```bash
# Build the cohort with the lever on. --chunk-fix is required above depth 4.
python script_utils/cpseq_search.py ladder-cohort \
    --out-dir $ZFS/cpseq/seed_ladder_collapsed \
    --corpus dataset_dir=...,n=20,label=LNR \
    --seed-depth 10 --collapse-seeds --chunk-fix --max-parallel-samples 15

# The fold job is UNCHANGED: it reads all three settings off the cohort. The sbatch's
# SAMPLES=1 default is accepted and overridden to the cohort's depth.
OUT_DIR=$ZFS/cpseq/seed_ladder_collapsed sbatch scripts/cpseq_ladder.sbatch
```

A shard that passes `--samples` disagreeing with the cohort's depth is refused rather than
overridden, and so is `--collapse-seeds` above depth 4 without `--chunk-fix`. Checks:
`python script_utils/test_cpseq_collapse.py` (47 assertions, no GPU).

### Not available: the cuequivariance kernels

Settled, so nobody re-opens it. `adapters/boltz2.py` hardcodes `--no_kernels` and that
stays. Its stated reason — that the kernels would replace the pinned torch 2.7.0+cu126 —
is **wrong**: 0.10.0 declares torch only as a `test` extra and installs cleanly into a
47 MB `pip --target` shadow with torch untouched (job 95843). The real reason is **API
drift**: on an A6000 with the driver present, boltz 2.2.1 calls
`triangle_multiplicative_update`, which 0.10.0 does not export, and every version that
does requires torch ≥ 2.11 (jobs 95856 config C, and 95853 arms K1/K2 —
`predictor_error`). Revisit only if boltz or the torch pin moves.

---

## 3. The cache key covers the settings, and must keep doing so

`input_hash(yaml_text, cfg)` is the prediction key and `Boltz2Config.to_dict()` is what
goes into it: `diffusion_samples`, `recycling_steps`, `sampling_steps`, `use_potentials`,
`seed`, `affinity`. Anything that changes the numerics of a prediction and is **not** in
there will silently mix two regimes under one key across the ~29k cached predictions —
inside comparisons whose entire design is that exactly one setting moves.

The kernel flag is the live example: it is absent from the config, which is the other
reason it could not simply be switched on. Any future inference-level change goes into
the hashed config or bumps `BOLTZ_REV` **first**.

`chunk_fix` and `max_parallel_samples` were added under this rule on 2026-11-03, and they
are **conditional keys**: each appears in `to_dict()` only when set away from its default.
That is a deliberate exception and the only one allowed here. The ~29k cached predictions
were keyed by this dict before the fields existed, so an unconditional key would change
every one of their hashes and re-predict the lot; a default-valued field describes a run
made exactly as those were, so its absence is accurate rather than a gap. Verified against
120 real cached records across five run directories — all 120 still reproduce their own
cache key. A run that actually turns one on keys apart, which is what this section
requires.

`chunk_fix_root` is deliberately **not** hashed. It is a filesystem path to a build whose
content `chunk_fix` and `CHUNKFIX_REV` already pin, and hashing it would file the same
prediction under two keys on two nodes.
