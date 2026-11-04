# Standard practice: measuring a walltime change

This is the method every walltime claim in this project has to be produced by. It exists
because on 2026-11-03 a measurement here produced a **5× speedup that was not real**, and
the only reason we know that is a control arm that someone almost did not bother to run.

Applies to both halves of a fold: Boltz-2 prediction on the GPU, and Rosetta scoring on
the CPU. The failure mode is identical in both.

---

## 1. The rule

> **Every timing comparison must include a control arm that changes nothing and occupies
> a different position in the run order.** If the control does not reproduce the baseline,
> the comparison is void and no number from it may be quoted.

Not "should". A timing run without a positional control cannot distinguish the thing being
measured from the cost of being measured first.

## 2. The evidence this rule is built on

`cpseq_kernels_ab.sbatch`, job **95856**, one A6000 on `preempt`, one input, three
configurations run sequentially inside the one job:

| arm | what changed | wall |
|---|---|---|
| A | production, `--no_kernels` | **177 s** |
| B | `PYTHONPATH` adds a shadow dir; **`--no_kernels` still passed** | **47 s** |
| C | shadow dir + kernels enabled | 23 s (`rc=1`, no structure) |

**A and B are the same computation.** Identical flags, identical seed, identical input,
same interpreter. The only difference is that A ran first. The 3.8× is cold-start:
page cache on the 2.2 GB checkpoint, CUDA JIT, model build.

B existed solely to be a null, and it was the most valuable arm in the job.

### What it invalidated

`cpseq_fold_cost_probe.sbatch`, job **95853**, same card, same input:

```
A1  unpatched chunking  ds=1  ->  227 s     (ran FIRST)
A2  patched chunking    ds=1  ->   45 s     (ran second)
```

This was reported as "the chunk-count fix is worth 5×". It is not. Netting out the ~3.8×
positional effect that job 95856 measured leaves the chunk correction worth roughly
**1.3×** — a real but unremarkable gain that was about to be written up as the headline
result.

Note the probe's cost model, `T(n, ds) = F + n·(K + ds·s)`, treats the per-invocation
fixed cost `F` as a constant. That is a *warm* `F`. The model is sound; feeding it a cold
first measurement is not.

### What survived

A2 vs A3 — both patched, both warm, sequential in one job:

```
A2  ds=1   ->  45 s   ( 1 structure)
A3  ds=10  -> 148 s   (10 structures)
```

`s = (148 − 45) / 9 = 11.4 s` per sample; `F + K = 45 − 11.4 = 33.6 s`. Warm-vs-warm, so
quotable. This is what supports the shared-trunk estimate (16 seeds as one invocation:
720 s → 216 s, 3.3×) — and that estimate still needs its own equivalence check before it
is adopted, which is a separate question from timing.

## 3. The same thing on the CPU side

`cpseq_score_cost_probe.sbatch`, job **95812**, `cpu` partition, 10 folds:

```
[arm1] full       10/10 fold(s) in 793.3s =>  79.33s/fold     (ran FIRST)
[arm1] skip_relax 10/10 fold(s) in  62.7s =>   6.27s/fold     (ran second)
```

So the two FastRelax runs are ~92% of `score_structure`, and `validity_report` is a flat
~0.15 s (not the 1.5–2.2 s estimated in `scorer.py`). But **the skip arm ran second**, so
the same positional caveat applies. It is smaller here than on the GPU — PyRosetta
initialises once per process and arm 1 already paid it — but the page cache on the staged
structures is still warm for arm 2, and it has not been quantified. It needs the same
null arm: run `full` again at the end and check it reproduces 79 s.

Pool scaling is worse, because the confound is *monotone with the independent variable*.
Worker counts are measured in ascending order `1, 2, 4, 8, 16`, so every larger pool runs
warmer than the one before it, and warm-up is silently added to the apparent scaling
curve. **Measure worker counts in both ascending and descending order**, or repeat the
first count last. A scaling curve measured in one direction only is not evidence.

## 4. Checklist

Before quoting any timing number:

- [ ] **A positional control arm ran** — same configuration as the baseline, different
      position in the order — and reproduced the baseline. State this explicitly.
- [ ] **The first measurement was discarded or repeated.** The first invocation in a job
      is a warm-up, not a data point.
- [ ] **The comparison is within one job.** Across jobs is not comparable even on the
      same hardware.
- [ ] **No second benchmark job was co-scheduled on the same node.** Jobs 95853 and 95856
      were both put on gpu1 at once; they held separate A6000s but shared CPUs and memory
      bandwidth, so cross-job numbers from that pair (A1's 227 s vs A's 177 s) are
      uninterpretable. Serialise benchmark jobs with `--dependency=afterany`.
- [ ] **The sampled set is random, including within a complex.** `cpseq_score_cost_probe`
      took `rows[:take]` per parent, i.e. the earliest journal rows — which are
      disproportionately the *reference* peptide, folded first. Its sample came out 2/10
      pre-relax-invalid against 60.9% across the full `llm_ladder` journals. That is a
      sampling artefact, not a finding, and it biases every per-fold statistic.
- [ ] **Node, GPU model and the full configuration are on every row.** `sacct` is dead
      here (slurmdbd unreachable, and it still exits 0), so the job's own log and its
      append-only results file are the only surviving provenance.
- [ ] **Which arms are warm-vs-warm is stated in the write-up**, not left for a reader to
      infer.

## 5. Why position costs so much here

Worth understanding rather than just guarding against:

- The Boltz-2 checkpoint is 2.2 GB on NFS. Cold, that read dominates; warm, it is free.
- CUDA kernel autotuning and JIT happen on first use.
- `prefetch_predictions` already exploits this deliberately: boltz costs "44 s per
  invocation plus 26 s per structure", which is why batching 17 structures per invocation
  is worth it. The per-invocation cost that batching amortises is *the same cost* that
  contaminates a first measurement. We were already relying on the effect and still got
  caught by it.

## 6. Scope

Every `*_cost_probe`, every A/B of a setting, and any claim of the form "X is N× faster".
It does not apply to correctness checks, which have no run-order dependence — but note
that an *equivalence* check (does config B produce the same distribution as A?) is a
correctness check and must not be conflated with the timing arm that shares its job.
