# LP-mixing -- can the existing CPSea AE be reused?

A 2-hour gate in front of a 2-GPU-day stage. CPU-cheap to read, GPU-cheap to run, and it
answered no.

Run it with `bash scripts/submit_lp_mixing.sh --submit ae-roundtrip`.

---

## 1. The decision

The LP-mixing flow arms need an autoencoder checkpoint to pin. The pipeline was built to
train a shared LP+CP autoencoder (`shared_ae_lpcp_128`), which costs ~2 GPU-days on the
critical path before the two 2-day flow arms can even start.

The alternative is to pin the existing `finetune_full_128` for **both** arms. That keeps
mix-vs-control single-variable, and additionally makes the arms comparable to the whole
v4/bondunroll lineage, which regressed its latents against exactly that checkpoint.

It is only sound if that AE actually represents linear peptides. If it does not, reusing it
makes **"LP mixing does not help" indistinguishable from "the AE cannot encode LP"** -- the
one confound that would waste the entire experiment. So: measure, do not assume.

## 2. What is measured

`diagnose_ae_latents.py` is run **twice under one AE checkpoint** -- once on CPSea metadata,
once on LP metadata -- and `lp_ae_roundtrip_compare.py` compares the two JSONs.

Both passes export the same `CPSEA_AE_CKPT_PATH`, which is what the Hydra config resolves
for `autoencoder_ckpt_path`. The comparator **refuses** (exit 1) if the two JSONs record
different `ae_ckpt` values, because then the comparison measures the checkpoints rather than
the corpora.

No subsetting is needed: LP-only and CPSea-only metadata already exist as separate parquets.
The job loads only `AutoEncoder`, never trains, and never touches the flow model -- so the
`NUM_CYCLIZATION_COND_TYPES` 4->5 checkpoint-pad issue does not apply here.

### The comparison is length-matched, and that is not cosmetic

Reconstruction error grows with peptide length, and the two corpora do not share a length
histogram. A pooled difference therefore mixes "the AE is worse on linear peptides" with
"the linear peptides are longer", and the pooled number can carry the opposite sign to the
per-length one.

So the headline reweights both corpora onto a shared length histogram, with weight per
length = `min(n_cpsea, n_lp)`. Neither corpus's length profile can then drive the result.
The pooled numbers are printed beside it to show the distortion.

## 3. The verdict rule, and the bug that inverted it

Three gates, in order:

1. **Power.** `effective_n < MIN_EFFECTIVE_N` (60) -> `UNDECIDED_UNDERPOWERED`. A smoke is
   12 examples; it must not be able to authorise 2 GPU-days.
2. **Absolute ceiling.** LP above `ABS_CEILING_A` (2.00 A) -> `TRAIN_SHARED_AE`. The ratio
   is beside the point; the latents do not carry the structure at all.
3. **Absolute floor.** LP at or below `ABS_NEGLIGIBLE_A` (0.30 A) -> the ratio test is
   **suppressed** and the verdict is `REUSE`. A ratio between two negligible numbers reports
   a large effect that means nothing. Otherwise the ratio decides, against
   `RATIO_TOLERANCE` (1.30).

`ABS_NEGLIGIBLE_A = 0.30` is set against what *consumes* these latents, not against the AE:
ring-bond windows calibrated on natives are 3.5-10 A CA-CA, bond-distance success is judged
at ~1.3 A, and downstream closure thresholds sit at 1-2 A. An AE reconstructing to 0.3 A is
nowhere near binding on any of them.

> **The bug, recorded because it points the way every such bug points.** On its first real
> run this gate printed `REUSE_FINETUNE_FULL_128`. The suppression test was reading the
> **pooled all-atom median (0.127 A)** while the ratio it was suppressing was built on the
> **length-matched sidechain value (0.320 A)**. A pooled median is robust to exactly the
> per-length tail that is the problem, so it hid the tail and licensed skipping 2 GPU-days.
>
> Two lessons. An absolute escape hatch must be judged on **the same statistic as the ratio
> it overrides**. And a threshold introduced after seeing data will tend to fire in the
> convenient direction -- `ABS_NEGLIGIBLE_A` was added after the smoke, and it did.
>
> Fixed: both absolute tests now read the length-matched value, and the output prints the
> worst per-length values plus a count above `ABS_NEGLIGIBLE_A`, so a tail can no longer
> hide behind an average.

Both real verdicts exit 0 -- this is a decision gate, not a failure, and a non-zero exit
would strand dependents in `DependencyNeverSatisfied` with no explanation. Only the
different-checkpoint misconfiguration exits 1.

`ae-roundtrip` is deliberately **excluded from `--submit all`**: its whole purpose is that a
human reads the verdict before committing the 2 days.

## 4. Result: train the shared AE

*Job 66045, 26 September 2026, effective n = 234, AE = `finetune_full_128` step-40000 EMA.*

| statistic | cpsea | lp | ratio |
|---|---|---|---|
| length-matched sidechain RMSD | **0.058 A** | **0.320 A** | **5.48x** |
| pooled all-atom median | 0.047 A | 0.127 A | 2.72x |
| posterior sigma^2 | 0.0014 | 0.0013 | -- |
| `kl_active_frac` median | 1.000 | 1.000 | -- |

The shape matters more than the ratio. **CPSea is uniform across every length (0.048-0.089);
LP is erratic (0.13-1.01).** Worst LP lengths: len 7 = **1.014 A**, len 9 = 0.677, len 13 =
0.452, len 14 = 0.363 -- **4 of 12 lengths above 0.30 A**.

Note what did *not* detect this: posterior sigma^2 and `kl_active_frac` are healthy and
essentially identical on both corpora. Only reconstruction sees it. A latent-statistics
check alone would have passed this AE.

**Verdict: `TRAIN_SHARED_AE`.** Sidechain precision is exactly what isopeptide closure
spends -- when the anchor residues are right, NZ-CG is already at ~1.3 A -- so a 0.32 A mean
with a 1.0 A tail is a real fraction of that budget.

### An open hypothesis, not a finding

The AE training log shows 42 binder-chain warnings, **all from `protfrag`**, of which **6
have `REAL_BREAK=1`** -- physically discontinuous "linear peptides", i.e. two fragments fed
in as one contiguous binder. Rate is roughly 0.1-0.4% of protfrag samples seen (imprecise:
it is unclear whether `global_step` counts batches or optimizer steps under
`accumulate_grad_batches=4`).

That is small enough not to threaten AE training, but it could account for part of the LP
reconstruction tail -- the len-7 outlier at 1.014 A is the shape a severed fragment would
produce. If so, part of the 5.48x gap is bad data rather than AE incapacity. Testable
cheaply by re-running the roundtrip with the warned ids excluded. **Untested so far; do not
cite it as a cause.**

## 5. Running it

```bash
# the gate (2 h, one a6000 -- pick the node from a fresh `myfree`)
bash scripts/submit_lp_mixing.sh --submit ae-roundtrip --gpu-nodelist <node>

# smoke: 3 batches x 4, into its OWN out dir
bash scripts/submit_lp_mixing.sh --submit ae-roundtrip --gpu-nodelist <node> --smoke
```

The smoke writes to `..._smoke/`. Both passes use fixed filenames
(`ae_latent_diag_{cpsea,lp}_train.json`), so sharing one directory risks the comparator
reading one corpus from the smoke and the other from the real run.

Knobs (via `--set VAR=value`): `RT_AE_CKPT`, `RT_CONFIG`, `RT_BATCHES`, `RT_BATCH_SIZE`,
`RT_WORKERS`, `RT_OUT_DIR`.

Outputs, under `$PROTEINA_ZFS_PATH/training_runs/diagnostics/lp_ae_roundtrip/`:

```
ae_latent_diag_cpsea_train.json   per-corpus diagnostic + by_length breakdown
ae_latent_diag_lp_train.json
ae_latent_diag_*.png
verdict.json                      verdict, thresholds, length_matched, per_length
```

`verdict.json` can be recomputed from the two JSONs without a GPU, which is how the first
run was re-scored after the threshold fix:

```bash
.venv/bin/python script_utils/lp_ae_roundtrip_compare.py \
  --cpsea-json <dir>/ae_latent_diag_cpsea_train.json \
  --lp-json    <dir>/ae_latent_diag_lp_train.json \
  --out        <dir>/verdict.json
```

## 6. State of the LP-mixing arm

- Mixed metadata **is built**: 2,489,756 train rows -- cpsea 2,444,209 / protfrag 42,123 /
  pepbench 3,424. LP is 1.8% naturally; the weighted sampler lifts it to 25%
  (`cpsea 0.75 / pepbench 0.06 / protfrag 0.19`). The `data` stage does not need re-running.
  Do not mistake `preprocessed/metadata/lp_*.parquet` (the LP-only side) for
  `preprocessed/metadata_mixed/` (the mix).
- `_lp_preflight_train` **passes**: all three sources present, sampled paths resolve.
- `--submit verify` **passes**: LINEAR=4 on every pepbench/protfrag row, cyclic rows carry
  0/2, all binders 5-16 residues. The full transform stack on LP rows is verified.
- `--submit ae` is the live stage, per this gate's verdict. It warm-starts from
  `complexa_ae.ckpt` -- the **pre-CPSea base**, not `finetune_full_128` -- so flow arms
  trained on the resulting AE are **not** comparable to the v4/bondunroll lineage.
- Then `flow-mix` and `flow-control` as siblings. Submitting one without the other produces
  an uninterpretable result.
