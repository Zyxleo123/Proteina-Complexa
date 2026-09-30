# CP → LP generator

Manufacture paired `(LP, CP, target)` triplets by turning bound **cyclic** peptides into
bound **linear** ones, so the LP→CP direction has supervision.

## Why this exists

The LP→CP project needs pairs: a linear peptide and the cyclic peptide it should become,
both bound to the same target. Two earlier routes to that pair are closed:

- **Provenance.** CPSea rows can be traced back to the AFDB fragment they were cut from,
  exactly (verified on all 2.44M rows). But those fragments are *already closed* —
  pre-cyclization terminal gap median 6.95 Å, travel 0.76 Å. The pair is an identity map,
  so it carries no transport signal.
- **Constructed corruption.** Pulling a ring open geometrically gives usable geometry, but
  the source sequence equals the target sequence, so the anchors are a leaked hint and the
  mutation budget is unsupervised.

What is left is to *generate* the LP side, and to make it look like a real bound linear
peptide rather than like a macrocycle with one bond deleted. That last clause is the whole
problem, and it is what the adversarial term is for.

## The checkpoint pairing — read this before changing anything

```
AE   (defines local_latents) : shared_ae_lpcp_128/frozen_ae.ckpt
flow (initialisation)        : cpsea_lpmix_from_v4cfg/cp2lp_init.ckpt
```

`local_latents` is **defined by the autoencoder**. A flow checkpoint trained against a
different AE emits latents in a different space, so decoding them with this AE produces
meaningless geometry from step zero — and every loss here (adversarial, contact, clash,
geometry) is computed on decoded all-atom coordinates, so nothing would catch it except
the samples being garbage.

This is why the init is the **LP-mixing** run and not `cpsea_bondunroll_pin20260828`,
which is otherwise the stronger CPSea checkpoint. Bondunroll was trained against
`finetune_full_128`, a CPSea-only AE that has never encoded a linear peptide. The lpmix
run is the only CPSea flow checkpoint trained against the shared LP/CP AE, and it
additionally arrives with the **LINEAR topology token already trained**, which this arm
depends on.

`cp2lp_init.ckpt` is a **pin** (`scripts/pin_cp2lp_init.sh`), not a live path: the lpmix
run may still be training and rewrites `last-EMA.ckpt` every 1500 steps, so two arms
launched an hour apart would otherwise start from different weights. The `.source` sidecar
records which checkpoint and when.

## How it works

**Source-CP conditioning.** The source CP is encoded once and frozen into `x_src_cp`,
which has the same tensor layout as `x_sc` — so the existing self-conditioning feature
classes read it verbatim under a different `mode_key`, and no new feature implementation
was needed. It is a *persistent condition*, not the evolving state:

| key | meaning |
|---|---|
| `x_t` / `x_sc` | the evolving LP state, integrated from noise to t=1 |
| `x_src_cp` | the frozen encoded source CP — the thing being edited away from |
| `src_cp_present` | explicit `[b]` flag, because zero is a legal coordinate |

**Topology.** The generated state is stamped `LINEAR`, which is an explicit request for
*no ring* and distinct from `UNSPECIFIED` (the CFG null). LINEAR makes the ring positional
encoding inactive and makes the bond loss skip the row, so nothing in the objective pulls
the generated termini back together.

**Differentiable rollout.** GAN updates integrate a short trajectory (`nsteps: 24`) with
gradients kept through the last `grad_steps` network evaluations — DRaFT-K, the same
mechanism the cyclization rollout fine-tune already uses. Export uses the full 200-step
deployed sampler; nothing should be exported from a 24-step rollout.

**Three terms in one optimizer step.**

1. *Real-LP flow matching* on PepBench/ProtFrag with the source condition absent. The
   anchor — the only term on real data, and what keeps the trunk producing LP geometry
   rather than whatever the discriminator currently rewards.
2. *CP-conditioned generation* scored by adversarial + contact + sequence + geometry +
   clash + opening losses.
3. *Discriminator* hinge update on length-matched real/fake pairs.

## The shortcut risk — the first number to look at

Real LPs sit on PepBench receptors; generated ones sit on CPSea receptors. A discriminator
handed rich receptor features can separate the two **perfectly without ever looking at the
peptide**, and the generator then receives gradient that is noise with respect to LP
quality.

Three defences, and one instrument:

- no global receptor description reaches the network — no length, no whole-chain
  composition, no absolute coordinates, no chain identity. Only the `k` nearest pocket
  residues per peptide residue, as distances plus identity.
- pocket identity defaults to a **5-way coarse chemistry class**, not 20-way residue type.
- real and fake are matched **by peptide length**; unmatched fakes are dropped rather than
  paired across lengths.
- **`train/cp2lp_d_shortcut_probe_acc`** — a head reading the interface block with the
  peptide masked out. Read it against `d_acc`, **not against 0.5**: the diagnostic quantity
  is the *gap*. A probe at 0.65 under a `d_acc` of 0.96 means the receptor carries ~15
  points over chance and the peptide contributes another ~30 on top, which is the regime we
  want. If the gap closes, drop `discriminator.pocket_identity` to `none` before touching
  anything else, and trust the contact-only arm's triplets. A single logged value is
  quantised (~8 examples per D step, so 0.75 is 6/8) and is never evidence of a trend — take
  running means.

Both real and generated peptides are shown to the discriminator as `autoencoder.decode`
output, so it cannot separate them on AE processing artefacts — a real LP is encoded and
decoded before it is shown, exactly like a generated one.

**That defence is incomplete, and the hole is the thing that actually broke the GAN arm.**
It equalises the *autoencoder*, not the *sampler*: a generated peptide additionally passes
through the 24-step differentiable rollout, and a real one does not. See
[the geometry tell](#the-geometry-tell--why-the-adversary-cannot-work-yet) — the resulting
artefacts are ~70x larger than anything the AE contributes, and they separate the two
classes perfectly. There is no probe for this; `shortcut_probe` masks the peptide, so it is
blind to a tell that lives *in* the peptide.

## Running it

```bash
# 0. metadata with receptor families kept apart (already built; rebuild to change eval_frac)
bash scripts/submit_cp2lp.sh --submit splits

# 1. preflight: do the checkpoints load together, does one batch survive
#    generator -> AE -> discriminator -> backward
bash scripts/submit_cp2lp.sh --submit smoke

# 2. both arms as siblings, so they queue concurrently
bash scripts/submit_cp2lp.sh --submit both

# 3. triplets from each arm (same CPs, same budget), plus the real reference
bash scripts/submit_cp2lp.sh --submit generate
bash scripts/submit_cp2lp.sh --submit reference

# 4. one table + figure comparing the arms against real
bash scripts/submit_cp2lp.sh --submit report

# or the whole chain with dependencies wired:
bash scripts/submit_cp2lp.sh --submit pilot
```

Configuration travels as one timestamped env file passed as a **positional** argument —
never `sbatch --export`, which sets `SLURM_GET_USER_ENV=1` and gets the job requeued and
held. Override with `--set KEY=value`.

## The ablation

`training_cp2lp_contactonly.yaml` differs from the GAN config in **one line**
(`cp2lp.adversarial.enabled`). Same init, same AE, same data, same conditioning, same
remaining losses, same rollout, same seed.

Without an adversary, nothing tells the generator what a real bound LP looks like. The
remaining terms say: keep the CP's contacts, keep its sequence, keep the backbone sane, do
not clash, get the termini apart. A peptide can satisfy all of that and still be an
obviously artificial object. Whether the adversarial term prices that is the question.

Read `terminal separation` and `min peptide-pocket distance` against the **real** column
first — those are the two the adversary should move. Then `CP contact retention`, which
both arms optimise directly and which should be roughly equal if the ablation is clean.

## Export gate

Stricter than the training losses. A sample is rejected unless:

- the decoded argmax sequence **equals** the source CP's sequence (identity is a premise
  of the task, not a tradeoff);
- the chain is chemically linear — C(L-1)–N(0) outside any bond window **and** the termini
  genuinely separated, so nothing downstream infers a cyclic connection;
- the backbone is intact.

**The backbone criterion is calibrated, and it passes real crystal peptides.** Its tolerance
is 0.02 nm on *every* `(i, i+1)` bond, which looks punishing until you measure it: real LPs
through the same code have a median bond deviation of 0.0033 nm and 97% of them (520/536)
clear the all-bonds test. So a `chain_intact` of 0 is a statement about the sample, not
about the gate — unlike the clash threshold, which had to be loosened because it rejected
43 of 60 natives.

Do **not**, however, read `geom_chain_intact_frac` off the *training* logs as an export
forecast. There it is measured on the deliberately short 24-step rollout, whose endpoint
carries bond errors ~70x larger than the AE contributes, so it reads 0 by construction. The
export path runs the full 200-step sampler; only that number is a forecast.

Rejected samples are still written to the manifest with a reason. The acceptance rate is
one of the numbers that says whether this works at all, and a silent drop makes it
unknowable. Accepted exports get an **OXT** built on the C-terminal residue: the decoder
never emits one (a macrocycle has no free C-terminus), and a carboxylate written without
it is read by parsers as amidated.

## Hardware

**a6000 (48 GB), pinned in `cp2lp_train.sbatch`.** A CUDA allocation failure inside a
rollout can kill the process with no Python traceback, which reads as a silent `FAILED`,
so the GPU class is not left to the scheduler.

**Peak memory is not constant across batches — this is the trap.** Pair features are
O(n_target²) and the receptor crop runs to 256 residues, so a batch of large receptors
costs several times one of small ones. Measured:

| load | result |
|---|---|
| 4 rollouts (max_cp 2 × samples 2), grad_steps 2 | OOMs a 24 GB a5000 (job 68250) |
| 8 rollouts (max_cp 4 × samples 2), grad_steps 2 | one Trainer step passes, then OOMs a **48 GB** a6000 after 8 min of real training, at `manual_backward` (job 68347) |
| 4 rollouts (max_cp 2 × samples 2), grad_steps 1 | current setting |

The knobs, in the order to turn them: `cp2lp.max_cp_per_step`, then
`cp2lp.rollout.grad_steps`, then `samples_per_cp` (last — dropping it to 1 kills the
within-CP diversity metric). The job also sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the OOM reported 1.62 GiB
reserved-but-unallocated, i.e. lost to fragmentation rather than in use.

**Any memory knob is part of the ablation's fixed config.** Tuning it on the GAN arm alone
stops the comparison isolating the adversarial term — cancel and relaunch both.

The smoke runs several Trainer steps at the configured load and reports peak allocated vs
total with a headroom warning, because a single passing step measures one lucky draw.

## Calibration notes carried over

- **Clash threshold is 0.20 nm, not 0.27.** The larger value rejects 43 of 60 native
  crystal complexes, so training against it pushes away from native-like packing.
- **Geometry is scored with the terminal bond absent.** Including the `(L-1, 0)` closure
  would tell the generator the ring it was asked to open is a defect to repair.
- **Features are appended at the END of each list.** Each feature factory concatenates in
  list order and projects with one Linear, and `_splice_pretrained_weights` copies the
  overlapping leading subtensor — so appending preserves every existing feature's weights
  and zero-initialises only the new columns. Inserting anywhere else silently re-binds
  trained weights to the wrong inputs.

## Run history and what it measured

Three generations so far. **The contact-only arm has won every comparison**, and the
adversarial arm has never beaten it on the metric they share.

### v1 (jobs 68417 / 68418, ~40k steps) — the ablation ran, and the adversary lost

Contact retention, windowed means:

| steps | GAN | contact-only |
|---|---|---|
| 0–2k | 0.164 | 0.179 |
| 10–12k | 0.140 | 0.150 |
| 20–22k | 0.164 | 0.238 |
| 37–39k | **0.158** | **0.435** |

The section above says the two should be "roughly equal if the ablation is clean". They are
not, and the GAN arm is not merely *slower* — at 39k it sits below its own starting value.
`contact_n_source` is 111.9 vs 111.6, so both arms saw statistically identical source CPs
and this is not a data difference.

### Why: the loss terms compete for a fixed gradient budget

The smoke's own per-term attribution into the flow network (job 68421) had already measured
it, and nobody had read it against a run:

| term | max‖g‖ | weight |
|---|---|---|
| sequence | **2.40** | 0.5 |
| geometry | 5.8e-2 | 0.5 |
| **contact** | **2.3e-2** | **1.0** |
| clash | 1.8e-2 | 0.5 |
| open | 1.3e-2 | 0.2 |

Contact carries the largest weight and a hundredth of sequence's gradient. That would be
survivable if the terms simply summed — but the generator's gradient is clipped to a fixed
norm (1.0, per optimizer, in `_step_optimizer`), and `train/cp2lp_grad_norm_preclip_g` runs
**11–52** against it. The clip is not an occasional safety net, it is the permanent
operating regime, so every term's effective step is scaled by `clip / norm` and the largest
term decides what the smallest gets.

The timing confirms it: in the contact-only arm retention only started moving *after*
sequence saturated (exact match 0.894 by 10–12k, then 0.150 → 0.238 → 0.435). In the GAN
arm the adversary took the freed budget instead — and that arm's own sequence converged
*slower* (0.698 vs 0.894 at 10–12k), so the adversary was slowing everything, with contact,
the weakest term, squeezed to zero progress.

`hinge_g_loss` is `-fake_logits.mean()`, the **non-saturating** form: constant gradient −1
w.r.t. the logit however confident D is. The hinge that vanishes is D's own, and in v1 both
its margins were passed (real +1.52, fake −1.83, `d_acc` 0.996), so D had largely stopped
learning while G kept pushing at full magnitude against a frozen critic — `loss_adv_g` rose
1.72 → 1.92. G was not starved; its gradient was live and unhelpful. Note that
`relu(1 - fake_logits)` would be a **no-op** fix at this operating point, since it has
gradient −1 everywhere below +1. The levers are `adversarial.weight` and `d_lr`.

### v2 (jobs 68914 / 68915) — a rebalance that regressed. Reverted.

Four changes: `adversarial.weight` 1.0 → 0.3, `d_lr` 2e-4 → 5e-5, `contact_sharpness_nm`
0.1 → 0.3, and a new `losses.sequence_decay`. Step-matched at 28–30k (v1's headline 0.435
was at 39k, so anything else would be an unfair comparison):

| metric | v1 gan | v1 ctl | v2 gan | v2 ctl |
|---|---|---|---|---|
| contact_retention | 0.162 | **0.349** | 0.182 | **0.185** |
| peptide_bond_mae_nm | 0.242 | 0.223 | **0.515** | **0.328** |
| clash_min_inter_nm | 0.074 | 0.038 | 0.046 | 0.052 |
| terminal_gap / seq_exact / real_lp_flow | — unchanged across all four — |

+0.02 on the GAN arm, −0.16 on the control, and 1.5–2.1x worse backbone geometry in both.
Both runs were clean (no requeue, OOM or non-finite gradient), so this is the config.

**The error was the sharpness.** Widening the soft-contact sigmoid widens the gradient's
support but *also softens the penalty*, and the softening wins. `lost = relu(c_src - c_gen)`,
and a wider sigmoid raises `c_gen` for a drifted pair while lowering `c_src` for the original
contact. For a source contact at 0.4 nm, loss at s=0.3 as a fraction of loss at s=0.1:

| drift | 0.9 nm | 1.2 nm | 1.5 nm | 2.0 nm |
|---|---|---|---|---|
| ratio | **0.52x** | 0.60x | 0.72x | 0.79x |

Worst exactly at the near distances where the recoverable contacts are. Widening the support
needs a term whose *magnitude* does not shrink with width — a distance hinge — not a wider
sigmoid.

The geometry regression is **not attributable** from this data: it hit both arms, and both
changes that reach the control (sharpness *and* `sequence_decay`) are present in both, so
they cannot be separated without a third arm. `sequence_decay` is the suspect on mechanism —
the sequence term may have been implicitly holding the latents in regions the decoder can
render. Both reverted together; the decay code stays, default-off, for a later isolated test.

What survived: the decay fired exactly as designed (`seq_weight_scale` 1.0 → 0.65, gradient
norm halved 21 → 11, confirming sequence was the dominant term), and the adversarial
rebalance hit its own target (shortcut probe 0.738 → 0.625, `d_acc` − probe gap 0.258 →
0.375). `adversarial.weight: 0.3` and `d_lr: 5e-5` were therefore **kept**.

### The geometry tell — why the adversary cannot work yet

`d_acc` reached **0.9995** and cutting `d_lr` fourfold did nothing, because the
discriminator is not working hard. Backbone bond deviation, real vs generated:

| | n | min | median | max |
|---|---|---|---|---|
| **real** LPs (reference job) | 536 | 0.0010 | 0.0033 | **0.0205** |
| **generated** (gan v1) | 40,133 | **0.0683** | 0.2621 | 1.9255 |
| **generated** (ctl v1) | 40,255 | **0.0581** | 0.2023 | 0.6280 |

The distributions are **completely disjoint** — the generated minimum is 3.3x above the real
maximum, and 0 of 40,133 batches ever fall below it. A single scalar threshold separates the
classes with 100% accuracy on every batch of the whole run. **The adversary is a
broken-backbone detector**, and no amount of weakening D changes that.

The cause is structural rather than a bug: fakes come from the 24-step differentiable
rollout, reals from a clean AE round-trip of a crystal structure. Two ways out, and the open
question is which:

1. **Make the paths symmetric** — noise real LPs to some `t` and re-integrate them through
   the same sampler, so both sides carry identical rollout artefacts. This is the same
   "same processing on both sides" principle already applied to the AE, and it is cheap.
2. **Fix the geometry** — if the generator's output is bad at *every* step count, the
   adversary is right to reject it and nothing is repairable until that is fixed.

`--submit generate` on a v1 checkpoint runs the full 200-step export sampler and decides
between them. If its bond MAE lands near the real distribution, only the short training
proxy is broken and (1) applies.

### The real column, measured

From `--submit reference` (n=536 real bound LPs, same code path as the generated ones):

| quantity | real | v1 gan | v1 ctl |
|---|---|---|---|
| `chain_intact` | **97%** (520/536) | 0 | 0 |
| `peptide_bond_mae_nm` | **0.0033** | 0.242 | 0.223 |
| `clash_min_inter_nm` | **0.262** | 0.074 | 0.038 |
| `terminal_gap_nm` | **2.14** | 1.75 | 1.78 |

Two things follow. **The export gate's 0.02 nm tolerance is correctly calibrated** — natives
sit 6x inside it — so this is *not* a repeat of the clash-threshold story and `chain_intact`
is a usable criterion. And both arms are under-opened and packed too tight against the
receptor, the control markedly so (0.038 vs 0.262, i.e. it buys its retention with clashes).
That last point partly exonerates the adversary: the GAN arm's drift to 0.074 was moving
*toward* native, which is why v2 rebalanced it rather than removing it.

Do not read `n_accepted: 0` from a reference run as a rejection — `write_real_reference`
returns `n_real` and never populates the acceptance counter that `cp2lp_generate.py` prints.

### Current state

- **Contact-only continues from v1's checkpoint** (`RUN_NAME=cp2lp_contactonly_v1`), with
  the reverted objective, which is bit-identical to what v1 trained under. It was still
  climbing at 40k and is the arm whose triplets we would actually use.
- **The GAN arm is a measurement, not a training run**, until the geometry tell is resolved.

Resuming is deliberate: warm-starting each arm from *its own* v1 weights would make the arms
differ by 40k steps of divergent training rather than by one config line.

## Traps found the hard way

- **`cp2lp_train.sbatch` passes `++run_name=${RUN_NAME}`, a Hydra override that beats the
  yaml.** Bumping `run_name` in a config is therefore not enough — the preamble's own
  `RUN_NAME` defaults must be bumped with it, or the new config writes into the *old* run
  directory and overwrites the checkpoints and wandb history of the arm you are comparing
  against. Guarded by `test_preamble_run_names_track_the_configs`.
- **`--submit generate` launches both arms, and `--set RUN_NAME=...` applies to the whole
  env file**, so it will point *both* at one checkpoint. `OUT` is keyed by `ARM`, so there
  is no file collision — just silently mislabelled triplets. Set the run name per arm.
- **The `reference` and `generate` stages both repoint the metadata at a single-source
  parquet**, but the weighted sampler is built from the train metadata and
  `source_fractions` must name exactly the sources present — so the inherited three-way
  training mix raised during `setup()` before the model even loaded. Fixed by clearing
  `dm.source_fractions`; nothing on that path trains.
- **A checkpoint being written by a live job is not safe to read.** The contact-only
  generate has to wait for a pinned copy while its arm is training, the same way
  `cp2lp_init.ckpt` is pinned.
- **The contact-only config inherits the GAN one** (`defaults: - /example/training_cp2lp_gan`),
  overriding only `run_name` and `adversarial.enabled`. Edit the GAN yaml and both arms move
  together — which is what keeps the ablation single-variable for free, and what
  `test_contactonly_differs_from_gan_in_exactly_one_knob` enforces.


## Files

| path | role |
|---|---|
| `src/proteinfoundation/cp2lp/conditioning.py` | `x_src_cp`, LINEAR topology request |
| `src/proteinfoundation/cp2lp/data.py` | role split, length-matched reservoir, family split |
| `src/proteinfoundation/cp2lp/discriminator.py` | realness logit + shortcut probe |
| `src/proteinfoundation/cp2lp/losses.py` | contact / sequence / geometry / clash / opening |
| `src/proteinfoundation/cp2lp/geometry.py` | open-chain primitives, all in nm |
| `src/proteinfoundation/cp2lp/module.py` | `CP2LPGenerator` LightningModule |
| `src/proteinfoundation/cp2lp/export.py` | triplet writer + real-LP reference |
| `script_utils/cp2lp_build_splits.py` | receptor-family-disjoint metadata |
| `script_utils/cp2lp_smoke.py` | preflight incl. per-term gradient attribution |
| `script_utils/cp2lp_generate.py` | triplet generation / reference measurement |
| `script_utils/cp2lp_report.py` | real-vs-generated table + figure (CPU) |
| `script_utils/test_cp2lp.py` | unit tests for the pure pieces |
