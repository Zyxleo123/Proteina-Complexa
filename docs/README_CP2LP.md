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

## Two generator modes

`cp2lp.mode` selects between them. They share the AE, the losses, the discriminator and
the export gate, and nothing else.

| | `onepass` (the direct editor) | `rollout` (legacy) |
|---|---|---|
| network evaluations per sample | **1** | 24 in training, 200 at export |
| `t` | fixed constant (0.5), an architecture input | integrated from 0 to 1 |
| what the readout emits | an edit **displacement** from zero-init heads | a flow velocity |
| real-LP flow term | off | on |
| training vs export path | the same function | different samplers |
| status | built and unit-tested; **not yet GPU-run** | three generations measured |

`onepass` is what the five-arm comparison below uses. `rollout` is kept unchanged so the
measured v1/v2 history stays reproducible; it is **not** part of that comparison, because
its generator objective differs too and it would be an architecture-plus-objective change
wearing one label.

## The direct one-pass editor

```
delta_X, delta_H = editor(X_src, H_src, target, sequence, z)
X_lp = X_src + delta_X
H_lp = H_src + delta_H
decoded_lp = frozen_AE.decode(X_lp, H_lp)
```

**Why it exists.** The rollout arm's adversary could not work, and the reason was
structural rather than a tuning problem — see [the geometry tell](#the-geometry-tell--why-the-adversary-cannot-work-in-rollout-mode).
Generated peptides came out of a 24-step differentiable trajectory and real ones out of a
clean AE round trip, so their backbone-bond distributions were **completely disjoint** and
a single scalar threshold separated the classes on every batch of a 40k run. The one-pass
editor removes the asymmetry instead of correcting for it: both classes now pass through
exactly one AE round trip and nothing else.

**The residual is a displacement, and that conversion is explicit.** The pretrained
coordinate head emits a flow *velocity*; reading it as an edit would start the editor at
`x_src + v(x_src, 0.5)`, an arbitrary jump. So the trunk gains its own pair of readouts
(`nn.enable_editor_residual`), zero-initialised, under a separate `delta` key that nothing
reading `v` can pick up. MEASURED on the real trunk: at initialisation `|delta| = 0`
exactly, so the first output is the AE round trip of the source CP — the identity edit.

**`t` is an architecture input, not a sampling coordinate.** It stays plumbed through
because removing the time embedding would put every pretrained layer out of distribution,
and it is pinned at `cp2lp.onepass.t_fixed` (0.5). Nothing schedules it, nothing integrates
over it, and there is no velocity-to-clean conversion anywhere on this path.

**Randomness is an explicit input.** One deterministic evaluation gives one LP per CP.
`editor_noise_seq` supplies 16 per-residue **invariant** Gaussian channels, appended last
in `nn.feats_seq`. Invariant and not a random 3-vector for a specific reason: the trunk
consumes absolute coordinates and is only rotation-*augmented*, so it cannot distinguish a
Cartesian perturbation from real geometry and would read the noise as structure.

**The noise adapter is initialised after the splice, and that ordering matters.**
`_splice_pretrained_weights` zero-fills an appended feature's columns, and the residual
heads are zero too — so on step one the trunk gets no gradient at all and the noise would
only become readable once the heads had moved. `train.py` re-initialises just the noise
columns with a small draw immediately after loading the init. MEASURED: the adapter lands
at columns `[91, 107)` of 107, i.e. exactly the last 16, confirming the append-at-end
invariant holds through the resolved config.

**On equivariance, honestly.** The trunk reads absolute CA coordinates through a `Linear`
and is rotation-augmented, not rotation-equivariant; translation is factored out upstream
by `CenteringTransform(center_mode=target)`. So "the residual rotates as a vector" is not a
property this architecture has, and a test asserting it would be wrong. What is enforced is
that the editor is **no more frame-sensitive than the pretrained velocity head**, measured
on the same pair of forwards. MEASURED on a reduced CPU trunk over three rigid motions:
ratio **0.93x / 1.09x / 1.10x**, and the noise tensor is bit-identical across frames. Read
only the ratio — the absolute deviations there (~1.4 relative) come from a randomly
initialised 2-layer network and say nothing about the trained model. The GPU smoke reports
the same ratio against the real trunk and fails above `--frame-tol` (default 1.5x).

**The real-LP flow term is off in this mode, and nothing replaces it.** It supervises a
velocity field that the residual heads are not connected to. No identity loss, no
distillation, no reconstruction objective was substituted. Real bound LPs still arrive
every batch and are still the discriminator's real class. The module *raises* if
`real_lp_flow_weight` is nonzero with `mode: onepass`.

## Generator updates, not `global_step` — and this changes how the v1/v2 tables read

MEASURED directly (`lightning 2.5.6`, manual optimization, accum 8, 64 batches):

| config | G steps | D steps | `trainer.global_step` |
|---|---|---|---|
| one optimizer | 8 | 0 | 8 |
| two optimizers | 8 | 8 | **16** |

`global_step` is the number of `optimizer.step()` calls **summed over optimizers**
(`lightning/pytorch/loops/optimization/manual.py` increments on every lightning
optimizer's `_on_after_step`). The project's EMA callback makes no difference. This arm
steps two optimizers, so an adversarial arm advances `global_step` twice as fast as the
control — and in v1 `train/cp2lp_d_match_frac` had mean **0.9999** over 40,133 logged
values, so the GAN arm stepped D on essentially every boundary.

**So the v1/v2 comparisons were never generator-update-matched.** Re-indexing v1's table:

| G updates | GAN | contact-only |
|---|---|---|
| ~10–11k | 0.164 | 0.150 |
| ~19–20k | 0.158 | 0.238 |
| ~38k | *never reached* | 0.435 |

The headline "0.158 vs 0.435" compared ~19k generator updates against ~38k. At matched
updates the GAN arm is slightly *ahead* around 10k, and the control's decisive 0.435 sits
at a budget the GAN arm never reached. **"The adversary lost" is not supported by that
data** — it is a statement about two different amounts of training. The claim is left in
the v1 section below as it was written, with this correction attached.

The one-pass arms therefore count generator updates themselves, in a persistent buffer:
`cp2lp.max_g_updates` ends the run, `cp2lp.eval_checkpoint_g_updates` writes retained
milestone checkpoints, the adversarial warmup is gated on it, and `train/cp2lp_g_updates`
is logged so every plot can be x-axed on the comparable quantity. The counter is inherited
from the fork, which is why the arms' budget reads `10300`.

**MEASURED: the fork was pinned at `g=250`, not 300** (pin job 69416). The warmup stopped
correctly at 300 and wrote `g000300.ckpt`, but `cp2lp_fork_pin.sbatch` hardcoded
`last.ckpt` and passed it as an explicit argument, which overrode `pin_cp2lp_fork.sh`'s
milestone preference — the helper was fixed and its CALLER was not. `last.ckpt` is written
on a *global_step* cadence (every 250), hence 250.

Consequences, in order of how much they matter: **none for the comparison** — all five arms
load the same file, so they share an identical initialisation and an identical budget. The
arms therefore get **10,050 post-warmup updates, not 10,000**, and their milestones land at
1,050 / 5,050 / 10,050 post-warmup. Quote those numbers, not the round ones. The caller is
fixed for the next fork and guarded by
`test_the_fork_pin_job_does_not_override_the_milestone_preference`; the running tranche was
NOT relaunched, because cancelling five arms to recover 50 updates of cosmetic alignment
would cost real GPU time for no scientific gain. `g_updates_in_fork` in the `.source`
sidecar is what made this visible at all.

## How the rollout mode works (legacy)

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

**This 24-vs-200 split applies to `mode: rollout` ONLY.** The one-pass editor has no
sampler on either path: training and export both call `generate_onepass`, which evaluates
the network exactly once, and `--nsteps` is accepted and ignored there with a warning. The
discrepancy this split caused — training-side `geom_chain_intact_frac` reading 0 by
construction while the export gate used a different sampler — does not exist in that mode.

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
[the geometry tell](#the-geometry-tell--why-the-adversary-cannot-work-in-rollout-mode) — the resulting
artefacts are ~70x larger than anything the AE contributes, and they separate the two
classes perfectly. There is no probe for this; `shortcut_probe` masks the peptide, so it is
blind to a tell that lives *in* the peptide.

## The five one-pass arms

**Status: the editor is GPU-verified; the arms have not been trained.** The smoke passed on
the real model (job 69354, a6000/gpu26) — see [Smoke, measured](#smoke-measured) — so the
wiring is confirmed. There are still no quality results, and none should be quoted from
this section.

| arm | direct editor + retention/geometry losses | GAN | extra change |
|---|---|---|---|
| `onepass_contactonly` | yes | off | none |
| `onepass_gan` | yes | on | none — **the reference** |
| `onepass_gan_ar1` | yes | on | approximate R1 only |
| `onepass_gan_noise` | yes | on | adaptive discriminator-input noise only |
| `onepass_gan_nce` | yes | on | local contrastive retention only |

Every addon is compared against `onepass_gan`, never against another addon, and no arm
inherits another's addon. `cp2lp.addon` takes `none|ar1|noise|nce`, all three default off,
and the module **raises** on any contradictory combination — an addon with no adversary is
rejected rather than silently becoming the contact-only arm again.

`onepass_contactonly` answers a different question: whether adversarial learning helps at
all. Keep it in the reading. An addon that beats the GAN while the GAN loses to the
control has improved a term that should not be switched on.

### The common fork

All five arms start from **one** checkpoint: a 300-generator-update warmup with the
adversary off and the shared non-adversarial losses, pinned by `scripts/pin_cp2lp_fork.sh`.
Five per-arm warmups would leave the arms differing by 300 updates of independently seeded
training before any addon was switched on, which is a live alternative explanation for any
addon result.

The discriminator is built in **every** arm (`cp2lp.enabled` constructs it regardless of
`adversarial.enabled`), so its random initialisation lands in the fork checkpoint too and
the four GAN arms get the *same* initial D — not the same seed and a hope that nothing
upstream consumed a different number of draws. `onepass_gan_noise` is the one exception,
and only by construction: its D has a noise-level input appended **last**, so the splice
copies the head's leading columns from the fork and zero-pads only the new ones.

The warmup is training, not a preflight gate. The smoke is what gates on correctness.

### Arm A — approximate R1 (`addon: ar1`)

```
loss_ar1 = mean((D(real_features + eps) - D(real_features))**2)
loss_D   = hinge_D + lambda_ar1 * loss_ar1
```

Adapted from the Adversarial Post-Training paper (arXiv:2501.08316, eq. 9–10), where the
point is that this is a finite-difference stand-in for R1's `||grad_x D(x)||^2` needing no
second-order backward. Real samples only, as there.

Two departures, both forced by the domain. **sigma is relative** (`0.01 x` each channel's
own std) rather than the paper's absolute 0.01: these feature blocks mix nm distances
(~0.2–3.0), dihedral components (−1..1) and soft contact counts (0..~40), and one absolute
sigma is a rounding error on one and a scramble of another. And **only continuous channels
are perturbed** — sequence probabilities, one-hot pocket identity, padding and the two
terminal flags are left exactly as they are, because Gaussian noise on a one-hot is not a
nearby real sample. VERIFIED: 20 of 42 feature columns are eligible at `k_pocket=4`, and
the untouched ones come back bit-identical.

`lambda: 10`, not the paper's 100 — this D is three transformer layers over ≤16 residues
against a hinge loss whose terms are O(1). A pilot value; `loss_d_ar1_weighted` is logged
beside `loss_d` so the ratio can be read off a run instead of guessed at again.

The paired forwards run with **dropout disabled**. With dropout live the two forwards
differ by their masks, and the penalty would measure the sampling variance of D's own
regularisation and call it input sensitivity — largest exactly when the input sensitivity
is smallest. VERIFIED: the penalty is 0 to 1e-10 at `sigma_rel=0`, and `D.training` is
restored afterwards.

This is a feature-space adaptation of approximate R1, **not** coordinate-space R1.

### Arm B — adaptive discriminator noise (`addon: noise`)

The same variance-preserving affine Gaussian corruption is applied to the continuous
feature channels of **both** classes before ordinary GAN scoring, and the level is handed to
D. Faithful to the Diffusion-GAN reference in five respects: the corruption law
(`sqrt(1-s^2) x + s eps`, the reference's `q_sample` with `s = sqrt(1-abar_t)`); the level
pool in which **half the entries are zero** and the rest follow its `priority`
distribution (probability proportional to the level index); the level as a D input; the
bang-bang controller on `mean(sign(real_logits))` against the reference's own target
**0.6**; and `p` controlling the eligible *range* of levels as it controls `T` there.

Changed: the schedule is parameterised directly by the injected standard deviation instead
of by a beta product, so `max_std: 0.2` reads as "at most a fifth of a channel's own
standard deviation" and can be checked against the features. Same affine law, same
normalisation.

`adapt_rate_examples: 5000` and not the reference's 100k (`ada_kimg=100`). That value is
sized for batches of 32–64 images; a D step here scores a handful of length-matched pairs,
so 100k examples would be tens of thousands of updates and `p` would never leave its
initial value inside a 10k-update pilot. A pilot value; `dgan_p` is logged.

**Two things to watch, because this arm can look like it is working when it is not.**
`d_acc` here is accuracy on *corrupted* features and **falls** as the controller raises the
noise — that is the controller working, not the discriminator improving. The number
comparable with the other four arms is `d_acc_clean`, logged beside it. And raising the
noise must not become a way to call a chemically broken peptide valid: the corruption is a
discriminator input only, every generator geometry loss and every exported structure uses
uncorrupted coordinates, and the export gate is unchanged. `chain_intact` and
`peptide_bond_mae_nm` are the guards.

The **shortcut probe reads clean features in every arm, including this one**, so that
cross-arm diagnostic stays comparable. It is detached from the trunk, so the choice shapes
nothing.

VERIFIED: 36–52% of draws are exactly clean over 1280 samples, the amplitude never exceeds
`max_std`, categorical channels come back bit-identical, gradient reaches the generator
through the corruption, and the controller drives `p` to 1.0 while D wins on reals and back
to 0.0 while it loses. This is a feature-space adaptation, not a claim that the features
follow physical diffusion.

### Arm C — local contrastive retention (`addon: nce`)

One generator-side term. For each selected interior residue, a continuous invariant local
descriptor — backbone CA distances at sequence offsets ±1..3, three backbone bond lengths,
and distances to the **same source-selected** pocket neighbours — is projected and
L2-normalised; the generated residue `i` must be classified against its own source residue
`i` among the other selected residues of that same peptide.

Adapted from CUT's PatchNCE: the formulation, within-peptide negatives (the reference
measured those beat minibatch-wide ones, and it holds more strongly here — a negative from
another complex is trivially distinguishable), the masked diagonal, temperature 0.07 and
the two-layer projection with L2 normalisation. Not its image generator, its convolutional
patch encoder, its layer set or its weight of 1.0 (`0.1` here, against a generator
objective whose other five terms already compete for a clipped gradient budget).

The residue-index correspondence is **exact** here in a way it never is for images: CP and
LP share a length and a sequence, so residue `i` is the same residue in both and CUT's
patch sampling is unnecessary.

**Residue identity is deliberately absent from the descriptor.** The sequence is pinned by
another term, so a descriptor carrying it would let the task be solved by reading the
sequence — position `i` is the only place with that identity — and the term would preserve
no geometry at all.

Excluded from scoring: the first and last **two** residues (opening the ring is the task,
so asking the termini to keep their local environment is asking the edit not to happen),
and any wrap-around neighbour (a descriptor encoding the closing bond would ask for the
macrocycle back). Positions are ranked by the source residue's own soft contact count, so
the budget goes to the residues that define the binding mode. Examples with fewer than two
valid positions are skipped and counted in `nce_n_skipped_examples` — a softmax over one
class is a constant.

**MEASURED trap: the loss has a floor of ~1.2–1.4, not 0.** At initialisation a random
projection leaves all descriptors nearly collinear, so the absolute value is not "how much
content was lost". It *is* monotone in how far the edit moved (1.39 / 1.40 / 1.58 / 1.66
for 0 / 0.05 / 0.2 / 0.6 nm of noise at 8 positions; 1.19 / 1.20 / 1.29 / 1.60 without the
projection). Read changes in `nce_loss`, never its level. VERIFIED: no gradient reaches the
source geometry.

### Smoke, measured

Job 69354, `onepass_gan`, a6000 (47.4 GiB), against the lpmix init (the fork did not exist
yet — the smoke is what gates on whether producing it is worth the GPU hours).

| check | result |
|---|---|
| editor-trunk calls, training | **1** |
| editor-trunk calls, export (`--nsteps 200` passed) | **1** — ignored, as designed |
| residual at initialisation | `dCA rms 0.000e+00 nm`, `dz rms 0.000e+00` — the identity edit |
| noise adapter span | columns `[315, 331)` of 331, i.e. exactly the last 16 |
| missing checkpoint keys beyond the expected | **0** |
| source / receptor tensors mutated | none |
| AE parameters trainable | 0 |
| residual-head gradient | 2 of 6 nonzero (only the two zero-init `Linear` weights can be) |
| trunk gradient, step one | 0 of 454 — **expected** from a zero-init head |
| trunk gradient, after one head update | **454 of 454**, no NaN, no Inf |
| frame sensitivity, editor / base velocity head | **0.734x** (tolerance 1.5x) |
| invariant noise unchanged under a rigid motion | yes |
| peak GPU | **7.9 GiB of 47.4 (83% headroom)**, largest receptor 154 residues |

**The identity edit makes the loss values self-checking, and they check out.** At
`|delta| = 0` the output *is* the source CP, so `contact_retention_frac` and `seq_identity`
are both exactly 1.0, `terminal_gap_nm` equals `src_terminal_gap_nm` to four decimals
(0.3874), and `loss_contact`, `loss_sequence` and `loss_geometry` are all exactly 0. The
only nonzero term is `loss_open = 0.0626`, which is `relu(0.45 - 0.3874)` exactly — the
open-chain hinge reading the source ring as closed, which is the one thing it should say.

**`global_step=7` against `g_updates=4`** — the double-counting confirmed in this arm, not
just in a toy.

**The editor uses about a sixth of the card the rollout arm needed.** 7.9 GiB against a
rollout configuration that OOMed a 48 GB a6000 at twice this load. **`max_cp_per_step` was nevertheless left at 2**, and the earlier framing of that headroom as
free gradient signal was wrong: the editor runs one network evaluation per sample, so 3x the
samples is 3x the compute per update. Raising it trades wall clock for gradient quality at an
unmeasured ratio, and `accum: 8` already gives 16 source CPs / 32 generated LPs per generator
update. The headroom's real value is that these arms would also fit an a5000, which matters
only if a6000 contention blocks the queue — and the GPU class must then be the same for all
five. The 83% figure is one draw: peak is O(n_target^2) and the crop runs to 256 residues
against the 233 largest measured.

**`grad_clip_val` was left at 1.0.** Raising it looked like the obvious lever given the
6-17.5 pre-clip norms, but at `lr: 5e-5` an unclipped norm of 17 is an effective step of
8.5e-4 — a 17x learning-rate increase in disguise, applied blind to five arms at once. The v1
finding is a reason to watch the per-term gradient attribution, not to unclip. If retention
stalls in all five arms *together*, that is the moment to test it, as its own single-variable
arm.

Absolute frame deviations, for the record: editor `rel_dev` 2.10, base velocity head 2.86.
Both far above 1, i.e. **neither is equivariant** — which is the honest statement about this
architecture. Only the ratio is meaningful.

#### All five arms smoked (jobs 69366–69370), with the addon magnitudes

| arm | `g_updates` / `global_step` | `grad_norm_preclip_g` | `loss_d` | peak GiB | addon term / its denominator |
|---|---|---|---|---|---|
| `onepass_contactonly` | 6 / 6 (**1:1**) | 17.2 | — | 7.3 | — |
| `onepass_gan` | 3 / 6 (**1:2**) | 13.6 | 3.97 | 7.2 | — |
| `onepass_gan_ar1` | 4 / 6 | 11.2 | 4.03 | 8.5 | **2.6e-08** of `loss_d` |
| `onepass_gan_noise` | 4 / 6 | 17.5 | 3.93 | 7.0 | `p` 0.0008 and rising |
| `onepass_gan_nce` | 4 / 7 | 6.06 | 4.07 | 8.0 | **0.709** of `loss_g_total` |

The `g_updates` / `global_step` columns are the cleanest possible demonstration of the
counting problem: **1:1 in the single-optimizer control, 1:2 in the adversarial arms.**

**THE CLIP IS ACTIVE FROM STEP ONE HERE TOO — the editor does not escape v1's
gradient-budget problem.** `grad_norm_preclip_g` is **6–17.5 against `grad_clip_val: 1.0`**,
so every term's effective step is scaled by `clip / norm` and the largest term still decides
what the smallest gets. That is the mechanism that pinned v1's contact retention at 0.158,
and it carries over unchanged. `grad_clip_val` is shared config, so raising it keeps the
comparison single-variable — it is the first lever to reach for if retention stalls in every
arm at once.

**aR1 is 2.6e-08 of the hinge at initialisation, which is why `lambda` is now the paper's
100 and not 10.** The penalty is QUADRATIC in D's logit shift, and a freshly initialised D is
nearly constant in its inputs — a 1%-of-std feature perturbation moves its logit by 1.0e-4,
so the penalty is 1.0e-08 before any weight is applied. There is nothing for an R1-type
term to penalise until D sharpens, and v1's D reached logits of ±1.5–1.8 within a few
thousand steps. **Re-read `loss_d_ar1_weighted / loss_d` at the `g=1300` milestone:** if it
is still below ~1e-3 there, arm A never became active and its result must be reported as
**NOT MEASURED**, not as "no effect".

The weight was raised from 10 to 100 because the argument for 10 rested on a premise the
measurement falsified — it assumed the penalty was O(1)-ish against an O(1) hinge, and it is
1e-08. The two failure modes are also not symmetric: too small makes arm A inert and its
false negative *indistinguishable from a true one*, while too large makes aR1 dominate D and
is visible in `d_acc` / `d_acc_clean`. Prefer the detectable failure. `sigma_rel` was left at
0.01 deliberately — changing two knobs in one arm makes an unhelpful result unattributable.

**`loss_g_total` is NOT comparable across arms, and arm C is why.** The contrastive term is
0.709 of that total at initialisation — partly because the *other* terms are near zero at
the identity edit, and partly because `nce_loss` has an irreducible floor of ~1.2–1.4 (a
random projection leaves the descriptors nearly collinear). A constant offset contributes no
gradient, but it does sit in `loss_g_total` forever, so compare `loss_contact`,
`loss_geometry` and the export metrics across arms — never the total.

`d_match_frac` reads 0.5 in every adversarial arm here. That is the 6-step smoke starting
with an EMPTY length-matched reservoir, not a problem: v1 reached 0.9999 once the reservoir
filled (~760 entries). Check it again after a few hundred updates.

Peak memory is **7.0–8.5 GiB** across all five and is **not monotone in receptor size** —
the 203- and 233-residue batches used less than the 154-residue one. The standing assumption
that peak is dominated by O(n_target^2) pair features held for the rollout arm, which
retained a trajectory; it is not the binding constraint for the editor.

### Early training, measured at matched G updates (band 700-1000 of 10,300)

**One run each, ~10% of the budget. Signals, not verdicts.**

| arm | d_acc | probe | dgan_p | retention | gap nm | dCA nm | bond MAE nm | pre-clip \|g\| |
|---|---|---|---|---|---|---|---|---|
| `contactonly` | — | — | — | 0.9971 | **0.504** | **0.0281** | **0.0066** | **0.74** |
| `gan` | 0.820 | 0.505 | — | 0.9219 | 1.134 | 0.1743 | 0.0506 | 7.96 |
| `gan_ar1` | 0.829 | 0.505 | — | 0.9202 | 1.103 | 0.1766 | 0.0498 | 7.24 |
| `gan_noise` | 0.823 | 0.505 | 0.050 | 0.9247 | 1.187 | 0.1755 | 0.0506 | 7.45 |
| `gan_nce` | 0.834 | 0.504 | — | **0.9384** | 1.011 | **0.1412** | 0.0431 | 7.77 |

**THE CONTACT-ONLY ARM BARELY EDITS, and that inverts v1's reading.** It moves CAs by 0.028
nm and opens the termini to 0.504 nm — i.e. just past the `min_open_gap_nm` hinge at 0.45 and
then stops, exactly as a flat-bottomed hinge predicts. Its retention of 0.9971 and its
excellent bond MAE are therefore not quality, they are *near-inaction*: a peptide that did
not move keeps every contact and every bond. The adversarial arms move 6x further and open to
1.0–1.19 nm, against a real-LP median of 2.14 nm. **The adversary is supplying the only
pressure toward "looks like a real bound LP"**, which is what the objective says it is for.
The cost is visible in the same row: bond MAE 0.05 nm against the export gate's 0.02, so on
current geometry the GAN arms would fail `chain_intact` and the control would pass. That
tension is the thing to watch, and it is why the comparison reports yield and quality
together.

**The clip only bites in the adversarial arms.** Pre-clip norm is **0.74 in the control** —
*below* `grad_clip_val: 1.0`, so nothing is clipped there and its terms sum freely — against
~7.5 in every GAN arm. So the adversarial term is most of the generator gradient, and v1's
gradient-budget competition applies to the GAN arms only. That asymmetry did not exist in v1,
where the control still carried the real-LP flow term.

`gan_nce` is the most differentiated addon so far: highest retention, lowest CA drift, least
terminal opening, best geometry of the GAN arms — the contrastive term preserving local
content at the cost of opening, which is what it was built to do.

#### The Diffusion-GAN controller was holding the noise near zero ON PURPOSE

`dgan_p` = 0.050 at this band looked like a misconfigured adaptation rate, and the first
conclusion drawn from it — "the controller is 10x too slow, arm B will never activate" — was
**WRONG**. The trajectory says otherwise:

- `p` changed **2198 times in 749 updates** (one adaptation per 0.3 updates), so it is not
  adapting rarely; it is oscillating near zero.
- `dgan_sign_real` averages **0.436, below the 0.6 target**. D is right on real examples only
  ~72% of the time, so the controller is correctly *lowering* the noise: its whole job is to
  make the task harder only while D is winning.
- `p` began climbing as D sharpened and reached 0.175 by g=999, projecting to saturate around
  **g ≈ 4500** — well inside the 10,300 budget.

**READING TRAP: `d_acc` is not `dgan_sign_real`.** `d_acc` (0.82) is accuracy over both
classes; `sign_real` (0.436) is the real class alone, and it is the controller's input. D here
is much better at rejecting fakes (`d_fake_logit` −0.78) than at confirming reals
(`d_real_logit` +1.15 with more spread), so the two numbers disagree and only one of them
drives the noise. No intervention was made on arm B.

**ANALYSIS TRAP, hit while producing the table above: `cp2lp_onepass_gan` is a PREFIX of
`cp2lp_onepass_gan_ar1`, `_noise` and `_nce`.** An unanchored `wandb/run-*{name}*` glob
resolves the reference arm to whichever addon arm sorts last, which made `gan` and `noise`
come out identical to four decimal places. Anchor on `-{run_name}-[0-9a-f]+/`.

### Mid-training, matched band g=3800-4200 (of 10,300)

| arm | retention | gap nm | dCA nm | bond MAE | **chain_intact** | min pocket nm | d_acc | probe | \|g\| |
|---|---|---|---|---|---|---|---|---|---|
| `contactonly` | 0.9982 | 0.492 | 0.0297 | **0.0059** | **0.772** | 0.2410 | — | — | 0.51 |
| `gan` | 0.9615 | 1.252 | 0.1675 | 0.0356 | 0.010 | 0.1369 | 0.908 | 0.576 | 5.65 |
| `gan_ar1` | 0.9560 | 1.334 | 0.1886 | 0.0412 | 0.005 | 0.1196 | 0.901 | 0.581 | 5.87 |
| `gan_noise` | 0.9526 | 1.385 | 0.1881 | 0.0422 | 0.006 | 0.1314 | 0.900 | 0.579 | 6.66 |
| `gan_nce` | **0.9643** | 1.142 | **0.1432** | **0.0334** | 0.016 | **0.1530** | 0.903 | 0.573 | 6.63 |

**`chain_intact` IS NOW AN EXPORT FORECAST, and this is the decisive column.** In rollout mode
it was not — it was measured on a 24-step trajectory while export used a 200-step sampler,
so it read 0 by construction. One-pass training and export call the *same* function, so this
number predicts the export gate directly. It says: **the control will export triplets at
~77% and the adversarial arms at ~1%.**

**So the tension is now crisp, and neither side is yet a usable triplet source.** The control
produces chemically-valid-but-conformationally-ring-like LPs: gap 0.492 against a 0.45 hinge
and a real-LP median of 2.14, CA drift 0.0297 nm. It passes the gate by *not moving*. The
adversarial arms produce LP-like conformations — gap 1.14–1.39, drift 0.14–0.19 — with
backbones that fail the 0.02 nm bond tolerance. One is valid and uninteresting; the other is
interesting and invalid.

**The trend is in the adversarial arms' favour.** Against the g=700–1000 band, `gan` bond MAE
fell 0.0506 → 0.0356 (−30%) while retention rose 0.9219 → 0.9615 and the gap kept opening.
Whether 0.0356 reaches 0.02 by g=10,300 is the open question, and the gate needs *every* bond
inside tolerance, not the mean — so this does not extrapolate safely.

The shortcut probe rose 0.505 → ~0.576 against `d_acc` 0.90. Read the **gap** (0.32), which
is the same healthy regime as v1 (probe 0.645–0.738 under `d_acc` 0.96–0.996).

#### Both noise and nce addons have now saturated, for opposite reasons

**`dgan_p` = 0.999 — the controller hit its ceiling**, ahead of the g≈4500 projection. But
`d_acc` on corrupted features is 0.900 against 0.910 clean: **even at maximum configured
corruption D is barely affected.** So the binding constraint is `max_std: 0.2`, not the
adaptation rate that first looked suspect. If arm B lands on top of the plain GAN, the honest
conclusion is "corruption up to 0.2x per-channel std does not change the outcome" — *not*
"Diffusion-GAN does not help".

**`nce_loss` = 0.096, far BELOW the ~1.2–1.4 floor measured at initialisation.** The
contrastive task is now nearly solved, so with weight 0.1 the term contributes ~0.01 and is
close to inert — after having shaped the trajectory (this arm has the best retention, the
lowest drift and the best geometry of the four adversarial arms).

**CAVEAT, and it matters for how arm C is reported: `nce_loss` → 0 does not prove retention.**
The projection head is trainable and sits on the generator's optimizer, so it can minimise
the loss by finding descriptors that identify position `i` easily rather than by the
generator preserving content — and the pocket-distance block is largely source-determined, so
it is identifiable almost for free. The honest instrument for arm C is its **retention**
column, not its loss.

### What decides

An addon is kept only if it improves valid LP quality **and** retention at **useful yield**
over `onepass_gan`. A lower discriminator accuracy, a nicer GAN loss curve or greater
terminal separation alone is not sufficient. An addon that fails is an informative result
and is not to be rescued by changing its retention weight, its backbone or its dataset —
a shared implementation or physical-objective fix applies to all five arms and restarts
from a new common fork.

## Evaluating the arms

A **fixed manifest**, built once by `script_utils/cp2lp_eval_manifest.py` and used
identically by every arm: 64 held-out CP parents, one per receptor family, spread evenly
over peptide length, with 8 fixed noise seeds each — 512 generations per arm. BUILT AND
VERIFIED: 64 parents / 64 distinct families, 5–6 per length across all 12 lengths (5..16),
bit-identical on a repeat run. Every row of the current eval split is `head_tail`, so there
is no chemistry axis to stratify over; that is recorded in the manifest summary rather than
left implicit.

The report (`script_utils/cp2lp_onepass_report.py`) enforces three things:

- **raw attempts and accepted exports are both reported, always.** An arm with one
  excellent survivor and near-zero yield tops every accepted-only quality column, and must
  be visible in `acceptance rate` and `coverage` instead.
- **a corrupt row is refused, never skipped.** Concurrent `O_APPEND` NUL-corrupts rows on
  this filesystem, and a reader that skips unparseable lines turns lost data into a
  silently smaller denominator. It names the file and the line.
- **deltas are against `onepass_gan`**, never against the previous addon.

Displacement from the source is measured **in the receptor frame with no superposition**.
The receptor is fixed and shared, so the peptides are already in one frame and the
difference *is* the pose change; independently aligning the generated peptide first would
report only the conformational change and hide a sample that slid off its binding mode.
`interface_ca_rmsd` restricts to source-contacting residues, and is **NaN rather than 0**
when there is no interface — a zero would average in as "the interface did not move".

Within-parent diversity counts **valid LPs only**. Diversity among broken chains is
distortion, not variety: a generator emitting different kinds of wreckage would otherwise
score as the most diverse arm in the table.

Cost — seconds per output, peak GPU, generation calls per sample, and the G-update budget
the checkpoint was at — is reported separately from quality, because the addons need
different computation and folding that into the verdict hides what is being paid for.

## Running it

### The one-pass arms

```bash
# 0. the fixed 64-parent / 8-seed evaluation manifest (once)
bash scripts/submit_cp2lp_onepass.sh --submit manifest

# 1. preflight: one trunk call per batch, frame check, arm exclusivity, one G/D update
bash scripts/submit_cp2lp_onepass.sh --submit smoke

# 2. the COMMON FORK, then pin it so all five arms start from identical weights
bash scripts/submit_cp2lp_onepass.sh --submit warmup
bash scripts/submit_cp2lp_onepass.sh --submit pin

# 3. all five arms as SIBLINGS, so they queue concurrently
bash scripts/submit_cp2lp_onepass.sh --submit arms
#    or one at a time: --submit onepass_gan_ar1

# 4. the fixed manifest through each arm, at one checkpoint milestone
bash scripts/submit_cp2lp_onepass.sh --submit evaluate
bash scripts/submit_cp2lp_onepass.sh --submit evaluate --set EVAL_CKPT_TAG=g005300-EMA

# 5. the five-arm table and figure
bash scripts/submit_cp2lp_onepass.sh --submit report

# or the whole chain with dependencies wired:
bash scripts/submit_cp2lp_onepass.sh --submit pilot
```

Milestone checkpoints are named by **generator** updates (`g001300`, `g005300`, `g010300`,
each with an `-EMA` twin), because that is the only budget the arms share; see the
`global_step` section above. `EVAL_CKPT_TAG` selects which one every arm is measured at.

### The legacy rollout arms

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
clear the all-bonds test.

RE-VERIFIED 2026-09-30 from the reference rows on disk rather than from this note: n=536,
median 0.0033 nm, **p90 0.0054**, max 0.0205, 520/536 = 97.0% intact. The value also sits on
a **plateau, not a knife edge** — 0.015 / 0.02 / 0.03 / 0.05 admit 99.6 / 99.8 / 100 / 100%
of natives by mean deviation — so it is a conservative pick inside a flat region.

**Which means the gate is not what is failing the adversarial arms.** Their mean bond MAE of
0.0356 nm is **6.5x the 90th percentile of real crystal peptides**, and it exceeds even the
most permissive tolerance that still admits every native (0.03). Loosening it would not
rescue them. This is the *opposite* of the clash-threshold story below, where 2.7 Å rejected
43 of 60 natives and the threshold genuinely was the bug — the discipline is the same, the
verdict is not.

One definition only: `losses.open_chain_geometry_loss` now defaults to
`geometry.PEPTIDE_BOND_TOL_NM`. The constant previously sat unreferenced beside a duplicated
literal, so editing it would have looked like moving the gate without moving it. Guarded by
`test_the_bond_tolerance_has_one_definition`. So a `chain_intact` of 0 is a statement about the sample, not
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

### v1 (jobs 68417 / 68418, ~40k `global_step`) — the ablation ran, and the adversary *appeared* to lose

> **CORRECTED.** `global_step` counts G+D optimizer steps, so the numbers below are *not*
> generator-update-matched: the GAN arm had half the generator updates of the control at
> every quoted step. See [Generator updates, not `global_step`](#generator-updates-not-global_step--and-this-changes-how-the-v1v2-tables-read).
> The measurements are unchanged; the conclusion drawn from them is withdrawn.

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

> The same correction applies: "step-matched at 28–30k" is `global_step`-matched, i.e.
> ~14–15k generator updates for the GAN arm against ~28–30k for the control. The
> *within-arm* v1-vs-v2 comparisons are unaffected, since both are adversarial arms
> counted the same way.

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

### The geometry tell — why the adversary cannot work in rollout mode

> **This is what the one-pass editor was built to fix**, and it is the one conclusion in
> this history that the update-accounting correction does not touch: the distributions are
> disjoint at *every* logged step in both arms, so no amount of extra training for either
> one closes it.

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
rollout, reals from a clean AE round-trip of a crystal structure. Two ways out were open:

1. **Make the paths symmetric** — noise real LPs to some `t` and re-integrate them through
   the same sampler, so both sides carry identical rollout artefacts.
2. **Fix the geometry** — if the generator's output is bad at *every* step count, the
   adversary is right to reject it.

**A third one was taken instead: remove the asymmetric path.** The one-pass editor has no
trajectory on either side, so a generated peptide and a real one now both pass through
exactly one AE round trip and the tell cannot exist by construction. That closes this
question for `mode: onepass` and leaves it open for `mode: rollout`, which nothing is
planned against.

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

**The one-pass editor is GPU-verified; the five arms have not been trained.** The smoke
passed on the real model (job 69354) — see [Smoke, measured](#smoke-measured). 42 new unit
tests pass alongside the 51 existing ones, and the evaluation manifest is built and
reproducible. No quality number exists yet.

Next actions, in order:

1. `--submit smoke` — the other four arms, so the aR1, corruption and contrastive paths are
   each exercised once (only the reference arm has run).
2. Decide `max_cp_per_step` and the GPU class, given 83% headroom, **before** launching —
   both are part of the fixed config.
3. `--submit warmup`, then `--submit pin` — the common fork.
4. `--submit arms` — five siblings, 10,000 post-warmup generator updates each.
5. `--submit evaluate` and `--submit report`.

The legacy rollout arms:

- **Contact-only continues from v1's checkpoint** (`RUN_NAME=cp2lp_contactonly_v1`), with
  the reverted objective, which is bit-identical to what v1 trained under.
- **The GAN arm is a measurement, not a training run.** The geometry tell is not fixable in
  that mode, which is why the work moved to `mode: onepass` rather than to a fourth
  rebalance of it.
- Both are now known not to have been generator-update-matched against each other; see the
  `global_step` section.

## Why the chain is not intact — the CA track, measured

`geom_chain_intact_frac` reads ~0.01 in every adversarial arm against ~0.79 in the control,
and the 0.02 nm tolerance is **not** the culprit (it sits on a plateau and natives are 6x
inside it — see [Calibration notes](#calibration-notes-carried-over)). So the backbones are
genuinely broken. `script_utils/cp2lp_ca_spacing_probe.py` asks which decoder input breaks
them. Run it with `scripts/submit_cp2lp_ca_probe.sh --submit decisive`; table the JSON with
`script_utils/cp2lp_ca_probe_report.py`.

### The native yardsticks, measured first

From the 101 complexes in `calibration_native/`, **peptide chains only** (what the editor
edits), through a direct PDB parse:

| quantity | n | median | sd | min | p1 | p99 | max |
|---|---|---|---|---|---|---|---|
| CA(i)-CA(i+1) | 1,233 | **3.838 A** | 0.054 | 3.034 | 3.743 | 3.906 | 3.980 |
| CA(i-1)-CA(i)-CA(i+1) | 1,132 | **97.3 deg** | 16.2 | 74.5 | 83.3 | 146.7 | 157.5 |

Two hard facts come out of this. **Nothing exceeds 4.00 A** in 1,233 native bonds, and with
ideal bond lengths and angles the CA-CA span is a function of omega alone whose ceiling is
**3.804 A** at omega=180 — so CA-CA is a rigid rod, not a range. And **nothing is below
74.5 deg** in 1,132 native bend angles. A CA trace outside those two windows is not a
peptide backbone, whatever a decoder then does with it.

### The mechanism

The frozen decoder's only inputs are `z_latent`, `ca_coors_nm` and `mask` (`module._decode`)
— topology never reaches it — so CA track plus latents is an *exhaustive* decomposition of
what can break a decoded backbone. The editor adds a free per-residue Cartesian residual to
the CA track (`editor.apply_residuals`: `edited[bb_ca] = src[bb_ca] + delta`), and
**nothing in the loss set constrains CA-CA spacing**: `open_chain_geometry_loss` scores
C(i)-N(i+1) and intra-residue N-CA / CA-C on the DECODED atom37, i.e. downstream of the
frozen decoder, so the cause is reachable only through the decoder's Jacobian.

`delta_ca_rms_nm` cannot see the problem, because it is the RMS of one Cartesian component
and so cannot tell the edit the task wants from the edit that breaks the chain. Translating
or smoothly bending the peptide moves every CA and changes no CA-CA distance; only the
**incoherent** part of the residual stretches the virtual bond. The probe therefore reports
`incoherence = RMS(|edited bond| - |source bond|) / (sqrt(2) * sigma_component)`, which is
0 for a rigid motion and 1 for an iid residual (both anchored in
`script_utils/test_cp2lp_ca_spacing_probe.py`).

### Dose response (eval64 manifest)

| run | g | sigma_c nm | incoherence | dCA-CA nm | in native sd | CA-CA sd nm | outside window | chain_intact |
|---|---|---|---|---|---|---|---|---|
| real LP reference | — | 0.0000 | n/a | 0.0000 | 0.0 | 0.0065 | 3.1% | **0.9609** |
| onepass_contactonly | 1300 | 0.0332 | 0.414 | 0.0194 | 4.0 | 0.0191 | 50.0% | 0.5156 |
| onepass_contactonly | 5300 | 0.0313 | 0.332 | 0.0147 | 3.1 | 0.0150 | 43.4% | 0.7188 |
| onepass_gan | 1300 | 0.1094 | 0.392 | 0.0607 | 12.6 | 0.0547 | 87.7% | **0.0000** |
| onepass_gan | 5300 | 0.1705 | 0.296 | 0.0713 | 14.9 | 0.0704 | 80.6% | **0.0156** |

**At matched g the two arms sit at the same incoherence** — 0.414 vs 0.392 at g=1300, 0.332
vs 0.296 at g=5300 — while sigma_c differs by 3.3x and then 5.4x. So the adversary is not
teaching the editor worse geometry; it is making the edit *bigger*. The gan arm hands the
decoder bonds from **1.0 A to 7.2 A** and bend angles down to **25.6 deg**.

Holding CA-CA inside one native sd needs incoherence under **2.0%** at the gan arm's sigma_c,
and under the export tolerance **8.3%**. The editor is at 30-39%. The control is viable only
because its edit is small enough that ~40% incoherence still lands within 4 native sd.

### The race, and why the adversarial arm loses it

Incoherence is **not** a fixed constant — the editor is slowly learning to be more coherent.
It is losing to the growth in edit size:

| arm | quantity | g=1300 | g=5300 | change |
|---|---|---|---|---|
| contactonly | edit size sigma_c | 0.0332 | 0.0313 | −5.8% |
| contactonly | incoherence | 0.414 | 0.332 | −19.7% |
| contactonly | **CA-CA damage** | 0.0194 | 0.0147 | **−24.4%** |
| contactonly | chain_intact | 0.5156 | 0.7188 | — |
| gan | edit size sigma_c | 0.1094 | 0.1705 | **+55.8%** |
| gan | incoherence | 0.392 | 0.296 | −24.6% |
| gan | **CA-CA damage** | 0.0607 | 0.0713 | **+17.5%** |
| gan | chain_intact | 0.0000 | 0.0156 | — |

Both arms clean up their residual at the same rate (−0.020 and −0.024 incoherence per 1,000 g
updates). The control holds its edit size flat, so its damage falls and `chain_intact` climbs
0.52 -> 0.72. The gan arm's edit grows 56%, which outruns the cleanup, so **its net CA-CA
damage rises 17.5% and `chain_intact` stays pinned near zero.**

Extrapolating the gan arm's cleanup rate linearly — **unsafe, and stated as such** — reaching
8.3% incoherence needs ~8,900 more generator updates and 2.0% needs ~11,500, against the
**5,000 remaining** in the `max_g_updates: 10300` budget. And that extrapolation is optimistic
twice over: it assumes the cleanup rate holds all the way down, and it ignores that sigma_c is
still growing. Do not read "incoherence is falling" as "this fixes itself".

### Apportionment: which input carries the damage

The same shape as the run's own `val/decode_*` ablation, but varying the EDITOR's residual
rather than the flow model's prediction. Scored by the export gate's all-bonds test:

| decode | contactonly | gan |
|---|---|---|
| source CA + source z (AE round trip = **ceiling**) | 0.9844 | 0.9844 |
| source CA + EDITED z (latents alone) | 0.7031 | 0.6250 |
| **EDITED CA + source z (CA alone)** | **0.2812** | **0.0469** |
| EDITED CA + EDITED z (the actual output) | 0.5156 | 0.0000 |
| projected CA + EDITED z (bond lengths repaired) | 0.6719 | 0.1406 |

**The CA track is the dominant damage channel**: swapping it alone takes 0.9844 to 0.0469
(gan) or 0.2812 (contactonly), while swapping the latents alone costs far less. Each row is
a valid causal intervention.

**The 2x2 does NOT decompose additively, and must not be read as if it did.** In the control,
`EDITED CA + source z` (0.2812) is *worse* than the real output (0.5156): the edited latents
**partially compensate** for the damage the edited CA track does, so breaking the pairing by
substituting a source latent reads worse than either channel's own contribution. The
compensation is arm-dependent — it is +0.234 in the control and -0.047 in the gan arm, i.e.
absent once the edit is large. Quote the single-channel rows as interventions, never as
shares of a total.

### Repairing bond lengths alone is not enough, and why

Projecting the edited CA track back onto CA-CA = 0.3809 nm (damped relaxation, converged to
5e-7 nm) and re-decoding with the same latents recovers only **0.000 -> 0.141** (gan) and
**0.516 -> 0.672** (control) — 15% and 36% of the deficit. Two reasons, both measured:

* **it leaves the second constraint violated, and nudges it worse.** gan bend-angle minimum:
  source 74.5 deg, edited 25.6, projected **21.8**; p99 148.0 / 161.7 / **170.0**. Enforcing
  bond lengths while the angles are free moves the angles. The control's edited angles are
  still native (min 74.9, median 97.2), which is why its lengths-only repair buys more.
* **`chain_intact` is an all-bonds test**, so it is the per-bond pass rate raised to the bond
  count (~9.25 here). Per-bond 0.52 -> 0.002 of peptides; 0.70 -> 0.037; **reaching 0.95 of
  peptides needs a per-bond pass rate of 0.9945.** Halving the mean bond error is worth
  almost nothing if a tail survives.

Read the counterfactual against its own ceiling, not against 1.0: projecting the **real** LPs
costs 0.9609 -> 0.9453, so a hard projection to a constant has an intrinsic 1.6-point price
(natives have CA-CA sd 0.0065 nm and a projection to one value removes that spread). A
flat-bottomed projection to the native window would not pay it. The projection target 0.3809
also sits 0.0029 nm below the peptide-only native median of 0.3838 — inside the tolerance,
but it is an offset, not a match.

### What this rules out, and what it implies

**Not a loss-weight problem.** The geometry term acts downstream of the frozen decoder while
the cause is an unconstrained CA trace upstream of it, and incoherence is the same in both
arms at matched g — it tracks training, not adversarial pressure. Within this parameterisation
the only lever on damage is edit size, and the control's 0.72 is bought by holding its edit
flat, i.e. by very nearly not moving. No setting of `geometry_weight` escapes that trade; it
only slides along it. The editor *is* learning to edit more cleanly, just far too slowly to
matter inside the budget, and more slowly than the adversary grows the edit.

**Not a tolerance problem**, and not a repeat of the clash-threshold story: the gate is
calibrated and the trace is genuinely non-physical.

**It is a parameterisation problem.** The editor's action space contains almost no
chain-preserving directions, yet the task needs a large edit (source terminal gap 0.35 nm,
real bound LPs 2.14 nm). The fix has to make a valid CA trace the only reachable output:

1. **Internal-coordinate residual (preferred).** Emit per-residue deltas on the CA-trace bend
   angles and virtual torsions, pin CA-CA at native, rebuild by NeRF. Zero-init still gives
   exactly the identity edit, so the pretrained splice and the "first output is the AE round
   trip" property both survive. Every reachable point satisfies both constraints by
   construction, there is no new loss weight competing for the clipped gradient budget, and
   ring opening *is* a torsional move so the action space matches the task. The known cost is
   conditioning: a torsion change swings the whole downstream chain.
2. **Differentiable projection layer (fallback).** Keep the Cartesian residual and pass the
   edited CA through a projection enforcing **both** constraints before decoding, flat-bottomed
   to the native windows so it does not pay the 1.6-point idealisation price. The projection's
   Jacobian annihilates the chain-breaking directions, so the adversary can only push in
   chain-preserving ones. Cheaper to build; costs forward-pass iterations.

A CA-CA penalty term is **not** on this list as a primary fix: it would be a seventh term in
the budget that already retires the sequence term to make room, and it addresses only the
constraint whose repair was measured to be insufficient on its own.

## The fix: a CA edit space with no invalid points

`src/proteinfoundation/cp2lp/ca_internal.py`. A CA trace becomes (seed frame, bond lengths,
bend angles, virtual torsions); an edit moves the bend angles, the torsions, and a rigid
transform of the whole trace. Bond lengths are not in the edit space by default and bend angles
are clamped to the measured native window, so **no reachable output is an invalid trace**. The
coherent/incoherent split the probe had to measure becomes structural: rigid motion is the
seed-frame delta, shape is theta/tau, and no parameter stretches the chain.

Zero deltas reproduce the input to 1e-5 nm, so the zero-initialised heads still make the
editor's first output exactly the AE round trip of the source CP — the property the pretrained
splice depends on.

### Measured at matched edit size (random edit, SOURCE latents, g=1300 AE)

`scripts/cp2lp_internal_probe.sbatch`. Both arms at identical RMS CA displacement, same frozen
decoder, same latents. `chain_intact`:

| sigma_c nm | RMS disp nm | cartesian | **internal** | cart CA-CA sd | int CA-CA sd | cart bend min | int bend min |
|---|---|---|---|---|---|---|---|
| 0.005 | 0.005 | 0.9375 | **0.9688** | 0.00905 | 0.00589 | 75.9 | 73.9 |
| 0.010 | 0.010 | 0.4219 | **0.9844** | 0.01514 | 0.00589 | 71.6 | 74.3 |
| 0.020 | 0.020 | 0.0625 | **0.9688** | 0.02923 | 0.00589 | 70.6 | 73.5 |
| 0.040 | 0.040 | 0.0000 | **0.7812** | 0.05300 | 0.00589 | 56.6 | 73.8 |
| 0.070 | 0.070 | 0.0000 | **0.4375** | 0.09226 | 0.00589 | 50.8 | 73.0 |
| 0.110 | 0.107 | 0.0000 | **0.3906** | 0.13780 | 0.00589 | 12.6 | 73.0 |
| 0.1705 | 0.167 | 0.0000 | 0.0469 | 0.19775 | 0.00589 | 5.2 | 73.0 |

Controls behave: `source` and `rigid_only` both read 0.9844 at every sigma, so a rigid motion of
the matched size costs exactly nothing and the harness is measuring what it claims.

**The internal space supports a 4-8x larger edit at matched `chain_intact`** — 0.020 vs 0.005
to hold ~0.95, 0.070 vs 0.010 to hold ~0.43, ~0.16 vs 0.020 to hold ~0.06. And the CA trace is
*exactly* native at every size: CA-CA sd pinned at the source's own 0.00589 nm and the bend
floor never below 73 deg, against the Cartesian arm's sd rising to 0.198 and its bend floor
collapsing to 5.2 deg.

For scale: the control arm trains at sigma_c 0.031-0.033 and scores 0.52-0.72. The internal
space at sigma_c 0.040 — a *larger* edit — scores 0.78.

### What it does NOT fix, and this is the important part

**A valid CA trace is necessary but not sufficient.** At the gan arm's own sigma_c of 0.1705 the
internal edit still only reaches 0.0469, with a perfectly native trace. So the damage at large
edit size is not CA invalidity — it is **CA-latent inconsistency**. The decoder places C(i) in
residue i's local frame and N(i+1) in residue i+1's, so the peptide bond spans two frames and
is intact only if the latents agree with the actual inter-frame geometry, which *is* the
torsion. Move tau without updating the latents and the bond opens, however native the trace.

This re-reads the earlier 2x2 honestly: the `edit_ca + src_z` cell (0.0469) was attributed to
the trace being invalid, and it is mostly not — making the trace perfectly valid moved it only
to 0.0781. Most of that cell is staleness between the two channels.

Two consequences:

* every number in the table above is a **FLOOR** for the parameterisation, not a prediction. It
  uses SOURCE latents, i.e. deliberately stale ones. A trained editor co-edits the latents, and
  the control arm is already measured doing exactly that — its edited latents recover +0.234 of
  `chain_intact` over a source-latent substitution.
* the remaining work is on the latent channel, not the CA channel. The CA channel is now
  constraint-exact and costs nothing to keep that way.

A `latent_oracle` column would settle how much of the residue is staleness, by re-encoding the
decoded structure. **Deliberately not implemented**: a faithful re-encode must rebuild the
encoder's whole input batch, and `coords` and `coords_nm` are different rigid frames ~19 A apart
that different feature factories read independently, so overriding only the nm coordinates
returns latents that mean nothing. The column would look authoritative and be wrong.

### Caveat on the random edit

A random torsional edit barely extends a peptide: at sigma_c 0.1705 the internal arm moved the
terminal gap only 0.352 -> 0.544 nm. The task needs a COORDINATED edit — driving the virtual
torsions toward trans takes a 16-mer to ~5 nm end to end. So the matched-size table understates
the parameterisation, and `script_utils/cp2lp_extension_tradeoff_probe.py` measures the curve
that matters instead: `chain_intact` against terminal gap along a coordinated walk toward the
extended corner, against real bound LPs' 2.14 nm median.

### Convention traps found while building it

* **The NeRF normal's sign disagreed with the extracted dihedral's.** Three of the four
  guarantee tests passed while the round trip was out by **3.6 nm** — the rebuild produced
  valid-LOOKING traces that were not the input. Pinned now by `test_round_trip_is_exact`.
* **`_dihedral` omitted the `-(p1 - p0)` negation**, so it was offset by exactly pi from IUPAC:
  anti read 0 instead of 180, and an extended trace came out at tau=0. Now pinned by physics
  rather than by convention — theta 91 deg / tau 50 reproduces the alpha-helix rise at 1.52 A
  per residue, theta 120 / tau 180 the beta-strand at 3.33 A. **`geometry.backbone_dihedrals`
  carries the same un-negated form.** Harmless there, because the discriminator sees the same
  relabelling on real and generated peptides alike, but those values are NOT Ramachandran
  phi/psi and must not be read as such.

## Traps found the hard way

- **A second-resolution timestamp is not a unique env-file name.** Two
  `submit_cp2lp_ca_probe.sh --submit one` invocations issued in the same second produced ONE
  env file: the second overwrote the first, and because a job reads its env file at RUN time
  rather than at submit time, *both* jobs then read the survivor. They probed the same arm
  and raced on one output path. Caught before the results were read, but nothing in the logs
  would have said so — both jobs succeed, and the two "different" runs simply agree. The
  submitter now puts `$$` in the name. Any submitter that writes a per-submission file needs
  more than a timestamp.

- **`trainer.global_step` is not the generator-update count.** Under manual optimization it
  sums `optimizer.step()` calls across optimizers, so an adversarial arm advances it twice
  as fast as a single-optimizer control. Measured, not inferred. Anything expressed in it —
  a budget, a warmup, a milestone, an x-axis — means a different amount of generator
  training in each arm. Use `cp2lp.max_g_updates` / `train/cp2lp_g_updates`. This already
  invalidated the v1/v2 step-matched comparisons once.
- **`g_updates` is a persistent buffer, so it is INHERITED from the fork.** That is
  deliberate (a requeue must not restart the schedule) and it means the arms' budgets are
  offset by the warmup's 300 — `max_g_updates: 10300` is 10,000 post-warmup updates. A
  budget of 10000 would quietly run 9700 and put every milestone 300 short of where the
  comparison says it is. Guarded by
  `test_arm_budgets_account_for_the_forks_inherited_update_count`.
- **The noise adapter must be initialised AFTER the pretrained splice and never on a
  resume.** The splice zero-fills an appended feature's columns; the residual heads are zero
  too; so with zero noise columns the trunk gets no gradient on step one and the editor
  learns a deterministic edit before discovering it has a random input. `train.py` calls
  `model.init_editor_noise_adapter()` in the one branch where both conditions hold.
- **An off-by-one in that adapter is silent and total** — it would randomise a *pretrained*
  feature's columns and leave the noise at zero, so the editor would be deterministic AND
  damaged. The span is computed from `feat_creators` order and asserted by
  `test_noise_adapter_writes_exactly_the_noise_columns`.
- **Standardise-corrupt-destandardise leaves float error on columns it did not touch.**
  Round-tripping a one-hot through `(v-mean)/std*std+mean` moves it by ~1e-8, which is
  numerically harmless but makes "categorical features are unchanged" unassertable. The
  `torch.where` selects the *final* value instead, so untouched columns are bit-identical.
- **In arm B, `d_acc` is accuracy on corrupted features and is supposed to fall.** Read
  `d_acc_clean` against the other arms. Reading `d_acc` would make the controller working
  look like the discriminator failing.
- **`nce_loss` has a floor of ~1.2–1.4, not 0**, because a random projection leaves the
  descriptors nearly collinear. Only changes in it are readable; the level is not "how much
  content was lost".

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
| `script_utils/test_cp2lp.py` | unit tests for the pure pieces (rollout mode) |
| `script_utils/cp2lp_ca_spacing_probe.py` | CA-track probe: residual decomposition, 2x2, projection counterfactual |
| `script_utils/cp2lp_ca_probe_report.py` | tables the probe JSONs (CPU, redrawable from disk) |
| `script_utils/test_cp2lp_ca_spacing_probe.py` | anchors the incoherence metric at its 0 and 1 limits |
| `scripts/cp2lp_ca_probe.sbatch` | one arm's probe |
| `scripts/submit_cp2lp_ca_probe.sh` | probe submitter (`--submit decisive`) |
| `src/proteinfoundation/cp2lp/editor.py` | the one-pass editor: batch, residual, frame check |
| `src/proteinfoundation/cp2lp/addons.py` | feature scales + the three add-ons, one per arm |
| `configs/example/training_cp2lp_onepass.yaml` | the shared base for all six one-pass runs |
| `configs/example/training_cp2lp_onepass_*.yaml` | the fork and the five arms, one line apart |
| `scripts/submit_cp2lp_onepass.sh` | the one-pass submitter (manifest/smoke/warmup/pin/arms/eval/report) |
| `scripts/pin_cp2lp_fork.sh` | snapshots the common fork the five arms share |
| `script_utils/cp2lp_onepass_smoke.py` | one-pass preflight incl. the trunk-call count |
| `script_utils/cp2lp_eval_manifest.py` | the fixed 64-parent / 8-seed evaluation manifest |
| `script_utils/cp2lp_onepass_report.py` | the five-arm comparison table + figure (CPU) |
| `script_utils/test_cp2lp_onepass.py` | unit tests for the editor and the three add-ons |
