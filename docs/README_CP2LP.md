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
from the fork (300), which is why the arms' budget reads `10300` — 10,000 post-warmup
updates, with the fork itself as the `g=300` evaluation point.

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
rollout configuration that OOMed a 48 GB a6000 at twice this load. Two consequences worth
deciding before launch rather than after: `max_cp_per_step` could go well above 2 (it is
the memory knob, and more source CPs per step is a better gradient estimate for the same
wall clock), and an a5000 is plausible if a6000 contention is the bottleneck. Both are part
of the config the comparison holds fixed, so **changing either after any arm has started
invalidates the comparison** — decide once, apply to all five. The 83% figure is also one
draw: peak is O(n_target^2) and the crop runs to 256 residues against the 154 measured here.

Absolute frame deviations, for the record: editor `rel_dev` 2.10, base velocity head 2.86.
Both far above 1, i.e. **neither is equivariant** — which is the honest statement about this
architecture. Only the ratio is meaningful.

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

## Traps found the hard way

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
