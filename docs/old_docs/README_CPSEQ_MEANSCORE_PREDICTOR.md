# A mean-score (Rosetta dG) predictor from the corpus-wide run — feasibility record

Started 2026-10-27. **Every number here was produced by execution**, by the three scripts
named at the end, from one run directory and one scoring version.

The goal is a model that predicts a cyclic peptide's mean Rosetta interface energy from
its sequence, cheaply enough to steer sequence generation instead of folding every
candidate. This document answers two questions and stops: **is the label usable**, and
**does a sequence encoder beat a trivial baseline**. No scorer has been trained for
deployment, and nothing here has guided a generation run.

Energies are Rosetta ref2015 REU and are **not** measured binding free energies.

---

## 0. Words used here

| name in code/data | what it means |
|---|---|
| fold | one Boltz-2 structure prediction of one sequence at one seed |
| label | the number the model is trained to predict: one sequence's energy summarised over its folds |
| within-target rho | Spearman correlation between predicted and labelled energy **among one target's candidates**, averaged over targets |
| ceiling | what a perfect predictor of the true latent mean would score on that metric, given that the label is itself a 10-seed estimate |
| reliability | the fraction of the label's within-target variance that is signal rather than seed noise |
| target (`parent_id`) | one receptor together with its native peptide |
| reference arm | the native peptide, cyclized, unchanged |
| `random_control` arm | the native peptide with 1–7 random substitutions |

---

## 1. The data, and why only this run

| | |
|---|---|
| run directory | `/data/user_data/yixiz/cpseq/pepbench_corpus_pot` |
| scoring version | `v3-cis-amide-muco-windows` only |
| stage | `ladder` (every sequence folded at the same 10 seeds: 101, 111, 202 … 909) |
| sequences | **3,690** (3,270 `random_control` + 420 `reference`) |
| targets | **420** |
| folds | **36,874** scored |
| peptide length | 5–16 residues |
| receptor size | 30–703 residues, median 165 |

Only this run is used, and only one of its three scoring versions. The same directory also
holds `v4-postrelax-validity` (2,893 folds) and `v5-junction-restrained-relax` (8,460),
which refine the junction differently; pooling them would make the label a mixture of
protocols rather than a quantity. That restriction is what "consistent label" has to mean
here, and it is enforced by `--scoring-version` taking exactly one value.

48 sequences were dropped for having fewer than 8 folds. 790 folds failed prediction and 5
produced no structure; both are excluded and counted in the emitted JSON.

---

## 2. The naive label is worthless, and clipping fixes it

**0.38 % of folds (140 of 37,072) come back from refinement at a positive interface
energy, up to +30,351 REU.** Ten of those pass every validity check, so the validity gate
does not remove them. The bulk of the distribution is well behaved — 1st to 99th
percentile spans −60.2 to −1.6 REU — so a handful of blow-ups dominate any mean.

Two independent estimators agree on what that costs. Split-half reliability splits each
sequence's 10 seeds into two interleaved halves, labels each half, and correlates the two
within target; the variance decomposition instead subtracts the seed-noise variance of the
mean from the observed within-target variance. They were computed separately and agree to
three decimals:

| label | split-half r | → full reliability | variance-decomposition reliability |
|---|---|---|---|
| mean of folds | −0.020 | **−0.04** | −0.02 |
| **mean of folds clipped at 0 REU** | 0.716 | **0.834** | **0.833** |
| median of folds | 0.684 | 0.813 | — |
| mean of valid folds only | 0.015 | 0.03 | 0.09 |

**The unclipped mean has reliability ≈ 0: it is a readout of which sequence happened to
draw a blow-up, not of the sequence.** Restricting to valid folds does not rescue it,
because blow-ups pass the gate. Clipping each fold at 0 REU before averaging lifts
reliability to 0.83.

The median scores 0.81 and is the project's convention for reporting energy
(`README_CPSEQ_PEPBENCH_SCALEUP` §2). The clipped mean is used here because it is
marginally more reliable and keeps a differentiable, per-fold form that a Gaussian-NLL
objective can supervise directly; `--agg median` selects the other.

Under the clipped mean, the within-target **signal** sd is **3.82 REU** against a seed
noise sd of 1.71 REU on the 10-fold mean. Between-target sd is **9.36 REU**, about 2.4×
the within-target signal, so a model that learns only which target a sequence belongs to
captures most of the pooled variance while ranking nothing. §4 shows that happening: every
model scores 0.60–0.94 pooled and 0.09–0.22 within target. That gap, not the variance
ratio, is why §3 insists on the within-target metric.

(An earlier draft quoted 43.95 REU for the between-target sd. That figure came from
*unclipped* means and was an artifact of the §2 blow-ups, which land in a few targets and
inflate the spread between target means fivefold. It is withdrawn.)

---

## 3. How a predictor is scored

**Within-target Spearman, averaged over held-out targets.** Three rules:

1. **Whole targets are held out** (`GroupKFold` on `parent_id`, 5 folds). The table is one
   row per sequence, so folds of a sequence cannot straddle the split.
2. **The reference row is excluded.** It is the only row per target with
   `n_mutations == 0`, so keeping it lets a model score by "is this the original" without
   ranking any design against another.
3. **Every score is quoted against the label's own ceiling**, computed by the same
   per-target averaging from the same folds: **0.715**. A perfect predictor of the true
   latent mean scores about 0.715 here, not 1.0. A raw rho is uninterpretable without it.

Global (pooled) rho is reported alongside and should be ignored: it mostly reads out which
target a sequence belongs to. Every model below scores 0.60–0.94 globally while scoring
0.09–0.22 within target.

---

## 4. Results

Ceiling 0.715. 3,270 sequences, 419 targets, reference excluded.

| features | within-target rho | fraction of ceiling |
|---|---|---|
| mutation count only (`n_mutations`, `identity_to_original`) | 0.001 | 0.00 |
| mean hydropathy, one scalar | −0.168 | 0.23 |
| composition deltas vs. the native peptide (22 feat) | 0.074 | 0.10 |
| absolute composition (24 feat) | 0.181 | 0.25 |
| **all sequence features, ridge (49 feat)** | **0.218** | **0.30** |
| all sequence features, gradient boosting | 0.158 | 0.22 |
| frozen ESM-2 650M, mean-pooled (1280d) | 0.117 | 0.16 |
| frozen ESM-2 + 3-residue wrap-around | 0.094 | 0.13 |
| frozen ESM-2 → 16 / 64 / 256 PCs | 0.107 / 0.116 / 0.094 | 0.15 / 0.16 / 0.13 |
| frozen ESM-2 + composition | 0.105–0.198 | 0.15–0.28 |
| post-fold Rosetta/pose readouts (**not usable for guidance**) | 0.701 | 0.98 |

### Sequence does carry signal, and it is not the mutation-count confound

Mutation count alone scores **0.001**, and removing all three mutation-count features from
the ridge moves it from 0.218 to 0.199. The 0.218 is therefore not "more edits are worse".

### A frozen ESM-2 peptide encoder loses to amino-acid composition

**0.117 against 0.218.** This is not a dimensionality artifact: reducing the embedding to
16, 64 or 256 principal components leaves it at 0.09–0.12, and concatenating it with the
composition vector never beats composition alone. The circular wrap-around makes it worse.

Two things this does not establish. The encoder is **frozen and mean-pooled**, so this is
a lower bound on ESM-2 rather than a verdict; fine-tuning or attention pooling could
change it. And a 5–16mer is far outside ESM-2's training distribution, where a mean over
~10 residue embeddings carries little beyond residue identity — which composition already
encodes, more directly.

### Most of the label is interface geometry, which sequence barely predicts

The label's within-target variation tracks pose descriptors, not sequence descriptors:

| post-fold readout | within-target rho vs. label | can sequence predict the readout? |
|---|---|---|
| shape complementarity `sc_value` | −0.469 | 0.189 |
| buried interface area `dSASA_int` | −0.421 | 0.243 |
| interface hydrogen bonds | −0.421 | — |
| local deformation proxy | 0.033 | 0.123 |
| per-sequence valid rate | −0.060 | 0.056 |

Given the fold, cheap descriptors recover the label at 0.701 of a 0.715 ceiling — the
label is nearly a deterministic function of the structure Boltz-2 produced. Sequence
predicts the dominant driver, buried area, at only 0.243, almost exactly what it predicts
the label at (0.218). **The bottleneck is that sequence weakly determines how much
interface the fold buries, and no peptide-only encoder can get around that.**

---

## 5. What this means for the proposed architecture

The staged plan was: peptide ESM-2 encoder → target structure encoder → cross-attention →
pooling → scalar head, with a variance head added later. Against the measurements:

- **The peptide encoder is not the limiting component, and ESM-2 is currently a
  regression.** Composition beats it. Building cross-attention on top of a component that
  starts below a 49-feature baseline is premature.
- **The target encoder is where the headroom is**, because the label's variance lives in
  pocket-dependent interface geometry. The data for it is on disk and complete: all 420
  targets have receptor structures, `pocket_contacts` and chain maps in
  `/data/user_data/yixiz/cpseq/pepbench_all/manifest/targets.csv`.
- **A variance head is not worth building yet.** It would be supervised by a 10-seed
  variance, whose relative standard error is √(2/9) ≈ 47 %, and the mean predictor it
  would sit on currently reaches 0.30 of ceiling. Settle the mean first.

### Two risks to the stated purpose, which is guiding generation

1. **The main learnable axis is hydrophobicity.** One scalar, mean hydropathy, gets 0.168
   of the 0.218 the full ridge gets — 77 % of it. Steering generation along a Rosetta dG
   predictor whose dominant direction is "bury more greasy surface" is a known failure
   mode, and it will not be visible in the dG metric being optimised. Any guidance run
   needs hydropathy drift tracked as a named diagnostic, not assumed benign.
2. **The training distribution is a tight shell around native peptides.** 2,077 of 3,270
   candidates are single substitutions, 633 are doubles, and there are 7.8 sequences per
   target. A scorer fitted here is interpolating within 1–3 edits of a natural peptide and
   will be out of distribution exactly where generation goes.

### Not yet addressed: homologous receptors across targets

Splits hold out whole targets, which handles the trivial case — only 2 PDB entries
contribute 2 targets each (4 of 420). It does **not** handle homology between different
PDB entries, and `family_id` is empty for all 420 rows in the manifest, so it cannot be
grouped on. Clustering the 420 receptor sequences and re-running the CV on those clusters
is required before any of these numbers is called generalization to unseen target
families.

---

## 6. Reproducing

```bash
# 1. the table: one row per sequence, label + per-fold values
python script_utils/cpseq_meanscore_dataset.py \
    --out-dir /data/user_data/yixiz/cpseq/pepbench_corpus_pot \
    --json /data/user_data/yixiz/cpseq/meanscore/dataset_v3.json

# 2. the ceiling, and the feature baselines
python script_utils/cpseq_meanscore_baseline.py \
    --dataset /data/user_data/yixiz/cpseq/meanscore/dataset_v3.json \
    --json /data/user_data/yixiz/cpseq/meanscore/baseline_v3.json

# 3. the frozen-ESM-2 probe (stage the weights to local SSD first: reading the
#    2.6 GB checkpoint straight off NAS hung for over four minutes)
rsync -a /home/yixiz/Proteina-Complexa/community_models/ckpts/ESM2/models--facebook--esm2_t33_650M_UR50D \
    /scratch/yixiz_esm2/
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python script_utils/cpseq_meanscore_esm_probe.py \
    --dataset /data/user_data/yixiz/cpseq/meanscore/dataset_v3.json \
    --cache-dir /scratch/yixiz_esm2 \
    --embed-cache /data/user_data/yixiz/cpseq/meanscore/esm_embed_v3.npz \
    --json /data/user_data/yixiz/cpseq/meanscore/esm_probe_v3.json
```

All three run on CPU except the probe, which used one L40S for about a minute inside an
existing allocation. The embedding cache makes the probe re-runnable without a GPU.
