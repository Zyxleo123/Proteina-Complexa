# Migrating the `cpseq_mlm` sequence proposer to another cluster

Written for whoever picks this up on the receiving cluster. It assumes you have the repo
and an `rclone` remote, and nothing else about your environment.

The model: a masked-LM that fills masked positions of a cyclic peptide given the receptor
pocket **sequence** and a cyclization-chemistry token. Trained on CPSea, warm-started from
`ChatterjeeLab/PepMLM-650M`. Full design notes in `README_CPSEQ_MLM_PROPOSER.md`.

---

## 1. What is in the bucket

```
<prefix>/
  README_CPSEQ_MLM_MIGRATION.md        this file
  RUN_STATUS.md                        the run's own results record
  corpus/                      386 MB  train_*.jsonl (24 shards) + val_*.jsonl (32) + stats
  runs/cpseq_mlm_pocket/
    manifest.json                      config, baselines, mask schedule actually used
    metrics.jsonl                      every logged step and eval -- all figures redraw from this
    PUBLISHED_STEP                     which hf_step* below is the one to use
    checkpoints/hf_step<N>/    2.5 GB  HF-format weights + tokenizer  <-- what you need
    checkpoints/last.pt        7.3 GB  weights + Adam state, ONLY for resuming (tier `all`)
  runs/cpseq_mlm_nopocket/             the target-free control arm, same layout
```

Two arms are published on purpose. `cpseq_mlm_pocket` is the model; `cpseq_mlm_nopocket` is
the paired control that makes "it reads the pocket" a measurable claim rather than an
assumption. Keep both or the central result is not reproducible.

## 2. Pull it

```bash
rclone copy b2:<bucket>/cpseq_mlm/runs/cpseq_mlm_pocket   $DEST/runs/cpseq_mlm_pocket   -P
rclone copy b2:<bucket>/cpseq_mlm/runs/cpseq_mlm_nopocket $DEST/runs/cpseq_mlm_nopocket -P
rclone copy b2:<bucket>/cpseq_mlm/corpus                  $DEST/corpus                  -P
rclone size b2:<bucket>/cpseq_mlm        # expect ~5.4 GB without last.pt, ~20 GB with
```

**Do not put any of this on a home filesystem.** Checkpoints, corpora and logs all belong
on scratch; a 2.5 GB checkpoint on a quota'd home directory is how a later run dies at a
save with no useful error.

## 3. Two environment traps that will stop you

Both were hit here and both are silent until they are not.

**(a) `transformers` 5.x may be unable to load ESM at all.** Every model class reaches
torchaudio on its import path:

```
AutoModelForMaskedLM -> modeling_utils -> loss.loss_utils -> loss_rnnt
                                          -> if is_torchaudio_available(): import torchaudio
```

If the environment has a torchaudio built against a *different* torch than the installed
one, the guard passes (it is installed) and the import then dies:

```
OSError: .../torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl
```

Check with one line before anything else:

```bash
python -c "from transformers import AutoModelForMaskedLM; print('ok')"
```

If it fails that way, the repo carries an import-only stub — put it first on the path:

```bash
export PYTHONPATH="$REPO/script_utils/_compat_stubs${PYTHONPATH:+:$PYTHONPATH}"
```

This is deliberately a shim and not an uninstall, because the venv here is shared by the
whole flow stack. If torchaudio is genuinely needed on your cluster, uninstall/reinstall it
matched to your torch instead and drop the stub.

**(b) `resize_token_embeddings` does not resize ESM's output bias.** `EsmLMHead` computes
`decoder(x) + self.bias`, and `bias` is a standalone `nn.Parameter` of width vocab. HF
resizes the input embedding and the decoder weight and leaves it alone, so adding this
task's 5 tokens produces, on the **first forward** and not at the resize:

```
RuntimeError: The size of tensor a (38) must match the size of tensor b (33) at non-singleton dimension 2
```

`cpseq_mlm_data.add_task_tokens` already handles it. You only meet this if you add tokens
by hand. **The published checkpoints are already resized**, so loading them needs no fix —
`add_task_tokens` is a no-op on them because the tokens are already in the tokenizer.

## 4. Smoke-test the download

```bash
export PYTHONPATH="$REPO/script_utils/_compat_stubs:$REPO/src"
python - <<'PY'
from proteinfoundation.cpseq.mlm_propose import MlmProposer
p = MlmProposer("$DEST/runs/cpseq_mlm_pocket/checkpoints/hf_step60000", device="cpu")
p.load(); print("pocket_enabled:", p.pocket_enabled)   # must print True
out = p.propose(
    original="EYTMRFRGISDKTS", edit_mask=[0, 6, 13],
    pocket_seq="RL/FELG/V/IENFFTTDPISLCY/HRM/QV/HF",
    cyc_type=0, n=3, temperature=1.0, order="random", seed=7)
for o in out: print(o.sequence, o.edited_positions, round(o.logprob, 2))
PY
```

Three things to check in that output: the length is unchanged, only positions 0/6/13 differ
from the original, and `pocket_enabled` is `True`. If it prints `False` the proposer did not
find `manifest.json` two levels up from the checkpoint and silently assumed an arm — pass
`pocket_enabled=True` explicitly, or keep `manifest.json` where the layout above puts it.

## 5. The input format, which you must get exactly right

```
<cls> <pocket residues …> <eos> <cyc:TYPE> <peptide, masked at edit positions> <eos>
```

- **The pocket is a sequence of fragments, not a chain.** Residues within 8 Å of the
  peptide, in receptor chain order, with `/` between residues that are **not consecutive**
  in the receptor numbering. A real example is 28 residues in **8 disjoint fragments**:
  `RL/FELG/V/IENFFTTDPISLCY/HRM/QV/HF`. The `/` becomes a learned `<gap>` token. Feeding a
  contiguous receptor chain instead is not a smaller version of the same input — it asserts
  adjacency that does not exist, and the fragmented-vs-contiguous difference was measured
  elsewhere in this project at 22–34 points of ring closure.
- `cyc_type` is an int: `0` mainchain, `1` disulfide, `2` isopeptide, `3` unspecified
  (the classifier-free null). Passing the wrong one is not a small error — for disulfide the
  anchors must be CYS, and the token is most of what determines them.
- Rebuild pockets with `script_utils/cpseq_build_mlm_corpus.py` so the radius, the gap rule
  and the chain roles match training. Do not hand-roll it.

## 6. Retraining or continuing

```bash
# continue the same run: drop last.pt into checkpoints/ and resubmit; the trainer resumes
bash scripts/submit_cpseq_mlm.sh --train-only CPSEQ_MLM_MAX_STEPS=60000

# from scratch on this corpus (no CPSea PDB tree needed)
bash scripts/submit_cpseq_mlm.sh --train-only CPSEQ_MLM_MAX_STEPS=60000 \
  CPSEQ_MLM_RUN_NAME=my_arm
```

`scripts/cpseq_mlm_env.sh` holds every default and is sourced by each job; the submitter
writes one timestamped env file per submission and passes it **positionally**.

**Do not switch that to `sbatch --export`.** On this project's cluster any explicit export
list makes sbatch set `SLURM_GET_USER_ENV=1`, slurmd then fails to rebuild the login
environment on the compute node, and the job is requeued and *held*, stranding every
dependent job. Your scheduler may differ, but the file mechanism works everywhere and costs
nothing.

Scheduler assumptions worth re-checking on the new cluster, all of them local to here:

| assumption | here | what to check |
|---|---|---|
| partitions | `cpu` for corpus, `general` (all a6000 48 GB) for training | your names, and that the GPU class is pinned by partition or `--nodelist` |
| submit ceiling | ~33 jobs, and **every array task counts** | the budget guard in the submitter (`CPSEQ_MLM_MAX_SUBMIT`) |
| accounting | `sacct` **and** `sacctmgr` are dead (slurmdbd unreachable, exit 0 with no rows) | if yours work, you can drop the "read the job's own .out" workarounds |
| log paths | absolute, under scratch | a relative `#SBATCH --output=` lands in the repo |

## 7. How to read the results, and what they do not say

From `metrics.jsonl`. At step 60000, per-position accuracy on withheld positions only:

| regime | pocket top-1 | pocket top-3 | control top-1 | `P(r\|pos,L)` | **`P(r\|pos,L,cyc)`** | model margin |
|---|---|---|---|---|---|---|
| denovo | 0.235 | 0.401 | 0.205 | 0.189 | 0.202 | +0.033 |
| budget | 0.261 | 0.442 | 0.239 | 0.191 | 0.213 | +0.048 |
| anchor | 0.555 | 0.651 | 0.565 | 0.426 | **0.526** | **+0.029** |
| span | 0.184 | 0.361 | 0.164 | 0.121 | 0.128 | **+0.056** |
| **all** | **0.267** | **0.430** | 0.246 | 0.204 | **0.228** | **+0.039** |
| shuffled pockets | 0.226 | 0.357 | 0.246 | — | — | — |

**Use the `P(r|pos,L,cyc)` column as the bar.** The model is handed the chemistry token, so
a prior without it is handicapped. Quoting the chemistry-free prior overstated this arm's
margin by 60% (+0.063 vs the honest +0.039).

Two things that inverts. On `anchor` the prior rises 0.426 -> 0.526 once it sees the
chemistry, so the model's margin there collapses to **+0.029** -- a terminal disulfide
anchor must be CYS, which is a lookup, not a prediction. And `span`, which is the weakest
regime by raw accuracy, is where the model contributes **most** (+0.056): a contiguous run
cannot be guessed from position and chemistry, while a ring anchor largely can.

**Read these three things and not more:**

1. `val/baseline_positional_acc` is the bar, **not** the flat amino-acid marginal (~0.10).
   The positional prior is a parameter-free lookup table of P(residue | position, length)
   fitted on train — it sees no pocket, no chemistry and no other residue. It is strong only
   because CPSea cyclizes terminally, which is also why it reaches 0.426 on `anchor`.
2. `val/pocket_gain` = accuracy with real pockets minus accuracy with pockets permuted
   across the panel. It is **exactly 0.0000** in the control arm, which is what certifies
   the instrument has no offset. In the pocket arm it grew monotonically from +0.012 at step
   500 to +0.042 at the end, and the top-3 gain is larger (+0.073).
3. Compare arms **at equal steps**. At 200 steps the control was *ahead*; that was warmup.

**What these numbers are not:**

- They are **per-position**, micro-averaged over pooled withheld positions, so long masks
  dominate (`denovo` contributes 2522 of 4757 scored positions). There is **no whole-mask
  exact-match metric** — filling an entire mask correctly is a much harsher number and is
  not measured here.
- Each prediction is made with **all other masked positions still masked** — a leave-k-out
  marginal from one forward. The proposer deploys differently, filling one position at a
  time with a re-forward. So this is not the accuracy of the sampling procedure.
- The priors are **top-1 only**. There is no top-3 baseline, so the top-3 column has no
  matching bar; random top-3 of 20 residues is 0.15.
- **A 2.4 training loss is not independently impressive.** In nats, measured on 2.44M
  train peptides: uniform-20 = 2.996, amino-acid marginal = 2.883, +chemistry = 2.841,
  +position&length = 2.690, +position&length&chemistry = 2.557, **model = 2.364**. The
  entire learnable span from uniform to model is 0.63 nats and the model's share beyond
  the best lookup table is 0.19 nats (ppl 12.89 -> 10.63, a 17.5% reduction). The floor is
  high because residue identity in a 5-16-mer is intrinsically ambiguous -- visible in the
  per-regime spread under one set of weights: `anchor` ppl 4.14 vs `span` ppl 14.14.
- The arm-vs-arm gap (pocket 0.267 vs control 0.246) is **not significant**: positions are
  clustered within peptides (4757 positions but only ~800 peptides, 5.9 each), giving a
  clustered z of +1.00 against an unclustered +2.44. One seed per arm. The pocket-shuffle
  gain is the sound evidence -- within-model, identical positions, monotone in training
  (r=+0.89, late mean +0.0406 +- 0.0018 = 23 sd from zero).
- Nothing here measures design success. These are teacher-forced infilling accuracies. Ring
  closure and Rosetta dG are this project's actual criteria, and the arm has not yet been
  run through the Boltz-2 gate against `mpnn_redesign`.

## 8. Provenance

- Corpus: CPSea, `cpsea_{train,val}.parquet`. Train 2,444,208 of 2,444,209 rows (one
  rejected, `empty_pocket`). Chemistry resolved from CONECT records, **not** from the
  metadata column, which is `"other"` for 57% of the set because it cannot separate
  disulfide from isopeptide. Resolved counts match the parquet exactly where it can label.
- Holdout: CPSea's splits are Foldseek-cluster disjoint with **all three pairwise
  intersections exactly 0** (train 555,712 clusters / val 30,872 / test 30,874). Training on
  `train` and validating on `val` is therefore already a cluster-level holdout, and the
  `cpseq` eval targets come from `test`.
- Warm start: `ChatterjeeLab/PepMLM-650M` (`EsmForMaskedLM`, 33 layers, hidden 1280, vocab
  33 before our 5 added tokens, maxpos 1026). `facebook/esm2_t33_650M_UR50D` is
  architecturally identical and is the control for whether the PepMLM warm start mattered —
  that ablation has **not** been run.
