# CPSea unmasking sequence proposer (`cpseq_mlm`)

A masked-LM that takes a partially-masked cyclic-peptide sequence, a receptor pocket and a
requested cyclization chemistry, and fills the masked positions. Built to be the proposal
step of `cpseq` search: one forward per filled position, so it slots in beside the random
mutation bands, the single-substitution scan, the MPNN redesigns and the Llama proposer and
is measured through the same Boltz-2 refold and the same frozen gate.

Started 2026-10-29. Everything below was verified by execution unless it says otherwise.

---

## 1. Why a masked LM, and not the flow co-generator's sequence pathway

The original plan was to warm-start from the sequence-conditioned co-generator checkpoint
and "isolate the sequence part". That turns out not to be a slice-out:

- `local_latents_transformer.py` has `output_parameterization = {bb_ca: v, local_latents:
  v}` and **no residue-type head**. The sequence is a *readout of the latents through the
  frozen AE decoder* (`utils/seq_conditioning.py` states this explicitly).
- Sequence enters the trunk as an **input feature** (`optional_res_type_seq_feat`, dim 20,
  all-zeros = unspecified) and never leaves as logits.

So isolating it means attaching a *new* 20-way head and training it; the warm start would
buy the trunk's representation, not the readout. Against that, generating a proposal
through the co-generator costs a 200–400 step ODE, and the search wants dozens of
proposals per target.

A masked LM gives the same interface for one forward per position and no AE in the loop.
The co-generator-head route remains the interesting stage 2, because it is the only option
that sees **pocket geometry** rather than pocket sequence.

## 2. Prior art this follows

| Work | What is taken from it |
|---|---|
| [PepMLM](https://arxiv.org/abs/2310.03842) (Nat Biotechnol 2025) | The whole shape: ESM-2 fine-tuned with span masking, target sequence first, masked binder at the C-terminus. Published for this exact task, sequence-only, no structure needed. Its 650M checkpoint is the warm start. |
| [PepTune](https://arxiv.org/abs/2412.17780) | States the gap we fill: sequence-only generators like PepMLM are limited to the 20 wild-type residues and so cannot express cyclic chemistry. Hence the explicit chemistry token. |
| [EvoDiff-OADM](https://elifesciences.org/reviewed-preprints/112029) | Order-agnostic masking, so infilling needs no special machinery — just mask and decode. |
| [Path Planning / P2](https://iclr.cc/virtual/2025/36160) (ICLR 2025) | Unmasking **order** is a first-class lever: +22% relative protein foldability from the order alone. Hence `order={confidence,random}`, recorded per proposal. |
| [Brooks et al.](https://arxiv.org/abs/2402.15020) (NeurIPS 2024) | An MLM gives marginals, not a joint. Hence one position per forward, never one-shot argmax over all masked positions. |
| [Gibbs sampling of protein MLMs](https://github.com/seanrjohnson/protein_gibbs_sampler) | One-position-at-a-time masking proposes sequences *close to the seed* — exactly the budget-constrained behaviour the search wants. |

## 3. Input layout

```
<cls> <pocket residues …> <eos> <cyc:TYPE> <peptide, masked at edit positions> <eos>
```

Two additions to PepMLM's layout:

- **`<cyc:TYPE>`** — `mainchain` / `disulfide` / `isopeptide` / `unspecified`. A ring is
  not a property of a sequence: the same 8-mer is a different molecule head-to-tail than
  disulfide-closed, so chemistry must be an input. `<cyc:unspecified>` is the
  classifier-free null, which `--cyc-dropout` (default 0.15) rewrites rows to.
- **`<gap>`** — between pocket residues that are not consecutive in the receptor.

The gap token is not cosmetic. A measured example:

```
pep (14): EYTMRFRGISDKTS   cyc=mainchain  anchors=(0,13)
pocket (28 res of 79): RL/FELG/V/IENFFTTDPISLCY/HRM/QV/HF
```

28 pocket residues in **8 disjoint fragments**. Without a marker the model is told those
fragments are one contiguous chain. Project memory measured a contiguous full receptor
against a fragmented pocket crop at **+22 to +34 points** of ring closure, so this is the
difference between conditioning on the real pocket and on a receptor that does not exist.

## 4. The chemistry label comes from CONECT, not from the metadata column

The parquet's `cyclization_type` is `other` for **57%** of CPSea (1.39M of 2.44M rows).
`cyclization/parse_labels.py` documents why: the column cannot separate disulfide from
isopeptide and lumps every isopeptide into `other`. Training on it would teach the model
that the majority chemistry is "unknown".

The corpus builder calls `infer_cyclization_label`, which resolves the actual bond. On a
300-row val sample that gave **mainchain 89 / isopeptide 173 / disulfide 38 and zero
unspecified** — i.e. it recovered 173 isopeptides the metadata column calls `other`.

It also returns the bond's endpoints, which the `anchor` mask regime needs. Measured
anchors land at `(0, L-1)`, consistent with CPSea being terminal-only.

## 5. The mask schedule is a mixture, not a rate

A model trained at one mask rate is asked, at inference, a question it never saw. The
schedule samples a **regime** per example and every metric is reported per regime; a
pooled accuracy over regimes of unequal difficulty is uninterpretable.

| Regime | Default weight | What it is |
|---|---|---|
| `denovo` | 0.25 | every position masked — PepMLM's own regime, and the search's round-0 query |
| `budget` | 0.45 | `k ~ U{1..floor(L/2)}` positions, uniformly placed — the search's edit query |
| `anchor` | 0.15 | the cyclization anchors (+0–2 more) — the LP→CP query, since what a linear peptide lacks is the anchor identities |
| `span` | 0.15 | one contiguous run — PepMLM masks spans |

`floor(L/2)` is exactly `cpseq.search.budget_for`, asserted equal in
`test_cpseq_mlm.py`, so a `budget`-regime draw cannot exceed the gate the `Proposal`
dataclass enforces.

## 6. What is measured, and against what

**Only withheld positions are scored.** `labels` is `-100` on every revealed position.
This project has twice produced a sequence metric that looked like a capability and was
not — `cond_recovery = 1.0` on a fully-revealed sequence, and an `seq_recovery ≈ 1.0` that
was 99.5% copying. A revealed position carrying a label is that bug.

Every eval reports, on the identical hidden positions:

- `marginal_argmax_acc` — always guess the commonest residue.
- `positional_argmax_acc` — **the baseline that matters.** CPSea cyclises terminally, so
  positions 0 and L−1 are anchor-enriched; a model that learns only "anchor at the ends,
  hydrophobic in the middle" beats the flat marginal while knowing nothing about any
  pocket. A sibling arm in this project landed at 9.77% against a 9.7% always-guess-leucine
  prior, which read as learning until the prior was computed.
- `val_pocketshuf/*` — **the pocket-shuffle control.** Same peptides, same masks, pockets
  permuted across examples. A pocket-conditioned model must lose accuracy;
  `val/pocket_gain` is the difference. Two arms in this project shipped a conditioning
  channel whose gates never opened, so this is not a formality.

The eval panel is **fixed** (one seed, stratified by regime) because every comparison here
is paired — checkpoint vs checkpoint, pocket vs no-pocket, real vs shuffled. Redrawing
masks per eval would put sampling noise on the measured axis, and with 5–16-residue
peptides that noise is large.

### The paired control arm

`--no-pocket` trains an otherwise-identical twin with the pocket segment dropped
entirely (not masked, not padded — a test asserts the peptide region is byte-identical and
only shifted). It is the other half of the pocket-shuffle question, and it doubles as the
cheap chemistry/foldability prior.

## 7. Running it

```bash
bash scripts/submit_cpseq_mlm.sh --smoke     # 1 shard, 3000 rows, 200 steps, both arms
bash scripts/submit_cpseq_mlm.sh --corpus    # corpus array only
bash scripts/submit_cpseq_mlm.sh --submit    # full chain
bash scripts/submit_cpseq_mlm.sh --submit CPSEQ_MLM_LR=2e-5 CPSEQ_MLM_MAX_STEPS=40000
```

The chain: `corpus(val)` and `corpus(train)` as sibling CPU array jobs, then the two
training arms as siblings on `afterok` of both — siblings because they are the paired
comparison and should run on the same corpus at the same time.

Conventions followed: config travels in a per-submission **env file passed positionally**
(never `sbatch --export`, which triggers the user-env-retrieval requeue-hold here); logs,
corpora, run dirs and checkpoints all land on scratch, never home; `set -euo pipefail` in
each job body; gates inside jobs `exit 0` with a reason so dependents are not stranded in
`DependencyNeverSatisfied`; corpus shards are resumable via `.done` markers and never share
an output file.

| Knob | Default | Note |
|---|---|---|
| `CPSEQ_MLM_MODEL` | `ChatterjeeLab/PepMLM-650M` | `facebook/esm2_t33_650M_UR50D` is the control for "did the PepMLM warm start help". Architecturally identical: `EsmForMaskedLM`, 33 layers, hidden 1280, vocab 33, maxpos 1026. |
| `CPSEQ_MLM_LR` | `1e-5` | Errs low on purpose: project memory records an LP-generator run diverging on an AF3-scale `1.8e-3` at small effective batch. |
| `CPSEQ_MLM_NUM_SHARDS` | 24 | Set by the **QOS submit ceiling (~33 jobs, array tasks count), not by throughput**: 24 shards + 2 trainers = 26 fits, 32 + 2 does not. See the throughput table below. |
| `CPSEQ_MLM_VAL_SHARDS` | 4 | Val is 134k rows against train's 2.44M. `--skip-val` reuses what is already on disk. |
| `CPSEQ_MLM_MAX_SUBMIT` | 30 | The submitter refuses rather than half-submitting a chain. An overflow is rejected by the QOS, not queued, which would leave the trainers waiting on a dependency that never resolves. `sacctmgr` cannot report the real limit — it fails against slurmdbd exactly as `sacct` does. |
| `CPSEQ_MLM_POCKET_RADIUS` | 8.0 Å | Heavy-atom to heavy-atom, matching `CroppingTransform2`'s `interface_threshold` rather than inventing a second definition of contact. |
| `CPSEQ_MLM_HOLDOUT_CLUSTERS` | unset | **Not needed, verified.** CPSea's splits are Foldseek-cluster disjoint with *zero* overlap — measured: train 555,712 clusters / val 30,872 / test 30,874, and all three pairwise intersections are 0. Since the `cpseq` eval targets come from the held-out test split, training on `train` and validating on `val` is already a cluster-level holdout. The knob stays for narrower holdouts. |

Training runs on `general` (all a6000, 48 GB) rather than `preempt`, so the GPU class is
pinned by the partition and cannot land on an 11 GB 2080 Ti.

### Corpus throughput: read the aggregate, not the per-shard rate

The extraction is I/O bound — 85% of wall time is NFS wait on a directory holding 2.44M
files. Measured:

| concurrent shards | per-shard | aggregate |
|---|---|---|
| 1 (idle node) | 51 rows/s | 51 rows/s |
| 2 | 15.1 rows/s | 30 rows/s |
| 32 | 6.3 rows/s | **202 rows/s** |

Each shard slows ~8× under a 32-task array and total throughput still rises 4×. The
per-shard collapse is contention, not waste, so **a falling per-shard rate is not a reason
to cut the shard count** — on first seeing it I did exactly that, wrongly. The limit on
shard count here is the QOS, not the filesystem.

## 8. Using it as a proposer

```python
from proteinfoundation.cpseq.mlm_propose import MlmProposer

p = MlmProposer("$RUN/checkpoints/hf_step20000")
props = p.propose(
    original="EYTMRFRGISDKTS", edit_mask=[0, 6, 13],
    pocket_seq="RL/FELG/V/IENFFTTDPISLCY/HRM/QV/HF",
    cyc_type=0,            # mainchain
    n=8, temperature=1.0, order="random", seed=1234,
)
```

- **The mask is the budget.** Revealed positions are copied verbatim, so
  `n_mutations <= floor(L/2)` holds by construction — nothing is rejected and resampled,
  so the proposal distribution is not quietly reshaped by a rejection filter. A mask larger
  than the budget is refused up front by `validate_request`, before any weight load.
- **CYS is excluded by default**, matching `search.default_alphabet`: a free cysteine can
  pair and close a ring nobody asked for, making the molecule differ from the chemistry its
  own label names.
- `order="random"` for search diversity, `order="confidence"` for a single best proposal.
- `score()` gives the masked-LM **pseudo**-perplexity (one forward per position, that
  position alone masked) — the quantity PepMLM reports. It is not a joint likelihood and
  must not be compared against an autoregressive model's perplexity as though it were.

## 9. Two traps that cost real time here

**`transformers` 5.x cannot load ESM in this venv without a shim.** Every model class
reaches torchaudio on its import path:

```
AutoModelForMaskedLM -> modeling_utils -> loss.loss_utils -> loss_rnnt
                                                             -> if is_torchaudio_available(): import torchaudio
```

The guard *passes* because torchaudio is installed, and the import then dies with
`undefined symbol: torch_library_impl` — its C extension was built against a different
torch than the installed 2.7.0+cu126. `grep -rn torchaudio src/ script_utils/ scripts/`
finds no user in this repo, so the fix is an import-only stub at
`script_utils/_compat_stubs/torchaudio/`, prepended to `PYTHONPATH` by the training job.
Deliberately *not* an uninstall: the venv is shared by the whole flow stack, and the stub
is additive and reversible. It is empty, so any real attribute access still fails loudly
rather than silently returning a mock.

**`resize_token_embeddings` does not resize ESM's output bias.** `EsmLMHead` computes
`self.decoder(x) + self.bias`, where `bias` is a standalone `nn.Parameter` of width vocab,
not the decoder Linear's bias. HF resizes the input embedding and the output embedding it
finds via `get_output_embeddings()`, and leaves that parameter alone, so adding the 5 task
tokens produces:

```
RuntimeError: The size of tensor a (38) must match the size of tensor b (33) at non-singleton dimension 2
```

The resize itself raises nothing — **only a forward pass finds it**, which is why
`test_resized_model_survives_a_forward_pass` does a forward rather than checking shapes.
`add_task_tokens` now grows the bias with zeros for the new entries, the neutral prior for
a token the model should never emit.

## 10. Status

- Corpus extractor: **built and validated** on 300 real val rows, 0 rejections. Vectorising
  the pocket cutoff took it from 0.41 s/row to 0.019 s/row (21×, byte-identical output);
  the pure-Python version would have cost 278 core-hours over CPSea.
- Data path: **21 unit tests pass**, covering the copy-metric trap, budget agreement with
  the search gate, gap-token handling, the no-pocket single variable, and the proposer gate.
- Training script and proposer: **written, smoke in flight.** No training run has completed,
  so **there are no accuracy numbers yet** and nothing here claims any.
- Not yet done: the eval-target cluster holdout list (§7), wiring the proposer into
  `cpseq_search.py` as an arm, and the head-to-head against MPNN redesign — which is the
  bar this arm has to clear, since MPNN is already a structure-conditioned unmasking
  proposer. Its edge should be that it needs no CP backbone at round 0 and knows the
  cyclization chemistry.
