# D-Flow for LP → CP: optimizing the source state of a frozen CP generator

Branch `dflow-lp2cp`, worktree `.claude/worktrees/dflow-lp2cp`.
Method reference: Ben-Hamu et al., *D-Flow: Differentiating through Flows for Controlled
Generation*, ICML 2024 — <https://proceedings.mlr.press/v235/ben-hamu24a.html>

## What this does

The pretrained target-conditioned CPSea flow is left exactly as it is. What gets optimized is
the **continuous state the deterministic sampler starts from**, by differentiating through the
whole integration trajectory and the autoencoder decode into a loss defined on the **final
decoded structure**. Gradient descent on the input of a frozen generative map.

This is **not** the per-step guidance already in `scripts/sdedit_guidance.py`. That reads
`dL/dx_t` at each Euler step, nudges the state, and throws the graph away — nothing connects
step *k* to step *k+1*, so it cannot trade a worse step now for a better final structure. Here
**one graph spans the entire rollout**.

### Three things had to be true, and only one of them already was

1. **Memory.** `partial_simulation` accepts `grad_enabled`, but keeps every step's activations
   alive; at the production `nsteps=400` that is 400 simultaneous network activations, which
   does not fit. `scripts/dflow_rollout.py` runs its own Euler loop with each step wrapped in
   `torch.utils.checkpoint`: memory becomes one forward plus `nsteps` small state tensors, at
   ~2× forward cost.

2. **Determinism.** The production sampler is an **SDE** — `configs/pipeline/model_sampling.yaml`
   sets `sampling_mode: sc` for both tracks, which draws a fresh `eps` every step. With fresh
   noise the map source → sample is not a function, and the gradient would be through one
   arbitrary realization that changes every optimizer step. Two deterministic integrators are
   exposed and they are **not the same experiment**:
   - `vf` — the pure probability-flow ODE, `delta_x = v * dt`. What D-Flow assumes. Default.
   - `sc` — the production SDE with its noise sequence **frozen** by reseeding per rollout, so
     `eps` is a constant and the gradient is correct.

3. **Fidelity to the production sampler.** A reimplemented loop that quietly differs would make
   every number here incomparable to everything else on this repo. `test_rollout_matches_partial_simulation`
   checks it **bit for bit** against `partial_simulation` on a real `ProductSpaceFlowMatcher`,
   for both integrators. That test is the load-bearing one.

### Two initializations, labelled separately and never pooled

| arm | `t_start` | what is optimized | status |
|---|---|---|---|
| `source` | (0, 0) | the source noise itself | **D-Flow as published** |
| `sdedit` | (`t_ca`, `t_lat`) > 0 | an intermediate SDEdit state built from the encoded input peptide | an **adaptation**, reported as its own arm |

The `sdedit` arm inherits SDEdit's own bias toward the input, so pooling the two would let that
bias be reported as a property of D-Flow. They are separate rows everywhere.

## The acceptance contract

`CONTRACT` in `scripts/dflow_losses.py` is the single source of truth, read by **both** the
differentiable loss and the hard gates — so a threshold cannot be changed for one without
changing it for the other. Fixed operational thresholds, not knobs to relax if the pass rate
disappoints:

| clause | threshold |
|---|---|
| ring | requested terminal chemistry closes, measured on the **decoded** structure with the repo's own `*_BOND_WINDOW_A` acceptance windows, **and** declared as a CONECT record |
| sequence | ≥ 80% identity to the **original LP**, ≤ 2 substitutions total (anchor changes counted), protected positions immutable |
| contacts | ≥ 80% of original peptide–receptor residue-pair contacts retained; contact = any heavy-atom pair ≤ 4.5 Å |
| pose | whole-peptide Cα RMSD ≤ 2 Å **and** original-interface-residue Cα RMSD ≤ 1 Å, in the receptor frame with **no superposition** |
| clash | no heavy-atom pair below `clash_min_A` (see calibration below) |
| energy | paired Rosetta interface score, CP − original LP, ≤ +5 REU; strict ≤ 0 tier reported alongside. **REU are not kcal/mol.** |

Reported but never substituted for the gates: mutation budgets 0/1/2/3 and identity tiers
90%/80%/70%.

### Why Cβ proximity is insufficient, and why the closure loss cannot be allowed to vanish

Cβ distance was measured on this repo at **AUC 0.462** for separating closed from open rings —
chance. The ring *shape* forms whether or not a bond exists. So Cβ enters the loss only as a
*reachability bracket*; acceptance reads the anchor-atom bond distance and nothing else.

Worse, `per_sample_requested_bond_distance` multiplies its distance by `atoms_valid`, and the
decoder zeroes coordinates of atoms the argmax residue does not have. For a disulfide request
on a sample whose endpoints did not decode as cysteines there is no SG atom, so the bond term
is **exactly zero with exactly zero gradient** — and measured abstention on the earlier SDEdit
grid was 83–100% for disulfide/isopeptide at `t_lat ≥ 0.6`. That is the common case, not an
edge case. So three terms carry closure and `ring_terms` **refuses to build an objective that
has only the vanishing one**:

- `anchor_ce` — cross-entropy on the **decoded sequence logits** at the two endpoints. Always
  defined, and it addresses the failure actually measured: isopeptide failure was ~50% anchor
  *identity*, with NZ–CG already at 1.3 Å once the residues were right.
- `anchor_cb` — flat-bottom on endpoint Cβ–Cβ with a chemistry-specific reachability window;
  differentiable even on a glycine endpoint via the virtual-Cβ fallback.
- `bond_fb` — flat-bottom on the true anchor-atom distance. Live once identity is right, which
  is when it matters: it sharpens geometry rather than finding it.

Do **not** reach for the decoder's pre-mask coordinates to "fix" the vanishing term: slots for
atoms a residue does not have are never supervised during AE training, so an SG position read
off a proline is an arbitrary number.

### Sequence retention is checked explicitly, on the decoded argmax

A latent-distance penalty does not guarantee sequence retention, and neither does freezing the
latent track: `t_lat = 1` holds the latent fixed, but decoding also reads the Cα trace, so a
backbone that moved can decode to a different residue with the latent untouched. The only
reliable statement is the one made on the final decoded sequence, which is what
`sequence_report` does. Identity is computed with **integer arithmetic** — `(~sub).float().mean()`
on 9 of 10 matches returns 0.8999999761581421, which is below 0.9, so a candidate exactly at a
tier would be rejected by float32 rounding alone.

### The ring is declared, and the declaration is verified from disk

A macrocycle written as plain ATOM records is, to every downstream consumer including Rosetta,
a linear peptide whose ends happen to be near each other. `scripts/dflow_pdb.py` writes the ring
bond as a CONECT record and `verify_ring` **reads the file back** to confirm it names the right
two atoms at a distance inside the acceptance window. Asserting that from the tensors would
prove nothing about what landed on disk.

**Known gap, stated rather than papered over:** `_declare_cyclic_bond` handles mainchain
(`declare_chemical_bond`) and disulfide (`add_disulfide_bond`) but returns False for
**isopeptide** — no residue-connection patch is wired up, so Rosetta scores a topology whose
closing bond it does not know about. That is not silently fixed; instead every row carries
`topology_declared`, FastRelax runs with `constrain_relax_to_start_coords(True)`, and
`ring_dist_before_A` / `ring_dist_after_A` measure whether the undeclared ring actually drifted.
The report keeps isopeptide's energy in its own tier.

## Instrument calibration — run *before* the experiment

`script_utils/dflow_calibrate.py` scores each **native crystal peptide against itself**. Results
on all 60 LNR test peptides (`evaluation_results/dflow_calibration_full/calibration.json`,
2026-09-28):

- **Arithmetic identities perfect, 60/60**: identity 1.0, 0 substitutions, contact retention
  1.0, Cα RMSD 0.0. A failure here would have been a code bug.
- **Positive control holds**: `gate_ring_closed` is 0/60 on these linear inputs, so the ring
  clause is being evaluated and is not passing vacuously.
- **One real fault caught.** At the initial `clash_min_A = 2.7 Å`, only **17 of 60 natives
  passed** — 2.7 Å is a hydrogen bond, not a clash. Native min heavy-atom distance to receptor:
  min 1.45 Å, p05 2.13 Å, median 2.64 Å.

  | `clash_min_A` | natives passing |
  |---|---|
  | 2.7 | 17/60 |
  | 2.5 | 46/60 |
  | 2.3 | 54/60 |
  | **2.0** | **59/60** |

  Left at 2.7 Å this would have driven `gate_clash`, and therefore joint success, to ~0 for
  every arm — a number that looks like a model result and is an artifact. **This project has
  already scored ten runs at 0.0 that way.** `clash_min_A` is now **2.0 Å**: below every real
  contact in the native set, but two non-bonded heavy atoms inside 2.0 Å is genuine overlap.

  This is calibration of the one clause stated *qualitatively* ("no severe clashes"). The
  numeric thresholds are fixed and were not moved.
- **Interface RMSD is usually not an independent test.** Interface residues / peptide length has
  median 0.92, and for **29 of 60** peptides the interface *is* the whole peptide — so the 1 Å
  interface clause simply supersedes the 2 Å whole-peptide clause on half the set. The report
  must not present them as two independent gates.

## Running it

Everything is submitted; nothing runs on a login node. Configuration travels as a **timestamped
env file passed positionally** — never `sbatch --export`, which sets `SLURM_GET_USER_ENV=1` and
gets jobs requeued and **held**, stranding every `afterok` dependent.

```bash
# smoke first: 1 peptide, 2 restarts, tiny grids, full chain
scripts/submit_dflow.sh --smoke --shards 1 --rosetta-shards 1 --nodelist gpu28

# the pilot
scripts/submit_dflow.sh --tag src --init-mode source \
  --shards 4 --rosetta-shards 4 --restarts 16 --opt-steps 60 \
  --opt-nsteps 30 --eval-nsteps 200 \
  --examples-file evaluation_results/_dflow_env/pilot_examples.txt \
  --exclude gpu29 --baselines "sdedit|<...>/edits_shard*.jsonl sdedit_guided|<...>/edits_shard*.jsonl"

scripts/submit_dflow.sh --dry     # print the plan, submit nothing
```

Stage graph — independent stages stay siblings so they queue concurrently, and everything that
only reads saved artefacts runs on CPU, off the GPU path:

```
calibrate (CPU) ........................... instrument check, no dependency
opt (GPU array) → rescore (CPU) → rosetta (CPU array) → report (CPU)
```

**Resumable by construction.** Each shard writes its **own** results file — concurrent
`O_APPEND` has NUL-corrupted rows on this cluster (88 of 144 lost) — and every stage skips
`run_key`s already present. `MAX_SECONDS` makes a job stop *cleanly* under the wall rather than
be killed at it, so hitting the limit means resubmitting the same command, not restarting. The
report reads only saved artefacts, so a new cut or a threshold question is a re-run of the
**report** job, not of the GPU stage.

### Coarse grid to optimize, fine grid to judge

Optimization runs on `--opt-nsteps` (short, so the graph fits and iterations are cheap); every
restart is then re-rolled and re-scored on `--eval-nsteps` (fine). `grid_consistent` records
whether the coarse and fine grids agreed on ring closure, which is how a numerical artifact of
the short grid gets caught rather than reported.

### Every restart is scored twice, and the pair is the point

`stage="init"` is the restart's **unoptimized** source rolled out and scored; `stage="opt"` is
the same source after optimization — same seed, same schedule, everything else identical.
Without that pairing a headline number cannot distinguish *the optimizer worked* from *16
restarts found a good sample*, which is the first thing to ask of a method whose selling point
is optimization. The report's paired table is computed only on restarts where **both** stages
scored, and reports the McNemar cells (gained / lost), not two marginal rates subtracted.

### One instrument for every arm

The GPU driver computes acceptance from its own tensors, but the **comparison** is made by
`script_utils/dflow_rescore.py`, which reads **PDBs** — for D-Flow, for SDEdit and for guided
SDEdit alike — through the same `accept()` with the same thresholds. The older arms' native
columns used a different contact definition (Cα–Cα within 10 Å) with no clash, sequence-budget
or interface-RMSD test at all, so they are carried through for reference and never used as the
verdict.

Two deliberate choices there:
- `--examples-file` scopes the baselines to the **same peptides** the D-Flow arms attempted, so
  the comparison is 8 peptides against 8, not 8 against 60.
- `file_ring_declared` is recorded but left **out** of the joint gate. The SDEdit runs predate
  the CONECT writer, so gating on it would make the baselines look worse for a reason about the
  writer rather than the structure. The joint gate's ring clause is `gate_ring_closed`, measured
  from anchor atoms and equally available to every arm. (Rosetta is unaffected either way —
  `_declare_cyclic_bond` takes explicit endpoints and never reads CONECT.)

**Staging must match.** The baselines were run on the **pocket-cropped** receptor
(`CPSea_data/lnr_pocket14`). LNR's full contiguous receptor differs from the ~14-fragment pocket
the model was trained on, and that mismatch is worth **+22 to +34 points of closure** on its own
— enough to have invalidated an earlier SDEdit sweep. `submit_dflow.sh` therefore defaults
`--metadata` to the pocket staging; an arm run on `lnr_staged` is a different experiment and
cannot be compared to these baselines.

## Pilot design

8 cluster-unique LNR test peptides, stratified by input N–C gap (closure is gap-gated on this
project, so a set drawn from one end of that distribution would answer a different question).
Pinned as an artefact in `evaluation_results/_dflow_env/pilot_examples.txt`:

| example | length | N–C gap (Å) | pocket receptor |
|---|---|---|---|
| LNR_2qos_C_A | 11 | 5.3 | 126 |
| LNR_4w50_B_F | 12 | 11.9 | 99 |
| LNR_3t6j_A_B | 5 | 15.2 | 227 |
| LNR_1t4f_M_P | 9 | 16.6 | 71 |
| LNR_1fiw_A_L | 14 | 20.2 | 113 |
| LNR_2np0_A_B | 15 | 25.8 | 83 |
| LNR_6fq4_A_B | 16 | 28.5 | 133 |
| LNR_1jrr_A_P | 14 | 45.6 | 230 |

× 3 chemistries × 16 restarts × 60 optimization updates, one objective setting, both init arms.

## Results

See `RESULTS_DFLOW_LP2CP.md` (written by the `report` stage into each run's
`report/RESULTS.md`, with the cross-arm summary consolidated by hand).
