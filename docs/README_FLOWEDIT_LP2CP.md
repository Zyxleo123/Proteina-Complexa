# FlowEdit LP -> CP: bound linear peptide -> cyclic peptide

Method, dependencies, and how to run it. Numbers live in
`evaluation_results/<run_tag>/report/RESULTS.md`, written by the report stage; this file is the
standing description and deliberately carries no result a later run would supersede.

Reference: Kulikov et al., *FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow
Models*, arXiv:2412.08629.

---

## 1. Why FlowEdit rather than SDEdit or DDIB

All three ask the same thing -- take a peptide the model already binds, make it cyclic, and do
not wreck the sequence or the pose -- and differ in how the input survives.

**SDEdit** noises the input and re-samples, so the input survives only as whatever the noise did
not destroy. The measured consequence on this model is stark: at `t_lat = 1` (frozen sequence)
closure is **1/59**, at every `t_ca`. Closure arrives only once the sequence is allowed to move a
long way -- the same failure the edit was meant to avoid, in a different coordinate.

**DDIB** needs an inversion, and an inversion is only as good as its round trip.

**FlowEdit** needs neither. It never starts from noise and never inverts. It integrates the
**difference** between two conditional velocity fields, so the trajectory begins at the input and
moves only insofar as the two conditionings disagree. Where the cyclic and linear fields agree,
the state does not move at all. That is a property of the construction, not of a tuned
hyperparameter.

It is asserted as a test: with source and target conditioning made identical the difference is
identically zero and the output must equal the input to the last bit
(`script_utils/test_flowedit.py::test_identical_conditioning_is_a_no_op`). If that ever breaks,
every retention number this pipeline reports is inflated by an amount nothing downstream can
detect.

**The corollary matters as much as the property.** If the model does not yet separate the two
conditionings, a faithful FlowEdit correctly does almost nothing -- which looks identical to a
broken sampler and calls for the opposite response (train longer, not fix code). So the sampler
reports `cond_separation_<track>` = mean |V_delta| / |v_target|. Read it before concluding
anything from a small edit.

---

## 2. The checkpoint dependency, stated plainly

FlowEdit needs a **source** and a **target** velocity field over the **same representation**.
Here they are one network under two conditionings:

| branch | `cyclization_type_cond` |
|---|---|
| source | `LINEAR` (index 4) |
| target | the requested chemistry -- `MAINCHAIN` 0 / `DISULFIDE` 1 / `ISOPEPTIDE` 2 |

That single key drives **both** the type embedding and the ring positional encoding -- `LINEAR`
yields the all-zero ring feature through `is_ring_request` -- so the receptor conditioning is
identical between branches *by construction*, not by careful matching.

`LINEAR` must be the trained token, never `UNSPECIFIED` (index 3), which is the
classifier-free-guidance null: labelling the linear branch with the null would redefine what
`guidance_w > 1` pushes away from.

**So the model must have been trained on `LINEAR`, on a shared LP+CP autoencoder.** A CP-only
checkpoint predates the token; `CyclizationTypeSeqFeat` pads its embedding with a zero row and
`forward` raises on use. The driver turns that into an up-front refusal naming the missing
dependency (`--require-trained-linear`, default on).

### How to check a checkpoint really has it

The conditioning embedding is `nn.cond_factory.feat_creators.2.embedding.weight`. Measured:

| checkpoint | shape | LINEAR row |
|---|---|---|
| `cpsea_lpmix_from_v4cfg` (step 3000) | (5, 256) | L2 = 15.24, in line with the other rows (15.2-17.7) |
| `cpsea_bondunroll_pin20260828` (CP-only) | (4, 256) | absent |

A zero-norm row 4 means padded-and-untrained, and any "FlowEdit" result from it is not one.

### State of the chain

| dependency | state |
|---|---|
| Shared LP+CP autoencoder (`shared_ae_lpcp_128`) | trained; step-40000 EMA pinned to `frozen_ae.ckpt` |
| LP-mixing flow arm (`cpsea_lpmix_from_v4cfg`) | training (job 67667); LINEAR trained by step 3000 |
| Mixed LP+CP metadata, LNR held out by id and sequence | built, 2,489,756 train rows |

The autoencoder was **not** reusable. The existing CPSea AE (`finetune_full_128`) reconstructs
linear peptides at **0.320 A** side-chain RMSD against **0.058 A** for cyclics -- 5.5x,
length-matched, with a tail to 1.014 A -- and side-chain precision is exactly what isopeptide
closure spends. Reusing it would have made "LP mixing does not help" indistinguishable from "the
AE cannot encode LP". See `README_LP_AE_ROUNDTRIP.md`.

The flow arm warm-starts from the base `complexa.ckpt`, not a v4 CPSea checkpoint, so its loss
curves are **not** comparable to the v4/bondunroll lineage. Read `val_gen/cyc/*_success`, not
train loss, and say which step a result came from.

---

## 3. The method, adapted to this model

### 3.1 The time convention is flipped

The paper runs `t = 1` (noise) -> `t = 0` (clean). This codebase's `interpolate` is
`x_t = (1-t)*x_0 + t*x_1` with `x_0` noise, so **`t = 0` is noise, `t = 1` is clean**, and the
sampler integrates upward with `x += v*dt`.

Writing the paper's `t` as `1 - s` for our `s`:

| paper | here |
|---|---|
| `Z_t^src = (1-t)*X^src + t*n` | `(1-s)*n + s*X^src` -- exactly `interpolate(x_0=n, x_1=X^src, t=s)` |
| `Z^FE <- Z^FE - dt*V_delta`, `dt < 0` | `Z^FE <- Z^FE + ds*V_delta`, integrating `s` upward |

The interpolant is therefore form-identical, and we call the model's own `interpolate` so masking
and stochastic centering are not re-derived. The update is the ordinary forward Euler step with
the velocity replaced by the velocity **difference**.

Pinned by an analytic test: a constant velocity difference `c` must move the state by exactly
`c*(1 - t_start)`. Getting `dt` backwards, or integrating downward, flips that result's sign.

### 3.2 Two generated tracks, with independent `t`

The flow is a product space over `bb_ca` (CA coordinates) and `local_latents` (the autoencoder's
per-residue sequence + side-chain code). `t` is sampled **independently per track** in training,
so a per-track start time is in-distribution rather than a hack.

The editing interval therefore has a per-track form, and it is the axis that matters:

- **`t_lat = 1` freezes the sequence** -- its `dt` is identically 0, so the track never leaves the
  clean source. The direct analogue of the paper's `n_max` edit-strength knob.
- **`t_ca`** sets how far the backbone may travel.
- **`edit_end_frac`** is the analogue of `n_min`: past it the delta is dropped and the plain
  **target** field finishes the trajectory.
- **`n_avg`** is the paper's noise averaging. Each draw costs **two** network forwards.

### 3.3 Why the integrator is forced to pure `vf`

`simulation_step`'s score-based modes (`sc`, `vf_ss`, `vf_tsr`) convert velocity to score via
`vf_to_score(x_t, v, t)`, an identity that holds only when `v` is the true field **at** `x_t`. A
velocity *difference* is not a velocity field at `Z^FE`, so that conversion has no meaning and the
injected noise would not be the paper's algorithm. `sampling_mode` is forced to `vf` with
`gt = 0`; `ddib_invert.py` forces the same thing for the same class of reason.

`center_every_step` is forced **off**: the peptide is scored in the receptor frame, and
re-centering the binder each step would move it out of that frame.

### 3.4 Cost

A FlowEdit candidate costs `nsteps * 2 * n_avg` forwards, **not** `nsteps`. At the sampling
config's `nsteps = 400` that is **800 per candidate**; the SDEdit arm costs 400. Billing them
alike overstates the control by 2x.

---

## 4. The acceptance contract

Fixed thresholds, one code path for both arms (`proteinfoundation.editing.contract`). The sweep
knobs are separate reporting columns and never move the bar.

| gate | threshold |
|---|---|
| ring closed | the real bond atoms of the requested chemistry, in window, on anchors that can host it |
| severe clashes | 0 non-bonded heavy-atom pairs < 2.0 A |
| sequence identity to the **original** LP | >= 80% |
| substitutions (anchors included) | <= 2 |
| protected residues | unchanged |
| peptide-receptor contact retention | >= 80% of pairs with any heavy atom <= 4.5 A |
| whole-peptide CA RMSD | <= 2 A, receptor frame, no independent alignment |
| original-interface CA RMSD | <= 1 A, same frame |
| paired Rosetta interface score, CP - LP, **after relaxation** | <= +5 REU (<= 0 also reported) |

Three deliberate refusals.

**C-beta proximity is not closure.** A Cb-Cb window is satisfied by two ends merely lying near
each other, which is what a peptide's ends do anyway; `cyc_cb_window_success` saturates and
separates nothing. Closure is judged on the real bond atoms *and* on anchor identities that can
host the chemistry -- so a PRO/PRO "isopeptide" fails at 1.33 A. Not hypothetical: anchor identity
is ~50% of measured isopeptide failure. In the first smoke every disulfide and isopeptide request
came back with `bond_dist_A = NaN` because the terminal anchors were VAL/ALA, LYS/LYS and
MET/ASP -- none of which can host the requested bond. NaN there means "not measurable", and it is
reported as such rather than as a large distance.

**Contacts are heavy-atom, not CA.** The 10 A CA-CA proxy used elsewhere in this repo both misses
side-chain-mediated contacts and invents contacts between residues whose side chains point apart.
Crystal inputs pass their real atom-presence mask, because an unresolved atom sits at coordinate
(0,0,0) and would otherwise become a real atom at the frame origin.

**Neither RMSD superposes the peptide.** An independent fit would rotate the binder back onto its
input and hide exactly the movement inside the binding site the edit must be priced for.

### The clash threshold is calibrated against natives

Measured on all 60 LNR natives, counting non-bonded heavy-atom pairs (the declared ring bond and
sequence-adjacent residues excluded):

| threshold | natives passing |
|---|---|
| 1.4 A | 60/60 |
| **2.0 A** | **59/60** |
| 2.2 A | 54/60 |

The median native's closest non-bonded heavy-atom pair sits at **2.64 A**, so 2.0 A is well below
a normal native contact while still catching a steric impossibility. A gate that fails the natives
is measuring the instrument, not the designs.

The one native rejected at 2.0 A is `LNR_1bjr_E_I`, whose closest peptide-receptor pair is
**1.45 A** -- a real crystal close contact. It is `ds[0]`, so a single-example check on `ds[0]`
reports the gate as broken; the test asserts 19/20 and names the exception. Its clashes are
inherited by any edit of it, so a near-identity FlowEdit result on 1bjr still shows 3-5 clashes
that the edit did not cause.

### The autoencoder floor, reported beside the contract

The latent track is an autoencoder code, so decoding an **unedited** latent already differs from
the crystal. If the AE alone mis-decodes two residues, the <= 2 substitution budget is spent
before any editing happens. Every row therefore carries `ae_rt_*` -- the same contract applied to
the unedited round trip -- and `ae_rt_consumes_budget` states it outright. Charging that to the
editing method would be a misattribution.

CA RMSD needs no such care: `bb_ca`'s clean sample **is** the true input CA trace
(`coords_nm[:, :, 1, :]`), so the CA comparison never passes through the AE.

---

## 5. Rosetta, and the isopeptide topology

Both sides go through one function, one scorefunction, one coordinate-constrained `FastRelax`. The
single difference is that the CP declares its closing bond and the LP has none to declare, because
it is linear -- the manipulation, not a confound. **REU are not kcal/mol** and are not converted.

`dG_separated` is not comparable across targets, so only the **paired** delta on one complex is
used. Complexes must be written **during sampling**: the loader hands out a target-centred crop
whose frame is redrawn per process, so a peptide saved without its receptor can never be put back
into it. There is no offline fix; an arm run without `--pdb-dir` has to be re-run.

`FastRelax` is stochastic, so a single score is a draw. Shortlisted candidates and their LP
baselines are each scored `--repeats` times (default 3) and the per-repeat list is kept beside the
median. The LP baseline is cached per input complex -- it does not depend on the candidate, and
redrawing it per candidate would add noise to every delta.

### Isopeptide: resolved, with a narrower limitation

This was recorded as an open gap on the belief that a Lys-NZ <-> Asp/Glu bond needed a custom
residue-connection patch. **It does not.** Rosetta ships one: the `SidechainConjugation` variant,
what ubiquitin/SUMO isopeptide linkages use. Applying `VariantType.SIDECHAIN_CONJUGATION` to both
partners adds a connection at exactly the conjugating atom -- measured on this build, **LYS -> NZ**,
**ASP -> CG**, **GLU -> CD** -- after which the ordinary `declare_chemical_bond` succeeds.

What remains is the residue type set's limitation, not the chemistry's: **ASN and GLN have no
`SidechainConjugation` variant here** (`ResidueTypeSet.cc:980` refuses it), though a Lys-Asn/Gln
isopeptide is the same CG/CD amide. Those rows keep
`binder_rosetta_cyclic_topology_declared=False` and are **excluded** from claimed isopeptide
successes rather than scored as if closed. The subset is live, not theoretical --
`cyclization.allow_asn_gln_isopeptide` is true in training.

### Declared is not closed

Declaring a bond does not make a ring close; relax will not pull ends together across a gap
(verified: 17.5 A stays 17.5 A). What it prevents is the opposite error -- `fa_rep` reading two
bonded atoms at 1.33 A as a clash and prying an already-closed ring apart.

So closure after relaxation is **measured**, not assumed. The first real scoring run made the
case: a candidate returned `cyclic_topology_declared = 1.0` and a respectable
`dG_separated = -18.3 REU` while its NZ-CG distance was **7.47 A** before relax and **9.46 A**
after. Without `ring_bond_distance_{norelax,after_relax}` that row reads as a scored isopeptide
success.

One convention to respect: `_declare_cyclic_bond` maps binder-local index `k` to PDB resSeq
`k+1`, which is `utils.pdb_utils.to_pdb`'s per-chain-restarting numbering. A complex written with
original residue numbering fails endpoint mapping outright (the right failure), so the scoring
stage derives endpoints from the file's own resids rather than assuming `0..L-1`.

---

## 6. Data

The **14 A pocket crop** of the 60-complex LNR test set
(`CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet`), not full-chain LNR.

LNR ships a contiguous receptor where CPSea trained on a ~14-fragment pocket, and that mismatch
alone cost 22-34 points of closure -- it invalidated an earlier SDEdit sweep. The earlier edit work
used the 14 A crop, so this keeps the SDEdit comparison paired. LNR is held out of the LP-mixing
training data by PDB id **and** by receptor sequence identity (>= 40%).

`CPSea_data` is gitignored and so absent from a worktree; `DATA_REPO` points at the main checkout.
The parquet's `path` column is absolute, so the structures resolve from anywhere.

---

## 7. Running it

```bash
# 0. the gating dependency, if it is not already training
bash scripts/submit_lp_mixing.sh --submit flow-mix

# 1. one real end-to-end smoke: both arms, full scoring, small
bash scripts/submit_flowedit.sh --submit smoke \
  --gpu-nodelist <node from a fresh myfree> --mem 32G

# 2. the paired sweep: both arms as siblings, same settings, same output dir
bash scripts/submit_flowedit.sh --submit paired --shards 8 \
  --set 'T_LAT=0.6 0.4 0.2' --set 'SEEDS=0 1 2 3 4 5 6 7' --set NSTEPS=400

# 3. paired dG on the shortlist, then the document (CPU)
bash scripts/submit_flowedit.sh --submit rosetta
bash scripts/submit_flowedit.sh --submit report
```

Every knob in `scripts/_flowedit_preamble.sh` is settable with `--set VAR=value`. The two arms are
**siblings**, not a chain: they are the comparison, so they queue concurrently, and submitting one
without the other produces an uninterpretable result.

Pass `--mem` when the free a6000 sits on a box whose RAM is mostly taken: the sbatch files ask
64 G, which is right for a sweep shard and too much to schedule for a smoke.

### Conventions this chain follows, and why

- Configuration travels as **one env file passed as a positional argument**. Never
  `sbatch --export`: any explicit export list sets `SLURM_GET_USER_ENV=1`, slurmd then fails to
  rebuild the login environment on the compute node, and the job is requeued and **held**,
  stranding every dependent on `Dependency`.
- **Every value in that file is quoted.** The file is sourced, and a bare `SEEDS=0 1` is not a
  two-word assignment -- bash reads it as "run the command `1`", and the job dies with
  `1: command not found` (measured, job 67817). Most of this pipeline's knobs are space-separated
  lists.
- `set -euo pipefail` is in the **job body**, not only the preamble. That is what caught the bug
  above: had it lived only in the preamble, the failed source would have left the job running with
  an empty configuration and still exited 0, releasing dependents behind a stage that did nothing.
- The preamble resolves `REPO` from `$SLURM_SUBMIT_DIR` with an absolute fallback, never
  `BASH_SOURCE` -- Slurm copies the script to `/var/spool/slurm/job<id>/slurm_script`, so a
  self-relative source silently sources nothing. It is self-sufficient under `env -i`.
- `PYTHONPATH` puts the worktree's `src` ahead of the venv's editable install, which points at the
  main checkout; without it `proteinfoundation.editing` is invisible.
- **One output file per shard.** Concurrent `O_APPEND` has NUL-corrupted rows on this filesystem
  (88 of 144 lost), and both loaders **refuse** corrupt lines rather than skipping them, because a
  skipped line is silent data loss and a summary over the survivors looks clean.
- **Only successful candidates count as done.** A failed row is retried on resume: the usual
  reason a resume happens after failures is that the cause was fixed, and treating failures as
  done would skip exactly the candidates the fix was for.
- GPU generation and CPU scoring are separate jobs on separate partitions: asking for a GPU to
  aggregate JSONL just queues the aggregation behind real work.
- The CPU stages **gate on their own input and exit 0** when the GPU stage produced nothing, so
  dependents report that instead of dying in `DependencyNeverSatisfied`.
- The moving `last-EMA.ckpt` is snapshotted per run tag (`FE_PIN_CKPT=1`), so every shard in a
  sweep scores the same model and "which model was this?" stays answerable. Each pin is ~2.9 GB --
  set `FE_PIN_CKPT=0` and point `FLOW_CKPT` at an existing pin for a throwaway re-run, and delete
  pins you no longer need. This filesystem has run out of quota before, mid-checkpoint-write.
- Batch the seeds, but give each candidate its **own** RNG. Drawing a whole batch from one
  generator makes a result reproducible only at the same batch size and position, which silently
  breaks resume -- a resumed shard batches whatever is left.
- Failure rows carry a traceback. A repeated one-line error message says nothing about where, and
  localising one cost a full GPU round trip.

---

## 8. Known limitations

- **ASN/GLN isopeptides cannot be Rosetta-scored with their ring** (section 5). Excluded, not faked.
- **The flow arm warm-starts from base `complexa.ckpt`**, so it is not loss-comparable to the
  v4/bondunroll lineage and has to learn cyclization largely from scratch.
- **Search scale is bounded by the forward cost** (section 3.4). 256 candidates per LP per
  chemistry across 60 LPs and 3 chemistries is ~46k candidates, about 37M forwards; the sweep is
  sharded and its scope is set by explicit runtime limits rather than by the aspiration.
- **CPSea is terminal-only.** Every training ring bonds at `(0, L-1)` and the inference path
  derives endpoints as first/last valid residue, so no interior-bridge edit is reachable. The
  primary comparison holds peptide length and terminal bond placement fixed, which matches that
  constraint rather than working around it.
- **Whole-peptide RMSD is the weaker of the two RMSD gates.** Interface-residue RMSD supersedes it
  on about half this set, so a candidate can pass the 2 A whole-peptide bar while moving the
  residues that actually bind.
- **Anchor identity gates side-chain chemistries.** A disulfide needs CYS at both termini and an
  isopeptide needs LYS plus ASP/GLU; when FlowEdit makes 0 substitutions those requests cannot
  close at all. Terminal anchor grafting (`scripts/anchor_graft.py`) exists for exactly this and is
  the natural next lever.
