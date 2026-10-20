# Sequence → target → cyclic-peptide outcome dataset (`cpseq`)

First tranche of
`(candidate sequence, target context, cyclization chemistry) → predicted CP–target
complexes → validity / interface / strain / uncertainty labels`.

Started 2026-10-19. **Everything stated here was verified by inspection or execution.**
Where something is unverified it says so. Where an arm is incomplete it says that too,
rather than substituting another model.

---

## 1. Model availability, as measured

| Arm | Source pin | Runtime | Weights | Status |
|---|---|---|---|---|
| **Boltz-2** | `upstream/boltz` `b1ebfc46` (2026-05-29) | own venv, `$ZFS/envs/boltz2` — boltz 2.2.1, torch 2.7.0+cu126, numpy 1.26.4 | `boltz_cache/` 6.2 GB, 45,227 CCD mols extracted | **running, produces data** |
| **AfCycDesign** | `community_models/colabdesign`, Zenodo `10.5281/zenodo.13309081` | complexa venv (colabdesign, jax, haiku all import) | AF2 params 5.3 GB | **running, produces data** |
| **HighFold3** | `upstream/highfold3` `2ff1ea31` (2025-09-24) | `$ZFS/envs/highfold3`, CPython 3.11 + AF3 `a8ecdb2d` | AF3 `af3.bin` 1.1 GB | **capability verified by probe; runtime build attempted — see §7** |

Each arm needed its own environment and none of the three could share one.

### Why Boltz-2 is isolated

It pins `numpy>=1.26,<2.0`, `einx==0.3.0`, `dm-tree==0.1.8` and its own torch, while the
complexa venv runs numpy ≥ 2 and torch 2.7.0+cu126 with openfold/atomworks on top.
Resolving them together downgrades one side, so the adapter shells out to a separate
interpreter. Installed non-editable from the pinned clone, so a later `git checkout` in
`upstream/` cannot silently change what runs.

**`--no_kernels` is required.** Boltz tries the cuequivariance triangular-multiplication
kernels and dies in the pairformer with `ModuleNotFoundError: cuequivariance_torch`.
Installing them is worse than skipping them: `cuequivariance_ops_torch_cu12` 0.13.0
requires `torch>=2.11` and pulls CUDA 13, which would replace this env's pinned torch and
the boltz built against it. The kernels are a speed optimisation, not part of the model.

### AfCycDesign — the encoding was already here

`src/proteinfoundation/evaluation/cyclic_offset.py` already implements the AfCycDesign
construction and is already wired into `rewards/alphafold2_reward.py`. It restricts
itself to `mainchain` and documents why: wrapping the **backbone** offset asserts that
residues 0 and L−1 are backbone neighbours, true only when the closing bond *is* the
peptide bond. The new work is the fixed-sequence complex-prediction path, not the
encoding.

---

## 2. Chemistry: capability table and specification

`src/proteinfoundation/cpseq/chemistry.py` is the single table, and it **computes** the
comparison set rather than asserting it.

| | mainchain (head-to-tail amide) | disulfide (SG–SG) | isopeptide |
|---|---|---|---|
| **Boltz-2** | `cyclic: true` on the peptide chain | explicit `bond` constraint on the two CYS SG atoms — **not** `cyclic` | expressible as a `bond` constraint |
| **AfCycDesign** | cyclic sequence-offset wrap | **linear_only** — backbone-only wrap cannot see a side-chain crosslink | **linear_only** |
| **HighFold3** | `--head_to_tail 1`, peptide first | `--disulfide_chain_res [[1,i,j]]`, peptide first | unsupported |
| **Rosetta (our scorer)** | strip terminus/cap variants → `declare_chemical_bond(N,C)` | `add_disulfide_bond` | **unsupported** — no residue-connection patch |

**The cross-model comparison therefore runs on head-to-tail `mainchain`** — the only
chemistry every predictor *and* the scorer support, and the one the existing CP workflow
uses. `disulfide` is a **separately labeled subset**: supported by Boltz-2, HighFold3 and
Rosetta, but `linear_only` for AfCycDesign. `isopeptide` is excluded because **Rosetta
cannot declare it** — a scorer gap, which is a different reason from a predictor gap, and
recorded as such.

Atom-level bonds, residue requirements and terminal states are in
`manifest/chemistry_spec.json`. Terminal state is part of the molecule: a head-to-tail
ring has **no free termini**, while a side-chain-closed ring keeps both.

Chemistry anchors count toward the mutation budget. A terminal disulfide needs CYS at
both ends; those substitutions are placed and counted *before* any free position is
drawn, and a band whose budget is smaller than the anchor cost says so rather than
overspending.

---

## 3. Candidate construction

`src/proteinfoundation/cpseq/candidates.py`. Four things kept honest:

- **Nominal band is a tag; the row carries the actual integers.** 12.5 % of a 13-mer is
  1.625 residues, so the row says `n_mutations=2`, `mutation_fraction=0.154`.
- **Identical replacements are impossible** — draws come from the alphabet minus the
  residue already present, so a no-op cannot inflate the count.
- **Collisions merge.** Two bands that round to the same sequence on a short peptide
  become one row carrying both band tags, not two rows that look independent.
- **Counts cap at `floor(L/2)`.** Rounding the 50 % band up at odd length gives 7 of 13 =
  46 % identity, outside the task's retention floor. Verified over L=5…16 and both
  chemistries: **minimum identity is exactly 0.500.**

CYS is excluded from random draws so no unrequested crosslink partner appears in a
molecule whose label names one ring.

---

## 4. Target contexts and the split

**The split is created before any variant exists** — `cpseq_build_dataset.py` runs
parse → cluster → split → generate, in that order, because variants first and split
second is how a mutant lands across a boundary from its own reference.

| | |
|---|---|
| LNR entries parsed | 93 |
| parent contexts in the 5–16mer range | **63** (30 rejected: 3–4mers and 17–27mers, each with its measured length as the reason) |
| receptor families | **54**, 5 multi-parent |
| parents selected | **12**, one per family, lengths 5,5,6,6,7,8,9,11,11,12,14,16 |
| mainchain candidates | **107** |
| disulfide candidates | **58** |
| `config_hash` | `4207a3085a004bba` |

Mainchain is 107 rather than 12 × 9 because one parent's bands coincide onto an identical
sequence. Disulfide is smaller by construction: two CYS anchors are charged first, so low
bands collapse into the reference arm.

**Receptor chain sets are computed from contact with the peptide, not read from
`test.txt`.** `README_TARGET_DISTRIBUTION_MATCHING.md` records LNR cases where a
contacting chain was discarded, and dropping a chain that touches the peptide changes the
pocket. One selected-pool parent has a second contacting chain `test.txt` omits; it is
kept and noted.

### Two pockets, deliberately

Boltz applies `max_distance` between the binder and **each** listed contact. Handing it
the descriptive 8 Å shell — 27 to 48 residues on these parents — would demand that all of
them sit within 6 Å of a 5–16mer at once. Nothing satisfies that, and an unsatisfiable
constraint does not localise a prediction, it distorts it.

- **descriptive pocket**: full 8 Å shell, 27–48 residues, used to characterise the site.
- **conditioning pocket**: the **6** nearest genuine contacts (≤ 5 Å to the parent's bound
  peptide), jointly satisfiable by construction, `force: false`, defined **once per
  parent** and reused unchanged by every mutant of it.

Both counts are in `manifest/targets.csv`. The pocket localises; it restrains no
individual peptide residue. Contact retention is not an objective here.

### Shared target conditioning, and one unavoidable difference

Per parent, cached: a **target-only template** (receptor chains only, renumbered 1..N,
coordinates copied verbatim, peptide chain excluded and the file read back to prove it)
and **one receptor MSA per unique receptor sequence** — keyed by sequence hash, not parent
id. **12 searches serve 165 candidates.** Candidate peptides get `msa: empty`: they are
synthetic sequences with no homologs, so a search each would buy nothing.

> **Conditioning difference, recorded not hidden:** AfCycDesign gets target chains and the
> target-only template but **no pocket**. ColabDesign's `hotspot` enters the *design
> loss*, which a fixed-sequence prediction never evaluates, so AF2 has no forward-pass
> pocket mechanism. Its rows carry `pocket_conditioning: "none"`.

### Generalization coverage — stated plainly

LNR goes to `dev` **whole**. It has already shaped this repository's preprocessing and
sampling decisions, so a held-out slice of it would not be independent. PepBench's
`train.cluster` / `valid.cluster` (4,157 / 114) are staged and are a pre-existing
clustering that could supply disjoint target families later. **Until those are built into
contexts, this tranche supports no cross-family generalization claim.**

---

## 5. Experimental controls

**Nothing staged in this project is an experimental CP–target complex.** CPSea_PDB's
entries are `<PDB>_<chain>_<start>_<end>_relaxed` — fragments cut from PDB proteins and
computationally cyclized — and CPBind's are AFDB accessions given the same treatment.
Useful data, not ground truth for a macrocycle.

So controls come from an RCSB structural query and each is **verified by measurement**
with our own evaluator before acceptance: the ring must close inside this repo's own
window or it is rejected with its measured distance as the reason. A literature claim of
cyclicity does not get a structure in.

The first pass returned 1SLD, 1VWB, 1VWN and 1VWC — four depositions of the *same* CHPQFC
peptide on the same receptor, which would have made six controls look far more diverse
than they are. Deduplicated by (peptide, receptor):

| PDB | chain | chemistry | L | sequence | closure bond |
|---|---|---|---|---|---|
| 5TU6 | B | mainchain | 7 | INPYLYP | 1.339 Å |
| 7K2M | P | mainchain | 7 | GEPETGE | 1.311 Å |
| 7K2F | C | mainchain | 7 | GAEETGE | 1.554 Å |
| 1SFI | I | mainchain | 14 | GRCTKSIPPICFPD | 1.334 Å |
| 1SLD | P | disulfide | 6 | CHPQFC | 2.048 Å |
| 1SLE | M | disulfide | 8 | CHPQGPPC | 2.026 Å |

These check **implementation and capability**, not prospective accuracy. All predate the
predictors' training data and may sit inside it.

---

## 6. The shared evaluator

One evaluator, applied identically to every arm. It does not know which predictor made
the structure it is given: same scorefunction (`ref2015`), same relax budget
(`FastRelax`, 1 repeat, restrained to input coordinates), same peptide topology, same
protonation policy (the shared PyRosetta init flags, recorded per row), same target
flexibility. Predictor-specific steering lives in the *prediction* record, in a different
file, so the two cannot be confused.

**Raw and refined are both kept**, with how far refinement moved the structure
(`binder_ca_rmsd_A`, `receptor_ca_rmsd_A`, `total_score_delta`), so a pose rescued by
relax cannot pass as a good prediction.

- **Validity/closure** — exact sequence, required atoms, closure bond **length and both
  angles and the torsion**, stereochemistry, backbone geometry, severe intra/inter
  clashes. Distance alone is not closure (§8 shows exactly why).
- **Interface energetics** — `dG_separated`, `dSASA_int`, and `dG_separated/dSASA_int ×
  100`, the last **null** (never 0 or ∞) when `dSASA_int ≤ 0`. Raw energy and area both
  kept. These are **Rosetta proxies, not measured binding free energies.**
- **Interface quality** — `delta_unsatHbonds`, `hbonds_int`, `sc_value`, `packstat`, the
  exact InterfaceAnalyzer definitions.
- **Peptide deformation** — the bound peptide is extracted **with its ring re-declared**
  (a split chain does not carry a declared bond, and without this every macrocycle would
  score strain-free), then relaxed without the receptor under the same budget. Labelled
  `local_deformation_energy_proxy`, `is_thermodynamic_free_energy: false`: the reference
  is the nearest free minimum to the bound pose, not a global unbound state.
- **Reliability** — per-predictor confidences in their own columns, sample spread,
  cross-predictor agreement with the **receptor** aligned.

Restraints are recorded, and reported energies are read with a scorefunction carrying
zero constraint weight, so no restraint term is ever counted as physical energy.

### Frozen tolerances

`manifest/geometry_tolerances.json`, written once then read-only. Bond windows are the
project standard; angle and torsion tolerances sit at the **worst experimental control
plus 5 %**, the way `derive_cyclic_thresholds.py` does it.

| chemistry | bond window | angle tol | torsion tol | n_controls |
|---|---|---|---|---|
| mainchain | 1.1–1.6 Å | ±7.54° | ±8.44° | 4 |
| disulfide | 1.8–2.3 Å | ±8.98° | ±55.46° | 2 |
| isopeptide | 1.1–1.6 Å | ±25° (fallback) | ±60° (fallback) | 0 |

The wide disulfide torsion is real — CB-SG-SG-CB genuinely occurs at both ±90°, and the
check folds the two signs together. `n_controls` travels with the numbers so a bar set
from two structures is visible as such.

A missing atom, a failed refinement, an undeclarable ring and an absent score are **not**
successful closure; each is recorded with its reason. A prediction failure is **not** an
experimental nonbinder label.

---

## 7. Verified findings

### Boltz-2: its two cyclic controls are not equally reliable

First real comparison, `lnr_1elr`, 2 samples each:

| chemistry | closure bond | reference | angles | torsion | clashes |
|---|---|---|---|---|---|
| mainchain | 1.337, 1.342 Å | 1.33 | 122/118 vs 121.7/116.2 | near trans | 0 intra |
| disulfide | 1.454, 3.584 Å | 2.05 | 85–95 vs 103.8 | — | 1, 3 intra |

The native `cyclic` flag closes the amide essentially perfectly. The explicit SG–SG
`bond` constraint does not — one sample too short to be a bond, one too long to be one.
**This is the concrete reason the two chemistries had to stay separate arms** rather than
one "cyclic" bucket.

### AfCycDesign: a ring at the right distance pointing the wrong way

`lnr_1elr`, mainchain: closure bond **1.251 Å — inside the 1.1–1.6 Å window** — while the
closure angles are off by **46.65°** and **28.66°** and the torsion by **135.42°**. A
distance-only check would have called that closed. It is two atoms near the right
distance in a chemically impossible arrangement, and it is why closure is measured as
length *and* angles *and* torsion.

The same samples show `dSASA_int = 0.0`, `nres_int = 0.0`, `i_ptm = 0.10` with
`pLDDT = 0.95`: the peptide is confidently **folded** but not **docked**. Being quantified
across the whole arm before it enters the results as a property rather than an anecdote.

### AfCycDesign yields 2 samples, not 4 — measured both ways

ColabDesign loads `model_{1,2}_ptm` when `use_templates=True` and `[1,2,3,4,5]` only when
it is False (`af/model.py:110-117`) — those are the monomer-ptm sets carrying template
weights. The binder protocol here *is* templated, so exactly 2 parameter sets exist;
asking for models 2+ raised `IndexError`, and no `num_models` argument changes it.

Re-seeding is not a substitute: the same parameter set at seed 42 and seed 777 gave
peptide CA coordinates identical to **0.000000 Å**. So `SAMPLE_CEILING = 2`, and
**best-of-2 is kept distinct from best-of-4** throughout the summaries.

### HighFold3: the ring lands on the FIRST chain, and the disulfide chain field is ignored

HighFold3's entire cyclic behaviour lives in `model/cyclic_peptide_matrix`, which ships
**only** as `model/__pycache__/cyclic_peptide_matrix.cpython-311.pyc` — no `.py` is
tracked. So the flags cannot be read, only exercised. The bytecode loads under CPython
3.11, and its own traceback reveals where it was built:
`/home/admin/.conda/envs/af3-cs/lib/python3.11/site-packages/alphafold3/model/cyclic_peptide_matrix.py`.

Probing `calc_cyclic_matrix_af3` on both the residue-wise and token-wise paths
`features.py:975,982` calls:

1. **The matrix is sized from where `residue_index` returns to 1, so the ring is applied
   to the first chain.** Peptide first + receptor second → 5×5, a correct wrapped ring on
   the 5-mer. Receptor first → 8×8, and the ring lands on the **receptor** while the
   peptide stays linear.

   **This is the trap, and it is silent.** `--head_to_tail` takes no chain argument, AF3's
   own examples list the protein first, and the wrong ordering still produces confident
   output — for a cyclized target protein. The adapter writes the peptide first, and that
   is not a style choice.

2. **`--disulfide_chain_res`'s documented chain field is ignored.** `[[1,1,6]]` and
   `[[2,1,6]]` return bit-identical matrices on both paths. The crosslink always goes to
   the first chain's block, so correctness again rests on ordering.

3. Residue indices are **1-based** (`[[1,2,5]]` sets the ±1 offset between 0-based rows 1
   and 4).

4. With no topology requested the function returns `[[0]]` — a 1×1 placeholder, so a naive
   elementwise diff against the "linear" case broadcasts instead of comparing.

Where it is applied the encoding is correct, and `list_cid_ss` encodes a **side-chain**
crosslink as a graph shortest path — exactly what the AF2 offset wrap cannot express, and
why HighFold3 can represent a disulfide ring that AfCycDesign genuinely cannot.

`adapters/highfold3.probe_cyclic_matrix()` re-runs all of this as a check, so a different
HighFold3 revision can be re-established rather than trusted from a docstring.

**Runtime version constraint, pinned from real history.** The bytecode needs CPython
3.11; current AF3 (`131ab9e7`) sets `requires-python = ">=3.12"`; AF3 `a8ecdb2d`
(2025-09-08, the last revision before HighFold3's snapshot) allows `>=3.11` with
`jax==0.4.34` and `dm-haiku==0.0.13`. The cluster has only 3.12, so the interpreter comes
from python-build-standalone. AF3's genetic databases are **not** fetched —
`installation.md` wants up to 1 TB against a 500 GB quota with ~225 GB free — and
alignments are supplied from the same per-target cache the other arms use.

### Environment traps worth recording

- **`MMSEQS_EXEC` and `FOLDSEEK_EXEC` point at binaries that do not exist.** `env.sh`
  exports both regardless. The family split must exist before variants, so
  `contexts.assign_families` falls back to a built-in pairwise clustering and records
  which path ran in `clustering_method.json`. The first fallback put all 63 receptors in
  **one** family — global alignment scatters chance matches across a long receptor,
  normalising by the shorter sequence clears 0.3 for unrelated pairs, and single linkage
  chains every family together from the first false edge. Local alignment with identity
  over aligned columns (what mmseqs reports) gives the 54 families above.
- **gemmi 0.6.5 misreads a spec-correct SEQRES block.** A 128-token SEQRES came back as
  129 residues (one extra LYS) while the deposited 1elr.pdb read exactly right (131 =
  131); byte-diffing showed identical layout. Boltz reads a PDB template by converting it
  with gemmi and aligning the entity's `full_sequence`, so this matters: a template
  sequence *longer* than its coordinates does not crash, it silently inserts unresolved
  residues. Templates are therefore handed to Boltz as **CIF** with the entity sequence
  derived from the model and verified equal to the polymer length on read-back.
- **PyRosetta 2026.39 dropped `InterfaceAnalyzerMover`'s string constructor** for a
  `DockingPartners` object. `evaluation/rosetta_energy.py` still passes a string, so the
  repo's existing Rosetta path does not run on the installed build either. Handled with
  the typed factory plus a fallback.
- **`packstat` returns 0.0 unless requested.** An uncomputed default, not a value, which
  would have read dataset-wide as a perfectly unpacked interface.
  `set_compute_packstat(True)` gives 0.76 on a crystal interface.
- **Rosetta types LNR 1elr's peptide as `MET:AcetylatedNtermProteinFull`** and declaring a
  head-to-tail bond exits the process with "doesnt have connection at N" and no
  traceback. Chemical caps are now stripped alongside plain terminus variants — and their
  removal is **reported**, because stripping an acetyl changes which molecule was scored.
- **The interactive node-hold job has 8 GiB, not the 32 GB squeue suggests.** Boltz-2 was
  OOM-killed there at exactly that ceiling (`oom_kill 2`, `memory.peak 8589303808`) after
  processing its inputs fine. `sacct` then put the successful step at **11.0 GiB**
  MaxRSS, which is how the sbatch request was sized (32 G, down from a guessed 48 G).

---

## 8. Architecture

```
src/proteinfoundation/cpseq/
  chemistry.py      capability table + atom-level chemistry spec; computes the
                    cross-model chemistry set rather than asserting it
  candidates.py     mutation bands, seeds, anchor accounting, dedup, identity floor
  contexts.py       parent contexts, receptor families, split (before variants)
  targets.py        target-only templates, CIF companion, cached per-target MSAs,
                    conditioning pocket
  geometry.py       measures validity/closure; decides nothing
  tolerances.py     frozen thresholds + the verdict that applies them
  scorer.py         the one shared chemistry-aware evaluator
  adapters/
    boltz2.py       cyclic flag for the amide, atom-pair bond for the disulfide
    afcycdesign.py  fixed-sequence complex prediction + the existing cyclic offset
    highfold3.py    peptide-first input, verified topology flags, probe

script_utils/
  cpseq_build_dataset.py      parents → families → split → candidates
  cpseq_prepare_targets.py    templates + MSA cache
  cpseq_fetch_controls.py     RCSB search → verify by measurement
  cpseq_freeze_tolerances.py  derive from controls, freeze
  cpseq_predict.py            stage 1, resumable, sharded, capability-gated
  cpseq_score.py              stage 2, resumable per sample
  cpseq_summarize.py          stage 3, joined tables

scripts/
  cpseq_env_boltz2.sbatch     isolated Boltz-2 env
  cpseq_env_highfold3.sbatch  CPython 3.11 + AF3 a8ecdb2d + HighFold3 overlay
  cpseq_predict.sbatch        GPU prediction, array-sharded
```

Prediction and scoring are **separate stages**, so rescoring never re-runs a predictor.
Task ids are `sha256(candidate, predictor, chemistry, revision, config)`, and the config
hash is part of the **output path** too — without that, a re-run at a different sample
count finds the old record and skips, leaving the tranche silently mixing settings.

---

## 9. Commands

```bash
cd /home/yixiz/Proteina-Complexa && source env.sh

# one-time environments
sbatch scripts/cpseq_env_boltz2.sbatch
sbatch scripts/cpseq_env_highfold3.sbatch

# dataset skeleton: parents -> families -> split -> candidates
.venv/bin/python script_utils/cpseq_build_dataset.py \
    --out-dir $ZFS/cpseq/tranche1 --n-parents 12

# shared target conditioning (cached; re-running is nearly free)
.venv/bin/python script_utils/cpseq_prepare_targets.py \
    --out-dir $ZFS/cpseq/tranche1 \
    --boltz-python $ZFS/envs/boltz2/.venv/bin/python

# experimental controls, verified by measurement
.venv/bin/python script_utils/cpseq_fetch_controls.py \
    --out-dir $ZFS/cpseq/controls --n 6

# freeze tolerances (once per tranche)
.venv/bin/python script_utils/cpseq_freeze_tolerances.py \
    --dataset-dir $ZFS/cpseq/tranche1 --controls $ZFS/cpseq/controls/controls.json

# stage 1 -- prediction
sbatch --array=0-3 scripts/cpseq_predict.sbatch PREDICTOR=boltz2 SHARD_COUNT=4 SAMPLES=4
sbatch --array=0-3 scripts/cpseq_predict.sbatch PREDICTOR=afcycdesign SHARD_COUNT=4 SAMPLES=4
#   LIMIT=2 for a preflight; FORCE=1 to re-run after an adapter fix

# stage 2 -- scoring (CPU; rescoring never re-predicts)
.venv/bin/python script_utils/cpseq_score.py \
    --dataset-dir $ZFS/cpseq/tranche1 --predictor boltz2 --shard-index 0 --shard-count 1

# stage 3 -- tables
.venv/bin/python script_utils/cpseq_summarize.py --dataset-dir $ZFS/cpseq/tranche1
```

### To expand the tranche

Raise `--n-parents` (63 LNR parents qualify, 54 families) and re-run
`cpseq_build_dataset.py` → `cpseq_prepare_targets.py` → the prediction arrays. Existing
work is skipped by record, so expansion is incremental. For a genuine held-out split,
build contexts from PepBench's `train.cluster`/`valid.cluster` and pass
`dev_fraction < 1.0` to `contexts.split_by_family`.

---

## 10. Status

- [x] Predictor availability measured; all three arms' chemistry capability verified
- [x] Chemistry capability table + spec, checked against `cyclization.constants`
- [x] Candidate builder (bands, seeds, anchors, dedup, ≥50 % identity floor)
- [x] Parent contexts, 54 receptor families, split built **before** variants
- [x] Shared target conditioning: target-only templates + 12 cached MSAs for 165 candidates
- [x] Six experimental controls, each verified by measurement, deduplicated by system
- [x] Frozen chemistry-specific tolerances with provenance
- [x] Shared chemistry-aware evaluator (validity, interface, quality, deformation)
- [x] Boltz-2 adapter — running, both chemistries
- [x] AfCycDesign adapter — running, mainchain (2-sample ceiling recorded)
- [x] HighFold3 adapter + topology verification; runtime build attempted
- [ ] Full first-tranche collection and scoring (in progress — §11)
- [ ] Cross-predictor pose agreement on the full set
- [ ] PepBench-derived held-out target families

---

## 11. Results

### Prediction: complete for both working arms

| arm | candidates | samples | completion | exact sequence | measured compute |
|---|---|---|---|---|---|
| Boltz-2 mainchain + disulfide | 165 | **660** (4/candidate) | **165/165, zero failures** | 100 % | **2.76 GPU-h** (4 shards, 13.5 GB MaxRSS) |
| AfCycDesign mainchain | 107 | **214** (2/candidate) | **107/107** | 100 % | **~0.8 GPU-h** (5.4 GB MaxRSS) |
| AfCycDesign disulfide | 58 | 0 | n/a — `chemistry_unsupported` | n/a | none |
| HighFold3 | — | 0 | runtime build incomplete (§7) | — | — |

**874 predicted complexes**, ~3.6 GPU-hours against the 24 GPU-hour cap. Every arm kept
the supplied sequence exactly; sequence compliance was verified from the output
structures, not assumed from the input.

The 58 AfCycDesign disulfide candidates are **recorded, not silently absent**: each has a
row with `status: chemistry_unsupported` and the capability table's reason. A backbone
offset wrap cannot express a side-chain crosslink, and the arm was not run with the
nearest available switch and relabelled.

### Scoring: partial, and the shape is already clear

Scoring is the bottleneck, at ~20 s/sample (`pose_from_pdb` 23 s cold, FastRelax 25 s,
InterfaceAnalyzer 2 s). The user's 13 other queued jobs leave no room under
`QOSMaxSubmitJobPerUser`, so most scoring ran on this allocation's 2 cores. Coverage at
time of writing is a deterministic stride slice of shards, i.e. an unbiased subset, and
the remaining shards resume with one command (§9).

| arm × chemistry | samples scored | completion | **validity of completed** | median best refined `dG_separated` |
|---|---|---|---|---|
| Boltz-2 mainchain | 47 | 100 % | **31.9 %** | **−32.26** |
| Boltz-2 disulfide | 24 | 100 % | **0 %** | null (no valid sample) |

### Why the two Boltz chemistries diverge so sharply

This is the tranche's clearest finding, and it is a chemistry result rather than a
scoring artefact.

| | closing bond dev | angle_i dev | angle_j dev | torsion dev |
|---|---|---|---|---|
| **mainchain** (`cyclic: true`) | median **0.01 Å**, p90 0.04 | median 1.40°, p90 6.09° | median 1.27°, p90 4.11° | median 3.92°, p90 12.53° |
| **disulfide** (explicit `bond`) | median **0.52 Å**, p90 1.30 | median 17.52°, p90 46.28° | median 15.18°, p90 31.89° | median **54.45°**, p90 78.73° |

Boltz's native head-to-tail flag closes the amide to within a hundredth of an Ångström of
the reference. Its explicit SG–SG bond constraint does not produce a bonded disulfide at
all: 20 of 24 samples fail the 1.8–2.3 Å window outright, with angles ~15–18° off and
torsions ~54° off.

**So the two controls are not interchangeable, and the capability table's insistence on
separating them was load-bearing.** Had the dataset pooled them under one "cyclic" label,
a 0 %-valid chemistry would have been averaged into a 32 %-valid one.

### What limits mainchain validity — reported, not hidden

The mainchain ring geometry is mostly excellent, so the 31.9 % figure is set by the
strictness of the frozen bar and by clashes, not by bad rings:

- **12 samples** fail only the closure torsion, against a ±8.44° tolerance derived from
  **4** experimental controls. Median torsion deviation is 3.92°, p90 is 12.53° — so the
  bar sits inside the predicted distribution's upper tail. A bar set from four crystals
  is a weak bar, which is why `n_controls` travels with it and the full deviation
  distribution is in `tables/samples.csv`. Anyone preferring a looser criterion can
  recompute validity from the stored geometry **without rescoring**.
- **17 samples** carry a peptide–receptor clash and **14** an intra-peptide clash. These
  are not marginal contacts: minimum inter-chain distances run 1.23–1.80 Å and the worst
  intra-peptide pair is **0.295 Å**, i.e. two heavy atoms essentially superimposed.
  Against an N···O hydrogen bond at 2.8–3.2 Å, a 2.2 Å cutoff with zero tolerance is the
  right call, and these samples are genuinely broken structures.

### AfCycDesign: folded but not docked

Across its scored samples the arm shows `dSASA_int = 0.0`, `nres_int = 0.0` and
`i_ptm ≈ 0.10` against `pLDDT ≈ 0.95`. The peptide is confidently folded and not placed
on the receptor, so its area-normalised interface score is **null** by construction (not
0, not ∞) and its interface terms are largely undefined.

Two causes, both recorded rather than worked around:

1. **No pocket conditioning is available.** ColabDesign's `hotspot` enters the design
   loss, which a fixed-sequence prediction never evaluates, so unlike Boltz and HighFold3
   this arm cannot be told where on the receptor to look.
2. **Its ring geometry is poor even when the distance looks right** — the preflight case
   closed at 1.251 Å, inside the window, with angles off by 46.7° and 28.7° and the
   torsion off by 135°.

There is a real trade-off behind (1): AF-Multimer would dock a peptide far better, but
the cyclic offset patch operates on the monomer relative-encoding path, which is what
AfCycDesign itself uses. Choosing multimer would buy docking at the cost of the ring
encoding — i.e. a different arm, not a better configuration of this one.

### Not yet measured

- **Cross-predictor pose agreement** needs both arms scored on shared candidates;
  `cpseq_summarize.py` computes it (receptor-aligned) once scoring covers overlapping
  samples.
- **Control pose recovery.** The six controls are staged through the identical pipeline
  (`cpseq_build_control_contexts.py`, 6 parents, 6 native-sequence candidates, templates
  and MSAs cached) but not yet predicted. The command is in §9; nothing about it is
  blocked.
- **Deformation proxy** is computed per scored sample and present in the rows; its
  cross-arm comparison waits on the same coverage.

### Honest scope statement

This is a complete **prediction** collection (874 complexes, two arms, both chemistries
where supported) with a **partial but unbiased scoring** pass, built on 12 target
families from development data. It is not enough to establish scorer generalization, and
LNR going to `dev` whole means it supports no cross-family claim at all. The Rosetta
numbers are proxies, not measured binding free energies, and the controls check
implementation rather than prospective accuracy.
