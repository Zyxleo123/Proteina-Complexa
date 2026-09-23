# LP → CP pose-decoy training set — Milestone 1: repo inventory and gap list

Read this before writing any of the pose-decoy build. It records what already exists in the
repository for each piece of the design, what is genuinely missing, and five measurements
taken during the inventory that change how the build should be written.

Snapshot: 21 September 2026, branch `cpsea`, working tree at `8d6ac02` plus the uncommitted
Stage-0 files. No build code existed or was written at the time of this survey.

The design being inventoried: synthetic `(source, target)` pairs where the target is a
*decoyed* cyclic pose rather than the pocket's native one, so that two examples sharing a
pocket have different correct answers and the "return this pocket's canonical cyclic
structure" shortcut cannot fit. Every finding below is graded against whether it preserves
that property.

---

## 1. What exists

### 1.1 CPSea complex index / loader

Two sets, both live:

| set | root | train / val / test | clusters (train) |
|---|---|---|---|
| `CPSea_PDB` | `$ZFS/CPSea/CPSea_PDB` | 11,048 / 726 / 645 | 5,311 |
| `CPSea_full` | `$ZFS/CPSea/CPSea_full/CPSea` | 2,444,209 / 134,394 / 135,811 | 555,712 |

Metadata is parquet with an identical schema across both, consumed by
`configs/dataset/unified/cpsea_peptide.yaml`:

```
example_id, path, binder_chain_id, cluster_id, split,
peptide_length, receptor_length, cyclization_type,
source_path, conect_pairs_kept, config_hash
```

**Pocket / peptide separation is by chain**, fixed at preprocess time by
`script_utils/preprocess_cpsea.py`: chain A is the receptor, chain B the binder, heavy atoms
only, no ACE/NME caps, standard residues only. Each file carries the header
`REMARK 999 CHAIN MAP: R->A (target), L->B (binder)`.

**Linkage type is two different things — do not mix them.** The metadata `cyclization_type`
column is a coarse 3-way string `{head_tail, disulfide, other}` (train: 3,514 / 1,162 /
6,372). It cannot distinguish isopeptide from anything else: every LYS:NZ↔ASP:CG bond in
CPSea lands in `other`. The real label comes from CONECT records via
`src/proteinfoundation/cyclization/parse_labels.py:130` `infer_cyclization_label`, which
returns `(i, j, type, has_cyclization)` over `{MAINCHAIN, DISULFIDE, ISOPEPTIDE}` and refuses
to guess when the bond is not observed. That function is what the build should reuse; it also
yields the cut-bond atoms directly.

**Multi-segment keying.** `example_id = {PDB}_{chain}_{start}_{end}_relaxed`, so the domain
key is `{PDB}_{chain}`. On `cpsea_train` (11,048 rows): 4,631 PDBs, 7,067 domains, and 2,395
domains carrying two or more peptide segments.

### 1.2 Frozen-pocket restrained minimization — close to the Deliverable C requirement

`scripts/soft_closure_project.py` (805 lines) already implements most of it:

- `amber14-all.xml` + `implicit/obc2.xml`, `CutoffNonPeriodic`, no constraints.
- Receptor frozen via `system.setParticleMass(idx, 0.0)` (`:361`) — pocket fully fixed while
  still contributing to the energy, which is exactly what the deliverable asks for.
- `ExampleContext` (`:330`) builds force field, frozen receptor, restraints and index vectors
  **once per example**; replicas differ only by `setPositions`. Its docstring records that
  this is worth ~63 s of a ~65 s replica. That is the throughput anchor for the whole build.
- Flat-bottom contact restraints, one representative heavy-atom pair per interface residue,
  chosen on the unperturbed pose so a replica's perturbation is something the restraints pull
  back out of rather than a new reference they lock in.
- `perturb_backbone_torsions` (`:154`) — already one corruption knob.
- `geometry_report` (`:199`) — contact retention, minimum peptide–receptor heavy-atom
  distance, CA-step chain geometry, CA-RMSD and max deviation to input.
- Accept gates on closure / retention / clash / chain, each threshold a CLI argument.

Two further wrappers exist:

- `src/proteinfoundation/utils/pr_alternative_utils.py:541` `openmm_relax` — three-stage
  restraint ramp, LJ-repulsion ramp, MD shakes, backbone-only restraints, accept-to-best
  bookkeeping.
- `scripts/relax_lnr_complexes.py` — AlphaFold's own protocol (`amber99sb`, 10 kcal/mol/Å²
  harmonic restraints on every heavy atom, L-BFGS to 10 kJ/mol/nm), with the OpenMM-8 units
  trap documented in the source.

So the requirement to *randomize force field and step count across the dataset* already has
two force fields and three restraint schemes written and debugged.

### 1.3 Pocket cropping

`scripts/restage_lnr_pocket.py` crops the receptor to a heavy-atom shell at `--radius` while
**preserving original residue numbering**, so deleted residues appear as numbering gaps —
CPSea's own convention. `docs/README_TARGET_DISTRIBUTION_MATCHING.md` carries the calibration:

| set | segments | max_run | seg_med | frag<5 | tgt_len |
|---|---|---|---|---|---|
| **CPSea (reference)** | 13.0 | 34.0 | 2.0 | 59.3% | 106.0 |
| LNR 14 Å crop | 11.0 | 32.0 | 2.0 | 59.4% | 90.5 |

`segments` plateaus at 9–14 for every radius from 6 to 18 Å, which is why an early sweep
wrongly concluded the radius knob was exhausted. `max_run` and `frag<5` are the live
statistics. The auditor is `scripts/audit_target_distribution.py`, which takes any number of
staged parquets side by side.

### 1.4 PepBench and LNR loaders

- **PepBench + ProtFrag**: `script_utils/preprocess_lp.py` →
  `$ZFS/LPData/preprocessed/metadata/lp_train.parquet`, 45,547 rows split 3,424 PepBench /
  42,123 ProtFrag, in the same staged schema as CPSea.
- **LNR**: `scripts/build_lnr_metadata.py` → 60 complexes, staged five ways under
  `CPSea_data/` (`lnr_staged`, `lnr_pocket`, `lnr_pocket14`, `lnr_gapped31`,
  `lnr_relaxed_*`). It already computes `nc_gap_angstrom` (N of residue 0 → C of residue
  L−1) plus best disulfide and isopeptide candidate anchor pairs.

### 1.5 Split / leakage utilities

The LP-mixing holdout is implemented at `script_utils/preprocess_lp.py:398`
`build_lnr_filter` and `:435` `lnr_exclusions`, with exactly the defaults the design calls
for: `--lnr-identity-threshold 0.40`, `--lnr-kmer-size 6`,
`--lnr-containment-prefilter 0.20`. It excludes on PDB-ID match **and** receptor-sequence
identity, using k-mer containment as a prefilter. It is sequence-based only.

### 1.6 Feature / geometry utilities

`mdtraj 1.10.2` in `.venv` covers six of the eight feature-profile entries directly:
`compute_rg`, `shrake_rupley`, `baker_hubbard`, `kabsch_sander`, `compute_phi` / `compute_psi`,
`compute_dssp`.

In-repo and reusable:

| what | where |
|---|---|
| SASA, shape complementarity | `pr_alternative_utils.py:301`, `:222` |
| interface H-bonds | `rewards/tmol_reward.py:708` |
| signed dihedral | `utils/angle_utils.py:30` |
| intra-peptide clash count | `cyclization/scoring.py:77` |
| chirality validity | `cyclization/scoring.py:60` |
| contact set / retention / CA-RMSD | `scripts/sdedit_cyclize.py:145`, `:226`, `:266` |
| ideal side-chain placement | `scripts/rotamer.py` (validated on 1,945 crystal side chains) |
| terminal anchor grafting | `scripts/anchor_graft.py` |

---

## 2. Measurements taken during the inventory

These five change the design and are the reason this document is longer than a file list.

### 2.1 The gap-coverage problem is larger than the brief implies

Sampling 250–300 complexes per set and measuring N(residue 0) → C(residue L−1):

| set | p5 | p25 | **med** | p75 | p95 | max | **% in 20–45 Å** |
|---|---|---|---|---|---|---|---|
| CPSea native, head_tail | 1.3 | 1.3 | **1.4** | 1.4 | 1.4 | 1.4 | **0.0%** |
| CPSea native, disulfide | 4.9 | 6.2 | **7.0** | 7.7 | 8.4 | 9.0 | **0.0%** |
| CPSea native, other (isopeptide) | 5.5 | 7.0 | **8.1** | 9.1 | 10.4 | 11.1 | **0.0%** |
| **PepBench (real linear)** | 10.3 | 17.0 | **21.6** | 24.8 | 34.7 | 45.3 | **56.7%** |
| ProtFrag (real linear) | 9.5 | 14.0 | 18.1 | 22.7 | 32.0 | 45.2 | 38.7% |

The corruption module must move the median by roughly 14–20 Å and place about 57% of the mass
above 20 Å. CPSea's median peptide is 13 residues, whose fully-extended end-to-end length is
about 45 Å, so the top of the target range means a near-fully-extended peptide.

That is reachable in principle — real bound linear binders are extended, with PepBench sitting
at roughly 70% of full extension — but it is a *global* conformational change, not a local
one. A windowed torsion perturbation near the cut can produce it, since a single phi change at
residue 2 swings residues 3–13 rigidly, but that motion sweeps the peptide body out of the
pocket, which is precisely what the contact-retention filter is there to drop.

**The coverage requirement and the contact filter are therefore in direct tension, and the
specified knob list contains no move that extends the peptide *along* the pocket surface.**
The recommendation is to add one contact-aware extension move rather than rely on rejection
sampling to find those conformations by chance. The adversarial-tuning loop is the right place
to settle the parameterisation, but the missing move should be added before the pilot, not
discovered at the tuning milestone.

### 2.2 The excision artifact is real and universal

Checked across eight complexes: in every one, receptor chain A has **zero residues** in the
peptide's residue-number range. The peptide in chain B also keeps its original numbering, so
the hole is named twice — once by geometry and once by `resseq`. Receptors run 98–156 residues
in 11–22 segments, with 9–14 fragments shorter than 5 residues.

The shortcut-removal deliverable is not speculative on this dataset. It is present in every
record and it is a free "where does the peptide go" cue with no counterpart at inference.

### 2.3 CPSea's receptor is already pocket-cropped, and it *is* the 14 Å calibration reference

The 14 Å radius was chosen because it makes *LNR* match *CPSea*. Applying a 14 Å crop back to
CPSea is therefore not a no-op: CPSea's `tgt_len` is 106 while the LNR 14 Å crop lands at 90.5.
Any re-crop of CPSea needs an audit with `audit_target_distribution.py` before adoption, not
an assumption that it changes nothing.

### 2.4 The within-pocket competitor arm is well-populated but needs a spatial check

On `cpsea_train`: 2,395 domains with two or more segments, giving 6,756 within-domain segment
pairs, of which **3,055 (45.2%) are sequence-disjoint across 1,173 domains**; 2,935 pairs are
disjoint *and* at least 10 residues apart.

The other 55% are overlapping windows of the same site — for example `1A0P_A_182_196` against
`1A0P_A_183_196` — and are near-duplicates, not competitors. Sequence disjointness is
necessary but not sufficient: two disjoint segments can still bind the same surface, so a
spatial check on shared receptor contacts is required before the arm is built.

### 2.5 Adversarial validation has two confounds built into the data

PepBench's median peptide length is **9** against CPSea's **13**. PepBench's median receptor is
**223** residues against CPSea's **139**, because PepBench receptors are complete chains rather
than pocket crops.

Four of the profile features — buried SASA fraction, terminal exposure, contact order, and
intra-peptide H-bond count — are sensitive to one or both. Unless PepBench is pocket-cropped to
match and length is controlled, the classifier will reach a high AUC on staging artifacts and
the feature importances will name the wrong knob. That is the exact failure the "high AUC plus
importances names the wrong knob" step is designed to catch, so it must be removed from the
comparison rather than diagnosed by it.

---

## 3. Gap list

### 3.1 Genuinely absent — must be written

1. **Ring opening.** No code in the repository cuts a ring bond. Searching for
   `cut_bond|open_ring|break_bond|linearize|decyclize` returns only test names. Everything
   here runs linear → cyclic; this build needs cyclic → linear.
2. **Rigid-body decoy transform** with the amplitude curriculum and bin labels.
3. **Corruption module.** Only `perturb_backbone_torsions` exists — one knob of seven,
   un-parameterised and not independently loggable.
4. **Ring-closure restraints during minimization.** `soft_closure_project.py` has a *pull*
   restraint that drags termini together down a ladder; the decoy-settling step needs a *hold*
   restraint that keeps an already-closed ring closed. Different force, same machinery.
5. **A no-op / randomised-preprocessing mode**, plus per-record preprocessing provenance.
6. **Filters for backbone geometry** (bond lengths and angles, omega, chirality, Ramachandran
   outliers) and for "relaxed back toward target". Clash and contact-retention filters exist;
   these do not.
7. **Contact order** and the **six-DOF terminal frame** — the only two profile features mdtraj
   does not provide.
8. **The adversarial validation loop** and the **dataset-level identifiability audit**.
   Nothing comparable exists.
9. **A FoldSeek cluster partition extended against LNR.** The existing holdout is
   sequence-only. `cluster_id` arrives pre-supplied in `CPSea_PDB_Cluster.tsv`, and there is
   **no `foldseek` or `mmseqs` binary on PATH** and no cluster-partition runner in the repo.
10. **Sharded columnar writer plus manifest.** Sharding precedent exists at
    `scripts/extract_peptide_surfaces.py:729` (stable stride partition so relaunches do not
    reshuffle ownership), and a manifest convention exists in `mix_manifest.json` (timestamp,
    inputs, seed, counts), but no writer for this schema.
11. **YAML-driven config for a data script.** The requirement is every constant in a YAML; the
    repo's convention for non-Hydra scripts is argparse plus a sourced sbatch preamble plus a
    per-submission env file (`scripts/_lp_mixing_preamble.sh`). Putting the constants in a YAML
    and having the preamble point at it satisfies both.

### 3.2 The environment split, which forces the job layout

|  | `.venv` | `.venv_openmm` |
|---|---|---|
| openmm / pdbfixer | ✗ | ✓ 8.6.0 / 1.12 |
| scikit-learn | ✓ 1.9.0 | ✗ |
| biotite / biopython / mdtraj | ✓ | ✗ |
| pyarrow / pandas | ✓ | ✓ |
| pyrosetta | ✓ | ✗ |

The two are mutually exclusive by design — OpenMM is not installable alongside the training
environment's pins. So minimization jobs run in `.venv_openmm` and write parquet, while
feature-profile, classifier, audit and report jobs run in `.venv` and read it.

`freesasa` is missing from both, so SASA goes through mdtraj's `shrake_rupley` or the
Biopython Shrake-Rupley path in `pr_alternative_utils.py`. Neither `xgboost` nor `lightgbm` is
installed, so the gradient-boosted arm of the adversarial classifier is sklearn's
`HistGradientBoostingClassifier`.

### 3.3 Test fixture

`CPSea_data/CPSea_sample_100/` is an existing 100-complex sample with preprocessed metadata.
The 20-complex fixture should be drawn from `CPSea_PDB` rather than from this sample: the
sample is AFDB-derived (`AF-A0A086B0B8-F1_...`) while the build set is PDB-derived, and the
tests should exercise the same provenance as the build.

---

## 4. Resolved at the Milestone 1 reply

The open question and both default decisions were answered. Recorded here so the
inventory stays readable on its own; the geometry that follows from them is in
`README_M15_CEILING_AUDIT.md`.

### 4.1 Set decision -- the build set is CPSea_full

The plan's "2.71M" is CPSea_full total exactly (2,444,209 + 134,394 + 135,811 =
2,714,414), which pins every count in the plan to the AFDB side. The "~72-75k" is a
filtered subset of it, and the filter was never defined; it binds only at Milestone 5 and
is deferred until then. **Do not scan for it and do not guess it.**

**CPSea_PDB is held out entirely** as the AFDB->PDB intermediate-domain check: no training
on it, no pilot complexes drawn from it, no test fixture from it. This reverses section
3.3 above -- the 20-complex fixture comes from `CPSea_data/CPSea_sample_100/` (staged at
`CPSea_data/preprocessed_sample100/`), whose AFDB provenance is the correct match to the
build rather than the mismatch that section flagged.

Consequence for section 2.4: the 2,395-domain figure is CPSea_PDB and does not transfer.
Deliverable E is recomputed on CPSea_full by
`script_utils/m15_deliverable_e.py index`.

### 4.2 Both default decisions confirmed

PepBench only for the adversarial reference, and the CONECT atom pair recorded verbatim
alongside both linkage taxonomies.

### 4.3 The gap-coverage tension (section 2.1) -- resolved without a new move

The measurement stands; the conclusion drawn from it did not. **Section 5's contact filter
was measuring the wrong quantity.** Global contact retention against the target forces the
source to be a small perturbation of the target, which is what caps the gap. A real
extended linear binder does not retain a compact ring's contacts -- it makes its own, and
shares only an anchor subset with the cyclic analogue.

The gate becomes:

- **anchor retention** -- anchors defined from the target as its top-k interface residues
  by buried surface area; the source must retain *those* contacts.
- **independent plausibility** -- the source must be a well-formed bound pose in its own
  right (no severe clash, sane buried fraction), not a near-copy of the target.

Non-anchor residues are then free to extend. Global retention stays as a **recorded
metric, not a gate**, so the ceiling audit can use it.

The contact-aware extension move recommended above is **not** added yet. It is downstream
of the ceiling audit, which may change the coverage target it exists to serve.

### 4.4 The chain-terminal gap is the wrong discriminator for bridged linkages

For a disulfide or isopeptide macrocycle the bonded atoms are interior side chains, so
N(res 0) -> C(res L-1) measures the tails, not the ring. The section-8 profile gains
**bridge span**: the CB-CB distance (CA for GLY) between the two CONECT-bonded atoms'
residues in the native cyclic state, and for linear peptides the distribution over every
`(i, j)` pair that could host a bridge plus the best pair per peptide. That is the feature
that says whether a given real linear peptide is cyclizable at all, and where.

### 4.5 Adversarial validation -- all three confounds addressed

Pocket-crop PepBench before profiling and audit with `audit_target_distribution.py` on
`max_run` / `frag<5`; fit a **control classifier on staging features alone** (peptide
length, receptor length, segment count, fragment counts) and report both AUCs at every
iteration -- if the control alone reaches high AUC the main number is void until it does
not; and stratify by peptide length rather than pooling.

### 4.6 Deliverable E gains a spatial gate

Sequence disjointness is necessary, not sufficient. The gate is Jaccard over the two
segments' receptor contact sets below a threshold, plus a minimum separation between
interface centroids. If the surviving count on CPSea_full is small the arm is **deferred,
not forced**.

### 4.7 FoldSeek -- what it would take

No `foldseek` and no `mmseqs` on PATH, and no cluster-partition runner in the repo. But
this is a 10-minute unblock, not a project:

- **Static binary, no compile, no root, no conda environment.**
  `https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz` is 12 MB and answers HTTP 200
  from the login node; `conda` also exists at `/zfsauton/scratch/yixiz/miniconda3` and
  bioconda carries it, but the tarball is the smaller commitment.
- The node pool is mixed, so fetch **both** `-avx2` and `-sse41` and pick at runtime from
  `/proc/cpuinfo`. The login node reports avx2 and avx512f; compute nodes advertise no
  Slurm features, so their instruction sets are not knowable from the scheduler.

**What it is actually needed for is narrower than it looked.** Both build sets already
ship a cluster partition:
`CPSea_full/CPSea/CPSea_properties/CPSea_Cluster.tsv` (228 MB, 2.7M rows,
`Cluster_Center` / `Cluster_Member`) and the CPSea_PDB equivalent. So FoldSeek is not
needed to partition CPSea at all -- only to place **LNR and PepBench** into that existing
partition.

**The sequence-only interim is not adequate, and should not be adopted quietly.** The
existing holdout (`preprocess_lp.py:398` `build_lnr_filter`) excludes on PDB-ID match plus
*receptor* sequence identity with k-mer containment. Extending the cluster partition
instead needs the **binder** placed, and CPSea binders are 5-16 residues: at that length
k-mer containment at `k = 6` is close to a coin flip, and two unrelated 8-mers share
6-mers by chance often enough that a threshold tuned to be safe rejects almost everything.
The honest interim is therefore to keep the current receptor-sequence holdout as the
leakage guard -- which is what it already is -- and to state plainly that **no structural
cluster partition against LNR exists yet**, rather than to present a peptide-sequence
k-mer partition as one.

---

## 5. Original open question (superseded by 4.1, kept for the record)

Which complex set is the "~72-75k" full-build target? It did not correspond to anything
locatable: `CPSea_PDB` train is 11,048 (12,419 total); `CPSea_full` train is 2,444,209
with 555,712 clusters; `LPData` is 45,547. A filesystem scan for a parquet in that range
timed out on ZFS before completing, so absence was never proven. Answered above: it is a
filter over CPSea_full, defined at Milestone 5.
