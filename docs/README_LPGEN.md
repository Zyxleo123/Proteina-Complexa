# LP generator milestone — common manifest

Status as of 2026-10-12. **Step 1 of the LP generator plan is built, run and verified, and
the step-3 POTFlow repairs are confirmed against the source. None of the four training arms
has been run.**

The milestone trains four LP generators (PPFlow, POTFlow, PepGLAD fixed-sequence,
RFdiffusion3) on linear peptide–target complexes, each with its native representation and
objective. This document covers the shared data layer they all read, and records what
upstream work each arm still needs.

## The manifest

```
/zfsauton/scratch/yixiz/LPData/preprocessed/metadata_lpgen/
  lpgen_manifest.parquet        43,734 rows, 19 columns
  lpgen_val_panel.parquet       32 complexes, 32 families, lengths 5–16
  lpgen_manifest.summary.json   counts, weights, and every gate's result
  lpgen_excluded.csv            1,893 dropped rows, each with its reason
```

Build it with:

```bash
./scripts/submit_lpgen.sh --submit manifest          # ~30 s on the cpu partition
./scripts/submit_lpgen.sh --submit manifest --smoke  # 2,000 rows, separate output dir
```

All four arms read this one file. The alternative — letting each upstream repo use its own
index, cache and split files — is how three arms end up trained on three different subsets
and the comparison stops meaning anything. Each arm's adapter must report its accepted ids
against this manifest.

| split | rows | families | sources |
|---|---|---|---|
| `train` | 33,089 | 7,186 | protfrag 30,507 / pepbench 2,582 |
| `val` | 4,028 | 833 | protfrag 3,725 / pepbench 303 |
| `heldout` | 6,617 | 1,482 | protfrag 6,084 / pepbench 533 |

`heldout` is reserved for the later CP-sequence comparison and is neither trained nor
validated on.

## What was already done, and is not redone

`preprocess_lp.py` had already converted PepBench and ProtFrag into one on-disk layout:
chain `A` receptor, chain `B` peptide, no hydrogens, no non-standard residues, no HETATM,
and the LNR benchmark filtered out (161 entries, by PDB id and by receptor-sequence
identity). This step does not touch structures. **Every one of the 45,627 input rows read
cleanly** — zero unreadable files, zero non-canonical residues, zero missing backbone
atoms, zero metadata/structure length disagreements. The preprocessing is sound.

The peptide length window is 5–16. That is not an arbitrary inherited cap: CPSea peptides,
which are the source of the CP sequences this milestone eventually serves, are themselves
capped at 16 (p99 = 16, 100% ≤ 16). The two windows match by construction. If the CP side
is ever extended, the LP preprocessing must be re-run — 23,000+ PepBench/ProtFrag rows
were rejected for length and would come back into scope.

## Five things the manifest adds

**1. Peptide sequence.** Preprocessing recorded `receptor_seq` (it needed one for the LNR
identity filter) but never the peptide sequence, and the metadata carried only
`peptide_length`. PepGLAD's fixed-sequence mode conditions on the sequence and RFD3 reveals
a fraction of it, so both arms were unimplementable without this column. It is read back
out of chain `B` of the written PDB — the structure the arms actually train on — so a
sequence disagreeing with the coordinates is a detected error, not a silent one.
Cross-checked against biotite on 40 source-stratified structures: **0 mismatches**.

**2. A leak-free family-level split.** The inherited split came from PepBench's own
`train.txt`/`valid.txt`, and it leaked: ProtFrag fragments had been cut from the same PDB
entries as 4 of the 80 PepBench validation complexes (families `2m14`, `2yvc`, `7xbd`), so
those receptors appeared in training as fragments. Assignment here is a pure function of
the family key, which makes that class of leak unrepresentable — a parent and all of its
fragment derivatives share a `cluster_id` and therefore always land on the same side.
Verified: **0 cross-split family overlap**.

**3. Reserved families.** Families that `cp2lp_build_splits.py` holds out for CP→LP
evaluation are marked `heldout`, reusing `family_of` and `in_eval_split` at the same
`(eval_frac=0.15, seed=0)`. Verified: **0 train/val families are CP→LP eval families**.
Note the relation is containment, not equality — `heldout` reserves 1,076 families beyond
those appearing in the CP→LP eval files, because that eval's real-LP side is PepBench-only.
Reserving more is safe; reserving less would contaminate the later comparison.

**4. Measured topology — this one changed the data.** `preprocess_lp.py` wrote every LP row
`cyclization_type="linear"` with `conect_pairs_kept=0`, because PepBench and ProtFrag ship
no CONECT records at all. That label is therefore an assumption inherited from the source,
not a measurement, and a cyclic peptide passes it unchallenged. Measured from coordinates:

- **76 peptides carry an intra-peptide disulfide** at ~2.03 Å, several between the first
  and last residue — CP topology wearing an LP label.
- **10 peptides are closed head-to-tail**, N(first)–C(last) at 1.305–1.337 Å for eight of
  them, which is the peptide-bond length. A tight cluster at exactly that distance is a
  real bond, not accidental proximity.
- **1 peptide** has an SG–SG contact at 1.572 Å, shorter than any real S–S; recorded under
  its own reason rather than counted as a disulfide.

All 87 are excluded by default (`--keep-crosslinked` readmits the 86 cyclic ones) and
recorded in `lpgen_excluded.csv`. This matters more than 0.19% of rows suggests: all but
one are PepBench, which carries 24% of each batch off 7.5% of the rows, so they would have
been drawn at roughly 3× their row share while training an *LP* generator.

**5. Residue-key collision gate — this one cost 1,806 rows.** Two genuinely different
residues can share one `(chain, resseq, icode)` key in the preprocessed PDBs. Measured over
the whole pool: **1,806 of 45,540 structures (3.97%), every one of them ProtFrag, none
PepBench** — consistent with `preprocess_lp.py` merging a multi-chain receptor into a single
chain `A` without renumbering, so two source chains both carrying residue 82 land on one
key. One file had `res 52` holding a TYR *followed by* a PRO, and `res 82` holding four
residues' worth of atoms.

Two separate harms, which is why it is a hard exclusion:

* atomworks refuses such a file outright (`Duplicate atom names detected in the same residue
  -- cannot infer struct_conn`), so RFD3 would have silently dropped ~4% of the training set
  through the dataloader's fallback retries;
* **359 structures (0.79%) collide inside the peptide chain**, and because this builder and
  `preprocess_lp.py` key on the same tuple, the collided group counts as *one* residue — so
  `peptide_length` and `peptide_seq` are understated and **agree with each other**, which is
  exactly why the length check could not see it. Those 359 rows had a wrong peptide sequence.

The reader also had to change to detect it: atom names were stored in a `set`, which
collapses duplicates before anything can notice, so it now counts atoms per residue too.
Recovering the 4% would mean renumbering residues on chain merge in `preprocess_lp.py`.

## Source sampling weights

Weights were **not** invented here. The project already configures them in
`training_cpsea_lpmix_from_v4cfg.yaml`: `pepbench: 0.06`, `protfrag: 0.19` inside a 25%
linear budget, which renormalised over the LP-only pool is **24% / 76%**. That ratio was
chosen ~sqrt-proportionally to pool size, between the natural ratio (which buries PepBench
at 7.5% of rows) and an even one (which would redraw its rows ~91× an epoch).

The plan's fallback of 80% PepBench / 20% ProtFrag is the *inverse* of this, and would make
PepBench repeat ~150× an epoch against ProtFrag's ~2×. Since weights are explicitly
configured, the plan's own rule says reuse them — so 24/76 is the default. `--set
PEPBENCH_FRAC=0.8` selects the binder-focused ratio, and the summary records which was
used, so an arm trained either way is identifiable after the fact.

Weights are emitted as a per-row `sample_weight` over train rows, normalised to sum to 1,
so every arm's sampler draws the same mixture. The realised mixture is integrated from the
weights and reported — it is not the row ratio, and reporting row counts instead would be
off by an order of magnitude.

## Validation panel

`lpgen_val_panel.parquet` holds 32 held-out PepBench complexes, one per family, filling the
length axis round-robin before taking a second example from any length bin — so it spans
all 12 lengths from 5 to 16 with 32 distinct receptor families. Deterministic: sorted, no
sampling. ProtFrag is excluded from the panel because its fragments come from monomeric
contexts, so their "interface" is intramolecular packing rather than a binding event.

## Native export for PPFlow and POTFlow

```
/zfsauton/scratch/yixiz/LPData/preprocessed/lpgen_native/
  train/{example_id}_B/{peptide,pocket}.pdb      33,089 complexes
  val/{example_id}_B/{peptide,pocket}.pdb         4,028 complexes
  {train,val}_names.pt      torch-loadable name lists (PPFlow's include_pdb_path)
  names.txt                 POTFlow's blocklist, deliberately EMPTY
  lpgen_export.summary.json
```

Build with `./scripts/submit_lpgen.sh --submit export` (~2.5 min, cpu partition), or
`--submit all` to chain it behind a manifest rebuild with `afterok`. Both arms want a
directory per complex with a peptide file and a receptor file and differ only in filenames
and membership mechanism, so one exporter serves both — and because both read the same
manifest, a difference between the arms cannot be a difference in examples seen.
**33,089 + 4,028 exported, 0 failures.** `--ppflow-aliases` (default on) additionally writes
`peptide_repaired.pdb` / `receptor_repaired.pdb`, the names PPFlow's loader opens; **no FoldX
repair is applied**, and the summary records that.

**Membership becomes manifest-driven for free.** POTFlow's dataset *is* its directory
listing (`all_pdbs = os.listdir(self.structure_dir)`), with the LMDB cache keyed on a
separate `name` field — which is why changing the cache name cannot change the split.
Exporting one directory per split makes `os.listdir` return exactly the manifest's rows, so
no loader patch is needed at all. The exporter also refuses to run if a split directory
contains anything the current export did not write, because a leftover directory from an
earlier, larger run would otherwise silently enlarge the training set (`EXPORT_CLEAN=1` by
default makes a rerun authoritative). PPFlow keeps its own native path: it filters one
`data_dir` through `include_pdb_path`, so it gets `{split}_names.pt`.

**`heldout` is not exported**, so no arm can point a `structure_dir` at the families
reserved for the later CP-sequence comparison.

**Directory names end in `_B` on purpose.** POTFlow recovers the peptide chain as
`data['id'][-1]` — the last character of the directory name, which is why PepMerge's
entries are named `1a07_C`. A name ending in anything else hands POTFlow the wrong chain
letter and nothing errors.

### The pocket crop is measured, not chosen

POTFlow reads `pocket.pdb`, not a full receptor, and our receptors run to 1,339 residues.
The cutoff was measured against PepMerge, the dataset POTFlow's released configuration
points at: across sampled entries the farthest residue *included* in its `pocket.pdb` sits
9.37–9.98 Å from the nearest peptide heavy atom, and the nearest residue *excluded* sits
10.01–10.53 Å away. The definition is therefore **exactly 10.0 Å, heavy atom to heavy atom,
whole residues** — inclusive at the boundary. Our export reproduces it: pocket median 60
residues against receptor median 109, inside PepMerge's own 0.14–0.46 pocket fraction.
Do not round this value; it is a measurement of the distribution the released checkpoint
was trained on. A residue is kept whole if *any* heavy atom is inside, and hydrogens never
pull a residue in.

Complexes with fewer than 4 pocket residues are rejected rather than exported — a peptide
with no receptor nearby is not a bound complex, and training on it would teach the
target-conditioning trunk to ignore an empty target. None occurred.

### TER/END records matter here

The exporter passes ATOM lines through verbatim apart from renumbering serials, so
coordinates, occupancies, b-factors and element columns cannot be corrupted by
reformatting. `TER` and `END` are written to the column spec, with `TER` carrying the final
residue's name, chain and number.

This is not cosmetic. A parser matches `line[0:6]` against `"TER   "`, so a bare `TER\n`
is silently an *unrecognized record* — and POTFlow runs `mkdssp` on `peptide.pdb`, DSSP is
strict about chain termination, and upstream swallows every DSSP failure in a bare
`except:` that relabels the entire peptide as helix with no log line. A malformed TER is
therefore a direct path into POTFlow's worst silent failure mode. Verified: Biopython parses
every exported file with **zero** warnings, and peptide sequences round-trip from the
manifest through the export with 0 mismatches on a source-stratified sample.

## Arm priority: lead with RFdiffusion3

**Decision 2026-10-12: POTFlow is parked; the milestone leads with the maintained
baselines.** Scored on documentation and maintenance state rather than on the papers:

| | PPFlow | PepGLAD | **foundry / RFD3** | POTFlow |
|---|---|---|---|---|
| upstream | `EDAPINENUT/ppflow` | `THUNLP-MT/PepGLAD` | `RosettaCommons/foundry` | `CMACH508/potflow` |
| env spec | `requirements.txt` | **none** | `pyproject.toml` | **none** |
| docs dir | no | no | **14 files** | no |
| test files | 0 | 0 | **120** | 0 |
| CI workflows | no | 1 | **4** | no |
| last commit | 2025-05-17 | 2025-04-16 | **2026-10-09** | — |

RFD3 is in a different category: actively maintained, packaged, tested, documented, with
public weights and complete training code. It also fits this project specifically — it runs
on **AtomWorks**, which this repository already patches. PPFlow is the natural second arm
(it has an env spec, and the native export plus its `include_pdb_path` list already serve
it). PepGLAD shipping **no environment specification at all** puts it in the same class of
risk as POTFlow, so it goes last.

POTFlow's repairs are built, tested and idempotent (see below) and can be picked up
whenever; nothing there is wasted. It is parked because it is unmaintained research code:
no public repo findable, no environment specification, a README whose entire Installation
section is `git clone`, and seven verified defects including a module-level crash.

### Re-scored 2026-10-14 on TASK FIT, which reorders everything

The table above scores maintenance. Maintenance picked RFD3 correctly but ranked the other
three on the wrong axis, because it never asked whether an arm can express the eventual
task. Stating that task precisely is what separates them:

> target + CP sequence → an LP with low binding energy to that target, sequence similarity
> to the CP as high as possible, substitutions only, fixed length, **new conformation and
> binding mode allowed**.

This is **similarity-budgeted codesign**. Fixed-sequence generation is its *k=0* corner and
de novo codesign is its *k=L* corner; the milestone needs the dial in between. So the
discriminating capability is **per-position sequence conditioning combined with a free
peptide pose** — pin a chosen subset of identities, generate the rest, and let the whole
peptide move. Scored on that:

| arm | per-position sequence control | peptide pose | fit |
|---|---|---|---|
| **foundry / RFD3** | **exact** — task-2 reveals 80–95% of identities, generates the rest | generated | **best** |
| **DiffPepBuilder / DiffPepDock** | **native per-residue** (see below) | fully diffused | **best** |
| UniMoMo | **none** | generated | poor |
| PepGLAD | all-or-nothing | generated | wrong granularity |
| PPFlow / PepFlow | plausible, undocumented | generated | unverified |

Live upstream state, read from the GitHub API on 2026-10-14 rather than from the papers:

| | **DiffPepBuilder** | **UniMoMo** | PepFlow | PPFlow | PepGLAD |
|---|---|---|---|---|---|
| upstream | `YuzheWangPKU/DiffPepBuilder` | `kxz18/UniMoMo` | `Ced3-han/PepFlowww` | `EDAPINENUT/ppflow` | `THUNLP-MT/PepGLAD` |
| last push | **2025-10-24** | **2025-12-07** | 2025-05-25 | 2025-05-17 | 2025-04-16 |
| license | MIT | MIT | MIT | **none** | MIT |
| env spec | `environment.yml` | 2 conda ymls | `environment.yml` | `requirements.txt` | **none** |
| released weights | **2** (1.24 GB each) | 1 | 2 | — | yes |
| trained on | **PepPC-F / PepPC** | **PepBench + ProtFrag** | PepMerge | PepMerge | **PepBench + ProtFrag** |
| tests / CI | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 1 |

Note that **PepBench and ProtFrag are PepGLAD's own datasets** and PepPC-F is
DiffPepBuilder's, so "has published results on our data" is a checkable property here, not a
vague one. Outside RFD3, though, **not one of these five has a single test file** — so
"well documented" tops out at README + env spec + released weights.

Staged commits: `diffpepbuilder` at `c19eb4f` (upstream 2025-10-24), `unimomo` at `f58f117`
(upstream 2025-12-07), in `$ZFS/lpgen/upstream/upstream_manifest.json`.

### Why DiffPepBuilder fits, read from the staged source

Three per-residue channels, and crucially the sequence channel is **decoupled from the
coordinate mask**:

- **`diffused_mask = chain_feats['ligand_mask']`** (`data/pdb_data_loader.py:191`) — the
  whole peptide is always diffused, so the pose is free and a new binding mode is allowed
  no matter what is pinned on the sequence side.
- **`fixed_mask = 1 - diffused_mask`** (`:192`) is therefore *not* the pinning handle: it is
  the exact complement of the diffusion mask, so pinning an identity through it would freeze
  that residue's coordinates too. This is the trap to avoid.
- **The `aatype` token channel is masked for every diffused position** —
  `aatype = torch.where(fixed_mask == 0, full_like(aatype, mask_idx), aatype)`
  (`model/embedding.py:169`), with a dedicated `[MASK]` type at index `num_aatypes` (`:115`).
- **The real sequence channel is a per-residue ESM-2 embedding**, `use_esm_embed: True`,
  `raw_esm_size: 1280` downsampled to 512 (`config/base.yaml:53-55`), appended to the
  **node** features *and* cross-concatenated into the **pair** features
  (`model/embedding.py:179-180`). It is independent of `diffused_mask`.
- **Output is a per-residue distribution over identities**: `aa_logits_pred`,
  `[batch, N_res, 20]` (`model/score_network.py:177`).

The two released checkpoints are exactly the two corners, and `mask_lig_seq` is the switch
(`experiments/train.py:575`):

```python
if self._data_conf.mask_lig_seq:                       # k = L, de novo
    final_aa_prob = F.softmax(infer_out['aa_logits_pred'] / temperature, dim=-1)
    final_aatype  = torch.multinomial(...)             # SAMPLE the sequence
else:                                                  # k = 0, docking
    final_aatype  = gt_aatype                          # USE the given sequence
```

and the two inference entry points build `esm_embed` themselves, bypassing the training
loader's unconditional wipe at `:200`:

- **de novo** (`experiments/run_inference.py:175-179`) tiles the **mean receptor embedding**
  across the peptide — no sequence information at all;
- **docking** (`experiments/run_docking.py:153-156`) passes the **real per-residue embedding
  of the whole complex, peptide included**, straight through, computed from a FASTA by
  `experiments/process_batch_dock.py:210-219`.

So DiffPepDock genuinely is sequence-conditioned, and intermediate *k* is a **per-position
interpolation between two mechanisms both checkpoints were trained with**: real embedding at
pinned positions, mean placeholder at editable ones, `gt_aatype` kept at pinned positions and
sampled from `aa_logits_pred` at editable ones. No architecture change is needed for the
first curve.

Two honest caveats:

- **Intermediate *k* is off-distribution for both checkpoints.** `diffpepdock_v1` was
  fine-tuned with the peptide sequence *always* revealed (`config/finetune.yaml`
  `mask_lig_seq: False`) and `diffpepbuilder_v1` with it *always* wiped. Each corner is
  in-distribution for exactly one checkpoint and neither was trained on the middle. The fix
  is the same recipe RFD3 is already running — train with a **random reveal fraction** — which
  is why the ΔG-vs-*k* measurement comes before any training, not after.
- **Identities are sampled independently per position** (`torch.multinomial` over flattened
  positions), so pinned residues inform the sampled ones only through the structure and the
  ESM/pair features, never through sequence-level coupling.

### DiffPepBuilder's preprocessing REWRITES ITS INPUT PDBs IN PLACE

Read this before pointing `process_batch_dock.py` at anything. Its docking preprocessing
does not merely read the input — it **edits the files on disk**:

```python
chain_map = _build_chain_map(chain_ids, ligand=lig_chain)
if any(chain_map[c] != c for c in chain_ids):
    renumber_rec_chain(file_path, receptor_info_path, in_place=True, chain_map=chain_map)
```
(`experiments/preprocess_utils.py:211-215`, called from `process_file`)

The docstring states the intent: *"ensure ligand→'A'"*. Our complexes are **chain A =
receptor, chain B = peptide**, so declaring `"lig_chain": "B"` makes
`_build_chain_map(chain_ids, ligand="B")` return `{"B": "A", "A": "B"}` (`:48-57`) — our
receptor and peptide chain labels are **swapped in the file itself**, and the receptor-info
JSON is rewritten too.

Why this is not a small matter: **RFD3 reads those very same complex PDBs by path out of
`lpgen_manifest.parquet`.** Pointing DiffPepBuilder's preprocessing at the canonical data
root would silently relabel the chains underneath the RFD3 arm — one arm's preprocessing
corrupting another arm's training data, with nothing raising an error, and the manifest's
own length/sequence checks would still pass because only the labels moved.

So the adapter must **copy the selected complexes into a per-run scratch directory and
point `--pdb_dir` at the copy**, never at `$ZFS/LPData/preprocessed/...`, and should assert
that the resolved `--pdb_dir` is not under the data root before running.

Two useful facts from the same reading:

- **`--peptide_seq_path` is optional**: *"If not provided, the script will use the ligand
  chain from the PDB file."* With `lig_chain` set, `lig_seq = extract_lig_seq(file_path,
  "A")` (`:224`) takes the native peptide sequence straight from the structure. That is the
  **k=0 baseline with no sequence plumbing at all** — DiffPepDock re-docks the native
  sequence. For *k>0*, supply a FASTA (`>id` → sequence, parsed by `read_peptide_seq`,
  which rejects duplicate ids).
- **`pocket_cutoff` defaults to 10** — the same 10.0 Å heavy-atom pocket definition this
  document already measured against PepMerge for the native export. The two arms agree on
  the pocket by coincidence of convention, which is worth keeping rather than overriding.
- Priority is `lig_chain > hotspots > motif`, and supplying both warns and ignores the
  motif/hotspots (`:225-230`).

### The k=0 DiffPepDock baseline RUNS, and the no-rewrite guarantee is measured

**Smoke run PASSED, job 84139 (gpu28, a5000).** 3 complexes x 2 samples in 74.75 s
(~12.5 s/sample), model 103.66 M parameters.

Two contracts were checked rather than assumed:

- **k=0 preserves the sequence exactly.** The generated chain A of `1d4t` reads
  `KSLTIYAQVQK`, identical to the staged input. That is `mask_lig_seq=False` doing what the
  source says: `final_aatype = gt_aatype` instead of sampling from `aa_logits_pred`.
- **The pose is genuinely re-docked, not copied** -- which is what makes the baseline
  informative at all:

  | target | n_res | raw CA RMSD (A) | superposed (A) |
  |---|---|---|---|
  | 1d4t | 11 | 3.59 / 3.76 | 1.98 / 2.09 |
  | 1bjr | 10 | 6.93 / 6.74 | 5.09 / 4.77 |
  | 1ee5 | 19 | 3.56 / 3.75 | 2.74 / 2.97 |

  Raw exceeds superposed throughout, so part of the deviation is placement in the pocket and
  part is conformation. Two samples of the same target differ, so there is real sampling
  variation. A near-zero raw RMSD would have meant the run was echoing the crystal pose and
  the baseline was worthless; that is the check to repeat on any future k.

**The no-rewrite guarantee held in three consecutive runs** (84127, 84128, 84139), each
reporting `hash CHANGED: 0` over the staged inputs with exactly the predicted number of
`*_processed.pdb` siblings. The verify runs BEFORE sampling, so a failure costs no GPU time.

Note the output receptor is the pocket crop, not the full chain (~365 atoms for `1d4t`),
consistent with `pocket_cutoff: 10`. And chain A carries ~5 atoms per residue (N, CA, C, O,
CB): this model predicts backbone frames plus 1 psi and 2 chi angles, so exported poses are
not all-atom. Anything downstream that needs full side chains has to pack them -- which is
the shared evaluator's job, not this job's.

### The k=0 reference numbers, on 80 complexes

**Job 84144 (80 targets x 8 poses = 640, 48.6 min, 4.55 s/pose on an a5000).** Scored by
`script_utils/lpgen_dpb_score.py` on the cpu partition; rows in `score/poses.jsonl`, summary
in `score/summary.json`, so every table below is re-derivable without a GPU.

| metric | center_pos_noise ON | center_pos_noise OFF |
|---|---|---|
| **k=0 contract: identical sequence** | **1.0000 (80/80 targets)** | **1.0000 (80/80)** |
| median raw CA RMSD | 5.62 A | 5.63 A |
| median superposed CA RMSD | 2.29 A | 2.32 A |
| median interface raw CA RMSD | 5.61 A | 5.60 A |
| best-of-8 raw (median over targets) | 5.35 A | 5.33 A |
| pairwise pose diversity | 1.17 A | 1.29 A |
| poses <= 2 A interface | 0.066 | 0.066 |
| targets with >= 1 hit | 0.100 | 0.113 |

**The k=0 contract holds exactly**: every one of 640 poses carries the input sequence
byte-for-byte. That is what makes the similarity axis measurable at all, and it is computed
rather than trusted.

**`center_pos_noise` is a no-op, and the ablation above was run because it did not look like
one.** `docking.yaml` ships `center_pos_noise: True, center_pos_noise_std: 5.0`, which
together with a 5.6 A placement error and a near-constant within-target error (0.17 A) made
a very plausible story: `@fn.lru_cache` on `_process_csv_row` draws the offset once per
target, so all 8 repeats would share it. The paired ablation says otherwise, and
`parse_chain_feats` (`data/utils.py:451-480`) explains why:

```python
bb_center += np.random.normal(0, center_pos_noise_std, bb_center.shape)
centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]   # BOTH chains
coordinate_bias = pdb_center + bb_center                                  # added back on write
```

The offset translates **receptor and peptide together**, leaving relative geometry
unchanged, and `coordinate_bias` restores it at write time. An exact round-trip: a no-op for
an SE(3)-equivariant network and for RMSD. Do not "fix" it, and do not cite it to explain a
pose error.

**So the 5.6 A is genuine**: handed the native sequence and the native pocket, DiffPepDock
reproduces the crystal pose for only ~10% of targets, while getting the internal
conformation to 2.3 A. The error is placement, not fold. **The k=0 anchor is therefore
loose**, and every k>0 point is compared against an instrument with a large noise floor.

Three protocol gaps against their published setting, named rather than assumed away:

* **8 poses, not their default 32.** But measured pairwise diversity is only 1.2-1.3 A, so
  4x more samples explores a ~1.3 A neighbourhood and is unlikely to close a 5.6 A placement
  error. That the obvious explanation is weak is itself worth knowing.
* **No relaxation** -- we run `run_postprocess=False`.
* **No selection.** Their postprocess ranks by Rosetta interface energy; we have no ranking
  signal yet, so these are medians over *unselected* poses. Calling them "docking accuracy"
  would overstate what was measured. Closing this is the next step: packing plus dG through
  the shared evaluator.

Two reading notes. **Pairwise diversity is measured between poses, not inferred from the
spread of RMSD-to-native** -- several poses can each sit 5 A from the native while being far
from each other, so the spread of a distance cannot stand in for diversity. And
`num_repeat_per_eval_sample` works via the custom sampler in `run_docking.py:169-209`, which
expands each target index N consecutive times; the `np.repeat` line commented out at
`pdb_data_loader.py:82` is the superseded mechanism, not a bug.

### Six lazy-import patches, because the deps are unsatisfiable on python 3.9

`run_docking.py` could not be imported at all: its postprocessing stack needs **pdbfixer,
openmm AND pyrosetta at module level**, even with `postprocess.run_postprocess=False`, which
is how we run it. Three hops, each discoverable only by running:

1. `run_docking.py:29` -> `analysis.postprocess` (job 84127)
2. `run_docking.py:35` -> `experiments.train:42` -> `analysis.metrics:18` -> same (job 84138)
3. `analysis.metrics:21` -> `analysis.postprocess_utils:24` -> `pdbfixer` (+ `openmm.app:26`)

**Installing them is not an option on this arm's python.** `pdbfixer` publishes exactly one
version, 1.12.0, source-only, which requires `openmm>=8.2` -- and openmm's last release with
cp39 wheels is **8.1.1**. Measured against the PyPI index 2026-10-14. The arm pins python
3.9.16 (and vendors openfold, which is version-sensitive), so satisfying the pair would mean
conda, a python bump, or compiling openmm in an env job.

So `lpgen_patch_upstream.py` gained six asserted idempotent edits, each deferring an import
into the single function that uses it. The chain is **provably** closed rather than merely
importable: exactly four files in the tree import those three packages at module level
(`postprocess_utils`, `postprocess`, `amber_minimize`, `cleanup`), all on the postprocessing
side; `amber_minimize` and `cleanup` are reachable only through `postprocess`; `postprocess`
is deferred in both of its importers; and `postprocess_utils` is reachable only through
`postprocess` and the one line deferred above. Verified with all three packages absent:

```
pdbfixer installed: False | openmm installed: False | pyrosetta installed: False
analysis.metrics      OK, ALL_METRICS = 10
experiments.train     OK
```

Because the deferral sits inside the using function, turning postprocessing back ON still
raises the same ImportError at the same point -- it does not silently skip the work.

**One env-shape trap worth keeping.** A uv venv ships no `setuptools`, while the conda export
this was translated from pinned `setuptools==68.0.0`, so `pkg_resources` was simply absent
and `import wandb` (module-level in `experiments/train.py`) died deep inside
`wandb.sdk.launch.builder.build` -- on a GPU node, after preprocessing had already run
(job 84128). `setuptools==68.0.0` is now pinned, and the env probe list gained `wandb` and
`pkg_resources`: **probe what the entry point imports, not what looks load-bearing**, since
every module in the original probe list imported fine while the entry point was broken.

### Why UniMoMo does not fit, despite the best numbers on our data

UniMoMo is PepGLAD's successor by the same author, trained on **our** PepBench+ProtFrag and
tested on **LNR, which is also our holdout set**, where it reports beating PepGLAD, PepFlow
and RFdiffusion (37.59% AAR, ΔG −28.72). It also has the environment spec PepGLAD lacks. It
is still demoted to a **calibration arm**, for a reason read from its source:
`api/templates/pep.py` builds the peptide as **all-GLY dummy blocks carrying a single `C`
atom each**, and nothing in the generation path accepts residue identities. The only
similarity dial is latent noise — the entangled regime this project has already measured
(abstention is pure `t_lat`; identity is not geometry). Reproduce its LNR numbers to get a
published bar on our exact data; do not build the CP pipeline on it.

Also: its full training recipe wants **8×80 GB**, which no partition here provides, so only
fine-tuning from its released checkpoint is in scope.

### A leak that gates every fine-tune from released weights

Our split is **family-level** over PepBench+ProtFrag. PepGLAD's and UniMoMo's released
weights were trained on **PepBench's own `train.txt`** — the split this document already
records as leaky. So those checkpoints have very likely seen families our manifest assigns
to `val` and `heldout`, which would contaminate both the 32-complex validation panel and the
later CP-sequence comparison. DiffPepBuilder is exposed the same way through PepPC-F.

This is cheap to measure before trusting any fine-tune: `train.txt`/`valid.txt` are on disk
at `$ZFS/LPData/PepBench/train_valid/`, and the manifest carries `cluster_id` per row.

And separately: "already trained on our data" buys confidence that a method works on this
distribution. It does **not** license skipping the common manifest — every arm is still
fine-tuned and scored against `lpgen_manifest.parquet`, or the comparison stops meaning
anything.

### The first RFD3 tranche diverged, and it was never a fine-tune

**MEASURED, job 83990 (gpu31, cancelled at 46 min of a 24 h allocation).** The run was
training from **random weights**, and its sequence head diverged:

| | early | late |
|---|---|---|
| token-level sequence loss | 3.36 → **1.94** | 6.96 → 22.2 → 33.8 → **44.6** |
| seq recovery | 0.05 → **0.56** | 0.058 → 0.049 → **0.000** → 0.011 |
| structure MSE | 0.040 | **0.016** — fine throughout |

So this is **sequence-specific**: the sequence head learned to 0.56 recovery and then
collapsed to zero while the structure loss kept improving, with two
`X_noisy_L: Tensor contains NaNs!` warnings (both on protfrag rows). Config: `base_lr:
0.0018`, `warmup_steps: 1000`, `batch_size: 1`, `grad_accum_steps: 3` — an effective batch
of **3 examples**. The run reached ~833 optimizer steps (~2500 examples), so the LR was
still *ramping* toward 1.8e-3 when it blew up. 1.8e-3 is AF3's large-batch pretraining LR.
A follow-up run (84043) lowered it to 1e-4.

Note the step accounting when reading these logs: progress is printed as `Epoch N Batch M`
over `n_examples_per_epoch: 2400` at `batch_size: 1`, so **batches are examples, and
optimizer steps are batches ÷ 3**. Reading the batch counter as the step counter overstates
training by 3×. Also, the only LR value that reaches the log is `lr: 0` — the real schedule
position is not logged, so it has to be derived or logging has to be added.

**The released weights were available the whole time.** The train job's banner said they
"are not on this cluster", which reads as unobtainable and was acted on as such — jobs
83990 and 84043 both ran from random init. In fact `foundry list-available` offers `rfd3`
and the CLI ships in the arm's own env:

```bash
./scripts/submit_lpgen.sh --submit weights --set STAGE_ARMS=foundry
# -> $ZFS/lpgen/weights/foundry/rfd3_latest.ckpt   2,690,316,669 bytes
./scripts/submit_lpgen.sh --submit train --set TRAIN_CKPT_PATH=<that path>
```

The banner now names the path and prints that line when it finds the checkpoint present but
unused. It deliberately does **not** adopt it automatically: silently turning a
from-scratch run into a fine-tune would change what the job means without the submitter
asking. And when it does become a fine-tune, the inherited-scheduler/EMA trap below applies —
check the effective LR on the first logged step before attributing anything to the LP data.

### Released checkpoints: the `weights` stage

```bash
./scripts/submit_lpgen.sh --submit weights --set STAGE_ARMS=diffpepbuilder,unimomo
```

Downloads to `$ZFS/lpgen/weights/<arm>/`, **outside** the upstream trees, so re-staging an
arm does not discard 2.5 GB. Published md5s (Zenodo API, 2026-10-14) are verified after
download and recorded in `weights_manifest.json`; the GitHub release for UniMoMo advertises
no checksum, so that row is recorded as `recorded_unverified` rather than claimed as
verified. Downloads land in a `.part` file and are moved only on success — an interrupted
job must not leave a truncated checkpoint that the presence check then treats as complete,
and `curl --fail` matters for the same reason: without it an HTTP error body is written
*into* the `.pth`, which fails much later as a confusing unpickling error.

| arm | file | size | md5 |
|---|---|---|---|
| diffpepbuilder | `diffpepbuilder_v1.pth` | 1242.3 MB | `5ab7d914a979db4f11ac778a51dd2241` **verified** |
| diffpepbuilder | `diffpepdock_v1.pth` | 1242.3 MB | `60579b59e97a4a2109f2b31b4fda185a` |
| diffpepbuilder | `esm2_t33_650M_UR50D.pt` | ~2.5 GB | not published |
| unimomo | `model.ckpt` | — | not published |
| foundry | `rfd3_latest.ckpt` | 2690.3 MB | installed by `foundry install`, no URL |

`foundry` is not a row in `WEIGHTS_SPEC` but a branch of its own, because it ships an
installer that knows the checkpoint registry and the current RFD3 revision. It is pointed at
the weights store explicitly: its default is `~/.foundry/checkpoints`, and home is a quota'd
volume that also holds this repository.

### DiffPepBuilder's environment: translated, not conda-created

Its `environment.yml` is a fully-pinned `conda env export` (python 3.9.16, torch 2.1.0
cu118), and we do **not** run `conda env create` on it: it pins build strings from the
`defaults` channel (`openssl=3.0.12=h7f8727e_0`, `ca-certificates=2023.7.22`, and
`cudatoolkit=10.2.89` alongside `cuda-runtime=11.8`) that no longer resolve together. The
`pip:` section of that same export is exact and portable, so `_lpgen_arm_requirements`
installs it verbatim under uv and translates only the torch stack.

**Two of its pins do not exist as wheels** — measured against the PyPI index 2026-10-14:
`openmm`'s earliest published wheel is **8.1.1** (it pins 8.0.0) and `pdbfixer` publishes
exactly **one** version, 1.12.0 (it pins 1.9). Both are conda-forge packages. So they are
omitted, sampling must run with `postprocess.amber_relax=False`, and relaxation happens in
the common CPU evaluator instead. An env built with those two lines present cannot resolve
at all. PyRosetta is omitted for a different reason: scoring belongs on the CPU partition
through the one evaluator every arm shares.

Per-arm python version and constraints are now resolved by `_lpgen_arm_python` /
`_lpgen_arm_constraints_file` / `_lpgen_arm_extra_constraints`, because the previous globals
defaulted to foundry's `ci-constraints.txt` and `atomworks<3` for **every** arm — which
would have silently pinned torch 2.7.1 over DiffPepBuilder's own tested 2.1.0.

### RFD3 setup facts, read from the source

Training entry point is `uv run python models/rfd3/src/rfd3/train.py experiment=pretrain`,
with Hydra config groups under `models/rfd3/configs/` (`experiment/` ships `pretrain`,
`debug`, `test-uncond`, `test-unindexed` — an LP experiment is a new file there, as the plan
requires). The trainer is Lightning **Fabric** (`src/foundry/trainers/fabric.py`), not
PyTorch Lightning.

**Fine-tune initialization is already model-only for the optimizer, but not for everything.**
`CheckpointConfig` defaults to `reset_optimizer: False`, which reads alarmingly, but
`rfd3_base.yaml` sets `skip_optimizer_loading: True` and the trainer computes
`reset_optimizer = bool(ckpt_config.reset_optimizer or self.skip_optimizer_loading)` — so
the optimizer is fresh with the stock config. What still carries over:

- **Scheduler and EMA cannot be reset.** `CheckpointConfig`'s own docstring is
  `TODO: Implement reset_scheduler and reset_ema`. A fine-tune therefore inherits the
  pretraining run's LR-schedule position and EMA state. For a bounded tranche this matters:
  if the released checkpoint sits near the floor of a long decay, the arm will look
  undertrained for reasons that have nothing to do with the data. Check the effective LR on
  the first logged step before attributing anything to the LP data.
- **The epoch counter resumes and is incremented**, then stopping conditions are evaluated
  immediately: `self.state["current_epoch"] += 1`, then
  `if max_epochs is not None and current_epoch >= max_epochs: should_stop = True`. Stock
  `max_epochs: 100_000` makes this latent — but it fires exactly when someone bounds an LP
  tranche the obvious way (`max_epochs: 10`), and the run then trains **zero steps and exits
  cleanly**, which is indistinguishable from a completed tranche. Bound the run by epochs
  *above* the checkpoint's epoch, or by wall clock, never by a small `max_epochs`.

**Install with upstream's own constraints, not latest-resolves.** This is not a preference,
it is required: a plain `uv pip install -e .` resolved **atomworks 3.0.1**, which *removed*
`StructuralDatasetWrapper` — the class **both** rfd3 dataset configs instantiate
(`train/pdb/base_no_weights.yaml` and `train/rfd3_monomer_distillation.yaml`). The training
data path simply cannot be built on 3.x. foundry's `ci-constraints.txt` pins
`torch==2.7.1`, `rdkit==2025.3.6`, and its header documents the same atomworks fragility;
`pyproject.toml` says only `atomworks[ml]>=2.1.1`, which does not cap the major. The env
job therefore installs with `-c ci-constraints.txt` plus `atomworks<3`, giving
**torch 2.7.1+cu126 / atomworks 2.2.1**, and its verification checks
`StructuralDatasetWrapper` *by name* — importing `atomworks.ml.datasets` successfully
proves nothing about whether the data path exists.

**GPU reality, measured.** With the constrained env the pre-Ampere partitions are usable:

| GPU | arch | bf16 | measured (2048³ matmul) |
|---|---|---|---|
| Tesla V100 | sm_70 | emulated | **7.14 TFLOP/s** (fp32 4.67) |
| RTX 2080 Ti | sm_75 | emulated | **5.25 TFLOP/s** (fp32 4.21) |
| a5000 / a6000 | sm_86 | native | not yet measured |
| h200 | sm_90 | native | not yet measured |

- `precision: bf16-mixed` needs **no change** to run on either legacy GPU. On both,
  `is_bf16_supported()` is `True` while `is_bf16_supported(including_emulation=False)` is
  `False` — bf16 is emulated, yet still *faster* than fp32.
- An earlier revision of this file claimed V100 could not run RFD3 at all
  (`no kernel image available`). That was true only of the **unconstrained** cu130 env
  (torch 2.14.1, whose arch list omits sm_70 because CUDA 13 dropped Volta) — the env we
  must not use anyway. Under the pinned cu126 build, V100 is fine.
- Still size tranches on Ampere+: ~5–7 TFLOP/s is roughly an order of magnitude below
  native-bf16 Ampere, so legacy is for the one-minibatch **preflight**, not for a 24-hour
  tranche. But preflight can proceed now instead of queueing.
- **gpu14 is a broken node: exclude it.** Both the foundry torch *and* this project's own
  torch 2.7.0+cu126 fail there with `CUDA unknown error` at `cuda.init()`, while gpu17 and
  gpu18 — same model, partition and driver — pass. `device_count()` returns 1 before
  failing and `myfree` lists it AVAILABLE, so it looks healthy. Use `--exclude=gpu14`.

Other knobs for sizing a tranche: `n_examples_per_epoch: 2400` makes an epoch a controllable
unit, `checkpoint_every_n_epochs: 10`, `validate_every_n_epochs: 4`, `grad_accum_steps: 3`
(upstream notes it is overridden by their `launch.sh`). HBPLUS is needed only for
hydrogen-bond conditioning, which the plan disables for the first run — so it is not a
dependency here.

Datasets already on disk: `PepMerge` (10,348 entries, PPFlow/POTFlow format),
`PepPC`/`PepPC-F` (PepGLAD), plus the preprocessed PepBench and ProtFrag behind the manifest.

### RFD3 reads the manifest directly — no conversion needed

The custom-dataset template (`configs/datasets/train/rfd3_monomer_distillation.yaml`) is an
`atomworks.ml.datasets.PandasDataset` over **a parquet with `example_id` and `path`
columns**, wrapped in `StructuralDatasetWrapper` with a `GenericDFParser`. That is exactly
`lpgen_manifest.parquet`'s shape, so RFD3 consumes the manifest as-is and reads our
two-chain complex PDBs by path. The native per-complex export is for PPFlow/POTFlow; RFD3
does not need it.

**Do NOT copy `b_factor_min: 70` from that template.** It is a pLDDT filter, and the
distillation set is AF2 models, where the B-factor column holds pLDDT and higher is better.
Our data is experimental, where the same column means crystallographic disorder and lower
is better — and measured over our manifest it splits by source:

| source | B-factor |
|---|---|
| PepBench | **entirely 0.000** in every structure sampled |
| ProtFrag | real values, 41.6–121.8, mean 78.9 |

So `b_factor_min: 70` would reject **100% of PepBench** — the real bound-peptide half that
carries 24% of every batch — and keep only monomeric fragments, silently. The experimental
`configs/datasets/train/pdb/base.yaml` correctly does not set it; that is the template to
follow. (A stray `999.99` B-factor also appears in at least one structure; harmless while
nothing filters on the column, but do not start.)

Still to resolve for the dataset config: cropping must retain full peptide chains
(`crop_contiguous_probability` / `crop_spatial_probability` in the template, against 5–16
residue peptides and receptors up to 1,339), and atomworks warns that `CCD_MIRROR_PATH` and
`PDB_MIRROR_PATH` are unset — needed for PDB-mirror training, to be confirmed for the
by-path custom route.

### Verified against the real AtomWorks stack

Loaded LP complexes through `PandasDataset` + `GenericDFParser` in the foundry env
(probe job 80890). Results, measured:

- **The manifest loads as-is: 6/6 and 3/3, zero failures.** RFD3 needs no data conversion.
- **`pn_unit_iid` separates the two chains exactly**: `A_1` = receptor (74–417 residues),
  `B_1` = peptide (5–14), and both counts match the manifest's `receptor_length` /
  `peptide_length` on every example. So `pn_unit_iid == "B_1"` is an exact handle on the
  peptide — and independently, **the peptide is the smaller protein unit in 6/6**, so the
  binder can be pinned without relying on a chain letter.
- **`pn_unit_iid_colnames` is load-bearing, confirmed empirically.** With
  `[receptor_chain_ids, peptide_chain_id]` → `PPICondition` accepts **6/6**. With `null`,
  the value the distillation template ships → it accepts **0/3**. The `pn_unit_iid`
  annotations exist either way (they are derived from the structure), but
  `query_pn_unit_iids` — the key `PPICondition` gates on — is populated *only* from those
  named columns. Copy the template's `null` and the binder condition silently never fires.
- Annotations present on the **raw loaded** array: `chain_id`, `chain_iid`, `chain_type`,
  `pn_unit_id`, `pn_unit_iid`, `pn_unit_entity`, `molecule_iid`, `res_id`, `res_name`,
  `atom_name`, `element`, `occupancy`, `b_factor`, `is_polymer`, `is_backbone_atom`,
  `is_aromatic`, `hetero`, `ins_code`, `transformation_id`. **`is_protein` is NOT** — the
  transform pipeline adds it, which is why the training conditions run inside the pipeline
  rather than at load time.
- Example keys: `atom_array`, `atom_array_stack`, `query_pn_unit_iids`, `chain_info`,
  `ligand_info`, `metadata`, `extra_info`, `assembly_id`, `example_id`, `path`.

Write the LP dataset config against the **modern** `PandasDataset(loader=...)` form, not
`StructuralDatasetWrapper`: that class is a *deprecated shim* even in 2.2.1 ("Use
PandasDataset with the appropriate loader function instead") and 3.x removed it, which is
precisely what breaks the two shipped configs. The loader's parameter is
**`metadata_row_parser`** — the shim passed it positionally, so `dataset_parser` looks
right and fails only at first item access.

Also: `/tmp` is **node-local** here. A job script staged in a login node's temp directory
is invisible to the compute node; put job inputs on `$ZFS`.

### RFD3 LP arm: data path and condition, verified end to end

Built and verified on CPU (jobs 80908/80936/80938). `./scripts/submit_lpgen.sh --submit
stage` then `--submit env` stages and installs; the configs live in **this** repo under
`configs/rfd3_lp/` and the condition in `src/proteinfoundation/lpgen/`, reached by
`PYTHONPATH=$REPO/src` and `hydra.searchpath=[...,file://$REPO/configs/rfd3_lp]`, so nothing
of ours is patched into the third-party tree.

```
configs/rfd3_lp/datasets/train/lpgen.yaml          the manifest as a PandasDataset
configs/rfd3_lp/datasets/conditions/lp_binder.yaml the LP binder condition
configs/rfd3_lp/datasets/lpgen_base.yaml           the `datasets` group (no PDB mirror)
configs/rfd3_lp/experiment/lp_finetune.yaml        the entry point
src/proteinfoundation/lpgen/rfd3_conditions.py     LPBinderCondition (+13 tests)
```

Verified: dataset builds at **33,089** rows (the train split exactly), **4/4 and 8/8
examples pull through the full transform pipeline** to model-ready features, and the motif
annotations come out as designed — binder pinned to `B_1` on every example, target fully
given, **peptide coordinates never given** (`is_motif_atom` and
`is_motif_atom_with_fixed_coord` are zero on the peptide throughout), and the two tasks both
firing (full-sequence reveals 113/113 and 114/114; partial reveals 29–85 of 43–113 atoms).
`ppi` was sampled on 8 of 10 items, matching the configured frequency of 8/10.

**Three things in the stock condition are wrong for this task**, and one constrains the fix:

* `PPICondition` picks the binder with `np.random.choice(query_pn_unit_iids)` — a coin flip
  that makes the **receptor** the generated chain half the time. `LPBinderCondition` pins it
  (the peptide is the smaller protein `pn_unit` in 6/6 measured examples, and the counts
  match the manifest's length columns).
* Stock reveals nothing of the binder's sequence. Ours reveals all of it, or a random
  80–95% of residues, per the plan's two tasks.
* Neither can be done with the native probabilities: both
  `sample_is_motif_atom_with_fixed_seq` and `sample_fix_motif_coordinates` are
  all-or-nothing *per-example* coin flips over the whole motif set, so they cannot treat
  receptor and peptide differently. `PPICondition` sets the annotations directly for the
  same reason, and that is the pattern followed.
* **The class must keep `name = "ppi"`.** That literal is branched on in five places —
  `pipelines.py:227` (the full-binder crop) and `:458`, `design_transforms.py:589` and
  `:637`, `hbonds.py:304`. Any other name silently skips all five, including the crop that
  keeps the peptide whole, with nothing reported.

**Config settings that are not defaults and matter:**

| setting | stock | ours | why |
|---|---|---|---|
| `train_conditions.ppi.frequency` | **0.0** | 8.0 | binder conditioning is *disabled* in both `design_base` and `pretrain` |
| `keep_full_binder_in_spatial_crop` | False | True | a cropped peptide is a corrupted target |
| `meta_conditioning_probabilities.full_binder_crop` | 0.75 | 1.0 | 0.75 corrupts a quarter of examples |
| `crop_contiguous_probability` | 0.25 | 0.0 | the contiguous route has **no** binder protection |
| `add_ppi_hotspots` | 0.75 | 0.0 | at inference the input is target + sequence, no hotspots |
| `calculate_hbonds` | 0.2 | 0.0 | plan disables it; also avoids needing HBPLUS |
| `b_factor_min` | 70 *(distillation)* | null | would reject 100% of PepBench |
| `paths.data.residue_cache_dir` | `/net/tukwila/...` | null | an upstream developer's path; `None` means "no cache" and is explicitly supported |

**Four traps hit while wiring this, all silent:**

1. `pn_unit_iid_colnames` must carry the **atomworks** `pn_unit_iid` values (`A_1`, `B_1`),
   not bare chain ids. Passing `A`/`B` composes, instantiates and *loads* without complaint,
   then dies deep in the spatial crop with `No crop center found!` because
   `is_query_pn_unit` matches nothing. The manifest now carries
   `receptor_pn_unit_iid` / `peptide_pn_unit_iid`; measured, the suffix is `_1` for all 237
   examples checked (`transformation_id` is always `1` in our single-assembly PDBs).
2. Crop overrides must sit at `dataset.transform`, not as a sibling of it. As a sibling they
   set `transform: null` — discarding the entire inherited transform block — *and* pass the
   crop keys to `PandasDataset` as constructor kwargs.
3. The sub-dataset include must target `@train`, not `@train.lpgen`; the file already nests
   its body under an `lpgen:` key, so the latter yields `train.lpgen.lpgen.dataset`, which
   composes fine and registers no dataset.
4. `columns_to_load` must include `split`, because `filters` queries it and columns are
   loaded *before* filters run.

The GPU preflight is done, including checkpoint save/reload — see below. Still open:
validation (see the note on what RFD3 validation actually *is*), and fine-tuning from the
released RFD3 weights, which need downloading.

### GPU preflight: PASSED (plan step 4)

`./scripts/submit_lpgen.sh --submit preflight` — a few real optimizer steps from **fresh
weights**, so it needs no released checkpoint. Result on an a6000 (job 82738), 48 batches,
one epoch, checkpoint written, **exit 0**:

| | |
|---|---|
| peak GPU memory | **22,544 of 49,140 MiB (46%)** on a6000 |
| GPU utilisation | peak 100%, mean 46% |
| throughput | 1.29 s/batch → **3.88 s/optimizer step** (grad_accum 3) |
| realised mixture | protfrag 77.1% / pepbench **22.9%** (configured 24/76) |

**A 24 GPU-hour tranche is therefore ~22,300 optimizer steps, ~66,900 examples, ~2.0 passes
over the 33,089 train rows.** That is a real but modest budget; it is a spending cap, not a
convergence claim.

**The a5000 (24 GB) is not enough.** Measured at `crop_size: 384`,
`diffusion_batch_size_train: 16`: 12 batches survived at 89% peak, then 48 batches OOM'd
(`Tried to allocate 1.23 GiB`, 1.18 GiB free). The first OOM also reported **4.70 GiB
"reserved by PyTorch but unallocated"**, so fragmentation was a real part of it and the job
now exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as torch's own message
advises. The submitter excludes gpu28 (the only a5000 in `general`) along with gpu14.

**Per-row sampling weights had to be wired explicitly, and their absence is nearly silent.**
Without a `weights` sibling of `dataset`, foundry samples uniformly over rows and says so in
one line: `No weights or sampler provided for dataset: None, using uniform weights with
replacement`. Uniform over rows is **92% ProtFrag**, against the configured 24% PepBench —
and the first preflight duly drew 12/12 ProtFrag. `proteinfoundation.lpgen.rfd3_sampling.
weights_from_column` reads the manifest's `sample_weight`; verified by drawing through the
same `WeightedRandomSampler` foundry uses (PepBench: 7.8% of rows → 24.4% of draws) and
then confirmed in the training loop by the ratio table above. It refuses zero weights by
default, because a zero-weight row is never sampled and that is indistinguishable from the
row being absent — which is how a filter/weights split mismatch would hide.
`calculate_weights_for_pdb_dataset_df` (what the PDB datasets use) cannot serve: it derives
weights from PDB cluster/composition columns the manifest lacks.

**Three more environment traps, all of which abort the run before any training:**

1. **Every path in the stock `paths` config is upstream-developer-specific** —
   `log_dir: /net/scratch/$USER/...`, `/projects/ml/...`, `/squash/...`, `/net/tukwila/...`.
   Most are inert strings for datasets we do not use, but `log_dir` is **created** at
   startup, so the stock value aborts with `PermissionError: /net`. Overridden, along with
   `failed_examples_dir` (pointed somewhere real, the counterpart to
   `n_fallback_retries: 0`) and `cif_cache_dir`.
2. **`logger` is a config group.** Writing `logger: csv` as a top-level value sets it to the
   *string* `"csv"`, and `instantiate_loggers` dies with
   `'str' object has no attribute 'items'`. It belongs in `defaults` as
   `- override /logger: csv`.
3. **RFD3's logging callbacks parse `example_id` with atomworks' own four-field brace
   format** (`{[datasets]}{pdb_id}{assembly_id}{[query_pn_unit_iids]}`) and raise
   `Invalid example ID format ... 0 matches found (expected 4)` on anything else — in
   `on_train_batch_start`, i.e. *after* the model and data path already work. The manifest
   now carries a conforming `rfd3_example_id` (a second column; `example_id` remains this
   project's primary key), with the source in the dataset-name field so that the
   sampling-ratio table reports the PepBench/ProtFrag mixture. `cluster_id` could not be
   used in the `pdb_id` slot because it repeats across rows and the id must stay unique.

Also: a `--set` value containing spaces must be **quoted** when the submitter writes the env
file, or the job's `source` treats the second token as a command (`b=2: command not found`)
and dies before any work.

### Checkpoint reload: the epoch-stop trap, now MEASURED

The warning in `lp_finetune.yaml` about bounding a run with `max_epochs` was reasoning from
the source; it has now been run. Reloading the preflight's own checkpoint
(`current_epoch = 0`, `global_step = 16`):

| `max_epochs` | outcome |
|---|---|
| **1** | **ZERO training. 0 epochs finished, 0 checkpoints written, GPU util peak 0% / mean 0%, memory peak 1,600 MiB — and `exit code: 0`.** |
| 3 | trains epochs **1 and 2**, exit 0, GPU util peak 100% / mean 50% |

So a checkpointed run bounded by a small `max_epochs` does not merely stop early: it never
touches the GPU, writes nothing, and **reports success**. Nothing in the exit status, the
log level, or the job state distinguishes it from a completed tranche. The data path is even
built — the LP sampling weights are logged — it is simply never iterated. Bound a tranche by
wall clock (the SLURM time limit), or by `checkpoint_epoch + 1 + desired_epochs`.

Note also that `max_epochs: 3` from a checkpoint at epoch 0 buys **2** epochs, not 3: the
load does `current_epoch += 1`, so the first epoch index is consumed by the checkpoint.

**Size tranches on the RESUME number, not the fresh-weights number.** Peak GPU memory was
**46%** of the a6000 training from scratch but **71%** resuming from a checkpoint
(34,698 of 49,140 MiB). The from-scratch preflight understates what a fine-tune needs.

`global_step` came back as **16** for 48 batches at `grad_accum_steps: 3` — i.e. it counts
optimizer steps, not micro-batches, for this single-optimizer setup. (Worth stating because
this project has been bitten the other way: with two optimizers, Lightning's `global_step`
counted both and invalidated a step-matched comparison.)

`skip_optimizer_loading: true` is visible in the resolved config of both runs, confirming the
optimizer is genuinely fresh on a reload.

### What RFD3 "validation" actually is (and why the LP panel is not wired yet)

RFD3's validation is **sampling-based design metrics, not a held-out loss**.
`AADesignTrainer.validation_step` runs a rollout forward pass and then builds a predicted
atom-array stack and scores design metrics; the val datasets are `ContigJsonDataset` over a
contig JSON with `is_inference: True`. There is no teacher-forced val-loss path.

For this project that is the right instrument rather than a limitation — the repository's own
history is that teacher-forced metrics made "the model works" unsupportable on CPSea, and
that sampled closure sits *below* teacher-forced closure. A val loss here would be the
misleading number.

But wiring the 32-complex LP panel into it needs two things that belong to the *next*
milestone, not this one: a contig JSON describing each target, and a decision about how
"the peptide sequence is given" is expressed as an **inference** condition. The plan defers
CP-sequence inference procedures to its section 6 on purpose, and `PPICondition`-style
training conditions are not used when `is_inference: True`. Bolting on a loss-style val
instead would produce exactly the number this project already learned not to trust.

So: `lpgen_val_panel.parquet` is built and held out (32 PepBench complexes, 32 families,
lengths 5–16), the trainer runs without validation and says so, and `validate_every_n_epochs`
is set but inert. The decision to make is whether the LP arm validates by regenerating panel
peptides from a contig spec (RFD3-native, needs the inference-conditioning design) or whether
this milestone's report is built from the preflight metrics plus a separate sampling pass.

### First training run DIVERGED — and the fix was inert twice before it took

`./scripts/submit_lpgen.sh --submit train` launches one tranche; resubmitting the same
`TRAIN_RUN_NAME` auto-resumes. Everything lands on scratch (see the note on home below).

**Job 83990, from scratch on an a6000: the sequence head diverged ~60% into epoch 0.**
`Token Lvl Sequence Loss` sat at 1.5–3.4 for ~53 logging intervals — genuinely learning,
below the ~3.0 uniform-prior baseline, with `Seq Recovery` often 0.4–0.8 — then broke:

```
... 2.31  4.41  1.86  5.90  6.96 12.45 14.46 19.02 23.34 24.33 ... 52.67 ... 54.15
```

monotonic thereafter, with recovery pinned at 0.02–0.08. **The structure losses stayed
healthy throughout** (MSE still falling, lDDT flat-ish), so nothing in the headline numbers
announced it.

Cause: the stock recipe is `AF3Scheduler` with `base_lr: 1.8e-3` and 1000 warmup steps —
AF3's schedule for AF3's batch size. Our effective batch was **3 structures**
(`batch_size: 1` × `grad_accum_steps: 3`; note `diffusion_batch_size_train: 16` is the
diffusion-sample dimension *within* one structure, not 16 complexes). Divergence hit around
optimizer step 485 of warmup, i.e. an instantaneous LR of ~8.7e-4. Nothing aborted, because
the stock trainer ships `error_if_grad_nonfinite: false` — so the blown-up head kept
training for ~300 more steps and was then **checkpointed**.

Fixes now in `lp_finetune.yaml`: `base_lr: 1e-4`, `grad_accum_steps: 16` (effective batch 16;
accumulation costs step time, not memory, and the preflight ran at 46% of the card with
mean utilisation under 50%), and `error_if_grad_nonfinite: True`.

**The LR override was a silent no-op twice before it worked**, which is worth recording
because nothing flagged either attempt:

1. A **top-level `optimizer:`** key. `optimizer` is nested under `model`
   (`model/rfd3_base.yaml`: `- optimizers/adam@optimizer`), so this created an unused
   top-level branch.
2. **`model.optimizer.lr_scheduler`** — a path that *does not exist* in the stock config.
   The patch created it. The resolved config then showed two identical-looking AF3 blocks,
   the invented one at 1e-4 and the real one still at 1.8e-3, and a run launched that way
   trained at the stock rate. The trainer reads exactly one node:
   `instantiate(train_cfg.model.lr_scheduler, ...)` in `construct_scheduler`. Caught only by
   tracing that call.
3. Assigning a **partial dict** to the right path still broke it: `lr_scheduler` is filled by
   a defaults-list entry, so a partial assignment **replaces** the node — it resolved to
   exactly `{base_lr: 0.0001}`, dropping `_target_` and, worse, `warmup_steps`. Losing the
   warmup would have made divergence *more* likely. The block is now restated in full.

Verify an override landed by reading the **resolved** config at the path the code reads,
not by seeing the value appear somewhere in the tree.

**The diverged checkpoint was moved out of `ckpt/`** into `_diverged/` with a README,
because the training script auto-resumes from the newest checkpoint in `ckpt/` — leaving it
there would have silently continued a diverged run.

**Silent dataloader fallbacks:** epoch 0 logged **38 transform failures** over ~2,400
examples (~1.6%), absorbed by `n_fallback_retries: 4` and visible only by grepping. 36 were
`MaskPolymerResiduesWithUnresolvedFrameAtoms: operands could not be broadcast together with
shape (1,) (0,)` (all ProtFrag — an empty array reaching the frame-atom mask), and 2 were
`CropSpatialLikeAF3: No crop center found!`. Not fatal, but it means the arm trains on
slightly less data than the manifest claims, and the preflight's `n_fallback_retries: 0` is
what would surface them.

**A from-scratch run will not produce a usable model in this budget**, and that is not what
the plan asks for. 24 GPU-hours at effective batch 16 is ~4,200 optimizer steps / ~2 passes
over 33k examples — nothing for an AF3-class model from random init. The plan says to start
from the released RFD3 weights; this run's value is proving the data path and optimisation
are stable, not producing a model. Fetching those weights is the blocking item for a real
tranche.

### Released RFD3 weights: fetched, verified, and loading cleanly

```
$ZFS/lpgen/weights/rfd3_released/rfd3_foundry_2025_12_01_remapped.ckpt
  2,690,316,669 bytes   sha256 9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1
  from https://files.ipd.uw.edu/pub/rfd3/   (server Last-Modified 2025-12-03)
```

Fetched as a job with `curl -C -` (so a retry resumes) and a byte-count check against the
server's `content-length`, because a truncated 2.7 GB checkpoint would otherwise load
partially and look like a weight-name mismatch.

**What is in it:** `current_epoch = 570`, `global_step = 42825`, 1,616 tensors totalling
336,080,102 parameters, split `model.` (808) / `shadow.` (808) — the shadow half is the EMA
copy, so ~168M real parameters. It carries optimizer state, which our trainer skips
(`skip_optimizer_loading: true`).

**`current_epoch = 570` makes the epoch-stop trap live, not hypothetical.** Loading it sets
`current_epoch = 571`, so **`max_epochs` must exceed 571** or the run trains zero steps and
exits 0. The stock `100_000` is fine; this is why `lp_finetune.yaml` does not override it.
A preflight against this checkpoint *does* need the override (`trainer.max_epochs=572`),
since the preflight config sets `max_epochs: 1`.

**The weights load completely: 0 parameters fell back to REINIT.** This mattered to check,
because foundry's loader is `default_policy: COPY` / `fallback_policy: REINIT` and its
docstring says "all model weights are re-initialized; we overwrite with the checkpoint
weights where appropriate" — a name mismatch would load partially, silently, and the run
would look like a fine-tune while being close to scratch. Both the train and preflight jobs
now count those fallbacks automatically and print either "all parameters came from the
checkpoint -- a genuine fine-tune" or a loud breakdown of absent-vs-shape-mismatch. The
`model.`/`shadow.` prefixes line up exactly, which is what "remapped" in the filename means.

The weights are unmistakably live: against random init, `Seq Recovery` went 0.053 → 0.70–0.98
and `Token Lvl Sequence Loss` 3.36 → 0.04–2.3.

**But do not read `Seq Recovery` as sequence prediction.** Our LP condition *reveals* the
peptide sequence — all of it in the full-sequence task, 80–95% in the partial one — so
recovery is mostly measuring whether the model copies a sequence it was given. This project
has made that mistake before (`cond_recovery = 1.0` was only the copy test). The informative
quantity is recovery on the *hidden* residues of the partial task, which is not what this
metric reports.

`torch.load(weights_only=True)` is refused by this file, so inspection needed the full
unpickler. Acceptable for the IPD's own release, worth stating.

**The preflight is fragile against the ~1.6% bad ProtFrag examples**, by design: it sets
`n_fallback_retries: 0` so a failure is loud, and a run of a few dozen batches has a real
chance of drawing one and dying on
`MaskPolymerResiduesWithUnresolvedFrameAtoms: operands could not be broadcast ... (1,) (0,)`.
That is what killed the weight-verification preflight *after* it had already printed the
verdict. Training absorbs them (`n_fallback_retries: 4`). The defect is not a missing
frame atom: the failing structures have N/CA/C on every residue of both chains, so the
cause is still open and the manifest cannot yet gate it.

### Fine-tune running, and how to tell divergence from an outlier

`./scripts/submit_lpgen.sh --submit train --set TRAIN_RUN_NAME=lp_finetune_v1 --set
TRAIN_CKPT_PATH=<released ckpt>` — job 84072 on an a6000, initialized from the released
weights with 0 REINIT fallbacks, run dir on scratch, resumable by resubmitting the same name.

Robust (median) first-third vs last-third over the first ~36 logging intervals:

| metric | first third | last third | |
|---|---|---|---|
| Token Lvl Sequence Loss | 0.482 | 0.403 | improving |
| Seq Recovery | 0.934 | 0.943 | improving |
| **Mean Lddt Protein** | 0.220 | **0.233** | improving |

**`Mean Lddt Protein` is the metric to watch**, because it is the only one of the three our
conditioning does not hand the model for free. `Seq Recovery` near 0.94 largely reflects the
LP condition *revealing* the peptide sequence — the copy test, not prediction.

**A single-example outlier is not divergence, and a max-over-values sentinel cannot tell the
difference.** My first watcher flagged this run at seq loss 9.62. That value sits between
neighbours of 0.46 and 0.27, and recovery at that interval dipped to 0.135 before returning
to 0.96 — one hard example, at `batch_size 1`, where every logged value is essentially a
single example. Medians are unaffected (0.482 → 0.403) and all three metrics improve.

The replacement sentinel (`$ZFS/lpgen/probe/watch_train.sh <jobid> [threshold]
[consecutive]`) requires **N consecutive** intervals above the threshold. Validated against
both runs' real logs:

```
job 83990 (diverged): longest consecutive >6.0 = 37  -> FLAGGED
job 84072 (healthy) : longest consecutive >6.0 =  1  -> not flagged
```

Which is the general point: the genuine divergence was a *monotonic climb over ~30
intervals*, not a spike. Judge these series by trend and by robust statistics, not by
extremes — and prefer medians, because at batch size 1 the mean is outlier-dominated too
(the mean first-third/last-third comparison on this same healthy run reads 0.734 → 1.352,
i.e. "getting worse", purely from that one batch).

### What two epochs of fine-tuning actually show: nothing measurable yet

Job 84072, first checkpoint `epoch-0572.ckpt` written (even-epoch cadence from the odd start
at 571, as predicted). Quartile medians over 2 epochs (~4,800 examples, **0.15 of one pass**
over the 33,089 train rows):

| metric | Q1 | Q2 | Q3 | Q4 | verdict |
|---|---|---|---|---|---|
| Total Loss | 0.298 | 0.276 | 0.259 | 0.271 | no trend (0.9 SE) |
| **Mse Loss Mean** | 0.0080 | 0.0098 | 0.0072 | 0.0094 | **flat (0.0 SE)** |
| Mse Loss High T | 0.0052 | 0.0050 | 0.0041 | 0.0043 | possibly down |
| Token Lvl Sequence Loss | 0.339 | 0.080 | 0.044 | 0.042 | improved, then **plateaued** |
| Mean Lddt Protein | 0.226 | 0.227 | 0.222 | 0.225 | flat (0.08 SE) |

**The only thing that moved is the copy task.** Our LP condition reveals the peptide
sequence, so `Token Lvl Sequence Loss` falling to 0.04 and `Seq Recovery` rising to 0.988
measure how well the model reproduces a sequence it was handed — and that saturated within
the first half-epoch. Every metric that is not free is flat within noise.

**The loss weighting is not the explanation.** I assumed the sequence term might be eating
the gradient; it is not. `diffusion_loss.weight: 4.0` against `sequence_loss.weight: 0.1`,
and at current values the weighted contributions are structure 4.0 x 0.010 = 0.040 versus
sequence 0.1 x 0.047 = 0.005 — structure dominates ~8.5x. The hypothesis was wrong.

Three readings remain open, and the training metrics cannot separate them:

1. **Far too early.** 0.15 of a pass. Nothing should be expected yet.
2. **Little headroom on the objective.** The model was pretrained at PDB scale; an MSE of
   0.01 on 33k peptide-protein complexes may simply mean it already fits them, in which case
   the fine-tune's value would appear in *generation*, never in training loss.
3. The fine-tune genuinely is not learning anything useful for this task.

**Which is why the missing validation matters more than the missing epochs.** `lDDT` as
logged is computed at a randomly sampled noise level — per-interval sd 0.05 against a mean
of 0.24 — so it is a weak instrument even with ~100 samples per half. The quantity that
would discriminate is sampled generation on the held-out panel, and that is exactly what is
not wired (see the note on what RFD3 validation is). Continuing to train has real but
limited information value until it exists: this project's own history is that teacher-forced
training metrics were what made "the model works" unsupportable on CPSea.

### In-loop generative validation WORKS — and says the fine-tune is mildly HARMFUL

Job 84270 (resumed from `epoch-0574.ckpt`), 12 epochs at effective batch 16, validating the
full 64-spec panel every 2 epochs. Six validation passes, metrics and structures both saved:
`val_metrics/validation_output_all_epochs.csv` (384 rows) and
`val_structures/predictions/epoch_NNN/lp_panel/*.cif.gz` with per-model JSON.

Paired comparison, same 64 specs at epoch 576 vs 586:

| metric | delta | significance | direction |
|---|---|---|---|
| `radius_of_gyration` | **+0.381 Å** | 7.2 SE | worse (expanding) |
| `fraction_chainbreaks` | +0.00083 | 4.0 SE | worse |
| `mean_valid_sidechain_percent` | −0.00309 | 3.2 SE | worse |
| `frac_clashing` | +0.00074 | 2.6 SE | worse |
| `mean_clash_percent` | +0.00026 | 0.6 SE | noise |

**Four of five are significantly worse, all in the same direction, on a paired design.** The
absolute sizes are small (clash fraction 0.04% -> 0.11%) but the monotone Rg growth at 7.2 SE
is hard to dismiss: the generated peptides are expanding, which is what drifting out of a
compact bound pose looks like. Both tasks (`fullseq` and `partialseq`) show the same pattern.

A file-corruption confound was ruled out before drawing this conclusion: a newer note warns
that DiffPepBuilder's preprocessing rewrites input PDBs **in place** and that RFD3 reads the
same files. Checked — every sampled training PDB and all 32 panel PDBs still carry their
2026-09-08 preprocessing mtime, so nothing was rewritten under the running job.

**But these metrics cannot settle the question, and should not be read as "fine-tuning is
bad".** They score the whole predicted structure, which is 550 receptor atoms against 64
peptide atoms, and they measure *physical plausibility* — clash-free, connected, chemically
valid — not whether the peptide is in the right place. None of the 29 columns reports
distance to the native pose, pocket occupancy, or interface contact recovery. It is entirely
possible for plausibility to drift slightly while binding accuracy improves, or vice versa.
`helix_fraction` is 0.0000 at every epoch, which for 5-16mers is expected rather than
alarming (the P-SEA survey of our own panel found short peptides essentially all coil).

So the next measurement is peptide-specific scoring of the dumped CIFs against the natives,
and it is cheap: the structures for all six epochs are already on disk, so it is offline CPU
work that can be done retroactively and does not disturb training. This repository already
has the pieces — interface RMSD superseding whole-peptide RMSD, the clash threshold
calibrated against native crystals at 2.0 Å, and the Rosetta dG sidecar.

### Scoring the generations against natives — and four silent corrupters

`script_utils/lpgen_score_val_structures.py` reads the dumped `.cif.gz` files and measures
accuracy, not plausibility. It is offline CPU work and retroactive: every epoch's structures
are already on disk, so no retraining is needed to score an earlier checkpoint.

**Metric names and the 2.0 Å interface-hit threshold match `lpgen_dpb_score.py`**, because
arms that are scored differently cannot be compared. Adopting its decomposition:
`raw_ca_rmsd` (native frame: placement *and* conformation), `superposed_ca_rmsd` (peptide-only
Kabsch: conformation alone), and the difference between them is the placement error — "wrong
pose" versus "wrong fold". One RFD3-specific wrinkle: its *raw* frame is **not** the native
one, so raw requires the receptor superposition below; a docking arm's poses already are.

Four things, each measured on real output, each of which yields a plausible-looking wrong
number rather than an error:

1. **Residue ids are off by one.** Generated CIFs are 1-based (A 1..74, B 1..6); our natives
   are 0-based (0..73, 0..5). Matching on `res_id` aligns the wrong residues and scores a
   minority of atoms — it matched 383 of 614 and produced a 4.06 Å "receptor RMSD" for a
   receptor that is in fact rigidly fixed.
2. **Sidechains are renamed to virtual atoms:** `V0, V1, V2...` where the native has
   `CG, CD, CE...`, because the trainer sets `cleanup_virtual_atoms: False`. Sidechains cannot
   be matched by name at all, so scoring is backbone-level (`N, CA, C, O, CB`). Residue
   *names* do match, which is what makes sequence identity checkable.
3. **The output is recentred.** The receptor is rigidly fixed by the contig spec, yet its raw
   deviation from the native is a near-uniform 1.88 Å (mean ≈ max) — RFD3 logs "Setting origin
   as COM of fixed motif". Superposing on the receptor drops it to **0.019 Å**, confirming
   rigidity. Skip that step and a peptide reads ~12 Å when ~10 Å of it is global offset.
4. **Atom order differs** between generated and native, so no element-wise comparison is
   valid even though the counts match exactly (550 receptor / 64 peptide in both).

`receptor_rmsd_after_fit` is therefore a per-row QC gate: it must be ~0, and a larger value
means that row's peptide RMSD is meaningless rather than merely bad.

### The two arms are not yet comparable

`lpgen_dpb_score.py` has already produced a DiffPepBuilder k=0 baseline:

```
80 LNR targets, 640 poses
raw CA 5.63 A | superposed CA 2.32 A | interface raw 5.60 A
best-of-N raw 5.33 A | poses <= 2.0 A interface 0.066 | targets any hit 0.113
```

That is the **80-target LNR set**. RFD3 validation runs on the **32-complex LP panel** drawn
from our own `val` split. Same metrics, different populations — so the numbers must not be
put side by side until one common evaluation set is chosen. Picking it is a decision worth
making before any arm number is quoted, not after.

### POTFlow repairs, verified against the code

The repo is `github.com/CMACH508/potflow` (not findable by search; no `EDAPINENUT/potflow`,
and PepFlow's single config contains neither named key). Every repair the plan lists is
confirmed in the source. Three are more severe than the plan describes:

**`--debug` cannot be turned off, and the run crashes.** `train.py:29` is
`add_argument('--debug', action='store_true', default=True)` — with `store_true` *and*
`default=True`, the flag is `True` whether or not it is passed, so there is no CLI route to
`False`. The `else` branch (`train.py:55–67`) that assigns `log_dir` and creates `ckpt_dir`
therefore never runs, while the training loop unconditionally does
`os.path.join(ckpt_dir, '%d.pt' % it)` every `val_freq` iterations (`train.py:199`). An
ordinary `python train.py` does not silently skip checkpointing — it dies with
`NameError: ckpt_dir`, and only after `val_freq` iterations of real training have been
spent. Fix the default to `False` *and* keep `store_true`.

**`validate()` is defined and never called.** `def validate` at `train.py:169` is the only
occurrence of the string in the file. There is no validation anywhere; `val_freq` gates
checkpointing only, and `'avg_val_loss'` is commented out in the saved dict — so "best
checkpoint" is not merely unimplemented, it is uncomputable as shipped. The loop also saves
only on `it % val_freq == 0`, so unless `max_iters` is an exact multiple the final state is
lost. All three of the plan's asks here (call validate, save latest/best, save final) are
genuinely needed.

**The all-helix DSSP fallback is a bare `except:`.** `pep_dataloader.py` wraps the DSSP call
in `except: pep_dssp = None`, and the `else` branch then does
`ss = ss + ['H']*generate_mask.sum()` with no log line. A bare except swallows everything,
including a missing DSSP binary — so if DSSP is simply not installed, **100% of the training
set silently becomes all-helix** and the secondary-structure prior, which is POTFlow's
distinguishing feature, trains against a constant. The nearby
`ss = ss + [ss[-1]]*(...)` also fabricates labels by repeating the last one when DSSP comes
back short. Replace with an explicit loop/unknown policy and log the count, as planned.

Confirmed as described:

- **Interpolant flags.** `learn_angle.yaml` has `sample_trans_prior: False` and
  `ot_sampler: False`. Worth more than the plan claims, though: `test_angle.yaml`
  (inference) sets **both `True`**. The shipped defaults therefore train under one
  interpolant and sample under another, so enabling both for training *aligns* train with
  the inference config rather than being an arbitrary choice — a stronger reason than "an
  arm with them disabled is an ablation".

| config | `sample_trans_prior` | `ot_sampler` |
|---|---|---|
| `learn_angle.yaml` | False | False |
| `learn_angle_partial_t.yaml` | False | False |
| `learn_foldx.yaml` | False | True |
| `learn_ss.yaml` | True | — |
| `learn_gmm_prior.yaml` | True | — |
| `test_angle.yaml` (inference) | True | True |

- **Dataset membership is a directory listing.** `pep_dataloader.py:134` is
  `all_pdbs = os.listdir(self.structure_dir)`, with the cache at
  `f'{self.name}_structure_cache.lmdb'`. So `name` picks the cache *filename* and has no
  effect on membership — changing it cannot enforce a different split, exactly as the plan
  says. Default `structure_dir` is `./Data/PepMerge_new/`; `PepMerge` (10,348 entries) is
  already on disk. Membership must become manifest-driven off `lpgen_manifest.parquet`.

One extra hazard not in the plan: the repo ships **two copies** of the same module, at
`potflow/utils/data.py` and `utils/data.py` (both defining `SS2SEG`). Which one is imported
depends on `sys.path` order, so a repair applied to one may silently not take effect. Pin
the import before editing either.

## The first validation curve measured nothing, and why

The RFD3 arm produced six validation epochs (576–586, 512 models each). **Both halves of that
curve — pose and sequence — were read through broken instruments.** Recorded here because both
failures are silent and both would have been quoted as model results.

### The pose half: `ori_token`

`ori_token` is not a feature or a hint. It is the choice of coordinate **origin**: `set_com`
does `atom_array.coord = atom_array.coord - center`, so the structure is translated until that
point is (0,0,0), and because the diffusion prior is centred on the origin, **that is where the
generated peptide materialises.** It appears only in rfd3's *inference* code, never in a
training transform.

Supplying nothing is not neutral. rfd3 then logs "Setting origin as COM of fixed motif" and
centres on the **receptor** — a median **17.8 Å** from the real binding site on this panel — so
every peptide condensed inside the protein. Raw CA RMSD ~21 Å with a superposed RMSD of ~3.1 Å
is the signature: right fold, wrong place.

**Scope, which matters more than the bug: training was never affected.** `lpgen_base.yaml` sets
`center_option: diffuse`, which resolves to `center_mask = ~is_motif_atom_with_fixed_coord`,
i.e. the **peptide**. Training always put the peptide at the origin. Only validation fell back
to the receptor centroid, so **the checkpoints are sound and fixing the measurement costs no
retraining.** A corollary: the one monotonic trend in the broken curve — clashes *and* contacts
both rising at p≈1e-6 — is not the model learning to interpenetrate the protein core. It is
what a model does when its prior is dropped inside the receptor. Do not read it as pathology.

The replacement origin is **calibrated, not inherited** (distance to the native peptide COM,
32 panel complexes; pocket = receptor atoms within `POCKET_CUTOFF_A` = 10 Å of the peptide):

| candidate | median | max |
|---|---|---|
| receptor centroid (the broken default) | 17.83 | 24.95 |
| pocket centroid, no push | 6.89 | 16.66 |
| **pocket + 6 Å outward — chosen** | **3.27** | 12.92 |
| pocket + 10 Å (rfd3's own stock value) | 5.23 | 11.54 |

rfd3's stock 10 Å **overshoots** on this panel. The pocket-derived origin is computed from
**receptor atoms only**, so the peptide's pose is never fed back in; the peptide only selects
*which* receptor atoms line the pocket, which is the same information a user supplies by naming
hotspots, and the plan's declared inference input is "target/pocket plus CP sequence".
`--ori-mode native_pep` is retained but is a pose-reconstruction learning check **only** — it
uses the answer's location and must never be quoted as site placement. The builder refuses a
degenerate direction (pocket centroid coincident with the receptor core) rather than emitting
NaNs that rfd3 would silently translate the whole structure by.

Note also that this arm trains with `add_ppi_hotspots: 0.0`, so it has **no learned
site-conditioning channel at all**. It can do "pose given site"; it can never honestly be asked
to *find* a site.

### The sequence half: the metric scored a copy

`SequenceLoss` weights both its cross-entropy and its `seq_recovery` by
`loss_input["sequence_valid_mask"]`, and `AddGroundTruthSequence` builds that mask as
`~np.isin(res_names, ["UNK","X","DX","<G>"])` — it excludes only *unknown* residues, **not** the
`is_motif_atom_with_fixed_seq` positions. But `EncodeAF3TokenLevelFeatures` writes the **true
residue one-hot into the input features** at exactly those positions. So the metric scores
positions whose answer the model was handed.

Measured over the 33,089 training rows under our real reveal schedule: median 121 tokens scored
per example, **51.4% of examples hide zero identities**, and hidden tokens are **0.54% of all
scored tokens**. `Seq Recovery ≈ 1.0` is therefore 99.46% copy. It remains a useful *tripwire* —
it is what caught the LR divergence, collapsing to 0.02–0.08 — but it is not a capability.

Splitting it (`--contigs` in `lpgen_score_val_structures.py`) gives the real numbers:

| | value | reading |
|---|---|---|
| given positions | **1.0000**, 0 of 400 wrong | copy check: conditioning IS honoured |
| hidden positions | **4.95%** | vs **9.7%** "always guess leucine" and 5.0% uniform |

**The sequence head is at chance.** (An earlier 8.3% figure was inferred from pooled identity
assuming one hidden position per spec; the contigs actually hide 36 positions across 32 specs,
so the direct split supersedes it.)

Supervision *is* present — the masking operates on `atom_array[token_starts]`, a fancy-index
**copy**, so `AddGroundTruthSequence` still reads true residues. Chance accuracy is a real
result, not a no-supervision artifact. The cause is more likely dosage: the schedule hides a
mean of **0.74 positions per example**, so `reveal_frac_min` (0.80) needs widening before any
sequence capability is a fair expectation.

### Where sequence can and cannot be constrained

The **predicted** sequence never reaches the geometry. `forward_with_recycle` carries only
`D_II_self` and `X_L_self`; `S_I` comes from the input features and is never overwritten by a
prediction; and the sampler's only consumer of `sequence_logits_I` is an entropy trajectory for
logging, while the coordinate update `X_L = X_noisy_L + step_scale*d_t*delta_L` never reads it.
**So biasing output logits is inert for design** — it changes the letters, not the pose. Every
sequence constraint that should affect structure must enter on the **input** side.

That input is binary per position: exact one-hot, or `GAP`. A simplex vector is *shape*-valid,
since `class EmbeddingLayer(nn.Linear)` is a linear map over the float one-hot rather than an
`nn.Embedding` lookup, so it yields the probability-weighted average of residue embeddings — but
it is off-distribution, because training saw only vertices. **Trap: `GAP` is a learned 21st
token, not the flat distribution, so "certain → masked" cannot be annealed by flattening the
one-hot**; flattening walks toward the amino-acid centroid, which means nothing to the model.
The in-distribution options are choosing *which* positions to open, or ensembling over masks and
filtering. Leave-one-out masking yields marginals that do **not** compose into a joint sequence;
mask jointly to design.

### Re-scoring saved checkpoints without retraining

Since training was sound, the eight saved checkpoints can be re-measured directly. The mechanism
is the **epoch-stop trap, used on purpose**: loading a checkpoint does `current_epoch += 1` and
then sets `should_stop` if `current_epoch >= max_epochs`
(`foundry/trainers/fabric.py:429-435`) — normally the reason a small `max_epochs` makes a
fine-tune train zero steps and exit 0. But `prevalidate` runs at **:447, before** the
`while not self.should_stop` loop at :459. So `ckpt_path=<ckpt> trainer.prevalidate=True
trainer.max_epochs=1` validates exactly once and never trains, and prevalidate decrements
`current_epoch` so dumps land in `epoch_<N>/` matching the checkpoint.

    ./scripts/submit_lpgen.sh --submit reval          # all 8 checkpoints + the scorer
    REVAL_LAST_N=1 ./scripts/submit_lpgen.sh --submit reval   # newest only, as a smoke

`scripts/lpgen_rfd3_reval.sbatch` runs one checkpoint per job, each with its own
`paths.output_dir` (a shared tree would mean concurrent appends to one `metrics.csv`, which is
measured to NUL-corrupt rows here), and asserts afterwards that no training epoch started and
that prevalidation actually ran. `lpgen_reval_score.sbatch` is a CPU sibling behind them on
`afterany` — not `afterok`, so one checkpoint that OOMs cannot strand the scorer and leave the
rest unexplained — and it writes one CSV per checkpoint plus an assembled `reval_curve.csv`.

**Write corrected contigs to a new path, never in place.** A running job holds the old JSON, and
rewriting it would put two ORI conventions into a single curve. Hence
`lpgen_val_contigs_pocket.json` alongside the original, with the builder logging the ORI mode,
the push, and a per-mode count, and warning loudly on any fallback.

## Next

1. **RFD3 (lead) — TRAINING NOW.** Staged, installed, preflighted, and the first tranche
   is running (job 83990 on gpu31, `--submit train`). Watch two things in its `.out`: the
   effective LR on the first logged step, because a fine-tune inherits the released
   checkpoint's schedule position and EMA (neither is resettable); and the count of
   `Transforms failed at stage MaskPolymerResiduesWithUnresolvedFrameAtoms` lines, which
   atomworks swallows via fallback retries — **26 examples in the first ~975 batches**, so
   the run will not fail, but it is the silent-subset-shrink mode this manifest exists to
   prevent and it needs diagnosing rather than riding.
2. **DiffPepBuilder (second, and the priority arm).** Staged at `c19eb4f`; weights
   downloading; env recipe written. The next step is deliberately **not** training — it is
   the gating measurement, which needs no training at all: run DiffPepDock at *k=0* (CP
   sequence revealed, `mask_lig_seq: False`) on our LNR set to get the zero-mutation
   baseline the plan requires, then sweep *k* by pinning identities per position and
   measuring **realized similarity and Rosetta ΔG against *k***. That produces the
   similarity-vs-affinity curve directly from released weights. Only then decide whether a
   random-reveal-fraction fine-tune is worth a tranche.
3. **UniMoMo (calibration only).** Reproduce its published LNR peptide numbers with
   `model.ckpt` to get a published bar on our exact data. Do not build the CP pipeline on
   it; it cannot be given a sequence.
4. **PPFlow / PepGLAD (off the critical path).** PPFlow: point `data_dir` at the native
   export and `include_pdb_path` at `{split}_names.pt`; separate its `--finetune` branch
   from resumption. PepGLAD is superseded by UniMoMo, its own authors' successor.
5. **POTFlow (parked).** Patch set is ready via `lpgen_patch_upstream.py`. Remaining work is
   an environment (it declares none): `lmdb`, `easydict`, `pot`, `torchcfm` are the four on
   the training path; `deepspeed`/`dllogger`/`openmm`/`pdbfixer`/`simtk`/`e3nn`/`esm` are
   vendored-openfold internals or alternate models and are likely not needed.
6. Per-arm preflight: one real minibatch forward/backward/step, checkpoint save+reload, two
   complete exported samples, and recorded peak memory and step throughput.

Training-loss values are not comparable across these architectures, and a low LP
reconstruction loss does not establish success from a CP sequence.
