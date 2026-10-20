# Sequence → target → cyclic-peptide outcome dataset (`cpseq`)

Build log for the first tranche of
`(candidate sequence, target context, cyclization chemistry) → predicted CP–target
complexes → validity / interface / strain / uncertainty labels`.

Started 2026-10-19. This file records what was **verified by inspection or
execution**, not what was planned. Anything unverified says so.

---

## Session constraint that shapes the schedule

The VS Code session runs inside a `node-hold` Slurm job (2 CPUs, 32 GB, 1 GPU,
A6000). Job `10859131` on `babel-t9-32` had 24 minutes left when this work
started, so nothing heavy ran in-session: every expensive stage is an `sbatch`
job that survives the session. `preempt` is `PreemptMode=REQUEUE`, so each job is
written idempotent and resumable.

---

## Predictor arms: availability as measured

| Arm | Source pin | Env | Weights | Status |
|---|---|---|---|---|
| Boltz-2 | `upstream/boltz` `b1ebfc46` (2026-05-29) | own venv, **building** (job 10860559) | `boltz_cache/` 6.2 GB present: `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `ccd.pkl`, `mols.tar` | env build queued |
| AfCycDesign / cyclic ColabDesign | `community_models/colabdesign` (installed in complexa venv) | complexa venv — `colabdesign`, `alphafold`, `jax`, `haiku` all import | AF2 params present (`params_model_{1,2}{,_ptm,_multimer_v3}.npz`, 5.3 GB) | **ready**; repo already carries the cyclic encoding (below) |
| HighFold3 | `upstream/highfold3` `2ff1ea31` (2025-09-24) | not built | AF3 `af3.bin` 1.1 GB present | blocked on an AF3 install + a **CPython 3.11** interpreter (below) |

### Boltz-2

Needs its own venv: it pins `numpy>=1.26,<2.0`, `einx==0.3.0`, `dm-tree==0.1.8`
and its own torch, while the complexa venv runs numpy ≥ 2 and torch 2.7.0+cu126
with openfold/atomworks on top. Resolving them together would downgrade one side,
so the adapter shells out to a separate interpreter —
`scripts/cpseq_env_boltz2.sbatch`, installing non-editable from the pinned clone
so a later `git checkout` in `upstream/` cannot silently change what runs.

`BOLTZ_CACHE` must point at the cache dir and contain `mols/` extracted from
`mols.tar`; the env job does that extraction (checked idempotent).

### AfCycDesign — the cyclic encoding is already in this repo

`src/proteinfoundation/evaluation/cyclic_offset.py` already implements the
AfCycDesign construction (wrap the pairwise sequence offset around the ring for
the binder block only) and is already wired into
`src/proteinfoundation/rewards/alphafold2_reward.py` behind a `cyclic_offset`
flag, with `cyclic_offset_applied` recorded per call. It restricts itself to
`mainchain` via `CYCLIC_OFFSET_SUPPORTED_TYPES = {"mainchain"}` and says why:
wrapping the **backbone** offset asserts that residues 0 and L−1 are backbone
neighbours, which is true only when the closing bond is the peptide bond itself.
A disulfide or isopeptide ring closes through side chains, so the wrap would
state something false about the molecule.

This matches the task's constraint that AfCycDesign's head-to-tail positional
encoding must not be advertised as support for arbitrary crosslinks. The adapter
work here is therefore *fixed-sequence complex prediction* (design/hallucination
off), not re-deriving the encoding.

### HighFold3 — two verified constraints

**1. The cyclic module ships only as compiled bytecode.** HighFold3's topology
control lives in `model/cyclic_peptide_matrix` (`calc_cyclic_matrix_af3`, called
from `model/features.py:975,982` for both the residue-wise and token-wise
matrices). The repository contains **only**
`model/__pycache__/cyclic_peptide_matrix.cpython-311.pyc` — there is no
`model/cyclic_peptide_matrix.py`. Verified with `git ls-files model` on the
pinned clone: the `.pyc` is a tracked file and the `.py` is absent.

Consequences, both real:
- the arm is pinned to **CPython 3.11 exactly** (a `.pyc` only loads under the
  interpreter whose magic number it was built with — our complexa venv is 3.12);
- the CycPOEM construction cannot be read as source, so what the flags do must be
  established behaviourally rather than by reading the implementation.

This is a constraint, not a dead end: a 3.11 venv plus an AF3 install can run it.

**2. Documented topology controls** (`README.md`, and `run_alphafold.py:63,74`,
`:829`):
- `--head_to_tail` — bool, "whether the cyclization is in the form of
  head_to_tail";
- `--disulfide_chain_res [[1,3,11]]` — `[[chain, Cys_i, Cys_j]]`.

`head_to_tail` is threaded as a single global bool from the flag through
`featurisation.py:45` → `pipeline.py:156` → `features.py:914` into
`calc_cyclic_matrix_af3(residue_index, head_to_tail=..., list_cid_ss=...)`. It is
**not** given a chain id, while the disulfide control **is** (`[[chain, i, j]]`).
Since `calc_cyclic_matrix_af3` receives only `residue_index` for the head-to-tail
case, which chain the ring is applied to must be determined by that function from
where `res_id` restarts — i.e. by chain-boundary inference, not by an argument.
**Which chain it actually affects is therefore still unverified** and is exactly
the thing the task says to check: it needs a behavioural test (predict a
2-chain complex, read back the matrix the module prints — `features.py` already
`print`s both matrices) before any HighFold3 number enters the dataset.

AF3's genetic databases do not fit the quota (see
`docs/RESOURCES_COLLECTED.md`); AF3 accepts precomputed alignments inline, so
MSAs get computed once per target context and cached.

---

## Chemistry support, as implemented in this project today

The project's own closure chemistries are `mainchain` / `disulfide` /
`isopeptide` (`src/proteinfoundation/cyclization/constants.py`), with
`UNSPECIFIED` and `LINEAR` as conditioning-only indices.

Atom-level bonds already pinned in the codebase:

| Chemistry | Closing bond | Reference geometry (`cyclization/linkage_geometry.py`) |
|---|---|---|
| `mainchain` | `C` of last residue → `N` of first | angles 116.2° / 121.7°, ω ≈ 180° (Engh & Huber) |
| `disulfide` | `SG`–`SG` | CB-SG-SG 103.8°, CB-SG-SG-CB ≈ ±90°, **sign-symmetric** (both handednesses are real) |
| `isopeptide` | Lys `NZ` → side-chain carbonyl C (`CG` Asp/Asn, `CD` Glu/Gln) | angles 120° / 117°, planar amide ω ≈ 180° |

**Scorer support is narrower than the dataset's chemistry vocabulary.**
`src/proteinfoundation/evaluation/rosetta_energy.py` declares the closing bond on
the pose before relax/scoring for `mainchain` (strip terminus variants, then
`declare_chemical_bond(i,"N",j,"C")`) and `disulfide` (`add_disulfide_bond`), and
**refuses isopeptide** — an arbitrary Lys-NZ↔Asp/Glu bond needs a residue-
connection patch plain amino-acid types do not carry. It reports this as
`binder_rosetta_cyclic_topology_declared=False` rather than scoring a ring that
was silently left open.

So the common supported chemistry across evaluator *and* all three predictors is
**head-to-tail `mainchain` amide on canonical residues**, which is also the
chemistry the existing CP workflow uses. `disulfide` is a separate labeled
subset (supported by Rosetta and by HighFold3's explicit control; **not**
supported by the AfCycDesign offset wrap, and in Boltz the head-to-tail cyclic
flag is *not* a substitute for it). `isopeptide` is unsupported by the shared
evaluator and is marked unsupported rather than scored.

---

## Reusable pieces found in the repo (do not rewrite)

| Need | Existing code |
|---|---|
| Rosetta interface metrics with cyclic bond declared | `evaluation/rosetta_energy.py` — `InterfaceAnalyzerMover`, `pack_separated=True`, `dG_separated`, `dSASA_int`, `delta_unsatHbonds`, `hbonds_int`, `sc_value`, `packstat`, plus `*_norelax` diagnostics so a failed relax stays visible |
| Closure geometry measurement from a PDB | `evaluation/cyclization_pdb_metrics.py` |
| Chemistry-specific geometry tolerances | `script_utils/derive_cyclic_thresholds.py` |
| AF2 cyclic refold + confidence | `rewards/alphafold2_reward.py` (+ `evaluation/cyclic_offset.py`) |
| Target staging into chain A = receptor / B = binder | `scripts/build_lnr_metadata.py` (coordinates never transformed) |
| Target-family split material | PepBench `train.cluster` (4,157) / `valid.cluster` (114) — a pre-existing clustering, so no need to invent one |
| LP references + targets | `/data/user_data/yixiz/LNR` — `test.txt` 93 targets, 96 PDBs (development data) |
| Slurm conventions | positional **env-file** config (never `sbatch --export`), shared `_*_preamble.sh`, per-shard `rows_<task>.jsonl` so there is one writer per file, `--open-mode=append`, logs under `/home` |

PyRosetta imports in the complexa venv (checked), so the shared evaluator runs
there as CPU work.

---

## First build (config_hash `4207a3085a004bba`)

`python script_utils/cpseq_build_dataset.py --out-dir $ZFS/cpseq/tranche1 --n-parents 12`

| | |
|---|---|
| LNR entries parsed | 93 |
| parent contexts in the 5–16mer ring range | **63** (30 rejected: 3–4mers and 17–27mers, each with its measured length as the reason) |
| receptor families | **54**, 5 of them multi-parent |
| split | `dev` whole — see below |
| parents selected for the tranche | **12**, one per family, lengths 5,5,6,6,7,8,9,11,11,12,14,16 |
| mainchain candidates (cross-model set) | **107** |
| disulfide candidates (labeled subset) | **58** |
| minimum identity to reference | **0.500** |

Mainchain is 107 and not 12 × 9 because one parent's bands coincide onto an
identical sequence; it merges and carries both band tags. Disulfide is smaller by
construction — two CYS anchors are charged to the budget first, so the low bands
have nothing left to spend and collapse into the reference arm.

Counts are capped at `floor(L/2)`: rounding the 50 % band up at an odd length
gives 7 of 13, which is 46 % identity and outside the retention constraint. The
cap binds only at the top band and odd lengths, and says so in `note`.

### Missing generalization coverage, stated plainly

LNR goes to `dev` **whole**. It has already shaped this repository's
preprocessing and sampling decisions, so a held-out slice of it would not be an
independent test set. PepBench's `train.cluster` / `valid.cluster` (4,157 / 114)
are staged and are a pre-existing clustering that could supply disjoint target
families for a future evaluation split; until those are built into contexts,
**this tranche supports no generalization claim across target families.**

### Tool paths in `.env` that do not resolve

`MMSEQS_EXEC` and `FOLDSEEK_EXEC` both point into the complexa venv's `bin`, and
neither binary exists there. `env.sh` exports both regardless. The family split
has to exist *before* variants are generated, so `contexts.assign_families` falls
back to a built-in pairwise clustering and records which path ran in
`clustering_method.json`.

The first fallback put all 63 receptors in **one** family: global alignment
scatters chance matches across a long receptor, normalising by the shorter
sequence then clears 0.3 for unrelated pairs, and single linkage chains every
family together from the first false edge. Local alignment with identity over the
aligned columns — what mmseqs reports — gives the 54 families above.

---

## Status

- [x] Predictor availability measured; two HighFold3 constraints verified
- [x] Chemistry capability table grounded in this repo's own code, checked against
      `cyclization.constants`
- [x] Candidate builder (bands, seeds, anchor accounting, dedup, identity floor)
- [x] Parent contexts, receptor families, split — built **before** variants
- [x] First skeleton build: 12 parents, 165 candidates
- [x] Boltz-2 adapter: inputs verified for both chemistries
- [ ] Boltz-2 env (job `10860559`, still `PENDING` on `preempt`)
- [ ] AfCycDesign adapter (encoding already in repo; needs fixed-sequence complex path)
- [ ] HighFold3 / AF3 env (needs CPython 3.11 + AF3 install; MSAs precomputed)
- [ ] Target-only templates + cached receptor MSAs per parent
- [ ] Shared chemistry-aware evaluator
- [ ] Per-predictor end-to-end preflight on shared cases
- [ ] Bounded first tranche (cap: 24 GPU-h, 300 CPU core-h)
- [ ] ~6 experimental CP–target complexes as technical controls
