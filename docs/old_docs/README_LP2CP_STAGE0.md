# LP → CP Stage 0: inventory and cost profile

Stage 0 of the LP → CP plan asks two things before any new training is committed: does the
existing system still exist and run, and what does one structure cost. This document records
the read-only inventory (verified 2026-09-17 against `8d6ac02`) and describes the job chain
that measures the cost.

## 1. Inventory — what is actually on disk

Everything the plan lists as "may need downloading" is already staged. No new download is
required to run the Stage 0 profile or the Stage 1 feasibility tranche.

| Resource | Path | State |
|---|---|---|
| CP flow checkpoint (pin) | `store/cpsea_bondunroll_pin20260828/checkpoints/last-EMA.ckpt` | present, `global_step=46500` |
| CP autoencoder | `$ZFS/Proteina-Complexa/training_runs/store/finetune_full_128/checkpoints/chk_epoch=00000002_step=000000040000-EMA.ckpt` | present, 4.1 GB |
| Base (pre-CPSea) checkpoints | `$ZFS/Proteina-Complexa/ckpts/complexa{,_ae}.ckpt` | present |
| Preprocessed CPSea | `$ZFS/CPSea/CPSea_full/CPSea/preprocessed/metadata/cpsea_{train,val,test}.parquet` | present: 2,444,209 / 134,394 / 135,811 rows |
| LNR (raw) | `$ZFS/LNR/pdbs` + `test.txt` | present |
| LNR (staged, 3 crops) | `CPSea_data/lnr_{staged,pocket,pocket14}/metadata/lnr_test_pocket.parquet` | 60 rows each, all input PDBs resolve |
| MEET validation | `$ZFS/LPData/MEET_valid/{index.txt,data.bin}` | present, 100 entries |
| PepBench | `$ZFS/LPData/PepBench/train_valid/` | present: 6,142 PDBs, 4,157 train / 114 valid ids |
| ProtFrag | `$ZFS/LPData/ProtFrag/` | present |
| Scoring environment | `.venv/bin/python`, PyRosetta | imports cleanly |

The AE pin is consistent across every flow run that matters: `cpsea_bondunroll_pin20260828`,
`cpsea_seqcond_from_v4cfg` and `cpsea_seqcond_v2` all record the same
`autoencoder_ckpt_path`, so their losses and closure numbers are mutually comparable. Keep
`CPSEA_AE_CKPT_PATH` pinned to it; `env.sh` already defaults to exactly this file.

### Three findings that change the plan

**1. There is no production sequence-conditioned checkpoint.** The plan's first action is
"locate the actual successful sequence-conditioned checkpoint". It does not exist at a usable
scale: `cpsea_seqcond_from_v4cfg` stopped at `global_step=6000` and `cpsea_seqcond_v2` at
`global_step=3000`, both against a 46,500-step warm start. The mechanism is fully wired —
`training.sequence_conditioning` draws a per-residue reveal, `val_generation.seq_cond` runs a
conditioned ODE, and `use_residue_type_feature` accepts a `[b, n]` mask end to end — but
compliance was only ever measured at `keep_frac=1.0`, which is the copy test. **Compliance
under a partial reveal is unmeasured.** That matters directly, because a partial reveal is the
natural encoding of a mutation budget: reveal the protected positions, leave the budget
positions free. The mechanism is the right one; the evidence that the model honours it is not
there yet.

**2. A hard mutation budget is not currently enforceable, only measurable.** The SDEdit path
is explicit about this by design: substitutions are an *outcome* of the latent-track noise
level `t_lat`, not a constraint, because pre-mutating a residue would hand the AE an input it
never saw. `n_substitutions` is recorded per row, so the budget can be measured and filtered
post hoc — but "minimize the loss subject to at most K substitutions" is a rejection sampler
today, not a constrained search. Closing that gap is the same work as item 1: sequence
conditioning is the only channel that can *request* specific identities at specific positions.
Note also that `scripts/sdedit_cyclize.py` does not currently set `use_residue_type_feature`
at all, so the reveal mask reaches the model only through the `val_generation` path.

**3. Motif-restricted preservation metrics do not exist.** The rows carry
`ca_rmsd_to_input_A`, `ca_max_dev_to_input_A` and `contact_retention`, but all three are
**all-residue and CA-only**. The plan's success criterion is motif RMSD ≤ 0.5 Å and worst-atom
displacement ≤ 1 Å over *selected sidechain functional-group atoms*, receptor-aligned. Two
things are missing: a motif selection (which residues, which atoms) and the atom-level
displacement computed over it. The receptor-frame half is already solid — the sampled peptide
is produced in the target-centred frame, so displacement is a direct subtraction with no
superposition step that could hide movement away from the binding site, and
`scripts/complex_frame.py` refuses to write a complex whose Kabsch residual exceeds tolerance
rather than writing it misplaced. The AE round-trip check (`scripts/submit_ae_roundtrip.sh`)
has the same shape of gap: `masked_atom37_coord_metrics` aggregates over every valid atom, so
it cannot report the reconstruction floor *on the motif atoms specifically*.

Two smaller notes for whoever writes Stage 1:

- All 60 LNR staged rows carry `cyclization_type == "other"`, so `--native-type` is unusable
  on this set; the chemistry must be requested explicitly.
- `CPSea_data/lnr_pocket14` receptors run 57–287 residues (median 90.5), a 5× spread. Cost
  quoted as a single mean over an unstated pocket mixture is not a number Stage 1 can be
  sized from, which is why the profile below splits by pocket size.

## 2. The cost profile

Four jobs, split by resource and chained `afterok`:

| | job | resource | what it does |
|---|---|---|---|
| 1 | `stage0_profile_select.sbatch` | CPU | picks the N smallest and N largest pockets, writes `profile/cases.{txt,json}` |
| 2 | `stage0_profile_gpu.sbatch` | 1× a6000 | runs the generations, writes `profile/run_meta.json` |
| 3 | `stage0_profile_rosetta.sbatch` | CPU | interface dG under `/usr/bin/time -v`, writes `profile/rosetta_cost.json` |
| 4 | `stage0_profile_report.sbatch` | CPU | `profile/stage0_profile.{md,json}` |

The GPU job is a **wrapper around `scripts/sdedit_sweep.sbatch`**, not a copy of it. A profile
is only useful if it measured the code path Stage 1 will run; a parallel invocation with its
own argument list would drift from the production arm the first time either changed. Two
default-off env knobs were added to that arm to make this possible: `EXAMPLES`/`EXAMPLES_FILE`
(restrict to an explicit case list) and `PROFILE` (stamp per-edit cost on each row). An env
file that predates them produces byte-identical behaviour.

Default measurement: 8 cases × 1 seed × 2 arms = **16 generations** at `nsteps=400`,
`t_ca=0.8`, `t_lat=0.4`, disulfide. The two arms are unguided and the intended closure
guidance (`w=50`), because guidance adds a gradient step per sampler step and its cost is a
separate measurement rather than a correction to the unguided one.

### Running it

```bash
# 1. ALWAYS check the pool first. If an a6000 shows AVAILABLE, pin it with --nodelist;
#    otherwise let the default --exclude route the job to any a6000 in `general`.
myfree

# 2. Smoke the chain (few GPU-minutes; cost numbers from this run are NOT the profile).
bash scripts/submit_stage0_profile.sh --smoke

# 3. The real profile.
bash scripts/submit_stage0_profile.sh
```

Size it for an **a6000 (48 GB)**. `general` is a mixed pool, so the submitter defaults to
`--exclude=gpu28` (the a5000): `MEM_PROFILE=a5000`/`a5000_min` silently force
`self_cond=false`, which would profile a different model than the one Stage 1 runs. An
`rtx_2080_ti` (11 GB) can die on a CUDA allocation with no Python traceback — which reads as a
mysterious silent `FAILED`. That exclude list is a snapshot of the pool, not a fact about it;
re-check `myfree` before each launch.

### Reading the report

Two things the report is built to not get wrong, and that any hand analysis must also get
right:

- **Abstention.** When the cyclization head finds no candidate anchor pair it abstains and
  `cyc/*_bond_success` comes back `NaN`. `NaN` is truthy in Python, so a naive filter scores a
  100 %-abstention cell as 100 % closure. Every closure figure gates on
  `requested_type_satisfied` first and prints the abstained count beside the closure rate.
- **Setup overhead.** Model load is paid once per *job* — a 2.9 GB flow checkpoint plus a
  4.1 GB AE off ZFS — not once per structure. Generation seconds and setup seconds are kept
  apart all the way to the extrapolation, which adds setup back in per job. Folding setup into
  a per-structure mean would make a 16-structure probe badly overstate the tranche it is being
  used to size.

The projection at the end is total compute at the measured per-unit rates, checked against the
plan's 24 GPU-hour / 300 CPU-core-hour caps. Parallel workers shorten elapsed time without
reducing the total. Re-render it against a different tranche size without re-sampling
anything:

```bash
.venv/bin/python script_utils/stage0_profile_report.py \
  --results-dir evaluation_results/stage0_profile_<stamp> --n-structures 800
```

### Prior evidence for the expected order of magnitude

A completed de novo run (`uncondgen_gapped31`, job 43819, RTX A6000) did 240 edits in 3,426 s
at the sampler's own `nsteps`, i.e. **~14.3 s/structure**, with no per-structure memory or
CPU-side figures recorded. At that rate the plan's 1,360-structure Stage 1 ceiling is roughly
5.4 GPU-hours, comfortably inside the 24-hour cap. That is a prior, not the measurement: it
averages over an unstated pocket mixture, has setup smeared into it, and says nothing about
the guided arm. The profile exists to replace it.
