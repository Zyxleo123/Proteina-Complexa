#!/bin/bash
#SBATCH --job-name=lpgen_dpbdock
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_dpbdock_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_dpbdock_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# DiffPepDock on the staged LNR complexes: the k=0 (zero-mutation) baseline.
#
# k=0 means the peptide sequence is supplied, not designed. Upstream expresses that as
# `data.mask_lig_seq=False`, which makes the writer use `gt_aatype` instead of sampling from
# `aa_logits_pred` (experiments/train.py:575) -- while `diffused_mask = ligand_mask` keeps
# the WHOLE peptide diffused, so the pose is free and a new binding mode is allowed. That
# combination is the baseline the plan asks for: "A zero-mutation LP baseline tests whether
# sequence edits improve the score."
#
# Preprocessing and sampling share this one GPU job on purpose: the ESM-2 embedding step in
# preprocessing wants the GPU too, and it writes the metadata CSV that sampling reads.
# It is skipped when its output already exists, so a requeue resumes instead of redoing it.
#
# Rosetta/Amber postprocessing is OFF here. Scoring runs afterwards on the CPU partition
# through the evaluator every arm shares -- and our env deliberately has neither pyrosetta
# nor openmm (their pinned versions have no wheels; see _lpgen_preamble.sh).
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "dpbdock"

ARM_TREE="${UPSTREAM_DIR}/diffpepbuilder"
ARM_PY="${ENVS_DIR}/diffpepbuilder/bin/python"

# Gates print why and exit 0: a non-zero exit here would strand any afterok dependent in
# DependencyNeverSatisfied, which is indistinguishable from "upstream never ran".
if [[ ! -x "$ARM_PY" ]]; then
  echo "GATE: no diffpepbuilder env at ${ARM_PY}" >&2
  echo "GATE: ./scripts/submit_lpgen.sh --submit env --set STAGE_ARMS=diffpepbuilder" >&2
  exit 0
fi
if [[ ! -f "${DPB_STAGE_DIR}/receptor_info.json" ]]; then
  echo "GATE: no staged complexes at ${DPB_STAGE_DIR}" >&2
  echo "GATE: ./scripts/submit_lpgen.sh --submit dpbstage" >&2
  exit 0
fi
DOCK_CKPT="${DOCK_CKPT:-${WEIGHTS_DIR}/diffpepbuilder/diffpepdock_v1.pth}"
if [[ ! -f "$DOCK_CKPT" ]]; then
  echo "GATE: no DiffPepDock checkpoint at ${DOCK_CKPT}" >&2
  echo "GATE: ./scripts/submit_lpgen.sh --submit weights --set STAGE_ARMS=diffpepbuilder" >&2
  exit 0
fi

# Every entry point calls pyrootutils.setup_root(indicator=[".git"], dotenv=True), so `root`
# is found by walking up to the clone's .git -- and the tree ships NO .env, so BASE_PATH,
# which the configs read as ${oc.env:BASE_PATH}, has to come from here.
export BASE_PATH="$ARM_TREE"
export PYTHONPATH="$ARM_TREE"
cd "$ARM_TREE"

# Keep the HF/torch caches off home (quota'd, and it holds the repo).
export TORCH_HOME="${TORCH_HOME:-${LPGEN_ROOT}/torch_home}"
mkdir -p "$TORCH_HOME"

PROC_DIR="${DPB_PROC_DIR:-${DPB_STAGE_DIR}/processed}"
RUN_DIR="${DPB_RUN_DIR:-${LPGEN_RUNS_DIR}/${RUN_ID}_dpbdock}"
mkdir -p "$PROC_DIR" "$RUN_DIR"

echo "[dpbdock] tree  = ${ARM_TREE}"
echo "[dpbdock] stage = ${DPB_STAGE_DIR}"
echo "[dpbdock] proc  = ${PROC_DIR}"
echo "[dpbdock] run   = ${RUN_DIR}"
echo "[dpbdock] ckpt  = ${DOCK_CKPT}"
echo "[dpbdock] k     = 0 (mask_lig_seq=False; native sequence supplied, pose free)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------- preprocessing
META="${PROC_DIR}/metadata_test.csv"
if [[ -s "$META" && "${DPB_FORCE_PREPROCESS:-0}" != "1" ]]; then
  echo "[dpbdock] preprocessing already done: ${META} ($(wc -l < "$META") lines incl. header)"
else
  echo "[dpbdock] preprocessing ${DPB_STAGE_DIR} -> ${PROC_DIR}"
  # No --peptide_seq_path: upstream then "use[s] the ligand chain from the PDB file", i.e.
  # the native sequence read straight out of chain A. That IS k=0, with no FASTA to build.
  # pocket_cutoff is left at its default of 10, which is the same 10.0 A heavy-atom
  # whole-residue definition measured against PepMerge for the native export.
  "$ARM_PY" experiments/process_batch_dock.py \
    --pdb_dir "$DPB_STAGE_DIR" \
    --write_dir "$PROC_DIR" \
    --receptor_info_path "${DPB_STAGE_DIR}/receptor_info.json" \
    --max_batch_size "${DPB_ESM_BATCH:-8}"
fi

# PROVE the inputs were not rewritten. The staged layout (peptide as chain A) is supposed to
# make upstream's in-place renumber unreachable; this turns that from a belief into a
# measurement, and it runs BEFORE sampling so a failure is caught before the GPU time.
echo "=================================================================="
echo "[dpbdock] verifying no staged input was modified by preprocessing:"
if ! "$PY" "${REPO}/script_utils/lpgen_dpb_stage.py" --out-dir "$DPB_STAGE_DIR" --verify; then
  echo "[dpbdock] FATAL: preprocessing rewrote a staged input. Results would be untrustworthy." >&2
  exit 1
fi
echo "=================================================================="

# ---------------------------------------------------------------------- docking
# Overrides, and why each one:
#   num_gpus/use_ddp  - docking.yaml ships num_gpus: 8 and use_ddp: True
#   eval_batch_size   - its 128 is sized for 8 GPUs
#   run_postprocess   - needs pyrosetta + openmm, which this env deliberately lacks
#   mask_lig_seq      - set explicitly so the log records which k this run is
"$ARM_PY" experiments/run_docking.py \
  --config-name=docking \
  data.val_csv_path="$META" \
  data.mask_lig_seq=False \
  data.num_repeat_per_eval_sample="${DPB_N_SAMPLES:-8}" \
  experiment.num_gpus=1 \
  experiment.use_ddp=False \
  experiment.eval_batch_size="${DPB_BATCH:-8}" \
  experiment.eval_ckpt_path="$DOCK_CKPT" \
  experiment.eval_dir="$RUN_DIR" \
  postprocess.run_postprocess=False \
  ${EXTRA_DOCK_ARGS:-}

echo "=================================================================="
echo "[dpbdock] outputs under ${RUN_DIR}:"
find "$RUN_DIR" -name '*.pdb' | wc -l | xargs echo "  pdbs:"
find "$RUN_DIR" -maxdepth 2 -type d | head -5
echo "[dpbdock] done"
