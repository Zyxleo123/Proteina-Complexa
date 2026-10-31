#!/bin/bash
#SBATCH --job-name=cpseq_mask_arm
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Arm E/F/G proposal stage: the LLM names positions, ProteinMPNN fills them.
#
# Two models, one job, in sequence: Llama-3.1-8B writes the masks and is released before
# ProteinMPNN runs. They are not co-resident, and neither competes with Boltz-2 -- folding
# is a separate stage, which is the constraint that made arm D a separate job too.
#
# RESUMABLE, because preempt requeues. The mask stage writes one JSON per parent and the
# fill stage reads them, so a requeue that lands after the masks are written skips straight
# to the fill. Delete the masks directory to force a re-ask.
#
# Config travels in a FILE passed positionally (never `sbatch --export`, which triggers
# the user-env-retrieval requeue-hold on this cluster).
#
# Usage:
#   sbatch -p preempt --qos=preempt_qos scripts/cpseq_mask_arm.sbatch <env-file>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="${1:-}"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: ${ENV_FILE:-<none>}"; exit 1; }
source "$ENV_FILE"

: "${MASK_OUT_DIR:?MASK_OUT_DIR not set in $ENV_FILE}"
ROUND="${MASK_ROUND:-1}"
N_MASKS="${MASK_N_MASKS:-12}"
ARMS="${MASK_ARMS:-llm_mask_mpnn_fill,random_mask_mpnn_fill,llm_mask_random_fill}"
SEQS_PER_MASK="${MASK_SEQS_PER_MASK:-1}"
MPNN_TEMP="${MASK_MPNN_TEMP:-0.2}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

echo "=== $(date) $(hostname) job=${SLURM_JOB_ID:-none} ==="
nvidia-smi -L || true
echo "  out=$MASK_OUT_DIR round=$ROUND n_masks=$N_MASKS arms=$ARMS"
echo "  requested: gpu=${SLURM_JOB_GRES:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=64G"

[[ -f "$MASK_OUT_DIR/cohort.json" ]] || {
  echo "GATE FAILED: no cohort.json in $MASK_OUT_DIR. Nothing to propose for. Exiting 0 so dependents report the gap."
  exit 0
}

MASKS_DIR="$MASK_OUT_DIR/masks/round$ROUND"
MASK_DONE="$MASKS_DIR/index.json"

# ---- stage 1: the LLM names positions ------------------------------------------------
if [[ -f "$MASK_DONE" ]]; then
  echo "--- masks already written ($MASK_DONE); skipping the LLM stage ---"
else
  echo "--- stage 1: LLM mask round ---"
  HF_HUB_OFFLINE=1 "$PY" -u script_utils/cpseq_llm_mask_round.py \
    --out-dir "$MASK_OUT_DIR" \
    --round "$ROUND" \
    --n-masks "$N_MASKS" \
    ${MASK_PARENTS:+--parents "$MASK_PARENTS"}
fi

# ---- stage 2: ProteinMPNN fills them -------------------------------------------------
echo "--- stage 2: fill ---"
"$PY" -u script_utils/cpseq_mask_mpnn_fill.py \
  --out-dir "$MASK_OUT_DIR" \
  --masks-dir "$MASKS_DIR" \
  --round "$ROUND" \
  --arms "$ARMS" \
  --seqs-per-mask "$SEQS_PER_MASK" \
  --temperature "$MPNN_TEMP" \
  ${MASK_PARENTS:+--parents "$MASK_PARENTS"}

echo "=== $(date) finished ==="
