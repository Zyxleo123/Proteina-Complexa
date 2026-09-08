#!/bin/bash
#SBATCH --job-name=sdedit_traj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# GPU arm: re-run named SDEdit edits and dump every integration frame
# (scripts/sdedit_trajectory.py). One job, all cases -- each is only a few minutes.
#
# MUST run on an RTX A6000: the sweep being reproduced ran on one, and a different GPU
# class perturbs floating point enough to flip a marginal closure. The submitter pins
# this by excluding the one non-A6000 node on the partition.
#
# Do NOT sbatch directly -- use scripts/submit_sdedit_traj.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${TRAJ_DIR:?TRAJ_DIR not set}"
: "${FLOW_CKPT_PATH:?FLOW_CKPT_PATH not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"
: "${CASES:?CASES not set}"
FLOW_CKPT_NAME="${FLOW_CKPT_NAME:-last-EMA.ckpt}"

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Node: $(hostname)  GPU: ${GPU_NAME}"
case "$GPU_NAME" in
  *A6000*) ;;
  *) echo "WARNING: expected an RTX A6000 (the sweep's GPU); got '${GPU_NAME}'."
     echo "         Trajectories may not reproduce the swept sample exactly." ;;
esac

"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable"; exit 1; }
[[ -f "${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}" ]] || { echo "FATAL: flow ckpt missing: ${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}"; exit 1; }
[[ -f "${CPSEA_AE_CKPT_PATH}" ]] || { echo "FATAL: AE ckpt missing: ${CPSEA_AE_CKPT_PATH}"; exit 1; }
[[ -f "${LNR_METADATA}" ]] || { echo "FATAL: LNR metadata missing: ${LNR_METADATA}"; exit 1; }

mkdir -p "$TRAJ_DIR"
echo "flow ckpt: ${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}"
echo "AE ckpt:   ${CPSEA_AE_CKPT_PATH}"

# Unquoted on purpose: CASES is a space-separated list that must word-split into argv.
# shellcheck disable=SC2086
srun "$PY" scripts/sdedit_trajectory.py \
  --ckpt-path "$FLOW_CKPT_PATH" \
  --ckpt-name "$FLOW_CKPT_NAME" \
  --metadata "$LNR_METADATA" \
  --out-dir "$TRAJ_DIR" \
  --nsteps "${NSTEPS:-0}" \
  --cases ${CASES}

echo "DONE trajectories -> ${TRAJ_DIR}"
