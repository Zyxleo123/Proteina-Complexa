#!/bin/bash
#SBATCH --job-name=sdedit_cases
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# GPU: re-sample a NAMED handful of edits, writing the receptor-inclusive complexes.
#
# Why re-sample instead of scoring what is on disk: the trajectory/slider runs saved the
# peptide alone, and the CPSea loader's frame is redrawn per process, so those peptides have
# no recoverable placement in their receptor (see scripts/complex_frame.py). Re-running is
# the fix, and it is honest: same pinned checkpoint, same AE, same seed and same grid point,
# so the structure produced here is the one the slider already shows.
#
# Do NOT sbatch directly -- use scripts/submit_slider_scores.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${FLOW_CKPT_PATH:?FLOW_CKPT_PATH not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"
: "${CASE_EXAMPLES:?CASE_EXAMPLES not set}"
FLOW_CKPT_NAME="${FLOW_CKPT_NAME:-last-EMA.ckpt}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable"; exit 1; }
[[ -f "${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}" ]] || { echo "FATAL: flow ckpt missing"; exit 1; }
[[ -f "${CPSEA_AE_CKPT_PATH}" ]] || { echo "FATAL: AE ckpt missing"; exit 1; }

mkdir -p "$RESULTS_DIR"
echo "examples: ${CASE_EXAMPLES}"
echo "grid: t_ca=${T_CA_GRID} t_lat=${T_LAT_GRID} seeds=${SEEDS} chems=${CYC_TYPES}"

# shellcheck disable=SC2086
srun "$PY" scripts/sdedit_cyclize.py \
  --ckpt-path "$FLOW_CKPT_PATH" \
  --ckpt-name "$FLOW_CKPT_NAME" \
  --metadata "$LNR_METADATA" \
  --examples ${CASE_EXAMPLES} \
  --out "${RESULTS_DIR}/edits_shard0.jsonl" \
  --pdb-dir "${RESULTS_DIR}/pdbs" \
  --complex-pdb-dir "${RESULTS_DIR}/complexes" \
  --cyc-types ${CYC_TYPES} \
  --t-ca ${T_CA_GRID} \
  --t-lat ${T_LAT_GRID} \
  --seeds ${SEEDS} \
  --nsteps "${NSTEPS:-0}"

echo "DONE cases -> ${RESULTS_DIR}"
