#!/bin/bash
#SBATCH --job-name=soft_closure_summarize
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 5 (CPU): the paired-arm comparison. Reads saved JSONL only.
# Do NOT sbatch directly -- use scripts/submit_soft_closure.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${PROJ_DIR:?PROJ_DIR not set}"

if ! compgen -G "${RESULTS_DIR}/control/edits_shard*.jsonl" > /dev/null; then
  echo "GATE: no control edits under ${RESULTS_DIR}/control -- nothing to compare."
  [[ -f "${PROJ_DIR}/projection_summary.json" ]] && cat "${PROJ_DIR}/projection_summary.json"
  exit 0
fi

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
srun "$PY" -u scripts/summarize_soft_closure.py \
  --results-dir "$RESULTS_DIR" \
  --proj-dir "$PROJ_DIR" \
  --cyc-type "${CYC_TYPE:-mainchain}"

echo "DONE summary -> ${RESULTS_DIR}/summary.json"
