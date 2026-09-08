#!/bin/bash
#SBATCH --job-name=ae_roundtrip_sum
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU arm: aggregates + plots the saved JSONL rows. Asks for no GPU on purpose -- this only
# reads artefacts, so it must not queue behind real GPU work, and the figure re-renders
# cheaply without re-running the model.
#
# Submitted by scripts/submit_ae_roundtrip.sh with --dependency=afterok on the GPU job.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set in $ENV_FILE}"
: "${LNR_STAGE_DIR:?LNR_STAGE_DIR not set in $ENV_FILE}"
CPSEA_VAL_METADATA="${CPSEA_VAL_METADATA:-${CPSEA_DATA_PATH}/preprocessed/metadata/cpsea_val.parquet}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

# Decision gate INSIDE the job: an upstream stage that produced nothing must explain itself
# and exit 0, not strand this job's own dependents in DependencyNeverSatisfied.
shopt -s nullglob
ROWS=( "${RESULTS_DIR}"/*.jsonl )
if (( ${#ROWS[@]} == 0 )); then
  echo "GATE: no result rows in ${RESULTS_DIR} -- the GPU arm ran but scored nothing."
  echo "GATE: read the ae_roundtrip job's .out before trusting this chain. Exiting 0."
  exit 0
fi

"$PY" scripts/summarize_ae_roundtrip.py \
  --results "${ROWS[@]}" \
  --lnr-metadata "${LNR_STAGE_DIR}/metadata/lnr_test.parquet" \
  --cpsea-metadata "$CPSEA_VAL_METADATA" \
  --out-dir "${RESULTS_DIR}/summary"
echo "DONE"
