#!/bin/bash
#SBATCH --job-name=sdedit_sum
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU arm: reads the shard JSONLs and builds the trade-off tables/figures. No GPU on
# purpose -- it only reads artefacts, so figures re-render without re-running the sampler.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs
ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"
: "${RESULTS_DIR:?RESULTS_DIR not set}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

shopt -s nullglob
ROWS=( "${RESULTS_DIR}"/edits_shard*.jsonl )
if (( ${#ROWS[@]} == 0 )); then
  echo "GATE: no shard results in ${RESULTS_DIR} -- the GPU array produced nothing."
  echo "GATE: read the sdedit_sweep array .out files before trusting this chain. Exiting 0."
  exit 0
fi
"$PY" scripts/summarize_sdedit_sweep.py --results "${ROWS[@]}" --out-dir "${RESULTS_DIR}/summary"
echo "DONE"
