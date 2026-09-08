#!/bin/bash
#SBATCH --job-name=ddib_invert_sum
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU arm: table + figure from the JSONL the GPU job wrote. Reads saved rows only, so the
# figure can be re-rendered without a second inversion.

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
ROWS=("${RESULTS_DIR}"/invert_*.jsonl)
if [[ ${#ROWS[@]} -eq 0 ]]; then
  # Gate failed, not "never ran": say so and exit 0 so nothing downstream is stranded
  # in DependencyNeverSatisfied with no explanation.
  echo "GATE: no invert_*.jsonl in ${RESULTS_DIR} -- the GPU arm produced nothing. Nothing to summarize."
  exit 0
fi

"$PY" scripts/summarize_ddib_invert.py \
  --results-dir "$RESULTS_DIR" \
  --out-prefix "${RESULTS_DIR}/ddib_invert"
echo "DONE"
