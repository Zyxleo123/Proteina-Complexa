#!/bin/bash
#SBATCH --job-name=stage0_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 0, step 4: turn the artefacts the previous three jobs wrote into the cost table and
# the Stage 1 projection.
#
# CPU only, and it recomputes nothing: everything it needs is already on disk as JSONL rows
# and JSON sidecars, so the table can be re-rendered with a different projection target (or
# after a bug fix in the report) without re-sampling a single structure.
#
# Do NOT sbatch directly -- use scripts/submit_stage0_profile.sh.

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
EDITS=("${RESULTS_DIR}"/edits_shard*.jsonl)
shopt -u nullglob
if [[ ${#EDITS[@]} -eq 0 ]]; then
  # GATE, not a failure: the GPU stage produced nothing. Exit 0 and say which stage is the
  # hole, rather than exiting non-zero and leaving a chain of DependencyNeverSatisfied.
  echo "GATE: no edits_shard*.jsonl under ${RESULTS_DIR}."
  echo "GATE: the GPU profile stage produced nothing -- read its .out before rerunning."
  exit 0
fi

"$PY" script_utils/stage0_profile_report.py \
  --results-dir "$RESULTS_DIR" \
  --n-structures "${PROJECT_STRUCTURES:-1360}"

echo "DONE stage0 report -> ${RESULTS_DIR}/profile/stage0_profile.md"
