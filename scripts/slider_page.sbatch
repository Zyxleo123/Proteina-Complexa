#!/bin/bash
#SBATCH --job-name=slider_page
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU: rebuild the scrubable viewer page, pairing each trajectory with the Rosetta dG of its
# input and its edited complex. Reads only saved artefacts -- sprites and JSONL -- so the page
# re-renders without sampling or scoring anything.
#
# Do NOT sbatch directly -- use scripts/submit_slider_scores.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${SLIDER_INDEX:?SLIDER_INDEX not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"
ROSETTA_DIR="${ROSETTA_DIR:-${RESULTS_DIR}/rosetta}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

if [[ ! -f "$SLIDER_INDEX" ]]; then
  echo "GATE: no slider index at ${SLIDER_INDEX} -- nothing to build. Exiting 0."
  exit 0
fi

shopt -s nullglob
ROWS=("${ROSETTA_DIR}"/rosetta_shard*.jsonl)
shopt -u nullglob
if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "NOTE: no Rosetta rows under ${ROSETTA_DIR}; building the page WITHOUT paired energies."
  echo "NOTE: every case will say so rather than borrowing a number from another attempt."
fi

"$PY" scripts/build_slider_page.py \
  --slider-index "$SLIDER_INDEX" \
  ${ROWS[0]:+--rosetta "${ROWS[@]}"} \
  --out "${RESULTS_DIR}/cyclization_step_viewer.html" \
  --run-label "${PAGE_LABEL:-$(basename "$RESULTS_DIR")}" \
  --checkpoint "$(basename "$(dirname "${FLOW_CKPT_PATH:-unknown}")")"

echo "DONE page -> ${RESULTS_DIR}/cyclization_step_viewer.html"
