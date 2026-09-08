#!/bin/bash
#SBATCH --job-name=sdedit_rosetta_plot
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU: the before/after dG figure + table. Reads only the scorer's JSONL, so restyling the
# figure never costs a Rosetta run, let alone a GPU one.
#
#   sbatch scripts/sdedit_rosetta_plot.sbatch <env-file> [arm]
#
# Do NOT sbatch directly -- use scripts/submit_sdedit_rosetta.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
ARM="${2:-}"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
EDIT_DIR="${RESULTS_DIR}${ARM:+/$ARM}"
ROSETTA_DIR="${ROSETTA_DIR:-${EDIT_DIR}/rosetta}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

shopt -s nullglob
ROWS=("${ROSETTA_DIR}"/rosetta_shard*.jsonl)
shopt -u nullglob
if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "GATE: no rosetta_shard*.jsonl under ${ROSETTA_DIR}."
  echo "GATE: read the scorer's .out files -- the usual cause is an SDEdit arm run without"
  echo "GATE: --complex-pdb-dir, whose complexes cannot be rebuilt afterwards. Exiting 0."
  exit 0
fi

echo "plotting ${#ROWS[@]} shard file(s) from ${ROSETTA_DIR}"
"$PY" scripts/plot_rosetta_before_after.py \
  --rosetta "${ROWS[@]}" \
  --out-dir "${ROSETTA_DIR}/figures" \
  --title "Rosetta interface dG: linear input vs cyclized edit${ARM:+ (${ARM})}"

# The interactive page: every scored attempt, filterable, with the ledger table. Same inputs
# as the PNG, so the two can never disagree.
"$PY" scripts/build_rosetta_report.py \
  --rosetta "${ROWS[@]}" \
  --out "${ROSETTA_DIR}/figures/ring_cost_ledger.html" \
  --run-label "$(basename "$(dirname "$(dirname "$EDIT_DIR")")")${ARM:+ / ${ARM}}" \
  --checkpoint "$(basename "$(dirname "${FLOW_CKPT_PATH:-unknown}")")"
