#!/bin/bash
#SBATCH --job-name=lp2cp_testeval_aggregate
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 3 (CPU): merge closure rows + Rosetta dG shards into summary.json + summary.png. Pure disk
# read, redrawable any time. Do NOT sbatch directly -- use scripts/submit_lp2cp_testeval.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${EVAL_OUT_DIR:?EVAL_OUT_DIR not set in $ENV_FILE}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

srun "$PY" script_utils/aggregate_lp2cp_testeval.py \
  --out-dir "$EVAL_OUT_DIR" \
  --rosetta-subdir "${EVAL_ROSETTA_SUBDIR:-rosetta_dg}"
