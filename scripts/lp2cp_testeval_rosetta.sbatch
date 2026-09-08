#!/bin/bash
#SBATCH --job-name=lp2cp_testeval_rosetta
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# Stage 2 (CPU array): score Rosetta interface dG on the complexes stage 1 dumped. Each array task
# is one shard -- it scores a disjoint slice of the step_*/ dirs and writes its OWN rows_<task>.jsonl,
# so there is exactly one writer per file (no concurrent-append NUL corruption). Resumable: a shard
# re-run skips complexes already in its own out file.
#
# Do NOT sbatch directly -- use scripts/submit_lp2cp_testeval.sh. Config in the positional env file.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${EVAL_DUMP_DIR:?EVAL_DUMP_DIR not set in $ENV_FILE}"
: "${EVAL_ROSETTA_OUT_DIR:?EVAL_ROSETTA_OUT_DIR not set in $ENV_FILE}"

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
SHARD_COUNT="${EVAL_SHARD_COUNT:-16}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$EVAL_ROSETTA_OUT_DIR"
OUT="$EVAL_ROSETTA_OUT_DIR/rows_${SHARD_INDEX}.jsonl"

echo "Node: $(hostname)  shard=${SHARD_INDEX}/${SHARD_COUNT}"
if [[ ! -d "$EVAL_DUMP_DIR" ]]; then
  echo "GATE FAILED: no dump dir at $EVAL_DUMP_DIR (stage 1 produced nothing). Exiting 0."
  exit 0
fi

srun "$PY" scripts/score_val_gen_rosetta.py \
  --dump-dir "$EVAL_DUMP_DIR" \
  --out "$OUT" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"
