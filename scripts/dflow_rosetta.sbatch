#!/bin/bash
#SBATCH --job-name=dflow_rosetta
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# CPU stage: paired Rosetta interface scoring, CP candidate minus its ORIGINAL LP
# (script_utils/dflow_rosetta.py). FastRelax is the expensive part of this experiment and it
# does not touch the GPU, so it runs as its own CPU array off the GPU path.
#
# Array, one shard per task, each writing its OWN output file -- concurrent O_APPEND has
# corrupted rows on this cluster. Resumable: the script skips score_keys already present and
# ROSETTA_MAX_SECONDS makes it stop cleanly instead of being killed at the wall.
#
# Do NOT sbatch directly -- use scripts/submit_dflow.sh.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/_dflow_preamble.sh" "$@"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
dflow_preflight_python

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ROSETTA_SHARDS="${ROSETTA_SHARDS:-1}"
ROSETTA_DIR="${ROSETTA_DIR:-${RESULTS_DIR}/rosetta}"
mkdir -p "$ROSETTA_DIR"

# Scored from the RE-SCORED rows, not the arm's own: that way the shortlist Rosetta spends
# its time on is the one the shared instrument selected, and `--only-passing` means the same
# thing for every arm. Falls back to the raw D-Flow rows if rescoring was skipped.
ROWS_GLOB="${ROSETTA_ROWS:-${RESCORE_DIR:-${RESULTS_DIR}/rescored}/rescored_*.jsonl}"
# shellcheck disable=SC2206
ROWS_FILES=( $ROWS_GLOB )
if [[ ! -f "${ROWS_FILES[0]}" ]]; then
  echo "note: no rescored rows at ${ROWS_GLOB}; falling back to raw D-Flow rows"
  # shellcheck disable=SC2206
  ROWS_FILES=( ${RESULTS_DIR}/dflow_shard*.jsonl )
  [[ -f "${ROWS_FILES[0]}" ]] || { echo "FATAL: no rows to score"; exit 1; }
fi

echo "shard ${SHARD_INDEX}/${ROSETTA_SHARDS}"
echo "rows:  ${#ROWS_FILES[@]} file(s)"
echo "repeats ${ROSETTA_REPEATS:-1}  stage ${ROSETTA_STAGE:-opt}"

"$DFLOW_PY" script_utils/dflow_rosetta.py \
  --rows "${ROWS_FILES[@]}" \
  --out "${ROSETTA_DIR}/rosetta_shard${SHARD_INDEX}.jsonl" \
  --repeats "${ROSETTA_REPEATS:-1}" \
  --stage "${ROSETTA_STAGE:-opt}" \
  --max-per-example "${ROSETTA_MAX_PER_EXAMPLE:-0}" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$ROSETTA_SHARDS" \
  --max-seconds "${ROSETTA_MAX_SECONDS:-0}" \
  ${ROSETTA_ONLY_PASSING:+--only-passing} \
  ${ROSETTA_NO_RELAX:+--no-relax}

echo "DONE rosetta shard ${SHARD_INDEX}"
