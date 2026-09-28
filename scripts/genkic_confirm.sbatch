#!/bin/bash
#SBATCH --job-name=genkic_confirm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# Re-score the SELECTED candidates with many more independent relax/score repeats, to test
# whether a pass at the energy bar survives its own measurement noise.
#
# This stage exists because selection and noise interact badly. The median within-candidate
# spread over 3 repeats is 8.4 REU and 60% of candidates exceed the whole +/-5 REU bar, so
# taking the best of a shortlist will put something under the bar by luck alone. The first
# accepted candidate passed at ddG +4.67 with a 1.04 REU spread -- a 0.33 REU margin, inside
# its own noise. Re-measuring with more repeats asks the different, correct question: does the
# MEDIAN still pass?
#
# Cheap: only accepted candidates and energy-only near misses are re-scored, which is a handful
# of structures rather than the whole pool. A candidate that also fails RMSD or contacts is
# skipped, because no number of relax repeats moves a geometric criterion.
#
# Do NOT sbatch directly -- use scripts/submit_genkic.sh. Config in the positional env file.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa/.claude/worktrees/genkic}/scripts/_genkic_preamble.sh"
[[ -f "$PREAMBLE" ]] || { echo "FATAL: preamble not found: $PREAMBLE"; exit 1; }
# shellcheck source=/dev/null
source "$PREAMBLE"

_genkic_consume_args "$@"
_genkic_defaults
_genkic_venv

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${CONFIRM_SHARDS:-4}"
_genkic_banner "confirm shard ${SHARD_INDEX}/${SHARD_COUNT}"

# Gate inside the job, exit 0: "nothing reached the bar yet" is a legitimate outcome, and a
# non-zero exit would report it as a crash.
shopt -s nullglob
SHARDS=("${ROWS_DIR}"/rows_*.jsonl)
shopt -u nullglob
if [[ ${#SHARDS[@]} -eq 0 ]]; then
  echo "GATE FAILED: no rows in ${ROWS_DIR} to select candidates from. Exiting 0."
  exit 0
fi

CONFIRM_DIR="${CONFIRM_DIR:-${OUT_DIR}/confirm}"
mkdir -p "$CONFIRM_DIR"
OUT="${CONFIRM_DIR}/rows_${SHARD_INDEX}.jsonl"

ROWS_ARGS=("${ROWS_DIR}")
# Confirm the SDEdit arm's winners too, under the identical protocol -- otherwise the
# comparison would hold the two arms to different evidential standards.
if [[ -d "${SDEDIT_ROWS_DIR}" ]]; then
  ROWS_ARGS+=("${SDEDIT_ROWS_DIR}")
  echo "including SDEdit rows from ${SDEDIT_ROWS_DIR}"
fi

echo "confirming with ${CONFIRM_REPEATS:-15} repeats per candidate AND per LP baseline"
srun python script_utils/confirm_candidates.py \
  --rows "${ROWS_ARGS[@]}" \
  --metadata "${METADATA}" \
  --out "$OUT" \
  --repeats "${CONFIRM_REPEATS:-15}" \
  --relax-repeats "${RELAX_REPEATS}" \
  --near-miss-reu "${CONFIRM_NEAR_MISS_REU:-5.0}" \
  --max-candidates "${CONFIRM_MAX:-0}" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"

echo "shard ${SHARD_INDEX} wrote $(wc -l < "$OUT" 2>/dev/null || echo 0) rows to $OUT"
