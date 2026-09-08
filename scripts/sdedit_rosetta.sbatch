#!/bin/bash
#SBATCH --job-name=sdedit_rosetta
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# CPU arm: Rosetta interface dG on the SDEdit input (before) and the edited complex (after).
# No GPU on purpose -- it only reads structures the sampler already wrote, which is also what
# lets the figure be re-rendered without re-sampling.
#
# Run as an ARRAY, sharded by input peptide; each task writes its OWN JSONL (concurrent
# O_APPEND to one file has NUL-corrupted rows on this cluster).
#
#   sbatch scripts/sdedit_rosetta.sbatch <env-file> [arm]
#
# `arm` (optional) names a subdirectory of RESULTS_DIR -- `projected` / `control` for the
# soft-closure run. Omitted, RESULTS_DIR itself holds the edits.
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
SHARD_COUNT="${ROSETTA_SHARDS:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

# The upstream GATE: no edits means the SDEdit arm produced nothing (or was gated itself).
# Exit 0 so the plot job still runs and reports it instead of dying on Dependency.
shopt -s nullglob
EDITS=("${EDIT_DIR}"/edits_shard*.jsonl)
shopt -u nullglob
if [[ ${#EDITS[@]} -eq 0 ]]; then
  echo "GATE: no edits_shard*.jsonl under ${EDIT_DIR} -- the SDEdit arm produced nothing."
  echo "GATE: nothing to score. Exiting 0."
  exit 0
fi

mkdir -p "$ROSETTA_DIR"
echo "Node: $(hostname)"
echo "arm=${ARM:-<root>}  edits=${#EDITS[@]} file(s)  shard ${SHARD_INDEX}/${SHARD_COUNT}"
echo "complexes: ${COMPLEX_DIR:-${EDIT_DIR}/complexes}"

ARGS=(--edits "${EDITS[@]}"
      --out "${ROSETTA_DIR}/rosetta_shard${SHARD_INDEX}.jsonl"
      --complex-dir "${COMPLEX_DIR:-${EDIT_DIR}/complexes}"
      --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT")
[[ -n "${ROSETTA_METADATA:-}" ]] && ARGS+=(--metadata "$ROSETTA_METADATA")
[[ "${ROSETTA_ONLY_SCORABLE:-1}" == "1" ]] && ARGS+=(--only-scorable)
[[ "${ROSETTA_ONLY_CLOSED:-0}" == "1" ]] && ARGS+=(--only-closed)
[[ -n "${ROSETTA_MAX_PER_EXAMPLE:-}" ]] && ARGS+=(--max-per-example "$ROSETTA_MAX_PER_EXAMPLE")

srun "$PY" scripts/score_sdedit_rosetta.py "${ARGS[@]}"

echo "DONE rosetta shard ${SHARD_INDEX} (arm=${ARM:-<root>})"
