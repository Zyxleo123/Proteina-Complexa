#!/bin/bash
#SBATCH --job-name=precheck_decoy_score
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#
# Pre-build check 1, stage C: Rosetta interface dG + geometry, before and after
# minimization. Back in .venv, which is where PyRosetta lives.
#
# Measured cost: ~36 s for the first call (PyRosetta init) then ~2 s per structure.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv venv

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
_precheck_banner "decoy score shard ${SHARD}"

shopt -s nullglob
mins=("${DECOY_DIR}"/minimized_shard*.jsonl)
if (( ${#mins[@]} == 0 )); then
  echo "GATE: no minimized rows under ${DECOY_DIR}. Exiting 0."
  exit 0
fi

EXTRA=(); [[ "${DECOY_NO_ROSETTA:-0}" == "1" ]] && EXTRA+=(--no-rosetta)

python -u "${REPO}/script_utils/precheck_decoy_score.py" \
  --config "${PRECHECK_CONFIG}" --minimized-dir "${DECOY_DIR}" \
  --out "${DECOY_DIR}/scored_shard${SHARD}.jsonl" \
  --shard "${SHARD}" --n-shards "${DECOY_SHARDS}" "${EXTRA[@]}"
