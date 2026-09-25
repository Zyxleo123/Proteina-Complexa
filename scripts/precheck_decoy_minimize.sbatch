#!/bin/bash
#SBATCH --job-name=precheck_decoy_min
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#
# Pre-build check 1, stage B: frozen-pocket minimization. Runs in .venv_openmm, which
# CANNOT coexist with .venv -- hence a stage of its own rather than a function call.
#
# Measured cost: ~28 s per pose (16 poses per complex, ~450 s per complex). Resumable at
# pose granularity, so a wall-clock kill costs the tail and not the shard.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv openmm

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
_precheck_banner "decoy minimize shard ${SHARD}"

MINCFG="${DECOY_DIR}/minimize.json"
if [[ ! -f "${MINCFG}" ]]; then
  echo "GATE: ${MINCFG} missing. The submitter writes it from the YAML, because the config"
  echo "loader imports yaml/pandas that .venv_openmm does not carry. Exiting 0."
  exit 0
fi

shopt -s nullglob
man=("${DECOY_DIR}"/manifest_shard*.jsonl)
if (( ${#man[@]} == 0 )); then
  echo "GATE: no stage-A manifests under ${DECOY_DIR}. Exiting 0 so dependents report the"
  echo "same gate failure rather than stranding on DependencyNeverSatisfied."
  exit 0
fi

python -u "${REPO}/script_utils/precheck_decoy_minimize.py" \
  --manifest-dir "${DECOY_DIR}" --out-dir "${DECOY_DIR}" \
  --minimize-config "${MINCFG}" --shard "${SHARD}" --n-shards "${DECOY_SHARDS}"
