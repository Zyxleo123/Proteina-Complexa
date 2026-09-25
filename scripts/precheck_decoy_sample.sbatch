#!/bin/bash
#SBATCH --job-name=precheck_decoy_sample
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#
# Pre-build check 1, stage A: stratified complex selection + perturbed pose writing.
#
# Array task 0 owns the SELECTION. Every other task waits for selection.csv to appear and
# then reuses it: a per-shard selection would have each shard sampling different complexes
# and silently turn one stratified set into N unrelated ones.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv venv

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
_precheck_banner "decoy sample shard ${SHARD}"
mkdir -p "${DECOY_DIR}"
SEL="${DECOY_DIR}/selection.csv"

if [[ "${SHARD}" != "0" ]]; then
  # Stage A's selection is measured (burial needs SASA), so it is built once by task 0.
  for _ in $(seq 1 240); do [[ -f "${SEL}" ]] && break; sleep 15; done
  [[ -f "${SEL}" ]] || { echo "GATE: selection.csv never appeared; task 0 failed."; exit 0; }
fi

EXTRA=(); [[ "${DECOY_LIMIT:-0}" != "0" ]] && EXTRA+=(--limit "${DECOY_LIMIT}")

python -u "${REPO}/script_utils/precheck_decoy_sample.py" \
  --config "${PRECHECK_CONFIG}" --out-dir "${DECOY_DIR}" \
  --selection "${SEL}" --shard "${SHARD}" --n-shards "${DECOY_SHARDS}" "${EXTRA[@]}"
