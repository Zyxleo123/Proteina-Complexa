#!/bin/bash
#SBATCH --job-name=precheck_profile
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#
# Pre-build check 3b: feature profiles over LNR, PepBench and CPSea cyclics.
#
# The array is indexed over (set x shard) so all three sets queue concurrently rather than
# one waiting on another. PROFILE_SETS names the sets; the array index is decoded against
# PROFILE_SHARDS.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv venv

PROFILE_SETS="${PROFILE_SETS:-lnr pepbench cpsea}"
read -r -a SETS <<< "${PROFILE_SETS}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
SET_I=$(( IDX / PROFILE_SHARDS ))
SHARD=$(( IDX % PROFILE_SHARDS ))
SET_NAME="${SETS[$SET_I]:-}"
[[ -n "${SET_NAME}" ]] || { echo "array index ${IDX} maps to no set; nothing to do"; exit 0; }

_precheck_banner "profile ${SET_NAME} shard ${SHARD}"
mkdir -p "${PROFILE_DIR}"

EXTRA=()
[[ "${PROFILE_LIMIT:-0}" != "0" ]] && EXTRA+=(--limit "${PROFILE_LIMIT}")

python -u "${REPO}/script_utils/precheck_profile.py" profile \
  --config "${PRECHECK_CONFIG}" \
  --set "${SET_NAME}" \
  --out "${PROFILE_DIR}/${SET_NAME}_shard${SHARD}.jsonl" \
  --shard "${SHARD}" \
  --n-shards "${PROFILE_SHARDS}" \
  "${EXTRA[@]}"
