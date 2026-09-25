#!/bin/bash
#SBATCH --job-name=precheck_feas
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Pre-build check 2: LNR ring-closure feasibility frontier. CPU only, no GPU, no OpenMM.
#
# One shard of LNR complexes per array task, one output file each. NEVER a shared output
# file: concurrent O_APPEND from several jobs NUL-corrupts rows on this filesystem (88 of
# 144 lost, measured), and the aggregator's job is then to refuse the corrupt lines rather
# than skip them.
#
# Resumable: the shard skips complexes already present in its own output, so hitting the
# wall clock means resubmitting rather than restarting.

# In the job body, not only the preamble: if the source below fails, a job without this
# keeps running under the system python and exits 0, releasing every afterok dependent
# behind a stage that did no work.
set -euo pipefail

# A batch script cannot locate files relative to itself -- Slurm copies it to
# /var/spool/slurm/job<id>/slurm_script, so $(dirname "${BASH_SOURCE[0]}") is the spool
# directory and sourcing a shared preamble that way silently sources nothing.
REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv venv
_precheck_banner "feasibility frontier"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${FEAS_DIR}"

EXTRA=()
[[ "${FEAS_LIMIT}" != "0" ]] && EXTRA+=(--limit "${FEAS_LIMIT}")
[[ "${FEAS_MAX_SOLVES}" != "0" ]] && EXTRA+=(--max-solves "${FEAS_MAX_SOLVES}")
[[ -n "${FEAS_CHEMISTRIES}" ]] && EXTRA+=(--chemistries "${FEAS_CHEMISTRIES}")

[[ -f "${BRIDGE_WINDOWS}" ]] || {
  echo "FATAL: bridge windows not found: ${BRIDGE_WINDOWS}" >&2
  echo "These are MEASURED on CPSea natives by the Milestone 1.5 calibration. Without" >&2
  echo "them the bridged chemistries would depend on an asserted textbook window." >&2
  exit 1
}

python -u "${REPO}/script_utils/precheck_lnr_feasibility.py" \
  --config "${PRECHECK_CONFIG}" \
  --windows "${BRIDGE_WINDOWS}" \
  --out "${FEAS_DIR}/lnr_shard${SHARD}.jsonl" \
  --shard "${SHARD}" \
  --n-shards "${FEAS_SHARDS}" \
  "${EXTRA[@]}"

echo "shard ${SHARD} done: ${FEAS_DIR}/lnr_shard${SHARD}.jsonl"
