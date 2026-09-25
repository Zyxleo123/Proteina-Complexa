#!/bin/bash
#SBATCH --job-name=precheck_decoy_report
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Pre-build check 1, stage D: the regression report. Reads saved artefacts only, so the
# tables redraw without rescoring a single pose.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv venv
_precheck_banner "decoy calibration report"

shopt -s nullglob
scored=("${DECOY_DIR}"/scored_shard*.jsonl)
if (( ${#scored[@]} == 0 )); then
  echo "GATE: no scored shards under ${DECOY_DIR}. Exiting 0."
  exit 0
fi

python -u "${REPO}/script_utils/precheck_decoy_report.py" \
  --config "${PRECHECK_CONFIG}" --score-dir "${DECOY_DIR}" --out-dir "${OUT_DIR}"
