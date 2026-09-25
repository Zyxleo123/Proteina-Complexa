#!/bin/bash
#SBATCH --job-name=precheck_pin
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#
# Pre-build check 0: record the pinned SDEdit baseline and prove its files still exist.
# CPU only. The confirmation run is a GPU job and is PRINTED, not submitted -- GPU work here
# goes through sbatch after checking `myfree` for an available node.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv venv
_precheck_banner "pin baseline"

mkdir -p "${PIN_DIR}"
python -u "${REPO}/script_utils/precheck_pin_baseline.py" \
  --config "${PRECHECK_CONFIG}" --out-dir "${PIN_DIR}"
