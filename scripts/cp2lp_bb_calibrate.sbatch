#!/bin/bash
#SBATCH --job-name=cp2lp_bb_calib
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Measure the backbone template library Arm B's torsion builder constructs chains with, and
# the template reconstruction FLOOR that its edit RMSDs have to be read against.
#
# CPU: it reads 101 native PDBs off disk and does arithmetic. Asking for a GPU to do this
# would only queue it behind real work.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "bb-calibrate"

cd "$REPO"

NATIVE_ROOT="${NATIVE_ROOT:-${REPO}/calibration_native}"
CALIB_OUT="${CALIB_OUT:-${TRIPLET_ROOT}/bb_template/backbone_template.json}"

if [[ ! -d "${NATIVE_ROOT}" ]]; then
  echo "GATE: no native calibration set at ${NATIVE_ROOT} -- nothing to measure."
  exit 0
fi

mkdir -p "$(dirname "${CALIB_OUT}")"

"${PY}" script_utils/cp2lp_bb_calibrate.py \
  --native-root "${NATIVE_ROOT}" \
  --report-floor \
  --out "${CALIB_OUT}"

echo "[done] template written to ${CALIB_OUT}"
