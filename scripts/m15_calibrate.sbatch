#!/bin/bash
#SBATCH --job-name=m15_calib
#SBATCH --output=logs/m15_calib_%j.out
#SBATCH --error=logs/m15_calib_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Bridge distance windows, measured on CPSea natives with CONECT-resolved linkages.
# Every ceiling downstream is stated against these windows, so asserting them from
# textbook chemistry instead would make the whole audit rest on a guess.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_m15_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_m15_preamble.sh"

_m15_consume_args "$@"
_m15_defaults
_m15_banner calibrate
_m15_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

mkdir -p "$OUT_DIR" "$REPO/logs"
EXTRA=()
[[ "$SMOKE" == "1" ]] && EXTRA=(--metadata "$REPO/CPSea_data/preprocessed_sample100/metadata/cpsea_train.parquet")

cd "$REPO"
"$PY" script_utils/m15_ceiling_audit.py calibrate \
  --config "$M15_CONFIG" \
  --out "$WINDOWS_JSON" \
  "${EXTRA[@]}"

echo "[calibrate] done -> $WINDOWS_JSON"
