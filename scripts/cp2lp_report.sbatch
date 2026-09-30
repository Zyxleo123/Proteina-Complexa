#!/bin/bash
#SBATCH --job-name=cp2lp_report
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU: the report reads exported JSONL rows and PDBs and draws a figure. Asking for a GPU
# to draw a figure just queues behind real work.
#
# Everything it needs is on disk, so the table and the figure can be re-rendered after a
# threshold changes without regenerating a single peptide.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "report"

cd "$REPO"
mkdir -p "$REPORT_DIR"

ARGS=()
for arm in gan contactonly; do
  d="${TRIPLET_ROOT}/${arm}"
  if _cp2lp_gate_generated "$d"; then
    ARGS+=(--generated "$d" --label "$arm")
  fi
done

if [[ "${#ARGS[@]}" -eq 0 ]]; then
  # Gate, not failure: neither arm has produced triplets yet. Exit 0 and say so.
  echo "GATE: no arm has produced triplets under ${TRIPLET_ROOT}. Nothing to report."
  exit 0
fi

if compgen -G "${REFERENCE_DIR}/**/real_reference*.jsonl" >/dev/null 2>&1 || \
   [[ -n "$(find "${REFERENCE_DIR}" -name 'real_reference*.jsonl' -size +0 2>/dev/null | head -1)" ]]; then
  ARGS+=(--real "$REFERENCE_DIR")
else
  echo "NOTE: no real reference under ${REFERENCE_DIR} -- the report runs without a real column."
fi

set -x
"$PY" script_utils/cp2lp_report.py "${ARGS[@]}" --out-dir "$REPORT_DIR"
