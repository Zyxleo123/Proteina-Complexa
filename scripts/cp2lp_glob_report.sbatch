#!/bin/bash
#SBATCH --job-name=cp2lp_glob_report
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 3: the sufficiency regressions and the markdown table, for every arm that produced
# rows. Reads saved JSONL only -- nothing here touches the model, so the tables can be
# re-rendered as often as the reading changes without re-running a GPU job.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "glob-report"

echo "host     : ${SLURMD_NODENAME:-$(uname -n 2>/dev/null || echo '?')}"

GLOB_DIR="${GLOB_DIR:-${TRIPLET_ROOT}/glob_probe}"
cd "$REPO"

shopt -s nullglob
ROWS=("${GLOB_DIR}"/*_rows.jsonl)
shopt -u nullglob

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "GATE: no *_rows.jsonl under ${GLOB_DIR} -- every probe job gated out or none ran."
  echo "      This is a gate, not a failure: exiting 0."
  exit 0
fi

rc=0
for rows in "${ROWS[@]}"; do
  tag="$(basename "$rows" _rows.jsonl)"
  summary="${GLOB_DIR}/${tag}.json"
  if [[ ! -f "$summary" ]]; then
    echo "SKIP ${tag}: rows present but no ${summary} (probe died after the dump?)"
    continue
  fi
  echo "---- ${tag}"
  # Per-arm exit codes are collected rather than fatal: one arm's corrupt rows should not
  # stop the others from being reported.
  "$PY" script_utils/cp2lp_glob_report.py \
    --rows "$rows" \
    --probe-summary "$summary" \
    --out "${GLOB_DIR}/${tag}_report.json" \
    --markdown "${GLOB_DIR}/${tag}_report.md" || rc=1
done

exit "$rc"
