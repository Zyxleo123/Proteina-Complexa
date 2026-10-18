#!/bin/bash
#SBATCH --job-name=cc_report
#SBATCH --output=logs/cc_report_%j.out
#SBATCH --error=logs/cc_report_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Tables, figures and the verdict, from whatever landed on disk. Reads only JSONL, so the
# figures re-render without re-running any minimization.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_corruption_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_corruption_preamble.sh"

_cc_consume_args "$@"
_cc_defaults
_cc_banner "report"
_cc_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

cd "$REPO"
mkdir -p "$REPORT_DIR"

"$PY" script_utils/corruption_ceiling_report.py \
  --ladder-glob "${LADDER_DIR}/ladder_shard*.jsonl" \
  --rosetta-glob "${ROSETTA_DIR}/dg_shard*.jsonl" \
  --out-dir "$REPORT_DIR" \
  --shortfall-tol-A "$SHORTFALL_TOL_A" \
  --retention-min "$RETENTION_MIN" \
  --min-heavy-A "$MIN_HEAVY_A" \
  --dg-tol "$DG_TOL"

echo "[report] done -> $REPORT_DIR"
