#!/bin/bash
#SBATCH --job-name=cp2lp_splits
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU: build the receptor-family-disjoint train/eval metadata. Pure pandas over a parquet.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "splits"

MIXED="${MIXED_META:-${LPDATA}/mixed_train.parquet}"
[[ -f "$MIXED" ]] || { echo "FATAL: missing $MIXED" >&2; exit 1; }

cd "$REPO"
set -x
"$PY" script_utils/cp2lp_build_splits.py \
  --mixed "$MIXED" \
  --out-dir "$LPDATA" \
  --eval-frac "${EVAL_FRAC:-0.15}" \
  --seed "${SPLIT_SEED:-0}" \
  --max-cp-train "${MAX_CP_TRAIN:-200000}"
