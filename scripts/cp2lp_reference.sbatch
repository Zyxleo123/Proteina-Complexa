#!/bin/bash
#SBATCH --job-name=cp2lp_reference
#SBATCH --partition=general
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Measure REAL bound linear peptides from the held-out receptor families, through the same
# AE round trip and the same scoring code that the generated peptides go through. This is
# the report's real column; a comparison run through two measurement paths measures the
# paths as much as it measures the peptides.
#
# Needs a GPU only for the AE encode/decode, and only on ~500 short peptides.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "reference"
_cp2lp_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

RUN_DIR="${STORE}/${RUN_NAME}"
if [[ ! -d "${RUN_DIR}/checkpoints" ]]; then
  echo "GATE: no checkpoints under ${RUN_DIR} -- need a run dir for its exp_config. Nothing measured."
  exit 0
fi
if [[ ! -f "${EVAL_REAL_META}" ]]; then
  echo "GATE: ${EVAL_REAL_META} missing -- run cp2lp_build_splits.py. Nothing measured."
  exit 0
fi

cd "$REPO"
mkdir -p "$REFERENCE_DIR"

set -x
"$PY" script_utils/cp2lp_generate.py \
  --run-dir "$RUN_DIR" \
  --cp-parquet "$EVAL_REAL_META" \
  --out-dir "$REFERENCE_DIR" \
  --reference-mode \
  --n-cp -1 \
  --batch-size "$GEN_BATCH" \
  --device cuda
