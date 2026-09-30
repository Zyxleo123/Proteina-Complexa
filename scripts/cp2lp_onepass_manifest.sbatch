#!/bin/bash
#SBATCH --job-name=cp2lp_op_manifest
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Build the fixed evaluation manifest once. CPU: it reads a parquet and writes a smaller
# one.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "onepass-manifest"

cd "$REPO"
set -x
"$PY" script_utils/cp2lp_eval_manifest.py \
  --eval-parquet "$EVAL_META" \
  --out-parquet "${EVAL_MANIFEST:-${LPDATA}/cp2lp_eval64.parquet}" \
  --n-parents "$EVAL_N_CP" \
  --seeds "$EVAL_SEEDS"
