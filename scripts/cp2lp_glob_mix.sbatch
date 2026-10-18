#!/bin/bash
#SBATCH --job-name=cp2lp_glob_mix
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 1 of the glob-dependence probe: build the held-out TWO-ROLE parquet.
#
# No existing split has both roles in one batch -- cp2lp_eval is cyclic-only, and
# cp2lp_eval_real_lp is linear-only -- and the only split that mixes them is the one the
# discriminator trained on. Metadata only, so this runs on the CPU partition: asking for a
# GPU to concatenate two parquets just queues behind real work.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "glob-mix"

echo "host     : ${SLURMD_NODENAME:-$(uname -n 2>/dev/null || echo '?')}"

GLOB_DIR="${GLOB_DIR:-${TRIPLET_ROOT}/glob_probe}"
GLOB_MIX="${GLOB_MIX:-${GLOB_DIR}/cp2lp_glob_mixed_heldout.parquet}"
GLOB_LINEAR_FRAC="${GLOB_LINEAR_FRAC:-0.25}"
GLOB_NO_STRATIFY="${GLOB_NO_STRATIFY:-0}"
GLOB_CP_SOURCE="${GLOB_CP_SOURCE:-${EVAL_META}}"

for f in "$GLOB_CP_SOURCE" "$EVAL_REAL_META"; do
  if [[ ! -f "$f" ]]; then
    echo "GATE: missing ${f}; build the splits first:"
    echo "      ${PY} script_utils/cp2lp_build_splits.py --mixed ${LPDATA}/mixed_train.parquet --out-dir ${LPDATA}"
    exit 0
  fi
done

cd "$REPO"
mkdir -p "$GLOB_DIR"

STRAT_FLAG=""
[[ "$GLOB_NO_STRATIFY" == "1" ]] && STRAT_FLAG="--no-stratify"

set -x
"$PY" script_utils/cp2lp_glob_mix.py \
  --cp-parquet "$GLOB_CP_SOURCE" \
  --lp-parquet "$EVAL_REAL_META" \
  --out "$GLOB_MIX" \
  --linear-frac "$GLOB_LINEAR_FRAC" \
  ${STRAT_FLAG}
