#!/usr/bin/env bash
#SBATCH --job-name=cpseq_arm_box
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/logs/cpseq_arm_box_%j.out
#
# Rank the proposal arms: one box per arm over every complex with a baseline, plus the
# winning arm's chosen sequence against the native in absolute REU.
#
# CPU only, and deliberately so: this reads journals already on disk and draws. Nothing
# is folded and nothing is re-scored, so asking for a GPU would only queue the figure
# behind real work. It stages its rows to JSON before drawing, so re-rendering costs
# nothing and never needs the run directory again:
#
#   $PY script_utils/cpseq_arm_box.py --from-rows <fig_dir>/<name>_rows.json \
#       --fig-dir <fig_dir>
#
# This is the cohort-level companion to scripts/cpseq_seed_box.sbatch. That one draws
# four hand-picked complexes in per-fold detail; four complexes cannot rank five arms,
# and this draws the quantity that can. Both read the same journals under the same gate,
# so if their orderings ever disagree, the pooled one here is the estimator's.
#
# Submit:  sbatch scripts/cpseq_arm_box.sbatch
# Config:  positional args -- a path to source, or VAR=value. Never `sbatch --export`
#          (an explicit export list sets SLURM_GET_USER_ENV=1, slurmd fails to rebuild
#          the login environment, and the job is requeued and HELD).

set -euo pipefail

for arg in "$@"; do
  if [ -f "$arg" ]; then echo "[cfg] sourcing $arg"; # shellcheck disable=SC1090
    . "$arg"
  elif [[ "$arg" == *=* ]]; then echo "[cfg] export $arg"; export "$arg"
  fi
done

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
CPSEQ="${CPSEQ:-$ZFS/cpseq}"
PY="${PY:-$REPO/.venv/bin/python}"

RUN_DIR="${RUN_DIR:-$CPSEQ/llm_ladder}"
SCORING_VERSION="${SCORING_VERSION:-v3-cis-amide-muco-windows}"
ARMS="${ARMS:-random_control,random_mask_mpnn_fill,llm_mask_mpnn_fill,llm_mask_random_fill,llm_search}"
FIG_DIR="${FIG_DIR:-$CPSEQ/figs/arm_box}"
FIG_NAME="${FIG_NAME:-arm_box}"
PANEL="${PANEL:-both}"

echo "=== cpseq_arm_box ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=cpu cpus=${SLURM_CPUS_PER_TASK:-2} mem=8G time=0:20:00"
nvidia-smi -L 2>/dev/null | head -2 || echo "gpu:      none (CPU-only, as intended)"
echo "run dir:  $RUN_DIR"
echo "gate:     $SCORING_VERSION"
echo "arms:     $ARMS"
echo "fig dir:  $FIG_DIR"

[ -x "$PY" ] || { echo "FATAL: no venv python at $PY" >&2; exit 1; }
[ -d "$RUN_DIR/state" ] || { echo "FATAL: no journals under $RUN_DIR/state" >&2; exit 1; }

mkdir -p "$FIG_DIR" "$CPSEQ/logs"
cd "$REPO"

echo "--- figure ---"
"$PY" script_utils/cpseq_arm_box.py \
  --out-dir "$RUN_DIR" \
  --arms "$ARMS" \
  --scoring-version "$SCORING_VERSION" \
  --fig-dir "$FIG_DIR" \
  --fig-name "$FIG_NAME" \
  --panel "$PANEL"

echo "[cpseq_arm_box] DONE"
