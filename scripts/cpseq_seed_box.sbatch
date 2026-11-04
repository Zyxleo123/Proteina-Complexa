#!/usr/bin/env bash
#SBATCH --job-name=cpseq_seed_box
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/logs/cpseq_seed_box_%j.out
#
# Draw the per-seed box plots behind the cross-fitted selected-candidate number.
#
# CPU only, and deliberately so: this reads journals already on disk and draws. Nothing
# is folded and nothing is re-scored, so asking for a GPU would only queue the figure
# behind real work. It also writes its rows to JSON before drawing, so re-rendering the
# figure costs nothing and never needs the run directory again:
#
#   $PY script_utils/cpseq_seed_box.py --from-rows <fig_dir>/<name>_rows.json \
#       --fig-dir <fig_dir>
#
# The estimator drawn is `cpseq_selected_candidate_report.py --pilot` -- the one that
# chose the arm, i.e. --draws all --directions 2 --min-ref-valid 3
# --energy-key refined_dG_separated. CHECK_REPORT points at that report's own --json so
# every delta in the figure is asserted equal to the number it illustrates; a figure that
# silently reimplements the selection rule is how the two drift apart.
#
# Submit:  sbatch scripts/cpseq_seed_box.sbatch
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
ARM="${ARM:-random_control}"
SCORING_VERSION="${SCORING_VERSION:-v3-cis-amide-muco-windows}"
# Four complexes, chosen for what they show rather than for how they read: the two that
# sit on the pooled -3.65 REU median (lnr_1t5z -3.57, pepbench_A_E_pdb6wmq -3.60), one
# mid-range (lnr_5y59 -5.65) and the largest win on the cohort (pepbench_C_A_pdb2jz3
# -16.48). All four have a reference at >= 3 valid folds AND a measurement in both
# cross-fit directions, so no panel rests on half the estimator.
COMPLEXES="${COMPLEXES:-lnr_1t5z,pepbench_A_E_pdb6wmq,lnr_5y59,pepbench_C_A_pdb2jz3}"
FIG_DIR="${FIG_DIR:-$CPSEQ/figs/seed_box}"
FIG_NAME="${FIG_NAME:-seed_box_${ARM}}"
# The pilot report's JSON for this same arm, run on the same run dir and gate. Written by
# the REPORT_JSON step below when absent, so one submission is self-sufficient.
REPORT_JSON="${REPORT_JSON:-$FIG_DIR/${ARM}_pilot.json}"

echo "=== cpseq_seed_box ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=cpu cpus=${SLURM_CPUS_PER_TASK:-2} mem=8G time=0:20:00"
nvidia-smi -L 2>/dev/null | head -2 || echo "gpu:      none (CPU-only, as intended)"
echo "run dir:  $RUN_DIR"
echo "arm:      $ARM"
echo "gate:     $SCORING_VERSION"
echo "cplx:     $COMPLEXES"
echo "fig dir:  $FIG_DIR"

[ -x "$PY" ] || { echo "FATAL: no venv python at $PY" >&2; exit 1; }
[ -d "$RUN_DIR/state" ] || { echo "FATAL: no journals under $RUN_DIR/state" >&2; exit 1; }

mkdir -p "$FIG_DIR"
cd "$REPO"

# The number the figure illustrates, recomputed here rather than quoted from a doc, so
# the assertion below compares two live computations on the same journals.
if [ ! -s "$REPORT_JSON" ]; then
  echo "--- pilot report (the number being illustrated) ---"
  "$PY" script_utils/cpseq_selected_candidate_report.py \
    --out-dir "$RUN_DIR" \
    --scoring-version "$SCORING_VERSION" \
    --arm "$ARM" \
    --pilot --per-complex \
    --json "$REPORT_JSON"
fi

echo "--- figure ---"
"$PY" script_utils/cpseq_seed_box.py \
  --out-dir "$RUN_DIR" \
  --arm "$ARM" \
  --scoring-version "$SCORING_VERSION" \
  --complexes "$COMPLEXES" \
  --fig-dir "$FIG_DIR" \
  --fig-name "$FIG_NAME" \
  --check-against "$REPORT_JSON"

echo "[cpseq_seed_box] DONE"
