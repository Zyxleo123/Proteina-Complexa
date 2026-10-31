#!/usr/bin/env bash
#SBATCH --job-name=cpseq_rescore
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_rescore_%x_%A_%a.out
#
# Re-score folds already on disk under a chosen scoring protocol. CPU ONLY -- no
# --gres, so it runs under preempt_cpu_qos and never queues behind a GPU.
#
# Why this is not `cpseq_ladder_score.sbatch`
# -------------------------------------------
# That script runs `ladder`, whose journal dedupes on (sequence, seed, stage, arm) --
# a key that carries NO scoring version. Re-running it after a protocol change reports
# every fold as already done and rescores nothing. `cpseq_rescore_folds.py` walks the
# journal instead and re-evaluates with `force_score=True`, which re-reads the retained
# raw structure, redoes the relax and the gate, and appends a row under the new version.
# Predictions are a cache hit, so no GPU is touched.
#
# ONE ARRAY TASK PER COMPLEX, and that is not negotiable: the journal is per complex and
# two processes appending to it would interleave rows. Parallelism comes from the array.
#
# SCORING_VERSION picks the protocol. Naming an OLDER version is the normal case: when
# an arm was scored under a newer protocol than its baseline, re-scoring the arm DOWN to
# the baseline's version is cheaper than re-scoring the baseline up, and it is the same
# like-for-like comparison either way. Folds already carrying the target version are
# skipped, so the cost is only the rows that actually differ.
#
# Scores are cached per (scoring_version, input_hash) and the journal is append-only, so
# nothing is overwritten: every protocol's verdict stays on disk and a reader picks one
# by filtering on `scoring_version`.
#
# Resources: FastRelax is single-threaded, so 2 CPUs is one for the relax and one for
# the driver. Measured 43.7 s of relax per fold, so a 300-fold complex is about 3.7 h.
#
# Preflight: sbatch scripts/cpseq_rescore.sbatch PARENTS=lnr_1elr LIMIT=3
# Run:       sbatch --array=0-19%20 scripts/cpseq_rescore.sbatch SCORING_VERSION=v3-cis-amide-muco-windows
# Resume:    the same line again; folds already at the target version are skipped.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi

OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/cpseq/llm_ladder}"
SCORING_VERSION="${SCORING_VERSION:-}"
CHEMISTRY="${CHEMISTRY:-mainchain}"
LIMIT="${LIMIT:-0}"
ARMS="${ARMS:-}"
PARENTS="${PARENTS:-}"
MAX_DRAW_INDEX="${MAX_DRAW_INDEX:--1}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"

# One parent per array task, read from the cohort in its recorded order so a task id
# means the same complex across submissions. An explicit PARENTS overrides it, which is
# what the preflight uses.
if [ -z "$PARENTS" ]; then
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
  PARENTS=$("$PY" - "$OUT_DIR" "$IDX" <<'EOF'
import json, sys
from pathlib import Path
cohort = json.loads((Path(sys.argv[1]) / "cohort.json").read_text())
ids = [e["parent_id"] if isinstance(e, dict) else e for e in cohort["cohort"]]
i = int(sys.argv[2])
print(ids[i] if i < len(ids) else "")
EOF
)
  if [ -z "$PARENTS" ]; then
    echo "array task ${SLURM_ARRAY_TASK_ID:-0} is past the end of the cohort; nothing to do"
    exit 0
  fi
fi

echo "=== cpseq_rescore ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=none"
echo "out_dir:   $OUT_DIR"
echo "parents:   $PARENTS"
echo "version:   ${SCORING_VERSION:-<current default>}"

EXTRA=()
[ -n "$SCORING_VERSION" ] && EXTRA+=(--scoring-version "$SCORING_VERSION")
[ -n "$ARMS" ] && EXTRA+=(--arms "$ARMS")
[ "$LIMIT" != "0" ] && EXTRA+=(--limit "$LIMIT")
[ "$MAX_DRAW_INDEX" != "-1" ] && EXTRA+=(--max-draw-index "$MAX_DRAW_INDEX")

# -u because this log is the only view into a task that runs for hours.
srun "$PY" -u script_utils/cpseq_rescore_folds.py \
  --out-dir "$OUT_DIR" \
  --parents "$PARENTS" \
  --chemistry "$CHEMISTRY" \
  "${EXTRA[@]}"

echo "=== cpseq_rescore OK ($PARENTS) ==="
