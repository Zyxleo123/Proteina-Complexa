#!/usr/bin/env bash
# Submit the filler temperature-sweep fold chain: predict (GPU) -> score (CPU) -> report.
#
# docs/README_CPSEQ_FOLD_PIPELINE_STANDARD.md is the decision record for the shape. The
# sweep already PROPOSED (scripts/cpseq_fill_temp_sweep.sbatch) and its cohort + per-arm
# per-parent proposals are on disk, so there is no prepare stage -- this chains the two
# fold passes and the readout.
#
# Why two passes. Scoring was 54% of a fold's wall time on `llm_ladder` and needs no
# card. The predict pass pays for Boltz-2 on an a6000 and writes no rows; the score pass
# pays for the PyRosetta relax on a `cpu` node (88 cores, uncontended against training),
# pooling SCORE_WORKERS processes per complex. Predictions key on `input_hash`, so this
# changes no result -- only who pays, and the queue position.
#
# Resume. ~3k folds from earlier waves are already on disk (t02 and the random arm,
# partial; t05/t08/t10 are empty -- the sweep's live question). `journal.has` skips what
# is paid for, so both passes compute only the missing folds.
#
# All configuration goes into ONE timestamped env file passed as a positional argument --
# never `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets jobs requeued+HELD.
#
#   bash scripts/submit_cpseq_fill_sweep_fold.sh
#   SKIP_INVALID_RELAX=1 bash scripts/submit_cpseq_fill_sweep_fold.sh   # drop invalid relaxes
#   DRY_RUN=1 bash scripts/submit_cpseq_fill_sweep_fold.sh             # print, submit nothing

set -euo pipefail

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
cd "$REPO"

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
OUT_DIR="${OUT_DIR:-${ZFS}/cpseq/llm_ladder_pot}"
PARENTS_FILE="${PARENTS_FILE:-${ZFS}/cpseq/sweep_t/fold_parents.txt}"
ARMS="${ARMS:-mpnn_fill_t02 mpnn_fill_t05 mpnn_fill_t08 mpnn_fill_t10 shared_mask_random_fill}"
GATE="${GATE:-v3-cis-amide-muco-windows}"

RUN_ID="${RUN_ID:-filltsweep_$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="${WORK_DIR:-${ZFS}/cpseq/sweep_t/${RUN_ID}}"
LOG_DIR="${ZFS}/cpseq_af3/logs"

# One upper-bound array; the jobs exit 0 for lines past the end of the parents file.
ARRAY_MAX="${ARRAY_MAX:-$(( $(grep -vcE '^[[:space:]]*(#|$)' "$PARENTS_FILE") - 1 ))}"
PRED_CONCURRENCY="${PRED_CONCURRENCY:-5}"   # GPU tasks at once; all general a6000 are busy
SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-10}"
SCORE_WORKERS="${SCORE_WORKERS:-16}"
SKIP_INVALID_RELAX="${SKIP_INVALID_RELAX:-0}"

mkdir -p "$WORK_DIR" "$LOG_DIR"

ENV_FILE="${WORK_DIR}/env.sh"
cat > "$ENV_FILE" <<EOF
# Written by submit_cpseq_fill_sweep_fold.sh at $(date -Is). Passed to each job as a
# positional argument; the job sources any argument that is a path.
OUT_DIR=${OUT_DIR}
PARENTS_FILE=${PARENTS_FILE}
ARMS="${ARMS}"
MASK_ROUND=1
CHEMISTRY=mainchain
RETENTION=rescorable
LIMIT=0
SCORE_WORKERS=${SCORE_WORKERS}
SKIP_INVALID_RELAX=${SKIP_INVALID_RELAX}
PYTHON_EXEC=${REPO}/.venv/bin/python
BOLTZ_PYTHON=${ZFS}/envs/boltz2/.venv/bin/python
BOLTZ_CACHE=${ZFS}/boltz_cache
EOF

echo "=== fill temperature-sweep fold chain ==="
echo "run id:    $RUN_ID"
echo "env file:  $ENV_FILE"
echo "out_dir:   $OUT_DIR"
echo "parents:   $PARENTS_FILE  ($((ARRAY_MAX + 1)) complexes)"
echo "arms:      $ARMS"
echo "predict:   general a6000, array 0-${ARRAY_MAX}%${PRED_CONCURRENCY}  (GPU, PREDICT_ONLY=1)"
echo "score:     cpu, array 0-${ARRAY_MAX}%${SCORE_CONCURRENCY}, ${SCORE_WORKERS} workers/task  (skip_inv=${SKIP_INVALID_RELAX})"
echo

# Submitting from inside an allocation leaks SLURM_* into the child sbatch; scrub it so
# each job creates a fresh context instead of inheriting this shell's.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v" || true; done

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 -- the three commands that would be submitted:"
  echo "  sbatch --parsable --partition=general --array=0-${ARRAY_MAX}%${PRED_CONCURRENCY} \\"
  echo "         scripts/cpseq_fill_sweep_fold.sbatch $ENV_FILE PREDICT_ONLY=1"
  echo "  sbatch --parsable --dependency=afterok:<pred> --array=0-${ARRAY_MAX}%${SCORE_CONCURRENCY} \\"
  echo "         scripts/cpseq_fill_sweep_score.sbatch $ENV_FILE"
  echo "  python script_utils/cpseq_fill_sweep_report.py --out-dir $OUT_DIR  (CPU readout, run by hand after score)"
  exit 0
fi

PRED=$(sbatch --parsable --partition=general \
  --array="0-${ARRAY_MAX}%${PRED_CONCURRENCY}" \
  scripts/cpseq_fill_sweep_fold.sbatch "$ENV_FILE" PREDICT_ONLY=1)
echo "predict:  $PRED  (general a6000, array 0-${ARRAY_MAX}%${PRED_CONCURRENCY}, GPU)"

# afterok on the whole predict array: scoring starts once every complex has its
# predictions. A missing prediction makes the score task exit 0 at its gate, so a
# partially-finished predict pass does not strand the chain.
SCORE=$(sbatch --parsable --dependency="afterok:${PRED}" \
  --array="0-${ARRAY_MAX}%${SCORE_CONCURRENCY}" \
  scripts/cpseq_fill_sweep_score.sbatch "$ENV_FILE")
echo "score:    $SCORE  (cpu, array 0-${ARRAY_MAX}%${SCORE_CONCURRENCY}, ${SCORE_WORKERS} workers/task)"

echo
echo "watch:  squeue -u \$USER -o '%.12i %.18j %.9P %.8T %.10M %R'"
echo "report (after score finishes, CPU, cheap):"
echo "  ${REPO}/.venv/bin/python script_utils/cpseq_fill_sweep_report.py --out-dir ${OUT_DIR}"
