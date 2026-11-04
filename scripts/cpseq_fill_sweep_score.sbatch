#!/usr/bin/env bash
#SBATCH --job-name=cpseq_fill_tscore
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_fill_tscore_%x_%A_%a.out
#
# The scoring half of the filler temperature sweep. CPU ONLY -- no --gres, deliberately.
# See docs/README_CPSEQ_FOLD_PIPELINE_STANDARD.md sec. 1 and 1a.
#
# cpseq_fill_sweep_fold.sbatch PREDICT_ONLY=1 pays for the Boltz-2 predictions on an
# a6000 and stops. This pays for the PyRosetta relaxes on a `cpu` node (88 cores,
# lov[1-2,5-9], uncontended against training), which on `llm_ladder` was 54% of a fold's
# wall time -- more than half of a "GPU" run was the card idle behind a relax.
#
# Changes NO result. `evaluate` keys predictions by `input_hash` and scores by
# (scoring_version, input_hash), so predict+score fills the same two caches as one job.
# If a prediction is MISSING the writer scores it, which without a GPU fails -- the right
# failure: it says the predict pass did not finish, not that a different set was scored.
#
# PER-COMPLEX PARALLELISM is the point. The shard axis in cpseq_search.py is the PARENT
# list, so this one-complex-per-task array gets nothing from sharding; parallelism comes
# from --score-workers, where worker processes SCORE and the parent WRITES. The journal
# is per complex and concurrent O_APPEND to it NUL-corrupts rows on this cluster, so a
# single writer is mandatory; the workers return records over a pipe. Only predictions
# already on disk are handed to workers (a pool cannot serialise evaluate.GPU_LOCK across
# processes), which is why this is a scoring-pass tool paired with PREDICT_ONLY=1.
#
# 16 CPUs / 64G because PyRosetta is ~1.5 GB per worker on a 422-residue pose; memory
# caps this, not cores. The pool does not plateau for any reason that carries to
# production -- raise --cpus-per-task and --mem together to go past 16.
#
# Arms score SERIALLY inside one task, same as the predict pass: every arm appends to the
# SAME complex journal, and the shared `reference` is scored once (first arm) then
# cache-hit. All 5 arms share one mask set; only the filler/temperature differs.
#
# SKIP_INVALID_RELAX=1 drops the relax on folds the v3 gate already failed (~60.9% of
# folds, ~58% of scoring hours). OFF by default: a skipped row carries no refined energy,
# deformation proxy or prerelax/postrelax gap, so it cannot be rescored to v4/v5. The
# sweep report is a v3 read and does not use those fields, but that is a judgement about
# THIS analysis, so it is opted into per submission and recorded on every row.
#
# Resumable: folds are appended keyed by (sequence, seed, stage, arm); re-run the same
# array line to resume.
#
# Submit (after the predict pass):
#   sbatch --array=0-9%10 scripts/cpseq_fill_sweep_score.sbatch \
#       PARENTS_FILE=/zfsauton/scratch/yixiz/cpseq/sweep_t/fold_parents.txt
#   ... SKIP_INVALID_RELAX=1     # to drop invalid-fold relaxes

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi

OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/cpseq/llm_ladder_pot}"
ARMS="${ARMS:-mpnn_fill_t02 mpnn_fill_t05 mpnn_fill_t08 mpnn_fill_t10 shared_mask_random_fill}"
MASK_ROUND="${MASK_ROUND:-1}"
CHEMISTRY="${CHEMISTRY:-mainchain}"
RETENTION="${RETENTION:-rescorable}"
LIMIT="${LIMIT:-0}"
SCORE_WORKERS="${SCORE_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
SKIP_INVALID_RELAX="${SKIP_INVALID_RELAX:-0}"
PARENTS_FILE="${PARENTS_FILE:-/zfsauton/scratch/yixiz/cpseq/sweep_t/fold_parents.txt}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
BOLTZ_PY="${BOLTZ_PYTHON:-/zfsauton/scratch/yixiz/envs/boltz2/.venv/bin/python}"
BOLTZ_CACHE="${BOLTZ_CACHE:-/zfsauton/scratch/yixiz/boltz_cache}"

echo "=== cpseq_fill_sweep_score (CPU) ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "out_dir:   $OUT_DIR"
echo "arms:      $ARMS"
echo "workers:   $SCORE_WORKERS"
echo "skip_inv:  $SKIP_INVALID_RELAX"

[ -f "$PARENTS_FILE" ] || { echo "FATAL: no PARENTS_FILE at $PARENTS_FILE" >&2; exit 1; }
IDX="${SLURM_ARRAY_TASK_ID:-0}"
PARENT=$(grep -v '^[[:space:]]*#' "$PARENTS_FILE" | grep -v '^[[:space:]]*$' \
         | sed -n "$((IDX + 1))p")
if [ -z "$PARENT" ]; then
  echo "array task $IDX is past the end of $PARENTS_FILE -- nothing to score, exiting 0"
  exit 0
fi
echo "parent:    $PARENT  (line $((IDX + 1)) of $PARENTS_FILE)"

if [ ! -d "$OUT_DIR/predictions/$PARENT" ]; then
  echo
  echo "GATE: no predictions under $OUT_DIR/predictions/$PARENT -- the predict pass never"
  echo "      ran for this complex. Exiting 0 so the chain's state stays legible rather"
  echo "      than looking like a crash. Run cpseq_fill_sweep_fold.sbatch PREDICT_ONLY=1"
  echo "      first."
  exit 0
fi

EXTRA=()
[ "$SKIP_INVALID_RELAX" != "0" ] && EXTRA+=(--skip-invalid-relax)

# Arms score SERIALLY -- one writer on this parent's journal. No --predict-only and no
# --prefetch-ahead: this pass scores, with every prediction already on disk.
for ARM in $ARMS; do
  echo "=== [$PARENT] scoring arm $ARM ==="
  srun "$PY" -u script_utils/cpseq_search.py ladder \
    --out-dir "$OUT_DIR" \
    --parents "$PARENT" \
    --mask-arm "$ARM" --mask-round "$MASK_ROUND" \
    --chemistry "$CHEMISTRY" \
    --samples 1 \
    --retention "$RETENTION" \
    --limit "$LIMIT" \
    --score-workers "$SCORE_WORKERS" \
    --boltz-python "$BOLTZ_PY" \
    --boltz-cache "$BOLTZ_CACHE" \
    "${EXTRA[@]}"
done

echo "=== cpseq_fill_sweep_score OK ($PARENT) ==="
