#!/usr/bin/env bash
#SBATCH --job-name=cpseq_ladder_score
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_ladder_score_%x_%A_%a.out
#
# The scoring half of the ladder: relax and gate structures that are already predicted.
# CPU ONLY -- no --gres, so it runs under preempt_cpu_qos and never queues behind a GPU.
#
# Why this is a separate job from the folding
# -------------------------------------------
# Profiled per fold: 14.3 s of Boltz-2 prediction on the GPU against 43.7 s of PyRosetta
# relax on one core. Run together, three quarters of a GPU job's wall time was Rosetta
# with the card idle -- 19.2 CPU-h of the 25.9 "GPU-h" the ES pilot billed. Splitting
# them moves that load to a queue with no GPU contention and leaves the card doing only
# what needs it.
#
# This changes NO result. `evaluate.py` keys predictions by `input_hash` and scores by
# `(scoring_version, input_hash)`, so the same plan run twice fills the same two caches
# in the same way; the only difference is which job pays. Run
# `cpseq_ladder.sbatch PREDICT_ONLY=1` first, then this with the same OUT_DIR, shard
# layout and LLM_ROUND.
#
# If a prediction is missing, `evaluate` will try to make one and fail without a GPU.
# That is the right failure: it says the predict pass did not finish, rather than
# silently scoring a different set of folds. Re-run the predict pass and then this.
#
# Resources:
#   - 3 CPUs, and WORKERS=3 sub-shards running side by side in the one job. FastRelax
#     is single-threaded, so a 1-worker job leaves two thirds of its allowance idle --
#     and the per-user cap on preempt_cpu_qos is 20 JOBS but 64 CPUs, so one relax per
#     job uses 20 of the 64 CPUs we are allowed. Three workers per job reaches 60 and
#     triples scoring throughput: 1,648 -> 4,943 folds/h, which is what makes depth 10
#     on a 160-complex corpus fit four days instead of needing eleven.
#   - 24G: ~1.5 GB of PyRosetta per worker on a 422-residue pose, times three, plus the
#     database and headroom.
#   - 8 h: at 43.7 s/fold a shard of ~150 folds needs ~2 h; the margin is for the larger
#     PepBench receptors. Resumable either way -- the journal skips scored folds, so a
#     requeue costs only the fold it was in the middle of.
#
# Usage:
#   sbatch --array=0-2%3 scripts/cpseq_ladder_score.sbatch OUT_DIR=<run> SHARD_COUNT=3
#   sbatch --array=0-2%3 scripts/cpseq_ladder_score.sbatch OUT_DIR=<run> SHARD_COUNT=3 LLM_ROUND=2

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

OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/cpseq/seed_ladder}"
# SHARD_COUNT is the number of JOBS; each runs WORKERS sub-shards in parallel, so the
# work is split SHARD_COUNT*WORKERS ways. Sub-shard w of job k is global shard
# k*WORKERS+w, which keeps every complex with exactly one writer -- the property the
# per-complex journals depend on.
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-3}}"
LIMIT="${LIMIT:-0}"
SAMPLES="${SAMPLES:-1}"
CHEMISTRY="${CHEMISTRY:-mainchain}"
PARENTS="${PARENTS:-}"
RETENTION="${RETENTION:-full}"
LLM_ROUND="${LLM_ROUND:-0}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
# Still passed, and still unused: with every prediction a cache hit, the adapter is
# never invoked. Kept so the two passes are configured identically and a missing
# prediction fails loudly in the adapter rather than silently somewhere else.
BOLTZ_PY="${BOLTZ_PYTHON:-/data/user_data/yixiz/envs/boltz2/.venv/bin/python}"
BOLTZ_CACHE="${BOLTZ_CACHE:-/data/user_data/yixiz/boltz_cache}"

echo "=== cpseq_ladder_score (CPU only) ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=none"
echo "shard:     ${SHARD_INDEX}/${SHARD_COUNT}  retention=$RETENTION  limit=$LIMIT"
echo "arm:       $([ "$LLM_ROUND" = "0" ] && echo 'random_control (draw stream)' || echo "llm_search (proposal rounds 1..$LLM_ROUND)")"
echo "out_dir:   $OUT_DIR"
echo "====================================="

EXTRA=()
[ -n "$PARENTS" ] && EXTRA+=(--parents "$PARENTS")
[ "$LLM_ROUND" != "0" ] && EXTRA+=(--llm-round "$LLM_ROUND")

# No --predict-only and no --prefetch-ahead: this pass scores, and there is nothing to
# overlap with when every prediction is already on disk.
#
# The sub-shards run as plain background processes rather than as `srun` steps: several
# concurrent steps inside one allocation contend for the job's task slots and serialise
# unless each is given --exclusive, which is more moving parts than this needs. Each is
# an ordinary single-threaded Python process writing to its own complexes.
GLOBAL_COUNT=$(( SHARD_COUNT * WORKERS ))
echo "sub-shards: $WORKERS of $GLOBAL_COUNT global shard(s), starting at $(( SHARD_INDEX * WORKERS ))"

pids=()
for w in $(seq 0 $(( WORKERS - 1 ))); do
  gidx=$(( SHARD_INDEX * WORKERS + w ))
  "$PY" -u script_utils/cpseq_search.py ladder \
    --out-dir "$OUT_DIR" \
    --shard-index "$gidx" \
    --shard-count "$GLOBAL_COUNT" \
    --chemistry "$CHEMISTRY" \
    --samples "$SAMPLES" \
    --retention "$RETENTION" \
    --limit "$LIMIT" \
    --boltz-python "$BOLTZ_PY" \
    --boltz-cache "$BOLTZ_CACHE" \
    "${EXTRA[@]}" 2>&1 | sed "s/^/[w$w] /" &
  pids+=($!)
done

# Fail the job if ANY sub-shard failed, so a dependent stage does not start on a
# partially scored generation. Without this the shell reports only the last pipeline.
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
[ "$rc" -eq 0 ] || { echo "a sub-shard failed"; exit 1; }

echo "=== cpseq_ladder_score OK (shard ${SHARD_INDEX}) ==="
