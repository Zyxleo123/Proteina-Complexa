#!/usr/bin/env bash
#SBATCH --job-name=cpseq_search
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_search_%x_%A_%a.out
#
# Bounded cyclic-peptide sequence search: arms A and B, then fresh-seed confirmation.
#
# Why this job holds a GPU *and* does PyRosetta, when the rest of the project splits
# them: the search is closed-loop. Round n+1's proposals are a function of round n's
# scores, so prediction and scoring must alternate inside one process. The batch
# pipeline's separation (`cpseq_predict.sbatch` GPU / `cpseq_score.sbatch` CPU) exists so
# a rescore never re-predicts, and that property is preserved here by the two separate
# caches rather than by two separate jobs.
#
# Resources are measured, not guessed:
#   - 55 s per evaluation marginal (boltz-2 1 sample + the full Rosetta protocol),
#     measured on lnr_1elr over three fresh evaluations after warm-up. The FIRST
#     evaluation in a process costs ~185 s because PyRosetta init and the boltz model
#     load are paid once, so a job that evaluates 2 candidates looks 3x slower per
#     candidate than one that evaluates 60.
#   - 32G and 2 CPUs: identical to cpseq_predict.sbatch, whose comment records why
#     (boltz-2 MaxRSS 11.0 GiB measured; 48G/4-CPU sat PENDING on preempt for an hour).
#   - 4:00:00 per parent: ~65 evaluations at 55 s is ~1 h, and the longest parent
#     (lnr_6seo, 16-mer on a large receptor) is slower per evaluation. The margin is for
#     that, not for a stall.
#
# One array task per parent. Parents are independent -- no proposal crosses a parent --
# so this is the natural shard, and a preempted task resumes only its own parent.
#
# Resumable, which preempt's REQUEUE requires: every evaluation is appended to that
# parent's outcomes.jsonl and keyed by (sequence, seed, stage, arm), so a requeued task
# skips what it already paid for instead of double-counting it in the yield.
#
# Config comes from a positional env file or VAR=value arguments -- never
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets jobs held.
#
# Submit:    sbatch --array=0-5 scripts/cpseq_search.sbatch MODE=screen SHARD_COUNT=6
# Preflight: sbatch scripts/cpseq_search.sbatch MODE=screen SHARD_COUNT=6 LIMIT=2
# Confirm:   sbatch --array=0-5 scripts/cpseq_search.sbatch MODE=confirm SHARD_COUNT=6
# Seeds:     sbatch --array=0-5 --time=6:00:00 scripts/cpseq_search.sbatch MODE=confirm \
#              SHARD_COUNT=6 SEEDS=101,202,303,404,505,606,707,808,909,111,222,333,444,555,666,777

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

MODE="${MODE:-screen}"
OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/cpseq/search_pilot}"
SHARD_COUNT="${SHARD_COUNT:-1}"
# An array task id wins when present; otherwise an explicit SHARD_INDEX lets a single
# non-array job take a specific shard. Without the fallback every non-array submission
# claims shard 0, which with another shard running means two writers on one parent.
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
LIMIT="${LIMIT:-0}"
ARMS="${ARMS:-guided_search,random_control}"
SAMPLES="${SAMPLES:-1}"
CHEMISTRY="${CHEMISTRY:-mainchain}"
PARENTS="${PARENTS:-}"
# What to keep per prediction once scored, and whether arm A needs the guided arm.
# Both default to the pilot's behaviour so an existing submission is unchanged.
RETENTION="${RETENTION:-full}"
RANDOM_POLICY="${RANDOM_POLICY:-match_guided}"
# Confirmation seeds. The default (empty) uses the script's CONFIRM_SEEDS = 101,202,303.
# The 16-seed set exists because 3 seeds cannot size the arm differences: measured
# seed-to-seed spread on this cohort is 4.3 REU median, 21.97 REU max.
SEEDS="${SEEDS:-}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
BOLTZ_PY="${BOLTZ_PYTHON:-/data/user_data/yixiz/envs/boltz2/.venv/bin/python}"
BOLTZ_CACHE="${BOLTZ_CACHE:-/data/user_data/yixiz/boltz_cache}"

echo "=== cpseq_search ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "mode:      $MODE  shard=${SHARD_INDEX}/${SHARD_COUNT}  arms=$ARMS  seeds=${SEEDS:-default}"
echo "retention: $RETENTION  random_policy: $RANDOM_POLICY"
echo "out_dir:   $OUT_DIR"
nvidia-smi -L || echo "no nvidia-smi"

EXTRA=()
[ -n "$PARENTS" ] && EXTRA+=(--parents "$PARENTS")
if [ "$MODE" = "screen" ]; then
  EXTRA+=(--arms "$ARMS" --limit "$LIMIT" --random-policy "$RANDOM_POLICY")
fi
# Only screen and confirm evaluate structures, so only they accept a retention tier;
# report would reject the flag.
if [ "$MODE" = "screen" ] || [ "$MODE" = "confirm" ]; then
  EXTRA+=(--retention "$RETENTION")
fi
if [ "$MODE" = "confirm" ] && [ -n "$SEEDS" ]; then
  EXTRA+=(--seeds "$SEEDS")
fi

srun "$PY" script_utils/cpseq_search.py "$MODE" \
  --out-dir "$OUT_DIR" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --chemistry "$CHEMISTRY" \
  --samples "$SAMPLES" \
  --boltz-python "$BOLTZ_PY" \
  --boltz-cache "$BOLTZ_CACHE" \
  "${EXTRA[@]}"

echo "=== cpseq_search OK ($MODE shard ${SHARD_INDEX}) ==="
