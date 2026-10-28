#!/usr/bin/env bash
#SBATCH --job-name=cpseq_ladder
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_ladder_%x_%A_%a.out
#
# The seed ladder: every random draw, and the unchanged cyclized peptide, folded at the
# SAME ten seeds.
#
# Why this run exists. Every earlier random-substitution run folded each candidate once.
# Boltz-2 returns a different structure for the same sequence on a different seed, and
# the spread is large: sd 3.67 REU on the computed energy, and a pass/fail check applied
# to a single fold is a coin flip at a ~44 % pass rate. Four results were reported from
# those single folds and then withdrawn (README_CPSEQ_PEPBENCH_SCALEUP sec. 3) -- in each
# case the apparent effect was the extra folds the designs got and the reference did not.
# One fold per candidate is a filter. It is not a measurement.
#
# So: nothing is promoted and nothing is filtered here. Each draw's result is its pass
# RATE over ten folds and its median energy over those folds, and the unchanged peptide
# is folded ten times too, so both sides of the comparison carry the same fold count.
#
# Resources are measured, not guessed:
#   - 75 s per fold mean (63 s median, 110 s p90), from 6,382 finished folds of the
#     PepBench run. The first fold in a process costs ~185 s because PyRosetta init and
#     the Boltz model load are paid once.
#   - 650 folds per complex (64 draws + 1 reference, x 10 seeds) is ~13.5 h, which is
#     LONGER than this 8 h limit on purpose: see the resume note below.
#   - 32G and 2 CPUs: identical to cpseq_search.sbatch, whose comment records why
#     (Boltz-2 MaxRSS 11.0 GiB measured; 48G/4-CPU sat PENDING on preempt for an hour).
#
# One array task per complex, and that is not negotiable: two processes appending to one
# complex's journal would interleave and corrupt rows that cost a GPU-hour to replace.
# Parallelism comes from the array, never from splitting a complex.
#
# A task that runs out of time or is preempted is RESUBMITTED, not restarted: every fold
# is appended to the complex's outcomes.jsonl keyed by (sequence, seed, stage, arm), so
# the next run skips what is already paid for. Two waves of 8 h finish a complex.
#
# The evaluation order is what makes an interrupted wave usable. Every draw is folded at
# seed rank k before any draw is folded at rank k+1, so at any cut-off the folds per
# draw differ by at most one. A half-finished run is the whole experiment at fewer
# folds, never a few draws at ten folds and the rest at none.
#
# Config comes from a positional env file or VAR=value arguments -- never
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets jobs held.
#
# Preflight: sbatch scripts/cpseq_ladder.sbatch PARENTS=<one_complex> LIMIT=3 SHARD_COUNT=1
# Run:       sbatch --array=0-19%20 scripts/cpseq_ladder.sbatch SHARD_COUNT=20
# Resume:    the same line again; finished folds are skipped.

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
SHARD_COUNT="${SHARD_COUNT:-1}"
# An array task id wins when present; otherwise an explicit SHARD_INDEX lets a single
# non-array job take a specific shard. Without the fallback every non-array submission
# claims shard 0, which with another shard running means two writers on one complex.
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
LIMIT="${LIMIT:-0}"
SAMPLES="${SAMPLES:-1}"
CHEMISTRY="${CHEMISTRY:-mainchain}"
PARENTS="${PARENTS:-}"
# Ten folds per draw is ten times the structures. `rescorable` keeps the gzipped raw
# complex, so a scoring-protocol change can still re-score without re-folding -- the one
# property worth paying storage for -- at ~50 KB per fold instead of ~1.31 MB.
RETENTION="${RETENTION:-rescorable}"
# Arm D. 0 (the default) folds the random draw stream; N folds the LLM proposals from
# rounds 1..N written by cpseq_llm_propose.py, which replaces the stream entirely. The
# rows are tagged arm=llm_search, so the two arms can share one OUT_DIR -- and should,
# because the reference folds are then shared and the comparison is properly paired.
LLM_ROUND="${LLM_ROUND:-0}"
# Overlap folding with scoring. A task folds every seed and only then scores, so the GPU
# sits idle for the whole scoring phase -- 51.5 % boltz duty cycle measured on the live
# corpus run. With this set, the next seed folds on a background thread while the current
# one is relaxed. It fills the same cache with the same inputs in the same order, so no
# result changes. Default off: the corpus run should adopt it deliberately.
PREFETCH_AHEAD="${PREFETCH_AHEAD:-0}"
# Predict and stop, writing no outcome rows. The scoring pass is the same plan run by
# `cpseq_ladder_score.sbatch`, which needs no GPU because every prediction is then a
# cache hit. Measured 14.3 s prediction vs 43.7 s relax per fold: without the split,
# three quarters of this GPU job's wall time is Rosetta with the card idle.
PREDICT_ONLY="${PREDICT_ONLY:-0}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
BOLTZ_PY="${BOLTZ_PYTHON:-/data/user_data/yixiz/envs/boltz2/.venv/bin/python}"
BOLTZ_CACHE="${BOLTZ_CACHE:-/data/user_data/yixiz/boltz_cache}"

echo "=== cpseq_ladder ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "shard:     ${SHARD_INDEX}/${SHARD_COUNT}  retention=$RETENTION  limit=$LIMIT"
echo "mode:      $([ "$PREDICT_ONLY" = "0" ] && echo 'predict+score' || echo 'PREDICT ONLY (score separately on CPU)')"
echo "prefetch:  $([ "$PREFETCH_AHEAD" = "0" ] && echo 'sequential (fold all, then score)' || echo 'ahead (fold next seed while scoring)')"
echo "arm:       $([ "$LLM_ROUND" = "0" ] && echo 'random_control (draw stream)' || echo "llm_search (proposal rounds 1..$LLM_ROUND)")"
echo "out_dir:   $OUT_DIR"
nvidia-smi -L || echo "no nvidia-smi"

EXTRA=()
[ -n "$PARENTS" ] && EXTRA+=(--parents "$PARENTS")
[ "$LLM_ROUND" != "0" ] && EXTRA+=(--llm-round "$LLM_ROUND")
[ "$PREFETCH_AHEAD" != "0" ] && EXTRA+=(--prefetch-ahead)
[ "$PREDICT_ONLY" != "0" ] && EXTRA+=(--predict-only)

# -u because this log is the only view into a task that runs for hours: Python block-
# buffers stdout when it is a file, so without it the per-fold lines sit in a 4 KB
# buffer and a running task looks identical to a hung one.
srun "$PY" -u script_utils/cpseq_search.py ladder \
  --out-dir "$OUT_DIR" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --chemistry "$CHEMISTRY" \
  --samples "$SAMPLES" \
  --retention "$RETENTION" \
  --limit "$LIMIT" \
  --boltz-python "$BOLTZ_PY" \
  --boltz-cache "$BOLTZ_CACHE" \
  "${EXTRA[@]}"

echo "=== cpseq_ladder OK (shard ${SHARD_INDEX}) ==="
