#!/usr/bin/env bash
#SBATCH --job-name=cpseq_predict
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_predict_%x_%A_%a.out
#
# Stage 1: structure prediction for candidate sequences, one predictor per submission.
#
# Memory is 48G on purpose. The interactive node-hold job this was developed in has
# 8 GiB, and boltz-2 was OOM-killed there at exactly that ceiling (cgroup oom_kill=2,
# memory.peak 8589303808) after successfully processing its inputs. 48G is the first
# request with real headroom over that; trim it once `sacct -o MaxRSS` says what the
# arm actually peaks at.
#
# The log goes to /home because /data/user_data is not mounted on the login node and
# Slurm resolves --output before the job body runs.
#
# Resumable, which preempt's REQUEUE requires: every task writes its own record.json
# and a task with a record is skipped, so a requeued job continues instead of restarting.
#
# Config comes from a positional env file or VAR=value arguments -- never
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets jobs held.
#
# Submit:   sbatch scripts/cpseq_predict.sbatch PREDICTOR=boltz2
# Preflight: sbatch scripts/cpseq_predict.sbatch PREDICTOR=boltz2 LIMIT=1
# Array:    sbatch --array=0-7 scripts/cpseq_predict.sbatch PREDICTOR=boltz2 SHARD_COUNT=8

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

set -a; source "$REPO/.env"; set +a

: "${PREDICTOR:?PREDICTOR not set (boltz2 | afcycdesign | highfold3)}"
DATASET_DIR="${DATASET_DIR:-$ZFS/cpseq/tranche1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SAMPLES="${SAMPLES:-4}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-0}"
CHEMISTRIES="${CHEMISTRIES:-}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"

echo "=== cpseq_predict ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=${SLURM_JOB_GRES:-?}"
echo "predictor: $PREDICTOR  shard=${SHARD_INDEX}/${SHARD_COUNT}  samples=$SAMPLES  limit=$LIMIT"
echo "dataset:   $DATASET_DIR"
nvidia-smi -L || echo "WARNING: no GPU visible"

ARGS=(
  --dataset-dir "$DATASET_DIR"
  --predictor "$PREDICTOR"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
  --samples "$SAMPLES"
  --seed "$SEED"
  --limit "$LIMIT"
  --af2-params-dir "$AF2_DIR"
)
[ -n "$CHEMISTRIES" ] && ARGS+=(--chemistries "$CHEMISTRIES")
# FORCE=1 re-runs tasks that already have a record.json. Needed after an adapter fix,
# because the default skip-if-recorded is what makes the job requeue-safe.
[ "${FORCE:-0}" = "1" ] && ARGS+=(--force)

srun "$PY" script_utils/cpseq_predict.py "${ARGS[@]}"

echo "=== cpseq_predict OK ==="
