#!/usr/bin/env bash
#SBATCH --job-name=cpseq_score
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_score_%x_%A_%a.out
#
# Stage 2: score predicted complexes with the shared evaluator. CPU only -- PyRosetta
# does all the work here, so this runs under preempt_cpu_qos and never competes for a
# GPU with stage 1.
#
# Resumable per SAMPLE: each shard appends to its own rows_<predictor>_<shard>.jsonl and
# skips (task_id, sample_index) pairs already in it. One writer per file, so a requeue
# cannot interleave two writers into a corrupt line.
#
# Config via positional env file or VAR=value arguments -- never `sbatch --export`.
#
# Submit:  sbatch scripts/cpseq_score.sbatch PREDICTOR=boltz2
# Array:   sbatch --array=0-3 scripts/cpseq_score.sbatch PREDICTOR=boltz2 SHARD_COUNT=4

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

: "${PREDICTOR:?PREDICTOR not set}"
DATASET_DIR="${DATASET_DIR:-$ZFS/cpseq/tranche1}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
LIMIT="${LIMIT:-0}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"

echo "=== cpseq_score ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M"
echo "predictor: $PREDICTOR  shard=${SHARD_INDEX}/${SHARD_COUNT}"

srun "$PY" script_utils/cpseq_score.py \
  --dataset-dir "$DATASET_DIR" \
  --predictor "$PREDICTOR" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --limit "$LIMIT"

echo "=== cpseq_score OK ==="
