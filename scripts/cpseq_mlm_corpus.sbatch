#!/bin/bash
#SBATCH --job-name=cpseq_mlm_corpus
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_corpus_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_corpus_%A_%a.out
#
# Extract the (pocket sequence, peptide sequence, chemistry) corpus from CPSea. CPU only:
# it parses PDBs and computes one distance block per example, and asking for a GPU would
# just queue behind real work.
#
# ARRAY JOB, one shard per task, one OUTPUT FILE per shard. Concurrent O_APPEND writes to
# a shared file were measured on this filesystem to NUL-corrupt rows (88 of 144 lost), so
# shards never share output and the reader refuses a corrupt line rather than skipping it.
#
# Resumable because --requeue is on: a shard whose .done marker exists exits immediately.
#
# Do NOT sbatch directly -- use scripts/submit_cpseq_mlm.sh, which writes the env file
# this sources as $1. Config travels in a FILE passed positionally, never via
# `sbatch --export`, which triggers the user-env-retrieval requeue-hold on this cluster.

set -euo pipefail          # in the body: a failed source must not run under system python and exit 0

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

ENV_FILE="${1:-}"
[ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ] || { echo "FATAL: env file not found: ${ENV_FILE:-<none>}"; exit 1; }
source "$REPO/scripts/cpseq_mlm_env.sh"
source "$ENV_FILE"

SPLIT="${CPSEQ_MLM_SPLIT:?CPSEQ_MLM_SPLIT not set}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${CPSEQ_MLM_NUM_SHARDS}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
# Python block-buffers stdout when it is a file rather than a TTY, so a running job's log
# stays empty and the job looks hung. Costs nothing here; the script prints rarely.
export PYTHONUNBUFFERED=1

echo "=== cpseq_mlm_corpus ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_JOB_ID:-none}[$SHARD]"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M partition=${SLURM_JOB_PARTITION:-?}"
echo "split:     $SPLIT  shard $SHARD/$NSHARDS"
echo "out:       $CPSEQ_MLM_CORPUS_DIR"
echo "radius:    $CPSEQ_MLM_POCKET_RADIUS"

METADATA="$CPSEA_METADATA_DIR/cpsea_${SPLIT}.parquet"
if [ ! -f "$METADATA" ]; then
  echo "GATE FAILED: metadata parquet missing: $METADATA"
  echo "Nothing to extract. Exiting 0 so dependents can report the same gate rather than"
  echo "stranding in DependencyNeverSatisfied with no explanation."
  exit 0
fi

mkdir -p "$CPSEQ_MLM_CORPUS_DIR"
ARGS=(
  --metadata "$METADATA"
  --split "$SPLIT"
  --out-dir "$CPSEQ_MLM_CORPUS_DIR"
  --shard "$SHARD"
  --num-shards "$NSHARDS"
  --pocket-radius "$CPSEQ_MLM_POCKET_RADIUS"
  --workers "$CPSEQ_MLM_WORKERS"
)
# Row cap per shard. Only a smoke uses it: with NUM_SHARDS=1 a shard is the WHOLE split,
# and the train split is 2.44M rows (~13 core-hours), so a "smoke" without this is a
# half-day job that proves nothing a few hundred rows would not.
if [ -n "${CPSEQ_MLM_LIMIT:-}" ]; then
  ARGS+=(--limit "$CPSEQ_MLM_LIMIT")
  echo "limit:     $CPSEQ_MLM_LIMIT rows (smoke)"
fi
echo "cmd: $PY script_utils/cpseq_build_mlm_corpus.py ${ARGS[*]}"
"$PY" script_utils/cpseq_build_mlm_corpus.py "${ARGS[@]}"

echo "=== shard $SHARD done ==="
