#!/bin/bash
#SBATCH --job-name=cpseq_corpus_inventory
#SBATCH --partition=cpu
#SBATCH --account=users
#SBATCH --qos=qos_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/logs/%x_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/logs/%x_%A_%a.out
#
# Inventory a corpus split through the real prepare filter: how many targets are
# eligible, why the rest are not, and how many DISTINCT receptor MSAs a staging run would
# have to fetch. Parse only -- no templates, no network, no GPU -- because those three
# numbers are what size the expensive stages, and they are obtainable without paying for
# them.
#
# CPU partition on purpose: this reads PDBs and computes pocket distances. Asking for a
# GPU to parse text would just queue behind real work.
#
# Two modes in one script, so the defaults live in one place:
#   array task  -> one shard of the split
#   INV_MERGE=1 -> combine every shard, write the summary and the eligible split list
#
# RESUMABLE: a shard whose JSON exists is skipped, so a time limit means resubmitting.
#
# Config travels in a FILE passed positionally, never `sbatch --export` (which triggers
# the user-env-retrieval requeue-hold on this cluster).
#
# Usage:
#   sbatch --array=0-19%20 scripts/cpseq_corpus_inventory.sbatch <env-file>
#   sbatch --dependency=afterok:<id> scripts/cpseq_corpus_inventory.sbatch <env-file> INV_MERGE=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
cd "$REPO"

# Positional args: a path is sourced, a VAR=value is exported. Never --export.
for a in "$@"; do
  if [[ -f "$a" ]]; then
    # shellcheck disable=SC1090
    source "$a"
  elif [[ "$a" == *=* ]]; then
    export "${a?}"
  fi
done

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
# Unbuffered, or a working job's .out stays empty for hours and looks hung -- and with
# sacct dead there is no other way to tell.
export PYTHONUNBUFFERED=1

INV_ROOT="${INV_ROOT:-$ZFS/LPData/PepBench/train_valid}"
INV_LIST="${INV_LIST:-train.txt}"
INV_SOURCE="${INV_SOURCE:-PepBench}"
INV_OUT="${INV_OUT:-$ZFS/cpseq/pepbench_train_inventory}"
INV_SHARDS="${INV_SHARDS:-20}"
INV_MERGE="${INV_MERGE:-0}"
# Every cache a receptor MSA might already sit in, so the fetch count reported is NET new.
INV_MSA_CACHES="${INV_MSA_CACHES:-$ZFS/cpseq/msa_cache $ZFS/cpseq/pepbench_valid/msa_cache $ZFS/cpseq/tranche1/msa_cache}"

mkdir -p "$ZFS/cpseq/logs" "$INV_OUT"

echo "=== $(date) $(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none} ==="
echo "  requested: cpus=${SLURM_CPUS_PER_TASK:-?} mem=16G partition=${SLURM_JOB_PARTITION:-?}"
echo "  root=$INV_ROOT list=$INV_LIST shards=$INV_SHARDS merge=$INV_MERGE"
echo "  out=$INV_OUT"
[[ -f "$INV_ROOT/$INV_LIST" ]] || { echo "FATAL: no split list at $INV_ROOT/$INV_LIST"; exit 1; }

if [[ "$INV_MERGE" == "1" ]]; then
  # shellcheck disable=SC2086
  "$PY" script_utils/cpseq_corpus_inventory.py \
    --root "$INV_ROOT" --list-file "$INV_LIST" --source "$INV_SOURCE" \
    --out-dir "$INV_OUT" --merge --msa-cache-dirs $INV_MSA_CACHES
  echo "=== cpseq_corpus_inventory MERGE OK ==="
else
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
  "$PY" script_utils/cpseq_corpus_inventory.py \
    --root "$INV_ROOT" --list-file "$INV_LIST" --source "$INV_SOURCE" \
    --out-dir "$INV_OUT" --shard-index "$IDX" --shard-count "$INV_SHARDS"
  echo "=== cpseq_corpus_inventory OK (shard $IDX) ==="
fi
