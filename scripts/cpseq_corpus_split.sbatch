#!/bin/bash
#SBATCH --job-name=cpseq_corpus_split
#SBATCH --partition=cpu
#SBATCH --account=users
#SBATCH --qos=qos_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/logs/%x_%j.out
#
# Split the inventory's eligible.txt into cluster shares, grouped by receptor sequence so
# no receptor MSA is fetched on two clusters. CPU only: this reads PDBs to recover
# receptor hashes (the same parse the inventory ran), no GPU, no network.
#
# Config travels in a FILE passed positionally, never `sbatch --export` (which triggers
# the user-env-retrieval requeue-hold on this cluster).
#
# Usage:
#   sbatch scripts/cpseq_corpus_split.sbatch <env-file>

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
export PYTHONUNBUFFERED=1

SPLIT_ROOT="${SPLIT_ROOT:-$ZFS/LPData/PepBench/train_valid}"
SPLIT_ELIGIBLE="${SPLIT_ELIGIBLE:-$ZFS/cpseq/pepbench_train_inventory/eligible.txt}"
SPLIT_SOURCE="${SPLIT_SOURCE:-PepBench}"
SPLIT_OUT="${SPLIT_OUT:-$ZFS/cpseq/pepbench_train_split}"
SPLIT_NAMES="${SPLIT_NAMES:-autonlab babel}"

mkdir -p "$ZFS/cpseq/logs" "$SPLIT_OUT"

echo "=== $(date) $(hostname) job=${SLURM_JOB_ID:-none} ==="
echo "  requested: cpus=${SLURM_CPUS_PER_TASK:-?} mem=16G partition=${SLURM_JOB_PARTITION:-?}"
echo "  root=$SPLIT_ROOT eligible=$SPLIT_ELIGIBLE"
echo "  out=$SPLIT_OUT names=$SPLIT_NAMES"
[[ -f "$SPLIT_ELIGIBLE" ]] || { echo "FATAL: no eligible list at $SPLIT_ELIGIBLE"; exit 1; }

# shellcheck disable=SC2086
"$PY" script_utils/cpseq_corpus_split.py \
  --root "$SPLIT_ROOT" --eligible "$SPLIT_ELIGIBLE" --source "$SPLIT_SOURCE" \
  --out-dir "$SPLIT_OUT" --names $SPLIT_NAMES

echo "=== cpseq_corpus_split OK ==="
