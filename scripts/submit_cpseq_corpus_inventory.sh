#!/bin/bash
# Inventory the PepBench train split through the real prepare filter, as an afterok chain.
#
#   A shards (CPU, array)  parse each slice of the split; eligible / rejected / receptors
#   B merge  (CPU, 1 task) combine, write inventory_summary.json + eligible.txt
#
# This submits ONLY the inventory. It fetches nothing and folds nothing: it exists to
# replace the extrapolated pool size (~3,540, scaled from a 1,000-target sample's 14.9%
# rejection rate) with a measured one, and to report how many DISTINCT receptor MSAs a
# staging run would have to fetch from api.colabfold.com before anyone commits to that.
#
# Config travels as a timestamped env FILE passed positionally, never `sbatch --export`.
#
# Usage:
#   bash scripts/submit_cpseq_corpus_inventory.sh
#   DRY_RUN=1 bash scripts/submit_cpseq_corpus_inventory.sh
#   INV_LIST=all.txt INV_OUT=<dir> bash scripts/submit_cpseq_corpus_inventory.sh

set -euo pipefail

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
INV_ROOT="${INV_ROOT:-$ZFS/LPData/PepBench/train_valid}"
INV_LIST="${INV_LIST:-train.txt}"
INV_SOURCE="${INV_SOURCE:-PepBench}"
INV_OUT="${INV_OUT:-$ZFS/cpseq/pepbench_train_inventory}"
INV_SHARDS="${INV_SHARDS:-20}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO"
[[ -f "$INV_ROOT/$INV_LIST" ]] || { echo "FATAL: no split list at $INV_ROOT/$INV_LIST"; exit 1; }
N=$(awk 'NF>=3' "$INV_ROOT/$INV_LIST" | wc -l | tr -d ' ')

ENV_DIR="$INV_OUT/_chain"
mkdir -p "$ENV_DIR" "$ZFS/cpseq/logs"
ENV_FILE="$ENV_DIR/inventory_$(date +%Y%m%d_%H%M%S).env"
cat > "$ENV_FILE" <<EOF
# Written by submit_cpseq_corpus_inventory.sh at $(date -Iseconds).
INV_ROOT=$INV_ROOT
INV_LIST=$INV_LIST
INV_SOURCE=$INV_SOURCE
INV_OUT=$INV_OUT
INV_SHARDS=$INV_SHARDS
PYTHON_EXEC=$REPO/.venv/bin/python
EOF

echo "split:  $INV_ROOT/$INV_LIST  ($N entries, >=3 fields)"
echo "shards: $INV_SHARDS  (~$(( (N + INV_SHARDS - 1) / INV_SHARDS )) entries each)"
echo "out:    $INV_OUT"
echo "env:    $ENV_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN -- nothing submitted."
  echo "  A: sbatch --array=0-$((INV_SHARDS-1))%${INV_SHARDS} scripts/cpseq_corpus_inventory.sbatch $ENV_FILE"
  echo "  B: sbatch --dependency=afterok:<A> scripts/cpseq_corpus_inventory.sbatch $ENV_FILE INV_MERGE=1"
  exit 0
fi

# Scrub SLURM_* so each child sbatch builds a fresh context instead of inheriting this
# shell's allocation (this is normally run from an interactive job on lov4).
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v" || true; done

A=$(sbatch --parsable --array="0-$((INV_SHARDS-1))%${INV_SHARDS}" \
      scripts/cpseq_corpus_inventory.sbatch "$ENV_FILE")
echo "A shards: $A"

B=$(sbatch --parsable --dependency="afterok:$A" \
      scripts/cpseq_corpus_inventory.sbatch "$ENV_FILE" INV_MERGE=1)
echo "B merge:  $B  (afterok:$A)"

echo
echo "chain: A=$A -> B=$B"
echo "summary will land at: $INV_OUT/inventory_summary.json"
echo "logs: $ZFS/cpseq/logs/cpseq_corpus_inventory_${A}_*.out"
