#!/usr/bin/env bash
# Split the PepBench eligible set into receptor-grouped cluster shares (autonlab | babel).
# One CPU job. Submits only -- no work of its own. Config travels as a timestamped env
# file passed positionally (never `sbatch --export`).
#
# Usage:
#   bash scripts/submit_cpseq_corpus_split.sh
#   DRY_RUN=1 bash scripts/submit_cpseq_corpus_split.sh
#   SPLIT_NAMES="autonlab babel" SPLIT_OUT=<dir> bash scripts/submit_cpseq_corpus_split.sh

set -euo pipefail

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
cd "$REPO"

SPLIT_ROOT="${SPLIT_ROOT:-$ZFS/LPData/PepBench/train_valid}"
SPLIT_ELIGIBLE="${SPLIT_ELIGIBLE:-$ZFS/cpseq/pepbench_train_inventory/eligible.txt}"
SPLIT_SOURCE="${SPLIT_SOURCE:-PepBench}"
SPLIT_OUT="${SPLIT_OUT:-$ZFS/cpseq/pepbench_train_split}"
SPLIT_NAMES="${SPLIT_NAMES:-autonlab babel}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$SPLIT_ELIGIBLE" ]] || { echo "FATAL: no eligible list at $SPLIT_ELIGIBLE"; exit 1; }
N=$(awk 'NF>=3' "$SPLIT_ELIGIBLE" | wc -l | tr -d ' ')

ENV_DIR="$SPLIT_OUT/_chain"
mkdir -p "$ENV_DIR" "$ZFS/cpseq/logs"
ENV_FILE="$ENV_DIR/split_$(date +%Y%m%d_%H%M%S).env"
cat > "$ENV_FILE" <<EOF
# Written by submit_cpseq_corpus_split.sh at $(date -Iseconds).
SPLIT_ROOT=$SPLIT_ROOT
SPLIT_ELIGIBLE=$SPLIT_ELIGIBLE
SPLIT_SOURCE=$SPLIT_SOURCE
SPLIT_OUT=$SPLIT_OUT
SPLIT_NAMES="$SPLIT_NAMES"
PYTHON_EXEC=$REPO/.venv/bin/python
EOF

echo "eligible: $SPLIT_ELIGIBLE  ($N entries)"
echo "out:      $SPLIT_OUT"
echo "names:    $SPLIT_NAMES"
echo "env:      $ENV_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN -- would submit:"
  echo "  sbatch scripts/cpseq_corpus_split.sbatch $ENV_FILE"
  exit 0
fi

# Scrub SLURM_* so the child sbatch builds a fresh context instead of inheriting this
# shell's allocation (this is normally run from an interactive job on lov4).
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v" || true; done

sbatch scripts/cpseq_corpus_split.sbatch "$ENV_FILE"
