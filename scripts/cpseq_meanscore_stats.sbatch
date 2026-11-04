#!/bin/bash
#SBATCH --job-name=ms_stats
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_stats_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_stats_%j.out

# Stage 1: build the fold table, then measure which statistic over a sequence's ~10 folds
# is the most reliable target score. Everything downstream trains on what this picks.
#
# CPU partition on purpose: this only reads saved JSONL and does rank statistics.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

mkdir -p "${MEANSCORE_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo
echo "### 1. build the fold table"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_dataset.py" \
  --out-dir "${MEANSCORE_SOURCE}" \
  --scoring-version "${MEANSCORE_SCORING_VERSION}" \
  --arms "${MEANSCORE_ARMS}" \
  --agg mean \
  --min-folds "${MEANSCORE_MIN_FOLDS}" \
  --json "${MEANSCORE_DATASET}"

echo
echo "### 2. aggregator reliability sweep (the target-score decision)"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_reliability.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --json "${MEANSCORE_RELIABILITY}" \
  --k "${MEANSCORE_K_PER_HALF}" \
  --n-rep "${MEANSCORE_N_REP}"

echo
echo "### 3. hand-built feature baselines (the bar the network must clear)"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_baseline.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --json "${MEANSCORE_BASELINE}"

echo
echo "MEANSCORE_STATS_DONE $(date -Is)"
