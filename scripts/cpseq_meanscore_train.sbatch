#!/bin/bash
#SBATCH --job-name=ms_train
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_train_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_train_%A_%a.out

# Stage 2: train the mutation-effect scorer, one array task per (aggregator, loss).
#
# CPU partition on purpose: the model is ~23k parameters over ~2k rows. A GPU would only
# queue behind real work.
#
# This job is a DECISION GATE as well as a trainer: if stage 1 never produced the fold
# table, it says so and exits 0 rather than non-zero, because a non-zero exit would
# strand every afterok dependent in DependencyNeverSatisfied with no explanation.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
CFG="$(meanscore_config_for_task "${TASK}")"
if [[ -z "${CFG}" ]]; then
  echo "GATE: array task ${TASK} has no configured (agg, loss) pair."
  echo "      ${MEANSCORE_N_CONFIGS} configs are defined; nothing to do."
  echo "MEANSCORE_TRAIN_SKIPPED $(date -Is)"
  exit 0
fi
AGG="$(awk '{print $1}' <<<"${CFG}")"
LOSS="$(awk '{print $2}' <<<"${CFG}")"

if [[ ! -s "${MEANSCORE_DATASET}" ]]; then
  echo "GATE FAILED: no fold table at ${MEANSCORE_DATASET}"
  echo "  This means stage 1 (cpseq_meanscore_stats.sbatch) did not produce it."
  echo "  Distinguish from 'upstream never ran': check for an ms_stats_*.out log."
  echo "  Exiting 0 so dependents report their own gate rather than vanishing."
  echo "MEANSCORE_TRAIN_SKIPPED $(date -Is)"
  exit 0
fi

OUT="${MEANSCORE_DIR}/scorer_${AGG}_${LOSS}${MEANSCORE_TAG}.json"

# Optional pretrained structure-aware features. Checked here so a missing feature file is
# a named gate rather than an argparse error 20 minutes into the queue.
EXTRA_ARGS=()
if [[ -n "${MEANSCORE_EXTRA}" ]]; then
  for f in ${MEANSCORE_EXTRA}; do
    if [[ ! -s "$f" ]]; then
      echo "GATE FAILED: requested extra-feature file is missing or empty: $f"
      echo "  Produced by cpseq_meanscore_mpnn.sbatch. Exiting 0 so dependents report"
      echo "  their own gate rather than vanishing."
      echo "MEANSCORE_TRAIN_SKIPPED $(date -Is)"
      exit 0
    fi
  done
  EXTRA_ARGS=(--extra-features ${MEANSCORE_EXTRA})
  echo "extra features: ${MEANSCORE_EXTRA}"
fi

# Resumable: an array task that already wrote its result is not repaid on a resubmit.
if [[ -s "${OUT}" ]]; then
  echo "already done: ${OUT} exists, skipping (delete it to force a rerun)"
  echo "MEANSCORE_TRAIN_DONE $(date -Is)"
  exit 0
fi

echo
echo "### task ${TASK}: agg=${AGG} loss=${LOSS}"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_scorer.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --json "${OUT}" \
  --agg "${AGG}" \
  --loss "${LOSS}" \
  --protocol "${MEANSCORE_PROTOCOL}" \
  --seeds "${MEANSCORE_SEEDS}" \
  --epochs "${MEANSCORE_EPOCHS}" \
  --device cpu \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo
echo "MEANSCORE_TRAIN_DONE $(date -Is)"
