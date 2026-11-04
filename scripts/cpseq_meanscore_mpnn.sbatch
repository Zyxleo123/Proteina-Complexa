#!/bin/bash
#SBATCH --job-name=ms_mpnn
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_mpnn_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_mpnn_%j.out

# Structure-aware features from pretrained ProteinMPNN, no folding.
#
# Rationale, from the feature baselines: post-fold readouts reach within-target rho 0.781
# against a ~0.80 ceiling while sequence-only features reach 0.222, so the scorer's
# missing input is structural context -- and it cannot fold, since avoiding the fold is
# the point. Each target's reference peptide has a native complex on disk, so candidates
# are threaded onto that backbone and scored by a model that sees the receptor.
#
# CPU partition: ProteinMPNN is ~1.7 M parameters.
#
# Three steps, cheapest first, each resumable, so a time limit never costs a finished one.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! -s "${MEANSCORE_DATASET}" ]]; then
  echo "GATE FAILED: no fold table at ${MEANSCORE_DATASET}"
  echo "  Stage 1 (cpseq_meanscore_stats.sbatch) produces it. Exiting 0 so dependents"
  echo "  report their own gate instead of vanishing into DependencyNeverSatisfied."
  echo "MEANSCORE_MPNN_SKIPPED $(date -Is)"
  exit 0
fi

echo
echo "### 1. resolve each target to a native complex + peptide chain"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_structures.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --json "${MEANSCORE_STRUCTURES}"

echo
echo "### 2. backbone marginals  log p(aa_i | backbone)  -- one pass per target"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_mpnn_feats.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --structures "${MEANSCORE_STRUCTURES}" \
  --json "${MEANSCORE_MPNN_BB}" \
  --model-name "${MEANSCORE_MPNN_MODEL}" \
  --n-draw-marginal "${MEANSCORE_MPNN_DRAW_MARGINAL}" \
  --skip-ar --resume

echo
echo "### 3. autoregressive score  p(s_i | s_<i, backbone)  -- one pass per candidate"
echo "    (on a 1-target probe this was NO better than the marginals: -0.092 vs -0.142,"
echo "     at ~25x the cost. Run anyway on all 20 so that is measured, not assumed.)"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_mpnn_feats.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --structures "${MEANSCORE_STRUCTURES}" \
  --json "${MEANSCORE_MPNN_AR}" \
  --model-name "${MEANSCORE_MPNN_MODEL}" \
  --n-draw-marginal "${MEANSCORE_MPNN_DRAW_MARGINAL}" \
  --n-draw-ar "${MEANSCORE_MPNN_DRAW_AR}" \
  --batch-size "${MEANSCORE_MPNN_BATCH}" \
  --resume

echo
echo "MEANSCORE_MPNN_DONE $(date -Is)"
