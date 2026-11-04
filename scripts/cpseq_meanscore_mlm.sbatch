#!/bin/bash
#SBATCH --job-name=ms_mlm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_mlm_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_mlm_%j.out

# Pocket-conditioned sequence features from the project's own fine-tuned PepMLM-650M.
#
# A 3-target probe already found the all-masked pocket marginal to be NULL (+0.021 mean
# within-target rho, against ProteinMPNN's -0.130), which matches the measured pocket
# ceiling of ~0.20 nats. This job exists so that null is measured on all 20 targets
# rather than asserted from three, and because `mlm_sub_logodds` -- the mutation-effect
# form, log p(mutant) - log p(wildtype) in the candidate's own context -- is a different
# feature from the marginal and has not been measured at all.
#
# CPU partition: the pocket crop is only ~40 residues, so a forward is ~55 tokens and the
# whole set is minutes even on CPU. The earlier worry about needing a GPU was based on
# feeding a full receptor, which is not what this model was trained on.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${ZFS}/hf"
# Shadow the venv's broken torchaudio: transformers 5.x reaches it on the path to every
# model class and the import dies with `undefined symbol: torch_library_impl`, because the
# extension was built against a different torch than the installed 2.7.0. Same shim the
# MLM training job uses.
export PYTHONPATH="${REPO}/script_utils/_compat_stubs${PYTHONPATH:+:$PYTHONPATH}"

MLM_CKPT="${MEANSCORE_MLM_CKPT:-${ZFS}/cpseq_mlm/runs/cpseq_mlm_pocket/checkpoints/hf_step60000}"
MLM_OUT="${MEANSCORE_MLM_FEATS:-${MEANSCORE_DIR}/mlm_feats_v3.json}"

for need in "${MEANSCORE_DATASET}" "${MEANSCORE_STRUCTURES}"; do
  if [[ ! -s "${need}" ]]; then
    echo "GATE FAILED: missing ${need}"
    echo "  dataset comes from cpseq_meanscore_stats.sbatch, structures from"
    echo "  cpseq_meanscore_mpnn.sbatch step 1. Exiting 0 so dependents report their own"
    echo "  gate instead of vanishing into DependencyNeverSatisfied."
    echo "MEANSCORE_MLM_SKIPPED $(date -Is)"
    exit 0
  fi
done
if [[ ! -d "${MLM_CKPT}" ]]; then
  echo "GATE FAILED: no MLM checkpoint at ${MLM_CKPT}"
  echo "MEANSCORE_MLM_SKIPPED $(date -Is)"
  exit 0
fi

echo
echo "### pocket-conditioned MLM features (marginals + substituted-position log-odds)"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_mlm_feats.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --structures "${MEANSCORE_STRUCTURES}" \
  --model "${MLM_CKPT}" \
  --json "${MLM_OUT}" \
  --batch-size "${MEANSCORE_MLM_BATCH:-8}" \
  --device cpu --resume

echo
echo "MEANSCORE_MLM_DONE $(date -Is)"
