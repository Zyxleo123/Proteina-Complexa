#!/bin/bash
#SBATCH --job-name=ms_esm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_esm_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_esm_%j.out

# Does a frozen pretrained ESM-2 peptide encoder beat amino-acid composition at ranking
# mean dG? `cpseq_meanscore_esm_probe.py` has existed since 2026-10-28 and had never been
# run; the weights it defaults to (facebook/esm2_t33_650M_UR50D) were not on disk, which
# is most of why. They are now.
#
# This probe has NO target information by construction -- every peptide is embedded alone.
# That is the point: it measures the peptide encoder's own contribution, and the gap it
# leaves is the budget for a target encoder. Compare it to the ProteinMPNN features from
# cpseq_meanscore_mpnn.sbatch, which DO see the receptor.
#
# CPU partition: 2k peptides of 5-16 residues through a 650M encoder is a few minutes of
# compute, and the GPU partitions are contended by real work.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${ZFS}/hf"

if [[ ! -s "${MEANSCORE_DATASET}" ]]; then
  echo "GATE FAILED: no fold table at ${MEANSCORE_DATASET} (stage 1 produces it)."
  echo "MEANSCORE_ESM_SKIPPED $(date -Is)"
  exit 0
fi

ESM_MODEL="${MEANSCORE_ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"
echo
echo "### frozen ESM-2 probe: ${ESM_MODEL}"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_esm_probe.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --esm "${ESM_MODEL}" \
  --cache-dir "${ZFS}/hf/hub" \
  --embed-cache "${MEANSCORE_DIR}/esm_embed_cache.npz" \
  --json "${MEANSCORE_DIR}/esm_probe_v3.json"

echo
echo "MEANSCORE_ESM_DONE $(date -Is)"
