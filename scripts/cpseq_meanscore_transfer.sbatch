#!/bin/bash
#SBATCH --job-name=ms_transfer
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_transfer_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_transfer_%j.out

# Transfer arm stage 2: does the proxy survive a change of prediction protocol?
# Reads saved JSON only, so re-runnable at zero cost.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

# Distinguish "upstream gate declined" from "upstream never ran": the first is a result,
# the second is a broken chain, and a bare missing file looks identical.
if [[ ! -s "${MEANSCORE_POT_DATASET}" ]]; then
  echo "GATE: ${MEANSCORE_POT_DATASET} absent."
  echo "      If ms_potfeats printed MEANSCORE_POTFEATS_SKIPPED, the transfer arm has no"
  echo "      data and this is the expected outcome. If it printed nothing, the chain is"
  echo "      broken -- read its .out before concluding anything."
  echo "MEANSCORE_TRANSFER_SKIPPED $(date -Is)"
  exit 0
fi
for f in "${MEANSCORE_POT_MPNN_BB}" "${MEANSCORE_POT_MPNN_AR}" "${MEANSCORE_DATASET}"; do
  if [[ ! -s "$f" ]]; then
    echo "GATE: missing $f"
    echo "MEANSCORE_TRANSFER_SKIPPED $(date -Is)"
    exit 0
  fi
done

"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_transfer.py" \
  --train-dataset "${MEANSCORE_DATASET}" \
  --train-features "${MEANSCORE_MPNN_BB}" "${MEANSCORE_MPNN_AR}" \
  --test-dataset "${MEANSCORE_POT_DATASET}" \
  --test-features "${MEANSCORE_POT_MPNN_BB}" "${MEANSCORE_POT_MPNN_AR}" \
  --agg "${MEANSCORE_TRIAGE_AGG}" \
  --json "${MEANSCORE_TRANSFER}"

echo
echo "MEANSCORE_TRANSFER_DONE $(date -Is)"
