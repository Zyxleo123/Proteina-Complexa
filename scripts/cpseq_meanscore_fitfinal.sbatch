#!/bin/bash
#SBATCH --job-name=ms_fitfinal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_fitfinal_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_fitfinal_%j.out

# Stage 5: fit the SHIPPING predictor on all 20 targets and save it as a usable artifact.
# The sweep's verdict is that the winner is a ridge on composition-delta + ProteinMPNN,
# not the mutation-effect network, so this fits that and writes coefficients rather than
# leaving the result as a table nobody can apply.
#
# Reads saved JSON only, so re-runnable at zero cost.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

missing=0
for f in "${MEANSCORE_DATASET}" ${MEANSCORE_TRIAGE_FEATS}; do
  if [[ ! -s "$f" ]]; then echo "GATE: missing or empty input: $f"; missing=1; fi
done
if (( missing )); then
  echo "GATE FAILED: upstream artefacts absent; run the stats/mpnn stages first."
  echo "MEANSCORE_FITFINAL_SKIPPED $(date -Is)"
  exit 0
fi

"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_fit_final.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --extra-features ${MEANSCORE_TRIAGE_FEATS} \
  --agg "${MEANSCORE_TRIAGE_AGG}" \
  --out "${MEANSCORE_MODEL}"

echo
echo "MEANSCORE_FITFINAL_DONE $(date -Is)"
