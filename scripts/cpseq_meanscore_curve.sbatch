#!/bin/bash
#SBATCH --job-name=ms_curve
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_curve_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_curve_%j.out

# Is the proxy data-hungry? A learning curve over the NUMBER OF TRAINING TARGETS, with
# target diversity and row volume separated -- they are confounded in the natural curve
# (5 targets is ~500 rows, 19 is ~1900) and they correspond to different things to go buy.
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
  echo "MEANSCORE_CURVE_SKIPPED $(date -Is)"
  exit 0
fi

"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_learning_curve.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --extra-features ${MEANSCORE_TRIAGE_FEATS} \
  --agg "${MEANSCORE_TRIAGE_AGG}" \
  --draws "${MEANSCORE_CURVE_DRAWS}" \
  --json "${MEANSCORE_CURVE}"

echo
echo "MEANSCORE_CURVE_DONE $(date -Is)"
