#!/bin/bash
#SBATCH --job-name=ms_triage
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_triage_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_triage_%j.out

# Stage 4: the deployment number. The sweep ends on a within-target Spearman rho; a search
# does not spend correlation, it spends folds at ~238 s each. This converts the winning
# ridge into "how many folds does it save", with a shuffled control that must come out at
# 1.0 or nothing else in the table counts.
#
# Reads saved JSON only (dataset + MPNN features), so it is re-runnable at zero cost and
# the verdict is redrawable without refolding or retraining anything.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

# Gate: without the dataset and the MPNN features there is nothing to triage. Print why
# and exit 0 -- a non-zero exit would strand every afterok dependent with no explanation.
missing=0
for f in "${MEANSCORE_DATASET}" ${MEANSCORE_TRIAGE_FEATS}; do
  if [[ ! -s "$f" ]]; then
    echo "GATE: missing or empty input: $f"
    missing=1
  fi
done
if (( missing )); then
  echo "GATE FAILED: upstream artefacts absent. Run the stats/mpnn stages first."
  echo "MEANSCORE_TRIAGE_SKIPPED $(date -Is)"
  exit 0
fi

echo "--- harness controls first: a shuffled selector must score ~1.0x ---"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_triage.py" --self-test

echo
echo "--- fold-budget triage on the real data ---"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_triage.py" \
  --dataset "${MEANSCORE_DATASET}" \
  --extra-features ${MEANSCORE_TRIAGE_FEATS} \
  --agg "${MEANSCORE_TRIAGE_AGG}" \
  --json "${MEANSCORE_TRIAGE}"

echo
echo "MEANSCORE_TRIAGE_DONE $(date -Is)"
