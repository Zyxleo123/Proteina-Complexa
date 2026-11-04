#!/bin/bash
#SBATCH --job-name=ms_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_report_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_report_%j.out

# Stage 3: one table and one verdict over every trained config. Reads saved JSON only, so
# it is re-runnable at zero cost and the verdict is redrawable without retraining.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_report.py" \
  --dir "${MEANSCORE_DIR}" \
  --json "${MEANSCORE_DIR}/verdict.json"

echo
echo "MEANSCORE_REPORT_DONE $(date -Is)"
