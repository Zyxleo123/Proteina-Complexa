#!/bin/bash
# Submitter for the AE round-trip go/no-go. Calls sbatch and nothing else.
#
#   bash scripts/submit_ae_roundtrip.sh [--partition P] [--nodelist N] [--time T] [--smoke]
#
# Writes ONE timestamped env file per submission and passes its PATH as a positional
# argument to each batch script. Deliberately NOT `sbatch --export`: any explicit export
# list sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the
# compute node, and the job is requeued and HELD, stranding every dependent job.
#
# Stage 1 (GPU) scores both arms; stage 2 (CPU) aggregates and plots, chained afterok.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"
NODELIST="gpu28"
TIME="02:00:00"
SUM_TIME="00:20:00"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    --smoke)     SMOKE=1;        shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ae_roundtrip_${STAMP}"
[[ $SMOKE == 1 ]] && RUN_ID="ae_roundtrip_smoke_${STAMP}"

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_ae_roundtrip.sh at ${STAMP}. One file per submission.
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
LNR_ROOT=/zfsauton/scratch/yixiz/LNR
LNR_STAGE_DIR=${REPO}/CPSea_data/lnr_staged
CONFIG_NAME=example/training_cpsea_peptide_smoke
CONTROL_LIMIT=$([[ $SMOKE == 1 ]] && echo 8 || echo 0)
SEED=42
EOF
echo "env file: $ENV_FILE"
cat "$ENV_FILE"
echo

# Submitting from inside an allocation would otherwise leak this shell's SLURM_* into the
# child job's environment and confuse its resource view.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

GPU_ID=$(sbatch --parsable \
  --partition="$PARTITION" --nodelist="$NODELIST" --time="$TIME" \
  --job-name="${RUN_ID}" \
  scripts/ae_roundtrip.sbatch "$ENV_FILE")
echo "submitted GPU arm:  job $GPU_ID  (${PARTITION}/${NODELIST})"

SUM_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$SUM_TIME" \
  --dependency=afterok:"$GPU_ID" \
  --job-name="${RUN_ID}_sum" \
  scripts/ae_roundtrip_summarize.sbatch "$ENV_FILE")
echo "submitted CPU summary: job $SUM_ID  (afterok:$GPU_ID)"

echo
echo "results will land in: ${REPO}/evaluation_results/${RUN_ID}"
echo "watch:  tail -f slurm_logs/${RUN_ID}_${GPU_ID}.out"
