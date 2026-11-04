#!/bin/bash
# Submit the cpseq mean-dG proxy-scorer chain. This script only calls sbatch.
#
#   stats  (fold table + aggregator reliability + feature baselines)
#     |-- afterok --> train  (array over (aggregator, loss))
#                       |-- afterok --> report  (one table, one verdict)
#
# Configuration travels as a timestamped env FILE passed as a POSITIONAL argument, never
# through `sbatch --export`: an explicit export list sets SLURM_GET_USER_ENV=1, slurmd
# then fails to rebuild the login environment, and the job is requeued and HELD, stranding
# every dependent. Overrides: pass VAR=value to this script.
#
# Usage:
#   scripts/submit_cpseq_meanscore.sh                       # defaults
#   scripts/submit_cpseq_meanscore.sh MEANSCORE_EPOCHS=240  # override
#   STATS_ONLY=1 scripts/submit_cpseq_meanscore.sh          # just the statistics

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_DIR="${ZFS}/cpseq/meanscore/env"
LOG_DIR="${ZFS}/slurm_logs"
mkdir -p "${ENV_DIR}" "${LOG_DIR}" "${ZFS}/cpseq/meanscore"

ENV_FILE="${ENV_DIR}/meanscore_${STAMP}.env"
{
  echo "# cpseq meanscore submission ${STAMP}"
  echo "export REPO=${REPO}"
  for kv in "$@"; do
    [[ "$kv" == *=* ]] && echo "export ${kv}"
  done
} > "${ENV_FILE}"
echo "env file  ${ENV_FILE}"
cat "${ENV_FILE}"

# Number of array tasks comes from the one place the sweep is defined.
N_CONFIGS="$(
  # shellcheck disable=SC1090
  source "${REPO}/scripts/_cpseq_meanscore_preamble.sh" >/dev/null 2>&1
  echo "${MEANSCORE_N_CONFIGS}"
)"
LAST=$((N_CONFIGS - 1))

# Submitting from inside an allocation would otherwise leak SLURM_* into the child jobs.
for v in $(env | grep -oE '^SLURM_[A-Z_]+' || true); do unset "$v" || true; done

echo
echo "+ sbatch --parsable ${REPO}/scripts/cpseq_meanscore_stats.sbatch ${ENV_FILE}"
STATS_ID="$(sbatch --parsable "${REPO}/scripts/cpseq_meanscore_stats.sbatch" "${ENV_FILE}")"
echo "  stats   job ${STATS_ID}"

if [[ -n "${STATS_ONLY:-}" ]]; then
  echo
  echo "STATS_ONLY set; stopping after stage 1."
  echo "watch: tail -f ${LOG_DIR}/ms_stats_${STATS_ID}.out"
  exit 0
fi

echo "+ sbatch --parsable --dependency=afterok:${STATS_ID} --array=0-${LAST} ${REPO}/scripts/cpseq_meanscore_train.sbatch ${ENV_FILE}"
TRAIN_ID="$(sbatch --parsable --dependency="afterok:${STATS_ID}" --array="0-${LAST}" \
  "${REPO}/scripts/cpseq_meanscore_train.sbatch" "${ENV_FILE}")"
echo "  train   job ${TRAIN_ID}  (array 0-${LAST}, ${N_CONFIGS} configs)"

echo "+ sbatch --parsable --dependency=afterok:${TRAIN_ID} ${REPO}/scripts/cpseq_meanscore_report.sbatch ${ENV_FILE}"
REPORT_ID="$(sbatch --parsable --dependency="afterok:${TRAIN_ID}" \
  "${REPO}/scripts/cpseq_meanscore_report.sbatch" "${ENV_FILE}")"
echo "  report  job ${REPORT_ID}"

cat <<EOF

submitted
  stats   ${STATS_ID}
  train   ${TRAIN_ID}  (array 0-${LAST})
  report  ${REPORT_ID}

watch
  squeue -u \$USER
  tail -f ${LOG_DIR}/ms_stats_${STATS_ID}.out
  tail -f ${LOG_DIR}/ms_train_${TRAIN_ID}_0.out
  cat ${LOG_DIR}/ms_report_${REPORT_ID}.out

NOTE: sacct is dead here (slurmdbd unreachable, and it exits 0 with no rows), so a
failed array task is only visible in its own log. Each task prints a terminal marker --
MEANSCORE_TRAIN_DONE or MEANSCORE_TRAIN_SKIPPED -- so audit with:
  grep -l MEANSCORE_TRAIN_DONE ${LOG_DIR}/ms_train_${TRAIN_ID}_*.out | wc -l
EOF
