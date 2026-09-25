#!/bin/bash
#SBATCH --job-name=precheck_feas_report
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Pre-build check 2, report stage. Reads only saved artefacts, so it is CPU-only and can be
# re-run to redraw the tables without recomputing a single frontier.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv venv
_precheck_banner "feasibility report"

# A gate that failed upstream must exit 0 with an explanation, not non-zero: a non-zero
# exit strands every dependent job in DependencyNeverSatisfied with nothing to read.
shopt -s nullglob
shards=("${FEAS_DIR}"/*.jsonl)
if (( ${#shards[@]} == 0 )); then
  echo "GATE: no frontier shards under ${FEAS_DIR}."
  echo "The upstream array wrote nothing. This is a gate failure, not a crash --"
  echo "exiting 0 so any dependent stage reports the same thing rather than vanishing."
  exit 0
fi
echo "found ${#shards[@]} shard file(s)"

python -u "${REPO}/script_utils/precheck_lnr_report.py" \
  --config "${PRECHECK_CONFIG}" \
  --feas-dir "${FEAS_DIR}" \
  --out-dir "${OUT_DIR}"

echo "report: ${OUT_DIR}/PRECHECK_TASK2_REPORT.md"
