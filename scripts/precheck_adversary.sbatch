#!/bin/bash
#SBATCH --job-name=precheck_adversary
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Pre-build checks 3b (reference distributions), 3c (adversarial harness) and 3e (OOD).
# Reads only the profile shards, so it is CPU-only and re-runnable without recomputing a
# single feature.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_precheck_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
source "${REPO}/scripts/_precheck_preamble.sh"

_precheck_consume_args "$@"
_precheck_defaults
_precheck_venv venv
_precheck_banner "reference distributions + adversary + OOD"

shopt -s nullglob
shards=("${PROFILE_DIR}"/*.jsonl)
if (( ${#shards[@]} == 0 )); then
  echo "GATE: no profile shards under ${PROFILE_DIR}."
  echo "The upstream profile array wrote nothing. Exiting 0 so this reads as a gate"
  echo "failure rather than stranding dependents in DependencyNeverSatisfied."
  exit 0
fi
echo "found ${#shards[@]} profile shard(s)"

python -u "${REPO}/script_utils/precheck_profile.py" reference \
  --config "${PRECHECK_CONFIG}" \
  --profile-dir "${PROFILE_DIR}" \
  --out-dir "${OUT_DIR}"

# The positive control (cpsea vs pepbench) runs first and gates the rest; it is configured
# in the YAML, not passed here. The extra pairs are the comparisons of interest.
python -u "${REPO}/script_utils/precheck_adversary.py" \
  --config "${PRECHECK_CONFIG}" \
  --rows "${OUT_DIR}/profile_rows.parquet" \
  --out-dir "${OUT_DIR}" \
  --pairs "${ADVERSARY_PAIRS:-cpsea:lnr lnr:pepbench}"

python -u "${REPO}/script_utils/precheck_task3_report.py" \
  --config "${PRECHECK_CONFIG}" \
  --out-dir "${OUT_DIR}"

echo "report: ${OUT_DIR}/PRECHECK_TASK3_REPORT.md"
