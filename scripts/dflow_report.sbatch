#!/bin/bash
#SBATCH --job-name=dflow_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU stage: join structure, sequence and energy into the contract's tables
# (script_utils/dflow_report.py). Reads only saved artefacts, so the tables can be redrawn
# without recomputing anything -- a threshold question or a new cut is a re-run of THIS job,
# not of the GPU stage.
#
# Do NOT sbatch directly -- use scripts/submit_dflow.sh.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/_dflow_preamble.sh" "$@"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
dflow_preflight_python

RESCORE_DIR="${RESCORE_DIR:-${RESULTS_DIR}/rescored}"
ROSETTA_DIR="${ROSETTA_DIR:-${RESULTS_DIR}/rosetta}"
REPORT_DIR="${REPORT_DIR:-${RESULTS_DIR}/report}"
mkdir -p "$REPORT_DIR"

# shellcheck disable=SC2206
RESCORED=( ${RESCORE_DIR}/rescored_*.jsonl )
if [[ ! -f "${RESCORED[0]}" ]]; then
  # A gate that failed upstream must say so and exit 0: a non-zero exit here would strand
  # nothing (this is the last stage), but the distinction between "gate failed" and
  # "upstream never ran" is the thing worth printing.
  echo "GATE FAILED: no rescored rows in ${RESCORE_DIR}."
  echo "Either the rescore stage did not run, or it produced nothing. Nothing to report."
  exit 0
fi

# shellcheck disable=SC2206
ROSETTA=( ${ROSETTA_DIR}/rosetta_shard*.jsonl )
ROS_ARG=()
if [[ -f "${ROSETTA[0]}" ]]; then
  ROS_ARG=(--rosetta "${ROSETTA[@]}")
else
  echo "note: no Rosetta rows in ${ROSETTA_DIR}; the energy clause will be unscored and"
  echo "      gate_joint_full will be false for every candidate. Structural tables stand."
fi

CAL_ARG=()
[[ -f "${RESULTS_DIR}/calibration.json" ]] && CAL_ARG=(--calibration "${RESULTS_DIR}/calibration.json")

# COST_SUMMARY lines carry the honest compute unit (network evaluations, wall seconds) and
# live only in the GPU jobs' logs.
# shellcheck disable=SC2206
COST_LOGS=( ${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_logs/dflow_opt_*.out )
COST_ARG=()
[[ -f "${COST_LOGS[0]}" ]] && COST_ARG=(--cost-log "${COST_LOGS[@]}")

"$DFLOW_PY" script_utils/dflow_report.py \
  --rescored "${RESCORED[@]}" \
  "${ROS_ARG[@]}" \
  "${CAL_ARG[@]}" \
  "${COST_ARG[@]}" \
  --out-dir "$REPORT_DIR" \
  --run-id "${RUN_ID:-dflow}" \
  --results-dir "$RESULTS_DIR"

echo
echo "DONE report -> ${REPORT_DIR}/RESULTS.md"
