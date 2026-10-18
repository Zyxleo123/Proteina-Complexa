#!/bin/bash
#SBATCH --job-name=dflow_rescore
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU stage: re-score every arm's written complexes through ONE acceptance instrument
# (script_utils/dflow_rescore.py). No GPU -- this only reads PDBs off disk, and asking for a
# GPU to parse files just queues behind real work.
#
# This is what makes the comparison legitimate: the D-Flow rows, the SDEdit rows and the
# guided-SDEdit rows are all judged here by the same `accept()` with the same thresholds,
# rather than by whichever function each arm happened to write its own columns with.
#
# Do NOT sbatch directly -- use scripts/submit_dflow.sh.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/_dflow_preamble.sh" "$@"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
dflow_preflight_python

RESCORE_DIR="${RESCORE_DIR:-${RESULTS_DIR}/rescored}"
mkdir -p "$RESCORE_DIR"

# Arms to re-score. One space-separated entry per arm, fields separated by "|":
#
#     <arm name>|<glob of rows jsonl>[|<stage>]
#
# The D-Flow arm is always present; baseline arms are added by the submitter from
# BASELINE_ARMS so the same instrument reads the older SDEdit runs. A glob that matches
# nothing is FATAL here rather than a silently empty arm -- an arm missing from the report
# because its path was wrong looks exactly like an arm that failed.
ARMS="${RESCORE_ARMS:-dflow|${RESULTS_DIR}/dflow_shard*.jsonl}"

for ARM in ${ARMS}; do
  IFS="|" read -r A_NAME A_GLOB A_STAGE <<< "$ARM"
  [[ -n "${A_NAME:-}" && -n "${A_GLOB:-}" ]] || { echo "FATAL: malformed arm: $ARM"; exit 1; }

  # shellcheck disable=SC2206
  A_FILES=( $A_GLOB )
  [[ -f "${A_FILES[0]}" ]] || { echo "FATAL: arm '${A_NAME}' matched no file: ${A_GLOB}"; exit 1; }

  echo
  echo "=== arm ${A_NAME}: ${#A_FILES[@]} file(s)${A_STAGE:+ stage=${A_STAGE}}"
  STAGE_ARG=()
  [[ -n "${A_STAGE:-}" && "$A_STAGE" != "-" ]] && STAGE_ARG=(--stage "$A_STAGE")

  "$DFLOW_PY" script_utils/dflow_rescore.py \
    --rows "${A_FILES[@]}" \
    --arm "$A_NAME" \
    --out "${RESCORE_DIR}/rescored_${A_NAME}.jsonl" \
    --limit "${RESCORE_LIMIT:-0}" \
    --max-seconds "${RESCORE_MAX_SECONDS:-0}" \
    "${STAGE_ARG[@]}" \
    ${EXAMPLES_FILE:+--examples-file "$EXAMPLES_FILE"} \
    ${RESCORE_DECLARED_GATE:+--declared-gate}
done

echo "DONE rescore -> ${RESCORE_DIR}"
