#!/bin/bash
#SBATCH --job-name=m15_report
#SBATCH --output=logs/m15_report_%j.out
#SBATCH --error=logs/m15_report_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
# Reads only files on disk, so every table and figure is redrawable without recomputing
# anything. Depends on its upstream stages with afterANY: a stage that gated out has
# nothing to contribute, and the report's job is to say which ones those were.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_m15_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_m15_preamble.sh"

_m15_consume_args "$@"
_m15_defaults
_m15_banner report
_m15_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

cd "$REPO"
shopt -s nullglob globstar

CEIL_GLOBS=("${CEILING_DIR}/"*.jsonl)
if [[ ${#CEIL_GLOBS[@]} -eq 0 ]]; then
  echo "GATE: no ceiling shards under ${CEILING_DIR}. The audit stage produced nothing --"
  echo "GATE: read its .out before trusting any other stage in this run. Exiting 0."
  exit 0
fi
echo "[report] ${#CEIL_GLOBS[@]} ceiling shard(s)"

ARGS=(--ceiling-glob "${CEILING_DIR}/*.jsonl" --out-dir "$OUT_DIR" --config "$M15_CONFIG")
[[ -f "$WINDOWS_JSON" ]] && ARGS+=(--windows "$WINDOWS_JSON")
[[ -f "${TIMING_DIR}/replica_timing.summary.json" ]] && ARGS+=(--timing "${TIMING_DIR}/replica_timing.summary.json")
[[ -f "${DELIV_E_DIR}/deliverable_e_index.json" ]] && ARGS+=(--e-index "${DELIV_E_DIR}/deliverable_e_index.json")
[[ -d "${DELIV_E_DIR}/spatial" ]] && ARGS+=(--e-spatial-glob "${DELIV_E_DIR}/spatial/*.jsonl")
[[ -n "${ACHIEVED_GLOB}" ]] && ARGS+=(--achieved-glob "${ACHIEVED_GLOB}")

"$PY" script_utils/m15_report.py "${ARGS[@]}"
echo
echo "================= REPORT ================="
cat "${OUT_DIR}/M15_CEILING_REPORT.md"
