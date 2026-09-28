#!/bin/bash
#SBATCH --job-name=genkic_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Aggregate the search shards into the arm's results tables. Reads ONLY saved rows, so the
# report is re-runnable as often as the tables need re-cutting without touching Rosetta --
# which is also why it is a separate CPU job rather than a tail on the search: asking for
# search resources to render a table would just queue it behind real work.
#
# Submitted with --dependency=afterany, not afterok: a shard that hit its walltime still
# produced rows worth aggregating, and partial coverage is reported explicitly (every target
# in the metadata appears, with "NO ROWS" where that is the truth) rather than silently
# shrinking the denominator.
#
# Do NOT sbatch directly -- use scripts/submit_genkic.sh. Config in the positional env file.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa/.claude/worktrees/genkic}/scripts/_genkic_preamble.sh"
[[ -f "$PREAMBLE" ]] || { echo "FATAL: preamble not found: $PREAMBLE"; exit 1; }
# shellcheck source=/dev/null
source "$PREAMBLE"

_genkic_consume_args "$@"
_genkic_defaults
_genkic_venv
_genkic_banner "report"

# A decision gate belongs INSIDE the job: exiting non-zero here would strand nothing (this is
# the last stage) but would also report "failed" for the legitimate case of an upstream stage
# that produced no rows. Distinguish the two explicitly and exit 0 either way.
shopt -s nullglob
SHARDS=("${ROWS_DIR}"/rows_*.jsonl)
shopt -u nullglob
if [[ ${#SHARDS[@]} -eq 0 ]]; then
  echo "GATE FAILED: no rows_*.jsonl in ${ROWS_DIR}."
  if [[ -d "${ROWS_DIR}" ]]; then
    echo "  The rows directory exists but is empty: the search ran and wrote nothing."
  else
    echo "  The rows directory does not exist: the search never ran (upstream never started)."
  fi
  echo "Exiting 0 -- this is a gate, not a crash."
  exit 0
fi
echo "aggregating ${#SHARDS[@]} shard file(s), $(cat "${SHARDS[@]}" | wc -l) rows total"

EXTRA=()
if [[ -d "${SDEDIT_ROWS_DIR}" ]]; then
  EXTRA+=(--sdedit-rows "${SDEDIT_ROWS_DIR}")
  echo "including SDEdit comparison rows from ${SDEDIT_ROWS_DIR} "\
"($(cat "${SDEDIT_ROWS_DIR}"/rows_*.jsonl 2>/dev/null | wc -l) rows)"
else
  echo "no SDEdit rows at ${SDEDIT_ROWS_DIR}: the comparison section will say so explicitly."
fi

python script_utils/aggregate_genkic.py \
  --rows-dir "${ROWS_DIR}" \
  --metadata "${METADATA}" \
  --out-dir "${REPORT_DIR}" \
  "${EXTRA[@]}"

echo
echo "report written to ${REPORT_DIR}"
