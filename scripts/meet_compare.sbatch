#!/bin/bash
#SBATCH --job-name=meet_compare
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU arm: MEET vs LNR de novo closure, matched on peptide length and input N-C gap.
# Reads only saved JSONL rows, so the tables and figure re-render without a GPU.
#
# Do NOT sbatch directly -- use scripts/submit_meet_compare.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs
ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${MEET_RESULTS:?MEET_RESULTS not set}"
: "${LNR_RESULTS:?LNR_RESULTS not set}"
: "${COMPARE_OUT:?COMPARE_OUT not set}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

shopt -s nullglob
for D in "$MEET_RESULTS" "$LNR_RESULTS"; do
  ROWS=( "${D}"/edits_shard*.jsonl )
  if (( ${#ROWS[@]} == 0 )); then
    echo "GATE: no shard results in ${D} -- the upstream GPU array produced nothing."
    echo "GATE: read that array's .out files before trusting this chain. Exiting 0."
    exit 0
  fi
done

"$PY" scripts/compare_meet_lnr.py \
  --arm meet "$MEET_RESULTS" \
  --arm lnr  "$LNR_RESULTS" \
  --match-max-length "${MATCH_MAX_LENGTH:-13}" \
  --out-dir "$COMPARE_OUT"
echo "DONE"
