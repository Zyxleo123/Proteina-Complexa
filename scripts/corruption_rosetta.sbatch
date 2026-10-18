#!/bin/bash
#SBATCH --job-name=cc_rosetta
#SBATCH --output=logs/cc_rosetta_%A_%a.out
#SBATCH --error=logs/cc_rosetta_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Rosetta interface dG for the dumped complexes. A CPU sidecar, off the OpenMM job: it
# needs a different interpreter and shards differently. One output file per array task.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_corruption_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_corruption_preamble.sh"

_cc_consume_args "$@"
_cc_defaults
_cc_banner "rosetta"
_cc_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${NUM_SHARDS:-1}"
cd "$REPO"

# Gate, not failure: if the ladder stage wrote nothing there is nothing to score, and a
# non-zero exit here would strand the report in DependencyNeverSatisfied with no reason.
shopt -s nullglob
LADDER_FILES=("$LADDER_DIR"/ladder_shard*.jsonl)
shopt -u nullglob
if [[ ${#LADDER_FILES[@]} -eq 0 ]]; then
  echo "GATE: no ladder JSONL in $LADDER_DIR -- the ladder stage produced nothing."
  echo "GATE: exiting 0 so the report still runs and says which stages contributed."
  exit 0
fi

mkdir -p "$ROSETTA_DIR"
OUT="${ROSETTA_DIR}/dg_shard${SHARD}.jsonl"

"$PY" script_utils/corruption_ceiling_rosetta.py \
  --ladder "${LADDER_FILES[@]}" \
  --out "$OUT" \
  --shard-index "$SHARD" --shard-count "$NSHARDS"

echo "[rosetta] done -> $OUT"
