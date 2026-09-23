#!/bin/bash
#SBATCH --job-name=m15_espat
#SBATCH --output=logs/m15_espat_%A_%a.out
#SBATCH --error=logs/m15_espat_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
# Section 6 / Deliverable E, stage 2: the spatial check. Sequence-disjoint is necessary,
# not sufficient -- two disjoint segments can bind the same surface, which makes them
# near-duplicates rather than competitors, and an arm built on them removes no shortcut.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_m15_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_m15_preamble.sh"

_m15_consume_args "$@"
_m15_defaults
_m15_banner deliverable-e-spatial
_m15_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

PAIRS="${DELIV_E_DIR}/deliverable_e_candidate_pairs.parquet"
if [[ ! -f "$PAIRS" ]]; then
  echo "GATE: no candidate pairs at $PAIRS -- the index stage did not produce them."
  echo "GATE: exiting 0 so the report still runs and says the spatial check is missing."
  exit 0
fi

# An empty candidate set is a RESULT (the arm is deferred, not forced), not a failure.
N_PAIRS="$("$PY" -c "import pandas as pd; print(len(pd.read_parquet('$PAIRS')))")"
if [[ "$N_PAIRS" -eq 0 ]]; then
  echo "GATE: 0 candidate pairs survived the index stage. Deliverable E has nothing to"
  echo "GATE: check spatially; the arm is deferred on evidence. Exiting 0."
  exit 0
fi

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${NUM_SHARDS:-1}"
cd "$REPO"
mkdir -p "${DELIV_E_DIR}/spatial"

"$PY" script_utils/m15_deliverable_e.py spatial \
  --config "$M15_CONFIG" \
  --pairs "$PAIRS" \
  --out "${DELIV_E_DIR}/spatial/pairs_shard${SHARD}.jsonl" \
  --shard "$SHARD" --num-shards "$NSHARDS"

echo "[deliverable-e-spatial] done -> ${DELIV_E_DIR}/spatial/pairs_shard${SHARD}.jsonl"
