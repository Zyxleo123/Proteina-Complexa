#!/bin/bash
#SBATCH --job-name=cc_ladder
#SBATCH --output=logs/cc_ladder_%A_%a.out
#SBATCH --error=logs/cc_ladder_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Pulls the terminal CA pair APART up a ladder with the receptor frozen. One output file
# PER ARRAY TASK: concurrent O_APPEND to a shared file NUL-corrupts rows on this
# filesystem, and a corrupted row is worse than a missing one because the aggregate still
# computes.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_corruption_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_corruption_preamble.sh"

_cc_consume_args "$@"
_cc_defaults
_cc_banner "ladder"
_cc_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }
_cc_preflight_openmm || { echo "FATAL: openmm preflight failed" >&2; exit 1; }

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${NUM_SHARDS:-1}"
PT="$PER_TYPE"
[[ "$SMOKE" == "1" ]] && PT=1

cd "$REPO"
mkdir -p "$LADDER_DIR" "$PDB_DIR"
OUT="${LADDER_DIR}/ladder_shard${SHARD}.jsonl"

# shellcheck disable=SC2086  # LADDER_A is a deliberate word-split list
"$PY_OPENMM" script_utils/corruption_ceiling.py \
  --metadata "$CPSEA_META" \
  --out "$OUT" \
  --pdb-dir "$PDB_DIR" \
  --per-type "$PT" \
  --shard "$SHARD" --num-shards "$NSHARDS" \
  --seed "$SEED" \
  --ladder-A $LADDER_A \
  --k-contact "$K_CONTACT" \
  --exclude-termini "$EXCLUDE_TERMINI" \
  --pull-rounds "$PULL_ROUNDS" \
  --max-iterations "$MAX_ITERATIONS" \
  --converge-tol-A "$CONVERGE_TOL_A" \
  --stall-eps-A "$STALL_EPS_A" \
  ${REQUIRE_RING_OPEN:+$([[ "$REQUIRE_RING_OPEN" == "1" ]] && echo --require-ring-open)}

echo "[ladder] done -> $OUT"
