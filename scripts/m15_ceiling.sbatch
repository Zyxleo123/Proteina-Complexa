#!/bin/bash
#SBATCH --job-name=m15_ceil
#SBATCH --output=logs/m15_ceil_%A_%a.out
#SBATCH --error=logs/m15_ceil_%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
# Per-complex ceiling scan. One output file PER ARRAY TASK: concurrent O_APPEND to a
# shared file NUL-corrupts rows on this filesystem, and a corrupted row is worse than a
# missing one because the aggregate still computes.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_m15_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_m15_preamble.sh"

_m15_consume_args "$@"
_m15_defaults
: "${AUDIT_SET:?AUDIT_SET not set (pass AUDIT_SET=lnr as a positional argument)}"
_m15_banner "ceiling:${AUDIT_SET}"
_m15_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

# Gate, not failure: if calibration did not run, say so and exit 0. A non-zero exit here
# strands the report job in DependencyNeverSatisfied with nothing explaining why.
if [[ ! -f "$WINDOWS_JSON" ]]; then
  echo "GATE: no bridge windows at $WINDOWS_JSON -- the calibrate stage did not produce"
  echo "GATE: its output. Nothing to audit against; exiting 0 so dependents still run."
  exit 0
fi

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${NUM_SHARDS:-1}"
cd "$REPO"

read -r META SRC MAXEX VFRAC < <("$PY" -c "
import sys; sys.path.insert(0, '.')
from script_utils import m15_config
c = m15_config.load('$M15_CONFIG')
s = c['audit']['sets']['$AUDIT_SET']
print(m15_config.resolve(s['metadata']), s.get('source_filter') or '-',
      s.get('max_examples', 0), c['audit']['validate_frac'])
")
[[ "$SRC" == "-" ]] && SRC=""
[[ "$SMOKE" == "1" ]] && { MAXEX=6; VFRAC=0; }

if [[ ! -f "$META" ]]; then
  echo "GATE: metadata for set '$AUDIT_SET' not found: $META"
  echo "GATE: exiting 0 -- this set contributes no rows, the others still report."
  exit 0
fi

mkdir -p "$CEILING_DIR"
OUT="${CEILING_DIR}/${AUDIT_SET}_shard${SHARD}.jsonl"
echo "[ceiling] set=$AUDIT_SET meta=$META shard=$SHARD/$NSHARDS max=$MAXEX validate=$VFRAC"

"$PY" script_utils/m15_ceiling_audit.py audit \
  --config "$M15_CONFIG" \
  --windows "$WINDOWS_JSON" \
  --metadata "$META" \
  --out "$OUT" \
  --label "$AUDIT_SET" \
  ${SRC:+--source-filter "$SRC"} \
  --max-examples "$MAXEX" \
  --shard "$SHARD" --num-shards "$NSHARDS" \
  --validate-frac "$VFRAC"

echo "[ceiling] done -> $OUT"
