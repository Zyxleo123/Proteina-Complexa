#!/bin/bash
#SBATCH --job-name=genkic_search
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# The substantive search (CPU array). Each array task takes a disjoint slice of the staged
# targets and writes its OWN rows_<shard>.jsonl -- exactly one writer per file. Concurrent
# O_APPEND to a shared file has NUL-corrupted rows on this cluster (88 of 144 lost), and the
# aggregator REFUSES a corrupt line rather than skipping it, so a shared file would not
# merely lose rows, it would stop the run.
#
# Resumable: the driver reads its own out file and skips finished row_keys, and TIME_BUDGET_S
# stops it from STARTING a new target it cannot finish. Hitting the walltime therefore means
# resubmitting the identical command, not restarting.
#
# No GPU: GenKIC, the bridge anneal, FastRelax and InterfaceAnalyzerMover are all CPU. Asking
# for a GPU here would only queue this behind real GPU work.
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

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${SEARCH_SHARDS}"
_genkic_banner "search shard ${SHARD_INDEX}/${SHARD_COUNT}"

[[ -f "$METADATA" ]] || { echo "FATAL: metadata not found: $METADATA"; exit 1; }

OUT="${ROWS_DIR}/rows_${SHARD_INDEX}.jsonl"
if [[ -f "$OUT" ]]; then
  echo "resuming: $OUT already has $(wc -l < "$OUT") rows"
fi

EXTRA=()
[[ -n "${EXAMPLES}" ]] && EXTRA+=(--examples ${EXAMPLES})
[[ -n "${PROTECTED}" ]] && EXTRA+=(--protected "${PROTECTED}")

srun python scripts/genkic_cyclize.py \
  --metadata "$METADATA" \
  --out "$OUT" \
  --pdb-dir "${PDB_DIR}" \
  --chemistries ${CHEMISTRIES} \
  --budgets ${BUDGETS} \
  --attempts-per-call "${ATTEMPTS_PER_CALL}" \
  --calls "${CALLS}" \
  --min-solutions "${MIN_SOLUTIONS}" \
  --max-released "${MAX_RELEASED}" \
  --shortlist "${SHORTLIST}" \
  --dump-top "${DUMP_TOP}" \
  --relax-repeats "${RELAX_REPEATS}" \
  --cp-repeats "${CP_REPEATS}" \
  --lp-repeats "${LP_REPEATS}" \
  --perturb-deg "${PERTURB_DEG}" \
  --seed "${SEED}" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --time-budget-s "${TIME_BUDGET_S}" \
  "${EXTRA[@]}"

echo "shard ${SHARD_INDEX} wrote $(wc -l < "$OUT") rows to $OUT"
