#!/bin/bash
#SBATCH --job-name=genkic_sdedit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# The comparison arm: grade EXISTING SDEdit outputs under the same contract as the GenKIC
# search. Independent of the search array -- it reads structures produced weeks ago -- so it
# is submitted as a sibling and queues concurrently rather than behind it.
#
# Each array task takes a disjoint slice of the example_ids and writes its own
# rows_<shard>.jsonl. One writer per file, same as the search.
#
# This stage requires an SDEdit run that wrote `complexes/`. A run that wrote only `pdbs/`
# holds peptide-only structures, and the contract's contact, interface-RMSD and paired-dG
# criteria are not computable without the receptor -- this repo has measured that a peptide
# saved alone cannot be re-united with its receptor afterwards (two loads of one example
# differ by ~2.7 A). The gate below refuses such a run by name instead of producing a table
# of NaNs that would read as SDEdit failing the contract.
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
SHARD_COUNT="${SDEDIT_SHARDS:-12}"
_genkic_banner "sdedit grading shard ${SHARD_INDEX}/${SHARD_COUNT}"

# Gate INSIDE the job, exiting 0: a non-zero exit here would strand the report stage on
# DependencyNeverSatisfied with no explanation, when the honest outcome is "the comparison
# arm had no gradable input".
if [[ -z "${SDEDIT_DIR}" ]]; then
  echo "GATE FAILED: SDEDIT_DIR is empty -- no SDEdit run named for comparison."
  echo "  Pass one, e.g. SDEDIT_DIR=${REPO}/evaluation_results/sdedit_pocket14_20260906_213400"
  echo "Exiting 0 -- this is a gate, not a crash."
  exit 0
fi
if [[ ! -d "${SDEDIT_DIR}" ]]; then
  echo "GATE FAILED: SDEDIT_DIR is not a directory: ${SDEDIT_DIR}. Exiting 0."
  exit 0
fi
shopt -s nullglob
EDITS=("${SDEDIT_DIR}"/edits_shard*.jsonl)
shopt -u nullglob
if [[ ${#EDITS[@]} -eq 0 ]]; then
  echo "GATE FAILED: no edits_shard*.jsonl in ${SDEDIT_DIR}. Exiting 0."
  exit 0
fi
if [[ ! -d "${SDEDIT_DIR}/complexes" ]]; then
  echo "GATE FAILED: ${SDEDIT_DIR} has no complexes/ directory."
  echo "  This run wrote peptide-only structures. Contact retention, interface-frame RMSD and"
  echo "  paired interface dG are not computable without the receptor, and a peptide saved"
  echo "  alone cannot be re-united with its receptor after the fact. Pick a run with"
  echo "  complexes/, or re-run SDEdit with complex writing enabled."
  echo "Exiting 0 -- this is a gate, not a crash."
  exit 0
fi
echo "grading ${#EDITS[@]} edits file(s) from ${SDEDIT_DIR}"
echo "  complexes present: $(ls "${SDEDIT_DIR}/complexes" | wc -l)"

SDEDIT_ROWS_DIR="${SDEDIT_ROWS_DIR:-${OUT_DIR}/sdedit_rows}"
mkdir -p "$SDEDIT_ROWS_DIR"
OUT="${SDEDIT_ROWS_DIR}/rows_${SHARD_INDEX}.jsonl"
if [[ -f "$OUT" ]]; then
  echo "resuming: $OUT already has $(wc -l < "$OUT") rows"
fi

EXTRA=()
[[ -n "${PROTECTED}" ]] && EXTRA+=(--protected "${PROTECTED}")

srun python script_utils/score_sdedit_contract.py \
  --edits "${EDITS[@]}" \
  --out "$OUT" \
  --pdb-dir "${OUT_DIR}/sdedit_pdbs" \
  --chemistries ${CHEMISTRIES} \
  --shortlist "${SDEDIT_SHORTLIST:-${SHORTLIST}}" \
  --dump-top "${DUMP_TOP}" \
  --relax-repeats "${RELAX_REPEATS}" \
  --lp-repeats "${LP_REPEATS}" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --time-budget-s "${TIME_BUDGET_S}" \
  "${EXTRA[@]}"

echo "shard ${SHARD_INDEX} wrote $(wc -l < "$OUT") rows to $OUT"
