#!/bin/bash
#SBATCH --job-name=cpseq_fill_tsweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# Phase A1 proposal stage: fill ONE shared mask set at several ProteinMPNN temperatures.
#
# What is under test. The ladder pilot found the filler optimising the wrong moment of
# the distribution: at the locked T=0.2, MPNN gives better TYPICAL draws than a random
# filler (+1.12 REU at matched masker, per-complex normalised) but a WORSE tail
# (median best draw -6.56 against random fill's -10.04). A best-of-N search spends only
# the tail, so near-argmax sampling is the suspect. This sweep raises T and asks whether
# the tail widens.
#
# Why no LLM stage. The masks come from `cpseq_make_shared_masks.py` and every arm fills
# the SAME ones, so a difference between arms is attributable to the filler and to
# nothing else. `cpseq_mask_arm.sbatch` re-asks the LLM per round, which would let the
# positions move between arms and confound exactly the axis being measured.
#
# Why CPU. ProteinMPNN here is one subprocess per mask on a peptide of L<=16; measured
# ~12 s per call, almost all of it process start and weight load, so a GPU would buy
# nothing. The expensive stage is the folding that follows, which is a separate job.
#
# Why a random-fill arm is included. It is the incumbent the temperature sweep has to
# beat on the tail, and on shared masks it costs no MPNN time at all. Without it the
# sweep could only say "T=0.8 beats T=0.2" and not the thing actually asked, which is
# whether any temperature recovers the random filler's tail while keeping MPNN's median.
#
# One array task per arm. Tasks write to disjoint `mask_proposals/<arm>/` trees, so they
# are safe to run concurrently -- unlike the fold stage, where one task per complex is
# mandatory because two writers would interleave a journal.
#
# Usage:
#   sbatch --array=0-4 scripts/cpseq_fill_temp_sweep.sbatch <env-file>
#   sbatch --array=0-4 scripts/cpseq_fill_temp_sweep.sbatch OUT_DIR=... MASKS_DIR=...

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"
mkdir -p slurm_logs

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done

OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/cpseq/llm_ladder_pot}"
MASKS_DIR="${MASKS_DIR:-/zfsauton/scratch/yixiz/cpseq/sweep_t/masks}"
ROUND="${ROUND:-1}"
SEQS_PER_MASK="${SEQS_PER_MASK:-1}"
PARENTS="${PARENTS:-}"
# The same parents file the fold stage shards over, so one env file fixes the scope of
# both stages. The fold stage takes ONE line per array task (two writers on a complex's
# journal would interleave); proposing has no such constraint and no journal to corrupt,
# so this task takes the WHOLE file. Without it the propose stage would silently cover
# every parent in MASKS_DIR, which is the same 10 in phase A but 391 in phase B.
PARENTS_FILE="${PARENTS_FILE:-}"
if [ -z "$PARENTS" ] && [ -n "$PARENTS_FILE" ]; then
  [ -f "$PARENTS_FILE" ] || { echo "FATAL: no PARENTS_FILE at $PARENTS_FILE" >&2; exit 1; }
  PARENTS=$(grep -v '^[[:space:]]*#' "$PARENTS_FILE" | grep -v '^[[:space:]]*$' \
            | paste -sd, -)
  [ -n "$PARENTS" ] || { echo "FATAL: $PARENTS_FILE lists no parents" >&2; exit 1; }
fi
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"

# The sweep grid. Index -> (arm label, filler, temperature). The label is what lands in
# the journal and what `--mask-arm` folds later, so it has to name the setting: these
# arms share one OUT_DIR with the pilot arms on purpose (shared reference folds, paired
# comparison), and a label like `llm_mask_mpnn_fill` would silently merge with it.
ARM_LABELS=(mpnn_fill_t02 mpnn_fill_t05 mpnn_fill_t08 mpnn_fill_t10 shared_mask_random_fill)
ARM_FILLS=(mpnn          mpnn          mpnn          mpnn          random)
ARM_TEMPS=(0.2           0.5           0.8           1.0           0.2)

IDX="${SLURM_ARRAY_TASK_ID:-${ARM_INDEX:-0}}"
if [ "$IDX" -ge "${#ARM_LABELS[@]}" ]; then
  echo "=== array task $IDX is past the end of the grid (${#ARM_LABELS[@]} arms); exiting 0"
  exit 0
fi
ARM="${ARM_LABELS[$IDX]}"
FILL="${ARM_FILLS[$IDX]}"
TEMP="${ARM_TEMPS[$IDX]}"

echo "=== cpseq_fill_temp_sweep ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "arm:      $ARM  (fill=$FILL temperature=$TEMP)"
echo "masks:    $MASKS_DIR  round=$ROUND"
echo "out_dir:  $OUT_DIR"
[ -n "$PARENTS" ] && echo "parents:  $PARENTS"

[ -f "$MASKS_DIR/index.json" ] || {
  echo "FATAL: no shared masks at $MASKS_DIR/index.json."
  echo "Build them first:"
  echo "  $PY script_utils/cpseq_make_shared_masks.py --out-dir $OUT_DIR \\"
  echo "      --dest $MASKS_DIR --n-masks 12 --round $ROUND"
  exit 1
}

# -u: this log is the only view into a task that runs for tens of minutes.
"$PY" -u script_utils/cpseq_mask_mpnn_fill.py \
  --out-dir "$OUT_DIR" \
  --masks-dir "$MASKS_DIR" \
  --round "$ROUND" \
  --arms "$ARM" \
  --fill "$FILL" \
  --run-tag "$ARM" \
  --temperature "$TEMP" \
  --seqs-per-mask "$SEQS_PER_MASK" \
  ${PARENTS:+--parents "$PARENTS"}

echo "=== cpseq_fill_temp_sweep OK ($ARM) ==="
