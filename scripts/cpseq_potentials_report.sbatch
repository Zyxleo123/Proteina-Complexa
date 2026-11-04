#!/usr/bin/env bash
#SBATCH --job-name=cpseq_pot_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/potcheck/logs/report_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/potcheck/logs/report_%j.out
#
# Stage 3: the answer. CPU only -- it reads journals, so it must not ask for a GPU.
#
# Two numbers, and they are not interchangeable:
#   * the PAIRED change on the complexes BOTH runs scored. This is the decision: whether
#     the arm's advantage over the native peptide survives potentials being on.
#   * the ENLARGED set, reported separately. Potentials raise the reference's pass rate,
#     so complexes the OFF run reported as `no_base` can acquire a baseline. That is a
#     gain, but it changes WHICH complexes are being compared, and an aggregate that
#     moved for that reason has said nothing about the arm.
#
# This job is a DECISION GATE, so a missing input prints why and exits 0. A non-zero exit
# would look identical to the job having crashed, and there is nothing downstream to
# protect by failing.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

# shellcheck source=/dev/null
source "$REPO/scripts/_cpseq_potentials_preamble.sh"
_pot_consume_args "$@"
_pot_defaults
_pot_banner "report"

mkdir -p "$WORK_DIR/on"

if [ ! -d "$POT_DIR/state" ]; then
  echo
  echo "GATE: no journals under $POT_DIR/state -- the fold stage produced nothing."
  echo "      This is 'upstream never ran', not 'the check failed'. Exiting 0 so the"
  echo "      state of the chain is legible rather than looking like a crash."
  exit 0
fi

ROWS=$(cat "$POT_DIR"/state/*/outcomes.jsonl 2>/dev/null | wc -l || echo 0)
echo "ON journal rows: $ROWS"

for arm in $ARMS; do
  OFF_JSON="$WORK_DIR/off/pilot_${arm}.json"
  ON_JSON="$WORK_DIR/on/pilot_${arm}.json"

  echo
  echo "================ $arm ================"
  echo "--- potentials ON, pilot estimator ---"
  if ! "$PY" script_utils/cpseq_selected_candidate_report.py \
        --out-dir "$POT_DIR" \
        --scoring-version "$GATE" \
        --arm "$arm" \
        --pilot \
        --per-complex \
        --json "$ON_JSON"; then
    echo "  (no printable ON report for $arm yet -- skipping its comparison)"
    continue
  fi

  if [ ! -f "$OFF_JSON" ]; then
    echo "  GATE: no OFF baseline at $OFF_JSON -- prepare did not redraw this arm."
    continue
  fi

  echo
  echo "--- PAIRED: off vs on, same estimator, same gate, same sequences ---"
  "$PY" script_utils/cpseq_potentials_compare.py \
      --off "$OFF_JSON" --on "$ON_JSON" \
      --label-off "pot off" --label-on "pot on" \
      --json "$WORK_DIR/compare_${arm}.json"
done

echo
echo "=== cpseq_potentials_report OK ==="
echo "artifacts under $WORK_DIR:"
ls -1 "$WORK_DIR" "$WORK_DIR/off" "$WORK_DIR/on" 2>/dev/null | sed 's/^/    /'
