#!/usr/bin/env bash
#SBATCH --job-name=cpseq_pot_fold
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/potcheck/logs/fold_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/potcheck/logs/fold_%A_%a.out
#
# Stage 2: fold one complex's draws, and the unchanged peptide, with inference
# potentials ON. See docs/README_CPSEQ_POTENTIALS_CHECK_PLAN.md.
#
# The sequences are NOT re-proposed. They are the ones the source run folded, copied
# into this run's proposal tree by cpseq_fork_cohort.py, so the ONLY thing differing
# between the two runs is the predictor setting. The reference peptide is folded here
# too, and shared across arms in this directory, because a ΔdG that paired a
# potentials-on arm against a potentials-off reference would not be a comparison.
#
# ONE ARRAY TASK PER COMPLEX, and that is not negotiable: the journal is per complex and
# two processes appending to it would interleave rows that cost a GPU-hour to replace.
# Parallelism comes from the array, never from splitting a complex.
#
# GPU CLASS IS PART OF THE REQUEST, not a detail. gpu12-19 are rtx_2080_ti with 11 GB and
# are excluded: a Boltz-2 allocation failure can kill the process with no Python
# traceback, which reads as a mysterious silent FAILED. What remains is a6000 (48 GB, on
# `general`) and v100 (on `legacy`). hostname and nvidia-smi -L are printed at startup
# because slurmdbd is unreachable here -- `sacct` returns nothing and still exits 0 -- so
# this log is the only surviving record of which card a task actually got.
#
# ONE PARTITION PER SUBMISSION. Each partition here carries its own restricted QoS
# (`AllowQos=qos_general` on general, `qos_legacy` on legacy), so a comma-separated
# --partition list is rejected outright with "Invalid qos specification" -- measured
# 2026-11-01. To use both pools the submitter sends two arrays over disjoint index
# ranges of the SAME parents file, overriding --partition on each.
#
# Resources are the source run's measured ones: Boltz-2 MaxRSS 11.0 GiB, hence 32G; the
# relax is single-threaded, so 2 CPUs is one for it and one for the driver.
#
# MEASURED HERE 2026-11-01, preflight job 92358 on gpu25 (A6000 48 GB), potentials ON,
# lnr_1t5z (L=11, 16 draws + reference = 17 sequences):
#   * 17 structures batched per boltz invocation, ~601 s per seed => ~35 s per structure.
#   * 10 seeds of prediction is ~1.7 h; the 170 relaxes that follow are ~2.1 h at the
#     source run's measured 43.7 s each. ~3.8 h per complex, over half of it Rosetta
#     with the card idle -- which is why PREFETCH_AHEAD defaults on for this check.
#   * 8 h per task therefore fits one complex, so resume is a backstop, not the norm.
#
# `--limit` DOES NOT BOUND A PREFLIGHT while batch prediction is on, which is the
# default. It caps the SCORING loop, and that loop runs only after the whole plan has
# been predicted for a seed -- so `--limit 2` still pays for all 17 predictions per seed,
# and job 92358 spent its entire hour predicting without scoring one fold. Bound a
# preflight with a short --time, a single --parents, and --no-batch-predict instead.
#
# A task that times out or is preempted is RESUBMITTED, not restarted: every fold is
# appended to the complex's outcomes.jsonl keyed by (sequence, seed, stage, arm), so the
# next run skips what is already paid for. Re-running the same array line IS the resume.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

# shellcheck source=/dev/null
source "$REPO/scripts/_cpseq_potentials_preamble.sh"
_pot_consume_args "$@"
_pot_defaults
_pot_banner "fold (potentials ON)"

PARENTS_FILE="$(_pot_parents_file "$LEAD_ARM")"
[ -f "$PARENTS_FILE" ] || { echo "FATAL: no parents file at $PARENTS_FILE"; exit 1; }

IDX="${SLURM_ARRAY_TASK_ID:-0}"
PARENT=$(grep -v '^[[:space:]]*#' "$PARENTS_FILE" | grep -v '^[[:space:]]*$' \
         | sed -n "$((IDX + 1))p")
if [ -z "$PARENT" ]; then
  # Exit 0, not 1. An array sized past the end of the parent list must not strand the
  # dependent report job in DependencyNeverSatisfied with nothing explaining why.
  echo "array task $IDX is past the end of $PARENTS_FILE -- nothing to fold, exiting 0"
  exit 0
fi
echo "parent:    $PARENT  (line $((IDX + 1)) of $PARENTS_FILE)"
echo "gpu mem:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"

# The cohort must say potentials are ON and must name a scoring protocol. Checked here as
# well as inside `ladder` so a misconfigured run dies in seconds instead of after a model
# load -- and because a cohort with no scoring_version would have every fold scored under
# the module default, which the v3 report cannot read at all.
"$PY" - "$POT_DIR" <<'EOF'
import json, sys
from pathlib import Path
c = json.loads((Path(sys.argv[1]) / "cohort.json").read_text())
assert c.get("use_potentials") is True, \
    f"cohort says use_potentials={c.get('use_potentials')!r}; this job folds the ON arm"
assert c.get("scoring_version"), \
    "cohort has no scoring_version: every fold would be scored under the module default"
print(f"[check] use_potentials={c['use_potentials']} "
      f"scoring_version={c['scoring_version']}")
EOF

ARM_FLAGS=()
case "$LEAD_ARM" in
  llm_search)     ARM_FLAGS=(--llm-round "$LLM_ROUNDS") ;;
  *mask*)         ARM_FLAGS=(--mask-arm "$LEAD_ARM" --mask-round "$MASK_ROUNDS") ;;
  random_control) ARM_FLAGS=() ;;
  *) echo "FATAL: unknown arm $LEAD_ARM"; exit 1 ;;
esac
echo "arm flags: ${ARM_FLAGS[*]:-<random draw stream>}"

# -u because this log is the only view into a task that runs for hours: Python block-
# buffers stdout when it is a file, so without it a running task is indistinguishable
# from a hung one.
EXTRA=()
# PREDICT_ONLY=1 splits this job in two instead of overlapping its halves.
#
# Measured on `llm_ladder`'s 22,642 v3 folds (2026-11-03): 37.8 s median of Boltz-2
# prediction against 29.6 s median of Rosetta scoring, so 54 % of that run's 327
# job-hours was CPU work holding an a6000. --prefetch-ahead hides some of that inside
# ONE GPU job; --predict-only lets the two halves be SCHEDULED apart, so the scoring
# half runs on `cpu` (88-core nodes, uncontended) instead of queueing for a card that
# all seven `general` nodes currently have fully allocated.
#
# Changes no result: predictions are keyed by `input_hash` and scores by
# (scoring_version, input_hash), so the same plan run twice fills the same two caches
# the same way. Only who pays changes. Follow with cpseq_potentials_score.sbatch over
# the same WORK_DIR and parents file.
if [ "${PREDICT_ONLY:-0}" != "0" ]; then
  EXTRA+=(--predict-only)
  echo "mode:      PREDICT ONLY -- no scoring here; run cpseq_potentials_score.sbatch next"
  # --prefetch-ahead would be meaningless: there is no scoring phase in this job for the
  # prefetch to run ahead OF, and passing both would just start a thread that races the
  # main loop to fill the same cache.
  [ "${PREFETCH_AHEAD:-1}" != "0" ] && \
    echo "note:      PREFETCH_AHEAD ignored under PREDICT_ONLY (nothing to overlap with)"
else
  # Overlap prediction with scoring. Measured above: without it the card is idle for the
  # 2.1 h of relax that follows 1.7 h of folding. It fills the same cache with the same
  # inputs in the same order, so no result changes -- only who pays for the wall time.
  [ "${PREFETCH_AHEAD:-1}" != "0" ] && EXTRA+=(--prefetch-ahead)
fi

srun "$PY" -u script_utils/cpseq_search.py ladder \
  --out-dir "$POT_DIR" \
  --parents "$PARENT" \
  --chemistry "$CHEMISTRY" \
  --samples 1 \
  --retention "$RETENTION" \
  --limit "$LIMIT" \
  --boltz-python "$BOLTZ_PY" \
  --boltz-cache "$BOLTZ_CACHE" \
  "${ARM_FLAGS[@]}" \
  "${EXTRA[@]}"

echo "=== cpseq_potentials_fold OK ($PARENT) ==="
