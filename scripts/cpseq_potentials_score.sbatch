#!/usr/bin/env bash
#SBATCH --job-name=cpseq_pot_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/potcheck/logs/score_%A_%a.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/potcheck/logs/score_%A_%a.out
#
# The scoring half of the potentials check. CPU ONLY -- no --gres, deliberately.
#
# Why this job exists
# -------------------
# Measured on `llm_ladder`'s 22,642 v3 folds (2026-11-03): 37.8 s median of Boltz-2
# prediction against 29.6 s median of Rosetta scoring per fold, so scoring is 54 % of
# that run's 327 job-hours -- more than half of a "GPU" run was PyRosetta with the card
# idle. `cpseq_potentials_fold.sbatch PREDICT_ONLY=1` now pays for the predictions on an
# a6000 and stops; this pays for the relaxes on a `cpu` node, where there are 88 cores
# per node and no contention with training jobs.
#
# This changes NO result. `evaluate` keys predictions by `input_hash` and scores by
# (scoring_version, input_hash), so the plan run as two passes fills the same two caches
# as the plan run as one. If a prediction is MISSING, the fold is scored by the writer,
# which without a GPU fails -- and that is the right failure: it says the predict pass
# did not finish, rather than silently scoring a different set of folds.
#
# Per-complex parallelism, which is the point
# -------------------------------------------
# The shard axis in `cpseq_search.py` is the PARENT list (`cohort[idx::count]`), so
# `cpseq_ladder_score.sbatch`'s WORKERS trick -- several sub-shard PROCESSES, each
# owning whole complexes -- buys nothing here: this array is one complex per task by
# design, because the journal is per complex and concurrent O_APPEND to one file
# NUL-corrupts rows on this cluster. So the parallelism comes from --score-workers,
# where the worker processes SCORE and the parent WRITES. One writer, N relaxes at once.
#
# Resources: 16 CPUs and 64G because PyRosetta is ~1.5 GB per worker on a 422-residue
# pose. Memory is what caps this, not cores -- cpseq_score_cost_probe.py measures where
# the scaling actually stops, and SCORE_WORKERS should be set from that rather than from
# the core count.
#
# SKIP_INVALID_RELAX=1 additionally drops the relax on folds the v3 gate has already
# failed (60.9 % of folds; 58.2 % of scoring hours). It is OFF by default because it
# changes what those rows carry -- no refined energy, no deformation proxy, no
# prerelax-to-postrelax gap, and so no rescore to v4/v5. The paired potentials
# comparison is a v3 comparison and does not read those fields, but that is a judgement
# about THIS analysis, so it is opted into per submission and recorded on every row.
#
# Resumable: every fold is appended to the complex's outcomes.jsonl keyed by
# (sequence, seed, stage, arm), so re-running the same array line is the resume.
#
# Submit (after the predict pass):
#   sbatch --array=0-9%10 scripts/cpseq_potentials_score.sbatch <env-file>
#   sbatch --array=0-9%10 scripts/cpseq_potentials_score.sbatch <env-file> SKIP_INVALID_RELAX=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

# shellcheck source=/dev/null
source "$REPO/scripts/_cpseq_potentials_preamble.sh"
_pot_consume_args "$@"
_pot_defaults
_pot_banner "score (CPU, potentials ON)"

SCORE_WORKERS="${SCORE_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
SKIP_INVALID_RELAX="${SKIP_INVALID_RELAX:-0}"
echo "workers:   $SCORE_WORKERS"
echo "skip_inv:  $SKIP_INVALID_RELAX"

PARENTS_FILE="$(_pot_parents_file "$LEAD_ARM")"
[ -f "$PARENTS_FILE" ] || { echo "FATAL: no parents file at $PARENTS_FILE"; exit 1; }

IDX="${SLURM_ARRAY_TASK_ID:-0}"
PARENT=$(grep -v '^[[:space:]]*#' "$PARENTS_FILE" | grep -v '^[[:space:]]*$' \
         | sed -n "$((IDX + 1))p")
if [ -z "$PARENT" ]; then
  # Exit 0: an array sized past the end of the parent list must not strand the dependent
  # report job in DependencyNeverSatisfied with nothing explaining why.
  echo "array task $IDX is past the end of $PARENTS_FILE -- nothing to score, exiting 0"
  exit 0
fi
echo "parent:    $PARENT  (line $((IDX + 1)) of $PARENTS_FILE)"

if [ ! -d "$POT_DIR/predictions/$PARENT" ]; then
  echo
  echo "GATE: no predictions under $POT_DIR/predictions/$PARENT -- the predict pass"
  echo "      never ran for this complex. Exiting 0 so the state of the chain stays"
  echo "      legible rather than looking like a crash. Run the fold stage with"
  echo "      PREDICT_ONLY=1 first."
  exit 0
fi

ARM_FLAGS=()
case "$LEAD_ARM" in
  llm_search)     ARM_FLAGS=(--llm-round "$LLM_ROUNDS") ;;
  *mask*)         ARM_FLAGS=(--mask-arm "$LEAD_ARM" --mask-round "$MASK_ROUNDS") ;;
  random_control) ARM_FLAGS=() ;;
  *) echo "FATAL: unknown arm $LEAD_ARM"; exit 1 ;;
esac
echo "arm flags: ${ARM_FLAGS[*]:-<random draw stream>}"

EXTRA=()
[ "$SKIP_INVALID_RELAX" != "0" ] && EXTRA+=(--skip-invalid-relax)

# No --predict-only and no --prefetch-ahead: this pass scores, and with every prediction
# already on disk there is nothing to overlap with.
#
# -u because this log is the only view into a task that runs for hours: Python
# block-buffers stdout when it is a file, so without it a running task is
# indistinguishable from a hung one.
srun "$PY" -u script_utils/cpseq_search.py ladder \
  --out-dir "$POT_DIR" \
  --parents "$PARENT" \
  --chemistry "$CHEMISTRY" \
  --samples 1 \
  --retention "$RETENTION" \
  --limit "$LIMIT" \
  --score-workers "$SCORE_WORKERS" \
  --boltz-python "$BOLTZ_PY" \
  --boltz-cache "$BOLTZ_CACHE" \
  "${ARM_FLAGS[@]}" \
  "${EXTRA[@]}"

echo "=== cpseq_potentials_score OK ($PARENT) ==="
