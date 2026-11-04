#!/usr/bin/env bash
#SBATCH --job-name=cpseq_fill_tfold
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_fill_tfold_%x_%A_%a.out
#
# Fold the filler temperature sweep (Phase A1), potentials ON.
#
# The companion to scripts/cpseq_fill_temp_sweep.sbatch. That job PROPOSED -- it filled
# one shared mask set at several ProteinMPNN temperatures and wrote the sequences to
# `mask_proposals/<arm>/round1/<parent>.jsonl`. Nothing is re-proposed here: `cpseq_search.py
# ladder --mask-arm <arm> --mask-round 1` LOADS those exact sequences and folds them, so
# the ONLY thing differing between arms is the filler and its temperature -- which is the
# axis the sweep exists to measure. The report (cpseq_fill_sweep_report.py) then reads
# these folds at MATCHED n.
#
# PREDICT ON THE GPU, SCORE ON THE CPU -- the standard fold shape, see
# docs/README_CPSEQ_FOLD_PIPELINE_STANDARD.md sec. 1. With PREDICT_ONLY=1 (how the
# submitter runs this) the job pays only for Boltz-2 prediction and writes no outcome
# rows; cpseq_fill_sweep_score.sbatch then pays for the ~54% of wall time that is
# PyRosetta relax on a `cpu` node, pooling SCORE_WORKERS processes. Predictions key on
# `input_hash`, so the plan run as predict+score fills the same cache as one job would.
# Left at PREDICT_ONLY=0 this folds AND scores on the GPU serially, which is the old
# single-job behaviour kept only as a fallback.
#
# ONE ARRAY TASK PER PARENT, and inside it ALL arms fold SERIALLY. This is not negotiable:
# the journal is per complex and every arm appends to the SAME complex journal, so two
# arms of one parent running at once would interleave rows that cost a GPU-hour to
# replace. The array gives parallelism ACROSS complexes; arms within a complex are serial
# by construction. Folding the arms together in one task is also what lets the unchanged
# `reference` be folded once (by the first arm) and cache-hit by the rest -- a ΔdG that
# paired an arm against a reference it did not share would not be a comparison.
#
# RESUMABLE. Every fold is appended keyed by (sequence, seed, stage, arm), and the ladder
# skips what `journal.has` already holds. Most of t02 and the random arm are already on
# disk from earlier waves; t05/t08/t10 are not. Re-running the same array line IS the
# resume -- it computes only the missing folds. The ladder's rank-major seed order makes
# an interrupted wave the whole experiment at fewer folds, never a few arms at full depth
# and the rest at none, so a task that hits the 8 h limit is resubmitted, not restarted.
#
# GPU CLASS IS PART OF THE REQUEST. a6000 (48 GB) on `general`: a Boltz-2 allocation
# failure on an 11 GB rtx_2080_ti kills the process with no Python traceback, which reads
# as a silent FAILED. hostname and nvidia-smi -L go into the log at startup because
# slurmdbd is unreachable here -- `sacct` returns nothing and exits 0 -- so this log is
# the only surviving record of which card a task got and of which arms it finished.
#
# Config: a positional env file or VAR=value args. NEVER `sbatch --export`, which sets
# SLURM_GET_USER_ENV=1 and gets jobs requeued and HELD.
#
# Usage:
#   sbatch --array=0-9 scripts/cpseq_fill_sweep_fold.sbatch \
#       PARENTS_FILE=/zfsauton/scratch/yixiz/cpseq/sweep_t/fold_parents.txt

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi

OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/cpseq/llm_ladder_pot}"
ARMS="${ARMS:-mpnn_fill_t02 mpnn_fill_t05 mpnn_fill_t08 mpnn_fill_t10 shared_mask_random_fill}"
MASK_ROUND="${MASK_ROUND:-1}"
CHEMISTRY="${CHEMISTRY:-mainchain}"
RETENTION="${RETENTION:-rescorable}"
LIMIT="${LIMIT:-0}"
PREFETCH_AHEAD="${PREFETCH_AHEAD:-1}"
PREDICT_ONLY="${PREDICT_ONLY:-0}"
PARENTS_FILE="${PARENTS_FILE:-/zfsauton/scratch/yixiz/cpseq/sweep_t/fold_parents.txt}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
BOLTZ_PY="${BOLTZ_PYTHON:-/zfsauton/scratch/yixiz/envs/boltz2/.venv/bin/python}"
BOLTZ_CACHE="${BOLTZ_CACHE:-/zfsauton/scratch/yixiz/boltz_cache}"

echo "=== cpseq_fill_sweep_fold ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "out_dir:   $OUT_DIR"
echo "arms:      $ARMS"
echo "mode:      $([ "$PREDICT_ONLY" != "0" ] && echo 'predict-only (GPU; score on cpu after)' || echo 'predict+score (fallback)')"
nvidia-smi -L 2>/dev/null || echo "gpu:       none"

[ -f "$PARENTS_FILE" ] || { echo "FATAL: no PARENTS_FILE at $PARENTS_FILE" >&2; exit 1; }
IDX="${SLURM_ARRAY_TASK_ID:-0}"
PARENT=$(grep -v '^[[:space:]]*#' "$PARENTS_FILE" | grep -v '^[[:space:]]*$' \
         | sed -n "$((IDX + 1))p")
if [ -z "$PARENT" ]; then
  echo "array task $IDX is past the end of $PARENTS_FILE -- nothing to fold, exiting 0"
  exit 0
fi
echo "parent:    $PARENT  (line $((IDX + 1)) of $PARENTS_FILE)"

# The cohort must say potentials are ON and name a scoring protocol -- checked here so a
# misconfigured run dies in seconds, not after a model load, and so no fold is scored
# under the module default (which the v3 report cannot read).
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
c = json.loads((Path(sys.argv[1]) / "cohort.json").read_text())
assert c.get("use_potentials") is True, \
    f"cohort says use_potentials={c.get('use_potentials')!r}; the sweep folds the ON arm"
assert c.get("scoring_version"), \
    "cohort has no scoring_version: every fold would be scored under the module default"
print(f"[check] use_potentials={c['use_potentials']} scoring_version={c['scoring_version']}")
EOF

EXTRA=()
[ "$PREFETCH_AHEAD" != "0" ] && EXTRA+=(--prefetch-ahead)
# PREDICT_ONLY writes predictions to the input_hash cache and no outcome rows, so the
# per-journal single-writer rule does not even bind this pass; the scoring pass on `cpu`
# writes the rows. Inert when scoring runs in the same job (PREDICT_ONLY=0).
[ "$PREDICT_ONLY" != "0" ] && EXTRA+=(--predict-only)

# Arms fold SERIALLY -- one writer on this parent's journal. The first arm folds the
# shared reference; later arms cache-hit it.
for ARM in $ARMS; do
  echo "=== [$PARENT] folding arm $ARM ==="
  srun "$PY" -u script_utils/cpseq_search.py ladder \
    --out-dir "$OUT_DIR" \
    --parents "$PARENT" \
    --mask-arm "$ARM" --mask-round "$MASK_ROUND" \
    --chemistry "$CHEMISTRY" \
    --samples 1 \
    --retention "$RETENTION" \
    --limit "$LIMIT" \
    --boltz-python "$BOLTZ_PY" \
    --boltz-cache "$BOLTZ_CACHE" \
    "${EXTRA[@]}"
done

echo "=== cpseq_fill_sweep_fold OK ($PARENT) ==="
