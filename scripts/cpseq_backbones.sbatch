#!/usr/bin/env bash
#SBATCH --job-name=cpseq_backbones
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_backbones_%x_%A_%a.out
#
# Arm C backbone supply: sample every runnable CP generator UNCONDITIONALLY on the same
# LNR cohort, at the same n, with the same seed and sampler settings -- one array task per
# checkpoint, so the four arms are progressed equally and a closure comparison between
# them is matched rather than assembled from whatever each run happened to produce.
#
# WHY THIS JOB EXISTS, AND WHY IT IS SEPARATE FROM cpseq_search.sbatch
# -------------------------------------------------------------------
# Arm C was backbone-starved, not proposal-starved: under the CB + displacement gate only
# 2 of 12 `bondunroll` backbones were admitted (both on one parent), so 5 of 6 parents had
# nothing to design onto. Two samples per parent at ~17 % admission makes that arithmetic,
# not bad luck. The fix is a better and deeper backbone supply, which is a pure generator
# job -- no Boltz, no search state -- so it does not belong inside the closed-loop search.
#
# MEMORY: 64G, not the 32G the search jobs use. Measured: `cpsea_cyc_ringpe_bond_v4` loads
# its own recovered 961M frozen AE on top of a 2.8G EMA checkpoint and enables the extra
# cyclization link/bond heads and cross-sequence pair features; at 32G it was killed
# (exit 137, host OOM) AFTER the autoencoder had loaded successfully. The OOM is host RAM
# at model construction, not VRAM, so the GPU type is left unpinned.
#
# UNCONDITIONAL IS THE POINT. `--seq-cond` is deliberately NOT passed. The plan forbids
# emulating absent unconditional support, and arm C needs conformations that are not
# conditioned on the original LP sequence -- its whole purpose is to propose from
# backbones a Boltz fold of that sequence cannot reach. `cpsea_cyc_ringpe_bond_v4` is the
# genuinely sequence-unconditional model (no seq-conditioning section in its config at
# all); `seqcond_v2`'s unconditional path is trained-but-eroded (p = 0.5, warm-started
# from bondunroll) and is the worst closer of the CPSea arms unconditionally, which is
# exactly why it must be measured here rather than assumed.
#
# Resumable, which preempt's REQUEUE requires: eval_val_gen_testset.py skips batches whose
# closure row is already on disk, and each task owns its own --out-dir.
#
# Submit:   sbatch --array=0-3 scripts/cpseq_backbones.sbatch
# One arm:  sbatch --array=0 scripts/cpseq_backbones.sbatch N_REPEAT=2
# Preflight: sbatch --array=0-3 scripts/cpseq_backbones.sbatch N_REPEAT=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi

OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/cpseq/search_pilot}"
STORE="${STORE:-/data/user_data/yixiz/Proteina-Complexa/training_runs/store}"
TEST_PARQUET="${TEST_PARQUET:-$OUT_DIR/backbones/metadata/cohort6_mainchain.parquet}"
# 6 parents x 8 repeats = 48 per checkpoint, the same n as the seq-cond closure row this
# is being compared against, so no comparison is made across different sample sizes.
N_COMPLEXES="${N_COMPLEXES:-6}"
N_REPEAT="${N_REPEAT:-8}"
# Small on purpose: the sampler re-seeds per batch as seed*100003+batch_idx, and numpy
# rejects anything over 4294967295 -- a date-shaped seed (20261021) overflows that and
# kills the run after the model has loaded. Anything under 42948 is safe. The value only
# has to be IDENTICAL across the four arms, which is what makes them matched.
SEED="${SEED:-1021}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
IDX="${SLURM_ARRAY_TASK_ID:-${ARM_INDEX:-0}}"

# name | run dir | explicit ckpt (empty = the run's own last-EMA.ckpt)
ARMS=(
  "gen_v4_uncond|cpsea_cyc_ringpe_bond_v4|"
  "gen_bondunroll_uncond|cpsea_bondunroll_from_v4cfg|"
  "gen_seqcond_v2_uncond|cpsea_seqcond_v2|checkpoints/last-v3-EMA.ckpt"
  "gen_lpmix_uncond|cpsea_lpmix_from_v4cfg|"
)
if [ "$IDX" -ge "${#ARMS[@]}" ]; then
  echo "array task $IDX >= ${#ARMS[@]} arms; nothing to do" >&2
  exit 0
fi
IFS='|' read -r NAME RUN CKPT <<< "${ARMS[$IDX]}"

echo "=== cpseq_backbones ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "arm:       $NAME  (task $IDX of ${#ARMS[@]})"
echo "run_dir:   $STORE/$RUN"
echo "ckpt:      ${CKPT:-<last-EMA.ckpt>}"
echo "sampling:  n_complexes=$N_COMPLEXES n_repeat=$N_REPEAT seed=$SEED  UNCONDITIONAL"
nvidia-smi -L || echo "no nvidia-smi"

[ -f "$TEST_PARQUET" ] || { echo "missing test parquet: $TEST_PARQUET" >&2; exit 1; }

EXTRA=()
[ -n "$CKPT" ] && EXTRA+=(--ckpt "$STORE/$RUN/$CKPT")

# No --seq-cond: see the header. Each arm writes its own dir so a requeue resumes only it.
srun "$PY" script_utils/eval_val_gen_testset.py \
  --run-dir "$STORE/$RUN" \
  --test-parquet "$TEST_PARQUET" \
  --n-complexes "$N_COMPLEXES" \
  --n-repeat "$N_REPEAT" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR/backbones/$NAME" \
  "${EXTRA[@]}"

echo "=== cpseq_backbones OK ($NAME) ==="
