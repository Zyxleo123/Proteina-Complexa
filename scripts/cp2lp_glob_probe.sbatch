#!/bin/bash
#SBATCH --job-name=cp2lp_glob_probe
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 2: is `d_acc` reading ring openness? One arm per job.
#
# Re-scores the discriminator's own length-matched real/fake pairs with the terminal gap and
# the radius of gyration held at a shared constant. See script_utils/cp2lp_glob_probe.py for
# why those two columns are the suspect and why a collapse proves dependence while a null
# only bounds it.
#
# Inference only -- no optimizer, no Trainer, no retained graph -- but it still runs the
# arm's generator and the frozen AE decode, and pair features are O(n_target^2) with the
# receptor crop at 256 residues. a6000/48G is for that peak, not for compute: a CUDA OOM
# here can surface with no Python traceback at all.
#
# Each arm writes its OWN summary and rows file. Concurrent O_APPEND to a shared file
# NUL-corrupts rows on this filesystem (88 of 144 lost, measured).
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-onepass_internal_clashnorm}"
_cp2lp_defaults
_cp2lp_banner "glob-probe"

echo "host     : ${SLURMD_NODENAME:-$(uname -n 2>/dev/null || echo '?')}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi -L unavailable"

GLOB_DIR="${GLOB_DIR:-${TRIPLET_ROOT}/glob_probe}"
GLOB_MIX="${GLOB_MIX:-${GLOB_DIR}/cp2lp_glob_mixed_heldout.parquet}"
GLOB_N_BATCHES="${GLOB_N_BATCHES:-200}"
GLOB_BATCH="${GLOB_BATCH:-16}"
# NON-EMA by default and deliberately: the EMA files are ~0.6 GB smaller and may track the
# generator only, and a checkpoint without discriminator weights would leave a randomly
# initialised D whose ~0.5 accuracy reads as "no dependence". The probe hard-fails on a
# missing discriminator.* key rather than measuring that, but the default should not invite
# it either.
GLOB_CKPT_TAG="${GLOB_CKPT_TAG:-g010300}"

RUN_DIR="${STORE}/${RUN_NAME}"
CKPT="${RUN_DIR}/checkpoints/${GLOB_CKPT_TAG}.ckpt"

# Soft gates: exit 0 so the report stage is not stranded on DependencyNeverSatisfied with no
# explanation. Each one distinguishes "this arm has not got there yet" from "upstream never
# ran", which a non-zero exit would not.
if [[ ! -f "$CKPT" ]]; then
  echo "GATE: ${ARM} has no ${GLOB_CKPT_TAG}.ckpt under ${RUN_DIR}/checkpoints."
  ls -1 "${RUN_DIR}/checkpoints" 2>/dev/null | sed 's/^/      have: /' || echo "      (no checkpoints dir)"
  exit 0
fi
if [[ ! -f "$GLOB_MIX" ]]; then
  echo "GATE: no mixed parquet at ${GLOB_MIX} -- stage 1 (cp2lp_glob_mix) did not produce it."
  exit 0
fi

cd "$REPO"
mkdir -p "$GLOB_DIR"
# A suffix keeps a re-run from overwriting an earlier one's rows. The first `decisive`
# submission (jobs 72051/72052) predates the population audit, so its files are kept.
GLOB_TAG_SUFFIX="${GLOB_TAG_SUFFIX:-}"
TAG="${ARM}__${GLOB_CKPT_TAG}${GLOB_TAG_SUFFIX}"

set -x
"$PY" script_utils/cp2lp_glob_probe.py \
  --run-dir "$RUN_DIR" \
  --ckpt "$CKPT" \
  --cp-parquet "$GLOB_MIX" \
  --out "${GLOB_DIR}/${TAG}.json" \
  --rows-out "${GLOB_DIR}/${TAG}_rows.jsonl" \
  --n-batches "$GLOB_N_BATCHES" \
  --batch-size "$GLOB_BATCH" \
  --device cuda
