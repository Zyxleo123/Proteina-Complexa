#!/bin/bash
#SBATCH --job-name=cp2lp_gradnorm
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Weighted per-term generator gradient norms at the clash arm's fork point, with the clash
# term scored under BOTH normalisations on the same decoded output.
#
# a6000 for the same reason the training arms pin it: the editor forward, the AE decode of
# both the generated and the source structures, and the discriminator all have to be
# resident, and a 24G card dies here with a CUDA allocation failure and NO Python traceback.
#
# Short and read-only -- it writes one JSON and no checkpoints. No --requeue: if it is
# preempted, rerun it; there is no partial state worth resuming.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "clash-gradnorm"
_cp2lp_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

cd "$REPO"
mkdir -p slurm_logs

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Measure at the arm's OWN fork point by default -- the seeded `last.ckpt` that this arm and
# its control both resume from. That is the state the decision applies to; measuring at the
# warmup fork would describe a model 10,000 updates younger.
GRADNORM_CKPT="${GRADNORM_CKPT:-${STORE}/cp2lp_onepass_internal_clashnorm/checkpoints/last.ckpt}"
GRADNORM_CONFIG="${GRADNORM_CONFIG:-example/training_cp2lp_onepass_internal_clashnorm}"
GRADNORM_BATCHES="${GRADNORM_BATCHES:-4}"
GRADNORM_OUT="${GRADNORM_OUT:-${STORE}/cp2lp_onepass_internal_clashnorm/clash_gradnorm_${SLURM_JOB_ID:-local}.json}"

if [[ ! -f "$GRADNORM_CKPT" ]]; then
  echo "ERROR: no checkpoint at ${GRADNORM_CKPT}" >&2
  echo "       Seed it first: bash scripts/pin_cp2lp_resume.sh cp2lp_onepass_internal cp2lp_onepass_internal_clashnorm" >&2
  exit 1
fi

echo "ckpt    : ${GRADNORM_CKPT}"
echo "config  : ${GRADNORM_CONFIG}"
echo "batches : ${GRADNORM_BATCHES}"
echo "out     : ${GRADNORM_OUT}"

set -x
"$PY" script_utils/cp2lp_clash_gradnorm.py \
  --config "$GRADNORM_CONFIG" \
  --ckpt "$GRADNORM_CKPT" \
  --batches "$GRADNORM_BATCHES" \
  --out "$GRADNORM_OUT"
