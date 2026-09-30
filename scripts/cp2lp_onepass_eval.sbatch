#!/bin/bash
#SBATCH --job-name=cp2lp_op_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Generate the fixed evaluation set for ONE one-pass arm: every parent in the manifest,
# under every seed in it, with the same checkpoint milestone in every arm.
#
# a5000 is enough -- inference only, one network evaluation per sample, no discriminator
# and no retained graph. The editor is the cheapest generation path in this project.
#
# Resumable: each shard drops a .done marker and writes its OWN manifest. Concurrent
# O_APPEND to a shared file NUL-corrupts rows on this filesystem, and the report REFUSES a
# corrupt row rather than averaging past it.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-onepass_gan}"
_cp2lp_defaults
_cp2lp_banner "onepass-eval"
_cp2lp_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

RUN_DIR="${STORE}/${RUN_NAME}"
CKPT="${RUN_DIR}/checkpoints/${EVAL_CKPT_TAG}.ckpt"
if [[ ! -f "$CKPT" ]]; then
  echo "GATE: ${ARM} has no ${EVAL_CKPT_TAG}.ckpt under ${RUN_DIR}/checkpoints."
  echo "      Milestones are written at the configured GENERATOR-update counts"
  echo "      (cp2lp.eval_checkpoint_g_updates), as g001300/g005300/g010300[-EMA].ckpt."
  ls -1 "${RUN_DIR}/checkpoints" 2>/dev/null | sed 's/^/      have: /' || true
  exit 0
fi

MANIFEST="${EVAL_MANIFEST:-${LPDATA}/cp2lp_eval64.parquet}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "GATE: no evaluation manifest at ${MANIFEST}."
  echo "      Build it: bash scripts/submit_cp2lp_onepass.sh --submit manifest"
  exit 0
fi

cd "$REPO"
OUT="${TRIPLET_ROOT}/${ARM}__${EVAL_CKPT_TAG}"
mkdir -p "$OUT"

# `--n-cp -1` takes EVERY parent in the manifest rather than the first N: the manifest is
# already exactly the intended set, and a count here could silently truncate it.
set -x
"$PY" script_utils/cp2lp_generate.py \
  --run-dir "$RUN_DIR" \
  --ckpt "$CKPT" \
  --cp-parquet "$MANIFEST" \
  --out-dir "$OUT" \
  --n-cp -1 \
  --samples-per-cp "$EVAL_SAMPLES_PER_CP" \
  --seeds "$EVAL_SEEDS" \
  --batch-size "$EVAL_BATCH" \
  --expect-parents "$EVAL_N_CP" \
  --write-rejected \
  --device cuda
