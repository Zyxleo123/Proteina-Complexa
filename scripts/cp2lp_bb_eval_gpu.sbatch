#!/bin/bash
#SBATCH --job-name=cp2lp_bb_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# GPU half of a checkpoint evaluation: generate the panel's backbones and dump them with
# their receptors. Packing is a separate CPU job -- asking for a GPU to run Rosetta would
# queue it behind real training.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-bb_torsion}"
_cp2lp_defaults
_cp2lp_banner "bb-eval-gpu"
cd "$REPO"

BB_PANEL="${BB_PANEL:-${LPDATA}/cp2lp_bb_panel24.parquet}"
BB_SEEDS="${BB_SEEDS:-0,1,2,3}"
BB_EVAL_ROOT="${BB_EVAL_ROOT:-${TRIPLET_ROOT}/bb_eval}"
CKPT="${BB_EVAL_CKPT:-${STORE}/${RUN_NAME}/${EVAL_CKPT_TAG}.ckpt}"
OUT="${BB_EVAL_ROOT}/${EVAL_CKPT_TAG}"

if [[ ! -f "${BB_PANEL}" ]]; then
  echo "GATE: no panel at ${BB_PANEL}. Build it first:"
  echo "  bash scripts/submit_cp2lp_backbone.sh --submit panel"
  exit 0
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "GATE: no checkpoint at ${CKPT} -- this milestone has not been reached yet."
  echo "      Available in ${STORE}/${RUN_NAME}:"
  ls -1 "${STORE}/${RUN_NAME}" 2>/dev/null | sed 's/^/        /' || echo "        (run dir missing)"
  exit 0
fi

mkdir -p "${OUT}"
"${PY}" script_utils/cp2lp_bb_evaluate.py \
  --config "${CONFIG_NAME}" \
  --ckpt "${CKPT}" \
  --ckpt-tag "${EVAL_CKPT_TAG}" \
  --manifest "${BB_PANEL}" \
  --out-dir "${OUT}" \
  --seeds "${BB_SEEDS}" \
  --source-torsion-baseline

echo "[done] dumps under ${OUT}"
