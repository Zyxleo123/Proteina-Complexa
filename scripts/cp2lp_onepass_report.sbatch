#!/bin/bash
#SBATCH --job-name=cp2lp_op_report
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# The five-arm comparison table and figure. CPU, and redrawable: everything it reads is a
# JSONL row already on disk, so a threshold can change and the table can be regenerated
# without sampling a single peptide.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "onepass-report"

cd "$REPO"
OUT="${REPORT_DIR}/onepass_${EVAL_CKPT_TAG}"
mkdir -p "$OUT"

ARGS=()
for arm in onepass_contactonly onepass_gan onepass_gan_ar1 onepass_gan_noise onepass_gan_nce; do
  d="${TRIPLET_ROOT}/${arm}__${EVAL_CKPT_TAG}"
  if _cp2lp_gate_generated "$d"; then
    ARGS+=(--arm "${arm}=${d}")
  fi
done
if [[ "${#ARGS[@]}" -eq 0 ]]; then
  echo "GATE: no arm has produced triplets at ${EVAL_CKPT_TAG}. Nothing to report."
  exit 0
fi

# The real-LP reference column, if the legacy reference stage has produced one. Optional:
# the arm-vs-arm comparison stands without it, but every geometry number is much easier to
# read against real bound LPs measured by the same code.
REF_ARG=()
[[ -d "${REFERENCE_DIR}" ]] && REF_ARG=(--real-reference "${REFERENCE_DIR}")

set -x
"$PY" script_utils/cp2lp_onepass_report.py \
  "${ARGS[@]}" \
  "${REF_ARG[@]}" \
  --out-dir "$OUT" \
  --reference-arm onepass_gan \
  --control-arm onepass_contactonly
