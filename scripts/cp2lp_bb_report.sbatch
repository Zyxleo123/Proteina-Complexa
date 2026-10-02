#!/bin/bash
#SBATCH --job-name=cp2lp_bb_report
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# The arm comparison. CPU and REDRAWABLE: everything it reads is already a row on disk, so a
# threshold can change and the tables and the figure regenerate without sampling or packing.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
# Arm-agnostic stage: it reads EVAL_META / the packed rows and nothing arm-specific.
# Pinned anyway so the banner does not advertise the preamble's `gan` default and
# name the legacy rollout config in a backbone-arm job's log.
ARM="${ARM:-bb_torsion}"
_cp2lp_defaults
_cp2lp_banner "bb-report"
cd "$REPO"

BB_EVAL_ROOT="${BB_EVAL_ROOT:-${TRIPLET_ROOT}/bb_eval}"
OUT="${BB_EVAL_ROOT}/${EVAL_CKPT_TAG}"

shopt -s nullglob
FILES=("${OUT}"/packed__*.jsonl)
if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "GATE: no packed__*.jsonl under ${OUT} -- the packing stage never ran, so there is no"
  echo "      completed-LP result to report. A backbone-only summary would need the attempts"
  echo "      files instead, and must be labelled as such."
  exit 0
fi

"${PY}" script_utils/cp2lp_bb_report.py \
  --packed-dir "${OUT}" \
  --out-dir "${OUT}/report" \
  --ckpt-tag "${EVAL_CKPT_TAG}" \
  --plot

echo "[done] report under ${OUT}/report"
