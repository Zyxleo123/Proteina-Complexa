#!/bin/bash
#SBATCH --job-name=flowedit_report
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Aggregate to tables + the results document. Reads only saved artefacts, so every figure
# and number re-renders without re-sampling or re-scoring anything.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/worktrees/flowedit}/scripts/_flowedit_preamble.sh"
[[ -f "${PREAMBLE}" ]] || { echo "FATAL: preamble not found: ${PREAMBLE}"; exit 1; }
# shellcheck disable=SC1090
source "${PREAMBLE}"
_fe_consume_args "$@"
_fe_defaults
_fe_banner "report"

shopt -s nullglob
CANDS=("${CAND_DIR}"/cand_*.jsonl)
ROS=("${ROSETTA_DIR}"/rosetta_shard*.jsonl)
shopt -u nullglob
if [[ ${#CANDS[@]} -eq 0 ]]; then
  echo "GATE: no candidate files under ${CAND_DIR}; nothing to report. Exiting 0."
  exit 0
fi
echo "candidates: ${#CANDS[@]} file(s), rosetta: ${#ROS[@]} file(s)"

"${PY}" script_utils/flowedit_report.py \
  --candidates "${CANDS[@]}" \
  ${ROS[0]+--rosetta "${ROS[@]}"} \
  --out-dir "${OUT_DIR}/report" \
  --run-tag "${RUN_TAG}"

echo "DONE report -> ${OUT_DIR}/report"
