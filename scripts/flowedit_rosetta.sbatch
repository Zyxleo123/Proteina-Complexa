#!/bin/bash
#SBATCH --job-name=flowedit_rosetta
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j_%a.out
#SBATCH --error=slurm_logs/%x_%j_%a.out
#
# Paired interface dG on the shortlist. CPU: it reads artefacts the GPU stage wrote, so
# asking for a GPU would just queue this behind real work.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/worktrees/flowedit}/scripts/_flowedit_preamble.sh"
[[ -f "${PREAMBLE}" ]] || { echo "FATAL: preamble not found: ${PREAMBLE}"; exit 1; }
# shellcheck disable=SC1090
source "${PREAMBLE}"
_fe_consume_args "$@"
_fe_defaults
_fe_banner "rosetta"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${ROSETTA_DIR}" slurm_logs

shopt -s nullglob
CANDS=("${CAND_DIR}"/cand_*.jsonl)
shopt -u nullglob
# A gate that fails must say why and exit 0: a non-zero exit would strand every dependent in
# DependencyNeverSatisfied with no explanation, and "the GPU stage wrote nothing" is a
# different situation from "this stage broke".
if [[ ${#CANDS[@]} -eq 0 ]]; then
  echo "GATE: no candidate files under ${CAND_DIR} -- the generation stage produced nothing."
  echo "Nothing to score. Exiting 0 so dependents can report the same thing."
  exit 0
fi
echo "scoring from ${#CANDS[@]} candidate file(s)"

"${PY}" script_utils/flowedit_rosetta.py \
  --candidates "${CANDS[@]}" \
  --out "${ROSETTA_DIR}/rosetta_shard${SHARD_INDEX}.jsonl" \
  --repeats "${ROSETTA_REPEATS}" \
  --per-group "${ROSETTA_SHORTLIST}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count "${SLURM_ARRAY_TASK_COUNT:-1}"

echo "DONE rosetta shard ${SHARD_INDEX}"
