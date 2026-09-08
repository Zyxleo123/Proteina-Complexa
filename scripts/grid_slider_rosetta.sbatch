#!/bin/bash
#SBATCH --job-name=grid_slider_dg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# CPU arm: targeted Rosetta dG for the viz cases only. The sweep's own Rosetta pass is
# chemistry-biased (its per-example cap was monopolised by disulfide), so mainchain and
# isopeptide grid points have no dG. This scores EVERY grid point of the chosen peptide(s),
# reading the complexes the sweep already wrote -- so the viewer has a before/after energy
# for every combination the selectors can reach. Independent of the GPU trajectory stage.
#
# Do NOT sbatch directly -- use scripts/submit_sdedit_grid_slider.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${SWEEP_DIR:?SWEEP_DIR not set}"
: "${ROSETTA_OUT_DIR:?ROSETTA_OUT_DIR not set}"
: "${VIZ_EXAMPLES:?VIZ_EXAMPLES not set}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_COUNT="${ROSETTA_SHARDS:-1}"

mkdir -p "$ROSETTA_OUT_DIR"
echo "Node: $(hostname)  shard ${SHARD_INDEX}/${SHARD_COUNT}"
echo "examples: ${VIZ_EXAMPLES}"

# Unquoted VIZ_EXAMPLES: a space-separated id list that must word-split into argv.
# shellcheck disable=SC2086
srun "$PY" scripts/score_sdedit_rosetta.py \
  --edits "${SWEEP_DIR}"/edits_shard*.jsonl \
  --complex-dir "${SWEEP_DIR}/complexes" \
  --examples ${VIZ_EXAMPLES} \
  --max-per-example 0 \
  --out "${ROSETTA_OUT_DIR}/rosetta_shard${SHARD_INDEX}.jsonl" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"

echo "DONE rosetta shard ${SHARD_INDEX} -> ${ROSETTA_OUT_DIR}"
