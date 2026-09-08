#!/bin/bash
#SBATCH --job-name=relax_lnr
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# ARM C stage 1 (CPU): AlphaFold-style restrained Amber minimization of the LNR complexes.
# Job ARRAY -- each task takes a shard and writes its OWN log (concurrent O_APPEND to one
# file has NUL-corrupted rows on this cluster).
#
# No GPU on purpose: this is OpenMM CPU minimization, and asking for a GPU would only queue
# it behind real training work.
#
# Do NOT sbatch directly -- use scripts/submit_uncond_relaxed.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"

: "${RELAX_DIR:?RELAX_DIR not set}"
: "${SRC_METADATA:?SRC_METADATA not set}"
: "${OPENMM_PYTHON:?OPENMM_PYTHON not set}"
SHARD_COUNT="${RELAX_SHARDS:-6}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

# OpenMM lives in its own venv -- it cannot coexist with the training environment's pins.
# This is NOT the project .venv and must not be swapped for it.
[[ -x "$OPENMM_PYTHON" ]] || { echo "FATAL: OpenMM venv missing: $OPENMM_PYTHON
Recreate it with:  uv venv --python 3.11 .venv_openmm && \\
  uv pip install --python .venv_openmm/bin/python openmm pdbfixer numpy pandas pyarrow"; exit 1; }
"$OPENMM_PYTHON" -c "import openmm, pdbfixer, pandas" \
  || { echo "FATAL: OpenMM venv incomplete on $(hostname)"; exit 1; }
[[ -f "$SRC_METADATA" ]] || { echo "FATAL: source metadata missing: $SRC_METADATA"; exit 1; }

export OPENMM_CPU_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "Node: $(hostname)  OPENMM_CPU_THREADS=${OPENMM_CPU_THREADS}  shard ${SHARD_INDEX}/${SHARD_COUNT}"
"$OPENMM_PYTHON" -c "import openmm; print('OpenMM', openmm.version.version)"

srun "$OPENMM_PYTHON" -u scripts/relax_lnr_complexes.py \
  --in-metadata "$SRC_METADATA" \
  --out-dir "$RELAX_DIR" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"

echo "DONE relax shard ${SHARD_INDEX}"
