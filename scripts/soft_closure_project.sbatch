#!/bin/bash
#SBATCH --job-name=soft_closure_project
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 1 (CPU): OpenMM staged "soft closure" projection of the LNR linear peptides whose
# head-to-tail gap the SDEdit sweep could never span. See scripts/soft_closure_project.py.
#
# CPU-only on purpose: this is force-field minimization of a few thousand atoms, and OpenMM's
# CUDA path would just queue behind real GPU work for no gain at this size.
#
# Do NOT sbatch directly -- use scripts/submit_soft_closure.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"

: "${PROJ_DIR:?PROJ_DIR not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"
: "${OPENMM_PYTHON:?OPENMM_PYTHON not set}"

# OpenMM cannot coexist with the training environment's pins, so it lives in its own venv.
# This is NOT the project .venv and must not be swapped for it.
[[ -x "$OPENMM_PYTHON" ]] || { echo "FATAL: OpenMM venv missing: $OPENMM_PYTHON
Recreate it with:  uv venv --python 3.11 .venv_openmm && \\
  uv pip install --python .venv_openmm/bin/python openmm pdbfixer numpy pandas pyarrow"; exit 1; }
"$OPENMM_PYTHON" -c "import openmm, pdbfixer, pandas" \
  || { echo "FATAL: OpenMM venv is incomplete on $(hostname)"; exit 1; }
[[ -f "$LNR_METADATA" ]] || { echo "FATAL: LNR metadata missing: $LNR_METADATA"; exit 1; }

export OPENMM_CPU_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "Node: $(hostname)   OPENMM_CPU_THREADS=${OPENMM_CPU_THREADS}"
"$OPENMM_PYTHON" -c "import openmm; print('OpenMM', openmm.version.version)"

# Unquoted on purpose: space-separated lists that must word-split into argv.
# shellcheck disable=SC2086
srun "$OPENMM_PYTHON" -u scripts/soft_closure_project.py \
  --metadata "$LNR_METADATA" \
  --out-dir "$PROJ_DIR" \
  --examples ${TEST_BED_EXAMPLES} \
  --cyc-type "${CYC_TYPE:-mainchain}" \
  --n-projections "${N_PROJECTIONS:-16}" \
  --torsion-sigma-deg "${TORSION_SIGMA_DEG:-12.0}" \
  --k-pull "${K_PULL:-2000.0}" \
  --k-contact "${K_CONTACT:-500.0}" \
  --contact-tol-A "${CONTACT_TOL_A:-0.75}" \
  --contacts-per-residue "${CONTACTS_PER_RESIDUE:-1}" \
  --ladder-A ${LADDER_A} \
  --max-iterations "${MIN_MAX_ITERATIONS:-1000}" \
  --stop-cb-A "${STOP_CB_A:-9.0}" \
  --accept-retention "${ACCEPT_RETENTION:-0.8}" \
  --keep-per-example "${KEEP_PER_EXAMPLE:-4}" \
  --seed "${PROJ_SEED:-0}"

echo "DONE projection -> ${PROJ_DIR}"
