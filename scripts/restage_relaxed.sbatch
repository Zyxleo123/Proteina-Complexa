#!/bin/bash
#SBATCH --job-name=restage_relaxed
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# ARM C stage 2 (CPU): crop the RELAXED complexes to the CPSea-style pocket.
# Relax first, crop second -- cropping first would minimize severed chain ends into the
# vacuum they were cut from.
#
# Do NOT sbatch directly -- use scripts/submit_uncond_relaxed.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RELAX_DIR:?RELAX_DIR not set}"
: "${SRC_METADATA:?SRC_METADATA not set}"
: "${POCKET_DIR:?POCKET_DIR not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

# GATE: if stage 1 produced nothing, say so and exit 0. A non-zero exit here would strand
# every downstream job in DependencyNeverSatisfied with no explanation.
shopt -s nullglob
RELAXED=( "${RELAX_DIR}"/pdbs/*.pdb )
if (( ${#RELAXED[@]} == 0 )); then
  echo "GATE: no relaxed PDBs in ${RELAX_DIR}/pdbs -- the relax array produced nothing."
  echo "GATE: read the relax_lnr array .out files. Exiting 0."
  exit 0
fi
echo "${#RELAXED[@]} relaxed complexes found"

"$PY" scripts/restage_lnr_pocket.py \
  --in-metadata "$SRC_METADATA" \
  --pdb-dir "${RELAX_DIR}/pdbs" \
  --out-dir "$POCKET_DIR" \
  --radius "${POCKET_RADIUS:-18.0}"

# The generation stage reads LNR_METADATA; make sure stage 2 actually produced it.
[[ -f "$LNR_METADATA" ]] || { echo "FATAL: expected $LNR_METADATA to exist after restaging"; exit 1; }
echo "DONE restage -> $LNR_METADATA"
