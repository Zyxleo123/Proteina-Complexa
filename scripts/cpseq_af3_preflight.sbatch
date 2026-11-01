#!/usr/bin/env bash
#SBATCH --job-name=cpseq_af3_preflight
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_af3_preflight_%j.out
#
# Does the AF3 install fold a cyclic-peptide complex on a GPU here, and what does it
# write? See script_utils/cpseq_af3_preflight.py for what this answers and what it
# deliberately does not.
#
# Resources, and why:
#   a6000 (48 GB)  the GPU class with free capacity on `general`. AF3's memory scales
#                  with tokens, and a ligand-tokenised 7-mer plus a ~180-residue
#                  receptor is a few hundred tokens -- small. The 2080 Ti nodes (11 GB)
#                  are deliberately avoided: this project has already lost time to a
#                  CUDA allocation failure killing a process with no Python traceback.
#   mem=96G        AF3's installation notes ask for >=64 GB host RAM, and `build_data`
#                  showed the CCD side of it is genuinely memory-hungry.
#   NOT the debug or legacy partitions: 2 h is too short for a first run that includes
#                  jax compilation, and the GPU classes there are too small.
#
# Submit:  sbatch scripts/cpseq_af3_preflight.sbatch
# Config:  positional args -- a path to source, or VAR=value. Never `sbatch --export`.

set -euo pipefail

for arg in "$@"; do
  if [ -f "$arg" ]; then echo "[cfg] sourcing $arg"; # shellcheck disable=SC1090
    . "$arg"
  elif [[ "$arg" == *=* ]]; then echo "[cfg] export $arg"; export "$arg"
  fi
done

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
CPSEQ="${CPSEQ:-$ZFS/cpseq}"
AF3_ENV="${AF3_ENV:-$ZFS/envs/af3}"
AF3_PYTHON="${AF3_PYTHON:-$AF3_ENV/.venv/bin/python}"
AF3_RUN="${AF3_RUN:-$ZFS/upstream/alphafold3/run_alphafold.py}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:-$ZFS/Proteina-Complexa/community_models/ckpts/AF3}"
PY="${PY:-$REPO/.venv/bin/python}"
PDB_ID="${PDB_ID:-5TU6}"
OUT_DIR="${OUT_DIR:-$CPSEQ/af3_preflight/$PDB_ID}"
SEED="${SEED:-101}"

# Every batch script in this project prints its host, GPU and request into the .out,
# because `sacct` is dead here (slurmdbd unreachable, and it exits 0 with no rows) so
# there is no accounting record to recover the node from after the fact.
echo "=== cpseq_af3_preflight ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=general gres=gpu:a6000:1 cpus=${SLURM_CPUS_PER_TASK:-8} mem=96G time=4:00:00"
nvidia-smi -L || echo "WARNING: nvidia-smi reported nothing"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "af3 env:  $AF3_ENV"
echo "weights:  $AF3_MODEL_DIR/af3.bin"
echo "control:  $PDB_ID"
echo "out:      $OUT_DIR"

for f in "$AF3_PYTHON" "$AF3_RUN" "$PY"; do
  [ -x "$f" ] || [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
[ -s "$AF3_MODEL_DIR/af3.bin" ] || { echo "FATAL: no af3.bin in $AF3_MODEL_DIR" >&2; exit 1; }
[ -s "$CPSEQ/controls/controls.json" ] || {
  echo "FATAL: no controls.json -- run scripts/cpseq_gate_rebuild.sbatch first" >&2
  exit 1
}

cd "$REPO"
mkdir -p "$OUT_DIR"

"$PY" script_utils/cpseq_af3_preflight.py \
  --controls "$CPSEQ/controls/controls.json" \
  --pdb-id "$PDB_ID" \
  --out-dir "$OUT_DIR" \
  --af3-python "$AF3_PYTHON" \
  --af3-run-script "$AF3_RUN" \
  --af3-model-dir "$AF3_MODEL_DIR" \
  --jax-cache-dir "${JAX_CACHE_DIR:-$ZFS/cpseq_af3/jax_cache}" \
  --seed "$SEED"
rc=$?

echo "=== cpseq_af3_preflight exit $rc ==="
exit $rc
