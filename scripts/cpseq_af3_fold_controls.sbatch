#!/usr/bin/env bash
#SBATCH --job-name=cpseq_af3_ctrl
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_af3_ctrl_%j.out
#
# Fold the deposited crystal macrocycles with AF3, peptide as a SMILES ligand.
#
# This is the calibration, and it is the gate on folding any design. These four complexes
# have known answers, so a pose that misses here is the instrument rather than the design.
# Boltz-2's bar on the same controls (docs/README_CPSEQ_DATASET.md sec. 7): receptor-aligned
# peptide CA RMSD 1.42 A median mainchain, 0.40 A best, every control within 2.13 A.
#
# Only the 4 MAINCHAIN controls are folded. The 2 disulfide ones (1SLD, 1SLE) are a
# different molecule and this arm cannot express them as a head-to-tail SMILES; the
# driver reports the skip rather than silently folding the wrong chemistry.
#
# Alignments come from boltz's own mmseqs2 client, cached, so these folds see the same
# alignment a Boltz-2 fold of the same receptor would. One AF3 call per complex carries
# all seeds, and the jax compilation cache is shared, so the second complex is cheaper
# than the first -- the per-fold cost in the summary table is the number to size the
# cohort run from.
#
# Submit:  sbatch scripts/cpseq_af3_fold_controls.sbatch
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
PY="${PY:-$REPO/.venv/bin/python}"
AF3_PYTHON="${AF3_PYTHON:-$ZFS/envs/af3/.venv/bin/python}"
AF3_RUN="${AF3_RUN:-$ZFS/upstream/alphafold3/run_alphafold.py}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:-$ZFS/Proteina-Complexa/community_models/ckpts/AF3}"
BOLTZ_PYTHON="${BOLTZ_PYTHON:-$ZFS/envs/boltz2/.venv/bin/python}"
MSA_CACHE="${MSA_CACHE:-$CPSEQ/msa_cache}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-$ZFS/cpseq_af3/jax_cache}"
SEEDS="${SEEDS:-101,202,303,404,505}"
REPRESENTATION="${REPRESENTATION:-polymer}"
OUT_DIR="${OUT_DIR:-$CPSEQ/af3_controls_$REPRESENTATION}"
ONLY="${ONLY:-}"

echo "=== cpseq_af3_fold_controls ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=general gres=gpu:a6000:1 cpus=${SLURM_CPUS_PER_TASK:-8} mem=96G time=8:00:00"
nvidia-smi -L || echo "WARNING: nvidia-smi reported nothing"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "seeds:    $SEEDS"
echo "peptide:  $REPRESENTATION"
echo "msa:      $MSA_CACHE (boltz mmseqs2 client)"
echo "out:      $OUT_DIR"

for f in "$PY" "$AF3_PYTHON" "$BOLTZ_PYTHON"; do
  [ -x "$f" ] || { echo "FATAL: missing interpreter $f" >&2; exit 1; }
done
[ -f "$AF3_RUN" ] || { echo "FATAL: missing $AF3_RUN" >&2; exit 1; }
[ -s "$AF3_MODEL_DIR/af3.bin" ] || { echo "FATAL: no af3.bin in $AF3_MODEL_DIR" >&2; exit 1; }
[ -s "$CPSEQ/controls/controls.json" ] || {
  echo "FATAL: no controls.json -- run scripts/cpseq_gate_rebuild.sbatch first" >&2
  exit 1
}

cd "$REPO"
mkdir -p "$OUT_DIR" "$MSA_CACHE" "$JAX_CACHE_DIR"

EXTRA=()
[ -n "$ONLY" ] && EXTRA+=(--only "$ONLY")

"$PY" script_utils/cpseq_af3_fold.py \
  --controls "$CPSEQ/controls/controls.json" \
  --out-dir "$OUT_DIR" \
  --chemistry mainchain \
  --representation "$REPRESENTATION" \
  --seeds "$SEEDS" \
  --num-diffusion-samples 1 \
  --af3-python "$AF3_PYTHON" \
  --af3-run-script "$AF3_RUN" \
  --af3-model-dir "$AF3_MODEL_DIR" \
  --boltz-python "$BOLTZ_PYTHON" \
  --msa-cache "$MSA_CACHE" \
  --jax-cache-dir "$JAX_CACHE_DIR" \
  "${EXTRA[@]}"
rc=$?

echo "=== cpseq_af3_fold_controls exit $rc ==="
exit $rc
