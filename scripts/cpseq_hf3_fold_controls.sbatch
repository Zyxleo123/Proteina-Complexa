#!/usr/bin/env bash
#SBATCH --job-name=cpseq_hf3_ctrl
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_hf3_ctrl_%j.out
#
# THE GATE. Fold the deposited crystal macrocycles with HighFold3 and measure whether the
# ring actually closes. Nothing downstream runs until this passes.
#
# Why a gate and not a formality: stock AF3 accepted both of its native representations at
# the input layer and produced the wrong chemistry both times --
#   SMILES ligand   ring closed at 1.372 A, but 18/20 folds kept D-amino acids after relax
#   polymer + bond  chirality correct, 0/20 rings closed, gaps 2.15-16.64 A, OXT on every
#                   C-terminus, i.e. the declared bond never entered the chemical graph
# HighFold3's cyclic capability is currently established only by probing its
# position-encoding matrix (verified: ring follows chain 1, peptide-first 5x5). A matrix
# probe is not a prediction coming back closed.
#
# And the failure would not be caught downstream: under v5 scoring an open ring is FORCED
# shut by the relax restraint and then certified valid -- 12/19 on folds that were 0/20
# closed, with energies from 0.0 to +49,500 REU. So closure is measured here, at fold
# time, before any scoring can launder it.
#
# Weights: HighFold3 is AF3 a8ecdb2d, so it loads the same af3.bin.
#
# Submit:  sbatch scripts/cpseq_hf3_fold_controls.sbatch
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
HF3_ENV="${HF3_ENV:-$ZFS/envs/highfold3}"
HF3_PYTHON="${HF3_PYTHON:-$HF3_ENV/.venv/bin/python}"
HF3_RUN="${HF3_RUN:-$ZFS/upstream/highfold3/run_alphafold.py}"
MODEL_DIR="${MODEL_DIR:-$ZFS/Proteina-Complexa/community_models/ckpts/AF3}"
BOLTZ_PYTHON="${BOLTZ_PYTHON:-$ZFS/envs/boltz2/.venv/bin/python}"
MSA_CACHE="${MSA_CACHE:-$CPSEQ/msa_cache}"
SEEDS="${SEEDS:-101,202,303,404,505}"
OUT_DIR="${OUT_DIR:-$CPSEQ/hf3_controls}"
ONLY="${ONLY:-}"

echo "=== cpseq_hf3_fold_controls ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=general gres=gpu:a6000:1 cpus=${SLURM_CPUS_PER_TASK:-8} mem=64G time=4:00:00"
nvidia-smi -L || echo "WARNING: nvidia-smi reported nothing"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "hf3 env:  $HF3_ENV"
echo "weights:  $MODEL_DIR/af3.bin"
echo "seeds:    $SEEDS"
echo "out:      $OUT_DIR"

for f in "$PY" "$HF3_PYTHON" "$BOLTZ_PYTHON"; do
  [ -x "$f" ] || { echo "FATAL: missing interpreter $f" >&2; exit 1; }
done
[ -f "$HF3_RUN" ] || { echo "FATAL: missing $HF3_RUN" >&2; exit 1; }
[ -s "$MODEL_DIR/af3.bin" ] || { echo "FATAL: no af3.bin in $MODEL_DIR" >&2; exit 1; }
[ -s "$CPSEQ/controls/controls.json" ] || {
  echo "FATAL: no controls.json -- run scripts/cpseq_gate_rebuild.sbatch first" >&2
  exit 1
}

# AF3 REFUSES to run on a compute-capability 7.x GPU (V100, T4) unless this XLA pass is
# disabled, and says so by name:
#   ValueError: For devices with GPU compute capability 7.x the ENV XLA_FLAGS must
#   include "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter".
# Set from the DETECTED capability rather than hardcoded, because this cluster's pool is
# mixed (a6000 is 8.6, v100 is 7.0, h200 is 9.0) and a job lands where there is room.
# Applying it unconditionally would disable the pass on GPUs that want it.
HF3_EXTRA=()
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
echo "gpu compute capability: ${CC:-unknown}"
case "$CC" in
  7.*)
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
    echo "XLA_FLAGS: $XLA_FLAGS  (required for compute capability $CC)"
    # Second 7.x requirement, also refused by name:
    #   ValueError: ... the --flash_attention_implementation must be set to "xla".
    # This is SLOWER than the default triton kernel, so a V100 fold costs more than an
    # a6000 fold -- relevant when sizing the main run from this one's timings.
    HF3_EXTRA+=(--extra-arg=--flash_attention_implementation=xla)
    echo "flash attention: xla (required for compute capability $CC; slower than default)"
    ;;
  *)
    echo "XLA_FLAGS: unset (not needed at compute capability ${CC:-unknown})"
    ;;
esac

cd "$REPO"
mkdir -p "$OUT_DIR" "$MSA_CACHE"

EXTRA=("${HF3_EXTRA[@]}")
[ -n "$ONLY" ] && EXTRA+=(--only "$ONLY")

"$PY" script_utils/cpseq_hf3_fold.py \
  --controls "$CPSEQ/controls/controls.json" \
  --out-dir "$OUT_DIR" \
  --chemistry mainchain \
  --seeds "$SEEDS" \
  --hf3-python "$HF3_PYTHON" \
  --hf3-run-script "$HF3_RUN" \
  --model-dir "$MODEL_DIR" \
  --boltz-python "$BOLTZ_PYTHON" \
  --msa-cache "$MSA_CACHE" \
  "${EXTRA[@]}"
rc=$?

echo "=== cpseq_hf3_fold_controls exit $rc ==="
exit $rc
