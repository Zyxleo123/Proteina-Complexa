#!/usr/bin/env bash
#SBATCH --job-name=cpseq_hf3_ctrl
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_hf3_ctrl_%j.out
#
# HighFold3 on the six crystal macrocycle controls -- the gate a second predictor must
# pass before its opinion of a designed sequence means anything.
#
# Why this is a job and not an interactive run: predictors in this project get OOM-killed
# in a node-hold job's step_extern cgroup, which reports `returncode = -9` and no Python
# traceback. That cost a debugging cycle on the PepBench preflight; see
# README_CPSEQ_SEARCH_PILOT.md sec. 11.
#
# Memory: AF3's installation notes ask for >=64 GB, which is sized for ~5000-token
# systems. These are 128-310 tokens, so 48G is the request; if a task is killed with -9,
# raise it rather than theorising.
#
# Submit:  sbatch scripts/cpseq_hf3_controls.sbatch
# Resume:  identical -- per-control output dirs are keyed by input hash

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done

ZFS="${ZFS:-/data/user_data/yixiz}"
DATASET_DIR="${DATASET_DIR:-$ZFS/cpseq/controls_tranche}"
OUT_DIR="${OUT_DIR:-$ZFS/cpseq/hf3_controls}"
MODEL_DIR="${MODEL_DIR:-$ZFS/models/af3}"
NUM_SEEDS="${NUM_SEEDS:-1}"
PARENTS="${PARENTS:-}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"

echo "=== cpseq_hf3_controls ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M"
echo "dataset:   $DATASET_DIR"
echo "out:       $OUT_DIR"
nvidia-smi -L || echo "no nvidia-smi"
cat "$ZFS/envs/highfold3/env_provenance.json" 2>/dev/null || true

EXTRA=()
[ -n "$PARENTS" ] && EXTRA+=(--parents "$PARENTS")

srun "$PY" -u script_utils/cpseq_highfold3_controls.py \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$OUT_DIR" \
  --model-dir "$MODEL_DIR" \
  --num-seeds "$NUM_SEEDS" \
  "${EXTRA[@]}"

echo "=== cpseq_hf3_controls DONE ==="
