#!/bin/bash
#SBATCH --job-name=hotspot_probe
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/Proteina-Complexa/logs/%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/Proteina-Complexa/logs/%x_%j.out
#
# Does the contact/hotspot construction pipeline work on the LP datasets?
#
# Hotspots are receptor residues at an interface, so this only works if the LP rows carry
# a receptor. PepBench (3,433 train) and ProtFrag (42,275 train) are peptide-protein
# complexes -- chain A receptor, chain B peptide -- and the LP mixing config runs the same
# `ContactTransform` with byte-identical settings as CPSea. This measures that instead of
# trusting it: a conditioning channel that is silently zero looks exactly like one that
# works, which is how `hotspot_mask_seq` went unnoticed.
#
# CPU partition deliberately: this is dataloading plus feature tensors, no model forward
# and no GPU. Asking for a GPU to audit a transform would just queue behind real work.
#
# `--output=` is an absolute scratch path because it is static -- a relative path lands in
# the repo on home.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/script_utils/hotspot_pipeline_probe.py" ]] || REPO="$REPO_DEFAULT"
cd "$REPO"

# Config travels as a positional arg / env file, never via `sbatch --export` (which sets
# SLURM_GET_USER_ENV=1, fails the login-env rebuild, and leaves the job requeued+held).
for a in "$@"; do
  if [[ -f "$a" ]]; then
    # shellcheck disable=SC1090
    source "$a"
  elif [[ "$a" == *=* ]]; then
    export "${a?}"
  fi
done

PY="${PY:-${REPO}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-example/training_cpsea_lpmix_from_v4cfg}"
NUM_BATCHES="${NUM_BATCHES:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/Proteina-Complexa/logs/hotspot_probe}"
TAG="${TAG:-$(echo "$CONFIG_NAME" | tr '/' '_')_${SLURM_JOB_ID:-local}}"

echo "host      : ${SLURMD_NODENAME:-$(uname -n)}"
echo "resources : partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=48G"
echo "python    : $PY"
echo "config    : $CONFIG_NAME"
echo "batches   : $NUM_BATCHES x $BATCH_SIZE"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi -L unavailable (expected on cpu partition)"

[[ -x "$PY" ]] || { echo "GATE: no interpreter at $PY"; exit 0; }

mkdir -p "$OUT_DIR"
set -x
"$PY" script_utils/hotspot_pipeline_probe.py \
  --config-name "$CONFIG_NAME" \
  --num-batches "$NUM_BATCHES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --out "${OUT_DIR}/${TAG}.json" \
  --rows-out "${OUT_DIR}/${TAG}_rows.jsonl"
