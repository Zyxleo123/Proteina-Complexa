#!/bin/bash
#SBATCH --job-name=ddib_invert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# GPU arm of the DDIB inversion probe (see scripts/ddib_invert.py): integrate the public
# Proteina-Complexa flow BACKWARDS from each LNR linear peptide and record, at a grid of
# times, whether the termini approach each other faster than the pose is destroyed.
#
# Job ARRAY over shards of the input peptides; each task writes its OWN results file
# (concurrent O_APPEND has NUL-corrupted rows on this cluster). Each task walks every
# track arm in TRACK_ARMS sequentially.
#
# Do NOT sbatch directly -- use scripts/submit_ddib_invert.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${SRC_FLOW_CKPT:?SRC_FLOW_CKPT not set}"
: "${SRC_AE_CKPT:?SRC_AE_CKPT not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable"; exit 1; }
[[ -f "$SRC_FLOW_CKPT" ]] || { echo "FATAL: source flow ckpt missing: $SRC_FLOW_CKPT"; exit 1; }
[[ -f "$SRC_AE_CKPT" ]]   || { echo "FATAL: source AE ckpt missing: $SRC_AE_CKPT"; exit 1; }
[[ -f "$LNR_METADATA" ]]  || { echo "FATAL: LNR metadata missing: $LNR_METADATA"; exit 1; }

# The source flow and its OWN autoencoder. `Proteina.load_from_checkpoint` also honours
# CPSEA_AE_CKPT_PATH through the config, and env.sh points that at the CPSea finetuned AE --
# which is a DIFFERENT latent space. Override it here so nothing can pick up the wrong one.
export CPSEA_AE_CKPT_PATH="$SRC_AE_CKPT"

mkdir -p "$RESULTS_DIR"
echo "shard ${SHARD_INDEX}/${SHARD_COUNT}"
echo "source flow: $SRC_FLOW_CKPT"
echo "source AE:   $SRC_AE_CKPT"
echo "track arms:  ${TRACK_ARMS}"

# TRACK_ARMS is a comma-separated list of space-separated track sets, e.g.
#   "bb_ca,bb_ca local_latents"
IFS=',' read -r -a ARMS <<< "$TRACK_ARMS"
for ARM in "${ARMS[@]}"; do
  TAG="$(echo "$ARM" | tr ' ' '+')"
  echo "=== arm ${TAG} ==="
  # Unquoted on purpose: space-separated lists that must word-split into argv.
  # shellcheck disable=SC2086
  srun "$PY" scripts/ddib_invert.py \
    --flow-ckpt "$SRC_FLOW_CKPT" \
    --ae-ckpt "$SRC_AE_CKPT" \
    --metadata "$LNR_METADATA" \
    --out "${RESULTS_DIR}/invert_${TAG}_shard${SHARD_INDEX}.jsonl" \
    --invert-tracks $ARM \
    --record-ts ${RECORD_TS} \
    --roundtrip-ts ${ROUNDTRIP_TS} \
    --seeds ${SEEDS} \
    --nsteps "${NSTEPS:-0}" \
    --limit "${PEPTIDE_LIMIT:-0}" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT"
done

echo "DONE shard ${SHARD_INDEX}"
