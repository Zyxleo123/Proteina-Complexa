#!/bin/bash
#SBATCH --job-name=soft_closure_sdedit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# Stage 2 (GPU): the SDEdit arm. Run TWICE, on the same grid and the same seeds:
#
#   ARM=projected   the soft-closure-projected peptides
#   ARM=control     the SAME peptides, unprojected
#
# The control is not optional. The original sweep scored these peptides at one seed on a
# different grid, so a bare "projected closes N%" would be confounded with that difference;
# only the paired arms isolate the projection.
#
# Do NOT sbatch directly -- use scripts/submit_soft_closure.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
ARM="${2:?FATAL: pass the arm (projected|control) as the second argument}"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${PROJ_DIR:?PROJ_DIR not set}"
: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${FLOW_CKPT_PATH:?FLOW_CKPT_PATH not set}"

case "$ARM" in
  projected) METADATA="${PROJ_DIR}/metadata/projected.parquet" ;;
  control)   METADATA="${PROJ_DIR}/metadata/control.parquet" ;;
  *) echo "FATAL: unknown arm '$ARM' (expected projected|control)"; exit 1 ;;
esac

# The upstream GATE. A missing projected parquet means no replica passed every acceptance
# test -- a real result from the feasibility oracle, not a crash. Exit 0 so the dependent
# jobs still run and report it, instead of stranding them in DependencyNeverSatisfied.
if [[ ! -f "$METADATA" ]]; then
  if [[ -f "${PROJ_DIR}/projection_summary.json" ]]; then
    echo "GATE: arm '${ARM}' has no metadata at ${METADATA}."
    echo "GATE: the projection's own verdict:"
    cat "${PROJ_DIR}/projection_summary.json"
  else
    echo "GATE: upstream projection never produced output at ${PROJ_DIR}."
  fi
  echo "GATE: nothing to edit for arm '${ARM}'. Exiting 0."
  exit 0
fi

FLOW_CKPT_NAME="${FLOW_CKPT_NAME:-last-EMA.ckpt}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
OUT_DIR="${RESULTS_DIR}/${ARM}"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable"; exit 1; }
[[ -f "${FLOW_CKPT_PATH}/${FLOW_CKPT_NAME}" ]] || { echo "FATAL: flow ckpt missing"; exit 1; }
[[ -f "${CPSEA_AE_CKPT_PATH}" ]] || { echo "FATAL: AE ckpt missing"; exit 1; }

mkdir -p "$OUT_DIR"
echo "arm=${ARM}  inputs=${METADATA}  shard ${SHARD_INDEX}/${SHARD_COUNT}"

# Shards never share an output path: concurrent O_APPEND has NUL-corrupted rows here before.
# shellcheck disable=SC2086
srun "$PY" scripts/sdedit_cyclize.py \
  --ckpt-path "$FLOW_CKPT_PATH" \
  --ckpt-name "$FLOW_CKPT_NAME" \
  --metadata "$METADATA" \
  --out "${OUT_DIR}/edits_shard${SHARD_INDEX}.jsonl" \
  --pdb-dir "${OUT_DIR}/pdbs" \
  --cyc-types ${CYC_TYPE:-mainchain} \
  --t-ca ${T_CA_GRID} \
  --t-lat ${T_LAT_GRID} \
  --seeds ${SEEDS} \
  --nsteps "${NSTEPS:-0}" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"

echo "DONE arm=${ARM} shard ${SHARD_INDEX}"
