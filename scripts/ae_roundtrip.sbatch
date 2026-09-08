#!/bin/bash
#SBATCH --job-name=ae_roundtrip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# GPU arm of the AE round-trip go/no-go (see scripts/eval_ae_roundtrip.py for the why).
#
# Do NOT sbatch this directly -- use scripts/submit_ae_roundtrip.sh, which writes the env
# file this sources as $1 and fills in --partition/--nodelist/--time from a fresh `myfree`
# at submission time (availability changes run to run, so it is not hardcoded here).
#
# The AE checkpoint is ~3.9 GB and loading it OOM-killed a login shell, hence --mem=64G.
#
# set -euo pipefail lives HERE, not only in env.sh: a failed source must not leave this
# running under the system python and still exit 0, which would release the summarize job
# behind a stage that did no work.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set in $ENV_FILE}"
: "${LNR_ROOT:?LNR_ROOT not set in $ENV_FILE}"
: "${LNR_STAGE_DIR:?LNR_STAGE_DIR not set in $ENV_FILE}"
CONFIG_NAME="${CONFIG_NAME:-example/training_cpsea_peptide_smoke}"
CPSEA_VAL_METADATA="${CPSEA_VAL_METADATA:-${CPSEA_DATA_PATH}/preprocessed/metadata/cpsea_val.parquet}"
CONTROL_LIMIT="${CONTROL_LIMIT:-0}"
SEED="${SEED:-42}"

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $PY"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable from $PY"; exit 1; }
echo "AE checkpoint: ${CPSEA_AE_CKPT_PATH}"
[[ -f "${CPSEA_AE_CKPT_PATH}" ]] || { echo "FATAL: AE checkpoint missing: ${CPSEA_AE_CKPT_PATH}"; exit 1; }

mkdir -p "$RESULTS_DIR"
LNR_METADATA="${LNR_STAGE_DIR}/metadata/lnr_test.parquet"

# Staging is seconds of pure file I/O, so it rides along here rather than costing a queue
# round-trip in its own CPU job. Idempotent: skipped when the parquet already exists.
if [[ -f "$LNR_METADATA" ]]; then
  echo "LNR already staged: $LNR_METADATA"
else
  echo "Staging LNR from $LNR_ROOT ..."
  "$PY" scripts/build_lnr_metadata.py --lnr-root "$LNR_ROOT" --out-dir "$LNR_STAGE_DIR"
fi

echo
echo "=== ARM 1/2: LNR linear peptides (the question) ==="
srun "$PY" scripts/eval_ae_roundtrip.py \
  --config-name "$CONFIG_NAME" \
  --metadata "$LNR_METADATA" \
  --arm lnr_linear_crystal \
  --out "${RESULTS_DIR}/lnr_linear_crystal.jsonl" \
  --seed "$SEED"

echo
echo "=== ARM 2/2: CPSea cyclic val (the in-distribution reference) ==="
# Without this arm an absolute RMSE is uninterpretable -- the result IS the comparison.
srun "$PY" scripts/eval_ae_roundtrip.py \
  --config-name "$CONFIG_NAME" \
  --metadata "$CPSEA_VAL_METADATA" \
  --arm cpsea_cyclic_val \
  --out "${RESULTS_DIR}/cpsea_cyclic_val.jsonl" \
  --limit "$CONTROL_LIMIT" \
  --seed "$SEED"

echo
echo "Rows written:"
wc -l "${RESULTS_DIR}"/*.jsonl
echo "DONE"
