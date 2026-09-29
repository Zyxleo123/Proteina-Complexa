#!/bin/bash
#SBATCH --job-name=flowedit_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# FlowEdit (or the paired SDEdit control) candidate generation, one shard per array task.
#
# `set -euo pipefail` lives HERE, in the job body, not only in the preamble: if the source
# below fails, a preamble-only setting would leave the job running under the system python
# and still EXIT 0, releasing every afterok dependent behind a stage that did no work.
set -euo pipefail

PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/worktrees/flowedit}/scripts/_flowedit_preamble.sh"
[[ -f "${PREAMBLE}" ]] || { echo "FATAL: preamble not found: ${PREAMBLE}"; exit 1; }
# shellcheck disable=SC1090
source "${PREAMBLE}"
_fe_consume_args "$@"
_fe_defaults
_fe_banner "gen:${METHOD}"

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
echo "shard ${SHARD_INDEX}/${SHARD_COUNT}"

_fe_preflight || { echo "FATAL: preflight failed"; exit 1; }

# Pin the moving last-EMA to a stable path, ONCE per run tag, so every shard in this sweep
# scores the same model. Shards race here, so the pin is written to a temp then moved
# (atomic on this filesystem) and an existing pin is reused rather than replaced.
PINNED="${OUT_DIR}/flow_ckpt_pinned.ckpt"
if [[ "${FE_PIN_CKPT:-1}" == "1" ]]; then
  mkdir -p "${OUT_DIR}"
  _fe_pin_flow_ckpt "${FLOW_CKPT}" "${PINNED}" || { echo "FATAL: could not pin flow ckpt"; exit 1; }
  FLOW_CKPT="${PINNED}"
fi

mkdir -p "${CAND_DIR}" "${KEEP_DIR}" slurm_logs

# One output file per shard. Concurrent O_APPEND to a shared file NUL-corrupts rows on this
# filesystem, and the aggregator REFUSES corrupt lines rather than skipping them, so a
# shared path would turn into silent data loss at aggregation time.
OUT_JSONL="${CAND_DIR}/cand_${METHOD}_shard${SHARD_INDEX}.jsonl"

# shellcheck disable=SC2086
"${PY}" scripts/flowedit_cyclize.py \
  --method "${METHOD}" \
  --flow-ckpt "${FLOW_CKPT}" \
  --ae-ckpt "${AE_CKPT}" \
  --metadata "${METADATA}" \
  --config-name "${CONFIG_NAME}" \
  --sampling-config "${SAMPLING_CONFIG}" \
  --out "${OUT_JSONL}" \
  --cyc-types ${CYC_TYPES} \
  --t-ca ${T_CA} \
  --t-lat ${T_LAT} \
  --edit-end-frac ${EDIT_END_FRAC} \
  --n-avg ${N_AVG} \
  --nsteps "${NSTEPS}" \
  --seeds ${SEEDS} \
  --batch-seeds "${BATCH_SEEDS}" \
  --guidance-w "${GUIDANCE_W}" \
  --protected "${PROTECTED}" \
  --pdb-dir "${KEEP_DIR}" \
  --write-if "${WRITE_IF}" \
  --limit "${LIMIT}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count "${SHARD_COUNT}" \
  ${EXTRA_ARGS}

echo "DONE shard ${SHARD_INDEX} -> ${OUT_JSONL}"
