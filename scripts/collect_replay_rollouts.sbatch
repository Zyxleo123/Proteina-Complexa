#!/bin/bash
#SBATCH --job-name=cpsea_replay_collect
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Job body for reward-weighted-replay rollout collection (Stage 4). Do not
# sbatch this directly -- use scripts/submit_replay_collect.sh, which writes
# the env file this sources as $1.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"

: "${CKPT:?CKPT not set in $ENV_FILE}"
: "${OUT_DIR:?OUT_DIR not set in $ENV_FILE}"
: "${CONFIG_NAME:?CONFIG_NAME not set in $ENV_FILE}"
: "${K:?K not set in $ENV_FILE}"
: "${N_BATCHES:?N_BATCHES not set in $ENV_FILE}"
: "${NSTEPS:?NSTEPS not set in $ENV_FILE}"
: "${GATE_FLAGS:?GATE_FLAGS not set in $ENV_FILE}"
GEN_CHUNK="${GEN_CHUNK:-2}"

source env.sh
mkdir -p "$OUT_DIR"

# Prefer PYTHON_EXEC from env.sh (repo .venv). Bare `python` on SLURM often
# resolves to conda base and lacks proteinfoundation.
PYTHON_BIN="${PYTHON_EXEC:-${SLURM_SUBMIT_DIR:-$(pwd)}/.venv/bin/python}"

echo "Collecting replay rollouts: ckpt=$CKPT out=$OUT_DIR k=$K n_batches=$N_BATCHES nsteps=$NSTEPS gen_chunk=$GEN_CHUNK gates=[$GATE_FLAGS] python=$PYTHON_BIN"

# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/collect_cpsea_replay_rollouts.py \
  --ckpt "$CKPT" \
  --config "$CONFIG_NAME" \
  --out "$OUT_DIR" \
  --k "$K" \
  --n-batches "$N_BATCHES" \
  --nsteps "$NSTEPS" \
  --seed "${SEED:-0}" \
  --max-size "${MAX_SIZE:-10000}" \
  --gen-chunk "$GEN_CHUNK" \
  $GATE_FLAGS
