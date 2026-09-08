#!/bin/bash
#SBATCH --job-name=lp2cp_testeval_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 1 (GPU): run val_generation single-pass on a stratified sample of the CPSea TEST set for
# one frozen flow checkpoint. Writes closure_rows_*.jsonl and dumps every sampled complex under
# <out>/rosetta_dump/ for the CPU Rosetta stage. Do NOT sbatch directly -- use
# scripts/submit_lp2cp_testeval.sh, which writes the env file this sources as $1.
#
# Config travels in a FILE passed positionally (never `sbatch --export`, which triggers the
# user-env-retrieval requeue-hold).

set -euo pipefail          # in the body: a failed source must not run under system python + exit 0
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }

# env.sh FIRST, per-submission env file SECOND so the job's own namespaced EVAL_* config wins over
# whatever .env exports (see scripts/eval_val_gen_nsteps.sbatch for the collision this avoids).
source env.sh
source "$ENV_FILE"

: "${EVAL_RUN_DIR:?EVAL_RUN_DIR not set in $ENV_FILE}"
: "${EVAL_TEST_PARQUET:?EVAL_TEST_PARQUET not set in $ENV_FILE}"
: "${EVAL_OUT_DIR:?EVAL_OUT_DIR not set in $ENV_FILE}"

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable from $PY"; exit 1; }

CKPT="${EVAL_CKPT_PATH:-$EVAL_RUN_DIR/checkpoints/last-EMA.ckpt}"
if [[ -d "$CKPT" ]]; then
  echo "FATAL: checkpoint path is a directory, not a file: $CKPT"; exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "GATE FAILED: no checkpoint at $CKPT. Nothing to score. Exiting 0 so dependents report the gap."
  exit 0
fi
[[ -f "$EVAL_TEST_PARQUET" ]] || { echo "FATAL: test parquet missing: $EVAL_TEST_PARQUET"; exit 1; }

echo "Scoring $CKPT"
echo "  test=$EVAL_TEST_PARQUET n_complexes=${EVAL_N_COMPLEXES:-1000} n_repeat=${EVAL_N_REPEAT:-4}"
echo "  nsteps=${EVAL_NSTEPS:-inherit(400)} seed=${EVAL_SEED:-0} out=$EVAL_OUT_DIR"

srun "$PY" script_utils/eval_val_gen_testset.py \
  --run-dir "$EVAL_RUN_DIR" \
  --ckpt "$CKPT" \
  --test-parquet "$EVAL_TEST_PARQUET" \
  --n-complexes "${EVAL_N_COMPLEXES:-1000}" \
  --n-repeat "${EVAL_N_REPEAT:-4}" \
  ${EVAL_NSTEPS:+--nsteps "$EVAL_NSTEPS"} \
  --seed "${EVAL_SEED:-0}" \
  --rosetta-dump-max "${EVAL_ROSETTA_DUMP_MAX:-128}" \
  --out-dir "$EVAL_OUT_DIR"
