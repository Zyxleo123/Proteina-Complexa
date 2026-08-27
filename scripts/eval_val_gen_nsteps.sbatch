#!/bin/bash
#SBATCH --job-name=cpsea_valgen_nsteps
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Scores one FROZEN training checkpoint's val_generation at several ODE step counts.
# Do not sbatch this directly -- use scripts/submit_val_gen_nsteps.sh, which writes the
# env file this sources as $1 and picks partition/nodelist/time from `myfree` at submit
# time. Config travels in a FILE passed positionally, never via `sbatch --export`, which
# triggers the user-env-retrieval requeue-hold.
#
# No training happens here. One GPU, weights frozen, EMA weights by default.

set -euo pipefail          # in the body, not only the preamble: a failed `source` must not
                           # leave this running under the system python and then exit 0
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }

# ORDER MATTERS, and it is the reverse of the obvious one. env.sh (via .env) exports a pile
# of repo-wide variables, and the job's own config must WIN over them -- so env.sh is sourced
# FIRST and the per-submission env file second. Sourcing the env file first let .env's
# `CKPT_PATH=${LOCAL_CHECKPOINT_PATH}` shadow this script's checkpoint default, so job 36051
# gated out with "no checkpoint at .../ckpts" (a DIRECTORY) having scored nothing. Every
# variable this script reads from $ENV_FILE is also namespaced (EVAL_*) so a future addition
# to .env cannot silently reintroduce the collision.
source env.sh
source "$ENV_FILE"

: "${RUN_DIR:?RUN_DIR not set in $ENV_FILE}"
: "${EVAL_OUT_DIR:?EVAL_OUT_DIR not set in $ENV_FILE}"

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable from $PY"; exit 1; }

# Gate, not a crash: a missing checkpoint means the upstream training run never produced
# one. Exit 0 so a dependent aggregation job reports "no rows" instead of being stranded in
# DependencyNeverSatisfied with nothing said about why.
#
# EVAL_CKPT_PATH, not CKPT_PATH: see the sourcing note above. The submitter always writes
# this key (empty when defaulting), so an unset value here means the env file is stale.
CKPT="${EVAL_CKPT_PATH:-$RUN_DIR/checkpoints/last-EMA.ckpt}"
if [[ -d "$CKPT" ]]; then
  # The exact shape of the bug that produced a silent no-op: a DIRECTORY passes any
  # existence test, then torch.load dies on it minutes into a queued job.
  echo "FATAL: checkpoint path is a directory, not a file: $CKPT"
  echo "       EVAL_CKPT_PATH must name a .ckpt file. Non-zero on purpose -- this is a"
  echo "       misconfiguration, not a missing upstream, so it must not look like a clean gate."
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "GATE FAILED: no checkpoint at $CKPT -- $RUN_DIR has not produced one."
  echo "Nothing to score. Exiting 0 so dependents can run and report the gap."
  exit 0
fi

echo "Scoring $CKPT"
echo "  nsteps=${NSTEPS:-100,400} seeds=${SEEDS:-0,1,2} n_batches=${N_BATCHES:-8}"
echo "  rows -> $EVAL_OUT_DIR/rows_<arm>_<jobid>.jsonl  (exclusive to this job)"

# Resumable: rows already in $EVAL_OUT_DIR are skipped, so hitting the wall limit means
# resubmitting this exact job, not restarting the sweep.
srun "$PY" script_utils/eval_val_gen_nsteps.py \
  --run-dir "$RUN_DIR" \
  --ckpt "$CKPT" \
  ${EVAL_LABEL:+--label "$EVAL_LABEL"} \
  --nsteps "${NSTEPS:-100,400}" \
  --seeds "${SEEDS:-0,1,2}" \
  --n-batches "${N_BATCHES:-8}" \
  --out-dir "$EVAL_OUT_DIR"
