#!/bin/bash
#SBATCH --job-name=cp2lp_generate
#SBATCH --partition=general
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Write (generated LP, source CP, target) triplets from one trained arm, using the FULL
# deployed sampler -- not the short differentiable trajectory the GAN updates use.
#
# a5000 is enough here: inference only, no rollout graph, no discriminator, batch 4.
#
# Resumable: each shard drops a .done marker and is skipped on resubmission, and each
# writes its OWN manifest. Concurrent O_APPEND to a shared file NUL-corrupts rows on this
# filesystem, and the report REFUSES a corrupt row rather than silently averaging past it.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "generate"
_cp2lp_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

RUN_DIR="${STORE}/${RUN_NAME}"
if [[ ! -d "${RUN_DIR}/checkpoints" ]]; then
  # Upstream gate, not an error: exit 0 so dependents run and say so, rather than being
  # stranded in DependencyNeverSatisfied with nothing explaining why.
  echo "GATE: no checkpoints under ${RUN_DIR} -- the ${ARM} training arm has not produced one yet."
  exit 0
fi

cd "$REPO"
OUT="${TRIPLET_ROOT}/${ARM}"
mkdir -p "$OUT"

set -x
"$PY" script_utils/cp2lp_generate.py \
  --run-dir "$RUN_DIR" \
  --cp-parquet "$EVAL_META" \
  --out-dir "$OUT" \
  --n-cp "$GEN_N_CP" \
  --samples-per-cp "$GEN_SAMPLES_PER_CP" \
  --seeds "$GEN_SEEDS" \
  --nsteps "$GEN_NSTEPS" \
  --batch-size "$GEN_BATCH" \
  --device cuda
