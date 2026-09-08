#!/bin/bash
#SBATCH --job-name=lp_train_ae
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 2 (GPU): the SHARED sidechain VAE, trained on CPSea + PepBench + ProtFrag peptides.
#
# a6000 (48G) on `general`, matching every recent CPSea run. Not an a5000 and not a
# 2080 Ti: this is the same model and effective batch (32 x accum 4 = 128) that
# finetune_full_128 ran, and the point of the arm is that it differs from that run in
# data alone. A smaller card would force a batch change and confound the comparison --
# and a CUDA allocation failure kills the process with no Python traceback, which reads
# as a mysterious silent FAILED.
#
# --requeue + resume_from_last means the 2-day wall costs a requeue, not a restart.
#
# Do NOT sbatch directly -- use scripts/submit_lp_mixing.sh.
#
#   sbatch lp_train_ae.sbatch /path/to/env_file [VAR=value ...]

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_lp_mixing_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "${REPO}"
mkdir -p slurm_logs

# Non-interactive bash skips ~/.bashrc; load it so WANDB_API_KEY is available.
if [[ -z "${WANDB_API_KEY:-}" || ${#WANDB_API_KEY} -lt 40 ]] && [[ -f "${HOME}/.bashrc" ]]; then
  set +euo pipefail
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc" || true
  set -euo pipefail
fi

# shellcheck source=_lp_mixing_preamble.sh
source "${REPO}/scripts/_lp_mixing_preamble.sh"
# shellcheck source=/dev/null
source "${REPO}/env.sh"
_lp_consume_args "$@"
_lp_defaults
_lp_banner "ae"

if ! _lp_preflight_train; then
  echo "GATE FAILED: mixed metadata is not usable; not starting AE training." >&2
  echo "GATE FAILED: exiting 0 so dependents report a reason instead of DependencyNeverSatisfied." >&2
  exit 0
fi

CKPT_DIR="${PROTEINA_ZFS_PATH}/training_runs/store/${AE_RUN_NAME}/checkpoints"
if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
  echo "Found ${CKPT_DIR}/last.ckpt -> resuming (RESUME_FROM_LAST=${RESUME_FROM_LAST})"
else
  echo "Fresh run: no last.ckpt under ${CKPT_DIR} -> step 0 + new WandB run."
fi

# Everything the arm depends on lives in the config; only run identity, logging and the
# resume flag are set here, so the YAML stays the single source of truth for the recipe.
TRAIN_ARGV=(
  -m proteinfoundation.partial_autoencoder.train
  --config-name "${AE_CONFIG}"
  ++show_prog_bar=false
  ++run_name="${AE_RUN_NAME}"
  ++log.log_wandb=true
  ++log.wandb_project="${WANDB_PROJECT_AE}"
  ++log.wandb_entity="${WANDB_ENTITY:-zyxleo}"
  ++hardware.ncpus_per_task_train_="${SLURM_CPUS_PER_TASK:-8}"
  ++training.resume_from_last="${RESUME_FROM_LAST}"
)
if [[ "${SMOKE}" == "1" ]]; then
  TRAIN_ARGV+=(++opt.val_check_interval=50 ++opt.limit_train_batches=20 ++log.checkpoint=false)
fi
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${EXTRA_OVERRIDES}); TRAIN_ARGV+=("${extra[@]}")
fi

printf 'CMD: %s' "${REPO}/.venv/bin/python"; printf ' %q' "${TRAIN_ARGV[@]}"; printf '\n'
srun --unbuffered "${REPO}/.venv/bin/python" "${TRAIN_ARGV[@]}"
