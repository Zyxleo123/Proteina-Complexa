#!/bin/bash
#SBATCH --job-name=lp_train_flow
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
# Stage 3 (GPU): one flow arm. Run TWICE, as siblings so they queue concurrently:
#
#   ARM=mix       CPSea + 25% linear peptide data, LINEAR topology token
#   ARM=control   CPSea only -- same AE, same metadata files, same val filter
#
# The control is not optional and it is not v4. Both arms load the SHARED AE, which is a
# different AE from the one every previous CPSea run used, so nothing here is comparable
# to v4 -- only to each other. Running only the mix arm would leave "is it the linear
# data or the new AE?" unanswerable.
#
# a6000 (48G) on `general`: this is the v4 recipe including the unrolled bond loss, whose
# gradient-carrying unroll step builds a second network graph beside the flow loss's own.
# It OOMs a smaller card, and a CUDA allocation failure exits with no Python traceback.
#
# Do NOT sbatch directly -- use scripts/submit_lp_mixing.sh.
#
#   sbatch lp_train_flow.sbatch /path/to/env_file ARM=mix

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_lp_mixing_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "${REPO}"
mkdir -p slurm_logs

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

ARM="${ARM:?FATAL: pass ARM=mix or ARM=control}"
case "${ARM}" in
  mix)     CONFIG="${MIX_CONFIG}";  RUN_NAME="${MIX_RUN_NAME}" ;;
  control) CONFIG="${CTRL_CONFIG}"; RUN_NAME="${CTRL_RUN_NAME}" ;;
  *) echo "FATAL: unknown ARM='${ARM}' (expected mix|control)" >&2; exit 1 ;;
esac
_lp_banner "flow:${ARM}"
echo "config   : ${CONFIG}"
echo "run_name : ${RUN_NAME}"

if ! _lp_preflight_train; then
  echo "GATE FAILED: mixed metadata is not usable; not starting the ${ARM} arm." >&2
  echo "GATE FAILED: exiting 0 so dependents report a reason instead of DependencyNeverSatisfied." >&2
  exit 0
fi

# Both arms must load the SAME AE snapshot. Pinning is idempotent: an existing snapshot
# is reused, so whichever arm starts second inherits exactly what the first one loaded,
# even if the AE run has written newer checkpoints in between.
if ! _lp_pin_shared_ae; then
  echo "GATE FAILED: no shared AE checkpoint to pin; not starting the ${ARM} arm." >&2
  echo "GATE FAILED: exiting 0 so dependents are not stranded." >&2
  exit 0
fi

CKPT_DIR="${PROTEINA_ZFS_PATH}/training_runs/store/${RUN_NAME}/checkpoints"
if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
  echo "Found ${CKPT_DIR}/last.ckpt -> resuming (RESUME_FROM_LAST=${RESUME_FROM_LAST})"
else
  echo "Fresh run: no last.ckpt under ${CKPT_DIR} -> step 0 + new WandB run."
fi

# The mix ratio, the metadata paths and the arm's filters all live in the config, so the
# two arms differ by their --config-name and nothing else on this command line.
TRAIN_ARGV=(
  -m proteinfoundation.train
  --config-name "${CONFIG}"
  ++show_prog_bar=false
  ++run_name="${RUN_NAME}"
  ++log.log_wandb=true
  ++log.wandb_project="${WANDB_PROJECT_FLOW}"
  ++log.wandb_entity="${WANDB_ENTITY:-zyxleo}"
  ++hardware.ncpus_per_task_train_="${SLURM_CPUS_PER_TASK:-8}"
  ++autoencoder_ckpt_path="${SHARED_AE_CKPT}"
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
