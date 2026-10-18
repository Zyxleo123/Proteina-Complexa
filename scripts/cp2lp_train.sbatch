#!/bin/bash
#SBATCH --job-name=cp2lp_train
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
# One CP->LP arm. Run TWICE, as siblings so they queue concurrently:
#
#   ARM=gan          adversarial + contact losses
#   ARM=contactonly  the ablation: same everything, adversarial term off
#
# a6000 (48G), NOT a5000. This step holds more than the lpmix arm it is finetuned from:
# the flow forward on batch 16, PLUS a differentiable rollout over max_cp_per_step *
# samples_per_cp trajectories whose last grad_steps forwards keep activations, PLUS the AE
# decoder on both the generated and the source structures, PLUS the discriminator. The
# lpmix arm already needs 48G without any of that. A 24G card dies here with a CUDA
# allocation failure and NO Python traceback, which reads as a mysterious silent FAILED --
# so the GPU class is pinned rather than left to whatever the scheduler picks.
#
# Resumable by design: `training.resume_from_last=true` plus --requeue/--open-mode=append
# means hitting the time limit is a resubmit, not a restart.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "train"
_cp2lp_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }
_cp2lp_preflight_flow_ckpt || { echo "FATAL: flow init missing" >&2; exit 1; }

cd "$REPO"
mkdir -p slurm_logs

# The rollout allocates and frees large activation blocks every step, and job 68347 died
# with 1.62 GiB "reserved but unallocated" -- i.e. lost to fragmentation, not actually in
# use. Expandable segments let the allocator grow a block instead of needing a contiguous
# one, which is the specific remedy the OOM message itself suggests.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
  # A smoke run gets its OWN run_name: three concurrent jobs writing one run_name collide
  # on the checkpoint directory and the last writer wins.
  EXTRA+=("++run_name=${RUN_NAME}_smoke" "++opt.limit_train_batches=20" "++val_generation.enabled=false")
  # FORCE THE GRADIENT PROBE TO FIRE. It is scheduled on `g_updates % N == 0` with N=200, and
  # the fork checkpoint arrives carrying g_updates=250, so at 20 batches a smoke run reaches
  # neither 400 nor any other multiple -- the probe code would never execute and the smoke
  # would "pass" without touching the thing being smoked. This is exactly the path that logs
  # the anchor and adversarial terms, which are stashed during the step and differentiated
  # against a live graph, i.e. the kind of defect that otherwise surfaces hours into a run.
  EXTRA+=("++cp2lp.backbone.grad_probe_every_n_steps=1")
  # AND DROP THE ADVERSARIAL WARMUP, for the same reason. The generator's adversarial term is
  # gated on `g_updates >= cp2lp.adversarial.warmup_g_steps` (300), the fork arrives carrying
  # g_updates=250, and `accumulate_grad_batches=8` means 20 batches buy only ~2 g-updates per
  # epoch -- so a smoke crawls from 250 and barely reaches 300, if at all. MEASURED on job
  # 74076: `gradnorm/anchor_step` logged 520 times while `loss_adv_g` and
  # `gradnorm/adversarial` were logged ZERO times, i.e. the smoke validated half the probe and
  # silently skipped the adversarial half. Setting the warmup to 0 makes the branch run from
  # the first g-update, which is the only way a smoke covers it.
  EXTRA+=("++cp2lp.adversarial.warmup_g_steps=0")
else
  EXTRA+=("++run_name=${RUN_NAME}")
fi

# Hydra overrides appended verbatim, LAST so they win. Space-separated; set it from the
# submitter with --set. Deliberately not used to carry configuration a job needs to be
# correct -- that belongs in the env file and in a config -- only one-off probes like
# forcing a cadence or shortening a budget.
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206  (word splitting is the point: these are separate argv entries)
  EXTRA+=(${EXTRA_TRAIN_ARGS})
  echo "EXTRA_TRAIN_ARGS: ${EXTRA_TRAIN_ARGS}"
fi

# Report this job's peak host RAM on the way out; see `_cp2lp_mem_peak`.
trap '_cp2lp_mem_peak' EXIT

set -x
"$PY" -m proteinfoundation.train \
  --config-name "$CONFIG_NAME" \
  ++show_prog_bar=false \
  ++log.log_wandb="${LOG_WANDB:-true}" \
  ++log.wandb_project="$WANDB_PROJECT" \
  ++log.wandb_entity="$WANDB_ENTITY" \
  ++hardware.ncpus_per_task_train_="${SLURM_CPUS_PER_TASK:-8}" \
  ++autoencoder_ckpt_path="$AE_CKPT" \
  ++pretrain_ckpt_path="$INIT_CKPT" \
  ++dataset.unified.datamodule.metadata_file="$TRAIN_META" \
  ++dataset.unified.datamodule.val_metadata_file="$EVAL_META" \
  ++training.resume_from_last=true \
  "${EXTRA[@]}"
