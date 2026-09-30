#!/bin/bash
#SBATCH --job-name=cp2lp_smoke
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:40:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Preflight: do the selected checkpoints load together, and does one batch survive the
# round trip generator -> AE -> discriminator -> backward?
#
# a6000 (48G), same class as the pilot. Step 5 restores the CONFIGURED rollout load, so
# the Trainer step doubles as a memory preflight for the training job. MEASURED: four
# concurrent rollouts already fill a 24G a5000 (job 68250 OOMed there), so 24G is not enough.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "smoke"
_cp2lp_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }
_cp2lp_preflight_flow_ckpt || { echo "FATAL: flow init missing" >&2; exit 1; }

cd "$REPO"
"$PY" script_utils/cp2lp_smoke.py \
  --config "$CONFIG_NAME" \
  --device cuda \
  ${SMOKE_BATCH:+--batch-size "$SMOKE_BATCH"} \
  ${SMOKE_NSTEPS:+--nsteps "$SMOKE_NSTEPS"} \
  ${SMOKE_VALGEN:+--val-generation}
