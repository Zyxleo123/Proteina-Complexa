#!/bin/bash
#SBATCH --job-name=cp2lp_op_smoke
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
# Preflight for the one-pass editor. One small batch plus a few Trainer steps.
#
# a6000, the same class as the arms, and the batch keeps the LARGEST receptor it drew:
# pair features are O(n_target^2) with a crop to 256 residues, so a memory check run on a
# small receptor measures the wrong thing. The one-pass envelope is far below the rollout
# arms' (one forward instead of 24, no retained trajectory) but that is a claim this job
# is supposed to verify rather than assume.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-onepass_gan}"
_cp2lp_defaults
_cp2lp_banner "onepass-smoke"
_cp2lp_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

# THE SMOKE RUNS BEFORE THE FORK EXISTS -- it is what decides whether producing the fork is
# worth the GPU hours, so it cannot require it. Every check in it is a wiring check
# (trunk-call count, residual algebra, frame sensitivity, one G/D update), and none of them
# depends on WHICH checkpoint loaded, as long as the AE/flow pairing is valid. So fall back
# to the lpmix init and say so loudly rather than gating.
SMOKE_INIT="${INIT_CKPT}"
if [[ ! -f "${SMOKE_INIT}" ]]; then
  echo "[smoke] the common fork is not built yet (${SMOKE_INIT})"
  echo "[smoke] falling back to the lpmix init: ${FLOW_CKPT}"
  SMOKE_INIT="${FLOW_CKPT}"
fi
[[ -f "${SMOKE_INIT}" ]] || { echo "FATAL: no usable init checkpoint (${SMOKE_INIT})" >&2; exit 1; }
echo "[smoke] init: ${SMOKE_INIT}"
echo "[smoke] AE  : ${AE_CKPT}"

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set -x
"$PY" script_utils/cp2lp_onepass_smoke.py \
  --config "$CONFIG_NAME" \
  --device cuda \
  --init-ckpt "$SMOKE_INIT" \
  --max-cp "${SMOKE_MAX_CP:-2}"
