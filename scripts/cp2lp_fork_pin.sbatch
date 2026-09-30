#!/bin/bash
#SBATCH --job-name=cp2lp_fork_pin
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Snapshot the warmup's checkpoint into the COMMON FORK the five arms start from.
#
# A job rather than a shell step in the submitter, so it can sit on `--dependency=afterok`
# behind the warmup and the whole chain can be launched in one go. It only copies a file --
# CPU partition; asking for a GPU to copy a checkpoint just queues behind real work.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-onepass_warmup}"
_cp2lp_defaults
_cp2lp_banner "fork-pin"

cd "$REPO"
SRC="${STORE}/${FORK_RUN}/checkpoints/last.ckpt"
if [[ ! -f "$SRC" ]]; then
  # Gate, not failure. A non-zero exit here strands all five arms in
  # DependencyNeverSatisfied with nothing explaining why.
  echo "GATE: the warmup produced no checkpoint at ${SRC}."
  echo "      Read the warmup job's .out before resubmitting: a job that fails to source"
  echo "      its preamble exits 0 and releases its dependents having done no work."
  exit 0
fi
FORCE="${FORCE_PIN:-0}" PY="$PY" bash scripts/pin_cp2lp_fork.sh "$SRC"
