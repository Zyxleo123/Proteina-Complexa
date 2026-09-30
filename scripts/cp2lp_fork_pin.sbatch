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
# PASS NO SOURCE ARGUMENT. `pin_cp2lp_fork.sh` prefers the highest-numbered
# `g??????.ckpt` milestone -- the run's exact `max_g_updates` endpoint -- and falls back to
# `last.ckpt`, which is written on a *global_step* cadence and so lands a few updates short.
#
# MEASURED THE HARD WAY (pin job 69416): this line used to hardcode `last.ckpt` and pass it
# as $1, which the script treats as an explicit override. The milestone-preference fix
# therefore did nothing, and the fork was pinned at g=250 instead of g=300. Harmless to that
# comparison -- all five arms inherited the same 250 -- but it is the whole class of bug
# where the helper is fixed and its CALLER is not. Guarded by
# `test_the_fork_pin_job_does_not_override_the_milestone_preference`.
CKPT_DIR="${STORE}/${FORK_RUN}/checkpoints"
if ! ls -1 "${CKPT_DIR}"/g[0-9][0-9][0-9][0-9][0-9][0-9].ckpt "${CKPT_DIR}"/last.ckpt >/dev/null 2>&1; then
  # Gate, not failure. A non-zero exit here strands all five arms in
  # DependencyNeverSatisfied with nothing explaining why.
  echo "GATE: the warmup produced no usable checkpoint under ${CKPT_DIR}."
  echo "      Read the warmup job's .out before resubmitting: a job that fails to source"
  echo "      its preamble exits 0 and releases its dependents having done no work."
  ls -1 "${CKPT_DIR}" 2>/dev/null | sed 's/^/      have: /' || true
  exit 0
fi
FORCE="${FORCE_PIN:-0}" PY="$PY" bash scripts/pin_cp2lp_fork.sh
