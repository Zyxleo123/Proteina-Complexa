#!/bin/bash
# Submitter for the SDEdit slider sprite sheets. Calls sbatch and nothing else.
#
#   bash scripts/submit_sdedit_slider.sh [--env-file F] [--partition P] [--time T]
#
# Defaults to the newest env_files/sdedit_traj_*.env, i.e. the most recent trajectory run.
# CPU only -- asking for a GPU to draw figures just queues behind real work.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"; TIME="00:30:00"; ENV_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)  ENV_FILE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$ENV_FILE" ]]; then
  ENV_FILE="$(ls -t "${REPO}"/env_files/sdedit_traj_*.env 2>/dev/null | head -1 || true)"
fi
[[ -n "$ENV_FILE" && -f "$ENV_FILE" ]] || { echo "FATAL: no env file found; pass --env-file"; exit 1; }
echo "env file: $ENV_FILE"

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

JOB=$(sbatch --parsable \
  --partition="$PARTITION" --time="$TIME" --job-name="sdedit_slider" \
  scripts/sdedit_slider.sbatch "$ENV_FILE")
echo "submitted CPU slider render: job ${JOB}"
# shellcheck disable=SC1090
echo "output: $( source "$ENV_FILE"; echo "${TRAJ_DIR%/frames}/slider" )"
