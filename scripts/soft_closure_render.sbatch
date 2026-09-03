#!/bin/bash
#SBATCH --job-name=soft_closure_render
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 4 (CPU): the before/after PNG and the GIF. Reads only saved arrays, so re-running it
# to restyle a figure costs nothing and needs no GPU.
#
# Do NOT sbatch directly -- use scripts/submit_soft_closure.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${PROJ_DIR:?PROJ_DIR not set}"
: "${FIG_DIR:?FIG_DIR not set}"

STAGE_DIR="${PROJ_DIR}/stages"
if [[ ! -d "$STAGE_DIR" ]] || ! compgen -G "${STAGE_DIR}/*.npz" > /dev/null; then
  echo "GATE: no projection stage traces at ${STAGE_DIR} -- nothing to draw."
  if [[ -f "${PROJ_DIR}/projection_summary.json" ]]; then
    echo "GATE: the projection's own verdict:"
    cat "${PROJ_DIR}/projection_summary.json"
  fi
  exit 0
fi

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
TRAJ_ARG=()
if [[ -n "${TRAJ_DIR:-}" ]] && compgen -G "${TRAJ_DIR}/*.npz" > /dev/null; then
  TRAJ_ARG=(--sdedit-traj-dir "$TRAJ_DIR")
  echo "SDEdit frames found -- figures will carry the third panel."
else
  echo "No SDEdit frames at ${TRAJ_DIR:-<unset>} -- drawing input vs projection only."
fi

mkdir -p "$FIG_DIR"
srun "$PY" -u scripts/render_soft_closure.py \
  --stage-dir "$STAGE_DIR" \
  --out-dir "$FIG_DIR" \
  --per-example "${RENDER_PER_EXAMPLE:-1}" \
  --max-sdedit-frames "${RENDER_MAX_SDEDIT_FRAMES:-40}" \
  "${TRAJ_ARG[@]}"

echo "DONE figures -> ${FIG_DIR}"
