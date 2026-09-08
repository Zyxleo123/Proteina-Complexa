#!/bin/bash
#SBATCH --job-name=sdedit_render
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU arm: turn saved trajectories into GIFs (scripts/render_sdedit_gif.py). Reads only
# the .npz files, so styling can be re-run cheaply without re-sampling. No GPU: asking for
# one to draw a figure just queues behind real work.
#
# Do NOT sbatch directly -- use scripts/submit_sdedit_traj.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${TRAJ_DIR:?TRAJ_DIR not set}"
: "${GIF_DIR:?GIF_DIR not set}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)"

# Gate INSIDE the job, exiting 0: a non-zero exit here would strand nothing today, but the
# distinction between "upstream produced no frames" and "upstream never ran" is the thing
# worth printing.
if ! compgen -G "${TRAJ_DIR}/*.npz" > /dev/null; then
  echo "GATE: no .npz trajectories in ${TRAJ_DIR} -- the GPU stage produced no frames."
  echo "      Nothing to render. Exiting 0 so this is not mistaken for a crash."
  exit 0
fi

mkdir -p "$GIF_DIR"
srun "$PY" scripts/render_sdedit_gif.py \
  --traj-dir "$TRAJ_DIR" \
  --out-dir "$GIF_DIR" \
  --state "${RENDER_STATE:-x1}" \
  --max-frames "${MAX_FRAMES:-48}" \
  --fps "${FPS:-10}"

echo "DONE gifs -> ${GIF_DIR}"
ls -lh "${GIF_DIR}"/*.gif
