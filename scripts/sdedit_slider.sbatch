#!/bin/bash
#SBATCH --job-name=sdedit_slider
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU arm: trajectories -> scrubable sprite sheets (scripts/render_sdedit_slider.py).
# Reads only the .npz frames the GPU stage already wrote, so this is re-runnable as often
# as the styling changes, with no re-sampling and no GPU.
#
# Do NOT sbatch directly -- use scripts/submit_sdedit_slider.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${TRAJ_DIR:?TRAJ_DIR not set}"
SLIDER_DIR="${SLIDER_DIR:-${TRAJ_DIR%/frames}/slider}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)"

if ! compgen -G "${TRAJ_DIR}/*.npz" > /dev/null; then
  echo "GATE: no .npz trajectories in ${TRAJ_DIR} -- the GPU stage produced no frames."
  echo "      Nothing to render. Exiting 0 so this is not mistaken for a crash."
  exit 0
fi

mkdir -p "$SLIDER_DIR"
srun "$PY" scripts/render_sdedit_slider.py \
  --traj-dir "$TRAJ_DIR" \
  --out-dir "$SLIDER_DIR" \
  --states ${SLIDER_STATES:-xt} \
  --dpi "${SLIDER_DPI:-70}"

echo "DONE sprites -> ${SLIDER_DIR}"
du -sh "$SLIDER_DIR"

# The viewer page: the same sprites, with the Rosetta dG of the input and of the edited
# complex on either side of each trajectory. ROSETTA_DIR is optional -- without it the page
# still renders and says which cases have no paired energy, rather than omitting the panels.
shopt -s nullglob
ROS=()
for d in ${ROSETTA_DIRS:-${ROSETTA_DIR:-}}; do ROS+=("$d"/rosetta_shard*.jsonl); done
shopt -u nullglob
echo "pairing energies from ${#ROS[@]} rosetta shard file(s)"
"$PY" scripts/build_slider_page.py \
  --slider-index "${SLIDER_DIR}/slider_index.json" \
  ${ROS[0]:+--rosetta "${ROS[@]}"} \
  --out "${SLIDER_DIR}/cyclization_step_viewer.html" \
  --run-label "$(basename "$(dirname "$SLIDER_DIR")")" \
  --checkpoint "$(basename "$(dirname "${FLOW_CKPT_PATH:-unknown}")")"
