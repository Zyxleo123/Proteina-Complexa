#!/bin/bash
#SBATCH --job-name=grid_slider_build
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU arm: trajectories -> per-grid-point sprite sheets -> the (t_ca, t_lat, pass)-selectable
# viewer page. Reads only the .npz frames the GPU stage wrote and the targeted Rosetta rows,
# so it re-renders as often as the styling changes with no re-sampling and no GPU.
#
# Do NOT sbatch directly -- use scripts/submit_sdedit_grid_slider.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${TRAJ_DIR:?TRAJ_DIR not set}"
: "${SLIDER_DIR:?SLIDER_DIR not set}"
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

echo "sprites rendered:"; du -sh "$SLIDER_DIR"

# Rosetta rows are optional: without them the page still renders and marks every combination
# "not scored" rather than omitting the energy panels.
shopt -s nullglob
ROS=( "${ROSETTA_OUT_DIR:-}"/rosetta_shard*.jsonl )
shopt -u nullglob
echo "pairing energies from ${#ROS[@]} rosetta shard file(s)"

"$PY" scripts/build_grid_slider_page.py \
  --slider-index "${SLIDER_DIR}/slider_index.json" \
  ${ROS[0]:+--rosetta "${ROS[@]}"} \
  --out "${SLIDER_DIR}/cyclization_grid_viewer.html" \
  --run-label "$(basename "$(dirname "$SLIDER_DIR")")" \
  --checkpoint "$(basename "$(dirname "${FLOW_CKPT_PATH:-unknown}")")"

echo "DONE page -> ${SLIDER_DIR}/cyclization_grid_viewer.html"
