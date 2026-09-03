#!/bin/bash
#SBATCH --job-name=soft_closure_traj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 3 (GPU): dump the SDEdit INTEGRATION TRAJECTORY for the projected peptides, so the
# GIF can show the pull and the generative edit as one continuous animation.
#
# The case list cannot be written by the submitter: which replicas exist is decided by the
# projection's acceptance gates, hours later. It is derived here, from the parquet.
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
: "${TRAJ_DIR:?TRAJ_DIR not set}"
: "${FLOW_CKPT_PATH:?FLOW_CKPT_PATH not set}"

PROJECTED_METADATA="${PROJ_DIR}/metadata/projected.parquet"
if [[ ! -f "$PROJECTED_METADATA" ]]; then
  echo "GATE: no accepted projections at ${PROJ_DIR} -- no trajectories to dump. Exiting 0."
  exit 0
fi

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
FLOW_CKPT_NAME="${FLOW_CKPT_NAME:-last-EMA.ckpt}"

# One case per input peptide: the replica whose pull got closest, at the grid point the
# original sweep found best for mainchain (t_ca 0.8, t_lat 0.4 -- see the sweep summary).
CASES=$("$PY" - "$PROJECTED_METADATA" "${CYC_TYPE:-mainchain}" "${TRAJ_T_CA:-0.8}" \
                "${TRAJ_T_LAT:-0.4}" "${TRAJ_SEED:-0}" <<'PYEOF'
import sys
import pandas as pd
path, chem, t_ca, t_lat, seed = sys.argv[1:6]
df = pd.read_parquet(path)
best = df.sort_values("nc_gap_angstrom").groupby("parent_example_id", as_index=False).first()
print(" ".join(f"{r.example_id}:{chem}:{t_ca}:{t_lat}:{seed}" for r in best.itertuples()))
PYEOF
)
[[ -n "$CASES" ]] || { echo "GATE: parquet held no rows. Exiting 0."; exit 0; }
echo "Node: $(hostname)"
echo "cases: $CASES"

mkdir -p "$TRAJ_DIR"
# shellcheck disable=SC2086
srun "$PY" scripts/sdedit_trajectory.py \
  --ckpt-path "$FLOW_CKPT_PATH" \
  --ckpt-name "$FLOW_CKPT_NAME" \
  --metadata "$PROJECTED_METADATA" \
  --out-dir "$TRAJ_DIR" \
  --nsteps "${NSTEPS:-0}" \
  --cases $CASES

echo "DONE trajectories -> ${TRAJ_DIR}"
