#!/bin/bash
# Submitter for the SDEdit trajectory render. Calls sbatch and nothing else.
#
#   bash scripts/submit_sdedit_traj.sh [--partition P] [--nodelist N] [--time T] [--dry-run]
#
# Writes ONE timestamped env file and passes its PATH as a positional argument (never
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets jobs requeued and HELD).
#
# Stage 1 (GPU): re-run the named edits, dump every integration frame.
# Stage 2 (CPU): render those frames to GIFs, chained afterok.
#
# The cases below are drawn from evaluation_results/sdedit_20260902_143308 at the
# high-fidelity corner (t_ca=0.8, t_lat=0.6, seed 0) -- two mainchain successes, two
# mainchain failures, and the disulfide/isopeptide failures on LNR_1ky6_A_P, the same
# peptide mainchain succeeds on, which makes chemistry the only variable in that contrast.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# `general` minus gpu28 is exactly the set of RTX A6000 nodes: the sweep ran on an A6000
# and a different GPU class can perturb floating point enough to flip a marginal closure.
PARTITION="general"; EXCLUDE="gpu28"; NODELIST=""
TIME="02:00:00"; RENDER_TIME="00:30:00"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --exclude)   EXCLUDE="$2";   shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    --dry-run)   DRY=1;          shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="sdedit_traj_${STAMP}"

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<INNER
# Auto-written by scripts/submit_sdedit_traj.sh at ${STAMP}.
TRAJ_DIR=${REPO}/evaluation_results/${RUN_ID}/frames
GIF_DIR=${REPO}/evaluation_results/${RUN_ID}/gifs
LNR_METADATA=${REPO}/CPSea_data/lnr_staged/metadata/lnr_test.parquet
# Same pin the sweep used -- flow and AE must match or the trajectory is not the one scored.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
# 0 = the design sampler's own nsteps (400), which is what the sweep ran.
NSTEPS=0
RENDER_STATE=x1
MAX_FRAMES=48
FPS=10
CASES="LNR_4w50_B_F:mainchain:0.8:0.6:0 LNR_1ky6_A_P:mainchain:0.8:0.6:0 LNR_1jrr_A_P:mainchain:0.8:0.6:0 LNR_3c3o_A_B:mainchain:0.8:0.6:0 LNR_1ky6_A_P:disulfide:0.8:0.6:0 LNR_1ky6_A_P:isopeptide:0.8:0.6:0"
INNER
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

if [[ $DRY == 1 ]]; then
  echo "--dry-run: not submitting. Resolve-only check:"
  # shellcheck disable=SC1090
  ( source env.sh; source "$ENV_FILE"
    # shellcheck disable=SC2086
    "${PYTHON_EXEC:-$(pwd)/.venv/bin/python}" scripts/sdedit_trajectory.py \
      --ckpt-path "$FLOW_CKPT_PATH" --ckpt-name "$FLOW_CKPT_NAME" \
      --metadata "$LNR_METADATA" --out-dir "$TRAJ_DIR" --cases ${CASES} --dry-run )
  exit 0
fi

# Scrub SLURM_* so submitting from inside an allocation does not leak the parent job's env.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

PIN=(); [[ -n "$NODELIST" ]] && PIN=(--nodelist="$NODELIST")
[[ -z "$NODELIST" && -n "$EXCLUDE" ]] && PIN=(--exclude="$EXCLUDE")

GPU_ID=$(sbatch --parsable \
  --partition="$PARTITION" "${PIN[@]}" --time="$TIME" --job-name="${RUN_ID}" \
  scripts/sdedit_traj.sbatch "$ENV_FILE")
echo "submitted GPU trajectories: job ${GPU_ID} (${PARTITION}${NODELIST:+/$NODELIST}${EXCLUDE:+ excl $EXCLUDE})"

RENDER_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$RENDER_TIME" \
  --dependency=afterok:"$GPU_ID" --job-name="${RUN_ID}_gif" \
  scripts/sdedit_render.sbatch "$ENV_FILE")
echo "submitted CPU render:       job ${RENDER_ID} (afterok:${GPU_ID})"
echo
echo "frames: ${REPO}/evaluation_results/${RUN_ID}/frames"
echo "gifs:   ${REPO}/evaluation_results/${RUN_ID}/gifs"
