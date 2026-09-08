#!/bin/bash
# Submitter for the (t_ca, t_lat, pass)-selectable SDEdit step viewer. Calls sbatch only.
#
#   bash scripts/submit_sdedit_grid_slider.sh [--partition P] [--exclude N] [--nodelist N]
#                                             [--peptides "id1 id2"] [--chems "..."]
#                                             [--t-ca "..."] [--t-lat "..."] [--seeds "..."]
#                                             [--sweep-dir DIR] [--dry-run]
#
# Three stages, chained so the page waits on both the frames and the energies:
#   1  GPU   re-run each (peptide, chem, t_ca, t_lat, seed) and dump the ODE trajectory
#            (scripts/sdedit_traj.sbatch -> sdedit_trajectory.py)
#   2  CPU   targeted Rosetta dG for those peptides' complexes, EVERY grid point
#            (scripts/grid_slider_rosetta.sbatch)   -- sibling of stage 1
#   3  CPU   sprites + the selectable viewer page   (scripts/grid_slider_build.sbatch)
#            afterok stage 1 AND stage 2
#
# Writes ONE timestamped env file and passes its PATH as a positional argument (never
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets jobs requeued and HELD).

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# `general` minus gpu28 is exactly the A6000 nodes: the sweep ran on an A6000 and a different
# GPU class perturbs floating point enough to flip a marginal closure.
PARTITION="general"; EXCLUDE="gpu28"; NODELIST=""
GPU_TIME="03:00:00"; ROS_TIME="05:00:00"; BUILD_TIME="00:40:00"; DRY=0

# One peptide across all three chemistries makes chemistry the only variable, and each case
# already spans success -> failure across its own noise grid. Override with --peptides to add
# more. The sweep this visualizes: pocket-cropped LNR, the frozen bond-unroll pin.
PEPTIDES="LNR_3rc4_A_B"
CHEMS="mainchain disulfide isopeptide"
TCA="0.2 0.4 0.6 0.8"; TLAT="0.2 0.4 0.6 0.8"; SEEDS="0 1 2"
SWEEP_DIR="${REPO}/evaluation_results/sdedit_passk_pocket_20260907_223120"
LNR_METADATA="${REPO}/CPSea_data/lnr_pocket/metadata/lnr_test_pocket.parquet"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --exclude)   EXCLUDE="$2";   shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --peptides)  PEPTIDES="$2";  shift 2 ;;
    --chems)     CHEMS="$2";     shift 2 ;;
    --t-ca)      TCA="$2";       shift 2 ;;
    --t-lat)     TLAT="$2";      shift 2 ;;
    --seeds)     SEEDS="$2";     shift 2 ;;
    --sweep-dir) SWEEP_DIR="$2"; shift 2 ;;
    --metadata)  LNR_METADATA="$2"; shift 2 ;;
    --dry-run)   DRY=1;          shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -f "$LNR_METADATA" ]] || { echo "FATAL: metadata not found: $LNR_METADATA" >&2; exit 1; }
[[ -d "$SWEEP_DIR/complexes" ]] || { echo "FATAL: sweep complexes not found: $SWEEP_DIR/complexes" >&2; exit 1; }

# Build the CASES list = peptide x chem x t_ca x t_lat x seed (config, not work).
CASES=""
for pep in $PEPTIDES; do for chem in $CHEMS; do
  for tca in $TCA; do for tlat in $TLAT; do for sd in $SEEDS; do
    CASES+="${pep}:${chem}:${tca}:${tlat}:${sd} "
  done; done; done
done; done
CASES="${CASES% }"
N_CASES=$(wc -w <<< "$CASES")

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="sdedit_gridviz_${STAMP}"
mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<ENVEOF
# Auto-written by scripts/submit_sdedit_grid_slider.sh at ${STAMP}.
TRAJ_DIR=${REPO}/evaluation_results/${RUN_ID}/frames
SLIDER_DIR=${REPO}/evaluation_results/${RUN_ID}/slider
ROSETTA_OUT_DIR=${REPO}/evaluation_results/${RUN_ID}/rosetta
SWEEP_DIR=${SWEEP_DIR}
VIZ_EXAMPLES=${PEPTIDES}
ROSETTA_SHARDS=1
LNR_METADATA=${LNR_METADATA}
# Same frozen pin the sweep used -- flow and AE must match or the trajectory is not the one
# the sweep scored, and a marginal closure could flip.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
# 0 = the design sampler's own nsteps (400), exactly what the sweep ran.
NSTEPS=0
SLIDER_STATES=xt
SLIDER_DPI=70
CASES="${CASES}"
ENVEOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo
echo "cases: ${N_CASES}  (peptides=[${PEPTIDES}] chems=[${CHEMS}])"

if [[ $DRY == 1 ]]; then
  echo "--dry-run: resolve-only trajectory check, submitting nothing."
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
  --partition="$PARTITION" "${PIN[@]}" --time="$GPU_TIME" --job-name="${RUN_ID}_traj" \
  scripts/sdedit_traj.sbatch "$ENV_FILE")
echo "stage 1 GPU  trajectories  job ${GPU_ID} (${PARTITION}${EXCLUDE:+ excl $EXCLUDE})"

ROS_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$ROS_TIME" --job-name="${RUN_ID}_dg" \
  scripts/grid_slider_rosetta.sbatch "$ENV_FILE")
echo "stage 2 CPU  rosetta dG    job ${ROS_ID} (sibling of stage 1)"

BUILD_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$BUILD_TIME" \
  --dependency=afterok:"$GPU_ID":"$ROS_ID" --job-name="${RUN_ID}_build" \
  scripts/grid_slider_build.sbatch "$ENV_FILE")
echo "stage 3 CPU  sprites+page  job ${BUILD_ID} (afterok:${GPU_ID},${ROS_ID})"
echo
echo "page: ${REPO}/evaluation_results/${RUN_ID}/slider/cyclization_grid_viewer.html"
