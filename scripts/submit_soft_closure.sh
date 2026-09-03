#!/bin/bash
# Submitter for the soft-closure projection experiment. Calls sbatch and nothing else.
#
#   bash scripts/submit_soft_closure.sh [--smoke] [--gpu-partition P] [--gpu-nodelist N]
#                                       [--cpu-partition P] [--shards N] [--dry-run]
#
# Writes ONE timestamped env file and passes its PATH as a positional argument. Never
# `sbatch --export`: any explicit export list sets SLURM_GET_USER_ENV=1, slurmd then fails to
# rebuild the login environment on the compute node, and the job is requeued and HELD.
#
# Stage 1  CPU   OpenMM staged pull projection            (soft_closure_project.sbatch)
# Stage 2a GPU   SDEdit on the PROJECTED peptides   \  siblings: they queue concurrently
# Stage 2b GPU   SDEdit on the SAME peptides, raw   /   and are the paired comparison
# Stage 3  GPU   SDEdit trajectory dump for the GIF      (soft_closure_traj.sbatch)
# Stage 4  CPU   before/after PNG + GIF                  (soft_closure_render.sbatch)
# Stage 5  CPU   paired-arm table                        (soft_closure_summarize.sbatch)
#
# Stage 2b depends only on stage 1, so it never waits behind 2a.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

GPU_PARTITION="general"; GPU_NODELIST=""; CPU_PARTITION="cpu"
PROJ_TIME="04:00:00"; GPU_TIME="06:00:00"; CPU_TIME="01:00:00"
SHARDS=4; SMOKE=0; DRYRUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-partition) GPU_PARTITION="$2"; shift 2 ;;
    --gpu-nodelist)  GPU_NODELIST="$2";  shift 2 ;;
    --cpu-partition) CPU_PARTITION="$2"; shift 2 ;;
    --shards)        SHARDS="$2";        shift 2 ;;
    --smoke)         SMOKE=1;            shift ;;
    --dry-run)       DRYRUN=1;           shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="softclose_${STAMP}"; [[ $SMOKE == 1 ]] && RUN_ID="softclose_smoke_${STAMP}"
[[ $SMOKE == 1 ]] && { SHARDS=1; PROJ_TIME="00:40:00"; GPU_TIME="01:00:00"; }

# The test bed: the mainchain cases the LNR SDEdit sweep could not close. The first three
# never closed once in 20 grid points (best C-N 11.4 / 16.9 / 14.0 A -- not near-misses); the
# rest closed at most 3 times in 20 and all sit in the widest input-gap bin.
TEST_BED="LNR_1jrr_A_P LNR_4x3h_A_B LNR_3cvl_A_B LNR_4tzm_A_C LNR_6j0x_A_E LNR_5vao_C_H LNR_2z5n_A_B LNR_4piq_A_B"
[[ $SMOKE == 1 ]] && TEST_BED="LNR_3cvl_A_B LNR_4x3h_A_B"

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<ENVEOF
# Auto-written by scripts/submit_soft_closure.sh at ${STAMP}.
PROJ_DIR=${REPO}/evaluation_results/${RUN_ID}/projection
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}/sdedit
TRAJ_DIR=${REPO}/evaluation_results/${RUN_ID}/traj
FIG_DIR=${REPO}/evaluation_results/${RUN_ID}/figures
LNR_METADATA=${REPO}/CPSea_data/lnr_staged/metadata/lnr_test.parquet
# OpenMM cannot be installed alongside the training environment's pins, so it lives in its
# own venv. Recreate with:
#   uv venv --python 3.11 .venv_openmm
#   uv pip install --python .venv_openmm/bin/python openmm pdbfixer numpy pandas pyarrow
OPENMM_PYTHON=${REPO}/.venv_openmm/bin/python
# Same frozen pin the original LNR sweep used, so the flow model and the AE match and the
# only difference from that sweep is the projection.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
TEST_BED_EXAMPLES="${TEST_BED}"

# --- projection ---
CYC_TYPE=mainchain
N_PROJECTIONS=$([[ $SMOKE == 1 ]] && echo 2 || echo 16)
TORSION_SIGMA_DEG=12.0
K_PULL=2000.0
K_CONTACT=500.0
CONTACT_TOL_A=0.75
CONTACTS_PER_RESIDUE=1
LADDER_A="20 15 12 9 6 3"
MIN_MAX_ITERATIONS=$([[ $SMOKE == 1 ]] && echo 100 || echo 1000)
# Stop once the generator can finish the job: the sweep closed rings reliably from this range.
STOP_CB_A=9.0
ACCEPT_RETENTION=0.8
KEEP_PER_EXAMPLE=$([[ $SMOKE == 1 ]] && echo 1 || echo 4)
PROJ_SEED=0

# --- SDEdit arms (identical for projected and control) ---
# t_ca 0.8 / t_lat 0.2-0.4 was the best mainchain corner in the full grid; three seeds so a
# per-peptide "did it ever close" is not one coin flip.
SHARD_COUNT=${SHARDS}
T_CA_GRID="$([[ $SMOKE == 1 ]] && echo "0.8" || echo "0.8 0.6")"
T_LAT_GRID="$([[ $SMOKE == 1 ]] && echo "0.4" || echo "0.4 0.2")"
SEEDS="$([[ $SMOKE == 1 ]] && echo "0" || echo "0 1 2")"
NSTEPS=$([[ $SMOKE == 1 ]] && echo 50 || echo 0)

# --- trajectory / figures ---
TRAJ_T_CA=0.8
TRAJ_T_LAT=0.4
TRAJ_SEED=0
RENDER_PER_EXAMPLE=1
RENDER_MAX_SDEDIT_FRAMES=40
ENVEOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

if [[ $DRYRUN == 1 ]]; then
  echo "--dry-run: resolving the projection inputs locally, submitting nothing."
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  # shellcheck disable=SC2086
  "$OPENMM_PYTHON" -u scripts/soft_closure_project.py \
    --metadata "$LNR_METADATA" --out-dir "$PROJ_DIR" --examples ${TEST_BED_EXAMPLES} \
    --cyc-type "$CYC_TYPE" --n-projections "$N_PROJECTIONS" --dry-run
  exit 0
fi

# Submitting from inside an allocation would otherwise leak SLURM_* into the child jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

GPU_NODE_ARG=(); [[ -n "$GPU_NODELIST" ]] && GPU_NODE_ARG=(--nodelist="$GPU_NODELIST")

PROJ_ID=$(sbatch --parsable \
  --partition="$CPU_PARTITION" --time="$PROJ_TIME" --job-name="${RUN_ID}_proj" \
  scripts/soft_closure_project.sbatch "$ENV_FILE")
echo "stage 1  CPU  projection            job ${PROJ_ID} (${CPU_PARTITION})"

EDIT_ID=$(sbatch --parsable \
  --partition="$GPU_PARTITION" "${GPU_NODE_ARG[@]}" --time="$GPU_TIME" \
  --array=0-$((SHARDS - 1)) --dependency=afterok:"$PROJ_ID" --job-name="${RUN_ID}_projected" \
  scripts/soft_closure_sdedit.sbatch "$ENV_FILE" projected)
echo "stage 2a GPU  SDEdit / projected    job ${EDIT_ID} (afterok:${PROJ_ID}, ${SHARDS} shard(s))"

CTRL_ID=$(sbatch --parsable \
  --partition="$GPU_PARTITION" "${GPU_NODE_ARG[@]}" --time="$GPU_TIME" \
  --array=0-$((SHARDS - 1)) --dependency=afterok:"$PROJ_ID" --job-name="${RUN_ID}_control" \
  scripts/soft_closure_sdedit.sbatch "$ENV_FILE" control)
echo "stage 2b GPU  SDEdit / control      job ${CTRL_ID} (afterok:${PROJ_ID}, sibling of 2a)"

TRAJ_ID=$(sbatch --parsable \
  --partition="$GPU_PARTITION" "${GPU_NODE_ARG[@]}" --time="$GPU_TIME" \
  --dependency=afterok:"$PROJ_ID" --job-name="${RUN_ID}_traj" \
  scripts/soft_closure_traj.sbatch "$ENV_FILE")
echo "stage 3  GPU  trajectory dump       job ${TRAJ_ID} (afterok:${PROJ_ID}, sibling of 2a/2b)"

FIG_ID=$(sbatch --parsable \
  --partition="$CPU_PARTITION" --time="$CPU_TIME" \
  --dependency=afterok:"$TRAJ_ID" --job-name="${RUN_ID}_fig" \
  scripts/soft_closure_render.sbatch "$ENV_FILE")
echo "stage 4  CPU  PNG + GIF             job ${FIG_ID} (afterok:${TRAJ_ID})"

SUM_ID=$(sbatch --parsable \
  --partition="$CPU_PARTITION" --time="$CPU_TIME" \
  --dependency=afterok:"$EDIT_ID":"$CTRL_ID" --job-name="${RUN_ID}_sum" \
  scripts/soft_closure_summarize.sbatch "$ENV_FILE")
echo "stage 5  CPU  paired-arm table      job ${SUM_ID} (afterok:${EDIT_ID},${CTRL_ID})"

echo
echo "results: ${REPO}/evaluation_results/${RUN_ID}"
echo "figures: ${REPO}/evaluation_results/${RUN_ID}/figures"
