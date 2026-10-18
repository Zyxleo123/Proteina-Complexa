#!/bin/bash
# Submitter for the LP -> CP Stage 0 profile. Calls sbatch and nothing else.
#
#   bash scripts/submit_stage0_profile.sh [--nodelist gpu29] [--partition general]
#                                         [--cpu-partition cpu] [--exclude gpu28]
#                                         [--metadata PARQUET]
#                                         [--cases-per-end 4] [--seeds "0 1"]
#                                         [--cyc-type disulfide] [--t-ca 0.8] [--t-lat 0.4]
#                                         [--project-structures 1360] [--gpu-mem 24G]
#                                         [--smoke] [--dry-run]
#
# What it measures, and why in four jobs instead of one
# -----------------------------------------------------
#   1. select   (CPU)  which complexes, written down as an artefact
#   2. gpu      (GPU)  the generations, at production sampler settings
#   3. rosetta  (CPU)  interface dG on what the GPU job wrote
#   4. report   (CPU)  the cost table and the Stage 1 projection
#
# Split by RESOURCE, per the cluster rules: only step 2 needs a GPU, and asking for one to
# read a parquet or run a scorer just queues that work behind real training. Chained with
# --dependency=afterok, so a broken upstream stage stops the chain rather than producing a
# confident report over missing data.
#
# Writes ONE timestamped env file per submission and passes its PATH as a positional argument
# to each batch script. Deliberately NOT `sbatch --export`: any explicit export list sets
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the compute
# node, and the job is requeued and HELD, stranding every dependent job on Dependency.
#
# PICK THE NODE FROM A FRESH `myfree`. The GPU job is sized for a 48 GB a6000: MEM_PROFILE
# a5000/a5000_min silently force self_cond=false, which would profile a different model than
# the one Stage 1 runs, and an rtx_2080_ti (11 GB) can die on a CUDA allocation with no Python
# traceback, which reads as a mysterious silent FAILED.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"; CPU_PARTITION="cpu"; NODELIST=""
# `general` is a MIXED pool: gpu28 is an a5000, the rest a6000. Landing on the a5000 would
# silently profile a different model (MEM_PROFILE a5000/a5000_min force self_cond=false), so
# it is excluded by default rather than left to chance. Re-check against a fresh `myfree`
# before each launch -- this list is a snapshot, not a fact about the cluster.
EXCLUDE="gpu28"
# Host RAM for the GPU job, overriding the 64G in its #SBATCH header. 64G is inherited from
# the SDEdit sweep arm and is what PENDS the job when the a6000 pool is busy: the constraint
# is host memory, not GPUs -- a node can show a free a6000 and still have 4-28G of RAM left,
# and Slurm will hold the job until 64G frees up. Lower it to fit the node that is actually
# available. The floor is set by checkpoint loading (a 2.9 GB flow ckpt plus a 4.1 GB AE, each
# read to CPU before moving to the device, so peak is well above their sum); 24G has headroom
# over that but is an ESTIMATE from file sizes, not a measured MaxRSS. If it is too low the job
# dies as a Slurm OOM kill, which is loud and distinguishable from a CUDA failure -- raise it
# and resubmit rather than assuming the model is at fault.
GPU_MEM=""
GPU_TIME="02:00:00"; CPU_TIME="04:00:00"; REPORT_TIME="00:20:00"
METADATA="CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet"
CASES_PER_END=4; SEEDS="0"; CYC_TYPE="disulfide"; T_CA="0.8"; T_LAT="0.4"
PROJECT_STRUCTURES=1360; DRYRUN=0; SMOKE=0; NSTEPS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)               SMOKE=1;                 shift ;;
    --partition)           PARTITION="$2";          shift 2 ;;
    --cpu-partition)       CPU_PARTITION="$2";      shift 2 ;;
    --nodelist)            NODELIST="$2";           shift 2 ;;
    --exclude)             EXCLUDE="$2";            shift 2 ;;
    --gpu-mem)             GPU_MEM="$2";            shift 2 ;;
    --metadata)            METADATA="$2";           shift 2 ;;
    --cases-per-end)       CASES_PER_END="$2";      shift 2 ;;
    --seeds)               SEEDS="$2";              shift 2 ;;
    --cyc-type)            CYC_TYPE="$2";           shift 2 ;;
    --t-ca)                T_CA="$2";               shift 2 ;;
    --t-lat)               T_LAT="$2";              shift 2 ;;
    --gpu-time)            GPU_TIME="$2";           shift 2 ;;
    --project-structures)  PROJECT_STRUCTURES="$2"; shift 2 ;;
    --dry-run)             DRYRUN=1;                shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Resolve to an absolute path: the batch scripts run from the spool directory, not the repo.
[[ "$METADATA" = /* ]] || METADATA="${REPO}/${METADATA}"
[[ -f "$METADATA" ]] || { echo "FATAL: metadata not found: $METADATA" >&2; exit 1; }

# Smoke: prove the four-job chain end to end for a few GPU-minutes. One case per pocket end,
# one unguided arm, a coarse sampler. Its COST NUMBERS ARE NOT THE PROFILE -- nsteps=50 is an
# eighth of production, so quoting them would understate Stage 1 by roughly that factor. Its
# own RUN_ID keeps it from resuming onto, or being mistaken for, the real run.
if [[ $SMOKE == 1 ]]; then
  CASES_PER_END=1; SEEDS="0"; NSTEPS=50
  GUIDANCE_OVERRIDE="0|-|-|-|-|-"
  GPU_TIME="00:40:00"; CPU_TIME="01:00:00"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="stage0_profile_${STAMP}"
[[ $SMOKE == 1 ]] && RUN_ID="stage0_profile_smoke_${STAMP}"
RESULTS_DIR="${REPO}/evaluation_results/${RUN_ID}"

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_stage0_profile.sh at ${STAMP}.
RESULTS_DIR=${RESULTS_DIR}
LNR_METADATA=${METADATA}
EXAMPLES_FILE=${RESULTS_DIR}/profile/cases.txt
PROFILE_N_SMALL=${CASES_PER_END}
PROFILE_N_LARGE=${CASES_PER_END}

# The same frozen pin and AE the SDEdit and de novo arms used, so this profile describes the
# model Stage 1 will run rather than whichever checkpoint a live run had reached today.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt

# One production sampler setting, not a sweep. NSTEPS=0 means the design sampler's own value
# (400): profiling a cheaper screening setting would understate Stage 1's cost.
SHARD_COUNT=1
CYC_TYPES="${CYC_TYPE}"
T_CA_GRID="${T_CA}"
T_LAT_GRID="${T_LAT}"
SEEDS="${SEEDS}"
NSTEPS=${NSTEPS}
PEPTIDE_LIMIT=0

# Two arms at the same grid point: unguided, and the intended closure guidance. Guidance adds
# a gradient step per sampler step, so its cost is a separate measurement, not a correction.
GUIDANCE_ARMS="${GUIDANCE_OVERRIDE:-0|-|-|-|-|- 50|ca_mse,contact_hinge|decay|1.0|1|identity}"

# Both dirs are required. The peptide PDBs are the artefact; the COMPLEXES are what the CPU
# Rosetta stage needs, and they can only be written HERE -- the loader hands out a
# target-centred crop whose frame is redrawn per process, so a peptide saved without its
# receptor can never be re-united with it (scripts/complex_frame.py).
PDB_DIR=${RESULTS_DIR}/pdb
COMPLEX_DIR=${RESULTS_DIR}/complexes

# Projection target for the report: the plan's Stage 1 ceiling.
PROJECT_STRUCTURES=${PROJECT_STRUCTURES}
EOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

SB=(--partition="$PARTITION")
[[ -n "$NODELIST" ]] && SB+=(--nodelist="$NODELIST")
[[ -n "$EXCLUDE" && -z "$NODELIST" ]] && SB+=(--exclude="$EXCLUDE")
[[ -n "$GPU_MEM" ]] && SB+=(--mem="$GPU_MEM")

if [[ $DRYRUN == 1 ]]; then
  echo "--dry-run: env file written, nothing submitted."
  exit 0
fi

# Submitting from inside an allocation would otherwise leak this shell's SLURM_* into the
# child jobs and confuse their array/step accounting.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

JID_SEL=$(sbatch --parsable --partition="$CPU_PARTITION" --time="$REPORT_TIME" \
  scripts/stage0_profile_select.sbatch "$ENV_FILE")
echo "select  : $JID_SEL"

JID_GPU=$(sbatch --parsable "${SB[@]}" --time="$GPU_TIME" \
  --dependency=afterok:"$JID_SEL" \
  scripts/stage0_profile_gpu.sbatch "$ENV_FILE")
echo "gpu     : $JID_GPU"

JID_ROS=$(sbatch --parsable --partition="$CPU_PARTITION" --time="$CPU_TIME" \
  --dependency=afterok:"$JID_GPU" \
  scripts/stage0_profile_rosetta.sbatch "$ENV_FILE")
echo "rosetta : $JID_ROS"

JID_REP=$(sbatch --parsable --partition="$CPU_PARTITION" --time="$REPORT_TIME" \
  --dependency=afterok:"$JID_ROS" \
  scripts/stage0_profile_report.sbatch "$ENV_FILE")
echo "report  : $JID_REP"

echo
echo "results : ${RESULTS_DIR}"
echo "report  : ${RESULTS_DIR}/profile/stage0_profile.md"
echo "watch   : squeue -j ${JID_SEL},${JID_GPU},${JID_ROS},${JID_REP}"
echo "READ THE FIRST JOB'S .out BEFORE TRUSTING A CHAIN THAT 'SUCCEEDED'."
