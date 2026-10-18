#!/bin/bash
#
# Submitter for the D-Flow LP -> CP experiment. Calls sbatch and nothing else.
#
# Stage graph (independent stages stay siblings so they queue concurrently):
#
#     calibrate (CPU) ......................... instrument check, no dependency
#     opt (GPU array) -> rescore (CPU) -> rosetta (CPU array) -> report (CPU)
#
# Configuration travels as a timestamped ENV FILE passed as a POSITIONAL argument, never via
# `sbatch --export`: any explicit export list makes sbatch set SLURM_GET_USER_ENV=1, slurmd
# then fails to rebuild the login environment on the compute node, and the job is requeued
# and HELD with "(user env retrieval failed requeued held)", stranding every afterok
# dependent on `Dependency`. Measured on this cluster, jobs 22184 and 22243. A file also
# handles spaces, so the list-valued variables here need no comma encoding.
#
# Usage:
#   scripts/submit_dflow.sh --smoke                    # 1 peptide, 2 restarts, tiny grids
#   scripts/submit_dflow.sh                            # the pilot
#   scripts/submit_dflow.sh --init-mode sdedit --tag sd
#   scripts/submit_dflow.sh --dry                      # print the plan, submit nothing

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"; NODELIST=""; EXCLUDE=""
TIME="08:00:00"; CPU_TIME="04:00:00"; ROS_TIME="1-00:00:00"
SHARDS=4; ROSETTA_SHARDS=4
SMOKE=0; DRY=0; TAG=""
CLI_INIT=""; CLI_RESTARTS=""; CLI_OPT_STEPS=""; CLI_OBJECTIVES=""; CLI_CYC=""
CLI_LIMIT=""; CLI_EXAMPLES_FILE=""; CLI_OPT_NSTEPS=""; CLI_EVAL_NSTEPS=""
BASELINE_ARMS=""

# Which receptor representation the peptides are staged against. This is NOT a detail: LNR's
# full contiguous receptor differs from the ~14-fragment pocket the model was trained on, and
# that segmentation mismatch was measured on this project at +22 to +34 points of closure --
# enough to invalidate an earlier SDEdit sweep on its own. The SDEdit baselines this
# experiment compares against were run on `lnr_pocket14`, so the D-Flow arms default to the
# same staging; comparing across it would credit the method with a receptor artifact.
METADATA="CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)        SMOKE=1; shift ;;
    --dry)          DRY=1; shift ;;
    --tag)          TAG="$2"; shift 2 ;;
    --partition)    PARTITION="$2"; shift 2 ;;
    --nodelist)     NODELIST="$2"; shift 2 ;;
    --exclude)      EXCLUDE="$2"; shift 2 ;;
    --time)         TIME="$2"; shift 2 ;;
    --shards)       SHARDS="$2"; shift 2 ;;
    --rosetta-shards) ROSETTA_SHARDS="$2"; shift 2 ;;
    --init-mode)    CLI_INIT="$2"; shift 2 ;;
    --restarts)     CLI_RESTARTS="$2"; shift 2 ;;
    --opt-steps)    CLI_OPT_STEPS="$2"; shift 2 ;;
    --opt-nsteps)   CLI_OPT_NSTEPS="$2"; shift 2 ;;
    --eval-nsteps)  CLI_EVAL_NSTEPS="$2"; shift 2 ;;
    --objectives)   CLI_OBJECTIVES="$2"; shift 2 ;;
    --cyc-types)    CLI_CYC="$2"; shift 2 ;;
    --limit)        CLI_LIMIT="$2"; shift 2 ;;
    --examples-file) CLI_EXAMPLES_FILE="$2"; shift 2 ;;
    --baselines)    BASELINE_ARMS="$2"; shift 2 ;;
    --metadata)     METADATA="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
SUFFIX="${TAG:+_${TAG}}"
RUN_ID="dflow${SUFFIX}_${STAMP}"; [[ $SMOKE == 1 ]] && RUN_ID="dflow${SUFFIX}_smoke_${STAMP}"

mkdir -p "${REPO}/env_files" "${REPO}/slurm_logs"
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"

# The D-Flow arm is always re-scored; baselines are appended so the SAME instrument reads
# them. Each entry is <name>|<glob>[|<stage>].
# The arm NAME carries the tag, so two D-Flow runs that differ only in init mode are not
# pooled under one label when several runs are read into a single combined report.
ARM_NAME="dflow${TAG:+_${TAG}}"
RESCORE_ARMS="${ARM_NAME}|${REPO}/evaluation_results/${RUN_ID}/dflow_shard*.jsonl"
[[ -n "$BASELINE_ARMS" ]] && RESCORE_ARMS="${RESCORE_ARMS} ${BASELINE_ARMS}"

cat > "$ENV_FILE" <<EOF
# D-Flow LP -> CP run ${RUN_ID}, written $(date -Is) by $(basename "${BASH_SOURCE[0]}").
# Sourced by scripts/_dflow_preamble.sh as a positional argument.

RUN_ID=${RUN_ID}
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
PDB_DIR=${REPO}/evaluation_results/${RUN_ID}/pdbs
COMPLEX_DIR=${REPO}/evaluation_results/${RUN_ID}/complexes
Z_DIR=${REPO}/evaluation_results/${RUN_ID}/z
RESCORE_DIR=${REPO}/evaluation_results/${RUN_ID}/rescored
ROSETTA_DIR=${REPO}/evaluation_results/${RUN_ID}/rosetta

# Pinned by absolute path so "the checkpoint" means the same thing next week. This is the
# flow trained against \$CPSEA_AE_CKPT_PATH; evaluating it against any other AE is the
# pin trap that has produced incomparable loss curves on this project before.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
LNR_METADATA=${REPO}/${METADATA}

SHARD_COUNT=${SHARDS}
$([[ -n "$CLI_EXAMPLES_FILE" ]] && echo "EXAMPLES_FILE=${CLI_EXAMPLES_FILE}")

# --- initialization -------------------------------------------------------------------
# source = D-Flow as published (optimize the source noise, t_start = 0).
# sdedit = optimize an intermediate SDEdit state. That is an ADAPTATION, not the published
# method, and it inherits SDEdit's own bias toward the input, so it is reported as its own
# arm and never pooled with source.
INIT_MODE=${CLI_INIT:-source}
T_CA=0.8
T_LAT=0.4

# --- sampler --------------------------------------------------------------------------
# vf = the pure probability-flow ODE. The production sampler is an SDE (sampling_mode: sc)
# that draws fresh noise every step, which would make source -> sample not a function and
# the gradient one arbitrary realization.
SAMPLER_MODE=vf
OPT_NSTEPS=$([[ $SMOKE == 1 ]] && echo 10 || echo ${CLI_OPT_NSTEPS:-30})
EVAL_NSTEPS=$([[ $SMOKE == 1 ]] && echo 50 || echo ${CLI_EVAL_NSTEPS:-200})

# --- optimization ---------------------------------------------------------------------
RESTARTS=$([[ $SMOKE == 1 ]] && echo 2 || echo ${CLI_RESTARTS:-16})
OPT_STEPS=$([[ $SMOKE == 1 ]] && echo 5 || echo ${CLI_OPT_STEPS:-60})
LR_CA=0.02
LR_LAT=0.05
GRAD_CLIP=1.0
SEED=0
LOG_EVERY=$([[ $SMOKE == 1 ]] && echo 1 || echo 10)

# Objective arms, space separated; "-" is the script default. Start small and expand what
# looks promising rather than running a blind grid.
OBJECTIVES="${CLI_OBJECTIVES:--}"
CYC_TYPES="${CLI_CYC:-$([[ $SMOKE == 1 ]] && echo "disulfide" || echo "mainchain disulfide isopeptide")}"
PEPTIDE_LIMIT=$([[ $SMOKE == 1 ]] && echo 1 || echo ${CLI_LIMIT:-0})

# Stop cleanly under the wall instead of being killed at it: the shard file is append-only
# and resume skips finished run_keys, so hitting this means resubmitting the same command.
MAX_SECONDS=$([[ $SMOKE == 1 ]] && echo 1800 || echo 25200)

# --- scoring --------------------------------------------------------------------------
RESCORE_ARMS="${RESCORE_ARMS}"
ROSETTA_SHARDS=${ROSETTA_SHARDS}
ROSETTA_STAGE=both
# Deliberately NOT --only-passing. Gating Rosetta on the structural conjunction would mean
# that an arm which fails a structural clause produces no energy data at all, and "we cannot
# say" would be indistinguishable from "it scored badly". The cost is bounded by
# ROSETTA_MAX_PER_EXAMPLE instead, which takes the best candidates per example by objective,
# so every arm gets a paired ddG column whether or not it qualifies. The report separates
# qualified from unqualified; the scorer does not have to.
ROSETTA_ONLY_PASSING=${ROSETTA_ONLY_PASSING:-}
ROSETTA_MAX_PER_EXAMPLE=$([[ $SMOKE == 1 ]] && echo 2 || echo 6)
ROSETTA_REPEATS=1
ROSETTA_MAX_SECONDS=$([[ $SMOKE == 1 ]] && echo 1800 || echo 82800)
CALIBRATE_LIMIT=$([[ $SMOKE == 1 ]] && echo 5 || echo 0)
EOF

echo "run id:   ${RUN_ID}"
echo "env file: ${ENV_FILE}"
echo "results:  ${REPO}/evaluation_results/${RUN_ID}"
echo

NODE_ARG=(); [[ -n "$NODELIST" ]] && NODE_ARG=(--nodelist="$NODELIST")
EXCL_ARG=(); [[ -n "$EXCLUDE" ]]  && EXCL_ARG=(--exclude="$EXCLUDE")
ARRAY_MAX=$((SHARDS - 1)); ROS_MAX=$((ROSETTA_SHARDS - 1))

if [[ $DRY == 1 ]]; then
  echo "--dry: not submitting. Env file written above; stages would be:"
  echo "  calibrate : sbatch --partition=$PARTITION --time=$CPU_TIME scripts/dflow_calibrate.sbatch $ENV_FILE"
  echo "  opt       : sbatch --partition=$PARTITION ${NODE_ARG[*]} --time=$TIME --array=0-$ARRAY_MAX scripts/dflow_opt.sbatch $ENV_FILE"
  echo "  rescore   : sbatch --partition=$PARTITION --time=$CPU_TIME --dependency=afterok:<opt> scripts/dflow_rescore.sbatch $ENV_FILE"
  echo "  rosetta   : sbatch --partition=$PARTITION --time=$ROS_TIME --array=0-$ROS_MAX --dependency=afterok:<rescore> scripts/dflow_rosetta.sbatch $ENV_FILE"
  echo "  report    : sbatch --partition=$PARTITION --time=$CPU_TIME --dependency=afterok:<rosetta> scripts/dflow_report.sbatch $ENV_FILE"
  exit 0
fi

# Submitting from inside an allocation would otherwise leak this shell's SLURM_* into the
# child job's environment and confuse its own array/task identity.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

CAL_ID=$(sbatch --parsable --partition="$PARTITION" --time="$CPU_TIME" \
  scripts/dflow_calibrate.sbatch "$ENV_FILE")
echo "calibrate : job ${CAL_ID} (independent -- the instrument check)"

OPT_ID=$(sbatch --parsable --partition="$PARTITION" "${NODE_ARG[@]}" "${EXCL_ARG[@]}" \
  --time="$TIME" --array="0-${ARRAY_MAX}" \
  scripts/dflow_opt.sbatch "$ENV_FILE")
echo "opt       : job ${OPT_ID} (${SHARDS} shard(s), ${PARTITION}${NODELIST:+/$NODELIST})"

RES_ID=$(sbatch --parsable --partition="$PARTITION" --time="$CPU_TIME" \
  --dependency="afterok:${OPT_ID}" \
  scripts/dflow_rescore.sbatch "$ENV_FILE")
echo "rescore   : job ${RES_ID} (afterok ${OPT_ID})"

ROS_ID=$(sbatch --parsable --partition="$PARTITION" --time="$ROS_TIME" \
  --array="0-${ROS_MAX}" --dependency="afterok:${RES_ID}" \
  scripts/dflow_rosetta.sbatch "$ENV_FILE")
echo "rosetta   : job ${ROS_ID} (${ROSETTA_SHARDS} shard(s), afterok ${RES_ID})"

REP_ID=$(sbatch --parsable --partition="$PARTITION" --time="$CPU_TIME" \
  --dependency="afterok:${ROS_ID}" \
  scripts/dflow_report.sbatch "$ENV_FILE")
echo "report    : job ${REP_ID} (afterok ${ROS_ID})"

echo
echo "watch: squeue -u \$USER -j ${CAL_ID},${OPT_ID},${RES_ID},${ROS_ID},${REP_ID}"
echo "logs:  ${REPO}/slurm_logs/"
