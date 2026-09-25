#!/bin/bash
# Submitter for the LP -> CP pre-build check suite. Calls sbatch and nothing else.
#
#   bash scripts/submit_precheck.sh [--task0] [--task1] [--task2] [--task3] [--all]
#                                   [--smoke] [--partition P] [--nodelist N] [--exclude N]
#                                   [--run-id ID] [--shards-feas N] [--shards-profile N]
#                                   [--shards-decoy N]
#
# Tasks 1, 2 and 3 are INDEPENDENT and are submitted as siblings, so they queue
# concurrently rather than one waiting on another. Stages WITHIN a task chain on
# --dependency=afterok.
#
# Writes ONE timestamped env file and passes its PATH as a positional argument. Never
# `sbatch --export`: any explicit export list (--export=ALL,K=v, --export=K=v and
# --export=NONE alike) sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD -- stranding every
# dependent job on Dependency. Measured here, jobs 22184 and 22243.
#
# Everything here is CPU-only. Task 0's confirmation run is the one GPU job in the suite and
# it is NOT submitted from here: the pin job prints its exact command, to be run after
# checking `myfree` for an available node.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"; NODELIST=""; EXCLUDE=""
T0=0; T1=0; T2=0; T3=0; SMOKE=0; RESUME_T1=0
RUN_ID=""; FEAS_SHARDS=4; PROFILE_SHARDS=6; DECOY_SHARDS=16
FEAS_TIME="08:00:00"; PROFILE_TIME="04:00:00"
# Walltimes stay at or below 8h. MEASURED 2026-09-24: a 24h request for the minimize
# stage sat PENDING forever on `general` while every job at 8h or less ran -- the partition
# will not schedule it. The identical job ran fine at 2h in the smoke, which is the control
# that pins the cause. Minimize needs ~2.3h/shard (28 s/pose x 300 poses), scoring ~20min,
# so 8h is already several times the measured need. Do NOT raise these "to be safe".
SAMPLE_TIME="06:00:00"; MIN_TIME="08:00:00"; SCORE_TIME="08:00:00"; REPORT_TIME="00:30:00"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task0) T0=1; shift ;;
    --task1) T1=1; shift ;;
    --resume-task1) T1=1; RESUME_T1=1; shift ;;
    --task2) T2=1; shift ;;
    --task3) T3=1; shift ;;
    --all)   T0=1; T1=1; T2=1; T3=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --exclude)   EXCLUDE="$2";   shift 2 ;;
    --run-id)    RUN_ID="$2";    shift 2 ;;
    --shards-feas)    FEAS_SHARDS="$2";    shift 2 ;;
    --shards-profile) PROFILE_SHARDS="$2"; shift 2 ;;
    --shards-decoy)   DECOY_SHARDS="$2";   shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if (( T0 + T1 + T2 + T3 == 0 )); then
  echo "nothing selected. Pass --task0/--task1/--task2/--task3 or --all." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
[[ -n "$RUN_ID" ]] || RUN_ID="precheck_${STAMP}"
[[ $SMOKE == 1 ]] && RUN_ID="precheck_smoke_${STAMP}"
OUT_DIR="${REPO}/evaluation_results/${RUN_ID}"

mkdir -p slurm_logs env_files "${OUT_DIR}"
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"

if (( SMOKE )); then
  # A capped solver budget makes feasibility levels report INFEASIBLE that a full budget
  # would close, so a smoke run is never a result. The preamble prints that on every job.
  FEAS_SHARDS=1; PROFILE_SHARDS=1; DECOY_SHARDS=1
  FEAS_TIME="01:00:00"; PROFILE_TIME="00:30:00"
  SAMPLE_TIME="00:40:00"; MIN_TIME="02:00:00"; SCORE_TIME="01:00:00"
fi

cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_precheck.sh at ${STAMP}. Passed as a positional argument.
RUN_ID=${RUN_ID}
OUT_DIR=${OUT_DIR}
PRECHECK_CONFIG=${REPO}/configs/pose_decoy/precheck.yaml
# Bridge distance windows MEASURED on CPSea natives by the Milestone 1.5 calibration.
# Asserting a textbook window here would make every bridged feasibility call a guess.
BRIDGE_WINDOWS=${REPO}/evaluation_results/m15_20260922_184436/bridge_windows.json
FEAS_DIR=${OUT_DIR}/feasibility
PROFILE_DIR=${OUT_DIR}/profile
DECOY_DIR=${OUT_DIR}/decoys
PIN_DIR=${OUT_DIR}/pin
FEAS_SHARDS=${FEAS_SHARDS}
PROFILE_SHARDS=${PROFILE_SHARDS}
DECOY_SHARDS=${DECOY_SHARDS}
PROFILE_SETS="lnr pepbench cpsea"
ADVERSARY_PAIRS="cpsea:lnr lnr:pepbench"
SMOKE=${SMOKE}
$(if (( SMOKE )); then cat <<SMK
FEAS_LIMIT=2
FEAS_MAX_SOLVES=8
FEAS_CHEMISTRIES=mainchain
PROFILE_LIMIT=20
DECOY_LIMIT=2
SMK
fi)
EOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

# Scrub SLURM_* so a submission made from inside an allocation does not inherit that job's
# array/task variables into the new jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

NODE_ARG=()
[[ -n "$NODELIST" ]] && NODE_ARG=(--nodelist="$NODELIST")
[[ -n "$EXCLUDE" ]] && NODE_ARG+=(--exclude="$EXCLUDE")
SB=(sbatch --parsable --partition="$PARTITION" "${NODE_ARG[@]}")

# ----------------------------------------------------------------- task 0: pin
if (( T0 )); then
  PIN_ID=$("${SB[@]}" --time="$REPORT_TIME" --job-name="${RUN_ID}_pin" \
    scripts/precheck_pin.sbatch "$ENV_FILE")
  echo "task 0  pin baseline          : job ${PIN_ID}"
  echo "         (the GPU confirmation command is PRINTED by that job, not submitted)"
fi

# ------------------------------------------------- task 1: decoy filter calibration
if (( T1 )); then
  # Written here, not in the job: the OpenMM environment has no yaml/pandas to read the
  # config with, and the two environments cannot be merged.
  .venv/bin/python - "$ENV_FILE" <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
env = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v.strip('"')
os.environ.setdefault("ZFS", "/zfsauton/scratch/yixiz")
os.environ.setdefault("CPSEA_FULL_META", os.environ["ZFS"] + "/CPSea/CPSea_full/CPSea/preprocessed/metadata")
os.environ.setdefault("LP_META", os.environ["ZFS"] + "/LPData/preprocessed/metadata")
from script_utils import precheck_config
cfg = precheck_config.load(env["PRECHECK_CONFIG"])
out = env["DECOY_DIR"]
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "minimize.json"), "w") as fh:
    json.dump(cfg["decoy_calibration"]["minimize"], fh, indent=2)
print(f"wrote {out}/minimize.json")
PY

  MIN_DEP=()
  if (( RESUME_T1 )); then
    # Stage A already produced its manifests on disk, so the sample array is skipped and
    # minimize starts with no upstream dependency. Requires --run-id pointing at the run
    # whose manifests exist; the stage is resumable per pose, so re-running is safe.
    SAMPLE_ID="(skipped: --resume-task1)"
    shopt -s nullglob
    _man=("${DECOY_DIR:-${OUT_DIR}/decoys}"/manifest_shard*.jsonl)
    if (( ${#_man[@]} == 0 )); then
      echo "FATAL: --resume-task1 needs stage-A manifests under ${OUT_DIR}/decoys," >&2
      echo "and none are there. Pass --run-id of the run that produced them." >&2
      exit 1
    fi
    echo "resume: found ${#_man[@]} stage-A manifest(s); skipping the sample array"
  else
    SAMPLE_ID=$("${SB[@]}" --time="$SAMPLE_TIME" --array=0-$((DECOY_SHARDS - 1)) \
      --job-name="${RUN_ID}_dsample" scripts/precheck_decoy_sample.sbatch "$ENV_FILE")
    MIN_DEP=(--dependency=afterok:"$SAMPLE_ID")
  fi
  MIN_ID=$("${SB[@]}" --time="$MIN_TIME" --array=0-$((DECOY_SHARDS - 1)) \
    "${MIN_DEP[@]}" --job-name="${RUN_ID}_dmin" \
    scripts/precheck_decoy_minimize.sbatch "$ENV_FILE")
  SCORE_ID=$("${SB[@]}" --time="$SCORE_TIME" --array=0-$((DECOY_SHARDS - 1)) \
    --dependency=afterok:"$MIN_ID" --job-name="${RUN_ID}_dscore" \
    scripts/precheck_decoy_score.sbatch "$ENV_FILE")
  # afterANY on the report: a slow scoring shard hitting its wall clock should cost sample
  # size, not the whole report.
  DREP_ID=$("${SB[@]}" --time="$REPORT_TIME" --dependency=afterany:"$SCORE_ID" \
    --job-name="${RUN_ID}_drep" scripts/precheck_decoy_report.sbatch "$ENV_FILE")
  echo "task 1  decoy calibration     : sample ${SAMPLE_ID} -> min ${MIN_ID} -> score ${SCORE_ID} -> report ${DREP_ID}"
fi

# ------------------------------------------------- task 2: LNR closure feasibility
if (( T2 )); then
  FEAS_ID=$("${SB[@]}" --time="$FEAS_TIME" --array=0-$((FEAS_SHARDS - 1)) \
    --job-name="${RUN_ID}_feas" scripts/precheck_feasibility.sbatch "$ENV_FILE")
  FREP_ID=$("${SB[@]}" --time="$REPORT_TIME" --dependency=afterany:"$FEAS_ID" \
    --job-name="${RUN_ID}_feasrep" scripts/precheck_feasibility_report.sbatch "$ENV_FILE")
  echo "task 2  LNR feasibility       : array ${FEAS_ID} -> report ${FREP_ID}"
fi

# --------------------------------- task 3: profile + reference + adversary + OOD
if (( T3 )); then
  N_SETS=3
  PROF_ID=$("${SB[@]}" --time="$PROFILE_TIME" \
    --array=0-$((N_SETS * PROFILE_SHARDS - 1)) \
    --job-name="${RUN_ID}_profile" scripts/precheck_profile.sbatch "$ENV_FILE")
  ADV_ID=$("${SB[@]}" --time="$REPORT_TIME" --dependency=afterany:"$PROF_ID" \
    --job-name="${RUN_ID}_adv" scripts/precheck_adversary.sbatch "$ENV_FILE")
  echo "task 3  profile + adversary   : array ${PROF_ID} -> adversary ${ADV_ID}"
fi

echo
echo "results: ${OUT_DIR}"
(( SMOKE )) && echo "SMOKE RUN -- capped budgets and limits. NOT a result."
exit 0
