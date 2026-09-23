#!/bin/bash
# Submitter for Milestone 1.5 -- the geometric ceiling audit. Calls sbatch and nothing else.
#
#   bash scripts/submit_m15_ceiling_audit.sh --submit <stage> [options]
#
# Stages:
#   calibrate      CPU   bridge distance windows, measured on CPSea natives
#   ceiling        CPU   per-complex ceiling scan (array, one job per set x shard)
#   timing         CPU   section 2: the setup/marginal replica ratio
#   deliverable-e  CPU   section 6: CPSea_full competitor pairs + the spatial check
#   report         CPU   tables + figures from whatever landed on disk
#   all            the whole DAG
#
# Shape of the DAG:
#
#   calibrate ──afterok──> ceiling(array) ─┐
#   timing ────────────────────────────────┼──afterany──> report
#   e-index ──afterok──> e-spatial(array) ─┘
#
# `timing` and `deliverable-e` are SIBLINGS of the ceiling chain, not behind it, so they
# queue concurrently. `report` depends with afterANY: a stage that gated itself out (no
# input, empty candidate set) exits 0 with an explanation, and the report's job is to say
# which stages contributed. afterok there would strand the report on a deliberate gate.
#
# Everything here is CPU-only -- no GPU is requested anywhere, because asking for one to
# measure geometry just queues behind real work.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGE=""; DRYRUN=0; SMOKE=0
CPU_PARTITION="cpu"
# TIME_TIME fits 20 complexes x 10 replicas at ~81 s each (~4.6 h) with headroom. The
# timing job rewrites its summary after every complex, so a wall-clock kill degrades to a
# partial result rather than to nothing.
CALIB_TIME="02:00:00"; CEIL_TIME="06:00:00"; TIME_TIME="12:00:00"
EIDX_TIME="06:00:00"; ESPAT_TIME="06:00:00"; REPORT_TIME="01:00:00"
RUN_ID=""
declare -a EXTRA_ENV=()

usage() { sed -n '2,30p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)         STAGE="$2";         shift 2 ;;
    --cpu-partition)  CPU_PARTITION="$2"; shift 2 ;;
    --run-id)         RUN_ID="$2";        shift 2 ;;
    --ceil-time)      CEIL_TIME="$2";     shift 2 ;;
    --espat-time)     ESPAT_TIME="$2";    shift 2 ;;
    # Anything VAR=value lands in the env file, so every knob in _m15_preamble.sh is
    # settable without editing a script.
    --set)            EXTRA_ENV+=("$2");  shift 2 ;;
    --smoke)          SMOKE=1;            shift ;;
    --dry-run)        DRYRUN=1;           shift ;;
    -h|--help)        usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 >&2 ;;
  esac
done
[[ -n "$STAGE" ]] || { echo "FATAL: --submit <stage> is required" >&2; usage 1 >&2; }
case "$STAGE" in
  calibrate|ceiling|timing|deliverable-e|report|all) ;;
  *) echo "FATAL: unknown stage '$STAGE'" >&2; usage 1 >&2 ;;
esac

# Submitting from inside an allocation leaks SLURM_* into the child sbatch, which then
# inherits an array index or a job id that is not its own.
for v in $(env | sed -n 's/^\(SLURM_[A-Z_]*\)=.*/\1/p'); do unset "$v" || true; done

[[ -n "$RUN_ID" ]] || RUN_ID="m15_$( [[ $SMOKE -eq 1 ]] && echo smoke_ )$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO}/evaluation_results/${RUN_ID}"
ENV_DIR="${REPO}/evaluation_results/_m15_env"
mkdir -p "$OUT_DIR" "$ENV_DIR" "${REPO}/logs"

# ONE env file per submission, passed as a POSITIONAL argument. Arguments travel through
# the job record untouched; `sbatch --export` does not -- any explicit export list sets
# SLURM_GET_USER_ENV=1, slurmd fails to rebuild the login environment on the compute node,
# and the job is requeued and HELD.
ENV_FILE="${ENV_DIR}/${RUN_ID}.env"
{
  echo "# written by submit_m15_ceiling_audit.sh at $(date -Is)"
  echo "REPO=${REPO}"
  echo "RUN_ID=${RUN_ID}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "SMOKE=${SMOKE}"
  for kv in "${EXTRA_ENV[@]}"; do echo "$kv"; done
} > "$ENV_FILE"
echo "env file: $ENV_FILE"

# Shard counts come from the YAML so the config stays the single source of truth.
read -r LNR_SHARDS PEP_SHARDS E_SHARDS AUDIT_SETS < <("${REPO}/.venv/bin/python" -c "
import sys; sys.path.insert(0, '${REPO}')
import os
os.environ.setdefault('CPSEA_FULL_META', '/zfsauton/scratch/yixiz/CPSea/CPSea_full/CPSea/preprocessed/metadata')
os.environ.setdefault('LP_META', '/zfsauton/scratch/yixiz/LPData/preprocessed/metadata')
from script_utils import m15_config
c = m15_config.load('${REPO}/configs/pose_decoy/m15.yaml')
sets = c['audit']['sets']
print(sets['lnr']['shards'], sets['pepbench']['shards'],
      c['deliverable_e']['spatial_shards'], ','.join(sets))
")
if [[ $SMOKE -eq 1 ]]; then LNR_SHARDS=1; PEP_SHARDS=1; E_SHARDS=1; fi

submit() {  # submit <description> <sbatch args...>
  local desc="$1"; shift
  if [[ $DRYRUN -eq 1 ]]; then
    # To stderr: stdout is the job id this function returns to its caller.
    { printf 'DRY-RUN %-16s sbatch' "$desc"; printf ' %q' "$@"; printf '\n'; } >&2
    echo "999999"
    return
  fi
  local out; out="$(sbatch "$@")"
  echo "$desc: $out" >&2
  awk '{print $NF}' <<<"$out"
}

CALIB_JID=""; EIDX_JID=""; declare -a REPORT_DEPS=()

if [[ "$STAGE" == "calibrate" || "$STAGE" == "ceiling" || "$STAGE" == "all" ]]; then
  CALIB_JID="$(submit calibrate \
    --partition="$CPU_PARTITION" --time="$CALIB_TIME" \
    scripts/m15_calibrate.sbatch "$ENV_FILE")"
  REPORT_DEPS+=("$CALIB_JID")
fi

if [[ "$STAGE" == "ceiling" || "$STAGE" == "all" ]]; then
  DEP=(); [[ -n "$CALIB_JID" ]] && DEP=(--dependency=afterok:"$CALIB_JID")
  IFS=',' read -ra SETS <<< "$AUDIT_SETS"
  for s in "${SETS[@]}"; do
    case "$s" in lnr) N="$LNR_SHARDS" ;; pepbench) N="$PEP_SHARDS" ;; *) N=1 ;; esac
    JID="$(submit "ceiling-$s" \
      --partition="$CPU_PARTITION" --time="$CEIL_TIME" \
      --array="0-$((N - 1))" "${DEP[@]}" \
      scripts/m15_ceiling.sbatch "$ENV_FILE" "AUDIT_SET=$s" "NUM_SHARDS=$N")"
    REPORT_DEPS+=("$JID")
  done
fi

if [[ "$STAGE" == "timing" || "$STAGE" == "all" ]]; then
  JID="$(submit timing \
    --partition="$CPU_PARTITION" --time="$TIME_TIME" \
    scripts/m15_timing.sbatch "$ENV_FILE")"
  REPORT_DEPS+=("$JID")
fi

if [[ "$STAGE" == "deliverable-e" || "$STAGE" == "all" ]]; then
  EIDX_JID="$(submit e-index \
    --partition="$CPU_PARTITION" --time="$EIDX_TIME" \
    scripts/m15_deliverable_e_index.sbatch "$ENV_FILE")"
  JID="$(submit e-spatial \
    --partition="$CPU_PARTITION" --time="$ESPAT_TIME" \
    --array="0-$((E_SHARDS - 1))" --dependency=afterok:"$EIDX_JID" \
    scripts/m15_deliverable_e_spatial.sbatch "$ENV_FILE" "NUM_SHARDS=$E_SHARDS")"
  REPORT_DEPS+=("$EIDX_JID" "$JID")
fi

if [[ "$STAGE" == "report" || "$STAGE" == "all" ]]; then
  DEP=()
  if [[ ${#REPORT_DEPS[@]} -gt 0 ]]; then
    DEP=(--dependency=afterany:"$(IFS=:; echo "${REPORT_DEPS[*]}")")
  fi
  submit report \
    --partition="$CPU_PARTITION" --time="$REPORT_TIME" \
    "${DEP[@]}" \
    scripts/m15_report.sbatch "$ENV_FILE" >/dev/null
fi

echo "submitted stage: $STAGE"
echo "results         : $OUT_DIR"
echo "report will be  : $OUT_DIR/M15_CEILING_REPORT.md"
