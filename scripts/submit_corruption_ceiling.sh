#!/bin/bash
# Submitter for the corruption-ceiling experiment. Calls sbatch and nothing else.
#
#   bash scripts/submit_corruption_ceiling.sh --submit <stage> [options]
#
# The question: how far can a CPSea complex's termini be pulled apart before the interface
# stops being worth training on? It gates the synthetic-decyclization route, because the
# measured pre-cyclization gap is 6.95 A median (max 10.71 A over n=400 AFDB parents) with
# 0.76 A median closure travel -- so CPSea holds no large LP->CP move, and corruption is
# the only possible source of transport signal. Inference needs 16.3 A (LNR) to 21.6 A
# (PepBench). If the ceiling lands below ~16 A there is no real-data fallback.
#
# Stages:
#   ladder     CPU   frozen-receptor pull of the terminal CA pair up a ladder (array)
#   rosetta    CPU   interface dG for the dumped complexes (array, sidecar)
#   report     CPU   tables + figures + the verdict, from whatever is on disk
#   render     CPU   the CP -> LP pull as a filmstrip, from the dumped complexes (PyMOL)
#   all        the whole DAG
#
# Shape of the DAG:
#
#   ladder(array) ──afterok──> rosetta(array) ──afterany──┬──> report
#                                                        └──> render
#
# `report` depends with afterANY: a stage that gated itself out (no input, no PyRosetta)
# exits 0 with an explanation, and the report's job is to say which stages contributed.
# afterok there would strand the report on a deliberate gate.
#
# Everything is CPU-only. Asking for a GPU to minimize geometry just queues behind real work.
#
# Re-running the report alone is safe and cheap -- pass the SAME --run-id, since every path
# hangs off it:
#   bash scripts/submit_corruption_ceiling.sh --submit report --run-id cc_20260927_...

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGE=""; DRYRUN=0; SMOKE=0
CPU_PARTITION="cpu"
LADDER_TIME="04:00:00"; ROSETTA_TIME="08:00:00"; REPORT_TIME="01:00:00"
RENDER_TIME="02:00:00"
LADDER_SHARDS=6; ROSETTA_SHARDS=8
RUN_ID=""
declare -a EXTRA_ENV=()

usage() { sed -n '2,33p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)          STAGE="$2";          shift 2 ;;
    --cpu-partition)   CPU_PARTITION="$2";  shift 2 ;;
    --run-id)          RUN_ID="$2";         shift 2 ;;
    --ladder-shards)   LADDER_SHARDS="$2";  shift 2 ;;
    --rosetta-shards)  ROSETTA_SHARDS="$2"; shift 2 ;;
    --ladder-time)     LADDER_TIME="$2";    shift 2 ;;
    --rosetta-time)    ROSETTA_TIME="$2";   shift 2 ;;
    --render-time)     RENDER_TIME="$2";    shift 2 ;;
    # Anything VAR=value lands in the env file, so every knob in _corruption_preamble.sh
    # is settable without editing a script. e.g. --set K_CONTACT=500 for the held arm.
    --set)             EXTRA_ENV+=("$2");   shift 2 ;;
    --smoke)           SMOKE=1;             shift ;;
    --dry-run)         DRYRUN=1;            shift ;;
    -h|--help)         usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 >&2 ;;
  esac
done
[[ -n "$STAGE" ]] || { echo "FATAL: --submit <stage> is required" >&2; usage 1 >&2; }
case "$STAGE" in
  ladder|rosetta|report|render|all) ;;
  *) echo "FATAL: unknown stage '$STAGE'" >&2; usage 1 >&2 ;;
esac

# Submitting from inside an allocation leaks SLURM_* into the child sbatch, which then
# inherits an array index or a job id that is not its own.
for v in $(env | sed -n 's/^\(SLURM_[A-Z_]*\)=.*/\1/p'); do unset "$v" || true; done

[[ -n "$RUN_ID" ]] || RUN_ID="cc_$( [[ $SMOKE -eq 1 ]] && echo smoke_ )$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO}/evaluation_results/${RUN_ID}"
ENV_DIR="${REPO}/evaluation_results/_cc_env"
mkdir -p "$OUT_DIR" "$ENV_DIR" "${REPO}/logs"

if [[ $SMOKE -eq 1 ]]; then LADDER_SHARDS=1; ROSETTA_SHARDS=1; fi

# ONE env file per submission, passed as a POSITIONAL argument. Arguments travel through
# the job record untouched; `sbatch --export` does not -- any explicit export list sets
# SLURM_GET_USER_ENV=1, slurmd fails to rebuild the login environment on the compute node,
# and the job is requeued and HELD. A file also handles spaces, so LADDER_A needs no
# comma encoding.
ENV_FILE="${ENV_DIR}/${RUN_ID}.env"
{
  echo "# written by submit_corruption_ceiling.sh at $(date -Is)"
  echo "REPO=${REPO}"
  echo "RUN_ID=${RUN_ID}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "SMOKE=${SMOKE}"
  for kv in "${EXTRA_ENV[@]}"; do echo "$kv"; done
} > "$ENV_FILE"
echo "env file: $ENV_FILE"
echo "run dir : $OUT_DIR"

submit() {  # submit <description> <sbatch args...>
  local desc="$1"; shift
  if [[ $DRYRUN -eq 1 ]]; then
    # To stderr: stdout is the job id this function returns to its caller.
    { printf 'DRY-RUN %-10s sbatch' "$desc"; printf ' %q' "$@"; printf '\n'; } >&2
    echo "999999"
    return
  fi
  local out; out="$(sbatch "$@")"
  echo "$desc: $out" >&2
  awk '{print $NF}' <<<"$out"
}

LADDER_JID=""; ROSETTA_JID=""

if [[ "$STAGE" == "ladder" || "$STAGE" == "all" ]]; then
  LADDER_JID="$(submit ladder \
    --partition="$CPU_PARTITION" --time="$LADDER_TIME" \
    --array="0-$((LADDER_SHARDS - 1))" \
    scripts/corruption_ladder.sbatch "$ENV_FILE" "NUM_SHARDS=$LADDER_SHARDS")"
fi

if [[ "$STAGE" == "rosetta" || "$STAGE" == "all" ]]; then
  DEP=(); [[ -n "$LADDER_JID" ]] && DEP=(--dependency=afterok:"$LADDER_JID")
  ROSETTA_JID="$(submit rosetta \
    --partition="$CPU_PARTITION" --time="$ROSETTA_TIME" \
    --array="0-$((ROSETTA_SHARDS - 1))" "${DEP[@]}" \
    scripts/corruption_rosetta.sbatch "$ENV_FILE" "NUM_SHARDS=$ROSETTA_SHARDS")"
fi

if [[ "$STAGE" == "report" || "$STAGE" == "all" ]]; then
  DEP=()
  if [[ -n "$ROSETTA_JID" ]]; then
    DEP=(--dependency=afterany:"$ROSETTA_JID")
  elif [[ -n "$LADDER_JID" ]]; then
    DEP=(--dependency=afterany:"$LADDER_JID")
  fi
  submit report \
    --partition="$CPU_PARTITION" --time="$REPORT_TIME" "${DEP[@]}" \
    scripts/corruption_report.sbatch "$ENV_FILE" >/dev/null
fi

if [[ "$STAGE" == "render" || "$STAGE" == "all" ]]; then
  # A SIBLING of report, not behind it: both read the same files on disk, so they queue
  # concurrently rather than the pictures waiting on the tables.
  DEP=()
  if [[ -n "$ROSETTA_JID" ]]; then
    DEP=(--dependency=afterany:"$ROSETTA_JID")
  elif [[ -n "$LADDER_JID" ]]; then
    DEP=(--dependency=afterany:"$LADDER_JID")
  fi
  submit render \
    --partition="$CPU_PARTITION" --time="$RENDER_TIME" "${DEP[@]}" \
    scripts/corruption_render.sbatch "$ENV_FILE" >/dev/null
fi

echo
echo "watch:   squeue -u \$USER -o '%.10i %.14j %.8T %.10M %R'"
echo "verdict: cat ${OUT_DIR}/report/verdict.json"
echo "figures: ${OUT_DIR}/render/filmstrip_combined.png"
