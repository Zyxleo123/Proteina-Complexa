#!/bin/bash
# Submitter for the replay/geometry-loss ablation training configs. Only calls
# sbatch -- all real work happens in scripts/train_cpsea_replay_ablation.sbatch.
#
# Partition/QOS/node are NOT hardcoded: checked via `myfree` at submission time
# (state changes constantly) rather than baked into the .sbatch file. As of the
# last check, the only free a6000 nodes (a5000 must be avoided -- it silently
# forces self_cond=false, see project memory) were gpu1 and gpu27, both only on
# debug/preempt (general's a6000 nodes were all full). debug caps at 12h, too
# short for a multi-day run, so this defaults to preempt + gpu1. RE-RUN `myfree`
# yourself before submitting -- availability shifts by the minute; override
# NODELIST/PARTITION below if gpu1 is no longer free.
#
# Usage:
#   CONFIG_NAME=example/training_cpsea_replay_noloss_from_v4cfg \
#     RUN_NAME=cpsea_replay_noloss_from_v4cfg \
#     bash scripts/submit_replay_ablation_train.sh
#
#   CONFIG_NAME=example/training_cpsea_geom_noreplay_from_v4cfg \
#     RUN_NAME=cpsea_geom_noreplay_from_v4cfg \
#     bash scripts/submit_replay_ablation_train.sh
#
# Configurable via environment variable (defaults shown):
#   CONFIG_NAME      (required)
#   RUN_NAME         (required)
#   PARTITION        preempt
#   QOS              qos_preempt
#   NODELIST         ''          EMPTY BY DEFAULT -- let Slurm pick. Pinning one node is
#                                how this script produced two separate indefinite pends
#                                (a node full, then a node under a MAINT reservation).
#                                GRES already pins the GPU class, which was the only real
#                                reason to name a node. Set it only to reproduce a run on
#                                specific hardware.
#   GRES             gpu:a6000:1 GPU class, typed. a6000 = 48G; `general` also contains
#                                gpu28, an a5000 (24G), which this excludes automatically.
#                                Do NOT relax to `gpu:1` -- a job sized for an a6000 dies
#                                on a smaller card with no Python traceback.
#   EXCLUDE          ''          extra nodes to avoid. Nodes under a reservation that
#                                starts before TIME elapses are added automatically.
#   TIME             7-00:00:00  (preempt's own max; this is a multi-day run)
#   EXTRA_OVERRIDES  ''          extra Hydra overrides, space-separated, appended
#                                LAST so they win. Chiefly for smoke-testing a new
#                                arm before committing days of GPU to it, e.g.
#                                  EXTRA_OVERRIDES="++opt.limit_train_batches=20 \
#                                    ++opt.max_epochs=1 ++log.log_wandb=false"
#                                Passed through the env FILE, not `sbatch --export`
#                                (which triggers the user-env-retrieval requeue-hold).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${CONFIG_NAME:?Set CONFIG_NAME, e.g. example/training_cpsea_replay_noloss_from_v4cfg}"
: "${RUN_NAME:?Set RUN_NAME}"

PARTITION="${PARTITION:-preempt}"
QOS="${QOS:-qos_preempt}"
NODELIST="${NODELIST:-}"
GRES="${GRES:-gpu:a6000:1}"
EXCLUDE="${EXCLUDE:-}"
TIME="${TIME:-7-00:00:00}"

echo "Re-checking node availability with myfree before submitting..."
myfree || echo "(myfree not available in this shell -- proceeding with configured NODELIST=$NODELIST anyway)"

# Guard: TIME > the partition's MaxTime does NOT fail at submit. The job is accepted and
# then pends forever with reason `PartitionTimeLimit`, which looks exactly like "the
# cluster is busy" and wastes a day before anyone runs `squeue -o %R`. This bites
# whenever PARTITION is overridden away from the default without also lowering TIME:
# preempt allows 7 days, general only 2. Fail loudly at submit instead.
_slurm_time_to_sec() {
  local t="$1"
  case "$t" in
    UNLIMITED|unlimited|INFINITE|infinite) echo 999999999; return;;
  esac
  local days=0 rest="$t"
  if [[ "$rest" == *-* ]]; then days="${rest%%-*}"; rest="${rest#*-}"; fi
  local h=0 m=0 sec=0
  case "$(tr -cd ':' <<< "$rest" | wc -c)" in
    2) IFS=: read -r h m sec <<< "$rest";;
    1) IFS=: read -r m sec <<< "$rest";;
    0) m="$rest";;
  esac
  echo $(( 10#${days:-0} * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${sec:-0} ))
}

PART_MAXTIME="$(scontrol show partition "$PARTITION" 2>/dev/null | tr ' ' '\n' | sed -n 's/^MaxTime=//p' | head -1)"
if [[ -n "$PART_MAXTIME" ]]; then
  if (( $(_slurm_time_to_sec "$TIME") > $(_slurm_time_to_sec "$PART_MAXTIME") )); then
    echo "FATAL: TIME=$TIME exceeds partition '$PARTITION' MaxTime=$PART_MAXTIME."
    echo "       The job would be ACCEPTED and then pend forever (reason: PartitionTimeLimit)."
    echo "       Either set TIME=$PART_MAXTIME (training checkpoints to last.ckpt and resumes),"
    echo "       or submit to a partition whose MaxTime covers it (preempt allows 7-00:00:00)."
    exit 1
  fi
  echo "TIME=$TIME is within $PARTITION MaxTime=$PART_MAXTIME"
fi

# Same class of silent pend: a QOS the partition does not allow.
PART_QOS="$(scontrol show partition "$PARTITION" 2>/dev/null | tr ' ' '\n' | sed -n 's/^AllowQos=//p' | head -1)"
if [[ -n "$PART_QOS" && "$PART_QOS" != "ALL" && ",$PART_QOS," != *",$QOS,"* ]]; then
  echo "FATAL: QOS=$QOS is not in partition '$PARTITION' AllowQos=$PART_QOS."
  echo "       Use QOS=$PART_QOS (general -> qos_general, preempt -> qos_preempt)."
  exit 1
fi

# Guard: a node whose MAINT reservation begins before this job would finish cannot run
# it, and Slurm expresses that as `ReqNodeNotAvail, May be reserved for other job` -- the
# third distinct flavour of "accepted, then pends forever" this script has hit. Find those
# nodes and route around them instead of waiting to discover it in squeue.
DEADLINE=$(( $(date +%s) + $(_slurm_time_to_sec "$TIME") ))
RESERVED_NODES=""
while IFS= read -r line; do
  [[ -n "$line" ]] || continue
  r_nodes="$(sed -n 's/.*[[:space:]]Nodes=\([^[:space:]]*\).*/\1/p' <<< "$line")"
  r_start="$(sed -n 's/.*StartTime=\([^[:space:]]*\).*/\1/p' <<< "$line")"
  [[ -n "$r_nodes" && -n "$r_start" ]] || continue
  r_epoch="$(date -d "${r_start//T/ }" +%s 2>/dev/null)" || continue
  if (( r_epoch < DEADLINE )); then
    RESERVED_NODES+="$(scontrol show hostnames "$r_nodes" 2>/dev/null | tr '\n' ',')"
    echo "Reservation on $r_nodes starts $r_start, before this ${TIME} job would finish."
  fi
done < <(scontrol show reservation --oneliner 2>/dev/null)
RESERVED_NODES="${RESERVED_NODES%,}"

if [[ -n "$RESERVED_NODES" ]]; then
  if [[ -n "$NODELIST" ]] && grep -qE "(^|,)($(tr ',' '|' <<< "$RESERVED_NODES"))(,|$)" <<< "$NODELIST"; then
    echo "FATAL: NODELIST=$NODELIST is under a reservation starting before TIME=$TIME elapses."
    echo "       The job would be ACCEPTED and then pend forever"
    echo "       (reason: ReqNodeNotAvail, May be reserved for other job)."
    echo "       Drop NODELIST (GRES=$GRES already pins the GPU class), or shorten TIME"
    echo "       to finish before the reservation starts."
    exit 1
  fi
  EXCLUDE="${EXCLUDE:+$EXCLUDE,}$RESERVED_NODES"
  echo "Excluding reserved node(s): $RESERVED_NODES"
fi

mkdir -p slurm_logs .slurm_envs
ENV_FILE=".slurm_envs/train_ablation_$(date +%Y%m%d_%H%M%S)_$$.env"
# `printf %q`, NOT a plain heredoc. The batch script SOURCES this file, so an
# unquoted `EXTRA_OVERRIDES=a b c` is parsed as "run command `b` with EXTRA_OVERRIDES=a
# in its environment" -- bash reports `b: command not found`, the source fails, and with
# `set -e` the job dies before step 1 having produced a 102-byte log that says nothing
# about the run. Measured: jobs 35346/35347. %q emits shell-safe quoting for spaces,
# quotes and anything else an override list can contain.
{
  printf 'CONFIG_NAME=%q\n' "$CONFIG_NAME"
  printf 'RUN_NAME=%q\n' "$RUN_NAME"
  printf 'EXTRA_OVERRIDES=%q\n' "${EXTRA_OVERRIDES:-}"
} > "$ENV_FILE"

echo "Wrote $ENV_FILE"
echo "Submitting scripts/train_cpsea_replay_ablation.sbatch $ENV_FILE"
echo "  partition=$PARTITION qos=$QOS time=$TIME gres=$GRES"
echo "  nodelist=${NODELIST:-<any>} exclude=${EXCLUDE:-<none>}"
echo "  extra_overrides=${EXTRA_OVERRIDES:-<none>}"

SBATCH_ARGS=(
  --partition="$PARTITION"
  --qos="$QOS"
  --time="$TIME"
  --gres="$GRES"
)
# Both optional: an empty --nodelist/--exclude is a submission error, not a no-op.
[[ -n "$NODELIST" ]] && SBATCH_ARGS+=( --nodelist="$NODELIST" )
[[ -n "$EXCLUDE" ]] && SBATCH_ARGS+=( --exclude="$EXCLUDE" )

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN set -- would submit:"
  echo "  sbatch ${SBATCH_ARGS[*]} scripts/train_cpsea_replay_ablation.sbatch $ENV_FILE"
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" scripts/train_cpsea_replay_ablation.sbatch "$ENV_FILE"
