#!/bin/bash
# Submitter for the val_generation nsteps-fidelity test. Only calls sbatch -- all real work
# is in scripts/eval_val_gen_nsteps.sbatch (GPU, one job per arm, siblings so they queue
# concurrently) and scripts/summarize_val_gen_nsteps.sbatch (CPU, --dependency=afterok on
# all three).
#
# THE QUESTION. Arm B (rollout_finetune / DRaFT-K) moved its own sampled closure
# 0.281 -> 0.500 over 589 rollout triggers and still came out within noise of v4 on every
# val_gen linkage type. Its rollouts train at nsteps=100; val_gen scores at nsteps=400. If
# the skill it learned is fidelity-specific, the null is the instrument's fault. This
# re-scores the FINISHED weights of three arms at both step counts -- no training, and the
# arms share a noise draw at each (nsteps, seed, batch), so the comparison is paired.
#
# nsteps=400 is the CONTROL: it must reproduce each arm's training-logged val_gen closure.
# If it does not, stop -- something other than nsteps moved, and no row in the table means
# anything yet.
#
# Usage:
#   bash scripts/submit_val_gen_nsteps.sh                # all three arms + summary
#   ARMS="cpsea_rollout_draft_from_v4cfg" bash scripts/submit_val_gen_nsteps.sh
#   DRY_RUN=1 bash scripts/submit_val_gen_nsteps.sh      # print, submit nothing
#
# Configurable via environment variable (defaults shown):
#   ARMS        the three ablation arms   space-separated store/ dir names
#   NSTEPS      100,400                   ODE step counts to score
#   SEEDS       0,1,2                     sampling seeds (independent noise draws)
#   N_BATCHES   8                         val batches per condition (training used 2)
#   OUT_DIR     evaluation_results/val_gen_nsteps   one rows_<arm>.jsonl per arm; resumable
#   PARTITION   general    a6000 lives here; `myfree` is re-run below, read it
#   QOS         qos_general
#   GRES        gpu:a6000:1   typed on purpose. a5000 (24G) silently forces self_cond=false,
#                             and a smaller card dies with no Python traceback.
#   TIME        08:00:00      resumable -- if it walls, resubmit the same command
#   CPU_PARTITION / CPU_QOS / CPU_TIME   cpu / qos_cpu / 00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

ARMS="${ARMS:-cpsea_cyc_ringpe_bond_v4 cpsea_rollout_draft_from_v4cfg cpsea_bondunroll_from_v4cfg}"
NSTEPS="${NSTEPS:-100,400}"
SEEDS="${SEEDS:-0,1,2}"
N_BATCHES="${N_BATCHES:-8}"
OUT_DIR="${OUT_DIR:-evaluation_results/val_gen_nsteps}"
BASELINE_LABEL="${BASELINE_LABEL:-cpsea_cyc_ringpe_bond_v4}"

PARTITION="${PARTITION:-general}"
QOS="${QOS:-qos_general}"
GRES="${GRES:-gpu:a6000:1}"
TIME="${TIME:-08:00:00}"
EXCLUDE="${EXCLUDE:-}"
CPU_PARTITION="${CPU_PARTITION:-cpu}"
CPU_QOS="${CPU_QOS:-qos_cpu}"
CPU_TIME="${CPU_TIME:-00:30:00}"

echo "Re-checking node availability with myfree before submitting..."
myfree || echo "(myfree not available in this shell -- proceeding with PARTITION=$PARTITION)"

# Slurm ACCEPTS a job whose TIME exceeds the partition MaxTime and then pends it forever
# with reason PartitionTimeLimit, which looks exactly like a busy cluster. Fail at submit.
_t2s() {
  local t="$1" days=0 rest h=0 m=0 sec=0
  case "$t" in UNLIMITED|INFINITE) echo 999999999; return;; esac
  rest="$t"
  [[ "$rest" == *-* ]] && { days="${rest%%-*}"; rest="${rest#*-}"; }
  case "$(tr -cd ':' <<< "$rest" | wc -c)" in
    2) IFS=: read -r h m sec <<< "$rest";;
    1) IFS=: read -r m sec <<< "$rest";;
    0) m="$rest";;
  esac
  echo $(( 10#${days:-0}*86400 + 10#${h:-0}*3600 + 10#${m:-0}*60 + 10#${sec:-0} ))
}
_check_partition() {
  local part="$1" qos="$2" tlimit="$3"
  local maxt allowq
  maxt="$(scontrol show partition "$part" 2>/dev/null | tr ' ' '\n' | sed -n 's/^MaxTime=//p' | head -1)"
  if [[ -n "$maxt" ]] && (( $(_t2s "$tlimit") > $(_t2s "$maxt") )); then
    echo "FATAL: TIME=$tlimit exceeds partition '$part' MaxTime=$maxt (job would pend forever)."; exit 1
  fi
  allowq="$(scontrol show partition "$part" 2>/dev/null | tr ' ' '\n' | sed -n 's/^AllowQos=//p' | head -1)"
  if [[ -n "$allowq" && "$allowq" != "ALL" && ",$allowq," != *",$qos,"* ]]; then
    echo "FATAL: QOS=$qos not in partition '$part' AllowQos=$allowq."; exit 1
  fi
  echo "  $part: TIME=$tlimit within MaxTime=$maxt, QOS=$qos allowed"
}
_check_partition "$PARTITION" "$QOS" "$TIME"
_check_partition "$CPU_PARTITION" "$CPU_QOS" "$CPU_TIME"

mkdir -p slurm_logs .slurm_envs "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
GPU_JOB_IDS=()

for arm in $ARMS; do
  RUN_DIR="store/$arm"
  if [[ ! -d "$RUN_DIR/checkpoints" ]]; then
    echo "SKIP $arm: no $RUN_DIR/checkpoints -- that arm never trained here."
    continue
  fi
  ENV_FILE=".slurm_envs/valgen_nsteps_${arm}_${STAMP}_$$.env"
  # printf %q, not a plain heredoc: the batch script SOURCES this, and an unquoted value
  # containing a space is parsed as a command invocation. Measured: jobs 35346/35347, which
  # died before step 1 having logged 102 bytes saying nothing.
  {
    printf 'RUN_DIR=%q\n'        "$RUN_DIR"
    printf 'EVAL_LABEL=%q\n'     "$arm"
    # Always written, even when empty: the batch script's fallback must come from THIS file,
    # never from whatever .env happens to export. An unnamespaced CKPT_PATH is exactly how
    # job 36051 scored nothing.
    printf 'EVAL_CKPT_PATH=%q\n' "${EVAL_CKPT_PATH:-}"
    printf 'NSTEPS=%q\n'    "$NSTEPS"
    printf 'SEEDS=%q\n'     "$SEEDS"
    printf 'N_BATCHES=%q\n' "$N_BATCHES"
    # A DIRECTORY, not a file. The scoring job names its own rows_<arm>_<jobid>.jsonl inside
    # it, so no two processes can ever share a path. Convention was not enough twice over:
    # first three arms shared one file (88 of 144 rows destroyed by non-atomic concurrent
    # O_APPEND), then this submitter was run twice and gave each arm two writers anyway.
    printf 'EVAL_OUT_DIR=%q\n'   "$OUT_DIR"
  } > "$ENV_FILE"

  SB=( --partition="$PARTITION" --qos="$QOS" --time="$TIME" --gres="$GRES"
       --job-name="valgen_${arm}" )
  [[ -n "$EXCLUDE" ]] && SB+=( --exclude="$EXCLUDE" )

  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY_RUN: sbatch ${SB[*]} scripts/eval_val_gen_nsteps.sbatch $ENV_FILE"
    continue
  fi
  # All three are SIBLINGS -- no dependency between them, so they queue concurrently and
  # the wall clock is one arm's, not three.
  jid="$(sbatch --parsable "${SB[@]}" scripts/eval_val_gen_nsteps.sbatch "$ENV_FILE")"
  echo "submitted $arm -> job $jid"
  GPU_JOB_IDS+=( "$jid" )
done

SUM_ENV=".slurm_envs/valgen_summary_${STAMP}_$$.env"
{
  printf 'OUT_DIR=%q\n'        "$OUT_DIR"
  printf 'BASELINE_LABEL=%q\n' "$BASELINE_LABEL"
} > "$SUM_ENV"

SUM_SB=( --partition="$CPU_PARTITION" --qos="$CPU_QOS" --time="$CPU_TIME"
         --job-name=valgen_summary )
if (( ${#GPU_JOB_IDS[@]} )); then
  SUM_SB+=( --dependency="afterok:$(IFS=:; echo "${GPU_JOB_IDS[*]}")" )
fi

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: sbatch ${SUM_SB[*]} scripts/summarize_val_gen_nsteps.sbatch $SUM_ENV"
  exit 0
fi
sum_jid="$(sbatch --parsable "${SUM_SB[@]}" scripts/summarize_val_gen_nsteps.sbatch "$SUM_ENV")"
echo "submitted summary -> job $sum_jid (afterok on ${GPU_JOB_IDS[*]:-<none>})"
echo
echo "Rows accumulate as $OUT_DIR/rows_<arm>.jsonl. Re-aggregate any time, no GPU needed:"
echo "  .venv/bin/python script_utils/summarize_val_gen_nsteps.py --rows $OUT_DIR"
