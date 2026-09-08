#!/bin/bash
# Submitter for the LP->CP test-set val_generation eval. Only calls sbatch -- all real work is in
# the three stage scripts, chained with --dependency=afterok:
#
#   Stage 1  lp2cp_testeval_gpu.sbatch        GPU   sample val_gen on 1000 stratified TEST complexes,
#                                                   dump every complex for Rosetta
#   Stage 2  lp2cp_testeval_rosetta.sbatch    CPU   array of shards; score interface dG on the dump
#   Stage 3  lp2cp_testeval_aggregate.sbatch  CPU   merge closure + dG -> summary.json + summary.png
#
# Everything is resumable: re-running this exact command skips batches/complexes already on disk.
# Config travels to each job in a per-submission env file passed as a POSITIONAL arg (never
# `sbatch --export`, which triggers the user-env-retrieval requeue-hold).
#
# Usage:
#   bash scripts/submit_lp2cp_testeval.sh                 # full pipeline, defaults below
#   DRY_RUN=1 bash scripts/submit_lp2cp_testeval.sh       # print sbatch lines, submit nothing
#   N_COMPLEXES=64 EVAL_NSTEPS=40 SHARD_COUNT=4 bash scripts/submit_lp2cp_testeval.sh   # smoke
#
# Configurable via environment variable (defaults shown):
#   RUN_DIR       store/cpsea_bondunroll_pin20260828   the LP->CP flow ckpt (SDEdit pins this one)
#   EVAL_CKPT_PATH  ''  (=> <RUN_DIR>/checkpoints/last-EMA.ckpt; EMA is what val_gen logged)
#   TEST_PARQUET  $CPSEA test metadata (cpsea_test.parquet) -- resolved below
#   N_COMPLEXES   1000        distinct stratified, cluster-unique test complexes
#   N_REPEAT      4           samples per complex (val_generation default)
#   EVAL_NSTEPS   ''          ODE steps; empty => inherit design sampler's 400 (same as val_gen)
#   SEED          0
#   OUT_DIR       evaluation_results/lp2cp_testeval/<run>   stable per-run so resubmits resume
#   SHARD_COUNT   16          Rosetta CPU array width
#   GPU_PARTITION/GPU_QOS/GPU_GRES/GPU_TIME    general / qos_general / gpu:a6000:1 / 08:00:00
#   CPU_PARTITION/CPU_QOS/CPU_TIME             preempt / qos_preempt / 08:00:00   (Rosetta shards)
#   AGG_PARTITION/AGG_QOS/AGG_TIME             cpu / qos_cpu / 00:20:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

RUN_DIR="${RUN_DIR:-store/cpsea_bondunroll_pin20260828}"
RUN_BASE="$(basename "$RUN_DIR")"
EVAL_CKPT_PATH="${EVAL_CKPT_PATH:-}"

# Resolve the test parquet from .env's CPSea root if not given. Falls back to the known absolute path.
_DEFAULT_TEST="/zfsauton/scratch/yixiz/CPSea/CPSea_full/CPSea/preprocessed/metadata/cpsea_test.parquet"
TEST_PARQUET="${TEST_PARQUET:-$_DEFAULT_TEST}"

N_COMPLEXES="${N_COMPLEXES:-1000}"
N_REPEAT="${N_REPEAT:-4}"
EVAL_NSTEPS="${EVAL_NSTEPS:-}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-evaluation_results/lp2cp_testeval/$RUN_BASE}"
SHARD_COUNT="${SHARD_COUNT:-16}"
ROSETTA_DUMP_MAX="${ROSETTA_DUMP_MAX:-128}"

GPU_PARTITION="${GPU_PARTITION:-general}"
GPU_QOS="${GPU_QOS:-qos_general}"
GPU_GRES="${GPU_GRES:-gpu:a6000:1}"   # a6000 typed: a5000 silently forces self_cond=false; smaller cards die
GPU_TIME="${GPU_TIME:-08:00:00}"
GPU_EXCLUDE="${GPU_EXCLUDE:-}"

CPU_PARTITION="${CPU_PARTITION:-preempt}"   # account 'users' lacks qos_general; preempt is CPU-only here
CPU_QOS="${CPU_QOS:-qos_preempt}"
CPU_TIME="${CPU_TIME:-08:00:00}"

AGG_PARTITION="${AGG_PARTITION:-cpu}"
AGG_QOS="${AGG_QOS:-qos_cpu}"
AGG_TIME="${AGG_TIME:-00:20:00}"

DUMP_DIR="$OUT_DIR/rosetta_dump"
ROSETTA_OUT_DIR="$OUT_DIR/rosetta_dg"

echo "Re-checking node availability with myfree before submitting..."
myfree || echo "(myfree not available in this shell -- proceeding with GPU_PARTITION=$GPU_PARTITION)"

# Slurm ACCEPTS a job whose TIME exceeds the partition MaxTime and then pends it forever
# (reason PartitionTimeLimit), which looks like a busy cluster. Fail loudly at submit instead.
_t2s() {
  local t="$1" days=0 rest h=0 m=0 sec=0
  case "$t" in UNLIMITED|INFINITE) echo 999999999; return;; esac
  rest="$t"; [[ "$rest" == *-* ]] && { days="${rest%%-*}"; rest="${rest#*-}"; }
  case "$(tr -cd ':' <<< "$rest" | wc -c)" in
    2) IFS=: read -r h m sec <<< "$rest";; 1) IFS=: read -r m sec <<< "$rest";; 0) m="$rest";;
  esac
  echo $(( 10#${days:-0}*86400 + 10#${h:-0}*3600 + 10#${m:-0}*60 + 10#${sec:-0} ))
}
_check_partition() {
  local part="$1" qos="$2" tlimit="$3" maxt allowq
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
_check_partition "$GPU_PARTITION" "$GPU_QOS" "$GPU_TIME"
_check_partition "$CPU_PARTITION" "$CPU_QOS" "$CPU_TIME"
_check_partition "$AGG_PARTITION" "$AGG_QOS" "$AGG_TIME"

mkdir -p slurm_logs .slurm_envs "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ---- Stage 1 env (GPU) -------------------------------------------------------------------------
GPU_ENV=".slurm_envs/lp2cp_testeval_gpu_${STAMP}_$$.env"
{
  printf 'EVAL_RUN_DIR=%q\n'          "$RUN_DIR"
  printf 'EVAL_CKPT_PATH=%q\n'        "$EVAL_CKPT_PATH"
  printf 'EVAL_TEST_PARQUET=%q\n'     "$TEST_PARQUET"
  printf 'EVAL_N_COMPLEXES=%q\n'      "$N_COMPLEXES"
  printf 'EVAL_N_REPEAT=%q\n'         "$N_REPEAT"
  printf 'EVAL_NSTEPS=%q\n'           "$EVAL_NSTEPS"
  printf 'EVAL_SEED=%q\n'             "$SEED"
  printf 'EVAL_OUT_DIR=%q\n'          "$OUT_DIR"
  printf 'EVAL_ROSETTA_DUMP_MAX=%q\n' "$ROSETTA_DUMP_MAX"
} > "$GPU_ENV"

# ---- Stage 2 env (Rosetta array) ---------------------------------------------------------------
ROS_ENV=".slurm_envs/lp2cp_testeval_rosetta_${STAMP}_$$.env"
{
  printf 'EVAL_DUMP_DIR=%q\n'        "$DUMP_DIR"
  printf 'EVAL_ROSETTA_OUT_DIR=%q\n' "$ROSETTA_OUT_DIR"
  printf 'EVAL_SHARD_COUNT=%q\n'     "$SHARD_COUNT"
} > "$ROS_ENV"

# ---- Stage 3 env (aggregate) -------------------------------------------------------------------
AGG_ENV=".slurm_envs/lp2cp_testeval_aggregate_${STAMP}_$$.env"
{
  printf 'EVAL_OUT_DIR=%q\n'         "$OUT_DIR"
  printf 'EVAL_ROSETTA_SUBDIR=%q\n'  "rosetta_dg"
} > "$AGG_ENV"

GPU_SB=( --partition="$GPU_PARTITION" --qos="$GPU_QOS" --time="$GPU_TIME" --gres="$GPU_GRES"
         --job-name="lp2cp_testeval_gpu_${RUN_BASE}" )
[[ -n "$GPU_EXCLUDE" ]] && GPU_SB+=( --exclude="$GPU_EXCLUDE" )
ROS_SB=( --partition="$CPU_PARTITION" --qos="$CPU_QOS" --time="$CPU_TIME"
         --array="0-$((SHARD_COUNT-1))%${SHARD_COUNT}" --job-name="lp2cp_testeval_rosetta_${RUN_BASE}" )
AGG_SB=( --partition="$AGG_PARTITION" --qos="$AGG_QOS" --time="$AGG_TIME"
         --job-name="lp2cp_testeval_agg_${RUN_BASE}" )

echo
echo "Plan:"
echo "  ckpt:        ${EVAL_CKPT_PATH:-$RUN_DIR/checkpoints/last-EMA.ckpt}"
echo "  test:        $TEST_PARQUET  (n_complexes=$N_COMPLEXES, n_repeat=$N_REPEAT, nsteps=${EVAL_NSTEPS:-inherit-400})"
echo "  out:         $OUT_DIR"
echo "  rosetta:     $SHARD_COUNT shards -> $ROSETTA_OUT_DIR"

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN -- would submit:"
  echo "  sbatch ${GPU_SB[*]} scripts/lp2cp_testeval_gpu.sbatch $GPU_ENV"
  echo "  sbatch --dependency=afterok:<gpu> ${ROS_SB[*]} scripts/lp2cp_testeval_rosetta.sbatch $ROS_ENV"
  echo "  sbatch --dependency=afterok:<ros> ${AGG_SB[*]} scripts/lp2cp_testeval_aggregate.sbatch $AGG_ENV"
  exit 0
fi

# Scrub SLURM_* so a submit from inside an allocation does not trigger user-env retrieval.
GPU_JID="$(env $(compgen -v | grep '^SLURM_' | sed 's/^/-u /') \
  sbatch --parsable "${GPU_SB[@]}" scripts/lp2cp_testeval_gpu.sbatch "$GPU_ENV")"
echo "submitted GPU stage -> job $GPU_JID"

ROS_JID="$(env $(compgen -v | grep '^SLURM_' | sed 's/^/-u /') \
  sbatch --parsable --dependency="afterok:$GPU_JID" "${ROS_SB[@]}" \
  scripts/lp2cp_testeval_rosetta.sbatch "$ROS_ENV")"
echo "submitted Rosetta array -> job $ROS_JID (afterok:$GPU_JID)"

AGG_JID="$(env $(compgen -v | grep '^SLURM_' | sed 's/^/-u /') \
  sbatch --parsable --dependency="afterok:$ROS_JID" "${AGG_SB[@]}" \
  scripts/lp2cp_testeval_aggregate.sbatch "$AGG_ENV")"
echo "submitted aggregate -> job $AGG_JID (afterok:$ROS_JID)"
echo
echo "Results land in $OUT_DIR/summary.json and $OUT_DIR/summary.png"
echo "Re-aggregate any time (no GPU): .venv/bin/python script_utils/aggregate_lp2cp_testeval.py --out-dir $OUT_DIR"
