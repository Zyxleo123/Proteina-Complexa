#!/usr/bin/env bash
# Queue one evolutionary-search pilot: a few complexes from one corpus, the LLM arm run
# as generations of elites and offspring, and the matched random control at the same
# budget in the same run directory.
#
# Why one corpus per invocation
# -----------------------------
# `ladder-cohort` refuses to mix corpora that do not share one frozen geometry gate, and
# tranche1 (LNR) and pepbench_valid do not. That is the right refusal -- rows judged by
# two different gates are not comparable -- and it costs nothing here, because every
# comparison this pilot makes is WITHIN a complex: the LLM arm against the random
# control against the unmutated reference, all folded by the same gate.
#
# The shape of a run
# ------------------
#   ladder LIMIT=8        fold the reference first, so generation 1 can see its
#                         interface instead of proposing blind. The draws it folds on
#                         the way are the control's own, so nothing is spent twice.
#   propose  --round k    the elites are read from the SELECTION half of the folds paid
#                         for so far, and each one gets its own prompt asking for
#                         offspring of it. SHARDED, because this stage is serial in the
#                         complex count -- measured 1.0 min per complex per generation,
#                         which is 0.3 h for this pilot but 7.6 h at 160 complexes and
#                         102 h at 2,140, all of it on the critical path. The model load
#                         is 12 s warm, so a shard costs almost nothing to start.
#   ladder   LLM_ROUND=k  one array task per complex: folds rounds 1..k, skipping what
#                         the journal already has. This is what makes a requeue cheap.
#   ... repeated GENERATIONS times, each propose waiting on the whole previous array.
#   ladder LLM_ROUND=0    the random control, at the cohort's full draw count. Submitted
#                         without a dependency on the chain: it shares the reference
#                         folds and is otherwise independent, so it backfills alongside.
#
# The control is NOT optional and NOT smaller. The measured seed-to-seed spread on this
# cohort is 4.3 REU median and up to 22 REU, larger than most effects being chased, so
# an arm folded more times than its baseline would show an advantage made of folds.
#
# Usage:
#   scripts/cpseq_launch_es_pilot.sh OUT_DIR=/data/user_data/yixiz/cpseq/es_pilot_lnr \
#       SHARDS=3 GENERATIONS=3 ELITES=4 OFFSPRING=3
set -euo pipefail

REPO="${REPO:-/home/yixiz/Proteina-Complexa}"
cd "$REPO"

for kv in "$@"; do
  [[ "$kv" == *=* ]] || { echo "bad argument (want VAR=value): $kv" >&2; exit 2; }
  export "${kv?}"
done

OUT_DIR="${OUT_DIR:?set OUT_DIR=<run dir holding cohort.json>}"
SHARDS="${SHARDS:?set SHARDS=<number of complexes>}"
GENERATIONS="${GENERATIONS:-3}"
ELITES="${ELITES:-4}"
OFFSPRING="${OFFSPRING:-3}"
# `full` because the plan keeps the relaxed complexes: buried unsatisfied polars, shape
# complementarity, buried surface area and exposed hydrophobicity are all computed from
# them afterwards, and the other tiers delete them as derived products. At pilot size
# (~1.3 MB per fold, a few thousand folds) that is a few GB, which is the right trade.
RETENTION="${RETENTION:-full}"
# Split each folding stage into a GPU predict pass and a CPU score pass. On by default:
# measured 14.3 s of prediction against 43.7 s of relax per fold, so an unsplit stage
# bills three quarters of its "GPU" time to Rosetta with the card idle. It changes no
# result -- both passes key the same caches identically. SPLIT=0 restores the single
# combined GPU job, which is simpler to debug on a small run.
SPLIT="${SPLIT:-1}"
# How many array tasks may RUN at once (the %N on --array). Capped because every array
# task counts individually against the per-user QOS limits -- 24 jobs on preempt_qos
# (GPU) and 20 on preempt_cpu_qos -- so asking for more does not finish sooner, it only
# fills the queue with our own pending tasks and blocks anything else we want to submit.
# Measured the hard way: the PepBench chain's first stage sat on
# QOSMaxJobsPerUserLimit because the 24 GPU slots were already ours.
MAX_CONCURRENT_GPU="${MAX_CONCURRENT_GPU:-12}"
# 20 is the per-user job cap on preempt_cpu_qos, and each score job takes 3 CPUs, so
# this saturates the 64-CPU allowance at 60. It is what the four-day corpus budget was
# sized against; lower it when something else of yours needs CPU slots at the same time.
MAX_CONCURRENT_CPU="${MAX_CONCURRENT_CPU:-20}"
# Propose-stage shards. Default: one shard per ~8 complexes, at least 1, capped at the
# number of complexes -- a shard per complex would pay the model load more often than it
# saves. Override explicitly for a big cohort.
PROPOSE_SHARDS="${PROPOSE_SHARDS:-0}"
if [[ "$PROPOSE_SHARDS" == "0" ]]; then
  PROPOSE_SHARDS=$(( (SHARDS + 7) / 8 ))
  [[ "$PROPOSE_SHARDS" -lt 1 ]] && PROPOSE_SHARDS=1
fi
DRY="${DRY:-0}"

[[ -f "$OUT_DIR/cohort.json" ]] || { echo "no cohort at $OUT_DIR/cohort.json" >&2; exit 2; }

# "0-<last>%<at most N at a time>", so a 160-complex run queues 160 tasks but only ever
# holds the cap's worth of slots.
arr() {  # arr <n_tasks> <cap>
  local n="$1" cap="$2"
  local c=$(( n < cap ? n : cap ))
  echo "0-$((n - 1))%${c}"
}

sub() {  # echo the command, submit it, return the job id
  echo "    + sbatch $*" >&2
  if [[ "$DRY" == "1" ]]; then echo "DRY$RANDOM"; return; fi
  sbatch --parsable "$@"
}

last_array=""

# One folding stage. Split: GPU predicts, then a CPU array scores what it predicted.
# Echoes the job id the next stage must wait on.
fold_stage() {
  local name="$1" dep="$2" llm_round="$3"
  if [[ "$SPLIT" == "1" ]]; then
    local pj sj
    pj=$(sub --job-name="${name}_predict" --dependency="afterok:$dep" \
      --array="$(arr "$SHARDS" "$MAX_CONCURRENT_GPU")" scripts/cpseq_ladder.sbatch \
      OUT_DIR="$OUT_DIR" SHARD_COUNT="$SHARDS" LLM_ROUND="$llm_round" \
      RETENTION="$RETENTION" PREDICT_ONLY=1)
    sj=$(sub --job-name="${name}_score" --dependency="afterok:$pj" \
      --array="$(arr "$SHARDS" "$MAX_CONCURRENT_CPU")" scripts/cpseq_ladder_score.sbatch \
      OUT_DIR="$OUT_DIR" SHARD_COUNT="$SHARDS" LLM_ROUND="$llm_round" \
      RETENTION="$RETENTION")
    echo "$pj $sj"
  else
    local j
    j=$(sub --job-name="$name" --dependency="afterok:$dep" \
      --array="$(arr "$SHARDS" "$MAX_CONCURRENT_GPU")" scripts/cpseq_ladder.sbatch \
      OUT_DIR="$OUT_DIR" SHARD_COUNT="$SHARDS" LLM_ROUND="$llm_round" \
      RETENTION="$RETENTION")
    echo "- $j"
  fi
}

echo "[es-pilot] $OUT_DIR"
echo "[es-pilot] $SHARDS complex(es), $GENERATIONS generation(s) of "\
"$ELITES x $OFFSPRING = $((ELITES * OFFSPRING))/complex, retention=$RETENTION, "\
"propose shards=$PROPOSE_SHARDS"

# Step 0: the reference, and the first few control draws.
seed_job=$(sub --job-name=es_ref --array="$(arr "$SHARDS" "$MAX_CONCURRENT_GPU")" \
  scripts/cpseq_ladder.sbatch \
  OUT_DIR="$OUT_DIR" SHARD_COUNT="$SHARDS" LIMIT=8 RETENTION="$RETENTION")
echo "[es-pilot] reference folds: $seed_job"
last_array="$seed_job"

for g in $(seq 1 "$GENERATIONS"); do
  if [[ "$PROPOSE_SHARDS" -gt 1 ]]; then
    p=$(sub --job-name="es_propose_g$g" --dependency="afterok:$last_array" \
      --array="$(arr "$PROPOSE_SHARDS" "$MAX_CONCURRENT_GPU")" \
      scripts/cpseq_llm_propose.sbatch \
      OUT_DIR="$OUT_DIR" ROUND="$g" ELITES="$ELITES" OFFSPRING_PER_ELITE="$OFFSPRING" \
      SHARD_COUNT="$PROPOSE_SHARDS" FEEDBACK_FOLDS=select)
  else
    p=$(sub --job-name="es_propose_g$g" --dependency="afterok:$last_array" \
      scripts/cpseq_llm_propose.sbatch \
      OUT_DIR="$OUT_DIR" ROUND="$g" ELITES="$ELITES" OFFSPRING_PER_ELITE="$OFFSPRING" \
      SHARD_COUNT=1 FEEDBACK_FOLDS=select)
  fi
  read -r fp fs <<<"$(fold_stage "es_fold_g$g" "$p" "$g")"
  echo "[es-pilot] generation $g: propose $p -> predict $fp -> score $fs"
  last_array="$fs"
done

# The matched random control, at the cohort's full draw count. No dependency on the
# chain: it only needs the reference folds, which step 0 pays for.
read -r cp cs <<<"$(fold_stage es_control "$seed_job" 0)"
echo "[es-pilot] random control: predict $cp -> score $cs"

echo "[es-pilot] last generation array: $last_array"
echo "[es-pilot] when both finish:"
echo "    .venv/bin/python script_utils/cpseq_search.py confirm --out-dir $OUT_DIR"
echo "    .venv/bin/python script_utils/cpseq_search.py report  --out-dir $OUT_DIR"
