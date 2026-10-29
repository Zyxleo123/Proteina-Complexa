#!/bin/bash
# Submitter for the CPSea sequence-proposer (masked-LM) arms. Calls sbatch and nothing
# else -- no extraction, no training, no checks that belong inside a job.
#
# Chain:
#   corpus(val)   ]
#   corpus(train) ]-- array jobs, siblings so they queue concurrently
#        |
#        +-- afterok --> train: pocket-conditioned arm   ]
#        +-- afterok --> train: target-free control arm  ]-- siblings, one GPU each
#
# The two training arms are SIBLINGS, not a sequence: they are the paired comparison this
# build exists to make, so they should run at the same time on the same corpus rather than
# hours apart.
#
# Usage:
#   bash scripts/submit_cpseq_mlm.sh --smoke     # 1 shard, 200 steps, both arms
#   bash scripts/submit_cpseq_mlm.sh --corpus    # corpus only
#   bash scripts/submit_cpseq_mlm.sh --submit    # full chain
#
# Overrides are passed as VAR=value and land in the per-submission env file:
#   bash scripts/submit_cpseq_mlm.sh --submit CPSEQ_MLM_LR=2e-5 CPSEQ_MLM_MAX_STEPS=40000

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/scripts/cpseq_mlm_env.sh"

MODE=""
SKIP_VAL=0
OVERRIDES=()
for arg in "$@"; do
  case "$arg" in
    --smoke|--submit|--corpus|--train-only) MODE="$arg" ;;
    # Val is 5% of CPSea and changes only when the pocket radius does, so rebuilding it
    # with every train corpus wastes submission slots against a QOS ceiling that counts
    # array tasks. It also AVOIDS a mixed shard grouping: re-running val at a different
    # shard count leaves two sets of filenames covering the same rows, which the trainer
    # now refuses outright (see cpseq_mlm_data.assert_one_shard_grouping).
    --skip-val) SKIP_VAL=1 ;;
    *=*) OVERRIDES+=("$arg") ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[ -n "$MODE" ] || { echo "pick one of --smoke | --corpus | --train-only | --submit" >&2; exit 2; }

# Scrub SLURM_* so a submission made from inside an allocation does not inherit the
# parent job's identity and confuse the array indexing.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v" || true; done

STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_DIR="$CPSEQ_MLM_ROOT/env_files"
mkdir -p "$ENV_DIR" "$CPSEQ_MLM_LOG_DIR" "$CPSEQ_MLM_CORPUS_DIR" "$CPSEQ_MLM_RUNS_DIR"

NUM_SHARDS="$CPSEQ_MLM_NUM_SHARDS"
VAL_SHARDS="$CPSEQ_MLM_VAL_SHARDS"
MAX_STEPS="$CPSEQ_MLM_MAX_STEPS"
SUFFIX=""
EXTRA=()

# An override on the command line must be visible to THIS script, not only written into
# the env file the jobs source: the banner, the array sizes and the submit-budget check
# below all read these variables, and a banner that disagrees with what the jobs run is
# how a run gets filed under the wrong step count.
for kv in "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; do
  export "${kv?}"
done
NUM_SHARDS="$CPSEQ_MLM_NUM_SHARDS"
VAL_SHARDS="$CPSEQ_MLM_VAL_SHARDS"
MAX_STEPS="$CPSEQ_MLM_MAX_STEPS"
if [ "$MODE" = "--smoke" ]; then
  # A smoke run must exercise the same code path, not a shortened imitation of it: same
  # scripts, same env file mechanism, one shard and few steps.
  NUM_SHARDS=1
  VAL_SHARDS=1
  MAX_STEPS=200
  SUFFIX="_smoke"
  # CPSEQ_MLM_LIMIT is not optional in smoke mode: with one shard, a shard IS the whole
  # split, and the train split is 2.44M rows. Without it the "smoke" is a 13-hour job.
  EXTRA+=("CPSEQ_MLM_EVAL_EVERY=100" "CPSEQ_MLM_SAVE_EVERY=200" "CPSEQ_MLM_EVAL_N=200"
          "CPSEQ_MLM_LIMIT=3000" "CPSEQ_MLM_CORPUS_DIR=$CPSEQ_MLM_ROOT/corpus_smoke")
fi

write_env () {  # write_env <path> <extra VAR=value ...>
  local path="$1"; shift
  {
    echo "# cpseq_mlm submission $STAMP (mode $MODE)"
    echo "export CPSEQ_MLM_NUM_SHARDS=$NUM_SHARDS"
    echo "export CPSEQ_MLM_MAX_STEPS=$MAX_STEPS"
    for kv in "${EXTRA[@]+"${EXTRA[@]}"}" "${OVERRIDES[@]+"${OVERRIDES[@]}"}" "$@"; do
      echo "export $kv"
    done
  } > "$path"
  echo "$path"
}

echo "=== submit_cpseq_mlm ($MODE) ==="
echo "stamp:   $STAMP"
echo "corpus:  $CPSEQ_MLM_CORPUS_DIR"
echo "runs:    $CPSEQ_MLM_RUNS_DIR"
echo "shards:  train=$NUM_SHARDS val=$([ "$SKIP_VAL" = 1 ] && echo skipped || echo "$VAL_SHARDS")   steps: $MAX_STEPS"

# --- submit budget. Every array TASK counts as a submitted job against the QOS limit, so
# a chain that would overflow it gets half-submitted: the first array lands, the second is
# rejected with QOSMaxSubmitJobPerUserLimit, and the trainers then wait on a dependency
# that will never be satisfied. Checked here rather than discovered there.
PLANNED=0
VAL_PLANNED=$VAL_SHARDS
[ "$SKIP_VAL" = "1" ] && VAL_PLANNED=0
case "$MODE" in
  --corpus)     PLANNED=$((NUM_SHARDS + VAL_PLANNED)) ;;
  --train-only) PLANNED=2 ;;
  *)            PLANNED=$((NUM_SHARDS + VAL_PLANNED + 2)) ;;
esac
EXISTING="$(squeue -u "$USER" -h -o '%i' 2>/dev/null | wc -l)"
if [ "$((PLANNED + EXISTING))" -gt "$CPSEQ_MLM_MAX_SUBMIT" ]; then
  echo
  echo "REFUSING TO SUBMIT: this chain needs $PLANNED job slots, you already hold" >&2
  echo "$EXISTING, and the budget is CPSEQ_MLM_MAX_SUBMIT=$CPSEQ_MLM_MAX_SUBMIT." >&2
  echo >&2
  echo "The QOS rejects the overflow rather than queueing it, which leaves a chain" >&2
  echo "half-submitted. Options:" >&2
  echo "  * wait for running jobs to drain (squeue -u $USER)" >&2
  echo "  * lower the shard counts: CPSEQ_MLM_NUM_SHARDS=8 CPSEQ_MLM_VAL_SHARDS=2" >&2
  echo "  * raise the budget if you know the real ceiling: CPSEQ_MLM_MAX_SUBMIT=N" >&2
  exit 3
fi
echo "budget:  $PLANNED new + $EXISTING held <= $CPSEQ_MLM_MAX_SUBMIT"

CORPUS_DEPS=()

submit_corpus () {  # submit_corpus <split> <n_shards>
  local split="$1" n="$2"
  # The shard count goes in the env file too: the job divides the split by it, so a job
  # that disagreed with the --array range would silently extract the wrong row blocks.
  local env_file; env_file="$(write_env "$ENV_DIR/corpus_${split}_${STAMP}${SUFFIX}.env" \
      "CPSEQ_MLM_SPLIT=$split" "CPSEQ_MLM_NUM_SHARDS=$n")"
  local last=$((n - 1))
  local cmd=(sbatch --parsable --array="0-${last}" --job-name="cpseq_mlm_corpus_${split}"
             scripts/cpseq_mlm_corpus.sbatch "$env_file")
  echo "+ ${cmd[*]}"
  local jid
  if ! jid="$("${cmd[@]}")"; then
    echo "FATAL: sbatch rejected the $split corpus array. Nothing downstream was" >&2
    echo "submitted, so there is no half-built chain to clean up." >&2
    exit 4
  fi
  echo "  -> corpus[$split] job $jid ($n shards)"
  CORPUS_DEPS+=("$jid")
}

submit_train () {  # submit_train <run_name> <no_pocket 0|1> [dep]
  local run_name="$1" no_pocket="$2" dep="${3:-}"
  local env_file; env_file="$(write_env "$ENV_DIR/train_${run_name}_${STAMP}.env" \
      "CPSEQ_MLM_RUN_NAME=$run_name" "CPSEQ_MLM_NO_POCKET=$no_pocket")"
  local cmd=(sbatch --parsable --job-name="$run_name")
  [ -n "$dep" ] && cmd+=(--dependency="afterok:$dep")
  cmd+=(scripts/cpseq_mlm_train.sbatch "$env_file")
  echo "+ ${cmd[*]}"
  local jid; jid="$("${cmd[@]}")"
  echo "  -> train[$run_name] job $jid"
}

if [ "$MODE" != "--train-only" ]; then
  if [ "$SKIP_VAL" = "1" ]; then
    echo "skipping val corpus (--skip-val); expecting it already on disk"
  else
    submit_corpus val "$VAL_SHARDS"
  fi
  submit_corpus train "$NUM_SHARDS"
fi

if [ "$MODE" = "--corpus" ]; then
  echo "corpus only; stopping here."
  exit 0
fi

DEP=""
if [ "${#CORPUS_DEPS[@]}" -gt 0 ]; then
  DEP="$(IFS=:; echo "${CORPUS_DEPS[*]}")"
fi

submit_train "cpseq_mlm_pocket${SUFFIX}" 0 "$DEP"
submit_train "cpseq_mlm_nopocket${SUFFIX}" 1 "$DEP"

echo
echo "watch:  squeue -u \$USER -o '%.10i %.26j %.9T %.10M %R'"
echo "logs:   $CPSEQ_MLM_LOG_DIR"
echo "note:   sacct is broken on this cluster (exits 0 with no rows) -- use squeue while"
echo "        running, scontrol show job <id> just after, or the job's own .out."
