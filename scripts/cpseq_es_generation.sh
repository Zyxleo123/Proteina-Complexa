#!/usr/bin/env bash
# One generation of the LLM evolutionary search over the PepBench-train cohort.
#
# Submits the three stages of generation N as an afterok chain:
#
#   propose (GPU)  -> llm_proposals/roundN/<parent>.jsonl, folds nothing
#   predict (GPU)  -> Boltz-2 structures for rounds 1..N, no relax
#   score   (CPU)  -> PyRosetta relax + the frozen geometry gate
#
# Generations are NOT auto-chained onto each other. One generation is ~27,232 new folds
# (851 parents x 8 candidates x 4 seeds) = ~108 GPU-h of prediction plus ~331 CPU-h of
# relax, and `afterok` guards against a stage FAILING, not against a generation whose
# proposals were worthless. Read generation N's report before spending generation N+1.
#
# Why the cohort is rebuilt every generation
# ------------------------------------------
# `load_llm_proposals` concatenates rounds 1..N and numbers the draws globally, so the
# arm grows 8 -> 16 -> 24 while `cohort.json` still says `n_draws: 8`. Nothing truncates
# to `n_draws` -- every proposal on disk is folded either way -- but `budget_rungs` is
# what the report truncates to, so without the rebuild there is no B=16 rung to read
# generation 2 at. Rebuilding is safe and idempotent: the parent set is every prepared
# target, selection is seeded on the corpus label, and the gate hash is re-recorded from
# the same frozen tolerances file. It must happen BETWEEN generations, never while a fold
# job is in flight, or a running shard and its cohort would disagree about n_draws.
#
# Why the score stage is 5 jobs of 12 workers rather than 20 of 3
# ---------------------------------------------------------------
# `preempt_cpu_qos` caps a user at 50 SUBMITTED jobs and 64 CPUs. With other pipelines
# holding ~45 of those slots a 20-task array is refused outright
# (QOSMaxSubmitJobPerUserLimit), while 5 x 12 reaches the same 60 concurrent single-
# threaded relax workers -- the same 60 global shards -- inside 5 slots and 60 CPUs.
# Raise SCORE_SHARDS/SCORE_WORKERS back to 20/3 when the queue is quiet; the global shard
# count is SCORE_SHARDS*SCORE_WORKERS and only that product has to stay put.
#
# Usage:
#   scripts/cpseq_es_generation.sh GEN=1
#   scripts/cpseq_es_generation.sh GEN=2 AFTER=<gen1 score jobid>   # queue behind gen 1
#   scripts/cpseq_es_generation.sh GEN=2 DRY_RUN=1                  # print, submit nothing

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

for kv in "$@"; do
  case "$kv" in
    *=*) export "${kv?}" ;;
    *) echo "bad argument (want VAR=value): $kv" >&2; exit 2 ;;
  esac
done

GEN="${GEN:?set GEN=<generation, 1-based>}"
OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/cpseq/es_corpus_train}"
DATASET="${DATASET:-/data/user_data/yixiz/cpseq/pepbench_train}"
LABEL="${LABEL:-PepBenchTrain}"
N_PARENTS="${N_PARENTS:-851}"
CANDIDATES="${CANDIDATES:-8}"      # per complex per generation
SEED_DEPTH="${SEED_DEPTH:-4}"      # folds per candidate
ELITES="${ELITES:-2}"
OFFSPRING_PER_ELITE="${OFFSPRING_PER_ELITE:-4}"
PROPOSE_SHARDS="${PROPOSE_SHARDS:-20}"
PROPOSE_TIME="${PROPOSE_TIME:-3:00:00}"
PREDICT_SHARDS="${PREDICT_SHARDS:-20}"
SCORE_SHARDS="${SCORE_SHARDS:-5}"
SCORE_WORKERS="${SCORE_WORKERS:-12}"
SCORE_MEM="${SCORE_MEM:-56G}"
RETENTION="${RETENTION:-rescorable}"
AFTER="${AFTER:-}"                 # optional: jobid of the previous generation's score
PY="${PY:-$REPO/.venv/bin/python}"
LOG="${LOG:-$OUT_DIR/es_generations.log}"

# Cumulative draws after this generation. The ladder reads this from cohort.json and the
# report's rungs are prefixes of it, so it has to be the TOTAL, not this generation's 8.
DRAWS=$(( CANDIDATES * GEN ))

mkdir -p "$OUT_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "=== es generation $GEN  $(date -Is) ==="
echo "out_dir=$OUT_DIR draws_cumulative=$DRAWS seed_depth=$SEED_DEPTH candidates=$CANDIDATES"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry] would rebuild cohort at --draws $DRAWS --seed-depth $SEED_DEPTH"
  echo "[dry] would submit propose(ROUND=$GEN) -> predict -> score"
  exit 0
fi

# --- cohort ----------------------------------------------------------------------------
# Refuse to touch it while anything is folding into this directory: a shard that read
# n_draws=8 and a cohort that now says 16 would disagree about what rung it is filling.
INFLIGHT=$(squeue -u "$USER" -h -o '%i %j' 2>/dev/null \
            | grep -cE 'es_g[0-9]+_(predict|score)' || true)
if [ "${INFLIGHT:-0}" -gt 0 ] && [ -z "$AFTER" ]; then
  echo "FATAL: $INFLIGHT fold job(s) still queued or running for this run."
  echo "       Wait for them, or pass AFTER=<their jobid> to queue behind them."
  exit 1
fi

echo "[1/4] cohort -> $DRAWS draws, depth $SEED_DEPTH"
"$PY" script_utils/cpseq_search.py ladder-cohort \
  --out-dir "$OUT_DIR" \
  --corpus "dataset_dir=$DATASET,n=$N_PARENTS,label=$LABEL" \
  --draws "$DRAWS" --seed-depth "$SEED_DEPTH"

DEP_PROPOSE=()
[ -n "$AFTER" ] && DEP_PROPOSE=(--dependency="afterok:$AFTER")

echo "[2/4] propose: generation $GEN, $CANDIDATES candidate(s) per complex"
P=$(sbatch --parsable --job-name="es_g${GEN}_propose" \
      --array="0-$((PROPOSE_SHARDS - 1))%${PROPOSE_SHARDS}" --time="$PROPOSE_TIME" \
      "${DEP_PROPOSE[@]}" \
      scripts/cpseq_llm_propose.sbatch \
      OUT_DIR="$OUT_DIR" ROUND="$GEN" BATCH_SIZE="$CANDIDATES" \
      ELITES="$ELITES" OFFSPRING_PER_ELITE="$OFFSPRING_PER_ELITE" \
      SHARD_COUNT="$PROPOSE_SHARDS")
echo "      propose $P"

echo "[3/4] predict: rounds 1..$GEN, GPU, no relax"
D=$(sbatch --parsable --job-name="es_g${GEN}_predict" \
      --array="0-$((PREDICT_SHARDS - 1))%${PREDICT_SHARDS}" --dependency="afterok:$P" \
      scripts/cpseq_ladder.sbatch \
      OUT_DIR="$OUT_DIR" SHARD_COUNT="$PREDICT_SHARDS" LLM_ROUND="$GEN" PREDICT_ONLY=1)
echo "      predict $D  (afterok:$P)"

echo "[4/4] score: $SCORE_SHARDS job(s) x $SCORE_WORKERS worker(s) = $((SCORE_SHARDS * SCORE_WORKERS)) global shards"
S=$(sbatch --parsable --job-name="es_g${GEN}_score" \
      --array="0-$((SCORE_SHARDS - 1))%${SCORE_SHARDS}" --dependency="afterok:$D" \
      --cpus-per-task="$SCORE_WORKERS" --mem="$SCORE_MEM" --time=8:00:00 \
      scripts/cpseq_ladder_score.sbatch \
      OUT_DIR="$OUT_DIR" SHARD_COUNT="$SCORE_SHARDS" WORKERS="$SCORE_WORKERS" \
      LLM_ROUND="$GEN" RETENTION="$RETENTION")
echo "      score   $S  (afterok:$D)"

echo "generation $GEN queued: propose=$P predict=$D score=$S"
echo "next: scripts/cpseq_es_generation.sh GEN=$((GEN + 1)) AFTER=$S"
