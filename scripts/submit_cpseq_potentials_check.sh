#!/usr/bin/env bash
# Submit the folding-potentials check: prepare -> fold (array) -> report.
#
# docs/README_CPSEQ_POTENTIALS_CHECK_PLAN.md is the decision record. One setting moves,
# folding potentials off -> on, on the same cohort, sequences, seeds, gate and estimator
# that produced the pilot arm table.
#
# This submitter only calls sbatch. All configuration goes into ONE timestamped env file
# whose PATH is passed as a positional argument -- never `sbatch --export`, which sets
# SLURM_GET_USER_ENV=1, makes slurmd try to rebuild the login environment on the compute
# node, and lands the job requeued and HELD, stranding every dependent on Dependency.
# Arguments travel through the job record untouched.
#
#   bash scripts/submit_cpseq_potentials_check.sh                 # submit the chain
#   PARENT_SET=fixed bash scripts/submit_cpseq_potentials_check.sh
#   CONCURRENCY=8 bash scripts/submit_cpseq_potentials_check.sh
#   DRY_RUN=1 bash scripts/submit_cpseq_potentials_check.sh       # print, submit nothing

set -euo pipefail

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
cd "$REPO"

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
RUN_ID="${RUN_ID:-potcheck_$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="${WORK_DIR:-${ZFS}/cpseq/potcheck/${RUN_ID}}"
LOG_DIR="${ZFS}/cpseq/potcheck/logs"

SRC_DIR="${SRC_DIR:-${ZFS}/cpseq/llm_ladder}"
POT_DIR="${POT_DIR:-${ZFS}/cpseq/llm_ladder_pot}"
GATE="${GATE:-v3-cis-amide-muco-windows}"
LEAD_ARM="${LEAD_ARM:-llm_search}"
ARMS="${ARMS:-llm_search}"
# `fixed` is what the plan specifies: "the same 10-complex cohort that produced the pilot
# numbers", i.e. the 10 with a v3 baseline in the source run (6 PepBench, 4 LNR). That is
# also what its cost model is built on -- 10 complexes x (8-16 draws + 1 reference) x 10
# folds, about 1,000-1,700 folds. `all` folds the other 10 as well, which is the only way
# a complex that GAINS a baseline under potentials can be observed, but it roughly
# doubles the spend and is not what the plan asks for.
PARENT_SET="${PARENT_SET:-fixed}"
LLM_ROUNDS="${LLM_ROUNDS:-2}"
# The corpus paths recorded on the source cohort were absolute on the machine that built
# it. The fork rewrites the prefix and verifies the result exists.
REMAP="${REMAP:-/data/user_data/yixiz/cpseq=${ZFS}/cpseq}"
# An upper bound on the parent count; the fold job exits 0 for lines past the end of the
# parent list, so over-sizing is safe and the submitter needs no knowledge of the split.
ARRAY_MAX="${ARRAY_MAX:-9}"
# The GPU pools, as two arrays over DISJOINT index ranges of the same parents file. One
# submission per partition is forced, not chosen: each partition here carries its own
# restricted QoS (AllowQos=qos_general / qos_legacy), so a comma-separated --partition
# list is rejected with "Invalid qos specification". Splitting this way uses the a6000s
# and the v100s at once instead of queueing behind one pool.
SPLIT_AT="${SPLIT_AT:-4}"           # general takes 0..SPLIT_AT, legacy the rest
PART_A="${PART_A:-general}"
PART_B="${PART_B:-legacy}"
CONCURRENCY_A="${CONCURRENCY_A:-5}"
CONCURRENCY_B="${CONCURRENCY_B:-5}"
# The scoring pass. One task per complex on `cpu`, each pooling SCORE_WORKERS PyRosetta
# processes at ~1.5 GB each on a 422-residue pose.
#
# Set this as high as the core and memory request allow. The pool does NOT plateau for
# any reason that carries over to production.
#
# Job 95812, 10 folds, serial baseline 793.3 s. The apparent plateau at w=8 and w=16 is
# neither contention nor a scaling limit -- it is the task set being too small:
#
#   w    total CPU   longest fold   optimal makespan   measured   vs optimal
#   2      727.0 s        169.5 s           363.5 s    385.1 s        94.4%
#   4      733.8 s        174.1 s           183.4 s    235.8 s        77.8%
#   8      732.9 s        172.5 s           172.5 s    172.7 s        99.9%
#   16     728.8 s        172.0 s           172.0 s    172.3 s        99.8%
#
# Optimal makespan is max(longest task, total/workers), and no pool can beat it. At w=8
# and w=16 this pool is within 0.2% of optimal -- essentially perfect. The wall is pinned
# by ONE 172 s fold while the other nine total ~560 s, so past w~4 there is simply
# nothing left to overlap the long one with. Ten items cannot measure further.
#
# And there is NO contention term: total CPU work is 727-734 s at every worker count, and
# below serial's 791 s because the page cache is warm. 16 PyRosetta processes on an
# 88-core node do not slow each other down at all.
#
# On a real complex (~1,040 folds, ~73 s each => ~21 h of serial work) the tail is
# irrelevant: the pool stays total/workers-bound until roughly
# total/longest = 76,000/180 ~= 420 workers. So the real cap is the allocation -- one
# core and ~1.5 GB per worker. Raise --cpus-per-task and --mem together in
# cpseq_potentials_score.sbatch to go above 16.
SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-10}"
SCORE_WORKERS="${SCORE_WORKERS:-16}"

mkdir -p "$WORK_DIR" "$LOG_DIR"

# `#SBATCH --output=` is a static directive that cannot read a variable, and a relative
# path there resolves against the repo on the home volume. The directives carry absolute
# scratch paths; these overrides make the per-run location explicit as well.
ENV_FILE="${WORK_DIR}/env.sh"
cat > "$ENV_FILE" <<EOF
# Written by submit_cpseq_potentials_check.sh at $(date -Is). Passed to each job as a
# positional argument; the preamble sources any argument that is a path.
ZFS=${ZFS}
RUN_ID=${RUN_ID}
WORK_DIR=${WORK_DIR}
SRC_DIR=${SRC_DIR}
POT_DIR=${POT_DIR}
GATE=${GATE}
LEAD_ARM=${LEAD_ARM}
ARMS=${ARMS}
PARENT_SET=${PARENT_SET}
LLM_ROUNDS=${LLM_ROUNDS}
REMAP=${REMAP}
PYTHON_EXEC=${REPO}/.venv/bin/python
BOLTZ_PYTHON=${ZFS}/envs/boltz2/.venv/bin/python
BOLTZ_CACHE=${ZFS}/boltz_cache
SCORE_WORKERS=${SCORE_WORKERS}
EOF

echo "=== potentials check ==="
echo "run id:      $RUN_ID"
echo "work dir:    $WORK_DIR"
echo "env file:    $ENV_FILE"
echo "source (OFF):$SRC_DIR"
echo "fork   (ON): $POT_DIR"
echo "arm:         $LEAD_ARM   rounds 1..$LLM_ROUNDS"
echo "parents:     $PARENT_SET"
echo "predict:     $PART_A 0-${SPLIT_AT}%${CONCURRENCY_A} + $PART_B $((SPLIT_AT+1))-${ARRAY_MAX}%${CONCURRENCY_B}  (GPU)"
echo "score:       cpu 0-${ARRAY_MAX}%${SCORE_CONCURRENCY}, ${SCORE_WORKERS} workers/task"
echo

# Submitting from inside an allocation leaks SLURM_* into the child sbatch, which can
# make it inherit this shell's job context instead of creating a fresh one.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v" || true; done

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 -- the five commands that would be submitted:"
  echo "  sbatch --parsable scripts/cpseq_potentials_prepare.sbatch $ENV_FILE"
  echo "  # predict on the GPU, PREDICT_ONLY=1, two pools over disjoint index ranges"
  echo "  sbatch --parsable --dependency=afterok:<prep> --partition=$PART_A \\"
  echo "         --array=0-${SPLIT_AT}%${CONCURRENCY_A} scripts/cpseq_potentials_fold.sbatch $ENV_FILE PREDICT_ONLY=1"
  echo "  sbatch --parsable --dependency=afterok:<prep> --partition=$PART_B \\"
  echo "         --array=$((SPLIT_AT + 1))-${ARRAY_MAX}%${CONCURRENCY_B} scripts/cpseq_potentials_fold.sbatch $ENV_FILE PREDICT_ONLY=1"
  echo "  # score on the CPU, ${SCORE_WORKERS} worker processes per complex"
  echo "  sbatch --parsable --dependency=afterok:<predA>:<predB> \\"
  echo "         --array=0-${ARRAY_MAX}%${SCORE_CONCURRENCY} scripts/cpseq_potentials_score.sbatch $ENV_FILE"
  echo "  sbatch --parsable --dependency=afterok:<score> scripts/cpseq_potentials_report.sbatch $ENV_FILE"
  exit 0
fi

PREP=$(sbatch --parsable \
  --output="${LOG_DIR}/prepare_%j.out" --error="${LOG_DIR}/prepare_%j.out" \
  scripts/cpseq_potentials_prepare.sbatch "$ENV_FILE")
echo "prepare: $PREP"

# PREDICT on the GPU, SCORE on the CPU. This is the standard shape for every fold chain
# in this project -- see docs/README_CPSEQ_FOLD_PIPELINE_STANDARD.md -- and it is the
# default here rather than an option.
#
# Measured on `llm_ladder`'s 22,642 v3 folds: 37.8 s median of Boltz-2 prediction against
# 29.6 s median of Rosetta scoring, so more than half the wall time of a "GPU" job was
# PyRosetta with the card idle. Run as one job the two phases contend for the scarcest
# resource on the cluster; run as two they are scheduled apart, and the scoring half goes
# to `cpu` (88-core nodes, lov[1-2,5-9]) which is not contended against training.
#
# Nothing about a result changes: predictions are keyed by `input_hash` and scores by
# (scoring_version, input_hash), so the plan run as two passes fills the same two caches
# as the plan run as one.
PRED_A=$(sbatch --parsable --dependency="afterok:${PREP}" \
  --partition="$PART_A" --array="0-${SPLIT_AT}%${CONCURRENCY_A}" \
  --output="${LOG_DIR}/fold_%A_%a.out" --error="${LOG_DIR}/fold_%A_%a.out" \
  scripts/cpseq_potentials_fold.sbatch "$ENV_FILE" PREDICT_ONLY=1)
echo "predict A: $PRED_A  ($PART_A, array 0-${SPLIT_AT}%${CONCURRENCY_A}, GPU)"

PRED_B=$(sbatch --parsable --dependency="afterok:${PREP}" \
  --partition="$PART_B" --array="$((SPLIT_AT + 1))-${ARRAY_MAX}%${CONCURRENCY_B}" \
  --output="${LOG_DIR}/fold_%A_%a.out" --error="${LOG_DIR}/fold_%A_%a.out" \
  scripts/cpseq_potentials_fold.sbatch "$ENV_FILE" PREDICT_ONLY=1)
echo "predict B: $PRED_B  ($PART_B, array $((SPLIT_AT + 1))-${ARRAY_MAX}%${CONCURRENCY_B}, GPU)"

# One task per complex, each pooling SCORE_WORKERS processes internally. The shard axis in
# cpseq_search.py is the PARENT list, so an array that is already one complex per task
# gets nothing from sharding -- the parallelism has to come from inside the task, which is
# what --score-workers does: workers score, the parent writes, so the per-complex journal
# keeps its single writer. Concurrent O_APPEND to one journal NUL-corrupts rows here.
#
# afterok on an array waits for EVERY task, so scoring starts only once both GPU pools
# have finished predicting.
SCORE=$(sbatch --parsable --dependency="afterok:${PRED_A}:${PRED_B}" \
  --array="0-${ARRAY_MAX}%${SCORE_CONCURRENCY}" \
  --output="${LOG_DIR}/score_%A_%a.out" --error="${LOG_DIR}/score_%A_%a.out" \
  scripts/cpseq_potentials_score.sbatch "$ENV_FILE")
echo "score:     $SCORE  (cpu, array 0-${ARRAY_MAX}%${SCORE_CONCURRENCY}, ${SCORE_WORKERS} workers/task)"

REPORT=$(sbatch --parsable --dependency="afterok:${SCORE}" \
  --output="${LOG_DIR}/report_%j.out" --error="${LOG_DIR}/report_%j.out" \
  scripts/cpseq_potentials_report.sbatch "$ENV_FILE")
echo "report:    $REPORT"

echo
echo "watch:   squeue -u \$USER -o '%.10i %.18j %.9P %.8T %.10M %R'"
echo "logs:    $LOG_DIR"
echo "answer:  $WORK_DIR/compare_${LEAD_ARM}.json"
