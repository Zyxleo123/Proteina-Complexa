#!/usr/bin/env bash
# The full PepBench-ES gen-1 llm_mask_mpnn_fill run, COLLAPSED, as an afterok chain.
#
#   A refs    (GPU)  fold reference + draw0 at the cohort's depth, so the LLM can see
#                    each complex's interface before it proposes. predict+score in one
#                    job: references are cheap (2 collapsed invocations/complex) and the
#                    propose stage needs them SCORED in the journal, not just predicted.
#   B propose (GPU)  LLM names positions, ProteinMPNN fills -> mask_proposals/
#                    llm_mask_mpnn_fill/round1/. Reads the reference folds from A.
#   C predict (GPU)  Boltz-2 structures for the arm's draws, PREDICT_ONLY, no relax.
#   D score   (CPU)  PyRosetta relax + the v3 gate, WORKERS sub-shards per job.
#
# Collapsed: the out_dir's cohort.json carries collapse_seeds/chunk_fix/
# max_parallel_samples (set by `cpseq_fork_cohort.py --collapse`), and every stage reads
# them back, so diffusion_samples becomes the seed depth and each draw is ONE invocation.
# This is a FRESH out_dir on purpose -- the collapsed and per-seed regimes key apart in
# the journal, so they cannot share a directory (see cpseq-trunk-collapse-wired).
#
# Not chained onto the uncollapsed pepbench_es: that run (job 94175) stays as the
# per-seed baseline this collapsed run is meant to reproduce.
#
# Config travels as VAR=value positional args, never `sbatch --export` (which triggers
# the user-env-retrieval requeue-hold on this cluster).
#
#   bash scripts/submit_cpseq_pepbench_es_collapse.sh
#   DRY_RUN=1 bash scripts/submit_cpseq_pepbench_es_collapse.sh
#   AFTER=<jobid> bash scripts/submit_cpseq_pepbench_es_collapse.sh   # queue A behind it

set -euo pipefail

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
cd "$REPO"

OUT_DIR="${OUT_DIR:-/zfsauton/scratch/yixiz/cpseq/pepbench_es_collapse}"
ARM="${ARM:-llm_mask_mpnn_fill}"
MASK_ROUND="${MASK_ROUND:-1}"
N_MASKS="${N_MASKS:-12}"
MPNN_TEMP="${MPNN_TEMP:-0.2}"
RETENTION="${RETENTION:-rescorable}"

# Stage shapes. 20 shards over 80 complexes = 4 complexes/task, the es_refs layout.
REF_SHARDS="${REF_SHARDS:-20}"
REF_LIMIT="${REF_LIMIT:-20}"          # ref + draw0 at full depth = rung B=1
PRED_SHARDS="${PRED_SHARDS:-20}"
PRED_CONCURRENCY="${PRED_CONCURRENCY:-5}"
SCORE_SHARDS="${SCORE_SHARDS:-5}"
SCORE_WORKERS="${SCORE_WORKERS:-12}"   # 5 jobs x 12 = 60 global shards, under the cap
SCORE_MEM="${SCORE_MEM:-56G}"
SKIP_INVALID_RELAX="${SKIP_INVALID_RELAX:-0}"
AFTER="${AFTER:-}"

[ -f "$OUT_DIR/cohort.json" ] || { echo "FATAL: no cohort.json in $OUT_DIR -- build the"\
  " collapsed fork first (cpseq_fork_cohort.py --collapse)"; exit 1; }
COLLAPSE=$(python3 -c "import json;print(json.load(open('$OUT_DIR/cohort.json')).get('collapse_seeds'))")
NPAR=$(python3 -c "import json;print(len(json.load(open('$OUT_DIR/cohort.json'))['cohort']))")

echo "=== pepbench_es collapsed chain ==="
echo "out_dir:   $OUT_DIR   (collapse_seeds=$COLLAPSE, $NPAR complexes)"
echo "arm:       $ARM  round 1..$MASK_ROUND  (n_masks=$N_MASKS, mpnn T=$MPNN_TEMP)"
echo "A refs:    cpseq_ladder     general a6000, array 0-$((REF_SHARDS-1))%${REF_SHARDS}, LIMIT=$REF_LIMIT (predict+score)"
echo "B propose: cpseq_mask_arm   preempt GPU (LLM mask + MPNN fill)"
echo "C predict: cpseq_ladder     general a6000, array 0-$((PRED_SHARDS-1))%${PRED_CONCURRENCY}, PREDICT_ONLY"
echo "D score:   cpseq_ladder_score cpu, array 0-$((SCORE_SHARDS-1))%${SCORE_SHARDS}, ${SCORE_WORKERS} workers/task (skip_inv=$SKIP_INVALID_RELAX)"
echo

# Scrub SLURM_* so each child sbatch creates a fresh context rather than inheriting this
# allocation's.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v" || true; done

COMMON="OUT_DIR=$OUT_DIR"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 -- the four stages that would be submitted:"
  echo "  A: sbatch --array=0-$((REF_SHARDS-1))%${REF_SHARDS} scripts/cpseq_ladder.sbatch $COMMON SHARD_COUNT=$REF_SHARDS LIMIT=$REF_LIMIT RETENTION=$RETENTION"
  echo "  B: sbatch -p preempt --qos=qos_preempt --dependency=afterok:<A> scripts/cpseq_mask_arm.sbatch <env with MASK_OUT_DIR=$OUT_DIR ARMS=$ARM>"
  echo "  C: sbatch --array=0-$((PRED_SHARDS-1))%${PRED_CONCURRENCY} --dependency=afterok:<B> scripts/cpseq_ladder.sbatch $COMMON SHARD_COUNT=$PRED_SHARDS MASK_ARM=$ARM MASK_ROUND=$MASK_ROUND PREDICT_ONLY=1"
  echo "  D: sbatch --array=0-$((SCORE_SHARDS-1))%${SCORE_SHARDS} --dependency=afterok:<C> scripts/cpseq_ladder_score.sbatch $COMMON SHARD_COUNT=$SCORE_SHARDS WORKERS=$SCORE_WORKERS MASK_ARM=$ARM MASK_ROUND=$MASK_ROUND RETENTION=$RETENTION"
  exit 0
fi

# An env file for the propose stage (it reads a FILE, and MASK_* names).
ENV_DIR="$OUT_DIR/_chain"
mkdir -p "$ENV_DIR"
MASK_ENV="$ENV_DIR/mask_arm.env"
cat > "$MASK_ENV" <<EOF
# Written by submit_cpseq_pepbench_es_collapse.sh at $(date -Is).
MASK_OUT_DIR=$OUT_DIR
MASK_ROUND=$MASK_ROUND
MASK_N_MASKS=$N_MASKS
MASK_ARMS=$ARM
MASK_MPNN_TEMP=$MPNN_TEMP
MASK_SEQS_PER_MASK=1
PYTHON_EXEC=$REPO/.venv/bin/python
EOF

DEP_A=(); [ -n "$AFTER" ] && DEP_A=(--dependency="afterok:$AFTER")
A=$(sbatch --parsable "${DEP_A[@]}" \
      --array="0-$((REF_SHARDS-1))%${REF_SHARDS}" \
      scripts/cpseq_ladder.sbatch \
      OUT_DIR="$OUT_DIR" SHARD_COUNT="$REF_SHARDS" LIMIT="$REF_LIMIT" RETENTION="$RETENTION")
echo "A refs:    $A"

# mask_arm.sbatch's own #SBATCH --output is slurm_logs/ on the home volume; redirect to
# scratch (logs never go to home). preempt GPU per its usage line.
B=$(sbatch --parsable --dependency="afterok:$A" \
      -p preempt --qos=qos_preempt --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=3:00:00 \
      --output="/zfsauton/scratch/yixiz/cpseq_af3/logs/%x_%j.out" \
      --error="/zfsauton/scratch/yixiz/cpseq_af3/logs/%x_%j.out" \
      scripts/cpseq_mask_arm.sbatch "$MASK_ENV")
echo "B propose: $B  (afterok:$A)"

C=$(sbatch --parsable --dependency="afterok:$B" \
      --array="0-$((PRED_SHARDS-1))%${PRED_CONCURRENCY}" \
      scripts/cpseq_ladder.sbatch \
      OUT_DIR="$OUT_DIR" SHARD_COUNT="$PRED_SHARDS" \
      MASK_ARM="$ARM" MASK_ROUND="$MASK_ROUND" PREDICT_ONLY=1 RETENTION="$RETENTION")
echo "C predict: $C  (afterok:$B)"

# cpseq_ladder_score.sbatch's own #SBATCH are stale babel values
# (--partition=preempt --account=bapoczos --qos=preempt_cpu_qos, logs to /home); override
# every one to the autonlab cpu pool and send logs to scratch.
D=$(sbatch --parsable --dependency="afterok:$C" \
      --partition=cpu --account=users --qos=qos_cpu \
      --output="/zfsauton/scratch/yixiz/cpseq_af3/logs/%x_%A_%a.out" \
      --error="/zfsauton/scratch/yixiz/cpseq_af3/logs/%x_%A_%a.out" \
      --array="0-$((SCORE_SHARDS-1))%${SCORE_SHARDS}" \
      --cpus-per-task="$SCORE_WORKERS" --mem="$SCORE_MEM" --time=8:00:00 \
      scripts/cpseq_ladder_score.sbatch \
      OUT_DIR="$OUT_DIR" SHARD_COUNT="$SCORE_SHARDS" WORKERS="$SCORE_WORKERS" \
      MASK_ARM="$ARM" MASK_ROUND="$MASK_ROUND" RETENTION="$RETENTION" \
      SKIP_INVALID_RELAX="$SKIP_INVALID_RELAX")
echo "D score:   $D  (afterok:$C)"

echo
echo "chain: A=$A -> B=$B -> C=$C -> D=$D"
echo "watch: squeue -u \$USER -o '%.12i %.18j %.9P %.8T %.10M %R'"
echo "report (after D): .venv/bin/python script_utils/cpseq_fill_sweep_report.py --out-dir $OUT_DIR --arms $ARM"
