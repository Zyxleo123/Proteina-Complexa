#!/usr/bin/env bash
# Launch the PepBench train-split random-arm scale-up, once target preparation finishes.
#
# Waits for `cpseq_prepare_targets.py` to exit, selects every prepared target as the
# cohort, runs a single-parent preflight AS A JOB, and submits the array only if that
# preflight succeeded. The preflight-as-a-job step is not ceremony: processes adopted
# into a node-hold allocation's extern step get a memory ceiling below Boltz's ~11 GiB
# working set, and the failure is a bare SIGKILL with no traceback.
#
# Resumable by design. The journal key is (sequence, seed, stage, arm) and the per-arm
# budget is counted from the journal, so a preempted task continues rather than
# re-spending. Re-running this script after a partial run is safe.
set -euo pipefail

REPO=/home/yixiz/Proteina-Complexa
ZFS=/data/user_data/yixiz
DATA=$ZFS/cpseq/pepbench_train
OUT=$ZFS/cpseq/pepbench_train_search
LOG=${LOG:-$DATA/launch.log}

# 3,433 parents over 300 shards is ~11 parents and ~103 evaluations per task. Sharding
# rather than one task per parent matters twice over: the first evaluation in a process
# costs ~185 s against a ~55 s marginal cost (PyRosetta init and the model load are paid
# once), so 9 evaluations per process would carry ~16 % startup overhead; and ~2.4 h of
# work per task sits comfortably inside a 4 h limit on a partition that requeues.
SHARDS=${SHARDS:-300}
CONCURRENT=${CONCURRENT:-40}
RETENTION=${RETENTION:-rescorable}
ARMS=${ARMS:-random_control}
RANDOM_POLICY=${RANDOM_POLICY:-standalone}

exec >>"$LOG" 2>&1
echo "=== launch $(date -Is) ==="
echo "shards=$SHARDS concurrent=$CONCURRENT retention=$RETENTION arms=$ARMS policy=$RANDOM_POLICY"

cd "$REPO"
# shellcheck disable=SC1091
source env.sh >/dev/null 2>&1 || true
PY=$REPO/.venv/bin/python

echo "[1/4] waiting for target preparation"
while pgrep -f cpseq_prepare_targets >/dev/null; do sleep 60; done
if [ ! -f "$DATA/manifest/targets.csv" ]; then
  echo "FATAL: no manifest/targets.csv -- preparation did not complete"; exit 1
fi
N=$(( $(wc -l < "$DATA/manifest/targets.csv") - 1 ))
echo "      prepared targets: $N"
[ "$N" -lt 100 ] && { echo "FATAL: only $N targets prepared, refusing to launch"; exit 1; }

# The frozen tolerances must be the SAME file for every corpus, or PepBench and LNR
# numbers stop being comparable, which is the one comparison this run exists to make.
[ -f "$DATA/manifest/geometry_tolerances.json" ] || {
  echo "FATAL: manifest/geometry_tolerances.json missing"; exit 1; }

echo "[2/4] cohort = every prepared target (no prior tranche to stratify against)"
if [ ! -f "$OUT/cohort.json" ]; then
  "$PY" script_utils/cpseq_search.py select-cohort \
    --dataset-dir "$DATA" --out-dir "$OUT" --all-targets
else
  echo "      cohort.json exists, keeping it"
fi
NP=$("$PY" -c "import json;print(len(json.load(open('$OUT/cohort.json'))['cohort']))")
echo "      cohort parents: $NP"

echo "[3/4] preflight one parent, as a job"
P0=$("$PY" -c "import json;print(json.load(open('$OUT/cohort.json'))['cohort'][0])")
PF=$(sbatch --parsable --job-name=pbt_preflight --time=1:00:00 \
      scripts/cpseq_search.sbatch MODE=screen "OUT_DIR=$OUT" \
      "PARENTS=$P0" LIMIT=3 SHARD_COUNT=1 \
      "RETENTION=$RETENTION" "ARMS=$ARMS" "RANDOM_POLICY=$RANDOM_POLICY")
echo "      preflight job $PF on $P0"
while squeue -j "$PF" -h >/dev/null 2>&1 && [ -n "$(squeue -j "$PF" -h 2>/dev/null)" ]; do
  sleep 60
done
ST=$(sacct -j "$PF" -n -o State --parsable2 2>/dev/null | head -1 || echo UNKNOWN)
echo "      preflight state: $ST"
case "$ST" in
  COMPLETED) ;;
  *) echo "FATAL: preflight did not complete ($ST); not submitting the array."
     echo "       A -9/SIGKILL with no traceback is the OOM signature -- check the cgroup."
     exit 1 ;;
esac

echo "[4/4] submitting the array"
JID=$(sbatch --parsable --array=0-$((SHARDS-1))%"$CONCURRENT" \
      --job-name=pbt_search scripts/cpseq_search.sbatch \
      MODE=screen "OUT_DIR=$OUT" "SHARD_COUNT=$SHARDS" \
      "RETENTION=$RETENTION" "ARMS=$ARMS" "RANDOM_POLICY=$RANDOM_POLICY")
echo "      array job $JID ($SHARDS shards, $CONCURRENT concurrent)"
echo "=== submitted $(date -Is) ==="
