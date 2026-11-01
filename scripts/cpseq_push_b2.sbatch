#!/usr/bin/env bash
#SBATCH --job-name=cpseq_push
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/cpseq_push_%j.out
#
# Slurm wrapper around scripts/cpseq_push_b2.sh.
#
# It exists only because /data/user_data/yixiz is not mounted on the login
# node, so the push cannot be started from where it is submitted. The real
# logic -- what counts as core, what counts as folds, and the skip-if-present
# resume -- lives in the pushed script and is not duplicated here.
#
# The log goes to /home on purpose: Slurm resolves --output before the job
# body runs, and a path under /data cannot be created at submit time.
#
# CPU-only and I/O-bound: 2 CPUs (one for tar, two for zstd -T2), 4 GB, no
# --gres. Small requests backfill onto busy preempt nodes much sooner.
#
# Requeue-safe: an object that already exists in the bucket is skipped, so a
# preempted run is finished by the identical submit line, and --open-mode=append
# keeps the log from the earlier attempt.
#
# Submit:
#   sbatch scripts/cpseq_push_b2.sbatch b2:proteina/cpseq            # core, then folds
#   sbatch scripts/cpseq_push_b2.sbatch b2:proteina/cpseq core       # the numbers only
#   sbatch scripts/cpseq_push_b2.sbatch b2:proteina/cpseq folds
#   REBUILD=1 sbatch ... is NOT how to do it -- pass REBUILD=1 as an argument.

set -euo pipefail

DEST=""
WHAT=""
for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    b2:*|*:*/*) DEST="$arg" ;;
    core|folds|all) WHAT="$arg" ;;
    *) echo "WARN: ignoring unrecognised argument '$arg'" >&2 ;;
  esac
done
DEST="${DEST:-b2:proteina/cpseq}"
WHAT="${WHAT:-all}"

# A batch script cannot find files relative to itself: Slurm runs a spool copy.
REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -f "$REPO/scripts/cpseq_push_b2.sh" ] || REPO=/home/yixiz/Proteina-Complexa
PUSH="$REPO/scripts/cpseq_push_b2.sh"
[ -f "$PUSH" ] || { echo "FATAL: $PUSH not found" >&2; exit 1; }

echo "=== $(date) job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "repo=$REPO"
echo "dest=$DEST tier=$WHAT rebuild=${REBUILD:-0}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  squeue -j "$SLURM_JOB_ID" -o '%.10i %.4C %.8m %.10L %R' -h || true
fi

CPSEQ="${CPSEQ:-/data/user_data/yixiz/cpseq}"
[ -d "$CPSEQ" ] || {
  echo "FATAL: $CPSEQ not visible on $(hostname) -- is /data mounted here?" >&2
  exit 1
}

# ---- only one pusher at a time -------------------------------------------
# Several of these are submitted as siblings rather than a dependency chain:
# siblings are all eligible at once, so three queue entries get three chances
# at a backfill hole instead of three waits in series. The cost is that two
# could land together and upload the same object twice, so the losers stand
# down here.
#
# Two guards, because the first is not enough on its own. Slurm runs a spool
# COPY of this script, so a job submitted before this guard existed does not
# take the lock -- the squeue check catches that one too.
RUNNING_OTHER=$(squeue -u "$USER" -h -n cpseq_push -t R -o '%i' 2>/dev/null \
                | grep -v "^${SLURM_JOB_ID:-none}$" | head -1 || true)
if [ -n "$RUNNING_OTHER" ]; then
  echo "another cpseq_push is already RUNNING (job $RUNNING_OTHER) -- standing down"
  exit 0
fi

exec 9>"$CPSEQ/.push_b2.lock" || true
if ! flock -n 9; then
  echo "another cpseq_push holds the lock -- standing down"
  exit 0
fi
echo "holding the push lock"
df -h "$CPSEQ" | tail -1
echo "--- tree, apparent size (du overstates: NFS block padding) ---"
du -sh --apparent-size "$CPSEQ" 2>/dev/null || true

rc=0
bash "$PUSH" "$DEST" "$WHAT" || rc=$?

# ---- the morning report ---------------------------------------------------
# Written to /home, which IS mounted on the login node, so it can be read
# without holding an allocation. This is the whole point: the upload finishes
# unattended overnight and the answer is waiting on a path reachable from
# anywhere, rather than inside a node that has since expired.
REPORT="$HOME/cpseq_push_status.txt"
{
  echo "=== cpseq -> B2 push status ==="
  echo "written:   $(date)"
  echo "job:       ${SLURM_JOB_ID:-none} on $(hostname)"
  echo "dest:      $DEST"
  echo "tier:      $WHAT"
  echo "exit code: $rc  ($([ "$rc" -eq 0 ] && echo 'push script completed' || echo 'push script FAILED or was cut off'))"
  echo
  echo "--- what is in the bucket now ---"
  rclone ls "$DEST" 2>&1 | sort -k2 || echo "(could not list $DEST)"
  echo
  echo "--- totals ---"
  rclone size "$DEST" 2>&1 || true
  echo
  echo "--- expected, from the scan manifests ---"
  for t in core folds; do
    f="$CPSEQ/.push_b2/$t.dirs"
    [ -f "$f" ] && echo "$t: $(wc -l < "$f") directories expected as $t/*.tar.zst"
  done
  echo
  if [ "$rc" -ne 0 ]; then
    echo "NOT COMPLETE. Resume with the identical line -- objects already in the"
    echo "bucket are skipped, so this costs a listing and continues where it stopped:"
    echo "  sbatch scripts/cpseq_push_b2.sbatch $DEST $WHAT"
  else
    echo "COMPLETE for tier '$WHAT'. Nothing further to run."
  fi
} > "$REPORT" 2>&1
echo "--- report written to $REPORT ---"
cat "$REPORT"

echo "=== $(date) wrapper finished (rc=$rc) ==="
exit "$rc"
