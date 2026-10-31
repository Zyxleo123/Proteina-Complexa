#!/usr/bin/env bash
#SBATCH --job-name=cpseq_push
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=8:00:00
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
df -h "$CPSEQ" | tail -1
echo "--- tree, apparent size (du overstates: NFS block padding) ---"
du -sh --apparent-size "$CPSEQ" 2>/dev/null || true

bash "$PUSH" "$DEST" "$WHAT"

echo "=== $(date) wrapper finished ==="
