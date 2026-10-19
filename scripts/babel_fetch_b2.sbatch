#!/usr/bin/env bash
#SBATCH --job-name=cpsea_fetch
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/home/yixiz/cpsea_fetch_%j.out
#
# Pull the migration set from Backblaze B2 onto babel.
#
# The log goes to /home on purpose: /data/user_data is not mounted on the
# login node, so the directory cannot exist at submit time, and Slurm
# resolves --output before the script runs and will not create it.
#
# This is also a plain bash script -- the #SBATCH lines are comments -- so if
# you ever get a shell on a node that can see /data, you can just run it.
#
# Everything here is resumable, because preempt can kill the job at any
# moment:
#   - rclone copy skips objects already present locally
#   - each of the 55 processed chunks drops a marker on success, so a rerun
#     re-extracts only what is missing
#
# Submit:   sbatch scripts/babel_fetch_b2.sbatch b2:my-bucket/cpsea
# Resume:   the identical line

set -euo pipefail

DEST_URL=""
for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) DEST_URL="$arg" ;;
  esac
done
SRC_URL="${SRC_URL:-$DEST_URL}"
[ -n "$SRC_URL" ] || { echo "usage: sbatch $0 b2:bucket/prefix" >&2; exit 1; }

NEW="${NEW:-/data/user_data/yixiz}"
NEWREPO="${NEWREPO:-/home/yixiz/Proteina-Complexa}"
DST="$NEW/CPSea/CPSea_full/CPSea/preprocessed/processed"
STATE="$NEW/.fetch_state"
TRANSFERS="${TRANSFERS:-8}"

echo "=== $(date) job ${SLURM_JOB_ID:-none} on $(hostname) ==="
echo "source=$SRC_URL dest=$NEW"

mkdir -p "$NEW" "$DST" "$STATE" || true
if [ ! -d "$NEW" ]; then
  echo "FATAL: cannot create $NEW on $(hostname) -- is /data mounted here?" >&2
  exit 1
fi
df -h "$NEW" | tail -1

command -v rclone >/dev/null || { echo "FATAL: rclone not on PATH" >&2; exit 1; }

# A compute node may not have outbound internet even when the login node
# does. Find out before pretending to make progress.
if ! rclone lsd "${SRC_URL%%/*}" >/dev/null 2>&1; then
  echo "FATAL: cannot reach ${SRC_URL%%/*} from $(hostname)." >&2
  echo "If this works on the login node, compute nodes have no outbound" >&2
  echo "internet: fetch to /home there instead, then move it across." >&2
  exit 1
fi
echo "b2 reachable: ok"

# ---- core ----------------------------------------------------------------
echo
echo "=== core: checkpoints, both AEs, results, CPSea split metadata ==="
rclone copy "$SRC_URL/zfs" "$NEW" -P --transfers "$TRANSFERS" --checkers "$TRANSFERS"
rclone copy "$SRC_URL/repo" "$NEWREPO" -P --transfers "$TRANSFERS" --checkers "$TRANSFERS"
echo "core done"

# ---- the 2.7M preprocessed PDBs -----------------------------------------
echo
echo "=== processed: 55 chunks ==="
chunks="$(rclone lsf "$SRC_URL/chunks" 2>/dev/null | grep '\.tar\.zst$' || true)"
if [ -z "$chunks" ]; then
  echo "no chunks at $SRC_URL/chunks -- skipping (core is still complete)"
else
  # val and test first: ~135k files each, enough to prove the pipe before
  # train's 2.46M.
  ordered="$(printf '%s\n' "$chunks" | grep '^val_'  || true)
$(printf '%s\n' "$chunks" | grep '^test_' || true)
$(printf '%s\n' "$chunks" | grep '^train_' || true)"

  done_n=0; got=0
  for obj in $ordered; do
    [ -n "$obj" ] || continue
    name="${obj%.tar.zst}"
    if [ -f "$STATE/$name.done" ]; then done_n=$((done_n + 1)); continue; fi
    echo "--- $name $(date +%H:%M:%S)"
    if rclone cat "$SRC_URL/chunks/$obj" | zstd -d | tar -C "$DST" -xf -; then
      touch "$STATE/$name.done"; got=$((got + 1))
    else
      echo "chunk $name failed or interrupted; will retry on resubmit" >&2
      exit 1
    fi
  done
  echo "processed: $got this run, $done_n already done"
fi

echo
echo "=== $(date) finished ==="
for s in train val test; do
  echo "$s: $(ls -1 "$DST/$s" 2>/dev/null | wc -l) files"
done
echo "expected: train 2455265, val 135125, test 136463"
