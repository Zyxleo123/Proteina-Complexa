#!/usr/bin/env bash
# Push the migration set from rhea to Backblaze B2. Run on rhea, today.
#
# Why B2 and not straight to babel: access to rhea ends today, babel's queue
# does not cooperate on demand, and /data on babel is not writable from its
# login node. So the deadline is on getting data OFF rhea, not on babel
# receiving it. B2 decouples the two.
#
# Measured on rhea 2026-10-18:
#   rhea -> internet            85.3 MB/s
#   rhea -> bastion -> babel    16 MB/s
# so the cloud hop is also the faster one.
#
# Resumable: every object is checked for existence before upload, so a rerun
# skips what already landed. Safe to interrupt and restart.
#
# Setup first:
#   rclone config create b2 b2 account <keyID> key <applicationKey>
#   rclone lsd b2:
#
# Usage:
#   bash scripts/rhea_push_b2.sh b2:my-bucket/cpsea
#   bash scripts/rhea_push_b2.sh b2:my-bucket/cpsea core    # 41 GB core only

set -euo pipefail

DEST="${1:-}"
WHAT="${2:-all}"
[ -n "$DEST" ] || { echo "usage: $0 b2:bucket/prefix [core|processed|all]" >&2; exit 1; }

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
MAN="${MAN:-$ZFS/migration_min_full}"
CH="$MAN/chunks"
SRC="$ZFS/CPSea/CPSea_full/CPSea/preprocessed/processed"
TRANSFERS="${TRANSFERS:-16}"

command -v rclone >/dev/null || { echo "rclone not found" >&2; exit 1; }
rclone lsd "${DEST%%/*}" >/dev/null 2>&1 || {
  echo "FATAL: cannot reach ${DEST%%/*} -- is the remote configured?" >&2
  echo "  rclone config create b2 b2 account <keyID> key <applicationKey>" >&2
  exit 1
}

echo "=== $(date) pushing to $DEST ==="

# ---- the irreplaceable core ----------------------------------------------
# Checkpoints, both autoencoders, all results, and the CPSea split
# definitions. ~41 GB. Nothing here can be regenerated or re-downloaded.
if [ "$WHAT" = "all" ] || [ "$WHAT" = "core" ]; then
  echo
  echo "--- core: $ZFS via migration_min manifest ---"
  rclone copy --files-from "$ZFS/migration_min/tier0.files" \
    "$ZFS" "$DEST/zfs" -P --transfers "$TRANSFERS" --checkers "$TRANSFERS"

  echo
  echo "--- core: results from the repo tree on home ---"
  rclone copy --files-from "$MAN/tier0_home.files" \
    "$REPO" "$DEST/repo" -P --transfers "$TRANSFERS" --checkers "$TRANSFERS"

  # The manifests themselves, so the babel side knows what to expect.
  rclone copy "$ZFS/migration_min/tier0.files" "$DEST/manifests/" -P
  rclone copy "$MAN/tier0_home.files" "$DEST/manifests/" -P
  rclone copy "$CH" "$DEST/manifests/chunks" -P --transfers "$TRANSFERS"
  echo "core done"
fi

# ---- the 2.7M preprocessed PDBs ------------------------------------------
# Uploaded as 55 tar.zst objects rather than 2.7M files: per-object overhead
# would dominate, and scratch is 99% full so no tarball can be staged on
# disk. tar streams through zstd straight into rclone rcat, nothing touches
# local storage. Chunked rather than one stream so that both this upload and
# the download on babel are resumable -- babel's only partition is
# preemptible, and a 17 GB stream with no resume point would be lost.
if [ "$WHAT" = "all" ] || [ "$WHAT" = "processed" ]; then
  echo
  echo "--- processed: 55 chunks as tar.zst ---"
  [ -d "$CH" ] || { echo "FATAL: no chunk lists at $CH" >&2; exit 1; }

  existing="$(rclone lsf "$DEST/chunks" 2>/dev/null || true)"
  n=0; skipped=0
  for L in $(ls -1 "$CH"/val_*.list "$CH"/test_*.list "$CH"/train_*.list 2>/dev/null); do
    name="$(basename "$L" .list)"
    if printf '%s\n' "$existing" | grep -qx "$name.tar.zst"; then
      skipped=$((skipped + 1)); continue
    fi
    echo "  $name ($(wc -l < "$L") files) $(date +%H:%M:%S)"
    tar -C "$SRC" --ignore-failed-read -cf - -T "$L" \
      | zstd -3 -T2 \
      | rclone rcat "$DEST/chunks/$name.tar.zst"
    n=$((n + 1))
  done
  echo "processed done: $n uploaded, $skipped already there"
fi

echo
echo "=== $(date) finished ==="
rclone size "$DEST" 2>/dev/null || true
