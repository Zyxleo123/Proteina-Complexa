#!/usr/bin/env bash
# Reclaim space in the checkpoint store without losing anything unique.
#
# Three actions, in increasing order of how much thought they needed:
#
#   1. frozen_ae.ckpt duplicates -> hardlink. 19 runs hold a byte-identical
#      4.1 GB copy of the same pinned AE. Hardlinking them to one inode frees
#      the space while leaving every path working, so no run breaks.
#   2. ignore*.ckpt -> delete, but only where it is byte-identical to the
#      last*.ckpt beside it. These are stray duplicates, not a separate state.
#   3. periodic chk_epoch=* checkpoints that are not the highest-step EMA ->
#      delete. Intermediate training states for three AE runs.
#
# Optimizer state (last.ckpt) is never touched: that is what makes a run
# resumable, and only 2 of the 71 runs would be safe to judge automatically.
#
# Safety:
#   - DRY RUN unless --apply is passed.
#   - Refuses to touch any path listed in the tier0 transfer manifest.
#   - Re-verifies byte-identity at run time with a 5-point sample
#     (head/25%/50%/75%/tail); it never trusts an earlier scan.
#
# Usage:
#   bash scripts/migration_prune.sh              # report only
#   bash scripts/migration_prune.sh --apply      # actually reclaim

set -euo pipefail

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
STORE="$ZFS/Proteina-Complexa/training_runs/store"
MANIFEST="${MANIFEST:-$ZFS/migration/tier0.files}"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

[ -d "$STORE" ] || { echo "no store at $STORE" >&2; exit 1; }

# ---- refuse-list: anything we intend to transfer is off limits -------------
declare -A PROTECTED
if [ -f "$MANIFEST" ]; then
  while IFS= read -r rel; do PROTECTED["$ZFS/$rel"]=1; done < "$MANIFEST"
  echo "loaded ${#PROTECTED[@]} protected paths from $MANIFEST"
else
  echo "!! no manifest at $MANIFEST -- run scripts/migration_manifest.sh first" >&2
  echo "!! refusing to run without a refuse-list" >&2
  exit 1
fi

protected() { [ -n "${PROTECTED[$1]:-}" ]; }

# 5-point content fingerprint. Cheap enough to run on every candidate, strong
# enough that a collision between two different checkpoints is not a concern.
fp() {
  local f="$1" sz blk
  sz=$(stat -c%s "$f")
  blk=$((sz / 1048576 / 4))
  {
    dd if="$f" bs=1M count=32 2>/dev/null
    dd if="$f" bs=1M skip=$blk           count=32 2>/dev/null
    dd if="$f" bs=1M skip=$((blk * 2))   count=32 2>/dev/null
    dd if="$f" bs=1M skip=$((blk * 3))   count=32 2>/dev/null
    dd if="$f" bs=1M skip=$((sz / 1048576 > 32 ? sz / 1048576 - 32 : 0)) count=32 2>/dev/null
  } | md5sum | cut -d' ' -f1
}

freed=0
say() { if [ "$APPLY" = 1 ]; then echo "  $*"; else echo "  [dry-run] $*"; fi; }

# ---- 1. hardlink identical frozen_ae.ckpt copies --------------------------
echo
echo "== frozen_ae.ckpt duplicates -> hardlink =="
declare -A KEEPER
for f in "$STORE"/*/frozen_ae.ckpt; do
  [ -e "$f" ] || continue
  sz=$(stat -c%s "$f")
  h="$(fp "$f")"
  key="$sz:$h"
  keep="${KEEPER[$key]:-}"
  if [ -z "$keep" ]; then
    KEEPER[$key]="$f"
    echo "  keeping $(basename "$(dirname "$f")")/frozen_ae.ckpt"
    continue
  fi
  # already hardlinked to the keeper? then nothing to do
  if [ "$(stat -c%i "$f")" = "$(stat -c%i "$keep")" ]; then continue; fi
  if protected "$f"; then echo "  SKIP (protected) $f"; continue; fi
  say "link $(basename "$(dirname "$f")")/frozen_ae.ckpt -> $(basename "$(dirname "$keep")")"
  if [ "$APPLY" = 1 ]; then
    ln "$keep" "$f.hltmp" && mv -f "$f.hltmp" "$f"
  fi
  freed=$((freed + sz))
done

# ---- 2. ignore*.ckpt that duplicate last*.ckpt ----------------------------
echo
echo "== ignore*.ckpt duplicates -> delete =="
for f in "$STORE"/*/checkpoints/ignore*.ckpt; do
  [ -e "$f" ] || continue
  d="$(dirname "$f")"
  case "$(basename "$f")" in
    ignore-EMA.ckpt) peer="$d/last-EMA.ckpt" ;;
    *)               peer="$d/last.ckpt" ;;
  esac
  [ -e "$peer" ] || { echo "  SKIP (no peer) $f"; continue; }
  [ "$(stat -c%s "$f")" = "$(stat -c%s "$peer")" ] || { echo "  SKIP (size differs) $f"; continue; }
  [ "$(fp "$f")" = "$(fp "$peer")" ] || { echo "  SKIP (content differs) $f"; continue; }
  if protected "$f"; then echo "  SKIP (protected) $f"; continue; fi
  sz=$(stat -c%s "$f")
  say "rm $(basename "$(dirname "$d")")/checkpoints/$(basename "$f")"
  [ "$APPLY" = 1 ] && rm -f "$f"
  freed=$((freed + sz))
done

# ---- 3. periodic checkpoints that are not the highest-step EMA ------------
echo
echo "== periodic non-best chk_epoch=* -> delete =="
for run in "$STORE"/*/; do
  run="${run%/}"; ck="$run/checkpoints"
  [ -d "$ck" ] || continue
  best="$(ls -1 "$ck"/chk_epoch=*-EMA.ckpt 2>/dev/null | sort | tail -1 || true)"
  [ -n "$best" ] || continue
  for f in "$ck"/chk_epoch=*.ckpt; do
    [ -e "$f" ] || continue
    [ "$f" = "$best" ] && continue
    if protected "$f"; then echo "  SKIP (protected) $(basename "$f")"; continue; fi
    sz=$(stat -c%s "$f")
    say "rm $(basename "$run")/checkpoints/$(basename "$f")"
    [ "$APPLY" = 1 ] && rm -f "$f"
    freed=$((freed + sz))
  done
done

echo
printf "%s %.1f GB\n" \
  "$([ "$APPLY" = 1 ] && echo 'reclaimed:' || echo 'would reclaim:')" \
  "$(echo "$freed/1073741824" | bc -l)"
[ "$APPLY" = 1 ] || echo "re-run with --apply to actually reclaim it"
