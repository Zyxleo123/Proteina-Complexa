#!/usr/bin/env bash
# Build the file list to carry to the next cluster.
#
# The checkpoint store is ~700 GB because nothing ever prunes it: every run
# keeps every periodic checkpoint plus its EMA twin. Almost none of that is
# worth moving. This script keeps ONE checkpoint per run -- the highest-step
# EMA, which is what every eval actually loads -- plus the tiny config JSONs
# that say how the run was configured.
#
# Output (written to $OUTDIR, default $ZFS/migration):
#   tier0.files   rsync --files-from list, paths relative to $ZFS
#   tier0.sizes   same entries with sizes, largest first, for eyeballing
#   tier0.total   one line: the total transfer size
#
# Usage:
#   bash scripts/migration_manifest.sh            # real runs only
#   KEEP_SMOKE=1 bash scripts/migration_manifest.sh   # include smoke/overfit
#
# Nothing is copied here. This only writes lists.

set -euo pipefail

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
STORE="$ZFS/Proteina-Complexa/training_runs/store"
OUTDIR="${OUTDIR:-$ZFS/migration}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"

mkdir -p "$OUTDIR"
FILES="$OUTDIR/tier0.files"
: > "$FILES"

# Runs that only ever existed to prove a job launches. Dropping these is the
# difference between ~700 GB and something you can move in an afternoon.
SKIP_RE='(_smoke|_local$|^valgen_|^ringpe_bond_smoke$|^cpsea_overfit)'

emit() { # path -> append as $ZFS-relative if it exists
  [ -e "$1" ] || return 0
  printf '%s\n' "${1#"$ZFS"/}" >> "$FILES"
}

echo "scanning $STORE ..."
nkept=0; nskip=0
for run in "$STORE"/*/; do
  run="${run%/}"
  [ -d "$run" ] || continue
  name="$(basename "$run")"
  if [ "$KEEP_SMOKE" != "1" ] && printf '%s' "$name" | grep -qE "$SKIP_RE"; then
    nskip=$((nskip + 1)); continue
  fi

  # Records which AE this run was frozen against. A few hundred bytes, and
  # without it you cannot tell which AE to evaluate a flow checkpoint with.
  emit "$run/frozen_ae_source.txt"

  ck="$run/checkpoints"
  [ -d "$ck" ] || continue

  # Highest-step EMA checkpoint. The chk_epoch=...step=... names sort
  # lexicographically in step order because the digits are zero-padded, so a
  # plain sort is already numeric order. Fall back to last-EMA, then last.
  best="$(ls -1 "$ck"/chk_epoch=*-EMA.ckpt 2>/dev/null | sort | tail -1 || true)"
  [ -n "$best" ] || best="$(ls -1 "$ck"/last-EMA.ckpt 2>/dev/null | tail -1 || true)"
  [ -n "$best" ] || best="$(ls -1 "$ck"/last.ckpt 2>/dev/null | tail -1 || true)"

  if [ -z "$best" ]; then
    echo "  !! no checkpoint found in $name" >&2
    continue
  fi

  emit "$best"
  # Configs are a few KB each and make the checkpoint reloadable.
  for j in "$ck"/*.json; do emit "$j"; done
  nkept=$((nkept + 1))
done
echo "  kept $nkept runs, skipped $nskip smoke/overfit runs"

# Frozen-AE copies. 23 runs each hold a full 4.1 GB copy of the AE they were
# trained against, and 20 of those are byte-for-byte the same pinned
# finetune_full_128 EMA checkpoint -- ~80 GB of pure duplication. Carry one
# copy per DISTINCT size. That matters beyond saving space: at least one of
# the distinct copies is an orphan whose source run has since been deleted,
# so it is the only surviving copy of that AE.
echo "deduplicating frozen AE copies ..."
declare -A seen_ae
ndup=0
for f in "$STORE"/*/frozen_ae.ckpt; do
  [ -e "$f" ] || continue
  sz="$(stat -c%s "$f")"
  if [ -n "${seen_ae[$sz]:-}" ]; then
    ndup=$((ndup + 1)); continue
  fi
  seen_ae[$sz]=1
  emit "$f"
  echo "  keeping $(basename "$(dirname "$f")")/frozen_ae.ckpt  ($sz bytes)"
done
echo "  skipped $ndup duplicate copies"

# Small, irreplaceable, and cheap to carry: every metrics/results artefact.
# These are the numbers behind the write-ups; none of it can be regenerated
# without the GPU time that produced it.
echo "collecting results and metrics ..."
for d in \
  "$ZFS/Proteina-Complexa/training_runs/store" \
  "$ZFS/lpgen/runs" \
  "$ZFS/lpgen/logs"
do
  [ -d "$d" ] || continue
  find "$d" \
    \( -name '*.jsonl' -o -name '*.json' -o -name '*.csv' -o -name '*.npz' \) \
    -size -200M -print 2>/dev/null \
  | while read -r f; do printf '%s\n' "${f#"$ZFS"/}"; done >> "$FILES"
done

# The released third-party weights are re-downloadable, but the manifest that
# records WHICH revision was fetched is not.
emit "$ZFS/lpgen/weights/weights_manifest.json"

sort -u -o "$FILES" "$FILES"

echo "sizing ..."
( cd "$ZFS" && du -ch --files0-from=<(tr '\n' '\0' < "$FILES") 2>/dev/null | tail -1 ) \
  > "$OUTDIR/tier0.total" || echo "unknown" > "$OUTDIR/tier0.total"

( cd "$ZFS" && while read -r f; do
    [ -e "$f" ] && printf '%s\t%s\n' "$(stat -c%s "$f")" "$f"
  done < "$FILES" | sort -rn | head -60 ) > "$OUTDIR/tier0.sizes"

echo
echo "entries : $(wc -l < "$FILES")"
echo "total   : $(cat "$OUTDIR/tier0.total")"
echo "list    : $FILES"
echo "largest : $OUTDIR/tier0.sizes"
