#!/usr/bin/env bash
# Push the cyclic-peptide sequence-search results from babel to Backblaze B2.
#
# This is the babel -> auton leg, and it is the same shape as the rhea -> babel
# leg in `rhea_push_b2.sh`: upload to B2 and let the destination pull whenever a
# node frees up, rather than holding a direct transfer open between two
# clusters whose login nodes cannot both see the data.
#
# Why tar.zst and not `rclone copy`
# ---------------------------------
# The run tree is hundreds of thousands of files of a few KB each. Measured on
# `es_corpus_train`: 7,158 design npz at ~3.3 KB, 6,359 score JSON at ~5.8 KB,
# 852 prediction directories holding a 651-byte input.yaml and a 617-byte
# record.json. Per-object overhead would dominate a file-by-file upload, and B2
# charges per transaction. The same files are also NFS-block-padded on babel --
# `es_corpus_train` reads as 714 MB of disk for 127 MB of bytes -- so `du`
# overstates what actually has to move by roughly 5x.
#
# They are JSON and JSONL, so zstd -3 earns its keep on the way out.
#
# Resumable: one object per tier per run directory, and an object that already
# exists is skipped. An interrupted run is restarted with the identical line.
# Nothing is staged on local disk -- tar streams through zstd into `rclone
# rcat` -- so this does not care how full /data is.
#
# Usage:
#   bash scripts/cpseq_push_b2.sh b2:proteina/cpseq            # core, then folds
#   bash scripts/cpseq_push_b2.sh b2:proteina/cpseq core       # the journals only
#   bash scripts/cpseq_push_b2.sh b2:proteina/cpseq folds
#   REBUILD=1 bash scripts/cpseq_push_b2.sh b2:proteina/cpseq  # re-scan the tree

set -euo pipefail

DEST="${1:-}"
WHAT="${2:-all}"
[ -n "$DEST" ] || { echo "usage: $0 b2:bucket/prefix [core|folds|all]" >&2; exit 1; }

CPSEQ="${CPSEQ:-/data/user_data/yixiz/cpseq}"
LISTS="${LISTS:-$CPSEQ/.push_b2}"
ZLEVEL="${ZLEVEL:-3}"
ZTHREADS="${ZTHREADS:-2}"

command -v rclone >/dev/null || { echo "FATAL: rclone not on PATH" >&2; exit 1; }
command -v zstd >/dev/null || { echo "FATAL: zstd not on PATH" >&2; exit 1; }
[ -d "$CPSEQ" ] || { echo "FATAL: $CPSEQ does not exist -- is /data mounted here?" >&2; exit 1; }

# A compute node may have no outbound internet even where the login node does.
# Find that out before pretending to make progress. Note the bucket-scoped
# form: `rclone lsd b2:` fails on a bucket-scoped key with a permissions error
# that reads like a misconfigured remote.
rclone lsd "$DEST" >/dev/null 2>&1 || rclone lsd "${DEST%%/*}" >/dev/null 2>&1 || {
  echo "FATAL: cannot reach ${DEST%%/*} from $(hostname)." >&2
  echo "  configure with: rclone config create b2 b2 account <keyID> key <applicationKey>" >&2
  exit 1
}

echo "=== $(date) $(hostname): $CPSEQ -> $DEST (tier=$WHAT) ==="
mkdir -p "$LISTS"

# ---- what counts as a sequence, and what counts as a fold -----------------
# core:  every sequence that was proposed, what happened to it, and every
#        number computed from it. `state/` holds the per-target journals
#        (candidates.jsonl, outcomes.jsonl) -- the record of what was tried and
#        why it was kept or dropped, which nothing regenerates. `scores/` is
#        keyed by scoring version, `interfaces/` and `llm_proposals/` are the
#        text the LLM arm read and wrote, and the loose *.json/*.jsonl at the
#        top of each run are the summaries the reports in docs/ are built from.
# folds: `predictions/` (one input.yaml + record.json per fold) and `designs/`
#        (the retained peptide coordinates as npz). Bulkier, and a fold can in
#        principle be recomputed on a GPU -- at the cost of the GPU-hours, and
#        with Boltz-2's seed noise meaning a recomputed fold is not the same
#        fold. Worth carrying, but second in line.
CORE_DIRS='state scores interfaces llm_proposals'
FOLD_DIRS='predictions designs'

build_lists() {
  echo "--- scanning $CPSEQ (one pruned traversal; cached in $LISTS) ---"

  # Scan into a scratch directory and publish atomically at the end. A scan
  # killed partway must leave NO usable cache: on 2026-11-01 job 11004355 was
  # cut off by its time limit midway through the directory scan, which left a
  # populated core.files next to an empty core.dirs. The old freshness test
  # below was `-s core.files`, so the next two runs happily reused that
  # half-written plan and uploaded the loose summaries and nothing else,
  # reporting "0 directories expected" as though the tree were empty. Lists
  # are now published only once complete, and freshness is a marker file.
  local tmp="$LISTS/.scan.$$"
  rm -rf "$tmp"; mkdir -p "$tmp"

  # Shallow pass: loose summaries at the top of the tree and of each run
  # directory -- reports, cohort definitions, ladder/screen summaries, probes.
  find "$CPSEQ" -maxdepth 2 -type f \
    \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.tsv' \
       -o -name '*.txt' -o -name '*.log' \) \
    -not -path "$LISTS/*" -printf '%P\n' > "$tmp/core.files" 2>/dev/null || true

  # Deep pass: ONE traversal for every result directory name, with -prune so
  # the walk stops at each match instead of descending through the hundreds of
  # thousands of small files inside it. The previous version ran a separate
  # full-tree `find` per directory name -- six traversals of a 47 GB NFS tree,
  # each walking into every match. That is what ran past the time limit.
  local cexpr=() fexpr=()
  for d in $CORE_DIRS; do cexpr+=( -o -name "$d" ); done
  for d in $FOLD_DIRS; do fexpr+=( -o -name "$d" ); done
  find "$CPSEQ" -path "$LISTS" -prune -o \
    \( -type d \( "${cexpr[@]:1}" \) -printf 'C %P\n' -prune \) -o \
    \( -type d \( "${fexpr[@]:1}" \) -printf 'F %P\n' -prune \) \
    > "$tmp/dirs.raw" 2>/dev/null || true

  sed -n 's/^C //p' "$tmp/dirs.raw" > "$tmp/core.dirs"
  sed -n 's/^F //p' "$tmp/dirs.raw" > "$tmp/folds.dirs"
  : > "$tmp/folds.files"

  # `transfer/` is the staged HighFold3 input set: small, and it is what a
  # second predictor would be re-run from, so it travels with the core.
  [ -d "$CPSEQ/transfer" ] && echo "transfer" >> "$tmp/core.dirs"

  sort -u -o "$tmp/core.dirs" "$tmp/core.dirs"
  sort -u -o "$tmp/folds.dirs" "$tmp/folds.dirs"
  sort -u -o "$tmp/core.files" "$tmp/core.files"

  # A scan that found no result directories at all is far more likely to be a
  # broken scan than an empty tree. Refuse to publish it as a plan.
  if [ ! -s "$tmp/core.dirs" ] && [ ! -s "$tmp/folds.dirs" ]; then
    echo "FATAL: scan found no $CORE_DIRS or $FOLD_DIRS directories under $CPSEQ." >&2
    echo "  Not publishing an empty plan. Check the tree layout before retrying." >&2
    ls -la "$CPSEQ" | head -20 >&2
    rm -rf "$tmp"
    return 1
  fi

  for f in core.files core.dirs folds.dirs folds.files; do
    mv -f "$tmp/$f" "$LISTS/$f"
  done
  rm -rf "$tmp"
  date -u +%FT%TZ > "$LISTS/.scan_complete"

  echo "    core: $(wc -l < "$LISTS/core.files") loose files, $(wc -l < "$LISTS/core.dirs") dirs"
  echo "    folds: $(wc -l < "$LISTS/folds.dirs") dirs"
}

if [ "${REBUILD:-0}" = "1" ] || [ ! -f "$LISTS/.scan_complete" ]; then
  rm -f "$LISTS/.scan_complete"
  build_lists
else
  echo "--- reusing cached lists in $LISTS from $(cat "$LISTS/.scan_complete") (REBUILD=1 to re-scan) ---"
fi

existing="$(rclone lsf "$DEST" --recursive 2>/dev/null || true)"

# Stream one directory (or one file list) into a single object. Nothing hits
# local disk, so a full /data is irrelevant.
push_obj() {
  local obj="$1"; shift
  if printf '%s\n' "$existing" | grep -qxF "$obj"; then
    echo "    skip $obj (already there)"
    return 0
  fi
  echo "    push $obj  $(date +%H:%M:%S)"
  # --ignore-failed-read: a job may still be writing into the tree, and one
  # unreadable file must not abort an otherwise complete object.
  tar -C "$CPSEQ" --ignore-failed-read -cf - "$@" \
    | zstd -"$ZLEVEL" -T"$ZTHREADS" \
    | rclone rcat "$DEST/$obj"
}

# ---- core -----------------------------------------------------------------
if [ "$WHAT" = "all" ] || [ "$WHAT" = "core" ]; then
  echo
  echo "--- core: sequence journals, scores, interfaces, summaries ---"
  push_obj "core/loose.tar.zst" -T "$LISTS/core.files"
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    [ -d "$CPSEQ/$d" ] || continue
    push_obj "core/$(echo "$d" | tr '/' '_').tar.zst" "$d"
  done < "$LISTS/core.dirs"
  # The lists themselves, so the far side knows what was meant to arrive.
  rclone copy "$LISTS" "$DEST/manifests/" -P 2>/dev/null || true
  echo "core done"
fi

# ---- folds ----------------------------------------------------------------
if [ "$WHAT" = "all" ] || [ "$WHAT" = "folds" ]; then
  echo
  echo "--- folds: prediction records and design coordinates ---"
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    [ -d "$CPSEQ/$d" ] || continue
    push_obj "folds/$(echo "$d" | tr '/' '_').tar.zst" "$d"
  done < "$LISTS/folds.dirs"
  echo "folds done"
fi

echo
echo "=== $(date) finished ==="
rclone size "$DEST" 2>/dev/null || true
