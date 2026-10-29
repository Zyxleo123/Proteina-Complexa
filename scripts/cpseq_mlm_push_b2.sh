#!/usr/bin/env bash
# Push the cpseq_mlm sequence-proposer artefacts to Backblaze B2.
#
# Same arrangement as scripts/rhea_push_b2.sh and for the same reason: B2 decouples
# "get it off this cluster" from "the other cluster receives it", so neither waits on the
# other's queue or login-node write permissions.
#
# WHAT GOES, AND WHY EACH TIER EXISTS. The run directory is 44 GB, most of which is not
# worth moving:
#
#   corpus    386 MB  The extracted (pocket, peptide, chemistry) JSONL. Small, and it cost
#                     108 core-hours to build (24 shards x 4.5 h), so it is by far the best
#                     value per byte here. Rebuilding it needs the CPSea PDB tree, which is
#                     2.44M files and a far bigger migration than this.
#   weights   2.5 GB  per arm. The HF-format directory -- what you need to PROPOSE or
#                     evaluate. This is the tier to take if you take one thing.
#   results   ~1 MB   manifest.json + metrics.jsonl per arm. Every number in the writeup is
#                     redrawable from these without a GPU.
#   resume    7.3 GB  per arm. `last.pt` = weights + Adam moments, needed only to CONTINUE
#                     training. Skipped by default: it is 3x the size of the weights and is
#                     useless for inference.
#
# Smoke runs (`*_smoke`) are never pushed -- they are 200-step throwaways.
#
# Resumable: rclone skips objects that already match, so a rerun costs a listing.
#
# Setup (once, if `rclone listremotes` does not show b2:):
#   rclone config create b2 b2 account <keyID> key <applicationKey>
#   rclone lsd b2:
#
# Usage:
#   bash scripts/cpseq_mlm_push_b2.sh b2:my-bucket/cpseq_mlm            # corpus+weights+results
#   bash scripts/cpseq_mlm_push_b2.sh b2:my-bucket/cpseq_mlm weights    # weights+results only
#   bash scripts/cpseq_mlm_push_b2.sh b2:my-bucket/cpseq_mlm all        # + last.pt (adds 15 GB)

set -euo pipefail

DEST="${1:-}"
WHAT="${2:-default}"
[ -n "$DEST" ] || {
  echo "usage: $0 b2:bucket/prefix [weights|corpus|default|all]" >&2; exit 1; }

ROOT="${CPSEQ_MLM_ROOT:-/zfsauton/scratch/yixiz/cpseq_mlm}"
RUNS="$ROOT/runs"
TRANSFERS="${TRANSFERS:-16}"
# Production arms only. A glob would pick up the smoke runs.
ARMS="${ARMS:-cpseq_mlm_pocket cpseq_mlm_nopocket}"

command -v rclone >/dev/null || { echo "rclone not found" >&2; exit 1; }
rclone lsd "${DEST%%:*}:" >/dev/null 2>&1 || {
  echo "FATAL: cannot reach remote '${DEST%%:*}:' -- is it configured?" >&2
  echo "  rclone config create b2 b2 account <keyID> key <applicationKey>" >&2
  exit 1
}

echo "=== $(date) cpseq_mlm -> $DEST  (tier: $WHAT) ==="

push_results () {
  for arm in $ARMS; do
    [ -d "$RUNS/$arm" ] || { echo "  skip $arm (no run dir)"; continue; }
    echo "--- results: $arm"
    rclone copy "$RUNS/$arm" "$DEST/runs/$arm" \
      --include "manifest.json" --include "metrics.jsonl" -P
  done
}

push_weights () {
  for arm in $ARMS; do
    # The LAST hf_step* by numeric step, not by mtime: a resumed run rewrites older
    # directories' mtimes, and `ls -t` would then pick a stale checkpoint.
    local latest
    latest="$(ls -d "$RUNS/$arm"/checkpoints/hf_step* 2>/dev/null \
              | sed 's/.*hf_step//' | sort -n | tail -1)"
    [ -n "$latest" ] || { echo "  skip $arm (no hf_step* checkpoint)"; continue; }
    echo "--- weights: $arm @ step $latest (2.5 GB)"
    rclone copy "$RUNS/$arm/checkpoints/hf_step$latest" \
      "$DEST/runs/$arm/checkpoints/hf_step$latest" --transfers "$TRANSFERS" -P
    # Record which step is the published one, so the receiving side does not have to
    # guess when several checkpoints are present.
    echo "$latest" | rclone rcat "$DEST/runs/$arm/PUBLISHED_STEP"
  done
}

push_corpus () {
  echo "--- corpus (386 MB)"
  rclone copy "$ROOT/corpus" "$DEST/corpus" \
    --include "*.jsonl" --include "*.stats.json" --transfers "$TRANSFERS" -P
}

push_resume () {
  for arm in $ARMS; do
    [ -f "$RUNS/$arm/checkpoints/last.pt" ] || continue
    echo "--- resume state: $arm (7.3 GB)"
    rclone copy "$RUNS/$arm/checkpoints/last.pt" \
      "$DEST/runs/$arm/checkpoints/" --transfers "$TRANSFERS" -P
  done
}

case "$WHAT" in
  weights)  push_results; push_weights ;;
  corpus)   push_corpus ;;
  default)  push_results; push_weights; push_corpus ;;
  all)      push_results; push_weights; push_corpus; push_resume ;;
  *) echo "unknown tier '$WHAT' (weights|corpus|default|all)" >&2; exit 1 ;;
esac

# The migration instructions travel WITH the data. A reader who finds the bucket and not
# the repo still needs to know what the pocket format is and which venv trap to expect.
if [ -f "$(dirname "$0")/../docs/README_CPSEQ_MLM_MIGRATION.md" ]; then
  rclone copy "$(dirname "$0")/../docs/README_CPSEQ_MLM_MIGRATION.md" "$DEST/" -P
fi
[ -f "$ROOT/RUN_STATUS.md" ] && rclone copy "$ROOT/RUN_STATUS.md" "$DEST/" -P

echo
echo "=== done. Verify from the other side with:"
echo "  rclone size   $DEST"
echo "  rclone tree   $DEST --level 3"
echo "  rclone check  $DEST/corpus $ROOT/corpus --one-way   # on THIS cluster"
