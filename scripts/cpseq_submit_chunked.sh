#!/usr/bin/env bash
# Submit a range of array indices in whatever chunks the QOS submit cap currently allows.
#
# Why this exists. This cluster caps submitted (pending + running) array TASKS per user,
# and the cap is not a round number you can hardcode -- measured here, a 5-task array was
# rejected at 72 queued tasks while a 2-task array went through. So the usable chunk size
# is a property of the moment, not of the script, and the only reliable way to find it is
# to ask sbatch and back off.
#
# Retrying the whole range as one array (what a naive watcher does) stalls indefinitely:
# it needs the full width free at one instant, which never happens while the queue is
# busy. This submits the largest chunk that fits, right now, and comes back for the rest
# -- so the work starts draining immediately instead of waiting for a gap that may not
# come. Phase B over ~391 complexes is where this stops being a convenience.
#
# Every chunk is a separate job id covering a disjoint slice of PARENTS_FILE, which is
# safe: the ladder's unit of work is one complex per array task precisely so that no two
# tasks ever write the same journal.
#
# Usage:
#   cpseq_submit_chunked.sh <first> <last> <interval_s> -- sbatch args...
# The script appends `--array=<lo>-<hi>` itself; do not pass one.

set -uo pipefail

FIRST="${1:?first array index}"; shift
LAST="${1:?last array index}"; shift
INTERVAL="${1:?interval seconds}"; shift
[ "${1:-}" = "--" ] && shift

next="$FIRST"
while [ "$next" -le "$LAST" ]; do
  remaining=$(( LAST - next + 1 ))
  placed=0
  # Largest first, so a quiet queue is filled in one submission rather than many.
  for width in 16 8 4 2 1; do
    [ "$width" -gt "$remaining" ] && continue
    hi=$(( next + width - 1 ))
    OUT=$(sbatch --parsable --array="${next}-${hi}" "$@" 2>&1)
    if [ $? -eq 0 ]; then
      echo "[$(date '+%H:%M:%S')] job $OUT  <- array ${next}-${hi} (${width} task(s))"
      next=$(( hi + 1 )); placed=1; break
    fi
    if ! grep -q "QOSMaxSubmitJobPerUserLimit" <<<"$OUT"; then
      echo "[$(date '+%H:%M:%S')] FATAL: sbatch failed for a reason that is not the cap:" >&2
      echo "$OUT" >&2
      exit 1
    fi
  done
  if [ "$placed" -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] capped at $(squeue -u "$USER" -h -r | wc -l) queued; ${remaining} task(s) left, waiting ${INTERVAL}s"
    sleep "$INTERVAL"
  fi
done

echo "[$(date '+%H:%M:%S')] all array indices ${FIRST}-${LAST} submitted"
