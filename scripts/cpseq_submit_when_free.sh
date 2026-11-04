#!/usr/bin/env bash
# Retry one sbatch until the QOS submit limit lets it through.
#
# This cluster caps submitted (pending + running) array tasks per user, and a sweep whose
# arms are submitted in a loop will have its LAST arms rejected while the earlier ones
# sit in the queue. Dropping the rejected arm is the thing not to do: in a filler sweep
# the arm that happens to be last is as load-bearing as any other, and a sweep missing
# its control answers a different question than the one asked.
#
# Only QOSMaxSubmitJobPerUserLimit is retried. Any other sbatch failure is a real error
# and exits non-zero rather than spinning forever on a bad command line.
#
# Usage: cpseq_submit_when_free.sh <interval_s> <max_tries> -- sbatch args...

set -uo pipefail

INTERVAL="${1:?interval seconds}"; shift
MAX_TRIES="${1:?max tries}"; shift
[ "${1:-}" = "--" ] && shift

for try in $(seq 1 "$MAX_TRIES"); do
  OUT=$(sbatch --parsable "$@" 2>&1)
  RC=$?
  if [ "$RC" -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] submitted job $OUT (try $try)"
    exit 0
  fi
  if ! grep -q "QOSMaxSubmitJobPerUserLimit" <<<"$OUT"; then
    echo "[$(date '+%H:%M:%S')] FATAL: sbatch failed for a reason that is not the submit cap:" >&2
    echo "$OUT" >&2
    exit 1
  fi
  echo "[$(date '+%H:%M:%S')] try $try/$MAX_TRIES: at the submit cap ($(squeue -u "$USER" -h -r | wc -l) tasks queued); waiting ${INTERVAL}s"
  sleep "$INTERVAL"
done

echo "[$(date '+%H:%M:%S')] FATAL: still capped after $MAX_TRIES tries" >&2
exit 1
