#!/usr/bin/env bash
# Snapshot the CP->LP generator's flow initialisation to a stable path.
#
# WHY A PIN. The lpmix run this initialises from may still be training, and it rewrites
# `last-EMA.ckpt` every `last_ckpt_every_n_steps`. Pointing an arm directly at that file
# means two arms launched an hour apart silently start from different weights -- which
# destroys the one thing the gan/contactonly comparison depends on -- and a job that reads
# during a write loads a torn file.
#
# Also keeps `=` out of the filename: Hydra's override grammar mis-parses those, which is
# why the AE pin exists in the same form.
#
# Usage:
#   bash scripts/pin_cp2lp_init.sh                       # pin lpmix last-EMA
#   bash scripts/pin_cp2lp_init.sh /path/to/other.ckpt   # pin something else
#   FORCE=1 bash scripts/pin_cp2lp_init.sh               # overwrite an existing pin
set -euo pipefail

STORE="${STORE:-/zfsauton/scratch/yixiz/Proteina-Complexa/training_runs/store}"
SRC="${1:-${STORE}/cpsea_lpmix_from_v4cfg/checkpoints/last-EMA.ckpt}"
DST="${DST:-${STORE}/cpsea_lpmix_from_v4cfg/cp2lp_init.ckpt}"

[[ -f "$SRC" ]] || { echo "ERROR: source checkpoint not found: $SRC" >&2; exit 1; }

if [[ -f "$DST" && "${FORCE:-0}" != "1" ]]; then
  echo "[pin] reusing existing pin: $DST"
  [[ -f "${DST}.source" ]] && sed 's/^/[pin]   /' "${DST}.source"
  echo "[pin] set FORCE=1 to re-pin. Re-pinning mid-experiment makes arms incomparable."
  exit 0
fi

before="$(stat -c '%Y %s' "$SRC")"
cp "$SRC" "${DST}.tmp"
sync
after="$(stat -c '%Y %s' "$SRC")"

# A checkpoint rewritten while we copied it gives a torn file that loads with plausible
# shapes and wrong weights. Detect it instead of shipping it.
if [[ "$before" != "$after" ]]; then
  rm -f "${DST}.tmp"
  echo "ERROR: $SRC changed during the copy (was [$before], now [$after])." >&2
  echo "       The training job just wrote a new checkpoint. Wait and retry." >&2
  exit 1
fi

mv "${DST}.tmp" "$DST"
{
  echo "$SRC"
  date -u +"pinned_utc=%Y-%m-%dT%H:%M:%SZ"
  echo "size=$(stat -c '%s' "$DST")"
} > "${DST}.source"

echo "[pin] wrote $DST"
sed 's/^/[pin]   /' "${DST}.source"
