#!/usr/bin/env bash
# Snapshot the COMMON FORK -- the warmup run's checkpoint that all five one-pass arms
# start from.
#
# WHY A PIN, same reasoning as scripts/pin_cp2lp_init.sh but with more riding on it. The
# whole design of this tranche is that the five arms differ by ONE config line, so they
# must start from one set of weights: the editor's, the trunk's, AND the discriminator's
# random initialisation (which is built in every arm regardless of `adversarial.enabled`
# and therefore lands in this checkpoint). `last.ckpt` is rewritten every 250 steps, so
# five arms launched minutes apart would otherwise fork from different weights and every
# addon result would have "a different warmup" as a live alternative explanation.
#
# RAW WEIGHTS, NOT EMA, on purpose. The arms CONTINUE training from here; resuming from an
# EMA snapshot would restart optimisation from an average of the last few hundred updates
# rather than from the trajectory itself. EMA is for export and evaluation, which is where
# `cp2lp_generate.py` reads `-EMA` files.
#
# Usage:
#   bash scripts/pin_cp2lp_fork.sh                       # pin the warmup's last.ckpt
#   bash scripts/pin_cp2lp_fork.sh /path/to/other.ckpt
#   FORCE=1 bash scripts/pin_cp2lp_fork.sh               # overwrite an existing pin
set -euo pipefail

STORE="${STORE:-/zfsauton/scratch/yixiz/Proteina-Complexa/training_runs/store}"
FORK_RUN="${FORK_RUN:-cp2lp_onepass_fork}"
CKPT_DIR="${STORE}/${FORK_RUN}/checkpoints"

# PREFER THE G-UPDATE MILESTONE over `last.ckpt`, and the difference is not cosmetic.
#
# `last.ckpt` is written by `last_ckpt_every_n_steps` (250), which triggers on
# `global_step`; the warmup STOPS at `max_g_updates` (300). So `last.ckpt` is whatever the
# last periodic save caught -- g=250 -- while `g000300.ckpt` is the run's actual endpoint.
# Pinning `last.ckpt` would make the docs' "the fork is the g=300 evaluation point" false,
# and the arms would inherit 250 and get 10,050 post-warmup updates against a budget that
# reads 10300.
#
# It would not break the COMPARISON (all five arms inherit the same number either way), but
# it would quietly make the budget arithmetic wrong, and that is the sort of thing that is
# discovered while reading a results table. Highest-numbered milestone wins.
if [[ -z "${1:-}" ]]; then
  # `|| true` is load-bearing under `set -e`: with no match, `ls` exits 2, and a failing
  # command substitution inside an assignment aborts the script -- so this died with a bare
  # exit 2 before reaching the "run the warmup first" message below, which is the single
  # most useful line in the file.
  SRC="$(ls -1 "${CKPT_DIR}"/g[0-9][0-9][0-9][0-9][0-9][0-9].ckpt 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "${SRC}" ]]; then
    echo "[pin] using the G-update milestone: $(basename "${SRC}")"
  else
    SRC="${CKPT_DIR}/last.ckpt"
    echo "[pin] NOTE: no g??????.ckpt milestone found; falling back to last.ckpt."
    echo "[pin]       That file is written on a global_step cadence, so it is probably a"
    echo "[pin]       few updates SHORT of max_g_updates. Harmless for the comparison (all"
    echo "[pin]       arms inherit the same count) but the budget arithmetic in the docs"
    echo "[pin]       assumes the milestone. Check g_updates_in_fork below."
  fi
else
  SRC="$1"
fi
DST="${DST:-${STORE}/${FORK_RUN}/${FORK_RUN}.ckpt}"

[[ -f "$SRC" ]] || {
  echo "ERROR: fork checkpoint not found: $SRC" >&2
  echo "       Run the warmup first: bash scripts/submit_cp2lp_onepass.sh --submit warmup" >&2
  exit 1
}

if [[ -f "$DST" && "${FORCE:-0}" != "1" ]]; then
  echo "[pin] reusing existing fork pin: $DST"
  [[ -f "${DST}.source" ]] && sed 's/^/[pin]   /' "${DST}.source"
  echo "[pin] set FORCE=1 to re-pin. Re-pinning after any arm has started makes the arms incomparable."
  exit 0
fi

before="$(stat -c '%Y %s' "$SRC")"
cp "$SRC" "${DST}.tmp"
sync
after="$(stat -c '%Y %s' "$SRC")"
if [[ "$before" != "$after" ]]; then
  rm -f "${DST}.tmp"
  echo "ERROR: $SRC changed during the copy (was [$before], now [$after])." >&2
  echo "       The warmup job is still writing. Wait for it to finish and retry." >&2
  exit 1
fi
mv "${DST}.tmp" "$DST"

# The arms inherit `g_updates` from this checkpoint (it is a persistent buffer), which is
# why their budgets read 10300 rather than 10000. Recording it makes that offset checkable
# instead of folklore.
g_updates="$(
  "${PY:-/zfsauton2/home/yixiz/Proteina-Complexa/.venv/bin/python}" - "$DST" <<'PYEOF' 2>/dev/null || echo "unknown"
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(sd["state_dict"].get("g_updates", -1)))
PYEOF
)"

{
  echo "$SRC"
  date -u +"pinned_utc=%Y-%m-%dT%H:%M:%SZ"
  echo "size=$(stat -c '%s' "$DST")"
  echo "g_updates_in_fork=${g_updates}"
} > "${DST}.source"

echo "[pin] wrote $DST"
sed 's/^/[pin]   /' "${DST}.source"
echo "[pin] the five arms inherit g_updates=${g_updates}; their max_g_updates is that + 10000."
