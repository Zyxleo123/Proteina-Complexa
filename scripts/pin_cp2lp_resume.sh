#!/usr/bin/env bash
# Snapshot a RESUME point so two continuations can fork from bit-identical state.
#
# WHY THIS EXISTS, AND WHY `pin_cp2lp_fork.sh` COULD NOT BE REUSED.
#
# `pin_cp2lp_fork.sh` pins a checkpoint that the arms then load through
# `pretrain_ckpt_path`, and train.py's comment on that branch is explicit: "load pre-trained
# checkpoint and DON'T load optimizer states". That is the right behaviour for the warmup
# fork, where every arm is meant to start optimisation fresh from one set of weights.
#
# It is the WRONG behaviour for a mid-run A/B. The clash-normalisation arm forks from a run
# that is 10,300 generator updates deep; loading it as a pretrain would discard the Adam
# moments and restart the LR warmup, so the arm would differ from its control in the loss
# normalisation AND in its optimiser state. MEASURED: both `last.ckpt` and `g010300.ckpt`
# carry `optimizer_states`, `lr_schedulers` and `loops` (checked by listing the zip's
# data.pkl without loading a single tensor -- these files are ~3 GB and this shell has
# ~1.4 GB of headroom, so `torch.load` here is an exit 137).
#
# So the fork has to happen through Lightning's RESUME path instead: seed the new run's own
# checkpoint directory with a copy of the source run's `last.ckpt`, and
# `training.resume_from_last=true` picks it up with the optimiser state intact.
#
# WHICH FILE. The control (job 70232) resumes `cp2lp_onepass_internal` from its OWN
# `last.ckpt`, so that is the file to copy -- not `g010300.ckpt`. The milestone is ~50
# g-updates further on (`last.ckpt` is written on a `global_step` cadence of 250, the
# milestone exactly at the budget), and forking the new arm from the milestone while the
# control forks from `last.ckpt` would start the two arms at different weights.
#
# TIMING IS LOAD-BEARING. `last.ckpt` is rewritten every 250 global steps once the source
# run resumes, so this must be run while that run is NOT running. The copy is guarded: if
# the file changes underneath it the copy is discarded and the script fails loudly.
#
# WANDB. `wandb_run_id.txt` is deliberately NOT copied. It lives beside the checkpoints and
# is how a resumed run re-attaches to its existing wandb run -- copying it would make the
# new arm log into the CONTROL's wandb history, interleaving two different experiments in
# one set of curves.
#
# Usage:
#   bash scripts/pin_cp2lp_resume.sh <src_run> <dst_run>
#   bash scripts/pin_cp2lp_resume.sh cp2lp_onepass_internal cp2lp_onepass_internal_clashnorm
#   FORCE=1 bash scripts/pin_cp2lp_resume.sh <src> <dst>     # overwrite an existing seed
set -euo pipefail

STORE="${STORE:-/zfsauton/scratch/yixiz/Proteina-Complexa/training_runs/store}"

SRC_RUN="${1:-}"
DST_RUN="${2:-}"
[[ -n "$SRC_RUN" && -n "$DST_RUN" ]] || {
  echo "usage: bash scripts/pin_cp2lp_resume.sh <src_run> <dst_run>" >&2
  exit 2
}

SRC_DIR="${STORE}/${SRC_RUN}/checkpoints"
DST_DIR="${STORE}/${DST_RUN}/checkpoints"

[[ -d "$SRC_DIR" ]] || { echo "ERROR: no such source run: $SRC_DIR" >&2; exit 1; }

# The EMA sidecar is copied with it. The EMA callback keeps the shadow weights in
# `last-EMA.ckpt`; resuming without it would continue training from the right weights but
# with an EMA re-seeded from them, so every exported `-EMA` checkpoint for the next few
# thousand updates would be an average over a window the control averaged differently.
FILES=(last.ckpt last-EMA.ckpt)

if [[ -f "${DST_DIR}/last.ckpt" && "${FORCE:-0}" != "1" ]]; then
  echo "[pin] ${DST_DIR}/last.ckpt already exists -- refusing to overwrite."
  [[ -f "${DST_DIR}/resume.source" ]] && sed 's/^/[pin]   /' "${DST_DIR}/resume.source"
  echo "[pin] FORCE=1 overrides. Re-seeding after the arm has started DISCARDS its progress."
  exit 0
fi

# Refuse to run while the source run is live. A copy taken mid-write is a torn checkpoint,
# and the guard below would catch it -- but saying so up front is cheaper than a 3 GB copy
# that is then thrown away.
if squeue -u "$(whoami)" -h -o '%T %Z' 2>/dev/null | grep -q '^RUNNING'; then
  echo "[pin] NOTE: you have RUNNING jobs. This script only needs the SOURCE run to be"
  echo "[pin]       idle; the mtime guard below is what actually enforces that."
fi

mkdir -p "$DST_DIR"

for f in "${FILES[@]}"; do
  src="${SRC_DIR}/${f}"
  [[ -f "$src" ]] || { echo "ERROR: missing ${src}" >&2; exit 1; }
  before="$(stat -c '%Y %s' "$src")"
  echo "[pin] copying ${f} ($(stat -c '%s' "$src") bytes) ..."
  cp "$src" "${DST_DIR}/${f}.tmp"
  sync
  after="$(stat -c '%Y %s' "$src")"
  if [[ "$before" != "$after" ]]; then
    rm -f "${DST_DIR}/${f}.tmp"
    echo "ERROR: ${src} changed during the copy (was [$before], now [$after])." >&2
    echo "       The source run is still training. Wait for it to stop and retry." >&2
    exit 1
  fi
  mv "${DST_DIR}/${f}.tmp" "${DST_DIR}/${f}"
done

{
  echo "seeded_from=${SRC_DIR}"
  date -u +"pinned_utc=%Y-%m-%dT%H:%M:%SZ"
  for f in "${FILES[@]}"; do
    echo "${f}_size=$(stat -c '%s' "${DST_DIR}/${f}")"
    echo "${f}_md5=$(md5sum "${DST_DIR}/${f}" | cut -d' ' -f1)"
  done
} > "${DST_DIR}/resume.source"

echo "[pin] seeded ${DST_DIR}"
sed 's/^/[pin]   /' "${DST_DIR}/resume.source"
echo "[pin] the md5s above are what make 'both arms forked from the same state' checkable."
