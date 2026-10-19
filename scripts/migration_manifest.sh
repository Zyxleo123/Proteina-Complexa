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
REPO_DIR="${REPO_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# PROFILE=full     every real run's EMA weights (~205 GB)
# PROFILE=minimal  only what the sequence-conditioned follow-up needs, plus
#                  both autoencoders. See RUNS_ONLY below.
PROFILE="${PROFILE:-full}"

# The minimal set, and why each member is in it:
#   cpsea_seqcond_v2            the model being followed up
#   cpsea_bondunroll_from_v4cfg its pretrain_ckpt_path, needed to relaunch
#   finetune_full_128           the CPSea AE (seqcond's autoencoder_ckpt_path)
#   shared_ae_lpcp_128          the CPSea+LP AE, via its frozen_ae.ckpt
#   cpsea_lpmix_from_v4cfg      the flow model that uses that shared AE
RUNS_ONLY="${RUNS_ONLY:-
cpsea_seqcond_v2
cpsea_bondunroll_from_v4cfg
finetune_full_128
shared_ae_lpcp_128
cpsea_lpmix_from_v4cfg
}"

# Arms still being worked. For these we also carry last.ckpt -- the non-EMA
# checkpoint holding optimizer state -- so training can resume on the new
# cluster rather than restart. Everything else gets EMA weights only, which
# is enough to evaluate and sample but not to continue training.
LIVE_ARMS="${LIVE_ARMS:-
cpsea_cyc_ringpe_bond_v4
shared_ae_lpcp_128
finetune_full_128
cpsea_lpmix_from_v4cfg
cpsea_seqcond_v2
cp2lp_bb_torsion
cp2lp_bb_cartesian
surface_receptor_full_v2
}"

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

# Choose the checkpoint with the highest embedded global_step from the globs
# given. With zero or one match this is free; only a genuine ambiguity costs
# a read, and even then only of the zip header, not the 2.9 GB of weights.
PY="${PY:-$REPO_DIR/.venv/bin/python}"

# Three-point content fingerprint: head, middle and tail. Enough to tell two
# multi-GB checkpoints apart without reading either in full.
fp() {
  local f="$1" sz mb
  sz=$(stat -c%s "$f"); mb=$((sz / 1048576))
  {
    dd if="$f" bs=1M count=32 2>/dev/null
    dd if="$f" bs=1M skip=$((mb / 2)) count=32 2>/dev/null
    dd if="$f" bs=1M skip=$(( mb > 32 ? mb - 32 : 0 )) count=32 2>/dev/null
  } | md5sum | cut -d' ' -f1
}
pick_latest() {
  local cands=() f
  for f in "$@"; do [ -e "$f" ] && cands+=("$f"); done
  case ${#cands[@]} in
    0) return 0 ;;
    1) printf '%s\n' "${cands[0]}"; return 0 ;;
  esac
  if [ -x "$PY" ] && [ -f "$REPO_DIR/script_utils/ckpt_step.py" ]; then
    local best
    best="$("$PY" -I "$REPO_DIR/script_utils/ckpt_step.py" --max "${cands[@]}" 2>/dev/null || true)"
    if [ -n "$best" ]; then printf '%s\n' "$best"; return 0; fi
  fi
  # Fall back to name order, and say so -- a silent fallback here is how the
  # wrong checkpoint travels.
  echo "  !! cannot read steps in $(dirname "${cands[0]}"); falling back to name order" >&2
  printf '%s\n' "$(printf '%s\n' "${cands[@]}" | sort | tail -1)"
}

echo "scanning $STORE ..."
nkept=0; nskip=0
for run in "$STORE"/*/; do
  run="${run%/}"
  [ -d "$run" ] || continue
  name="$(basename "$run")"
  if [ "$PROFILE" = "minimal" ]; then
    printf '%s\n' $RUNS_ONLY | grep -qx "$name" || { nskip=$((nskip + 1)); continue; }
  elif [ "$KEEP_SMOKE" != "1" ] && printf '%s' "$name" | grep -qE "$SKIP_RE"; then
    nskip=$((nskip + 1)); continue
  fi

  # Records which AE this run was frozen against. A few hundred bytes, and
  # without it you cannot tell which AE to evaluate a flow checkpoint with.
  emit "$run/frozen_ae_source.txt"

  ck="$run/checkpoints"
  [ -d "$ck" ] || continue

  # Pick the furthest-trained EMA checkpoint. Filenames cannot be trusted for
  # this: on requeue Lightning writes last.ckpt, then last-v1.ckpt and so on,
  # and the suffix does not track time. cpsea_seqcond_v2 is the case in point
  # -- bare last.ckpt is step 2400 while last-v3.ckpt is step 3000, so naming
  # or mtime would both have carried a stale checkpoint. When more than one
  # candidate exists, read the step out of each and take the max.
  best="$(pick_latest "$ck"/chk_epoch=*-EMA.ckpt)"
  [ -n "$best" ] || best="$(pick_latest "$ck"/last*-EMA.ckpt)"
  [ -n "$best" ] || best="$(pick_latest "$ck"/last*.ckpt)"

  if [ -z "$best" ]; then
    echo "  !! no checkpoint found in $name" >&2
    continue
  fi

  emit "$best"
  # Configs are a few KB each and make the checkpoint reloadable.
  for j in "$ck"/*.json; do emit "$j"; done
  # The wandb run id is what lets a resumed run keep logging to the same
  # chart instead of starting a second one. 34 bytes.
  emit "$ck/wandb_run_id.txt"

  # Live arms additionally get optimizer state, so they can be resumed. In
  # the minimal profile every run kept is one we intend to continue.
  if [ "$PROFILE" = "minimal" ] || printf '%s\n' $LIVE_ARMS | grep -qx "$name"; then
    nonema=()
    for f in "$ck"/last*.ckpt; do
      case "$f" in
        *-EMA.ckpt) ;;
        *) [ -e "$f" ] && nonema+=("$f") ;;
      esac
    done
    if [ ${#nonema[@]} -gt 0 ]; then
      opt="$(pick_latest ${nonema[@]+"${nonema[@]}"})"
      if [ -n "$opt" ]; then
        emit "$opt"
        echo "  + optimizer state for $name: $(basename "$opt")"
      fi
    fi
  fi
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
  rn="$(basename "$(dirname "$f")")"
  if [ "$PROFILE" = "minimal" ]; then
    printf '%s\n' $RUNS_ONLY | grep -qx "$rn" || continue
  fi
  sz="$(stat -c%s "$f")"
  # For a pure AE run, frozen_ae.ckpt is often a byte-copy of that run's own
  # best checkpoint -- shared_ae_lpcp_128 is exactly that. Drop whichever we
  # already carry, but only on a content match: a size match alone could
  # silently hand a run the wrong autoencoder, and the flow configs point at
  # the frozen_ae.ckpt path by name, so we keep that one and drop the twin.
  for sib in $(grep "^Proteina-Complexa/training_runs/store/$rn/checkpoints/.*\.ckpt\$" "$FILES" 2>/dev/null || true); do
    [ -e "$ZFS/$sib" ] || continue
    [ "$(stat -c%s "$ZFS/$sib")" = "$sz" ] || continue
    if [ "$(fp "$ZFS/$sib")" = "$(fp "$f")" ]; then
      echo "  $rn: frozen_ae.ckpt is $(basename "$sib"); dropping the twin"
      grep -vxF "$sib" "$FILES" > "$FILES.tmp" && mv "$FILES.tmp" "$FILES"
    fi
  done
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

# ---- lpgen ---------------------------------------------------------------
# The LP generator arm trains into $ZFS/lpgen/runs, not into the store above,
# so none of it is picked up by the loop over $STORE.
echo "collecting lpgen checkpoints ..."
for run in $ZFS/lpgen/runs/*/; do
  run="${run%/}"; name="$(basename "$run")"
  # Preflight and smoke runs exist to prove a job launches; their checkpoints
  # are not worth 1.3 GB each.
  printf '%s' "$name" | grep -qE '(_preflight$|_hsprobe$|smoke|^_)' && continue
  newest="$(ls -1 "$run"/ckpt/*.ckpt 2>/dev/null | sort | tail -1 || true)"
  [ -n "$newest" ] || continue
  emit "$newest"
  echo "  + $name/$(basename "$newest")"
done

# RFD3 base weights. Unlike the DiffPepBuilder weights -- which have URLs and
# md5s in scripts/_lpgen_preamble.sh and are listed in weights_manifest.json
# -- these two have no recorded provenance anywhere, and foundry is a private
# registry. Carry one copy; the pair are byte-identical.
emit "$ZFS/lpgen/weights/rfd3_released/rfd3_foundry_2025_12_01_remapped.ckpt"

sort -u -o "$FILES" "$FILES"

# ---- CPSea dataset identity ----------------------------------------------
# The bulk of CPSea (290 GB full + 68 GB CPBind) is public on Zenodo
# (record 16794716) and is re-downloadable, so it does not cross the network.
# What is NOT re-derivable is exactly which structures these runs were
# trained and evaluated on -- and the Zenodo record warns a newer version is
# coming, so a fresh download may not be the same data. The index and
# properties files pin that: 2,714,414 rows for ~617 MB.
echo "collecting CPSea dataset identity ..."
for f in \
  "$ZFS/CPSea/CPSea_full/CPSea/CPSea_index.txt" \
  "$ZFS/CPSea/CPBind/CPBind_index.txt" \
  "$ZFS/CPSea/CPSea_PDB/CPSea_PDB_index.txt"
do emit "$f"; done
for d in \
  "$ZFS/CPSea/CPSea_full/CPSea/CPSea_properties" \
  "$ZFS/CPSea/CPBind/CPBind_properties" \
  "$ZFS/CPSea/CPSea_PDB/CPSea_PDB_properties"
do
  [ -d "$d" ] || continue
  find "$d" -type f -size -200M -print 2>/dev/null \
    | while read -r f; do printf '%s\n' "${f#"$ZFS"/}"; done >> "$FILES"
done

# The eval data root is only 1.6 GB in total, so carry it whole rather than
# re-downloading and re-preprocessing it.
find "$ZFS/CPSea/CPSea_PDB" -type f -size -200M -print 2>/dev/null \
  | while read -r f; do printf '%s\n' "${f#"$ZFS"/}"; done >> "$FILES"

# The preprocessed tree is OUR format, written by script_utils/preprocess_cpsea.py,
# so unlike the raw structures it cannot be re-downloaded -- only regenerated.
# Split it by what regeneration would and would not reproduce:
#
#   metadata/ (147 MB)  cpsea_train/val/test.parquet -- the split assignment
#                       itself. cpsea_seqcond_v2 trained against these exact
#                       files. Re-preprocessing could land a different split,
#                       which would quietly contaminate any evaluation of the
#                       existing checkpoint. Always carried.
#   manifest/ (222 MB)  audit CSVs, rejected.csv, preprocess_stats.json -- the
#                       record of what was dropped and why. Always carried.
#   processed/ (~97 GB) the tensors. Regenerable from the Zenodo raw given a
#                       long job over 2.71M structures. Opt in with
#                       WITH_PREPROCESSED=1 to carry them instead.
PRE="$ZFS/CPSea/CPSea_full/CPSea/preprocessed"
for d in metadata manifest; do
  [ -d "$PRE/$d" ] || continue
  find "$PRE/$d" -type f -size -200M -print 2>/dev/null \
    | while read -r f; do printf '%s\n' "${f#"$ZFS"/}"; done >> "$FILES"
done
if [ "${WITH_PREPROCESSED:-0}" = "1" ] && [ -d "$PRE/processed" ]; then
  echo "  including preprocessed/processed (~97 GB) ..."
  find "$PRE/processed" -type f -print 2>/dev/null \
    | while read -r f; do printf '%s\n' "${f#"$ZFS"/}"; done >> "$FILES"
fi

sort -u -o "$FILES" "$FILES"

# ---- the repo tree on home ----------------------------------------------
# Despite the rule that artefacts live on $ZFS, 46 GB of results accumulated
# under the repo on home. Almost all of it is bulk structures -- 362k PDBs
# across inference/ and evaluation_results/ -- which regenerate from the
# checkpoints we are carrying. The numbers behind every figure and table are
# in the CSV/JSON/JSONL/npz beside them, and that is only ~500 MB.
# These paths are relative to the REPO, not $ZFS, so they get their own list
# and their own rsync invocation.
REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
HFILES="$OUTDIR/tier0_home.files"
: > "$HFILES"
if [ -d "$REPO" ]; then
  echo "collecting results from the repo tree on home ..."
  for d in evaluation_results inference outputs logs slurm_logs calibration_native assets surfaces; do
    [ -d "$REPO/$d" ] || continue
    find "$REPO/$d" -type f \
      \( -name '*.csv' -o -name '*.json' -o -name '*.jsonl' -o -name '*.npz' \
         -o -name '*.png' -o -name '*.txt' -o -name '*.log' -o -name '*.fa' \) \
      -size -200M -print 2>/dev/null \
    | while read -r f; do printf '%s\n' "${f#"$REPO"/}"; done >> "$HFILES"
  done
  sort -u -o "$HFILES" "$HFILES"
  ( cd "$REPO" && du -ch --files0-from=<(tr '\n' '\0' < "$HFILES") 2>/dev/null | tail -1 ) \
    > "$OUTDIR/tier0_home.total" || echo "unknown" > "$OUTDIR/tier0_home.total"
  echo "  home entries : $(wc -l < "$HFILES")"
  echo "  home total   : $(cat "$OUTDIR/tier0_home.total")"
fi

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
