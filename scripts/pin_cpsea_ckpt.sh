#!/bin/bash
# Snapshot a LIVE training run's last-EMA.ckpt into a frozen experiment name.
#
# Why: cpsea_peptide_batch_design_experiments.sh hardcodes `++ckpt_name=last-EMA.ckpt` and
# reloads it ONCE PER TARGET. Against a still-training run that pointer moves between targets,
# so a single "run" silently benchmarks several different models -- and a Lightning save landing
# mid-read gives a torn file. Copying to a dated experiment dir makes the design run reproducible
# and lets the manifest record a checkpoint that will still mean the same thing next week.
#
# Usage: bash scripts/pin_cpsea_ckpt.sh <SRC_EXPERIMENT> [PIN_NAME]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?usage: pin_cpsea_ckpt.sh <SRC_EXPERIMENT> [PIN_NAME]}"
PIN="${2:-${SRC%%_from_*}_pin$(date +%Y%m%d)}"

SRC_DIR="${REPO}/store/${SRC}/checkpoints"
PIN_DIR="${REPO}/store/${PIN}/checkpoints"
SRC_CKPT="${SRC_DIR}/last-EMA.ckpt"

[[ -f "${SRC_CKPT}" ]] || { echo "ERROR: no checkpoint at ${SRC_CKPT}" >&2; exit 1; }
if [[ -e "${PIN_DIR}/last-EMA.ckpt" ]]; then
  echo "Pin already exists: ${PIN_DIR}/last-EMA.ckpt -- refusing to overwrite." >&2
  echo "Delete it or pass a different PIN_NAME." >&2
  exit 1
fi

# The source run is still training. Copy, then verify the source did not save underneath us;
# an mtime change during the copy means the bytes we read may straddle two checkpoints.
mkdir -p "${PIN_DIR}"
for attempt in 1 2 3; do
  BEFORE=$(stat -c %Y "${SRC_CKPT}")
  echo "Attempt ${attempt}: copying $(du -h "${SRC_CKPT}" | cut -f1) (src mtime $(date -d @"${BEFORE}" '+%F %T')) ..."
  cp "${SRC_CKPT}" "${PIN_DIR}/last-EMA.ckpt"
  AFTER=$(stat -c %Y "${SRC_CKPT}")
  if [[ "${BEFORE}" == "${AFTER}" ]]; then
    echo "OK: source unchanged during copy -- snapshot is coherent."
    break
  fi
  echo "  source was rewritten mid-copy; retrying."
  rm -f "${PIN_DIR}/last-EMA.ckpt"
  [[ "${attempt}" == "3" ]] && { echo "ERROR: could not get a clean copy in 3 tries." >&2; exit 1; }
done

# The batch script reads exp_config_<EXP>.json from the checkpoint dir to sanity-check
# latent_normalization, so it must be renamed to match the PIN experiment name.
cp "${SRC_DIR}/exp_config_${SRC}.json" "${PIN_DIR}/exp_config_${PIN}.json"
[[ -f "${SRC_DIR}/data_config_${SRC}.json" ]] && \
  cp "${SRC_DIR}/data_config_${SRC}.json" "${PIN_DIR}/data_config_${PIN}.json"

# Provenance: a bare copy of last-EMA.ckpt is otherwise unattributable.
cat > "${REPO}/store/${PIN}/PIN_SOURCE.txt" <<PROV
source_experiment=${SRC}
source_path=${SRC_CKPT}
source_mtime=$(date -d @"${BEFORE}" '+%F %T %Z')
pinned_at=$(date '+%F %T %Z')
pinned_by=pin_cpsea_ckpt.sh
note=source run was still training when pinned; last-EMA.ckpt has since moved on
PROV

echo ""
echo "Pinned experiment: ${PIN}"
echo "  ckpt : ${PIN_DIR}/last-EMA.ckpt"
echo "  prov : ${REPO}/store/${PIN}/PIN_SOURCE.txt"
echo ""
echo "Design from it with:  EXPERIMENT=${PIN} ..."
