#!/bin/bash
#SBATCH --job-name=restage_gapmore
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU arm: builds the two "gap it more" target sets. Reads staged PDBs, writes staged PDBs
# and parquets -- no model, no GPU.
#
#   ARM D-phys  14 A pocket crop. Shortens receptor runs PHYSICALLY (real deletions ->
#               real backbone breaks). Median segments/max_run/tgt_len 13.0 / 33.5 / 101.5,
#               which matches CPSea val (14 / 31 / 117.5) on all three at once, where the
#               18 A Arm B crop sits at 11 / 51 / 134.
#   ARM D-num   Arm B's own PDBs with the receptor RENUMBERED so no run exceeds 31.
#               Coordinates and atom set byte-identical to Arm B; only the featurization
#               changes. Isolates "short segments as a feature" from "smaller receptor".
#
# Run both: D-phys alone confounds run length with receptor size and with the loss of real
# (distal) receptor atoms; D-num alone changes only the feature but writes numbering that
# disagrees with the geometry. Together they separate the two.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs
source env.sh
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

SRC_FULL="CPSea_data/lnr_staged/metadata/lnr_test.parquet"
SRC_POCKET="CPSea_data/lnr_pocket/metadata/lnr_test_pocket.parquet"
for f in "$SRC_FULL" "$SRC_POCKET"; do
  [[ -f "$f" ]] || { echo "GATE: missing upstream metadata $f -- restage Arm B first. Exiting 0."; exit 0; }
done

echo "=== ARM D-phys: 14 A crop from the ORIGINAL full-receptor staging ==="
"$PY" scripts/restage_lnr_pocket.py \
  --in-metadata "$SRC_FULL" \
  --out-dir CPSea_data/lnr_pocket14 \
  --radius 14.0

echo
echo "=== ARM D-num: renumber Arm B so no receptor run exceeds 31 ==="
"$PY" scripts/inject_numbering_gaps.py \
  --in-metadata "$SRC_POCKET" \
  --out-dir CPSea_data/lnr_gapped31 \
  --max-seg-len 31

echo
echo "=== AUDIT: do the new sets match the training distribution? ==="
# Read max_run and frag<5, not `segments` -- segment COUNT plateaus at every crop radius and
# will look matched even when the run lengths are 20 residues too long. See
# docs/README_TARGET_DISTRIBUTION_MATCHING.md.
"$PY" scripts/audit_target_distribution.py \
  --set "CPSea_ctl (ref)=CPSea_data/control/cpsea_val_control.parquet" \
  --set "LNR orig full=${SRC_FULL}" \
  --set "LNR 18A (ArmB)=${SRC_POCKET}" \
  --set "LNR 14A (D-phys)=CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet" \
  --set "LNR renum31 (D-num)=CPSea_data/lnr_gapped31/metadata/lnr_test_gapped.parquet"

echo
echo "DONE"
