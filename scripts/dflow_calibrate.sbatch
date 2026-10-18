#!/bin/bash
#SBATCH --job-name=dflow_calibrate
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU stage, and it runs BEFORE the experiment rather than after: feed the acceptance
# instrument the native crystal peptides and check it agrees with the answer
# (script_utils/dflow_calibrate.py).
#
# This project has scored ten runs at 0.0 once already for reasons that were entirely about
# the instrument -- miniprotein thresholds applied to macrocycles. Identity, substitutions,
# contact retention and RMSD against self are arithmetic and must come out perfect; the two
# that are judgement calls are `clash_min_A` (real crystals carry short heavy-atom contacts)
# and whether the interface-RMSD clause simply supersedes the whole-peptide one. The ring
# gate is EXPECTED to fail on these linear inputs -- that failure is the positive control.
#
# It is a gate, not a dependency to be silently skipped: if it fails, the pass rates from the
# rest of the run are not interpretable. It still exits 0 on a failed check so the chain is
# not stranded in DependencyNeverSatisfied -- read the report.
#
# Do NOT sbatch directly -- use scripts/submit_dflow.sh.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/_dflow_preamble.sh" "$@"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${LNR_METADATA:?LNR_METADATA not set}"
dflow_require_file "${LNR_METADATA}" "LNR metadata"
dflow_preflight_python

mkdir -p "$RESULTS_DIR"

"$DFLOW_PY" script_utils/dflow_calibrate.py \
  --metadata "$LNR_METADATA" \
  --out "${RESULTS_DIR}/calibration.json" \
  --cyc-type "${CALIBRATE_CYC_TYPE:-disulfide}" \
  --limit "${CALIBRATE_LIMIT:-0}"

echo "DONE calibrate -> ${RESULTS_DIR}/calibration.json"
