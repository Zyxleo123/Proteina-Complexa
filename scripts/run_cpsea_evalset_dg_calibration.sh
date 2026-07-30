#!/bin/bash
# Compute harness-consistent native Rosetta dG bars for the 96 CPSea EVAL targets, so Phase 0
# joint success becomes computable on the eval set.
#
# Why this is needed even though cpsea_eval_targets.yaml already lists a native_rosetta_dG
# ------------------------------------------------------------------------------------------
# That field comes from the CPSea dataset's precomputed CPSea_PDB_Affinity.tsv, a DIFFERENT
# Rosetta protocol. On the five natives we scored both ways it disagreed with our own harness by
# -5.9 to +3.2 REU with inconsistent sign -- near/above the FastRelax noise floor (~1.5 REU
# median). Joint success gates design dG (our harness) against the native bar, so the bar must be
# produced by the SAME `compute_rosetta_interface_metrics_single` the designs went through, or the
# comparison is an artifact of differing Rosetta treatment -- exactly what calibrate_rosetta_dg.py
# exists to prevent. The Affinity.tsv value stays a rough proxy, not the headline bar.
#
# Prereq (run once, cheap, no GPU): materialize the native complexes into the layout the calibrator
# globs. Idempotent, so re-running is safe:
#   python script_utils/link_evalset_natives.py
#
# This job is PyRosetta-only (FastRelax + interface dG) -- NO GPU, NO AF2, NO analysis pipeline.
#
# Usage:
#   sbatch scripts/run_cpsea_evalset_dg_calibration.sh
#   # then re-run Phase 0 pointed at the new bars:
#   python script_utils/phase0_baseline.py \
#     --design-glob "evaluation_results/search_binder_local_pipeline_cpsea_eval_*singlepass_sh0_*/binder_results_*.csv" \
#     --native-report calibration_native/rosetta_dg_report_evalset.json \
#     --out evaluation_results/phase0_evalset.json \
#     --per-design-csv evaluation_results/phase0_evalset_per_design.csv
#
#SBATCH --job-name=cpsea_evalset_dg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/native_calib/%x_%j.out

set -euo pipefail

# Under sbatch "$0" is a spool copy, so "$(dirname "$0")/.." lands outside the repo (the bug that
# bit run_cpsea_native_calibration.sh). SLURM_SUBMIT_DIR is where sbatch was invoked.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

if [[ ! -f script_utils/calibrate_rosetta_dg.py ]]; then
  echo "ERROR: $REPO_DIR is not the Proteina-Complexa repo root (submit sbatch from the repo)." >&2
  exit 1
fi

mkdir -p slurm_logs/native_calib
source env.sh

# Materialize (idempotent) so a bare `sbatch` works even if the link step was skipped.
python script_utils/link_evalset_natives.py

NATIVE_GLOB="calibration_native/cpsea_eval_*/processed/*.pdb"
N_NAT=$(compgen -G "$NATIVE_GLOB" | wc -l || true)
if (( N_NAT == 0 )); then
  echo "ERROR: no eval natives matched $NATIVE_GLOB -- run link_evalset_natives.py first." >&2
  exit 1
fi
echo "Scoring ${N_NAT} eval natives through the design harness (relaxed + norelax)..."

# cpsea_eval_* glob keeps this off the old 5-target report; write to a separate file so the
# canonical 5-target rosetta_dg_report.json is untouched.
python script_utils/calibrate_rosetta_dg.py \
  --native-glob "$NATIVE_GLOB" \
  --design-glob "evaluation_results/search_binder_local_pipeline_cpsea_eval_*singlepass_sh0_*/binder_results_*.csv" \
  --out calibration_native/rosetta_dg_report_evalset.json \
  --natives-csv calibration_native/rosetta_natives_evalset.csv

echo "Done. Native bars -> calibration_native/rosetta_dg_report_evalset.json"
echo "Re-run Phase 0 with --native-report calibration_native/rosetta_dg_report_evalset.json"
