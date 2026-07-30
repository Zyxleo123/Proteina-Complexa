#!/bin/bash
# Measure the ACHIEVABLE CEILING for CPSea cyclic-peptide binder metrics, by scoring the
# real crystallographic peptide of each target through the exact harness the designs use.
#
# Why this must run before any Phase 0 baseline
# ---------------------------------------------
# Measured over 298 recorded CPSea designs on 1E2T, the stock `protein_binder` criteria:
#
#     pLDDT   >= 0.9    -> best design observed 0.830   ->   0 / 298 pass
#     i_pAE*31 <= 7.0   -> best design observed 0.370*31 ->  0 / 298 pass
#     scRMSD_ca < 1.5   ->                                  250 / 298 pass
#     ring closed       -> 179 / 191, and scored NOWHERE
#
# Two of three criteria have zero headroom. They do not rank models, they veto all of
# them, which is the entire explanation of the standing 0% pass rate. Recalibrating from
# the DESIGN distribution would be circular ("success = what we already produce"), so the
# reference point has to be a known-good molecule: the native peptide.
#
# Each target is run twice -- cyclic offset OFF then ON -- because the offset fix is
# itself a hypothesis. OFF reproduces the current instrument exactly (regression check);
# ON folds the actual macrocycle. The difference is the measurement of what cutting the
# ring was costing, and only the ON numbers should be used to set thresholds.
#
# NOTE: only 1E2T and 1J7K are head-to-tail (mainchain), so only those two get the wrap.
# 1GYT / 1M46 / 1MU2 are isopeptide: the offset is deliberately skipped for them (a
# backbone wrap would assert a peptide bond that does not exist), and their ON run is
# expected to equal their OFF run. That equality is a useful check, not a wasted job.
#
# Usage:
#   sbatch scripts/run_cpsea_native_calibration.sh
#   # then:
#   python script_utils/derive_cyclic_thresholds.py evaluation_results --out configs/cyclic_thresholds.json
#
#SBATCH --job-name=cpsea_native_calib
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/native_calib/%x_%j.out

set -euo pipefail

# Under sbatch, "$0" is a COPY of this script in the slurm spool dir
# (/var/spool/slurmd/job<N>/slurm_script), so "$(dirname "$0")/.." resolves outside the
# repo and every relative path below silently targets the wrong tree. SLURM_SUBMIT_DIR is
# the directory sbatch was invoked from; fall back to BASH_SOURCE for a plain shell run.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

# Fail loudly and immediately if we are not actually in the repo, rather than producing an
# empty results tree that looks like "the run happened and found nothing".
if [[ ! -f configs/evaluate_cpsea_native_calibration.yaml ]]; then
  echo "ERROR: $REPO_DIR is not the Proteina-Complexa repo root (submit sbatch from the repo)." >&2
  exit 1
fi

mkdir -p slurm_logs/native_calib
source env.sh

# Native linkage chemistry per target (see memory: 1E2T/1J7K mainchain, rest isopeptide).
declare -A NATIVE_TYPE=(
  [cpsea_1E2T]=mainchain
  [cpsea_1J7K]=mainchain
  [cpsea_1GYT]=isopeptide
  [cpsea_1M46]=isopeptide
  [cpsea_1MU2]=isopeptide
)

FAILURES=()

for TARGET in cpsea_1E2T cpsea_1J7K cpsea_1GYT cpsea_1M46 cpsea_1MU2; do
  TYPE="${NATIVE_TYPE[$TARGET]}"
  for MODE in off on; do
    if [[ "$MODE" == "on" ]]; then
      CYCLIC_ARGS="++metric.cyclic_offset=true ++metric.cyclization_type=${TYPE}"
    else
      CYCLIC_ARGS="++metric.cyclic_offset=false"
    fi

    RUN_NAME="native_calib_${TARGET}_cyc${MODE}"
    EXPECTED_DIR="evaluation_results/${RUN_NAME}"

    echo "=== native calibration: ${TARGET} (native type=${TYPE}) cyclic_offset=${MODE} ==="
    # Every override needs the '++' prefix. The CLI parses `++run_name=` itself and then
    # re-appends `++run_name=<parsed>` LAST (cli_runner.py:663), so a bare `run_name=` is
    # silently outranked by the config default -- which sends all 10 runs to one directory
    # to overwrite each other. Bare overrides do reach hydra for other keys, which makes
    # the failure especially deceptive: the right targets get evaluated, into the wrong dir.
    #
    # One bad target must not lose the sweep, but a swept-under failure that still exits 0
    # is worse: it looks exactly like a successful run that produced no data. Record and
    # re-raise at the end.
    if ! complexa analysis configs/evaluate_cpsea_native_calibration.yaml \
        "++run_name=${RUN_NAME}" \
        "++dataset.task_name=${TARGET}" \
        "++sample_storage_path=./calibration_native/${TARGET}/processed" \
        ${CYCLIC_ARGS}; then
      echo "ERROR: ${TARGET} cyclic_offset=${MODE} FAILED" >&2
      FAILURES+=("${TARGET}:cyc${MODE}")
      continue
    fi

    # A zero exit is not proof the results landed where the derive script will look for
    # them. Verify the run actually created its own directory, so a silently-ignored
    # run_name override is caught here rather than 10 runs later.
    if ! compgen -G "${EXPECTED_DIR}/binder_results_*.csv" > /dev/null; then
      echo "ERROR: ${TARGET} cyc${MODE} exited 0 but produced no results in ${EXPECTED_DIR}" >&2
      echo "       (run_name override ignored? check for a shared output dir)" >&2
      FAILURES+=("${TARGET}:cyc${MODE}:no-output")
    fi
  done
done

if (( ${#FAILURES[@]} > 0 )); then
  echo "FAILED ${#FAILURES[@]} of 10 calibration runs: ${FAILURES[*]}" >&2
  echo "Thresholds derived from a partial sweep would be anchored on whichever natives" >&2
  echo "happened to succeed, so fix these before running derive_cyclic_thresholds.py." >&2
  exit 1
fi

echo "All 10 calibration runs succeeded. Derive thresholds with:"
echo "  python script_utils/derive_cyclic_thresholds.py evaluation_results --out configs/cyclic_thresholds.json"
