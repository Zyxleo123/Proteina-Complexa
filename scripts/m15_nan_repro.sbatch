#!/bin/bash
#SBATCH --job-name=m15_nanrep
#SBATCH --output=logs/m15_nanrep_%j.out
#SBATCH --error=logs/m15_nanrep_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
# Isolate the `Particle coordinate is NaN` failure from the replica-timing run before the
# decoy build inherits it. Reuses the timing stage's force-field and restraint settings from
# the same YAML block, so the reproducer exercises the code path that actually failed rather
# than an approximation of it.
#
# --cpus-per-task=1 matches m15_timing.sbatch: OpenMM's CPU platform takes every core it is
# given, and a divergence that depends on thread count would not reproduce under a different
# core budget.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_m15_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_m15_preamble.sh"

_m15_consume_args "$@"
_m15_defaults
_m15_banner nan-repro
_m15_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

[[ -x "$PY_OPENMM" ]] || { echo "FATAL: no openmm venv at $PY_OPENMM" >&2; exit 1; }
export OPENMM_CPU_THREADS=1

cd "$REPO"
mkdir -p "$TIMING_DIR"

# The complex that failed in m15_20260922_184436. Overridable from the env file via
# NAN_TARGET, so this job stays useful for the next NaN rather than being single-use.
NAN_TARGET="${NAN_TARGET:-AF-A0A537RZW2-F1_0_130_140_relaxed_relaxed}"
NAN_CONTROLS="${NAN_CONTROLS:-3}"
NAN_REPEATS="${NAN_REPEATS:-2}"
# One output file per run, never a shared one: two nan-repro runs differing only by whether
# the ladder is walked are a PAIRED comparison, and writing both to nan_repro.jsonl would
# silently destroy the earlier arm. NAN_OUT_TAG suffixes the pair apart.
NAN_OUT_TAG="${NAN_OUT_TAG:-}"
NAN_OUT="${TIMING_DIR}/nan_repro${NAN_OUT_TAG:+_${NAN_OUT_TAG}}.jsonl"
if [[ -e "$NAN_OUT" ]]; then
  echo "GATE: ${NAN_OUT} already exists. Set NAN_OUT_TAG=<name> to write a new arm rather"
  echo "GATE: than overwriting a previous one. Exiting 0 so nothing downstream is stranded."
  exit 0
fi

read -r META NREP SIG CPR KP KC CTOL TOL MAXIT PLAT SEED RUNGS < <("$PY" -c "
import sys; sys.path.insert(0, '.')
from script_utils import m15_config
t = m15_config.load('$M15_CONFIG')['timing']
print(m15_config.resolve(t['metadata']), t['n_replicas'], t['torsion_sigma_deg'],
      t['contacts_per_residue'], t['k_pull'], t['k_contact'], t['contact_tol_A'],
      t['tolerance'], t['max_iterations'], t['platform'], t['seed'], t['ladder_rungs'])
")
[[ "$SMOKE" == "1" ]] && { NREP=2; NAN_CONTROLS=1; NAN_REPEATS=1; }

echo "[nan-repro] target=${NAN_TARGET} controls=${NAN_CONTROLS} repeats=${NAN_REPEATS}"

# A reproducer that reproduces the bug must still exit 0: the finding IS the result, and a
# non-zero exit would strand anything chained behind it in DependencyNeverSatisfied while
# looking like an infrastructure failure rather than a measurement.
"$PY_OPENMM" script_utils/m15_nan_repro.py \
  --metadata "$META" \
  --out "$NAN_OUT" \
  --target "$NAN_TARGET" \
  --n-controls "$NAN_CONTROLS" \
  --repeats "$NAN_REPEATS" \
  --n-replicas "$NREP" \
  --torsion-sigma-deg "$SIG" --contacts-per-residue "$CPR" \
  --k-pull "$KP" --k-contact "$KC" --contact-tol-A "$CTOL" \
  --tolerance "$TOL" --max-iterations "$MAXIT" \
  --ladder-rungs "$RUNGS" ${NAN_LADDER:+--with-ladder} ${NAN_NO_LADDER:+--no-ladder} \
  --platform "$PLAT" --seed "$SEED"

echo
echo "[nan-repro] rows    -> ${NAN_OUT}"
echo "[nan-repro] verdict -> ${NAN_OUT%.jsonl}.verdict.json"
