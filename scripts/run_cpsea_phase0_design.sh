#!/bin/bash
# Phase 0 / per-linkage design run over the 96-target CPSea evaluation set.
#
# Thin driver over ~/slurm/cpsea/cpsea_peptide_batch_design_experiments.sh, which already
# handles GPU allocation, AE-checkpoint resolution, the run manifest, and resume. This adds
# only what that script has no opinion about: WHICH targets, WHICH seed, and how to shard.
#
# What this run is for
# --------------------
# Every recorded CPSea design predates per-target `cyclization_type` in the targets dict, so
# type conditioning was inert and the requested linkage was never recorded. That makes the
# per-linkage breakdown circular (`observed == mainchain` is defined BY closure) and left
# disulfide with n=21 incidental samples. This run fixes both: the 96-target set is balanced
# 32/32/32 by PARSED native chemistry, and each target carries its own `cyclization_type`, so
# the model is told what to build and the eval scores it against that request.
#
# Defaults chosen deliberately
# ----------------------------
#   SEARCH=single-pass  The control. One ODE integration, no lookahead, no in-loop AF2. It
#                       measures the MODEL rather than the search, and prior tuning found
#                       test-time compute bought nothing on binding anyway. It is also ~10x
#                       cheaper, which is what makes 96 targets affordable at all.
#   SEED                Explicit and stamped into RUN_NAME. Recorded designs have no seed
#                       column, so they are not individually reproducible; every run from
#                       here should be.
#   NSTEPS=400          Matches the recorded baseline so this run is comparable to it. The
#                       50/100/200 arms are a separate sweep (see NSTEPS below) -- do not
#                       mix step counts inside one RUN_NAME or the comparison is unequal
#                       sampling compute, which is exactly the confound to avoid.
#
# Cost
# ----
# Single-pass runs AF2 once per FINAL sample, so cost ~ TARGETS x (N_BEST + N_OVERSAMPLE).
# At the defaults that is 96 x 12 = 1152 refold+Rosetta evaluations. Budget hours, not
# minutes, and shard it.
#
# Usage:
#   # Smoke first -- 3 targets, one per chemistry, 2 candidates each.
#   SMOKE=true bash scripts/run_cpsea_phase0_design.sh
#
#   # Full run, sharded across 4 allocations (launch each from the login node)
#   NUM_SHARDS=4 SHARD=0 bash scripts/run_cpsea_phase0_design.sh
#
#   # nsteps arm for the scaling comparison
#   NSTEPS=100 bash scripts/run_cpsea_phase0_design.sh
#
# Env:
#   SEED         generation seed (default 0); stamped into RUN_NAME
#   NSTEPS       ODE integration steps (default 400); stamped into RUN_NAME
#   EXPERIMENT   training run to design from (default cpsea_cyc_ringpe_bond_v4)
#   N_BEST       candidates kept per target (default 10)
#   SHARD        this shard index (default 0)
#   NUM_SHARDS   total shards (default 1)
#   SMOKE        true -> 3 targets, N_BEST=2
#   SEARCH       override the single-pass default at your own risk

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"

TARGETS_YAML="configs/targets/cpsea_eval_targets.yaml"
if [[ ! -f "${TARGETS_YAML}" ]]; then
  echo "ERROR: ${TARGETS_YAML} not found. Generate it first:" >&2
  echo "  python script_utils/select_cpsea_eval_targets.py --per-type 32" >&2
  exit 1
fi

SEED="${SEED:-0}"
NSTEPS="${NSTEPS:-400}"
EXPERIMENT="${EXPERIMENT:-cpsea_cyc_ringpe_bond_v4}"
N_BEST="${N_BEST:-10}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SMOKE="${SMOKE:-false}"
SEARCH="${SEARCH:-single-pass}"

# Parse the targets YAML with the repo venv, NOT whatever `python` the caller's shell resolves to
# (a base conda env lacks `pyyaml` -> "No module named 'yaml'" -> 0 targets selected -> the shard
# error). Overridable via PYTHON_EXEC.
PY="${PYTHON_EXEC:-${REPO}/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: python not found at $PY (set PYTHON_EXEC)"; exit 1; }
"$PY" -c "import yaml" 2>/dev/null || { echo "ERROR: $PY lacks pyyaml; set PYTHON_EXEC to an interpreter that has it"; exit 1; }

# Read target names straight from the generated file, so this can never drift from the set
# that was actually selected and appended to targets_dict.yaml.
mapfile -t ALL_TARGETS < <("$PY" - "$TARGETS_YAML" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))["target_dict_cfg"]
# Sort by chemistry then name so shards stay balanced across linkage types: a shard that
# happened to be all-isopeptide would give a per-linkage result that is really a shard effect.
for name in sorted(d, key=lambda k: (d[k]["cyclization_type"], k)):
    print(name)
PY
)

if [[ "${SMOKE}" == "true" ]]; then
  # One target per chemistry, so a smoke run still exercises all three code paths.
  mapfile -t ALL_TARGETS < <("$PY" - "$TARGETS_YAML" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))["target_dict_cfg"]
seen = {}
for name in sorted(d):
    seen.setdefault(d[name]["cyclization_type"], name)
print("\n".join(seen.values()))
PY
)
  N_BEST=2
  echo "SMOKE mode: ${#ALL_TARGETS[@]} targets, N_BEST=${N_BEST}"
fi

# Round-robin sharding over the chemistry-sorted list keeps each shard type-balanced.
SHARD_TARGETS=()
for i in "${!ALL_TARGETS[@]}"; do
  if (( i % NUM_SHARDS == SHARD )); then
    SHARD_TARGETS+=("${ALL_TARGETS[$i]}")
  fi
done

if (( ${#SHARD_TARGETS[@]} == 0 )); then
  echo "ERROR: shard ${SHARD}/${NUM_SHARDS} selected no targets." >&2
  exit 1
fi

RUN_NAME="phase0_s${SEED}_n${NSTEPS}_${SEARCH//-/}"
if [[ "${SMOKE}" == "true" ]]; then
  RUN_NAME="smoke_${RUN_NAME}"
fi
if (( NUM_SHARDS > 1 )); then
  RUN_NAME="${RUN_NAME}_sh${SHARD}"
fi

echo "=================================================================="
echo " Phase 0 design run"
echo "   run name    : ${RUN_NAME}"
echo "   experiment  : ${EXPERIMENT}"
echo "   search      : ${SEARCH}"
echo "   seed        : ${SEED}"
echo "   nsteps      : ${NSTEPS}"
echo "   n_best      : ${N_BEST}"
echo "   targets     : ${#SHARD_TARGETS[@]} (shard ${SHARD}/${NUM_SHARDS} of ${#ALL_TARGETS[@]})"
echo "=================================================================="

# Seed and nsteps go through EXTRA_GEN_ARGS, which the batch script appends to GEN_ARGS.
# (That hook was added for this; it is empty by default so other callers are unaffected.)
export EXTRA_GEN_ARGS="++seed=${SEED} ++generation.args.nsteps=${NSTEPS}"
export EXPERIMENTS="${EXPERIMENT}"
export TARGETS="${SHARD_TARGETS[*]}"
export N_BEST
export SEARCH
export SLURM_JOB_NAME="${RUN_NAME}"

exec bash ~/slurm/cpsea/cpsea_peptide_batch_design_experiments.sh "${RUN_NAME}"
