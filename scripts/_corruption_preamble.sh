# Shared preamble for the corruption-ceiling experiment.
#
# Sourced by every scripts/corruption_*.sbatch. Every knob has a default here; the
# submitter writes ONE per-submission env file and passes its PATH as a positional
# argument.
#
# Never `sbatch --export`: any explicit export list (--export=ALL,K=v, --export=K=v and
# --export=NONE alike) sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD -- stranding every
# dependent job on Dependency with no error anyone will see.
#
# Contract:
#   _cc_consume_args "$@"   # sources any arg that is a path, exports VAR=value args
#   _cc_defaults            # fills in anything still unset
#   _cc_banner <stage>
#   _cc_preflight           # cheap gate; returns non-zero, the CALLER decides the exit code

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# ---------------------------------------------------------------- arg handling
_cc_consume_args() {
  local a
  for a in "$@"; do
    if [[ -f "${a}" ]]; then
      echo "[preamble] sourcing env file: ${a}"
      # shellcheck source=/dev/null
      set -a; source "${a}"; set +a
    elif [[ "${a}" == *=* ]]; then
      echo "[preamble] export ${a}"
      export "${a?}"
    elif [[ -n "${a}" ]]; then
      echo "[preamble] WARNING: ignoring unrecognized argument: ${a}" >&2
    fi
  done
}

# ------------------------------------------------------------------- defaults
_cc_defaults() {
  ZFS="${ZFS:-/zfsauton/scratch/yixiz}"

  # CPSea_full is the build set. CPSea_PDB stays held out as the AFDB->PDB
  # intermediate-domain check; pointing this at it silently spends the held-out set.
  CPSEA_META="${CPSEA_META:-${ZFS}/CPSea/CPSea_full/CPSea/preprocessed/metadata/cpsea_train.parquet}"

  RUN_ID="${RUN_ID:-cc_$(date +%Y%m%d_%H%M%S)}"
  OUT_DIR="${OUT_DIR:-${REPO}/evaluation_results/${RUN_ID}}"
  LADDER_DIR="${LADDER_DIR:-${OUT_DIR}/ladder}"
  PDB_DIR="${PDB_DIR:-${OUT_DIR}/complexes}"
  ROSETTA_DIR="${ROSETTA_DIR:-${OUT_DIR}/rosetta}"
  REPORT_DIR="${REPORT_DIR:-${OUT_DIR}/report}"
  RENDER_DIR="${RENDER_DIR:-${OUT_DIR}/render}"

  # Rendering. One example per chemistry by default, chosen as the MEDIAN final separation so
  # the figure is representative rather than the prettiest ladder in the run.
  RENDER_PER_CHEMISTRY="${RENDER_PER_CHEMISTRY:-1}"
  RENDER_PANEL_PX="${RENDER_PANEL_PX:-700}"
  # 1 skips ray tracing: draft quality, ~10x faster. Useful while iterating on the layout.
  RENDER_FAST="${RENDER_FAST:-0}"
  # Space-separated example_ids to render instead of the per-chemistry medians.
  RENDER_EXAMPLES="${RENDER_EXAMPLES:-}"
  # 1 keeps the per-rung PNGs (~7 MB) so the filmstrip layout can be re-assembled without
  # re-rendering in PyMOL, which is the expensive half (~25 min for 3 examples).
  RENDER_KEEP_PANELS="${RENDER_KEEP_PANELS:-0}"

  # Complexes per chemistry (head_tail / disulfide / other), before sharding.
  PER_TYPE="${PER_TYPE:-20}"
  # Ascending target terminal CA-CA separations. The range that decides the question:
  # CPSea's pre-cyclization median is 6.95 A, LNR's is 16.3 A, PepBench's is 21.6 A.
  LADDER_A="${LADDER_A:-8 10 12 14 16 18 20 24}"
  # 0 lets the interface break so retention is an OUTCOME -- that is the measurement.
  # Non-zero holds the interface instead and asks whether the gap is reachable at all.
  K_CONTACT="${K_CONTACT:-0}"
  EXCLUDE_TERMINI="${EXCLUDE_TERMINI:-2}"
  # Rounds of minimization per rung, at fixed r0. One bounded minimize() per rung made the
  # achieved gap a function of the iteration budget rather than of the pocket, so a rung is
  # now pulled until it converges or visibly stops moving.
  PULL_ROUNDS="${PULL_ROUNDS:-12}"
  MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
  CONVERGE_TOL_A="${CONVERGE_TOL_A:-0.5}"
  STALL_EPS_A="${STALL_EPS_A:-0.05}"
  # 1 fails an example whose ring could not be opened rather than recording it and moving on.
  REQUIRE_RING_OPEN="${REQUIRE_RING_OPEN:-0}"
  SEED="${SEED:-0}"

  # Report acceptance thresholds.
  SHORTFALL_TOL_A="${SHORTFALL_TOL_A:-1.5}"
  RETENTION_MIN="${RETENTION_MIN:-0.70}"
  MIN_HEAVY_A="${MIN_HEAVY_A:-2.0}"
  DG_TOL="${DG_TOL:-5.0}"

  PY="${PY:-${REPO}/.venv/bin/python}"
  PY_OPENMM="${PY_OPENMM:-${REPO}/.venv_openmm/bin/python}"
  SMOKE="${SMOKE:-0}"
  export PYTHONUNBUFFERED=1
}

_cc_banner() {
  echo "=================================================================="
  # `hostname` is not on PATH on every compute node here; SLURMD_NODENAME always is.
  echo "job     : ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ [task ${SLURM_ARRAY_TASK_ID}]} on ${SLURMD_NODENAME:-$(uname -n 2>/dev/null || echo '?')}"
  echo "stage   : ${1:-?}"
  echo "repo    : ${REPO}"
  echo "meta    : ${CPSEA_META}"
  echo "out dir : ${OUT_DIR}"
  echo "ladder  : ${LADDER_A}   k_contact=${K_CONTACT}   per_type=${PER_TYPE}"
  echo "smoke   : ${SMOKE}"
  echo "=================================================================="
}

# ------------------------------------------------------------------ preflight
# Returns non-zero on failure. The caller decides whether that is a hard error (this stage
# cannot run) or a gate (an upstream stage produced no input, in which case the job must
# print why and exit 0 so dependents are not stranded in DependencyNeverSatisfied).
_cc_preflight() {
  local ok=0
  [[ -x "${PY}" ]]        || { echo "ERROR: no venv python at ${PY}" >&2; ok=1; }
  [[ -f "${CPSEA_META}" ]] || { echo "ERROR: missing CPSea metadata: ${CPSEA_META}" >&2; ok=1; }
  return "${ok}"
}

# The OpenMM stage runs under a DIFFERENT interpreter, so it gets its own check: the
# import is what actually fails on a bad venv, and it fails ~60 s into the first example
# rather than at submission.
_cc_preflight_openmm() {
  local ok=0
  [[ -x "${PY_OPENMM}" ]] || { echo "ERROR: no openmm venv at ${PY_OPENMM}" >&2; return 1; }
  "${PY_OPENMM}" -c "
import sys; sys.path.insert(0, '${REPO}/scripts')
import openmm, pdbfixer  # noqa: F401
from soft_closure_project import ATOMS_FOR_TYPE
assert 'ca_termini' in ATOMS_FOR_TYPE, 'ca_termini anchor missing from ATOMS_FOR_TYPE'
print('[preflight] openmm', openmm.__version__, 'ok; ca_termini anchor present')
" || ok=1
  return "${ok}"
}

# `$(dirname "${BASH_SOURCE[0]}")` does NOT work inside a batch script: Slurm copies it to
# /var/spool/slurm/job<id>/slurm_script before running, so that resolves to the spool
# directory and the source silently succeeds against nothing. Every corruption_*.sbatch
# uses $SLURM_SUBMIT_DIR with an absolute fallback instead, and carries its own
# `set -euo pipefail` so a failed source cannot leave the job running under the system
# python and still exiting 0.
