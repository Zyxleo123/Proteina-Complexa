# Shared preamble for the Milestone 1.5 geometric ceiling audit.
#
# Sourced by every scripts/m15_*.sbatch. Every knob has a default here; the submitter
# writes ONE per-submission env file and passes its PATH as a positional argument.
#
# Never `sbatch --export`: any explicit export list (--export=ALL,K=v, --export=K=v and
# --export=NONE alike) sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD -- stranding every
# dependent job on Dependency with no error anyone will see.
#
# Contract:
#   _m15_consume_args "$@"   # sources any arg that is a path, exports VAR=value args
#   _m15_defaults            # fills in anything still unset
#   _m15_banner <stage>
#   _m15_preflight           # cheap gate; returns non-zero, the CALLER decides the exit code

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# ---------------------------------------------------------------- arg handling
_m15_consume_args() {
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
_m15_defaults() {
  ZFS="${ZFS:-/zfsauton/scratch/yixiz}"

  # The build set is CPSea_full. CPSea_PDB is held out ENTIRELY as the AFDB->PDB
  # intermediate-domain check: no training on it, no pilot complexes from it, no test
  # fixture from it. Pointing any of these at CPSea_PDB silently spends the held-out set.
  CPSEA_FULL_META="${CPSEA_FULL_META:-${ZFS}/CPSea/CPSea_full/CPSea/preprocessed/metadata}"
  LP_META="${LP_META:-${ZFS}/LPData/preprocessed/metadata}"
  export CPSEA_FULL_META LP_META

  M15_CONFIG="${M15_CONFIG:-${REPO}/configs/pose_decoy/m15.yaml}"
  RUN_ID="${RUN_ID:-m15_$(date +%Y%m%d_%H%M%S)}"
  OUT_DIR="${OUT_DIR:-${REPO}/evaluation_results/${RUN_ID}}"

  WINDOWS_JSON="${WINDOWS_JSON:-${OUT_DIR}/bridge_windows.json}"
  CEILING_DIR="${CEILING_DIR:-${OUT_DIR}/ceiling}"
  TIMING_DIR="${TIMING_DIR:-${OUT_DIR}/timing}"
  DELIV_E_DIR="${DELIV_E_DIR:-${OUT_DIR}/deliverable_e}"

  # Which audit sets to run; must be keys under `audit.sets` in the YAML.
  AUDIT_SETS="${AUDIT_SETS:-lnr pepbench}"

  # Glob(s) of previously measured contact_retention, used only to rescale achieved /
  # ceiling. Empty is fine -- the report then omits that section rather than inventing it.
  ACHIEVED_GLOB="${ACHIEVED_GLOB:-${REPO}/evaluation_results/sdedit*/**/*.jsonl}"

  PY="${PY:-${REPO}/.venv/bin/python}"
  PY_OPENMM="${PY_OPENMM:-${REPO}/.venv_openmm/bin/python}"
  SMOKE="${SMOKE:-0}"
  export PYTHONUNBUFFERED=1
}

_m15_banner() {
  echo "=================================================================="
  # `hostname` is not on PATH on every compute node here; SLURMD_NODENAME always is.
  echo "job     : ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ [task ${SLURM_ARRAY_TASK_ID}]} on ${SLURMD_NODENAME:-$(uname -n 2>/dev/null || echo '?')}"
  echo "stage   : ${1:-?}"
  echo "repo    : ${REPO}"
  echo "config  : ${M15_CONFIG}"
  echo "out dir : ${OUT_DIR}"
  echo "smoke   : ${SMOKE}"
  echo "=================================================================="
}

# ------------------------------------------------------------------ preflight
# Returns non-zero on failure. The caller decides whether that is a hard error (this stage
# cannot run) or a gate (an upstream stage did not produce its input, in which case the
# job must print why and exit 0 so dependents are not stranded in DependencyNeverSatisfied).
_m15_preflight() {
  local ok=0
  [[ -x "${PY}" ]] || { echo "ERROR: no venv python at ${PY}" >&2; ok=1; }
  [[ -f "${M15_CONFIG}" ]] || { echo "ERROR: missing config: ${M15_CONFIG}" >&2; ok=1; }
  [[ -f "${CPSEA_FULL_META}/cpsea_train.parquet" ]] || {
    echo "ERROR: missing CPSea_full metadata: ${CPSEA_FULL_META}/cpsea_train.parquet" >&2; ok=1; }

  # An unexpanded $VAR in the YAML surfaces much later as a "no such file" naming a
  # literal dollar sign; the loader refuses instead, so exercise it here where it is cheap.
  if [[ ${ok} -eq 0 ]]; then
    "${PY}" -c "
import sys; sys.path.insert(0, '${REPO}')
from script_utils import m15_config
cfg = m15_config.load('${M15_CONFIG}')
print('[preflight] config ok:', ', '.join(sorted(cfg)))
" || ok=1
  fi
  return "${ok}"
}

# `$(dirname "${BASH_SOURCE[0]}")` does NOT work inside a batch script: Slurm copies it to
# /var/spool/slurm/job<id>/slurm_script before running, so that resolves to the spool
# directory and the source silently succeeds against nothing. Every m15_*.sbatch uses
# $SLURM_SUBMIT_DIR with an absolute fallback instead, and carries its own
# `set -euo pipefail` so a failed source cannot leave the job running under the system
# python and still exiting 0.
