# Shared preamble for the LP -> CP pre-build check suite.
#
# Sourced by every scripts/precheck_*.sbatch. Every knob has a default here; the submitter
# writes ONE per-submission env file and passes its PATH as a positional argument.
#
# Never `sbatch --export`: any explicit export list (--export=ALL,K=v, --export=K=v and
# --export=NONE alike) sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD -- stranding every
# dependent job on Dependency with no error anyone will see. Measured on this cluster,
# jobs 22184 and 22243. Arguments travel through the job record untouched, so a file path
# as a positional argument is the mechanism that works.
#
# Contract:
#   _precheck_consume_args "$@"   # sources any arg that is a path, exports VAR=value args
#   _precheck_defaults            # fills in anything still unset
#   _precheck_venv <venv|openmm>  # activates the right interpreter and PROVES it
#   _precheck_banner <stage>
#
# `set -euo pipefail` belongs in the JOB BODY, not only here: if sourcing this file fails,
# a job without it keeps running under the system python and exits 0, releasing every
# afterok dependent behind a stage that did no work.

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# ---------------------------------------------------------------- arg handling
_precheck_consume_args() {
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
_precheck_defaults() {
  ZFS="${ZFS:-/zfsauton/scratch/yixiz}"

  # CPSea_full is the build set. CPSea_PDB is held out ENTIRELY as the AFDB->PDB
  # intermediate-domain check (README_POSE_DECOY_INVENTORY 4.1); pointing any of these at
  # it silently spends the held-out set.
  CPSEA_FULL_META="${CPSEA_FULL_META:-${ZFS}/CPSea/CPSea_full/CPSea/preprocessed/metadata}"
  LP_META="${LP_META:-${ZFS}/LPData/preprocessed/metadata}"
  export ZFS CPSEA_FULL_META LP_META

  PRECHECK_CONFIG="${PRECHECK_CONFIG:-${REPO}/configs/pose_decoy/precheck.yaml}"
  RUN_ID="${RUN_ID:-precheck_$(date +%Y%m%d_%H%M%S)}"
  OUT_DIR="${OUT_DIR:-${REPO}/evaluation_results/${RUN_ID}}"

  # The calibrated bridge distance windows. MEASURED on CPSea natives by the Milestone 1.5
  # run; asserting textbook chemistry here would make every bridged feasibility call depend
  # on a guess.
  BRIDGE_WINDOWS="${BRIDGE_WINDOWS:-${REPO}/evaluation_results/m15_20260922_184436/bridge_windows.json}"

  FEAS_DIR="${FEAS_DIR:-${OUT_DIR}/feasibility}"
  PROFILE_DIR="${PROFILE_DIR:-${OUT_DIR}/profile}"
  DECOY_DIR="${DECOY_DIR:-${OUT_DIR}/decoys}"
  PIN_DIR="${PIN_DIR:-${OUT_DIR}/pin}"

  FEAS_SHARDS="${FEAS_SHARDS:-4}"
  PROFILE_SHARDS="${PROFILE_SHARDS:-6}"
  DECOY_SHARDS="${DECOY_SHARDS:-16}"

  # Smoke knobs. Non-empty means the run is NOT a result: a capped solver budget makes
  # levels report infeasible that a full budget would close.
  SMOKE="${SMOKE:-0}"
  FEAS_LIMIT="${FEAS_LIMIT:-0}"
  FEAS_MAX_SOLVES="${FEAS_MAX_SOLVES:-0}"
  FEAS_CHEMISTRIES="${FEAS_CHEMISTRIES:-}"

  export PRECHECK_CONFIG RUN_ID OUT_DIR BRIDGE_WINDOWS
  export FEAS_DIR PROFILE_DIR DECOY_DIR PIN_DIR
  export FEAS_SHARDS PROFILE_SHARDS DECOY_SHARDS
  export SMOKE FEAS_LIMIT FEAS_MAX_SOLVES FEAS_CHEMISTRIES

  mkdir -p "${OUT_DIR}"
}

# ------------------------------------------------------------------ interpreter
# The two environments are mutually exclusive by design -- OpenMM is not installable
# alongside the training environment's pins -- so minimization runs in .venv_openmm and
# writes parquet, while every feature / classifier / report job runs in .venv and reads it.
_precheck_venv() {
  local which="${1:-venv}" path
  case "${which}" in
    venv)   path="${REPO}/.venv" ;;
    openmm) path="${REPO}/.venv_openmm" ;;
    *) echo "[preamble] FATAL: unknown venv '${which}'" >&2; return 1 ;;
  esac
  [[ -x "${path}/bin/python" ]] || { echo "[preamble] FATAL: no interpreter at ${path}" >&2; return 1; }
  # shellcheck source=/dev/null
  source "${path}/bin/activate"
  export PYTHONPATH="${REPO}:${REPO}/src:${PYTHONPATH:-}"

  # Prove the environment is the one this stage needs, rather than discovering it three
  # hours in. A missing import here is a fast, legible failure.
  case "${which}" in
    venv)   python - <<'PY' || return 1
import mdtraj, numpy, pandas, pyarrow, scipy, sklearn, yaml  # noqa: F401
print(f"[preamble] .venv ok: mdtraj {mdtraj.__version__}, sklearn {sklearn.__version__}")
PY
    ;;
    openmm) python - <<'PY' || return 1
import openmm, pdbfixer  # noqa: F401
print(f"[preamble] .venv_openmm ok: openmm {openmm.__version__}")
PY
    ;;
  esac
}

_precheck_banner() {
  echo "================================================================"
  echo "stage   : ${1:-?}"
  echo "run id  : ${RUN_ID}"
  echo "out dir : ${OUT_DIR}"
  echo "config  : ${PRECHECK_CONFIG}"
  # `hostname` is not on PATH on every compute node here, and a bare $(hostname) prints a
  # command-not-found to stderr on each job. SLURMD_NODENAME is set by Slurm itself.
  echo "job     : ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_TASK_ID:-none} on ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
  [[ "${SMOKE}" == "1" ]] && echo "SMOKE   : yes -- this run is NOT a result"
  echo "================================================================"
}
