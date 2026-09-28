# Shared preamble for the Rosetta/GenKIC LP -> CP search arm.
#
# Sourced by every scripts/genkic_*.sbatch. Every knob has a default here; the submitter
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
#   _genkic_consume_args "$@"   # sources any arg that is a path, exports VAR=value args
#   _genkic_defaults            # fills in anything still unset
#   _genkic_venv                # activates .venv and PROVES PyRosetta imports
#   _genkic_banner <stage>
#
# `set -euo pipefail` belongs in the JOB BODY, not only here: if sourcing this file fails,
# a job without it keeps running under the system python and exits 0, releasing every
# afterok dependent behind a stage that did no work.
#
# This arm does NOT source env.sh. env.sh carries wandb credentials and the CPSea AE
# checkpoint pin, and this arm touches neither -- it is pure PyRosetta on staged crystal
# complexes, with no GPU, no model and no AE. Staying off env.sh keeps every job
# self-sufficient under `env -i`.

# REPO must be the GenKIC WORKTREE, not the main checkout: the search code lives only here
# and is untracked on the cpsea branch. A default pointing at the main checkout would run a
# different (older) tree and look like it worked.
REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa/.claude/worktrees/genkic}"

# ---------------------------------------------------------------- arg handling
_genkic_consume_args() {
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
_genkic_defaults() {
  # The 60 staged LNR complexes: bound linear peptides with their receptors, each keeping
  # its source crystal's own residue numbering (which is why the search maps binder indices
  # by chain order, never by pdb2pose).
  METADATA="${METADATA:-${REPO}/CPSea_data/lnr_staged/metadata/lnr_test.parquet}"

  RUN_ID="${RUN_ID:-genkic_$(date +%Y%m%d_%H%M%S)}"
  OUT_DIR="${OUT_DIR:-${REPO}/evaluation_results/${RUN_ID}}"
  ROWS_DIR="${ROWS_DIR:-${OUT_DIR}/rows}"
  PDB_DIR="${PDB_DIR:-${OUT_DIR}/pdbs}"
  REPORT_DIR="${REPORT_DIR:-${OUT_DIR}/report}"

  CHEMISTRIES="${CHEMISTRIES:-mainchain disulfide isopeptide}"
  BUDGETS="${BUDGETS:-0 1 2}"
  PROTECTED="${PROTECTED:-}"

  # Search effort. attempts * calls * |window ladder| closure attempts per (chemistry,
  # budget, anchor option). The contract prizes candidate QUALITY over single-sample
  # closure rate, so the search keeps going past the first closed structure and only a
  # ranked shortlist pays for relaxation.
  ATTEMPTS_PER_CALL="${ATTEMPTS_PER_CALL:-2000}"
  CALLS="${CALLS:-4}"
  # Closures GenKIC collects per call before the lowest-energy selector picks one. At 1 the
  # search stops at the first closed structure, which the contract rules out.
  MIN_SOLUTIONS="${MIN_SOLUTIONS:-8}"
  MAX_RELEASED="${MAX_RELEASED:-6}"
  SHORTLIST="${SHORTLIST:-8}"
  DUMP_TOP="${DUMP_TOP:-2}"
  RELAX_REPEATS="${RELAX_REPEATS:-1}"
  # Independent relaxation/scoring runs per shortlisted CANDIDATE (the contract asks for 3).
  # Distinct from RELAX_REPEATS, which is FastRelax cycles inside ONE trajectory and so says
  # nothing about run-to-run spread.
  CP_REPEATS="${CP_REPEATS:-3}"
  LP_REPEATS="${LP_REPEATS:-1}"
  PERTURB_DEG="${PERTURB_DEG:-45.0}"
  SEED="${SEED:-1}"

  SEARCH_SHARDS="${SEARCH_SHARDS:-12}"
  # Stop STARTING new targets after this long, so a shard ends cleanly inside its walltime
  # instead of being killed mid-target. Resume by resubmitting: finished row_keys are skipped.
  TIME_BUDGET_S="${TIME_BUDGET_S:-14400}"

  EXAMPLES="${EXAMPLES:-}"
  SMOKE="${SMOKE:-0}"

  # Existing SDEdit outputs, graded under the SAME contract for the comparison arm. Empty
  # means the comparison stage reports that it had nothing to grade, rather than silently
  # producing a one-sided table.
  #
  # The run MUST have written complexes/: a peptide-only run cannot be graded on contacts,
  # interface-frame RMSD or paired dG at all. `sdedit_pocket14_...` is the default because the
  # pocket crop is the corrected receptor segmentation (the full-chain runs' numbers are
  # superseded), and it covers all 60 targets.
  SDEDIT_DIR="${SDEDIT_DIR:-${REPO}/evaluation_results/sdedit_pocket_20260906_180646}"
  SDEDIT_ROWS_DIR="${SDEDIT_ROWS_DIR:-${OUT_DIR}/sdedit_rows}"
  SDEDIT_SHARDS="${SDEDIT_SHARDS:-12}"
  SDEDIT_SHORTLIST="${SDEDIT_SHORTLIST:-${SHORTLIST}}"

  export METADATA RUN_ID OUT_DIR ROWS_DIR PDB_DIR REPORT_DIR
  export CHEMISTRIES BUDGETS PROTECTED
  export ATTEMPTS_PER_CALL CALLS MIN_SOLUTIONS MAX_RELEASED SHORTLIST DUMP_TOP
  export RELAX_REPEATS CP_REPEATS LP_REPEATS PERTURB_DEG SEED
  export SEARCH_SHARDS TIME_BUDGET_S EXAMPLES SMOKE
  export SDEDIT_DIR SDEDIT_ROWS_DIR SDEDIT_SHARDS SDEDIT_SHORTLIST

  mkdir -p "${OUT_DIR}" "${ROWS_DIR}" "${PDB_DIR}" "${REPORT_DIR}"
}

# ------------------------------------------------------------------ interpreter
_genkic_venv() {
  local path="${REPO}/.venv"
  [[ -x "${path}/bin/python" ]] || {
    echo "[preamble] FATAL: no interpreter at ${path}" >&2; return 1; }
  # shellcheck source=/dev/null
  source "${path}/bin/activate"
  export PYTHONPATH="${REPO}:${REPO}/src:${REPO}/script_utils:${PYTHONPATH:-}"

  # Prove PyRosetta imports and initializes before spending walltime. A missing or
  # unlicensed PyRosetta here is a fast, legible failure instead of a mystery three hours in.
  python - <<'PY' || return 1
import pandas, pyrosetta  # noqa: F401
from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init
_ensure_pyrosetta_init()
print(f"[preamble] .venv ok: pyrosetta {pyrosetta.__file__.split('/site-packages/')[-1]}, "
      f"pandas {pandas.__version__}")
PY
}

_genkic_banner() {
  echo "================================================================"
  echo "stage      : ${1:-?}"
  echo "run id     : ${RUN_ID}"
  echo "out dir    : ${OUT_DIR}"
  echo "metadata   : ${METADATA}"
  echo "chemistries: ${CHEMISTRIES}"
  echo "budgets    : ${BUDGETS}"
  echo "effort     : attempts=${ATTEMPTS_PER_CALL} calls=${CALLS} min_solutions=${MIN_SOLUTIONS} max_released=${MAX_RELEASED}"
  echo "shortlist  : ${SHORTLIST} (cp_repeats=${CP_REPEATS}, relax_repeats=${RELAX_REPEATS}, lp_repeats=${LP_REPEATS})"
  # `hostname` is not on PATH on every compute node here, and a bare $(hostname) prints a
  # command-not-found to stderr on each job. SLURMD_NODENAME is set by Slurm itself.
  echo "job        : ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_TASK_ID:-none} on ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
  echo "cpus/mem   : ${SLURM_CPUS_PER_TASK:-?} cpus, ${SLURM_MEM_PER_NODE:-?} MB"
  [[ "${SMOKE}" == "1" ]] && echo "SMOKE      : yes -- this run is NOT a result"
  echo "================================================================"
}
