# Shared preamble for the folding-potentials check.
#
# The check: flip ONE setting -- Boltz-2 inference potentials, off -> on -- on the same
# cohort, the same sequences, the same seeds, the same gate and the same estimator that
# produced the pilot arm table, and see whether the arm's advantage over the native
# peptide survives. docs/README_CPSEQ_POTENTIALS_CHECK_PLAN.md is the decision record.
#
# Sourced by every scripts/cpseq_potentials_*.sbatch. Every knob has a default here; the
# submitter writes ONE per-submission env file and passes its PATH as a positional
# argument.
#
# Never `sbatch --export`: any explicit export list (--export=ALL,K=v, --export=K=v and
# --export=NONE alike) sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD -- stranding every
# dependent job on Dependency with no error anyone will see. Arguments travel through the
# job record untouched, so a file path as a positional argument is the mechanism that
# works.
#
# Contract:
#   _pot_consume_args "$@"    # sources any arg that is a path, exports VAR=value args
#   _pot_defaults             # fills in anything still unset
#   _pot_banner <stage>       # node, GPU, resources, and the settings that matter
#
# `set -euo pipefail` belongs in the JOB BODY, not only here: if sourcing this file
# fails, a job without it keeps running under the system python and exits 0, releasing
# every afterok dependent behind a stage that did no work.
#
# A batch script cannot locate files relative to itself -- Slurm copies it to
# /var/spool/.../slurm_script before running -- so $SLURM_SUBMIT_DIR with an absolute
# fallback is how the repo is found.

REPO="${REPO:-${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa

# ---------------------------------------------------------------- arg handling
_pot_consume_args() {
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
_pot_defaults() {
  # Detected, not hardcoded. This work started on babel, where the scratch root is
  # /data/user_data/yixiz, and moved to autonlab, where it is /zfsauton/scratch/yixiz.
  # A fixed default would silently point a bare `sbatch` at a root that does not exist
  # on the cluster it is running on, and the first symptom would be "no cohort" several
  # jobs in. The submitter's env file overrides this anyway; the detection is for the
  # case where someone runs a job script by hand.
  if [ -z "${ZFS:-}" ]; then
    if [ -d /data/user_data/yixiz ]; then
      ZFS=/data/user_data/yixiz
    else
      ZFS=/zfsauton/scratch/yixiz
    fi
  fi

  # The run being tested, and its potentials-on sibling. SRC_DIR is READ-ONLY throughout
  # this check -- the -5.23 / -7.23 numbers live there and must stay exactly as they are
  # on disk and in the published bundle.
  SRC_DIR="${SRC_DIR:-${ZFS}/cpseq/llm_ladder}"
  POT_DIR="${POT_DIR:-${ZFS}/cpseq/llm_ladder_pot}"

  # v3, and not negotiable here: the numbers being compared against are v3 numbers, and
  # `llm_ladder` is gate-mixed (v3 22,642 rows / v5 6,150 / v4 1,087), so the report
  # refuses to print from it without a named gate.
  GATE="${GATE:-v3-cis-amide-muco-windows}"

  # `llm_search` leads. It is the only arm with no v5 rows (3,150 v3 / 0 v5), so its v3
  # cohort is whole, and it is the arm that reproduces its recorded number to the digit
  # -- which is what certifies the estimator. `random_mask_mpnn_fill` runs alongside it
  # so the section 0.1 tie stays honest rather than one arm being quietly promoted.
  LEAD_ARM="${LEAD_ARM:-llm_search}"
  ARMS="${ARMS:-llm_search random_mask_mpnn_fill}"

  # How many proposal rounds each arm folds. These must match what the SOURCE run folded,
  # or the fork measures a different draw depth and best-of-N makes that alone move the
  # number. The prepare stage reads the rounds actually present on disk and checks these
  # against them.
  LLM_ROUNDS="${LLM_ROUNDS:-2}"
  MASK_ROUNDS="${MASK_ROUNDS:-2}"

  # Which complexes to fold.
  #   fixed  the complexes the SOURCE run actually scored (status ok in its own report).
  #          This is the decision set: the plan's "same 10 complexes" comparison, and
  #          the only set on which a paired change is a statement about the setting.
  #   all    every parent in the cohort. Needed to OBSERVE the enlarged set -- a complex
  #          that gains a baseline under potentials can only show up if its draws were
  #          folded -- but it roughly doubles the cost, which would put this check above
  #          the ~90 GPU-h corpus run it is supposed to gate. So: not the default, and
  #          best run as a second wave for the LEAD arm only (the reference folds are
  #          shared across arms in one directory, so that wave pays for them once).
  PARENT_SET="${PARENT_SET:-fixed}"

  RUN_ID="${RUN_ID:-potcheck_$(date +%Y%m%d_%H%M%S)}"
  # Reports, parent lists and the env file. Deliberately NOT inside POT_DIR: that
  # directory is a run directory whose shape other tools expect, and a stray analysis
  # folder in it is how a later glob picks up the wrong thing.
  WORK_DIR="${WORK_DIR:-${ZFS}/cpseq/potcheck/${RUN_ID}}"

  RETENTION="${RETENTION:-rescorable}"
  LIMIT="${LIMIT:-0}"
  CHEMISTRY="${CHEMISTRY:-mainchain}"
  PY="${PYTHON_EXEC:-${REPO}/.venv/bin/python}"
  BOLTZ_PY="${BOLTZ_PYTHON:-${ZFS}/envs/boltz2/.venv/bin/python}"
  BOLTZ_CACHE="${BOLTZ_CACHE:-${ZFS}/boltz_cache}"

  export ZFS SRC_DIR POT_DIR GATE LEAD_ARM ARMS LLM_ROUNDS MASK_ROUNDS \
         PARENT_SET RUN_ID WORK_DIR RETENTION LIMIT CHEMISTRY PY \
         BOLTZ_PY BOLTZ_CACHE REPO
}

# The per-arm parent list the fold stage reads. One file per arm, because an arm whose
# proposals are missing for a complex has a different runnable set.
_pot_parents_file() { echo "${WORK_DIR}/parents_${1}.txt"; }

# ------------------------------------------------------------------- banner
# hostname, GPU model and the resource request go into every .out at startup. With
# slurmdbd unreachable there is no accounting record to recover the node from after the
# fact, and a job that died on a smaller card than it was sized for can otherwise look
# like a mysterious silent failure.
_pot_banner() {
  echo "=== cpseq_potentials: ${1:-stage} ==="
  echo "node:      $(hostname)"
  echo "job:       ${SLURM_JOB_ID:-local}  array_task=${SLURM_ARRAY_TASK_ID:-none}"
  echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=${CUDA_VISIBLE_DEVICES:-none}"
  echo "repo:      ${REPO}"
  echo "src_dir:   ${SRC_DIR}   (READ-ONLY)"
  echo "pot_dir:   ${POT_DIR}"
  echo "work_dir:  ${WORK_DIR}"
  echo "gate:      ${GATE}"
  echo "arms:      ${ARMS}   lead=${LEAD_ARM}"
  echo "parents:   ${PARENT_SET}"
  nvidia-smi -L 2>/dev/null || echo "gpu:       none (CPU stage)"
  "${PY}" -c 'import sys; print("python:    " + sys.executable)' 2>/dev/null \
    || echo "python:    MISSING at ${PY}"
}
