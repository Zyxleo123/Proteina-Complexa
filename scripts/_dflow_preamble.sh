#!/bin/bash
# Shared preamble for every D-Flow LP -> CP job. Sourced, never executed.
#
# Two things here are not optional on this cluster:
#
# 1. PYTHONPATH must point at THIS checkout's `src`. The venv installs
#    `proteinfoundation` editable via `_editable_impl_proteinfoundation.pth`, which
#    hardcodes the MAIN repo's path -- so a job run from a git worktree silently imports
#    the main checkout's source and any change made here has no effect. Measured: without
#    this line `import proteinfoundation` resolves outside the worktree.
#
# 2. Configuration arrives as a positional ENV FILE, never via `sbatch --export`. Any
#    explicit export list -- `--export=ALL,K=v`, `--export=K=v` and `--export=NONE` alike --
#    makes sbatch set SLURM_GET_USER_ENV=1; slurmd then tries to rebuild the login
#    environment on the compute node, the lookup fails, and the job is requeued and HELD
#    with "(user env retrieval failed requeued held)", stranding every afterok dependent on
#    `Dependency`. Measured on this cluster, jobs 22184 and 22243.
#
# `set -euo pipefail` belongs in each job BODY as well as here: a failed `source` would
# otherwise leave the job running under the system python and still exit 0, releasing every
# afterok dependent behind a stage that did no work.

set -euo pipefail

# A batch script cannot locate itself: Slurm copies it to /var/spool/.../slurm_script, so
# ${BASH_SOURCE[0]} points at the spool directory. Use SLURM_SUBMIT_DIR with a fallback.
DFLOW_REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$DFLOW_REPO"

if [[ -f "${DFLOW_REPO}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${DFLOW_REPO}/env.sh"
fi

# Argument handling: a path is sourced as an env file, VAR=value is exported directly.
# Arguments travel through the job record untouched, which is what makes this safe.
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    echo "sourcing env file: $arg"
    # shellcheck disable=SC1090
    source "$arg"
  elif [[ "$arg" == *=* ]]; then
    echo "exporting: $arg"
    export "${arg?}"
  fi
done

export PYTHONPATH="${DFLOW_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
DFLOW_PY="${PYTHON_EXEC:-${DFLOW_REPO}/.venv/bin/python}"
export DFLOW_PY

mkdir -p "${DFLOW_REPO}/slurm_logs"

# Every job prints where it landed and what it got: with slurmdbd unreachable here `sacct`
# returns nothing and still exits 0, so the node and the GPU are recoverable only from the
# job's own log. A pass on one node and a failure on another is a hardware difference until
# proven otherwise, and a CUDA OOM can kill the process with no Python traceback at all.
echo "=============================================================="
echo "host:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none} ${SLURM_ARRAY_TASK_ID:+array task $SLURM_ARRAY_TASK_ID}"
echo "partition: ${SLURM_JOB_PARTITION:-n/a}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} gres=${SLURM_JOB_GRES:-none}"
nvidia-smi -L 2>/dev/null || echo "gpus:      none visible"
echo "python:    ${DFLOW_PY}"
echo "repo:      ${DFLOW_REPO}"
echo "started:   $(date -Is)"
echo "=============================================================="

dflow_require_file() {
  local path="$1" what="$2"
  [[ -f "$path" ]] || { echo "FATAL: ${what} missing: ${path}"; exit 1; }
}

# Confirm `proteinfoundation` resolves inside THIS checkout, not the main one the editable
# install points at. A job that silently imports the wrong source tree produces results for
# code nobody changed, which is indistinguishable from a null result.
dflow_preflight_python() {
  DFLOW_REPO="$DFLOW_REPO" "$DFLOW_PY" - <<'PY'
import os, pathlib, sys
import proteinfoundation
resolved = pathlib.Path(proteinfoundation.__file__).resolve()
repo = pathlib.Path(os.environ["DFLOW_REPO"]).resolve()
print(f"proteinfoundation -> {resolved}")
if repo not in resolved.parents:
    sys.exit(
        f"FATAL: proteinfoundation resolved to {resolved}, which is OUTSIDE {repo}. "
        "The editable install's .pth hardcodes another checkout; PYTHONPATH did not win."
    )
PY
}
