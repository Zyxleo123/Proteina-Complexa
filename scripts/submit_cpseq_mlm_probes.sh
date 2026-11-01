#!/bin/bash
# Submit the cpseq_mlm probe chain: is the plateau the task, or the model?
#
#   dump (GPU)  ->  softacc (CPU, afterok)        substitution tolerance + calibration
#   ambiguity (CPU, sibling)                      the data-side ceiling, no model involved
#
# The two branches are SIBLINGS, not a chain: the ambiguity probe needs no checkpoint, so
# making it wait behind a GPU job would only delay it. The dump->softacc edge is a real
# dependency and uses `afterok`.
#
# This submitter only calls sbatch. It does no work of its own, writes one timestamped env
# file per submission, and passes that file's path as a POSITIONAL argument -- never via
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets the job requeued and held
# with "(user env retrieval failed requeued held)", stranding every dependent job.
#
# Usage:
#   scripts/submit_cpseq_mlm_probes.sh                       # defaults: pocket run, highest ckpt
#   CPSEQ_MLM_PROBE_RUN=cpseq_mlm_nopocket scripts/submit_cpseq_mlm_probes.sh
#   CPSEQ_MLM_PROBE_CKPT=hf_step58000 scripts/submit_cpseq_mlm_probes.sh
#   SKIP_AMBIGUITY=1 scripts/submit_cpseq_mlm_probes.sh      # dump+softacc only
#   SKIP_DUMP=1 scripts/submit_cpseq_mlm_probes.sh           # re-reduce an existing dump

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source "$REPO/scripts/cpseq_mlm_env.sh"

# Scrub SLURM_* from this shell before calling sbatch. Submitting from inside an existing
# allocation otherwise leaks that job's resource environment into the new submissions --
# which is how a CPU reduction ends up inheriting a GPU request, or a --mem that does not
# match its own #SBATCH line.
for v in $(env | grep -oE '^SLURM_[A-Z_]+' || true); do unset "$v" || true; done

STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_DIR="$CPSEQ_MLM_ROOT/env_files"
mkdir -p "$ENV_DIR" "$CPSEQ_MLM_LOG_DIR" "$CPSEQ_MLM_PROBE_DIR"
ENV_FILE="$ENV_DIR/probes_${CPSEQ_MLM_PROBE_RUN}_${STAMP}.env"

# Only the knobs this submission actually pins are written, so the shared preamble stays
# the single source of defaults and the env file records the DECISIONS rather than a copy
# of every variable. A file (not a variable list) also means spaces need no encoding.
{
  echo "# cpseq_mlm probe chain, submitted $STAMP from $(hostname)"
  echo "export CPSEQ_MLM_PROBE_RUN='${CPSEQ_MLM_PROBE_RUN}'"
  echo "export CPSEQ_MLM_PROBE_CKPT='${CPSEQ_MLM_PROBE_CKPT}'"
  echo "export CPSEQ_MLM_PROBE_SHUFFLE='${CPSEQ_MLM_PROBE_SHUFFLE}'"
  echo "export CPSEQ_MLM_PROBE_NBOOT='${CPSEQ_MLM_PROBE_NBOOT}'"
  echo "export CPSEQ_MLM_CORPUS_DIR='${CPSEQ_MLM_CORPUS_DIR}'"
  echo "export CPSEQ_MLM_PROBE_DIR='${CPSEQ_MLM_PROBE_DIR}'"
  echo "export CPSEQ_MLM_PRIORS_CACHE='${CPSEQ_MLM_PRIORS_CACHE}'"
  echo "export CPSEQ_MLM_AMBIG_GLOBS='${CPSEQ_MLM_AMBIG_GLOBS}'"
} > "$ENV_FILE"

echo "env file:  $ENV_FILE"
echo "run:       $CPSEQ_MLM_PROBE_RUN"
echo "ckpt:      ${CPSEQ_MLM_PROBE_CKPT:-<highest hf_step*>}"
echo "probe dir: $CPSEQ_MLM_PROBE_DIR"
echo

# Verify the preamble is self-sufficient under a stripped environment, the way a compute
# node sees it. A missing default shows up here as an unbound variable instead of as a job
# that runs with an empty path and writes its output somewhere unexpected.
if ! env -i bash -c "set -u; source '$REPO/scripts/cpseq_mlm_env.sh'; \
     : \"\$CPSEQ_MLM_PROBE_DIR\" \"\$CPSEQ_MLM_PRIORS_CACHE\" \"\$CPSEQ_MLM_AMBIG_GLOBS\"" 2>/dev/null; then
  echo "FATAL: scripts/cpseq_mlm_env.sh is not self-sufficient under 'env -i'."
  echo "A compute node does not inherit the login environment, so a default that only"
  echo "resolves here would be empty in the job. Fix the preamble before submitting."
  exit 1
fi

DUMP_JOB=""
if [ "${SKIP_DUMP:-0}" != "1" ]; then
  DUMP_JOB=$(sbatch --parsable \
    --job-name="dump_${CPSEQ_MLM_PROBE_RUN}" \
    scripts/cpseq_mlm_dump_preds.sbatch "$ENV_FILE")
  echo "submitted dump      : $DUMP_JOB   (GPU, general)"
fi

if [ -n "$DUMP_JOB" ]; then
  SOFT_JOB=$(sbatch --parsable \
    --dependency=afterok:"$DUMP_JOB" \
    --job-name="softacc_${CPSEQ_MLM_PROBE_RUN}" \
    scripts/cpseq_mlm_softacc.sbatch "$ENV_FILE")
  echo "submitted softacc   : $SOFT_JOB   (CPU, afterok:$DUMP_JOB)"
else
  SOFT_JOB=$(sbatch --parsable \
    --job-name="softacc_${CPSEQ_MLM_PROBE_RUN}" \
    scripts/cpseq_mlm_softacc.sbatch "$ENV_FILE")
  echo "submitted softacc   : $SOFT_JOB   (CPU, no dependency -- re-reducing existing dumps)"
fi

if [ "${SKIP_AMBIGUITY:-0}" != "1" ]; then
  AMBIG_JOB=$(sbatch --parsable \
    --job-name="ambiguity" \
    scripts/cpseq_mlm_ambiguity.sbatch "$ENV_FILE")
  echo "submitted ambiguity : $AMBIG_JOB   (CPU, sibling -- no dependency)"
fi

echo
echo "watch:  squeue -u $USER"
echo "logs:   $CPSEQ_MLM_LOG_DIR/cpseq_mlm_{dump,softacc,ambiguity}_*_<jobid>.out"
echo "out:    $CPSEQ_MLM_PROBE_DIR"
echo
echo "Read the dump job's .out before trusting the chain: it gates on reproducing the"
echo "run's own val/acc/all, and a gate failure exits 0 by design."
