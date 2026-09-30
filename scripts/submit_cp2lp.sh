#!/usr/bin/env bash
# Submitter for the CP->LP generator. Calls sbatch and nothing else -- no work of its own.
#
# Configuration travels as ONE timestamped env file passed as a POSITIONAL argument.
# Never `sbatch --export`: any explicit export list (including --export=NONE) sets
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the compute
# node, and the job is requeued and HELD -- stranding every dependent on Dependency.
#
# Stages:
#   splits       CPU  build the receptor-family-disjoint train/eval metadata
#   smoke        GPU  checkpoints load together; one batch through gen -> AE -> D -> backward
#   gan          GPU  the adversarial arm
#   contactonly  GPU  the ablation (same everything, adversarial term off)
#   both         GPU  gan + contactonly as SIBLINGS, so they queue concurrently
#   generate     GPU  triplets from each trained arm (same CPs, same budget)
#   reference    GPU  real bound LPs measured by the same code
#   report       CPU  one table + figure comparing the arms against real
#   pilot        the whole chain: smoke -> both -> (generate, reference) -> report
#
# Usage:
#   bash scripts/submit_cp2lp.sh --submit smoke
#   bash scripts/submit_cp2lp.sh --submit pilot
#   bash scripts/submit_cp2lp.sh --submit gan --set GEN_SEEDS=0,1,2
#   bash scripts/submit_cp2lp.sh --dry-run pilot
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

STAGE=""
DRY=0
declare -a OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)  STAGE="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; STAGE="${2:-}"; shift 2 ;;
    --set)     OVERRIDES+=("${2:-}"); shift 2 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$STAGE" ]] || { echo "need --submit <stage>; see --help" >&2; exit 2; }

mkdir -p slurm_logs env_files

# One timestamped env file per submission, KEPT. Do not clean these up while any job still
# references one: the training arms carry --requeue and re-source their env file on
# restart, so a missing file means the job silently falls back to every preamble default --
# a different experiment wearing the same run_name. They are a few hundred bytes each, and
# the timestamp is what makes a past submission reproducible.
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="${REPO}/env_files/cp2lp_${STAGE}_${STAMP}.env"
{
  echo "# CP->LP submission ${STAMP}, stage=${STAGE}"
  echo "REPO=${REPO}"
  for kv in "${OVERRIDES[@]:-}"; do
    [[ -n "$kv" ]] && echo "$kv"
  done
} > "$ENV_FILE"
echo "env file: $ENV_FILE"
sed 's/^/  /' "$ENV_FILE"

# Submitting from inside an allocation leaks SLURM_* into the child job's environment,
# where it confuses the new job's own bookkeeping. Scrub them before calling sbatch.
scrub_env() {
  env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NNODES \
      -u SLURM_NTASKS -u SLURM_CPUS_PER_TASK -u SLURM_MEM_PER_NODE \
      -u SLURM_JOB_NAME -u SLURM_ARRAY_TASK_ID -u SLURM_SUBMIT_DIR \
      -u SLURM_JOB_PARTITION -u SLURM_GET_USER_ENV "$@"
}

submit() {
  local script="$1"; shift
  local -a deps=()
  local -a extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dep) [[ -n "${2:-}" ]] && deps+=("$2"); shift 2 ;;
      *)     extra+=("$1"); shift ;;
    esac
  done
  local -a cmd=(sbatch)
  if [[ "${#deps[@]}" -gt 0 ]]; then
    local joined
    joined="$(IFS=:; echo "${deps[*]}")"
    cmd+=("--dependency=afterok:${joined}")
  fi
  cmd+=("$script" "$ENV_FILE" "${extra[@]}")
  echo "+ ${cmd[*]}" >&2
  if [[ "$DRY" == "1" ]]; then
    echo "DRYRUN"
    return 0
  fi
  local out
  out="$(scrub_env "${cmd[@]}")"
  echo "$out" >&2
  echo "${out##* }"
}

case "$STAGE" in
  splits)
    JID="$(submit scripts/cp2lp_splits.sbatch)"
    echo "splits: $JID"
    ;;

  smoke)
    JID="$(submit scripts/cp2lp_smoke.sbatch)"
    echo "smoke: $JID"
    ;;

  gan|contactonly)
    JID="$(submit scripts/cp2lp_train.sbatch "ARM=${STAGE}")"
    echo "${STAGE}: $JID"
    ;;

  both)
    # Siblings, not a chain: the two arms are independent and must queue concurrently.
    G="$(submit scripts/cp2lp_train.sbatch ARM=gan)"
    C="$(submit scripts/cp2lp_train.sbatch ARM=contactonly)"
    echo "gan: $G   contactonly: $C"
    ;;

  generate)
    G="$(submit scripts/cp2lp_generate.sbatch ARM=gan)"
    C="$(submit scripts/cp2lp_generate.sbatch ARM=contactonly)"
    echo "generate gan: $G   contactonly: $C"
    ;;

  reference)
    JID="$(submit scripts/cp2lp_reference.sbatch ARM=gan)"
    echo "reference: $JID"
    ;;

  report)
    JID="$(submit scripts/cp2lp_report.sbatch)"
    echo "report: $JID"
    ;;

  pilot)
    S="$(submit scripts/cp2lp_smoke.sbatch)"
    # Both arms wait on the smoke: there is no point burning two GPU-days to discover the
    # checkpoints do not load together.
    G="$(submit scripts/cp2lp_train.sbatch --dep "$S" ARM=gan)"
    C="$(submit scripts/cp2lp_train.sbatch --dep "$S" ARM=contactonly)"
    # Generation waits on its OWN arm only, so a failure in one arm does not block the other.
    GG="$(submit scripts/cp2lp_generate.sbatch --dep "$G" ARM=gan)"
    GC="$(submit scripts/cp2lp_generate.sbatch --dep "$C" ARM=contactonly)"
    # The reference needs a run dir for its exp_config, nothing more.
    R="$(submit scripts/cp2lp_reference.sbatch --dep "$G" ARM=gan)"
    RP="$(submit scripts/cp2lp_report.sbatch --dep "$GG" --dep "$GC" --dep "$R")"
    echo
    echo "pilot chain:"
    echo "  smoke        $S"
    echo "  train gan    $G   (afterok $S)"
    echo "  train ctrl   $C   (afterok $S)"
    echo "  gen gan      $GG  (afterok $G)"
    echo "  gen ctrl     $GC  (afterok $C)"
    echo "  reference    $R   (afterok $G)"
    echo "  report       $RP  (afterok $GG,$GC,$R)"
    ;;

  *)
    echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
