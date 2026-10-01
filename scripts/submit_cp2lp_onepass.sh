#!/usr/bin/env bash
# Submitter for the ONE-PASS CP->LP editor and its five arms. Calls sbatch and nothing
# else -- no work of its own.
#
# Configuration travels as ONE timestamped env file passed as a POSITIONAL argument.
# Never `sbatch --export`: any explicit export list (including --export=NONE) sets
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the compute
# node, and the job is requeued and HELD -- stranding every dependent on Dependency.
#
# THE FIVE ARMS ARE DISJOINT EXPERIMENTS. They share the AE, the fork checkpoint, the seed,
# the source-CP order, the generator-noise stream, the data split, the batch limits, the
# accumulation, the learning rates, the clipping, the loss weights and the evaluation
# manifest. They differ by ONE config line each. Every addon is compared against
# `onepass_gan`; none is compared against another addon.
#
# Stages:
#   manifest    CPU  build the fixed 64-parent / 8-seed evaluation manifest (once)
#   smoke       GPU  the compact preflight, ONE PER ARM as siblings (trunk-call count,
#                    frame check, arm exclusivity, one real G/D update with the addon live)
#   smoke-one   GPU  the same, for a single arm (--set SMOKE_ARM=onepass_gan_nce)
#   warmup      GPU  the COMMON FORK: 300 generator updates, adversary off
#   pin         CPU  snapshot the fork so all five arms start from identical weights
#   arms        GPU  all five arms as SIBLINGS, so they queue concurrently
#   <arm name>  GPU  one arm: onepass_contactonly | onepass_gan | onepass_gan_ar1
#                            | onepass_gan_noise | onepass_gan_nce
#   evaluate    GPU  the fixed manifest through each arm at EVAL_CKPT_TAG
#   report      CPU  the five-arm comparison table and figure
#   pilot            the whole chain with dependencies wired
#
# Usage:
#   bash scripts/submit_cp2lp_onepass.sh --submit smoke
#   bash scripts/submit_cp2lp_onepass.sh --submit pilot
#   bash scripts/submit_cp2lp_onepass.sh --submit evaluate --set EVAL_CKPT_TAG=g005300-EMA
#   bash scripts/submit_cp2lp_onepass.sh --dry-run pilot
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ARMS=(onepass_contactonly onepass_gan onepass_gan_ar1 onepass_gan_noise onepass_gan_nce)

# NOT a member of ARMS. The five-arm addon comparison is a closed experiment -- adding a sixth
# member would change what that table means. `onepass_internal` is a separate question (does
# the CA edit space matter?) with its own control (`onepass_gan`), so it is submitted on its
# own and compared pairwise.
EXTRA_ARMS=(onepass_internal)

STAGE=""
DRY=0
declare -a OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)  STAGE="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; STAGE="${2:-}"; shift 2 ;;
    --set)     OVERRIDES+=("${2:-}"); shift 2 ;;
    -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$STAGE" ]] || { echo "need --submit <stage>; see --help" >&2; exit 2; }

mkdir -p slurm_logs env_files

# One timestamped env file per submission, KEPT. Do not delete these while any job still
# references one: the arms carry --requeue and re-source their env file on restart, so a
# missing file means the job falls back to every preamble default -- a different experiment
# wearing the same run_name.
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="${REPO}/env_files/cp2lp_onepass_${STAGE}_${STAMP}.env"
{
  echo "# CP->LP one-pass submission ${STAMP}, stage=${STAGE}"
  echo "REPO=${REPO}"
  for kv in "${OVERRIDES[@]:-}"; do
    [[ -n "$kv" ]] && echo "$kv"
  done
} > "$ENV_FILE"
echo "env file: $ENV_FILE"
sed 's/^/  /' "$ENV_FILE"

# `--set` values must also be visible to THIS script, not only to the job. Several stages read
# an override to decide what to submit -- `smoke-one` reads SMOKE_ARM -- and without this the
# documented `--submit smoke-one --set SMOKE_ARM=onepass_gan_nce` silently submitted the
# DEFAULT arm: the value reached the env file, the job then had ARM= passed positionally by
# this script, and a positional argument beats the env file. Both jobs succeed and the two
# "different" smokes simply agree, so nothing in the logs says it happened.
for kv in "${OVERRIDES[@]:-}"; do
  [[ -n "$kv" && "$kv" == *=* ]] && export "${kv?}"
done

# Submitting from inside an allocation leaks SLURM_* into the child job, where it confuses
# the new job's own bookkeeping. Scrub before calling sbatch.
scrub_env() {
  env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NNODES \
      -u SLURM_NTASKS -u SLURM_CPUS_PER_TASK -u SLURM_MEM_PER_NODE \
      -u SLURM_JOB_NAME -u SLURM_ARRAY_TASK_ID -u SLURM_SUBMIT_DIR \
      -u SLURM_JOB_PARTITION -u SLURM_GET_USER_ENV "$@"
}

submit() {
  local script="$1"; shift
  local -a deps=() extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dep) [[ -n "${2:-}" ]] && deps+=("$2"); shift 2 ;;
      *)     extra+=("$1"); shift ;;
    esac
  done
  local -a cmd=(sbatch)
  if [[ "${#deps[@]}" -gt 0 ]]; then
    local joined; joined="$(IFS=:; echo "${deps[*]}")"
    cmd+=("--dependency=afterok:${joined}")
  fi
  cmd+=("$script" "$ENV_FILE" "${extra[@]}")
  echo "+ ${cmd[*]}" >&2
  if [[ "$DRY" == "1" ]]; then echo "DRYRUN"; return 0; fi
  local out; out="$(scrub_env "${cmd[@]}")"
  echo "$out" >&2
  echo "${out##* }"
}

is_arm() {
  local a
  for a in "${ARMS[@]}" "${EXTRA_ARMS[@]}"; do [[ "$1" == "$a" ]] && return 0; done
  return 1
}

case "$STAGE" in
  manifest)
    echo "manifest: $(submit scripts/cp2lp_onepass_manifest.sbatch)" ;;

  smoke)
    # ONE SMOKE PER ARM, as siblings. The addon-specific paths -- the corruption, the aR1
    # penalty, the contrastive term -- are only reachable through their own arm's config, so
    # smoking `onepass_gan` alone would leave three of the five arms unexercised. Siblings
    # rather than one looping job so a failure in one arm does not hide the other four.
    #
    # It needs no fork: the sbatch falls back to the lpmix init and says so. The smoke is
    # what gates on whether producing the fork is worth the GPU hours.
    echo "one smoke per arm (siblings):"
    for arm in "${ARMS[@]}"; do
      printf '  %-22s %s\n' "$arm" "$(submit scripts/cp2lp_onepass_smoke.sbatch "ARM=${arm}")"
    done ;;

  smoke-one)
    echo "smoke: $(submit scripts/cp2lp_onepass_smoke.sbatch "ARM=${SMOKE_ARM:-onepass_gan}")" ;;

  warmup)
    echo "warmup (the fork): $(submit scripts/cp2lp_train.sbatch ARM=onepass_warmup)" ;;

  pin)
    echo "pin: $(submit scripts/cp2lp_fork_pin.sbatch ARM=onepass_warmup)" ;;

  arms)
    echo "five arms as siblings (they must queue concurrently, not chain):"
    for arm in "${ARMS[@]}"; do
      printf '  %-22s %s\n' "$arm" "$(submit scripts/cp2lp_train.sbatch "ARM=${arm}")"
    done ;;

  evaluate)
    echo "evaluate at EVAL_CKPT_TAG (default last-EMA):"
    for arm in "${ARMS[@]}"; do
      printf '  %-22s %s\n' "$arm" "$(submit scripts/cp2lp_onepass_eval.sbatch "ARM=${arm}")"
    done ;;

  report)
    echo "report: $(submit scripts/cp2lp_onepass_report.sbatch)" ;;

  pilot)
    M="$(submit scripts/cp2lp_onepass_manifest.sbatch)"
    S="$(submit scripts/cp2lp_onepass_smoke.sbatch ARM=onepass_gan)"   # the reference arm gates the chain
    # The warmup waits on the smoke: there is no point burning GPU hours on a fork whose
    # editor turns out to be miswired.
    W="$(submit scripts/cp2lp_train.sbatch --dep "$S" ARM=onepass_warmup)"
    P="$(submit scripts/cp2lp_fork_pin.sbatch --dep "$W" ARM=onepass_warmup)"
    declare -a AJ=() EJ=()
    for arm in "${ARMS[@]}"; do
      j="$(submit scripts/cp2lp_train.sbatch --dep "$P" "ARM=${arm}")"
      AJ+=("$j")
      # Each arm's evaluation waits on ITS OWN arm only, so one arm failing does not block
      # the other four from being measured.
      EJ+=("$(submit scripts/cp2lp_onepass_eval.sbatch --dep "$j" "ARM=${arm}" --dep "$M")")
    done
    declare -a RD=()
    for j in "${EJ[@]}"; do RD+=(--dep "$j"); done
    R="$(submit scripts/cp2lp_onepass_report.sbatch "${RD[@]}")"
    echo
    echo "pilot chain:"
    echo "  manifest   $M"
    echo "  smoke      $S"
    echo "  warmup     $W   (afterok $S)"
    echo "  fork pin   $P   (afterok $W)"
    for i in "${!ARMS[@]}"; do
      printf '  %-22s train %s (afterok %s)   eval %s\n' "${ARMS[$i]}" "${AJ[$i]}" "$P" "${EJ[$i]}"
    done
    echo "  report     $R   (afterok every eval)"
    echo
    echo "NOTE: the arms' budgets are in GENERATOR updates (cp2lp.max_g_updates), not"
    echo "      global_step -- global_step counts G+D optimizer steps under manual"
    echo "      optimization, so it advances twice as fast in an adversarial arm." ;;

  *)
    if is_arm "$STAGE"; then
      echo "${STAGE}: $(submit scripts/cp2lp_train.sbatch "ARM=${STAGE}")"
    else
      echo "unknown stage: $STAGE" >&2
      echo "arms are: ${ARMS[*]}" >&2
      exit 2
    fi ;;
esac
