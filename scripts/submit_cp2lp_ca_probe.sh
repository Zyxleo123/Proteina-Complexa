#!/usr/bin/env bash
# Submitter for the CA-spacing probe. Calls sbatch and nothing else -- no work of its own.
#
# WHAT IT ANSWERS. Every adversarial one-pass arm exports at ~1% `chain_intact` while the
# control exports at ~79%, and the 0.02 nm bond tolerance is NOT at fault (it is calibrated
# on a plateau, natives sit 6x inside it). So the backbones are genuinely broken, and the
# probe asks which decoder input breaks them -- see `script_utils/cp2lp_ca_spacing_probe.py`
# for the mechanism and the readings.
#
# Configuration travels as ONE timestamped env file passed as a POSITIONAL argument.
# Never `sbatch --export`: any explicit export list (including --export=NONE) sets
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the compute
# node, and the job is requeued and HELD.
#
# Stages:
#   decisive    GPU  the three runs that settle the mechanism, as siblings:
#                    onepass_gan (broken, intact ~1%), onepass_contactonly (working, ~79%),
#                    and the real-LP reference (native CA-CA through the same code path)
#   all         GPU  every arm at the common milestone, for the delta-vs-intact dose
#                    response, plus the reference
#   one         GPU  a single arm (--set ARM=onepass_gan_nce)
#   reference   GPU  the real-LP reference alone
#
# The arms are probed at `g001300` by default, NOT at `last`: g001300 is finalised and
# immutable, it exists for all six runs so the comparison is single-variable, and the
# mechanism is already fully present there (the gan arm's bond MAE is 0.027 nm by g=1300,
# above the 0.02 gate). `last.ckpt` is being rewritten by the live jobs and reading it races
# the checkpoint writer. Non-EMA is deliberate too: the logged `geom_chain_intact_frac` the
# probe is explaining comes from the live weights, so EMA would answer a different question.
#
# Usage:
#   bash scripts/submit_cp2lp_ca_probe.sh --submit decisive
#   bash scripts/submit_cp2lp_ca_probe.sh --submit all
#   bash scripts/submit_cp2lp_ca_probe.sh --submit one --set ARM=onepass_gan_nce
#   bash scripts/submit_cp2lp_ca_probe.sh --dry-run decisive
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ALL_ARMS=(onepass_contactonly onepass_gan onepass_gan_ar1 onepass_gan_noise onepass_gan_nce)

STAGE=""
DRY=0
declare -a OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)  STAGE="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; STAGE="${2:-}"; shift 2 ;;
    --set)     OVERRIDES+=("${2:-}"); shift 2 ;;
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$STAGE" ]] || { echo "need --submit <stage>; see --help" >&2; exit 2; }

mkdir -p slurm_logs env_files

# The PID is in the name, not just the timestamp. MEASURED the hard way: two
# `--submit one` invocations issued in the same second produced ONE env file, the second
# overwrote the first, and both jobs read the survivor -- so both probed the same arm and
# raced on one output path. A second-resolution stamp is not a unique name.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="${REPO}/env_files/cp2lp_ca_probe_${STAGE}_${STAMP}.env"
{
  echo "# CP->LP CA-spacing probe ${STAMP}, stage=${STAGE}"
  echo "REPO=${REPO}"
  for kv in "${OVERRIDES[@]:-}"; do
    [[ -n "$kv" ]] && echo "$kv"
  done
} > "$ENV_FILE"
echo "env file: $ENV_FILE"
sed 's/^/  /' "$ENV_FILE"

# Submitting from inside an allocation leaks SLURM_* into the child job. Scrub first.
scrub_env() {
  env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NNODES \
      -u SLURM_NTASKS -u SLURM_CPUS_PER_TASK -u SLURM_MEM_PER_NODE \
      -u SLURM_JOB_NAME -u SLURM_ARRAY_TASK_ID -u SLURM_SUBMIT_DIR \
      -u SLURM_JOB_PARTITION -u SLURM_GET_USER_ENV "$@"
}

submit() {
  local -a cmd=(sbatch scripts/cp2lp_ca_probe.sbatch "$ENV_FILE" "$@")
  echo "+ ${cmd[*]}" >&2
  if [[ "$DRY" == "1" ]]; then echo "DRYRUN"; return 0; fi
  local out; out="$(scrub_env "${cmd[@]}")"
  echo "$out" >&2
  echo "${out##* }"
}

case "$STAGE" in
  decisive)
    # The pair that defines the tension, plus the native yardstick. The reference needs an
    # arm only for its run dir and AE; it never uses the editor.
    B="$(submit ARM=onepass_gan)"
    C="$(submit ARM=onepass_contactonly)"
    R="$(submit ARM=onepass_gan PROBE_REFERENCE=1)"
    echo "gan(broken)      $B"
    echo "contactonly(ok)  $C"
    echo "real reference   $R"
    ;;
  all)
    for a in "${ALL_ARMS[@]}"; do
      J="$(submit "ARM=${a}")"
      echo "${a}  ${J}"
    done
    J="$(submit ARM=onepass_warmup PROBE_CKPT_TAG=g000300)"
    echo "fork(g000300)  ${J}"
    R="$(submit ARM=onepass_gan PROBE_REFERENCE=1)"
    echo "real reference ${R}"
    ;;
  one)
    J="$(submit)"
    echo "one  ${J}"
    ;;
  reference)
    R="$(submit ARM=onepass_gan PROBE_REFERENCE=1)"
    echo "real reference ${R}"
    ;;
  *)
    echo "unknown stage: $STAGE (see --help)" >&2; exit 2 ;;
esac
