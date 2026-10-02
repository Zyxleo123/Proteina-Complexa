#!/usr/bin/env bash
# Submitter for the two BACKBONE REPRESENTATION arms. Calls sbatch and nothing else -- no
# work of its own.
#
# Configuration travels as ONE timestamped env file passed as a POSITIONAL argument. Never
# `sbatch --export`: any explicit export list (including --export=NONE) sets
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the compute
# node, and the job is requeued and HELD -- stranding every dependent on Dependency.
#
# THE TWO ARMS ARE ONE EXPERIMENT WITH ONE VARIABLE. They share the trunk, the fork
# checkpoint, the AE, the discriminator architecture and its initialisation, the data split,
# the batch limits, the accumulation, the learning rates, the clipping, the loss weights, the
# movement budget, the packer and the optimizer-step budget. They differ in how a network
# output becomes a backbone, and in nothing else:
#
#   bb_cartesian   a free Cartesian residual on N, CA, C, O
#   bb_torsion     residual phi/psi/omega rotations + a rigid root transform
#
# Stages:
#   calibrate   CPU  measure the backbone template library and its reconstruction floor
#   pack-smoke  CPU  both arms + a COMPLETE Rosetta packing example each, on real natives
#   smoke       GPU  both arms on the REAL trunk: one-trunk-call count, zero-output
#                    identity, raw validity against the source's own, per-term gradient
#                    norms, the real-LP anchor, and a full training_step
#   arms        GPU  BOTH PILOTS as siblings, so they queue concurrently
#   <arm name>  GPU  one pilot: bb_cartesian | bb_torsion
#   panel       CPU  the FIXED 24-source / 4-paired-seed evaluation panel, built once
#   evaluate    GPU+CPU  per arm: generate the panel -> pack it all-atom; then one report
#                    over both arms. Set EVAL_CKPT_TAG to pick the milestone.
#   report      CPU  re-render the tables and the figure from rows already on disk
#   pilot            calibrate -> pack-smoke -> smoke -> arms, with dependencies wired
#
# Usage:
#   bash scripts/submit_cp2lp_backbone.sh --submit smoke
#   bash scripts/submit_cp2lp_backbone.sh --submit arms
#   bash scripts/submit_cp2lp_backbone.sh --dry-run pilot
#   bash scripts/submit_cp2lp_backbone.sh --submit arms --set SMOKE=1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ARMS=(bb_cartesian bb_torsion)

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

# One timestamped env file per submission, KEPT. Do not delete these while any job still
# references one: the pilots carry --requeue and re-source their env file on restart, so a
# missing file means the job falls back to every preamble default -- a different experiment
# wearing the same run_name.
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="${REPO}/env_files/cp2lp_bb_${STAGE}_${STAMP}.env"
{
  echo "# CP->LP backbone-arm submission ${STAMP}, stage=${STAGE}"
  echo "REPO=${REPO}"
  for kv in "${OVERRIDES[@]:-}"; do
    [[ -n "$kv" ]] && echo "$kv"
  done
} > "$ENV_FILE"
echo "env file: $ENV_FILE"
sed 's/^/  /' "$ENV_FILE"

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
  local -a deps=() extra=() sbopts=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dep)    [[ -n "${2:-}" ]] && deps+=("$2"); shift 2 ;;
      --sbopt)  sbopts+=("$2"); shift 2 ;;
      *)        extra+=("$1"); shift ;;
    esac
  done
  local -a cmd=(sbatch)
  if [[ "${#deps[@]}" -gt 0 ]]; then
    local joined; joined="$(IFS=:; echo "${deps[*]}")"
    cmd+=("--dependency=afterok:${joined}")
  fi
  [[ "${#sbopts[@]}" -gt 0 ]] && cmd+=("${sbopts[@]}")
  cmd+=("$script" "$ENV_FILE" "${extra[@]}")
  echo "+ ${cmd[*]}" >&2
  if [[ "$DRY" == "1" ]]; then echo "DRYRUN"; return 0; fi
  local out; out="$(scrub_env "${cmd[@]}")"
  echo "$out" >&2
  echo "${out##* }"
}

case "$STAGE" in
  calibrate)
    echo "calibrate: $(submit scripts/cp2lp_bb_calibrate.sbatch)" ;;

  pack-smoke)
    echo "pack-smoke: $(submit scripts/cp2lp_bb_pack_smoke.sbatch)" ;;

  smoke)
    # ONE JOB FOR BOTH ARMS, unlike the five-arm tranche's one-smoke-per-arm. The whole
    # point here is a single-variable comparison, and running the two arms in one process
    # under one seed is what makes the printed contrast between them comparable at all --
    # the first version used two processes and reported chain_intact 0.5000 against 1.0000
    # at identical zero edit, purely because the two dataloaders drew different batches.
    echo "smoke (both arms, shared seed): $(submit scripts/cp2lp_bb_smoke.sbatch)" ;;

  arms)
    echo "both pilots as siblings (they must queue concurrently, not chain):"
    for arm in "${ARMS[@]}"; do
      printf '  %-14s %s\n' "$arm" "$(submit scripts/cp2lp_train.sbatch "ARM=${arm}")"
    done ;;

  bb_cartesian|bb_torsion)
    echo "$STAGE: $(submit scripts/cp2lp_train.sbatch "ARM=${STAGE}")" ;;

  panel)
    echo "panel: $(submit scripts/cp2lp_bb_panel.sbatch)" ;;

  evaluate)
    # One GPU dump per arm, then ONE CPU packing job per arm, then ONE report over both.
    # The report is a sibling of nothing -- it needs every arm's packed rows, so it waits on
    # all of them.
    echo "evaluate at EVAL_CKPT_TAG=${EVAL_CKPT_TAG:-last-EMA}:"
    declare -a packs=()
    for arm in "${ARMS[@]}"; do
      g="$(submit scripts/cp2lp_bb_eval_gpu.sbatch "ARM=${arm}")"
      k="$(submit scripts/cp2lp_bb_pack_eval.sbatch --dep "$g" "ARM=${arm}")"
      packs+=("$k")
      printf '  %-14s gpu=%s pack=%s\n' "$arm" "$g" "$k"
    done
    dep=()
    for k in "${packs[@]}"; do dep+=(--dep "$k"); done
    echo "  report        $(submit scripts/cp2lp_bb_report.sbatch "${dep[@]}")" ;;

  report)
    echo "report: $(submit scripts/cp2lp_bb_report.sbatch)" ;;

  pilot)
    # The CPU stages first and as SIBLINGS of each other: the calibration and the packing
    # smoke read different things and neither needs the other.
    cal="$(submit scripts/cp2lp_bb_calibrate.sbatch)"
    pack="$(submit scripts/cp2lp_bb_pack_smoke.sbatch)"
    echo "calibrate : $cal"
    echo "pack-smoke: $pack"
    # The GPU smoke gates the pilots: it is what decides whether the arms are worth the
    # GPU hours, and it has already caught four defects that review did not.
    smoke="$(submit scripts/cp2lp_bb_smoke.sbatch --dep "$pack")"
    echo "smoke     : $smoke"
    for arm in "${ARMS[@]}"; do
      printf '  %-14s %s\n' "$arm" "$(submit scripts/cp2lp_train.sbatch --dep "$smoke" "ARM=${arm}")"
    done ;;

  *)
    echo "unknown stage: $STAGE" >&2
    echo "stages: calibrate | pack-smoke | smoke | arms | ${ARMS[*]} | panel |" >&2
    echo "        evaluate | report | pilot" >&2
    exit 2 ;;
esac
