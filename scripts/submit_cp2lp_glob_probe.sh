#!/usr/bin/env bash
# Submitter for the glob-dependence probe. Calls sbatch and nothing else.
#
# WHAT IT ANSWERS. Every CP->LP arm pins `d_acc` near 0.98 and nothing moves it. The
# clash-norm arm fixed a real steric defect and `d_acc` did not budge (0.9872 vs 0.9879);
# the torsion arm reaches near-perfect chain geometry and still scores 0.9840. Both of the
# two tells docs/old_docs/README_CP2LP.md names have now been manipulated and neither moved the
# number -- so `d_acc` is reading something else.
#
# The suspect is in the feature list, not in the data: `build_features` feeds D the terminal
# gap and the radius of gyration as explicit global columns, and real bound LPs sit at
# ~2.14 nm separation while the arms reach 1.04-1.25. If D leans on those, `d_acc` restates
# the unsolved task ("did you open the ring all the way") rather than judging realism, and
# no loss-term experiment can move it. `d_shortcut_probe_acc` cannot see this: it reads
# `glob[:, 2:]` and masks the peptide besides.
#
# Stages:
#   decisive   the two arms whose nulls motivated the probe, as siblings behind the mix:
#              onepass_internal_clashnorm (the clash arm, d_acc 0.9872 at g020300) and
#              bb_torsion (near-perfect geometry, d_acc 0.9840 at g010300), then the report
#   all        every arm that has a discriminator and a milestone, then the report
#   one        a single arm (--set ARM=bb_cartesian GLOB_CKPT_TAG=g010300)
#   mix        stage 1 alone, to inspect the held-out parquet before spending a GPU
#   report     stage 3 alone, to re-render the tables from saved rows
#
# Each arm is probed at its OWN final milestone, because the arms do not share one: the
# clash pair ran to g020300 and the backbone pair to g010300. The milestone travels per
# submission rather than as one global default, so a shared tag cannot silently gate out
# half the comparison.
#
# NON-EMA checkpoints, deliberately. The EMA files may track the generator only, and a
# checkpoint without discriminator weights would leave a randomly initialised D scoring
# ~0.5 -- which reads as "no dependence" from a model that never learned anything. The probe
# hard-fails on a missing `discriminator.*` key, but the default should not invite it.
#
# Configuration travels as ONE timestamped env file passed as a POSITIONAL argument.
# Never `sbatch --export`: any explicit export list (including --export=NONE) sets
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the compute
# node, and the job is requeued and HELD, stranding every dependent job on Dependency.
#
# Usage:
#   bash scripts/submit_cp2lp_glob_probe.sh --submit release   # the interface-column pass
#   bash scripts/submit_cp2lp_glob_probe.sh --submit decisive
#   bash scripts/submit_cp2lp_glob_probe.sh --submit one --set ARM=bb_cartesian
#   bash scripts/submit_cp2lp_glob_probe.sh --submit report
#   bash scripts/submit_cp2lp_glob_probe.sh --dry-run release
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# arm:milestone. The milestone is part of the identity of the measurement here -- see the
# header -- so it is not factored out into a default.
ALL_ARMS=(
  onepass_internal_clashnorm:g020300
  onepass_internal:g020300
  bb_torsion:g010300
  bb_cartesian:g010300
  onepass_gan:g010300
)

# THE RELEASE PAIR, and the reason they are a stage of their own rather than entries in
# ALL_ARMS. The first glob probe held the openness columns (0, 1) and measured an EXACT null,
# 0.9817 -> 0.9817, which leaves (2, 3) -- `contact_count` and `mean_min_heavy_nm` -- as the
# only place a glob tell can live. Those two columns were never held out. The probe now holds
# them, and the release arms are the right population to hold them ON:
#
# MEASURED from the packed release eval, best-of-3 seeds, selected by the predeclared order:
#
#                                   min receptor dist (p50)   clashing
#     cp2lp_bb_release_cold                   1.23 A            74.0%
#     cp2lp_bb_release_warm                   1.65 A            61.5%
#     ..._cold__source  (zero edit)           2.70 A             0.0%
#     real bound LPs (backbone, n=536)        2.62 A               --
#
# So the IDENTICAL Rosetta protocol packs a real backbone with zero clashes and the generated
# backbones at 1.2-1.7 A, which is inside a covalent bond length. Something is paying the
# generator to interpenetrate, and `contact_count` is the suspect with a mechanism: it rewards
# closeness monotonically, unbounded, with no overlap penalty, and `backbone_steric_loss` is
# the only term opposing it. A collapse under `contactblind` names the adversarial term as the
# cause; a null under `allblind` sends the search to the trunk instead.
#
# The milestones differ because the warm arm's budget is cumulative (it inherits 10300).
RELEASE_ARMS=(
  bb_release_cold:g010300
  bb_release_warm:g020350
)

STAGE=""
DRY=0
declare -a OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)  STAGE="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; STAGE="${2:-}"; shift 2 ;;
    --set)     OVERRIDES+=("${2:-}"); shift 2 ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$STAGE" ]] || { echo "need --submit <stage>; see --help" >&2; exit 2; }

mkdir -p slurm_logs env_files

# PID in the name, not just the timestamp: two invocations in the same second otherwise
# produce ONE env file, the second overwrites the first, and both jobs read the survivor.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ENV_FILE="${REPO}/env_files/cp2lp_glob_${STAGE}_${STAMP}.env"
{
  echo "# CP->LP glob-dependence probe ${STAMP}, stage=${STAGE}"
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
  local script="$1"; shift
  local -a cmd=(sbatch)
  # A leading dep=<spec> argument becomes --dependency; everything else is a VAR=value the
  # batch script's preamble exports.
  if [[ "${1:-}" == dep=* ]]; then
    cmd+=("--dependency=${1#dep=}"); shift
  fi
  cmd+=("scripts/${script}" "$ENV_FILE" "$@")
  echo "+ ${cmd[*]}" >&2
  if [[ "$DRY" == "1" ]]; then echo "DRYRUN"; return 0; fi
  local out; out="$(scrub_env "${cmd[@]}")"
  echo "$out" >&2
  echo "${out##* }"
}

# The probes are SIBLINGS behind the mix, not a chain, so they queue concurrently. The
# report waits on all of them with afterany rather than afterok: a gated-out arm exits 0,
# but an arm that OOMs should still not block the arms that succeeded -- and the report's
# own gate distinguishes "no rows" from "never ran".
case "$STAGE" in
  mix)
    M="$(submit cp2lp_glob_mix.sbatch)"
    echo "mix  ${M}"
    ;;
  report)
    R="$(submit cp2lp_glob_report.sbatch)"
    echo "report  ${R}"
    ;;
  one)
    M="$(submit cp2lp_glob_mix.sbatch)"
    P="$(submit cp2lp_glob_probe.sbatch "dep=afterok:${M}")"
    R="$(submit cp2lp_glob_report.sbatch "dep=afterany:${P}")"
    echo "mix     ${M}"
    echo "probe   ${P}"
    echo "report  ${R}"
    ;;
  decisive|all|release)
    if [[ "$STAGE" == "decisive" ]]; then
      ARMS=(onepass_internal_clashnorm:g020300 bb_torsion:g010300)
    elif [[ "$STAGE" == "release" ]]; then
      ARMS=("${RELEASE_ARMS[@]}")
    else
      ARMS=("${ALL_ARMS[@]}")
    fi
    M="$(submit cp2lp_glob_mix.sbatch)"
    echo "mix     ${M}"
    deps=""
    for spec in "${ARMS[@]}"; do
      arm="${spec%%:*}"; tag="${spec##*:}"
      J="$(submit cp2lp_glob_probe.sbatch "dep=afterok:${M}" "ARM=${arm}" "GLOB_CKPT_TAG=${tag}")"
      echo "probe   ${arm} @ ${tag}  ${J}"
      deps="${deps}${deps:+:}${J}"
    done
    R="$(submit cp2lp_glob_report.sbatch "dep=afterany:${deps}")"
    echo "report  ${R}"
    ;;
  *)
    echo "unknown stage: $STAGE (see --help)" >&2; exit 2 ;;
esac
