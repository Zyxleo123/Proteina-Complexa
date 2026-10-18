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
#   release     GPU  the PRESERVATION-RELEASE pair as siblings: the torsion arm with
#                    `cp2lp.losses.preserve_weight: 0.0`, warm-started from the trained
#                    torsion generator and cold-started from the fork. Cold-vs-torsion is
#                    single-variable in the loss; warm-vs-cold is single-variable in the
#                    init. See configs/example/training_cp2lp_bb_release.yaml.
#   shape       GPU  the STERIC SHAPE arm, alone: bb_release_cold with
#                    `cp2lp.losses.steric_depth_gain: 2.0` and nothing else changed, so the
#                    already-finished cold arm is its control. Makes DEEP overlap
#                    disproportionately expensive, because the weight (1.0, the largest),
#                    the threshold (0.20 vs real 0.262) and "steric is drowned out"
#                    (gradnorm 0.78 of 2.42) are all eliminated by measurement -- and the
#                    arms still ended INSIDE a covalent bond length. Read
#                    `steric_min_inter_nm` rising toward 0.262 while `terminal_gap_nm`
#                    HOLDS near 2.14; one without the other is a null.
#                    See configs/example/training_cp2lp_bb_shape.yaml.
#   retain      GPU  the RETENTION LADDER: preserve_weight 0.1/0.25/0.5 as three siblings,
#                    each warm-started from the release-COLD endpoint (its measured state is
#                    the preserve=0 anchor). Penalises interface EDIT, not openness, because
#                    the gap/retention tension is an edit-size confound: partial
#                    corr(gap,retention|edit) = +0.005 vs marginal -0.641. A win is a rung at
#                    edit ~0.3 nm with terminal_gap HOLDING in the real IQR (1.575-2.300).
#                    3 rungs not 4: ~12.4 GB each and the quota is not queryable.
#                    See configs/example/training_cp2lp_bb_retain.yaml.
#   evaluate-release GPU+CPU  the release pair: per-arm EMA endpoint -> pack -> one report
#                    over both. The arms have DIFFERENT milestone numbers (warm inherits
#                    10300), so they share an EVAL_CKPT_TAG label and pass per-arm weights.
#                    Chain it behind the pilots with RELEASE_DEP=<id>:<id>.
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

# The PRESERVATION-RELEASE pair, a separate list so `--submit arms` keeps meaning the
# original two-arm representation comparison and nothing re-reads that experiment.
RELEASE_ARMS=(bb_release_warm bb_release_cold)

# The store root, duplicated from `_cp2lp_preamble.sh` because this submitter is a plain
# script and never sources it. `test_submitter_store_root_matches_preamble` asserts the two
# agree, so a moved store cannot leave this pointing at a stale tree.
STORE_ROOT="${STORE_ROOT:-/zfsauton/scratch/yixiz/Proteina-Complexa/training_runs/store}"

STAGE=""
DRY=0
declare -a OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)  STAGE="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; STAGE="${2:-}"; shift 2 ;;
    --set)     OVERRIDES+=("${2:-}"); shift 2 ;;
    -h|--help) sed -n '2,78p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
  # SINGLE-QUOTE THE VALUE. The preamble sources this file under `set -a`, so a bare
  # `K=a b` would export K=a and then try to RUN `b`. Unquoted worked only because every
  # override so far has been a single token (SMOKE=1); the moment one carries a space --
  # EXTRA_TRAIN_ARGS is the first -- it silently exports half the value and executes the
  # rest. Quoting is also why configuration travels as a FILE rather than as sbatch args.
  for kv in "${OVERRIDES[@]:-}"; do
    [[ -n "$kv" ]] || continue
    k="${kv%%=*}"; v="${kv#*=}"
    if [[ "$kv" != *=* ]]; then
      echo "FATAL: --set expects KEY=VALUE, got: $kv" >&2; exit 2
    fi
    printf "%s='%s'\n" "$k" "${v//\'/\'\\\'\'}"
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

# A SMOKE RUN MUST NOT RESERVE TWO DAYS. cp2lp_train.sbatch asks for `--time=2-00:00:00`,
# which is right for a real arm and ruinous for a 20-batch check: measured on job 74049, a
# 48h request on `general` was given an estimated start of +10h with Reason=Priority, because
# nothing that long backfills into the gaps the free a6000s are reserved for. A short request
# fits those gaps, which is the whole point of a smoke. `sbatch` CLI options override the
# script's own `#SBATCH` directives, so this needs no second copy of the job script.
#
# Applied automatically rather than left as a flag to remember: the failure mode is not an
# error, it is a smoke that silently sits in the queue for half a day.
SMOKE_TIME="${SMOKE_TIME:-00:30:00}"
train_sbopts() {
  local kv
  for kv in "${OVERRIDES[@]:-}"; do
    [[ "$kv" == SMOKE=1 ]] || continue
    echo "--sbopt"; echo "--time=${SMOKE_TIME}"
    return 0
  done
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

# STAGE ALIASES, resolved BEFORE the case. `case` evaluates its word once, so setting STAGE
# inside a branch -- even with a `;;&` fallthrough -- can never reach a different branch; the
# first attempt at this silently submitted nothing.
if [[ "$STAGE" == "retain-low" ]]; then
  # The low-weight ladder, at the exemption the w025 tranche favoured (k=1). A separate entry
  # point because those rungs showed 0.25 is ALREADY PAST the knee at every exemption level --
  # all three collapsed to edit ~0.06 nm with the gap pinned on the 0.45 open-hinge floor -- so
  # the informative range lies below it. With the running w025_x1 these four rungs
  # (0.02 / 0.05 / 0.1 / 0.25) form one continuous weight scan at fixed k=1.
  RETAIN_ARMS="bb_retain_w002_x1 bb_retain_w005_x1 bb_retain_w010_x1"
  STAGE=retain
fi

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

  bb_cartesian|bb_torsion|bb_release_warm|bb_release_cold|bb_shape|bb_shape_warm|bb_retain_w0[0-9][0-9]|bb_retain_w0[0-9][0-9]_x[0-9])
    echo "$STAGE: $(submit scripts/cp2lp_train.sbatch "ARM=${STAGE}")" ;;

  shape)
    # THE STERIC SHAPE ARM, alone. A single-variable child of bb_release_cold -- it differs in
    # `run_name` and `steric_depth_gain` and in nothing else -- so it needs no sibling to be
    # interpretable: `cp2lp_bb_release_cold` IS its control and is already trained.
    #
    # It starts from the same pinned fork as the cold arm, so unlike `release` there is no
    # other arm's milestone to check for first.
    # THE SHAPE PAIR, as siblings so they queue concurrently rather than serialising.
    #
    #   cold  fork start, 10050 updates. Differs from the FINISHED cp2lp_bb_release_cold in
    #         exactly 2 resolved keys (run_name + steric_depth_gain), so that arm is its
    #         control and no new baseline has to be trained.
    #   warm  starts from release_cold's g010300 -- the state that MEASURED 0.0705 nm of
    #         interpenetration -- so the sign of the effect is readable in ~1-2h instead of
    #         ~18h. Read it as a before/after within its own run, NOT against cold's endpoint.
    #
    # WHY THE WARM CHECKPOINT IS CHECKED HERE AND NOT IN THE JOB, same as `release`: without it
    # the job dies with a FileNotFoundError from train.py after a queue wait that can be hours,
    # and release_cold's milestone is the only copy of that result.
    shape_warm_ckpt="${SHAPE_WARM_CKPT:-${STORE_ROOT}/cp2lp_bb_release_cold/checkpoints/g010300.ckpt}"
    if [[ ! -s "$shape_warm_ckpt" ]]; then
      echo "FATAL: shape warm-start checkpoint missing or empty: $shape_warm_ckpt" >&2
      echo "  The warm shape arm starts from the release-COLD endpoint. Without it, submit" >&2
      echo "  only the cold arm: --submit bb_shape" >&2
      exit 1
    fi
    # The config states this path too; a disagreement would be invisible in the job log,
    # because `++pretrain_ckpt_path=${INIT_CKPT}` is a Hydra override and BEATS the yaml.
    cfg_ckpt="$(grep -m1 '^pretrain_ckpt_path:' configs/example/training_cp2lp_bb_shape_warm.yaml | awk '{print $2}')"
    if [[ "$cfg_ckpt" != "$shape_warm_ckpt" ]]; then
      echo "FATAL: warm-start path disagrees between the preamble and the config." >&2
      echo "  preamble/SHAPE_WARM_CKPT : $shape_warm_ckpt" >&2
      echo "  config pretrain_ckpt_path: $cfg_ckpt" >&2
      exit 1
    fi
    echo "shape warm-start checkpoint: $shape_warm_ckpt ($(du -h "$shape_warm_ckpt" | cut -f1))"
    echo "shape pair as siblings (cold's control is the finished cp2lp_bb_release_cold):"
    mapfile -t sbo < <(train_sbopts)
    for arm in bb_shape bb_shape_warm; do
      printf '  %-18s %s\n' "$arm" \
        "$(submit scripts/cp2lp_train.sbatch "${sbo[@]}" "ARM=${arm}")"
    done ;;

  release)
    # THE PRESERVATION-RELEASE PAIR, as siblings. `preserve_weight: 0.0` on the torsion arm;
    # warm from the trained torsion generator, cold from the fork.
    #
    # WHY THE WARM ARM NEEDS ITS CHECKPOINT CHECKED HERE AND NOT IN THE JOB. The warm arm is
    # the only thing in this repo that starts from another arm's milestone, and if that file
    # is missing the job dies ~20 minutes into a queue wait with a FileNotFoundError from
    # `train.py`. Both milestones are also the only copy of the torsion result.
    warm_ckpt="${RELEASE_WARM_CKPT:-${STORE_ROOT}/cp2lp_bb_torsion/checkpoints/g010300.ckpt}"
    if [[ ! -s "$warm_ckpt" ]]; then
      echo "FATAL: warm-start checkpoint missing or empty: $warm_ckpt" >&2
      echo "  The warm arm starts from the TRAINED torsion generator. Without it, submit" >&2
      echo "  only the cold arm: --submit bb_release_cold" >&2
      exit 1
    fi
    echo "warm-start checkpoint: $warm_ckpt ($(du -h "$warm_ckpt" | cut -f1))"
    echo "release pair as siblings (they must queue concurrently, not chain):"
    for arm in "${RELEASE_ARMS[@]}"; do
      printf '  %-18s %s\n' "$arm" "$(submit scripts/cp2lp_train.sbatch "ARM=${arm}")"
    done ;;

  retain)
    # THE EXEMPTION TRANCHE: one k=0 control plus two terminal-exempt rungs at the SAME weight,
    # so the exemption is the single variable. The plain 0.1/0.5 rungs were cancelled once it
    # was clear a k=0 ladder charges for the terminal travel that opens the ring -- see
    # `source_interface_weights`' exemption block.
    RETAIN_ARMS=(${RETAIN_ARMS:-bb_retain_w025 bb_retain_w025_x1 bb_retain_w025_x2})

    # RIGHT-SIZE THE WALLTIME. cp2lp_train.sbatch asks for `--time=2-00:00:00`, which is correct
    # for a 10050-update arm and ruinous for a 3000-update rung: MEASURED on the shape arms at
    # 501 g-updates/hour, a rung needs ~6.0 h, and a 48 h request cannot backfill into the
    # windows the free a6000s are held for. The first ladder submission sat PENDING with
    # estimated starts ~18 h out (jobs 74276/74279/74280, StartTime 2026-10-07T14:14-16:06)
    # purely because of this -- the same defect the smoke path already fixes.
    #
    # A TIGHT WALLTIME IS SAFE HERE BECAUSE THE RUN IS RESUMABLE. `training.resume_from_last` is
    # true and `last_ckpt_every_n_steps` is 250, so hitting the limit means RESUBMITTING the
    # same arm and continuing from `last.ckpt`, not restarting. That is the whole reason a
    # ~33% margin is enough rather than a 8x one.
    RETAIN_TIME="${RETAIN_TIME:-08:00:00}"
    mapfile -t sbo < <(train_sbopts)
    sbo+=(--sbopt "--time=${RETAIN_TIME}")
    # THE RETENTION LADDER: preserve_weight 0.1 / 0.25 / 0.5, three siblings so they queue
    # concurrently. Every rung warm-starts from the release-COLD endpoint, whose measured state
    # IS the preserve=0 anchor of the ladder (edit 0.650 nm, retention 0.489, gap 1.957 nm).
    #
    # WHY preserve_weight AND NOT AN OPENNESS TERM. The gap/retention tension is an edit-size
    # confound: partial corr(gap, retention | edit) = +0.005 against corr(gap, retention) =
    # -0.641, with corr(gap, edit) = +0.918. Retention is spent on edit MAGNITUDE, and
    # preserve_weight is the only term that charges for interface displacement.
    #
    # THREE RUNGS AND NOT FOUR, deliberately: each costs ~12.4 GB (one milestone 3.6+2.9 GB
    # plus the rolling last/last-EMA pair 2.9+2.9 GB), and both backbone arms were once killed
    # mid-checkpoint-write by `Errno 122 Disk quota exceeded` at ~39 GB. The quota is not
    # queryable here -- `quota` and `lfs` are both absent -- so the footprint is bounded by
    # running fewer rungs rather than by checking. Prune each rung's last*.ckpt once its
    # milestone is written; they are redundant then.
    retain_ckpt="${SHAPE_WARM_CKPT:-${STORE_ROOT}/cp2lp_bb_release_cold/checkpoints/g010300.ckpt}"
    if [[ ! -s "$retain_ckpt" ]]; then
      echo "FATAL: ladder warm-start checkpoint missing or empty: $retain_ckpt" >&2
      echo "  Every rung starts from the release-COLD endpoint; without it there is no ladder." >&2
      exit 1
    fi
    echo "ladder warm-start: $retain_ckpt ($(du -h "$retain_ckpt" | cut -f1))"
    echo "retention ladder as siblings (preserve=0 anchor is the finished cp2lp_bb_release_cold):"
    for arm in "${RETAIN_ARMS[@]}"; do
      printf '  %-18s %s\n' "$arm" \
        "$(submit scripts/cp2lp_train.sbatch "${sbo[@]}" "ARM=${arm}")"
    done ;;

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

  evaluate-release)
    # Evaluation for the PRESERVATION-RELEASE pair. Separate from `evaluate` because the two
    # tranches cannot share one invocation: `evaluate` iterates `ARMS` and relies on every arm
    # having the SAME milestone number, and the release arms do not -- the warm arm inherits
    # 10300 G-updates from the torsion checkpoint, so its endpoint is 20350 against the cold
    # arm's 10300.
    #
    # HOW THAT IS RECONCILED. `OUT` inside the eval/pack/report scripts is keyed by
    # `EVAL_CKPT_TAG` ALONE and is deliberately shared across arms, because the report takes
    # one `--packed-dir` and groups the rows `by_arm`. So both arms are given the same tag --
    # a LABEL for the shared comparison directory -- plus a per-arm `BB_EVAL_CKPT` naming its
    # own weights. That uses the escape hatch the eval script already has, rather than adding
    # a mechanism.
    #
    # THE EMA SIDECARS, not the raw milestones: EMA weights are the evaluation weights here
    # (the stage default is `last-EMA`), and the raw copy is for resuming optimisation.
    #
    # DEPENDENCIES, PER ARM, which is the point. Pass
    #   RELEASE_DEPS="bb_release_cold=72085,bb_release_warm=72166"
    # and each arm's evaluation waits ONLY on its own training job. A single shared
    # `--dependency=afterok:72085:72166` would mean a failure in either pilot leaves BOTH
    # evaluations on DependencyNeverSatisfied -- so one arm dying would silently cost the
    # other arm's result, which is already on disk and perfectly evaluable. `RELEASE_DEP`
    # still applies one dependency list to every arm, for the case where that is what is
    # wanted.
    #
    # The report waits on whichever pack jobs were submitted, so if one arm's chain never
    # runs the report is the only casualty, and `--submit report` re-renders it later from
    # the rows that did land. Each eval job also GATES on its checkpoint existing and exits 0
    # with an explanation, so a chain that fires early says why.
    TAG="${EVAL_CKPT_TAG:-release_end}"
    declare -A END=( [bb_release_cold]=g010300-EMA [bb_release_warm]=g020350-EMA )
    declare -A ADEP=()
    IFS=',' read -r -a _pairs <<< "${RELEASE_DEPS:-}"
    for kv in "${_pairs[@]:-}"; do
      [[ "$kv" == *=* ]] && ADEP["${kv%%=*}"]="${kv#*=}"
    done
    echo "evaluate-release into the shared dir EVAL_CKPT_TAG=${TAG}:"
    declare -a rpacks=()
    for arm in "${RELEASE_ARMS[@]}"; do
      declare -a rdep=()
      d="${ADEP[$arm]:-${RELEASE_DEP:-}}"
      [[ -n "$d" ]] && rdep=(--dep "$d")
      # `cp2lp_` + THE FULL ARM NAME is the run_name -- `bb_release_cold` -> run
      # `cp2lp_bb_release_cold`, matching the preamble's dispatch table. The `bb_` prefix is
      # NOT stripped; stripping it yields `cp2lp_release_cold`, a directory that does not
      # exist, and the eval job would then gate on a missing checkpoint and exit 0 having
      # evaluated nothing. `test_submitter_release_eval_paths` asserts this against the
      # preamble so it cannot drift back.
      ck="${STORE_ROOT}/cp2lp_${arm}/checkpoints/${END[$arm]}.ckpt"
      g="$(submit scripts/cp2lp_bb_eval_gpu.sbatch "${rdep[@]}" \
            "ARM=${arm}" "EVAL_CKPT_TAG=${TAG}" "BB_EVAL_CKPT=${ck}")"
      k="$(submit scripts/cp2lp_bb_pack_eval.sbatch --dep "$g" \
            "ARM=${arm}" "EVAL_CKPT_TAG=${TAG}")"
      rpacks+=("$k")
      printf '  %-18s gpu=%s pack=%s\n' "$arm" "$g" "$k"
      printf '  %-18s ckpt=%s\n' "" "$ck"
    done
    dep=()
    for k in "${rpacks[@]}"; do dep+=(--dep "$k"); done
    echo "  report             $(submit scripts/cp2lp_bb_report.sbatch "${dep[@]}" "EVAL_CKPT_TAG=${TAG}")" ;;

  evaluate-retain)
    # THE VERDICT STAGE for the retention tranche. Everything measured in-run so far is a PROXY
    # (`preserve_ca_rmsd_nm`, `terminal_gap_nm`); `contact_retention` only exists after all-atom
    # packing, which is what this chain does.
    #
    # The shape arms ride along because they share the question "what did this cost the
    # interface" and the report groups rows `by_arm` from ONE shared --packed-dir, so adding
    # them here is free and makes the comparison one table instead of two.
    #
    # EMA SIDECARS, not the raw milestones: EMA weights are the evaluation weights.
    #
    # !! THE k=0 MILESTONE IS NOT REPRESENTATIVE OF ITS OWN RUN. MEASURED: `cp2lp_bb_retain_w025`
    # surged late -- edit median 0.147 over its last 3000 batches but 1.187 over its last 100,
    # with 282 of the last 400 above 1.0 -- and g013300 was written inside that surge. Its four
    # siblings are stable to within 10% over the same windows. So read that one row as "a
    # destabilised k=0 checkpoint", NOT as "k=0 at equilibrium", and do not draw the exemption
    # trend through it.
    TAG="${EVAL_CKPT_TAG:-retain_end}"
    declare -A REND=(
      [bb_retain_w025]=g013300-EMA      [bb_retain_w025_x1]=g013300-EMA
      [bb_retain_w025_x2]=g013300-EMA   [bb_retain_w005_x1]=g013300-EMA
      [bb_retain_w010_x1]=g013300-EMA   [bb_retain_w002_x1]=g013300-EMA
      [bb_shape]=g010300-EMA            [bb_shape_warm]=g020350-EMA
    )
    RETAIN_EVAL_ARMS=(${RETAIN_EVAL_ARMS:-bb_retain_w005_x1 bb_retain_w010_x1 bb_retain_w025_x1 bb_retain_w025 bb_retain_w025_x2 bb_shape bb_shape_warm})
    echo "evaluate-retain into the shared dir EVAL_CKPT_TAG=${TAG}:"
    declare -a tpacks=()
    for arm in "${RETAIN_EVAL_ARMS[@]}"; do
      ck="${STORE_ROOT}/cp2lp_${arm}/checkpoints/${REND[$arm]}.ckpt"
      # GATE IN THE SUBMITTER, not in the job: a missing EMA sidecar makes the GPU job exit 0
      # having evaluated nothing, which then releases a pack job that packs nothing and a report
      # that reports a confidently wrong denominator. Checked here so it is visible at submit.
      if [[ ! -s "$ck" ]]; then
        echo "  SKIP ${arm}: no $(basename "$ck") (did its milestone land?)" >&2
        continue
      fi
      g="$(submit scripts/cp2lp_bb_eval_gpu.sbatch "ARM=${arm}" "EVAL_CKPT_TAG=${TAG}" "BB_EVAL_CKPT=${ck}")"
      k="$(submit scripts/cp2lp_bb_pack_eval.sbatch --dep "$g" "ARM=${arm}" "EVAL_CKPT_TAG=${TAG}")"
      tpacks+=("$k")
      printf '  %-20s gpu=%s pack=%s\n' "$arm" "$g" "$k"
    done
    dep=()
    for k in "${tpacks[@]}"; do dep+=(--dep "$k"); done
    echo "  report               $(submit scripts/cp2lp_bb_report.sbatch "${dep[@]}" "EVAL_CKPT_TAG=${TAG}")" ;;

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
    echo "        evaluate | evaluate-release | report | pilot | release |" >&2
    echo "        ${RELEASE_ARMS[*]}" >&2
    exit 2 ;;
esac
