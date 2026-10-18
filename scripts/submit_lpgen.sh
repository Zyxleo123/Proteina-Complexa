#!/bin/bash
# Submitter for the LP-generator milestone. Calls sbatch and nothing else.
#
# Stages:
#   manifest   build the ONE common LP manifest every arm trains against (cpu, ~minutes)
#   export     write the per-complex layout PPFlow and POTFlow read (cpu, ~minutes)
#   stage      clone the upstream arm repos, pin commits, apply our patch set (cpu)
#   weights    download + verify the arms' RELEASED checkpoints (cpu, ~2.5 GB per arm)
#   dpbstage   stage LNR complexes for DiffPepDock, rewrite-proof layout (cpu, seconds)
#   dpbdock    DiffPepDock k=0 baseline on the staged complexes (1 gpu)
#   dpbscore   score those poses against the staged natives (cpu, reads PDBs only)
#   env        build one venv per arm (cpu, slow: torch is ~2.5 GB per env)
#   preflight  RFD3 LP arm: a few real optimizer steps on one GPU (plan step 4)
#   hsprobe    HOTSPOT GATE (cpu): measure the realised hotspot dosage on our LP data
#              through the real transform stack, before spending GPU time on the arm
#   hscontigs  build the two hotspot validation panels (cpu, seconds)
#   train      RFD3 LP arm: one training tranche; resubmit the same TRAIN_RUN_NAME to resume
#   reval      re-validate ALREADY-SAVED checkpoints against the corrected panel (1 gpu
#              job per checkpoint, siblings) + one cpu job that assembles the curve
#   all        manifest, then export behind it
#
# Arms are added here as they get adapters, and each one is a sibling of the others so a
# missing upstream dependency in one arm does not block the rest. Current priority, which
# is set by CONDITIONING GRANULARITY rather than by the papers (see README_LPGEN.md):
#   foundry/RFD3    exact per-position sequence control; training now
#   diffpepbuilder  per-residue sequence channel + fully diffused peptide; priority arm
#   unimomo         calibration only -- no way to supply residue identities at all
#   ppflow/pepglad  off the critical path; potflow parked
set -euo pipefail

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# cpu partition: the manifest is pure file I/O. Asking for a GPU would queue it behind
# real training work for no benefit.
CPU_PARTITION="${CPU_PARTITION:-cpu}"
MANIFEST_TIME="${MANIFEST_TIME:-01:00:00}"
EXPORT_TIME="${EXPORT_TIME:-02:00:00}"
STAGE_TIME="${STAGE_TIME:-00:40:00}"
# 2.5 GB over two Zenodo records; generous because Zenodo throughput varies a lot.
WEIGHTS_TIME="${WEIGHTS_TIME:-02:00:00}"
DPBSTAGE_TIME="${DPBSTAGE_TIME:-00:20:00}"
DPBDOCK_TIME="${DPBDOCK_TIME:-04:00:00}"
DPBSCORE_TIME="${DPBSCORE_TIME:-00:40:00}"
ENV_TIME="${ENV_TIME:-02:00:00}"
# GPU for the preflight. a6000 by default: bf16 is EMULATED below Ampere, so a timing taken
# on a 2080 Ti or V100 cannot be extrapolated to a tranche. gpu14 is excluded because CUDA
# init fails there for every torch build while its siblings pass -- and it still shows as
# AVAILABLE in myfree.
GPU_PARTITION="${GPU_PARTITION:-general}"
# gpu28 is the only a5000 (24 GB) in `general`, and MEASURED it cannot hold the LP arm at
# crop_size 384 / diffusion_batch_size_train 16: peak 89% over 12 batches, then OOM at 48.
# Excluding it leaves the a6000s (48 GB). gpu14 is excluded because CUDA init fails there
# for every torch build while its siblings pass -- and myfree still lists it AVAILABLE.
GPU_EXCLUDE="${GPU_EXCLUDE:-gpu14,gpu28}"
PREFLIGHT_TIME="${PREFLIGHT_TIME:-01:00:00}"
# The hotspot gate reads examples through the real transform stack on CPU. 200 examples x
# 2 passes; generous because the spatial crop is the slow part and runs on every example.
HSPROBE_TIME="${HSPROBE_TIME:-01:30:00}"
# Reads 32 PDBs and writes two JSON files.
HSCONTIGS_TIME="${HSCONTIGS_TIME:-00:20:00}"
# One tranche per submission. The plan's cap is 24 GPU-hours per arm; the run is bounded by
# THIS, never by a small trainer.max_epochs (which makes a resumed run a silent no-op).
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
# One panel rollout per checkpoint. In-loop validation of the 64 specs costs roughly a fifth
# of a 2.4 h epoch, so 2 h is generous; the scorer only reads saved structures.
REVAL_TIME="${REVAL_TIME:-02:00:00}"
# Overridable because the binding constraint is often FREE CPUS on the one available GPU
# node, not the GPU itself: gpu28 sat AVAILABLE with 3/4 GPUs free but only 6 free CPUs,
# so an 8-CPU request queued behind nothing in particular. Validation is a GPU-bound
# rollout over 64 specs; it does not need many loader workers.
REVAL_CPUS="${REVAL_CPUS:-4}"
REVAL_SCORE_TIME="${REVAL_SCORE_TIME:-01:00:00}"

STAGE=""
RUN_ID=""
SMOKE=0
DRYRUN=0
EXTRA_ENV=()

usage() {
  sed -n '2,12p' "$0"
  cat <<'EOF'

  --submit <stage>   manifest | export | stage | weights | dpbstage | dpbdock | dpbscore | env | preflight | hsprobe | hscontigs | train | reval | all
  --run-id <id>      override the generated run id
  --set K=V          set any preamble knob (repeatable), e.g. --set PEPBENCH_FRAC=0.8
  --smoke            tiny run: 2000 rows, panel of 8 (writes to a _smoke manifest dir)
  --dry-run          print the sbatch lines without submitting
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)   STAGE="$2";          shift 2 ;;
    --run-id)   RUN_ID="$2";         shift 2 ;;
    --set)      EXTRA_ENV+=("$2");   shift 2 ;;
    --smoke)    SMOKE=1;             shift ;;
    --dry-run)  DRYRUN=1;            shift ;;
    -h|--help)  usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 >&2 ;;
  esac
done
[[ -n "$STAGE" ]] || { echo "FATAL: --submit <stage> is required" >&2; usage 1 >&2; }
case "$STAGE" in
  manifest|export|stage|weights|dpbstage|dpbdock|dpbscore|env|preflight|hsprobe|hscontigs|train|reval|all) ;;
  *) echo "FATAL: unknown stage '$STAGE'" >&2; usage 1 >&2 ;;
esac

# Submitting from inside an allocation leaks SLURM_* into the child sbatch, which then
# inherits a job id that is not its own.
for v in $(env | sed -n 's/^\(SLURM_[A-Z_]*\)=.*/\1/p'); do unset "$v" || true; done

[[ -n "$RUN_ID" ]] || RUN_ID="lpgen_$( [[ $SMOKE -eq 1 ]] && echo smoke_ )$(date +%Y%m%d_%H%M%S)"
# Scratch, never home: home is quota'd and holds the repo. See _lpgen_preamble.sh.
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
LPGEN_ROOT="${LPGEN_ROOT:-${ZFS}/lpgen}"
ENV_DIR="${LPGEN_ENV_DIR:-${LPGEN_ROOT}/env}"
LOG_DIR="${LPGEN_LOG_DIR:-${LPGEN_ROOT}/logs}"
mkdir -p "$ENV_DIR" "$LOG_DIR"

# ONE env file per submission, passed as a POSITIONAL argument. Arguments travel through
# the job record untouched; `sbatch --export` does not -- any explicit export list sets
# SLURM_GET_USER_ENV=1, slurmd fails to rebuild the login environment on the compute node,
# and the job is requeued and HELD.
ENV_FILE="${ENV_DIR}/${RUN_ID}.env"
{
  echo "# written by submit_lpgen.sh at $(date -Is)"
  echo "REPO=${REPO}"
  echo "RUN_ID=${RUN_ID}"
  if [[ $SMOKE -eq 1 ]]; then
    # A smoke run must not overwrite the real manifest that arms resolve by path.
    echo "MANIFEST_DIR=/zfsauton/scratch/yixiz/LPData/preprocessed/metadata_lpgen_smoke"
    echo "MANIFEST_LIMIT=2000"
    echo "PANEL_SIZE=8"
    echo "NATIVE_DIR=/zfsauton/scratch/yixiz/LPData/preprocessed/lpgen_native_smoke"
    echo "EXPORT_LIMIT=200"
  fi
  # Quote the VALUE side. The env file is sourced by the job, so a multi-word value
  # written bare (e.g. EXTRA_PREFLIGHT_ARGS=a=1 b=2) makes the shell treat the second
  # token as a command: `b=2: command not found`, and the job dies before any work.
  for kv in "${EXTRA_ENV[@]}"; do
    printf '%s=%q\n' "${kv%%=*}" "${kv#*=}"
  done
} > "$ENV_FILE"
echo "env file: $ENV_FILE"

submit() {  # submit <description> <sbatch args...>
  local desc="$1"; shift
  # Override the scripts' static #SBATCH --output/--error, which are repo-relative and so
  # would write job logs onto home.
  set -- --output="${LOG_DIR}/${desc}_%j.out" --error="${LOG_DIR}/${desc}_%j.out" "$@"
  if [[ $DRYRUN -eq 1 ]]; then
    { printf 'DRY-RUN %-10s sbatch' "$desc"; printf ' %q' "$@"; printf '\n'; } >&2
    echo "999999"
    return
  fi
  local out; out="$(sbatch "$@")"
  echo "$desc: $out" >&2
  awk '{print $NF}' <<<"$out"
}

cd "$REPO"

MANIFEST_JID=""

if [[ "$STAGE" == "manifest" || "$STAGE" == "all" ]]; then
  MANIFEST_JID="$(submit manifest \
    --partition="$CPU_PARTITION" --time="$MANIFEST_TIME" \
    scripts/lpgen_manifest.sbatch "$ENV_FILE")"
fi

if [[ "$STAGE" == "export" || "$STAGE" == "all" ]]; then
  DEP=(); [[ -n "$MANIFEST_JID" ]] && DEP=(--dependency=afterok:"$MANIFEST_JID")
  submit export \
    --partition="$CPU_PARTITION" --time="$EXPORT_TIME" "${DEP[@]}" \
    scripts/lpgen_export.sbatch "$ENV_FILE" >/dev/null
fi

# stage and env are siblings of the data stages, not children: the upstream trees do not
# depend on the manifest, and an arm whose clone or install fails must not strand the
# data pipeline (or the other arms).
STAGE_JID=""
if [[ "$STAGE" == "stage" ]]; then
  STAGE_JID="$(submit stage \
    --partition="$CPU_PARTITION" --time="$STAGE_TIME" \
    scripts/lpgen_stage.sbatch "$ENV_FILE")"
fi

# weights is a sibling of stage, not a child: the checkpoints live outside the trees, so
# downloading them neither needs a clone nor is invalidated by a re-stage.
if [[ "$STAGE" == "weights" ]]; then
  submit weights \
    --partition="$CPU_PARTITION" --time="$WEIGHTS_TIME" \
    scripts/lpgen_weights.sbatch "$ENV_FILE" >/dev/null
fi

# dpbstage only reads LNR and writes a fresh staging dir, so it depends on nothing.
if [[ "$STAGE" == "dpbstage" ]]; then
  submit dpbstage \
    --partition="$CPU_PARTITION" --time="$DPBSTAGE_TIME" \
    scripts/lpgen_dpb_stage.sbatch "$ENV_FILE" >/dev/null
fi

if [[ "$STAGE" == "env" ]]; then
  DEP=(); [[ -n "$STAGE_JID" ]] && DEP=(--dependency=afterok:"$STAGE_JID")
  submit env \
    --partition="$CPU_PARTITION" --time="$ENV_TIME" "${DEP[@]}" \
    scripts/lpgen_env.sbatch "$ENV_FILE" >/dev/null
fi

if [[ "$STAGE" == "dpbdock" ]]; then
  submit dpbdock \
    --partition="$GPU_PARTITION" --time="$DPBDOCK_TIME" --exclude="$GPU_EXCLUDE" \
    scripts/lpgen_dpb_dock.sbatch "$ENV_FILE" >/dev/null
fi

# dpbscore reads saved PDBs, so it belongs on the cpu partition, not behind GPU work.
if [[ "$STAGE" == "dpbscore" ]]; then
  submit dpbscore \
    --partition="$CPU_PARTITION" --time="$DPBSCORE_TIME" \
    scripts/lpgen_dpb_score.sbatch "$ENV_FILE" >/dev/null
fi

if [[ "$STAGE" == "preflight" ]]; then
  submit preflight \
    --partition="$GPU_PARTITION" --time="$PREFLIGHT_TIME" --exclude="$GPU_EXCLUDE" \
    scripts/lpgen_rfd3_preflight.sbatch "$ENV_FILE" >/dev/null
fi

if [[ "$STAGE" == "train" ]]; then
  submit train \
    --partition="$GPU_PARTITION" --time="$TRAIN_TIME" --exclude="$GPU_EXCLUDE" \
    scripts/lpgen_rfd3_train.sbatch "$ENV_FILE" >/dev/null
fi

# The hotspot GATE. CPU, and deliberately not a dependency of `train`: it is a measurement
# whose verdict a human reads, not a pass/fail the pipeline can act on. A low live-hotspot
# rate is information, not an error -- so making the fine-tune wait on it with `afterok`
# would strand the arm on a job that was never meant to block it.
if [[ "$STAGE" == "hsprobe" ]]; then
  submit hsprobe \
    --partition="$CPU_PARTITION" --time="$HSPROBE_TIME" \
    scripts/lpgen_hotspot_probe.sbatch "$ENV_FILE" >/dev/null
fi

if [[ "$STAGE" == "hscontigs" ]]; then
  submit hscontigs \
    --partition="$CPU_PARTITION" --time="$HSCONTIGS_TIME" \
    scripts/lpgen_hotspot_contigs.sbatch "$ENV_FILE" >/dev/null
fi

if [[ "$STAGE" == "reval" ]]; then
  # The checkpoints to re-score. Resolved HERE rather than in the job so the fan-out is
  # visible in the submission and each checkpoint gets its own job, its own output tree and
  # its own log -- a shared output tree would mean concurrent appends to one metrics.csv.
  CKPT_DIR="${REVAL_CKPT_DIR:-${ZFS}/lpgen/runs/${REVAL_RUN_NAME:-lp_finetune_v1}/ckpt}"
  CONTIGS="${REVAL_CONTIGS:-${ZFS}/LPData/preprocessed/metadata_lpgen/lpgen_val_contigs_pocket.json}"
  if [[ ! -f "$CONTIGS" ]]; then
    echo "FATAL: corrected contigs not found: $CONTIGS" >&2
    echo "  build them first:" >&2
    echo "  python script_utils/lpgen_build_val_contigs.py \\" >&2
    echo "      --panel \$ZFS/LPData/preprocessed/metadata_lpgen/lpgen_val_panel.parquet \\" >&2
    echo "      --out $CONTIGS --ori-mode pocket --ori-push 6.0" >&2
    exit 1
  fi
  shopt -s nullglob
  CKPTS=("$CKPT_DIR"/*.ckpt)
  shopt -u nullglob
  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "FATAL: no checkpoints in $CKPT_DIR" >&2; exit 1
  fi
  # Newest first when only some can run: the latest checkpoint is the one that decides
  # whether the arm is improving.
  if [[ -n "${REVAL_LAST_N:-}" ]]; then
    mapfile -t CKPTS < <(printf '%s\n' "${CKPTS[@]}" | sort -r | head -n "$REVAL_LAST_N")
  fi
  # REVAL_EPOCHS="572,578,582,586" re-scores exactly those epochs. A re-eval costs about a
  # GPU-hour per checkpoint, so a few SPACED points answer "is anything improving" for a
  # fraction of a full sweep; fill the gaps in only if something moves. Named epochs are
  # matched exactly and a miss is fatal, so a typo cannot silently shrink the curve.
  if [[ -n "${REVAL_EPOCHS:-}" ]]; then
    SEL=()
    IFS=',' read -r -a WANT <<< "$REVAL_EPOCHS"
    for w in "${WANT[@]}"; do
      w="${w// /}"
      hit=""
      for ck in "${CKPTS[@]}"; do
        e="$(sed -n 's/^epoch-0*\([0-9][0-9]*\)$/\1/p' <<<"$(basename "$ck" .ckpt)")"
        if [[ "$e" == "$w" ]]; then hit="$ck"; break; fi
      done
      if [[ -z "$hit" ]]; then
        echo "FATAL: REVAL_EPOCHS asked for epoch ${w}, which is not in ${CKPT_DIR}" >&2
        echo "  available: $(for c in "${CKPTS[@]}"; do basename "$c" .ckpt; done | tr '\n' ' ')" >&2
        exit 1
      fi
      SEL+=("$hit")
    done
    CKPTS=("${SEL[@]}")
  fi
  # Adopt jobs submitted earlier (e.g. a smoke) so ONE scorer waits on the whole set rather
  # than running early against a partial curve.
  PRIOR_JIDS=()
  if [[ -n "${REVAL_WAIT_FOR:-}" ]]; then
    IFS=',' read -r -a PRIOR_JIDS <<< "$REVAL_WAIT_FOR"
    echo "also waiting on already-running reval job(s): ${PRIOR_JIDS[*]}"
  fi
  echo "re-evaluating ${#CKPTS[@]} checkpoint(s) against $(basename "$CONTIGS")"
  REVAL_JIDS=()
  for ck in "${CKPTS[@]}"; do
    tag="$(basename "$ck" .ckpt)"
    jid="$(submit "reval_${tag}" \
      --partition="$GPU_PARTITION" --time="$REVAL_TIME" --exclude="$GPU_EXCLUDE" \
      --cpus-per-task="$REVAL_CPUS" \
      scripts/lpgen_rfd3_reval.sbatch "$ENV_FILE" "REVAL_CKPT=${ck}" "REVAL_CONTIGS=${CONTIGS}")"
    REVAL_JIDS+=("$jid")
  done
  # afterANY, not afterok: a single checkpoint that OOMs or hits the time limit must not
  # strand the scorer, which would leave every other checkpoint unscored and unexplained.
  ALL_JIDS=("${REVAL_JIDS[@]}" ${PRIOR_JIDS[@]+"${PRIOR_JIDS[@]}"})
  DEP="--dependency=afterany:$(IFS=:; echo "${ALL_JIDS[*]}")"
  submit reval_score \
    --partition="$CPU_PARTITION" --time="$REVAL_SCORE_TIME" "$DEP" \
    scripts/lpgen_reval_score.sbatch "$ENV_FILE" "REVAL_CONTIGS=${CONTIGS}" >/dev/null
fi

echo "submitted stage: $STAGE"
echo "watch: squeue -u \$USER -o '%.10i %.14j %.8T %.10M %R'"
