#!/bin/bash
#SBATCH --job-name=lpgen_train
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_train_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_train_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# Train the RFD3 LP arm. One tranche per submission; resubmit to continue.
#
# EVERYTHING lands on scratch -- checkpoints are 2.6 GB each and home is a quota'd volume
# that also holds the repository.
#
# RESUMABLE BY RESUBMISSION. The run directory is keyed on TRAIN_RUN_NAME, not on a
# timestamp, so resubmitting the same name finds the previous checkpoints and continues.
# `ckpt_path` accepts a DIRECTORY and loads the latest checkpoint in it
# (trainers/fabric.py), which is what the auto-resume below relies on. Hitting the time
# limit therefore costs one resubmission, not the tranche.
#
# DO NOT ADD `trainer.max_epochs` HERE. Measured (jobs 83983/83985): loading a checkpoint
# does `current_epoch += 1` and then immediately evaluates
# `if current_epoch >= max_epochs: should_stop = True`, so a small max_epochs makes a
# resumed run train ZERO steps -- 0 epochs, 0 checkpoints, GPU util 0% -- and exit 0,
# which is indistinguishable from a finished tranche. The tranche is bounded by the SLURM
# time limit instead, and the stock `max_epochs: 100_000` is left alone.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "train"

if ! _lpgen_require_manifest; then
  echo "[train] GATE not satisfied -- exiting 0 so dependents are not stranded"
  exit 0
fi

ARM_TREE="${UPSTREAM_DIR}/foundry"
ARM_PY="${ENVS_DIR}/foundry/bin/python"
for p in "$ARM_TREE/models/rfd3/src/rfd3/train.py" "$ARM_PY"; do
  [[ -e "$p" ]] || { echo "[train] GATE missing: $p (run --submit stage / --submit env)" >&2; exit 0; }
done

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-lp_scratch_v1}"
RUN_DIR="${LPGEN_RUNS_DIR}/${TRAIN_RUN_NAME}"
CKPT_DIR="${RUN_DIR}/ckpt"
mkdir -p "$CKPT_DIR"

case "$RUN_DIR" in
  "$REPO"*|"$HOME"*) echo "[train] FATAL: run dir is on home: $RUN_DIR" >&2; exit 1 ;;
esac

# Auto-resume. A directory with at least one checkpoint means continue; otherwise this is a
# fresh start from RANDOM WEIGHTS -- said loudly, because "fine-tune" and "from scratch" are
# very different runs and the distinction is otherwise invisible in the log.
RESUME_ARGS=()
N_CKPT=$(find "$CKPT_DIR" -maxdepth 1 -name '*.ckpt' 2>/dev/null | wc -l)
if [[ -n "${TRAIN_CKPT_PATH:-}" ]]; then
  echo "[train] ===== INITIALIZING FROM AN EXPLICIT CHECKPOINT ====="
  echo "[train]   ${TRAIN_CKPT_PATH}"
  RESUME_ARGS=("ckpt_path=${TRAIN_CKPT_PATH}")
elif [[ "$N_CKPT" -gt 0 ]]; then
  echo "[train] ===== RESUMING: ${N_CKPT} checkpoint(s) in ${CKPT_DIR} ====="
  find "$CKPT_DIR" -maxdepth 1 -name '*.ckpt' -printf '  %f  %TY-%Tm-%Td %TH:%TM  %s bytes\n' | sort
  RESUME_ARGS=("ckpt_path=${CKPT_DIR}")
else
  echo "[train] ===== FRESH START FROM RANDOM WEIGHTS ====="
  echo "[train] This is NOT a fine-tune, and the plan asks for one ('start from the released"
  echo "[train] RFD3 weights; use model-only initialization')."
  # The previous wording claimed the released weights "are not on this cluster", which read
  # as unobtainable and was acted on as such: jobs 83990 and 84043 both trained from random
  # init. They are in fact one CLI call away -- `foundry list-available` offers rfd3 and the
  # CLI ships in the arm's own env. So say where they are, and if they are already present,
  # print the exact line that uses them.
  #
  # Deliberately NOT auto-adopting a found checkpoint: silently turning a from-scratch run
  # into a fine-tune would change what the job means without the submitter asking for it.
  FOUNDRY_RFD3_CKPT="${FOUNDRY_RFD3_CKPT:-${WEIGHTS_DIR}/foundry/rfd3_latest.ckpt}"
  if [[ -f "$FOUNDRY_RFD3_CKPT" ]]; then
    echo "[train] WARNING: released RFD3 weights ARE present and are NOT being used:"
    echo "[train]   ${FOUNDRY_RFD3_CKPT}  ($(stat -c %s "$FOUNDRY_RFD3_CKPT") bytes)"
    echo "[train] To fine-tune instead, resubmit with:"
    echo "[train]   --submit train --set TRAIN_CKPT_PATH=${FOUNDRY_RFD3_CKPT}"
  else
    echo "[train] Download them first (one CPU job, no GPU needed):"
    echo "[train]   ./scripts/submit_lpgen.sh --submit weights --set STAGE_ARMS=foundry"
    echo "[train] then resubmit with --set TRAIN_CKPT_PATH=<the path it prints>"
  fi
  # A from-scratch run at AF3's pretraining LR diverged in job 83990: the sequence head
  # reached 0.56 recovery, then collapsed to 0.00 with token loss 1.9 -> 44.6 while the
  # structure MSE kept improving, alongside two X_noisy_L NaN warnings. base_lr 0.0018 with
  # an effective batch of 3 examples (batch_size 1 x grad_accum_steps 3) is the suspect.
  echo "[train] NOTE: from-scratch at base_lr 0.0018 diverged (job 83990); check the"
  echo "[train] effective LR on the first logged step before attributing anything to data."
fi

export PYTHONPATH="${REPO}/src"
"$ARM_PY" -c "
import proteinfoundation.lpgen.rfd3_conditions as c, proteinfoundation.lpgen.rfd3_sampling as s
assert c.__file__.startswith('${REPO}/src'), c.__file__
assert c.LPBinderCondition.name == 'ppi'
print('[train] our modules resolve:', c.__file__)
"

# The first OOM on a 24 GB card showed 4.70 GiB reserved-but-unallocated; torch's own
# message recommends this, and it costs nothing on a card with headroom.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MEMLOG="${RUN_DIR}/gpu_mem_${SLURM_JOB_ID:-local}.csv"
( while true; do
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits >> "$MEMLOG" 2>/dev/null || true
    sleep 30
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# ---------------------------------------------------------------- checkpoint retention
# foundry never prunes (trainers/fabric.py:815 has no save_top_k), so without this a
# multi-day run fills the quota on its own. Sweep once now -- which clears whatever the
# PREVIOUS job in this run name accumulated -- then on a timer, because a single job can
# outlive its own disk budget. The log goes to RUN_DIR and NOT to CKPT_DIR: resume
# resolves a directory with `sorted(iterdir())[-1]` and does not filter for *.ckpt
# (fabric.py:948), so a `prune.log` sitting in there would be loaded AS the checkpoint.
PRUNE_LOG="${RUN_DIR}/ckpt_prune_${SLURM_JOB_ID:-local}.log"
prune_ckpts() {
  "$PY" -I "${REPO}/script_utils/lpgen_prune_ckpts.py" \
    --ckpt-dir "$CKPT_DIR" \
    --keep-newest "${CKPT_KEEP_NEWEST}" \
    --keep-every "${CKPT_KEEP_EVERY}" >> "$PRUNE_LOG" 2>&1 || true
}
echo "[train] checkpoint retention: keep newest ${CKPT_KEEP_NEWEST} + every ${CKPT_KEEP_EVERY}th epoch"
echo "[train]   sweep log: ${PRUNE_LOG}"
prune_ckpts
tail -1 "$PRUNE_LOG" 2>/dev/null | sed 's/^/[train]   /'
if [[ "${CKPT_PRUNE_INTERVAL_SEC}" -gt 0 ]]; then
  ( while true; do sleep "${CKPT_PRUNE_INTERVAL_SEC}"; prune_ckpts; done ) &
  PRUNER=$!
  trap 'kill $SAMPLER $PRUNER 2>/dev/null || true' EXIT
fi

cd "$ARM_TREE"
echo "[train] experiment=${TRAIN_EXPERIMENT} run_dir=${RUN_DIR}"
set +e
"$ARM_PY" models/rfd3/src/rfd3/train.py \
  "experiment=${TRAIN_EXPERIMENT}" \
  "hydra.searchpath=[pkg://rfd3.configs,pkg://configs,file://${REPO}/configs/rfd3_lp]" \
  "paths.output_dir=${RUN_DIR}" \
  "${RESUME_ARGS[@]}" \
  ${EXTRA_TRAIN_ARGS:-} 2>&1 | tee -a "${RUN_DIR}/train_${SLURM_JOB_ID:-local}.log"
RC=${PIPESTATUS[0]}
set -e
# ${PRUNER:-} because the in-job sweep is optional and this runs under `set -u`.
kill $SAMPLER ${PRUNER:-} 2>/dev/null || true

echo "=================================================================="
echo "[train] train.py exit code: ${RC}"
if [[ -s "$MEMLOG" ]]; then
  "$PY" - "$MEMLOG" <<'PYEOF'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 3]
if rows:
    used = [int(r[0]) for r in rows]; total = int(rows[0][1]); util = [int(r[2]) for r in rows]
    print(f"[train] GPU memory: peak {max(used)} MiB of {total} MiB ({max(used)/total:.0%})")
    print(f"[train] GPU util  : peak {max(util)}%, mean {sum(util)/len(util):.0f}%")
    if max(util) == 0:
        print("[train] WARNING: GPU utilisation never rose above 0% -- no training happened. "
              "If a checkpoint was loaded, check trainer.max_epochs against its epoch.")
PYEOF
fi
STAGE_LABEL=train
LOGF="${RUN_DIR}/train_${SLURM_JOB_ID:-local}.log"

# WAS IT REALLY A FINE-TUNE? foundry's weight loader has `default_policy: COPY` with
# `fallback_policy: REINIT`, and its docstring is explicit: "By default, all model weights
# are re-initialized; we overwrite with the checkpoint weights where appropriate". So a
# checkpoint whose parameter names do not match the model loads *partially*, every miss is
# silently re-initialized, and the run looks like a fine-tune while being closer to
# scratch. Each miss does log one warning, so count them.
if grep -q 'Loading checkpoint from' "$LOGF" 2>/dev/null; then
  N_REINIT=$(grep -c "Falling back to policy: 'WeightLoadingPolicy.REINIT'" "$LOGF" 2>/dev/null; true)
  N_SHAPE=$(grep -c 'Shape mismatch for' "$LOGF" 2>/dev/null; true)
  N_MISSING=$(grep -c 'not found in checkpoint' "$LOGF" 2>/dev/null; true)
  echo "[${STAGE_LABEL}] weight load: ${N_REINIT} parameter(s) fell back to REINIT "
  echo "[${STAGE_LABEL}]              (${N_MISSING} absent from ckpt, ${N_SHAPE} shape mismatch)"
  if [[ "$N_REINIT" -gt 0 ]]; then
    echo "[${STAGE_LABEL}] WARNING: those parameters were NOT initialized from the checkpoint."
    echo "[${STAGE_LABEL}] A large count means this is not really a fine-tune. First few:"
    grep "Falling back to policy: 'WeightLoadingPolicy.REINIT'" "$LOGF" | head -5 | sed 's/^/    /'
  else
    echo "[${STAGE_LABEL}] all parameters came from the checkpoint -- a genuine fine-tune"
  fi
fi

# Final sweep, after the last checkpoint has settled: the timer may well have fired last
# while a checkpoint was still too new to judge, and this is the state the next job
# inherits. The newest is always kept, so this never costs the resume point.
prune_ckpts
echo "[train] checkpoints now in ${CKPT_DIR}:"
find "$CKPT_DIR" -maxdepth 1 -name '*.ckpt' -printf '  %f  %s bytes\n' | sort | tail -5
echo "[train] checkpoint disk: $(du -sh "$CKPT_DIR" 2>/dev/null | cut -f1) ($(tail -1 "$PRUNE_LOG" 2>/dev/null))"
echo "[train] resubmit the SAME --set TRAIN_RUN_NAME=${TRAIN_RUN_NAME} to continue from the latest"
exit "$RC"
