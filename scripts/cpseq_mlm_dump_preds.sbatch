#!/bin/bash
#SBATCH --job-name=cpseq_mlm_dump
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_dump_%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_dump_%x_%j.out
#
# Dump per-position predictions for a cpseq_mlm checkpoint on its FIXED eval panel. One
# forward pass over ~770 encoded examples with a 650M model: minutes, not hours.
#
# This is the only GPU step of the probe chain. Everything downstream is a CPU reduction
# over its output file, which is the point -- each new question about the model's errors
# should cost a CPU job, not another GPU pass.
#
# `general` rather than `debug`: this partition is all a6000 (48 GB), so the GPU class is
# pinned by the partition and cannot silently land on an 11 GB 2080 Ti, where a 650M model
# at these sequence lengths would die in the CUDA allocator with no Python traceback.
#
# Do NOT sbatch directly -- use scripts/submit_cpseq_mlm_probes.sh. Config travels in a
# FILE passed positionally, never via `sbatch --export` (an explicit export list sets
# SLURM_GET_USER_ENV=1 and the job is requeued and held).

set -euo pipefail          # in the body: a failed source must not run under system python and exit 0

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

ENV_FILE="${1:-}"
[ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ] || { echo "FATAL: env file not found: ${ENV_FILE:-<none>}"; exit 1; }
source "$REPO/scripts/cpseq_mlm_env.sh"
source "$ENV_FILE"

PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
export PYTHONUNBUFFERED=1
# Shadow the venv's broken torchaudio; `transformers` 5.x reaches it on the path to every
# model class. See scripts/cpseq_mlm_train.sbatch for the full symptom.
export PYTHONPATH="$REPO/script_utils/_compat_stubs${PYTHONPATH:+:$PYTHONPATH}"

RUN_NAME="${CPSEQ_MLM_PROBE_RUN:?CPSEQ_MLM_PROBE_RUN not set}"
RUN_DIR="$CPSEQ_MLM_RUNS_DIR/$RUN_NAME"
OUT_DIR="$CPSEQ_MLM_PROBE_DIR/$RUN_NAME"

echo "=== cpseq_mlm_dump_preds ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none}"
echo "request:   gpu=${SLURM_GPUS_ON_NODE:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi -L || true
echo "run:       $RUN_NAME -> $RUN_DIR"
echo "ckpt:      ${CPSEQ_MLM_PROBE_CKPT:-<highest hf_step*>}"
echo "corpus:    $CPSEQ_MLM_CORPUS_DIR"
echo "out:       $OUT_DIR"

# --- gates. Each prints why and exits 0: a non-zero exit would strand the reduction job
# in DependencyNeverSatisfied with no explanation, and "the training run is missing" is a
# different problem from "this job crashed".
if [ ! -d "$RUN_DIR" ]; then
  echo "GATE FAILED: run dir missing: $RUN_DIR"
  echo "The training run never happened, or ran under a different CPSEQ_MLM_RUNS_DIR."
  echo "Nothing to dump. Exiting 0."
  exit 0
fi
if [ ! -f "$RUN_DIR/manifest.json" ]; then
  echo "GATE FAILED: $RUN_DIR/manifest.json missing."
  echo "The panel is rebuilt from the manifest's seed and panel_size; without it the dump"
  echo "would score a DIFFERENT panel and no comparison against the training curve would"
  echo "be valid. Exiting 0."
  exit 0
fi
if ! compgen -G "$RUN_DIR/checkpoints/hf_step*" > /dev/null; then
  echo "GATE FAILED: no hf_step* checkpoint under $RUN_DIR/checkpoints."
  echo "Distinguishing this from 'training never ran': the run dir EXISTS, so it started"
  echo "and saved nothing in HF format -- check keep_last_n_hf and the .out. Exiting 0."
  exit 0
fi
if [ ! -d "$CPSEQ_MLM_CORPUS_DIR" ]; then
  echo "GATE FAILED: corpus dir missing: $CPSEQ_MLM_CORPUS_DIR"
  echo "The panel cannot be rebuilt without the val shards it was drawn from. Exiting 0."
  exit 0
fi

mkdir -p "$OUT_DIR"

# The expected pooled accuracy, pulled from the run's OWN metrics.jsonl at the checkpoint
# step, is passed to the dump as a hard gate. This is the check that catches a panel that
# has silently drifted -- a differently sharded corpus changes row order, which changes the
# panel, which would make every downstream paired number quietly wrong while still looking
# like a plausible accuracy.
read -r CKPT_NAME CHECK_ACC CHECK_ACC_SHUF < <("$PY" - "$RUN_DIR" "${CPSEQ_MLM_PROBE_CKPT:-}" <<'PYEOF'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
want = sys.argv[2] if len(sys.argv) > 2 else ""
ckpts = sorted((d for d in (run_dir / "checkpoints").glob("hf_step*") if d.is_dir()),
               key=lambda d: int(d.name.removeprefix("hf_step")))
ckpt = (run_dir / "checkpoints" / want) if want else ckpts[-1]
step = int(ckpt.name.removeprefix("hf_step"))

acc = acc_shuf = ""
for line in (run_dir / "metrics.jsonl").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if d.get("eval") and d.get("step") == step:
        acc = d.get("val/acc/all", "")
        acc_shuf = d.get("val_pocketshuf/acc/all", "")
print(ckpt.name, acc if acc != "" else "NONE", acc_shuf if acc_shuf != "" else "NONE")
PYEOF
)
echo "resolved:  ckpt=$CKPT_NAME expected acc=$CHECK_ACC shuffled=$CHECK_ACC_SHUF"

DUMP="$OUT_DIR/preds_${CKPT_NAME}.jsonl"
ARGS=(
  --run-dir "$RUN_DIR"
  --ckpt "$CKPT_NAME"
  --corpus-dir "$CPSEQ_MLM_CORPUS_DIR"
  --out "$DUMP"
)
# No recorded accuracy for this step means the checkpoint was saved on an interval the
# eval did not land on. The dump still runs; it just cannot be gated, and says so, rather
# than being compared against the nearest eval and reported as verified.
if [ "$CHECK_ACC" != "NONE" ]; then
  ARGS+=(--check-acc "$CHECK_ACC")
else
  echo "[warn] no eval row at this step in metrics.jsonl; dumping WITHOUT the panel gate"
fi

echo "cmd: $PY script_utils/cpseq_mlm_dump_preds.py ${ARGS[*]}"
"$PY" script_utils/cpseq_mlm_dump_preds.py "${ARGS[@]}"

if [ "${CPSEQ_MLM_PROBE_SHUFFLE:-1}" = "1" ]; then
  echo
  echo "=== shuffled-pocket control ==="
  # --shuffle-seed must be the trainer's --seed: `evaluate` is called with
  # shuffle_seed=args.seed, so any other value dumps a DIFFERENT control than the one the
  # training curve reported.
  SEED="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['seed'])" "$RUN_DIR/manifest.json")"
  SHUF_ARGS=(
    --run-dir "$RUN_DIR"
    --ckpt "$CKPT_NAME"
    --corpus-dir "$CPSEQ_MLM_CORPUS_DIR"
    --out "$OUT_DIR/preds_${CKPT_NAME}_pocketshuf.jsonl"
    --shuffle-pockets
    --shuffle-seed "$SEED"
  )
  if [ "$CHECK_ACC_SHUF" != "NONE" ]; then
    SHUF_ARGS+=(--check-acc "$CHECK_ACC_SHUF")
  fi
  echo "cmd: $PY script_utils/cpseq_mlm_dump_preds.py ${SHUF_ARGS[*]}"
  "$PY" script_utils/cpseq_mlm_dump_preds.py "${SHUF_ARGS[@]}"
fi

echo
echo "=== dump done ==="
ls -la "$OUT_DIR"
