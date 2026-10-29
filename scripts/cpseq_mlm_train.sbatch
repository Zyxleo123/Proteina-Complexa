#!/bin/bash
#SBATCH --job-name=cpseq_mlm_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_train_%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_train_%x_%j.out
#
# Fine-tune ESM-2/PepMLM into the CPSea cyclic-peptide unmasking proposer. One arm per
# submission; `CPSEQ_MLM_NO_POCKET=1` selects the target-free control.
#
# `general` rather than `preempt`: this partition is all a6000 (48 GB), so the GPU class
# is pinned by the partition and cannot silently land on an 11 GB 2080 Ti. A 650M
# fine-tune at these sequence lengths needs ~12 GB, so the headroom is deliberate rather
# than measured to the edge.
#
# Resumable: the trainer reloads checkpoints/last.pt and continues, so a requeue after the
# 2-day limit costs one checkpoint interval rather than the run.
#
# Do NOT sbatch directly -- use scripts/submit_cpseq_mlm.sh. Config travels in a FILE
# passed positionally, never via `sbatch --export`.

set -euo pipefail          # in the body: a failed source must not run under system python and exit 0

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

ENV_FILE="${1:-}"
[ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ] || { echo "FATAL: env file not found: ${ENV_FILE:-<none>}"; exit 1; }
source "$REPO/scripts/cpseq_mlm_env.sh"
source "$ENV_FILE"

RUN_NAME="${CPSEQ_MLM_RUN_NAME:?CPSEQ_MLM_RUN_NAME not set}"
RUN_DIR="$CPSEQ_MLM_RUNS_DIR/$RUN_NAME"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
# Unbuffered so the loss curve is readable in the log while the job runs, not only after
# it exits. A block-buffered training log is indistinguishable from a hung job.
export PYTHONUNBUFFERED=1

# Shadow the venv's broken torchaudio. `transformers` 5.x reaches it on the path to every
# model class (modeling_utils -> loss.loss_utils -> loss_rnnt), its guard passes because
# the package IS installed, and the import then dies with
#   OSError: .../torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl
# because that extension was built against a different torch than the installed 2.7.0.
# This prepends an import-only stub instead of uninstalling from the venv the whole flow
# stack shares. See script_utils/_compat_stubs/torchaudio/__init__.py.
export PYTHONPATH="$REPO/script_utils/_compat_stubs${PYTHONPATH:+:$PYTHONPATH}"

echo "=== cpseq_mlm_train ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none}"
echo "request:   gpu=${SLURM_GPUS_ON_NODE:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M partition=${SLURM_JOB_PARTITION:-?}"
nvidia-smi -L || true
echo "run:       $RUN_NAME -> $RUN_DIR"
echo "model:     $CPSEQ_MLM_MODEL"
echo "arm:       ${CPSEQ_MLM_NO_POCKET:-0}  (1 = target-free control)"
echo "HF_HOME:   $HF_HOME"

# --- gate: the corpus must exist and be non-trivial. A trainer that silently validates
# on two shards is worse than one that refuses.
if [ ! -d "$CPSEQ_MLM_CORPUS_DIR" ]; then
  echo "GATE FAILED: corpus dir missing: $CPSEQ_MLM_CORPUS_DIR"
  echo "The corpus array did not run, or ran against a different root. Exiting 0."
  exit 0
fi
N_TRAIN_SHARDS=$(find "$CPSEQ_MLM_CORPUS_DIR" -maxdepth 1 -name 'train_*.jsonl' | wc -l)
if [ "$N_TRAIN_SHARDS" -lt 1 ]; then
  echo "GATE FAILED: no train_*.jsonl in $CPSEQ_MLM_CORPUS_DIR"
  echo "Distinguishing this from 'upstream never ran': the directory EXISTS, so the array"
  echo "ran and produced nothing -- read its .stats.json reasons. Exiting 0."
  exit 0
fi
echo "corpus:    $N_TRAIN_SHARDS train shard(s)"

ARGS=(
  --corpus-dir "$CPSEQ_MLM_CORPUS_DIR"
  --run-dir "$RUN_DIR"
  --model "$CPSEQ_MLM_MODEL"
  --max-steps "$CPSEQ_MLM_MAX_STEPS"
  --batch-size "$CPSEQ_MLM_BATCH_SIZE"
  --accum "$CPSEQ_MLM_ACCUM"
  --lr "$CPSEQ_MLM_LR"
  --warmup "$CPSEQ_MLM_WARMUP"
  --eval-every "$CPSEQ_MLM_EVAL_EVERY"
  --save-every "$CPSEQ_MLM_SAVE_EVERY"
  --eval-n "$CPSEQ_MLM_EVAL_N"
  --cyc-dropout "$CPSEQ_MLM_CYC_DROPOUT"
  --seed "$CPSEQ_MLM_SEED"
)
if [ "${CPSEQ_MLM_NO_POCKET:-0}" = "1" ]; then
  ARGS+=(--no-pocket)
fi
if [ -n "${CPSEQ_MLM_HOLDOUT_CLUSTERS:-}" ]; then
  ARGS+=(--holdout-clusters "$CPSEQ_MLM_HOLDOUT_CLUSTERS")
fi
if [ -n "${CPSEQ_MLM_TRAIN_LIMIT:-}" ]; then
  ARGS+=(--train-limit "$CPSEQ_MLM_TRAIN_LIMIT")
fi

mkdir -p "$RUN_DIR"
echo "cmd: $PY script_utils/cpseq_train_mlm.py ${ARGS[*]}"
"$PY" script_utils/cpseq_train_mlm.py "${ARGS[@]}"

echo "=== train done ==="
