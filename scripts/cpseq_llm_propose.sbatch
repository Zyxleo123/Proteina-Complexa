#!/usr/bin/env bash
#SBATCH --job-name=cpseq_llm_propose
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_40GB|VRAM_48GB|VRAM_80GB|VRAM_96GB|VRAM_141GB
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_llm_propose_%x_%A_%a.out
#
# Arm D's proposal stage: one GENERATION of LLM-proposed sequences for every complex in
# a cohort. It folds nothing. `cpseq_search.py ladder --llm-round N` folds what this
# writes.
#
# A generation is an evolutionary step, not a batch: the elites are the best sequences so
# far by mean post-relax interface dG over their valid folds, read from the SELECTION
# half of the folds only, and each one gets its own prompt asking for offspring of it.
# The substitution budget stays measured against the original, so generations cannot
# compound into a walk off the budget the arms share.
#
# Why this is its own job rather than a call inside the fold loop
# --------------------------------------------------------------
# Llama-3.1-8B in bf16 holds ~16.5 GiB of GPU memory (measured on an A6000). Boltz-2
# needs the GPU too, and the fold loop runs it as a subprocess, so a resident LLM would
# compete with the thing it exists to serve. One load here serves every complex in the
# round -- 11.9 s warm, ~95 s cold off the NAS -- and the GPU is free again before any
# folding starts. Same reason arm C's ProteinMPNN stage is separate.
#
# The GPU is selected by VRAM, not by model name. `preempt` includes 11 GiB cards
# (2080Ti) on which a 16.5 GiB model cannot load at all, and a CUDA allocation failure can
# kill the process with no Python traceback -- so the small cards must be excluded. But
# pinning `gpu:A6000:1` excludes the L40S, A100 and RTX_PRO_6000 nodes, which all fit
# easily: measured 2026-10-24, that pin left this job PENDING on Priority for 25 minutes
# while three RTX_PRO_6000 nodes sat partly idle. The constraint admits every card with
# >= 40 GiB (126 nodes against the A6000 subset) and still cannot land on one too small.
#
# Resources are measured, not guessed:
#   - GPU 16.5 GiB for the model, plus a few hundred MB of activations at these prompt
#     lengths (~4,000 prompt tokens, <=700 generated).
#   - Host 32G: the load peaks around 17 GiB transiently. Do not run this inside a shared
#     interactive allocation -- measured 2026-10-24, a VS Code session's cgroup sat at
#     48.5 of 51.5 GiB with other work in it and the load died with std::bad_alloc.
#   - 2 CPUs: PLIP is single-threaded (~1.2 s/complex) and the tokenizer is light.
#   - ~50 s per sampling attempt, up to --max-attempts per complex, so a 20-complex
#     round needs ~70 min at the worst case and far less typically. 1 h is the default
#     because short jobs backfill on preempt; pass --time for a bigger round.
#   - PLIP interface summaries cost ~1.2 s per complex and are cached under
#     <out-dir>/interfaces/, so later rounds re-read rather than recompute them.
#
# Resumable: the proposal file for a round is rewritten, and the interface cache is keyed
# on the prediction's input_hash, so a requeued task redoes only the model calls. Nothing
# downstream double-counts, because the fold stage dedupes on the journal.
#
# Config comes from a positional env file or VAR=value arguments -- never
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets jobs held.
#
# Preflight: sbatch scripts/cpseq_llm_propose.sbatch OUT_DIR=<run> ROUND=1 PARENTS=<one>
# Run:       sbatch scripts/cpseq_llm_propose.sbatch OUT_DIR=<run> ROUND=1
# Then fold: sbatch --array=0-19%20 scripts/cpseq_ladder.sbatch \
#                OUT_DIR=<run> SHARD_COUNT=20 LLM_ROUND=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
if [[ ! -f "$REPO/script_utils/cpseq_llm_propose.py" ]]; then
  REPO=/home/yixiz/Proteina-Complexa
fi

# Positional env file first, then VAR=value overrides.
if [[ $# -gt 0 && -f "${1:-}" ]]; then
  # shellcheck disable=SC1090
  source "$1"
  shift
fi
for kv in "$@"; do
  [[ "$kv" == *=* ]] || { echo "bad argument (want VAR=value): $kv" >&2; exit 2; }
  export "${kv?}"
done

OUT_DIR="${OUT_DIR:?set OUT_DIR=<run directory holding cohort.json>}"
ROUND="${ROUND:?set ROUND=<proposal round, 1-based>}"
MODEL="${MODEL:-/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-0}"   # 0 = ELITES x OFFSPRING_PER_ELITE
ELITES="${ELITES:-4}"
OFFSPRING_PER_ELITE="${OFFSPRING_PER_ELITE:-4}"
ELITE_MIN_DISTANCE="${ELITE_MIN_DISTANCE:-2}"
FEEDBACK_FOLDS="${FEEDBACK_FOLDS:-select}"
INTERFACES_BEST="${INTERFACES_BEST:-2}"
INTERFACES_WORST="${INTERFACES_WORST:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
SHARD_COUNT="${SHARD_COUNT:-1}"
PARENTS="${PARENTS:-}"
PY="${PY:-$REPO/.venv/bin/python}"

echo "=============================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_JOB_ID:-none} array task ${SLURM_ARRAY_TASK_ID:-none}"
echo "request       : ${SLURM_CPUS_PER_TASK:-?} cpu, ${SLURM_MEM_PER_NODE:-?} MB, gpu ${SLURM_JOB_GRES:-?}"
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
echo "out dir       : $OUT_DIR"
echo "generation    : $ROUND   ${ELITES} elite(s) x ${OFFSPRING_PER_ELITE} offspring"
echo "model         : $MODEL"
echo "feedback folds: $FEEDBACK_FOLDS"
echo "shard         : $SHARD_INDEX/$SHARD_COUNT  parents='${PARENTS}'"
echo "=============================================================="

cd "$REPO"

# Keep HF entirely offline: the weights are a local directory and this job must never
# reach for the hub, where the repo is gated and a 401 would read as a missing file.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARGS=(
  --out-dir "$OUT_DIR"
  --round "$ROUND"
  --model "$MODEL"
  --batch-size "$BATCH_SIZE"
  --elites "$ELITES"
  --offspring-per-elite "$OFFSPRING_PER_ELITE"
  --elite-min-distance "$ELITE_MIN_DISTANCE"
  --feedback-folds "$FEEDBACK_FOLDS"
  --interfaces-best "$INTERFACES_BEST"
  --interfaces-worst "$INTERFACES_WORST"
  --max-attempts "$MAX_ATTEMPTS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
)
[[ -n "$PARENTS" ]] && ARGS+=(--parents "$PARENTS")
[[ "${DRY_RUN:-0}" == "1" ]] && ARGS+=(--dry-run)

srun "$PY" -u script_utils/cpseq_llm_propose.py "${ARGS[@]}"

echo "done: $(date -Is)"
