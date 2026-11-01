#!/bin/bash
#SBATCH --job-name=cpseq_mlm_softacc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_softacc_%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_softacc_%x_%j.out
#
# Score the prediction dump under substitution-tolerant metrics, and report calibration.
# Answers two questions the exact-match curve cannot:
#   1. are the model's misses CONSERVATIVE substitutions, more so than a lookup's?
#   2. is the ~2.36-nat loss an entropy floor, or overconfidence?
#
# CPU partition, no GPU: this only reads the dump and the corpus. Asking for a GPU to
# average a scoring matrix would queue behind real training work for nothing.
#
# --mem=48G: the train-side priors are built from 2.58M corpus rows, measured at ~1.7 GB
# of row objects (pockets are small -- median 45 residues), so this is ~25x headroom. It is
# deliberate rather than tight because a host-RAM OOM kill leaves no traceback and MaxRSS
# is not observable after the fact on this cluster (slurmdbd is unreachable).
#
# Do NOT sbatch directly -- use scripts/submit_cpseq_mlm_probes.sh.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

ENV_FILE="${1:-}"
[ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ] || { echo "FATAL: env file not found: ${ENV_FILE:-<none>}"; exit 1; }
source "$REPO/scripts/cpseq_mlm_env.sh"
source "$ENV_FILE"

PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
export PYTHONUNBUFFERED=1

RUN_NAME="${CPSEQ_MLM_PROBE_RUN:?CPSEQ_MLM_PROBE_RUN not set}"
OUT_DIR="$CPSEQ_MLM_PROBE_DIR/$RUN_NAME"

echo "=== cpseq_mlm_softacc ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M partition=${SLURM_JOB_PARTITION:-?}"
echo "dumps in:  $OUT_DIR"
echo "corpus:    $CPSEQ_MLM_CORPUS_DIR"
echo "n_boot:    $CPSEQ_MLM_PROBE_NBOOT"

# --- gate: the dump must exist. Distinguish "the GPU job failed its own gate and exited 0"
# from "this job is broken": both leave no dump, but only the first is expected.
if [ ! -d "$OUT_DIR" ] || ! compgen -G "$OUT_DIR/preds_*.jsonl" > /dev/null; then
  echo "GATE FAILED: no preds_*.jsonl under $OUT_DIR"
  echo "The dump job either never ran or hit one of its own gates and exited 0 -- read its"
  echo ".out before re-submitting this one. Exiting 0."
  exit 0
fi

mkdir -p "$CPSEQ_MLM_PROBE_DIR"

# Every dump in the directory is reduced, so the real-pocket arm and the shuffled control
# are scored by one job with one priors cache -- the priors are a property of the training
# corpus, not of the arm, and rebuilding them per dump would waste the 2.58M-row read.
shopt -s nullglob
for DUMP in "$OUT_DIR"/preds_*.jsonl; do
  BASE="$(basename "$DUMP" .jsonl)"
  echo
  echo "################ $BASE"
  "$PY" script_utils/cpseq_mlm_softacc.py \
    --dump "$DUMP" \
    --corpus-dir "$CPSEQ_MLM_CORPUS_DIR" \
    --priors-cache "$CPSEQ_MLM_PRIORS_CACHE" \
    --out "$OUT_DIR/softacc_${BASE#preds_}.json" \
    --n-boot "$CPSEQ_MLM_PROBE_NBOOT"
done

echo
echo "=== softacc done ==="
ls -la "$OUT_DIR"
