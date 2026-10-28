#!/usr/bin/env bash
#SBATCH --job-name=cpseq_prep
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_prep_%x_%j.out
#
# Prepare every target of a corpus: target-only templates, the CIF the predictor is
# handed, the conditioning pocket, and one cached receptor alignment per unique receptor
# sequence.
#
# CPU and network only -- no GPU, hence `preempt_cpu_qos`. The cost is almost entirely
# alignment fetches against the free ColabFold server, measured at a 1.6 s median per
# query, rate-limited here to `MSA_DELAY_S` out of courtesy to a shared service rather
# than out of necessity.
#
# Resumable, which matters because preempt will kill this: `fetch_receptor_msa`
# short-circuits on an existing `<sequence-hash>.a3m`, and the manifest is checkpointed
# after every fetch. A requeued run re-queries nothing it already holds. The cache is
# keyed by sequence hash and is therefore portable: seeding a new corpus directory by
# copying another's `msa_cache/` makes every shared receptor a cache hit.
#
# `manifest/targets.csv` is written once at the end, so a run killed mid-way leaves the
# alignments (the expensive part) and no manifest. Re-running rebuilds the manifest
# cheaply from the cache.
#
# Submit:
#   sbatch scripts/cpseq_prepare_corpus.sbatch \
#     OUT_DIR=/data/user_data/yixiz/cpseq/pepbench_all \
#     LIST_FILE=all.txt SOURCE=PepBench

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi

OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/cpseq/pepbench_all}"
CORPUS_ROOT="${CORPUS_ROOT:-/data/user_data/yixiz/LPData/PepBench/train_valid}"
LIST_FILE="${LIST_FILE:-all.txt}"
SOURCE="${SOURCE:-PepBench}"
MSA_DELAY_S="${MSA_DELAY_S:-2.0}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
BOLTZ_PY="${BOLTZ_PYTHON:-/data/user_data/yixiz/envs/boltz2/.venv/bin/python}"

echo "=== cpseq_prepare_corpus ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M (no GPU)"
echo "corpus:    $CORPUS_ROOT/$LIST_FILE"
echo "out_dir:   $OUT_DIR"
echo "cached:    $(ls "$OUT_DIR/msa_cache" 2>/dev/null | grep -c '\.a3m$' || echo 0) receptor alignment(s) already on disk"

srun "$PY" -u script_utils/cpseq_prepare_targets.py \
  --out-dir "$OUT_DIR" \
  --lnr-root "$CORPUS_ROOT" \
  --list-file "$LIST_FILE" \
  --source "$SOURCE" \
  --msa-delay-s "$MSA_DELAY_S" \
  --boltz-python "$BOLTZ_PY"

echo "=== cpseq_prepare_corpus OK ==="
