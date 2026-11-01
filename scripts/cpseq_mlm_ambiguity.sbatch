#!/bin/bash
#SBATCH --job-name=cpseq_mlm_ambiguity
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_ambiguity_%x_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/cpseq_mlm_ambiguity_%x_%j.out
#
# Estimate the task's irreducible accuracy and loss from the DATA, with no model involved.
# This is the sibling of the dump/softacc chain, not a dependant of it: it needs no
# checkpoint, so it queues concurrently rather than waiting behind a GPU job.
#
# --mem=96G: the probe streams the corpus (rows are never retained) but holds one key table
# per grouping, and the model grouping has one entry per distinct (pocket, length,
# chemistry) context -- measured at ~5994 distinct contexts per 6000 rows on the smoke
# corpus, so expect ~2.5M entries at ~400 bytes. That is ~1 GB of dicts; the request is
# large because the `cpu` partition has 768 GB per node and a host-RAM OOM kill here would
# leave no traceback.
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

echo "=== cpseq_mlm_ambiguity ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M partition=${SLURM_JOB_PARTITION:-?}"
echo "corpus:    $CPSEQ_MLM_CORPUS_DIR"
echo "globs:     $CPSEQ_MLM_AMBIG_GLOBS"

if [ ! -d "$CPSEQ_MLM_CORPUS_DIR" ]; then
  echo "GATE FAILED: corpus dir missing: $CPSEQ_MLM_CORPUS_DIR"
  echo "This probe reads only the corpus, so there is nothing it can do. Exiting 0."
  exit 0
fi

mkdir -p "$CPSEQ_MLM_PROBE_DIR"
OUT="$CPSEQ_MLM_PROBE_DIR/ambiguity.json"

echo "cmd: $PY script_utils/cpseq_mlm_ambiguity_probe.py --corpus-dir $CPSEQ_MLM_CORPUS_DIR --globs $CPSEQ_MLM_AMBIG_GLOBS --out $OUT"
"$PY" script_utils/cpseq_mlm_ambiguity_probe.py \
  --corpus-dir "$CPSEQ_MLM_CORPUS_DIR" \
  --globs "$CPSEQ_MLM_AMBIG_GLOBS" \
  --out "$OUT"

echo
echo "=== ambiguity done ==="
cat <<'NOTE'
NOTE on reading this: the 'model' grouping is confounded here by construction. The pocket
is extracted by radius around the peptide, so rows sharing a pocket string are
near-duplicate complexes rather than two peptides against one target. Groups offering only
one distinct peptide are dropped, and the drop count is printed -- but if the usable count
is small, the model-side ceiling is NOT established by this probe, and the calibration
columns in softacc are the better test of the plateau claim.
NOTE
