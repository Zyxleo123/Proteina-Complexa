#!/bin/bash
#SBATCH --job-name=valgen_rosetta
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU sidecar: score Rosetta interface dG on the complexes val_generation dumped to disk
# (see src/proteinfoundation/eval/val_gen_dump.py and scripts/score_val_gen_rosetta.py).
# This is deliberately CPU-only -- Rosetta relax must never share the GPU training path.
#
# Do not sbatch directly -- use scripts/submit_val_gen_rosetta.sh, which picks an AVAILABLE
# CPU node from `myfree` and writes the env file this sources as $1. That env file must set
# DUMP_DIR and OUT_JSONL (and may set MAX_PER_STEP, PYTHON_EXEC).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"

: "${DUMP_DIR:?DUMP_DIR not set in $ENV_FILE}"
: "${OUT_JSONL:?OUT_JSONL not set in $ENV_FILE}"
MAX_PER_STEP="${MAX_PER_STEP:-0}"

source env.sh
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)"
echo "Python: $PY"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable from $PY"; exit 1; }

echo "Scoring dump_dir=$DUMP_DIR -> out=$OUT_JSONL (max_per_step=$MAX_PER_STEP)"
"$PY" scripts/score_val_gen_rosetta.py \
    --dump-dir "$DUMP_DIR" \
    --out "$OUT_JSONL" \
    --max-per-step "$MAX_PER_STEP"
echo "Done."
