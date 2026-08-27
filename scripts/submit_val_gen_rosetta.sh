#!/bin/bash
# Submitter for the val_generation Rosetta dG sidecar. Only calls sbatch -- all real work
# happens in scripts/score_val_gen_rosetta.sbatch (which runs scripts/score_val_gen_rosetta.py).
#
# This is a CPU job: Rosetta relax must never run on the GPU training path. It reads the complex
# PDBs the training loop dumped (set `val_generation.rosetta_dump_dir` in the training config) and
# appends interface dG rows to OUT_JSONL, resumably -- safe to re-run or to loop while training
# is still writing new step_*/ dirs.
#
# Usage:
#   DUMP_DIR=/path/to/val_gen_rosetta_dump \
#     OUT_JSONL=/path/to/val_gen_dg.jsonl \
#     bash scripts/submit_val_gen_rosetta.sh
#
# Configurable via environment variable (defaults shown):
#   DUMP_DIR      (required)  root dir the training loop wrote step_*/ into
#   OUT_JSONL     (required)  output JSONL of scored complexes
#   MAX_PER_STEP  0           cap complexes scored per step (0 = all dumped)
#   PARTITION     preempt     CPU-only job here; account "users" lacks qos_general (denied)
#   QOS          qos_preempt
#   TIME          08:00:00
#   NODELIST     ''
#   EXCLUDE      ''
#   DRY_RUN      ''           set to print the sbatch line without submitting

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${DUMP_DIR:?Set DUMP_DIR (the val_generation.rosetta_dump_dir from the training config)}"
: "${OUT_JSONL:?Set OUT_JSONL}"

MAX_PER_STEP="${MAX_PER_STEP:-0}"
JOB_NAME="${JOB_NAME:-valgen_rosetta}"   # per-run so concurrent runs' sidecars don't block each other
# preempt/qos_preempt: the account "users" is NOT associated with qos_general, so submitting to
# general fails with "Access/permission denied" -- which also silently killed the auto-submit from
# inside the training job. preempt is what the training jobs themselves use. This is a CPU-only job
# (no --gres), so it backfills onto any preempt node with free CPUs; if preempted it just resumes.
PARTITION="${PARTITION:-preempt}"
QOS="${QOS:-qos_preempt}"
TIME="${TIME:-08:00:00}"
NODELIST="${NODELIST:-}"
EXCLUDE="${EXCLUDE:-}"

echo "Re-check CPU node availability with myfree before submitting..."
myfree || echo "(myfree not available in this shell -- proceeding anyway)"

mkdir -p slurm_logs .slurm_envs
ENV_FILE=".slurm_envs/valgen_rosetta_$(date +%Y%m%d_%H%M%S)_$$.env"
{
  printf 'DUMP_DIR=%q\n' "$DUMP_DIR"
  printf 'OUT_JSONL=%q\n' "$OUT_JSONL"
  printf 'MAX_PER_STEP=%q\n' "$MAX_PER_STEP"
} > "$ENV_FILE"

echo "Wrote $ENV_FILE"
echo "Submitting scripts/score_val_gen_rosetta.sbatch $ENV_FILE"
echo "  partition=$PARTITION qos=$QOS time=$TIME"
echo "  dump_dir=$DUMP_DIR out=$OUT_JSONL max_per_step=$MAX_PER_STEP"

SBATCH_ARGS=( --partition="$PARTITION" --qos="$QOS" --time="$TIME" --job-name="$JOB_NAME" )
[[ -n "$NODELIST" ]] && SBATCH_ARGS+=( --nodelist="$NODELIST" )
[[ -n "$EXCLUDE" ]] && SBATCH_ARGS+=( --exclude="$EXCLUDE" )

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN set -- would submit:"
  echo "  sbatch ${SBATCH_ARGS[*]} scripts/score_val_gen_rosetta.sbatch $ENV_FILE"
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" scripts/score_val_gen_rosetta.sbatch "$ENV_FILE"
