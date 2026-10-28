#!/usr/bin/env bash
#SBATCH --job-name=cpseq_hf3_des
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --array=1-20%8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_hf3_des_%A_%a.out
#
# HighFold3's opinion of the searched sequences -- the second predictor the transfer test
# needs.
#
# The searches select a sequence because BOLTZ-2's pose of it scored well under Rosetta.
# That cannot distinguish a good peptide from one that suits Boltz-2. HighFold3 is an
# independent cyclic-peptide predictor that already passed the admission gate
# (`cpseq_highfold3_controls.py`): 0.64-3.39 A receptor-aligned peptide CA RMSD on six
# crystal macrocycles, peptide never templated. afcycdesign failed the same gate at
# 34.67 A and was discarded, which is why the gate exists.
#
# ONE TASK PER PARENT, both its sequences. The searched peptide and the native it is
# compared against are folded in the same task, so they share a node, a GPU, an alignment,
# the recycle count and the seed set. Splitting them across tasks would let the pair land
# on different hardware, which is the confound that wrecked the first determinism
# measurement (one run on babel-y9-16, the other on babel-x5-28).
#
# Scope: 20 parents x 2 sequences x 8 seeds = 320 folds. 10 parents are LNR (tranche1) and
# 10 are PepBench (pepbench_valid); the task list carries each parent's own dataset dir
# because the two corpora have separate manifests.
#
# Resources are MEASURED, not guessed from AF3's notes. The controls job (10882775, six
# single-seed predictions, receptors 121-296 residues) completed in 13:32 with a python-step
# MaxRSS of 9.3 GiB against a 48G request -- 5x oversized, and oversizing is what keeps a
# job out of backfill on a partition this busy. Receptors here reach 413 residues, so 20G is
# roughly 2x the measured peak. At ~135 s per single-seed prediction, two sequences at 8
# seeds each is ~24 min plus JAX's first-call compile, so 1h. If a task dies with a bare
# SIGKILL (-9), raise the memory rather than theorising.
#
# Note the partition floor: memory is clamped to 8000M per CPU, so 4 CPUs would force 32G
# whatever --mem says. 2 CPUs is enough -- `--norun_data_pipeline` means no genetic search,
# the driver runs one prediction subprocess at a time, and the mmCIF conversion is
# single-threaded.
#
# Resumable, which preempt's REQUEUE requires: each task writes its own
# `folds_<array task id>.jsonl` and the resume set is read from EVERY `folds*.jsonl`
# present, so tasks never share a journal yet all see each other's finished work. A
# prediction already on disk is also reused, keyed by input hash, so a requeued task
# re-runs nothing it had finished.
#
# This job FOLDS ONLY. Rosetta scoring is CPU work and belongs in its own job; the
# structures land as PDB beside the mmCIF, converted and checked against the requested
# peptide before being recorded.
#
# Submit:   sbatch scripts/cpseq_hf3_designs.sbatch
# Resume:   identical -- resubmitting skips recorded sequences and cached predictions
# One task: sbatch --array=1 scripts/cpseq_hf3_designs.sbatch
# Inspect:  add DRY_RUN=1 (prints the predictor command, touches no GPU)

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

TASKS="${TASKS:-/data/user_data/yixiz/cpseq/transfer/hf3_tasks.tsv}"
OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/cpseq/transfer/hf3}"
MODEL_DIR="${MODEL_DIR:-/data/user_data/yixiz/models/af3}"
CHEMISTRY_FROM="${CHEMISTRY_FROM:-/data/user_data/yixiz/cpseq/llm_ladder}"
FOLDS="${FOLDS:-8}"
SEED="${SEED:-42}"
NUM_RECYCLES="${NUM_RECYCLES:-10}"
DRY_RUN="${DRY_RUN:-0}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array_task=${SLURM_ARRAY_TASK_ID:-none}"
nvidia-smi -L || true
echo "request: gpu:1 cpus=${SLURM_CPUS_PER_TASK:-?} mem=20G time=1:00:00"

[ -f "$TASKS" ] || { echo "missing task list: $TASKS" >&2; exit 2; }

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
LINE=$(sed -n "${TASK_ID}p" "$TASKS")
[ -n "$LINE" ] || { echo "no task on line $TASK_ID of $TASKS" >&2; exit 2; }

PARENT=$(printf '%s' "$LINE" | cut -f1)
DATASET_DIR=$(printf '%s' "$LINE" | cut -f2)
SEQ_SOURCE=$(printf '%s' "$LINE" | cut -f3)

echo "task $TASK_ID: parent=$PARENT dataset=$DATASET_DIR source=$SEQ_SOURCE folds=$FOLDS"

for p in "$SEQ_SOURCE" "$DATASET_DIR/manifest/targets.csv" "$MODEL_DIR"; do
  [ -e "$p" ] || { echo "missing input: $p" >&2; exit 2; }
done

EXTRA=()
[ "$DRY_RUN" = "1" ] && EXTRA+=(--dry-run)

python script_utils/cpseq_highfold3_designs.py \
  --dataset-dir "$DATASET_DIR" \
  --seq-source "$SEQ_SOURCE" \
  --out-dir "$OUT_DIR" \
  --model-dir "$MODEL_DIR" \
  --chemistry-from "$CHEMISTRY_FROM" \
  --parents "$PARENT" \
  --folds "$FOLDS" \
  --seed "$SEED" \
  --num-recycles "$NUM_RECYCLES" \
  --journal-tag "$TASK_ID" \
  "${EXTRA[@]}"

echo "task $TASK_ID done"
