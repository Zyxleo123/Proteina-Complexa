#!/usr/bin/env bash
#SBATCH --job-name=cpseq_score_cost
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/scorecost/logs/probe_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/scorecost/logs/probe_%j.out
#
# Profile the SCORING half of a cpseq fold and the two levers that cut it. The companion
# to cpseq_fold_cost_probe.sbatch, which profiles the GPU half.
#
# CPU ONLY, and that is the point being demonstrated as well as used: every structure
# this reads is already predicted, so none of this work needs a card. The `cpu` partition
# has 88-core nodes (lov[1-2,5-9]) against `general`'s contended a6000s, which is why the
# pool arm can ask for 16 cores here without queueing behind training jobs.
#
# Resources: 16 CPUs for the pool-scaling arm, and 64G because PyRosetta is ~1.5 GB per
# worker on a 422-residue pose -- 16 workers is ~24 GB plus the databases and headroom.
# Memory, not cores, is what caps this, which is exactly what the arm is measuring.
#
# Read-only on the run directory: structures are copied out and decompressed under
# WORK, and every file the scorer writes is redirected there. Safe against a live run.
#
# Submit:
#   sbatch scripts/cpseq_score_cost_probe.sbatch
#   sbatch scripts/cpseq_score_cost_probe.sbatch RUN_DIR=/path/to/run N=48

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

# Positional arguments, never `sbatch --export`: an explicit export list sets
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on the compute
# node, and the job lands requeued and HELD. Arguments travel through the job record
# untouched.
for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
# llm_ladder_pot by default: it is the only run whose predictions are on THIS cluster
# (llm_ladder shipped as journals only, so its raw_pdb_paths point at babel), and it is
# v3-gated, which is the protocol the skip arm applies to.
RUN_DIR="${RUN_DIR:-${ZFS}/cpseq/llm_ladder_pot}"
RUN_ID="${RUN_ID:-scorecost_$(date +%Y%m%d_%H%M%S)}"
WORK="${WORK:-${ZFS}/cpseq/scorecost/${RUN_ID}}"
OUT="${OUT:-${WORK}/results.jsonl}"
N="${N:-24}"
WORKERS="${WORKERS:-1,2,4,8,16}"
CHEMISTRY="${CHEMISTRY:-mainchain}"
GATE="${GATE:-v3-cis-amide-muco-windows}"
PY="${PYTHON_EXEC:-${REPO}/.venv/bin/python}"

mkdir -p "$WORK"

# hostname and the resource request go into the log at startup: slurmdbd is unreachable
# here, `sacct` returns nothing and still exits 0, so this log is the only surviving
# record of where the job ran and what it was given.
echo "=== cpseq_score_cost_probe ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=none"
echo "repo:      ${REPO}"
echo "run_dir:   ${RUN_DIR}   (READ-ONLY)"
echo "work:      ${WORK}"
echo "folds:     ${N}   workers=${WORKERS}   gate=${GATE}"
"$PY" -c 'import sys; print("python:    " + sys.executable)'
echo "=============================="

# -u because a long job's log is the only view into it: Python block-buffers stdout when
# it is a file, so without this a running probe is indistinguishable from a hung one.
srun "$PY" -u script_utils/cpseq_score_cost_probe.py \
  --run-dir "$RUN_DIR" \
  --work "$WORK/staged" \
  --out "$OUT" \
  --n "$N" \
  --workers "$WORKERS" \
  --chemistry "$CHEMISTRY" \
  --scoring-version "$GATE"

echo
echo "=== cpseq_score_cost_probe OK ==="
echo "rows: $OUT"
