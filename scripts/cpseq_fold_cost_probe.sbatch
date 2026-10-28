#!/usr/bin/env bash
#SBATCH --job-name=cpseq_costprobe
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_costprobe_%j.out
#
# Where a Boltz-2 fold's wall time goes, and whether ten seeds can become one
# invocation with forty diffusion samples without changing the answer.
#
# A6000 is pinned, not left to the scheduler: the 44 s + 26 s/structure figure this
# probe is extending was measured on an A6000, and preempt also hands out Blackwell
# RTX_PRO_6000 and others. A timing taken on a different card would not compose with
# the number it is meant to refine.
#
# Must be a job. Run inside the node-hold allocation, boltz's working set is above the
# extern step's ceiling and the process dies to a bare SIGKILL with no traceback --
# which is how the earlier batching comparison managed to "pass" while producing
# nothing at all.
#
# Requeue-safe in the only sense that matters here: the results file is append-only and
# every row carries the full configuration that produced it, so a restarted probe adds
# a second independent measurement rather than corrupting the first.
#
# Submit: sbatch scripts/cpseq_fold_cost_probe.sbatch
#         sbatch scripts/cpseq_fold_cost_probe.sbatch /path/to/other/input.yaml

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/script_utils" ] || REPO=/home/yixiz/Proteina-Complexa

ZFS=/data/user_data/yixiz
PROBE=$ZFS/cpseq/_opt_probe
WORK=${WORK:-$PROBE/work}
OUT=${OUT:-$PROBE/results.jsonl}

# Default input: a real production fold, receptor 414 aa + 12-mer cyclic peptide, with
# its cached MSA, target template and pocket constraint all intact. Overridable as $1.
INPUT=${1:-$ZFS/cpseq/pepbench_corpus_pot/predictions/pepbench_1_3_pdb1gff/08e2460e4ef2670c/input.yaml}

echo "=== cpseq_fold_cost_probe ==="
echo "node:    $(hostname)"
echo "job:     ${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "request: -p preempt --qos=preempt_qos --gres=gpu:A6000:1 --cpus-per-task=4 --mem=32G --time=2:00:00"
echo "repo:    $REPO"
echo "input:   $INPUT"
echo "out:     $OUT"
nvidia-smi -L
echo

[ -f "$INPUT" ] || { echo "FATAL: no input yaml at $INPUT" >&2; exit 1; }

mkdir -p "$WORK" "$(dirname "$OUT")" /home/yixiz/slurm_logs

# The repo venv, only to run the probe itself; it imports the cpseq adapter (which
# imports nothing from boltz) and shells out to the boltz venv for every prediction.
PY=$REPO/.venv/bin/python
[ -x "$PY" ] || { echo "FATAL: no repo venv python at $PY" >&2; exit 1; }

exec "$PY" "$REPO/script_utils/cpseq_fold_cost_probe.py" \
  --input-yaml "$INPUT" \
  --work "$WORK" \
  --out "$OUT"
