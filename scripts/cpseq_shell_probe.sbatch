#!/usr/bin/env bash
#SBATCH --job-name=cpseq_shell_probe
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_shell_probe_%j.out
#
# Pair full-complex relax against an interface-shell relax on retained raw structures.
# CPU ONLY -- no --gres, so this runs under preempt_cpu_qos and never competes with the
# folding jobs for a GPU. It reads structures already on disk and predicts nothing.
#
# Resources, from the profile this probe exists to generalise:
#   - 8 CPUs: each structure is one full relax (~105 s) plus one shell relax (~12 s) in
#     its own process. 50 structures / 8 workers x ~2 min = ~13 min, so 4 h is generous
#     and short jobs backfill well on preempt.
#   - 24G: PyRosetta holds ~1.5 GB per process with a 422-residue pose, x8 workers, plus
#     the database. Measured peak well under this.
#   - NOT requeue-safe in the strict sense: a preempted probe restarts from the top. It
#     is ~13 min of CPU, so that is cheaper than making it resumable.
#
# Usage:
#   sbatch scripts/cpseq_shell_probe.sbatch
#   sbatch scripts/cpseq_shell_probe.sbatch SHELL=8.0 N=40

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

for kv in "$@"; do
  [[ "$kv" == *=* ]] || { echo "bad argument (want VAR=value): $kv" >&2; exit 2; }
  export "${kv?}"
done

ZFS="${ZFS:-/data/user_data/yixiz}"
SHELL_A="${SHELL:-10.0}"
N="${N:-25}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
OUT="${OUT:-$ZFS/cpseq/shell_relax_probe_shell${SHELL_A}.json}"
PY="${PY:-$REPO/.venv/bin/python}"

echo "=== cpseq_shell_probe ==="
echo "host:    $(hostname)"
echo "job:     ${SLURM_JOB_ID:-local}"
echo "request: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=none"
echo "shell:   ${SHELL_A} A   n=${N}/run   jobs=${JOBS}"
echo "out:     $OUT"
echo "========================="

srun "$PY" -u script_utils/cpseq_shell_relax_probe.py \
  --out-dir "$ZFS/cpseq/es_pilot_lnr" \
  --out-dir "$ZFS/cpseq/es_pilot_pep" \
  --n "$N" --shell "$SHELL_A" --jobs "$JOBS" --out "$OUT"

echo "done: $(date -Is)"
