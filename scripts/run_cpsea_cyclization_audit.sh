#!/bin/bash
# Full-dataset scan of the CPSea head-to-tail claim (~2.44M structures).
#
# The audit is pure PDB I/O, so it is CPU-only and embarrassingly parallel; the
# login node has 2 cores, which is why this exists as an array job.
#
#   sbatch scripts/run_cpsea_cyclization_audit.sh
#   # then, once all tasks finish:
#   python script_utils/summarize_cpsea_audit_shards.py slurm_logs/cyc_audit
#
#SBATCH --job-name=cpsea_cyc_audit
#SBATCH --partition=general
#SBATCH --array=0-31
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/cyc_audit/shard_%a.out

set -euo pipefail
# Under sbatch, "$0" is a copy in the slurm spool dir, not this file, so
# "$(dirname "$0")/.." resolves outside the repo and mkdir below dies on a read-only
# path. SLURM_SUBMIT_DIR is where sbatch was invoked; BASH_SOURCE covers a plain run.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p slurm_logs/cyc_audit
source env.sh

python script_utils/audit_cpsea_cyclization_endpoints.py \
    --splits train val test \
    --all \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --out-csv "slurm_logs/cyc_audit/shard_${SLURM_ARRAY_TASK_ID}.csv"
