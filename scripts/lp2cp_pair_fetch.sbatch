#!/bin/bash
#SBATCH --job-name=lp2cp_fetch
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# Stage 2 (CPU, array): download the AFDB parent models.
#
# This is the one stage whose feasibility was NOT established before it was written: the
# compute nodes' egress to alphafold.ebi.ac.uk is untested, and from the login node a
# known-good control accession 404s exactly as a dead one does. So the job checks two
# control accessions first and, if both fail, says "this is egress, not the data", writes
# a marker the `pair` stage gates on, and exits 0 -- a non-zero exit would strand every
# afterok dependent in DependencyNeverSatisfied with no explanation of which gate tripped.
#
# Resumable: a model already on disk at over 1 KB is skipped, so a job that hits the wall
# can simply be resubmitted. One manifest per array task; shards never share an output
# file, because concurrent O_APPEND NUL-corrupts rows on this filesystem.
#
#   sbatch --array=0-7 lp2cp_pair_fetch.sbatch /path/to/env_file

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_lp2cp_pair_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "${REPO}"
mkdir -p slurm_logs

# shellcheck source=_lp2cp_pair_preamble.sh
source "${REPO}/scripts/_lp2cp_pair_preamble.sh"
# shellcheck source=/dev/null
source "${REPO}/env.sh"
_p2_consume_args "$@"
_p2_defaults
_p2_banner "fetch"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
echo "shard ${SHARD} of ${FETCH_SHARDS}"

PARENTS="${PAIR_WORK}/parents.tsv"
if [[ ! -f "${PARENTS}" ]]; then
  echo "GATE: no parent list at ${PARENTS}."
  echo "GATE: run --submit parents first. Exiting 0 so dependents are not stranded."
  exit 0
fi

"${PY}" script_utils/lp2cp_provenance.py fetch \
  --parents "${PARENTS}" \
  --models-dir "${MODELS_DIR}" \
  --out-dir "${FETCH_LOG_DIR}" \
  --shard "${SHARD}" \
  --num-shards "${FETCH_SHARDS}" \
  --version "${AFDB_VERSION}" \
  --timeout "${FETCH_TIMEOUT}" \
  --retries "${FETCH_RETRIES}" \
  --sleep "${FETCH_SLEEP}"

echo
echo "fetch shard ${SHARD} done."
