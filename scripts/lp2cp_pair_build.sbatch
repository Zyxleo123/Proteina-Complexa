#!/bin/bash
#SBATCH --job-name=lp2cp_pair
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#
# Stage 3 (CPU, array): build the LP/CP pairs.
#
# For each CPSea complex: slice the parent AFDB model at the provenance residue range,
# rigid-fit the parent onto the CPSea receptor using the receptor residues they share by
# id (the excised range excluded, or the peptide would be fitted as part of the pocket),
# and write a complex whose chain A is CPSea's own receptor and chain B is the parent's
# peptide in that receptor's frame.
#
# The paired record also carries what the diagnosis needs: the sequence delta (the free
# mutation-budget label), the terminal gap on both sides, the receptor fit residual, and
# the LP->CP peptide displacement.
#
# Resumable: the shard's own jsonl is read back and finished example_ids are skipped. A
# NUL-corrupted line is FATAL rather than skipped -- skipping it would rebuild pairs the
# shard already has while reporting a clean run.
#
#   sbatch --array=0-7 lp2cp_pair_build.sbatch /path/to/env_file

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
_p2_banner "pair"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
echo "shard ${SHARD} of ${PAIR_SHARDS}"

TODO="${PAIR_WORK}/pairs_todo.parquet"
if [[ ! -f "${TODO}" ]]; then
  echo "GATE: no work list at ${TODO}."
  echo "GATE: run --submit parents first. Exiting 0 so dependents are not stranded."
  exit 0
fi

# A real misconfiguration, so this one exits non-zero: re-sharding an existing build
# duplicates pairs into different shards and both copies look valid.
_p2_assert_shard_count "${PAIRS_DIR}" "${PAIR_SHARDS}" || exit 1

"${PY}" script_utils/lp2cp_provenance.py pair \
  --todo "${TODO}" \
  --models-dir "${MODELS_DIR}" \
  --lp-pdb-dir "${LP_PDB_DIR}" \
  --out-dir "${PAIRS_DIR}" \
  --shard "${SHARD}" \
  --num-shards "${PAIR_SHARDS}" \
  --version "${AFDB_VERSION}" \
  --min-anchor "${MIN_ANCHOR}"

echo
echo "pair shard ${SHARD} done."
