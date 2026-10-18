#!/bin/bash
#SBATCH --job-name=lp2cp_parents
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 1 (CPU): decode CPSea provenance and choose the parent set.
#
# Emits the two work lists every later stage reads:
#   ${PAIR_WORK}/parents.tsv        unique AFDB accessions to download
#   ${PAIR_WORK}/pairs_todo.parquet one row per pair to build
#
# Refuses to proceed if any row's residue span disagrees with its peptide_length. That
# equality is the whole basis for treating the id's range as the peptide's address, and it
# held on all 2,444,209 train rows when this was written -- if it ever breaks, pairs would
# be built against the wrong residues and every downstream number would be quietly wrong.
#
# 64G because the decode reads the full metadata table into pandas.
#
#   sbatch lp2cp_pair_parents.sbatch /path/to/env_file

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
_p2_banner "parents"

_p2_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

META_ARGS=()
for split in ${CPSEA_SPLITS}; do
  META_ARGS+=("${CPSEA_FULL_META}/${split}.parquet")
done

CU_FLAG="--cluster-unique"
[[ "${CLUSTER_UNIQUE}" == "1" ]] || CU_FLAG="--no-cluster-unique"

# shellcheck disable=SC2086
"${PY}" script_utils/lp2cp_provenance.py parents \
  --metadata "${META_ARGS[@]}" \
  --out-dir "${PAIR_WORK}" \
  --parent-limit "${PARENT_LIMIT}" \
  --max-rows "${MAX_ROWS}" \
  --cyc-types ${CYC_TYPES} \
  "${CU_FLAG}"

echo
echo "parents stage done."
