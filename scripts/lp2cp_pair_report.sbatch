#!/bin/bash
#SBATCH --job-name=lp2cp_pair_report
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 4 (CPU): diagnose the built pair set and render the report.
#
# Reads only saved artefacts -- the pair shards plus a cached npz of reference terminal
# gaps -- so the figures redraw without rebuilding a pair or re-profiling a reference.
#
# The question it exists to answer is section 2: CPSea's fragments were selected BECAUSE
# they were cyclizable, which is selection on the target variable, so the parents may sit
# at small terminal gaps while real LP binders sit at 16-21 A. `protfrag` is the control
# (AFDB fragments, like CPSea's parents) and `pepbench` the target (real bound binders).
# Tracking protfrag but not pepbench IS the bias, quantified.
#
# It also reports the mutation-budget labels the coupling gives for free, the transport
# amplitude against the 2 A floor, and the receptor fit residual.
#
#   sbatch lp2cp_pair_report.sbatch /path/to/env_file

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
_p2_banner "report"

LP_META_ARG=()
if [[ -f "${LP_METADATA}" ]]; then
  LP_META_ARG=(--lp-metadata "${LP_METADATA}")
else
  echo "WARNING: no LP metadata at ${LP_METADATA}; the pepbench/protfrag reference" >&2
  echo "         columns will be absent and section 2's verdict will read UNDECIDED." >&2
fi

LNR_ARG=()
[[ -d "${LNR_ROOT}" ]] && LNR_ARG=(--lnr-root "${LNR_ROOT}")

"${PY}" script_utils/lp2cp_pair_report.py \
  --pairs-dir "${PAIRS_DIR}" \
  --out-dir "${REPORT_DIR}" \
  --ref-sample "${REF_SAMPLE}" \
  "${LP_META_ARG[@]}" \
  "${LNR_ARG[@]}"

echo
echo "report stage done: ${REPORT_DIR}/report.md"
