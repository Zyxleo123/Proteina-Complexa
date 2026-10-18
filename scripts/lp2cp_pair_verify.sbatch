#!/bin/bash
#SBATCH --job-name=lp2cp_verify
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 0 (CPU): prove the pairing is correct before anything is downloaded or built.
#
#   1. the provenance parser and the shard partition (`selftest`)
#   2. the geometry, end to end, against a synthesised parent: a real CPSea complex with
#      its two chains merged into one and rigidly moved. The pairing must undo that
#      exactly, which makes receptor fit ~0, n_mut 0, and the LP/CP terminal gaps equal.
#      Then the same parent with only its peptide displaced 8 A: the receptor fit must be
#      unchanged (the excised range really is excluded from the anchors) and the measured
#      amplitude must read 8 A, not merely "nonzero".
#   3. the data roots the later stages read.
#
# (2) is what a parser test cannot cover, and it is where the silent failures live: a
# peptide fitted as part of the pocket, or an amplitude scaled wrong, both produce
# plausible-looking numbers.
#
#   sbatch lp2cp_pair_verify.sbatch /path/to/env_file

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
_p2_banner "verify"

echo "--- [1/3] provenance parser + shard partition"
"${PY}" script_utils/lp2cp_provenance.py selftest

echo
echo "--- [2/3] pairing geometry against a synthesised parent"
"${PY}" script_utils/test_lp2cp_provenance.py

echo
echo "--- [3/3] data roots"
_p2_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }
for f in "${LP_METADATA}"; do
  if [[ -f "${f}" ]]; then
    echo "  ok      ${f}"
  else
    echo "  MISSING ${f}  (report section 2 will read UNDECIDED without it)"
  fi
done
[[ -d "${LNR_ROOT}" ]] && echo "  ok      ${LNR_ROOT}" || echo "  MISSING ${LNR_ROOT} (optional)"

echo
echo "Verification passed."
