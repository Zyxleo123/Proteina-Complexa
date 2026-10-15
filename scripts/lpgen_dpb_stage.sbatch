#!/bin/bash
#SBATCH --job-name=lpgen_dpbstage
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_dpbstage_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_dpbstage_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# Stage the LNR complexes for DiffPepDock, in the one layout that keeps upstream's
# preprocessing from rewriting its inputs (peptide as chain A => identity chain map).
#
# Pure file I/O over 93 small PDBs, so this is a cpu-partition job. Asking for a GPU to
# rewrite chain labels would queue it behind real training for no benefit.
#
# See script_utils/lpgen_dpb_stage.py for why the layout matters and what the --verify
# pass proves.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "dpbstage"

# The project venv is enough: the stager uses only the standard library.
[[ -x "$PY" ]] || { echo "FATAL: python not executable: ${PY}" >&2; exit 1; }

echo "[dpbstage] lnr=${LNR_DIR}"
echo "[dpbstage] out=${DPB_STAGE_DIR}"

"$PY" "${REPO}/script_utils/lpgen_dpb_stage.py" \
  --lnr-dir "$LNR_DIR" \
  --out-dir "$DPB_STAGE_DIR" \
  --data-root "$DPB_DATA_ROOT_GUARD" \
  --limit "$DPB_STAGE_LIMIT"

echo "=================================================================="
echo "[dpbstage] baseline verify (nothing has run against it yet, so this MUST pass):"
"$PY" "${REPO}/script_utils/lpgen_dpb_stage.py" --out-dir "$DPB_STAGE_DIR" --verify

echo "[dpbstage] staged files:"
ls -1 "$DPB_STAGE_DIR" | head -5
echo "[dpbstage] ... $(ls -1 "$DPB_STAGE_DIR"/*.pdb 2>/dev/null | wc -l) pdbs total"
echo "[dpbstage] done"
