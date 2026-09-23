#!/bin/bash
#SBATCH --job-name=m15_eidx
#SBATCH --output=logs/m15_eidx_%j.out
#SBATCH --error=logs/m15_eidx_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
# Section 6 / Deliverable E, stage 1: exact counts over the whole 2.44M-row CPSea_full
# index. Parquet only -- no structures are opened here, so the counts are exact rather
# than sampled. 96G because the pair enumeration holds the multi-segment rows in memory.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_m15_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_m15_preamble.sh"

_m15_consume_args "$@"
_m15_defaults
_m15_banner deliverable-e-index
_m15_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

cd "$REPO"
mkdir -p "$DELIV_E_DIR"

CFG="$M15_CONFIG"
if [[ "$SMOKE" == "1" ]]; then
  # A smoke must not read the 2.44M-row index; point at the 90-row sample instead, via a
  # copy of the config so the real one is never edited in place.
  CFG="${DELIV_E_DIR}/m15_smoke.yaml"
  "$PY" -c "
import sys, yaml; sys.path.insert(0, '.')
from script_utils import m15_config
c = yaml.safe_load(open('$M15_CONFIG'))
c['deliverable_e']['metadata'] = 'CPSea_data/preprocessed_sample100/metadata/cpsea_train.parquet'
c['deliverable_e']['spatial_sample_cap'] = 50
yaml.safe_dump(c, open('$CFG', 'w'))
print('[smoke] wrote $CFG')
"
fi

"$PY" script_utils/m15_deliverable_e.py index --config "$CFG" --out-dir "$DELIV_E_DIR"
echo "[deliverable-e-index] done -> $DELIV_E_DIR/deliverable_e_index.json"
