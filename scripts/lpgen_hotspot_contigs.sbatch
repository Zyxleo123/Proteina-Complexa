#!/bin/bash
#SBATCH --job-name=lpgen_hscontigs
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_hscontigs_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_hscontigs_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Build the two hotspot validation panels. CPU only -- it reads PDBs and writes JSON.
#
# Both files are built by ONE job so they cannot drift apart: they must share the hotspot
# cutoff, the request fraction and the ori push, or the in-loop curve and the obedience table
# are measuring different requests and the decoy contrast stops being interpretable.
#
#   lpgen_val_contigs_hotspot.json        hs_native + hs_none       (in-loop, 2 specs/complex)
#   lpgen_val_contigs_hs_obedience.json   + hs_decoy, hs_decoy_orinative (reval, 4/complex)
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "hscontigs"

if [[ ! -f "${VAL_PANEL}" ]]; then
  echo "GATE: validation panel absent: ${VAL_PANEL}" >&2
  echo "GATE: build it first -> ./scripts/submit_lpgen.sh --submit manifest" >&2
  echo "[hscontigs] exiting 0 so dependents are not stranded"
  exit 0
fi

COMMON=(
  --panel "${VAL_PANEL}"
  --hotspot-cutoff "${HS_CUTOFF}"
  --hotspot-frac "${HS_FRAC}"
  --decoy-radius "${HS_DECOY_RADIUS}"
  --decoy-min-separation "${HS_DECOY_MIN_SEP}"
  --ori-push "${HS_ORI_PUSH}"
  --pocket-cutoff "${POCKET_CUTOFF_A}"
)

echo "[hscontigs] ===== in-loop panel: hs_native + hs_none ====="
"$PY" "${REPO}/script_utils/lpgen_build_val_contigs.py" \
  "${COMMON[@]}" --hotspot-mode native --out "${HS_CONTIGS}"

echo
echo "[hscontigs] ===== obedience panel: + hs_decoy, hs_decoy_orinative ====="
"$PY" "${REPO}/script_utils/lpgen_build_val_contigs.py" \
  "${COMMON[@]}" --hotspot-mode obedience --out "${HS_OBEDIENCE_CONTIGS}"

echo
echo "=================================================================="
for f in "${HS_CONTIGS}" "${HS_OBEDIENCE_CONTIGS}"; do
  n=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f")
  echo "[hscontigs] ${f}: ${n} specs"
done
echo "[hscontigs] in-loop panel is wired by configs/rfd3_lp/datasets/val/lp_panel_hotspot.yaml"
echo "[hscontigs] obedience panel: --submit reval --set REVAL_CONTIGS=${HS_OBEDIENCE_CONTIGS}"
