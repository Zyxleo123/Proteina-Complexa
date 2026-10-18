#!/bin/bash
#SBATCH --job-name=lpgen_hsprobe
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_hsprobe_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_hsprobe_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# Measure RFD3's hotspot conditioning on our LP data. CPU ONLY -- no model, no GPU.
#
# This is the gate in front of the hotspot fine-tune. It builds the REAL arm config and
# iterates REAL examples through the REAL transform stack, so what it reports is what
# training would see. Asking for a GPU would queue a pure-dataloader job behind training.
#
# It exits 2 (not 1) when an invariant fails, so the submitter can tell "the conditioning is
# wrong" apart from "the job could not run at all". A low live-hotspot rate is NOT a failure:
# that is the measurement, and the dosage decision belongs to whoever reads it.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "hsprobe"

if ! _lpgen_require_manifest; then
  echo "[hsprobe] GATE not satisfied -- exiting 0 so dependents are not stranded"
  exit 0
fi

ARM_TREE="${UPSTREAM_DIR}/foundry"
ARM_PY="${ENVS_DIR}/foundry/bin/python"
for p in "$ARM_TREE/models/rfd3/configs/train.yaml" "$ARM_PY"; do
  [[ -e "$p" ]] || { echo "[hsprobe] GATE missing: $p (run --submit stage / --submit env)" >&2; exit 0; }
done

OUT_DIR="${HSPROBE_DIR:-${LPGEN_RUNS_DIR}/${RUN_ID}_hsprobe}"
mkdir -p "$OUT_DIR"
case "$OUT_DIR" in
  "$REPO"*|"$HOME"*) echo "[hsprobe] FATAL: out dir is on home: $OUT_DIR" >&2; exit 1 ;;
esac

# Our condition class and our configs both live in THIS repo. PYTHONPATH makes the class
# importable from the arm's env; hydra.searchpath makes the configs visible. PROJECT_ROOT is
# read by rfd3's own train.yaml composition and by `_config_path` in train.py.
export PYTHONPATH="${REPO}/src"
export PROJECT_ROOT="${ARM_TREE}"

cd "$ARM_TREE"
echo "[hsprobe] experiment=${HSPROBE_EXPERIMENT} n=${HSPROBE_N} passes=${HSPROBE_PASSES}"
set +e
"$ARM_PY" "${REPO}/script_utils/lpgen_hotspot_probe.py" \
  --config-dir "${ARM_TREE}/models/rfd3/configs" \
  --searchpath "pkg://rfd3.configs,pkg://configs,file://${REPO}/configs/rfd3_lp" \
  --experiment "${HSPROBE_EXPERIMENT}" \
  --n "${HSPROBE_N}" \
  --passes "${HSPROBE_PASSES}" \
  --seed "${HSPROBE_SEED}" \
  --out-dir "$OUT_DIR" 2>&1 | tee "${OUT_DIR}/hsprobe.log"
RC=${PIPESTATUS[0]}
set -e

echo "=================================================================="
echo "[hsprobe] exit code: ${RC}"
case "$RC" in
  0) echo "[hsprobe] every invariant PASSED. Read the live-hotspot rate in the summary"
     echo "[hsprobe] before launching the arm: if the channel is live on only a few percent"
     echo "[hsprobe] of examples, raise max_ppi_hotspots_frac_to_provide rather than hoping." ;;
  2) echo "[hsprobe] an INVARIANT FAILED -- the arm would train on wrong conditioning."
     echo "[hsprobe] Do not launch the fine-tune until this is understood." ;;
  *) echo "[hsprobe] the probe could not run; this says nothing about the conditioning." ;;
esac
echo "[hsprobe] outputs in ${OUT_DIR}"
exit "$RC"
