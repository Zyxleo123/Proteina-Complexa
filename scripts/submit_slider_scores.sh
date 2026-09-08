#!/bin/bash
# Score the cases the step viewer shows -- and only those. Calls sbatch and nothing else.
#
#   bash scripts/submit_slider_scores.sh [--slider-index PATH] [--gpu-partition P]
#        [--gpu-nodelist N] [--cpu-partition P] [--dry-run]
#
# Stage 1 GPU  re-sample the viewer's cases, writing complexes  (sdedit_cases.sbatch)
# Stage 2 CPU  Rosetta dG before + after                        (sdedit_rosetta.sbatch)
# Stage 3 CPU  rebuild the viewer page with the energies        (slider_page.sbatch)
#
# The cases are UNPROJECTED: staged crystal complexes straight from lnr_test.parquet, the
# same inputs the trajectories were sampled from. Re-sampling is required because those runs
# saved the peptide without its receptor and the loader frame is not reproducible across
# processes -- but with the same pin, seed and grid point the structure is the same one.
#
# One timestamped env file, passed as a POSITIONAL argument -- never `sbatch --export`.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SLIDER_INDEX="${REPO}/evaluation_results/sdedit_traj_20260902_163025/slider/slider_index.json"
GPU_PARTITION="general"; GPU_NODELIST=""; CPU_PARTITION="cpu"
GPU_TIME="02:00:00"; CPU_TIME="04:00:00"; PAGE_TIME="00:20:00"; DRYRUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slider-index)  SLIDER_INDEX="$2";  shift 2 ;;
    --gpu-partition) GPU_PARTITION="$2"; shift 2 ;;
    --gpu-nodelist)  GPU_NODELIST="$2";  shift 2 ;;
    --cpu-partition) CPU_PARTITION="$2"; shift 2 ;;
    --dry-run)       DRYRUN=1;           shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
[[ -f "$SLIDER_INDEX" ]] || { echo "FATAL: no slider index at $SLIDER_INDEX" >&2; exit 1; }

PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
# Read the cases out of the index itself rather than restating them: the page and the scores
# then cannot drift apart, and a case added to the viewer is scored without editing this file.
# One field per LINE: the example list itself contains spaces, so a single whitespace-split
# read would shift every field along by three and hand the sampler a peptide id as a t_ca.
mapfile -t CASE_FIELDS < <("$PY" - "$SLIDER_INDEX" <<'PYEOF'
import json, sys
idx = json.load(open(sys.argv[1]))
def start(v): return v[0] if isinstance(v, list) and v else v
print(" ".join(sorted({c["example_id"] for c in idx})))
print(" ".join(sorted({c["chem"] for c in idx})))
print(" ".join(str(v) for v in sorted({round(float(start(c["t_ca"])), 4) for c in idx})))
print(" ".join(str(v) for v in sorted({round(float(start(c["t_lat"])), 4) for c in idx})))
PYEOF
)
[[ ${#CASE_FIELDS[@]} -eq 4 ]] || { echo "FATAL: could not read cases from $SLIDER_INDEX" >&2; exit 1; }
EXAMPLES="${CASE_FIELDS[0]}"; CHEMS="${CASE_FIELDS[1]}"
T_CA="${CASE_FIELDS[2]}";     T_LAT="${CASE_FIELDS[3]}"
echo "cases from $(basename "$SLIDER_INDEX"):"
echo "  examples: $EXAMPLES"
echo "  chems:    $CHEMS"
echo "  grid:     t_ca $T_CA / t_lat $T_LAT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="sliderscore_${STAMP}"
mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<ENVEOF
# Auto-written by scripts/submit_slider_scores.sh at ${STAMP}.
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
SLIDER_INDEX=${SLIDER_INDEX}
PAGE_LABEL=${RUN_ID}
# Unprojected staged crystal inputs -- the same metadata the trajectories were sampled from.
LNR_METADATA=${REPO}/CPSea_data/lnr_staged/metadata/lnr_test.parquet
# The pin the trajectories used. A different checkpoint would sample a different structure
# and the energies would describe something the viewer does not show.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
CASE_EXAMPLES="${EXAMPLES}"
CYC_TYPES="${CHEMS}"
T_CA_GRID="${T_CA}"
T_LAT_GRID="${T_LAT}"
SEEDS="0"
NSTEPS=0
# Score every edit: open rings and abstentions included, since the viewer shows them too.
ROSETTA_SHARDS=1
ROSETTA_ONLY_SCORABLE=0
ENVEOF
echo; echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

if [[ $DRYRUN == 1 ]]; then
  echo "--dry-run: resolving the sampler locally, submitting nothing."
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  # shellcheck disable=SC2086
  "$PY" scripts/sdedit_cyclize.py --ckpt-path "$FLOW_CKPT_PATH" --metadata "$LNR_METADATA" \
    --examples ${CASE_EXAMPLES} --out /dev/null --cyc-types ${CYC_TYPES} \
    --t-ca ${T_CA_GRID} --t-lat ${T_LAT_GRID} --seeds ${SEEDS} --dry-run
  exit 0
fi

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
GPU_NODE_ARG=(); [[ -n "$GPU_NODELIST" ]] && GPU_NODE_ARG=(--nodelist="$GPU_NODELIST")

EDIT_ID=$(sbatch --parsable \
  --partition="$GPU_PARTITION" "${GPU_NODE_ARG[@]}" --time="$GPU_TIME" \
  --job-name="${RUN_ID}_edits" scripts/sdedit_cases.sbatch "$ENV_FILE")
echo "stage 1 GPU  re-sample cases   job ${EDIT_ID} (${GPU_PARTITION}${GPU_NODELIST:+/$GPU_NODELIST})"

ROS_ID=$(sbatch --parsable \
  --partition="$CPU_PARTITION" --time="$CPU_TIME" --array=0-0 \
  --dependency=afterok:"$EDIT_ID" --job-name="${RUN_ID}_dg" \
  scripts/sdedit_rosetta.sbatch "$ENV_FILE")
echo "stage 2 CPU  Rosetta dG        job ${ROS_ID} (afterok:${EDIT_ID})"

PAGE_ID=$(sbatch --parsable \
  --partition="$CPU_PARTITION" --time="$PAGE_TIME" \
  --dependency=afterok:"$ROS_ID" --job-name="${RUN_ID}_page" \
  scripts/slider_page.sbatch "$ENV_FILE")
echo "stage 3 CPU  rebuild the page  job ${PAGE_ID} (afterok:${ROS_ID})"
echo
echo "page: ${REPO}/evaluation_results/${RUN_ID}/cyclization_step_viewer.html"
