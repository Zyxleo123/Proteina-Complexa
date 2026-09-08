#!/bin/bash
# ARM A -- the in-distribution control for de novo cyclic-peptide generation.
#
#   bash scripts/submit_uncond_control.sh [--smoke] [--partition P] [--nodelist N]
#                                         [--shards N] [--time T] [--seeds "0 1 2 3"]
#
# The question this answers
# -------------------------
# `val_generation` logs ~0.85-0.96 ring closure on CPSea val; de novo generation on the
# LNR-staged targets scores ~0.45-0.52. Those two runs differ on TWO axes at once:
#
#   (1) target population  -- CPSea val (in-distribution, receptor stored pre-cropped to the
#                             pocket, ~13-18 disjoint segments) vs LNR-staged (full contiguous
#                             receptor chain, ~1 segment)
#   (2) cyclization type   -- native per example vs all 3 forced on every target
#
# This arm holds the PIPELINE fixed (same script, sampler, checkpoint, closure metric) and
# sets BOTH axes to the val_generation setting: CPSea val targets, native type. If closure
# comes back ~0.85 the pipeline is sound and the LNR gap is about the targets. If it comes
# back ~0.5, the gap is in this script's integration path (partial_simulation from t=0)
# rather than the data -- and Arm B would be chasing the wrong thing.
#
# Same pin/AE/sampler as scripts/submit_uncond_gen.sh, so the two are directly comparable.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"; NODELIST=""; TIME="08:00:00"; SUM_TIME="00:20:00"; SHARDS=6; SMOKE=0
SEEDS="0 1 2 3"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --shards)    SHARDS="$2";    shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    --seeds)     SEEDS="$2";     shift 2 ;;
    --smoke)     SMOKE=1;        shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

CONTROL_META="${REPO}/CPSea_data/control/cpsea_val_control.parquet"
[[ -f "$CONTROL_META" ]] || {
  echo "FATAL: control metadata missing: $CONTROL_META"
  echo "  build it first:  .venv/bin/python scripts/sample_cpsea_val_control.py \\"
  echo "                     --out $CONTROL_META --n 200"
  exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="uncondctl_${STAMP}"; [[ $SMOKE == 1 ]] && RUN_ID="uncondctl_smoke_${STAMP}"
[[ $SMOKE == 1 ]] && { SHARDS=1; TIME="01:00:00"; }

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_uncond_control.sh at ${STAMP}.
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
LNR_METADATA=${CONTROL_META}
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
SHARD_COUNT=${SHARDS}
# NATIVE type, exactly as val_generation does it: each example is asked for the chemistry it
# actually has. CYC_TYPES is ignored under --native-type but must stay non-empty for argparse.
NATIVE_TYPE=1
CYC_TYPES="mainchain"
# The de novo corner: both tracks from full noise. Identical to the LNR uncond run.
T_CA_GRID="0.0"
T_LAT_GRID="0.0"
SEEDS="$([[ $SMOKE == 1 ]] && echo "0" || echo "${SEEDS}")"
NSTEPS=$([[ $SMOKE == 1 ]] && echo 50 || echo 0)
PEPTIDE_LIMIT=$([[ $SMOKE == 1 ]] && echo 2 || echo 0)
PDB_DIR=${REPO}/evaluation_results/${RUN_ID}/pdb
EOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

NODE_ARG=(); [[ -n "$NODELIST" ]] && NODE_ARG=(--nodelist="$NODELIST")
ARRAY_ID=$(sbatch --parsable \
  --partition="$PARTITION" "${NODE_ARG[@]}" --time="$TIME" \
  --array=0-$((SHARDS - 1)) --job-name="${RUN_ID}" \
  scripts/sdedit_sweep.sbatch "$ENV_FILE")
echo "submitted GPU array: job ${ARRAY_ID} (${SHARDS} shard(s), ${PARTITION}${NODELIST:+/$NODELIST})"

SUM_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$SUM_TIME" \
  --dependency=afterok:"$ARRAY_ID" --job-name="${RUN_ID}_sum" \
  scripts/sdedit_summarize.sbatch "$ENV_FILE")
echo "submitted CPU summary: job ${SUM_ID} (afterok:${ARRAY_ID})"
echo
echo "results: ${REPO}/evaluation_results/${RUN_ID}"
