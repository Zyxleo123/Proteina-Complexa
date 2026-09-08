#!/bin/bash
# ARM C -- de novo generation on LNR targets that are BOTH pocket-cropped AND relaxed.
#
#   bash scripts/submit_uncond_relaxed.sh [--smoke] [--gpu-partition P] [--gpu-nodelist N]
#                                         [--cpu-partition P] [--shards N] [--seeds "0 1 2 3"]
#                                         [--radius R]
#
# The question
# ------------
# Arm B (pocket crop alone) lifted closure 0.45-0.52 -> 0.75-0.79. A residual remains against
# the in-distribution control, and the competing explanations have been measured and excluded:
#
#   crop radius     tapped out -- segments plateau at 9-11 for EVERY radius 8-20 A
#   peptide length  flat -- Arm B closure is 0.75-0.83 across all length bins
#   forced type     free -- forced ~= native on in-distribution targets, type_sat = 1.000
#
# What is left is structure provenance. The model trained on 2.44M structures that are 100%
# AlphaFold models and 100% energy-relaxed; it has never seen an experimental structure. LNR
# is 100% raw crystal. This arm removes that difference.
#
# Pipeline (split by resource, chained afterok)
# ---------------------------------------------
#   stage 1  CPU array  relax the FULL complexes (AF-style restrained Amber minimization)
#   stage 2  CPU        crop the relaxed complexes to the 18 A pocket
#   stage 3  GPU array  de novo generation, 3 forced types
#   stage 4  CPU        summarize
#
# Relax BEFORE cropping: cropping first would minimize severed chain ends into vacuum.
# Every stage is resumable, and stage 2 gates (exit 0 with a reason) if stage 1 made nothing.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

GPU_PARTITION="general"; GPU_NODELIST=""; CPU_PARTITION="cpu"
RELAX_TIME="04:00:00"; RESTAGE_TIME="00:30:00"; GPU_TIME="08:00:00"; SUM_TIME="00:20:00"
SHARDS=6; RELAX_SHARDS=6; SMOKE=0; SEEDS="0 1 2 3"; RADIUS="18.0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-partition) GPU_PARTITION="$2"; shift 2 ;;
    --gpu-nodelist)  GPU_NODELIST="$2";  shift 2 ;;
    --cpu-partition) CPU_PARTITION="$2"; shift 2 ;;
    --shards)        SHARDS="$2";        shift 2 ;;
    --seeds)         SEEDS="$2";         shift 2 ;;
    --radius)        RADIUS="$2";        shift 2 ;;
    --smoke)         SMOKE=1;            shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

OPENMM_PY="${REPO}/.venv_openmm/bin/python"
[[ -x "$OPENMM_PY" ]] || { echo "FATAL: OpenMM venv missing: $OPENMM_PY"; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="uncondrelax_${STAMP}"; [[ $SMOKE == 1 ]] && RUN_ID="uncondrelax_smoke_${STAMP}"
[[ $SMOKE == 1 ]] && { SHARDS=1; RELAX_SHARDS=1; GPU_TIME="01:00:00"; RELAX_TIME="01:00:00"; }

RELAX_DIR="${REPO}/CPSea_data/lnr_relaxed_${STAMP}"
POCKET_DIR="${REPO}/CPSea_data/lnr_relaxed_pocket_${STAMP}"

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_uncond_relaxed.sh at ${STAMP}.
SRC_METADATA=${REPO}/CPSea_data/lnr_staged/metadata/lnr_test.parquet
RELAX_DIR=${RELAX_DIR}
POCKET_DIR=${POCKET_DIR}
POCKET_RADIUS=${RADIUS}
RELAX_SHARDS=${RELAX_SHARDS}
OPENMM_PYTHON=${OPENMM_PY}
# Written by stage 2; stage 3 reads it. The path is deterministic, so it can be named here
# before the file exists.
LNR_METADATA=${POCKET_DIR}/metadata/lnr_test_pocket.parquet
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
# Same pin/AE as every other arm, so Arm C vs Arm B is single-variable: relaxation.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
SHARD_COUNT=${SHARDS}
CYC_TYPES="$([[ $SMOKE == 1 ]] && echo "disulfide" || echo "disulfide isopeptide mainchain")"
T_CA_GRID="0.0"
T_LAT_GRID="0.0"
SEEDS="$([[ $SMOKE == 1 ]] && echo "0" || echo "${SEEDS}")"
NSTEPS=$([[ $SMOKE == 1 ]] && echo 50 || echo 0)
PEPTIDE_LIMIT=$([[ $SMOKE == 1 ]] && echo 2 || echo 0)
PDB_DIR=${REPO}/evaluation_results/${RUN_ID}/pdb
EOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
GPU_NODE_ARG=(); [[ -n "$GPU_NODELIST" ]] && GPU_NODE_ARG=(--nodelist="$GPU_NODELIST")

RELAX_ID=$(sbatch --parsable \
  --partition="$CPU_PARTITION" --time="$RELAX_TIME" \
  --array=0-$((RELAX_SHARDS - 1)) --job-name="${RUN_ID}_relax" \
  scripts/relax_lnr.sbatch "$ENV_FILE")
echo "stage 1  CPU  relax      job ${RELAX_ID} (${RELAX_SHARDS} shard(s), ${CPU_PARTITION})"

RESTAGE_ID=$(sbatch --parsable \
  --partition="$CPU_PARTITION" --time="$RESTAGE_TIME" \
  --dependency=afterok:"$RELAX_ID" --job-name="${RUN_ID}_restage" \
  scripts/restage_relaxed.sbatch "$ENV_FILE")
echo "stage 2  CPU  pocket crop job ${RESTAGE_ID} (afterok:${RELAX_ID})"

GEN_ID=$(sbatch --parsable \
  --partition="$GPU_PARTITION" "${GPU_NODE_ARG[@]}" --time="$GPU_TIME" \
  --array=0-$((SHARDS - 1)) --dependency=afterok:"$RESTAGE_ID" --job-name="${RUN_ID}" \
  scripts/sdedit_sweep.sbatch "$ENV_FILE")
echo "stage 3  GPU  generation  job ${GEN_ID} (afterok:${RESTAGE_ID}, ${GPU_PARTITION}${GPU_NODELIST:+/$GPU_NODELIST})"

SUM_ID=$(sbatch --parsable \
  --partition="$CPU_PARTITION" --time="$SUM_TIME" \
  --dependency=afterok:"$GEN_ID" --job-name="${RUN_ID}_sum" \
  scripts/sdedit_summarize.sbatch "$ENV_FILE")
echo "stage 4  CPU  summary     job ${SUM_ID} (afterok:${GEN_ID})"
echo
echo "results: ${REPO}/evaluation_results/${RUN_ID}"
echo "NOTE: read the stage-1 .out before trusting the chain -- a gated stage 2 exits 0."
