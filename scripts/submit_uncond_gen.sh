#!/bin/bash
# Submitter for UNCONDITIONAL (de novo) cyclic-peptide generation on the SDEdit target set.
#
#   bash scripts/submit_uncond_gen.sh [--smoke] [--partition P] [--nodelist N]
#                                     [--shards N] [--time T] [--seeds "0 1 2 3"]
#                                     [--metadata PARQUET] [--tag NAME]
#
# ARM B usage (pocket-cropped receptors -- see scripts/restage_lnr_pocket.py):
#   bash scripts/submit_uncond_gen.sh --metadata CPSea_data/lnr_pocket/metadata/lnr_test_pocket.parquet \
#                                     --tag pocket
# Everything else is held fixed, so Arm B vs the original run is single-variable: the ONLY
# difference is which receptor residues the loader is handed.
#
# This is the de novo baseline for the SDEdit editing experiment. It reuses the SDEdit
# machinery (scripts/sdedit_cyclize.py + the two generic .sbatch arms), driven entirely
# through the env file, at the ONE grid corner that means "generate from scratch":
#
#     t_ca_start = 0.0   backbone starts from full noise (pose fully regenerated)
#     t_lat_start = 0.0  latent starts from full noise (sequence fully regenerated)
#
# So the encoded input peptide is discarded by interpolate(t=0) -- the only thing kept from
# each target is the receptor conditioning (same targets) and the peptide LENGTH (the mask).
# Each of the 3 cyclization types is REQUESTED via cyclization_type_cond, and closure is
# measured per requested type. The preservation columns (ca_rmsd_to_input, contact_retention,
# n_substitutions) are meaningless here by construction -- read cyc/*_bond_success only.
#
# Same pin, AE, and target set as scripts/submit_sdedit_sweep.sh so the two are comparable.
# Writes ONE timestamped env file and passes its PATH positionally (never `sbatch --export`).

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"; NODELIST=""; TIME="08:00:00"; SUM_TIME="00:20:00"; SHARDS=6; SMOKE=0
SEEDS="0 1 2 3"; TAG=""
METADATA="CPSea_data/lnr_staged/metadata/lnr_test.parquet"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --shards)    SHARDS="$2";    shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    --seeds)     SEEDS="$2";     shift 2 ;;
    --metadata)  METADATA="$2";  shift 2 ;;
    --tag)       TAG="$2";       shift 2 ;;
    --smoke)     SMOKE=1;        shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
# Resolve to an absolute path: the batch script runs from the spool dir, not the repo.
[[ "$METADATA" = /* ]] || METADATA="${REPO}/${METADATA}"
[[ -f "$METADATA" ]] || { echo "FATAL: metadata not found: $METADATA" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
SUFFIX="${TAG:+_${TAG}}"
RUN_ID="uncondgen${SUFFIX}_${STAMP}"; [[ $SMOKE == 1 ]] && RUN_ID="uncondgen${SUFFIX}_smoke_${STAMP}"
[[ $SMOKE == 1 ]] && { SHARDS=1; TIME="01:00:00"; }

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_uncond_gen.sh at ${STAMP}.
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
LNR_METADATA=${METADATA}
# Same frozen bond-unroll pin and AE as the SDEdit sweep so de novo vs edit are comparable.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
SHARD_COUNT=${SHARDS}
# The de novo corner: both tracks start from full noise. Not a grid -- a single point.
CYC_TYPES="$([[ $SMOKE == 1 ]] && echo "disulfide" || echo "disulfide isopeptide mainchain")"
T_CA_GRID="0.0"
T_LAT_GRID="0.0"
SEEDS="$([[ $SMOKE == 1 ]] && echo "0" || echo "${SEEDS}")"
NSTEPS=$([[ $SMOKE == 1 ]] && echo 50 || echo 0)
PEPTIDE_LIMIT=$([[ $SMOKE == 1 ]] && echo 2 || echo 0)
# Dump every generated peptide as a PDB so closures can be re-scored (Rosetta dG) later.
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