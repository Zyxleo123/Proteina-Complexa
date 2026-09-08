#!/bin/bash
# Submitter for the DDIB inversion probe. Calls sbatch and nothing else.
#
#   bash scripts/submit_ddib_invert.sh [--smoke] [--partition P] [--nodelist N]
#                                      [--shards N] [--time T]
#
# Stage 1: GPU job ARRAY -- reverse-integrate the PUBLIC Proteina-Complexa flow from each
#          LNR linear peptide, recording terminal gap / pose cost / interface at a grid of t.
# Stage 2: CPU summary -- table + figure, from the saved rows only.
#
# One timestamped env file, passed as a POSITIONAL argument (never `sbatch --export`, which
# sets SLURM_GET_USER_ENV=1 and gets jobs requeued and HELD).

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"; NODELIST=""; TIME="06:00:00"; SUM_TIME="00:20:00"; SHARDS=4; SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --shards)    SHARDS="$2";    shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    --smoke)     SMOKE=1;        shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ddib_invert_${STAMP}"; [[ $SMOKE == 1 ]] && RUN_ID="ddib_invert_smoke_${STAMP}"
[[ $SMOKE == 1 ]] && { SHARDS=1; TIME="01:00:00"; }

CKPTS="/zfsauton/scratch/yixiz/Proteina-Complexa/ckpts"

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF2
# Auto-written by scripts/submit_ddib_invert.sh at ${STAMP}.
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
LNR_METADATA=${REPO}/CPSea_data/lnr_staged/metadata/lnr_test.parquet
# The SOURCE model of the translation: the stock public flow, trained on ordinary bound
# peptides, paired with ITS OWN autoencoder. The CPSea AE was finetuned from this one, so
# the two are different latent spaces -- mixing them would silently reinterpret the
# local_latents track. The job exports CPSEA_AE_CKPT_PATH from SRC_AE_CKPT for this reason.
SRC_FLOW_CKPT=${CKPTS}/complexa.ckpt
SRC_AE_CKPT=${CKPTS}/complexa_ae.ckpt
SHARD_COUNT=${SHARDS}
# Two arms. bb_ca alone is the transferable one: CA coordinates are literally the same
# space in both models, so that inverted state can be handed to the cyclic model as-is.
# Adding local_latents is a fuller inversion but lands in the public AE's latent space,
# which the cyclic model does not share -- run to price what the restriction costs.
TRACK_ARMS="$([[ $SMOKE == 1 ]] && echo "bb_ca" || echo "bb_ca,bb_ca local_latents")"
RECORD_TS="$([[ $SMOKE == 1 ]] && echo "0.8 0.5 0.2" || echo "0.95 0.9 0.85 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05")"
# Invertibility check: re-integrate forward to t=1 and compare to the input. Short list --
# each entry costs a second pass. If these are not small, nothing else in the run means
# anything: the "latent" would not be an encoding of this peptide at all.
ROUNDTRIP_TS="$([[ $SMOKE == 1 ]] && echo "0.5" || echo "0.8 0.5 0.2")"
SEEDS="0"
NSTEPS=$([[ $SMOKE == 1 ]] && echo 50 || echo 400)
PEPTIDE_LIMIT=$([[ $SMOKE == 1 ]] && echo 2 || echo 0)
EOF2
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

NODE_ARG=(); [[ -n "$NODELIST" ]] && NODE_ARG=(--nodelist="$NODELIST")
ARRAY_ID=$(sbatch --parsable \
  --partition="$PARTITION" "${NODE_ARG[@]}" --time="$TIME" \
  --array=0-$((SHARDS - 1)) --job-name="${RUN_ID}" \
  scripts/ddib_invert.sbatch "$ENV_FILE")
echo "submitted GPU array: job ${ARRAY_ID} (${SHARDS} shard(s), ${PARTITION}${NODELIST:+/$NODELIST})"

SUM_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$SUM_TIME" \
  --dependency=afterok:"$ARRAY_ID" --job-name="${RUN_ID}_sum" \
  scripts/ddib_invert_summarize.sbatch "$ENV_FILE")
echo "submitted CPU summary: job ${SUM_ID} (afterok:${ARRAY_ID})"
echo
echo "results: ${REPO}/evaluation_results/${RUN_ID}"
