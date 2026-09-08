#!/bin/bash
# Submitter for the track-asymmetric SDEdit sweep. Calls sbatch and nothing else.
#
#   bash scripts/submit_sdedit_sweep.sh [--smoke] [--partition P] [--nodelist N]
#                                       [--shards N] [--time T]
#                                       [--metadata PARQUET] [--tag NAME]
#
# RE-RUN ON THE POCKET-CROPPED TARGETS:
#   bash scripts/submit_sdedit_sweep.sh --tag pocket \
#     --metadata CPSea_data/lnr_pocket/metadata/lnr_test_pocket.parquet
#
# The original sweep ran on the full contiguous LNR receptors, which reach the model as ~1
# segment where all CPSea training used a ~14-fragment pocket. On the de novo task that cost
# 22-34 points of ring closure, so every conclusion drawn from the original sweep -- the
# closure/retention frontier, "frozen sequence cannot close", the best (t_ca, t_lat) corner --
# needs re-deriving against the pocket-cropped set. See scripts/restage_lnr_pocket.py.
#
# Writes ONE timestamped env file and passes its PATH as a positional argument (never
# `sbatch --export`, which sets SLURM_GET_USER_ENV=1 and gets jobs requeued and HELD).
#
# Stage 1: GPU job ARRAY, one shard of input peptides each, one results file each.
# Stage 2: CPU summarize, chained afterok on the whole array.
# Stage 3: CPU Rosetta dG before/after, afterok on the array; the figure hangs off it with
#          afterANY, so a slow dG shard hitting the wall clock costs sample size, not the plot.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARTITION="general"; NODELIST=""; EXCLUDE=""; TIME="08:00:00"; SUM_TIME="00:20:00"; SHARDS=6; SMOKE=0
# Grid / seeds. Empty = keep the heredoc defaults below (so old invocations are unchanged).
# Override for a thorough sweep, e.g. --seeds "0 1 2" to enable pass@k. a5000 silently forces
# self_cond=false, so pin a6000 with --exclude gpu28 when running on `general`.
CLI_TCA=""; CLI_TLAT=""; CLI_SEEDS=""; CLI_CYC=""; CLI_GARMS=""
# FastRelax cost varies ~30x across receptors, so a fixed 12 h killed shards on the guided
# run and stranded the figure. The scorer resumes per structure; a longer window is free.
ROS_TIME="1-00:00:00"
TAG=""
# Default is the ORIGINAL staging, kept so existing invocations are unchanged. The
# pocket-cropped set is the one to use from now on -- see --tag usage below.
METADATA="CPSea_data/lnr_staged/metadata/lnr_test.parquet"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --metadata)  METADATA="$2";  shift 2 ;;
    --tag)       TAG="$2";       shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --exclude)   EXCLUDE="$2";   shift 2 ;;
    --t-ca)      CLI_TCA="$2";   shift 2 ;;
    --t-lat)     CLI_TLAT="$2";  shift 2 ;;
    --seeds)     CLI_SEEDS="$2"; shift 2 ;;
    --cyc-types) CLI_CYC="$2";   shift 2 ;;
    --guidance-arms) CLI_GARMS="$2"; shift 2 ;;
    --shards)    SHARDS="$2";    shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    --ros-time)  ROS_TIME="$2";  shift 2 ;;
    --smoke)     SMOKE=1;        shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
[[ "$METADATA" = /* ]] || METADATA="${REPO}/${METADATA}"
[[ -f "$METADATA" ]] || { echo "FATAL: metadata not found: $METADATA" >&2; exit 1; }
SUFFIX="${TAG:+_${TAG}}"
RUN_ID="sdedit${SUFFIX}_${STAMP}"; [[ $SMOKE == 1 ]] && RUN_ID="sdedit${SUFFIX}_smoke_${STAMP}"
[[ $SMOKE == 1 ]] && { SHARDS=1; TIME="01:00:00"; }

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_sdedit_sweep.sh at ${STAMP}.
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
# Peptide-only PDBs, and the receptor-inclusive complexes the Rosetta stage scores. The
# complexes MUST be written here, by the sampler: the CPSea loader's frame is redrawn per
# process, so an edited peptide saved without its receptor can never be put back into it.
PDB_DIR=${REPO}/evaluation_results/${RUN_ID}/pdbs
COMPLEX_DIR=${REPO}/evaluation_results/${RUN_ID}/complexes
ROSETTA_SHARDS=${SHARDS}
ROSETTA_ONLY_SCORABLE=1
# FastRelax is minutes per complex and the full grid is thousands of edits, so the dG arm
# scores a capped, evenly-spread subset per input peptide rather than the whole sweep.
ROSETTA_MAX_PER_EXAMPLE=$([[ $SMOKE == 1 ]] && echo 2 || echo 8)
LNR_METADATA=${METADATA}
# Frozen pin of the bond-unroll arm: the best closure model, and a pin so this run still
# means the same thing next week. Trained against \$CPSEA_AE_CKPT_PATH -- the AE the round-trip
# validated -- so flow and AE match, which is the trap that invalidates cross-run comparisons.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
SHARD_COUNT=${SHARDS}
# Phase B grid. Phase A pinned t_lat at 0.6 for every usable row (its t_lat=1.0 half died
# on the t<1 assert), so the SEQUENCE track has never actually been varied -- and it is the
# binding constraint: at t_lat=0.6 the sampled sequence admitted no disulfide anchor pair in
# 59/59 edits and no isopeptide pair in 51/59, so the head returned a null edge (-1) before
# geometry was ever consulted. Both chemistries were unmeasurable, not failing.
# So t_lat gets the resolution here: 1.0 is the frozen-sequence corner (zero substitutions,
# backbone-only edit -- the cheap control), and lowering it buys anchor placement.
CYC_TYPES="${CLI_CYC:-$([[ $SMOKE == 1 ]] && echo "disulfide" || echo "disulfide isopeptide mainchain")}"
T_CA_GRID="${CLI_TCA:-$([[ $SMOKE == 1 ]] && echo "0.4" || echo "0.2 0.4 0.6 0.8")}"
T_LAT_GRID="${CLI_TLAT:-$([[ $SMOKE == 1 ]] && echo "1.0" || echo "1.0 0.8 0.6 0.4 0.2")}"
SEEDS="${CLI_SEEDS:-0}"
NSTEPS=$([[ $SMOKE == 1 ]] && echo 50 || echo 0)
# Guidance arms (space-separated "w|loss|schedule|pow|k|mode|stride|graft" specs). Unset =
# the single unguided arm the sbatch defaults to. Quoted so the spaces survive `source`.
${CLI_GARMS:+GUIDANCE_ARMS="${CLI_GARMS}"}
PEPTIDE_LIMIT=$([[ $SMOKE == 1 ]] && echo 2 || echo 0)
EOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

NODE_ARG=(); [[ -n "$NODELIST" ]] && NODE_ARG=(--nodelist="$NODELIST")
[[ -n "$EXCLUDE" ]] && NODE_ARG+=(--exclude="$EXCLUDE")
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

ROS_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$ROS_TIME" \
  --array=0-$((SHARDS - 1)) --dependency=afterok:"$ARRAY_ID" --job-name="${RUN_ID}_dg" \
  scripts/sdedit_rosetta.sbatch "$ENV_FILE")
echo "submitted CPU Rosetta dG: job ${ROS_ID} (afterok:${ARRAY_ID}, ${ROS_TIME})"

ROSFIG_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$SUM_TIME" \
  --dependency=afterany:"$ROS_ID" --job-name="${RUN_ID}_dg_fig" \
  scripts/sdedit_rosetta_plot.sbatch "$ENV_FILE")
echo "submitted CPU dG figure:  job ${ROSFIG_ID} (afterany:${ROS_ID})"
echo
echo "results: ${REPO}/evaluation_results/${RUN_ID}"
