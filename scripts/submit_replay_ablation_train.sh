#!/bin/bash
# Submitter for the replay/geometry-loss ablation training configs. Only calls
# sbatch -- all real work happens in scripts/train_cpsea_replay_ablation.sbatch.
#
# Partition/QOS/node are NOT hardcoded: checked via `myfree` at submission time
# (state changes constantly) rather than baked into the .sbatch file. As of the
# last check, the only free a6000 nodes (a5000 must be avoided -- it silently
# forces self_cond=false, see project memory) were gpu1 and gpu27, both only on
# debug/preempt (general's a6000 nodes were all full). debug caps at 12h, too
# short for a multi-day run, so this defaults to preempt + gpu1. RE-RUN `myfree`
# yourself before submitting -- availability shifts by the minute; override
# NODELIST/PARTITION below if gpu1 is no longer free.
#
# Usage:
#   CONFIG_NAME=example/training_cpsea_replay_noloss_from_v4cfg \
#     RUN_NAME=cpsea_replay_noloss_from_v4cfg \
#     bash scripts/submit_replay_ablation_train.sh
#
#   CONFIG_NAME=example/training_cpsea_geom_noreplay_from_v4cfg \
#     RUN_NAME=cpsea_geom_noreplay_from_v4cfg \
#     bash scripts/submit_replay_ablation_train.sh
#
# Configurable via environment variable (defaults shown):
#   CONFIG_NAME  (required)
#   RUN_NAME     (required)
#   PARTITION    preempt
#   QOS          qos_preempt
#   NODELIST     gpu1        (gpu27 is the other currently-free a6000 fallback)
#   TIME         7-00:00:00  (preempt's own max; this is a multi-day run)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${CONFIG_NAME:?Set CONFIG_NAME, e.g. example/training_cpsea_replay_noloss_from_v4cfg}"
: "${RUN_NAME:?Set RUN_NAME}"

PARTITION="${PARTITION:-preempt}"
QOS="${QOS:-qos_preempt}"
NODELIST="${NODELIST:-gpu1}"
TIME="${TIME:-7-00:00:00}"

echo "Re-checking node availability with myfree before submitting..."
myfree || echo "(myfree not available in this shell -- proceeding with configured NODELIST=$NODELIST anyway)"

mkdir -p slurm_logs .slurm_envs
ENV_FILE=".slurm_envs/train_ablation_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENV_FILE" <<EOF
CONFIG_NAME=$CONFIG_NAME
RUN_NAME=$RUN_NAME
EOF

echo "Wrote $ENV_FILE"
echo "Submitting scripts/train_cpsea_replay_ablation.sbatch $ENV_FILE"
echo "  partition=$PARTITION qos=$QOS nodelist=$NODELIST time=$TIME"
sbatch \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --nodelist="$NODELIST" \
  --time="$TIME" \
  scripts/train_cpsea_replay_ablation.sbatch "$ENV_FILE"
