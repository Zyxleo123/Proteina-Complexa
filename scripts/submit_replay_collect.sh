#!/bin/bash
# Submitter for reward-weighted-replay rollout collection. Only calls sbatch --
# all real work happens in scripts/collect_replay_rollouts.sbatch.
#
# Writes one timestamped env file per submission (per this repo's Slurm
# convention: never `sbatch --export`, which forces SLURM_GET_USER_ENV=1 and can
# leave the job held with "user env retrieval failed") and passes its path as a
# positional argument to the batch script.
#
# Usage:
#   CKPT=/path/to/last-EMA.ckpt OUT_DIR=$ZFS/CPSea/replay_buffers/run1 \
#     GATE_CHIRALITY=true \
#     bash scripts/submit_replay_collect.sh
#
# Configurable via environment variable (defaults shown):
#   CKPT           (required)
#   OUT_DIR        (required)
#   CONFIG_NAME    example/training_cpsea_peptide_cyc_typecond
#   K              10
#   N_BATCHES      50
#   NSTEPS         200
#   SEED           0
#   MAX_SIZE       10000
#   GATE_CHIRALITY false   (--gate-chirality if true)
#   GATE_ANGLE     false   (--gate-angle if true)
#   GATE_DIHEDRAL  false   (--gate-dihedral if true)
#   GATE_CLASH     false   (--gate-clash if true)
#   GEN_CHUNK      2       candidates per generate() call (24GB-safe default)
#   TIME           04:00:00  Slurm timelimit (overrides #SBATCH --time). 1 day: 1-00:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${CKPT:?Set CKPT to the frozen checkpoint to collect from}"
: "${OUT_DIR:?Set OUT_DIR to the replay buffer output directory}"

CONFIG_NAME="${CONFIG_NAME:-example/training_cpsea_peptide_cyc_typecond}"
K="${K:-10}"
N_BATCHES="${N_BATCHES:-50}"
NSTEPS="${NSTEPS:-200}"
SEED="${SEED:-0}"
MAX_SIZE="${MAX_SIZE:-10000}"
GEN_CHUNK="${GEN_CHUNK:-2}"
TIME="${TIME:-04:00:00}"

GATE_FLAGS=""
[[ "${GATE_CHIRALITY:-false}" == "true" ]] && GATE_FLAGS="$GATE_FLAGS --gate-chirality"
[[ "${GATE_ANGLE:-false}" == "true" ]] && GATE_FLAGS="$GATE_FLAGS --gate-angle"
[[ "${GATE_DIHEDRAL:-false}" == "true" ]] && GATE_FLAGS="$GATE_FLAGS --gate-dihedral"
[[ "${GATE_CLASH:-false}" == "true" ]] && GATE_FLAGS="$GATE_FLAGS --gate-clash"

mkdir -p slurm_logs .slurm_envs
ENV_FILE=".slurm_envs/replay_collect_$(date +%Y%m%d_%H%M%S)_$$.env"
# Quote GATE_FLAGS: an unquoted leading space (e.g. GATE_FLAGS= --gate-chirality)
# is parsed by `source` as an empty assignment plus a command named --gate-chirality
# (exit 127). Same for any value that needs to survive whitespace.
cat > "$ENV_FILE" <<EOF
CKPT=$CKPT
OUT_DIR=$OUT_DIR
CONFIG_NAME=$CONFIG_NAME
K=$K
N_BATCHES=$N_BATCHES
NSTEPS=$NSTEPS
SEED=$SEED
MAX_SIZE=$MAX_SIZE
GEN_CHUNK=$GEN_CHUNK
GATE_FLAGS="$GATE_FLAGS"
EOF

echo "Wrote $ENV_FILE"
echo "Submitting scripts/collect_replay_rollouts.sbatch $ENV_FILE (time=$TIME)"
# CLI --time overrides the #SBATCH --time directive in the batch script.
sbatch --time="$TIME" scripts/collect_replay_rollouts.sbatch "$ENV_FILE"
