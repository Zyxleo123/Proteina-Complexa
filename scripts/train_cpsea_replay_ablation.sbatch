#!/bin/bash
#SBATCH --job-name=cpsea_ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Job body for the replay/geometry-loss ablation configs
# (training_cpsea_replay_noloss_from_v4cfg.yaml /
# training_cpsea_geom_noreplay_from_v4cfg.yaml). Do not sbatch this directly --
# use scripts/submit_replay_ablation_train.sh, which writes the env file this
# sources as $1 and fills in --partition/--nodelist/--time/--qos at submit time
# (chosen from `myfree` at submission, not hardcoded here, since availability
# changes run to run).
#
# Mirrors scripts/smoke_cpsea_cyc_cond.sbatch's invocation: srun the repo venv's
# python -m proteinfoundation.train directly, not the external
# ~/slurm/cpsea/cpsea_peptide_train_finetune.sh launcher (opaque to this repo).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"

: "${CONFIG_NAME:?CONFIG_NAME not set in $ENV_FILE}"
: "${RUN_NAME:?RUN_NAME not set in $ENV_FILE}"

source env.sh

PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "Python: $PY"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable from $PY"; exit 1; }

# Unquoted on purpose: EXTRA_OVERRIDES is a space-separated list of Hydra overrides
# that must word-split into separate argv entries. It is appended LAST so it beats both
# the config file and ++run_name. Empty by default, in which case this adds nothing.
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
echo "Launching config=$CONFIG_NAME run_name=$RUN_NAME extra=${EXTRA_OVERRIDES:-<none>}"
# shellcheck disable=SC2086
srun "$PY" -m proteinfoundation.train \
  --config-name "$CONFIG_NAME" \
  "++run_name=$RUN_NAME" \
  $EXTRA_OVERRIDES
