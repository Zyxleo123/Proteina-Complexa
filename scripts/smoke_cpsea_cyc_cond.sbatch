#!/bin/bash
#SBATCH --job-name=cpsea_cyc_cond_smoke
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Smoke test for cyclization-type conditioning (20 steps, single GPU).
#
#   sbatch scripts/smoke_cpsea_cyc_cond.sbatch          # conditioned run
#   MODE=control sbatch scripts/smoke_cpsea_cyc_cond.sbatch   # backward-compat run
#
# MODE=cond    (default) type_conditioning=true  -> the new path
# MODE=control          type_conditioning=false -> must reproduce pre-change behavior

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

source env.sh

MODE="${MODE:-cond}"

# The 160M binder trunk has dim_cond=256, so size the type embedding to match its
# siblings (t_emb_dim / idx_emb_dim = 256).
CYC_EMB_DIM=256

# `cyclization_type_emb` MUST be appended LAST. feats_cond_seq order is the column order of
# the concat feeding cond_factory.linear_out, and _splice_pretrained_weights (train.py) warm-
# starts that layer by copying the checkpoint's leading columns and zero-padding the rest.
# Appending last keeps the pretrained time-embedding columns aligned and makes the new type
# embedding contribute exactly zero at init -- i.e. an exact warm start from complexa.ckpt.
# Putting it anywhere else would shift the columns and silently corrupt the spliced weights.
FEATS_COND_SEQ='[time_emb_bb_ca,time_emb_local_latents,cyclization_type_emb]'

# CPSea-finetuned sidechain AE. This now matches env.sh/.env's default (the old
# cpsea_ae_finetune_full_BS32ACCUM4FIX run was deleted); kept pinned explicitly so this smoke
# test cannot drift onto a different AE than the runs it is smoke-testing. Pin the step-40000
# EMA, not last-EMA.ckpt -- the AE defines the local_latents target, so all comparable runs
# must load the same one. (ignore-EMA.ckpt is a legacy throwaway name; do not use it.)
#
# Exported rather than passed as a Hydra override: the filename contains '=' characters,
# which Hydra's override grammar would mis-parse. The config already reads this via
# `autoencoder_ckpt_path: ${oc.env:CPSEA_AE_CKPT_PATH}`, so exporting is both safer and
# the intended path.
AE_CKPT="/zfsauton/scratch/yixiz/Proteina-Complexa/training_runs/store/finetune_full_128/checkpoints/chk_epoch=00000002_step=000000040000-EMA.ckpt"
[[ -f "$AE_CKPT" ]] || { echo "FATAL: AE checkpoint not found: $AE_CKPT"; exit 1; }
export CPSEA_AE_CKPT_PATH="$AE_CKPT"
echo "AE checkpoint: $CPSEA_AE_CKPT_PATH"

COMMON=(
  --config-name example/training_cpsea_peptide_smoke
  +single=true
  +nolog=true
  ++show_prog_bar=true
  ++cyclization.enabled=true
)

if [[ "$MODE" == "control" ]]; then
  echo "=== CONTROL: cyclization head on, type conditioning OFF (backward-compat check) ==="
  ARGS=(
    "${COMMON[@]}"
    ++cyclization.type_conditioning=false
    ++run_name=cpsea_cyc_control_smoke
  )
else
  echo "=== COND: cyclization type conditioning ON ==="
  ARGS=(
    "${COMMON[@]}"
    ++cyclization.type_conditioning=true
    ++training.cyclization_type_dropout_rate=0.15
    ++nn.feats_cond_seq="$FEATS_COND_SEQ"
    ++nn.cyc_emb_dim=$CYC_EMB_DIM
    ++run_name=cpsea_cyc_cond_smoke
  )
fi

# Use the repo venv explicitly. A fresh SLURM shell sources ~/.bashrc, which puts conda's
# python first on PATH -- bare `python` there is NOT the venv and has no proteinfoundation.
# env.sh exports PYTHON_EXEC=<repo>/.venv/bin/python for exactly this reason.
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "Python: $PY"
"$PY" -c "import proteinfoundation" || { echo "FATAL: proteinfoundation not importable from $PY"; exit 1; }

srun "$PY" -m proteinfoundation.train "${ARGS[@]}"
