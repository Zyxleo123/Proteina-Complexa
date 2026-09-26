#!/bin/bash
#SBATCH --job-name=lp_ae_roundtrip
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Decision gate before the 2-day shared-AE stage: does the EXISTING CPSea-only AE
# (finetune_full_128) encode LINEAR peptides as well as it encodes cyclic ones?
#
# If it does, both flow arms can pin it and the AE stage is skipped -- saving 2 GPU-days
# on the critical path and making the arms comparable to the whole v4/bondunroll lineage.
# If it does not, reusing it would make "LP mixing does not help" indistinguishable from
# "the AE cannot encode LP", which is the one confound that would waste the whole
# experiment.
#
# Runs script_utils/diagnose_ae_latents.py TWICE under ONE AE checkpoint -- once on CPSea
# metadata, once on LP metadata -- then compares them length-matched.
#
# a6000 on `general`: same card as every recent CPSea run, and the AE is the same 4 GB
# model. Not an a5000 and not a 2080 Ti -- a CUDA allocation failure here kills the
# process with no Python traceback, which reads as a mysterious silent FAILED.
#
# Read-only: loads the AE, never trains, writes nothing outside its own out dir.
#
# Do NOT sbatch directly -- use scripts/submit_lp_mixing.sh --submit ae-roundtrip.
#
#   sbatch lp_ae_roundtrip.sbatch /path/to/env_file [VAR=value ...]

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_lp_mixing_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "${REPO}"
mkdir -p slurm_logs

# Non-interactive bash skips ~/.bashrc; load it so WANDB_API_KEY is available.
if [[ -z "${WANDB_API_KEY:-}" || ${#WANDB_API_KEY} -lt 40 ]] && [[ -f "${HOME}/.bashrc" ]]; then
  set +euo pipefail
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc" || true
  set -euo pipefail
fi

# env.sh BEFORE the env file: env.sh uses `${VAR:-default}`, so sourcing it second would
# not clobber the submission's choices -- but it also exports CPSEA_AE_CKPT_PATH, which
# the env file may legitimately override, and that only works in this order.
# shellcheck source=/dev/null
source "${REPO}/env.sh"
# shellcheck source=_lp_mixing_preamble.sh
source "${REPO}/scripts/_lp_mixing_preamble.sh"
_lp_consume_args "$@"
_lp_defaults
_lp_banner "ae-roundtrip"

PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The ONE AE both passes load. Defaulted by env.sh to the finetune_full_128 step-40000
# EMA -- the checkpoint every recent CPSea flow run regressed its latents against.
AE_CKPT="${RT_AE_CKPT:-${CPSEA_AE_CKPT_PATH}}"
RT_CONFIG="${RT_CONFIG:-example/training_cpsea_bondunroll_from_v4cfg}"
RT_BATCHES="${RT_BATCHES:-40}"
RT_BATCH_SIZE="${RT_BATCH_SIZE:-8}"
RT_WORKERS="${RT_WORKERS:-4}"
LP_META="${LP_META:-${LP_OUT_DIR}/metadata}"
OUT_DIR="${RT_OUT_DIR:-${PROTEINA_ZFS_PATH}/training_runs/diagnostics/lp_ae_roundtrip}"

# A smoke gets its OWN out dir. Both passes write fixed filenames
# (ae_latent_diag_{cpsea,lp}_train.json), so a smoke sharing the dir would either be
# overwritten by the real run or -- worse, if the two overlap -- have the comparator read
# one corpus from the smoke and the other from the full run.
if [[ "${SMOKE}" == "1" ]]; then
  RT_BATCHES=3; RT_BATCH_SIZE=4
  OUT_DIR="${OUT_DIR%/}_smoke"
fi

mkdir -p "${OUT_DIR}"

echo "AE ckpt   : ${AE_CKPT}"
echo "config    : ${RT_CONFIG}"
echo "cpsea meta: ${CPSEA_FULL_META}/cpsea_train.parquet"
echo "lp meta   : ${LP_META}/lp_train.parquet"
echo "out       : ${OUT_DIR}"
echo "batches   : ${RT_BATCHES} x ${RT_BATCH_SIZE}"

# ---- gate. Exit 0, not non-zero: a gate that fails hard strands every dependent job in
# DependencyNeverSatisfied with no explanation. Distinguish "gate failed" from "never ran".
gate_fail=0
[[ -f "${AE_CKPT}" ]] || { echo "GATE: AE checkpoint not found: ${AE_CKPT}"; gate_fail=1; }
[[ -f "${CPSEA_FULL_META}/cpsea_train.parquet" ]] || {
  echo "GATE: missing ${CPSEA_FULL_META}/cpsea_train.parquet"; gate_fail=1; }
[[ -f "${LP_META}/lp_train.parquet" ]] || {
  echo "GATE: missing ${LP_META}/lp_train.parquet -- run --submit data first"; gate_fail=1; }
if [[ ${gate_fail} -ne 0 ]]; then
  echo "GATE: preconditions unmet; exiting 0 so nothing is stranded."
  exit 0
fi

# CPSEA_AE_CKPT_PATH is what the Hydra config resolves for autoencoder_ckpt_path, so
# exporting it is how the AE gets pinned for BOTH passes. Same value twice, deliberately:
# if the two passes loaded different weights the comparison would measure the checkpoints
# rather than the corpora, and the compare script refuses that case.
export CPSEA_AE_CKPT_PATH="${AE_CKPT}"

run_pass() {  # run_pass <tag> <train_metadata>
  local tag="$1" meta="$2"
  echo
  echo "=================== pass: ${tag} ==================="
  "${PY}" script_utils/diagnose_ae_latents.py \
    --config-name "${RT_CONFIG}" \
    --split train \
    --metadata "${meta}" \
    --num-batches "${RT_BATCHES}" \
    --batch-size "${RT_BATCH_SIZE}" \
    --num-workers "${RT_WORKERS}" \
    --out "${OUT_DIR}" \
    --tag "${tag}"
}

run_pass cpsea "${CPSEA_FULL_META}/cpsea_train.parquet"
run_pass lp    "${LP_META}/lp_train.parquet"

echo
"${PY}" script_utils/lp_ae_roundtrip_compare.py \
  --cpsea-json "${OUT_DIR}/ae_latent_diag_cpsea_train.json" \
  --lp-json    "${OUT_DIR}/ae_latent_diag_lp_train.json" \
  --out        "${OUT_DIR}/verdict.json"

echo
echo "done. verdict: ${OUT_DIR}/verdict.json"
