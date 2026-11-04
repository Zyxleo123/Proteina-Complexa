#!/bin/bash
#SBATCH --job-name=ms_potfeats
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/slurm_logs/ms_potfeats_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/slurm_logs/ms_potfeats_%j.out

# Transfer arm stage 1: build the llm_ladder_pot dataset and its ProteinMPNN features.
# This is the compute; the analysis that reads it is a separate job so the verdict stays
# redrawable at zero cost.
#
# Structures are REUSED from structures_v3.json: llm_ladder_pot covers a subset of the
# same 20 targets, and the native complex for a target does not depend on which prediction
# protocol was used to score candidates against it.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
source "${REPO}/scripts/_cpseq_meanscore_preamble.sh"
meanscore_apply_args "$@"
meanscore_banner

if [[ ! -d "${MEANSCORE_POT_SOURCE}" ]]; then
  echo "GATE: ${MEANSCORE_POT_SOURCE} does not exist -- no transfer arm to build."
  echo "MEANSCORE_POTFEATS_SKIPPED $(date -Is)"
  exit 0
fi
if [[ ! -s "${MEANSCORE_STRUCTURES}" ]]; then
  echo "GATE: ${MEANSCORE_STRUCTURES} missing; run the mpnn stage first."
  echo "MEANSCORE_POTFEATS_SKIPPED $(date -Is)"
  exit 0
fi

echo "### 1. dataset from the potentials-ON run"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_dataset.py" \
  --out-dir "${MEANSCORE_POT_SOURCE}" \
  --scoring-version "${MEANSCORE_SCORING_VERSION}" \
  --arms "${MEANSCORE_ARMS}" \
  --min-folds "${MEANSCORE_POT_MIN_FOLDS}" \
  --json "${MEANSCORE_POT_DATASET}"

# Gate: an empty dataset is a real outcome (too few folds per sequence), not a crash.
rows=$("${PYTHON_EXEC}" -c "import json,sys; print(len(json.load(open('${MEANSCORE_POT_DATASET}'))['rows']))" 2>/dev/null || echo 0)
echo "rows in pot dataset: ${rows}"
if [[ "${rows}" -lt 40 ]]; then
  echo "GATE: only ${rows} rows survived the >=${MEANSCORE_POT_MIN_FOLDS}-fold filter."
  echo "      Too few for a per-target transfer test. Exiting 0 so dependents can report."
  echo "MEANSCORE_POTFEATS_SKIPPED $(date -Is)"
  exit 0
fi

echo
echo "### 2. backbone marginals (one pass per target)"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_mpnn_feats.py" \
  --dataset "${MEANSCORE_POT_DATASET}" \
  --structures "${MEANSCORE_STRUCTURES}" \
  --json "${MEANSCORE_POT_MPNN_BB}" \
  --model-name "${MEANSCORE_MPNN_MODEL}" \
  --batch-size "${MEANSCORE_MPNN_BATCH}" \
  --n-draw-marginal "${MEANSCORE_MPNN_DRAW_MARGINAL}" \
  --skip-ar --resume

echo
echo "### 3. autoregressive score (one pass per candidate; 0.46 s/candidate measured)"
"${PYTHON_EXEC}" "${REPO}/script_utils/cpseq_meanscore_mpnn_feats.py" \
  --dataset "${MEANSCORE_POT_DATASET}" \
  --structures "${MEANSCORE_STRUCTURES}" \
  --json "${MEANSCORE_POT_MPNN_AR}" \
  --model-name "${MEANSCORE_MPNN_MODEL}" \
  --batch-size "${MEANSCORE_MPNN_BATCH}" \
  --n-draw-marginal "${MEANSCORE_MPNN_DRAW_MARGINAL}" \
  --n-draw-ar "${MEANSCORE_MPNN_DRAW_AR}" --resume

echo
echo "MEANSCORE_POTFEATS_DONE $(date -Is)"
