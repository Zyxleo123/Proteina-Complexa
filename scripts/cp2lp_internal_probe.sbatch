#!/bin/bash
#SBATCH --job-name=cp2lp_int_probe
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Matched-size test of the internal-coordinate edit space against the free Cartesian one.
# Inference only; nothing is trained. See script_utils/cp2lp_internal_edit_probe.py for what
# the three arms are and why the size match is a precondition rather than a detail.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-onepass_gan}"
_cp2lp_defaults
_cp2lp_banner "internal-edit-probe"
echo "host     : $(uname -n 2>/dev/null || echo '?')"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi -L unavailable"

PROBE_CKPT_TAG="${PROBE_CKPT_TAG:-g001300}"
PROBE_OUT_DIR="${PROBE_OUT_DIR:-${TRIPLET_ROOT}/ca_probe}"
TARGET_SIGMA="${TARGET_SIGMA:-0.1705}"
SIGMA_SWEEP="${SIGMA_SWEEP:-}"

RUN_DIR="${STORE}/${RUN_NAME}"
CKPT="${RUN_DIR}/checkpoints/${PROBE_CKPT_TAG}.ckpt"
if [[ ! -f "$CKPT" ]]; then
  echo "GATE: ${ARM} has no ${PROBE_CKPT_TAG}.ckpt under ${RUN_DIR}/checkpoints."
  exit 0
fi
MANIFEST="${PROBE_MANIFEST:-${EVAL_MANIFEST}}"
[[ -f "$MANIFEST" ]] || { echo "GATE: no manifest at ${MANIFEST}."; exit 0; }

cd "$REPO"
mkdir -p "$PROBE_OUT_DIR"
# Two measurements share this job because they share the model load, which dominates the
# runtime. `matched` compares the two edit parameterisations at equal edit size; `extension`
# walks a COORDINATED edit toward the extended corner and reports chain_intact against
# terminal gap, which is the trade-off the trained editor would actually ride -- a random edit
# of the same size barely extends the peptide (0.352 -> 0.544 nm at sigma 0.1705), so the
# matched comparison alone understates the parameterisation.
PROBE_MODE="${PROBE_MODE:-matched}"
if [[ "$PROBE_MODE" == "extension" ]]; then
  set -x
  exec "$PY" script_utils/cp2lp_extension_tradeoff_probe.py \
    --run-dir "$RUN_DIR" \
    --ckpt "$CKPT" \
    --cp-parquet "$MANIFEST" \
    --out "${PROBE_OUT_DIR}/extension_tradeoff.json" \
    --n-batches "${PROBE_N_BATCHES:-8}" \
    --fractions "${FRACTIONS:-0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0}" \
    --device cuda
fi

set -x
"$PY" script_utils/cp2lp_internal_edit_probe.py \
  --run-dir "$RUN_DIR" \
  --ckpt "$CKPT" \
  --cp-parquet "$MANIFEST" \
  --out "${PROBE_OUT_DIR}/internal_edit__${SIGMA_TAG:-sigma${TARGET_SIGMA}}.json" \
  --n-batches "${PROBE_N_BATCHES:-8}" \
  --target-sigma-c "$TARGET_SIGMA" \
  ${SIGMA_SWEEP:+--sigma-sweep "$SIGMA_SWEEP"} \
  --device cuda
