#!/bin/bash
#SBATCH --job-name=cp2lp_ca_probe
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
# Measure the CA track the frozen decoder is actually handed, for ONE arm.
#
# Answers "why is the chain not intact" by decomposing the editor's residual into the part
# that moves the peptide (coherent -- the intended edit) and the part that stretches the
# CA(i)-CA(i+1) virtual bond (incoherent -- the only part that can break the backbone), and
# then re-decoding after projecting the CA track back onto valid spacing. See
# `script_utils/cp2lp_ca_spacing_probe.py` for the reasoning and the readings.
#
# Inference only, one network evaluation per sample, no discriminator and no retained graph,
# so an a6000 is chosen for RAM headroom rather than for compute. Each arm writes its OWN
# summary file -- concurrent O_APPEND to a shared file NUL-corrupts rows on this filesystem.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-onepass_gan}"
_cp2lp_defaults
_cp2lp_banner "ca-probe"

# The node and card are recoverable only from this log: there is no accounting database here.
echo "host     : $(uname -n 2>/dev/null || echo '?')"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi -L unavailable"

PROBE_CKPT_TAG="${PROBE_CKPT_TAG:-g001300}"
PROBE_N_BATCHES="${PROBE_N_BATCHES:-8}"
PROBE_BATCH="${PROBE_BATCH:-8}"
PROBE_OUT_DIR="${PROBE_OUT_DIR:-${TRIPLET_ROOT}/ca_probe}"
PROBE_REFERENCE="${PROBE_REFERENCE:-0}"

RUN_DIR="${STORE}/${RUN_NAME}"
CKPT="${RUN_DIR}/checkpoints/${PROBE_CKPT_TAG}.ckpt"

# Soft gate: a missing milestone means this arm has not reached it yet, which is a fact about
# the schedule and not an error. Exit 0 so nothing downstream is stranded on Dependency.
if [[ ! -f "$CKPT" ]]; then
  echo "GATE: ${ARM} has no ${PROBE_CKPT_TAG}.ckpt under ${RUN_DIR}/checkpoints -- not reached yet."
  ls -1 "${RUN_DIR}/checkpoints" 2>/dev/null | sed 's/^/      have: /' || echo "      (no checkpoints dir)"
  exit 0
fi

# `cp2lp_eval64.parquet` is the fixed evaluation manifest the arms are already scored on, so
# the probe measures the same peptides the comparison table does.
MANIFEST="${PROBE_MANIFEST:-${EVAL_MANIFEST}}"
if [[ "$PROBE_REFERENCE" == "1" ]]; then
  # Real bound LPs, through the identical encode/decode path. This is what makes the
  # generated CA-CA numbers comparable to anything.
  MANIFEST="${PROBE_REF_MANIFEST:-${EVAL_REAL_META}}"
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "GATE: no manifest at ${MANIFEST}."
  exit 0
fi

cd "$REPO"
mkdir -p "$PROBE_OUT_DIR"
TAG="${ARM}__${PROBE_CKPT_TAG}"
[[ "$PROBE_REFERENCE" == "1" ]] && TAG="real_reference"

EXTRA=()
[[ "$PROBE_REFERENCE" == "1" ]] && EXTRA+=(--reference-mode)

set -x
"$PY" script_utils/cp2lp_ca_spacing_probe.py \
  --run-dir "$RUN_DIR" \
  --ckpt "$CKPT" \
  --cp-parquet "$MANIFEST" \
  --out "${PROBE_OUT_DIR}/${TAG}.json" \
  --rows-out "${PROBE_OUT_DIR}/${TAG}_bonds.jsonl" \
  --n-batches "$PROBE_N_BATCHES" \
  --batch-size "$PROBE_BATCH" \
  "${EXTRA[@]}" \
  --device cuda
