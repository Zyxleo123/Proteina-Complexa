#!/bin/bash
#SBATCH --job-name=lpgen_preflight
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_preflight_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_preflight_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# GPU preflight for the RFD3 LP arm: a few real optimizer steps from fresh weights.
#
# The plan's step 4. Needs no released checkpoint -- `ckpt_path: null` still exercises the
# model build, the data path, the loss, the backward pass and the optimizer, and it is the
# run that produces the peak-memory and step-throughput numbers used to size a tranche.
#
# Resource notes (measured, see docs/README_LPGEN.md):
#   * `precision: bf16-mixed` runs on every GPU class here, but bf16 is EMULATED below
#     Ampere -- functional and faster than fp32, yet ~an order of magnitude off native. Use
#     a5000/a6000/h200 for anything whose timing you intend to extrapolate.
#   * gpu14 is a broken node: CUDA init fails there for every torch build while its
#     siblings pass, and it still advertises as AVAILABLE. The submitter excludes it.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "preflight"

if ! _lpgen_require_manifest; then
  echo "[preflight] GATE not satisfied -- exiting 0 so dependents are not stranded"
  exit 0
fi

ARM_TREE="${UPSTREAM_DIR}/foundry"
ARM_PY="${ENVS_DIR}/foundry/bin/python"
for p in "$ARM_TREE/models/rfd3/src/rfd3/train.py" "$ARM_PY"; do
  [[ -e "$p" ]] || { echo "[preflight] GATE missing: $p (run --submit stage / --submit env)" >&2; exit 0; }
done

RUN_DIR="${PREFLIGHT_DIR:-${REPO}/evaluation_results/${RUN_ID}_rfd3_preflight}"
mkdir -p "$RUN_DIR"

# Our condition class and our Hydra configs both live in THIS repo, not in the upstream
# tree. PYTHONPATH makes the class importable; hydra.searchpath makes the configs visible.
export PYTHONPATH="${REPO}/src"
echo "[preflight] asserting our module resolves from the arm env"
"$ARM_PY" -c "
import proteinfoundation.lpgen.rfd3_conditions as m
print('  resolved:', m.__file__)
assert m.__file__.startswith('${REPO}/src'), 'resolved outside the expected tree: ' + m.__file__
assert m.LPBinderCondition.name == 'ppi', 'condition name must stay ppi'
print('  LPBinderCondition.name =', m.LPBinderCondition.name)
"

# Sample GPU memory externally: the number we want is the peak, and an external sampler
# reports it even if the run dies partway, which is when it matters most.
MEMLOG="${RUN_DIR}/gpu_mem.csv"
( while true; do
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits >> "$MEMLOG" 2>/dev/null || true
    sleep 5
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# Fragmentation is a real factor here: the first OOM reported 4.70 GiB "reserved by
# PyTorch but unallocated" against 1.18 GiB free, and torch's own message recommends this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[preflight] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

cd "$ARM_TREE"
echo "[preflight] launching: experiment=${PREFLIGHT_EXPERIMENT}"
set +e
"$ARM_PY" models/rfd3/src/rfd3/train.py \
  "experiment=${PREFLIGHT_EXPERIMENT}" \
  "hydra.searchpath=[pkg://rfd3.configs,pkg://configs,file://${REPO}/configs/rfd3_lp]" \
  "paths.output_dir=${RUN_DIR}" \
  ${EXTRA_PREFLIGHT_ARGS:-} 2>&1 | tee "${RUN_DIR}/train.log"
RC=${PIPESTATUS[0]}
set -e
kill $SAMPLER 2>/dev/null || true

echo "=================================================================="
echo "[preflight] train.py exit code: ${RC}"
if [[ -s "$MEMLOG" ]]; then
  "$PY" - "$MEMLOG" <<'PYEOF'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 3]
if rows:
    used = [int(r[0]) for r in rows]
    total = int(rows[0][1])
    util = [int(r[2]) for r in rows]
    print(f"[preflight] GPU memory: peak {max(used)} MiB of {total} MiB "
          f"({max(used)/total:.0%}), {len(rows)} samples")
    print(f"[preflight] GPU util  : peak {max(util)}%, mean {sum(util)/len(util):.0f}%")
PYEOF
fi
STAGE_LABEL=preflight
LOGF="${RUN_DIR}/train.log"

# WAS IT REALLY A FINE-TUNE? foundry's weight loader has `default_policy: COPY` with
# `fallback_policy: REINIT`, and its docstring is explicit: "By default, all model weights
# are re-initialized; we overwrite with the checkpoint weights where appropriate". So a
# checkpoint whose parameter names do not match the model loads *partially*, every miss is
# silently re-initialized, and the run looks like a fine-tune while being closer to
# scratch. Each miss does log one warning, so count them.
if grep -q 'Loading checkpoint from' "$LOGF" 2>/dev/null; then
  N_REINIT=$(grep -c "Falling back to policy: 'WeightLoadingPolicy.REINIT'" "$LOGF" 2>/dev/null; true)
  N_SHAPE=$(grep -c 'Shape mismatch for' "$LOGF" 2>/dev/null; true)
  N_MISSING=$(grep -c 'not found in checkpoint' "$LOGF" 2>/dev/null; true)
  echo "[${STAGE_LABEL}] weight load: ${N_REINIT} parameter(s) fell back to REINIT "
  echo "[${STAGE_LABEL}]              (${N_MISSING} absent from ckpt, ${N_SHAPE} shape mismatch)"
  if [[ "$N_REINIT" -gt 0 ]]; then
    echo "[${STAGE_LABEL}] WARNING: those parameters were NOT initialized from the checkpoint."
    echo "[${STAGE_LABEL}] A large count means this is not really a fine-tune. First few:"
    grep "Falling back to policy: 'WeightLoadingPolicy.REINIT'" "$LOGF" | head -5 | sed 's/^/    /'
  else
    echo "[${STAGE_LABEL}] all parameters came from the checkpoint -- a genuine fine-tune"
  fi
fi

echo "[preflight] checkpoints written:"
find "$RUN_DIR" -name '*.ckpt' -o -name '*.pt' 2>/dev/null | head -10 || true
echo "[preflight] run dir: $RUN_DIR"
exit "$RC"
