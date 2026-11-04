#!/usr/bin/env bash
#SBATCH --job-name=cpseq_kern_install
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/kernels/logs/install_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/kernels/logs/install_%j.out
#
# Stage 1 of the Boltz-2 kernel check: install the cuequivariance triangle kernels into
# a SHADOW directory, leaving the production boltz env byte-identical.
#
# Why this exists. `adapters/boltz2.py` hardcodes `--no_kernels`, and its comment says
# the kernels are impossible here: cuequivariance_ops_torch_cu12 0.13.0 requires
# torch>=2.11 and pulls CUDA 13, which would replace this env's pinned torch
# 2.7.0+cu126. That is true OF 0.13.0. There are 18 older releases and the torch floor
# appears only in the last three -- checked on PyPI 2026-11-03:
#
#     0.13.0, 0.12.0, 0.11.1   torch>=2.11
#     0.10.0 and earlier       torch>=2.4, and only as a `test` extra
#
# At 0.10.0 the real dependency set is cuequivariance (pure python: networkx, numpy,
# opt-einsum, scipy, sympy), cuequivariance-ops-cu12 and scipy. No torch at all, so a
# normal install CANNOT move it. The cp312 x86_64 wheel matches this env's Python
# 3.12.13. So the conclusion in that comment was generalised from one version, and this
# job finds out.
#
# WHY --target AND NOT A COPIED ENV. The first attempt copied the 4.4 GB env and died
# mid-copy (job 95767, exit 1 with no stderr, 3.9 GB written) -- scratch was near its
# limit. A shadow directory is ~200 MB instead of 4.4 GB, and it is the idiom this
# project already uses for exactly this purpose: cpseq_fold_cost_probe.py shadows its
# patched boltz copy in via PYTHONPATH and never writes to the installed package,
# "because production tasks are reading it". Same reasoning, same mechanism.
#
# The production env is therefore never written to AT ALL -- not even copied. Stage 2
# runs the production python with PYTHONPATH pointing here, so the kernels are present
# for that one process and for nothing else.
#
# CPU only: a pip install needs no GPU, and the kernels cannot be exercised without one
# anyway. That is stage 2, cpseq_kernels_ab.sbatch, which pins an a6000.
#
# This job is a GATE and exits 0 whatever happens, writing its verdict to a status file
# stage 2 reads. A non-zero exit would strand the A/B in DependencyNeverSatisfied with
# nothing explaining why, and "the install failed" is a result worth having.

set -euo pipefail

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
# The PRODUCTION env, read-only throughout. Note the .venv: the boltz env lives at
# $ZFS/envs/boltz2/.venv, which is the path every job script passes as --boltz-python's
# parent, and pointing at $ZFS/envs/boltz2 finds no bin/python at all (job 95749).
BOLTZ_ENV="${BOLTZ_ENV:-$ZFS/envs/boltz2/.venv}"
SHADOW="${SHADOW:-$ZFS/cpseq/kernels/site}"
CUEQ_VERSION="${CUEQ_VERSION:-0.10.0}"
STATUS="${STATUS:-$ZFS/cpseq/kernels/install_status.json}"

mkdir -p "$(dirname "$STATUS")"

PYBIN="$BOLTZ_ENV/bin/python"

# One writer, one heredoc. The repo venv's python is used rather than the boltz env's,
# because this must still be able to record a verdict when the boltz python is the thing
# that is broken.
STATUS_PY="${STATUS_PY:-/zfsauton2/home/yixiz/Proteina-Complexa/.venv/bin/python}"
[ -x "$STATUS_PY" ] || STATUS_PY=python3

write_status() {
  "$STATUS_PY" - "$STATUS" "$1" "$2" "${SLURM_JOB_ID:-local}" "$SHADOW" <<'EOF'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "ok": sys.argv[2] == "ok", "stage": "install", "detail": sys.argv[3],
    "job_id": sys.argv[4], "shadow": sys.argv[5],
}, indent=2) + "\n")
EOF
  echo "[status] $1: $2"
}

# Written FIRST, as a failure. A status file is read by stage 2, and the previous run
# left a stale `ok:false` from a different job that this run's own banner then appeared
# to contradict -- so until this job proves otherwise, the recorded verdict is its own
# and says so. Every exit path below overwrites it.
write_status failed "job ${SLURM_JOB_ID:-local} started and has not yet reached a verdict"

echo "=== cpseq_kernels_install ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-local}"
echo "request:   cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M gpu=none"
echo "boltz env: $BOLTZ_ENV   (READ-ONLY, not copied)"
echo "shadow:    $SHADOW"
echo "cueq:      $CUEQ_VERSION"
echo "status:    $STATUS"
echo "space:     $(df -h "$ZFS" | tail -1)"
echo "============================="

if [ ! -x "$PYBIN" ]; then
  write_status failed "no boltz python at $PYBIN"
  exit 0
fi

TORCH_BEFORE=$("$PYBIN" -c 'import torch; print(torch.__version__)')
PYVER=$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "boltz torch:  $TORCH_BEFORE"
echo "boltz python: $PYVER"

# --- the install, into the shadow dir -------------------------------------------------
# --no-deps, with the one genuinely missing dependency named explicitly.
#
# Not laziness -- PYTHONPATH SHADOWING IS THE RISK HERE. Anything in $SHADOW takes
# precedence over the production env for stage 2's process, so letting pip resolve
# `cuequivariance`'s declared deps (networkx, numpy, opt-einsum, scipy, sympy) would
# drop a second numpy and scipy in front of boltz's own. A different numpy ahead on the
# path can change boltz's behaviour with no kernel involved at all, which would corrupt
# the very comparison this is for.
#
# Measured in the boltz env (2026-11-03): networkx 3.7, sympy 1.14.0, numpy 1.26.4 and
# scipy 1.13.1 are all PRESENT. Only opt_einsum is missing -- that is what killed job
# 95816's import check. So the shadow holds cuequivariance, its ops, and opt_einsum, and
# shadows nothing the production env already provides.
#
# `--no-deps` also keeps torch out: 0.10.0 declares it only as a `test` extra, but a
# resolver given latitude could still fetch 2.6 GB of wheel onto a near-full scratch.
mkdir -p "$SHADOW"
set +e
"$PYBIN" -m pip install --upgrade --target "$SHADOW" --no-deps \
  "cuequivariance-torch==$CUEQ_VERSION" \
  "cuequivariance==$CUEQ_VERSION" \
  "cuequivariance-ops-torch-cu12==$CUEQ_VERSION" \
  "cuequivariance-ops-cu12==$CUEQ_VERSION" \
  "opt-einsum" 2>&1 | tail -20
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then
  write_status failed "pip install exited $RC (see log above)"
  exit 0
fi
echo "shadow size: $(du -sh "$SHADOW" | cut -f1)"

# --- did the production env change? ---------------------------------------------------
# It must not have. --target writes only to $SHADOW, so this is a check that the claim
# in this header is true rather than merely intended.
TORCH_AFTER=$("$PYBIN" -c 'import torch; print(torch.__version__)' 2>&1) || {
  write_status failed "boltz env cannot import torch after install: $TORCH_AFTER"
  exit 0
}
echo "boltz torch after: $TORCH_AFTER"
if [ "$TORCH_AFTER" != "$TORCH_BEFORE" ]; then
  write_status failed "production torch moved $TORCH_BEFORE -> $TORCH_AFTER"
  exit 0
fi
if "$PYBIN" -c 'import cuequivariance_torch' 2>/dev/null; then
  write_status failed \
    "cuequivariance is importable from the PRODUCTION env without PYTHONPATH -- the \
install leaked out of the shadow dir and production folds would silently change"
  exit 0
fi
echo "production env clean: cuequivariance NOT importable without PYTHONPATH (correct)"

# --- can it import through the shadow, and does boltz still load? ---------------------
# The first real question: 0.10.0 may not expose the interface boltz 2.2.1 calls, and
# API drift across 0.10 -> 0.13 is plausible. That shows up here as an ImportError
# rather than silently at fold time.
#
# No GPU on this node, so an import that REQUIRES a device fails here and is still worth
# trying on stage 2's a6000. That case is recorded as ok with a note, not as a failure,
# so the A/B still runs and gets the last word.
set +e
IMP=$(PYTHONPATH="$SHADOW" "$PYBIN" -c '
import cuequivariance_torch as cet
print("cuequivariance_torch", getattr(cet, "__version__", "?"))
import cuequivariance_ops_torch
print("cuequivariance_ops_torch ok")
import boltz, torch
print("boltz", boltz.__version__, "torch", torch.__version__)
' 2>&1)
IRC=$?
set -e
echo "--- import check (PYTHONPATH=$SHADOW) ---"
echo "$IMP" | tail -12

if [ "$IRC" -ne 0 ]; then
  if echo "$IMP" | grep -qiE "cuda|no kernel image|device|driver|nvml|libcu"; then
    write_status ok "installed; import needs a GPU on this CPU node, stage 2 decides: $(echo "$IMP" | tail -1)"
  else
    write_status failed "import failed for a non-GPU reason: $(echo "$IMP" | tail -1)"
  fi
  exit 0
fi

write_status ok "installed into shadow and imports: $(echo "$IMP" | tr '\n' ' ')"
echo
echo "=== cpseq_kernels_install OK ==="
echo "shadow: $SHADOW   (production $BOLTZ_ENV untouched)"
echo "stage 2: sbatch scripts/cpseq_kernels_ab.sbatch"
