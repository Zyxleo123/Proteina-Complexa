#!/usr/bin/env bash
#SBATCH --job-name=cpseq_env_boltz2
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_env_boltz2_%j.out
#
# Build the isolated Boltz-2 environment for the sequence -> CP outcome dataset.
#
# Boltz-2 pins numpy<2.0, einx, dm-tree and its own torch; the complexa venv runs
# numpy>=2 and torch 2.7.0+cu126 with openfold/atomworks on top. Installing boltz
# into it would resolve one of the two away, so boltz gets its own venv and the
# adapter shells out to it. That is the "isolated environment" the task asks for.
#
# The log goes to /home because /data/user_data is not mounted on the login node,
# and Slurm resolves --output before the job body runs (it will not mkdir either).
#
# Idempotent and requeue-safe, which preempt requires:
#   - the venv is only created when absent
#   - pip install is skipped once `boltz` imports at the pinned revision
#   - mols.tar is only extracted when cache/mols is missing
#   - every stage drops a marker under $ENV_ROOT/markers
#
# Submit:  sbatch scripts/cpseq_env_boltz2.sbatch
# Resume:  the identical line

set -euo pipefail

ZFS="${ZFS:-/data/user_data/yixiz}"
ENV_ROOT="${ENV_ROOT:-$ZFS/envs/boltz2}"
VENV="$ENV_ROOT/.venv"
BOLTZ_SRC="${BOLTZ_SRC:-$ZFS/upstream/boltz}"
BOLTZ_CACHE="${BOLTZ_CACHE:-$ZFS/boltz_cache}"
# Pinned upstream revision; recorded in the marker so a rebuilt env is traceable
# back to the source it was built from.
BOLTZ_REV="${BOLTZ_REV:-b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc}"
# cu126 matches the driver the complexa env already runs against on these nodes.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

MARKERS="$ENV_ROOT/markers"
mkdir -p "$MARKERS"

echo "=== cpseq_env_boltz2 ==="
echo "node:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "venv:      $VENV"
echo "boltz src: $BOLTZ_SRC @ $BOLTZ_REV"
echo "cache:     $BOLTZ_CACHE"
nvidia-smi -L 2>/dev/null || echo "(no GPU in this job -- install only, as intended)"

# --------------------------------------------------------------- source revision
[ -d "$BOLTZ_SRC/.git" ] || { echo "FATAL: no boltz clone at $BOLTZ_SRC" >&2; exit 1; }
HAVE_REV="$(git -C "$BOLTZ_SRC" rev-parse HEAD)"
if [ "$HAVE_REV" != "$BOLTZ_REV" ]; then
  echo "FATAL: boltz clone is at $HAVE_REV, expected $BOLTZ_REV." >&2
  echo "       Refusing to build an env whose revision does not match the pin." >&2
  exit 1
fi

# ------------------------------------------------------------------------- venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "[venv] creating $VENV"
  python3.12 -m venv "$VENV"
else
  echo "[venv] present"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip wheel

# ------------------------------------------------------------------------ boltz
# `pip install` (not -e): the clone stays read-only, and the venv carries a built
# copy, so a later `git checkout` in upstream/ cannot silently change what runs.
if "$PY" -c "import boltz" 2>/dev/null && [ -f "$MARKERS/boltz_$BOLTZ_REV" ]; then
  echo "[boltz] already installed at $BOLTZ_REV"
else
  echo "[boltz] installing torch from $TORCH_INDEX"
  "$PY" -m pip install "torch==2.7.0" --index-url "$TORCH_INDEX"
  echo "[boltz] installing boltz from $BOLTZ_SRC"
  "$PY" -m pip install "$BOLTZ_SRC"
  "$PY" -c "import boltz; print('boltz', boltz.__version__ if hasattr(boltz,'__version__') else '(no __version__)')"
  touch "$MARKERS/boltz_$BOLTZ_REV"
fi

# ------------------------------------------------------------------------ cache
# Weights were fetched already (see docs/RESOURCES_COLLECTED.md); only the CCD
# mol tarball still needs unpacking. boltz looks for $BOLTZ_CACHE/mols.
for f in ccd.pkl boltz2_conf.ckpt boltz2_aff.ckpt mols.tar; do
  [ -s "$BOLTZ_CACHE/$f" ] || { echo "FATAL: missing $BOLTZ_CACHE/$f" >&2; exit 1; }
done
if [ -d "$BOLTZ_CACHE/mols" ] && [ -f "$MARKERS/mols_extracted" ]; then
  echo "[cache] mols/ already extracted ($(ls "$BOLTZ_CACHE/mols" | wc -l) entries)"
else
  echo "[cache] extracting mols.tar -> $BOLTZ_CACHE/mols"
  tar -xf "$BOLTZ_CACHE/mols.tar" -C "$BOLTZ_CACHE"
  [ -d "$BOLTZ_CACHE/mols" ] || { echo "FATAL: mols.tar did not produce mols/" >&2; exit 1; }
  touch "$MARKERS/mols_extracted"
fi

# ------------------------------------------------------------------- provenance
export BOLTZ_REV BOLTZ_CACHE
"$PY" - <<'PYEOF' > "$ENV_ROOT/env_provenance.json"
import json, subprocess, sys, os, importlib.metadata as md
def v(p):
    try: return md.version(p)
    except Exception: return None
print(json.dumps({
    "python": sys.version.split()[0],
    "boltz_version": v("boltz"),
    "torch": v("torch"),
    "numpy": v("numpy"),
    "rdkit": v("rdkit"),
    "cuda_available": __import__("torch").cuda.is_available(),
    "boltz_src_rev": os.environ.get("BOLTZ_REV"),
    "boltz_cache": os.environ.get("BOLTZ_CACHE"),
}, indent=2))
PYEOF
cat "$ENV_ROOT/env_provenance.json"

echo "=== cpseq_env_boltz2 OK ==="
