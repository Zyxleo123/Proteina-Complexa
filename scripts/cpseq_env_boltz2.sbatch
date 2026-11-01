#!/usr/bin/env bash
#SBATCH --job-name=cpseq_env_boltz2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_env_boltz2_%j.out
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

# Config by positional arg: a path to source, or VAR=value pairs. Never `sbatch
# --export` -- on this cluster any explicit export list makes sbatch set
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment, and the job
# is requeued and held.
for arg in "$@"; do
  if [ -f "$arg" ]; then echo "[cfg] sourcing $arg"; # shellcheck disable=SC1090
    . "$arg"
  elif [[ "$arg" == *=* ]]; then echo "[cfg] export $arg"; export "$arg"
  fi
done

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ENV_ROOT="${ENV_ROOT:-$ZFS/envs/boltz2}"
VENV="$ENV_ROOT/.venv"
BOLTZ_SRC="${BOLTZ_SRC:-$ZFS/upstream/boltz}"
BOLTZ_CACHE="${BOLTZ_CACHE:-$ZFS/boltz_cache}"
# Pinned upstream revision; recorded in the marker so a rebuilt env is traceable
# back to the source it was built from.
BOLTZ_REV="${BOLTZ_REV:-b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc}"
BOLTZ_REPO="${BOLTZ_REPO:-https://github.com/jwohlwend/boltz}"
BOLTZ_HF_REPO="${BOLTZ_HF_REPO:-boltz-community/boltz-2}"
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
# Clone if absent. On babel this script could assume the clone, because the clone had
# been made by hand before the env was first built; on a fresh cluster that assumption
# is just a FATAL on the first run, so the script now makes it itself.
if [ ! -d "$BOLTZ_SRC/.git" ]; then
  echo "[boltz] cloning $BOLTZ_REPO -> $BOLTZ_SRC"
  mkdir -p "$(dirname "$BOLTZ_SRC")"
  git clone --quiet "$BOLTZ_REPO" "$BOLTZ_SRC"
  git -C "$BOLTZ_SRC" checkout --quiet --detach "$BOLTZ_REV"
fi
HAVE_REV="$(git -C "$BOLTZ_SRC" rev-parse HEAD)"
if [ "$HAVE_REV" != "$BOLTZ_REV" ]; then
  echo "FATAL: boltz clone is at $HAVE_REV, expected $BOLTZ_REV." >&2
  echo "       Refusing to build an env whose revision does not match the pin." >&2
  exit 1
fi

# ------------------------------------------------------------------------- venv
if [ ! -x "$VENV/bin/python" ]; then
  # `python3.12` is on the login node's PATH but NOT on the compute nodes', so a bare
  # `python3.12 -m venv` fails here with "command not found" after the clone has already
  # succeeded. Resolve an interpreter explicitly instead of assuming the name resolves.
  if [ -z "${PY312:-}" ]; then
    for cand in "$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12" \
                "/zfsauton2/home/yixiz/Proteina-Complexa/.venv/bin/python" \
                "$(command -v python3.12 || true)" \
                "$(command -v python3 || true)"; do
      [ -x "$cand" ] || continue
      if "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' 2>/dev/null; then
        PY312="$cand"; break
      fi
    done
  fi
  [ -n "${PY312:-}" ] && [ -x "$PY312" ] || { echo "FATAL: no CPython >=3.12; set PY312=" >&2; exit 1; }
  echo "[venv] creating $VENV with $PY312 ($("$PY312" -V 2>&1))"
  "$PY312" -m venv "$VENV"
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
# On babel the weights were already on disk and only the CCD mol tarball needed
# unpacking. On a fresh cluster they have to be fetched: ~6.2 GB from the
# `boltz-community/boltz-2` HF repo. The repo file list is printed before the fetch, so
# an upstream rename shows up as a readable mismatch in this log rather than a bare
# KeyError.
#
# The URLs are taken from boltz's OWN constants (`src/boltz/main.py`: CCD_URL, MOL_URL,
# and the boltz2 checkpoint URLs) rather than guessed from a repo name, because they do
# not all live in one place: `ccd.pkl` is served from the **boltz-1** repo while
# `mols.tar` and both boltz2 checkpoints come from **boltz-2**. Assuming one repo for
# all four is what the first attempt did, and it failed on ccd.pkl alone.
#
# Resolved against the pinned clone, so this cannot drift from the revision being built.
BOLTZ_NEED="ccd.pkl mols.tar boltz2_conf.ckpt boltz2_aff.ckpt"
MISSING=""
for f in $BOLTZ_NEED; do [ -s "$BOLTZ_CACHE/$f" ] || MISSING="$MISSING $f"; done
if [ -n "$MISSING" ]; then
  echo "[cache] missing:$MISSING"
  mkdir -p "$BOLTZ_CACHE"
  echo "[cache] URLs as declared by $BOLTZ_SRC/src/boltz/main.py:"
  grep -nE "^(CCD_URL|MOL_URL|BOLTZ2_[A-Z_]*URL)" "$BOLTZ_SRC/src/boltz/main.py" \
    | sed 's/^/           /' || true
  url_for() {
    case "$1" in
      ccd.pkl)          echo "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl" ;;
      mols.tar)         echo "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar" ;;
      boltz2_conf.ckpt) echo "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt" ;;
      boltz2_aff.ckpt)  echo "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt" ;;
      *)                return 1 ;;
    esac
  }
  for f in $BOLTZ_NEED; do
    [ -s "$BOLTZ_CACHE/$f" ] && { echo "[cache] have $f"; continue; }
    u="$(url_for "$f")" || { echo "FATAL: no URL known for $f" >&2; exit 1; }
    echo "[cache] downloading $f from $u"
    # -C - resumes a partial file, so a preemption or a dropped link costs only the
    # remainder of the current file.
    curl -fL -C - --retry 5 --retry-delay 10 -o "$BOLTZ_CACHE/$f" "$u"
    echo "[cache]   -> $BOLTZ_CACHE/$f ($(stat -c%s "$BOLTZ_CACHE/$f") bytes)"
  done
fi
for f in $BOLTZ_NEED; do
  [ -s "$BOLTZ_CACHE/$f" ] || { echo "FATAL: missing $BOLTZ_CACHE/$f after fetch" >&2; exit 1; }
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
