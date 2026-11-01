#!/usr/bin/env bash
#SBATCH --job-name=cpseq_env_af3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_env_af3_%j.out
#
# Build a PLAIN AlphaFold3 runtime, for AF3 as the sequence folder in place of Boltz-2.
#
# Why plain AF3 and not the HighFold3 overlay (`cpseq_env_highfold3.sbatch`)
# -------------------------------------------------------------------------
# HighFold3 exists to teach AF3 that a POLYMER chain is cyclic, via a wrapped relative
# position encoding. This arm does not need that: the peptide is handed to AF3 as a
# LIGAND whose SMILES already contains the head-to-tail bond, so the ring is part of the
# chemistry rather than something the position encoding has to express. The overlay also
# replaces AF3's own data/feature pipeline with HighFold3's modified copy, which would
# make "we folded this with AF3" untrue in a way that is hard to see later.
#
# Plain AF3 at the current pin therefore also drops the whole CPython 3.11 apparatus the
# HighFold3 build needs (that pin exists only because HighFold3's cyclic module ships as
# a sourceless 3.11 .pyc). AF3 131ab9e7 wants >=3.12 and this cluster has 3.12.13.
#
# The genetic databases are NOT fetched -- installation.md wants up to 1 TB. AF3 accepts
# precomputed alignments inline, which is how the adapter supplies them: the same cached
# per-target ColabFold MSAs the Boltz-2 arm uses, so the two predictors see the same
# evolutionary input and differ only as folders.
#
# Idempotent and requeue-safe; every stage drops a marker under $ENV_ROOT/markers.
#
# Submit:  sbatch scripts/cpseq_env_af3.sbatch
# Resume:  the identical line
# Config:  positional args, either a path to an env file to source or VAR=value pairs.
#          NEVER `sbatch --export` -- on this cluster any explicit export list makes
#          sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
#          environment, and the job is requeued and held.

set -euo pipefail

for arg in "$@"; do
  if [ -f "$arg" ]; then echo "[cfg] sourcing $arg"; # shellcheck disable=SC1090
    . "$arg"
  elif [[ "$arg" == *=* ]]; then echo "[cfg] export $arg"; export "$arg"
  fi
done

ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ENV_ROOT="${ENV_ROOT:-$ZFS/envs/af3}"
VENV="$ENV_ROOT/.venv"
AF3_SRC="${AF3_SRC:-$ZFS/upstream/alphafold3}"
AF3_REV="${AF3_REV:-131ab9e7420066523c62546199243f7a9b3e3e78}"
AF3_REPO="${AF3_REPO:-https://github.com/google-deepmind/alphafold3}"
# Model parameters. 1.02 GB zstd-compressed; this URL is the one this project has always
# used (docs/RESOURCES_COLLECTED.md). Using them means accepting Google DeepMind's AF3
# model-parameter terms, which are non-commercial and forbid redistribution -- so the
# .bin stays on scratch and is never committed, copied into the repo, or pushed to B2.
AF3_WEIGHTS_URL="${AF3_WEIGHTS_URL:-https://storage.googleapis.com/alphafold3/af3.bin.zst}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:-$ZFS/Proteina-Complexa/community_models/ckpts/AF3}"
AF3_WEIGHTS_BYTES="${AF3_WEIGHTS_BYTES:-1020545840}"   # measured on the .zst, 2026-11-01
PY312="${PY312:-}"

MARKERS="$ENV_ROOT/markers"
mkdir -p "$MARKERS" "$AF3_MODEL_DIR" "$(dirname "$AF3_SRC")"

echo "=== cpseq_env_af3 ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=cpu cpus=${SLURM_CPUS_PER_TASK:-8} mem=48G time=4:00:00"
echo "af3:      $AF3_REV"
echo "env_root: $ENV_ROOT"
echo "weights:  $AF3_MODEL_DIR/af3.bin"
nvidia-smi -L 2>/dev/null | head -3 || echo "gpu:      none (cpu partition; jax CUDA is verified in the GPU preflight, not here)"
cmake --version | head -1
gcc --version | head -1

# ------------------------------------------------------------------ CPython 3.12
if [ -z "$PY312" ]; then
  for cand in "$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12" \
              "/zfsauton2/home/yixiz/Proteina-Complexa/.venv/bin/python" \
              "$(command -v python3.12 || true)"; do
    [ -x "$cand" ] || continue
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' 2>/dev/null; then
      PY312="$cand"; break
    fi
  done
fi
[ -n "$PY312" ] && [ -x "$PY312" ] || { echo "FATAL: no CPython >=3.12 found; set PY312=" >&2; exit 1; }
echo "python:   $PY312 ($("$PY312" -V 2>&1))"

# --------------------------------------------------------------- AF3 source at the pin
if [ ! -d "$AF3_SRC/.git" ]; then
  echo "[af3] cloning $AF3_REPO"
  git clone --quiet "$AF3_REPO" "$AF3_SRC"
fi
git -C "$AF3_SRC" fetch --quiet origin || true
git -C "$AF3_SRC" checkout --quiet --detach "$AF3_REV"
HAVE=$(git -C "$AF3_SRC" rev-parse HEAD)
[ "$HAVE" = "$AF3_REV" ] || { echo "FATAL: source is $HAVE, expected $AF3_REV" >&2; exit 1; }
echo "[af3] at $HAVE"
grep -m1 requires-python "$AF3_SRC/pyproject.toml" || true

# ------------------------------------------------------------------------------ venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "[venv] creating $VENV"
  "$PY312" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip wheel setuptools
echo "[venv] $("$PY" -V)"

# ------------------------------------------------------------------------- build AF3
# Name the compiler explicitly. Without this the build on babel died with
#   CMake Error ... Could not find the compiler specified in the environment
#   variable CXX: clang++
# with CXX unset and clang not installed -- scikit-build-core/CMake guesses clang++ and
# then reports it as if the environment had asked for it.
#
# The compiler here is gcc 15.3, where AF3's C++20 was last built against gcc 11.5.
# Newer gcc is stricter (the classic failure is a missing transitive <cstdint>), so if
# the system compiler fails this falls back to a conda-provided gcc 12 rather than
# leaving a half-built env. Both paths are recorded in env_provenance.json.
build_af3() {
  local cc="$1" cxx="$2"
  echo "[af3] building with CC=$cc CXX=$cxx (compiles C++ extensions; slow)"
  CC="$cc" CXX="$cxx" "$PY" -m pip install --no-cache-dir \
    -C cmake.define.CMAKE_C_COMPILER="$cc" \
    -C cmake.define.CMAKE_CXX_COMPILER="$cxx" \
    "$AF3_SRC"
}

if [ -f "$MARKERS/af3_$AF3_REV" ] && "$PY" -c "import alphafold3" 2>/dev/null; then
  echo "[af3] already installed"
else
  "$PY" -m pip install --quiet "scikit_build_core" "pybind11" "cmake>=3.28" "ninja" "numpy"
  AF3_CC=""; AF3_CXX=""
  if build_af3 "${CC:-gcc}" "${CXX:-g++}"; then
    AF3_CC="${CC:-gcc}"; AF3_CXX="${CXX:-g++}"
  else
    echo "[af3] system compiler failed; provisioning gcc 12 via conda"
    CONDA_PREFIX_GCC="$ENV_ROOT/toolchain"
    if [ ! -x "$CONDA_PREFIX_GCC/bin/x86_64-conda-linux-gnu-g++" ]; then
      CONDA_EXE="${CONDA_EXE:-$ZFS/miniconda3/bin/conda}"
      [ -x "$CONDA_EXE" ] || { echo "FATAL: need conda for the gcc fallback ($CONDA_EXE)" >&2; exit 1; }
      "$CONDA_EXE" create --quiet --yes --prefix "$CONDA_PREFIX_GCC" -c conda-forge \
        "gxx_linux-64=12" "gcc_linux-64=12"
    fi
    AF3_CC="$CONDA_PREFIX_GCC/bin/x86_64-conda-linux-gnu-gcc"
    AF3_CXX="$CONDA_PREFIX_GCC/bin/x86_64-conda-linux-gnu-g++"
    build_af3 "$AF3_CC" "$AF3_CXX"
  fi
  "$PY" -c "import alphafold3; print('[af3] import OK')"
  echo "$AF3_CC|$AF3_CXX" > "$ENV_ROOT/compiler.txt"
  touch "$MARKERS/af3_$AF3_REV"
fi

# -------------------------------------------------------------------------- jax + CUDA
# jax's cuda12 wheels carry their own CUDA runtime, so no `module load cuda` is needed
# and the node's driver is the only system dependency. Installed AFTER AF3 so AF3's own
# pin wins if it declares one.
if [ -f "$MARKERS/jax_cuda" ] && "$PY" -c "import jax" 2>/dev/null; then
  echo "[jax] already installed"
else
  JAXVER="$("$PY" -c "import importlib.metadata as m; print(m.version('jax'))" 2>/dev/null || echo "")"
  if [ -n "$JAXVER" ]; then
    echo "[jax] AF3 pinned jax==$JAXVER; installing the matching cuda12 extra"
    "$PY" -m pip install --no-cache-dir "jax[cuda12]==$JAXVER"
  else
    echo "[jax] no jax pin from AF3; installing jax[cuda12]"
    "$PY" -m pip install --no-cache-dir "jax[cuda12]"
  fi
  touch "$MARKERS/jax_cuda"
fi

# ------------------------------------------------------------------------- CCD data
# AF3 ships `build_data`, which parses libcifpp's components.cif into
# constants/converters/ccd.pickle. pip install does NOT run it, and without it every
# prediction dies at chemical_components.cached_ccd with FileNotFoundError on
# ccd.pickle -- AFTER the model has loaded, so it reads as a runtime fault rather than
# an incomplete install.
#
# It needs real memory: parsing the full CCD inside a node-hold job's step_extern cgroup
# gets SIGKILLed (exit 137, no traceback). That is why this is a build stage in a job
# with 48G and not a one-liner someone runs by hand.
AF3_PKG="$("$PY" -c 'import alphafold3, os; print(os.path.dirname(alphafold3.__file__))')"
CONVERTERS="$AF3_PKG/constants/converters"
# The artefact's NAME is version-dependent: AF3 a8ecdb2d wrote `ccd.pickle`, and the rev
# pinned here (131ab9e7) writes `ccd.msgpack` plus `chemical_component_sets.msgpack`.
# Checking only for the pickle made a successful build look like a failed one, so accept
# either and say which was found.
ccd_artifact() {
  for f in "$CONVERTERS/ccd.msgpack" "$CONVERTERS/ccd.pickle"; do
    [ -s "$f" ] && { echo "$f"; return 0; }
  done
  return 1
}
if CCD="$(ccd_artifact)"; then
  echo "[af3] CCD present: $(basename "$CCD") ($(stat -c%s "$CCD") bytes)"
else
  echo "[af3] build_data (parses components.cif; memory-hungry)"
  "$VENV/bin/build_data"
  CCD="$(ccd_artifact)" || {
    echo "FATAL: build_data produced no ccd.msgpack or ccd.pickle in $CONVERTERS" >&2
    ls -la "$CONVERTERS" >&2
    exit 1
  }
  echo "[af3] CCD built: $(basename "$CCD") ($(stat -c%s "$CCD") bytes)"
fi
touch "$MARKERS/ccd_data"

# ------------------------------------------------------------------- model parameters
# Resumable: curl -C - continues a partial .zst, and the decompressed .bin is only
# written once the compressed size matches what was measured. A truncated download that
# still decompresses is the failure mode worth spending a size check on.
ZST="$AF3_MODEL_DIR/af3.bin.zst"
BIN="$AF3_MODEL_DIR/af3.bin"
if [ -s "$BIN" ] && [ -f "$MARKERS/weights" ]; then
  echo "[weights] present ($(stat -c%s "$BIN") bytes)"
else
  echo "[weights] fetching $AF3_WEIGHTS_URL"
  curl -fL -C - --retry 5 --retry-delay 10 -o "$ZST" "$AF3_WEIGHTS_URL"
  GOT=$(stat -c%s "$ZST")
  if [ "$GOT" != "$AF3_WEIGHTS_BYTES" ]; then
    echo "WARNING: af3.bin.zst is $GOT bytes, expected $AF3_WEIGHTS_BYTES."
    echo "         Upstream may have republished; recording the actual size rather than failing."
  fi
  echo "[weights] decompressing"
  zstd -d -f -o "$BIN" "$ZST"
  [ -s "$BIN" ] || { echo "FATAL: decompression produced nothing" >&2; exit 1; }
  echo "[weights] af3.bin $(stat -c%s "$BIN") bytes"
  rm -f "$ZST"
  # The terms note that travelled with these weights was lost with the old cluster.
  cat > "$AF3_MODEL_DIR/TERMS.md" <<'TERMS'
# AlphaFold 3 model parameters

Fetched from `https://storage.googleapis.com/alphafold3/af3.bin.zst`, the source this
project has always recorded for them (`docs/RESOURCES_COLLECTED.md`).

Use is governed by Google DeepMind's AlphaFold 3 Model Parameters Terms of Use. The
terms this project operates under:

- **Non-commercial research use only.**
- **No redistribution.** These parameters are not committed to git, not copied into the
  repo tree, and not pushed to the B2 bucket that carries the rest of this project's
  artefacts. They live on scratch and are re-fetched per cluster from the URL above.
- Derived predictions are research output, not experimental measurements. Energies
  computed from them are Rosetta ref2015 REU and are not measured binding free energies.

If you are reading this on a new cluster: re-fetch rather than copy.
TERMS
  touch "$MARKERS/weights"
fi

# ------------------------------------------------------------------------------ verify
echo "[verify] import, CCD, weights, and whether jax sees a GPU here"
AF3_REV="$AF3_REV" AF3_MODEL_DIR="$AF3_MODEL_DIR" "$PY" - <<'PYEOF' | tee "$ENV_ROOT/env_provenance.json"
import importlib.metadata as md
import json, os, pathlib, sys

def v(p):
    try: return md.version(p)
    except Exception: return None

import alphafold3
pkg = pathlib.Path(alphafold3.__file__).parent
conv = pkg / "constants" / "converters"
ccd = next((p for p in (conv / "ccd.msgpack", conv / "ccd.pickle") if p.exists()), None)
bin_ = pathlib.Path(os.environ["AF3_MODEL_DIR"]) / "af3.bin"

rec = {
    "python": sys.version.split()[0],
    "af3_rev": os.environ.get("AF3_REV"),
    "alphafold3": v("alphafold3"),
    "jax": v("jax"),
    "jaxlib": v("jaxlib"),
    "dm-haiku": v("dm-haiku"),
    "numpy": v("numpy"),
    "rdkit": v("rdkit"),
    "ccd_artifact": ccd.name if ccd else None,
    "ccd_bytes": ccd.stat().st_size if ccd else None,
    "af3_bin_bytes": bin_.stat().st_size if bin_.exists() else None,
    "overlay": None,
    "note": "plain AF3, no HighFold3 overlay: the peptide is a SMILES ligand, so the "
            "head-to-tail bond is in the chemistry and needs no cyclic position encoding",
}
try:
    import jax
    rec["jax_devices"] = [str(d) for d in jax.devices()]
except Exception as exc:
    rec["jax_devices"] = f"unavailable here: {type(exc).__name__}: {exc}"
print(json.dumps(rec, indent=2))
PYEOF

echo "[verify] run_alphafold.py --help"
"$PY" "$AF3_SRC/run_alphafold.py" --help 2>&1 | head -15 || echo "WARNING: run_alphafold.py --help failed (above)"

echo "=== cpseq_env_af3 DONE ==="
