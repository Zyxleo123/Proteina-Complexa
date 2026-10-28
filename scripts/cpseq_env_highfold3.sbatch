#!/usr/bin/env bash
#SBATCH --job-name=cpseq_env_hf3
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_env_hf3_%j.out
#
# Build the HighFold3 runtime: AlphaFold3 under CPython 3.11, with HighFold3's model
# files overlaid, per its README ("refer to the conda version of AlphaFold3
# installation, and replace the feature files by highfold3/model").
#
# The version pin is forced and narrow, and both ends were established from evidence:
#
#   - HighFold3's cyclic module exists ONLY as model/__pycache__/
#     cyclic_peptide_matrix.cpython-311.pyc. No .py is tracked, so it must run under
#     CPython 3.11 exactly. (It loads there; its traceback shows it was built in a
#     conda env at .../python3.11/site-packages/alphafold3/model/.)
#   - current AF3 (131ab9e7) sets requires-python = ">=3.12".
#   - AF3 a8ecdb2d (2025-09-08), the last revision before HighFold3's 2025-09-24
#     snapshot, allows ">=3.11" with jax==0.4.34 and dm-haiku==0.0.13.
#
# So: AF3 @ a8ecdb2d + CPython 3.11, from python-build-standalone since the cluster
# has only 3.12.
#
# AF3's genetic databases are NOT fetched -- installation.md wants up to 1 TB against a
# 500 GB quota. AF3 accepts precomputed alignments inline, which is how the adapter
# supplies them (same cached per-target MSAs the other arms use).
#
# Idempotent and requeue-safe; each stage drops a marker.
#
# Submit:  sbatch scripts/cpseq_env_highfold3.sbatch
# Resume:  the identical line

set -euo pipefail

ZFS="${ZFS:-/data/user_data/yixiz}"
ENV_ROOT="${ENV_ROOT:-$ZFS/envs/highfold3}"
VENV="$ENV_ROOT/.venv"
PY311="${PY311:-$ZFS/pythons/python/bin/python3.11}"
AF3_SRC="${AF3_SRC:-$ZFS/upstream/alphafold3}"
AF3_REV="${AF3_REV:-a8ecdb2d7a433c5e9f8510a0f52434a0c55018c4}"
AF3_WORKTREE="${AF3_WORKTREE:-$ZFS/upstream/alphafold3_hf3}"
HF3_SRC="${HF3_SRC:-$ZFS/upstream/highfold3}"
HF3_REV="${HF3_REV:-2ff1ea31e59f3bac70e45aadde60a4a8fed6e199}"
MARKERS="$ENV_ROOT/markers"
mkdir -p "$MARKERS"

echo "=== cpseq_env_highfold3 ==="
echo "node:    $(hostname)"
echo "job:     ${SLURM_JOB_ID:-local}"
echo "python:  $PY311 ($("$PY311" -V 2>&1))"
echo "af3:     $AF3_REV -> $AF3_WORKTREE"
echo "hf3:     $HF3_REV"
cmake --version | head -1
gcc --version | head -1

[ -x "$PY311" ] || { echo "FATAL: no CPython 3.11 at $PY311" >&2; exit 1; }

# ------------------------------------------------------- AF3 source at the pinned rev
# A worktree, so the pinned 131ab9e clone's HEAD is left alone.
if [ ! -d "$AF3_WORKTREE/.git" ] && [ ! -f "$AF3_WORKTREE/pyproject.toml" ]; then
  echo "[af3] creating worktree at $AF3_REV"
  git -C "$AF3_SRC" worktree add --detach "$AF3_WORKTREE" "$AF3_REV"
else
  echo "[af3] worktree present"
fi
HAVE=$(git -C "$AF3_WORKTREE" rev-parse HEAD 2>/dev/null || echo none)
echo "[af3] worktree at $HAVE"
[ "$HAVE" = "$AF3_REV" ] || { echo "FATAL: worktree is $HAVE, expected $AF3_REV" >&2; exit 1; }
grep -m1 requires-python "$AF3_WORKTREE/pyproject.toml"

# ------------------------------------------------------------------------------ venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "[venv] creating $VENV"
  "$PY311" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip wheel setuptools
echo "[venv] $("$PY" -V)"

# ------------------------------------------------------------------- build AF3
if [ -f "$MARKERS/af3_$AF3_REV" ] && "$PY" -c "import alphafold3" 2>/dev/null; then
  echo "[af3] already installed"
else
  echo "[af3] installing build requirements"
  "$PY" -m pip install "scikit_build_core" "pybind11" "cmake>=3.28" "ninja" "numpy"
  # Name the compiler explicitly. Without this the build died with
  #   CMake Error ... Could not find the compiler specified in the environment
  #   variable CXX: clang++
  # even though CXX is unset in the shell and clang is not installed on these nodes --
  # scikit-build-core/CMake 4.4 guesses clang++ and then reports it as if the
  # environment had asked for it. gcc 11.5 is present and compiles AF3's required
  # C++20 (checked: `g++ -std=c++20` builds a trivial TU).
  export CC="${CC:-gcc}" CXX="${CXX:-g++}"
  echo "[af3] CC=$CC CXX=$CXX"
  echo "[af3] building (this compiles C++ extensions; slow)"
  # --no-build-isolation would hide the pinned build deps; left on deliberately.
  "$PY" -m pip install --no-cache-dir \
    -C cmake.define.CMAKE_C_COMPILER="$CC" \
    -C cmake.define.CMAKE_CXX_COMPILER="$CXX" \
    "$AF3_WORKTREE"
  "$PY" -c "import alphafold3; print('alphafold3 import OK')"
  # HighFold3's overlaid model/features.py imports chex, which pinned AF3 a8ecdb2d does
  # not declare -- so run_alphafold.py dies with ModuleNotFoundError after a build that
  # otherwise succeeded. 0.1.87 is the last release compatible with jax 0.4.34.
  "$PY" -m pip install --no-cache-dir "chex==0.1.87"
  touch "$MARKERS/af3_$AF3_REV"
fi

# --------------------------------------------- overlay HighFold3's model files
# HighFold3 ships a modified copy of AF3's model/ tree. Its own .pyc was built at
# site-packages/alphafold3/model/, which is where these belong.
SITE=$("$PY" -c "import alphafold3, pathlib; print(pathlib.Path(alphafold3.__file__).parent)")
echo "[hf3] overlaying onto $SITE"
HAVE_HF3=$(git -C "$HF3_SRC" rev-parse HEAD)
[ "$HAVE_HF3" = "$HF3_REV" ] || { echo "FATAL: highfold3 clone is $HAVE_HF3, expected $HF3_REV" >&2; exit 1; }

if [ ! -d "$SITE.af3_backup" ]; then
  cp -r "$SITE" "$SITE.af3_backup"
  echo "[hf3] kept a pristine AF3 copy at $SITE.af3_backup"
fi
# Copy HighFold3's trees over the installed package. `model` carries CycPOEM; the
# others are its modified data/feature pipeline.
for d in model data common constants parsers structure; do
  if [ -d "$HF3_SRC/$d" ]; then
    echo "[hf3]   $d/"
    cp -r "$HF3_SRC/$d/." "$SITE/$d/" 2>/dev/null || cp -r "$HF3_SRC/$d" "$SITE/"
  fi
done
# HighFold3's cyclic module ships ONLY as model/__pycache__/
# cyclic_peptide_matrix.cpython-311.pyc, with no .py tracked. Copying the overlay
# verbatim leaves it inside __pycache__, where Python will NOT find it: a __pycache__
# entry is a cache validated against a source file, so with no .py beside it the import
# fails with "unknown location". A sourceless module must sit directly in the package
# directory as <name>.pyc. This places it there, which is the only way this module is
# importable at all.
AF3_PKG="$("$PY" -c 'import alphafold3, os; print(os.path.dirname(alphafold3.__file__))')"
CPM_PYC="$AF3_PKG/model/__pycache__/cyclic_peptide_matrix.cpython-311.pyc"
if [ -f "$CPM_PYC" ]; then
  cp "$CPM_PYC" "$AF3_PKG/model/cyclic_peptide_matrix.pyc"
  echo "[hf3] placed sourceless cyclic_peptide_matrix.pyc in $AF3_PKG/model/"
else
  echo "WARNING: $CPM_PYC absent; the cyclic module will not import"
fi

touch "$MARKERS/hf3_overlay_$HF3_REV"

# ------------------------------------------------------------------- CCD data
# AF3 ships `build_data`, which parses libcifpp's components.cif into
# constants/converters/ccd.pickle. pip install does NOT run it, and without it every
# prediction dies at chemical_components.cached_ccd with FileNotFoundError on
# ccd.pickle -- after the model has loaded, so it looks like a runtime fault rather
# than an incomplete install.
#
# It needs real memory: parsing the full CCD inside a node-hold job's step_extern
# cgroup gets SIGKILLed (exit 137, no traceback). That is why this is a build stage in
# a job and not a one-liner someone runs by hand.
CCD_PICKLE="$AF3_PKG/constants/converters/ccd.pickle"
if [ -s "$CCD_PICKLE" ]; then
  echo "[af3] ccd.pickle present, skipping build_data"
else
  echo "[af3] build_data (parses components.cif; memory-hungry)"
  "$VENV/bin/build_data"
  [ -s "$CCD_PICKLE" ] || { echo "ERROR: build_data did not produce $CCD_PICKLE"; exit 1; }
  echo "[af3] ccd.pickle $(stat -c%s "$CCD_PICKLE") bytes"
fi
touch "$MARKERS/ccd_data"

# --------------------------------------------------------------------- verify
echo "[verify] cyclic module + topology probe"
"$PY" - <<'PYEOF'
import json, pathlib, sys
import numpy as np
import alphafold3
site = pathlib.Path(alphafold3.__file__).parent
try:
    from alphafold3.model import cyclic_peptide_matrix as cpm
except Exception as exc:
    print(json.dumps({"cyclic_module_import": f"FAILED: {type(exc).__name__}: {exc}"}))
    sys.exit(1)
pep_first = np.array([1,2,3,4,5, 1,2,3,4,5,6,7,8])
rec_first = np.array([1,2,3,4,5,6,7,8, 1,2,3,4,5])
a = np.array(cpm.calc_cyclic_matrix_af3(pep_first, head_to_tail=True, token_wise=False))
b = np.array(cpm.calc_cyclic_matrix_af3(rec_first, head_to_tail=True, token_wise=False))
print(json.dumps({
    "cyclic_module_import": "OK",
    "peptide_first_shape": list(a.shape),
    "receptor_first_shape": list(b.shape),
    "ring_follows_first_chain": list(a.shape) == [5, 5] and list(b.shape) == [8, 8],
}, indent=2))
PYEOF

echo "[verify] run_alphafold.py --help"
"$PY" "$HF3_SRC/run_alphafold.py" --help 2>&1 | head -20 || echo "WARNING: run_alphafold.py --help failed (see above)"

export AF3_REV HF3_REV
"$PY" - <<'PYEOF' > "$ENV_ROOT/env_provenance.json"
import json, sys, os, importlib.metadata as md
def v(p):
    try: return md.version(p)
    except Exception: return None
rec = {"python": sys.version.split()[0], "alphafold3": v("alphafold3"), "jax": v("jax"),
       "dm-haiku": v("dm-haiku"), "numpy": v("numpy"), "rdkit": v("rdkit"),
       "af3_rev": os.environ.get("AF3_REV"), "highfold3_rev": os.environ.get("HF3_REV")}
try:
    import jax
    rec["jax_devices"] = [str(d) for d in jax.devices()]
except Exception as exc:
    rec["jax_devices"] = f"unavailable: {type(exc).__name__}: {exc}"
print(json.dumps(rec, indent=2))
PYEOF

cat "$ENV_ROOT/env_provenance.json"

echo "=== cpseq_env_highfold3 DONE ==="
