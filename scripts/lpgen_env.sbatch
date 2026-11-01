#!/bin/bash
#SBATCH --job-name=lpgen_env
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_env_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_env_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
# Build one Python environment per arm.
#
# A separate venv per arm, not a shared one: the arms pin incompatible stacks (foundry
# resolves torch 2.14.1+cu126 against this project's 2.7.0+cu126), and installing an arm's
# dependencies into the project venv would silently change what every other job in this
# repository runs.
#
# Environments live on scratch beside the trees -- torch alone is ~2.5 GB per env and home
# has a quota.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "env"

UV_BIN="${UV_BIN:-/zfsauton2/home/yixiz/.local/bin/uv}"
[[ -x "$UV_BIN" ]] || { echo "FATAL: uv not executable at ${UV_BIN}" >&2; exit 1; }

# Keep uv's cache off home (quota) and make it shareable between arms.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ZFS}/lpgen/uv_cache}"
mkdir -p "$UV_CACHE_DIR" "$ENVS_DIR"

for arm in ${STAGE_ARMS//,/ }; do
  tree="${UPSTREAM_DIR}/${arm}"
  if [[ ! -d "$tree" ]]; then
    # A gate, not a crash: print why and keep going, so one unstaged arm does not
    # abandon the others.
    echo "[env] GATE ${arm}: not staged at ${tree}; run --submit stage first" >&2
    continue
  fi

  env_dir="${ENVS_DIR}/${arm}"
  echo "=================================================================="
  echo "[env] ${arm}: ${env_dir}"
  if [[ -d "$env_dir" && "$ENV_CLEAR" != "1" ]]; then
    echo "[env] GATE ${arm}: venv already exists at ${env_dir}" >&2
    echo "[env] GATE ${arm}: pass --set ENV_CLEAR=1 to rebuild it (this REPLACES the env)" >&2
    continue
  fi
  # Python version is per-arm: DiffPepBuilder's export pins 3.9.16 and vendors openfold,
  # which is version-sensitive. Installing it under 3.12 is not a smaller change than it
  # looks -- it changes which wheels exist at all.
  arm_py="$(_lpgen_arm_python "$arm")"
  echo "[env] ${arm}: python ${arm_py}"
  VENV_ARGS=(--python "$arm_py")
  [[ "$ENV_CLEAR" == "1" ]] && VENV_ARGS+=(--clear)
  "$UV_BIN" venv "${VENV_ARGS[@]}" "$env_dir"

  # Build the constraint list: the arm's own CI constraints plus ours. Both are resolved
  # PER ARM -- foundry's atomworks/torch pins must not leak onto an arm that pins its own
  # torch, which a global default would do silently.
  CARGS=()
  arm_cfile="$(_lpgen_arm_constraints_file "$arm")"
  arm_extra="$(_lpgen_arm_extra_constraints "$arm")"
  if [[ -n "$arm_cfile" && -f "${tree}/${arm_cfile}" ]]; then
    echo "[env] ${arm}: constraints <- ${arm_cfile}"
    CARGS+=(-c "${tree}/${arm_cfile}")
  fi
  if [[ -n "$arm_extra" ]]; then
    extra="${env_dir}.extra-constraints.txt"
    : > "$extra"
    for c in $arm_extra; do echo "$c" >> "$extra"; done
    echo "[env] ${arm}: extra constraints -> $(tr '\n' ' ' < "$extra")"
    CARGS+=(-c "$extra")
  fi

  target="$(_lpgen_arm_install_target "$arm")"
  arm_reqs="$(_lpgen_arm_requirements "$arm")"
  if [[ -n "$target" ]]; then
    echo "[env] ${arm}: editable install of ${target}"
    ( cd "$tree" && VIRTUAL_ENV="$env_dir" "$UV_BIN" pip install "${CARGS[@]}" -e "$target" )
  elif [[ -n "$arm_reqs" ]]; then
    # The arm ships only a fully-pinned `conda env export`. We install OUR translation of
    # it (see _lpgen_arm_requirements for why conda env create is not used) and write that
    # translation next to the venv, so what was installed stays auditable after the fact.
    reqs="${env_dir}.requirements.txt"
    printf '%s\n' "$arm_reqs" > "$reqs"
    echo "[env] ${arm}: translated requirements -> ${reqs}"
    # --no-config, and it is load-bearing. MEASURED (jobs 84019, 84021): this job's CWD is
    # the repository, so uv reads OUR pyproject.toml -- which sets
    #   [tool.uv] extra-index-url = ["https://download.pytorch.org/whl/cu126"]
    # Because uv only considers the FIRST index that carries a package, that redirects
    # every package to the PyTorch index, not just torch: the build failed on
    # `tqdm==4.66.1` ("found on .../cu126, but not at the requested version"), because that
    # index mirrors an old tqdm. An arm's environment must be a function of the ARM's
    # requirements, never of the directory the job happened to start in.
    VIRTUAL_ENV="$env_dir" "$UV_BIN" pip install --no-config "${CARGS[@]}" -r "$reqs"
  elif [[ -f "$tree/requirements.txt" ]]; then
    echo "[env] ${arm}: requirements.txt"
    VIRTUAL_ENV="$env_dir" "$UV_BIN" pip install "${CARGS[@]}" -r "$tree/requirements.txt"
  else
    # POTFlow and PepGLAD declare no dependencies at all. Guessing a set here would
    # bury that fact; the arm's own notes record what its imports need.
    echo "[env] ${arm}: NO packaging and NO requirements.txt -- venv created empty." >&2
    echo "[env] ${arm}: see docs/old_docs/README_LPGEN.md for the import-derived dependency list." >&2
  fi

  echo "[env] ${arm}: verifying"
  VIRTUAL_ENV="$env_dir" "$env_dir/bin/python" - "$arm" <<'PYEOF'
import importlib, sys
arm = sys.argv[1]
# Per-arm import smoke: enough to prove the install is usable, not a test suite.
probes = {
    # StructuralDatasetWrapper is checked BY NAME: it is the class both rfd3 dataset
    # configs instantiate, and atomworks 3.x removed it. Importing `atomworks.ml.datasets`
    # successfully proves nothing about whether the data path can be built.
    "foundry": ["torch", "atomworks", "atomworks.ml.datasets", "foundry", "rfd3", "hydra"],
    "ppflow":  ["torch"],
    "pepglad": ["torch"],
    "potflow": ["torch", "lmdb", "easydict", "ot", "torchcfm"],
    # esm is load-bearing, not incidental: the peptide SEQUENCE reaches DiffPepDock as a
    # per-residue ESM embedding (data/pdb_data_loader.py), which is the channel the
    # similarity-budgeted task depends on. hydra/omegaconf drive its configs; mdtraj and
    # tmtools are used by its evaluation path.
    # pyrootutils and wget are probed because EVERY DiffPepBuilder entry point imports them
    # at module level (and setup_root uses dotenv), so a missing one is an immediate crash
    # rather than a late failure. ml_collections is imported by its vendored openfold.
    # wandb and pkg_resources are probed because of a REAL escape (job 84128): the import
    # chain run_docking.py -> experiments.train -> wandb -> ... -> pkg_resources died on a
    # GPU node, after preprocessing had already run. A uv venv ships no setuptools, so
    # pkg_resources was absent while every module in the old probe list imported fine.
    # Probe what the ENTRY POINT imports, not just what looks load-bearing.
    "diffpepbuilder": ["torch", "esm", "hydra", "omegaconf", "Bio", "tmtools",
                       "ml_collections", "tree", "pyrootutils", "wget", "dotenv",
                       "wandb", "pkg_resources"],
    "unimomo": ["torch"],
}.get(arm, ["torch"])
bad = []
for m in probes:
    try:
        mod = importlib.import_module(m)
        print(f"  ok      {m:28s} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        bad.append(m)
        print(f"  FAIL    {m:28s} {type(exc).__name__}: {exc}")
# Named symbols that must exist, not just importable modules.
symbols = {
    "foundry": [
        ("atomworks.ml.datasets", "StructuralDatasetWrapper"),
        ("atomworks.ml.datasets", "PandasDataset"),
        ("atomworks.ml.datasets.parsers", "GenericDFParser"),
    ],
}.get(arm, [])
for mod_name, sym in symbols:
    try:
        mod = importlib.import_module(mod_name)
        getattr(mod, sym)
        print(f"  ok      {mod_name}.{sym}")
    except Exception as exc:
        bad.append(f"{mod_name}.{sym}")
        print(f"  FAIL    {mod_name}.{sym}  {type(exc).__name__}: {exc}")

if bad:
    print(f"IMPORT CHECK FAILED for {arm}: {bad}")
    raise SystemExit(1)
print(f"import check passed for {arm}")
PYEOF
  echo "[env] ${arm}: python -> ${env_dir}/bin/python"
done

echo "[env] done -> $ENVS_DIR"
