# Shared preamble for the LP-generator milestone (the four LP training arms).
#
# Sourced by every scripts/lpgen_*.sbatch. Every knob has a default here; the submitter
# writes ONE per-submission env file and passes its PATH as a positional argument.
#
# Never `sbatch --export`: any explicit export list (--export=ALL,K=v, --export=K=v and
# --export=NONE alike) sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD -- stranding every
# dependent job on Dependency with no error anyone will see.
#
# Contract:
#   _lpgen_consume_args "$@"   # sources any arg that is a path, exports VAR=value args
#   _lpgen_defaults            # fills in anything still unset
#   _lpgen_banner <stage>      # hostname + resource request into the .out; sacct is dead here
#   _lpgen_preflight           # cheap gate; returns non-zero, the CALLER decides the exit code

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# ---------------------------------------------------------------- arg handling
_lpgen_consume_args() {
  local a
  for a in "$@"; do
    if [[ -f "${a}" ]]; then
      echo "[preamble] sourcing env file: ${a}"
      # shellcheck source=/dev/null
      set -a; source "${a}"; set +a
    elif [[ "${a}" == *=* ]]; then
      echo "[preamble] export ${a}"
      export "${a?}"
    elif [[ -n "${a}" ]]; then
      echo "[preamble] WARNING: ignoring unrecognized argument: ${a}" >&2
    fi
  done
}

# ------------------------------------------------------------------- defaults
_lpgen_defaults() {
  ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
  PY="${PY:-${REPO}/.venv/bin/python}"

  # NOTHING THIS MILESTONE PRODUCES MAY LAND ON HOME. Home is a 249 GB quota'd volume that
  # also holds the repository; checkpoints here are 2.6 GB each. Every output path --
  # job logs, per-submission env files, run directories, checkpoints -- resolves under
  # this root on scratch. Note `#SBATCH --output=` is a STATIC directive and cannot read a
  # variable, so the submitter passes --output/--error explicitly to override it; a script
  # run by bare `sbatch` would otherwise write its log into the repo.
  LPGEN_ROOT="${LPGEN_ROOT:-${ZFS}/lpgen}"
  LPGEN_LOG_DIR="${LPGEN_LOG_DIR:-${LPGEN_ROOT}/logs}"
  LPGEN_ENV_DIR="${LPGEN_ENV_DIR:-${LPGEN_ROOT}/env}"
  LPGEN_RUNS_DIR="${LPGEN_RUNS_DIR:-${LPGEN_ROOT}/runs}"

  # Inputs: the already-preprocessed LP pool. preprocess_lp.py wrote chain A = receptor,
  # chain B = peptide, LNR filtered out. This milestone does not re-run that.
  LP_META_DIR="${LP_META_DIR:-${ZFS}/LPData/preprocessed/metadata}"
  LP_TRAIN="${LP_TRAIN:-${LP_META_DIR}/lp_train.parquet}"
  LP_VAL="${LP_VAL:-${LP_META_DIR}/lp_val.parquet}"

  RUN_ID="${RUN_ID:-lpgen_$(date +%Y%m%d_%H%M%S)}"
  # The manifest is DATA, not a run artefact: it lives beside the other metadata so every
  # arm (and every later run) resolves one path, rather than the newest run directory.
  MANIFEST_DIR="${MANIFEST_DIR:-${ZFS}/LPData/preprocessed/metadata_lpgen}"
  MANIFEST="${MANIFEST:-${MANIFEST_DIR}/lpgen_manifest.parquet}"
  VAL_PANEL="${VAL_PANEL:-${MANIFEST_DIR}/lpgen_val_panel.parquet}"

  # Split parameters. RESERVE_* must match cp2lp_build_splits.py (--eval-frac 0.15,
  # --seed 0) or `heldout` stops being the CP->LP eval set and the later CP-sequence
  # comparison is contaminated by families this milestone trained on.
  RESERVE_FRAC="${RESERVE_FRAC:-0.15}"
  RESERVE_SEED="${RESERVE_SEED:-0}"
  VAL_FRAC="${VAL_FRAC:-0.10}"
  VAL_SEED="${VAL_SEED:-7}"

  PANEL_SIZE="${PANEL_SIZE:-32}"
  PANEL_SOURCES="${PANEL_SOURCES:-pepbench}"

  # Empty reuses the configured lp-mix weights renormalised to LP-only (0.24 PepBench).
  # Set to 0.8 for the inverse, binder-focused ratio -- a deliberate departure, recorded
  # in the summary as such.
  PEPBENCH_FRAC="${PEPBENCH_FRAC:-}"

  # Native per-complex export (PPFlow / POTFlow layout).
  NATIVE_DIR="${NATIVE_DIR:-${ZFS}/LPData/preprocessed/lpgen_native}"
  # heldout is reserved for the later CP-sequence comparison; exporting it would invite
  # an arm to point its structure_dir at it.
  EXPORT_SPLITS="${EXPORT_SPLITS:-train,val}"
  # 10.0 A heavy-atom whole-residue crop. MEASURED against PepMerge, which POTFlow's
  # released config points at: included pocket residues reach 9.37-9.98 A, excluded
  # receptor residues start at 10.01-10.53 A. Not a guess -- do not round it.
  POCKET_CUTOFF_A="${POCKET_CUTOFF_A:-10.0}"
  # PPFlow opens peptide_repaired.pdb / receptor_repaired.pdb (upstream runs FoldX first).
  PPFLOW_ALIASES="${PPFLOW_ALIASES:-1}"
  EXPORT_WORKERS="${EXPORT_WORKERS:-16}"
  EXPORT_LIMIT="${EXPORT_LIMIT:--1}"
  # POTFlow's dataset IS os.listdir(structure_dir), so a leftover directory from an
  # earlier larger run would silently enlarge the training set. The exporter refuses;
  # this makes a rerun authoritative instead.
  EXPORT_CLEAN="${EXPORT_CLEAN:-1}"

  # ---------------------------------------------------------- upstream arm repos
  # Code lives on scratch, not home: home has a quota and these are third-party trees we
  # patch and may re-stage. Our PATCHES live in the project repo (lpgen_patch_upstream.py)
  # so they stay reviewable and version-controlled while the trees stay disposable.
  UPSTREAM_DIR="${UPSTREAM_DIR:-${ZFS}/lpgen/upstream}"
  ENVS_DIR="${ENVS_DIR:-${ZFS}/lpgen/envs}"
  # Arms to act on. Directory name == arm name, which is also what
  # lpgen_patch_upstream.py keys its edits on.
  #   foundry = RFdiffusion3 (the lead arm; rfd3 lives in models/rfd3 of that monorepo)
  STAGE_ARMS="${STAGE_ARMS:-foundry}"
  # 1 re-clones an arm that is already staged. Off by default so a rerun is cheap and
  # cannot silently discard local patches.
  STAGE_FORCE="${STAGE_FORCE:-0}"
  # Apply our patch set after cloning. Idempotent, and refuses an arm rather than
  # half-patching it.
  STAGE_PATCH="${STAGE_PATCH:-1}"
  # Install against upstream's OWN tested constraints rather than letting pip resolve
  # latest. Measured: a latest-resolves install of foundry produced atomworks 3.0.1, which
  # DROPPED `StructuralDatasetWrapper` -- the class BOTH rfd3 dataset configs instantiate --
  # so the training data path could not be built at all. foundry's ci-constraints.txt pins
  # torch==2.7.1 and its header documents the same atomworks fragility.
  # Relative to the arm's tree; empty disables.
  ARM_CONSTRAINTS_FILE="${ARM_CONSTRAINTS_FILE:-ci-constraints.txt}"
  # Extra constraints the upstream file does not express. atomworks<3 is required: the
  # configs were written against 2.x and `>=2.1.1` in pyproject does not cap the major.
  ARM_EXTRA_CONSTRAINTS="${ARM_EXTRA_CONSTRAINTS:-atomworks<3}"
  # 1 replaces an existing venv. Default 0: `uv venv` refuses to overwrite, and silently
  # clearing one would pull the environment out from under any job already using it.
  ENV_CLEAR="${ENV_CLEAR:-0}"

  # ------------------------------------------- DiffPepDock staging (k=0 baseline)
  # LNR ships raw multi-chain PDBs plus test.txt (93 rows:
  # pdb_id / receptor_chain / peptide_chain / flag). Read-only here -- the stager writes
  # fresh two-chain complexes rather than touching the source tree.
  LNR_DIR="${LNR_DIR:-${ZFS}/LNR}"
  # Per-run staging dir. It must NOT live under the data root: DiffPepBuilder's
  # get_motif_center_pos writes a *_processed.pdb SIBLING next to every input, and the
  # data root is where lpgen_manifest.parquet points RFD3 at its training complexes.
  # The stager refuses outright if this resolves inside DPB_DATA_ROOT_GUARD.
  DPB_STAGE_DIR="${DPB_STAGE_DIR:-${LPGEN_ROOT}/dpb_stage/${RUN_ID}}"
  DPB_DATA_ROOT_GUARD="${DPB_DATA_ROOT_GUARD:-${ZFS}/LPData}"
  DPB_STAGE_LIMIT="${DPB_STAGE_LIMIT:--1}"

  # ------------------------------------------------------- released checkpoints
  # Weights live on scratch (1.24 GB each) and NEVER in the trees: a re-stage wipes the
  # tree, and re-downloading 2.5 GB because of a `git clone` is pure waste.
  WEIGHTS_DIR="${WEIGHTS_DIR:-${ZFS}/lpgen/weights}"
  # 1 re-downloads a checkpoint that is already present AND verifies. Off by default.
  WEIGHTS_FORCE="${WEIGHTS_FORCE:-0}"
  # Published md5s, read from the Zenodo and GitHub release APIs on 2026-10-14. These are
  # checked after download: a truncated or HTML-error-page download is otherwise a file
  # that exists, has plausible size, and fails only later inside torch.load.
  #
  # ESM-2 650M is in this list because DiffPepBuilder's sequence channel IS an ESM
  # embedding (use_esm_embed: True, raw_esm_size 1280), so it is a hard dependency of the
  # arm rather than an optional extra. Upstream's PretrainedSequenceEmbedder otherwise
  # `wget.download`s it -- unverified, un-resumable, and INTO THE TREE at
  # {root}/experiments/checkpoints, which a re-stage then throws away. Routing it here
  # instead gives it the same .part-and-verify discipline as the released checkpoints, and
  # the symlink below lets upstream's own default path find it unmodified.
  WEIGHTS_SPEC="${WEIGHTS_SPEC:-$(cat <<'WEOF'
diffpepbuilder diffpepbuilder_v1.pth https://zenodo.org/api/records/12794439/files/diffpepbuilder_v1.pth/content 5ab7d914a979db4f11ac778a51dd2241
diffpepbuilder diffpepdock_v1.pth    https://zenodo.org/api/records/15398020/files/diffpepdock_v1.pth/content    60579b59e97a4a2109f2b31b4fda185a
diffpepbuilder esm2_t33_650M_UR50D.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt -
diffpepbuilder esm2_t33_650M_UR50D-contact-regression.pt https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt -
unimomo       model.ckpt             https://github.com/kxz18/UniMoMo/releases/download/v1.0/model.ckpt         -
WEOF
)}"

  # ------------------------------------------------------------- RFD3 preflight
  PREFLIGHT_EXPERIMENT="${PREFLIGHT_EXPERIMENT:-lp_preflight}"
  PREFLIGHT_DIR="${PREFLIGHT_DIR:-${LPGEN_RUNS_DIR}/${RUN_ID}_rfd3_preflight}"
  EXTRA_PREFLIGHT_ARGS="${EXTRA_PREFLIGHT_ARGS:-}"

  # ----------------------------------------------------------- RFD3 training
  TRAIN_EXPERIMENT="${TRAIN_EXPERIMENT:-lp_finetune}"
  # Keyed on a NAME, not a timestamp, so resubmitting the same name resumes.
  TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-lp_scratch_v1}"
  # Model-only init from the released RFD3 weights. They ARE available -- `foundry install
  # rfd3` via `--submit weights --set STAGE_ARMS=foundry` puts
  # ${WEIGHTS_DIR}/foundry/rfd3_latest.ckpt (2.69 GB) in place -- so a from-scratch run is
  # now a deliberate choice, not the only option. The plan asks for the fine-tune.
  # Empty still means from scratch; the train job prints the exact path if it finds the
  # checkpoint unused, rather than adopting it silently.
  TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-}"
  EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

  # ------------------------------------------- RFD3 re-evaluation of saved checkpoints
  # Re-scores checkpoints that already exist, so a measurement fix never costs retraining.
  # The mechanism is the epoch-stop behaviour that is normally a TRAP: loading a checkpoint
  # does `current_epoch += 1`, then `current_epoch >= max_epochs` sets should_stop -- but
  # `prevalidate` runs at trainers/fabric.py:447, BEFORE the `while not should_stop` loop at
  # :459. So ckpt + prevalidate + a small max_epochs validates once and trains nothing.
  REVAL_ROOT="${REVAL_ROOT:-${LPGEN_RUNS_DIR}/${TRAIN_RUN_NAME}_reval}"
  # The corrected panel. The ORI (where the diffusion prior is dropped) is derived from
  # RECEPTOR pocket atoms, not the native peptide COM: see lpgen_build_val_contigs.py.
  # Default deliberately NOT the in-loop file -- a running job holds that one, and
  # rewriting it in place would put two ORI conventions in one curve.
  REVAL_CONTIGS="${REVAL_CONTIGS:-${MANIFEST_DIR}/lpgen_val_contigs_pocket.json}"
  # Which checkpoints to re-score. Empty = every epoch-*.ckpt in the run's ckpt dir.
  REVAL_CKPT_DIR="${REVAL_CKPT_DIR:-${LPGEN_RUNS_DIR}/${TRAIN_RUN_NAME}/ckpt}"
  REVAL_CKPT="${REVAL_CKPT:-}"
  REVAL_SCORE_WORKERS="${REVAL_SCORE_WORKERS:-16}"
  EXTRA_REVAL_ARGS="${EXTRA_REVAL_ARGS:-}"

  MANIFEST_WORKERS="${MANIFEST_WORKERS:-16}"
  # >0 processes only the first N rows: a smoke run, and the summary says so.
  MANIFEST_LIMIT="${MANIFEST_LIMIT:--1}"
}

# -------------------------------------------------------------------- banner
# sacct is unreachable here (slurmdbd refuses the connection and sacct still exits 0), so
# the node and the resource request are recoverable ONLY from this .out. Print them.
_lpgen_banner() {
  local stage="${1:-?}"
  echo "=================================================================="
  echo "[lpgen:${stage}] job=${SLURM_JOB_ID:-none} host=$(hostname)"
  echo "[lpgen:${stage}] cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M partition=${SLURM_JOB_PARTITION:-?}"
  echo "[lpgen:${stage}] repo=${REPO}"
  echo "[lpgen:${stage}] py=${PY}"
  echo "[lpgen:${stage}] run_id=${RUN_ID}"
  echo "[lpgen:${stage}] date=$(date -Is)"
  nvidia-smi -L 2>/dev/null || echo "[lpgen:${stage}] no GPU (expected on the cpu partition)"
  echo "=================================================================="
}

# ------------------------------------------------------------------ preflight
_lpgen_preflight() {
  local ok=0
  [[ -x "${PY}" ]] || { echo "PREFLIGHT: python not executable: ${PY}" >&2; ok=1; }
  [[ -f "${LP_TRAIN}" ]] || { echo "PREFLIGHT: missing LP_TRAIN: ${LP_TRAIN}" >&2; ok=1; }
  [[ -f "${LP_VAL}" ]]   || { echo "PREFLIGHT: missing LP_VAL: ${LP_VAL}" >&2; ok=1; }
  return $ok
}

# Gate for stages that consume the manifest. A stage whose gate failed must print why and
# exit 0 -- a non-zero exit strands every afterok dependent in DependencyNeverSatisfied
# with no explanation, which is indistinguishable from "upstream never ran".
_lpgen_require_manifest() {
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "GATE: manifest absent: ${MANIFEST}" >&2
    echo "GATE: build it first -> scripts/submit_lpgen.sh --submit manifest" >&2
    return 1
  fi
  return 0
}

# ------------------------------------------------------------------ arm registry
# Upstream URL per arm. Kept here rather than in the sbatch so the submitter, the staging
# job and a human all read one list.
_lpgen_arm_url() {
  case "$1" in
    foundry) echo "https://github.com/RosettaCommons/foundry.git" ;;   # RFdiffusion3
    # DiffPepBuilder (de novo) + DiffPepDock (sequence-conditioned docking). The arm whose
    # CONDITIONING GRANULARITY matches the eventual task: the peptide sequence enters as a
    # PER-RESIDUE channel (aatype token in model/embedding.py, and a per-residue ESM
    # embedding in data/pdb_data_loader.py) while `diffused_mask = ligand_mask` keeps the
    # WHOLE peptide diffused -- so identities can be pinned without freezing the pose.
    diffpepbuilder) echo "https://github.com/YuzheWangPKU/DiffPepBuilder.git" ;;
    # PepGLAD's own successor, same author, trained on OUR PepBench+ProtFrag and tested on
    # LNR. Kept as a CALIBRATION arm only: api/templates/pep.py builds the peptide as
    # all-GLY dummy blocks with one C atom each, so there is NO slot to supply residue
    # identities and the only similarity dial is latent noise.
    unimomo) echo "https://github.com/kxz18/UniMoMo.git" ;;
    ppflow)  echo "https://github.com/EDAPINENUT/ppflow.git" ;;
    pepglad) echo "https://github.com/THUNLP-MT/PepGLAD.git" ;;
    potflow) echo "https://github.com/CMACH508/potflow.git" ;;         # parked
    *)       echo "" ;;
  esac
}

# ------------------------------------------------------- per-arm env recipe
# Constraints are foundry-specific and MUST NOT leak onto the other arms: `atomworks<3`
# is meaningless for an arm that never imports it, and a stray `-c ci-constraints.txt`
# would silently pin torch 2.7.1 over an arm's own tested version.
_lpgen_arm_constraints_file() {
  case "$1" in
    foundry) echo "${ARM_CONSTRAINTS_FILE}" ;;
    *)       echo "" ;;
  esac
}
_lpgen_arm_extra_constraints() {
  case "$1" in
    foundry) echo "${ARM_EXTRA_CONSTRAINTS}" ;;
    *)       echo "" ;;
  esac
}

# Python version per arm. DiffPepBuilder's environment.yml pins python=3.9.16 and vendors
# openfold, which is version-sensitive; UniMoMo ships env_cuda121.yaml on 3.9 too.
_lpgen_arm_python() {
  case "$1" in
    diffpepbuilder) echo "3.9" ;;
    unimomo)        echo "3.9" ;;
    *)              echo "3.12" ;;
  esac
}

# Requirement lines for arms whose only spec is a fully-pinned `conda env export`.
#
# We do NOT run `conda env create -f environment.yml` for these. That export pins build
# strings from the `defaults` channel (openssl=3.0.12=h7f8727e_0, ca-certificates=
# 2023.7.22, cudatoolkit=10.2.89 alongside cuda-runtime=11.8) which no longer resolve
# together, and it is linux-64//build-specific rather than portable. The `pip:` section of
# that same export IS exact and portable, so we install it verbatim and translate only the
# handful of conda-channel packages (torch stack, openmm, pdbfixer) to wheels.
#
# torch is pinned as a DIRECT WHEEL URL, not `torch==2.1.0+cu118` with an extra index.
# MEASURED (job 84019): with an extra index, uv resolves against the FIRST index that
# carries the package -- here a cu126 index -- and refuses with "no version of
# torch==2.1.0+cu118", suggesting --index-strategy unsafe-best-match. Naming the wheel
# avoids both the cross-index search and that flag's dependency-confusion exposure.
#
# The package list is derived from what the tree actually IMPORTS, not from what the export
# lists, because the export is a snapshot of someone's whole conda environment. Measured by
# grepping every .py in the tree (openfold included) for a module-level import:
#
#   DROPPED, imported NOWHERE: torchvision, torchaudio (conda installs them beside torch),
#   matplotlib (not mentioned in the repo at all), and mdtraj / biotite / easydict, which
#   appear ONLY inside the two demo notebooks' blanket `pip install` line.
#   mdtraj was not merely unused, it was the build failure: no cp39 wheel exists for it, so
#   uv compiled from source and its vendored tng_io.c fails on modern GCC
#   (-Wincompatible-pointer-types is an error now).
#
#   ADDED, missing from the export's pip section as transcribed but imported at MODULE level
#   by every entry point: pyrootutils (train.py, run_docking.py, run_inference.py,
#   process_batch_dock.py, process_receptor.py, process_dataset.py, split_dataset.py,
#   run_postprocess.py, SSbuilder.py) and wget (preprocess_utils.py). python-dotenv comes
#   with them because every one of those calls `pyrootutils.setup_root(..., dotenv=True)`.
#   Without these three the arm cannot start at all.
#   ml_collections is kept although their own code never imports it: VENDORED OPENFOLD does.
#
#   setuptools==68.0.0 is ADDED for an environment-shape reason, not an import one: a uv
#   venv ships no setuptools, while the conda env this was translated from pinned 68.0.0 --
#   so `pkg_resources` was simply absent. MEASURED (job 84128): `import wandb` (0.15.12,
#   imported at module level by experiments/train.py, which run_docking.py imports) walks
#   wandb.sdk.launch.builder.build and dies on `No module named 'pkg_resources'`. Pinned
#   rather than floated because modern setuptools is removing pkg_resources outright.
#
# Note `setup_root(indicator=[".git"], pythonpath=True, dotenv=True)`: `root` is found by
# walking up to a `.git` directory, so the staged clone's .git is load-bearing, and the
# tree ships no .env -- BASE_PATH, which the configs read as ${oc.env:BASE_PATH}, has to be
# exported by the job.
#
# PyRosetta is deliberately ABSENT. Their postprocess runs Rosetta interface analysis, but
# scoring belongs on the CPU partition through the evaluator every other arm is scored
# with -- one evaluator across arms is the whole point of the common manifest. Asking for
# a GPU to run Rosetta would just queue relaxation behind real training.
#
# openmm and pdbfixer are absent for a HARD reason, not a preference: the pins in that
# export do not exist as wheels. MEASURED against the PyPI index 2026-10-14 -- openmm's
# earliest published wheel is 8.1.1 (they pin 8.0.0) and pdbfixer publishes exactly one
# version, 1.12.0 (they pin 1.9). Both are conda-forge packages. So sampling must run with
# `postprocess.amber_relax=False`, and relaxation happens in the common CPU evaluator
# instead. An env built with those two lines present cannot resolve at all.
_lpgen_arm_requirements() {
  case "$1" in
    diffpepbuilder)
      cat <<'REQEOF'
torch @ https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp39-cp39-linux_x86_64.whl
setuptools==68.0.0
numpy==1.26.0
pandas==2.1.1
absl-py==2.0.0
antlr4-python3-runtime==4.9.3
biopython==1.81
dm-tree==0.1.8
fair-esm==2.0.0
GPUtil==1.4.0
hydra-core==1.3.2
hydra-joblib-launcher==1.2.0
joblib==1.3.2
ml-collections==0.1.1
omegaconf==2.3.0
pyrootutils==1.0.4
python-dotenv==1.0.0
scikit-learn==1.3.2
scipy==1.11.3
tmtools==0.0.3
tqdm==4.66.1
wandb==0.15.12
wget==3.2
REQEOF
      ;;
    *) echo "" ;;
  esac
}

# Editable-install root per arm (the directory holding pyproject/setup.py).
_lpgen_arm_install_target() {
  case "$1" in
    foundry) echo "." ;;      # rc-foundry is one hatch package incl. models/rfd3/src/rfd3
    *)       echo "" ;;       # other arms have no packaging; PYTHONPATH is used instead
  esac
}
