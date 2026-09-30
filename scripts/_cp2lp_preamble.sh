# Shared preamble for the CP->LP generator arms.
#
# Sourced by every scripts/cp2lp_*.sbatch. Every knob has a default here; the submitter
# writes ONE per-submission env file and passes its PATH as a positional argument.
#
# Never `sbatch --export`: any explicit export list (--export=ALL,K=v, --export=K=v and
# --export=NONE alike) sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD -- stranding every
# dependent job on Dependency with no error anyone will see.
#
# Contract:
#   _cp2lp_consume_args "$@"   # sources any arg that is a path, exports VAR=value args
#   _cp2lp_defaults            # fills in anything still unset
#   _cp2lp_banner <stage>
#   _cp2lp_preflight           # hard requirements; the CALLER decides the exit code
#   _cp2lp_gate_generated      # soft gate for downstream stages; exit 0 when unmet

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# ---------------------------------------------------------------- arg handling
_cp2lp_consume_args() {
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
_cp2lp_defaults() {
  ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
  STORE="${STORE:-${ZFS}/Proteina-Complexa/training_runs/store}"
  LPDATA="${LPDATA:-${ZFS}/LPData/preprocessed/metadata_mixed}"

  # THE PAIRING. `local_latents` is defined by the AE, so a flow checkpoint trained
  # against a different one decodes to meaningless geometry. cpsea_lpmix_from_v4cfg is the
  # only CPSea flow run trained against the shared LP/CP AE -- and it already has the
  # LINEAR topology token trained, which this arm needs.
  AE_CKPT="${AE_CKPT:-${STORE}/shared_ae_lpcp_128/frozen_ae.ckpt}"
  FLOW_CKPT="${FLOW_CKPT:-${STORE}/cpsea_lpmix_from_v4cfg/cp2lp_init.ckpt}"

  # Metadata built by script_utils/cp2lp_build_splits.py. Receptor families disjoint.
  TRAIN_META="${TRAIN_META:-${LPDATA}/cp2lp_train.parquet}"
  EVAL_META="${EVAL_META:-${LPDATA}/cp2lp_eval.parquet}"
  EVAL_REAL_META="${EVAL_REAL_META:-${LPDATA}/cp2lp_eval_real_lp.parquet}"

  ARM="${ARM:-gan}"                       # gan | contactonly
  case "${ARM}" in
    # These MUST track `run_name` in the configs: cp2lp_train.sbatch passes
    # `++run_name=${RUN_NAME}`, a Hydra override, which beats the yaml. Leaving them at a
    # previous version silently points a new config's run at the OLD run directory and
    # overwrites its checkpoints and wandb history -- the only record of the arm you are
    # comparing against.
    gan)         CONFIG_NAME="${CONFIG_NAME:-example/training_cp2lp_gan}";          RUN_NAME="${RUN_NAME:-cp2lp_gan_v2}" ;;
    contactonly) CONFIG_NAME="${CONFIG_NAME:-example/training_cp2lp_contactonly}";  RUN_NAME="${RUN_NAME:-cp2lp_contactonly_v2}" ;;
    *)           CONFIG_NAME="${CONFIG_NAME:-example/training_cp2lp_gan}";          RUN_NAME="${RUN_NAME:-cp2lp_${ARM}}" ;;
  esac

  WANDB_PROJECT="${WANDB_PROJECT:-cpsea_cp2lp}"
  WANDB_ENTITY="${WANDB_ENTITY:-zyxleo}"

  # Generation / export. Both arms MUST use the same CPs and the same budget or the
  # ablation compares sampling effort instead of comparing the objective.
  GEN_NSTEPS="${GEN_NSTEPS:-200}"
  GEN_N_CP="${GEN_N_CP:-200}"
  GEN_SAMPLES_PER_CP="${GEN_SAMPLES_PER_CP:-4}"
  GEN_SEEDS="${GEN_SEEDS:-0,1}"
  GEN_BATCH="${GEN_BATCH:-4}"

  TRIPLET_ROOT="${TRIPLET_ROOT:-${STORE}/cp2lp_triplets}"
  REFERENCE_DIR="${REFERENCE_DIR:-${TRIPLET_ROOT}/real_reference}"
  REPORT_DIR="${REPORT_DIR:-${TRIPLET_ROOT}/report}"

  PY="${PY:-${REPO}/.venv/bin/python}"
  SMOKE="${SMOKE:-0}"
  export PYTHONUNBUFFERED=1
}

_cp2lp_banner() {
  echo "=================================================================="
  # `hostname` is not on PATH on every compute node here; SLURMD_NODENAME always is.
  echo "job      : ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ [task ${SLURM_ARRAY_TASK_ID}]} on ${SLURMD_NODENAME:-$(uname -n 2>/dev/null || echo '?')}"
  echo "stage    : ${1:-?}    arm: ${ARM}"
  echo "config   : ${CONFIG_NAME}"
  echo "run_name : ${RUN_NAME}"
  echo "AE       : ${AE_CKPT}"
  echo "flow init: ${FLOW_CKPT}"
  echo "train    : ${TRAIN_META}"
  echo "eval     : ${EVAL_META}"
  echo "request  : ${SLURM_JOB_PARTITION:-?} gres=${SLURM_JOB_GRES:-?} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
  # There is no usable accounting database here, so the node and the GPU it actually got
  # are recoverable ONLY from this log. A job that dies on a card too small for it looks
  # like a silent FAILED with no traceback otherwise.
  nvidia-smi -L 2>/dev/null || echo "gpus     : (nvidia-smi unavailable -- CPU stage)"
  echo "smoke    : ${SMOKE}"
  echo "=================================================================="
}

# ------------------------------------------------------------------ preflight
# Hard requirements: this stage genuinely cannot run without them. Returns non-zero; the
# caller exits 1.
_cp2lp_preflight() {
  local ok=0
  [[ -x "${PY}" ]]           || { echo "ERROR: no venv python at ${PY}" >&2; ok=1; }
  [[ -f "${AE_CKPT}" ]]      || { echo "ERROR: missing AE checkpoint: ${AE_CKPT}" >&2; ok=1; }
  [[ -f "${TRAIN_META}" ]]   || { echo "ERROR: missing train metadata: ${TRAIN_META}" >&2
                                  echo "       build it: ${PY} script_utils/cp2lp_build_splits.py --mixed ${LPDATA}/mixed_train.parquet --out-dir ${LPDATA}" >&2; ok=1; }
  return "${ok}"
}

# The flow init is checked separately: a training job cannot start without it, but a
# generate/report job does not need it at all.
_cp2lp_preflight_flow_ckpt() {
  [[ -f "${FLOW_CKPT}" ]] || {
    echo "ERROR: missing flow init checkpoint: ${FLOW_CKPT}" >&2
    echo "       This must be a CPSea flow run trained against ${AE_CKPT}." >&2
    return 1
  }
  # A checkpoint whose AE differs from the one being loaded is the failure mode that
  # produces plausible-looking losses and meaningless geometry, so say which pair is in
  # use rather than only that the file exists.
  echo "[preflight] flow init: ${FLOW_CKPT}"
  echo "[preflight] AE       : ${AE_CKPT}"
  return 0
}

# ----------------------------------------------------------------------- gates
# A stage whose UPSTREAM produced nothing must print why and exit 0. A non-zero exit
# strands every afterok dependent in DependencyNeverSatisfied with no explanation, which
# is strictly worse than a job that ran and said "nothing to do".
_cp2lp_gate_generated() {
  local d="${1}"
  if [[ ! -d "${d}" ]]; then
    echo "GATE: ${d} does not exist -- the generate stage never ran. Nothing to report."
    return 1
  fi
  local n
  n="$(find "${d}" -name 'triplets*.jsonl' -size +0 2>/dev/null | wc -l)"
  if [[ "${n}" -eq 0 ]]; then
    echo "GATE: no non-empty triplets*.jsonl under ${d} -- the generate stage produced nothing."
    return 1
  fi
  echo "[gate] found ${n} triplet manifest(s) under ${d}"
  return 0
}
