# Shared preamble for the FlowEdit LP->CP chain. Sourced by every stage's batch script.
#
# Sourcing contract (see the repo's other preambles for why it is this shape):
#   * Every argument that is a readable FILE is sourced (the submitter's env file).
#   * Every argument of the form VAR=value is exported.
#   * `sbatch --export` is NEVER used: any explicit export list sets SLURM_GET_USER_ENV=1,
#     slurmd then fails to rebuild the login environment on the compute node, and the job is
#     requeued and HELD, stranding every dependent on Dependency.
#
# A batch script cannot locate files relative to itself -- Slurm copies it to
# /var/spool/slurm/job<id>/slurm_script -- so REPO comes from SLURM_SUBMIT_DIR with an
# absolute fallback, never from BASH_SOURCE.

_fe_repo() {
  if [[ -n "${REPO:-}" && -d "${REPO}/scripts" ]]; then echo "${REPO}"; return; fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
    echo "${SLURM_SUBMIT_DIR}"; return
  fi
  echo "/zfsauton2/home/yixiz/worktrees/flowedit"
}

_fe_consume_args() {
  local a
  for a in "$@"; do
    if [[ -f "${a}" ]]; then
      echo "[preamble] sourcing env file: ${a}"
      # shellcheck disable=SC1090
      source "${a}"
    elif [[ "${a}" == *=* ]]; then
      echo "[preamble] export ${a}"
      export "${a?}"
    fi
  done
}

_fe_defaults() {
  REPO="${REPO:-$(_fe_repo)}"
  cd "${REPO}" || { echo "FATAL: cannot cd to REPO=${REPO}" >&2; exit 1; }

  # Absolute fallbacks so the preamble is self-sufficient under `env -i`: a compute node
  # gets no login environment, and a default that silently resolved to /training_runs/...
  # would make the checkpoint paths wrong rather than missing.
  ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
  PROTEINA_ZFS_PATH="${PROTEINA_ZFS_PATH:-${ZFS}/Proteina-Complexa}"
  export ZFS PROTEINA_ZFS_PATH

  # The worktree's own src must precede the venv's editable install, which points at the
  # main checkout. Without this the new `proteinfoundation.editing` package is invisible.
  export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
  PY="${PY:-${REPO}/.venv/bin/python}"

  # ---- checkpoints (the hard dependency) ----
  # The flow model MUST have been trained on the LINEAR token: FlowEdit's source field is
  # `cyclization_type_cond=LINEAR`, and a CP-only checkpoint has an untrained zero row for
  # it. `last-EMA.ckpt` keeps moving, so a run that matters pins a snapshot first with
  # FE_PIN_CKPT=1 and then reads FLOW_CKPT.
  LPMIX_RUN_NAME="${LPMIX_RUN_NAME:-cpsea_lpmix_from_v4cfg}"
  LPMIX_STORE="${LPMIX_STORE:-${PROTEINA_ZFS_PATH}/training_runs/store/${LPMIX_RUN_NAME}}"
  FLOW_CKPT="${FLOW_CKPT:-${LPMIX_STORE}/checkpoints/last-EMA.ckpt}"
  # The AE the flow was TRAINED against. Pinning a different one silently reinterprets the
  # latent track, which no downstream number can detect.
  AE_CKPT="${AE_CKPT:-${PROTEINA_ZFS_PATH}/training_runs/store/shared_ae_lpcp_128/frozen_ae.ckpt}"

  # ---- data ----
  # NOT under ${REPO}: this is a git worktree and CPSea_data is gitignored, so it exists
  # only in the main checkout. The parquet's `path` column is absolute, so the structures
  # resolve from any working directory -- but the parquet itself has to be found here.
  DATA_REPO="${DATA_REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
  # The 14 A pocket crop, not full-chain LNR: LNR ships a contiguous receptor where CPSea
  # trained on a ~14-fragment pocket, and that mismatch alone cost 22-34 points of closure.
  # The earlier edit work used the 14 A crop, so this keeps the SDEdit comparison paired.
  METADATA="${METADATA:-${DATA_REPO}/CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet}"
  CONFIG_NAME="${CONFIG_NAME:-example/training_cpsea_peptide_smoke}"
  SAMPLING_CONFIG="${SAMPLING_CONFIG:-pipeline/model_sampling}"

  # ---- output ----
  RUN_TAG="${RUN_TAG:-flowedit_$(date +%Y%m%d_%H%M%S)}"
  OUT_DIR="${OUT_DIR:-${REPO}/evaluation_results/${RUN_TAG}}"
  # One file per job. Concurrent O_APPEND to a shared file NUL-corrupts rows on this
  # filesystem (measured: 88 of 144 rows lost), so shards never share an output path.
  CAND_DIR="${CAND_DIR:-${OUT_DIR}/candidates}"
  KEEP_DIR="${KEEP_DIR:-${OUT_DIR}/kept}"

  # ---- search ----
  CYC_TYPES="${CYC_TYPES:-mainchain disulfide isopeptide}"
  T_CA="${T_CA:-0.8}"
  T_LAT="${T_LAT:-0.6 0.4}"
  EDIT_END_FRAC="${EDIT_END_FRAC:-1.0}"
  N_AVG="${N_AVG:-1}"
  NSTEPS="${NSTEPS:-0}"          # 0 = the sampling config's own value (400)
  SEEDS="${SEEDS:-$(seq 0 7 | tr '\n' ' ')}"
  BATCH_SEEDS="${BATCH_SEEDS:-8}"
  GUIDANCE_W="${GUIDANCE_W:-1.0}"
  PROTECTED="${PROTECTED:-none}"
  WRITE_IF="${WRITE_IF:-geom}"
  LIMIT="${LIMIT:-0}"
  SHARD_COUNT="${SHARD_COUNT:-1}"
  METHOD="${METHOD:-flowedit}"   # flowedit | sdedit  (the paired comparison arm)
  EXTRA_ARGS="${EXTRA_ARGS:-}"

  # ---- Rosetta ----
  ROSETTA_REPEATS="${ROSETTA_REPEATS:-3}"
  ROSETTA_SHORTLIST="${ROSETTA_SHORTLIST:-64}"
  ROSETTA_DIR="${ROSETTA_DIR:-${OUT_DIR}/rosetta}"

  export REPO DATA_REPO PY PYTHONPATH FLOW_CKPT AE_CKPT METADATA CONFIG_NAME SAMPLING_CONFIG
  export RUN_TAG OUT_DIR CAND_DIR KEEP_DIR CYC_TYPES T_CA T_LAT EDIT_END_FRAC N_AVG
  export NSTEPS SEEDS BATCH_SEEDS GUIDANCE_W PROTECTED WRITE_IF LIMIT SHARD_COUNT METHOD
  export EXTRA_ARGS ROSETTA_REPEATS ROSETTA_SHORTLIST ROSETTA_DIR
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

_fe_banner() {
  echo "=================================================================="
  echo "job        : ${SLURM_JOB_ID:-local} on $(hostname)"
  echo "stage      : ${1:-?}"
  echo "repo       : ${REPO}"
  echo "out dir    : ${OUT_DIR}"
  echo "flow ckpt  : ${FLOW_CKPT}"
  echo "ae ckpt    : ${AE_CKPT}"
  echo "metadata   : ${METADATA}"
  # Every batch script must print these: with slurmdbd unreachable there is no accounting
  # record, so the node and GPU are recoverable ONLY from the job's own log.
  echo "resources  : ${SLURM_JOB_PARTITION:-?} ${SLURM_JOB_GRES:-${SLURM_GPUS_ON_NODE:-?}} "\
       "cpus=${SLURM_CPUS_ON_NODE:-?} mem=${SLURM_MEM_PER_NODE:-?}"
  nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi -- CPU stage)"
  echo "=================================================================="
}

# Snapshot the moving `last-EMA.ckpt` to a stable, `=`-free path and echo that path.
# Resume-safe: an existing snapshot is reused, never replaced -- swapping the flow model
# mid-search would make one output file two different experiments.
_fe_pin_flow_ckpt() {
  local src="${1}" dest="${2}"
  if [[ -f "${dest}" ]]; then
    echo "[pin] reusing existing flow snapshot: ${dest}"
    [[ -f "${dest}.source" ]] && echo "[pin] from: $(cat "${dest}.source")"
    return 0
  fi
  [[ -f "${src}" ]] || { echo "FATAL: no flow checkpoint at ${src}" >&2; return 1; }
  mkdir -p "$(dirname "${dest}")"
  cp "${src}" "${dest}.tmp" && mv "${dest}.tmp" "${dest}"
  printf '%s\n' "${src}" > "${dest}.source"
  # Record the step so "which model was this?" stays answerable.
  "${PY}" - "${dest}" <<'PYEOF' 2>/dev/null || true
import sys, torch
try:
    d = torch.load(sys.argv[1], map_location="meta", weights_only=False, mmap=True)
    print(f"[pin] global_step = {d.get('global_step')}")
except Exception as e:
    print(f"[pin] (could not read global_step: {type(e).__name__})")
PYEOF
  echo "[pin] flow ckpt pinned: ${src} -> ${dest}"
}

_fe_preflight() {
  local ok=0
  [[ -x "${PY}" ]] || { echo "ERROR: no venv python at ${PY}" >&2; ok=1; }
  [[ -f "${METADATA}" ]] || { echo "ERROR: missing metadata: ${METADATA}" >&2; ok=1; }
  [[ -f "${AE_CKPT}" ]] || { echo "ERROR: missing AE checkpoint: ${AE_CKPT}" >&2; ok=1; }
  [[ -f "${FLOW_CKPT}" ]] || {
    echo "ERROR: missing flow checkpoint: ${FLOW_CKPT}" >&2
    echo "       FlowEdit needs a model TRAINED on the LINEAR token. Train it with" >&2
    echo "       scripts/submit_lp_mixing.sh --submit flow-mix, then re-run." >&2
    ok=1
  }
  return "${ok}"
}
