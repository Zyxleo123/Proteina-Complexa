# Shared preamble for the provenance-paired LP -> CP dataset build.
#
# Sourced by every scripts/lp2cp_pair_*.sbatch. Every knob has a default here; the
# submitter writes ONE per-submission env file and passes its PATH as a positional
# argument. Never `sbatch --export`: any explicit export list (--export=ALL,K=v and
# --export=NONE alike) sets SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD -- stranding every
# dependent on Dependency.
#
# Contract:
#   _p2_consume_args "$@"   # sources any arg that is a path, exports VAR=value args
#   _p2_defaults            # fills in anything still unset
#   _p2_banner <stage>
#   _p2_preflight           # data roots + venv; returns non-zero on a real misconfig

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# ---------------------------------------------------------------- arg handling
_p2_consume_args() {
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
_p2_defaults() {
  ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
  PROTEINA_ZFS_PATH="${PROTEINA_ZFS_PATH:-${ZFS}/Proteina-Complexa}"

  # A smoke gets its OWN output root. Sharing one with a real build lets a 200-parent
  # smoke overwrite a 20,000-parent dataset's work lists, and the corruption is silent
  # because every path still resolves.
  if [[ "${SMOKE:-0}" == "1" ]]; then
    PAIR_ROOT="${PAIR_ROOT:-${ZFS}/LP2CP_pairs_smoke}"
    PARENT_LIMIT="${PARENT_LIMIT:-200}"
    FETCH_SHARDS="${FETCH_SHARDS:-2}"
    PAIR_SHARDS="${PAIR_SHARDS:-2}"
    REF_SAMPLE="${REF_SAMPLE:-200}"
    # Bound the metadata read too, or a 200-parent smoke still pulls 2.4M rows into pandas
    # and needs the full 64G. These are the FIRST rows in file order, which is an
    # alphabetical slice and not a sample -- fine for exercising the machinery, never for
    # building a dataset or reading a distribution.
    MAX_ROWS="${MAX_ROWS:-200000}"
  fi

  PAIR_ROOT="${PAIR_ROOT:-${ZFS}/LP2CP_pairs}"
  PAIR_WORK="${PAIR_WORK:-${PAIR_ROOT}/work}"
  MODELS_DIR="${MODELS_DIR:-${PAIR_ROOT}/afdb_models}"
  PAIRS_DIR="${PAIRS_DIR:-${PAIR_ROOT}/pairs}"
  LP_PDB_DIR="${LP_PDB_DIR:-${PAIR_ROOT}/lp_pdb}"
  FETCH_LOG_DIR="${FETCH_LOG_DIR:-${PAIR_ROOT}/fetch_manifests}"
  REPORT_DIR="${REPORT_DIR:-${PAIR_ROOT}/report}"

  # ---- CPSea side ----
  # The `full` root, matching every recent CPSea flow run and the pinned AE. NOT
  # $CPSEA_DATA_PATH, which is the smaller CPSea_PDB tree with different example ids.
  CPSEA_FULL_META="${CPSEA_FULL_META:-${ZFS}/CPSea/CPSea_full/CPSea/preprocessed/metadata}"
  CPSEA_SPLITS="${CPSEA_SPLITS:-cpsea_train}"

  # ---- selection ----
  PARENT_LIMIT="${PARENT_LIMIT:-20000}"
  CYC_TYPES="${CYC_TYPES:-all}"
  CLUSTER_UNIQUE="${CLUSTER_UNIQUE:-1}"
  MAX_ROWS="${MAX_ROWS:-0}"  # 0 = read the whole metadata table

  # ---- fetch ----
  # v4 is what CPSea's lineage used: the ID lists shipped in La-Proteina_AFDB_IDs.zip are
  # named `-model_v4`. A different version is a different structure, so changing this
  # changes what "the parent" means.
  AFDB_VERSION="${AFDB_VERSION:-v4}"
  FETCH_SHARDS="${FETCH_SHARDS:-8}"
  FETCH_TIMEOUT="${FETCH_TIMEOUT:-30}"
  FETCH_RETRIES="${FETCH_RETRIES:-2}"
  FETCH_SLEEP="${FETCH_SLEEP:-0.05}"

  # ---- pair ----
  PAIR_SHARDS="${PAIR_SHARDS:-8}"
  MIN_ANCHOR="${MIN_ANCHOR:-10}"

  # ---- diagnosis references ----
  LP_METADATA="${LP_METADATA:-${ZFS}/LPData/preprocessed/metadata/lp_train.parquet}"
  LNR_ROOT="${LNR_ROOT:-${ZFS}/LNR}"
  REF_SAMPLE="${REF_SAMPLE:-3000}"

  PY="${PY:-${REPO}/.venv/bin/python}"
  export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

_p2_banner() {
  echo "=================================================================="
  echo "job        : ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ [task ${SLURM_ARRAY_TASK_ID}]} on $(hostname)"
  echo "stage      : ${1:-?}"
  echo "repo       : ${REPO}"
  echo "pair root  : ${PAIR_ROOT}"
  echo "models     : ${MODELS_DIR}"
  echo "pairs      : ${PAIRS_DIR}"
  echo "selection  : parent_limit=${PARENT_LIMIT} cyc=${CYC_TYPES} max_rows=${MAX_ROWS}"
  echo "shards     : fetch=${FETCH_SHARDS} pair=${PAIR_SHARDS}"
  echo "smoke      : ${SMOKE:-0}"
  echo "=================================================================="
}

# ------------------------------------------------------------------ preflight
_p2_preflight() {
  local ok=0
  [[ -x "${PY}" ]] || { echo "ERROR: no venv python at ${PY}" >&2; ok=1; }

  local split f
  for split in ${CPSEA_SPLITS}; do
    f="${CPSEA_FULL_META}/${split}.parquet"
    [[ -f "${f}" ]] || { echo "ERROR: missing CPSea metadata: ${f}" >&2; ok=1; }
  done
  return "${ok}"
}

# Guard against re-sharding an existing build. `shard_of` partitions by a hash modulo the
# shard count, so changing PAIR_SHARDS half way through reassigns every example: the
# resume check then finds nothing done, and pairs already built get written a second time
# into a different shard. Both copies look valid.
_p2_assert_shard_count() {
  local dir="$1" want="$2" stamp="$1/.num_shards"
  mkdir -p "${dir}"
  if [[ -f "${stamp}" ]]; then
    local have; have="$(cat "${stamp}")"
    if [[ "${have}" != "${want}" ]]; then
      echo "FATAL: ${dir} was built with ${have} shards, this job wants ${want}." >&2
      echo "       Re-sharding duplicates every pair. Either keep the original count" >&2
      echo "       or start a fresh PAIR_ROOT." >&2
      return 1
    fi
  else
    echo "${want}" > "${stamp}"
  fi
  return 0
}
