# Shared preamble for the LP-mixing pipeline (linear peptide data mixed into CPSea).
#
# Sourced by every scripts/lp_*.sbatch. Every knob has a default here; the submitter
# writes ONE per-submission env file and passes its PATH as a positional argument.
# Never `sbatch --export`: any explicit export list sets SLURM_GET_USER_ENV=1, slurmd
# then fails to rebuild the login environment on the compute node, and the job is
# requeued and HELD -- stranding every dependent on Dependency.
#
# Contract:
#   _lp_consume_args "$@"   # sources any arg that is a path, exports VAR=value args
#   _lp_defaults            # fills in anything still unset
#   _lp_preflight_data      # gate for the data-prep stage
#   _lp_preflight_train     # gate before burning a GPU allocation

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# ---------------------------------------------------------------- arg handling
_lp_consume_args() {
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
_lp_defaults() {
  ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
  PROTEINA_ZFS_PATH="${PROTEINA_ZFS_PATH:-${ZFS}/Proteina-Complexa}"

  # ---- raw inputs (as downloaded from the PepBench Zenodo record) ----
  LP_RAW_ROOT="${LP_RAW_ROOT:-${ZFS}/LPData}"
  PEPBENCH_ROOT="${PEPBENCH_ROOT:-${LP_RAW_ROOT}/PepBench/train_valid}"
  PROTFRAG_ROOT="${PROTFRAG_ROOT:-${LP_RAW_ROOT}/ProtFrag}"
  LNR_ROOT="${LNR_ROOT:-${ZFS}/LNR}"

  # ---- preprocessed outputs ----
  LP_OUT_DIR="${LP_OUT_DIR:-${LP_RAW_ROOT}/preprocessed}"
  MIXED_META_DIR="${MIXED_META_DIR:-${LP_OUT_DIR}/metadata_mixed}"

  # ---- CPSea side ----
  # The `full` root, matching what every recent CPSea flow run and the pinned AE
  # (finetune_full_128) used. NOT $CPSEA_DATA_PATH, which is the smaller CPSea_PDB tree.
  CPSEA_FULL_META="${CPSEA_FULL_META:-${ZFS}/CPSea/CPSea_full/CPSea/preprocessed/metadata}"

  # ---- peptide length window ----
  # Must match CroppingTransform2's binder_min/max_length in the training configs, or
  # rows are written that the cropper will then reject at load time.
  PEPTIDE_MIN="${PEPTIDE_MIN:-5}"
  PEPTIDE_MAX="${PEPTIDE_MAX:-16}"

  # ---- LNR holdout ----
  LNR_IDENTITY_THRESHOLD="${LNR_IDENTITY_THRESHOLD:-0.40}"
  LNR_KMER_SIZE="${LNR_KMER_SIZE:-6}"
  LNR_CONTAINMENT_PREFILTER="${LNR_CONTAINMENT_PREFILTER:-0.20}"

  # Rows kept from CPSea's 134k-row val split when building the MIXED val file. The AE
  # validates on this mix; the flow arms filter it back down to cpsea-only. Small so
  # validation is fast and the linear side is actually visible.
  VAL_CPSEA_SAMPLE="${VAL_CPSEA_SAMPLE:-2000}"

  # ---- training ----
  STAGE="${STAGE:-}"
  AE_CONFIG="${AE_CONFIG:-training_ae_shared_lpcp}"
  AE_RUN_NAME="${AE_RUN_NAME:-shared_ae_lpcp_128}"
  # The AE checkpoint both flow arms load. Written by _lp_pin_shared_ae into a path with
  # no `=` characters (Hydra's override grammar mis-parses those, and AE checkpoint
  # filenames contain `epoch=...step=...`).
  SHARED_AE_CKPT="${SHARED_AE_CKPT:-${PROTEINA_ZFS_PATH}/training_runs/store/${AE_RUN_NAME}/frozen_ae.ckpt}"
  # Which checkpoint inside the AE run dir to pin. EMA, matching the pin on
  # finetune_full_128 -- the flow model regresses the AE's latents, so it must load the
  # same weights the latents were defined by.
  SHARED_AE_SOURCE="${SHARED_AE_SOURCE:-}"

  MIX_CONFIG="${MIX_CONFIG:-example/training_cpsea_lpmix_from_v4cfg}"
  CTRL_CONFIG="${CTRL_CONFIG:-example/training_cpsea_lpctrl_from_v4cfg}"
  MIX_RUN_NAME="${MIX_RUN_NAME:-cpsea_lpmix_from_v4cfg}"
  CTRL_RUN_NAME="${CTRL_RUN_NAME:-cpsea_lpctrl_from_v4cfg}"

  WANDB_PROJECT_AE="${WANDB_PROJECT_AE:-cpsea_ae}"
  WANDB_PROJECT_FLOW="${WANDB_PROJECT_FLOW:-cpsea_lp_mixing}"

  # true so a preempted / time-limited job resumes from last.ckpt on requeue rather than
  # restarting at step 0. These runs are days long; they WILL hit the 2-day wall.
  RESUME_FROM_LAST="${RESUME_FROM_LAST:-true}"

  SMOKE="${SMOKE:-0}"

  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

_lp_banner() {
  echo "=================================================================="
  echo "job        : ${SLURM_JOB_ID:-local} on $(hostname)"
  echo "stage      : ${1:-?}"
  echo "repo       : ${REPO}"
  echo "lp out     : ${LP_OUT_DIR}"
  echo "mixed meta : ${MIXED_META_DIR}"
  echo "=================================================================="
}

# ------------------------------------------------------------------ preflight
_lp_preflight_data() {
  local ok=0
  [[ -x "${REPO}/.venv/bin/python" ]] || { echo "ERROR: no venv python at ${REPO}/.venv/bin/python" >&2; ok=1; }

  local d
  for d in "${PEPBENCH_ROOT}" "${PROTFRAG_ROOT}" "${LNR_ROOT}"; do
    [[ -d "${d}" ]] || { echo "ERROR: missing input dir: ${d}" >&2; ok=1; }
  done
  local f
  for f in "${PEPBENCH_ROOT}/train.txt" "${PEPBENCH_ROOT}/valid.txt" \
           "${PROTFRAG_ROOT}/all.txt" "${LNR_ROOT}/test.txt"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing index file: ${f}" >&2; ok=1; }
  done
  for f in "${CPSEA_FULL_META}/cpsea_train.parquet" "${CPSEA_FULL_META}/cpsea_val.parquet"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing CPSea metadata: ${f}" >&2; ok=1; }
  done

  # Refusing to run without the LNR filter is the point of having a gate here: LNR is the
  # benchmark these runs will be judged on, and a leaked training set makes every number
  # downstream meaningless in a way no later check can detect.
  if [[ "${SKIP_LNR_FILTER:-0}" == "1" ]]; then
    echo "WARNING: SKIP_LNR_FILTER=1 -- LNR will be LEAKED into training data." >&2
    echo "WARNING: any LNR result from the resulting models is invalid." >&2
  fi
  return "${ok}"
}

_lp_preflight_train() {
  local ok=0
  [[ -x "${REPO}/.venv/bin/python" ]] || { echo "ERROR: no venv python at ${REPO}/.venv/bin/python" >&2; ok=1; }

  local f
  for f in "${MIXED_META_DIR}/mixed_train.parquet" "${MIXED_META_DIR}/mixed_val.parquet"; do
    [[ -f "${f}" ]] || {
      echo "ERROR: missing mixed metadata: ${f}" >&2
      echo "       Build it first: bash scripts/submit_lp_mixing.sh --submit data" >&2
      ok=1
    }
  done

  if [[ -z "${WANDB_API_KEY:-}" || ${#WANDB_API_KEY} -lt 40 ]]; then
    echo "ERROR: WANDB_API_KEY missing/invalid (set in ~/.bashrc or .env)" >&2
    ok=1
  fi

  # Prove the mixed metadata actually carries the sources the configs weight, and that
  # every path it points at exists. A source name typo raises at setup() -- after the
  # job has queued, allocated a GPU and loaded a checkpoint.
  if [[ ${ok} -eq 0 ]]; then
    "${REPO}/.venv/bin/python" - "${MIXED_META_DIR}/mixed_train.parquet" "${MIXED_META_DIR}/mixed_val.parquet" <<'PY' || ok=1
import sys
from pathlib import Path

import pyarrow.parquet as pq

EXPECTED = {"cpsea", "pepbench", "protfrag"}
bad = False
for split, path in zip(("train", "val"), sys.argv[1:3]):
    pf = pq.ParquetFile(path)
    if "dataset_source" not in pf.schema_arrow.names:
        print(f"ERROR: {path} has no dataset_source column", file=sys.stderr)
        bad = True
        continue
    tbl = pq.read_table(path, columns=["dataset_source"])
    counts = tbl.column("dataset_source").value_counts()
    got = {c["values"].as_py(): c["counts"].as_py() for c in counts}
    print(f"[preflight] {split}: {pf.metadata.num_rows:,} rows | {got}")
    if split == "train" and set(got) != EXPECTED:
        print(
            f"ERROR: train sources {sorted(got)} != {sorted(EXPECTED)}. The training configs' "
            "source_fractions must name every source present, or the run dies at setup().",
            file=sys.stderr,
        )
        bad = True

# Spot-check that the paths resolve. A metadata file built against a since-moved
# processed/ tree fails one example at a time, deep inside a dataloader worker.
tbl = pq.read_table(sys.argv[1], columns=["path", "dataset_source"])
paths, srcs = tbl.column("path").to_pylist(), tbl.column("dataset_source").to_pylist()
by_src: dict[str, list[str]] = {}
for p, s in zip(paths, srcs):
    if len(by_src.setdefault(s, [])) < 8:
        by_src[s].append(p)
for s, ps in sorted(by_src.items()):
    missing = [p for p in ps if not Path(p).is_file()]
    if missing:
        print(f"ERROR: {len(missing)}/{len(ps)} sampled {s} paths missing, e.g. {missing[0]}", file=sys.stderr)
        bad = True
    else:
        print(f"[preflight] {s}: {len(ps)}/{len(ps)} sampled structure paths resolve")
raise SystemExit(1 if bad else 0)
PY
  fi
  return "${ok}"
}

# Snapshot the AE checkpoint the flow arms load, to a path free of `=` characters.
# Resume-safe: an existing snapshot is reused, never silently replaced -- swapping the AE
# mid-run would change the local_latents regression target underneath the model.
_lp_pin_shared_ae() {
  local run_dir="${PROTEINA_ZFS_PATH}/training_runs/store/${AE_RUN_NAME}"
  local dest="${SHARED_AE_CKPT}"

  if [[ -f "${dest}" ]]; then
    echo "[pin] reusing existing shared AE snapshot: ${dest}"
    [[ -f "${dest}.source" ]] && echo "[pin] originally copied from: $(cat "${dest}.source")"
    return 0
  fi

  local src="${SHARED_AE_SOURCE}"
  if [[ -z "${src}" ]]; then
    # Prefer a numbered EMA checkpoint over last-EMA: `last` keeps moving, so pinning it
    # makes "which AE was this?" unanswerable a week later.
    src="$(ls -1 "${run_dir}"/checkpoints/chk_*-EMA.ckpt 2>/dev/null | sort | tail -1 || true)"
    [[ -z "${src}" ]] && src="$(ls -1 "${run_dir}"/checkpoints/last*-EMA.ckpt 2>/dev/null | tail -1 || true)"
  fi
  if [[ -z "${src}" || ! -f "${src}" ]]; then
    echo "ERROR: no shared-AE checkpoint found under ${run_dir}/checkpoints" >&2
    echo "       Train it first: bash scripts/submit_lp_mixing.sh --submit ae" >&2
    echo "       Or pin one explicitly with SHARED_AE_SOURCE=/abs/path.ckpt" >&2
    return 1
  fi

  mkdir -p "$(dirname "${dest}")"
  cp "${src}" "${dest}.tmp" && mv "${dest}.tmp" "${dest}"
  printf '%s\n' "${src}" > "${dest}.source"
  echo "[pin] shared AE pinned: ${src}"
  echo "[pin]                -> ${dest}"
}
