#!/bin/bash
#SBATCH --job-name=lp_build_data
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 1 (CPU): build the linear-peptide training data, then the mixed metadata.
#
# Nothing here needs a GPU -- it reads PDBs and writes parquet -- so asking for one would
# only queue this behind real work. 96G is for the mixed-metadata step: CPSea's full train
# metadata is 2.44M rows, which is why that step streams row groups instead of using
# pandas (a plain read OOMs a 32G box).
#
# Do NOT sbatch directly -- use scripts/submit_lp_mixing.sh, which writes the env file.
#
#   sbatch lp_build_data.sbatch /path/to/env_file [VAR=value ...]

# In the job body, not only the preamble: if the source below fails, an unguarded body
# keeps running under the system python and exits 0, releasing every afterok dependent
# behind a stage that did no work.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_lp_mixing_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "${REPO}"
mkdir -p slurm_logs

# shellcheck source=_lp_mixing_preamble.sh
source "${REPO}/scripts/_lp_mixing_preamble.sh"
_lp_consume_args "$@"
# env.sh BEFORE the env file's own values are re-asserted: env.sh unconditionally exports
# some of the same names, so sourcing it second would clobber this submission's choices.
# shellcheck source=/dev/null
source "${REPO}/env.sh"
_lp_consume_args "$@"
_lp_defaults
_lp_banner "data"

if ! _lp_preflight_data; then
  echo "GATE FAILED: inputs are not in place; not building data." >&2
  echo "GATE FAILED: exiting 0 so dependents report a reason instead of DependencyNeverSatisfied." >&2
  exit 0
fi

PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

LIMIT_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  echo "SMOKE=1: processing a small subset only. The resulting metadata is NOT trainable data."
  LIMIT_ARGS=(--limit 400)
fi
SKIP_ARGS=()
[[ "${SKIP_LNR_FILTER:-0}" == "1" ]] && SKIP_ARGS=(--skip-lnr-filter)

echo "--- [1/2] preprocess PepBench + ProtFrag -> ${LP_OUT_DIR}"
"${PY}" -m script_utils.preprocess_lp \
  --pepbench-root "${PEPBENCH_ROOT}" \
  --protfrag-root "${PROTFRAG_ROOT}" \
  --lnr-root "${LNR_ROOT}" \
  --out-dir "${LP_OUT_DIR}" \
  --peptide-min-length "${PEPTIDE_MIN}" \
  --peptide-max-length "${PEPTIDE_MAX}" \
  --lnr-identity-threshold "${LNR_IDENTITY_THRESHOLD}" \
  --lnr-kmer-size "${LNR_KMER_SIZE}" \
  --lnr-containment-prefilter "${LNR_CONTAINMENT_PREFILTER}" \
  --nproc "${SLURM_CPUS_PER_TASK:-16}" \
  "${LIMIT_ARGS[@]}" "${SKIP_ARGS[@]}"

# ProtFrag ships no validation split, so lp_val.parquet exists only if PepBench's
# valid.txt produced in-range rows. Pass it only when it is there; the builder warns and
# falls back to a CPSea-only val otherwise.
LP_VAL_ARGS=()
[[ -f "${LP_OUT_DIR}/metadata/lp_val.parquet" ]] && LP_VAL_ARGS=(--lp-val "${LP_OUT_DIR}/metadata/lp_val.parquet")

echo "--- [2/2] mix with CPSea -> ${MIXED_META_DIR}"
"${PY}" -m script_utils.build_mixed_metadata \
  --cpsea-train "${CPSEA_FULL_META}/cpsea_train.parquet" \
  --cpsea-val "${CPSEA_FULL_META}/cpsea_val.parquet" \
  --lp-train "${LP_OUT_DIR}/metadata/lp_train.parquet" \
  "${LP_VAL_ARGS[@]}" \
  --out-dir "${MIXED_META_DIR}" \
  --val-cpsea-sample "${VAL_CPSEA_SAMPLE}"

echo "--- done. Composition:"
cat "${MIXED_META_DIR}/mix_manifest.json"
