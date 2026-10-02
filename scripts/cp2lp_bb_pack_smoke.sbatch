#!/bin/bash
#SBATCH --job-name=cp2lp_bb_pack_smoke
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Both backbone arms plus a COMPLETE Rosetta packing example per arm, on real native
# complexes. CPU: the arms run on random head outputs here, so no trunk and no GPU are
# involved -- this smoke is about the representation and packing plumbing.
#
# This is the job that decides whether the experiment can be reported as holistic or only as
# backbone-only. If packing fails here it fails for the real evaluation too.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
_cp2lp_defaults
_cp2lp_banner "bb-pack-smoke"

cd "$REPO"

NATIVE_ROOT="${NATIVE_ROOT:-${REPO}/calibration_native}"
SMOKE_N="${SMOKE_N:-3}"
PACK_SEEDS="${PACK_SEEDS:-3}"
SMOKE_DIR="${SMOKE_DIR:-${TRIPLET_ROOT}/bb_pack_smoke/${SLURM_JOB_ID:-local}}"

if [[ ! -d "${NATIVE_ROOT}" ]]; then
  echo "GATE: no native complexes at ${NATIVE_ROOT} -- nothing to smoke."
  exit 0
fi

mkdir -p "${SMOKE_DIR}"

ARGS=(
  --native-root "${NATIVE_ROOT}"
  --n-complexes "${SMOKE_N}"
  --pack-seeds "${PACK_SEEDS}"
  --work-dir "${SMOKE_DIR}/candidates"
  --out "${SMOKE_DIR}/report.json"
)
[[ "${SKIP_PACKING:-0}" == "1" ]] && ARGS+=(--skip-packing)

echo "[smoke] unit tests first -- a broken invariant makes the rest meaningless"
"${PY}" -m pytest -q --no-header -p no:warnings \
  script_utils/test_cp2lp_backbone.py \
  script_utils/test_cp2lp_bb_arms.py \
  script_utils/test_cp2lp_bb_losses.py \
  script_utils/test_cp2lp_bb_discriminator.py

echo
echo "[smoke] arms + packing"
"${PY}" script_utils/cp2lp_bb_pack_smoke.py "${ARGS[@]}"

echo "[done] report at ${SMOKE_DIR}/report.json"
