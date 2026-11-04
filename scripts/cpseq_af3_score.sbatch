#!/usr/bin/env bash
#SBATCH --job-name=cpseq_af3_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_af3_score_%j.out
#
# Score AF3-folded complexes with the shared evaluator. CPU only -- this never predicts,
# which is the whole reason prediction and scoring are separate jobs in this project: a
# protocol change costs CPU and not GPU-hours.
#
# Defaults score the 5TU6 preflight fold against the deposited 5TU6 crystal under the
# SAME protocol, because a dG means nothing on its own. The native column is the
# reference the AF3 number is read against, and this project has already been bitten
# once by an evaluator whose absolute numbers looked reasonable and whose native check
# had never been run.
#
# Scored under v3 by default: the published -7.82 REU baseline is a
# `v3-cis-amide-muco-windows` number, so a number meant to sit beside it has to be one
# too. v3 judges validity on the RAW prediction and reports an energy from the relaxed
# structure -- a known defect of that version, carried deliberately for comparability,
# and both verdicts are on every row.
#
# Submit:  sbatch scripts/cpseq_af3_score.sbatch
# Config:  positional args -- a path to source, or VAR=value. Never `sbatch --export`.

set -euo pipefail

for arg in "$@"; do
  if [ -f "$arg" ]; then echo "[cfg] sourcing $arg"; # shellcheck disable=SC1090
    . "$arg"
  elif [[ "$arg" == *=* ]]; then echo "[cfg] export $arg"; export "$arg"
  fi
done

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
CPSEQ="${CPSEQ:-$ZFS/cpseq}"
PY="${PY:-$REPO/.venv/bin/python}"
SCORING_VERSION="${SCORING_VERSION:-v3-cis-amide-muco-windows}"
TOLERANCES="${TOLERANCES:-$CPSEQ/gate/geometry_tolerances.json}"

PDB_ID="${PDB_ID:-5TU6}"
SEQUENCE="${SEQUENCE:-INPYLYP}"
NATIVE="${NATIVE:-$CPSEQ/controls/pdb/$(echo "$PDB_ID" | tr 'A-Z' 'a-z').pdb}"
NATIVE_PEPTIDE_CHAIN="${NATIVE_PEPTIDE_CHAIN:-B}"
NATIVE_RECEPTOR_CHAINS="${NATIVE_RECEPTOR_CHAINS:-A}"
AF3_RUN_DIR="${AF3_RUN_DIR:-$CPSEQ/af3_preflight/$PDB_ID/af3_out}"
OUT_DIR="${OUT_DIR:-$CPSEQ/af3_scores/$PDB_ID}"

echo "=== cpseq_af3_score ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=cpu cpus=${SLURM_CPUS_PER_TASK:-4} mem=24G time=2:00:00"
nvidia-smi -L 2>/dev/null | head -2 || echo "gpu:      none (CPU-only, as intended)"
echo "version:  $SCORING_VERSION"
echo "gate:     $TOLERANCES"
echo "af3 dir:  $AF3_RUN_DIR"
echo "native:   $NATIVE"

[ -x "$PY" ] || { echo "FATAL: no venv python at $PY" >&2; exit 1; }
[ -s "$TOLERANCES" ] || {
  echo "FATAL: no frozen gate at $TOLERANCES -- run scripts/cpseq_gate_rebuild.sbatch" >&2
  exit 1
}

# Every per-fold model.cif AF3 wrote. AF3 puts one per seed/sample under
# seed-<N>_sample-<K>/, plus a top-level copy of the top-ranked one; the per-fold files
# are the folds, and the top-level copy is excluded so it is not counted twice.
mapfile -t CIFS < <(find "$AF3_RUN_DIR" -type f -name '*_model.cif' -path '*seed-*' | sort)
if [ "${#CIFS[@]}" -eq 0 ]; then
  echo "[score] no per-fold model.cif under seed-*/ ; falling back to any model.cif"
  mapfile -t CIFS < <(find "$AF3_RUN_DIR" -type f -name '*_model.cif' | sort)
fi
[ "${#CIFS[@]}" -gt 0 ] || { echo "FATAL: no model.cif under $AF3_RUN_DIR" >&2; exit 1; }
echo "folds:    ${#CIFS[@]}"

ARGS=()
for c in "${CIFS[@]}"; do
  # Label each fold by its seed/sample directory, so a row is traceable to the fold it
  # came from rather than to a position in a list.
  d="$(basename "$(dirname "$c")")"
  case "$d" in seed-*) label="$d" ;; *) label="toplevel" ;; esac
  ARGS+=(--cif "${label}=${c}")
  echo "            $label  <- $c"
done

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" script_utils/cpseq_af3_score.py \
  "${ARGS[@]}" \
  --sequence "$SEQUENCE" \
  --receptor-chains A \
  --chemistry mainchain \
  --native "$NATIVE" \
  --native-peptide-chain "$NATIVE_PEPTIDE_CHAIN" \
  --native-receptor-chains "$NATIVE_RECEPTOR_CHAINS" \
  --tolerances "$TOLERANCES" \
  --scoring-version "$SCORING_VERSION" \
  --work-dir "$OUT_DIR/work" \
  --out "$OUT_DIR/scores.jsonl"
rc=$?

echo "=== cpseq_af3_score exit $rc ==="
exit $rc
