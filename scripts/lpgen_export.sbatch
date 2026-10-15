#!/bin/bash
#SBATCH --job-name=lpgen_export
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_export_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_export_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
# Export the common manifest into the per-complex layout PPFlow and POTFlow read:
# one directory per complex, holding peptide.pdb and a 10 A pocket crop.
#
# Pure I/O and numpy distance work; no GPU.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "export"
_lpgen_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

# A gate, not a crash: print why and exit 0 so an afterok dependent is not stranded in
# DependencyNeverSatisfied with no explanation.
if ! _lpgen_require_manifest; then
  echo "[export] GATE not satisfied -- nothing exported, exiting 0 so dependents are not stranded"
  exit 0
fi

cd "$REPO"
mkdir -p "$NATIVE_DIR"

ARGS=(
  --manifest "$MANIFEST"
  --out-dir "$NATIVE_DIR"
  --splits "$EXPORT_SPLITS"
  --pocket-cutoff-A "$POCKET_CUTOFF_A"
  --workers "$EXPORT_WORKERS"
  --limit "$EXPORT_LIMIT"
)
[[ "$PPFLOW_ALIASES" == "1" ]] && ARGS+=(--ppflow-aliases)
[[ "$EXPORT_CLEAN" == "1" ]] && ARGS+=(--clean)

echo "[export] self-test"
"$PY" script_utils/test_lpgen_export_native.py

echo "[export] exporting"
"$PY" script_utils/lpgen_export_native.py "${ARGS[@]}"

echo "[export] done -> $NATIVE_DIR"
for s in ${EXPORT_SPLITS//,/ }; do
  if [[ -d "$NATIVE_DIR/$s" ]]; then
    echo "[export]   $s: $(find "$NATIVE_DIR/$s" -mindepth 1 -maxdepth 1 -type d | wc -l) complexes"
  fi
done
