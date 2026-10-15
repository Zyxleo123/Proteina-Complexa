#!/bin/bash
#SBATCH --job-name=lpgen_manifest
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_manifest_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_manifest_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
# Build the ONE common LP manifest every generator arm trains against.
#
# Pure I/O: it reads 45,708 small PDBs (1.7 GB) to recover the peptide sequence that
# preprocessing never recorded, then selects rows and fixes the split. No GPU -- asking
# for one would just queue this behind real training work.
#
# set -euo pipefail belongs HERE, in the body, not only in the preamble: otherwise a
# failed source leaves the job running under the system python and still exits 0,
# releasing every afterok dependent behind a stage that did no work.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script, so $BASH_SOURCE
# points at the spool directory -- sourcing relative to it silently sources nothing.
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "manifest"
_lpgen_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

cd "$REPO"
mkdir -p "$MANIFEST_DIR"

ARGS=(
  --lp-train "$LP_TRAIN"
  --lp-val "$LP_VAL"
  --out-dir "$MANIFEST_DIR"
  --reserve-frac "$RESERVE_FRAC"
  --reserve-seed "$RESERVE_SEED"
  --val-frac "$VAL_FRAC"
  --val-seed "$VAL_SEED"
  --panel-size "$PANEL_SIZE"
  --panel-sources "$PANEL_SOURCES"
  --workers "$MANIFEST_WORKERS"
  --limit "$MANIFEST_LIMIT"
)
# Unset means "reuse the configured lp-mix weights"; the script records which it used.
[[ -n "$PEPBENCH_FRAC" ]] && ARGS+=(--pepbench-frac "$PEPBENCH_FRAC")

# The unit tests are the smoke run, per the plan: they exercise the split purity, the
# insertion-code residue parse, the panel spread and the weight integration on synthetic
# data in a couple of seconds. Running them here means a broken builder fails before it
# writes a manifest that four arms would then train against.
echo "[manifest] self-test"
"$PY" script_utils/test_lpgen_manifest.py

echo "[manifest] building"
"$PY" script_utils/lpgen_manifest.py "${ARGS[@]}"

echo "[manifest] done -> $MANIFEST_DIR"
ls -la "$MANIFEST_DIR"
