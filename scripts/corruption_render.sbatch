#!/bin/bash
#SBATCH --job-name=cc_render
#SBATCH --output=logs/cc_render_%j.out
#SBATCH --error=logs/cc_render_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Renders the CP -> LP pull as a filmstrip, from the dumped complexes and the ladder JSONL.
# Reads only files on disk, so figures re-render without re-running any minimization.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_corruption_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_corruption_preamble.sh"

_cc_consume_args "$@"
_cc_defaults
_cc_banner "render"
_cc_preflight || { echo "FATAL: preflight failed" >&2; exit 1; }

cd "$REPO"

# Gate, not failure: nothing dumped means nothing to draw, and a non-zero exit here would only
# obscure the upstream stage that actually produced nothing.
shopt -s nullglob
LADDER_FILES=("$LADDER_DIR"/ladder_shard*.jsonl)
shopt -u nullglob
if [[ ${#LADDER_FILES[@]} -eq 0 ]]; then
  echo "GATE: no ladder JSONL in $LADDER_DIR -- nothing to render. Exiting 0."
  exit 0
fi
if [[ ! -d "$PDB_DIR" ]]; then
  echo "GATE: no dumped complexes at $PDB_DIR -- nothing to render. Exiting 0."
  exit 0
fi

# PyMOL lives in .venv, not .venv_openmm.
"$PY" -c "import pymol" 2>/dev/null || {
  echo "GATE: PyMOL is not importable in $PY. No figures written; exiting 0."; exit 0; }

mkdir -p "$RENDER_DIR"
"$PY" script_utils/corruption_ceiling_render.py \
  --ladder-glob "${LADDER_DIR}/ladder_shard*.jsonl" \
  --rosetta-glob "${ROSETTA_DIR}/dg_shard*.jsonl" \
  --complex-dir "$PDB_DIR" \
  --out-dir "$RENDER_DIR" \
  --per-chemistry "$RENDER_PER_CHEMISTRY" \
  --panel-px "$RENDER_PANEL_PX" \
  ${RENDER_FAST:+$([[ "$RENDER_FAST" == "1" ]] && echo --fast)} \
  ${RENDER_KEEP_PANELS:+$([[ "$RENDER_KEEP_PANELS" == "1" ]] && echo --keep-panels)} \
  ${RENDER_EXAMPLES:+--examples $RENDER_EXAMPLES}

echo "[render] done -> $RENDER_DIR"
