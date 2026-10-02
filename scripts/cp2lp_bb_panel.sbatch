#!/bin/bash
#SBATCH --job-name=cp2lp_bb_panel
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# The FIXED 24-source evaluation panel, built once and used by every arm at every milestone.
#
# Reuses `cp2lp_eval_manifest.py` rather than adding a second selector: it already picks one
# row per receptor family, spreads over peptide length, is deterministic by seeded hash, and
# reports its own coverage. 24 parents x 4 paired seeds = 96 attempts per arm per milestone.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
# Arm-agnostic stage: it reads EVAL_META / the packed rows and nothing arm-specific.
# Pinned anyway so the banner does not advertise the preamble's `gan` default and
# name the legacy rollout config in a backbone-arm job's log.
ARM="${ARM:-bb_torsion}"
_cp2lp_defaults
_cp2lp_banner "bb-panel"
cd "$REPO"

BB_PANEL="${BB_PANEL:-${LPDATA}/cp2lp_bb_panel24.parquet}"
BB_PANEL_N="${BB_PANEL_N:-24}"
BB_SEEDS="${BB_SEEDS:-0,1,2,3}"

"${PY}" script_utils/cp2lp_eval_manifest.py \
  --eval-parquet "${EVAL_META}" \
  --out-parquet "${BB_PANEL}" \
  --n-parents "${BB_PANEL_N}" \
  --seeds "${BB_SEEDS}"

echo "[done] panel at ${BB_PANEL}"
