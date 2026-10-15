#!/bin/bash
#SBATCH --job-name=lpgen_dpbscore
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_dpbscore_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_dpbscore_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Score DiffPepDock poses against the staged natives.
#
# CPU partition: this only reads saved PDBs. Asking for a GPU to compute RMSDs would queue
# a table behind real training. Results go to disk (JSONL rows + JSON summary) so tables and
# figures are re-derivable without re-running the GPU job.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "dpbscore"

# The run dir holds a timestamped `docking/<stamp>` subdirectory; accept either the eval_dir
# or the stamped dir, since the scorer globs recursively.
SCORE_RUN_DIR="${SCORE_RUN_DIR:-${DPB_RUN_DIR:-}}"
if [[ -z "$SCORE_RUN_DIR" || ! -d "$SCORE_RUN_DIR" ]]; then
  echo "GATE: SCORE_RUN_DIR not a directory: ${SCORE_RUN_DIR:-<unset>}" >&2
  exit 0
fi
SCORE_OUT_DIR="${SCORE_OUT_DIR:-${SCORE_RUN_DIR}/score}"

echo "[dpbscore] run   = ${SCORE_RUN_DIR}"
echo "[dpbscore] stage = ${DPB_STAGE_DIR}"
echo "[dpbscore] out   = ${SCORE_OUT_DIR}"

"$PY" "${REPO}/script_utils/lpgen_dpb_score.py" \
  --run-dir "$SCORE_RUN_DIR" \
  --stage-dir "$DPB_STAGE_DIR" \
  --out-dir "$SCORE_OUT_DIR" \
  --contact-cutoff "${SCORE_CONTACT_CUTOFF:-5.0}" \
  --success-a "${SCORE_SUCCESS_A:-2.0}"

echo "[dpbscore] done"
