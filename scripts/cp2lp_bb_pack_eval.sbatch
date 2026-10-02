#!/bin/bash
#SBATCH --job-name=cp2lp_bb_pack
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# CPU half: pack every dumped backbone and score it all-atom. THE JOB THAT DECIDES WHETHER
# THE RESULT IS HOLISTIC -- a report built without it is backbone-only and must say so.
#
# Resumable: rows are appended as they finish and a finished attempt is skipped, so a time
# limit is a resubmit. ONE OUTPUT FILE PER JOB -- concurrent O_APPEND to a shared file was
# measured to NUL-corrupt rows here.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_cp2lp_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_cp2lp_preamble.sh"

_cp2lp_consume_args "$@"
ARM="${ARM:-bb_torsion}"
_cp2lp_defaults
_cp2lp_banner "bb-pack-eval"
cd "$REPO"

BB_EVAL_ROOT="${BB_EVAL_ROOT:-${TRIPLET_ROOT}/bb_eval}"
OUT="${BB_EVAL_ROOT}/${EVAL_CKPT_TAG}"
PACK_SEEDS="${PACK_SEEDS:-3}"

shopt -s nullglob
FILES=("${OUT}"/attempts__*.jsonl)
if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "GATE: no attempts__*.jsonl under ${OUT} -- the GPU eval stage produced nothing."
  exit 0
fi

rc=0
for f in "${FILES[@]}"; do
  echo "[pack] ${f}"
  "${PY}" script_utils/cp2lp_bb_pack_eval.py \
    --attempts "${f}" \
    --out-dir "${OUT}" \
    --pack-seeds "${PACK_SEEDS}" \
    ${BB_SOURCE_PDB_ROOT:+--source-pdb-root "${BB_SOURCE_PDB_ROOT}"} \
    ${BB_KEEP_PDBS:+--keep-pdbs} || rc=1
done
exit "${rc}"
