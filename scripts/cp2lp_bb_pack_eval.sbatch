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

# THE SOURCE COMPLEXES, which is what makes `contact_retention` and the sequence gate exist
# at all. `--source-pdb-root` was always plumbed through to the pack script and was never set,
# so `score_packed` got `source_contacts=None`, never computed retention, and the report's
# retention gate silently passed on a NaN -- the 2026-10-04 run reported a "joint success"
# rate over three of its five documented criteria.
#
# Built as a symlink farm keyed by `<example_id>.pdb`, which is the layout the pack script
# expects, derived FROM THE MANIFEST rather than hardcoding a directory: a panel may draw its
# rows from several splits, and `path` is the only thing that knows where each one lives.
# These panel PDBs are real complexes with crystal sidechains, so they also seed the packer's
# chi angles instead of leaving it to guess.
BB_PANEL="${BB_PANEL:-${LPDATA}/cp2lp_bb_panel24.parquet}"
if [[ -z "${BB_SOURCE_PDB_ROOT:-}" ]]; then
  BB_SOURCE_PDB_ROOT="${OUT}/source_pdbs"
  mkdir -p "${BB_SOURCE_PDB_ROOT}"
  "${PY}" - "${BB_PANEL}" "${BB_SOURCE_PDB_ROOT}" <<'PYSRC'
import os, sys
import pandas as pd

panel, root = sys.argv[1], sys.argv[2]
df = pd.read_parquet(panel)
linked = missing = 0
for ex, path in zip(df["example_id"], df["path"]):
    dst = os.path.join(root, f"{ex}.pdb")
    if not os.path.exists(path):
        print(f"  MISSING source complex for {ex}: {path}")
        missing += 1
        continue
    # ATOMIC AND IDEMPOTENT, because the arms' pack jobs run CONCURRENTLY against this one
    # shared directory. The obvious "if exists: unlink; symlink" is a TOCTOU race: both jobs
    # see no link, both call symlink, and the loser dies with FileExistsError -- which is
    # exactly how job 73694 failed and stranded the report on DependencyNeverSatisfied.
    # Create under a per-process unique name, then os.replace, which is atomic on POSIX and
    # silently wins or loses rather than raising.
    tmp = f"{dst}.tmp{os.getpid()}"
    try:
        if os.path.islink(tmp) or os.path.exists(tmp):
            os.unlink(tmp)
        os.symlink(path, tmp)
        os.replace(tmp, dst)
    except OSError as exc:
        # A concurrent job may have completed the same link first; that is success, not
        # failure. Anything else is real.
        if not (os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(path)):
            raise
        print(f"  {ex}: linked concurrently by another job ({exc.__class__.__name__})")
    finally:
        if os.path.islink(tmp):
            os.unlink(tmp)
    linked += 1
print(f"[source-pdbs] linked {linked}, missing {missing}, into {root}")
if missing:
    # Loud, not fatal: the pack script skips a source it cannot find, and the REPORT now
    # raises on the resulting absent gate rather than scoring it as a pass. Better to see
    # the count here than to debug a KeyError three jobs later.
    print("[source-pdbs] WARNING: attempts for the missing sources will have no retention")
PYSRC
fi
echo "[pack] source-pdb-root: ${BB_SOURCE_PDB_ROOT}"

shopt -s nullglob
# THIS ARM'S attempts only, by default. The attempts filename carries the arm label, which is
# the run_name, so `attempts__${RUN_NAME}__*` selects exactly this job's share.
#
# WHY NOT ALL OF THEM. The arms' pack jobs run concurrently against one shared directory, and
# globbing everything made each job try to pack every arm: duplicated Rosetta work, both jobs
# appending while both read each other's `packed__*.jsonl` for the resume set, and a race on
# the source-symlink farm. Keeping each job to its own file makes the stage embarrassingly
# parallel instead of accidentally shared.
#
# BB_PACK_ALL=1 restores the old behaviour, for re-packing a directory in one job.
if [[ -n "${BB_PACK_ALL:-}" ]]; then
  FILES=("${OUT}"/attempts__*.jsonl)
else
  FILES=("${OUT}"/attempts__"${RUN_NAME}"__*.jsonl)
fi
if [[ "${#FILES[@]}" -eq 0 ]]; then
  echo "GATE: no attempts__${RUN_NAME}__*.jsonl under ${OUT}."
  echo "      Present: $(ls -1 "${OUT}"/attempts__*.jsonl 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
  echo "      Either the GPU eval stage for this arm produced nothing, or the arm label and"
  echo "      RUN_NAME disagree (the eval writes attempts__<--arm-label>__<tag>.jsonl)."
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
