#!/usr/bin/env bash
#SBATCH --job-name=cpseq_pot_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq/potcheck/logs/prepare_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/cpseq/potcheck/logs/prepare_%j.out
#
# Stage 1 of the folding-potentials check: build the potentials-ON sibling run, and
# redraw the potentials-OFF side of the comparison.
#
# CPU only -- no --gres. Everything here reads journals and copies small JSONL files;
# asking for a GPU to do it would queue this behind real work.
#
# What it does, in order:
#   1. Redraws the OFF report for the arm, with per-complex rows, from the source run's
#      journals. This is the baseline half of the paired comparison and costs nothing:
#      no structure is needed, which is why the published bundle ships journals only.
#   2. Forks the cohort into POT_DIR with use_potentials flipped on. The PROPOSALS are
#      copied (so the sequences are identical by construction) and the JOURNALS are not
#      (their key carries no inference settings, so a copy would retire every fold as
#      already paid for and the ON run would measure nothing).
#   3. Writes the parent list the fold array indexes by line.
#
# The fold array depends on this with afterok, so a failure here stops the GPU spend
# rather than producing a directory that folds the wrong thing.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "$REPO"

# shellcheck source=/dev/null
source "$REPO/scripts/_cpseq_potentials_preamble.sh"
_pot_consume_args "$@"
_pot_defaults
_pot_banner "prepare"

mkdir -p "$WORK_DIR/off"

# --- 1. the OFF baseline, redrawn from the source journals -------------------------
echo
echo "--- OFF baseline (source run, redrawn from journals) ---"
for arm in $ARMS; do
  echo "  arm: $arm"
  "$PY" script_utils/cpseq_selected_candidate_report.py \
      --out-dir "$SRC_DIR" \
      --scoring-version "$GATE" \
      --arm "$arm" \
      --pilot \
      --json "$WORK_DIR/off/pilot_${arm}.json" \
    | sed -n '/^  all/p'
done

# --- 2. the fork --------------------------------------------------------------------
echo
echo "--- fork: use_potentials off -> on ---"
# The gate rehash is an override and is recorded as one. The evidence: the acceptance
# THRESHOLDS on disk here are byte-identical to the `adopted` set in the project's own
# gate audit, while the authoritative hash also covers `note`, `clash_cutoff_note` and
# `margin` -- per-corpus prose that differs between every corpus, which is why
# llm_ladder, pepbench_corpus_pot and these corpora all record different hashes for the
# same numbers. Without this the fork cannot proceed at all; with it, the fork records
# both hashes so a later reader can see the override and judge it.
REHASH_REASON="${REHASH_REASON:-Thresholds verified identical to gate_audit_v3.json \
'adopted' (tolerances-subtree sha 63966225f4d03fc5); the authoritative gate hash also \
covers note/clash_cutoff_note/margin, which differ per corpus (llm_ladder dbbfd2c9, \
pepbench_corpus_pot 251422b1, these corpora d8c15393), so it is not comparable across \
corpora while the numbers are equal.}"

FORK_ARGS=(
  --src "$SRC_DIR"
  --dest "$POT_DIR"
  --use-potentials on
  --scoring-version "$GATE"
  --json "$WORK_DIR/fork_report.json"
)
# The source cohort was built on another machine and records absolute corpus paths.
[ -n "${REMAP:-}" ] && FORK_ARGS+=(--remap "$REMAP")
[ "${REHASH_GATE:-1}" = "1" ] && FORK_ARGS+=(--rehash-gate --rehash-reason "$REHASH_REASON")
# A second submission of this chain is a resume, not a fresh fork.
[ -d "$POT_DIR/state" ] && FORK_ARGS+=(--resume)

"$PY" script_utils/cpseq_fork_cohort.py "${FORK_ARGS[@]}"

# --- 3. the parent list ---------------------------------------------------------------
echo
echo "--- parent list ---"
PF="$(_pot_parents_file "$LEAD_ARM")"
"$PY" - "$POT_DIR" "$WORK_DIR/off/pilot_${LEAD_ARM}.json" "$PARENT_SET" "$PF" <<'EOF'
import json, sys
from pathlib import Path
pot, offjson, which, dest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cohort = json.loads((Path(pot) / "cohort.json").read_text())
allp = [e["parent_id"] if isinstance(e, dict) else e for e in cohort["cohort"]]
scored = []
if Path(offjson).exists():
    rows = json.loads(Path(offjson).read_text())["rungs"][0].get("per_complex", [])
    scored = [r["parent_id"] for r in rows if r["status"] == "ok"]
if which == "fixed":
    pick = [p for p in allp if p in set(scored)]
    note = (f"the {len(pick)} complexes the SOURCE run scored under this estimator; "
            f"the paired decision set")
else:
    pick = list(allp)
    note = (f"every parent in the cohort ({len(allp)}). The {len(allp) - len(scored)} "
            f"the source run could not score are folded too, so a complex that GAINS a "
            f"baseline under potentials can be observed -- reported separately from the "
            f"paired set, never mixed into it")
lines = [f"# parent set: {which} -- {note}",
         f"# source scored {len(scored)} of {len(allp)}", ""]
lines += pick
Path(dest).write_text("\n".join(lines) + "\n")
print(f"[parents] {which}: {len(pick)} parent(s) -> {dest}")
for p in pick:
    print(f"    {p}{'' if p in set(scored) else '   (no baseline under OFF)'}")
EOF

echo
echo "=== cpseq_potentials_prepare OK ==="
echo "fold the ON arm with:  sbatch --array=0-$(( $(grep -vc '^[[:space:]]*#\|^[[:space:]]*$' "$PF") - 1 ))%N scripts/cpseq_potentials_fold.sbatch <env>"
