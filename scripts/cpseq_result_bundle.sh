#!/usr/bin/env bash
# Push ONLY the sequences and fold records that a named result rests on.
#
# `cpseq_push_b2.sh` moves whole run trees, which for `llm_ladder` is tens of GB
# of predicted structures. The arm table of docs/README_CPSEQ_MPNN_FILL_SETTINGS.md
# section 0.2 does not rest on any of that: every number in it comes from
# `state/<parent>/outcomes.jsonl`, which is 21 MB for the whole 20-complex
# cohort and compresses like the JSONL it is. The structures are reproducible
# from the journals; the journals are the thing nothing regenerates.
#
# So this uploads one small object per result, with the replay instructions and
# the exact scripts inside it, rather than a tier of a tree.
#
# Contents:
#   cohort.json                    the seeds, rungs and parents
#   state/<parent>/*.jsonl         candidate + outcome journals (all arms)
#   REPLAY.md                      the command, the gate, the caveats
#   scripts/, script_utils/        the report code, pinned by git sha
#
# Usage:
#   bash scripts/cpseq_result_bundle.sh                       # build only, no upload
#   bash scripts/cpseq_result_bundle.sh b2:proteina/cpseq     # build and upload
#   WITH_SCORES=1 bash scripts/cpseq_result_bundle.sh ...     # + the v3 score JSONs
#   PARENTS=baseline bash scripts/cpseq_result_bundle.sh ...  # only the 10 scored
#
# PARENTS=all (the default) sends all 20, because "10 of 20" is part of the
# claim: a bundle holding only the 10 that have a baseline cannot show that the
# other 10 were measured and excluded, which is the difference between a result
# and a selected subset.

set -euo pipefail

DEST="${1:-}"
RUN_DIR="${RUN_DIR:-/data/user_data/yixiz/cpseq/llm_ladder}"
GATE="${GATE:-v3-cis-amide-muco-windows}"
NAME="${NAME:-$(basename "$RUN_DIR")_pilot_table}"
PARENTS="${PARENTS:-all}"
WITH_SCORES="${WITH_SCORES:-0}"
ZLEVEL="${ZLEVEL:-10}"
STAGE="${STAGE:-${TMPDIR:-/tmp}/cpseq_bundle_$$}"

REPO="${REPO:-/home/yixiz/Proteina-Complexa}"
[ -d "$RUN_DIR/state" ] || { echo "FATAL: no $RUN_DIR/state -- is /data mounted here?" >&2; exit 1; }
command -v zstd >/dev/null || { echo "FATAL: zstd not on PATH" >&2; exit 1; }

SHA="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY=""
git -C "$REPO" diff --quiet 2>/dev/null || DIRTY=" (dirty worktree)"

# Which parents. "baseline" asks the report code itself which complexes have one,
# so this script cannot drift from the table's own admission rule.
if [ "$PARENTS" = "baseline" ]; then
  mapfile -t PIDS < <(cd "$REPO" && python3 - "$RUN_DIR" "$GATE" <<'PY'
import collections, glob, json, sys
run, gate = sys.argv[1], sys.argv[2]
ref = collections.defaultdict(list)
for f in glob.glob(f"{run}/state/*/outcomes.jsonl"):
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("arm") == "reference" and r.get("scoring_version") == gate:
            ref[r["parent_id"]].append(bool(r.get("valid")))
for p in sorted(p for p, v in ref.items() if sum(v) >= 3):
    print(p)
PY
)
else
  mapfile -t PIDS < <(cd "$RUN_DIR/state" && ls -d */ 2>/dev/null | tr -d /)
fi
[ "${#PIDS[@]}" -gt 0 ] || { echo "FATAL: no parents selected" >&2; exit 1; }

echo "=== result bundle: $NAME ==="
echo "run_dir:  $RUN_DIR"
echo "gate:     $GATE"
echo "parents:  ${#PIDS[@]} ($PARENTS)"
echo "repo:     $SHA$DIRTY"

trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME/state" "$STAGE/$NAME/scripts" "$STAGE/$NAME/script_utils"

cp "$RUN_DIR/cohort.json" "$STAGE/$NAME/" 2>/dev/null || echo "  note: no cohort.json"
for p in "${PIDS[@]}"; do
  mkdir -p "$STAGE/$NAME/state/$p"
  cp "$RUN_DIR/state/$p"/*.jsonl "$STAGE/$NAME/state/$p/" 2>/dev/null || true
done

# The report code, so the bundle replays without the repo.
cp "$REPO/scripts/cpseq_pilot_table.sh" "$STAGE/$NAME/scripts/"
for f in cpseq_selected_candidate_report.py cpseq_ladder_report.py cpseq_arm_compare.py; do
  cp "$REPO/script_utils/$f" "$STAGE/$NAME/script_utils/"
done

if [ "$WITH_SCORES" = "1" ] && [ -d "$RUN_DIR/scores/$GATE" ]; then
  echo "  + score JSONs for $GATE"
  mkdir -p "$STAGE/$NAME/scores/$GATE"
  cp -r "$RUN_DIR/scores/$GATE/." "$STAGE/$NAME/scores/$GATE/"
fi

cat > "$STAGE/$NAME/REPLAY.md" <<EOF
# $NAME

Built $(date -Is) on $(hostname) from \`$RUN_DIR\`, repo \`$SHA\`$DIRTY.

Every number in the arm table of \`README_CPSEQ_MPNN_FILL_SETTINGS.md\` section 0.2
comes from the journals in \`state/\`. No predicted structure is needed to redraw it,
which is why none is here.

## Replay

    OUT_DIR=\$PWD REPO=\$PWD bash scripts/cpseq_pilot_table.sh

That is \`--pilot\`, i.e. \`--pool all --directions 2 --min-ref-valid 3
--energy-key refined_dG_separated\`, under \`--scoring-version $GATE\`. Each of
those four differs from the report's default, and the default answers a
different question (per-budget B off a nested prefix).

## What is in here

- \`cohort.json\` — seeds, budget rungs, parents, protocol
- \`state/<parent>/outcomes.jsonl\` — one row per (sequence, seed, gate); carries
  \`valid\`, \`refined_dG_separated\`, \`seed_rank\`, \`arm\`, \`scoring_version\`
- \`state/<parent>/candidates.jsonl\` — the proposals and their provenance
- \`script_utils/\` — the report code, unmodified
$([ "$WITH_SCORES" = "1" ] && echo "- \`scores/$GATE/\` — per-fold score records, keyed by input_hash")

Parents included: ${#PIDS[@]} ($PARENTS).

## Known caveats

- The gate must be named: this run holds v3, v4 and v5 rows, and the report
  refuses to print a comparison from a mixed set.
- 195 of the 200 v4 reference rows were never actually scored (null geometry,
  \`valid: false\`). Do not read a v4 pass rate off this run.
- 4 of the 10 complexes with a baseline have 0 or 1 valid reference fold in one
  cross-fit direction. Per-complex signs are solid; magnitudes are soft.
EOF

TARBALL="$STAGE/${NAME}.tar.zst"
tar -C "$STAGE" -cf - "$NAME" | zstd -"$ZLEVEL" -T2 -q -o "$TARBALL"
echo "built:    $TARBALL  ($(du -h "$TARBALL" | cut -f1))"

if [ -z "$DEST" ]; then
  KEEP="${KEEP:-$HOME/${NAME}.tar.zst}"
  cp "$TARBALL" "$KEEP"
  echo "no DEST given; kept at $KEEP"
  echo "upload with: bash scripts/cpseq_result_bundle.sh b2:proteina/cpseq"
  exit 0
fi

command -v rclone >/dev/null || { echo "FATAL: rclone not on PATH" >&2; exit 1; }
rclone lsd "$DEST" >/dev/null 2>&1 || rclone lsd "${DEST%%/*}" >/dev/null 2>&1 || {
  echo "FATAL: cannot reach ${DEST%%/*} from $(hostname)." >&2; exit 1; }

OBJ="$DEST/results/${NAME}.tar.zst"
# `rclone lsf` exits 0 for a missing object and prints nothing, so the exit code
# is not an existence test -- it has to be the output that is checked.
if [ -n "$(rclone lsf "$OBJ" 2>/dev/null)" ] && [ -z "${FORCE:-}" ]; then
  echo "exists:   $OBJ  (FORCE=1 to overwrite)"
else
  rclone copyto "$TARBALL" "$OBJ" --progress
  echo "uploaded: $OBJ"
fi
