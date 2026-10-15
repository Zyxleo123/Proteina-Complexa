#!/bin/bash
#SBATCH --job-name=lpgen_stage
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_stage_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_stage_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# Clone the upstream arm repositories, pin their commits, and apply our patch set.
#
# Compute nodes DO have outbound network here (measured: job 80479 on lov2 reached
# github over https and `git ls-remote` succeeded), so this runs as a job like everything
# else rather than on the login node.
#
# The trees are disposable and live on scratch; our CHANGES live in the project repo as
# asserted edits in script_utils/lpgen_patch_upstream.py, so they stay reviewable and a
# new upstream commit shows up as a loud refusal instead of a silent behaviour change.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "stage"

export GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=true
mkdir -p "$UPSTREAM_DIR"
cd "$REPO"

MANIFEST_OUT="${UPSTREAM_DIR}/upstream_manifest.json"
tmp_rows="$(mktemp)"
trap 'rm -f "$tmp_rows"' EXIT

for arm in ${STAGE_ARMS//,/ }; do
  url="$(_lpgen_arm_url "$arm")"
  if [[ -z "$url" ]]; then
    echo "FATAL: no URL registered for arm '${arm}' (see _lpgen_arm_url)" >&2
    exit 1
  fi
  dest="${UPSTREAM_DIR}/${arm}"

  if [[ -d "$dest/.git" && "$STAGE_FORCE" != "1" ]]; then
    echo "[stage] ${arm}: already staged at ${dest} (STAGE_FORCE=1 to re-clone)"
  else
    [[ -d "$dest" ]] && { echo "[stage] ${arm}: removing existing ${dest}"; rm -rf "$dest"; }
    echo "[stage] ${arm}: cloning ${url}"
    # --depth 1: we need the code and a recorded SHA, not the history. Note a later
    # re-clone of a moving branch lands on a NEWER commit -- the patch set's assertions
    # are what detect that, which is why they refuse instead of warning.
    git clone --depth 1 -q "$url" "$dest"
  fi

  sha="$(git -C "$dest" rev-parse HEAD)"
  date_iso="$(git -C "$dest" log -1 --format=%cI)"
  echo "[stage] ${arm}: ${sha} (upstream commit ${date_iso})"
  printf '%s\t%s\t%s\t%s\n' "$arm" "$url" "$sha" "$date_iso" >> "$tmp_rows"
done

"$PY" - "$MANIFEST_OUT" "$tmp_rows" <<'PYEOF'
import json, sys, datetime
out, rows = sys.argv[1], sys.argv[2]
arms = {}
for line in open(rows):
    arm, url, sha, cdate = line.rstrip("\n").split("\t")
    arms[arm] = {"url": url, "commit": sha, "upstream_commit_date": cdate}
json.dump({
    "staged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": "shallow clones; a re-clone of a moving branch may land on a newer commit. "
            "The asserted edits in lpgen_patch_upstream.py are the drift detector.",
    "arms": arms,
}, open(out, "w"), indent=2)
print(f"wrote {out}")
PYEOF

if [[ "$STAGE_PATCH" == "1" ]]; then
  # Only arms we actually have edits for; the others are staged pristine.
  patch_arms=""
  for arm in ${STAGE_ARMS//,/ }; do
    if "$PY" -c "
import sys; sys.path.insert(0,'script_utils')
from lpgen_patch_upstream import ARMS
sys.exit(0 if '${arm}' in ARMS else 1)
" 2>/dev/null; then
      patch_arms="${patch_arms:+${patch_arms},}${arm}"
    else
      echo "[stage] ${arm}: no patch set defined, staged pristine"
    fi
  done
  if [[ -n "$patch_arms" ]]; then
    echo "[stage] patching: ${patch_arms}"
    "$PY" script_utils/lpgen_patch_upstream.py \
      --upstream-dir "$UPSTREAM_DIR" --arms "$patch_arms" \
      --summary-json "${UPSTREAM_DIR}/patch_summary.json"
  fi
fi

echo "[stage] done -> $UPSTREAM_DIR"
cat "$MANIFEST_OUT"
