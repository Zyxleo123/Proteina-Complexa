#!/bin/bash
#SBATCH --job-name=stage0_rosetta
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 0, step 3: the CPU half of the cost profile -- Rosetta interface dG on the complexes
# the GPU job just wrote, measured for what it costs.
#
# This is deliberately NOT scripts/sdedit_rosetta.sbatch, even though it calls the same
# scorer. That arm shards across an array and reports nothing about its own cost; here the
# whole point is the cost, so the run is single-task (core-seconds from a sharded array would
# have to be summed across jobs) and wrapped in /usr/bin/time so user+system CPU time is read
# from the kernel rather than inferred from wall-clock x allocated cores. Those two differ by
# a lot for a scorer that spends its first seconds reading a database off ZFS.
#
# No GPU on purpose: it only reads structures the sampler already wrote. That separation is
# also what lets the Stage 1 tranche queue its scoring on the CPU partition instead of behind
# real GPU work.
#
# Do NOT sbatch directly -- use scripts/submit_stage0_profile.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
mkdir -p slurm_logs

ENV_FILE="$1"
[[ -f "$ENV_FILE" ]] || { echo "FATAL: env file not found: $ENV_FILE"; exit 1; }
source env.sh
source "$ENV_FILE"

: "${RESULTS_DIR:?RESULTS_DIR not set}"
PROFILE_DIR="${RESULTS_DIR}/profile"
ROSETTA_DIR="${RESULTS_DIR}/rosetta"
COMPLEX_DIR="${COMPLEX_DIR:-${RESULTS_DIR}/complexes}"
PY="${PYTHON_EXEC:-$(pwd)/.venv/bin/python}"

# GATE, not a failure: if the GPU job produced nothing there is nothing to score, and exiting
# non-zero here would strand the report job on DependencyNeverSatisfied with no explanation of
# which stage actually broke. Say which, and exit 0 so the report runs and reports the hole.
shopt -s nullglob
EDITS=("${RESULTS_DIR}"/edits_shard*.jsonl)
shopt -u nullglob
if [[ ${#EDITS[@]} -eq 0 ]]; then
  echo "GATE: no edits_shard*.jsonl under ${RESULTS_DIR} -- the GPU profile produced nothing."
  echo "GATE: nothing to score. Exiting 0 so the report job can say so."
  exit 0
fi
if [[ ! -d "$COMPLEX_DIR" ]]; then
  echo "GATE: no complexes at ${COMPLEX_DIR}. The GPU job must run with COMPLEX_DIR set --"
  echo "GATE: the loader frame is per-process, so a peptide saved without its receptor cannot"
  echo "GATE: be re-united with it afterwards (scripts/complex_frame.py). Exiting 0."
  exit 0
fi

mkdir -p "$ROSETTA_DIR" "$PROFILE_DIR"
echo "Node: $(hostname 2>/dev/null || echo unknown)  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "edits:     ${#EDITS[@]} file(s) under ${RESULTS_DIR}"
echo "complexes: ${COMPLEX_DIR}"

TIME_LOG="${PROFILE_DIR}/rosetta_time.txt"
# --only-scorable keeps the cost honest: an unscorable row costs nothing, so billing it would
# understate the per-structure figure that Stage 1 gets sized from.
set +e
/usr/bin/time -v -o "$TIME_LOG" \
  "$PY" scripts/score_sdedit_rosetta.py \
    --edits "${EDITS[@]}" \
    --out "${ROSETTA_DIR}/rosetta_shard0.jsonl" \
    --complex-dir "$COMPLEX_DIR" \
    --only-scorable \
    --shard-index 0 --shard-count 1
RC=$?
set -e
echo "scorer exit code: ${RC}"
[[ -f "$TIME_LOG" ]] && { echo "--- /usr/bin/time -v ---"; cat "$TIME_LOG"; }

# Machine-readable cost sidecar. Parsed here rather than in the report job so the numbers stay
# attached to the job that actually paid them, and survive the report being re-run later.
"$PY" - "$TIME_LOG" "${PROFILE_DIR}/rosetta_cost.json" "${ROSETTA_DIR}/rosetta_shard0.jsonl" \
       "${SLURM_CPUS_PER_TASK:-0}" "$RC" <<'PYEOF'
import json
import re
import sys
from pathlib import Path

time_log, out_json, rosetta_jsonl, cpus, rc = sys.argv[1:6]


def grab(text, label, cast=float):
    m = re.search(rf"{re.escape(label)}:\s*([0-9.:]+)", text)
    if not m:
        return None
    v = m.group(1)
    if ":" in v:  # h:mm:ss or m:ss
        parts = [float(p) for p in v.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return cast(v)


text = Path(time_log).read_text() if Path(time_log).is_file() else ""
user_s = grab(text, "User time (seconds)")
sys_s = grab(text, "System time (seconds)")
wall_s = grab(text, "Elapsed (wall clock) time (h:mm:ss or m:ss)")
max_rss_kb = grab(text, "Maximum resident set size (kbytes)")

n_rows = 0
p = Path(rosetta_jsonl)
if p.is_file():
    n_rows = sum(1 for line in p.read_text().splitlines() if line.strip())

core_s = None if (user_s is None or sys_s is None) else round(user_s + sys_s, 1)
record = {
    "exit_code": int(rc),
    "cpus_allocated": int(cpus) if str(cpus).isdigit() else None,
    "wall_seconds": wall_s,
    "user_seconds": user_s,
    "system_seconds": sys_s,
    "cpu_core_seconds": core_s,
    "max_rss_MB": None if max_rss_kb is None else round(max_rss_kb / 1024, 1),
    "n_scored_rows": n_rows,
    # Per SCORED ROW, not per generated structure: an unscorable generation never reaches the
    # scorer, so dividing by the generation count would report a cost the scorer never paid.
    "core_seconds_per_scored_row": None if (core_s is None or not n_rows) else round(core_s / n_rows, 2),
}
Path(out_json).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PYEOF

# The gate above already handled "nothing to score". A scorer that ran and FAILED is a real
# failure and must break the chain rather than hand the report an empty file to average.
exit "$RC"
