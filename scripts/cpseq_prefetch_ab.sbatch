#!/usr/bin/env bash
#SBATCH --job-name=cpseq_prefetch_ab
#SBATCH --partition=preempt
#SBATCH --account=bapoczos
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/yixiz/slurm_logs/cpseq_prefetch_ab_%j.out
#
# Does overlapping the fold phase with the score phase actually pay, and does it leave
# the results alone?
#
# A ladder task folds EVERY seed first and only then scores, so the GPU is idle for the
# whole scoring phase and the CPU for the whole folding phase. Measured on the live
# corpus run (133 records, 58 min, 6 tasks): 51.5 % boltz duty cycle, 157 s/fold end to
# end. `--prefetch-ahead` folds seed k+1 on a background thread while seed k is being
# relaxed. Folding is a subprocess and the relax is single-threaded PyRosetta -- sampled
# at 79-97 % of one core, nothing above one core's worth -- so they genuinely overlap.
#
# Both arms run BACK TO BACK IN ONE JOB, on one node and one GPU, because this is a
# timing comparison: two jobs could land on different cards, and preempt hands out
# several. That is also why no GPU type is pinned here, unlike the cost probe: the
# answer is a RATIO between two arms that share one card, so the card's model cancels,
# and asking for any GPU schedules far sooner on a busy partition.
#
# Each arm gets its OWN out_dir, so neither can serve the other from the prediction
# cache -- which would make the second arm look infinitely fast. That costs duplicate
# folding, and the duplication is the price of a clean read.
#
# Same parent, same draws, same seeds, same everything but the flag. The two arms must
# produce the SAME outcome rows, and the script diffs them: the overlap only changes who
# is waiting, so a difference in any scored value means the change is not what it claims.
#
# Requeue-safe: each arm's out_dir is wiped at the start of its run, so a restarted job
# re-measures from scratch rather than resuming into a half-populated cache and
# reporting a time that means nothing.
#
# Submit: sbatch scripts/cpseq_prefetch_ab.sbatch
#         sbatch scripts/cpseq_prefetch_ab.sbatch SEED_DEPTH=6 PARENT=pepbench_1_5_pdb1z7z

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/home/yixiz/Proteina-Complexa}"
[ -d "$REPO/src/proteinfoundation" ] || REPO=/home/yixiz/Proteina-Complexa
cd "$REPO"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) if [ -f "$arg" ]; then set -a; source "$arg"; set +a; fi ;;
  esac
done

ZFS=/data/user_data/yixiz
SRC_COHORT="${SRC_COHORT:-$ZFS/cpseq/pepbench_corpus_pot/cohort.json}"
ROOT="${ROOT:-$ZFS/cpseq/_opt_probe/ab}"
# One parent is enough: the question is per-task throughput, not cohort coverage.
PARENT="${PARENT:-}"
# Four seed groups, not ten. The first group can never overlap -- there is nothing to
# score while it folds -- so the measured gain understates the full-depth gain, and
# says so honestly, at 36 folds per arm instead of 90.
SEED_DEPTH="${SEED_DEPTH:-4}"
PY="${PYTHON_EXEC:-$REPO/.venv/bin/python}"
BOLTZ_PY="${BOLTZ_PYTHON:-$ZFS/envs/boltz2/.venv/bin/python}"
BOLTZ_CACHE="${BOLTZ_CACHE:-$ZFS/boltz_cache}"

echo "=== cpseq_prefetch_ab ==="
echo "node:    $(hostname)"
echo "job:     ${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "request: -p preempt --qos=preempt_qos --gres=gpu:A6000:1 --cpus-per-task=2 --mem=32G --time=4:00:00"
echo "cohort:  $SRC_COHORT"
echo "root:    $ROOT   seed_depth=$SEED_DEPTH"
nvidia-smi -L || echo "no nvidia-smi"
echo

[ -f "$SRC_COHORT" ] || { echo "FATAL: no cohort at $SRC_COHORT" >&2; exit 1; }

# ------------------------------------------------------- trimmed cohort per arm
for ARM in base ahead; do
  OUT="$ROOT/$ARM"
  rm -rf "$OUT"
  mkdir -p "$OUT"
  "$PY" -c "
import json, sys
src, out, parent, depth = sys.argv[1:5]
d = json.load(open(src))
p = parent or d['cohort'][0]
if p not in d['cohort']:
    raise SystemExit(f'FATAL: {p} is not in the cohort')
depth = int(depth)
d['cohort'] = [p]
d['n_parents'] = 1
d['parent_sources'] = {p: d['parent_sources'][p]}
d['seeds'] = d['seeds'][:depth]
d['seed_tiers'] = sorted({t for t in d['seed_tiers'] if t < depth} | {depth})
d['ab_trial'] = {'source_cohort': src, 'parent': p, 'seed_depth': depth}
json.dump(d, open(out + '/cohort.json', 'w'), indent=2)
print(f'  {out}: parent={p} draws={d[\"n_draws\"]} seeds={d[\"seeds\"]}')
" "$SRC_COHORT" "$OUT" "$PARENT" "$SEED_DEPTH"
done
echo

# ------------------------------------------------------------------- the two arms
run_arm () {
  local ARM="$1"; shift
  local OUT="$ROOT/$ARM"
  echo "=== arm: $ARM ==="
  local T0; T0=$(date +%s)
  # Not `set -e`-fatal: if one arm dies the other's number is still worth having, and
  # the report below says which arm produced nothing.
  srun "$PY" -u script_utils/cpseq_search.py ladder \
    --out-dir "$OUT" \
    --shard-index 0 --shard-count 1 \
    --chemistry mainchain --samples 1 \
    --retention rescorable --limit 0 \
    --boltz-python "$BOLTZ_PY" --boltz-cache "$BOLTZ_CACHE" \
    "$@" 2>&1 | sed "s/^/  [$ARM] /" || echo "  [$ARM] ARM FAILED (rc=$?)"
  local T1; T1=$(date +%s)
  echo "$((T1 - T0))" > "$OUT/.wall_s"
  echo "=== arm $ARM done in $((T1 - T0))s ==="
  echo
}

run_arm base
run_arm ahead --prefetch-ahead

# ------------------------------------------------------------------------ report
"$PY" - "$ROOT" <<'PYEOF'
import glob, json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for arm in ("base", "ahead"):
    d = root / arm
    wall = int((d / ".wall_s").read_text().strip()) if (d / ".wall_s").exists() else 0
    boltz_s, n_pred = 0.0, 0
    for p in glob.glob(str(d / "predictions/*/*/record.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("status") == "ok" and r.get("runtime_s"):
            boltz_s += r["runtime_s"]; n_pred += 1
    outcomes = []
    for p in glob.glob(str(d / "state/*/outcomes.jsonl")):
        for line in Path(p).read_text().splitlines():
            try:
                outcomes.append(json.loads(line))
            except Exception:
                pass
    good = [o for o in outcomes if not o.get("technical_failure")]
    rows[arm] = {"wall_s": wall, "boltz_s": round(boltz_s, 1), "n_pred": n_pred,
                 "n_folds": len(good), "outcomes": good}

print("\n=== prefetch A/B ===")
print(f"  {'arm':8s} {'wall':>8s} {'folds':>7s} {'s/fold':>8s} {'boltz s':>9s} {'duty':>7s}")
for arm, r in rows.items():
    spf = r["wall_s"] / r["n_folds"] if r["n_folds"] else 0
    duty = r["boltz_s"] / r["wall_s"] * 100 if r["wall_s"] else 0
    print(f"  {arm:8s} {r['wall_s']:7d}s {r['n_folds']:7d} {spf:7.1f}s "
          f"{r['boltz_s']:8.1f}s {duty:6.1f}%")

b, a = rows["base"], rows["ahead"]
if b["n_folds"] and a["n_folds"]:
    sb = b["wall_s"] / b["n_folds"]; sa = a["wall_s"] / a["n_folds"]
    print(f"\n  speedup: {sb / sa:.2f}x  ({sb:.1f} -> {sa:.1f} s/fold)")

# The arms must agree fold for fold. Compare on the key the journal itself uses, and on
# the scored values a result is read from -- a timing change that moved a number is not
# a timing change.
KEYS = ("valid", "status", "best_valid_dG_separated", "min_violation", "n_mutations")
def index(os_):
    return {(o["sequence"], o["prediction_seed"], o.get("arm", "")): o for o in os_}
ib, ia = index(b["outcomes"]), index(a["outcomes"])
shared = sorted(set(ib) & set(ia))
diffs = [(k, f, ib[k].get(f), ia[k].get(f))
         for k in shared for f in KEYS if ib[k].get(f) != ia[k].get(f)]
print(f"\n  rows: base={len(ib)} ahead={len(ia)} shared={len(shared)} "
      f"base_only={len(set(ib) - set(ia))} ahead_only={len(set(ia) - set(ib))}")
if diffs:
    print(f"  MISMATCH on {len(diffs)} field(s) -- the overlap changed a result:")
    for k, f, x, y in diffs[:15]:
        print(f"    {k[0]} seed={k[1]}  {f}: base={x!r} ahead={y!r}")
else:
    print(f"  identical on {', '.join(KEYS)} across all {len(shared)} shared rows")

json.dump({a_: {k: v for k, v in r.items() if k != "outcomes"} for a_, r in rows.items()},
          open(root / "ab_result.json", "w"), indent=2)
print(f"\n  wrote {root / 'ab_result.json'}")
PYEOF

echo "=== cpseq_prefetch_ab OK ==="
