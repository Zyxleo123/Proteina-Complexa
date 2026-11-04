#!/usr/bin/env bash
#SBATCH --job-name=cpseq_prep_cohort
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_prep_cohort_%j.out
#
# Rebuild the two corpora and the 20-complex cohort the MPNN-fill baseline was measured
# on, then build the cohort against the v3 protocol.
#
# The cohort is 10 LNR + 10 PepBench, selected uniformly without replacement with a seed
# derived from the corpus label (`cpseq_search.proposal_seed`). So the SAME twenty come
# back provided the eligible pool is the same, and the pool is "every target prepared in
# that dataset_dir" -- which is why both corpora are prepared here with the same list
# files and length window the pilot used. If the pool differs the cohort differs, and the
# cohort.json records n_selected against the pool size so that is visible rather than
# assumed.
#
# Templates are local; the MSAs are fetched from the free ColabFold server through
# boltz's own mmseqs2 client, rate-limited, and checkpointed after every fetch so an
# interrupt never re-queries. Measured previously at ~1.6 s/query, so this is minutes of
# network, not hours.
#
# The frozen gate is COPIED into each corpus's manifest, never re-derived per corpus:
# re-deriving would create a second gate and make the two corpora's numbers
# incomparable, which is the one comparison this run exists to make. `copied_from` and
# `copy_rationale` go inside the file, and `gate_sha256` ignores those keys, so the hash
# stays identical to the frozen original.
#
# Submit:  sbatch scripts/cpseq_prepare_cohort.sbatch
# Config:  positional args -- a path to source, or VAR=value. Never `sbatch --export`.

set -euo pipefail

for arg in "$@"; do
  if [ -f "$arg" ]; then echo "[cfg] sourcing $arg"; # shellcheck disable=SC1090
    . "$arg"
  elif [[ "$arg" == *=* ]]; then echo "[cfg] export $arg"; export "$arg"
  fi
done

REPO="${REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
CPSEQ="${CPSEQ:-$ZFS/cpseq}"
PY="${PY:-$REPO/.venv/bin/python}"
BOLTZ_PYTHON="${BOLTZ_PYTHON:-$ZFS/envs/boltz2/.venv/bin/python}"
GATE="${GATE:-$CPSEQ/gate/geometry_tolerances.json}"
SCORING_VERSION="${SCORING_VERSION:-v3-cis-amide-muco-windows}"

LNR_ROOT="${LNR_ROOT:-$ZFS/LNR}"
LNR_LIST="${LNR_LIST:-test.txt}"
LNR_OUT="${LNR_OUT:-$CPSEQ/tranche1}"
PB_ROOT="${PB_ROOT:-$ZFS/LPData/PepBench/train_valid}"
PB_LIST="${PB_LIST:-valid.txt}"
PB_OUT="${PB_OUT:-$CPSEQ/pepbench_valid}"

COHORT_OUT="${COHORT_OUT:-$CPSEQ/seed_ladder}"
N_LNR="${N_LNR:-10}"
N_PB="${N_PB:-10}"
DRAWS="${DRAWS:-64}"
SEED_DEPTH="${SEED_DEPTH:-10}"
MSA_DELAY_S="${MSA_DELAY_S:-2.0}"

echo "=== cpseq_prepare_cohort ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=cpu cpus=${SLURM_CPUS_PER_TASK:-8} mem=48G time=6:00:00"
nvidia-smi -L 2>/dev/null | head -2 || echo "gpu:      none (CPU-only, as intended)"
echo "version:  $SCORING_VERSION"
echo "gate:     $GATE"
echo "LNR:      $LNR_ROOT/$LNR_LIST -> $LNR_OUT"
echo "PepBench: $PB_ROOT/$PB_LIST -> $PB_OUT"
echo "cohort:   $N_LNR LNR + $N_PB PepBench -> $COHORT_OUT"

[ -x "$PY" ] || { echo "FATAL: no venv python at $PY" >&2; exit 1; }
[ -x "$BOLTZ_PYTHON" ] || { echo "FATAL: no boltz python at $BOLTZ_PYTHON -- run scripts/cpseq_env_boltz2.sbatch" >&2; exit 1; }
[ -s "$GATE" ] || { echo "FATAL: no frozen gate at $GATE -- run scripts/cpseq_gate_rebuild.sbatch" >&2; exit 1; }
[ -d "$LNR_ROOT/pdbs" ] || { echo "FATAL: no $LNR_ROOT/pdbs" >&2; exit 1; }
[ -d "$PB_ROOT/pdbs" ] || { echo "FATAL: no $PB_ROOT/pdbs" >&2; exit 1; }

cd "$REPO"

copy_gate() {
  local dest_dir="$1" label="$2"
  mkdir -p "$dest_dir/manifest"
  GATE="$GATE" DEST="$dest_dir/manifest/geometry_tolerances.json" LABEL="$label" \
  "$PY" - <<'PYEOF'
import hashlib, json, os, time
src, dest, label = os.environ["GATE"], os.environ["DEST"], os.environ["LABEL"]
doc = json.loads(open(src).read())
def gate_sha(d):
    th = {k: v for k, v in d.items() if k not in ("copied_from", "copy_rationale", "frozen_on")}
    return hashlib.sha256(json.dumps(th, sort_keys=True).encode()).hexdigest()
before = gate_sha(doc)
doc["copied_from"] = src
doc["copy_rationale"] = (
    f"Copied, not re-derived, for {label}. The tolerances come from four experimental "
    "crystal macrocycle controls and are a project standard, not a property of this "
    "corpus. Re-deriving per corpus would create a second gate and make this corpus's "
    "numbers incomparable with the other's, which is the comparison the run exists to "
    "make."
)
if os.path.exists(dest) and gate_sha(json.loads(open(dest).read())) != before:
    raise SystemExit(f"REFUSING: {dest} already holds DIFFERENT thresholds. A frozen "
                     f"gate is read-only; delete it deliberately if you mean to replace it.")
open(dest, "w").write(json.dumps(doc, indent=2))
after = gate_sha(json.loads(open(dest).read()))
print(f"[gate] {label}: {dest}")
print(f"[gate]   sha256 {after}  (unchanged by the copy: {after == before})")
PYEOF
}

# ------------------------------------------------------------------ 1. LNR
if [ -s "$LNR_OUT/manifest/targets.csv" ]; then
  echo "[1/4] LNR targets already prepared ($(wc -l < "$LNR_OUT/manifest/targets.csv") rows); skipping"
else
  echo "[1/4] preparing LNR targets (templates + cached MSAs)"
  "$PY" script_utils/cpseq_prepare_targets.py \
    --out-dir "$LNR_OUT" \
    --lnr-root "$LNR_ROOT" --list-file "$LNR_LIST" --source LNR \
    --msa-delay-s "$MSA_DELAY_S" \
    --boltz-python "$BOLTZ_PYTHON"
fi
copy_gate "$LNR_OUT" LNR

# ------------------------------------------------------------- 2. PepBench
if [ -s "$PB_OUT/manifest/targets.csv" ]; then
  echo "[2/4] PepBench targets already prepared ($(wc -l < "$PB_OUT/manifest/targets.csv") rows); skipping"
else
  echo "[2/4] preparing PepBench targets (templates + cached MSAs)"
  "$PY" script_utils/cpseq_prepare_targets.py \
    --out-dir "$PB_OUT" \
    --lnr-root "$PB_ROOT" --list-file "$PB_LIST" --source PepBench \
    --msa-delay-s "$MSA_DELAY_S" \
    --boltz-python "$BOLTZ_PYTHON"
fi
copy_gate "$PB_OUT" PepBench

# ------------------------------------------------------- 3. pool sizes, stated
echo "[3/4] eligible pools (the cohort selection depends on these)"
for d in "$LNR_OUT" "$PB_OUT"; do
  if [ -s "$d/manifest/targets.csv" ]; then
    echo "        $(basename "$d"): $(( $(wc -l < "$d/manifest/targets.csv") - 1 )) prepared targets"
  else
    echo "        $(basename "$d"): NO targets.csv"; exit 1
  fi
done

# ------------------------------------------------------------- 4. the cohort
if [ -s "$COHORT_OUT/cohort.json" ]; then
  echo "[4/4] cohort.json exists; not rebuilding (a rebuilt cohort can select a"
  echo "      different twenty and would orphan any folds already on disk)"
else
  echo "[4/4] building the 20-complex cohort at $SCORING_VERSION"
  "$PY" script_utils/cpseq_search.py ladder-cohort \
    --out-dir "$COHORT_OUT" \
    --corpus "dataset_dir=$LNR_OUT,n=$N_LNR,label=LNR" \
    --corpus "dataset_dir=$PB_OUT,n=$N_PB,label=PepBench" \
    --draws "$DRAWS" --seed-depth "$SEED_DEPTH" \
    --scoring-version "$SCORING_VERSION"
fi

COHORT_OUT="$COHORT_OUT" "$PY" - <<'PYEOF'
import json, os
c = json.loads(open(os.path.join(os.environ["COHORT_OUT"], "cohort.json")).read())
print(f"\n[cohort] {c['n_parents']} complexes, scoring_version={c.get('scoring_version')}")
print(f"[cohort] gate sha256 {c.get('geometry_tolerances_sha256','?')[:16]}")
print(f"[cohort] draws={c.get('n_draws')} seeds={len(c.get('seeds') or [])} "
      f"rungs={c.get('budget_rungs')}")
for corpus in c.get("corpora", []):
    print(f"[cohort]   {corpus['label']}: {corpus['n_selected']} of "
          f"{corpus.get('pool_rule','?')} -- seed {corpus.get('selection_seed')}")
for p in c["cohort"]:
    pid = p.get("parent_id", p) if isinstance(p, dict) else p
    print(f"             {pid}")
PYEOF

echo "=== cpseq_prepare_cohort DONE ==="
