#!/usr/bin/env bash
#SBATCH --job-name=cpseq_gate_rebuild
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_gate_rebuild_%j.out
#
# Rebuild the frozen acceptance gate on this cluster.
#
# The gate is the one thing in this pipeline that is NOT supposed to be re-derived
# casually: `geometry_tolerances.json` is written once and read-only thereafter, and
# `cpseq_search.gate_sha256` refuses to fold if it changes under a live cohort. The
# search pilot's instruction is "copy it, do not re-derive it", because re-deriving per
# corpus would create a second gate and make two corpora incomparable.
#
# That instruction assumed the file was reachable. Here it is not -- it lives only on
# babel, and the B2 bucket carries the docs and nothing else -- so it has to be rebuilt.
# Rebuilt, not reinvented: the control SET is pinned to the six entries this project
# froze (so the derivation sees the same structures), and the result is checked against
# the audit numbers published in docs/README_CPSEQ_PEPBENCH_SCALEUP.md section 4. If the
# rebuilt gate reproduces those, it is the same gate; if it does not, the run stops here
# rather than carrying an unexplained threshold into every downstream number.
#
# Three stages, each resumable:
#   1. fetch the six controls from RCSB and VERIFY each by measurement
#   2. derive and freeze the tolerances from them
#   3. audit the frozen gate on the controls and on 150 native CPSea macrocycles
#
# Submit:  sbatch scripts/cpseq_gate_rebuild.sbatch
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
GATE_DIR="${GATE_DIR:-$CPSEQ/gate}"
CONTROLS_DIR="${CONTROLS_DIR:-$CPSEQ/controls}"
PY="${PY:-$REPO/.venv/bin/python}"
CPSEA_META="${CPSEA_META:-$ZFS/CPSea/CPSea_full/CPSea/preprocessed/metadata/cpsea_val.parquet}"

# The frozen control set. Pinned rather than re-searched: the tolerances are derived
# from these structures, so a later RCSB query returning a different six would be a
# different acceptance gate wearing the same version label.
CONTROL_IDS="${CONTROL_IDS:-5TU6,7K2M,7K2F,1SFI,1SLD,1SLE}"
CONTROL_EXPECT="${CONTROL_EXPECT:-5TU6:mainchain,7K2M:mainchain,7K2F:mainchain,1SFI:mainchain,1SLD:disulfide,1SLE:disulfide}"

mkdir -p "$GATE_DIR" "$CONTROLS_DIR"

echo "=== cpseq_gate_rebuild ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local}"
echo "request:  partition=cpu cpus=${SLURM_CPUS_PER_TASK:-4} mem=32G time=2:00:00"
nvidia-smi -L 2>/dev/null | head -2 || echo "gpu:      none (CPU-only stage, as intended)"
echo "controls: $CONTROL_IDS"
echo "cpsea:    $CPSEA_META"
echo "python:   $PY ($("$PY" -V 2>&1))"
[ -x "$PY" ] || { echo "FATAL: no venv python at $PY" >&2; exit 1; }
[ -s "$CPSEA_META" ] || { echo "FATAL: no CPSea metadata at $CPSEA_META" >&2; exit 1; }

cd "$REPO"

# -------------------------------------------------------------- 1. controls
if [ -s "$CONTROLS_DIR/controls.json" ]; then
  echo "[1/3] controls.json present; skipping fetch"
else
  echo "[1/3] fetching + verifying the six pinned controls"
  "$PY" script_utils/cpseq_fetch_controls.py \
    --out-dir "$CONTROLS_DIR" \
    --pdb-ids "$CONTROL_IDS" \
    --expect "$CONTROL_EXPECT" \
    --n 6
fi
"$PY" - "$CONTROLS_DIR/controls.json" <<'PYEOF'
import json, sys
rows = json.loads(open(sys.argv[1]).read())
print(f"[1/3] {len(rows)} controls:")
for r in sorted(rows, key=lambda x: x["pdb_id"]):
    c = r["closure"]
    print(f"        {r['pdb_id']} chain {r['peptide_chain']:2s} {r['chemistry']:10s} "
          f"L={r['peptide_length']:2d} bond={c.get('bond_length_A')} A  {r['sequence']}")
PYEOF

# ------------------------------------------------------- 2. freeze tolerances
if [ -s "$GATE_DIR/geometry_tolerances.json" ]; then
  echo "[2/3] geometry_tolerances.json present; refusing to overwrite a frozen gate"
  echo "      (delete it deliberately if you really mean to re-freeze)"
else
  echo "[2/3] deriving and freezing tolerances from the controls"
  mkdir -p "$GATE_DIR"
  "$PY" script_utils/cpseq_freeze_tolerances.py \
    --dataset-dir "$GATE_DIR" \
    --controls "$CONTROLS_DIR/controls.json" \
    --out "$GATE_DIR/geometry_tolerances.json"
fi
echo "[2/3] frozen gate:"
cat "$GATE_DIR/geometry_tolerances.json"
echo
"$PY" - "$GATE_DIR/geometry_tolerances.json" <<'PYEOF'
import hashlib, json, sys
doc = json.loads(open(sys.argv[1]).read())
th = {k: v for k, v in doc.items() if k not in ("copied_from", "copy_rationale", "frozen_on")}
h = hashlib.sha256(json.dumps(th, sort_keys=True).encode()).hexdigest()
print(f"[2/3] gate_sha256 = {h}")
print("[2/3] every cohort built against this gate records that hash; `cmd_ladder` "
      "re-checks it before folding.")
PYEOF

# ------------------------------------------------------------- 3. gate audit
echo "[3/3] auditing the frozen gate on the controls and 150 native CPSea macrocycles"
"$PY" script_utils/cpseq_gate_audit.py \
  --controls "$CONTROLS_DIR/controls.json" \
  --cpsea-metadata "$CPSEA_META" \
  --n-cpsea 150 --seed 0 \
  --out "$GATE_DIR/gate_audit_v3.json"

# The published numbers this has to reproduce, from
# docs/README_CPSEQ_PEPBENCH_SCALEUP.md section 4. Printed side by side rather than
# asserted silently: a mismatch here means the rebuilt gate is NOT the gate the -7.82
# REU baseline was measured under, and every comparison downstream would be against a
# moved bar.
"$PY" - "$GATE_DIR/gate_audit_v3.json" <<'PYEOF'
import json, sys
a = json.loads(open(sys.argv[1]).read())

def tally(rows, key):
    ok = [r for r in rows if r.get("status") == "ok" and key in r]
    return sum(bool(r[key]) for r in ok), len(ok)

res = a.get("results") or {}
ctrl = res.get("controls") or []
nat = res.get("native_cpsea_head_tail") or []
if not ctrl or not nat:
    raise SystemExit(
        f"FATAL: gate_audit_v3.json has no rows to check "
        f"(top-level keys {sorted(a)}, results keys {sorted(res)})"
    )

# The criterion the published figures quote is planarity against the NEAREST permitted
# arrangement (`torsion_adopted`) -- that is the cis-or-trans-within-30-deg change v3
# made. The closing-bond window is a separate criterion and is what rejects 7K2F, so
# both are reported rather than collapsed into one "passing" number.
checks = [
    ("crystal controls, planarity (cis or trans, 30 deg)", ctrl, "torsion_adopted", "6/6"),
    ("native CPSea, planarity (cis or trans, 30 deg)",      nat,  "torsion_adopted", "117/150"),
    ("native CPSea, closing bond in MuCO window",           nat,  "bond_adopted",    "150/150"),
]

print("\n[3/3] rebuilt gate vs the published v3 audit")
print(f"{'criterion':52s} {'published':>11s} {'rebuilt':>11s}  verdict")
drift = []
for name, rows, key, published in checks:
    p, t = tally(rows, key)
    got = f"{p}/{t}"
    agree = got == published
    if not agree:
        drift.append((name, published, got))
    print(f"{name:52s} {published:>11s} {got:>11s}  {'match' if agree else 'DRIFT'}")

# 7K2F must still be the one control the bond window rejects, by name.
bond_fail = sorted(r["id"] for r in ctrl
                   if r.get("status") == "ok" and r.get("bond_adopted") is False)
print(f"\n[3/3] controls rejected on the closing-bond window: "
      f"{', '.join(bond_fail) if bond_fail else '(none)'}   expected: 7K2F")
if bond_fail != ["7K2F"]:
    drift.append(("controls failing bond window", "['7K2F']", str(bond_fail)))

if drift:
    print("\n[3/3] REFUSING: the rebuilt gate is not the gate the -7.82 REU baseline was")
    print("      measured under. Every downstream comparison would be against a moved bar.")
    for name, published, got in drift:
        print(f"        {name}: published {published}, rebuilt {got}")
    raise SystemExit(1)

print("\n[3/3] the rebuilt gate reproduces the published v3 audit exactly.")
print("[3/3] full rows are in gate_audit_v3.json; every figure redraws from it.")
PYEOF

echo "=== cpseq_gate_rebuild DONE ==="
echo "gate:  $GATE_DIR/geometry_tolerances.json"
echo "audit: $GATE_DIR/gate_audit_v3.json"
