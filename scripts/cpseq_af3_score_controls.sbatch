#!/usr/bin/env bash
#SBATCH --job-name=cpseq_af3_sc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --open-mode=append
#SBATCH --output=/zfsauton/scratch/yixiz/cpseq_af3/logs/cpseq_af3_sc_%A_%a.out
#
# Score every AF3 control fold against its own deposited crystal, by the identical
# protocol. One array task per complex.
#
# Why one task per complex, and one journal per task: concurrent O_APPEND to a shared
# file NUL-corrupts rows on this filesystem (measured elsewhere in this project at 88 of
# 144 rows lost). Parallelism comes from the array; each task owns its own scores.jsonl
# and the driver appends to it as each fold finishes, so a crash costs one fold rather
# than the whole task.
#
# The native is hetero-stripped first. Rosetta splits a chain on HOH/MG/DST into extra
# chains, and InterfaceAnalyzerMover then SEGFAULTS against a 2-partner interface spec --
# a bare signal with no Python stack. Keeping only ATOM records of the receptor and
# peptide chains gives the native the same two partners the AF3 fold has, which is what
# makes the two numbers comparable at all.
#
# SCORING_VERSION selects the protocol, and it changes the RELAX, not just the label:
#   v3/v4  bond-only restraint     -- holds the closing bond, lets the junction absorb strain
#   v5     junction restraint      -- holds angles and torsion too
# Measured on 5TU6 these differ by ~6.6 REU on the AF3 fold and ~2.1 REU the OTHER way on
# the native, so run both and read them separately. Never pool them.
#
# Submit:  sbatch --array=0-3 scripts/cpseq_af3_score_controls.sbatch
#          sbatch --array=0-3 scripts/cpseq_af3_score_controls.sbatch SCORING_VERSION=v5-junction-restrained-relax
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
SCORING_VERSION="${SCORING_VERSION:-v3-cis-amide-muco-windows}"
TOLERANCES="${TOLERANCES:-$CPSEQ/gate/geometry_tolerances.json}"
REPRESENTATION="${REPRESENTATION:-polymer}"
# One PyRosetta process per core. Rosetta is single-threaded per process and
# ~110 s/fold, so this is the main lever on wall time.
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
# Default ON. A fold whose ring is open cannot produce a meaningful energy:
# the relax restraint forces it shut and the number describes the forced
# structure, not the prediction. Set REQUIRE_CLOSURE=0 to score anyway.
REQUIRE_CLOSURE="${REQUIRE_CLOSURE:-1}"
FOLDS_DIR="${FOLDS_DIR:-$CPSEQ/af3_controls_$REPRESENTATION}"
OUT_ROOT="${OUT_ROOT:-$CPSEQ/af3_scores/${REPRESENTATION}_$SCORING_VERSION}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "=== cpseq_af3_score_controls ==="
echo "node:     $(hostname)"
echo "job:      ${SLURM_JOB_ID:-local} array task $IDX"
echo "request:  partition=cpu cpus=${SLURM_CPUS_PER_TASK:-4} mem=24G time=6:00:00"
nvidia-smi -L 2>/dev/null | head -2 || echo "gpu:      none (CPU-only, as intended)"
echo "version:  $SCORING_VERSION"
echo "peptide:  $REPRESENTATION"
echo "workers:  $WORKERS"
echo "closure:  require=$REQUIRE_CLOSURE"
echo "gate:     $TOLERANCES"

[ -x "$PY" ] || { echo "FATAL: no venv python at $PY" >&2; exit 1; }
[ -s "$TOLERANCES" ] || { echo "FATAL: no frozen gate at $TOLERANCES" >&2; exit 1; }
[ -s "$CPSEQ/controls/controls.json" ] || { echo "FATAL: no controls.json" >&2; exit 1; }

cd "$REPO"

# Which complex this task owns, and everything needed to score it.
read -r LABEL SEQ PEP_CHAIN REC_CHAINS NATIVE <<EOF
$(IDX="$IDX" CONTROLS="$CPSEQ/controls/controls.json" "$PY" - <<'PYEOF'
import json, os
rows = [c for c in json.loads(open(os.environ["CONTROLS"]).read())
        if c["chemistry"] == "mainchain"]
rows.sort(key=lambda c: c["pdb_id"])
idx = int(os.environ["IDX"])
if idx >= len(rows):
    raise SystemExit(f"array index {idx} beyond {len(rows)} mainchain controls")
c = rows[idx]
print(c["pdb_id"].upper(), c["sequence"], c["peptide_chain"],
      ",".join(c["receptor_chains"]), c["structure_path"])
PYEOF
)
EOF

echo "complex:  $LABEL  seq=$SEQ  peptide=$PEP_CHAIN  receptor=$REC_CHAINS"
echo "native:   $NATIVE"

OUT_DIR="$OUT_ROOT/$LABEL"
mkdir -p "$OUT_DIR/work"

# Hetero-strip the native: keep only ATOM records of the receptor and peptide chains.
NATIVE_CLEAN="$OUT_DIR/${LABEL}_clean.pdb"
KEEP="$PEP_CHAIN,$REC_CHAINS" "$PY" - "$NATIVE" "$NATIVE_CLEAN" <<'PYEOF'
import os, sys
keep = {c.strip() for c in os.environ["KEEP"].split(",") if c.strip()}
src, dest = sys.argv[1], sys.argv[2]
n = 0
with open(dest, "w") as out:
    for line in open(src):
        if not line.startswith("ATOM"):
            continue
        if line[21] not in keep or line[17:20].strip() == "HOH":
            continue
        out.write(line)
        n += 1
    out.write("END\n")
print(f"[native] kept {n} ATOM records of chains {sorted(keep)} -> {dest}")
PYEOF

# Every per-fold model.cif. AF3 also writes a top-level copy of the top-ranked sample;
# including it would double-weight one seed, so only seed-*/ files are taken.
mapfile -t CIFS < <(find "$FOLDS_DIR/$LABEL/af3_out" -type f -name '*_model.cif' \
                      -path '*seed-*' | sort)
[ "${#CIFS[@]}" -gt 0 ] || { echo "FATAL: no per-fold model.cif for $LABEL" >&2; exit 1; }
echo "folds:    ${#CIFS[@]}"

ARGS=()
for c in "${CIFS[@]}"; do
  d="$(basename "$(dirname "$c")")"
  ARGS+=(--cif "${d}=${c}")
done

"$PY" script_utils/cpseq_af3_score.py \
  "${ARGS[@]}" \
  --sequence "$SEQ" \
  --receptor-chains "$REC_CHAINS" \
  --chemistry mainchain \
  --representation "$REPRESENTATION" \
  --workers "$WORKERS" \
  $([ "$REQUIRE_CLOSURE" = "1" ] && echo --require-closure) \
  --native "$NATIVE_CLEAN" \
  --native-peptide-chain "$PEP_CHAIN" \
  --native-receptor-chains "$REC_CHAINS" \
  --tolerances "$TOLERANCES" \
  --scoring-version "$SCORING_VERSION" \
  --work-dir "$OUT_DIR/work" \
  --out "$OUT_DIR/scores.jsonl"
rc=$?

echo "=== cpseq_af3_score_controls $LABEL exit $rc ==="
exit $rc
