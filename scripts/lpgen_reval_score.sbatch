#!/bin/bash
#SBATCH --job-name=lpgen_reval_score
#SBATCH --output=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_reval_score_%j.out
#SBATCH --error=/zfsauton/scratch/yixiz/lpgen/logs/lpgen_reval_score_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# Score every re-evaluated checkpoint and assemble ONE curve. CPU only: this reads saved
# structures, so asking for a GPU would just queue it behind real training.
#
# ONE CSV PER CHECKPOINT, then a concatenation. Not one shared file: concurrent O_APPEND is
# measured to NUL-corrupt rows on this cluster. The scoring here is sequential within a
# single process, but the per-checkpoint files also make a partial run resumable and let a
# single bad checkpoint be re-scored without redoing the rest.
set -euo pipefail

REPO_DEFAULT=/zfsauton2/home/yixiz/Proteina-Complexa
REPO="${SLURM_SUBMIT_DIR:-$REPO_DEFAULT}"
[[ -f "$REPO/scripts/_lpgen_preamble.sh" ]] || REPO="$REPO_DEFAULT"
source "$REPO/scripts/_lpgen_preamble.sh"

_lpgen_consume_args "$@"
_lpgen_defaults
_lpgen_banner "reval_score"

if [[ ! -d "${REVAL_ROOT}" ]]; then
  echo "[reval_score] GATE: ${REVAL_ROOT} does not exist -- no checkpoint was re-evaluated" >&2
  exit 0
fi

shopt -s nullglob
ROOTS=("${REVAL_ROOT}"/*/val_structures/predictions)
shopt -u nullglob
if [[ ${#ROOTS[@]} -eq 0 ]]; then
  echo "[reval_score] GATE: no prediction trees under ${REVAL_ROOT}" >&2
  echo "[reval_score] the GPU re-eval jobs either failed or dumped nothing" >&2
  exit 0
fi
echo "[reval_score] found ${#ROOTS[@]} prediction tree(s)"

OUT_DIR="${REVAL_ROOT}/scores"
mkdir -p "$OUT_DIR"

N_OK=0
for root in "${ROOTS[@]}"; do
  tag="$(basename "$(dirname "$(dirname "$root")")")"
  out="${OUT_DIR}/${tag}.csv"
  if [[ -s "$out" ]]; then
    echo "[reval_score] ${tag}: already scored, skipping (delete ${out} to redo)"
    N_OK=$((N_OK + 1))
    continue
  fi
  echo "[reval_score] scoring ${tag}"
  # Per-checkpoint failures must not abort the others; a missing checkpoint is a gap in the
  # curve, not a reason to lose the checkpoints that did work.
  if "$PY" "${REPO}/script_utils/lpgen_score_val_structures.py" \
       --pred-root "$root" \
       --panel "${VAL_PANEL}" \
       --contigs "${REVAL_CONTIGS}" \
       --out "$out" \
       --workers "${REVAL_SCORE_WORKERS}"; then
    N_OK=$((N_OK + 1))
  else
    echo "[reval_score] WARNING: ${tag} failed to score" >&2
    rm -f "$out"
  fi
done

if [[ "$N_OK" -eq 0 ]]; then
  echo "[reval_score] nothing scored" >&2
  exit 1
fi

CURVE="${OUT_DIR}/reval_curve.csv"
"$PY" - "$OUT_DIR" "$CURVE" <<'PYEOF'
import sys, glob, os
import pandas as pd

out_dir, curve = sys.argv[1], sys.argv[2]
files = sorted(f for f in glob.glob(os.path.join(out_dir, "*.csv"))
               if not f.endswith("reval_curve.csv"))
frames = []
for f in files:
    d = pd.read_csv(f)
    # REFUSE a corrupt file rather than skipping its rows: a silently short curve is worse
    # than a loud failure, because the shape of the trend is the whole deliverable.
    if d.empty or "epoch" not in d.columns:
        raise SystemExit(f"FATAL: {f} has no usable rows -- refusing to assemble a partial curve")
    d["source_file"] = os.path.basename(f)
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df.to_csv(curve, index=False)
print(f"[reval_score] wrote {curve}: {len(df):,} rows over {df.epoch.nunique()} epoch(s)")

ok = df[df.status == "ok"] if "status" in df else df
if ok.empty:
    raise SystemExit("FATAL: no scored row")

print("\n[reval_score] CORRECTED CURVE (medians over models per epoch)")
print(f"  {'epoch':>6s} {'n':>5s} {'raw':>7s} {'superp':>7s} {'iface':>7s} "
      f"{'<=2A':>6s} {'anyhit':>7s} {'clash':>6s} {'contact':>8s} {'hidden':>7s}")
for ep, g in ok.groupby("epoch"):
    best = g.groupby("example_id").interface_raw_ca_rmsd.min()
    hid = g.seq_identity_hidden.dropna() if "seq_identity_hidden" in g else pd.Series(dtype=float)
    print(f"  {ep:6.0f} {len(g):5d} {g.raw_ca_rmsd.median():7.2f} "
          f"{g.superposed_ca_rmsd.median():7.2f} {g.interface_raw_ca_rmsd.median():7.2f} "
          f"{(g.interface_raw_ca_rmsd <= 2.0).mean():6.3f} {(best <= 2.0).mean():7.3f} "
          f"{g.n_clash_2A.median():6.0f} {g.n_contacts_5A.median():8.0f} "
          f"{(hid.mean() if len(hid) else float('nan')):7.3f}")

# The copy check is the tripwire that says whether the conditioning was honoured at all.
if "seq_identity_given" in ok:
    giv = ok.seq_identity_given.dropna()
    if len(giv):
        bad = int((giv < 0.9999).sum())
        print(f"\n  copy check (given identities): median {giv.median():.4f}, "
              f"{bad} of {len(giv):,} models wrong"
              + ("  <-- conditioning NOT honoured" if bad else "  <-- ok"))
PYEOF

echo "[reval_score] scores: ${OUT_DIR}"
