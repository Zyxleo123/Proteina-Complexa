#!/bin/bash
# Submitter for the MEET-vs-LNR de novo closure comparison (CPU only).
#
#   bash scripts/submit_meet_compare.sh --meet <results_dir> --lnr <results_dir> [--out DIR]
#
# Both arguments are `evaluation_results/uncondgen*` directories produced by
# scripts/submit_uncond_gen.sh. Compare LIKE WITH LIKE: a MEET run staged with a pocket crop
# belongs against the LNR *pocket* arm, not the full-chain arm -- receptor segmentation
# alone moves closure by tens of points (see scripts/restage_lnr_pocket.py).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MEET=""; LNR=""; OUT=""; PARTITION="general"; TIME="00:20:00"; MAXLEN=13
while [[ $# -gt 0 ]]; do
  case "$1" in
    --meet) MEET="$2"; shift 2 ;;
    --lnr)  LNR="$2";  shift 2 ;;
    --out)  OUT="$2";  shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --match-max-length) MAXLEN="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$MEET" && -n "$LNR" ]] || { echo "FATAL: --meet and --lnr are required" >&2; exit 1; }
[[ "$MEET" = /* ]] || MEET="${REPO}/${MEET}"
[[ "$LNR"  = /* ]] || LNR="${REPO}/${LNR}"
[[ -d "$MEET" ]] || { echo "FATAL: not a directory: $MEET" >&2; exit 1; }
[[ -d "$LNR"  ]] || { echo "FATAL: not a directory: $LNR"  >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
[[ -n "$OUT" ]] || OUT="${REPO}/evaluation_results/meet_vs_lnr_${STAMP}"
[[ "$OUT" = /* ]] || OUT="${REPO}/${OUT}"

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/meet_compare_${STAMP}.env"
cat > "$ENV_FILE" <<EOF2
# Auto-written by scripts/submit_meet_compare.sh at ${STAMP}.
MEET_RESULTS=${MEET}
LNR_RESULTS=${LNR}
COMPARE_OUT=${OUT}
MATCH_MAX_LENGTH=${MAXLEN}
EOF2
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
JID=$(sbatch --parsable --partition="$PARTITION" --time="$TIME" \
  --job-name="meet_compare" scripts/meet_compare.sbatch "$ENV_FILE")
echo "submitted CPU compare: job ${JID}"
echo "results: ${OUT}"
