#!/usr/bin/env bash
# Reproduce the arm table of docs/README_CPSEQ_MPNN_FILL_SETTINGS.md section 0.2.
#
# This is the estimator the ARM CHOICE was made on, and it is not the report's
# default. Section 3 of that document specifies it; `--pilot` is the four flags
# that implement it:
#
#   --pool all           select from every draw the arm produced for a complex,
#                        not from the first B of the stream
#   --directions 2       select on ranks [0,5) measure on [5,10), then the
#                        reverse, and average the two per complex
#   --min-ref-valid 3    a reference with fewer than 3 valid folds is a complex
#                        with NO BASELINE, reported as such, never scored
#   --energy-key refined_dG_separated
#                        the pilot's field, not the screening objective
#
# The fourth is a NO-OP on every run measured so far, and is kept only so the
# rule is stated in full. `best_valid_dG_separated` and `refined_dG_separated`
# are equal on all 14,602 v3 rows of `llm_ladder` and `pepbench_corpus_pot`
# where both exist -- they differ only in being null on an invalid fold, and
# `median_energy_in` already drops those. Verified by running the table both
# ways: identical to the digit. They would diverge under `diffusion_samples > 1`,
# where `best_valid` is a best-of-samples and `refined` is one structure; every
# run here is at 1.
#
# Without `--pilot` the same script answers a different question -- per-budget B
# off a nested prefix -- and those numbers are not comparable to these.
#
# The gate must be named. `llm_ladder` holds three (v3 22,642 rows / v5 6,150 /
# v4 1,087) and the report refuses to print a comparison from a mixed set. v3 is
# the gate: see section 5.2 for why not v4 and not v5.
#
# Usage:
#   bash scripts/cpseq_pilot_table.sh                      # print the table
#   OUT_JSON=/some/dir bash scripts/cpseq_pilot_table.sh   # also keep the JSON
#
# Verified 2026-11-01: `llm_search` reproduces its recorded -7.23 REU / 9 of 10
# exactly, which is what certifies this script against the original measurement.
# It is also the one arm with no v5 rows; the arms that moved are the ones whose
# folds were later split across gates.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-${REPO:-/home/yixiz/Proteina-Complexa}}"
[ -f "$REPO/script_utils/cpseq_selected_candidate_report.py" ] \
  || REPO=/home/yixiz/Proteina-Complexa

OUT_DIR="${OUT_DIR:-/data/user_data/yixiz/cpseq/llm_ladder}"
GATE="${GATE:-v3-cis-amide-muco-windows}"
OUT_JSON="${OUT_JSON:-}"
ARMS="${ARMS:-random_mask_mpnn_fill llm_search llm_mask_random_fill random_control llm_mask_mpnn_fill}"

cd "$REPO"
[ -d "$OUT_DIR/state" ] || { echo "FATAL: no $OUT_DIR/state -- is /data mounted here?" >&2; exit 1; }

echo "=== pilot-estimator arm table ==="
echo "out_dir: $OUT_DIR"
echo "gate:    $GATE"
echo "rule:    --pilot (= --pool all --directions 2 --min-ref-valid 3"
echo "                    --energy-key refined_dG_separated)"
echo
printf '%-24s %s\n' "arm" "cplx |   dG REU     w/l    sign p | no_base  pool"

for arm in $ARMS; do
  json_arg=()
  [ -n "$OUT_JSON" ] && json_arg=(--json "$OUT_JSON/pilot_${arm}.json")
  printf '%-24s' "$arm"
  python3 script_utils/cpseq_selected_candidate_report.py \
      --out-dir "$OUT_DIR" \
      --scoring-version "$GATE" \
      --arm "$arm" \
      --pilot \
      "${json_arg[@]}" \
    | sed -n '/^  all/p'
done

cat <<'EOF'

Caveats that travel with this table, both measured 2026-11-01:

  * The 10 complexes are the SAME 10 for every arm -- the reference is shared --
    and they are 6 PepBench, 4 LNR, out of a cohort of 10 and 10. The other 10
    have a reference below 3 valid folds (see section 5.3).
  * 4 of the 10 have only 0 or 1 valid reference fold in ONE of the two cross-fit
    directions, so that direction's baseline is a median over a single structure:
    lnr_5njx, lnr_5y59, lnr_6seo (2/1) and pepbench_A_B_pdb2phk (1/2).
    --min-ref-valid applies to the full depth, not per half. Treat the per-complex
    sign as solid and the magnitude as soft.
EOF
