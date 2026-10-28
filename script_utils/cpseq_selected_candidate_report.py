"""The cross-fit-selected candidate against the unchanged peptide, on BOTH axes.

`cpseq_ladder_report.py` already answers the energy question without selection bias:
choose the candidate on folds 0..half, measure it on half..depth, and the peptide on
half..depth too. What it does not do is report VALIDITY for that same candidate. Its
validity column compares the complex's AVERAGE draw to the peptide, which is a different
question -- "is a random mutant usually as foldable as the native" rather than "is the
sequence the search picked as foldable as the native".

Mixing the two is how a search gets credited with an energy win that came with a
validity cost nobody measured. So this script reports both numbers for ONE sequence: the
candidate the cross-fit selected, on the folds the selection never saw.

Selection rule is imported, not reimplemented, so it cannot drift from the report's:
lowest median energy over folds [0, half). Measurement is over [half, depth) on both
sides, always the same number of folds for candidate and peptide -- the rule that the
withdrawn best-of-B result broke.

A candidate selected on the first half that passes no fold in the second is NOT counted
as an energy loss; it is counted in its own `unmeasurable` column, because a missing
measurement is not a bad one. It IS counted in validity, where a pass rate of zero is a
real and meaningful answer.

The complex is the unit of analysis. Ten folds of one sequence are repeated measures,
and a test pooled over folds counts one complex as ten.

Usage:
    python script_utils/cpseq_selected_candidate_report.py \
        --out-dir /data/user_data/yixiz/cpseq/pepbench_corpus_pot
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_ladder_report import (  # noqa: E402
    DRAW_ARM,
    build,
    complete_prefix,
    load,
    sign_test,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--rungs", default="1,2,4,8")
    ap.add_argument("--select-reliability", type=int, default=0,
                    help="folds of the SELECTION half a candidate must pass before it "
                         "is eligible to be picked. 0 (the default) reproduces "
                         "cpseq_ladder_report's cross-fitted rule, which ranks on "
                         "energy alone. Set it to 3 (of 5) for the rule a pipeline "
                         "shipping a sequence would actually use -- fold acceptably "
                         "first, then be the best of what is left. The gate is applied "
                         "ONLY to the selection folds; applying it to the measurement "
                         "folds would be selecting on the outcome.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    depth = args.depth
    half = depth // 2
    state = build(load(out_dir), draw_arm=DRAW_ARM)
    rungs = [int(x) for x in args.rungs.split(",")]

    print(f"=== cross-fit-selected candidate vs unchanged peptide ===")
    print(f"out_dir: {out_dir}")
    print(f"selected on folds [0,{half}) by median energy; "
          f"both sides measured on folds [{half},{depth})\n")
    if args.select_reliability:
        print(f"selection gate: candidate must pass >= {args.select_reliability} "
              f"of the {half} selection folds to be eligible\n")
    print(f"  {'B':>2} {'cplx':>5} | {'dG REU':>8} {'w/l':>8} {'sign p':>9} "
          f"| {'d(valid)':>9} {'w/l':>8} {'sign p':>9} | {'unmeas':>6} {'unserved':>8}")

    results = []
    for B in rungs:
        e_d, e_w, e_l, unmeas = [], 0, 0, 0
        v_d, v_w, v_l = [], 0, 0
        n_cplx = 0
        n_unserved = 0
        for _pid, s in state.items():
            draws = s["draws"]
            ref = s["reference"]
            if ref is None or complete_prefix(draws, depth) < B:
                continue
            prefix = [draws[k] for k in sorted(draws) if k < B]
            if ref.n_folds(depth) < depth:
                continue
            pick = [(d.median_energy_in(0, half), d) for d in prefix
                    if d.median_energy_in(0, half) is not None
                    and d.n_pass_in(0, half) >= args.select_reliability]
            if not pick:
                # Every draw failed the gate on the selection folds. That is a real
                # outcome of the rule -- the search had nothing it would ship -- so the
                # complex is counted as unserved rather than quietly skipped, which
                # would score the rule only on the complexes it happened to answer.
                n_unserved += 1
                continue
            n_cplx += 1
            chosen = min(pick, key=lambda t: t[0])[1]

            # Validity: a zero pass rate in the held-out half is a real answer, so this
            # is computed for every selected candidate, including ones the energy
            # comparison has to drop.
            n_ho = chosen.n_folds(depth) and len(chosen.folds_in(half, depth))
            n_ref = len(ref.folds_in(half, depth))
            if n_ho and n_ref:
                dv = chosen.n_pass_in(half, depth) / n_ho - ref.n_pass_in(half, depth) / n_ref
                v_d.append(dv)
                if dv > 0:
                    v_w += 1
                elif dv < 0:
                    v_l += 1

            ref_e = ref.median_energy_in(half, depth)
            got = chosen.median_energy_in(half, depth)
            if ref_e is None:
                continue
            if got is None:
                unmeas += 1
                continue
            de = got - ref_e
            e_d.append(de)
            if de < 0:
                e_w += 1
            elif de > 0:
                e_l += 1

        row = {
            "B": B, "n_complexes": n_cplx,
            "n_unserved": n_unserved,
            "select_reliability_k_of_half": args.select_reliability,
            "energy_n": len(e_d),
            "energy_median_delta": st.median(e_d) if e_d else None,
            "energy_wins": e_w, "energy_losses": e_l,
            "energy_sign_p": sign_test(e_w, e_l) if (e_w + e_l) else None,
            "energy_unmeasurable": unmeas,
            "validity_n": len(v_d),
            "validity_median_delta": st.median(v_d) if v_d else None,
            "validity_mean_delta": st.fmean(v_d) if v_d else None,
            "validity_wins": v_w, "validity_losses": v_l,
            "validity_sign_p": sign_test(v_w, v_l) if (v_w + v_l) else None,
            "select_folds": [0, half], "measure_folds": [half, depth],
        }
        results.append(row)
        em = "-" if row["energy_median_delta"] is None else f"{row['energy_median_delta']:+8.2f}"
        vm = "-" if row["validity_median_delta"] is None else f"{row['validity_median_delta']:+9.3f}"
        ep = "-" if row["energy_sign_p"] is None else f"{row['energy_sign_p']:9.2e}"
        vp = "-" if row["validity_sign_p"] is None else f"{row['validity_sign_p']:9.2e}"
        print(f"  {B:>2} {n_cplx:>5} | {em} {f'{e_w}/{e_l}':>8} {ep} "
              f"| {vm} {f'{v_w}/{v_l}':>8} {vp} | {unmeas:>6} {n_unserved:>8}")

    print("\n  dG REU     selected candidate's median energy minus the peptide's, both")
    print("             over the held-out folds. Negative = the search's pick is better.")
    print("  d(valid)   the same candidate's pass RATE minus the peptide's, held-out")
    print("             folds. Negative = the search's pick folds acceptably less often.")
    print("  unmeas     selected on the first half, passed no fold in the second, so it")
    print("             has no held-out energy. Counted in validity, not in energy.")
    print("  Energies are Rosetta ref2015 REU, NOT measured binding free energies.")

    dest = Path(args.json) if args.json else out_dir / "selected_candidate_report.json"
    dest.write_text(json.dumps(
        {"out_dir": str(out_dir), "depth": depth, "half": half,
         "draw_arm": DRAW_ARM, "rungs": results}, indent=2))
    print(f"\n[selected-candidate-report] -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
