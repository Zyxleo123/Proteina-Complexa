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
    warn_if_mixed,
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
    ap.add_argument("--arm", default=DRAW_ARM,
                    help=f"which proposal arm's draws to select from (default "
                         f"{DRAW_ARM}). The reference folds are shared across arms, "
                         f"which is what keeps the arms paired.")
    ap.add_argument("--scoring-version", default=None,
                    help="report only folds judged by this gate. A run directory that "
                         "holds more than one version was judged by more than one gate, "
                         "and no selection rule can repair that -- so without this the "
                         "report says MIXED and refuses to print a comparison.")
    # --- the four axes on which this report's estimator differs from the one that chose
    # the arm (README_CPSEQ_MPNN_FILL_SETTINGS.md section 3). Defaults reproduce this
    # script's long-standing behaviour exactly, so no number already recorded moves.
    ap.add_argument("--pool", choices=("rungs", "all"), default="rungs",
                    help="'rungs' selects from the first B draws of the stream, one row "
                         "per B, and requires those B complete at depth. 'all' selects "
                         "from every draw the arm produced for that complex, which is "
                         "what the pilot table did -- an arm that proposed 7 draws on "
                         "one complex and 12 on another contributes both, so the pool "
                         "size is a property of the run rather than of the row.")
    ap.add_argument("--directions", type=int, choices=(1, 2), default=1,
                    help="1 selects on ranks [0,half) and measures on [half,depth). "
                         "2 also runs the reverse and averages the two per complex, "
                         "which is section 3's rule and uses every fold for both jobs.")
    ap.add_argument("--min-ref-valid", type=int, default=0,
                    help="valid folds the reference needs before the complex has a "
                         "baseline at all (section 3: 3). Complexes below it are "
                         "reported as 'no baseline', NOT as a loss and not dropped "
                         "silently -- a complex whose native peptide never folds is a "
                         "missing measurement, not a negative result.")
    ap.add_argument("--energy-key", default=None,
                    help="row field candidates are ranked and scored by. Defaults to "
                         "cpseq_ladder_report.ENERGY_KEY (the screening objective). The "
                         "pilot table used refined_dG_separated.")
    ap.add_argument("--pilot", action="store_true",
                    help="shorthand for the estimator that chose the arm: --pool all "
                         "--directions 2 --min-ref-valid 3 "
                         "--energy-key refined_dG_separated. Set explicitly by any of "
                         "those flags, which win over this one.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.pilot:
        given = set(sys.argv[1:])
        if "--pool" not in given:
            args.pool = "all"
        if "--directions" not in given:
            args.directions = 2
        if "--min-ref-valid" not in given:
            args.min_ref_valid = 3
        if "--energy-key" not in given:
            args.energy_key = "refined_dG_separated"

    out_dir = Path(args.out_dir)
    depth = args.depth
    half = depth // 2
    per_parent = load(out_dir)
    if warn_if_mixed(per_parent, args.scoring_version):
        return 1
    state = build(per_parent, draw_arm=args.arm,
                  scoring_version=args.scoring_version,
                  **({"energy_key": args.energy_key} if args.energy_key else {}))
    rungs = [int(x) for x in args.rungs.split(",")] if args.pool == "rungs" else [0]

    # (select_lo, select_hi, measure_lo, measure_hi) per direction.
    splits = [(0, half, half, depth)]
    if args.directions == 2:
        splits.append((half, depth, 0, half))

    print(f"=== cross-fit-selected candidate vs unchanged peptide ===")
    print(f"out_dir: {out_dir}   arm: {args.arm}")
    print(f"estimator: pool={args.pool} directions={args.directions} "
          f"min_ref_valid={args.min_ref_valid} "
          f"energy_key={args.energy_key or 'default'}")
    for lo, hi, mlo, mhi in splits:
        print(f"  select on ranks [{lo},{hi}) by median energy, "
              f"measure both sides on [{mlo},{mhi})")
    if args.directions == 2:
        print("  the two directions are averaged per complex before any median")
    if args.select_reliability:
        print(f"selection gate: candidate must pass >= {args.select_reliability} "
              f"of the {half} selection folds to be eligible")
    print()
    print(f"  {'B':>3} {'cplx':>5} | {'dG REU':>8} {'w/l':>8} {'sign p':>9} "
          f"| {'d(valid)':>9} {'w/l':>8} {'sign p':>9} "
          f"| {'unmeas':>6} {'unserved':>8} {'no_base':>8} {'pool':>5}")

    def pick_and_measure(cands, ref, sel, meas):
        """Select on `sel`, measure candidate and reference on `meas`.

        Returns (delta_energy | None, delta_validity | None, status), where status is
        'unserved' (no draw was eligible to select), 'no_baseline' (the reference has no
        measurable energy on the measuring side, so there is nothing to beat),
        'unmeasurable' (something was selected but it passed no measuring fold) or 'ok'.
        The last three are all missing numbers rather than bad ones, and they are kept
        apart because they say different things about the run.
        """
        slo, shi = sel
        mlo, mhi = meas
        pick = [(c.median_energy_in(slo, shi), c) for c in cands
                if c.median_energy_in(slo, shi) is not None
                and c.n_pass_in(slo, shi) >= args.select_reliability]
        if not pick:
            return None, None, "unserved"
        chosen = min(pick, key=lambda t: t[0])[1]

        dv = None
        n_ho = len(chosen.folds_in(mlo, mhi))
        n_ref = len(ref.folds_in(mlo, mhi))
        if n_ho and n_ref:
            dv = (chosen.n_pass_in(mlo, mhi) / n_ho
                  - ref.n_pass_in(mlo, mhi) / n_ref)

        ref_e = ref.median_energy_in(mlo, mhi)
        got = chosen.median_energy_in(mlo, mhi)
        if ref_e is None:
            return None, dv, "no_baseline"
        if got is None:
            return None, dv, "unmeasurable"
        return got - ref_e, dv, "ok"

    results = []
    for B in rungs:
        e_d, e_w, e_l, unmeas = [], 0, 0, 0
        v_d, v_w, v_l = [], 0, 0
        n_cplx = 0
        n_unserved = 0
        n_no_baseline = 0
        pools = []
        for _pid, s in state.items():
            draws = s["draws"]
            ref = s["reference"]
            if ref is None or ref.n_folds(depth) < depth:
                continue
            # A reference too rarely valid to measure is a complex with no baseline, not
            # a complex the arm lost on. Counted and reported, never scored.
            if args.min_ref_valid and ref.n_pass(depth) < args.min_ref_valid:
                n_no_baseline += 1
                continue
            if args.pool == "all":
                cands = [draws[k] for k in sorted(draws)]
            else:
                if complete_prefix(draws, depth) < B:
                    continue
                cands = [draws[k] for k in sorted(draws) if k < B]
            if not cands:
                continue

            des, dvs, served = [], [], False
            nobase = 0
            for slo, shi, mlo, mhi in splits:
                de, dv, status = pick_and_measure(cands, ref, (slo, shi), (mlo, mhi))
                if status == "unserved":
                    continue
                served = True
                if status == "no_baseline":
                    nobase += 1
                if de is not None:
                    des.append(de)
                if dv is not None:
                    dvs.append(dv)
            if not served:
                # Every draw failed the gate on the selection folds, in every direction.
                # That is a real outcome of the rule -- the search had nothing it would
                # ship -- so the complex is counted as unserved rather than quietly
                # skipped, which would score the rule only on the complexes it answered.
                n_unserved += 1
                continue
            n_cplx += 1
            pools.append(len(cands))

            # Validity first, and for every served complex -- including the ones the
            # energy comparison has to drop. A pass rate of zero is a real answer.
            if dvs:
                dv = st.fmean(dvs)
                v_d.append(dv)
                if dv > 0:
                    v_w += 1
                elif dv < 0:
                    v_l += 1

            if not des:
                # The reference itself was unmeasurable on every measuring side: no
                # baseline to beat, same class as --min-ref-valid, and not a loss.
                if nobase:
                    n_no_baseline += 1
                else:
                    unmeas += 1
                continue
            de = st.fmean(des)
            e_d.append(de)
            if de < 0:
                e_w += 1
            elif de > 0:
                e_l += 1

        row = {
            "B": B if args.pool == "rungs" else "all",
            "n_complexes": n_cplx,
            "n_unserved": n_unserved,
            "n_no_baseline": n_no_baseline,
            "mean_pool": st.fmean(pools) if pools else None,
            "pool": args.pool,
            "directions": args.directions,
            "min_ref_valid": args.min_ref_valid,
            "energy_key": args.energy_key,
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
            "splits": [list(s) for s in splits],
        }
        results.append(row)
        em = "-" if row["energy_median_delta"] is None else f"{row['energy_median_delta']:+8.2f}"
        vm = "-" if row["validity_median_delta"] is None else f"{row['validity_median_delta']:+9.3f}"
        ep = "-" if row["energy_sign_p"] is None else f"{row['energy_sign_p']:9.2e}"
        vp = "-" if row["validity_sign_p"] is None else f"{row['validity_sign_p']:9.2e}"
        pl = "-" if row["mean_pool"] is None else f"{row['mean_pool']:5.1f}"
        print(f"  {str(row['B']):>3} {n_cplx:>5} | {em} {f'{e_w}/{e_l}':>8} {ep} "
              f"| {vm} {f'{v_w}/{v_l}':>8} {vp} "
              f"| {unmeas:>6} {n_unserved:>8} {n_no_baseline:>8} {pl:>5}")

    print("\n  dG REU     selected candidate's median energy minus the peptide's, both")
    print("             over the held-out folds. Negative = the search's pick is better.")
    print("  d(valid)   the same candidate's pass RATE minus the peptide's, held-out")
    print("             folds. Negative = the search's pick folds acceptably less often.")
    print("  unmeas     selected on the first half, passed no fold in the second, so it")
    print("             has no held-out energy. Counted in validity, not in energy.")
    print("  no_base    reference below --min-ref-valid: no baseline to beat. Excluded")
    print("             from both columns, because it is a missing measurement.")
    print("  pool       mean candidates selected from, per complex.")
    print("  Energies are Rosetta ref2015 REU, NOT measured binding free energies.")

    # Named after the arm and the gate, so reporting a second arm or a second scoring
    # version cannot overwrite the first's file.
    stem = "selected_candidate_report"
    if args.arm != DRAW_ARM:
        stem += f"_{args.arm}"
    if args.scoring_version:
        stem += f"_{args.scoring_version}"
    dest = Path(args.json) if args.json else out_dir / f"{stem}.json"
    dest.write_text(json.dumps(
        {"out_dir": str(out_dir), "depth": depth, "half": half,
         "draw_arm": args.arm, "scoring_version": args.scoring_version,
         "rungs": results}, indent=2))
    print(f"\n[selected-candidate-report] -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
