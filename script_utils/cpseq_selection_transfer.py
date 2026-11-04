#!/usr/bin/env python3
"""Decompose a predictor-setting change into reference, selection, and scoring.

The paired report answers "does the arm still beat the peptide when the arm is allowed
to RE-SELECT under the new setting". That is one question, and it is the one the pilot
estimator asks, because selection is part of the estimator. It is not the only question,
and the two are easy to conflate:

  A. re-selected   select among the new run's folds, measure there.  <- the report
  B. transferred   take the sequence the OLD run selected, and measure THAT sequence in
                   the new run. Answers "does the winner we would have shipped survive?"

If A and B agree, selection is not where the change lives. If B is much better than A,
the new setting is picking worse candidates. If both moved by the same amount as the
REFERENCE moved, nothing happened to the arm at all and the baseline moved under it.

So this prints, per complex and pooled, on exactly the complexes both runs scored:
  * the reference's own median energy in each run     (did the baseline move?)
  * the selected candidate's median energy in each run
  * A and B as deltas against each run's own reference
  * whether the two runs even selected the same sequence

Selection and measurement reuse `cpseq_ladder_report`'s own Sequence objects and the
report's rule -- lowest median energy over the selection folds, measured over the
disjoint half, both directions averaged -- so this cannot drift from the number it is
decomposing.

Usage:
    python script_utils/cpseq_selection_transfer.py \
        --off-dir <off run> --on-dir <on run> \
        --arm llm_search --scoring-version v3-cis-amide-muco-windows
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_ladder_report import build, load  # noqa: E402

DEPTH = 10
HALF = DEPTH // 2
ENERGY_KEY = "refined_dG_separated"
MIN_REF_VALID = 3


def median_in(seq, lo: int, hi: int):
    """Median energy over the passing folds in [lo,hi).

    Delegates to `Sequence.median_energy_in`, which is the report's own rule, rather
    than recomputing it: this script exists to decompose that number, so a second
    definition of "the candidate's energy" would be free to disagree with it.
    """
    return seq.median_energy_in(lo, hi)


def pick(draws: dict, lo: int, hi: int):
    """The draw with the lowest median energy over the selection folds."""
    best, best_e = None, None
    for k in sorted(draws):
        e = median_in(draws[k], lo, hi)
        if e is None:
            continue
        if best_e is None or e < best_e:
            best, best_e = k, e
    return best


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off-dir", required=True)
    ap.add_argument("--on-dir", required=True)
    ap.add_argument("--arm", default="llm_search")
    ap.add_argument("--scoring-version", required=True)
    ap.add_argument("--label-off", default="off")
    ap.add_argument("--label-on", default="on")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    states = {}
    for tag, d in (("off", args.off_dir), ("on", args.on_dir)):
        states[tag] = build(load(Path(d)), draw_arm=args.arm,
                            scoring_version=args.scoring_version,
                            energy_key=ENERGY_KEY)

    splits = [(0, HALF, HALF, DEPTH), (HALF, DEPTH, 0, HALF)]
    rows = []
    for pid in sorted(set(states["off"]) & set(states["on"])):
        so, sn = states["off"][pid], states["on"][pid]
        ro, rn = so["reference"], sn["reference"]
        if ro is None or rn is None:
            continue
        if ro.n_folds(DEPTH) < DEPTH or rn.n_folds(DEPTH) < DEPTH:
            continue
        if ro.n_pass(DEPTH) < MIN_REF_VALID or rn.n_pass(DEPTH) < MIN_REF_VALID:
            continue

        refo, refn, a_vals, b_vals, same = [], [], [], [], []
        for slo, shi, mlo, mhi in splits:
            ko = pick(so["draws"], slo, shi)
            kn = pick(sn["draws"], slo, shi)
            r_o = median_in(ro, mlo, mhi)
            r_n = median_in(rn, mlo, mhi)
            if r_o is not None:
                refo.append(r_o)
            if r_n is not None:
                refn.append(r_n)
            # A: the ON run's own pick, measured in ON.
            if kn is not None and r_n is not None:
                e = median_in(sn["draws"][kn], mlo, mhi)
                if e is not None:
                    a_vals.append(e - r_n)
            # B: the OFF run's pick, measured in ON. The same draw index is the same
            # sequence in both runs by construction -- the proposals were copied, not
            # re-proposed -- which is what makes this transfer well defined.
            if ko is not None and r_n is not None and ko in sn["draws"]:
                e = median_in(sn["draws"][ko], mlo, mhi)
                if e is not None:
                    b_vals.append(e - r_n)
            if ko is not None and kn is not None:
                same.append(ko == kn)

        rows.append({
            "parent_id": pid,
            "ref_median_off": st.fmean(refo) if refo else None,
            "ref_median_on": st.fmean(refn) if refn else None,
            "ref_valid_off": ro.n_pass(DEPTH),
            "ref_valid_on": rn.n_pass(DEPTH),
            "reselected_on": st.fmean(a_vals) if a_vals else None,
            "transferred_on": st.fmean(b_vals) if b_vals else None,
            "same_pick": (all(same) if same else None),
            "n_same": sum(same), "n_dirs": len(same),
        })

    print(f"=== selection transfer: {args.label_off} -> {args.label_on} "
          f"(arm {args.arm}) ===")
    print(f"complexes scored by BOTH runs at full depth: {len(rows)}\n")
    print(f"  {'complex':<24} {'refE_off':>9} {'refE_on':>9} {'refv':>7} "
          f"{'A resel':>8} {'B transf':>9} {'pick':>6}")
    for r in rows:
        f = lambda v: "        -" if v is None else f"{v:+9.2f}"  # noqa: E731
        print(f"  {r['parent_id']:<24} {f(r['ref_median_off'])} {f(r['ref_median_on'])} "
              f"{r['ref_valid_off']:>3}->{r['ref_valid_on']:<3} "
              f"{('%+8.2f' % r['reselected_on']) if r['reselected_on'] is not None else '       -'} "
              f"{('%+9.2f' % r['transferred_on']) if r['transferred_on'] is not None else '        -'} "
              f"{r['n_same']}/{r['n_dirs']}")

    def med(key):
        v = [r[key] for r in rows if r[key] is not None]
        return st.median(v) if v else None

    print()
    print(f"  reference's own median energy:  {args.label_off} {med('ref_median_off'):+.2f}"
          f"   {args.label_on} {med('ref_median_on'):+.2f}"
          f"   (change {med('ref_median_on') - med('ref_median_off'):+.2f})")
    print(f"  A  re-selected under {args.label_on:<9} dG vs its own reference: "
          f"{med('reselected_on'):+.2f}")
    print(f"  B  {args.label_off}-selected, scored under {args.label_on}: "
          f"{med('transferred_on'):+.2f}")
    agree = sum(1 for r in rows if r["same_pick"])
    print(f"  the two runs chose the SAME sequence in both directions on "
          f"{agree}/{len(rows)} complexes")
    print()
    print("  A vs B separates 're-picking rescues the arm' from 'the winner transfers'.")
    print("  Either compared with the reference's own change says whether the arm moved")
    print("  or the BASELINE moved under it.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "off_dir": args.off_dir, "on_dir": args.on_dir, "arm": args.arm,
            "scoring_version": args.scoring_version, "per_complex": rows,
            "median_ref_off": med("ref_median_off"),
            "median_ref_on": med("ref_median_on"),
            "median_reselected_on": med("reselected_on"),
            "median_transferred_on": med("transferred_on"),
            "same_pick_complexes": agree, "n_complexes": len(rows),
        }, indent=2))
        print(f"[selection-transfer] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
