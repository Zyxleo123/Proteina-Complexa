#!/usr/bin/env python3
"""Pair one arm's per-complex deltas across two runs that differ by one setting.

The question
------------
`cpseq_selected_candidate_report.py --pilot` gives an arm's median ΔdG against the
native peptide. Run it on two directories and you have two medians -- but the difference
of two medians over two complex sets is not a measurement of the change between them.
Two things move it that have nothing to do with the setting under test:

  * **The complex set moves.** A complex is scored only if its reference folds valid at
    least `--min-ref-valid` times. Potentials raise the reference's pass rate, so the ON
    run can score complexes the OFF run reported as `no_base`. A median taken over a
    larger, differently-selected set is a different estimand.
  * **Complexes differ enormously from each other.** The per-complex spread on this
    cohort is ~5.7 REU (the arm table's own sd), against the ~1 REU effect this check is
    trying to resolve. Unpaired, the comparison is dominated by which complexes are in
    it.

So this script pairs per complex, and reports the FIXED set and the enlarged set
separately -- which is what the plan asks for in as many words.

Input is the `per_complex` block of two report JSONs, so the estimator is the report's
own, run twice. Nothing about the selection rule is reimplemented here; this script does
arithmetic on its output and nothing else. That matters because `llm_search`
reproducing its recorded -7.23 REU to the digit is what certifies that estimator, and a
second implementation would be free to disagree with the certified one.

Usage:
    python script_utils/cpseq_potentials_compare.py \
        --off pilot_llm_search_OFF.json --on pilot_llm_search_ON.json \
        --label-off "potentials off" --label-on "potentials on"
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p at q=0.5, ties excluded by the caller.

    Same rule as `cpseq_ladder_report.sign_test`; duplicated rather than imported only
    so that this script runs against a pair of JSON files with no repo on the path,
    which is how the published bundles are meant to be replayable.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def per_complex(path: Path) -> tuple[dict[str, dict], dict]:
    """The per-complex rows of a report JSON, keyed by parent, plus the aggregate row."""
    doc = json.loads(path.read_text())
    rungs = doc.get("rungs") or []
    if not rungs:
        raise SystemExit(f"{path}: no rungs in this report")
    if len(rungs) > 1:
        raise SystemExit(
            f"{path}: {len(rungs)} rungs. This comparison is for the --pilot estimator, "
            f"which has exactly one ('all'). A budget-rung report answers a different "
            f"question per rung and the two are not comparable -- see section 0.2."
        )
    row = rungs[0]
    rows = row.get("per_complex")
    if rows is None:
        raise SystemExit(
            f"{path}: no `per_complex` block. It is written by "
            f"cpseq_selected_candidate_report.py; a JSON produced before that key "
            f"existed has to be regenerated (it redraws from the journals, no folds "
            f"needed)."
        )
    return {r["parent_id"]: r for r in rows}, row


def describe(deltas: list[float]) -> str:
    if not deltas:
        return "n=0"
    med = st.median(deltas)
    mean = st.fmean(deltas)
    sd = st.stdev(deltas) if len(deltas) > 1 else 0.0
    se = sd / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    ses = f"{mean / se:.2f} SE" if se else "- SE"
    return (f"n={len(deltas)}  median={med:+.2f}  mean={mean:+.2f}  "
            f"sd={sd:.2f}  se={se:.2f}  ({ses})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", required=True, help="report JSON for the baseline run")
    ap.add_argument("--on", required=True, help="report JSON for the changed run")
    ap.add_argument("--label-off", default="off")
    ap.add_argument("--label-on", default="on")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    off, off_row = per_complex(Path(args.off))
    on, on_row = per_complex(Path(args.on))

    print("=== paired per-complex comparison, one setting changed ===")
    print(f"{args.label_off:>18}: {args.off}")
    print(f"{args.label_on:>18}: {args.on}")
    for name, row in ((args.label_off, off_row), (args.label_on, on_row)):
        print(f"  {name:>16} aggregate: n={row['n_complexes']} "
              f"dG={row['energy_median_delta']} "
              f"w/l={row['energy_wins']}/{row['energy_losses']} "
              f"no_base={row['n_no_baseline']} unserved={row['n_unserved']} "
              f"unmeas={row['energy_unmeasurable']} "
              f"draws={row['mean_draws']}")

    # The two runs must be the same arm and the same gate, or the pairing is comparing
    # two different things per complex and would look like a clean paired design while
    # doing it.
    for key in ("draw_arm", "scoring_version"):
        a = json.loads(Path(args.off).read_text()).get(key)
        b = json.loads(Path(args.on).read_text()).get(key)
        if a != b:
            print(f"\nREFUSING: the two reports disagree about {key}: {a!r} vs {b!r}. "
                  f"Pairing them would attribute an arm or gate difference to the "
                  f"setting under test.")
            return 1
    for key in ("draws", "directions", "min_ref_valid", "energy_key"):
        if off_row.get(key) != on_row.get(key):
            print(f"\nREFUSING: the two reports used a different estimator: "
                  f"{key}={off_row.get(key)!r} vs {on_row.get(key)!r}. The estimator is "
                  f"part of the number; only one thing may differ between these runs.")
            return 1

    both = sorted(set(off) | set(on))
    scored_off = {p for p, r in off.items() if r["status"] == "ok"}
    scored_on = {p for p, r in on.items() if r["status"] == "ok"}
    fixed = sorted(scored_off & scored_on)

    print(f"\n--- per complex ---")
    print(f"  {'complex':<28} {args.label_off:>12} {args.label_on:>12} "
          f"{'change':>9}   status")
    rows_out = []
    for pid in both:
        o, n = off.get(pid), on.get(pid)
        oe = o.get("energy_delta") if o else None
        ne = n.get("energy_delta") if n else None
        ostat = o["status"] if o else "not_folded"
        nstat = n["status"] if n else "not_folded"
        chg = f"{ne - oe:+9.2f}" if (oe is not None and ne is not None) else "        -"
        tag = ostat if ostat == nstat else f"{ostat} -> {nstat}"
        # A complex absent from one run was never FOLDED there -- typically because the
        # run covered a chosen subset -- which is a difference in what was submitted, not
        # a complex that changed class under the setting. Flagging it as a class change
        # invites reading the subset boundary as a result.
        mark = ""
        if ostat != nstat:
            mark = ("   <-- not folded in this run"
                    if "not_folded" in (ostat, nstat) else "   <-- class change")
        print(f"  {pid:<28} {('%+12.2f' % oe) if oe is not None else '           -'} "
              f"{('%+12.2f' % ne) if ne is not None else '           -'} {chg}   "
              f"{tag}{mark}")
        rows_out.append({"parent_id": pid, "off": oe, "on": ne,
                         "status_off": ostat, "status_on": nstat,
                         "paired": pid in fixed})

    # The headline. Paired on the complexes BOTH runs scored, which is the only set on
    # which "the arm's advantage held" is a statement about the setting rather than
    # about the sample.
    diffs = [on[p]["energy_delta"] - off[p]["energy_delta"] for p in fixed]
    print(f"\n--- FIXED set: the {len(fixed)} complex(es) both runs scored ---")
    off_v = [off[p]["energy_delta"] for p in fixed]
    on_v = [on[p]["energy_delta"] for p in fixed]
    print(f"  {args.label_off:>18} ΔdG vs peptide   median={st.median(off_v):+.2f}"
          if off_v else "  (empty)")
    if on_v:
        print(f"  {args.label_on:>18} ΔdG vs peptide   median={st.median(on_v):+.2f}")
    if diffs:
        worse = sum(1 for d in diffs if d > 0)
        better = sum(1 for d in diffs if d < 0)
        print(f"  change ({args.label_on} - {args.label_off}): {describe(diffs)}")
        print(f"  the arm got better on {better}, worse on {worse}, "
              f"sign p={sign_test(better, worse):.3g}")
        print(f"  (a POSITIVE change means the arm's advantage over the peptide shrank "
              f"under {args.label_on})")

    # The enlarged set, reported separately and never mixed into the number above.
    gained = sorted(scored_on - scored_off)
    lost = sorted(scored_off - scored_on)
    print(f"\n--- the complex set moved ---")
    print(f"  scored by {args.label_off:>14}: {len(scored_off)}")
    print(f"  scored by {args.label_on:>14}: {len(scored_on)}")
    print(f"  gained under {args.label_on}: {len(gained)}"
          + (f"  ({', '.join(gained)})" if gained else ""))
    print(f"  lost under   {args.label_on}: {len(lost)}"
          + (f"  ({', '.join(lost)})" if lost else ""))
    if gained:
        gv = [on[p]["energy_delta"] for p in gained]
        print(f"  the gained complexes' own ΔdG under {args.label_on}: "
              f"median={st.median(gv):+.2f}  (n={len(gv)})")
        print(f"  ENLARGED-set median over all {len(scored_on)} that "
              f"{args.label_on} scored: {st.median([on[p]['energy_delta'] for p in scored_on]):+.2f}")
        print(f"  Report this separately from the fixed-set number above. An aggregate "
              f"that moved because more\n  complexes cleared the baseline has not told "
              f"you anything about the arm.")

    out = {
        "off": {"path": args.off, "label": args.label_off, "aggregate": off_row},
        "on": {"path": args.on, "label": args.label_on, "aggregate": on_row},
        "per_complex": rows_out,
        "fixed_set": fixed,
        "fixed_n": len(fixed),
        "fixed_median_off": st.median(off_v) if off_v else None,
        "fixed_median_on": st.median(on_v) if on_v else None,
        "paired_change_median": st.median(diffs) if diffs else None,
        "paired_change_mean": st.fmean(diffs) if diffs else None,
        "paired_change_sd": st.stdev(diffs) if len(diffs) > 1 else None,
        "paired_better": sum(1 for d in diffs if d < 0),
        "paired_worse": sum(1 for d in diffs if d > 0),
        "paired_sign_p": sign_test(sum(1 for d in diffs if d < 0),
                                   sum(1 for d in diffs if d > 0)) if diffs else None,
        "gained": gained,
        "lost": lost,
        "enlarged_n": len(scored_on),
        "enlarged_median_on": (st.median([on[p]["energy_delta"] for p in scored_on])
                               if scored_on else None),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n[potentials-compare] -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
