#!/usr/bin/env python3
"""Arm vs arm, head to head, cross-fitted. Does the LLM proposer beat random draws?

`cpseq_ladder_report.py` answers "is this arm better than the unchanged peptide", once
per arm. That is not the same question as "is arm D better than arm A", and the
difference matters: both arms could beat the reference, or neither, while one is clearly
better than the other. This script does the paired comparison directly.

Everything about the cross-fit is **imported** from `cpseq_ladder_report`, not
reimplemented -- the same `Sequence`, the same `build`, the same fold-rank truncation.
A second implementation of "choose on the first half, measure on the second" would be
the obvious place for the two arms to end up being scored by subtly different rules.

The comparison, per complex and per rung B
------------------------------------------
For each arm independently: among its first B draws, keep those clearing the reliability
bar, pick the one with the best median energy over folds `0..depth/2`, then report that
candidate's median energy over folds `depth/2..depth`. Selection and measurement share
no fold, so neither arm can win on favourable noise.

Then the two arms are compared **within the complex** and counted across complexes with
a sign test. The complex is the unit of analysis, never the fold: ten folds of one
sequence are repeated measures, and this project has already published a pooled p-value
of 2.6e-6 that became 0.031 once the target was the unit.

Three outcomes are reported because they are different claims:

- **dG, cross-fitted** -- the headline. Negative means arm D found a lower-energy
  sequence than arm A on folds neither arm's selection touched.
- **dG, best-of-B** -- the same thing without the cross-fit, shown only so the gap
  between the two columns is visible. At B=8 the random arm's own advantage over the
  reference went from -4.66 to -0.06 across exactly that gap, so a head-to-head number
  that appears only in this column is selection bias.
- **validity** -- whether an arm finds an acceptable sequence AT ALL. An arm with worse
  energy but better coverage is not strictly worse, and for a corpus coverage is itself
  the deliverable.

A complex contributes to the energy comparison only when BOTH arms have a measurable
candidate. Complexes where one arm found nothing are reported separately as coverage,
never dropped silently -- an arm that only answers on easy complexes would otherwise
look better than one that answers everywhere.

Usage:
    python script_utils/cpseq_arm_compare.py --out-dir $ZFS/cpseq/llm_ladder
    python script_utils/cpseq_arm_compare.py --out-dir $ZFS/cpseq/llm_ladder \
        --arm-a random_control --arm-b llm_search --depth 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpseq_ladder_report import (  # noqa: E402
    ENERGY_KEY,
    Sequence,
    build,
    complete_prefix,
    load,
    sign_test,
    warn_if_mixed,
)


def arm_pick(
    draws: dict[int, Sequence],
    b: int,
    depth: int,
    k_needed: int,
    *,
    crossfit: bool,
) -> tuple[float | None, str]:
    """One arm's chosen candidate's energy at rung B, and why it is what it is.

    Returns `(energy, status)`. The status is never discarded by the caller: "no draw
    cleared the reliability bar" and "the chosen candidate passed no measurement fold"
    are different facts, and both differ from a number.
    """
    prefix = [draws[i] for i in range(b) if draws.get(i) is not None]
    if len(prefix) < b:
        return None, f"only {len(prefix)}/{b} draws present"
    half = depth // 2
    if not crossfit:
        ok = [d for d in prefix
              if d.n_pass(depth) >= k_needed and d.median_energy(depth) is not None]
        if not ok:
            return None, "no draw cleared the reliability bar"
        return min(d.median_energy(depth) for d in ok), "ok"

    if half < 1:
        return None, "fold depth too small to hold out a half"
    # Chosen on the FIRST half only.
    pick = [(d.median_energy_in(0, half), d) for d in prefix
            if d.median_energy_in(0, half) is not None]
    if not pick:
        return None, "no draw passed a selection-half fold"
    chosen = min(pick, key=lambda t: t[0])[1]
    got = chosen.median_energy_in(half, depth)
    if got is None:
        # Chosen on the first half, then passed no fold in the second. Not a win and not
        # a loss: it is a candidate the selection could not stand behind, and scoring it
        # either way would flatter or punish the arm for a missing measurement.
        return None, "chosen candidate passed no measurement fold"
    return got, "ok"


def arm_validity(draws: dict[int, Sequence], b: int, depth: int, k_needed: int) -> dict:
    """Whether this arm found anything acceptable at rung B, and how often it passed."""
    prefix = [draws[i] for i in range(b) if draws.get(i) is not None]
    n_pass = sum(d.n_pass(depth) for d in prefix)
    n_folds = sum(d.n_folds(depth) for d in prefix)
    return {
        "has_reliable": any(d.n_pass(depth) >= k_needed for d in prefix),
        "has_any_pass": any(d.n_pass(depth) >= 1 for d in prefix),
        "fold_pass_rate": n_pass / n_folds if n_folds else None,
        "n_folds": n_folds,
    }


def compare(
    state_a: dict[str, dict],
    state_b: dict[str, dict],
    rungs: list[int],
    depth: int,
    reliability: float,
) -> list[dict]:
    rows = []
    k_needed = math.ceil(reliability * depth)
    half = depth // 2
    shared = sorted(set(state_a) & set(state_b))

    for b in rungs:
        # A complex is usable at this rung only when BOTH arms have B complete draws and
        # the shared reference is fully folded. Matching this way is the whole point: a
        # rung where arm D has 8 draws and arm A has 4 compares budgets, not proposers.
        usable = []
        for pid in shared:
            sa, sb = state_a[pid], state_b[pid]
            if complete_prefix(sa["draws"], depth) < b:
                continue
            if complete_prefix(sb["draws"], depth) < b:
                continue
            usable.append(pid)
        if not usable:
            continue

        xf_diffs: list[float] = []
        xf_wins = xf_losses = 0
        bob_diffs: list[float] = []
        bob_wins = bob_losses = 0
        a_only = b_only = neither = 0
        cov_a = cov_b = 0
        rate_a: list[float] = []
        rate_b: list[float] = []
        per_complex = []

        for pid in usable:
            da, db = state_a[pid]["draws"], state_b[pid]["draws"]
            va = arm_validity(da, b, depth, k_needed)
            vb = arm_validity(db, b, depth, k_needed)
            cov_a += int(va["has_reliable"])
            cov_b += int(vb["has_reliable"])
            if va["fold_pass_rate"] is not None:
                rate_a.append(va["fold_pass_rate"])
            if vb["fold_pass_rate"] is not None:
                rate_b.append(vb["fold_pass_rate"])

            ea, sta = arm_pick(da, b, depth, k_needed, crossfit=True)
            eb, stb = arm_pick(db, b, depth, k_needed, crossfit=True)
            fa, _ = arm_pick(da, b, depth, k_needed, crossfit=False)
            fb, _ = arm_pick(db, b, depth, k_needed, crossfit=False)

            if ea is not None and eb is not None:
                d = eb - ea
                xf_diffs.append(d)
                if d < 0:
                    xf_wins += 1
                elif d > 0:
                    xf_losses += 1
            elif eb is not None:
                b_only += 1
            elif ea is not None:
                a_only += 1
            else:
                neither += 1

            if fa is not None and fb is not None:
                d2 = fb - fa
                bob_diffs.append(d2)
                if d2 < 0:
                    bob_wins += 1
                elif d2 > 0:
                    bob_losses += 1

            per_complex.append({
                "parent_id": pid,
                "crossfit_energy_arm_a": ea, "crossfit_status_arm_a": sta,
                "crossfit_energy_arm_b": eb, "crossfit_status_arm_b": stb,
                "bestofb_energy_arm_a": fa, "bestofb_energy_arm_b": fb,
                "reliable_arm_a": va["has_reliable"], "reliable_arm_b": vb["has_reliable"],
                "fold_pass_rate_arm_a": va["fold_pass_rate"],
                "fold_pass_rate_arm_b": vb["fold_pass_rate"],
            })

        rows.append({
            "B": b,
            "n_complexes": len(usable),
            "fold_depth": depth,
            "reliability_k_of_n": f"{k_needed}/{depth}",
            "select_folds": [0, half],
            "measure_folds": [half, depth],
            "n_energy_pairs": len(xf_diffs),
            "crossfit_median_delta": st.median(xf_diffs) if xf_diffs else None,
            "crossfit_wins_b": xf_wins,
            "crossfit_losses_b": xf_losses,
            "crossfit_sign_p": (
                sign_test(xf_wins, xf_losses) if (xf_wins + xf_losses) else None
            ),
            "bestofb_median_delta": st.median(bob_diffs) if bob_diffs else None,
            "bestofb_wins_b": bob_wins,
            "bestofb_losses_b": bob_losses,
            "bestofb_sign_p": (
                sign_test(bob_wins, bob_losses) if (bob_wins + bob_losses) else None
            ),
            "measurable_arm_b_only": b_only,
            "measurable_arm_a_only": a_only,
            "measurable_neither": neither,
            "coverage_reliable_arm_a": cov_a / len(usable),
            "coverage_reliable_arm_b": cov_b / len(usable),
            "fold_pass_rate_arm_a": st.mean(rate_a) if rate_a else None,
            "fold_pass_rate_arm_b": st.mean(rate_b) if rate_b else None,
            "per_complex": per_complex,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm-a", default="random_control", help="the baseline arm")
    ap.add_argument("--arm-b", default="llm_search", help="the arm under test")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--reliability", type=float, default=0.5)
    ap.add_argument("--json", default="arm_compare.json")
    ap.add_argument(
        "--scoring-version",
        default=None,
        help="keep only rows written by this scoring_version. Required in practice for "
        "any run spanning the 2026-10-22 gate recalibration: a baseline scored by the "
        "retired gate against an arm scored by the current one is not a comparison.",
    )
    ap.add_argument(
        "--energy-key",
        default=ENERGY_KEY,
        help="the row field candidates are ranked and scored by. Default is the "
        "screening objective. `refined_dG_separated_per_1000A2` re-ranks on dG per "
        "buried area, which rewards interface quality rather than interface size -- see "
        "the ENERGY_KEY note in cpseq_ladder_report.py for what that changes and for the "
        "caveat about the per-fold representative sample.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cohort = json.loads((out_dir / "cohort.json").read_text())
    depth = args.depth or len(cohort["seeds"])
    rungs = list(cohort["budget_rungs"])

    per_parent = load(out_dir)
    mixed = warn_if_mixed(per_parent, args.scoring_version)
    state_a = build(per_parent, draw_arm=args.arm_a, energy_key=args.energy_key,
                    scoring_version=args.scoring_version)
    state_b = build(per_parent, draw_arm=args.arm_b, energy_key=args.energy_key,
                    scoring_version=args.scoring_version)
    # `build` returns an entry per complex even when that arm has no draws there, so the
    # arms are filtered to complexes where they actually have some.
    state_a = {p: s for p, s in state_a.items() if s["draws"]}
    state_b = {p: s for p, s in state_b.items() if s["draws"]}

    print(f"=== arm comparison, {out_dir} ===")
    print(f"baseline  (arm A): {args.arm_a}  -- {len(state_a)} complex(es) with draws")
    print(f"under test (arm B): {args.arm_b}  -- {len(state_b)} complex(es) with draws")
    print(f"fold depth {depth}; selection folds 0-{depth // 2 - 1}, "
          f"measurement folds {depth // 2}-{depth - 1}")
    if not state_b:
        print(f"\n[arm-compare] {args.arm_b} has no folded draws yet -- nothing to "
              f"compare. Fold it first (ladder --llm-round N).")
        return 1

    rows = compare(state_a, state_b, rungs, depth, args.reliability)
    if not rows:
        both = sorted(set(state_a) & set(state_b))
        print(f"\n[arm-compare] no rung is complete in BOTH arms yet "
              f"({len(both)} shared complex(es)). Per complex, complete draws:")
        for pid in both[:25]:
            print(f"  {pid:28s} {args.arm_a}: "
                  f"{complete_prefix(state_a[pid]['draws'], depth):>3}   "
                  f"{args.arm_b}: {complete_prefix(state_b[pid]['draws'], depth):>3}")
        return 1

    print(f"\n{'':>3} {'':>5} | {'validity: reliable cand.':^26} | "
          f"{'dG cross-fitted (B vs A)':^30} | {'dG best-of-B':^22}")
    print(f"{'B':>3} {'cplx':>5} | {'armA':>6} {'armB':>6} {'passA':>6} {'passB':>6} | "
          f"{'dREU':>7} {'w/l':>7} {'sign p':>7} {'n':>3} | "
          f"{'dREU':>7} {'w/l':>7} {'p':>5}")
    for r in rows:
        xd = "-" if r["crossfit_median_delta"] is None else f"{r['crossfit_median_delta']:+7.2f}"
        xp = "-" if r["crossfit_sign_p"] is None else f"{r['crossfit_sign_p']:.3f}"
        bd = "-" if r["bestofb_median_delta"] is None else f"{r['bestofb_median_delta']:+7.2f}"
        bp = "-" if r["bestofb_sign_p"] is None else f"{r['bestofb_sign_p']:.3f}"
        pa = "-" if r["fold_pass_rate_arm_a"] is None else f"{r['fold_pass_rate_arm_a']:.3f}"
        pb = "-" if r["fold_pass_rate_arm_b"] is None else f"{r['fold_pass_rate_arm_b']:.3f}"
        print(f"{r['B']:>3} {r['n_complexes']:>5} | "
              f"{r['coverage_reliable_arm_a']:>6.3f} {r['coverage_reliable_arm_b']:>6.3f} "
              f"{pa:>6} {pb:>6} | "
              f"{xd:>7} {str(r['crossfit_wins_b']) + '/' + str(r['crossfit_losses_b']):>7} "
              f"{xp:>7} {r['n_energy_pairs']:>3} | "
              f"{bd:>7} {str(r['bestofb_wins_b']) + '/' + str(r['bestofb_losses_b']):>7} "
              f"{bp:>5}")

    report = {
        "out_dir": str(out_dir), "arm_a": args.arm_a, "arm_b": args.arm_b,
        "fold_depth": depth, "reliability": args.reliability,
        "energy_key": args.energy_key,
        "scoring_version": args.scoring_version,
        "scoring_version_mixed": mixed, "rungs": rows,
    }
    path = out_dir / args.json
    path.write_text(json.dumps(report, indent=2))
    print(f"\n[arm-compare] -> {path}")

    print("\nHow to read this:")
    print(f"  dREU is arm B minus arm A, per complex, then the median across complexes.")
    print(f"  NEGATIVE means {args.arm_b} found the lower-energy sequence.")
    print(f"  w/l counts complexes where {args.arm_b} won / lost; ties excluded.")
    print(f"  The cross-fitted column is the ANSWER. Candidates are chosen on folds "
          f"0-{depth // 2 - 1} and scored on folds {depth // 2}-{depth - 1}, so neither "
          f"arm can win by having had luckier folds during selection.")
    print(f"  The best-of-B column uses the same folds to choose and to score. It is "
          f"shown only so the gap is visible: the random arm's own advantage over the "
          f"unchanged peptide fell from -4.66 to -0.06 REU across exactly that gap at "
          f"B=8, so anything that appears here and not in the cross-fitted column is "
          f"selection bias, not a result.")
    print(f"  validity: 'reliable cand.' is the fraction of complexes where the arm "
          f"found a candidate passing the geometry gate in at least "
          f"{math.ceil(args.reliability * depth)}/{depth} folds. 'pass' is the mean "
          f"per-fold pass rate. An arm with worse energy but better coverage is not "
          f"strictly worse.")
    worst = [r for r in rows if r["measurable_arm_a_only"] or r["measurable_arm_b_only"]
             or r["measurable_neither"]]
    if worst:
        print("\n  Complexes excluded from the energy comparison at some rung because "
              "one arm had nothing measurable:")
        for r in worst:
            print(f"    B={r['B']}: {args.arm_b} only {r['measurable_arm_b_only']}, "
                  f"{args.arm_a} only {r['measurable_arm_a_only']}, "
                  f"neither {r['measurable_neither']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
