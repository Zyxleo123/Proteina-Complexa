#!/usr/bin/env python3
"""The cross-fit-selected candidate's INTERFACE against the unchanged peptide's.

`cpseq_selected_candidate_report.py` reports that candidate's energy and validity.
This reports what its interface is made of: confidence, buried unsatisfied polar atoms,
hydrogen bonds, shape complementarity and buried area -- the columns that say whether a
candidate binds better or merely binds BIGGER.

Why it exists
-------------
A run of this comparison on `llm_search` at 17 complexes found buried area up +119 A^2
(10 wins / 1 loss, p=0.012) while buried-unsat and interface hydrogen bonds did not move
(3/7 p=0.344 and 5/3 p=0.727). That pattern has an explanation in the objective rather
than in the proposer. Measured across 480 folds of this run by `cpseq_unsat_probe.py`,
within parent: buried area against the screening objective is rho -0.464 (15/17 parents
negative), while Rosetta's refined buried-unsat count against it is only +0.204. The
search is ranked on an energy that rewards burying surface, and burying surface is what
strands polar atoms -- so a proposer doing exactly as instructed produces that table.

`--energy-key` is therefore the point of this script, not a convenience. Re-ranking on
`refined_dG_separated_per_1000A2`, already recorded on every row, reverses those two
correlations (area falls to +0.255, unsat rises to +0.493, 16/17 parents). Running this
report under both keys asks whether the interface columns follow the objective, using the
folds already on disk and predicting nothing new.

What is held fixed
------------------
Selection is the imported cross-fit rule and nothing else: lowest median energy over
folds [0, half), measured over [half, depth). Candidate and peptide are measured over
the SAME folds, which is the rule the withdrawn best-of-B result broke. A candidate that
passes no measurement fold is `unmeasurable`, never a loss.

The complex is the unit of analysis. Ten folds of one sequence are repeated measures, and
a test pooled over folds counts one complex as ten -- this project has already published
a pooled 2.6e-6 that became 0.031 once the target was the unit.

Usage:
    python script_utils/cpseq_interface_compare.py \
        --out-dir /data/user_data/yixiz/cpseq/llm_ladder --arm llm_search
    python script_utils/cpseq_interface_compare.py \
        --out-dir /data/user_data/yixiz/cpseq/llm_ladder --arm llm_search \
        --energy-key refined_dG_separated_per_1000A2
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpseq_ladder_report import (  # noqa: E402
    ENERGY_KEY,
    Sequence,
    build,
    load,
    sign_test,
    warn_if_mixed,
)

# (label, row field, higher-is-better). The nested confidence fields are addressed as
# `confidence.<key>`; everything else is a flat row field.
#
# Raw AND refined buried-unsat are both listed deliberately. Relax repairs part of the
# defect -- `cpseq_unsat_probe.py` measured raw against refined at only rho 0.336 within
# parent -- so they are different claims, and a table carrying one of them silently would
# invite reading a relaxed number as what the predictor produced.
METRICS: tuple[tuple[str, str, bool], ...] = (
    ("pair ipTM", "confidence.protein_iptm", True),
    ("interface pLDDT", "confidence.complex_iplddt", True),
    # §11 recorded interface PAE as unrecoverable from `llm_ladder`, where
    # `interface_pae_for_pdb` contributed no keys to any of 15,204 rows. The ES pilot
    # runs DO carry it (`interface_pae_status` is "ok" on all 1,608 folds, median 8.46),
    # so it is reported wherever it exists and simply absent where it does not.
    ("interface PAE", "confidence.interface_pae", False),
    ("binder PAE", "confidence.binder_pae", False),
    ("buried unsat (refined)", "refined_delta_unsatHbonds", False),
    ("buried unsat (raw)", "raw_delta_unsatHbonds", False),
    ("interface H-bonds", "refined_hbonds_int", True),
    ("shape complementarity", "refined_sc_value", True),
    ("buried area A^2", "refined_dSASA_int", True),
    ("dG per 1000 A^2", "refined_dG_separated_per_1000A2", False),
)


def metric_of(row: dict, field: str) -> float | None:
    if field.startswith("confidence."):
        c = row.get("confidence") or {}
        v = c.get(field.split(".", 1)[1])
    else:
        v = row.get(field)
    return None if v is None else float(v)


def median_metric(seq: Sequence, field: str, lo: int, hi: int) -> float | None:
    """Median of one metric over the PASSING folds in [lo, hi).

    Passing folds only, matching how energy is taken: the interface of a structure the
    geometry gate rejected describes a pose that was thrown away, and averaging it in
    would let a rejected complex set a candidate's shape complementarity.
    """
    vals = [
        v
        for r in seq.folds_in(lo, hi)
        if r.get("valid") and (v := metric_of(r, field)) is not None
    ]
    return st.median(vals) if vals else None


def select_candidate(draws: dict[int, Sequence], half: int) -> Sequence | None:
    """The cross-fit pick: lowest median energy over the SELECTION half only.

    Deliberately the same rule as `cpseq_arm_compare.arm_pick`'s cross-fit branch, driven
    by whatever `energy_key` the Sequences were built with, so that changing the objective
    changes which candidate is chosen and nothing else about the comparison.
    """
    scored = [
        (e, i)
        for i, d in draws.items()
        if (e := d.median_energy_in(0, half)) is not None
    ]
    if not scored:
        return None
    return draws[min(scored)[1]]


def compare(state: dict[str, dict], depth: int) -> dict:
    half = depth // 2
    per_metric: dict[str, dict] = {
        label: {"field": f, "higher_is_better": hib, "pairs": []}
        for label, f, hib in METRICS
    }
    per_complex = []
    n_no_candidate = n_unmeasurable = 0

    for pid in sorted(state):
        ref = state[pid].get("reference")
        draws = state[pid].get("draws") or {}
        if ref is None or not draws:
            n_no_candidate += 1
            continue
        cand = select_candidate(draws, half)
        if cand is None:
            n_no_candidate += 1
            continue
        # The candidate must have SOMETHING on the measurement half, or there is no
        # measurement to report -- not a loss, its own column.
        if cand.n_pass_in(half, depth) == 0:
            n_unmeasurable += 1
            continue

        entry = {"parent_id": pid, "candidate": cand.draw_index}
        for label, field, _ in METRICS:
            c = median_metric(cand, field, half, depth)
            r = median_metric(ref, field, half, depth)
            entry[label] = {"cand": c, "ref": r}
            # A pair needs both sides measured on the held-out folds. Dropping a complex
            # where only one side has the metric keeps every reported delta paired.
            if c is not None and r is not None:
                per_metric[label]["pairs"].append((pid, c, r))
        per_complex.append(entry)

    out = {
        "fold_depth": depth,
        "selection_folds": [0, half],
        "measurement_folds": [half, depth],
        "n_complexes_reported": len(per_complex),
        "n_no_candidate": n_no_candidate,
        "n_unmeasurable": n_unmeasurable,
        "metrics": [],
        "per_complex": per_complex,
    }
    for label, field, hib in METRICS:
        pairs = per_metric[label]["pairs"]
        if not pairs:
            out["metrics"].append(
                {"metric": label, "field": field, "n": 0, "note": "no paired complex"}
            )
            continue
        cands = [c for _, c, _ in pairs]
        refs = [r for _, _, r in pairs]
        deltas = [c - r for _, c, r in pairs]
        # Wins counted in the metric's OWN direction, so the w/l column never has to be
        # read against a footnote about which way is better.
        wins = sum(1 for d in deltas if (d > 0) == hib and d != 0)
        losses = sum(1 for d in deltas if (d < 0) == hib and d != 0)
        out["metrics"].append(
            {
                "metric": label,
                "field": field,
                "higher_is_better": hib,
                "n": len(pairs),
                "median_cand": round(st.median(cands), 4),
                "median_ref": round(st.median(refs), 4),
                "median_delta": round(st.median(deltas), 4),
                "wins": wins,
                "losses": losses,
                "sign_p": round(sign_test(wins, losses), 4),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--arm", default="llm_search")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--energy-key", default=ENERGY_KEY)
    ap.add_argument("--json", default="")
    ap.add_argument("--scoring-version", default=None,
                    help="keep only rows from this scoring_version; see arm_compare.")
    args = ap.parse_args()

    per_parent = load(args.out_dir)
    if not per_parent:
        print(f"no journal rows under {args.out_dir}/state", file=sys.stderr)
        return 2
    mixed = warn_if_mixed(per_parent, args.scoring_version)
    state = build(per_parent, draw_arm=args.arm, energy_key=args.energy_key,
                  scoring_version=args.scoring_version)
    state = {k: v for k, v in state.items() if v.get("draws")}
    if not state:
        print(f"no folded draws for arm {args.arm!r}", file=sys.stderr)
        return 1

    depth = args.depth
    if depth <= 0:
        ranks = {
            k
            for v in state.values()
            for s in list(v["draws"].values()) + ([v["reference"]] if v["reference"] else [])
            for k in s.by_rank
        }
        depth = max(ranks) + 1 if ranks else 0
    if depth < 2:
        print(f"fold depth {depth} cannot hold out a half", file=sys.stderr)
        return 1

    rep = compare(state, depth)
    rep["out_dir"] = str(args.out_dir)
    rep["arm"] = args.arm
    rep["energy_key"] = args.energy_key
    rep["scoring_version"] = args.scoring_version
    rep["scoring_version_mixed"] = mixed

    print(
        f"{args.arm}, selected on dG field {args.energy_key!r} "
        f"over folds 0-{depth // 2 - 1}, measured on folds {depth // 2}-{depth - 1}"
    )
    print(
        f"{rep['n_complexes_reported']} complexes reported; "
        f"{rep['n_no_candidate']} with no selectable candidate, "
        f"{rep['n_unmeasurable']} selected but passing no measurement fold\n"
    )
    print(f"{'metric':<24} {'cand':>9} {'ref':>9} {'median d':>10} {'w/l':>8} {'p':>7}")
    for m in rep["metrics"]:
        if not m.get("n"):
            print(f"{m['metric']:<24} {'-':>9} {'-':>9} {'-':>10} {'-':>8} {'-':>7}")
            continue
        arrow = "" if m["higher_is_better"] else " v"
        print(
            f"{m['metric'] + arrow:<24} {m['median_cand']:>9.3f} {m['median_ref']:>9.3f} "
            f"{m['median_delta']:>+10.3f} "
            f"{str(m['wins']) + '/' + str(m['losses']):>8} {m['sign_p']:>7.3f}"
        )
    print("\n  'v' marks a metric where LOWER is better; w/l is already in that direction.")
    print(f"  n per metric: {', '.join(str(m.get('n', 0)) for m in rep['metrics'])}")

    if args.json:
        path = args.out_dir / args.json
        path.write_text(json.dumps(rep, indent=2))
        print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
