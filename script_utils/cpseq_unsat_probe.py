#!/usr/bin/env python3
"""Does the geometric buried-unsat detector track Rosetta's number, and is the defect
something a substitution could even repair?

Two questions, and the second one decides whether the first matters.

`README_CPSEQ_LLM_SEARCH.md` reports `delta_unsatHbonds` from Rosetta's
`InterfaceAnalyzerMover`: one integer per fold, no atom attribution. A proposer cannot
act on an integer, so `cpseq/unsat.py` reimplements the concept geometrically to get the
attribution. Whether that reimplementation measures the same thing is an open question,
and the honest order is to answer it before anything reaches a prompt:

1. **Agreement.** Across the threshold grid in `unsat.py`, which setting correlates with
   `raw_delta_unsatHbonds`, and how well? A geometric count that tracks nothing is not a
   substitute for the Rosetta term, however interpretable its atom list.

2. **Ceiling.** Of the unsatisfied atoms, how many are peptide SIDE-CHAIN atoms? Those
   are the only ones a sequence substitution can repair. A buried backbone amide and a
   receptor-side atom are real defects that no proposal can fix, and if they are most of
   the count then showing the model this information has a small ceiling no matter how
   well the detector agrees with Rosetta. This is the number that says whether to build
   the feature at all.

Both are read off the SAME folds the reported comparison used, raw structures off disk.
Nothing is re-predicted and Rosetta is not re-run: `raw_delta_unsatHbonds` is taken from
the journal rows, so the reference number is exactly the one the report quoted.

Sampling is stratified by parent and spans both the candidate and reference arms, because
the question is whether the detector tracks Rosetta ACROSS the range the search explores;
drawing only candidates would measure agreement on one tail.

Usage:
    python script_utils/cpseq_unsat_probe.py \
        --run-dir /data/user_data/yixiz/cpseq/llm_ladder \
        --per-parent 24 --jobs 2 \
        --out /data/user_data/yixiz/cpseq/llm_ladder/unsat_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.unsat import THRESHOLD_GRID, unsat_report  # noqa: E402


# --------------------------------------------------------------------------------------
# Correlation, without a scipy dependency in the report path
# --------------------------------------------------------------------------------------


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks, so ties do not manufacture ordering. Buried-unsat counts are small
    integers and ties are the common case, not an edge case."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None  # a constant column has no correlation, and 0.0 would imply it does
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


def load_rows(run_dir: Path) -> dict[str, list[dict]]:
    """Journal outcome rows by parent, keeping only what this probe can use."""
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((run_dir / "state").glob("*/outcomes.jsonl")):
        parent = path.parent.name
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            # A row without the Rosetta number has no reference to compare against, and a
            # row without a raw path has no structure to measure.
            if r.get("raw_delta_unsatHbonds") is None or not r.get("raw_pdb_path"):
                continue
            by_parent[parent].append(r)
    return by_parent


def sample(
    by_parent: dict[str, list[dict]], per_parent: int, seed: int
) -> list[tuple[str, dict]]:
    """Up to `per_parent` rows per parent, balanced across arms within each parent.

    Balanced rather than uniform because the arms are wildly unequal in count -- a
    uniform draw would be almost entirely candidates and would measure agreement on the
    candidate distribution alone.
    """
    rng = random.Random(seed)
    picked: list[tuple[str, dict]] = []
    for parent in sorted(by_parent):
        rows = by_parent[parent]
        by_arm: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_arm[str(r.get("arm"))].append(r)
        arms = sorted(by_arm)
        if not arms:
            continue
        quota = max(1, per_parent // len(arms))
        taken: list[dict] = []
        for arm in arms:
            pool = by_arm[arm]
            rng.shuffle(pool)
            taken.extend(pool[:quota])
        # Any shortfall from a thin arm is backfilled from the rest, so a parent with one
        # reference row still contributes `per_parent` folds.
        if len(taken) < per_parent:
            seen = {id(r) for r in taken}
            rest = [r for r in rows if id(r) not in seen]
            rng.shuffle(rest)
            taken.extend(rest[: per_parent - len(taken)])
        for r in taken[:per_parent]:
            picked.append((parent, r))
    return picked


# --------------------------------------------------------------------------------------
# One fold
# --------------------------------------------------------------------------------------


def measure(task: tuple[str, dict]) -> dict:
    parent, row = task
    rec = {
        "parent_id": parent,
        "sequence": row.get("sequence"),
        "arm": row.get("arm"),
        "stage": row.get("stage"),
        "input_hash": row.get("input_hash"),
        "prediction_seed": row.get("prediction_seed"),
        "valid": row.get("valid"),
        "rosetta_raw_delta_unsat": row.get("raw_delta_unsatHbonds"),
        "rosetta_raw_hbonds_int": row.get("raw_hbonds_int"),
        "rosetta_raw_dSASA_int": row.get("raw_dSASA_int"),
        "rosetta_raw_dG_separated": row.get("raw_dG_separated"),
        "ok": False,
        "error": None,
    }
    try:
        rep = unsat_report(
            row["raw_pdb_path"],
            peptide_sequence=row.get("sequence"),
            structure_stage="raw",
        )
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    if not rep.get("ok"):
        rec["error"] = rep.get("error")
        return rec
    rec["ok"] = True
    rec["sequence_matches"] = rep.get("sequence_matches")
    rec["peptide_chain"] = rep.get("peptide_chain")
    # One flat row per grid point, so the report can group without reparsing structures.
    rec["grid"] = [
        {
            "buried_max": g["buried_sasa_max_A2"],
            "desolv_min": g["desolvated_min_A2"],
            "newly_unsat_total": g["total"]["newly_unsat"],
            "newly_unsat_peptide": g["peptide"]["newly_unsat"],
            "newly_unsat_peptide_sc": g["peptide_sidechain"]["newly_unsat"],
            "newly_unsat_peptide_bb": g["peptide_backbone"]["newly_unsat"],
            "newly_unsat_receptor": g["receptor"]["newly_unsat"],
            "unsat_complex_total": g["total"]["unsat_complex"],
            "buried_complex_total": g["total"]["buried_complex"],
        }
        for g in rep["grid"]
    ]
    return rec


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def report(records: list[dict]) -> dict:
    usable = [r for r in records if r.get("ok")]
    out: dict = {
        "n_records": len(records),
        "n_usable": len(usable),
        "n_failed": len(records) - len(usable),
        "failures": {},
        "grid": [],
    }
    for r in records:
        if not r.get("ok"):
            key = str(r.get("error"))[:120]
            out["failures"][key] = out["failures"].get(key, 0) + 1
    if len(usable) < 3:
        return out

    ros = [float(r["rosetta_raw_delta_unsat"]) for r in usable]

    # A third question the stored rows can answer for free, and it bounds the other two:
    # does buried-unsat relate to the quantity the search actually selects on? The ladder
    # ranks candidates by `best_valid_dG_separated` (`cpseq_search.py:656`); unsat is
    # recorded and ignored. If a fold's unsat count carries no relation to its dG, then a
    # proposer that repaired a defect would be outranked by one that did not, and the
    # information would have nowhere to go even where the defect is real and fixable.
    with_dg = [
        r for r in usable if r.get("rosetta_raw_dG_separated") is not None
    ]
    if len(with_dg) >= 3:
        dg = [float(r["rosetta_raw_dG_separated"]) for r in with_dg]
        ros_dg = [float(r["rosetta_raw_delta_unsat"]) for r in with_dg]
        sa = [float(r.get("rosetta_raw_dSASA_int") or 0.0) for r in with_dg]
        out["objective_relation"] = {
            "n": len(with_dg),
            "spearman_rosetta_unsat_vs_dG": (
                None if (s := _spearman(ros_dg, dg)) is None else round(s, 3)
            ),
            "spearman_rosetta_unsat_vs_dSASA": (
                None if (s := _spearman(ros_dg, sa)) is None else round(s, 3)
            ),
            "spearman_dSASA_vs_dG": (
                None if (s := _spearman(sa, dg)) is None else round(s, 3)
            ),
            "note": (
                "dG is negative-is-better, so a POSITIVE unsat-vs-dG correlation means "
                "more unsatisfied atoms go with worse energy, which is the expected sign."
            ),
        }

    for gi, (bmax, dmin) in enumerate(THRESHOLD_GRID):
        tot = [float(r["grid"][gi]["newly_unsat_total"]) for r in usable]
        sc = [float(r["grid"][gi]["newly_unsat_peptide_sc"]) for r in usable]
        bb = [float(r["grid"][gi]["newly_unsat_peptide_bb"]) for r in usable]
        rec_side = [float(r["grid"][gi]["newly_unsat_receptor"]) for r in usable]
        n = len(usable)
        sum_tot = sum(tot)
        out["grid"].append(
            {
                "buried_max_A2": bmax,
                "desolv_min_A2": dmin,
                "mean_geom_total": round(sum_tot / n, 2),
                "mean_rosetta": round(sum(ros) / n, 2),
                "spearman_vs_rosetta": (
                    None if (s := _spearman(tot, ros)) is None else round(s, 3)
                ),
                "pearson_vs_rosetta": (
                    None if (p := _pearson(tot, ros)) is None else round(p, 3)
                ),
                # The ceiling. Of everything the detector flags, what share sits on a
                # peptide side chain, which is the only part a proposal can change.
                "mean_peptide_sidechain": round(sum(sc) / n, 2),
                "mean_peptide_backbone": round(sum(bb) / n, 2),
                "mean_receptor": round(sum(rec_side) / n, 2),
                "fixable_share": (
                    None if sum_tot <= 0 else round(sum(sc) / sum_tot, 3)
                ),
                "folds_with_a_fixable_defect": sum(1 for v in sc if v > 0),
                "folds_with_any_defect": sum(1 for v in tot if v > 0),
            }
        )
    # Ranked by rank-correlation, which is the relevant one: the detector would be used
    # to tell better from worse, not to reproduce Rosetta's absolute integer.
    out["best_by_spearman"] = sorted(
        (g for g in out["grid"] if g["spearman_vs_rosetta"] is not None),
        key=lambda g: -g["spearman_vs_rosetta"],
    )[:3]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--per-parent", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--limit", type=int, default=0, help="stop after this many folds (0 = no limit)"
    )
    ap.add_argument(
        "--from-jsonl",
        action="store_true",
        help="re-report from an existing --out file without measuring anything. The "
        "structures cost 1.5-7 s each and the rows hold everything the report reads, so "
        "a changed report must never be a reason to recompute them.",
    )
    args = ap.parse_args()

    if args.from_jsonl:
        if not args.out.exists():
            print(f"no rows at {args.out}", file=sys.stderr)
            return 2
        records = []
        for line in args.out.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except Exception:  # noqa: BLE001, S112
                continue
        return _emit(report(records), args.out)

    by_parent = load_rows(args.run_dir)
    if not by_parent:
        print(f"no usable journal rows under {args.run_dir}/state", file=sys.stderr)
        return 2
    tasks = sample(by_parent, args.per_parent, args.seed)
    if args.limit:
        tasks = tasks[: args.limit]
    print(
        f"{sum(len(v) for v in by_parent.values())} journal rows carry both a Rosetta "
        f"count and a raw path, across {len(by_parent)} parents"
    )
    print(f"measuring {len(tasks)} folds on {args.jobs} workers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    # Written as they land, so a preempted run keeps what it paid for and the report can
    # be regenerated from the JSONL without recomputing a single structure.
    with open(args.out, "w") as fh, ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, rec in enumerate(ex.map(measure, tasks, chunksize=1), start=1):
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(tasks):
                ok = sum(1 for r in records if r.get("ok"))
                print(f"  {i}/{len(tasks)}  ok={ok}", flush=True)

    return _emit(report(records), args.out)


def _emit(rep: dict, out_path: Path) -> int:
    rep_path = out_path.with_suffix(".report.json")
    rep_path.write_text(json.dumps(rep, indent=2))

    print(f"\nusable {rep['n_usable']}/{rep['n_records']}")
    if rep["failures"]:
        print("failures:")
        for k, v in sorted(rep["failures"].items(), key=lambda kv: -kv[1]):
            print(f"  {v:>5}  {k}")
    print(
        f"\n{'bur<=':>6} {'dsasa>=':>8} {'geom':>6} {'rosetta':>8} "
        f"{'spearman':>9} {'pearson':>8} {'pep_sc':>7} {'pep_bb':>7} {'recep':>7} "
        f"{'fixable':>8} {'n_fix>0':>8}"
    )
    for g in rep["grid"]:
        print(
            f"{g['buried_max_A2']:>6.1f} {g['desolv_min_A2']:>8.1f} "
            f"{g['mean_geom_total']:>6.2f} {g['mean_rosetta']:>8.2f} "
            f"{str(g['spearman_vs_rosetta']):>9} {str(g['pearson_vs_rosetta']):>8} "
            f"{g['mean_peptide_sidechain']:>7.2f} {g['mean_peptide_backbone']:>7.2f} "
            f"{g['mean_receptor']:>7.2f} {str(g['fixable_share']):>8} "
            f"{g['folds_with_a_fixable_defect']:>8}"
        )
    rel = rep.get("objective_relation")
    if rel:
        print(
            f"\nagainst the objective (n={rel['n']}): "
            f"rosetta unsat vs dG rho={rel['spearman_rosetta_unsat_vs_dG']}, "
            f"vs dSASA rho={rel['spearman_rosetta_unsat_vs_dSASA']}, "
            f"dSASA vs dG rho={rel['spearman_dSASA_vs_dG']}"
        )
        print(f"  {rel['note']}")
    print(f"\nreport: {rep_path}")
    print(f"rows:   {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
