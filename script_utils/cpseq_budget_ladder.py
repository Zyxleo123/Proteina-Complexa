#!/usr/bin/env python3
"""Budget ladder for the random-control arm, computed from screening rows already on disk.

The scale-up question is how many random draws per target to pay for. This answers it
from the realised PepBench screen without predicting anything new: each parent holds `n`
evaluated random draws, `v` of them gate-valid, so a budget-B arm is a without-replacement
draw of B from those n and its best-of-B statistics are hypergeometric order statistics.

Two quantities are reported separately because they answer different questions, and the
project's own seed-noise finding means only the first is trustworthy at the margin:

  coverage   P(a target gets >=1 gate-valid design within B draws). Concave in B, so
             there is a knee and it is the statistic the ladder is for.
  best dG    E[min refined dG_separated among the valid draws of B] | at least one valid.
             Monotone in B BY CONSTRUCTION -- it is a minimum over more draws -- and
             README_CPSEQ_SEARCH_PILOT sec. 7 measured a per-sequence spread across seeds
             of median 10.96 REU, which is larger than the whole ladder's span. It is
             printed because it is measurable, not because it justifies a larger budget.
  yield      E[# valid designs] = B * (v/n). LINEAR in B, so it has no knee: a corpus-
             building run maximises it by adding targets, not budget (coverage is concave
             in B while target count is linear in cost).

Prefix truncation of the journal is deliberately NOT used. Round 0 of the realised
schedule is the guided arm's single-substitution scan, which arm A count-matches, so
1102/1102 of its random draws are 1-mutant; a prefix below B=16 would therefore confound
budget with mutation count. Subsampling the realised draws avoids that.

Usage:
    python script_utils/cpseq_budget_ladder.py --out-dir $ZFS/cpseq/pepbench_search
    python script_utils/cpseq_budget_ladder.py --out-dir $ZFS/cpseq/pepbench_search \
        --arm guided_search          # for contrast only; see the caveat below
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics as st
from math import comb

# Measured marginal cost of one evaluation (README_CPSEQ_SEARCH_PILOT sec. 11), and the
# measured mean on-disk size of one prediction directory, raw and after pruning. These
# are measurements, not estimates; `cpseq_budget_ladder.py --audit-storage` re-derives
# the sizes from the tree.
GPU_S_PER_EVAL = 39.3
CPU_S_PER_EVAL = 46.3
MB_PER_PRED_RAW = 1.31
MB_PER_PRED_PRUNED = 0.188

BUDGETS = (1, 2, 4, 8, 12, 16, 24, 32)


def load_outcomes(out_dir: pathlib.Path, arm: str) -> dict[str, list[dict]]:
    """Screening rows per parent for one arm, technical failures excluded.

    A technical failure is not a negative result -- it is an evaluation that did not
    happen -- so counting it in the denominator would understate the arm's yield.
    """
    per: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted((out_dir / "state").glob("*/outcomes.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("arm") != arm or r.get("stage") != "screen":
                continue
            if r.get("technical_failure"):
                continue
            per[r["parent_id"]].append(r)
    return dict(per)


def coverage_at_b(n: int, v: int, b: int) -> float | None:
    """P(>=1 of the v valid rows appears in a without-replacement draw of b from n)."""
    if b > n:
        return None
    if v == 0:
        return 0.0
    if n - v < b:
        return 1.0
    return 1.0 - comb(n - v, b) / comb(n, b)


def expected_best_at_b(energies: list[float], n: int, b: int) -> float | None:
    """E[best (most negative) energy among the valid rows of b drawn] | >=1 valid.

    Rank all n rows with the invalid ones ranked worst. For valid energies sorted
    ascending e_1<=...<=e_v, the i-th is the sample's best iff it is drawn and none of
    the i-1 better rows are, so P = C(n-i, b-1)/C(n, b). Renormalising by the total of
    those probabilities conditions on the sample containing any valid row at all, which
    is why this is not an average over empty sets.
    """
    if b > n or not energies:
        return None
    num = 0.0
    p_any = 0.0
    for i, e_i in enumerate(sorted(energies), start=1):
        if n - i < b - 1:
            break
        p = comb(n - i, b - 1) / comb(n, b)
        num += p * e_i
        p_any += p
    return num / p_any if p_any > 0 else None


def validity_by_mutation_count(per: dict[str, list[dict]]) -> dict[int, dict]:
    """Gate pass rate against realised substitution count.

    This decides whether a standalone random arm needs to inherit the guided arm's
    realised count multiset (see `search.match_counts`, which currently requires it) or
    may freeze a simple distribution of its own.
    """
    agg: dict[int, dict] = collections.defaultdict(
        lambda: {"draws": 0, "valid": 0, "energies": []})
    for rows in per.values():
        for r in rows:
            b = agg[r["n_mutations"]]
            b["draws"] += 1
            if r.get("valid"):
                b["valid"] += 1
                if r.get("best_valid_dG_separated") is not None:
                    b["energies"].append(r["best_valid_dG_separated"])
    return dict(sorted(agg.items()))


def build_ladder(per_full: dict[str, list[dict]], budgets=BUDGETS) -> list[dict]:
    out = []
    n_tgt = len(per_full)
    for b in budgets:
        covs, bests, yields_ = [], [], []
        for rows in per_full.values():
            n = len(rows)
            valid = [r for r in rows if r.get("valid")]
            energies = [r["best_valid_dG_separated"] for r in valid
                        if r.get("best_valid_dG_separated") is not None]
            c = coverage_at_b(n, len(valid), b)
            if c is not None:
                covs.append(c)
                yields_.append(b * len(valid) / n)
            e = expected_best_at_b(energies, n, b)
            if e is not None:
                bests.append(e)
        evals = b * n_tgt
        out.append(dict(
            B=b, n_targets=n_tgt,
            coverage=st.mean(covs),
            targets_covered=st.mean(covs) * n_tgt,
            exp_best_dG=st.mean(bests) if bests else None,
            med_best_dG=st.median(bests) if bests else None,
            exp_valid_designs=sum(yields_),
            evals=evals,
            gpu_h=evals * GPU_S_PER_EVAL / 3600,
            cpu_h=evals * CPU_S_PER_EVAL / 3600,
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="a search output dir, e.g. pepbench_search")
    ap.add_argument("--arm", default="random_control")
    ap.add_argument("--min-n", type=int, default=32,
                    help="only parents with at least this many draws enter the ladder, so "
                         "the curve is not a mixture over different n")
    ap.add_argument("--json-out", default="",
                    help="defaults to <out-dir>/tables/budget_ladder_<arm>.json")
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out_dir)
    per = load_outcomes(out_dir, a.arm)
    if not per:
        print(f"no {a.arm} screening rows under {out_dir}/state")
        return 1

    full = {k: v for k, v in per.items() if len(v) >= a.min_n}
    short = {k: v for k, v in per.items() if len(v) < a.min_n}
    print(f"arm = {a.arm}")
    if a.arm != "random_control":
        print("  CAVEAT: subsampling a FEEDBACK-DRIVEN arm does not model a smaller-budget\n"
              "  run of it. Rounds 1+ read an archive built from every earlier round, so a\n"
              "  real budget-B guided run would have proposed different sequences. Only the\n"
              "  random arm's draws are exchangeable, which is what makes its ladder exact.")
    print(f"parents with rows      : {len(per)}")
    print(f"  in the ladder (n>={a.min_n}) : {len(full)}")
    if short:
        print(f"  partial, excluded    : {len(short)} (n = {sorted(len(v) for v in short.values())})")
    print()

    print("=== gate validity by realised substitution count (all parents) ===")
    print(f"{'n_mut':>6} {'draws':>7} {'valid':>7} {'rate':>7} {'med dG | valid':>15}")
    bym = validity_by_mutation_count(per)
    for k, b in bym.items():
        med = f"{st.median(b['energies']):.2f}" if b["energies"] else "--"
        print(f"{k:>6} {b['draws']:>7} {b['valid']:>7} {b['valid']/b['draws']:>7.3f} {med:>15}")
    print()

    ladder = build_ladder(full)
    print("=== budget ladder (exact hypergeometric over the realised draws) ===")
    print(f"{'B':>4} {'coverage':>9} {'targets':>8} {'E[best dG]':>11} {'E[#valid]':>10} "
          f"{'evals':>7} {'GPU-h':>7} {'CPU-h':>7}")
    for r in ladder:
        print(f"{r['B']:>4} {r['coverage']:>9.3f} {r['targets_covered']:>8.1f} "
              f"{r['exp_best_dG']:>11.2f} {r['exp_valid_designs']:>10.1f} "
              f"{r['evals']:>7} {r['gpu_h']:>7.1f} {r['cpu_h']:>7.1f}")
    print()

    print("=== marginal return, per 1000 further evaluations ===")
    print(f"{'B':>4} {'d coverage':>11} {'d dG':>8} {'cover/1k evals':>15}")
    prev = None
    for r in ladder:
        if prev is not None:
            dc = r["coverage"] - prev["coverage"]
            de = r["evals"] - prev["evals"]
            print(f"{r['B']:>4} {dc:>+11.3f} {r['exp_best_dG']-prev['exp_best_dG']:>+8.2f} "
                  f"{dc/de*1000:>15.3f}")
        prev = r
    print()

    reachable = sum(1 for rows in full.values() if any(r.get("valid") for r in rows))
    per_parent = [sum(1 for r in rows if r.get("valid")) for rows in full.values()]
    print(f"ceiling at n={a.min_n}: {reachable}/{len(full)} = {reachable/len(full):.3f} of "
          f"targets have >=1 valid design")
    print(f"valid draws per parent: med {st.median(per_parent):.0f}, min {min(per_parent)}, "
          f"max {max(per_parent)}, zero-yield parents {sum(1 for x in per_parent if x == 0)}")

    dest = pathlib.Path(a.json_out) if a.json_out else (
        out_dir / "tables" / f"budget_ladder_{a.arm}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(
        arm=a.arm, n_parents_in_ladder=len(full), min_n=a.min_n,
        ceiling_coverage=reachable / len(full),
        cost_constants=dict(gpu_s_per_eval=GPU_S_PER_EVAL, cpu_s_per_eval=CPU_S_PER_EVAL,
                            mb_per_pred_raw=MB_PER_PRED_RAW,
                            mb_per_pred_pruned=MB_PER_PRED_PRUNED),
        by_n_mutations={str(k): {"draws": v["draws"], "valid": v["valid"],
                                 "rate": v["valid"] / v["draws"]} for k, v in bym.items()},
        ladder=ladder), indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
