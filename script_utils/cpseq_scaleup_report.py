#!/usr/bin/env python3
"""Report a random-arm scale-up from whatever has finished. Safe to run mid-flight.

Designed to be run against a RUNNING search. It never assumes a parent is complete, and
it separates three different denominators that are easy to conflate when a run is partial:

  attempted   parents with at least one screening row
  complete    parents whose random arm reached the full budget AND have a reference row
  reportable  the `complete` set -- every rate below is computed on it, because a parent
              caught mid-budget has a truncated arm and would drag coverage down for a
              reason that is not about the method

A partial run is a smaller sample, not a biased one: shard assignment is
`cohort[i::n_shards]`, which is independent of anything about the target, so the parents
finished at any moment are an arbitrary subset rather than (say) the easy ones first.
That is what makes an interim number honest to report.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_scaleup_report.py --out-dir $ZFS/cpseq/pepbench_train_search
    python script_utils/cpseq_scaleup_report.py --out-dir ... --arm random_control --budget 8
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

REFERENCE_ARM = "reference"


def load(out_dir: Path) -> dict[str, list[dict]]:
    per: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted((out_dir / "state").glob("*/outcomes.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # A row still being appended to is not a corrupt run; skip it.
                continue
            per[r.get("parent_id", f.parent.name)].append(r)
    return dict(per)


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher p for a 2x2 table, exact, no scipy dependency."""
    def lc(n, k):
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    n = a + b + c + d
    row1, col1 = a + b, a + c
    p = 0.0
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    obs = lc(row1, a) + lc(n - row1, c) - lc(n, col1)
    for k in range(lo, hi + 1):
        term = lc(row1, k) + lc(n - row1, col1 - k) - lc(n, col1)
        if term <= obs + 1e-9:
            p += math.exp(term)
    return min(1.0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", default="random_control")
    ap.add_argument("--budget", type=int, default=8,
                    help="draws per parent the run was configured for")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    if not (out_dir / "state").exists():
        print(f"no state/ under {out_dir} -- nothing has been written yet")
        return 1

    per = load(out_dir)
    cohort_n = None
    cpath = out_dir / "cohort.json"
    if cpath.exists():
        cohort_n = len(json.loads(cpath.read_text()).get("cohort", []))

    design, ref, complete = {}, {}, []
    tech_fail = 0
    for pid, rows in per.items():
        d = [r for r in rows if r.get("arm") == a.arm and r.get("stage") == "screen"]
        rf = [r for r in rows if r.get("arm") == REFERENCE_ARM and r.get("stage") == "screen"]
        tech_fail += sum(1 for r in rows if r.get("technical_failure"))
        d = [r for r in d if not r.get("technical_failure")]
        rf = [r for r in rf if not r.get("technical_failure")]
        design[pid], ref[pid] = d, rf
        if len(d) >= a.budget and rf:
            complete.append(pid)

    print(f"=== scale-up status: {out_dir.name} ===")
    if cohort_n:
        print(f"cohort                 : {cohort_n} targets")
    print(f"attempted (any row)    : {len(per)}")
    print(f"complete (>={a.budget} draws + ref): {len(complete)}"
          + (f"  = {len(complete)/cohort_n:.3f} of cohort" if cohort_n else ""))
    print(f"technical failures     : {tech_fail}")
    if not complete:
        done = sorted((len(v) for v in design.values()), reverse=True)[:8]
        print(f"\nnothing complete yet; deepest parents have {done} draws")
        return 0

    dr = [r for pid in complete for r in design[pid][: a.budget]]
    rr = [r for pid in complete for r in ref[pid]]
    dv = sum(1 for r in dr if r.get("valid"))
    rv = sum(1 for r in rr if r.get("valid"))

    # Read the protocol version off the rows rather than naming one: a run that spans a
    # version bump, or an interim report on rows scored under the previous criteria, must
    # not be labelled with criteria that did not produce it.
    vers = sorted({r.get("scoring_version") for r in dr + rr if r.get("scoring_version")})
    vlabel = ", ".join(vers) if vers else "unrecorded"
    if len(vers) > 1:
        print(f"\nWARNING: rows span {len(vers)} scoring protocol versions ({vlabel}). "
              f"Validity labels are NOT comparable across them; rescore before pooling.")
    print(f"\n--- per-draw structural validity (scoring protocol: {vlabel}) ---")
    print(f"{a.arm:16} {dv:5d}/{len(dr):<5d} = {dv/len(dr):.3f}")
    print(f"{'reference':16} {rv:5d}/{len(rr):<5d} = {rv/len(rr):.3f}")
    if len(dr) and len(rr):
        p = fisher_exact_greater(dv, len(dr) - dv, rv, len(rr) - rv)
        print(f"Fisher, designs vs unmutated cyclic reference: p = {p:.3g}")

    cov = sum(1 for pid in complete
              if any(r.get("valid") for r in design[pid][: a.budget]))
    refcov = sum(1 for pid in complete if any(r.get("valid") for r in ref[pid]))
    print(f"\n--- per-target coverage at budget {a.budget} ---")
    print(f"targets with >=1 valid design   : {cov}/{len(complete)} = {cov/len(complete):.3f}")
    print(f"targets whose reference is valid: {refcov}/{len(complete)} = {refcov/len(complete):.3f}")

    de = [r["best_valid_dG_separated"] for r in dr
          if r.get("valid") and r.get("best_valid_dG_separated") is not None]
    re_ = [r["best_valid_dG_separated"] for r in rr
           if r.get("valid") and r.get("best_valid_dG_separated") is not None]
    print(f"\n--- interface energy among VALID rows (Rosetta ref2015 REU, not affinity) ---")
    for lab, e in ((a.arm, de), ("reference", re_)):
        if e:
            print(f"{lab:16} n={len(e):<5d} median {st.median(e):8.2f}  best {min(e):8.2f}")
    # Per-target best-vs-own-reference is the comparison that is not confounded by which
    # targets happen to be easy, so it is reported alongside the pooled numbers.
    beats = tot = 0
    for pid in complete:
        d_e = [r["best_valid_dG_separated"] for r in design[pid][: a.budget]
               if r.get("valid") and r.get("best_valid_dG_separated") is not None]
        r_e = [r["best_valid_dG_separated"] for r in ref[pid]
               if r.get("valid") and r.get("best_valid_dG_separated") is not None]
        if d_e and r_e:
            tot += 1
            beats += min(d_e) < min(r_e)
    if tot:
        print(f"\nper-target: best design beats its own reference on energy in "
              f"{beats}/{tot} = {beats/tot:.3f} of targets where both are valid")

    # The journal row flattens closure to `closure_*` scalars and drops which arrangement
    # the bond matched, so the cis/trans split is read from the score file instead. That
    # keeps this a pure reader: the alternative was adding the field to the driver's row
    # mid-run, which would have produced a corpus where only the later rows carry it.
    arr = collections.Counter()
    missing_scores = 0
    for r in dr:
        ih, ver = r.get("input_hash"), r.get("scoring_version")
        if not ih or not ver:
            continue
        sp = out_dir / "scores" / ver / f"{ih}.json"
        if not sp.exists():
            missing_scores += 1
            continue
        try:
            sj = json.loads(sp.read_text())
        except (json.JSONDecodeError, OSError):
            missing_scores += 1
            continue
        for smp in sj.get("samples", []) or []:
            v = (smp.get("closure") or {}).get("dihedral_arrangement")
            if v:
                arr[v] += 1
    if arr:
        tot_a = sum(arr.values())
        print("\n--- closing-amide arrangement (both are permitted; from score files) ---")
        if missing_scores:
            print(f"  ({missing_scores} rows had no readable score file)")
        for k, v in sorted(arr.items()):
            print(f"  {k:6} {v:5d}/{tot_a} = {v/tot_a:.3f}")

    preds = out_dir / "predictions"
    if preds.exists():
        n = sum(1 for _ in preds.glob("*/*"))
        print(f"\npredictions on disk: {n}")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps({
            "cohort_n": cohort_n, "attempted": len(per), "complete": len(complete),
            "technical_failures": tech_fail, "budget": a.budget, "arm": a.arm,
            "design_validity": [dv, len(dr)], "reference_validity": [rv, len(rr)],
            "coverage": [cov, len(complete)], "reference_coverage": [refcov, len(complete)],
            "design_median_dG": st.median(de) if de else None,
            "reference_median_dG": st.median(re_) if re_ else None,
            "arrangement": dict(arr), "scoring_versions": vers,
        }, indent=2))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
