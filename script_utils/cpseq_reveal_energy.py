#!/usr/bin/env python3
"""Interface energy of partial-reveal CP designs, against the unmutated CP reference.

Closure is necessary and not sufficient: a ring that does not bind is worthless. This
compares the corrected interface energy of designs made at several reveal fractions
against the SAME parents' unmutated cyclic reference, which is what `keep_frac = 1.0`
generates (median 0 substitutions from the LP sequence).

Two things it refuses to do, because both would manufacture an improvement:

**Energies are read only from samples that pass the frozen geometry gate.** An open or
clashing ring has no interface to score, however attractive its number. That makes the
valid subsets unequal across reveal fractions -- closure falls as more is designed -- so
the validity rate is printed beside every energy and the yield is never hidden behind a
median.

**The comparison is paired per parent.** Pooling across parents compares energies computed
on different receptors with different interface sizes, and the pooled difference can carry
the opposite sign to every per-parent one. So the headline is a per-parent delta against
that parent's own reference, and parents where either side has no valid sample are reported
as incomparable rather than dropped.

Energies are Rosetta ref2015 REU, not measured binding free energies.

Usage:
    python script_utils/cpseq_reveal_energy.py \
        --scored kf1.00=tables/scored_kf100.jsonl --scored kf0.75=tables/scored_kf075.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001, S112
            continue
    return rows


def refined_dg(r: dict) -> float | None:
    return ((r.get("rosetta") or {}).get("refined") or {}).get("dG_separated")


def refined_dsasa(r: dict) -> float | None:
    return ((r.get("rosetta") or {}).get("refined") or {}).get("dSASA_int")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--reference", default="kf1.00",
                    help="label of the unmutated-CP reference arm")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arms: dict[str, list[dict]] = {}
    for spec in args.scored:
        label, path = spec.split("=", 1)
        arms[label] = load(Path(path))

    print(f"{'arm':10s} {'n':>4s} {'valid':>6s} {'yield':>7s} {'n_mut med':>10s} "
          f"{'dG med':>8s} {'dG best':>8s} {'dSASA med':>10s} {'recovery':>9s}")
    print("-" * 86)
    per_arm = {}
    for label, rows in arms.items():
        scored = [r for r in rows if r.get("sample_index", -1) >= 0]
        if not scored:
            print(f"{label:10s} {'0':>4s}  no rows yet")
            continue
        valid = [r for r in scored if r.get("valid")]
        dgs = [refined_dg(r) for r in valid if refined_dg(r) is not None]
        sas = [refined_dsasa(r) for r in valid if refined_dsasa(r) is not None]
        muts = [r.get("n_mutations") for r in scored if r.get("n_mutations") is not None]
        recs = [r.get("cond_recovery") for r in scored if r.get("cond_recovery") is not None]
        per_arm[label] = {
            "n": len(scored), "n_valid": len(valid),
            "valid_yield": round(len(valid) / len(scored), 4) if scored else None,
            "median_n_mutations": st.median(muts) if muts else None,
            "median_valid_dG": round(st.median(dgs), 3) if dgs else None,
            "best_valid_dG": round(min(dgs), 3) if dgs else None,
            "median_valid_dSASA": round(st.median(sas), 1) if sas else None,
            "median_recovery": round(st.median(recs), 3) if recs else None,
        }
        a = per_arm[label]
        print(f"{label:10s} {a['n']:4d} {a['n_valid']:6d} {str(a['valid_yield']):>7s} "
              f"{str(a['median_n_mutations']):>10s} {str(a['median_valid_dG']):>8s} "
              f"{str(a['best_valid_dG']):>8s} {str(a['median_valid_dSASA']):>10s} "
              f"{str(a['median_recovery']):>9s}")

    # --- paired, per parent -----------------------------------------------------------
    ref_label = args.reference
    if ref_label not in arms:
        print(f"\n[energy] reference arm {ref_label!r} absent; no paired comparison")
        return 0

    def best_by_parent(rows: list[dict]) -> dict[str, float]:
        out: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if not r.get("valid"):
                continue
            d = refined_dg(r)
            if d is not None:
                out[r.get("parent_id")].append(d)
        return {k: min(v) for k, v in out.items()}

    ref_best = best_by_parent(arms[ref_label])
    pairs = []
    print(f"\n=== paired per parent: best valid dG vs {ref_label} (reference) ===")
    hdr = f"{'parent':12s} {'ref dG':>8s}"
    for label in arms:
        if label != ref_label:
            hdr += f" {label+' dG':>13s} {'delta':>8s}"
    print(hdr)
    parents = sorted({r.get("parent_id") for rows in arms.values() for r in rows
                      if r.get("parent_id")})
    for pid in parents:
        line = f"{pid:12s} {str(ref_best.get(pid, 'none')):>8s}"
        rec = {"parent_id": pid, "reference_best_dG": ref_best.get(pid)}
        for label, rows in arms.items():
            if label == ref_label:
                continue
            b = best_by_parent(rows).get(pid)
            delta = (round(b - ref_best[pid], 3)
                     if (b is not None and pid in ref_best) else None)
            line += f" {str(round(b,3) if b is not None else 'none'):>13s} {str(delta):>8s}"
            rec[f"{label}_best_dG"] = b
            rec[f"{label}_delta_vs_reference"] = delta
        print(line)
        pairs.append(rec)

    print("\nNegative delta = the design beats its parent's unmutated CP reference.")
    print("Rosetta ref2015 REU; valid samples only; see valid_yield above for the "
          "denominator each median rests on.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"per_arm": per_arm, "per_parent": pairs,
                                        "reference_arm": ref_label,
                                        "energy_units": "Rosetta ref2015 REU"}, indent=2))
        print(f"\n[energy] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
