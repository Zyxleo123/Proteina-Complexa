"""Phase 0: reproducible CPSea baseline on the validated instrument.

Success definition
------------------
Deliberately three separate numbers, never one blended score:

1. **Raw generation** -- did the requested macrocycle form? Measured over everything
   sampled, before any filtering. This is the model's unassisted output.
2. **Binding quality** -- Rosetta interface dG, gated against the *native* dG for that
   same target (see `calibrate_rosetta_dg.py`). Using the native as the bar is what
   makes this an external criterion rather than a restatement of the design
   distribution.
3. **Joint** -- both. Reported apart from its parts so a metric used to select
   candidates can never resurface as the headline.

AF2 confidence (pLDDT / i_pAE / scRMSD) is carried as **descriptive columns only**.
It was disproven as a gate for this molecule class: after calibrating its thresholds
against natives it passed 94% of designs, versus 0% before -- a constant either way.

Why dG and not AF2: dG passed the same native check, with a per-target hit rate
spanning 0%-73.5% and natives above the median design on 4/5 targets.

Known gaps in the recorded data (these need fresh generation runs, not more analysis):
  * ``nsteps`` is 400 for every recorded design, so the 50/100/200/400 scaling
    comparison cannot be made here.
  * No seed column, so runs are not individually reproducible from these records.
  * No chirality or clash metric exists in the codebase yet.
  * ``cyc_type_requested`` is null in every row -- type conditioning was not active at
    design time -- so the per-linkage breakdown is by OBSERVED chemistry. That answers
    "what did the model build", not "did it build what it was told", and the two are
    not interchangeable.
"""

import argparse
import glob
import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from proteinfoundation.result_analysis.cyclic_success import (  # noqa: E402
    CYC_DIST_COL,
    CYC_TYPE_OBS_COL,
    cyclization_validity_mask,
)

DESIGN_GLOB = "evaluation_results/search_binder_local_pipeline_cpsea_*/binder_results_*.csv"
DG_COL = "generated_binder_rosetta_dG_separated"
TOTAL_SCORE_COL = "generated_binder_rosetta_total_score_relaxed"
SEQ_COL = "self_sequence"
TARGET_COL = "task_name"

DESCRIPTIVE_COLS = {
    "pLDDT": "self_complex_pLDDT",
    "i_pAE": "self_complex_i_pAE",
    "scRMSD_ca": "self_binder_scRMSD_ca",
}

LENGTH_BINS = [(0, 11), (12, 13), (14, 15), (16, 999)]


def load_designs(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(pattern)):
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError) as e:
            logger.warning("Skipping unreadable %s: %s", path, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_native_dg(report_path: str) -> dict[str, float]:
    """Per-target native dG, the externally-anchored quality bar."""
    with open(report_path) as handle:
        report = json.load(handle)
    return {t: e["native_dG"] for t, e in report.get("relaxed", {}).items() if "native_dG" in e}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def peptide_length(df: pd.DataFrame) -> pd.Series:
    if SEQ_COL not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[SEQ_COL].astype(str).str.strip().str.len().replace(0, np.nan)


def pass_at_k(n: int, n_success: int, k: int) -> float:
    """Empirical pass@k from n repeated samples containing n_success successes.

    Uses the exact hypergeometric complement 1 - C(n-c, k)/C(n, k) -- the probability
    that a uniformly drawn size-k subset of the samples we actually have contains at
    least one success. This is NOT 1-(1-p)^k: that form assumes k independent draws at
    a fixed per-sample rate, which overstates pass@k whenever successes are clustered
    within a target (they are).
    """
    if n <= 0 or k <= 0 or n_success <= 0:
        return 0.0 if n > 0 else float("nan")
    if k > n:
        return float("nan")  # undefined: cannot draw k from n
    if n - n_success < k:
        return 1.0
    return 1.0 - (math.comb(n - n_success, k) / math.comb(n, k))


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI over TARGETS (the unit of independence), not designs.

    Designs within a target share a receptor and are not independent draws, so
    resampling designs would give a spuriously tight interval.
    """
    arr = np.asarray([v for v in values if not (v is None or (isinstance(v, float) and math.isnan(v)))])
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def summarize(df: pd.DataFrame, native_dg: dict[str, float], k_values: list[int]) -> dict:
    chem = cyclization_validity_mask(df)
    dg = _num(df, DG_COL)
    lengths = peptide_length(df)

    # Quality gate: at least as tight as the native complex for the SAME target.
    native_bar = df[TARGET_COL].map(native_dg) if TARGET_COL in df.columns else pd.Series(np.nan, index=df.index)
    quality = dg <= native_bar
    quality_scored = dg.notna() & native_bar.notna()
    joint = chem & quality.fillna(False)

    report: dict = {
        "n_designs": int(len(df)),
        "n_targets": int(df[TARGET_COL].nunique()) if TARGET_COL in df.columns else 0,
        "micro": {
            "raw_ring_closed_rate": float(chem.mean()),
            "quality_rate_vs_native_dG": float(quality[quality_scored].mean()) if quality_scored.any() else float("nan"),
            "joint_success_rate": float(joint.mean()),
            "n_quality_scored": int(quality_scored.sum()),
        },
        "descriptive_only_not_a_gate": {},
    }

    for label, col in DESCRIPTIVE_COLS.items():
        series = _num(df, col).dropna()
        if not series.empty:
            report["descriptive_only_not_a_gate"][label] = {
                "n": int(series.size),
                "median": round(float(series.median()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            }

    bond = _num(df, CYC_DIST_COL).dropna()
    if not bond.empty:
        report["bond_distance_A"] = {
            "n": int(bond.size),
            "median": round(float(bond.median()), 3),
            "iqr": [round(float(bond.quantile(0.25)), 3), round(float(bond.quantile(0.75)), 3)],
        }

    # ---- per target ----
    per_target = {}
    for target, sub in df.groupby(df[TARGET_COL].astype(str)):
        idx = sub.index
        sub_chem = chem.loc[idx]
        sub_joint = joint.loc[idx]
        sub_dg = dg.loc[idx].dropna()
        n = len(sub)
        n_joint = int(sub_joint.sum())
        entry = {
            "n_designs": n,
            "raw_ring_closed_rate": float(sub_chem.mean()),
            "joint_success_rate": float(sub_joint.mean()),
            "any_success": bool(sub_joint.any()),
            "native_dG": native_dg.get(target),
            "design_dG_median": round(float(sub_dg.median()), 3) if not sub_dg.empty else None,
            "design_dG_best": round(float(sub_dg.min()), 3) if not sub_dg.empty else None,
            "pass_at_k": {str(k): round(pass_at_k(n, n_joint, k), 4) for k in k_values},
        }
        per_target[target] = entry
    report["per_target"] = per_target

    # ---- macro (mean over targets; a target with more designs must not dominate) ----
    if per_target:
        closed = [v["raw_ring_closed_rate"] for v in per_target.values()]
        joint_rates = [v["joint_success_rate"] for v in per_target.values()]
        report["macro"] = {
            "raw_ring_closed_rate": float(np.mean(closed)),
            "raw_ring_closed_rate_ci95": bootstrap_ci(closed),
            "joint_success_rate": float(np.mean(joint_rates)),
            "joint_success_rate_ci95": bootstrap_ci(joint_rates),
            "target_coverage": float(np.mean([v["any_success"] for v in per_target.values()])),
        }
        report["macro"]["pass_at_k"] = {
            str(k): round(float(np.mean([v["pass_at_k"][str(k)] for v in per_target.values()])), 4)
            for k in k_values
        }

    # ---- by observed linkage chemistry ----
    #
    # WARNING, and the reason each row carries a flag: the three categories are not
    # assigned on equal terms. `_infer_chemistry` names disulfide/isopeptide from residue
    # identity (CYS-CYS, LYS-ASP/GLU) whether or not the ring closed, but mainchain has no
    # residue signature and is recognised ONLY by finding the backbone N and C already
    # bonded -- using the very same acceptance window that defines closure. So
    # `observed == "mainchain"` is logically equivalent to `closed`, and its closure rate
    # is 100% by construction, not by measurement.
    #
    # The bias runs the other way too: a mainchain attempt that failed to close has no
    # signature at all, so it lands in `unclassified` rather than in `mainchain`. That
    # bucket is reported here instead of being dropped, because silently discarding it
    # would hide most of the mainchain failures.
    #
    # A sound per-linkage comparison needs the REQUESTED type, which is null in every
    # recorded row. This table cannot substitute for it.
    if CYC_TYPE_OBS_COL in df.columns:
        by_type = {}
        obs = df[CYC_TYPE_OBS_COL].astype(str)
        for linkage, sub in df.groupby(obs):
            idx = sub.index
            key = "unclassified" if linkage in ("nan", "None", "") else linkage
            entry = {
                "n": len(sub),
                "raw_ring_closed_rate": float(chem.loc[idx].mean()),
                "joint_success_rate": float(joint.loc[idx].mean()),
            }
            if key == "mainchain":
                entry["closure_rate_is_circular"] = True
                entry["note"] = "observed==mainchain is defined BY closure; 100% is tautological"
            if key == "unclassified":
                entry["note"] = "no chemistry signature; holds the failed mainchain attempts"
            by_type[key] = entry
        report["by_observed_linkage"] = by_type
        report["by_observed_linkage_is_confounded"] = True

    # ---- by peptide length bin ----
    by_len = {}
    for lo, hi in LENGTH_BINS:
        sel = (lengths >= lo) & (lengths <= hi)
        if not bool(sel.any()):
            continue
        idx = df.index[sel]
        by_len[f"{lo}-{hi if hi < 999 else '+'}"] = {
            "n": int(sel.sum()),
            "raw_ring_closed_rate": float(chem.loc[idx].mean()),
            "joint_success_rate": float(joint.loc[idx].mean()),
        }
    report["by_length_bin"] = by_len

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-glob", default=DESIGN_GLOB)
    parser.add_argument("--native-report", default="calibration_native/rosetta_dg_report.json")
    parser.add_argument("--k", default="1,5,10,20", help="Comma-separated k values for pass@k")
    parser.add_argument("--out", default="evaluation_results/phase0_baseline.json")
    parser.add_argument("--per-design-csv", default="evaluation_results/phase0_per_design.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_designs(args.design_glob)
    if df.empty:
        logger.error("No design results matched %r", args.design_glob)
        return 1
    native_dg = load_native_dg(args.native_report)
    if not native_dg:
        logger.error("No native dG bars in %s; run calibrate_rosetta_dg.py first.", args.native_report)
        return 1

    k_values = [int(k) for k in args.k.split(",")]
    report = summarize(df, native_dg, k_values)

    m, macro = report["micro"], report.get("macro", {})
    print("\n================ PHASE 0 BASELINE ================")
    print(f"designs={report['n_designs']}  targets={report['n_targets']}")
    print("\n-- raw generation (unfiltered) --")
    print(f"  ring closed        micro {m['raw_ring_closed_rate'] * 100:5.1f}%   macro {macro.get('raw_ring_closed_rate', float('nan')) * 100:5.1f}%")
    print("\n-- binding quality (Rosetta dG vs native, external bar) --")
    print(f"  >= native dG       micro {m['quality_rate_vs_native_dG'] * 100:5.1f}%   (n scored {m['n_quality_scored']})")
    print("\n-- joint (both, kept separate from its parts) --")
    lo, hi = macro.get("joint_success_rate_ci95", (float("nan"),) * 2)
    print(f"  joint success      micro {m['joint_success_rate'] * 100:5.1f}%   macro {macro.get('joint_success_rate', float('nan')) * 100:5.1f}%  95% CI [{lo * 100:.1f}, {hi * 100:.1f}]")
    print(f"  target coverage    {macro.get('target_coverage', float('nan')) * 100:5.1f}%")
    print("\n-- pass@k (empirical, macro over targets) --")
    for k, v in macro.get("pass_at_k", {}).items():
        print(f"  pass@{k:<3s} {v * 100:5.1f}%")
    print("\n-- by observed linkage chemistry  [CONFOUNDED - see notes] --")
    for linkage, v in report.get("by_observed_linkage", {}).items():
        flag = "  <-- CIRCULAR" if v.get("closure_rate_is_circular") else ""
        print(f"  {linkage:<13s} n={v['n']:<5d} closed {v['raw_ring_closed_rate'] * 100:5.1f}%   joint {v['joint_success_rate'] * 100:5.1f}%{flag}")
    print("     mainchain closure is defined BY closure -> 100% is tautological, not measured.")
    print("     failed mainchain attempts have no signature and fall into 'unclassified'.")
    print("     A valid per-linkage split needs the REQUESTED type (null in all recorded rows).")
    print("\n-- by peptide length  [confounded with target: each target has a characteristic length] --")
    for b, v in report.get("by_length_bin", {}).items():
        print(f"  {b:<8s} n={v['n']:<5d} closed {v['raw_ring_closed_rate'] * 100:5.1f}%   joint {v['joint_success_rate'] * 100:5.1f}%")
    print("\n-- per target --")
    for t, v in report["per_target"].items():
        print(f"  {t:<12s} n={v['n_designs']:<4d} closed {v['raw_ring_closed_rate'] * 100:5.1f}%  joint {v['joint_success_rate'] * 100:5.1f}%  dG best {v['design_dG_best']} vs native {v['native_dG']}")
    print("\n-- AF2 confidence: DESCRIPTIVE ONLY, not a gate --")
    for label, v in report["descriptive_only_not_a_gate"].items():
        print(f"  {label:<10s} median {v['median']:.3f}  [{v['min']:.3f}, {v['max']:.3f}]")
    print("==================================================\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote %s", args.out)

    if args.per_design_csv:
        keep = [c for c in [TARGET_COL, SEQ_COL, DG_COL, TOTAL_SCORE_COL, CYC_DIST_COL, CYC_TYPE_OBS_COL, *DESCRIPTIVE_COLS.values()] if c in df.columns]
        out = df[keep].copy()
        out["ring_closed"] = cyclization_validity_mask(df).values
        out["peptide_length"] = peptide_length(df).values
        out.to_csv(args.per_design_csv, index=False)
        logger.info("Wrote %s", args.per_design_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
