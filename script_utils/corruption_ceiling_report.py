#!/usr/bin/env python
"""Turn the corruption-ceiling ladder into one number: how far can the termini be pulled
apart before the interface stops being worth training on?

Reads only files on disk (ladder JSONL + the Rosetta sidecar JSONL), so the figures and
tables re-render without re-running any minimization.

The comparison that matters, and why these constants:
  * 6.95 A  -- median PRE-cyclization terminal CA-CA of CPSea fragments (n=400 AFDB parents).
               The corpus starts here; it contains no larger move.
  * 16.3 A  -- LNR median terminal CA gap. The inference regime.
  * 21.6 A  -- PepBench median. 56.7% of PepBench sits in the 20-45 A band.

If the ceiling lands below ~16 A, synthetic corruption cannot manufacture the travel the
route needs, and there is no real-data fallback -- CPSea has no large-move example.

dG is used ONLY as a per-example delta against that example's own native rung, because
dG_separated is not comparable across targets.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CPSEA_PRE_GAP_A = 6.95
LNR_MEDIAN_A = 16.3
PEPBENCH_MEDIAN_A = 21.6
DG_COL = "binder_rosetta_dG_separated"


def read_jsonl(patterns, what):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Refusing is the point: a skipped corrupt row leaves an aggregate that
                    # still computes and is quietly missing data.
                    raise SystemExit(f"FATAL: corrupt {what} row {path}:{lineno}.")
    return pd.DataFrame(rows)


def is_reference(df):
    """The native and baseline rungs: references, never rungs of the ladder itself."""
    ref = df["is_native_rung"] == True  # noqa: E712
    if "is_baseline_rung" in df:
        ref = ref | (df["is_baseline_rung"] == True)  # noqa: E712
    return ref


def med(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else float("nan")


def build(ladder: pd.DataFrame, dg: pd.DataFrame) -> pd.DataFrame:
    ok = ladder[ladder.get("status").isna()] if "status" in ladder else ladder
    ok = ok[ok["ladder_complete"].notna()].copy() if "ladder_complete" in ok else ok.copy()
    if not dg.empty and DG_COL in dg:
        ok = ok.merge(dg[["pdb_tag", DG_COL]], on="pdb_tag", how="left")
        # The reference is the BASELINE rung -- minimized at the native separation, so it
        # carries the same force-field relaxation as the pulled rungs and the delta isolates
        # the opening. Falling back to the unminimized native rung bills the amber-vs-Rosetta
        # relaxation offset as damage: on the first smoke that offset was a flat +6 to +11 REU
        # for isopeptide, present already 0.25 A from native.
        ref_kind = "baseline"
        if "is_baseline_rung" in ok and bool((ok["is_baseline_rung"] == True).any()):  # noqa: E712
            ref_rows = ok[ok["is_baseline_rung"] == True]  # noqa: E712
        else:
            ref_kind = "native (UNMINIMIZED -- pre-baseline run)"
            ref_rows = ok[ok["is_native_rung"] == True]  # noqa: E712
        ok.attrs["dg_reference"] = ref_kind
        ok["dG_ref"] = ok["example_id"].map(
            ref_rows.set_index("example_id")[DG_COL].to_dict())
        ok["delta_dG"] = pd.to_numeric(ok[DG_COL], errors="coerce") - pd.to_numeric(
            ok["dG_ref"], errors="coerce")
    else:
        ok[DG_COL] = np.nan
        ok["delta_dG"] = np.nan
        ok.attrs["dg_reference"] = "none (no Rosetta rows)"
    return ok


def per_rung(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for gap, g in df[~is_reference(df)].groupby("target_gap_A"):
        out.append({
            "target_gap_A": float(gap), "n": len(g),
            "achieved_med_A": med(g["terminal_ca_A"]),
            "shortfall_med_A": med(g["shortfall_A"]),
            "retention_med": med(g["contact_retention"]),
            "min_heavy_med_A": med(g["min_pep_rec_heavy_A"]),
            "interior_rmsd_med_A": med(g["interior_ca_rmsd_A"]),
            "delta_dG_med": med(g["delta_dG"]),
            "n_dG": int(pd.to_numeric(g["delta_dG"], errors="coerce").notna().sum()),
            "frac_reached": float((g.get("limit_reason") == "reached").mean())
                            if "limit_reason" in g else float("nan"),
            "frac_stalled": float((g.get("limit_reason") == "stalled").mean())
                            if "limit_reason" in g else float("nan"),
            "frac_exhausted": float((g.get("limit_reason") == "rounds_exhausted").mean())
                              if "limit_reason" in g else float("nan"),
        })
    return pd.DataFrame(out).sort_values("target_gap_A").reset_index(drop=True)


def verdict(tbl: pd.DataFrame, a, attributed: bool = True) -> dict:
    """The largest terminal separation ACHIEVED with the interface still intact.

    Not "the largest rung whose setpoint was met". The ladder is cumulative, so each rung
    starts where the last one ended and the restraint routinely settles short of its own
    setpoint while the separation keeps climbing overall. Grading on setpoints answered the
    wrong question and got it badly wrong: the re-smoke ladder reached 21.8 A at retention
    0.89, and a setpoint-based ceiling called it 10 A and declared the inference regime out
    of reach. Shortfall is a diagnostic about ladder spacing and pull strength, not a
    pass/fail criterion.

    `attributed=False` means the rows carry no `limit_reason`, so nothing can be said about
    why a pull stopped and no ceiling is claimed. The first smoke's report degraded exactly
    that way -- "ceiling 8 A, does NOT reach the inference regime", off a ladder whose
    disulfide ring was never opened -- because every guard read a column that wasn't there.
    """
    reasons, ceiling, broken, instrument_limited = {}, None, False, []
    limited_by = "no rung passed"
    for _, r in tbl.iterrows():
        fails, gap = [], float(r["target_gap_A"])
        # Interface health only. Whether the restraint hit its setpoint is not in here.
        if not math.isnan(r["retention_med"]) and r["retention_med"] < a.retention_min:
            fails.append(f"contacts lost (retention {r['retention_med']:.2f})")
        if not math.isnan(r["min_heavy_med_A"]) and r["min_heavy_med_A"] < a.min_heavy_A:
            fails.append(f"clash (min heavy {r['min_heavy_med_A']:.2f} A)")
        if r["n_dG"] > 0 and not math.isnan(r["delta_dG_med"]) and r["delta_dG_med"] > a.dg_tol:
            fails.append(f"dG worse by {r['delta_dG_med']:.1f} REU")
        exh = r.get("frac_exhausted", float("nan"))
        if not attributed:
            instrument_limited.append(gap)
        elif not math.isnan(exh) and exh >= 0.5:
            # Still moving when the round budget ran out: the achieved separation is a lower
            # bound on this rung, not the pocket's limit.
            instrument_limited.append(gap)
        reasons[gap] = fails
        if fails:
            broken = True
        elif not broken:
            # Contiguous from the bottom: an interface that recovers above a failure is noise,
            # not a reach to rely on.
            ceiling = float(r["achieved_med_A"])

    if ceiling is not None:
        top = tbl.iloc[-1]
        if broken:
            limited_by = "interface"
        elif not attributed or (not math.isnan(top.get("frac_exhausted", float("nan")))
                                and top["frac_exhausted"] >= 0.5):
            limited_by = "instrument (achieved is a lower bound; raise --pull-rounds)"
        elif not math.isnan(top.get("frac_stalled", float("nan"))) and top["frac_stalled"] >= 0.5:
            limited_by = "pocket (the pull stopped winning)"
        else:
            limited_by = "ladder top (every rung reached; extend --ladder-A to find the limit)"

    trustworthy = bool(attributed and not instrument_limited)
    return {"ceiling_achieved_A": ceiling,
            "limited_by": limited_by,
            "instrument_limited_rungs_A": instrument_limited,
            "ceiling_is_trustworthy": trustworthy,
            "shortfalls_attributed": bool(attributed),
            "max_achieved_any_rung_A": (float(tbl["achieved_med_A"].max())
                                        if len(tbl) else None),
            "per_rung_failures": reasons,
            "reaches_lnr_median": bool(ceiling is not None and ceiling >= LNR_MEDIAN_A),
            "reaches_pepbench_median": bool(ceiling is not None and ceiling >= PEPBENCH_MEDIAN_A),
            "thresholds": {"retention_min": a.retention_min, "min_heavy_A": a.min_heavy_A,
                           "dg_tol_REU": a.dg_tol, "shortfall_tol_A": a.shortfall_tol_A}}


def figures(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    moved = df[~is_reference(df)]

    def marks(ax):
        for x, lab, c in ((CPSEA_PRE_GAP_A, "CPSea pre-cyc 6.95", "#888888"),
                          (LNR_MEDIAN_A, "LNR 16.3", "#c2410c"),
                          (PEPBENCH_MEDIAN_A, "PepBench 21.6", "#7c3aed")):
            ax.axvline(x, ls="--", lw=1, color=c)
            ax.annotate(lab, (x, ax.get_ylim()[1]), rotation=90, fontsize=7,
                        va="top", ha="right", color=c)

    f, ax = plt.subplots(figsize=(5.2, 4))
    lim = max(tbl["target_gap_A"].max() if len(tbl) else 24, 24) + 2
    ax.plot([0, lim], [0, lim], color="#cbd5e1", lw=1, label="asked = achieved")
    ax.plot(tbl["target_gap_A"], tbl["achieved_med_A"], "o-", color="#0f766e", label="achieved (median)")
    ax.set_xlabel("target terminal CA-CA (A)"); ax.set_ylabel("achieved (A)")
    ax.set_title("Does the pocket let the termini separate?"); ax.legend(fontsize=8)
    marks(ax); f.tight_layout(); f.savefig(fig_dir / "achieved_vs_target.png", dpi=150); plt.close(f)

    f, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    axes[0].scatter(moved["terminal_ca_A"], moved["contact_retention"], s=8, alpha=.45, color="#0f766e")
    axes[0].plot(tbl["achieved_med_A"], tbl["retention_med"], "o-", color="#0b3b36")
    axes[0].set_xlabel("achieved terminal CA-CA (A)"); axes[0].set_ylabel("contact retention")
    axes[1].scatter(moved["terminal_ca_A"], moved["delta_dG"], s=8, alpha=.45, color="#b45309")
    axes[1].plot(tbl["achieved_med_A"], tbl["delta_dG_med"], "o-", color="#7c2d12")
    axes[1].axhline(0, color="#94a3b8", lw=1)
    axes[1].set_xlabel("achieved terminal CA-CA (A)")
    axes[1].set_ylabel("delta interface dG vs native rung (REU)")
    for ax in axes:
        marks(ax)
    f.suptitle("What opening costs the interface")
    f.tight_layout(); f.savefig(fig_dir / "interface_vs_gap.png", dpi=150); plt.close(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder-glob", nargs="+", required=True)
    ap.add_argument("--rosetta-glob", nargs="+", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shortfall-tol-A", type=float, default=1.5)
    ap.add_argument("--retention-min", type=float, default=0.70)
    ap.add_argument("--min-heavy-A", type=float, default=2.0)
    ap.add_argument("--dg-tol", type=float, default=5.0, help="REU worse than native, allowed.")
    a = ap.parse_args()

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ladder = read_jsonl(a.ladder_glob, "ladder")
    if ladder.empty:
        print("GATE: no ladder rows on disk. The ladder stage produced nothing; exiting 0.")
        return 0
    dg = read_jsonl(a.rosetta_glob, "rosetta") if a.rosetta_glob else pd.DataFrame()

    # An all-failed ladder is a real failure, not a gate. Falling through to per_rung() here
    # raised `KeyError: is_native_rung` off a 60-example run where every example had died of
    # the same NameError -- a traceback that names a column instead of naming the problem.
    if "status" in ladder:
        failed = ladder[ladder["status"] == "failed"]
        usable = ladder[ladder["status"].isna()]
        if len(failed) and usable.empty:
            print(f"\nFATAL: every example failed ({failed['example_id'].nunique()} of them). "
                  f"No usable rows, so there is nothing to report.")
            print("Distinct errors:")
            for err, g in failed.groupby(failed.get("error", pd.Series(dtype=str))):
                print(f"  {g['example_id'].nunique():4d}x  {err}")
            return 1
        if len(failed):
            print(f"\n!! {failed['example_id'].nunique()} examples failed and are excluded; "
                  f"{usable['example_id'].nunique()} contribute rows.")

    df = build(ladder, dg)
    tbl = per_rung(df)
    # Can a shortfall be attributed at all in this run? Without limit_reason it cannot, and a
    # ceiling must not be claimed from it.
    attributed = bool("limit_reason" in df and df["limit_reason"].notna().any())
    ring_checked = "ring_opened" in df
    v = verdict(tbl, a, attributed=attributed)
    v["ring_opening_verified"] = ring_checked

    print(f"\ncomplexes: {df['example_id'].nunique()}   rows: {len(df)}   "
          f"dG rows: {0 if dg.empty else len(dg)}")
    print(f"dG reference: {df.attrs.get('dg_reference', '?')}")
    if str(df.attrs.get("dg_reference", "")).startswith("native"):
        print("  !! no baseline rung in this run, so every delta_dG carries the constant")
        print("     amber-vs-Rosetta relaxation offset on top of the cost of opening.")
    if "status" in ladder:
        nf = int(ladder["status"].eq("failed").sum())
        if nf:
            print(f"failed examples: {nf} (kept in the JSONL, excluded from the tables)")
    print("\n== per rung ==")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    if not attributed:
        print("\n!! these rows carry no `limit_reason`, so a shortfall cannot be attributed to")
        print("   the pocket rather than to the minimizer. NO ceiling is claimed below.")
    if not ring_checked:
        print("!! these rows carry no `ring_opened`, so ring opening was never verified. A ring")
        print("   that stayed shut reads as a pocket that resists at ~8 A.")

    print("\n== by chemistry: largest ACHIEVED separation with the interface intact ==")
    for cyc, g in df.groupby("cyc_type"):
        t = per_rung(g)
        cv = verdict(t, a, attributed=attributed)["ceiling_achieved_A"]
        print(f"  {cyc:<11} n={g['example_id'].nunique():<4} ceiling "
              f"{'none' if cv is None else f'{cv:.1f} A'}")

    # A still-closed ring turns the whole arm into a measurement of a macrocycle
    # stretching, which looks exactly like a pocket that resists. Say so before any verdict.
    if "ring_opened" in df:
        closed = df[df["ring_opened"] == False]  # noqa: E712
        if len(closed):
            print(f"\n!! {closed['example_id'].nunique()} complexes had a ring that could NOT "
                  f"be opened -- their ladders measure a CLOSED macrocycle stretching, not an "
                  f"opening, and their apparent ceiling is an artifact:")
            for cyc, g in closed.groupby("cyc_type"):
                print(f"     {cyc}: {g['example_id'].nunique()} complexes")

    print("\n== verdict ==")
    c = v["ceiling_achieved_A"]
    print(f"  ceiling (largest ACHIEVED separation, interface intact): "
          f"{'NO rung passed' if c is None else f'{c:.1f} A'}")
    print(f"  limited by: {v['limited_by']}")
    print(f"  max achieved on any rung: {v['max_achieved_any_rung_A']:.1f} A"
          if v["max_achieved_any_rung_A"] is not None else "")
    print(f"  reaches LNR median (16.3 A)      : {v['reaches_lnr_median']}")
    print(f"  reaches PepBench median (21.6 A) : {v['reaches_pepbench_median']}")
    if v["instrument_limited_rungs_A"]:
        print(f"  !! rungs {v['instrument_limited_rungs_A']} were INSTRUMENT-LIMITED, so this "
              f"ceiling is a lower bound on the instrument, not on the pocket.")
        print(f"     Re-run with a larger --pull-rounds before drawing any conclusion.")
    if not v["reaches_lnr_median"] and v["ceiling_is_trustworthy"]:
        print("  => corruption does NOT reach the inference regime on CPSea complexes.")
        print("     CPSea has no large-move example to fall back on, so the transport would")
        print("     be trained entirely below the gap range it meets at inference.")
    for gap, fails in sorted(v["per_rung_failures"].items()):
        if fails:
            print(f"    {gap:5g} A: {'; '.join(fails)}")

    tbl.to_csv(out_dir / "per_rung.csv", index=False)
    (out_dir / "verdict.json").write_text(json.dumps(v, indent=2))
    df.to_csv(out_dir / "rows.csv", index=False)
    figures(df, tbl, out_dir / "figures")
    print(f"\nwrote {out_dir}/per_rung.csv, verdict.json, rows.csv, figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
