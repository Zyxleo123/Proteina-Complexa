"""Report for pre-build check 2 -- the LNR closure feasibility frontier.

Reads the sharded JSONL the frontier job writes and emits the three deliverables:

  1. a per-target feasibility table across the (k, tolerance) sweep;
  2. the list of LNR targets infeasible under EVERY setting in the sweep;
  3. the validity-control comparison against the targets SDEdit never closed.

Two reporting rules this file enforces rather than leaves to the reader.

**A corrupt row is refused, never skipped.**  Concurrent appends to one file NUL-corrupt
rows on this filesystem, and a shard that silently drops them turns lost data into a
smaller-but-plausible number.  If any line fails to parse the job exits non-zero and names
the file.

**Every quoted median says which slice it came from.**  The Milestone 1.5 report had three
different legitimate medians in circulation at once (0.586 LNR, 0.584 PepBench, 0.594 on
the sampled-attempt subset), and a bracket bound whose slice is unstated cannot be
reproduced.  Tables here carry the n and the slice name in the same row.

Runs in `.venv`.  CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script_utils import precheck_config  # noqa: E402

CHEMISTRIES = ("mainchain", "disulfide", "isopeptide")


def _fmt(v, spec: str = "{:.3f}", missing: str = "-") -> str:
    """Format a cell, rendering a missing value as `-` rather than `nan`.

    A literal `nan` in a markdown table reads as a measured value that happened to be
    undefined; `-` reads as "not measured", which is what it is.
    """
    try:
        if v is None or pd.isna(v):
            return missing
        return spec.format(float(v))
    except (TypeError, ValueError):
        return missing


def load_rows(feas_dir: Path) -> pd.DataFrame:
    """Every shard, with corrupt lines REFUSED rather than skipped."""
    files = sorted(feas_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no shard files under {feas_dir}")
    rows, bad = [], []
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                bad.append(f"{f.name}:{n}: {type(exc).__name__}: {exc}")
    if bad:
        raise SystemExit(
            "REFUSING to aggregate: {} unparseable row(s). Concurrent appends NUL-corrupt "
            "rows on this filesystem, and skipping them would turn lost data into a "
            "smaller-but-plausible number.\n  {}".format(len(bad), "\n  ".join(bad[:10])))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"{feas_dir}: shards present but no rows")
    return df


def per_target_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (target, chemistry): the cheapest feasible k and what it retains.

    `k_min` is the headline. It is a residue COUNT, so unlike the retention ratio it
    carries no length contamination -- see README_M15_CEILING_AUDIT on why a trend read off
    the ratio against anything correlated with length is arithmetic.
    """
    out = []
    for (ex, chem), g in df.groupby(["example_id", "chemistry"], dropna=False):
        g = g.sort_values("k")
        if g.get("no_hostable_pair", pd.Series([0] * len(g))).fillna(0).astype(int).max() == 1:
            out.append({"example_id": ex, "chemistry": chem, "peptide_length": np.nan,
                        "no_hostable_pair": 1, "analytic_feasible_any": False,
                        "torsion_feasible_any": False, "k_min_analytic": np.nan,
                        "k_min_torsion": np.nan})
            continue
        an = g[g["analytic_feasible"].fillna(False).astype(bool)]
        to = g[g["torsion_feasible"].fillna(False).astype(bool)]
        rec = {
            "example_id": ex,
            "chemistry": chem,
            "peptide_length": int(g["peptide_length"].iloc[0]),
            "nc_gap_A": float(g["nc_gap_A"].iloc[0]),
            "n_contacts_ca10": int(g["n_contacts_ca10"].iloc[0]),
            "n_anchor_contacts": int(g["n_anchor_contacts"].iloc[0]),
            "anchor_fallback_to_all": int(g["anchor_fallback_to_all"].iloc[0]),
            "no_hostable_pair": 0,
            "analytic_feasible_any": bool(len(an)),
            "torsion_feasible_any": bool(len(to)),
            "k_min_analytic": int(an["k"].min()) if len(an) else np.nan,
            "k_min_torsion": int(to["k"].min()) if len(to) else np.nan,
            "refine_exhausted": bool(g["refine_exhausted"].fillna(False).astype(bool).any()),
        }
        # What the cheapest feasible solution actually preserves.  Both accountings are
        # reported: `strict` writes off every released residue and is a FLOOR, `perm` lets
        # a released residue keep a contact whose partner is still in reach and is the
        # UPPER bound.  The honest answer is the bracket, not either number alone.
        if len(to):
            r = to.sort_values("k").iloc[0]
            rec.update({
                "at_kmin_ret_all_strict": float(r["tor_all_ret_all_strict"]),
                "at_kmin_ret_all_perm": float(r["tor_all_ret_all_perm"]),
                "at_kmin_ret_anchor_strict": float(r["tor_anc_ret_anchor_strict"]),
                "at_kmin_ret_anchor_perm": float(r["tor_anc_ret_anchor_perm"]),
                "best_ret_anchor_strict": float(to["tor_anc_ret_anchor_strict"].max()),
            })
        out.append(rec)
    return pd.DataFrame(out)


def sweep_table(df: pd.DataFrame, tolerances: list[float], column: str) -> pd.DataFrame:
    """Fraction of targets feasible at each (k, tolerance) cell, per chemistry.

    Feasibility at `k` is cumulative -- "releasing AT MOST k residues" -- which is what the
    brief's sweep parameter means.  Torsion feasibility is monotone in k, but retention is
    NOT, so the cumulative maximum has to be taken over retention too rather than read off
    the row at exactly k.
    """
    recs = []
    for chem, gc in df.groupby("chemistry"):
        targets = gc["example_id"].nunique()
        for k in sorted(gc["k"].unique()):
            if k < 0:
                continue
            upto = gc[(gc["k"] <= k) & gc["torsion_feasible"].fillna(False).astype(bool)]
            best = upto.groupby("example_id")[column].max()
            for tol in tolerances:
                n_ok = int((best >= tol).sum())
                recs.append({"chemistry": chem, "k": int(k), "tolerance": tol,
                             "n_targets": targets, "n_feasible": n_ok,
                             "frac_feasible": n_ok / targets if targets else np.nan})
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--feas-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = precheck_config.load(args.config)
    acfg = cfg.get("lnr_feasibility") or {}
    tolerances = [float(t) for t in acfg.get("tolerance_grid", [0.0, 0.5, 0.7, 0.9])]
    controls = list(acfg.get("control_targets") or [])
    still_holding = set(acfg.get("control_still_holding") or [])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(Path(args.feas_dir))
    df.to_csv(out_dir / "feasibility_rows.csv", index=False)

    front = per_target_frontier(df)
    front.to_csv(out_dir / "frontier_by_target.csv", index=False)

    sweep_all = sweep_table(df, tolerances, "tor_all_ret_all_strict")
    sweep_anc = sweep_table(df, tolerances, "tor_anc_ret_anchor_strict")
    sweep_all.to_csv(out_dir / "sweep_all_contacts.csv", index=False)
    sweep_anc.to_csv(out_dir / "sweep_anchor_contacts.csv", index=False)

    # Infeasible under EVERY setting in the sweep: no k, no tolerance, any chemistry.
    never = (front.groupby("example_id")["torsion_feasible_any"].max() == False)  # noqa: E712
    never_ids = sorted(never[never].index.tolist())

    lines: list[str] = []
    A = lines.append
    A("# Pre-build check 2 -- LNR closure feasibility frontier\n")
    A(f"Complexes: {df['example_id'].nunique()}  |  "
      f"chemistries: {sorted(df['chemistry'].unique())}  |  "
      f"rows: {len(df)}\n")
    A("`k` is the number of residues RELEASED from their input conformation; the held")
    A("complement is a contiguous window kept at its native bound position. Retention is")
    A("CA-CA within 10 A, matched byte-for-byte to `sdedit_cyclize.py` and to the Milestone")
    A("1.5 ceiling, so these numbers can be divided into those.\n")
    A("Quote `k` for any claim about cost. The retention columns are RATIOS with peptide")
    A("length in the denominator, so a trend in them against length or gap is arithmetic")
    A("(README_M15_CEILING_AUDIT, section 2).\n")

    A("\n## 1. Cheapest feasible k, per chemistry\n")
    A("`k_min_torsion` is the confirmed answer; `k_min_analytic` is the permissive upper")
    A("bound from the CA-trace idealisation. The gap between them is how much the analytic")
    A("bound over-licenses.\n")
    A("| chemistry | n targets | feasible | median k_min (torsion) | median k_min (analytic) | median L |")
    A("|---|---|---|---|---|---|")
    for chem in CHEMISTRIES:
        g = front[front["chemistry"] == chem]
        if g.empty:
            continue
        ok = g[g["torsion_feasible_any"]]
        A(f"| {chem} | {len(g)} | {len(ok)} ({len(ok)/len(g):.0%}) | "
          f"{ok['k_min_torsion'].median() if len(ok) else float('nan'):.1f} | "
          f"{g['k_min_analytic'].median():.1f} | {g['peptide_length'].median():.0f} |")

    A("\n## 2. What the cheapest solution preserves\n")
    A("Both accountings, because either alone is misleading: `strict` writes off every")
    A("released residue and is therefore a FLOOR, `perm` keeps a released residue's contact")
    A("when its partner is still within reach and is the UPPER bound.\n")
    A("| chemistry | slice n | anchor strict | anchor perm | all strict | all perm |")
    A("|---|---|---|---|---|---|")
    for chem in CHEMISTRIES:
        g = front[(front["chemistry"] == chem) & front["torsion_feasible_any"]]
        if g.empty or "at_kmin_ret_anchor_strict" not in g:
            continue
        A(f"| {chem} | {len(g)} | "
          f"{g['at_kmin_ret_anchor_strict'].median():.3f} | "
          f"{g['at_kmin_ret_anchor_perm'].median():.3f} | "
          f"{g['at_kmin_ret_all_strict'].median():.3f} | "
          f"{g['at_kmin_ret_all_perm'].median():.3f} |")

    A("\n## 3. The (k, tolerance) sweep -- fraction of targets with a solution\n")
    A("Anchor retention, which is the gate the design actually needs: the anchors are the")
    A("interface the edit is defined to preserve, and global retention would instead force")
    A("the solution to be a near-copy of the input.\n")
    for chem in CHEMISTRIES:
        g = sweep_anc[sweep_anc["chemistry"] == chem]
        if g.empty:
            continue
        ks = sorted(g["k"].unique())[:8]
        A(f"\n**{chem}** (n = {int(g['n_targets'].iloc[0])} targets)\n")
        A("| tolerance | " + " | ".join(f"k<={k}" for k in ks) + " |")
        A("|---" * (len(ks) + 1) + "|")
        for tol in tolerances:
            cells = []
            for k in ks:
                m = g[(g["k"] == k) & (np.isclose(g["tolerance"], tol))]
                cells.append(f"{m['frac_feasible'].iloc[0]:.2f}" if len(m) else "-")
            A(f"| {tol:.1f} | " + " | ".join(cells) + " |")

    A("\n## 4. Targets infeasible under every setting in the sweep\n")
    if never_ids:
        A(f"{len(never_ids)} of {front['example_id'].nunique()}:\n")
        for t in never_ids:
            g = front[front["example_id"] == t]
            # `nc_gap_A` is absent entirely when every row took the no-hostable-pair branch,
            # so this reaches for the column rather than assuming the frame has it.
            gap = g["nc_gap_A"].dropna() if "nc_gap_A" in g else []
            A(f"- `{t}` (N-C gap {gap.iloc[0]:.1f} A)" if len(gap) else f"- `{t}`")
    else:
        A("None. Every target closes under some (k, chemistry) in the sweep.\n")
        A("This is not the same as saying every target is designable: the frontier is")
        A("permissive by construction -- no sterics, no receptor excluded volume, no")
        A("side-chain packing -- so it bounds what is geometrically possible, not what a")
        A("sampler will find.")

    A("\n## 5. Validity control\n")
    A("These never closed across 20 SDEdit grid points. If the solver calls them easily")
    A("feasible, either the solver or the SDEdit result needs explaining before any number")
    A("above is trusted.\n")
    A("One correction the brief does not carry: the paired soft-closure CONTROL later")
    A("closed 6 of 8 of that test bed UNPROJECTED, and `3cvl` was among those that closed.")
    A("So only `1jrr` and `4x3h` remain genuine never-closed hold-outs; `3cvl`'s failure")
    A("was a single-seed, wide-grid sampling artifact.\n")
    A("| target | status in SDEdit | chemistry | k_min (torsion) | L | N-C gap | anchor ret @ k_min |")
    A("|---|---|---|---|---|---|---|")
    for t in controls:
        g = front[front["example_id"] == t]
        if g.empty:
            A(f"| `{t}` | never closed | (absent from this run) | - | - | - | - |")
            continue
        status = ("never closed (still holds)" if t in still_holding
                  else "never closed (later closed in paired control)")
        for _, r in g.iterrows():
            A("| `{t}` | {status} | {chem} | {km} | {L} | {gap} | {ret} |".format(
                t=t, status=status, chem=r["chemistry"],
                km=_fmt(r.get("k_min_torsion"), "{:.0f}"),
                L=_fmt(r.get("peptide_length"), "{:.0f}"),
                gap=_fmt(r.get("nc_gap_A"), "{:.1f}"),
                ret=_fmt(r.get("at_kmin_ret_anchor_strict"), "{:.3f}")))

    A("\n### Reading the control\n")
    A("A LOW `k_min` on `1jrr` or `4x3h` means the geometry permits closure that the")
    A("sampler never found -- a sampler-side deficit, not a geometric one. A HIGH `k_min`")
    A("means the geometry itself is the obstacle and the SDEdit result is explained. The")
    A("two readings point at different remedies, so the number decides which.\n")

    if front["refine_exhausted"].fillna(False).any():
        n = int(front["refine_exhausted"].fillna(False).sum())
        A(f"\n> **{n} (target, chemistry) pairs exhausted the torsion budget.** Their")
        A("> `k_min_torsion` is an upper bound, not a measurement: a larger budget could")
        A("> only lower it. Do not quote them as infeasible.\n")

    (out_dir / "PRECHECK_TASK2_REPORT.md").write_text("\n".join(lines) + "\n")
    summary = {
        "n_targets": int(front["example_id"].nunique()),
        "n_rows": int(len(df)),
        "never_feasible": never_ids,
        "by_chemistry": {
            chem: {
                "n": int((front["chemistry"] == chem).sum()),
                "n_feasible": int(((front["chemistry"] == chem)
                                   & front["torsion_feasible_any"]).sum()),
                "median_k_min_torsion": float(
                    front[(front["chemistry"] == chem)
                          & front["torsion_feasible_any"]]["k_min_torsion"].median()),
            } for chem in CHEMISTRIES if (front["chemistry"] == chem).any()
        },
        "controls": {t: (front[front["example_id"] == t]
                         .set_index("chemistry")["k_min_torsion"].to_dict())
                     for t in controls},
    }
    (out_dir / "precheck_task2_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {out_dir}/PRECHECK_TASK2_REPORT.md")
    print(json.dumps(summary, indent=2, default=float)[:1500])


if __name__ == "__main__":
    main()
