"""Does the hope stand? Table + figure for the DDIB inversion probe (scripts/ddib_invert.py).

The question is NOT "do the termini get closer" -- collapsing a peptide to a point does that
too. It is whether the reverse ODE buys closure feasibility more cheaply than isotropic
noise does. So every number is reported against its paired control at the same t:

    _xt      state handed onward (what a cyclic generator would start from)
    _x1pred  the source model's clean prediction at that t
    _sdedit  interpolate(gaussian, input, t) -- the SDEdit arm, same t, same peptide

and the headline is a paired comparison, not a marginal: at each t, how many peptides have
a SMALLER terminal gap under inversion than under SDEdit, and what does each arm's gap cost
in pose (`ca_rmsd_to_input_A`) and interface (`contact_retention`).

Reads only the saved JSONL, so the figure is re-renderable without a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# A closed head-to-tail ring has CA(1)-CA(L) at ~3.8 A, exactly like a bonded pair. 6 A is
# the generous "a generator could pull this shut" band; 3.8 +- 0.5 would be already-closed.
REACHABLE_CA_A = 6.0
ARMS = [("_xt", "inverted state"), ("_x1pred", "model clean pred"), ("_sdedit", "SDEdit control")]


def load(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("invert_*.jsonl"))
    if not files:
        raise SystemExit(f"FATAL: no invert_*.jsonl under {results_dir}")
    rows = []
    for f in files:
        for lineno, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Refuse, never skip: a silently dropped row is a silently biased summary,
                # and NUL-corrupted rows from concurrent appends have happened here before.
                raise SystemExit(f"FATAL: corrupt row {f}:{lineno}. Delete the file and rerun.")
    df = pd.DataFrame(rows)
    print(f"{len(df)} rows from {len(files)} file(s)")
    if "status" in df:
        bad = df[df["status"] != "ok"]
        if len(bad):
            print(f"WARNING: {len(bad)} failed rows; first error: {bad.iloc[0].get('error')}")
        df = df[df["status"] == "ok"].copy()
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (tracks, t), g in df.groupby(["invert_tracks", "t"]):
        row = {"invert_tracks": tracks, "t": t, "n": len(g)}
        for suffix, _ in ARMS:
            for base in ("ca_end_gap_A", "ca_rmsd_to_input_A", "rg_A", "contact_retention"):
                col = f"{base}{suffix}"
                if col in g:
                    row[f"med_{col}"] = float(g[col].median())
            col = f"ca_end_gap_A{suffix}"
            if col in g:
                row[f"frac_reachable{suffix}"] = float((g[col] < REACHABLE_CA_A).mean())
        if "nc_gap_A_x1pred" in g:
            row["med_nc_gap_A_x1pred"] = float(g["nc_gap_A_x1pred"].median())
            row["med_seq_identity_x1pred"] = float(g["seq_identity_x1pred"].median())
        # The paired test: same peptide, same t, inversion vs its own SDEdit control.
        if {"ca_end_gap_A_xt", "ca_end_gap_A_sdedit"} <= set(g.columns):
            row["frac_inversion_tighter"] = float((g["ca_end_gap_A_xt"] < g["ca_end_gap_A_sdedit"]).mean())
        if "roundtrip_ca_rmsd_A" in g:
            rt = g["roundtrip_ca_rmsd_A"].dropna()
            if len(rt):
                row["med_roundtrip_ca_rmsd_A"] = float(rt.median())
                row["n_roundtrip"] = int(len(rt))
        out.append(row)
    return pd.DataFrame(out).sort_values(["invert_tracks", "t"], ascending=[True, False])


def figure(df: pd.DataFrame, summary: pd.DataFrame, out_png: Path) -> None:
    tracks = sorted(summary["invert_tracks"].unique())
    fig, axes = plt.subplots(2, len(tracks), figsize=(6 * len(tracks), 8), squeeze=False)
    for c, tr in enumerate(tracks):
        s = summary[summary["invert_tracks"] == tr].sort_values("t", ascending=False)
        ax = axes[0][c]
        for suffix, label in ARMS:
            col = f"med_ca_end_gap_A{suffix}"
            if col in s:
                ax.plot(s["t"], s[col], marker="o", label=label)
        med_in = df[df["invert_tracks"] == tr]["input_ca_end_gap_A"].median()
        ax.axhline(med_in, ls=":", c="k", label="input (median)")
        ax.axhline(REACHABLE_CA_A, ls="--", c="g", label=f"reachable ({REACHABLE_CA_A} A)")
        ax.invert_xaxis()  # the reverse pass runs right-to-left: t = 1 -> 0
        ax.set_xlabel("t along the reverse ODE (1 = input)")
        ax.set_ylabel("median CA(first)-CA(last) gap [A]")
        ax.set_title(f"closure feasibility -- invert {tr}")
        ax.legend(fontsize=8)

        # The cost axis. A gap bought by destroying the pose is not a gap worth having, so
        # the two are plotted against each other rather than each against t.
        ax = axes[1][c]
        for suffix, label in ARMS:
            gcol, rcol = f"med_ca_end_gap_A{suffix}", f"med_ca_rmsd_to_input_A{suffix}"
            if gcol in s and rcol in s:
                ax.plot(s[rcol], s[gcol], marker="o", label=label)
                for _, r in s.iterrows():
                    ax.annotate(f"{r['t']:.2f}", (r[rcol], r[gcol]), fontsize=6)
        ax.axhline(REACHABLE_CA_A, ls="--", c="g")
        ax.set_xlabel("median CA RMSD to input [A]  (pose spent)")
        ax.set_ylabel("median CA end gap [A]  (closure bought)")
        ax.set_title("gap at equal damage")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    df = load(Path(args.results_dir))
    if df.empty:
        raise SystemExit("FATAL: no successful rows to summarize.")
    summary = summarize(df)
    csv = Path(f"{args.out_prefix}_summary.csv")
    summary.to_csv(csv, index=False)
    print(f"wrote {csv}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        cols = [c for c in ["invert_tracks", "t", "n", "med_ca_end_gap_A_xt",
                            "med_ca_end_gap_A_x1pred", "med_ca_end_gap_A_sdedit",
                            "frac_reachable_xt", "frac_reachable_x1pred", "frac_reachable_sdedit",
                            "frac_inversion_tighter", "med_ca_rmsd_to_input_A_xt",
                            "med_contact_retention_xt", "med_nc_gap_A_x1pred",
                            "med_roundtrip_ca_rmsd_A"] if c in summary]
        print(summary[cols].to_string(index=False))

    print(f"\ninput median CA end gap: {df['input_ca_end_gap_A'].median():.2f} A; "
          f"input median N-C gap: {df['input_nc_gap_A'].median():.2f} A "
          f"over {df['example_id'].nunique()} peptides")
    figure(df, summary, Path(f"{args.out_prefix}_gap_vs_cost.png"))


if __name__ == "__main__":
    main()
