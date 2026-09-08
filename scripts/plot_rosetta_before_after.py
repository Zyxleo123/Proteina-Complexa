"""The before/after Rosetta figure: what cyclization cost the interface.

Reads only the JSONL that scripts/score_sdedit_rosetta.py wrote -- no PyRosetta, no model,
no GPU -- so the figure re-renders as often as its styling needs changing.

What it draws, and why in this form
-----------------------------------
`dG_separated` is not comparable ACROSS targets: a 357-residue receptor and a 294-residue
one have different interface scales, so a bare "mean dG after = -21" says nothing. Every
panel here is therefore built on the PAIRED quantity.

  1. Per-peptide dumbbell. One row per input, the before dG and the after dG joined by a
     line, so the reader sees the level AND the move. Sorted by the move, which puts the
     peptides that lost the most binding at the top where they get read first.
  2. Per-edit delta strip. Every attempt as one mark, split by chemistry, zero marked. This
     is where the spread lives: the dumbbell shows a median, and a median hides the fact
     that seeds disagree by tens of REU.

Closure is a SECOND encoding (filled = the requested ring closed, hollow = it did not), not
a second colour: the question "does closing the ring cost binding?" is exactly a comparison
between those two groups, and it must survive colour-blind viewing and greyscale printing.

Abstentions are excluded from every rate and every panel by default (`--include-abstained`
keeps them, drawn hollow). An edit where the model proposed a different chemistry never had
its ring measured, so calling it "not closed" would be a lie of the same shape as the NaN
trap in the sweep summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DG = "binder_rosetta_dG_separated"

# Categorical slots 1-3 of the validated reference palette, assigned in fixed order and never
# cycled: one hue per chemistry, stable whatever subset a run happens to contain.
CHEM_COLOR = {"mainchain": "#2a78d6", "disulfide": "#eb6834", "isopeptide": "#1baf7a"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#dcdbd4"
BEFORE_COLOR = "#868e96"  # the input is a reference level, not a series: neutral ink


def load(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(f"FATAL: corrupt row {p}:{lineno}. Do not trust this run.")
    if not rows:
        raise SystemExit(f"FATAL: no rows in {[str(p) for p in paths]}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rosetta", nargs="+", required=True, help="rosetta_*.jsonl shard files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="Rosetta interface dG, linear input vs cyclized edit")
    ap.add_argument("--include-abstained", action="store_true",
                    help="Keep edits where the model proposed a different chemistry.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load([Path(p) for p in args.rosetta])

    before = df[df["kind"] == "before"].set_index("example_id")
    after = df[df["kind"] == "after"].copy()
    if after.empty:
        raise SystemExit("FATAL: no `after` rows -- nothing to compare against the inputs.")
    if not args.include_abstained and "requested_type_satisfied" in after:
        n0 = len(after)
        after = after[after["requested_type_satisfied"] == 1]
        print(f"dropped {n0 - len(after)} abstained edits (model proposed another chemistry); "
              f"{len(after)} remain", flush=True)
    if after.empty:
        raise SystemExit("FATAL: every after-edit was an abstention; use --include-abstained "
                         "to plot them, but no ring was ever measured in this run.")

    after["dG_before"] = after["example_id"].map(before[DG]) if DG in before else np.nan
    after["ddG"] = after[DG] - after["dG_before"]
    after["closed"] = after.get("closed", 0).fillna(0).astype(int)
    scored = after.dropna(subset=[DG, "dG_before"])
    if scored.empty:
        raise SystemExit("FATAL: no edit has both a before and an after dG (NaN everywhere) -- "
                         "check the scorer's log for Rosetta failures.")

    # ---- summary ---------------------------------------------------------------------
    def block(g: pd.DataFrame) -> dict:
        return {"n_edits": int(len(g)), "n_peptides": int(g["example_id"].nunique()),
                "median_dG_before": round(float(g["dG_before"].median()), 3),
                "median_dG_after": round(float(g[DG].median()), 3),
                "median_ddG": round(float(g["ddG"].median()), 3),
                "frac_ddG_worse": round(float((g["ddG"] > 0).mean()), 3),
                "median_contact_retention": (round(float(g["contact_retention"].median()), 3)
                                             if "contact_retention" in g else None)}

    summary = {"all": block(scored),
               "by_chemistry": {c: block(g) for c, g in scored.groupby("cyc_type")},
               "closed": block(scored[scored["closed"] == 1]) if (scored["closed"] == 1).any() else None,
               "not_closed": block(scored[scored["closed"] == 0]) if (scored["closed"] == 0).any() else None}
    (out_dir / "rosetta_before_after.json").write_text(json.dumps(summary, indent=2))
    scored.to_csv(out_dir / "rosetta_before_after.csv", index=False)

    print(f"\n{'group':22s} {'n':>4s} {'dG before':>10s} {'dG after':>10s} {'ddG':>8s} {'% worse':>8s}")
    print("-" * 66)
    for name, blk in ([("all", summary["all"])]
                      + [(c, b) for c, b in summary["by_chemistry"].items()]
                      + [(k, summary[k]) for k in ("closed", "not_closed") if summary[k]]):
        print(f"{name:22s} {blk['n_edits']:4d} {blk['median_dG_before']:10.2f} "
              f"{blk['median_dG_after']:10.2f} {blk['median_ddG']:8.2f} "
              f"{100 * blk['frac_ddG_worse']:7.0f}%")

    # ---- figure ----------------------------------------------------------------------
    per_pep = (scored.groupby(["example_id", "cyc_type"])
               .agg(dG_before=("dG_before", "first"), dG_after=(DG, "median"),
                    ddG=("ddG", "median"), n=(DG, "size"), n_closed=("closed", "sum"))
               .reset_index().sort_values("ddG"))

    h = max(3.2, 0.32 * len(per_pep) + 1.6)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, h), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # -- panel 1: per-peptide dumbbell
    y = np.arange(len(per_pep))
    for yi, (_, r) in zip(y, per_pep.iterrows()):
        col = CHEM_COLOR.get(r["cyc_type"], MUTED)
        ax1.plot([r["dG_before"], r["dG_after"]], [yi, yi], color=col, lw=2, alpha=0.55,
                 solid_capstyle="round", zorder=1)
        ax1.plot(r["dG_before"], yi, "o", ms=8, color=BEFORE_COLOR, mec="white", mew=1.6, zorder=2)
        filled = r["n_closed"] > 0
        ax1.plot(r["dG_after"], yi, "o", ms=9, mfc=col if filled else "white", mec=col, mew=2,
                 zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{e.replace('LNR_', '')}" for e in per_pep["example_id"]], fontsize=9)
    ax1.set_ylim(-0.8, len(per_pep) - 0.2)
    ax1.set_xlabel("Rosetta dG_separated (REU)   ← stronger binding", fontsize=10, color=INK)
    ax1.set_title("Per peptide: input → cyclized (median of attempts)",
                  fontsize=11, color=INK, loc="left")
    for yi, (_, r) in zip(y, per_pep.iterrows()):
        # Direct-label the move only -- the two dG levels are already on the axis, and a number
        # on every mark is noise.
        ax1.annotate(f"{r['ddG']:+.0f}", (max(r['dG_before'], r['dG_after']), yi),
                     xytext=(7, 0), textcoords="offset points", va="center",
                     fontsize=8.5, color=MUTED)

    # -- panel 2: per-edit delta strip
    chems = [c for c in ("mainchain", "disulfide", "isopeptide") if c in set(scored["cyc_type"])]
    rng = np.random.default_rng(0)
    for xi, chem in enumerate(chems):
        g = scored[scored["cyc_type"] == chem]
        col = CHEM_COLOR[chem]
        for closed_flag, mfc in ((0, "white"), (1, col)):
            gg = g[g["closed"] == closed_flag]
            if gg.empty:
                continue
            jitter = rng.uniform(-0.17, 0.17, len(gg))
            ax2.plot(xi + jitter, gg["ddG"], "o", ms=7, mfc=mfc, mec=col, mew=1.6,
                     alpha=0.9, linestyle="none", zorder=3)
        med = float(g["ddG"].median())
        ax2.plot([xi - 0.32, xi + 0.32], [med, med], color=INK, lw=2.5, zorder=4,
                 solid_capstyle="round")
        ax2.annotate(f"median {med:+.1f}", (xi + 0.34, med), fontsize=8.5, color=MUTED,
                     va="center")
    ax2.axhline(0, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax2.set_xticks(range(len(chems)))
    ax2.set_xticklabels(chems, fontsize=10)
    ax2.set_xlim(-0.6, len(chems) - 0.15 + 0.6)
    ax2.set_ylabel("Δ dG_separated, cyclized − input (REU)", fontsize=10, color=INK)
    ax2.set_title("Per edit: the cost of the ring  (above 0 = weaker binding)",
                  fontsize=11, color=INK, loc="left")

    for ax in (ax1, ax2):
        ax.grid(axis="x" if ax is ax1 else "y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    handles = [Line2D([], [], marker="o", ls="", ms=8, mfc=BEFORE_COLOR, mec="white",
                      label="input (linear)")]
    handles += [Line2D([], [], marker="o", ls="", ms=8, mfc=CHEM_COLOR[c], mec=CHEM_COLOR[c],
                       label=f"{c}, ring closed") for c in chems]
    handles += [Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=CHEM_COLOR[c], mew=1.8,
                       label=f"{c}, not closed") for c in chems]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(args.title, fontsize=13, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    dest = out_dir / "rosetta_before_after.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nfigure:  {dest}\nsummary: {out_dir / 'rosetta_before_after.json'}\n"
          f"table:   {out_dir / 'rosetta_before_after.csv'}", flush=True)


if __name__ == "__main__":
    main()
