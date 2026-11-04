#!/usr/bin/env python3
"""All five proposal arms against ONE shared peptide baseline, per complex.

`cpseq_seed_box.py` draws one arm in full cross-fit detail. This draws every arm side by
side, which is a different question -- not "is this arm better than the peptide" but
"which arm is better" -- and it is only answerable because the reference folds are
SHARED across arms (`cpseq_ladder_report.DRAW_ARM`: the reference is written under its
own arm and read by every other). Both sides of every comparison here are therefore the
identical ten foldings of the unchanged peptide, not two independent foldings of it.

Two things this figure does that need stating, because they are where it could mislead:

1. **The held-out folds are POOLED over the two cross-fit directions.** Direction A
   measures ranks 5-9 of the sequence it picked, direction B measures ranks 0-4 of the
   sequence IT picked. Those are disjoint rank windows, so pooling them gives exactly ten
   held-out measurements per arm -- but from up to TWO different sequences when the
   directions disagree. The panel marks which arms that applies to. The reference pools
   the same way and is always one sequence, so its box is its ten folds.

2. **The box is the distribution; the number under it is the estimator.** The estimator
   averages two per-direction MEDIANS, which is not the median of the pooled folds, so
   the two are shown separately rather than one being passed off as the other. The
   number is what `cpseq_selected_candidate_report.py --pilot` reports for that complex.

Arms are ordered by their pooled headline, worst-to-best left-to-right, and that
headline is printed in the tick label -- so the per-complex panels can be read against
the cohort-level number instead of being mistaken for it. Four complexes cannot reorder
five arms; the pooled column is what ranks them.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_seed_box_arms.py \
        --out-dir /zfsauton/scratch/yixiz/cpseq/llm_ladder \
        --scoring-version v3-cis-amide-muco-windows \
        --complexes lnr_1t5z,pepbench_A_E_pdb6wmq,lnr_5y59,pepbench_C_A_pdb2jz3 \
        --fig-dir /zfsauton/scratch/yixiz/cpseq/figs/seed_box
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpseq_seed_box import (  # noqa: E402
    C_REF, C_SEL, INK, INK_2, SURFACE, PILOT_ENERGY_KEY, PILOT_MIN_REF_VALID,
    DEPTH, extract,
)

ARMS = ("random_control", "random_mask_mpnn_fill", "llm_mask_mpnn_fill",
        "llm_mask_random_fill", "llm_search")

# Short tick labels. The arm names encode mask-source + fill-source, and spelled out in
# full they are wider than the boxes they label, which forces a rotation that then
# collides with the next panel. The pairing that matters is kept visible: the two
# `+mpnn` arms differ only in WHERE the mask came from, and the two `llmmask` arms only
# in what filled it.
SHORT = {
    "random_control": "control",
    "random_mask_mpnn_fill": "randmask\n+mpnn",
    "llm_mask_mpnn_fill": "llmmask\n+mpnn",
    "llm_mask_random_fill": "llmmask\n+rand",
    "llm_search": "llm_search",
}


def held_out_folds(group_list, side: str) -> list[float]:
    """The ten held-out energies, pooled across the two cross-fit directions.

    Each direction contributes only the folds it MEASURES, so the two contributions sit
    in disjoint rank windows and no fold is counted twice. A fold that failed the gate
    contributes nothing, which is why this can return fewer than ten.
    """
    out = []
    for g in group_list:
        if g is None:
            continue
        mlo, mhi = g["measure"]
        out += [f["energy"] for f in g[f"{side}_folds"]
                if f["energy"] is not None and mlo <= f["rank"] < mhi]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--fig-name", default="seed_box_arms")
    ap.add_argument("--scoring-version", default=None)
    ap.add_argument("--complexes", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--energy-key", default=PILOT_ENERGY_KEY)
    ap.add_argument("--min-ref-valid", type=int, default=PILOT_MIN_REF_VALID)
    ap.add_argument("--pilot-json-dir", default=None,
                    help="where <arm>_pilot.json live, for the pooled headline printed "
                         "in each tick label. Defaults to --fig-dir.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    cplx = [c.strip() for c in args.complexes.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    pj_dir = Path(args.pilot_json_dir or args.fig_dir)

    # Per arm: the staged rows for these complexes, plus its cohort-level headline.
    per_arm, pooled = {}, {}
    for arm in arms:
        per_arm[arm] = extract(Path(args.out_dir), arm, args.scoring_version, cplx,
                               args.energy_key, args.min_ref_valid, args.depth)
        p = pj_dir / f"{arm}_pilot.json"
        if p.is_file():
            pooled[arm] = json.loads(p.read_text())["rungs"][0]["energy_median_delta"]

    # Worst to best, so the x-axis carries the ranking instead of an arbitrary order.
    arms.sort(key=lambda a: -(pooled.get(a) or 0.0))

    panels = [p["parent_id"] for p in per_arm[arms[0]]["panels"]]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n + 0.9, 5.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, pid in zip(axes[0], panels):
        ax.set_facecolor(SURFACE)
        plotted, labels, differs = [], [], []

        def box_at(x, vals, colour):
            if not vals:
                return
            bp = ax.boxplot([vals], positions=[x], widths=0.58, showfliers=False,
                            whis=(0, 100),
                            medianprops=dict(color=colour, lw=2.0),
                            boxprops=dict(color=colour, lw=1.2),
                            whiskerprops=dict(color=colour, lw=1.0),
                            capprops=dict(color=colour, lw=1.0))
            for b in bp["boxes"]:
                b.set_alpha(0.9)
            for i, v in enumerate(vals):
                ax.plot([x - 0.17 + 0.05 * ((i % 3) - 1)], [v], marker="o", ms=5.0,
                        lw=0, mfc=colour, mec=colour, alpha=0.95, zorder=3)
            ax.plot([x], [st.fmean(vals)], marker="D", ms=6.5, lw=0, mfc="none",
                    mec=colour, mew=1.6, zorder=5)

        # The reference, once. Identical folds for every arm, which is the premise.
        rec = next(p for p in per_arm[arms[0]]["panels"] if p["parent_id"] == pid)
        ref_v = held_out_folds(rec["groups"], "ref")
        box_at(0.0, ref_v, C_REF)
        plotted += ref_v
        labels.append(f"peptide\n(shared)\nn={len(ref_v)}/{args.depth}")

        for j, arm in enumerate(arms, start=1):
            rec = next((p for p in per_arm[arm]["panels"]
                        if p["parent_id"] == pid), None)
            if rec is None:
                labels.append(f"{arm}\n(no measurement)")
                continue
            vals = held_out_folds(rec["groups"], "sel")
            box_at(j * 1.15, vals, C_SEL)
            plotted += vals
            head = f"[{pooled[arm]:+.2f}]" if arm in pooled else ""
            labels.append(f"{SHORT.get(arm, arm)}\n{head}\n"
                          f"n={len(vals)}/{args.depth}")
            if not rec["same_pick"]:
                differs.append(j * 1.15)
            ax.annotate(f"{rec['delta_avg']:+.2f}", xy=(j * 1.15, 0.015),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="bottom", fontsize=8.5, color=INK)

        lo, hi = min(plotted), max(plotted)
        pad = max(1.0, 0.10 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad + 0.10 * (hi - lo))
        ax.set_xlim(-0.65, len(arms) * 1.15 + 0.65)
        ax.set_xticks([0.0] + [j * 1.15 for j in range(1, len(arms) + 1)])
        ax.set_xticklabels(labels, fontsize=7.6, color=INK_2, linespacing=1.45)
        # A dagger marks an arm whose two cross-fit directions picked DIFFERENT
        # sequences, i.e. whose ten points are five folds each of two peptides.
        for x in differs:
            ax.annotate("†", xy=(x, 0.985), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=10, color=INK_2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis="y", labelsize=8.5, colors=INK_2)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d2")
        ax.set_title(f"{pid}\nref valid {rec['ref_valid']}/{args.depth}",
                     fontsize=10, color=INK, linespacing=1.5)

    axes[0][0].set_ylabel(f"{args.energy_key}  (Rosetta ref2015 REU)",
                          fontsize=9.5, color=INK)
    handles = [
        Line2D([], [], marker="o", lw=0, ms=6.5, mfc=C_REF, mec=C_REF,
               label="unchanged peptide (shared across all arms)"),
        Line2D([], [], marker="o", lw=0, ms=6.5, mfc=C_SEL, mec=C_SEL,
               label="that arm's cross-fit-selected candidate"),
        Line2D([], [], marker="D", lw=0, ms=7, mfc="none", mec=INK_2, mew=1.6,
               label="diamond = mean; box bar = median"),
        Line2D([], [], lw=0, label="† directions picked different sequences"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle(
        "Five proposal arms against one shared peptide baseline — one point per seed\n"
        f"gate {args.scoring_version} · held-out folds pooled over both cross-fit "
        "directions · [bracketed] = that arm's pooled 10-complex headline",
        fontsize=10.5, color=INK, y=0.985, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = fig_dir / f"{args.fig_name}.{ext}"
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"[seed-box-arms] figure -> {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
