#!/usr/bin/env python3
"""Which arm is best, as a box over complexes -- and the winner against the native.

`cpseq_seed_box_arms.py` draws five arms on FOUR hand-picked complexes, one panel each.
Four complexes cannot rank five arms, and that figure says so. This draws the quantity
that can: one box per arm over EVERY complex with a baseline, where each point in a box
is one complex's cross-fitted held-out delta. That is the distribution behind the pooled
median `cpseq_selected_candidate_report.py --pilot` reports, so the ranking in this
figure is the ranking the estimator makes.

Two things this figure must not be mistaken for:

1. **A box here is nine complexes, not nine folds.** Each point is already a per-complex
   average of the two cross-fit directions, so fold noise is inside the point, not
   between points. The box therefore shows how much the arm's advantage varies BY TARGET,
   which is the variation that decides whether an arm generalises.

2. **It is not best-of-B.** The selected sequence is measured only on the five folds that
   did NOT choose it. Pooling all ten would recover the quantity
   `README_CPSEQ_PEPBENCH_SCALEUP` sec. 10 withdrew -- -4.66 REU that became -0.06 under
   cross-fitting -- because a candidate that wins a selection is partly the one whose
   folding noise happened to be favourable. There is no honest "the best sequence it
   finds" number that is not cross-fitted, so this figure does not offer one.

**The draw budget is not equal across arms and is printed under every box.** Best-of-B
improves with B, so an arm with more draws has an advantage that is not its proposal
rule. `random_control` is what makes the ranking readable anyway: it is the positional
control, it gets by far the most draws, and where it still loses to an arm on a quarter
of its budget, that arm's lead is not a budget artefact. Read every comparison against
the control's row, never between two arms alone.

Panel B puts the winning arm's chosen sequence beside the native peptide in absolute
REU, per complex, because panel A's zero line is the native and a reader is entitled to
ask what the native actually scored. Absolute energies are NOT pooled across complexes:
different receptors sit at different dG scales, so a pooled absolute box would be a
mixture of targets and not a measurement of anything. The delta is the poolable
quantity, and that is panel A.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_arm_box.py \
        --out-dir /zfsauton/scratch/yixiz/cpseq/llm_ladder \
        --scoring-version v3-cis-amide-muco-windows \
        --fig-dir /zfsauton/scratch/yixiz/cpseq/figs/arm_box

    # redraw from the staged rows, no journal read
    python script_utils/cpseq_arm_box.py \
        --from-rows /zfsauton/scratch/yixiz/cpseq/figs/arm_box/arm_box_rows.json \
        --fig-dir /zfsauton/scratch/yixiz/cpseq/figs/arm_box
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_seed_box import (  # noqa: E402
    C_REF, C_SEL, INK, INK_2, SURFACE, PILOT_ENERGY_KEY, PILOT_MIN_REF_VALID,
    DEPTH, extract,
)
from cpseq_seed_box_arms import ARMS, SHORT, held_out_folds  # noqa: E402

# De-emphasis for the arms that are not the winner. This is a NEUTRAL, not a sixth
# categorical hue: identity on this axis is carried by the tick label, and the only
# colour distinction the figure rests on is winner-vs-rest, which is one pair.
C_MUTE = "#9a9894"
GRID = "#e7e6e3"


def sign_p(wins: int, losses: int) -> float | None:
    """Exact two-sided sign test. None when there is nothing to test."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def stage(out_dir: Path, arms: list[str], scoring_version: str | None,
          energy_key: str, min_ref_valid: int, depth: int) -> dict:
    """Per-arm per-complex deltas, plus the held-out folds needed for panel B."""
    per_arm = {}
    for arm in arms:
        r = extract(out_dir, arm, scoring_version, None, energy_key,
                    min_ref_valid, depth, "median", "median")
        rows = {}
        for p in r["panels"]:
            if p["delta_avg"] is None:
                continue
            groups = [g for g in p["groups"] if g is not None]
            rows[p["parent_id"]] = {
                "delta": p["delta_avg"],
                "n_draws": p["n_draws"],
                "same_pick": bool(p["same_pick"]),
                "ref_folds": held_out_folds(groups, "ref"),
                "sel_folds": held_out_folds(groups, "sel"),
                "ref_sequence": p["ref_sequence"],
                "sel_sequences": sorted({g["sequence"] for g in groups}),
            }
        per_arm[arm] = rows
        print(f"[arm-box] {arm}: {len(rows)} complexes with a baseline")

    # Only complexes measured under EVERY arm can rank the arms: a box drawn over a
    # different complex set per arm would put the easy targets of one arm against the
    # hard targets of another and call the difference an arm effect.
    common = sorted(set.intersection(*[set(v) for v in per_arm.values()])) \
        if per_arm else []
    print(f"[arm-box] common to all {len(arms)} arms: {len(common)}")

    summary = {}
    for arm in arms:
        ds = [per_arm[arm][c]["delta"] for c in common]
        nd = [per_arm[arm][c]["n_draws"] for c in common]
        w, l = sum(d < 0 for d in ds), sum(d >= 0 for d in ds)
        summary[arm] = {
            "n": len(ds), "median": st.median(ds), "mean": st.fmean(ds),
            "wins": w, "losses": l, "sign_p": sign_p(w, l),
            "draws_median": st.median(nd), "draws_min": min(nd), "draws_max": max(nd),
            "n_all": len(per_arm[arm]),
        }

    return {
        "out_dir": str(out_dir), "scoring_version": scoring_version,
        "energy_key": energy_key, "min_ref_valid": min_ref_valid, "depth": depth,
        "arms": arms, "common": common, "per_arm": per_arm, "summary": summary,
    }


def head_to_head(rows: dict, a: str, b: str) -> dict:
    """`a` against `b`, paired by complex. Wins counts complexes where `a` is lower."""
    d = [rows["per_arm"][a][c]["delta"] - rows["per_arm"][b][c]["delta"]
         for c in rows["common"]]
    w, l = sum(x < 0 for x in d), sum(x >= 0 for x in d)
    return {"wins": w, "n": w + l, "median_gap": st.median(d), "sign_p": sign_p(w, l)}


def print_table(rows: dict) -> str:
    """The ranking as text, and the winner's name so the figure cannot disagree."""
    order = sorted(rows["arms"], key=lambda a: rows["summary"][a]["median"])
    print("\n=== pooled over the common complex set, pinned estimator "
          "(select=median, measure=median) ===")
    print(f"{'arm':24s} {'n':>3s} {'medianD':>9s} {'meanD':>9s} {'W/L':>6s} "
          f"{'sign p':>7s} {'draws':>12s}")
    for a in order:
        s = rows["summary"][a]
        p = "-" if s["sign_p"] is None else f"{s['sign_p']:.3f}"
        print(f"{a:24s} {s['n']:3d} {s['median']:+9.2f} {s['mean']:+9.2f} "
              f"{s['wins']:>3d}/{s['losses']:<2d} {p:>7s} "
              f"{s['draws_median']:>5.0f} [{s['draws_min']}-{s['draws_max']}]")
    print("\n  medianD  median over complexes of the per-complex held-out delta, the")
    print("           pooled headline. Negative = better than the native peptide.")
    print("  draws    candidates the arm chose from. NOT equal across arms; best-of-B")
    print("           improves with B, so read every row against random_control's.")

    best = order[0]
    print(f"\n[arm-box] best arm on this cohort: {best} "
          f"({rows['summary'][best]['median']:+.2f} REU)")

    # Head-to-head against every other arm, paired by complex. The pooled medians can
    # separate two arms that trade wins target by target, which is a different claim.
    print(f"\n=== {best} head-to-head, paired by complex ===")
    sep = []
    for a in order[1:]:
        h = head_to_head(rows, best, a)
        p = h["sign_p"]
        print(f"  vs {a:24s} wins {h['wins']}/{h['n']}  "
              f"median gap {h['median_gap']:+6.2f} REU  "
              f"sign p {'-' if p is None else f'{p:.3f}'}")
        if p is not None and p < 0.05:
            sep.append(a)
    if sep:
        print(f"\n[arm-box] {best} separates from: {', '.join(sep)}")
    else:
        print(f"\n[arm-box] {best} separates from NO other arm at p<0.05. The pooled")
        print("          median orders the arms; the paired tests do not support that")
        print("          order as a ranking. Quote 'every arm beats the native', not")
        print("          'this arm beats that one'.")
    return best


def _panel_b(axB, rows: dict, best: str, *, standalone: bool = False) -> None:
    """Panel B: the selected sequence against the native, per complex, absolute REU.

    Shared by the two-panel figure and the panel-B-only render, so the slide figure is
    the identical draw as the right half of the full one and cannot drift from it.
    """
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.lines import Line2D

    common = rows["common"]
    axB.set_facecolor(SURFACE)
    # Ordered by how much the winner gained, so the panel reads as a gradient rather
    # than an arbitrary complex order.
    cs = sorted(common, key=lambda c: rows["per_arm"][best][c]["delta"])
    xs = list(range(1, len(cs) + 1))
    for i, c in enumerate(cs, start=1):
        r = rows["per_arm"][best][c]
        rf, sf = r["ref_folds"], r["sel_folds"]
        if not rf or not sf:
            continue
        rm, sm = st.median(rf), st.median(sf)
        axB.plot([i, i], [rm, sm], color=INK_2, lw=1.3, zorder=2)
        for e in rf:
            axB.plot(i - 0.17, e, "o", ms=4.6, mfc=C_REF, mec=SURFACE, mew=0.9,
                     alpha=0.55, zorder=3)
        for e in sf:
            axB.plot(i + 0.17, e, "o", ms=4.6, mfc=C_SEL, mec=SURFACE, mew=0.9,
                     alpha=0.55, zorder=3)
        axB.plot(i - 0.17, rm, "s", ms=9.5, mfc=C_REF, mec=INK, mew=1.1, zorder=6)
        axB.plot(i + 0.17, sm, "D", ms=9.0, mfc=C_SEL, mec=INK, mew=1.1, zorder=6)
        axB.annotate(f"{r['delta']:+.1f}", xy=(i, min(rm, sm)),
                     xytext=(0, -15), textcoords="offset points", ha="center",
                     fontsize=8.4, color=INK,
                     fontweight="bold" if r["delta"] < 0 else "normal")

    axB.set_xticks(xs)
    axB.set_xticklabels([c.replace("pepbench_", "").replace("_pdb", " ")
                         for c in cs], rotation=38, ha="right", fontsize=8.4,
                        color=INK)
    axB.set_xlim(0.4, len(cs) + 0.6)
    axB.set_ylabel(f"{rows['energy_key']}   (REU, absolute)", fontsize=10, color=INK)
    title = (f"{best}'s chosen sequence vs the native peptide   ·   held-out folds only"
             if standalone else
             f"B.  {best}'s chosen sequence vs the native peptide   ·   "
             f"held-out folds only")
    axB.set_title(title, fontsize=11.5, color=INK, loc="left", pad=8)
    axB.grid(axis="y", color=GRID, lw=0.9)
    axB.set_axisbelow(True)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axB.spines[sp].set_color(INK_2)
    axB.legend(handles=[
        Line2D([], [], marker="s", ls="", mfc=C_REF, mec=INK, ms=9,
               label="native peptide (median of held-out folds)"),
        Line2D([], [], marker="D", ls="", mfc=C_SEL, mec=INK, ms=8.5,
               label=f"{best} selected (median of held-out folds)"),
        Line2D([], [], marker="o", ls="", mfc=C_MUTE, mec=SURFACE, ms=5,
               label="individual held-out folds"),
    ], loc="upper left", fontsize=8.6, frameon=False, labelcolor=INK)


def draw_panel_b(rows: dict, fig_dir: Path, fig_name: str, best: str) -> None:
    """Panel B alone, on its own axes, for a slide that wants only the per-complex view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.6, 5.8), facecolor=SURFACE)
    ax = fig.add_subplot(111)
    _panel_b(ax, rows, best, standalone=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.2)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = fig_dir / f"{fig_name}.{ext}"
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        print(f"[arm-box] -> {out}")
    plt.close(fig)


def draw(rows: dict, fig_dir: Path, fig_name: str, best: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order = sorted(rows["arms"], key=lambda a: -rows["summary"][a]["median"])
    common = rows["common"]

    fig = plt.figure(figsize=(15.6, 7.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.17,
                          left=0.055, right=0.985, top=0.715, bottom=0.14)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---------------- Panel A: one box per arm, over complexes ----------------
    axA.set_facecolor(SURFACE)
    data = [[rows["per_arm"][a][c]["delta"] for c in common] for a in order]
    pos = list(range(1, len(order) + 1))

    axA.axhline(0.0, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=2)
    axA.annotate("native peptide", xy=(len(order) + 0.46, 0.0),
                 xytext=(-2, 5), textcoords="offset points",
                 ha="right", va="bottom", fontsize=9, color=INK)

    bp = axA.boxplot(data, positions=pos, widths=0.56, patch_artist=True,
                     medianprops=dict(color=INK, lw=2.0), showfliers=False,
                     whiskerprops=dict(color=INK_2, lw=1.2),
                     capprops=dict(color=INK_2, lw=1.2))
    for a, patch in zip(order, bp["boxes"]):
        win = a == best
        patch.set_facecolor(C_SEL if win else C_MUTE)
        patch.set_alpha(0.30 if win else 0.22)
        patch.set_edgecolor(C_SEL if win else INK_2)
        patch.set_linewidth(1.8 if win else 1.2)

    # Nine complexes is few enough that the points ARE the evidence; the box is a
    # summary of them, so both are drawn and the points sit on top.
    rng = __import__("random").Random(0)
    for x, a in zip(pos, order):
        win = a == best
        for c in common:
            d = rows["per_arm"][a][c]["delta"]
            axA.plot(x + rng.uniform(-0.17, 0.17), d, "o", ms=6.0,
                     mfc=(C_SEL if win else C_MUTE), mec=SURFACE, mew=1.4,
                     alpha=0.95, zorder=5)

    ticks = []
    for a in order:
        s = rows["summary"][a]
        p = "" if s["sign_p"] is None else f"  p={s['sign_p']:.2f}"
        ticks.append(f"{SHORT.get(a, a)}\n{s['median']:+.2f} REU\n"
                     f"{s['wins']}/{s['wins'] + s['losses']} wins{p}\n"
                     f"{s['draws_median']:.0f} draws")
    axA.set_xticks(pos)
    axA.set_xticklabels(ticks, fontsize=8.6, color=INK)
    for lbl, a in zip(axA.get_xticklabels(), order):
        if a == best:
            lbl.set_color(C_SEL)
            lbl.set_fontweight("bold")
    axA.set_xlim(0.45, len(order) + 0.55)
    axA.set_ylabel("held-out Δ dG   (selected − native, REU)", fontsize=10, color=INK)
    axA.set_title(f"A.  Which arm is best   ·   {len(common)} complexes, "
                  f"cross-fitted, paired",
                  fontsize=11.5, color=INK, loc="left", pad=8)
    axA.grid(axis="y", color=GRID, lw=0.9)
    axA.set_axisbelow(True)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axA.spines[sp].set_color(INK_2)
    axA.annotate("better than native ↓", xy=(0.012, 0.028), xycoords="axes fraction",
                 fontsize=9, color=INK_2, style="italic")

    # ---------------- Panel B: the winner's pick vs the native ----------------
    _panel_b(axB, rows, best)

    s = rows["summary"][best]
    # The headline is the claim the PAIRED tests support, not the one the pooled
    # medians suggest. An arm can top the pooled median and still lose the head-to-head
    # count against the arm below it, which is exactly what happens here, so a title
    # reading "<arm> wins" would be contradicted by the figure's own numbers.
    h2h = {a: head_to_head(rows, best, a) for a in rows["arms"] if a != best}
    sep = [a for a, h in h2h.items() if h["sign_p"] is not None and h["sign_p"] < 0.05]
    worst = min(rows["arms"], key=lambda a: -rows["summary"][a]["median"])
    # Lowest sign-test p against the native, ties broken by the median. A None p means
    # the arm had nothing to test, which must sort last rather than crash the compare.
    cleanest = min(rows["arms"],
                   key=lambda a: (rows["summary"][a]["sign_p"]
                                  if rows["summary"][a]["sign_p"] is not None else 2.0,
                                  rows["summary"][a]["median"]))
    cs = rows["summary"][cleanest]

    single = len(rows["arms"]) == 1
    if single:
        head = (f"{best}: {s['wins']}/{s['n']} complexes beat the native peptide "
                f"(cross-fitted, p={s['sign_p']:.3f}, {s['draws_median']:.0f} draws)")
    elif sep:
        head = (f"All {len(rows['arms'])} proposal arms beat the native peptide; "
                f"{best} separates from {', '.join(sep)}")
    else:
        head = (f"All {len(rows['arms'])} proposal arms beat the native peptide — "
                f"but at n={s['n']} they do not separate from each other")
    fig.suptitle(head, fontsize=13.5, color=INK, x=0.055, ha="left", y=0.972)

    nearest = sorted(h2h.items(),
                     key=lambda kv: -(kv[1]["sign_p"] if kv[1]["sign_p"] is not None
                                      else -1.0))[:2]
    near_txt = ";  ".join(
        f"vs {a} {h['wins']}/{h['n']}"
        + ("" if h["sign_p"] is None else f" (p={h['sign_p']:.2f})")
        for a, h in nearest)
    # The control is the budget yardstick, but it is not in every arm set (the
    # potentials-ON cohort renames its arms), so the clause naming it is conditional.
    ctl = rows["summary"].get("random_control")
    ctl_txt = ("" if ctl is None else
               f" against random_control's {ctl['draws_median']:.0f}")
    fig.text(
        0.055, 0.925,
        "Each point in A is ONE complex: the candidate is selected on five folds and "
        "measured on the other five, then the two directions are averaged.  Not "
        "best-of-B — measuring the selected\nsequence on the folds that chose it "
        "inflated this gap to −4.66 REU where cross-fitting leaves −0.06.\n"
        f"Best pooled median is {best} ({s['median']:+.2f} REU), but PAIRED by complex "
        f"it is level with its neighbours — {near_txt}\n"
        "— so the left-to-right order in A is NOT a ranking the data support.   "
        f"{worst} is LAST ({rows['summary'][worst]['median']:+.2f} REU, "
        f"p={rows['summary'][worst]['sign_p']:.2f}).\n"
        f"The strongest claim this cohort carries is {cleanest}'s {cs['wins']}/{cs['n']}"
        f" sweep over the native (p={cs['sign_p']:.3f}) on {cs['draws_median']:.0f} "
        f"draws{ctl_txt}.\n"
        "Draw budgets differ across arms (bottom row of A).   Rosetta ref2015 REU, not "
        "measured binding free energies.",
        fontsize=8.9, color=INK_2, ha="left", va="top", linespacing=1.6)

    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = fig_dir / f"{fig_name}.{ext}"
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        print(f"[arm-box] -> {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", help="run directory to read journals from")
    ap.add_argument("--from-rows", help="redraw from staged rows, no journal read")
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--fig-name", default="arm_box")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--scoring-version", default=None)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--energy-key", default=PILOT_ENERGY_KEY)
    ap.add_argument("--min-ref-valid", type=int, default=PILOT_MIN_REF_VALID)
    ap.add_argument("--panel", choices=("both", "b"), default="both",
                    help="'b' draws only the per-complex panel (selected vs native), "
                         "for a slide that wants that half alone.")
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    if args.from_rows:
        rows = json.loads(Path(args.from_rows).read_text())
    else:
        if not args.out_dir:
            ap.error("one of --out-dir or --from-rows is required")
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        rows = stage(Path(args.out_dir), arms, args.scoring_version,
                     args.energy_key, args.min_ref_valid, args.depth)
        fig_dir.mkdir(parents=True, exist_ok=True)
        staged = fig_dir / f"{args.fig_name}_rows.json"
        staged.write_text(json.dumps(rows, indent=1))
        print(f"[arm-box] rows -> {staged}")

    best = print_table(rows)
    if args.panel == "b":
        draw_panel_b(rows, fig_dir, args.fig_name, best)
    else:
        draw(rows, fig_dir, args.fig_name, best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
