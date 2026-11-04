#!/usr/bin/env python3
"""Per-seed box plots: the unchanged peptide against the cross-fit-selected candidate.

`cpseq_selected_candidate_report.py --pilot` prints ONE number per complex. This draws
the folds that number is computed from, so the spread behind it is visible rather than
asserted. Nothing is folded and nothing is re-scored; it reads the same journals.

**The one thing this figure must not do is put all ten folds of the selected sequence in
one box.** Five of them CHOSE it. A box over all ten is the best-of-B quantity that
`README_CPSEQ_PEPBENCH_SCALEUP` sec. 10 withdrew -- a -4.66 REU advantage that became
-0.06 under cross-fitting -- because the candidate that wins a selection is partly the
one whose folding noise happened to be favourable, and that same noise is then reported
as its energy. So each selected sequence is drawn with its selection folds as HOLLOW
markers outside the box and its held-out folds as FILLED markers inside it, and the box
and its median line are computed only from the held-out five.

The estimator runs in both directions (select on ranks [0,5) and measure on [5,10), then
the reverse), which is what the pilot rule does, and the two directions can pick
different sequences. They are drawn as two groups per panel rather than averaged,
because averaging them is the last step of the estimator and this figure exists to show
the steps before it. The averaged value is printed in the panel title, and it is the
number that enters the pooled median.

Both sides of a group have the same fold count -- five against five, the same seeds --
which is rule 2 of the ladder report: the peptide gets the same number of folds as the
design it is compared against.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    # extract + draw
    python script_utils/cpseq_seed_box.py \
        --out-dir /zfsauton/scratch/yixiz/cpseq/llm_ladder \
        --scoring-version v3-cis-amide-muco-windows --arm random_control \
        --complexes lnr_1t5z,pepbench_A_E_pdb6wmq \
        --fig-dir $ZFS/cpseq/figs/seed_box

    # redraw from the staged rows, no journal read
    python script_utils/cpseq_seed_box.py --from-rows $ZFS/cpseq/figs/seed_box/rows.json \
        --fig-dir $ZFS/cpseq/figs/seed_box
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The pilot estimator's settings, named here so the figure cannot quietly be drawn under
# a different rule than the number it illustrates. These are
# `cpseq_selected_candidate_report.py --pilot`.
PILOT_ENERGY_KEY = "refined_dG_separated"
PILOT_MIN_REF_VALID = 3
DEPTH = 10

# dataviz reference palette, categorical slots 1 and 2 (light mode). This pair is the
# documented adjacent pair: worst-case CVD dE 9.1 and normal-vision dE 19.6, both above
# the gates, so identity survives colour-vision deficiency. Role within a series
# (selection fold vs held-out fold) is carried by MARKER FILL, not by a third hue, so no
# distinction in this figure rests on colour alone.
C_REF = "#2a78d6"
C_SEL = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


AGGS = ("median", "mean", "mean_clip")


def agg_energy_in(seq, lo: int, hi: int, how: str) -> float | None:
    """Summarise a sequence's passing folds in [lo, hi) by `how`.

    `Sequence.median_energy_in` is the project's rule and is what every recorded number
    was computed with. This generalises it so median and mean can be compared on the
    SAME folds, and asserts the median branch reproduces `median_energy_in` exactly --
    otherwise a comparison of aggregators would silently also be a comparison of two
    fold-selection rules.

    `mean_clip` charges each fold at min(energy, 0) before averaging. That is not
    cosmetic: 0.38 % of folds come back from refinement at up to +30,351 REU and some of
    them PASS the validity checks, and `cpseq_meanscore_dataset.py` measured the
    unclipped mean's split-half reliability at -0.02 against the clipped mean's 0.83 and
    the median's 0.81. An unclipped mean is therefore only safe on a cohort checked to
    have no positive folds -- which is what `--stats-table` reports per box.
    """
    e = [r[seq.energy_key] for r in seq.folds_in(lo, hi)
         if r.get("valid") and r.get(seq.energy_key) is not None]
    if not e:
        return None
    if how == "median":
        got = st.median(e)
        assert got == seq.median_energy_in(lo, hi), "median branch drifted"
        return got
    if how == "mean":
        return st.fmean(e)
    if how == "mean_clip":
        return st.fmean([min(x, 0.0) for x in e])
    raise ValueError(f"unknown aggregator: {how}")


def pick(cands, lo: int, hi: int, how: str = "median"):
    """The candidate with the lowest aggregated energy over the selection folds.

    Six lines duplicated from `cpseq_selected_candidate_report.main.pick_and_measure`,
    which is nested in `main()` and cannot be imported. Duplicating a selection rule is
    exactly how a figure drifts from the number it illustrates, so `--check-against`
    re-reads that report's own per-complex JSON and asserts every delta drawn here
    matches it. Run it; the figure is not trustworthy without it. The check is only
    meaningful at `how="median"`, which is the rule that report implements.
    """
    scored = [(agg_energy_in(c, lo, hi, how), c) for c in cands
              if agg_energy_in(c, lo, hi, how) is not None]
    if not scored:
        return None
    return min(scored, key=lambda t: t[0])[1]


def fold_points(seq, lo: int, hi: int, energy_key: str) -> list[dict]:
    """Per-seed rows in rank order: the energy, or None where the fold failed the gate.

    A failed fold is kept as a row with `energy: None` rather than dropped, so a panel
    can state how many of the ten seeds have no energy at all. Dropping them would draw
    a five-point box as if it were a ten-point one.
    """
    out = []
    for rank in range(lo, hi):
        row = seq.by_rank.get(rank)
        if row is None:
            out.append({"rank": rank, "energy": None, "valid": None, "missing": True})
            continue
        ok = bool(row.get("valid"))
        e = row.get(energy_key)
        out.append({
            "rank": rank,
            "energy": float(e) if (ok and e is not None) else None,
            "valid": ok,
            "missing": False,
        })
    return out


def box_stats(seq, lo: int, hi: int, energy_key: str) -> dict:
    """Every summary of one box's held-out folds, so none has to be recomputed later.

    Carries `n_positive` because it is the precondition on reading the mean at all: a
    box with no positive folds is one where mean and clipped mean are identical and the
    blow-up failure mode cannot be operating.
    """
    e = [r[energy_key] for r in seq.folds_in(lo, hi)
         if r.get("valid") and r.get(energy_key) is not None]
    if not e:
        return {"n": 0, "n_folds": hi - lo}
    return {
        "n": len(e), "n_folds": hi - lo,
        "median": st.median(e), "mean": st.fmean(e),
        "mean_clip": st.fmean([min(x, 0.0) for x in e]),
        "best": min(e), "worst": max(e), "spread": max(e) - min(e),
        "n_positive": sum(1 for x in e if x > 0),
    }


def extract(out_dir: Path, arm: str, scoring_version: str | None,
            complexes: list[str] | None, energy_key: str,
            min_ref_valid: int, depth: int,
            select_agg: str = "median", measure_agg: str = "median") -> dict:
    """Stage the per-seed rows the figure needs, straight from the journals."""
    from cpseq_ladder_report import build, load, warn_if_mixed

    per_parent = load(out_dir)
    if warn_if_mixed(per_parent, scoring_version):
        raise SystemExit("mixed scoring versions: pass --scoring-version")
    state = build(per_parent, draw_arm=arm, scoring_version=scoring_version,
                  energy_key=energy_key)

    half = depth // 2
    # (select_lo, select_hi, measure_lo, measure_hi), the pilot rule's two directions.
    splits = [(0, half, half, depth), (half, depth, 0, half)]

    panels = []
    for pid in (complexes if complexes else sorted(state)):
        if pid not in state:
            print(f"[seed-box] {pid}: not in run directory, skipped")
            continue
        s = state[pid]
        ref, draws = s["reference"], s["draws"]
        if ref is None or ref.n_folds(depth) < depth:
            print(f"[seed-box] {pid}: reference below depth, skipped")
            continue
        if ref.n_pass(depth) < min_ref_valid:
            # Same exclusion the estimator makes, and for the same reason: a complex
            # whose native peptide rarely folds has no baseline to beat. It is a missing
            # measurement, not a result, so it does not get a panel.
            print(f"[seed-box] {pid}: no baseline (ref_valid="
                  f"{ref.n_pass(depth)} < {min_ref_valid}), skipped")
            continue
        cands = [draws[k] for k in sorted(draws)]

        groups, deltas = [], []
        for slo, shi, mlo, mhi in splits:
            chosen = pick(cands, slo, shi, select_agg)
            if chosen is None:
                groups.append(None)
                continue
            ref_e = agg_energy_in(ref, mlo, mhi, measure_agg)
            got = agg_energy_in(chosen, mlo, mhi, measure_agg)
            delta = None if (ref_e is None or got is None) else got - ref_e
            if delta is not None:
                deltas.append(delta)
            groups.append({
                "select": [slo, shi],
                "measure": [mlo, mhi],
                "draw_index": chosen.draw_index,
                "sequence": next(iter(chosen.by_rank.values()))["sequence"],
                "ref_median_measure": ref_e,
                "sel_median_measure": got,
                "sel_median_select": agg_energy_in(chosen, slo, shi, select_agg),
                "delta": delta,
                # Every aggregator for both held-out boxes, so the figure can draw the
                # mean beside the median without a second pass over the journals.
                "ref_stats": box_stats(ref, mlo, mhi, energy_key),
                "sel_stats": box_stats(chosen, mlo, mhi, energy_key),
                # All ten folds of each side, so the figure can show the selection half
                # as well as the half that is actually measured.
                "ref_folds": fold_points(ref, 0, depth, energy_key),
                "sel_folds": fold_points(chosen, 0, depth, energy_key),
            })
        panels.append({
            "parent_id": pid,
            "corpus": s["corpus"],
            "ref_sequence": next(iter(ref.by_rank.values()))["sequence"],
            "ref_valid": ref.n_pass(depth),
            "n_draws": len(cands),
            "groups": groups,
            # The estimator averages the two directions per complex before any median,
            # so this is the value the pooled number is built from.
            "delta_avg": st.mean(deltas) if deltas else None,
            "same_pick": (groups[0] and groups[1]
                          and groups[0]["sequence"] == groups[1]["sequence"]),
        })

    return {
        "out_dir": str(out_dir),
        "arm": arm,
        "scoring_version": scoring_version,
        "depth": depth,
        "estimator": {
            "draws": "all", "directions": 2,
            "min_ref_valid": min_ref_valid, "energy_key": energy_key,
            "select_agg": select_agg, "measure_agg": measure_agg,
        },
        "panels": panels,
    }


def sweep_aggs(out_dir: Path, arm: str, scoring_version: str | None,
               energy_key: str, min_ref_valid: int, depth: int) -> dict:
    """The pooled estimate under every (select, measure) aggregator pair.

    Four panels cannot settle whether the mean is the better summary -- they are the
    four complexes that were picked to be drawn. This runs the SAME estimator over every
    complex with a baseline and changes one thing: how a sequence's folds are collapsed
    to one number. Selection and measurement are swept separately because they are
    different claims. Measuring by the mean asks "does the reported gap change"; also
    SELECTING by the mean asks "would we have shipped a different sequence", and only
    the second can change which candidate wins.

    `median/median` must reproduce the pinned estimator exactly; it is the control row
    and the table is not readable without it.
    """
    from cpseq_ladder_report import build, load, sign_test, warn_if_mixed

    per_parent = load(out_dir)
    if warn_if_mixed(per_parent, scoring_version):
        raise SystemExit("mixed scoring versions: pass --scoring-version")
    state = build(per_parent, draw_arm=arm, scoring_version=scoring_version,
                  energy_key=energy_key)
    half = depth // 2
    splits = [(0, half, half, depth), (half, depth, 0, half)]

    # Does the blow-up failure mode operate on this cohort at all? If no passing fold is
    # positive anywhere, `mean` and `mean_clip` are the same column and the -0.02
    # reliability result cannot be what distinguishes them here.
    n_pos = n_tot = 0
    for s in state.values():
        seqs = [s["reference"]] + list(s["draws"].values())
        for seq in seqs:
            if seq is None:
                continue
            for r in seq.folds(depth):
                v = r.get(energy_key)
                if r.get("valid") and v is not None:
                    n_tot += 1
                    n_pos += v > 0

    rows = []
    for s_agg in AGGS:
        for m_agg in AGGS:
            deltas, wins, losses, unmeas, no_base = [], 0, 0, 0, 0
            for _pid, s in state.items():
                ref, draws = s["reference"], s["draws"]
                if ref is None or ref.n_folds(depth) < depth:
                    continue
                if ref.n_pass(depth) < min_ref_valid:
                    no_base += 1
                    continue
                cands = [draws[k] for k in sorted(draws)]
                per_dir = []
                for slo, shi, mlo, mhi in splits:
                    chosen = pick(cands, slo, shi, s_agg)
                    if chosen is None:
                        continue
                    ref_e = agg_energy_in(ref, mlo, mhi, m_agg)
                    got = agg_energy_in(chosen, mlo, mhi, m_agg)
                    if ref_e is not None and got is not None:
                        per_dir.append(got - ref_e)
                if not per_dir:
                    unmeas += 1
                    continue
                d = st.mean(per_dir)
                deltas.append(d)
                wins += d < 0
                losses += d > 0
            rows.append({
                "select_agg": s_agg, "measure_agg": m_agg,
                "n": len(deltas),
                "median_delta": st.median(deltas) if deltas else None,
                "mean_delta": st.fmean(deltas) if deltas else None,
                "wins": wins, "losses": losses,
                "sign_p": sign_test(wins, losses),
                "unmeasurable": unmeas, "no_baseline": no_base,
            })

    print(f"=== aggregator sweep: {out_dir}  arm {arm}  gate {scoring_version} ===")
    # Whether clipping actually changes an answer is settled by the mean/mean_clip rows
    # being equal or not, NOT by this count: a positive fold only matters if it lands in
    # a half that some selected pick or reference is measured on. Both are printed,
    # because a cohort where they disagree is one where the unclipped mean is unsafe.
    print(f"passing folds with energy > 0 REU: {n_pos} of {n_tot}"
          + ("   (none -- mean and mean_clip must agree)" if n_pos == 0 else
             "   (compare the mean and mean_clip rows: equal means none of them"
             " landed in a measured half)"))
    print()
    print(f"  {'select':>10s} {'measure':>10s} {'n':>3s} {'pooled med':>11s} "
          f"{'pooled mean':>12s} {'w/l':>8s} {'sign p':>9s} {'unmeas':>7s}")
    for r in rows:
        tag = "  <- pinned" if (r["select_agg"] == "median"
                                and r["measure_agg"] == "median") else ""
        md = "-" if r["median_delta"] is None else f"{r['median_delta']:+11.2f}"
        mn = "-" if r["mean_delta"] is None else f"{r['mean_delta']:+12.2f}"
        print(f"  {r['select_agg']:>10s} {r['measure_agg']:>10s} {r['n']:3d} {md} "
              f"{mn} {str(r['wins']) + '/' + str(r['losses']):>8s} "
              f"{r['sign_p']:9.3f} {r['unmeasurable']:7d}{tag}")
    print()
    print("  pooled med   median across complexes of the per-complex delta -- the")
    print("               headline rule. pooled mean averages them instead, and is")
    print("               sensitive to the one or two complexes with a large delta.")
    print("  Energies are Rosetta ref2015 REU, NOT measured binding free energies.")
    return {"n_positive_folds": n_pos, "n_folds": n_tot, "rows": rows}


def check_against(rows: dict, report_json: Path) -> int:
    """Assert every drawn delta equals `cpseq_selected_candidate_report`'s own."""
    d = json.loads(report_json.read_text())
    rung = d["rungs"][0]
    ref = {r["parent_id"]: r for r in rung["per_complex"]}
    bad = 0
    for p in rows["panels"]:
        want = ref.get(p["parent_id"], {}).get("energy_delta")
        got = p["delta_avg"]
        if want is None or got is None:
            print(f"[check] {p['parent_id']}: report={want} figure={got} (not compared)")
            continue
        if abs(want - got) > 1e-6:
            print(f"[check] MISMATCH {p['parent_id']}: report={want:.4f} "
                  f"figure={got:.4f}")
            bad += 1
        else:
            print(f"[check] ok {p['parent_id']}: {got:+.3f} REU")
    print(f"[check] {'FAILED' if bad else 'all drawn deltas match the report'}")
    return bad


def _ylim(values: list[float]) -> tuple[float, float, float | None]:
    """Axis limits robust to refinement blow-ups, and the cut they were taken at.

    0.38 % of folds come back from refinement at up to +30,351 REU and some of them pass
    the validity checks (`cpseq_meanscore_dataset.py`), so a plain min/max would compress
    every real point onto one line. The median the estimator reports is unaffected by
    them, but the AXIS is not, so the axis is set from the inner points and anything
    outside is drawn at the boundary and counted in the panel. The data is never clipped.
    """
    v = sorted(values)
    if len(v) < 8:
        lo, hi = v[0], v[-1]
        pad = max(1.0, 0.12 * (hi - lo))
        return lo - pad, hi + pad, None
    q1, q3 = v[len(v) // 4], v[(3 * len(v)) // 4]
    iqr = max(q3 - q1, 1e-9)
    lo_cut, hi_cut = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    inner = [x for x in v if lo_cut <= x <= hi_cut]
    lo, hi = (min(inner), max(inner)) if inner else (v[0], v[-1])
    pad = max(1.0, 0.12 * (hi - lo))
    cut = hi + pad if any(x > hi for x in v) or any(x < lo for x in v) else None
    return lo - pad, hi + pad, cut


def draw(rows: dict, fig_dir: Path, fname: str = "seed_box") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    panels = [p for p in rows["panels"] if p["delta_avg"] is not None]
    if not panels:
        raise SystemExit("[seed-box] nothing measurable to draw")

    depth = rows["depth"]
    half = depth // 2
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 0.8, 5.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    # Two direction groups, each a (peptide, selected) pair. The x positions put the
    # pair close together and the groups apart, so the comparison the estimator makes is
    # the one the eye makes first.
    xs = {(0, "ref"): 0.0, (0, "sel"): 0.78, (1, "ref"): 2.0, (1, "sel"): 2.78}

    for ax, p in zip(axes[0], panels):
        ax.set_facecolor(SURFACE)
        plotted: list[float] = []
        for gi, g in enumerate(p["groups"]):
            if g is None:
                continue
            mlo, mhi = g["measure"]
            for side, colour in (("ref", C_REF), ("sel", C_SEL)):
                folds = g[f"{side}_folds"]
                x = xs[(gi, side)]
                held = [f["energy"] for f in folds
                        if f["energy"] is not None and mlo <= f["rank"] < mhi]
                other = [f["energy"] for f in folds
                         if f["energy"] is not None and not (mlo <= f["rank"] < mhi)]
                plotted += held + other
                if held:
                    bp = ax.boxplot([held], positions=[x], widths=0.52,
                                    showfliers=False, whis=(0, 100),
                                    medianprops=dict(color=colour, lw=2.0),
                                    boxprops=dict(color=colour, lw=1.2),
                                    whiskerprops=dict(color=colour, lw=1.0),
                                    capprops=dict(color=colour, lw=1.0))
                    for b in bp["boxes"]:
                        b.set_alpha(0.9)
                # Held-out folds: filled, inside the box, these are the measurement.
                # Selection folds (the other half): hollow, offset right, OUTSIDE the
                # box. For the peptide both halves are just folds -- it is never
                # selected -- so its hollow points are the other direction's
                # measurement, shown for completeness of the ten seeds.
                for vals, filled, dx in ((held, True, -0.14), (other, False, 0.18)):
                    for i, v in enumerate(vals):
                        jx = x + dx + 0.045 * ((i % 3) - 1)
                        ax.plot([jx], [v], marker="o", ms=5.5, lw=0,
                                mfc=colour if filled else "none",
                                mec=colour if filled else INK_2,
                                mew=1.0 if filled else 1.1,
                                alpha=0.95 if filled else 0.8, zorder=3)
                # The MEAN of the same held-out folds, as a diamond. The box's bar is the
                # median, so the gap between bar and diamond IS the asymmetry of that
                # box's tail: diamond above the bar means a bad outlier is dragging the
                # mean, which is the whole question the mean-vs-median choice turns on.
                # Shape, not colour, so it does not compete with series identity.
                stats = g[f"{side}_stats"]
                if stats.get("n"):
                    ax.plot([x], [stats["mean"]], marker="D", ms=6.5, lw=0,
                            mfc="none", mec=colour, mew=1.6, zorder=5)
                # How many of the five held-out folds carry an energy at all. A box over
                # two passing folds is not a box, and the median the estimator reports
                # for it is a median of two numbers -- so the count is on the figure
                # rather than left to be inferred from how many dots are visible.
                ax.annotate(f"n={len(held)}/{mhi - mlo}", xy=(x, 0.965),
                            xycoords=("data", "axes fraction"), ha="center",
                            va="top", fontsize=7.5, color=INK_2)
            if g["delta"] is not None:
                ax.annotate(f"{g['delta']:+.2f}",
                            xy=((xs[(gi, 'ref')] + xs[(gi, 'sel')]) / 2, 0.015),
                            xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=9, color=INK)

        lo, hi, cut = _ylim(plotted) if plotted else (0, 1, None)
        off = [v for v in plotted if v < lo or v > hi]
        # Headroom for the per-box n= labels, so they never sit on top of a point.
        ax.set_ylim(lo, hi + 0.16 * (hi - lo))
        ax.set_xlim(-0.6, 3.4)
        ax.set_xticks([(xs[(0, 'ref')] + xs[(0, 'sel')]) / 2,
                       (xs[(1, 'ref')] + xs[(1, 'sel')]) / 2])
        ax.set_xticklabels([f"select 0–{half-1}\nmeasure {half}–{depth-1}",
                            f"select {half}–{depth-1}\nmeasure 0–{half-1}"],
                           fontsize=8.5, color=INK_2)
        # Without this a narrow panel can place two ticks close enough that they round
        # to the same label and the axis reads "-25, -25".
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax.tick_params(axis="y", labelsize=8.5, colors=INK_2)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#d8d7d2")

        same = "same sequence both directions" if p["same_pick"] else \
               "different sequence per direction"
        sub = (f"held-out Δ = {p['delta_avg']:+.2f} REU   "
               f"({p['n_draws']} draws, ref valid {p['ref_valid']}/{depth})")
        ax.set_title(f"{p['parent_id']}\n{sub}\n{same}",
                     fontsize=9.5, color=INK, linespacing=1.5)
        if off:
            ax.annotate(f"{len(off)} fold(s) off-scale", xy=(0.98, 0.10),
                        xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=8, color=INK_2)

    axes[0][0].set_ylabel(f"{rows['estimator']['energy_key']}  (Rosetta ref2015 REU)",
                          fontsize=9.5, color=INK)

    handles = [
        Line2D([], [], marker="o", lw=0, ms=6.5, mfc=C_REF, mec=C_REF,
               label="unchanged peptide — held-out fold"),
        Line2D([], [], marker="o", lw=0, ms=6.5, mfc="none", mec=INK_2,
               label="other half (not in this direction's box)"),
        Line2D([], [], marker="o", lw=0, ms=6.5, mfc=C_SEL, mec=C_SEL,
               label="selected candidate — held-out fold"),
        Line2D([], [], marker="s", lw=0, ms=7, mfc="#ffffff", mec=INK_2,
               label="box bar = median, held-out folds only"),
        Line2D([], [], marker="D", lw=0, ms=7, mfc="none", mec=INK_2, mew=1.6,
               label="diamond = mean of the same folds"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"Cross-fit-selected candidate vs unchanged peptide — one point per seed\n"
        f"arm {rows['arm']} · gate {rows['scoring_version']} · "
        f"select by lowest median energy on half the seeds, measure on the other half",
        fontsize=10.5, color=INK, y=0.985, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = fig_dir / f"{fname}.{ext}"
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", help="run directory to read journals from")
    ap.add_argument("--from-rows", help="redraw from a staged rows.json, no journal read")
    ap.add_argument("--fig-dir", default=None,
                    help="required unless --sweep-aggs, which draws nothing")
    ap.add_argument("--fig-name", default="seed_box")
    ap.add_argument("--arm", default="random_control")
    ap.add_argument("--scoring-version", default=None)
    ap.add_argument("--complexes", default=None,
                    help="comma-separated parent_ids. Default: every complex with a "
                         "baseline, which is usually too many for one figure.")
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--energy-key", default=PILOT_ENERGY_KEY)
    ap.add_argument("--min-ref-valid", type=int, default=PILOT_MIN_REF_VALID)
    ap.add_argument("--select-agg", choices=AGGS, default="median",
                    help="how a candidate's selection folds are collapsed to the one "
                         "number it is ranked by. Changing this changes WHICH sequence "
                         "is picked.")
    ap.add_argument("--measure-agg", choices=AGGS, default="median",
                    help="how the held-out folds are collapsed on both sides. Changing "
                         "this changes only how the gap is reported.")
    ap.add_argument("--sweep-aggs", action="store_true",
                    help="run the estimator over EVERY complex with a baseline under "
                         "all nine (select, measure) aggregator pairs and print the "
                         "pooled table. No figure. This is the test of whether the mean "
                         "is the better summary; four drawn panels are not.")
    ap.add_argument("--sweep-json", default=None)
    ap.add_argument("--check-against", default=None,
                    help="a cpseq_selected_candidate_report --pilot --json file. Every "
                         "drawn delta is asserted equal to its energy_delta there.")
    args = ap.parse_args()

    if args.sweep_aggs:
        if not args.out_dir:
            ap.error("--sweep-aggs needs --out-dir")
        res = sweep_aggs(Path(args.out_dir), args.arm, args.scoring_version,
                         args.energy_key, args.min_ref_valid, args.depth)
        if args.sweep_json:
            Path(args.sweep_json).write_text(json.dumps(res, indent=1))
            print(f"[seed-box] sweep -> {args.sweep_json}")
        return 0

    if not args.fig_dir:
        ap.error("--fig-dir is required when drawing")
    fig_dir = Path(args.fig_dir)
    if args.from_rows:
        rows = json.loads(Path(args.from_rows).read_text())
    else:
        if not args.out_dir:
            ap.error("one of --out-dir or --from-rows is required")
        cplx = ([c.strip() for c in args.complexes.split(",") if c.strip()]
                if args.complexes else None)
        rows = extract(Path(args.out_dir), args.arm, args.scoring_version, cplx,
                       args.energy_key, args.min_ref_valid, args.depth,
                       args.select_agg, args.measure_agg)
        fig_dir.mkdir(parents=True, exist_ok=True)
        staged = fig_dir / f"{args.fig_name}_rows.json"
        staged.write_text(json.dumps(rows, indent=1))
        print(f"[seed-box] rows -> {staged}")

    bad = 0
    if args.check_against:
        bad = check_against(rows, Path(args.check_against))

    for path in draw(rows, fig_dir, args.fig_name):
        print(f"[seed-box] figure -> {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
