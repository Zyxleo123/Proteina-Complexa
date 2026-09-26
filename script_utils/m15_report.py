"""Milestone 1.5 report: turn the shard outputs into the tables and figures to decide on.

Reads only files on disk, so every figure is redrawable without recomputing anything.

The headline this exists to produce: `achieved / ceiling`.  A raw contact retention is not
interpretable on its own -- 0.40 is a failure against a ceiling of 0.95 and near-perfect
against a ceiling of 0.45 -- so until the ceiling is known, optimising against retention is
optimising against an unknown scale.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed categorical order, never cycled.  Validated with the dataviz palette checker
# (light surface): lightness band, chroma floor, CVD separation, normal-vision floor and
# contrast all pass.
HUE = {"mainchain": "#2f6fb8", "disulfide": "#d97706", "isopeptide": "#159467",
       "other": "#8b5cf6",
       # The terminal-pinned bridged arms: same hue family as their free counterpart,
       # darkened, so a plot reads free-vs-pinned as one chemistry rather than two.
       "disulfide_term": "#92400e", "isopeptide_term": "#0b5f43"}
INK = "#1f2328"
INK_MUTED = "#6b7280"
GRID = "#e5e7eb"
SURFACE = "#fcfcfb"

# `*_term` are the bridged arms with anchors pinned to (0, L-1).  The unsuffixed bridged
# arms maximise over all (i, j), which CPSea cannot reach -- it conditions on the chain
# termini, so those are the only anchors it can place a bridge between.  Both are carried
# so the free-vs-reachable difference is visible; only `*_term` is a target the model is
# measured against.
CHEMISTRIES = ("mainchain", "disulfide", "isopeptide",
               "disulfide_term", "isopeptide_term")
REACHABLE_CHEMISTRIES = ("mainchain", "disulfide_term", "isopeptide_term")
GAP_BINS = [0, 5, 10, 15, 20, 25, 30, 45, 1e9]
GAP_LABELS = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-45", ">45"]
LEN_BINS = [0, 8, 11, 14, 100]
LEN_LABELS = ["5-8", "9-11", "12-14", "15+"]


def read_jsonl(patterns: list[str]) -> pd.DataFrame:
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        # Refuse, do not skip: a truncated or NUL-corrupted row means the
                        # shard is unreliable, and silently dropping it turns data loss
                        # into a quietly wrong average.
                        raise SystemExit(
                            f"{path}: corrupt JSON line ({exc}). Concurrent writes to one "
                            f"file corrupt rows here -- every job must own its output file."
                        ) from exc
    return pd.DataFrame(rows)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    if df.empty:
        return "_(no rows)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [_fmt(r[c], nd) if isinstance(r[c], float) else str(r[c]) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------ tables
def ceiling_col(df: pd.DataFrame, chem: str) -> str:
    """The torsion-confirmed ceiling where it exists, else the analytic upper bound.

    Never silently mix them: the analytic column over-licenses at the boundary, so a table
    that falls back per row would compare a tight number against a loose one.
    """
    refined = f"{chem}_ceiling_refined"
    if refined in df and df[refined].notna().any():
        return refined
    return f"{chem}_ceiling_strict"


def free_vs_reachable(df: pd.DataFrame) -> pd.DataFrame:
    """Per-chemistry cost of restricting the bridge anchors to the chain termini.

    The free scan maximises over all (i, j); CPSea can only anchor at (0, L-1).  Paired
    per complex, so `delta_median` is the median of the per-row differences rather than
    the difference of the medians -- the two are not the same statistic, and only the
    paired one answers "what does this complex lose by being restricted".
    """
    rows = []
    for chem in ("disulfide", "isopeptide"):
        free_col, term_col = ceiling_col(df, chem), ceiling_col(df, f"{chem}_term")
        if free_col not in df or term_col not in df:
            continue
        both = df.dropna(subset=[free_col, term_col])
        feas_free = df.get(f"{chem}_feasible_any")
        feas_term = df.get(f"{chem}_term_feasible_any")
        rows.append({
            "chemistry": chem,
            "n_paired": len(both),
            "feasible_frac_free": float(feas_free.mean()) if feas_free is not None else float("nan"),
            "feasible_frac_terminal": float(feas_term.mean()) if feas_term is not None else float("nan"),
            "free_median": float(both[free_col].median()) if len(both) else float("nan"),
            "terminal_median": float(both[term_col].median()) if len(both) else float("nan"),
            "delta_median_paired": float((both[free_col] - both[term_col]).median()) if len(both) else float("nan"),
            "frac_terminal_within_0.05": float(((both[free_col] - both[term_col]) <= 0.05).mean()) if len(both) else float("nan"),
        })
    return pd.DataFrame(rows)


def ceiling_by_set(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, grp in df.groupby("label"):
        for chem in CHEMISTRIES:
            col = f"{chem}_ceiling_strict"
            if col not in grp:
                continue
            v = grp[col].dropna()
            rcol = f"{chem}_ceiling_refined"
            rv = grp[rcol].dropna() if rcol in grp else pd.Series(dtype=float)
            feas = grp.get(f"{chem}_feasible_any")
            exh = grp.get(f"{chem}_refine_exhausted")
            rows.append({
                "set": label, "chemistry": chem, "n": len(v),
                "feasible_frac": float(feas.mean()) if feas is not None else float("nan"),
                "analytic_median": float(v.median()) if len(v) else float("nan"),
                "refined_median": float(rv.median()) if len(rv) else float("nan"),
                # The bracket. `refined` writes off every free residue's contacts, so it is
                # a FLOOR, not a ceiling; `permissive` lets a free residue keep a contact
                # whose partner is still in reach, and is the real upper bound. Quote both
                # or quote neither -- a single number here is a claim the data does not
                # support.
                "permissive_median": (float(grp[f"{chem}_ceiling_permissive"].median())
                                      if f"{chem}_ceiling_permissive" in grp else float("nan")),
                "n_refined": len(rv),
                "refine_exhausted_frac": float(exh.mean()) if exh is not None else float("nan"),
            })
    return pd.DataFrame(rows)


def ceiling_vs_gap(df: pd.DataFrame, chem: str = "mainchain") -> pd.DataFrame:
    col = ceiling_col(df, chem)
    if col not in df:
        return pd.DataFrame()
    d = df.dropna(subset=[col, "nc_gap_A"]).copy()
    d["gap_bin"] = pd.cut(d["nc_gap_A"], GAP_BINS, labels=GAP_LABELS, right=False)
    d["len_bin"] = pd.cut(d["peptide_length"], LEN_BINS, labels=LEN_LABELS, right=True)
    g = d.groupby("gap_bin", observed=True).agg(
        n=(col, "size"), gap_median=("nc_gap_A", "median"),
        len_median=("peptide_length", "median"),
        ceiling_median=(col, "median"), ceiling_p25=(col, "quantile"),
    ).reset_index()
    # `quantile` above defaults to 0.5; recompute the quartiles explicitly.
    q = d.groupby("gap_bin", observed=True)[col].quantile([0.25, 0.75]).unstack()
    g["ceiling_p25"] = q[0.25].values
    g["ceiling_p75"] = q[0.75].values
    g["max_window_median"] = d.groupby("gap_bin", observed=True)[
        f"{chem}_max_feasible_window_len"].median().values
    return g


def ceiling_by_length(df: pd.DataFrame, chem: str = "mainchain") -> pd.DataFrame:
    col = ceiling_col(df, chem)
    if col not in df:
        return pd.DataFrame()
    d = df.dropna(subset=[col]).copy()
    d["len_bin"] = pd.cut(d["peptide_length"], LEN_BINS, labels=LEN_LABELS, right=True)
    return d.groupby("len_bin", observed=True).agg(
        n=(col, "size"), gap_median=("nc_gap_A", "median"),
        ceiling_median=(col, "median")).reset_index()


# ------------------------------------------------------- the ratio is the wrong instrument
# `ceiling_strict` is (contacts whose residue is in the held window) / (all contacts).  With
# contacts spread roughly evenly along the chain that is held_window / L, so the ceiling
# carries peptide length in its DENOMINATOR.  Two consequences, both of which bit the first
# read of this run:
#
#   * a length trend in the ceiling is arithmetic.  (L - k) / L rises with L at constant k,
#     so "longer peptides retain more" can be true of the ratio while the geometric cost k
#     is flat.  It says nothing about closure being easier.
#   * a gap trend can be masked.  In this dataset length rises monotonically with gap
#     (len_median 6 -> 15 across the bins past 10 A), so a rising denominator cancels a
#     falling numerator and the ceiling looks flat when the cost is not.
#
# `k` -- the number of residues that must be RELEASED to close -- has no such denominator,
# so every gap/length claim below is made on k and the ratio is reported beside it only to
# show the distortion.
def released_cost(df: pd.DataFrame, chem: str = "mainchain") -> pd.DataFrame:
    """Per-row released-residue cost, measured directly and as the ratio implies it.

    `k_direct` is the real quantity: L minus the window the refinement actually held.
    `k_implied` is L * (1 - ceiling), i.e. what you would infer from the ratio if contacts
    were uniform along the chain.  Their difference is not noise -- it measures how far the
    uniform-contact assumption is from this data, which is the assumption every ratio-based
    conclusion rests on.
    """
    col = ceiling_col(df, chem)
    wcol = f"{chem}_refined_window_len"
    if col not in df or wcol not in df:
        return pd.DataFrame()
    d = df.dropna(subset=[col, wcol, "nc_gap_A", "peptide_length"]).copy()
    d["k_direct"] = d["peptide_length"] - d[wcol]
    d["k_implied"] = d["peptide_length"] * (1.0 - d[col])
    d["gap_bin"] = pd.cut(d["nc_gap_A"], GAP_BINS, labels=GAP_LABELS, right=False)
    d["len_bin"] = pd.cut(d["peptide_length"], LEN_BINS, labels=LEN_LABELS, right=True)
    return d


def released_cost_vs_gap(df: pd.DataFrame, chem: str = "mainchain") -> pd.DataFrame:
    """The gap signal with the length denominator taken out."""
    d = released_cost(df, chem)
    if d.empty:
        return pd.DataFrame()
    col = ceiling_col(df, chem)
    g = d.groupby("gap_bin", observed=True).agg(
        n=("k_direct", "size"), gap_median=("nc_gap_A", "median"),
        len_median=("peptide_length", "median"), ceiling_median=(col, "median"),
        k_direct_median=("k_direct", "median"), k_implied_median=("k_implied", "median"),
        held_median=(f"{chem}_refined_window_len", "median")).reset_index()
    return g


def released_cost_by_length(df: pd.DataFrame, chem: str = "mainchain") -> pd.DataFrame:
    """The length 'signal'.  If k is flat here the ceiling's length trend was arithmetic."""
    d = released_cost(df, chem)
    if d.empty:
        return pd.DataFrame()
    col = ceiling_col(df, chem)
    return d.groupby("len_bin", observed=True).agg(
        n=("k_direct", "size"), len_median=("peptide_length", "median"),
        gap_median=("nc_gap_A", "median"), ceiling_median=(col, "median"),
        k_direct_median=("k_direct", "median"),
        k_implied_median=("k_implied", "median")).reset_index()


def ceiling_gap_within_length(df: pd.DataFrame, chem: str = "mainchain",
                              value: str = "k_direct") -> pd.DataFrame:
    """Gap stratified WITHIN length bins -- gap and length are collinear here, so the
    marginal tables cannot separate them and neither conclusion is safe without this."""
    d = released_cost(df, chem)
    if d.empty:
        return pd.DataFrame()
    piv = d.pivot_table(index="len_bin", columns="gap_bin", values=value,
                        aggfunc="median", observed=True)
    cnt = d.pivot_table(index="len_bin", columns="gap_bin", values=value,
                        aggfunc="size", observed=True)
    # A median over 1-2 complexes is not a measurement; blank those cells rather than
    # letting them carry a trend line.
    piv = piv.where(cnt >= 5)
    out = piv.reset_index()
    out.columns = [str(c) for c in out.columns]
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without a scipy dependency: Pearson on ranks."""
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def joint_regression(df: pd.DataFrame, chem: str = "mainchain") -> pd.DataFrame:
    """Regress each response on gap and length, alone and jointly.

    Standardised coefficients (z-scored predictors and response), so `beta_gap` and
    `beta_len` are directly comparable and the marginal-vs-joint shift is the confound made
    numeric.  Reported for both the ratio and the released-residue cost: if a predictor's
    marginal coefficient collapses when the other is controlled, the marginal table was
    reading the other variable.
    """
    d = released_cost(df, chem)
    if d.empty or len(d) < 10:
        return pd.DataFrame()

    def fit(sub: pd.DataFrame, response: str, preds: list[str]) -> dict:
        y = sub[response].to_numpy(float)
        keep = np.isfinite(y)
        X = np.column_stack([sub[p].to_numpy(float) for p in preds])
        keep &= np.isfinite(X).all(axis=1)
        y, X = y[keep], X[keep]
        if len(y) < 10 or y.std() == 0:
            return {}
        z = lambda a: (a - a.mean()) / (a.std() if a.std() else 1.0)  # noqa: E731
        Xz = np.column_stack([z(X[:, i]) for i in range(X.shape[1])] + [np.ones(len(y))])
        beta, *_ = np.linalg.lstsq(Xz, z(y), rcond=None)
        pred = Xz @ beta
        ss = ((z(y) - pred) ** 2).sum()
        out = {"n": int(len(y)), "r2": float(1.0 - ss / ((z(y) - z(y).mean()) ** 2).sum())}
        for p, b in zip(preds, beta[:-1]):
            out[f"beta_{'gap' if 'gap' in p else 'len'}"] = float(b)
        return out

    col = ceiling_col(df, chem)
    rows = []
    for slice_name, sub in (("all", d), (">10 A gap", d[d["nc_gap_A"] >= 10.0])):
        for resp, rlabel in ((col, "ceiling (ratio)"), ("k_direct", "k released")):
            for preds, plabel in ((["nc_gap_A"], "gap only"),
                                  (["peptide_length"], "length only"),
                                  (["nc_gap_A", "peptide_length"], "gap + length")):
                r = fit(sub, resp, preds)
                if r:
                    rows.append({"slice": slice_name, "response": rlabel, "model": plabel,
                                 **r})
    reg = pd.DataFrame(rows)
    if not reg.empty:
        reg = reg[["slice", "response", "model", "n", "beta_gap", "beta_len", "r2"]]
    return reg


def collinearity_note(df: pd.DataFrame, chem: str = "mainchain") -> dict:
    """How badly gap and length are entangled -- the reason the marginal tables mislead."""
    d = released_cost(df, chem)
    if d.empty:
        return {}
    g, L = d["nc_gap_A"].to_numpy(float), d["peptide_length"].to_numpy(float)
    past10 = d["nc_gap_A"] >= 10.0
    # Variance inflation. With two predictors VIF = 1 / (1 - r^2) between them. Above ~10
    # the individual joint coefficients are numerically unstable -- they can flip sign and
    # inflate without the fit getting worse -- so they must not be read as effect sizes.
    r = float(np.corrcoef(g, L)[0, 1]) if len(g) > 2 else float("nan")
    vif = float(1.0 / (1.0 - r * r)) if math.isfinite(r) and abs(r) < 1 else float("inf")
    return {
        "pearson_gap_length_all": r,
        "vif_gap_length": vif,
        "spearman_gap_length_all": _spearman(g, L),
        "spearman_gap_length_past10": _spearman(g[past10.to_numpy()], L[past10.to_numpy()]),
        "k_direct_median": float(d["k_direct"].median()),
        "k_implied_median": float(d["k_implied"].median()),
        "k_uniform_contact_bias": float((d["k_implied"] - d["k_direct"]).median()),
    }


def mainchain_median_by_slice(df: pd.DataFrame, chem: str = "mainchain") -> pd.DataFrame:
    """Pooled AND per-set medians in one table, labelled.

    The pooled median is not any set's median and must not be quoted as one: pepbench is
    600 of the 660 rows, so the pooled number is essentially pepbench's while the lnr slice
    sits slightly above it.  A bracket bound quoted without its slice is unreproducible.
    """
    col = ceiling_col(df, chem)
    if col not in df:
        return pd.DataFrame()
    d = df.dropna(subset=[col])
    rows = [{"slice": "POOLED (all sets)", "n": int(len(d)),
             "refined_median": float(d[col].median()),
             "frac_of_rows": 1.0}]
    for label, grp in d.groupby("label"):
        rows.append({"slice": str(label), "n": int(len(grp)),
                     "refined_median": float(grp[col].median()),
                     "frac_of_rows": float(len(grp) / len(d))})
    return pd.DataFrame(rows)


def bridge_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, grp in df.groupby("label"):
        row = {"set": label, "n": len(grp),
               "nc_gap_median_A": float(grp["nc_gap_A"].median()),
               "bridge_span_median_A": float(grp["bridge_span_med_A"].median()),
               "bridge_span_min_median_A": float(grp["bridge_span_min_A"].median())}
        for chem in ("disulfide", "isopeptide"):
            c = f"{chem}_hostable"
            if c in grp:
                row[f"{chem}_hostable_frac"] = float(grp[c].mean())
                row[f"{chem}_n_sites_median"] = float(grp[f"n_pairs_in_{chem}_window"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def validator_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("validator_")]
    if not cols or "validator_status" not in df:
        return pd.DataFrame()
    d = df[df["validator_status"] == "ok"]
    rows = []
    for tag in ("inside", "outside"):
        a, t, u = (f"validator_{tag}_analytic", f"validator_{tag}_torsion",
                   f"validator_{tag}_under_licensed")
        if a not in d:
            continue
        rows.append({
            "window": tag, "n": len(d),
            "analytic_feasible": float(d[a].mean()),
            "torsion_closed": float(d[t].mean()),
            "agree": float((d[a] == d[t]).mean()),
            "analytic_over_licensed": float((d[a] & ~d[t]).mean()),
            "analytic_UNDER_licensed": float(d[u].mean()),
            "median_residual_A": float(d[f"validator_{tag}_residual_A"].median()),
        })
    return pd.DataFrame(rows)


# Which column says a ring of each chemistry actually BONDED.  Not
# `cyc/cyc_cb_window_success`, which saturates and proves nothing.
CLOSURE_COL = {"mainchain": "cyc/mainchain_cn_bond_success",
               "disulfide": "cyc/disulfide_bond_success",
               "isopeptide": "cyc/isopeptide_bond_success"}


def achieved_over_ceiling(ceil: pd.DataFrame, achieved_globs: list[str]
                          ) -> tuple[pd.DataFrame, dict]:
    """Rescale a measured contact retention by what was geometrically available.

    Two filters, both load-bearing:

      * the requested chemistry must have been produced at all, and
      * the ring must actually have BONDED.

    Without the second the comparison is not like-for-like: the ceiling is retention
    *subject to closure*, and an open ring is free to keep every contact it started with.
    That alone pushed the measured ratio above 1.0 on the smoke -- i.e. above a quantity
    that is supposed to be an upper bound.
    """
    if not achieved_globs:
        return pd.DataFrame(), {}
    ach = read_jsonl(achieved_globs)
    if ach.empty or "contact_retention" not in ach:
        return pd.DataFrame(), {}
    ach = ach[(ach.get("status") == "ok") & ach["contact_retention"].notna()]
    diag = {"n_rows_ok": int(len(ach))}
    if "requested_type_satisfied" in ach:
        # Abstained rows carry NaN, and NaN is truthy -- a naive filter scores a
        # 100%-abstention cell as 100% success.  Gate on the flag, not on the metric.
        ach = ach[ach["requested_type_satisfied"].fillna(0).astype(float) > 0]
    diag["n_type_satisfied"] = int(len(ach))

    if "cyc_type" in ach:
        closed = pd.Series(False, index=ach.index)
        for chem, col in CLOSURE_COL.items():
            if col in ach:
                closed |= (ach["cyc_type"] == chem) & (ach[col].fillna(0).astype(float) > 0)
        ach = ach[closed]
    diag["n_closed"] = int(len(ach))
    if ach.empty:
        return pd.DataFrame(), diag

    key = ceil.set_index("example_id")
    cols = {chem: ceiling_col(ceil, chem) for chem in CHEMISTRIES}
    rows = []
    for (eid, chem), grp in ach.groupby(["example_id", ach.get("cyc_type", "mainchain")]):
        if eid not in key.index:
            continue
        col = cols.get(chem)
        if col is None or col not in key.columns:
            continue
        ceiling = float(key.loc[eid, col])
        if not math.isfinite(ceiling) or ceiling <= 0:
            continue
        best = float(grp["contact_retention"].max())
        med = float(grp["contact_retention"].median())
        rows.append({"example_id": eid, "chemistry": chem, "n_attempts": len(grp),
                     "achieved_median": med, "achieved_best": best, "ceiling": ceiling,
                     "frac_of_ceiling_median": med / ceiling,
                     "frac_of_ceiling_best": best / ceiling,
                     # A true upper bound is never exceeded.  When this fires, the
                     # CONTIGUITY assumption is the suspect: the ceiling holds ONE
                     # contiguous window rigid, while a sampler may keep several
                     # non-adjacent stretches near-native.  Count it, never clip it.
                     "exceeds_ceiling": int(best > ceiling + 1e-9)})
    out = pd.DataFrame(rows)
    if not out.empty:
        diag["n_complexes_exceeding_ceiling"] = int(out["exceeds_ceiling"].sum())
        diag["frac_exceeding_ceiling"] = float(out["exceeds_ceiling"].mean())
    return out, diag


# ----------------------------------------------------------------------------- figures
def make_figures(ceil: pd.DataFrame, rescaled: pd.DataFrame, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    def style(ax, xlabel, ylabel, title):
        ax.set_facecolor(SURFACE)
        ax.figure.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)

    # 1. The core deliverable: how much interface survives closure, against terminal gap.
    if "mainchain_ceiling_strict" in ceil:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for chem in CHEMISTRIES:
            col = ceiling_col(ceil, chem)
            if col not in ceil:
                continue
            d = ceil.dropna(subset=[col, "nc_gap_A"])
            ax.scatter(d["nc_gap_A"], d[col], s=26, color=HUE[chem], alpha=0.55,
                       edgecolor=SURFACE, linewidth=0.8, label=chem, zorder=3)
            b = d.copy()
            b["bin"] = pd.cut(b["nc_gap_A"], GAP_BINS, labels=GAP_LABELS, right=False)
            m = b.groupby("bin", observed=True).agg(x=("nc_gap_A", "median"),
                                                    y=(col, "median")).dropna()
            ax.plot(m["x"], m["y"], color=HUE[chem], linewidth=2.0, zorder=4)
        ax.set_ylim(0, 1.02)
        style(ax, "terminal N(0)-C(L-1) gap  (A)", "max contact retention under closure",
              "Ceiling: what closure leaves of the native interface")
        ax.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9, loc="lower left")
        fig.tight_layout()
        p = out_dir / "ceiling_vs_terminal_gap.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p.name)

    # 2. Head-to-tail against bridged, the section-4.4 question, as a paired dot plot.
    have = [c for c in CHEMISTRIES if f"{c}_ceiling_strict" in ceil]
    if len(have) >= 2:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for k, chem in enumerate(have):
            v = ceil[ceiling_col(ceil, chem)].dropna()
            if v.empty:
                continue
            ax.scatter(v, np.full(len(v), k) + np.random.default_rng(k).normal(0, 0.06, len(v)),
                       s=22, color=HUE[chem], alpha=0.45, edgecolor=SURFACE, linewidth=0.7,
                       zorder=3)
            ax.plot([v.quantile(0.25), v.quantile(0.75)], [k, k], color=HUE[chem],
                    linewidth=3.0, solid_capstyle="round", zorder=4)
            ax.scatter([v.median()], [k], s=95, color=HUE[chem], edgecolor=SURFACE,
                       linewidth=1.6, zorder=5)
            ax.annotate(f"{v.median():.2f}", (v.median(), k), textcoords="offset points",
                        xytext=(0, 13), ha="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(have)))
        ax.set_yticklabels(have, color=INK)
        ax.set_xlim(0, 1.05)
        style(ax, "max contact retention under closure", "",
              "Bridged closure keeps more of the interface than head-to-tail")
        fig.tight_layout()
        p = out_dir / "ceiling_by_chemistry.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p.name)

    # 3. Section 3: the bridge span, which is the feature a bridged macrocycle actually
    #    has to satisfy -- the terminal gap measures the free tails, not the ring.
    if "bridge_span_min_A" in ceil:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for name, col, hue in (("terminal N-C gap", "nc_gap_A", HUE["mainchain"]),
                               ("best bridge span (CB-CB)", "bridge_span_min_A",
                                HUE["disulfide"])):
            v = np.sort(ceil[col].dropna().values)
            if not len(v):
                continue
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=hue, linewidth=2.0,
                    label=name, zorder=3)
        ax.axvspan(3.1, 7.8, color=HUE["isopeptide"], alpha=0.10, zorder=1)
        ax.annotate("bridge-compatible\nCB-CB window", (5.4, 0.06), ha="center",
                    color=INK_MUTED, fontsize=9)
        ax.set_ylim(0, 1.02)
        style(ax, "distance  (A)", "cumulative fraction of complexes",
              "Terminal gap is the wrong discriminator for a bridged ring")
        ax.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9, loc="lower right")
        fig.tight_layout()
        p = out_dir / "bridge_span_vs_terminal_gap.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p.name)

    # 4. The headline, when a measured retention is available to rescale.
    if not rescaled.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        d = rescaled.sort_values("ceiling").reset_index(drop=True)
        for k, r in d.iterrows():
            hue = HUE.get(r["chemistry"], HUE["other"])
            ax.plot([r["achieved_median"], r["ceiling"]], [k, k], color=GRID, linewidth=1.6,
                    zorder=2)
            ax.scatter([r["ceiling"]], [k], s=44, color=hue, alpha=0.35,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.scatter([r["achieved_median"]], [k], s=44, color=hue, edgecolor=SURFACE,
                       linewidth=0.8, zorder=4)
        ax.set_yticks([])
        ax.set_xlim(0, 1.02)
        style(ax, "contact retention", "complexes, sorted by ceiling",
              "Achieved against available: the gap that is actually addressable")
        ax.annotate("open = ceiling   filled = achieved", (0.02, len(d) * 0.96),
                    color=INK_MUTED, fontsize=9)
        fig.tight_layout()
        p = out_dir / "achieved_vs_ceiling.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p.name)
    return made


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ceiling-glob", nargs="*", default=[])
    ap.add_argument("--windows", type=Path, default=None)
    ap.add_argument("--timing", type=Path, default=None)
    ap.add_argument("--e-index", type=Path, default=None)
    ap.add_argument("--e-spatial-glob", nargs="*", default=[])
    ap.add_argument("--achieved-glob", nargs="*", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md: list[str] = ["# Milestone 1.5 -- geometric ceiling audit", ""]
    summary: dict[str, object] = {}

    ceil = read_jsonl(args.ceiling_glob) if args.ceiling_glob else pd.DataFrame()
    if not ceil.empty:
        ceil = ceil[ceil.get("status") == "ok"].reset_index(drop=True)

    # -- gate, not crash: a missing upstream stage must say so and leave the rest intact.
    if ceil.empty:
        md += ["> **Ceiling stage produced no rows.** Check the audit job's `.out` before",
               "> reading anything below as a result.", ""]
    else:
        summary["n_complexes"] = int(len(ceil))
        md += ["## 1. Ceiling by set and chemistry", "",
               "How much of the native CA-CA 10 A interface can survive ring closure, when a",
               "contiguous residue window is held at its bound position and the rest is",
               "released to close.", "",
               "**This is a BRACKET, not a number. Read `refined_median` and",
               "`permissive_median` together.**", "",
               "- `refined_median` is a **FLOOR**. It counts a contact only when its residue",
               "  is inside the held window, writing off every released residue -- but a",
               "  residue that moves 2 A does not lose a 10 A contact. Measured: 35% of",
               "  complexes beat it, 86% for mainchain. Do not quote it alone.",
               "- `permissive_median` is the **upper bound**: a released residue keeps a",
               "  contact whose partner is still within its reach. 0% exceedance.",
               "- `analytic_median` is the pre-refinement bound, kept only to show what the",
               "  cone-vs-sphere idealisation was worth. Never quote it.", "",
               "**Where the bracket is tight the chemistry is answered; where it is wide the",
               "ceiling question is open.** On the first full run bridged closure read as",
               "answered (1.000 both ends) and head-to-tail did not ([~0.59, 1.000]) -- but",
               "that bridged number is the FREE scan, which maximises over anchor pairs the",
               "generator cannot use. Read section 1b before quoting it.", "",
               "Both ends stay permissive about physics: no sterics, no Ramachandran, no",
               "receptor excluded volume. A bridged ceiling also assumes the anchor residues",
               "can carry the chemistry; `*_anchor_native_compatible` in the per-complex rows",
               "says whether they natively do.", "",
               md_table(ceiling_by_set(ceil)), ""]

        fvr = free_vs_reachable(ceil)
        if not fvr.empty:
            md += ["### 1b. Free bridge against the reachable one", "",
                   "The bridged rows above maximise over every (i, j) pair with |i-j| >= 3.",
                   "CPSea conditions on the chain TERMINI, so the only anchors it can place",
                   "a bridge between are (0, L-1). A ceiling taken over unreachable pairs is",
                   "not a target the model can be held to, so the `*_term` columns repeat the",
                   "scan with the anchors pinned there, same spec and same refinement.", "",
                   "`delta_median_paired` is the median per-complex drop, not the difference",
                   "of the two medians. Where it is near zero the free ceiling was reachable",
                   "anyway and section 1's bridged number stands; where it is large, the",
                   "bridged advantage over head-to-tail is partly an artifact of scanning",
                   "pairs the generator cannot reach.", "",
                   md_table(fvr, nd=3), ""]
            for r in fvr.itertuples(index=False):
                summary[f"{r.chemistry}_terminal_median"] = r.terminal_median
                summary[f"{r.chemistry}_free_minus_terminal_median"] = r.delta_median_paired

        msl = mainchain_median_by_slice(ceil)
        if not msl.empty:
            md += ["### Which slice a quoted median is", "",
                   "The pooled median is not any set's median. Quote the slice with the",
                   "number or it cannot be reproduced.", "",
                   md_table(msl), ""]
            summary["mainchain_refined_median_pooled"] = float(
                msl.loc[msl["slice"] == "POOLED (all sets)", "refined_median"].iloc[0])

        md += ["## 2. Ceiling against terminal gap (head-to-tail)", "",
               "**Read section 2b before drawing a gap or length conclusion from these two",
               "tables.** `ceiling` is a ratio with peptide length in its denominator, and in",
               "this dataset length rises monotonically with gap, so neither marginal table",
               "separates the two.", "",
               md_table(ceiling_vs_gap(ceil), nd=2), "",
               "### by peptide length", "", md_table(ceiling_by_length(ceil), nd=2), ""]

        # ---- 2b: the confound, made numeric.  This exists because the first read of the
        # marginal tables above produced two conclusions ("past 10 A the gap stops
        # mattering", "short peptides are the constrained case") that are artifacts of the
        # ratio's denominator, not findings about closure.
        rc_gap = released_cost_vs_gap(ceil)
        if not rc_gap.empty:
            coll = collinearity_note(ceil)
            md += ["## 2b. Released-residue cost -- the same data without the ratio", "",
                   "`ceiling_strict` = (contacts inside the held window) / (all contacts), so",
                   "under roughly uniform contacts it is `held / L`. Peptide length is in the",
                   "denominator. `k` is the number of residues that must be RELEASED to close",
                   "(`L - held`); it is the geometric cost and carries no denominator.", "",
                   f"- gap/length rank correlation: **{coll.get('spearman_gap_length_all', float('nan')):.2f}** "
                   f"over all rows, **{coll.get('spearman_gap_length_past10', float('nan')):.2f}** past 10 A. "
                   "Gap and length are entangled, so the marginal tables above are each",
                   "  partly reading the other variable.",
                   f"- `k_implied - k_direct` median: **{coll.get('k_uniform_contact_bias', float('nan')):+.2f}** "
                   "residues. This is the uniform-contact assumption's error, measured rather",
                   "  than assumed; near zero means the ratio's algebra holds on this data.",
                   "", md_table(rc_gap, nd=2), "",
                   "### by peptide length", "",
                   "If `k_direct_median` is flat down this table while `ceiling_median` climbs,",
                   "the ceiling's length trend is arithmetic -- the same closure cost spread",
                   "over a longer chain -- and *not* evidence that long peptides close more",
                   "cheaply.", "",
                   md_table(released_cost_by_length(ceil), nd=2), ""]

            piv_k = ceiling_gap_within_length(ceil, value="k_direct")
            piv_c = ceiling_gap_within_length(ceil, value=ceiling_col(ceil, "mainchain"))
            if not piv_k.empty:
                md += ["### Gap stratified within length bins", "",
                       "Rows = length bin, columns = gap bin. Cells with n < 5 are blank. This",
                       "is the table that separates the two effects; the marginal tables",
                       "cannot.", "",
                       "`k_direct` (residues released to close -- lower is better):", "",
                       md_table(piv_k, nd=2), "",
                       "`ceiling` (the ratio, for comparison):", "",
                       md_table(piv_c, nd=2), ""]

            reg = joint_regression(ceil)
            if not reg.empty:
                vif = coll.get("vif_gap_length", float("nan"))
                md += ["### Joint regression, standardised coefficients", "",
                       "Predictors and response z-scored, so `beta_gap` and `beta_len` are",
                       "comparable. Compare a predictor's `gap only` / `length only` row",
                       "against its `gap + length` row: a coefficient that collapses when the",
                       "other variable is controlled was reading that other variable.", "",
                       f"**Variance inflation: VIF = {vif:.1f}.** " + (
                           "Above 10, so the individual `gap + length` coefficients are "
                           "unstable -- they can flip sign and inflate without the fit "
                           "getting worse. Read the SIGN and the r2, and take effect sizes "
                           "from the stratified table above, not from these coefficients."
                           if vif > 10 else
                           "Below 10, so the joint coefficients are stable enough to read as "
                           "effect sizes."), "",
                       "The `k released` rows are the trustworthy ones: `k` has no length in",
                       "its denominator, so its coefficients mean what they say.", "",
                       md_table(reg, nd=3), ""]
                summary["joint_regression"] = reg.to_dict("records")
            summary["collinearity"] = coll

        md += ["## 3. Bridge-span profile", "",
               "The chain-terminal gap measures the free tails, not the ring: for a",
               "disulfide or isopeptide macrocycle the bonded atoms are interior side",
               "chains. `hostable_frac` is the fraction of peptides carrying at least one",
               "(i, j) pair already inside the measured bond window.", "",
               md_table(bridge_profile_table(ceil)), ""]

        vt = validator_table(ceil)
        if not vt.empty:
            md += ["## 4. Validator: analytic bound against a torsion-space solve", "",
                   "`analytic_UNDER_licensed` is the only fatal column. The bound is",
                   "permissive by construction, so over-licensing is expected and bounds the",
                   "ceiling's looseness; under-licensing would mean the ceiling is too low",
                   "and a real deficit could hide behind it.", "",
                   md_table(vt), ""]
            summary["validator_under_licensed"] = float(vt["analytic_UNDER_licensed"].max())

    resc, resc_diag = (achieved_over_ceiling(ceil, args.achieved_glob)
                       if not ceil.empty else (pd.DataFrame(), {}))
    if not resc.empty:
        summary["frac_of_ceiling_median"] = float(resc["frac_of_ceiling_median"].median())
        summary["achieved_ceiling_diagnostics"] = resc_diag
        exceed = resc_diag.get("frac_exceeding_ceiling", 0.0)
        md += ["## 5. Achieved / ceiling", "",
               "The number the deliverable metric should be rescaled to before anything is",
               "optimised against it.", "",
               f"Filtered to rows where the requested chemistry was produced AND the ring "
               f"actually bonded: {resc_diag.get('n_rows_ok', 0):,} ok -> "
               f"{resc_diag.get('n_type_satisfied', 0):,} right type -> "
               f"{resc_diag.get('n_closed', 0):,} closed. The ceiling is retention *subject "
               f"to closure*, so an open ring is not comparable to it.", ""]
        if exceed > 0:
            md += [f"> **{exceed:.0%} of complexes exceed `refined`.** That is not a sampler "
                   f"beating physics -- it means `refined` is not an upper bound. It counts "
                   f"a contact as kept only when its residue is INSIDE the held window, so "
                   f"every released residue is written off wholesale. A residue that moves "
                   f"2 A does not lose a 10 A CA-CA contact, so the write-off is simply "
                   f"wrong, and `refined` is a FLOOR on retention-under-closure.", "",
                   f"> The true bound is `permissive_median` in section 1. Where the two are "
                   f"far apart the ceiling question is **unanswered for that chemistry**, "
                   f"and `frac_of_ceiling` below should not be quoted. Closing that gap "
                   f"needs conformational sampling under the closure constraint, not a "
                   f"tighter bound -- which is a Milestone-2 decision, not a patch.", ""]
        md += [
               md_table(resc.groupby("chemistry").agg(
                   n=("example_id", "size"),
                   achieved_median=("achieved_median", "median"),
                   ceiling_median=("ceiling", "median"),
                   frac_of_ceiling_median=("frac_of_ceiling_median", "median"),
                   frac_of_ceiling_best=("frac_of_ceiling_best", "median"),
                   n_exceeding=("exceeds_ceiling", "sum"),
               ).reset_index()), ""]
        resc.to_csv(args.out_dir / "achieved_over_ceiling.csv", index=False)

    if args.windows and args.windows.exists():
        w = json.loads(args.windows.read_text())
        rows = [{"type": t, **{k: v for k, v in win.items()}} for t, win in w["windows"].items()]
        rows = [{**r, "n_observed": w["observed"].get(r["type"], {}).get("n")} for r in rows]
        md += ["## 6. Bridge windows, measured from CPSea natives", "",
               f"From {w['n_labelled']} CONECT-resolved linkages "
               f"({w['n_sampled']} sampled, {w['n_unreadable']} unreadable). "
               f"[p1, p99] plus {w['pad_A']} A.", "",
               md_table(pd.DataFrame(rows), nd=2), ""]
        summary["bridge_windows"] = w["windows"]

    t = None
    if args.timing and args.timing.exists():
        t = json.loads(args.timing.read_text())
    elif args.timing:
        # A time-limited timing job leaves rows but no summary. Recomputing from the rows
        # is strictly better than omitting the section: the medians are over whatever
        # completed, and `partial` says so.
        rows_path = args.timing.parent / (args.timing.name.replace(".summary.json", ".jsonl"))
        rows = read_jsonl([str(rows_path)]) if rows_path.exists() else pd.DataFrame()
        if not rows.empty:
            rows = rows[rows.get("status") == "ok"]
        if not rows.empty:
            t = {"n_complexes": int(len(rows)), "partial": True,
                 "n_replicas_each": int(rows["n_replicas_each"].iloc[0])
                 if "n_replicas_each" in rows else -1,
                 "setup_median_s": float(rows["t_setup_s"].median()),
                 "marginal_median_s": float(rows["t_replica_median_s"].median()),
                 "ratio_median": float(rows["setup_over_marginal"].median()),
                 "s_per_decoy_at_8": float(rows["s_per_decoy_at_8"].median()),
                 "s_per_decoy_at_30": float(rows["s_per_decoy_at_30"].median()),
                 "s_per_decoy_at_100": float(rows["s_per_decoy_at_100"].median())}
            print(f"[report] no timing summary; recomputed from {len(rows)} rows")
    if t is not None:
        n_done, n_want = t.get("n_complexes", 0), t.get("n_complexes_requested", 0)
        partial = (f" **(PARTIAL -- {n_done} of {n_want} complexes; check the job log for "
                   f"whether it was killed or a complex failed)**") if t.get("partial") else ""
        md += [f"## 7. Replica economics{partial}", "",
               f"n = {t['n_complexes']} complexes x {t.get('n_replicas_each', '?')} replicas.", "",
               f"- context setup: **{t['setup_median_s']:.1f} s** per complex",
               f"- marginal per replica: **{t['marginal_median_s']:.2f} s**",
               f"- setup / marginal: **{t['ratio_median']:.3f}x** "
               f"({'setup dominates' if t['ratio_median'] > 1 else 'the minimization dominates'})", "",
               f"Amortised wall-clock per decoy: {t['s_per_decoy_at_8']:.1f} s at 8 decoys, "
               f"{t['s_per_decoy_at_30']:.1f} s at 30, {t['s_per_decoy_at_100']:.1f} s at 100.", ""]
        summary["timing"] = t

    if args.e_index and args.e_index.exists():
        idx = json.loads(args.e_index.read_text())
        md += ["## 8. Deliverable E on CPSea_full", "",
               md_table(pd.DataFrame([{k: v for k, v in idx.items()
                                       if isinstance(v, (int, float))}]), nd=4), ""]
        sp = read_jsonl(args.e_spatial_glob) if args.e_spatial_glob else pd.DataFrame()
        if not sp.empty:
            sp = sp[sp.get("status") == "ok"]
        if not sp.empty and args.config:
            from script_utils import m15_config
            ecfg = m15_config.load(args.config)["deliverable_e"]
            jac, sep = float(ecfg["max_contact_jaccard"]), float(ecfg["min_centroid_separation_A"])
            keep = (sp["contact_jaccard"] <= jac) & (
                sp["centroid_separation_A"].isna() | (sp["centroid_separation_A"] >= sep))
            p_hat = float(keep.mean())
            n = len(sp)
            se = math.sqrt(max(p_hat * (1 - p_hat), 1e-12) / n)
            pop = idx.get(f"n_disjoint_and_ge{ecfg['min_sequence_separation']}_apart", 0)
            md += [f"Spatial gate: Jaccard <= {jac}, centroid separation >= {sep} A.", "",
                   f"- scored pairs: **{n:,}** (of {pop:,} candidates, "
                   f"{100 * n / pop if pop else float('nan'):.1f}% sampled)",
                   f"- surviving fraction: **{p_hat:.3f}** "
                   f"[{max(0, p_hat - 1.96 * se):.3f}, {min(1, p_hat + 1.96 * se):.3f}]",
                   f"- projected surviving pairs on CPSea_full: **{int(p_hat * pop):,}**",
                   f"- median contact Jaccard: {sp['contact_jaccard'].median():.3f}",
                   f"- median superposition RMSD: "
                   f"{sp['superpose_rmsd_A'].median():.3f} A (a large value invalidates the "
                   f"separation column, not the Jaccard one)", ""]
            summary["deliverable_e_surviving_frac"] = p_hat
            summary["deliverable_e_projected_pairs"] = int(p_hat * pop)

    figs = make_figures(ceil, resc, args.out_dir / "figures") if not ceil.empty else []
    if figs:
        md += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in figs] + [""]

    (args.out_dir / "M15_CEILING_REPORT.md").write_text("\n".join(md))
    (args.out_dir / "m15_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not ceil.empty:
        ceil.to_csv(args.out_dir / "ceiling_rows.csv", index=False)
    print(f"[report] -> {args.out_dir / 'M15_CEILING_REPORT.md'}  ({len(figs)} figures)")


if __name__ == "__main__":
    main()
