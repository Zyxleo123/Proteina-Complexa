"""Aggregate the track-asymmetric SDEdit sweep into the preservation/closure trade-off.

The whole question is a trade-off, so a single scalar cannot answer it. Every edit has a
closure outcome and a preservation cost, and the useful output is the frontier between
them across the (t_ca, t_lat) grid:

    closure          did the requested ring actually form (bond within tolerance)
    ca_rmsd_to_input pose displacement, receptor frame, no superposition
    n_substitutions  sequence cost (0 by construction at t_lat = 1.0)
    contact_retention fraction of input interface contacts kept

CPU only; reads saved rows, so the figures re-render without re-running the sampler.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# cyclic_geometry_metrics already emits a per-chemistry bond-success rate. Each row is scored
# against the chemistry it REQUESTED -- a disulfide arm must not be graded on the mainchain key.
#
# Closure is reported TWO WAYS, side by side and INDEPENDENT of each other -- never chained,
# never one gating the other. They answer different questions and disagree informatively:
#
#   BOND  the anchor-atom distance in its two-sided window (C-N / SG-SG / NZ-CG). Strict, and
#         chemistry-SPECIFIC: only defined where the sampled sequence carries the required
#         anchors, which is why it abstains on most frozen-sequence rows.
#   CB    Endpoint CB-CB in [3.0, 8.0] A, computed here from `term_cb_dist_A` -- the distance
#         between the FIRST and LAST valid residue, read off the structure. Chemistry-agnostic
#         and anchor-free, so it stays defined where BOND abstains, and it measures ring SHAPE
#         rather than bond formation.
#
#         It is NOT `cyc/cyc_cb_window_success`, despite the name. That field is keyed to the
#         model's PREDICTED edge (i, j), which is -1 on an abstention; the distance is then
#         written as a sentinel 0.0, which falls outside the window and is scored as an OPEN
#         ring. On the full-chain guided sweep that marked 100% of abstained rows as failures
#         (1514/1514 disulfide, 1349/1349 isopeptide) -- rebilling a sequence-budget failure as
#         a geometry failure, which is the exact pooling `attach_closure` exists to prevent.
#
#         CB is a near-superset of BOND, not an independent draw: measured on pocket14,
#         P(CB | bond) = 0.996-1.000 while P(CB | not bond) = 0.48-0.74. The GAP between the two
#         columns is the useful readout -- it is the population where the ring shape formed but
#         no bond did, which is dominated by anchor IDENTITY (wrong residue at i/j), not by
#         gross geometry. Report both; never substitute CB for BOND, least of all on mainchain,
#         where the peptide bond IS the definition of head-to-tail cyclization.
#
# An earlier note here claimed the CB window "saturates and therefore proves nothing". That is
# true only on in-distribution CPSea targets (0.993-0.998); on the LNR arms it ranges 0.66-0.97
# and tracks the pocket-crop fix as clearly as BOND does, so it is reported, not dropped.
CLOSURE_KEY_FOR_TYPE = {
    "mainchain": "cyc/mainchain_cn_bond_success",
    "disulfide": "cyc/disulfide_bond_success",
    "isopeptide": "cyc/isopeptide_bond_success",
}
# The count backing each rate. The per-type scorable subsets differ by construction (a
# disulfide is only scorable where the SAMPLED sequence carries two CYS), so a rate without
# its count cannot distinguish "good at disulfides" from "scored on three lucky samples".
# The chemistry-agnostic criterion, reported alongside (never instead of) the bond keys.
# Anchor-free endpoint geometry, written by `cb_closure_geometry` in scripts/sdedit_cyclize.py.
# Runs predating that block carry neither field; see the fallback in `attach_closure`.
CB_DIST_KEY = "term_cb_dist_A"
CB_WINDOW_A = (3.0, 8.0)
# Legacy, predicted-edge-keyed fields. Only used as a fallback for old runs, and only after the
# abstention sentinel is stripped -- see the module docstring for why they are not the default.
CB_LEGACY_DIST_KEY = "cyc/cyc_cb_dist_pred_A"

COUNT_KEY_FOR_TYPE = {
    "mainchain": "cyc/n_valid_mainchain",
    "disulfide": "cyc/n_valid_disulfide",
    "isopeptide": "cyc/n_valid_isopeptide",
}

# Similarity-guidance settings. They only enter the grouping when they actually VARY, so an
# unguided sweep summarises exactly as it did before guidance existed.
GUID_COLS = ["guidance_w", "guidance_loss", "guidance_schedule", "guidance_exclude_termini",
             "guidance_mode"]


def guidance_arms(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fills the unguided arm's null settings and returns the guidance columns that vary."""
    if "guidance_w" not in df.columns:
        return df, []
    df = df.copy()
    df["guidance_w"] = pd.to_numeric(df["guidance_w"], errors="coerce").fillna(0.0)
    for c in GUID_COLS[1:]:
        if c in df.columns:
            # The unguided rows carry None by construction; "off" keeps them a visible group
            # instead of being dropped by groupby.
            df[c] = df[c].where(df["guidance_w"] > 0, "off").fillna("off").astype(str)
    df["arm"] = df.apply(
        lambda r: "unguided" if r["guidance_w"] <= 0 else
        f"w={r['guidance_w']:g} {r.get('guidance_loss', '')} "
        f"{r.get('guidance_schedule', '')} k={r.get('guidance_exclude_termini', '')} "
        f"{r.get('guidance_mode', '')}", axis=1)
    varying = [c for c in GUID_COLS if c in df.columns and df[c].nunique(dropna=False) > 1]
    return df, varying


def load_rows(paths: list[Path]) -> pd.DataFrame:
    """Loads shard JSONLs, REFUSING corrupt lines instead of skipping them."""
    records = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"FATAL: results file missing: {path}")
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            if "\x00" in line:
                raise SystemExit(f"FATAL: NUL byte in {path}:{lineno} -- file is corrupt.")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"FATAL: unparseable row {path}:{lineno}: {exc}")
    if not records:
        raise SystemExit("FATAL: no rows loaded.")
    return pd.DataFrame(records)


def attach_closure(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `closed`, `n_scorable` and `abstained`, from the key matching the row's REQUESTED type.

    A missing closure value has TWO causes that must not be pooled:

      abstained   the head emitted a null edge (`pred_cyc_type == -1`), because the sampled
                  SEQUENCE admitted no candidate anchor pair of the requested chemistry.
                  A sequence-budget failure, upstream of geometry -- the ring was never
                  attempted, so it is not evidence about the model's geometry.
      unscorable  an edge was predicted but the per-chemistry bond key is absent.

    Pooling them into one NaN is what let "disulfide closure is unmeasured" read as a
    geometry result when the head had in fact abstained on 59/59 edits.
    """
    closed, scorable = [], []
    for _, r in df.iterrows():
        ck = CLOSURE_KEY_FOR_TYPE.get(r.get("cyc_type"))
        nk = COUNT_KEY_FOR_TYPE.get(r.get("cyc_type"))
        closed.append(float(r[ck]) if ck and ck in df.columns and pd.notna(r.get(ck)) else np.nan)
        scorable.append(float(r[nk]) if nk and nk in df.columns and pd.notna(r.get(nk)) else np.nan)
    df = df.copy()
    df["closed"] = closed
    df["n_scorable"] = scorable
    # The second criterion, computed in parallel over the same rows and never gated on the
    # first. It is defined on ABSTAINED rows too -- that is the whole point of having an
    # anchor-free column, and those rows carry real information (on pocket14 only 6-13% of
    # abstained rows have their endpoints in window, i.e. an abstention usually is an open
    # ring). Its denominator is therefore LARGER than the bond-scorable one, which is why
    # `n_scorable_cb` is aggregated separately below; never read the two means against a
    # single n.
    if CB_DIST_KEY in df.columns:
        cb_dist = pd.to_numeric(df[CB_DIST_KEY], errors="coerce")
    elif CB_LEGACY_DIST_KEY in df.columns:
        # An old run, before the anchor-free block existed. Its only CB distance is keyed to the
        # predicted edge, so blank the abstention sentinel (an exact 0.0, which no real CB pair
        # can take) rather than let it read as an open ring. CB is then reported on that run's
        # ATTEMPTED rows only, which is a smaller claim -- and the honest one.
        cb_dist = pd.to_numeric(df[CB_LEGACY_DIST_KEY], errors="coerce")
        cb_dist = cb_dist.where(cb_dist > 0)
    else:
        cb_dist = pd.Series(np.nan, index=df.index)
    lo, hi = CB_WINDOW_A
    df["cb_dist_A"] = cb_dist
    df["closed_cb"] = np.where(
        cb_dist.notna(), ((cb_dist >= lo) & (cb_dist <= hi)).astype(float), np.nan
    )
    df["abstained"] = (df["pred_cyc_type"] < 0).astype(float) if "pred_cyc_type" in df else np.nan
    return df


def _passk_estimator(n: int, c: int, k: int) -> float:
    """Chen et al. unbiased pass@k: P(at least one of k draws passes) from c/n observed.

    n = attempts (seeds) for this (example, cell), c = how many closed, k = draw budget.
    Exact expectation over the C(n,k) subsets, not the plug-in 1-(1-c/n)^k, so it is right
    for the small n we can afford here. Requires k <= n; the caller skips examples with n<k.
    """
    if c >= n:
        return 1.0
    if c <= 0:
        return 0.0
    # 1 - C(n-c, k)/C(n, k): probability a size-k subset misses every passing draw.
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k(df: pd.DataFrame, group_cols: list[str], out_dir: Path) -> None:
    """pass@1 and pass@k per (cyc_type, t_ca, t_lat), for both the BOND and CB criteria.

    A "pass" is a CLOSED ring, abstention counted as failure (fillna 0) -- pass@k is an
    end-to-end usefulness number, so an edit that never placed an anchor is a miss, not a
    hole in the denominator. Seeds are the k budget: for each input peptide in a cell we have
    up to len(SEEDS) attempts, and pass@k averages the unbiased per-peptide estimator over
    peptides. pass@1 is exactly the per-attempt closure rate; pass@{max seeds} is "did this
    peptide ever close in the seeds we drew". Reported for every k in 1..max_seeds so the
    sample-efficiency curve is visible, not just its endpoints.
    """
    # The cell is the full grouping, INCLUDING any varying guidance columns -- otherwise a
    # multi-arm run pools its arms into one cell and reads them as extra "seeds", so the
    # unguided/identity/dps arms would be indistinguishable and their seeds conflated. On an
    # unguided sweep group_cols has no guidance columns, so this is exactly (cyc, t_ca, t_lat).
    cell_cols = [c for c in group_cols if c in df.columns]
    if not any(c in cell_cols for c in ("cyc_type", "t_ca_start", "t_lat_start")) \
            or "example_id" not in df.columns:
        return
    crit = [("bond", "closed"), ("cb", "closed_cb")]
    crit = [(name, col) for name, col in crit if col in df.columns]
    if not crit:
        return

    # k budget is capped by the smallest per-(example,cell) attempt count actually present,
    # so we never ask for pass@k where some peptide has fewer than k seeds.
    per_ex = df.groupby(cell_cols + ["example_id"]).size()
    max_k = int(per_ex.min()) if len(per_ex) else 1
    max_k = max(1, min(max_k, 8))
    ks = list(range(1, max_k + 1))

    rows = []
    for cell, g in df.groupby(cell_cols, dropna=False):
        cell = cell if isinstance(cell, tuple) else (cell,)
        rec = dict(zip(cell_cols, cell))
        rec["n_examples"] = int(g["example_id"].nunique())
        rec["seeds_per_example"] = float(g.groupby("example_id").size().median())
        for name, col in crit:
            closed = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
            by_ex = g.assign(_c=closed).groupby("example_id")["_c"].agg(["size", "sum"])
            for k in ks:
                elig = by_ex[by_ex["size"] >= k]
                if len(elig) == 0:
                    rec[f"pass@{k}_{name}"] = float("nan")
                    continue
                vals = [_passk_estimator(int(r["size"]), int(r["sum"]), k)
                        for _, r in elig.iterrows()]
                rec[f"pass@{k}_{name}"] = float(np.mean(vals))
        rows.append(rec)

    table = pd.DataFrame(rows).sort_values(cell_cols)
    table.to_csv(out_dir / "sdedit_passk.csv", index=False)
    print(f"\npass@k (k budget = seeds; abstention counted as a miss), max k = {max_k}:")
    print(table.round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(args.results)
    n_failed = int((df.get("status", "ok") == "failed").sum()) if "status" in df else 0
    if n_failed:
        print(f"WARNING: {n_failed} edits failed:")
        for _, r in df[df.status == "failed"].head(10).iterrows():
            print(f"    {r.get('run_key')}: {r.get('error')}")
    df = df[df.get("status", "ok") == "ok"].copy()
    print(f"{len(df)} successful edits over "
          f"{df['example_id'].nunique()} peptides, {df['cyc_type'].nunique()} chemistries")

    df, varying_guid = guidance_arms(df)
    if varying_guid:
        print(f"\nguidance arms present; grouping also by {varying_guid}")
        if "guid_ca_disp_A" in df.columns:
            # An arm whose guidance never moved anything is a plumbing failure that otherwise
            # reads as "guidance did not help".
            push = df.groupby("arm")["guid_ca_disp_A"].median()
            print("median CA displacement added by guidance, per arm (A):")
            print(push.round(3).to_string())
            inert = [a for a, v in push.items() if a != "unguided" and (pd.isna(v) or v <= 1e-6)]
            if inert:
                print(f"WARNING: guidance never moved anything in arm(s) {inert} -- treat their "
                      f"numbers as UNGUIDED, not as a null result.")
        if "guid_bond_valid_frac" in df.columns:
            # For a CLOSURE-guided run this is the first thing to read. The bond-distance term
            # is gated on the anchor atoms existing in the decoded structure, so on an
            # abstaining edit (no cysteine -> no SG) its gradient is identically zero. An arm
            # near 0 here did not fail to help: it was never applied.
            bv = df.groupby("arm")["guid_bond_valid_frac"].mean()
            if bv.notna().any():
                print("\nfraction of guided steps where the bond term had its anchor ATOMS, "
                      "per arm:")
                print(bv.round(3).to_string())
                dead = [a for a, v in bv.items()
                        if a != "unguided" and pd.notna(v) and v <= 0.01]
                if dead:
                    print(f"NOTE: the bond term never had anchor atoms in arm(s) {dead}. That "
                          "is the expected result for a bond-only arm on the abstaining "
                          "population, and it means those rows measure the OTHER terms only.")

    df = attach_closure(df)
    if df["closed"].notna().sum() == 0:
        print("WARNING: no per-chemistry closure key present; preservation metrics only.")
    else:
        diag = df.groupby("cyc_type").apply(
            lambda g: pd.Series({
                "n": len(g),
                "abstained": float(g["abstained"].mean()) if "abstained" in g else np.nan,
                "unscorable": float(g["closed"].isna().mean()),
                "closure_given_attempted": float(g.loc[g["abstained"] == 0, "closed"].mean())
                if "abstained" in g and (g["abstained"] == 0).any() else np.nan,
            }), include_groups=False)
        print("\nwhy closure is missing, by chemistry -- `abstained` is the head emitting a null "
              "edge because\nthe sampled SEQUENCE had no valid anchor pair (a sequence-budget "
              "failure, NOT a geometry\nresult); `closure_given_attempted` conditions on the "
              "rows where a ring was actually tried:")
        print(diag.round(3).to_string())

    group_cols = ["cyc_type", "t_ca_start", "t_lat_start"] + varying_guid
    agg: dict[str, str | list] = {
        "ca_rmsd_to_input_A": "median",
        "n_substitutions": "median",
        "contact_retention": "median",
        "example_id": "count",
    }
    if "closed" in df:
        agg["closed"] = "mean"
    # The CB criterion is aggregated independently of `closed`: its mean is over the rows
    # where CB is defined, which is a DIFFERENT (larger) set than the bond-scorable rows.
    # Reporting both means side by side is the point -- do not combine them.
    if "closed_cb" in df:
        agg["closed_cb"] = "mean"
    if "cb_dist_A" in df:
        agg["cb_dist_A"] = "median"
    if "abstained" in df:
        agg["abstained"] = "mean"
    if "n_scorable" in df:
        agg["n_scorable"] = "sum"
    if "closed_cb" in df:
        df["n_scorable_cb"] = df["closed_cb"].notna().astype(float)
        agg["n_scorable_cb"] = "sum"
    if "requested_type_satisfied" in df:
        agg["requested_type_satisfied"] = "mean"
    if "guid_ca_disp_A" in df:
        agg["guid_ca_disp_A"] = "median"
    agg = {k: v for k, v in agg.items() if k in df.columns or k == "example_id"}

    table = df.groupby(group_cols, dropna=False).agg(agg).rename(columns={"example_id": "n"})
    table = table.reset_index().sort_values(group_cols)
    table.to_csv(args.out_dir / "sdedit_grid.csv", index=False)
    print("\n" + table.to_string(index=False))

    # The verdict table for a closure-guided run on the abstaining cells: what fraction of
    # edits got a ring ATTEMPTED at all, per arm. Closure-given-attempted is reported beside
    # it because an arm that raises attempts while lowering closure has moved the population,
    # not improved it.
    if varying_guid and "abstained" in df.columns:
        rescue = df.groupby(["cyc_type", "arm"], dropna=False).apply(
            lambda g: pd.Series({
                "n": len(g),
                "attempted": float(1.0 - g["abstained"].mean()),
                "closure_given_attempted": float(g.loc[g["abstained"] == 0, "closed"].mean())
                if (g["abstained"] == 0).any() else np.nan,
                # Abstention counted as failure -- the number that decides whether the arm
                # is worth anything end to end.
                "closure_unconditional": float(g["closed"].fillna(0.0).mean()),
            }), include_groups=False).reset_index()
        rescue.to_csv(args.out_dir / "sdedit_abstention_rescue.csv", index=False)
        print("\nabstention rescue, per chemistry x arm:")
        print(rescue.round(3).to_string(index=False))

    pass_at_k(df, group_cols, args.out_dir)

    df.to_csv(args.out_dir / "sdedit_rows.csv", index=False)
    (args.out_dir / "sdedit_summary.json").write_text(json.dumps({
        "n_edits": int(len(df)),
        "n_failed": n_failed,
        "n_peptides": int(df["example_id"].nunique()),
        "closure_keys": CLOSURE_KEY_FOR_TYPE,
        "grid": json.loads(table.to_json(orient="records")),
    }, indent=2, default=str))

    # --- trade-off figure: preservation cost on x, closure on y, one panel per chemistry
    chems = sorted(df["cyc_type"].unique())
    fig, axes = plt.subplots(1, max(len(chems), 1), figsize=(5.2 * max(len(chems), 1), 4.4), squeeze=False)
    for ax, chem in zip(axes[0], chems):
        sub = table[table.cyc_type == chem]
        for t_lat, grp in sub.groupby("t_lat_start"):
            grp = grp.sort_values("t_ca_start")
            y = grp["closed"] if "closed" in grp else np.full(len(grp), np.nan)
            ax.plot(grp["ca_rmsd_to_input_A"], y, "o-", label=f"t_lat={t_lat}")
            for _, r in grp.iterrows():
                ax.annotate(f"{r['t_ca_start']:.1f}", (r["ca_rmsd_to_input_A"],
                            r["closed"] if "closed" in r else np.nan), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("median CA displacement from input (A)")
        ax.set_ylabel("closure rate")
        ax.set_title(f"{chem}\n(labels = t_ca start)")
        ax.legend(fontsize=8)
    fig.suptitle("Track-asymmetric SDEdit: closure vs preservation", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out_dir / "sdedit_tradeoff.png", dpi=150, bbox_inches="tight")
    print(f"\nfigure: {args.out_dir / 'sdedit_tradeoff.png'}")

    # --- the guidance question, drawn as the question: does the guided frontier sit ABOVE
    # the unguided one, or does it just slide along it? Sliding along means guidance is a
    # reparameterization of t_ca and buys nothing.
    if varying_guid and "arm" in df.columns and df["arm"].nunique() > 1:
        arm_table = df.groupby(["cyc_type", "arm", "t_ca_start"], dropna=False).agg(
            closed=("closed", "mean"),
            ca_rmsd=("ca_rmsd_to_input_A", "median"),
            retention=("contact_retention", "median"),
            n=("example_id", "count"),
        ).reset_index()
        arm_table.to_csv(args.out_dir / "sdedit_guidance_arms.csv", index=False)
        for xcol, xlabel, fname in (
            ("ca_rmsd", "median CA displacement from input (A)", "sdedit_guidance_rmsd.png"),
            ("retention", "median interface contact retention", "sdedit_guidance_retention.png"),
        ):
            fig2, axes2 = plt.subplots(1, max(len(chems), 1),
                                       figsize=(5.4 * max(len(chems), 1), 4.6), squeeze=False)
            for ax, chem in zip(axes2[0], chems):
                sub2 = arm_table[arm_table.cyc_type == chem]
                for arm, grp in sub2.groupby("arm"):
                    grp = grp.sort_values("t_ca_start")
                    style = "ko--" if arm == "unguided" else "o-"
                    ax.plot(grp[xcol], grp["closed"], style, label=arm, lw=2 if arm == "unguided" else 1.4)
                    for _, r in grp.iterrows():
                        ax.annotate(f"{r['t_ca_start']:.1f}", (r[xcol], r["closed"]), fontsize=7,
                                    xytext=(3, 3), textcoords="offset points")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("closure rate")
                ax.set_title(f"{chem}\n(labels = t_ca start; dashed black = unguided)")
                ax.legend(fontsize=7)
            fig2.suptitle("Similarity guidance: does it beat the SDEdit frontier or slide along it?",
                          y=1.02)
            fig2.tight_layout()
            fig2.savefig(args.out_dir / fname, dpi=150, bbox_inches="tight")
            print(f"figure: {args.out_dir / fname}")


if __name__ == "__main__":
    main()
