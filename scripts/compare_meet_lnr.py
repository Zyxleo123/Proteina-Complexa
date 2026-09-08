"""Compare de novo ring closure on MEET targets vs LNR targets, matched on length and gap.

The question
------------
De novo (unconditional) generation closes ~45-52% of requested rings on the LNR targets
(evaluation_results/uncondgen_20260902_151235) against ~90% reported natively. Two
explanations are confounded in that number:

    target OOD          LNR receptors are crystal complexes; CPSea trained on AFDB pockets.
    geometric demand    LNR peptides sit in pockets that hold the termini far apart, so the
                        ring the model is asked for requires a large displacement.

MEET separates them. Its targets come from the SAME 8.64M AFDB domains CPSea used, so if
closure on MEET is high while LNR stays low AT MATCHED length and N-C gap, the gap is about
the target distribution. If the two agree once matched, it was the geometry all along.

Definitions (identical on both sides -- both metadata files are written by the same code
paths, see scripts/build_meet_metadata.py and scripts/build_lnr_metadata.py):

    input_nc_gap_angstrom   first residue's backbone N to last residue's carbonyl C, in the
                            NATIVE input peptide. At the de novo corner the model never sees
                            that peptide; the gap is a property of the POCKET -- how far
                            apart it holds a bound peptide's termini.
    input_peptide_length    number of real residues (MEET's ACE/NME caps are stripped).

Rows are graded per REQUESTED chemistry and abstentions are separated from failures, by
reusing summarize_sdedit_sweep's own attach_closure -- an abstained row (no anchor pair in
the sampled sequence) is NOT a geometry failure and must not be pooled with one.

CPU only; reads saved JSONL rows, so every table and figure re-renders without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_sdedit_sweep import attach_closure, load_rows  # noqa: E402

GAP_BINS = [0.0, 10.0, 15.0, np.inf]
GAP_LABELS = ["<=10A", "10-15A", ">15A"]
# MEET stops at 13 residues, so anything above it cannot be length-matched and is reported
# separately rather than silently dragging the LNR mean.
LEN_BINS = [4.5, 7.5, 10.5, 13.5, np.inf]
LEN_LABELS = ["5-7", "8-10", "11-13", "14+"]


def load_arm(results_dir: Path, label: str) -> pd.DataFrame:
    shards = sorted(results_dir.glob("edits_shard*.jsonl"))
    if not shards:
        raise SystemExit(f"FATAL: no edits_shard*.jsonl under {results_dir}")
    df = load_rows(shards)
    n_failed = int((df.get("status", "ok") == "failed").sum()) if "status" in df else 0
    df = df[df.get("status", "ok") == "ok"].copy()
    df = attach_closure(df)
    df["arm"] = label
    df["n_failed_edits"] = n_failed
    for c in ("t_ca_start", "t_lat_start"):
        if c in df.columns:
            bad = df[pd.to_numeric(df[c], errors="coerce").fillna(-1) != 0.0]
            if len(bad):
                raise SystemExit(
                    f"FATAL: {results_dir} is not a de novo run -- {len(bad)} rows have "
                    f"{c} != 0. Comparing an edit run against a de novo run is meaningless.")
    # A run read before it finishes is fine -- the sampler completes each target's whole
    # (chemistry x seed) block before moving to the next, so any prefix is chemistry-balanced
    # and the targets it covers are unordered w.r.t. length and N-C gap. But the reader must
    # be TOLD it is looking at a preview, or a partial arm silently reads as a final result.
    cells = df.groupby("example_id").size()
    full = int(cells.max()) if len(cells) else 0
    n_complete = int((cells == full).sum())
    n_partial = int((cells < full).sum())
    print(f"[{label}] {len(df)} ok edits, {n_failed} failed, {len(cells)} targets "
          f"({n_complete} complete at {full} edits each, {n_partial} still in flight), "
          f"from {results_dir}")
    if n_partial:
        print(f"[{label}] PARTIAL RUN -- treat every number below as a preview.")
    df["arm_n_targets_seen"] = len(cells)
    df["arm_n_targets_complete"] = n_complete
    return df


def rate(g: pd.DataFrame) -> pd.Series:
    """Closure conditioned on ATTEMPTED, plus the abstention rate that makes it readable.

    `closed` is NaN on abstained rows and NaN is truthy under a naive filter, which has
    previously scored a 100%-abstention cell as 100% closure. Gate on `abstained` explicitly.
    """
    att = g[g["abstained"] == 0]
    scored = att["closed"].dropna()
    return pd.Series({
        "n_edits": len(g),
        "n_targets": g["example_id"].nunique(),
        "abstain_rate": float(g["abstained"].mean()) if len(g) else np.nan,
        "n_attempted": len(scored),
        "closure": float(scored.mean()) if len(scored) else np.nan,
        # Wilson-free: a plain binomial SE is enough to see whether a bin can carry a claim.
        "se": float(np.sqrt(scored.mean() * (1 - scored.mean()) / len(scored)))
        if len(scored) else np.nan,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", nargs=2, metavar=("LABEL", "RESULTS_DIR"),
                    required=True, help="Repeatable: --arm meet <dir> --arm lnr <dir>")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--match-max-length", type=int, default=13,
                    help="Length ceiling for the MATCHED comparison. MEET stops at 13.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat([load_arm(Path(d), label) for label, d in args.arm], ignore_index=True)
    if "input_nc_gap_angstrom" not in df.columns:
        raise SystemExit("FATAL: rows carry no input_nc_gap_angstrom -- re-run with metadata "
                         "that has the column; binning is the whole point of this script.")

    df["gap_bin"] = pd.cut(df["input_nc_gap_angstrom"], GAP_BINS, labels=GAP_LABELS,
                           right=True, include_lowest=True)
    df["len_bin"] = pd.cut(df["input_peptide_length"], LEN_BINS, labels=LEN_LABELS)
    df.to_csv(args.out_dir / "rows.csv", index=False)

    out: dict[str, object] = {"arms": {l: d for l, d in args.arm},
                              "match_max_length": args.match_max_length,
                              "coverage": {
                                  str(a): {"targets_seen": int(g["arm_n_targets_seen"].iloc[0]),
                                           "targets_complete": int(g["arm_n_targets_complete"].iloc[0]),
                                           "edits": int(len(g))}
                                  for a, g in df.groupby("arm", observed=True)}}

    print("\n=== overall, per requested chemistry (UNMATCHED -- read the matched table below) ===")
    overall = df.groupby(["arm", "cyc_type"], observed=True).apply(rate, include_groups=False)
    print(overall.round(3).to_string())
    overall.round(4).to_csv(args.out_dir / "overall.csv")
    out["overall"] = json.loads(overall.reset_index().to_json(orient="records"))

    matched = df[df["input_peptide_length"] <= args.match_max_length].copy()
    print(f"\n=== length-matched to <= {args.match_max_length} residues, by N-C gap bin ===")
    by_gap = matched.groupby(["cyc_type", "gap_bin", "arm"], observed=True).apply(
        rate, include_groups=False)
    print(by_gap.round(3).to_string())
    by_gap.round(4).to_csv(args.out_dir / "by_gap_bin.csv")
    out["by_gap_bin"] = json.loads(by_gap.reset_index().to_json(orient="records"))

    print(f"\n=== length-matched pooled over chemistry, by N-C gap bin ===")
    pooled = matched.groupby(["gap_bin", "arm"], observed=True).apply(rate, include_groups=False)
    print(pooled.round(3).to_string())
    pooled.round(4).to_csv(args.out_dir / "by_gap_bin_pooled.csv")
    out["by_gap_bin_pooled"] = json.loads(pooled.reset_index().to_json(orient="records"))

    print("\n=== length-matched, by length bin ===")
    by_len = matched.groupby(["len_bin", "arm"], observed=True).apply(rate, include_groups=False)
    print(by_len.round(3).to_string())
    by_len.round(4).to_csv(args.out_dir / "by_length_bin.csv")
    out["by_length_bin"] = json.loads(by_len.reset_index().to_json(orient="records"))

    # The verdict, stated as the residual gap AFTER matching. A bin with fewer than 20
    # attempted edits per arm cannot carry it, and is reported as such rather than averaged in.
    verdict = []
    for gb in GAP_LABELS:
        cell = pooled.loc[gb] if gb in pooled.index.get_level_values(0) else None
        if cell is None or len(cell) < 2:
            verdict.append({"gap_bin": gb, "note": "one arm has no data in this bin"})
            continue
        labels = list(cell.index)
        row = {"gap_bin": gb}
        for l in labels:
            row[f"{l}_closure"] = round(float(cell.loc[l, "closure"]), 4)
            row[f"{l}_n"] = int(cell.loc[l, "n_attempted"])
        thin = [l for l in labels if cell.loc[l, "n_attempted"] < 20]
        row["underpowered"] = thin
        if len(labels) == 2:
            row["delta"] = round(float(cell.loc[labels[0], "closure"]
                                       - cell.loc[labels[1], "closure"]), 4)
            row["delta_is"] = f"{labels[0]} minus {labels[1]}"
        verdict.append(row)
    out["residual_gap_after_matching"] = verdict
    print("\n=== residual gap after matching (pooled over chemistry) ===")
    print(json.dumps(verdict, indent=2))

    (args.out_dir / "comparison.json").write_text(json.dumps(out, indent=2))

    # Figure: closure vs gap bin, one panel per chemistry, one line per arm.
    types = [t for t in ("mainchain", "disulfide", "isopeptide") if t in set(matched["cyc_type"])]
    fig, axes = plt.subplots(1, len(types) + 1, figsize=(4 * (len(types) + 1), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, types + ["pooled"]):
        sub = matched if t == "pooled" else matched[matched["cyc_type"] == t]
        tab = sub.groupby(["gap_bin", "arm"], observed=True).apply(rate, include_groups=False)
        for label in sorted(set(matched["arm"])):
            xs, ys, es = [], [], []
            for i, gb in enumerate(GAP_LABELS):
                if (gb, label) in tab.index:
                    xs.append(i); ys.append(tab.loc[(gb, label), "closure"])
                    es.append(tab.loc[(gb, label), "se"])
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label)
        ax.set_xticks(range(len(GAP_LABELS))); ax.set_xticklabels(GAP_LABELS)
        ax.set_title(t); ax.set_xlabel("input N-C gap"); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("closure | attempted")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"de novo ring closure, length-matched to <= {args.match_max_length} residues")
    fig.tight_layout()
    fig.savefig(args.out_dir / "closure_by_gap.png", dpi=160)
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
