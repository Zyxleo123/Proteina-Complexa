"""Aggregate the LP->CP test-set val_generation eval: closure rates + Rosetta interface dG.

Reads what the GPU driver and the CPU Rosetta sidecar wrote to disk and produces a single
summary.json plus a redrawable figure. No GPU, no model -- pure disk read, so the figure can be
re-rendered cheaply after more Rosetta shards land.

INPUTS (under --out-dir)
  closure_rows_*.jsonl   one row per sampled batch: {metrics: {val_gen/cyc/*, val_gen/geom/*, ...}}
  rosetta_dg/*.jsonl     one row per scored complex: {cyclization_type, dG_separated, ...}

The closure rates are n-WEIGHTED across batches: each batch logs a rate (e.g. mainchain closure)
and the count it was computed over (n_valid_mainchain), and a naive mean of the per-batch rates
would over-weight batches with few of that linkage. mean = sum(rate_i * n_i) / sum(n_i).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


# (rate key, count key) pairs -- the count is the denominator the rate was measured over.
CLOSURE = {
    "mainchain": ("val_gen/cyc/mainchain_cn_bond_success", "val_gen/cyc/n_valid_mainchain"),
    "disulfide": ("val_gen/cyc/disulfide_bond_success", "val_gen/cyc/n_valid_disulfide"),
    "isopeptide": ("val_gen/cyc/isopeptide_bond_success", "val_gen/cyc/n_valid_isopeptide"),
    "cb_window_all": ("val_gen/cyc/cyc_cb_window_success", "val_gen/cyc/n_valid_cyc"),
    "type_satisfied": ("val_gen/cyc/type_satisfied", "val_gen/cyc/n_valid_cyc"),
}
GEOM = {  # value key, weighted by n_valid_cyc
    "ca_ca_nm": "val_gen/geom/ca_ca_nm",
    "ca_ca_viol_frac": "val_gen/geom/ca_ca_viol_frac",
}


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or "\x00" in line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _wmean(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """(weighted mean, total weight) over finite (value, weight) pairs; NaN if no weight."""
    num = den = 0.0
    for v, w in pairs:
        if w and math.isfinite(v) and math.isfinite(w):
            num += v * w
            den += w
    return (num / den if den else float("nan")), den


def _stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    median = xs_sorted[n // 2] if n % 2 else (xs_sorted[n // 2 - 1] + xs_sorted[n // 2]) / 2
    return {"n": n, "mean": sum(xs) / n, "median": median, "min": xs_sorted[0], "max": xs_sorted[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="Eval dir with closure_rows_*.jsonl + rosetta_dg/.")
    ap.add_argument("--rosetta-subdir", default="rosetta_dg", help="Subdir of scored dG JSONL shards.")
    ap.add_argument("--figure", default=None, help="Path for the PNG (default <out-dir>/summary.png).")
    args = ap.parse_args()

    out_dir = args.out_dir
    closure_rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "closure_rows_*.jsonl"))):
        closure_rows.extend(_read_jsonl(f))
    print(f"closure batches: {len(closure_rows)}")

    closure_summary = {}
    for name, (rk, ck) in CLOSURE.items():
        pairs = []
        for r in closure_rows:
            m = r.get("metrics", {})
            if rk in m and ck in m:
                pairs.append((float(m[rk]), float(m[ck])))
        mean, tot = _wmean(pairs)
        closure_summary[name] = {"rate": mean, "n": tot}
    for name, vk in GEOM.items():
        pairs = []
        for r in closure_rows:
            m = r.get("metrics", {})
            if vk in m and "val_gen/cyc/n_valid_cyc" in m:
                pairs.append((float(m[vk]), float(m["val_gen/cyc/n_valid_cyc"])))
        mean, tot = _wmean(pairs)
        closure_summary[name] = {"value": mean, "n": tot}

    # Rosetta dG, overall + per requested/predicted linkage type recorded in the manifest.
    rosetta_rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, args.rosetta_subdir, "*.jsonl"))):
        rosetta_rows.extend(_read_jsonl(f))
    print(f"rosetta scored complexes: {len(rosetta_rows)}")

    def _dg(rows):
        return [float(r["dG_separated"]) for r in rows if r.get("dG_separated") is not None]

    by_type = defaultdict(list)
    for r in rosetta_rows:
        by_type[r.get("cyclization_type") or "linear_scored"].append(r)
    rosetta_summary = {"overall": _stats(_dg(rosetta_rows))}
    for t, rs in sorted(by_type.items()):
        rosetta_summary[t] = _stats(_dg(rs))

    summary = {
        "n_closure_batches": len(closure_rows),
        "n_rosetta_complexes": len(rosetta_rows),
        "ckpt": closure_rows[0].get("ckpt_name") if closure_rows else None,
        "ckpt_global_step": closure_rows[0].get("ckpt_global_step") if closure_rows else None,
        "nsteps": closure_rows[0].get("nsteps") if closure_rows else None,
        "n_repeat": closure_rows[0].get("n_repeat") if closure_rows else None,
        "closure": closure_summary,
        "rosetta_dG_separated": rosetta_summary,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")

    # ---- Figure (best-effort; the numbers above are the deliverable) -------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axc, axd) = plt.subplots(1, 2, figsize=(12, 4.5))
        types = ["mainchain", "disulfide", "isopeptide", "cb_window_all", "type_satisfied"]
        rates = [(closure_summary[t]["rate"] or 0) for t in types]
        ns = [int(closure_summary[t]["n"] or 0) for t in types]
        bars = axc.bar(types, rates, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D"])
        for b, r, nn in zip(bars, rates, ns):
            axc.text(b.get_x() + b.get_width() / 2, (r or 0) + 0.02, f"{r:.2f}\nn={nn}",
                     ha="center", va="bottom", fontsize=8)
        axc.set_ylim(0, 1.15)
        axc.set_ylabel("closure rate (sampled)")
        axc.set_title(f"Sampled cyclization closure\nca_ca={closure_summary['ca_ca_nm']['value']:.3f} nm")
        axc.tick_params(axis="x", rotation=20)

        dg_groups = [(t, _dg(rs)) for t, rs in sorted(by_type.items())]
        dg_groups = [(t, d) for t, d in dg_groups if d]
        if dg_groups:
            axd.boxplot([d for _, d in dg_groups], labels=[t for t, _ in dg_groups], showmeans=True)
            axd.axhline(0, color="grey", lw=0.8, ls="--")
            axd.set_ylabel("Rosetta dG_separated (REU)")
            axd.set_title(f"Interface dG by linkage (n={len(rosetta_rows)})")
            axd.tick_params(axis="x", rotation=20)
        else:
            axd.text(0.5, 0.5, "no Rosetta dG yet", ha="center", va="center")
            axd.axis("off")

        fig.suptitle(f"LP->CP ckpt {summary['ckpt']} @ step {summary['ckpt_global_step']} "
                     f"-- {len(rosetta_rows)} complexes, nsteps={summary['nsteps']}", fontsize=10)
        fig.tight_layout()
        fig_path = args.figure or os.path.join(out_dir, "summary.png")
        fig.savefig(fig_path, dpi=130)
        print(f"Wrote {fig_path}")
    except Exception as e:  # a figure failure must never lose the summary
        print(f"(figure skipped: {e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
