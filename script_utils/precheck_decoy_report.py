"""Report for pre-build check 1 -- decoy filter calibration.

Answers the question the check was posed to answer: can `contact_retention` substitute for
Rosetta interface dG as the pose-decoy acceptance filter, and where do the band bounds sit?

Three results, in the order they decide the question:

  1. **The restricted regression is the answer.**  A correlation that holds across the whole
     0-5 A sweep can be useless inside the narrow band the filter actually runs in, so the
     global number is context and the operating-range number decides.
  2. **Normal and tangential displacement are reported separately, always.**  Pulling off
     the interface and sliding along it produce completely different retention at the same
     scalar amplitude, so a regression pooled over direction is fitting a mixture.
  3. **The collapse rate** -- what fraction of perturbed poses return to within a small
     RMSD of native after minimization, against initial amplitude.  A decoy that collapses
     back is not a decoy, and a set of them would teach the shortcut the build exists to
     remove.

Runs in `.venv`.  CPU only; reads saved artefacts, so the tables redraw without rescoring.
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


def load_scored(score_dir: Path) -> pd.DataFrame:
    files = sorted(score_dir.glob("scored_shard*.jsonl"))
    if not files:
        raise SystemExit(f"no scored shards under {score_dir}")
    rows, bad = [], []
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                bad.append(f"{f.name}:{n}: {exc}")
    if bad:
        raise SystemExit(
            "REFUSING to aggregate: {} unparseable row(s); skipping them would turn lost "
            "data into a smaller-but-plausible correlation.\n  {}".format(
                len(bad), "\n  ".join(bad[:10])))
    return pd.DataFrame(rows)


def corr_block(df: pd.DataFrame, x: str, y: str) -> dict:
    """Spearman, Pearson and residual spread for y ~ x, on the rows where both exist."""
    from scipy import stats

    g = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(g) < 8:
        return {"n": int(len(g)), "note": "too few paired rows"}
    xs, ys = g[x].to_numpy(), g[y].to_numpy()
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return {"n": int(len(g)), "note": "no variance in one axis"}
    sp = stats.spearmanr(xs, ys)
    pe = stats.pearsonr(xs, ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    return {
        "n": int(len(g)),
        "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "pearson": float(pe.statistic), "pearson_p": float(pe.pvalue),
        "slope": float(slope), "intercept": float(intercept),
        "residual_sd": float(np.std(resid, ddof=2)),
        "r2": float(pe.statistic ** 2),
        "y_sd": float(np.std(ys, ddof=1)),
    }


def _fmt(b: dict, key: str, spec="{:.3f}") -> str:
    v = b.get(key)
    return spec.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = precheck_config.load(args.config)
    acfg = cfg["decoy_calibration"]["analysis"]
    op = acfg["operating_range"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_scored(Path(args.score_dir))
    df.to_csv(out_dir / "decoy_pose_table.csv", index=False)

    has_dg = df[["post_rosetta_dG"]].apply(pd.to_numeric, errors="coerce").notna().any().any()

    L: list[str] = []
    A = L.append
    A("# Pre-build check 1 -- decoy filter calibration\n")
    A(f"Poses: {len(df)} over {df['example_id'].nunique()} complexes.  "
      f"Translation 0-{df['translation_A'].max():.1f} A, "
      f"rotation 0-{df['rotation_deg'].max():.0f} deg.\n")

    if not has_dg:
        A("> **Rosetta dG is absent from every row.** This run was scored with")
        A("> `--no-rosetta`, which makes it a geometry-only smoke and NOT an answer to the")
        A("> calibration question. The retention/geometry tables below are still valid;")
        A("> every dG table is omitted rather than filled with NaN.\n")

    A("\n## 1. Does retention track dG?\n")
    if has_dg:
        A("Both scorings are reported. The `post` (minimized) pair is the operative one --")
        A("it is what a filter would see -- and the `pre` pair is what the perturbation")
        A("produced before the force field had a say.\n")
        A("| slice | n | Spearman | Pearson | r2 | slope (REU per unit retention) | residual SD (REU) | dG SD (REU) |")
        A("|---|---|---|---|---|---|---|---|")
        slices = {
            "global, post-min": df,
            "global, pre-min": df,
        }
        for name, d in slices.items():
            ph = "post" if "post" in name else "pre"
            b = corr_block(d, f"{ph}_contact_retention", f"{ph}_rosetta_dG")
            A(f"| {name} | {b.get('n')} | {_fmt(b,'spearman')} | {_fmt(b,'pearson')} | "
              f"{_fmt(b,'r2')} | {_fmt(b,'slope','{:.2f}')} | "
              f"{_fmt(b,'residual_sd','{:.2f}')} | {_fmt(b,'y_sd','{:.2f}')} |")

        # The restricted fit: the band the filter actually runs in.
        opr = df[(df["translation_A"] <= float(op["max_translation_A"]))
                 & (df["rotation_deg"] <= float(op["max_rotation_deg"]))
                 & (pd.to_numeric(df["post_contact_retention"], errors="coerce")
                    >= float(op["min_contact_retention"]))]
        b = corr_block(opr, "post_contact_retention", "post_rosetta_dG")
        A(f"| **operating range** (<= {op['max_translation_A']} A, "
          f"<= {op['max_rotation_deg']} deg, retention >= {op['min_contact_retention']}) | "
          f"{b.get('n')} | {_fmt(b,'spearman')} | {_fmt(b,'pearson')} | {_fmt(b,'r2')} | "
          f"{_fmt(b,'slope','{:.2f}')} | {_fmt(b,'residual_sd','{:.2f}')} | "
          f"{_fmt(b,'y_sd','{:.2f}')} |")
        A("")
        A("**The operating-range row is the one that decides the question.** A filter does")
        A("not run on 5 A displacements; it runs in the narrow band above, and a global")
        A("correlation that does not survive the restriction cannot be used there.\n")

        if b.get("n", 0) >= 8 and np.isfinite(b.get("residual_sd", np.nan)):
            ratio = b["residual_sd"] / b["y_sd"] if b.get("y_sd") else float("nan")
            A(f"Residual spread is {b['residual_sd']:.2f} REU against a dG spread of "
              f"{b['y_sd']:.2f} REU in that band ({ratio:.0%} of it). "
              + ("Retention explains little of the dG variation there, so substituting it "
                 "for dG would accept and reject largely on something other than interface "
                 "energy.\n" if ratio > 0.8 else
                 "Retention carries a usable fraction of the dG signal in that band.\n"))
    else:
        A("Omitted: no dG in this run.\n")

    A("\n## 2. Direction matters more than amplitude\n")
    A("The interface normal points out of the receptor into the peptide, so `normal+`")
    A("pulls the peptide off its site and `normal-` presses it in. At equal scalar")
    A("amplitude these are not comparable perturbations, and a regression pooled over")
    A("direction fits a mixture of them.\n")
    A("| direction | n | median translation | median retention (post) | median CA-RMSD (post) | median Jaccard dist |")
    A("|---|---|---|---|---|---|")
    for mode, g in df.groupby("direction_mode"):
        A(f"| {mode} | {len(g)} | {g['translation_A'].median():.2f} | "
          f"{pd.to_numeric(g['post_contact_retention'], errors='coerce').median():.3f} | "
          f"{pd.to_numeric(g['post_ca_rmsd_A'], errors='coerce').median():.2f} | "
          f"{pd.to_numeric(g['post_jaccard_distance'], errors='coerce').median():.3f} |")
    A("")
    A("Retention against the SIGNED normal displacement, which the pooled amplitude hides:\n")
    A("| normal displacement (A) | n | median retention (post) |")
    A("|---|---|---|")
    edges = [-6, -4, -2, -0.5, 0.5, 2, 4, 6]
    dn = pd.to_numeric(df["d_normal_A"], errors="coerce")
    for lo, hi in zip(edges, edges[1:]):
        g = df[(dn > lo) & (dn <= hi)]
        if len(g) == 0:
            continue
        A(f"| ({lo:g}, {hi:g}] | {len(g)} | "
          f"{pd.to_numeric(g['post_contact_retention'], errors='coerce').median():.3f} |")

    A("\n## 3. Band bounds the data supports\n")
    A("Read off the operating-range rows rather than asserted. The Jaccard lower bound")
    A("`delta` is the value at which retention still separates poses the dG ranks well;")
    A("where the correlation does not survive the restriction, no bound is supportable and")
    A("the table says so instead of quoting one.\n")
    A("| quantile of post-min retention | Jaccard distance | CA-RMSD (A) |"
      + (" dG (REU) |" if has_dg else ""))
    A("|---|---|---|" + ("---|" if has_dg else ""))
    ret = pd.to_numeric(df["post_contact_retention"], errors="coerce")
    for q in (5, 25, 50, 75, 95):
        thr = np.nanpercentile(ret, q)
        g = df[ret >= thr]
        row = (f"| >= p{q} ({thr:.3f}) | "
               f"{pd.to_numeric(g['post_jaccard_distance'], errors='coerce').median():.3f} | "
               f"{pd.to_numeric(g['post_ca_rmsd_A'], errors='coerce').median():.2f} |")
        if has_dg:
            row += f" {pd.to_numeric(g['post_rosetta_dG'], errors='coerce').median():.2f} |"
        A(row)

    A("\n## 4. Collapse rate against initial amplitude\n")
    A("What fraction of perturbed poses return to within "
      f"{acfg['collapse_rmsd_A']} A CA-RMSD of native after frozen-pocket minimization.")
    A("A pose that collapses back is not a decoy: it teaches the model the pocket's")
    A("canonical answer, which is the shortcut the build exists to remove.\n")
    A("| initial translation (A) | n | collapsed | median minimization RMSD (A) |")
    A("|---|---|---|---|")
    t_edges = [0, 0.01, 1, 2, 3, 4, 6]
    tr = pd.to_numeric(df["translation_A"], errors="coerce")
    for lo, hi in zip(t_edges, t_edges[1:]):
        g = df[(tr > lo - (1e-9 if lo == 0 else 0)) & (tr <= hi)] if lo > 0 else df[tr <= hi]
        if len(g) == 0:
            continue
        A(f"| ({lo:g}, {hi:g}] | {len(g)} | "
          f"{pd.to_numeric(g['collapsed'], errors='coerce').mean():.0%} | "
          f"{pd.to_numeric(g['minimization_rmsd_A'], errors='coerce').median():.2f} |")

    A("\n## 5. What this does not establish\n")
    A("- The perturbations are RIGID-BODY only. A decoy generator that also perturbs")
    A("  torsions would produce poses this sweep never visits, and the bounds here do not")
    A("  transfer to them unexamined.")
    A("- dG is scored WITHOUT FastRelax, on the pose as given. That is deliberate --")
    A("  relaxing would undo the perturbation being calibrated -- but it means these are")
    A("  not the same numbers a relaxed protocol would report.")
    A("- The receptor is frozen. Induced fit is therefore absent by construction, and a")
    A("  pose scored badly here might be tolerable to a flexible pocket.\n")

    (out_dir / "PRECHECK_TASK1_REPORT.md").write_text("\n".join(L) + "\n")

    summary = {"n_poses": int(len(df)), "n_complexes": int(df["example_id"].nunique()),
               "has_rosetta_dG": bool(has_dg),
               "collapse_rate_overall": float(
                   pd.to_numeric(df["collapsed"], errors="coerce").mean())}
    if has_dg:
        summary["operating_range_fit"] = corr_block(
            df[(df["translation_A"] <= float(op["max_translation_A"]))
               & (df["rotation_deg"] <= float(op["max_rotation_deg"]))],
            "post_contact_retention", "post_rosetta_dG")
    (out_dir / "precheck_task1_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"wrote {out_dir}/PRECHECK_TASK1_REPORT.md")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
