"""Fit the sufficiency side of the glob-dependence probe, and tabulate the verdict.

Reads the JSONL `cp2lp_glob_probe.py` dumps and answers two questions the GPU job cannot:

  SUFFICIENCY -- how much of `d_acc` is reachable from the global columns ALONE? Logistic
  regressions on the terminal gap by itself, on gap+Rg, on all four `glob` columns, and on
  the two columns the existing shortcut probe is allowed to see (`glob[:, 2:]`), each
  cross-validated. If gap alone approaches D's own accuracy on the identical rows, `d_acc`
  is a readout of ring openness and not of realism.

  USAGE -- does D's decision MOVE with the gap? Spearman between `terminal_gap_nm` and
  `d_logit`, computed WITHIN each class. The pooled correlation is confounded by the label
  (reals have both larger gaps and higher logits), so only the within-class figure says
  anything; it is the correlational companion to the GPU job's counterfactual.

TWO SPLITTING DECISIONS THAT CHANGE THE ANSWER.

* GROUPED BY BATCH, not shuffled. `samples_per_cp` copies of one CP are near-duplicates and
  land in the same batch; a shuffled split would put a peptide's twin in the training fold
  and inflate every accuracy here.
* LENGTH IS REPORTED AS A CONTROL, not a feature. D is scored on length-matched pairs
  (`d_match_frac` 1.0), so `peptide_length` alone must come out at chance. If it does not,
  the matching has failed and every other number in this table is reading a length tell --
  so it is printed first, as a gate rather than a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GAP = "terminal_gap_nm"
RG = "radius_of_gyration_nm"
CONTACTS = "contact_count"
MINHEAVY = "mean_min_heavy_nm"

FEATURE_SETS: dict[str, list[str]] = {
    "gap only": [GAP],
    "gap + Rg (the openness pair)": [GAP, RG],
    "all 4 glob columns": [GAP, RG, CONTACTS, MINHEAVY],
    "glob[2:] (what the shortcut probe sees)": [CONTACTS, MINHEAVY],
    "peptide_length (control, must be ~0.5)": ["peptide_length"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rows", required=True, help="JSONL from cp2lp_glob_probe.py")
    p.add_argument("--probe-summary", required=True, help="JSON from cp2lp_glob_probe.py")
    p.add_argument("--out", required=True, help="output JSON summary")
    p.add_argument("--markdown", default=None, help="optional markdown table path")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def cv_scores(
    df: pd.DataFrame, cols: list[str], folds: int, seed: int
) -> tuple[float, float, int]:
    """Grouped-CV accuracy and AUC for a logistic regression on `cols`."""
    x = df[cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    groups = df["batch"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = int(min(folds, n_groups))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), n_splits

    preds = np.zeros_like(y, dtype=np.float64)
    hard = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
        if len(np.unique(y[tr])) < 2:
            preds[te] = 0.5
            hard[te] = y[tr][0] if len(tr) else 0
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
        model.fit(x[tr], y[tr])
        preds[te] = model.predict_proba(x[te])[:, 1]
        hard[te] = model.predict(x[te])
    acc = float((hard == y).mean())
    auc = float(roc_auc_score(y, preds)) if len(np.unique(y)) > 1 else float("nan")
    return acc, auc, n_splits


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    if not rows:
        logger.error(f"no rows in {args.rows}")
        return 1
    df = pd.DataFrame(rows)
    probe = json.loads(Path(args.probe_summary).read_text())

    n_real = int((df["label"] == 1).sum())
    n_fake = int((df["label"] == 0).sum())
    logger.info(f"{len(df)} rows: {n_real} real, {n_fake} generated, {df['batch'].nunique()} batches")

    # ---- D's own accuracy on these exact rows, from the dumped logits
    d_acc_rows = float(
        (((df["label"] == 1) & (df["d_logit"] > 0)) | ((df["label"] == 0) & (df["d_logit"] <= 0)))
        .mean()
    )

    out: dict[str, object] = {
        "n_rows": len(df),
        "n_real": n_real,
        "n_generated": n_fake,
        "n_batches": int(df["batch"].nunique()),
        "ckpt": probe.get("ckpt"),
        "g_updates": probe.get("g_updates"),
        "d_acc_on_these_rows": d_acc_rows,
        "d_acc_baseline_gpu": probe.get("d_acc_baseline"),
        "ablations": {
            k: probe.get(k)
            for k in ("d_acc_openblind", "d_acc_gapblind", "d_acc_rgblind", "delta_openblind")
        },
        "verdict_dependence": probe.get("verdict"),
    }

    # ---- distribution of the suspect columns, by class
    dist = {}
    for col in (GAP, RG):
        dist[col] = {
            "real_mean": float(df.loc[df.label == 1, col].mean()),
            "real_median": float(df.loc[df.label == 1, col].median()),
            "gen_mean": float(df.loc[df.label == 0, col].mean()),
            "gen_median": float(df.loc[df.label == 0, col].median()),
            "overlap_frac": float(
                (
                    (df.loc[df.label == 0, col] > df.loc[df.label == 1, col].quantile(0.05)).mean()
                )
            ),
        }
    out["distributions"] = dist

    # ---- sufficiency
    suff = {}
    for name, cols in FEATURE_SETS.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(f"skipping '{name}': missing columns {missing}")
            continue
        acc, auc, n_splits = cv_scores(df, cols, args.folds, args.seed)
        suff[name] = {"cv_accuracy": acc, "cv_auc": auc, "n_splits": n_splits}
        logger.info(f"  {name:<42s} acc={acc:.4f} auc={auc:.4f}")
    out["sufficiency"] = suff

    # ---- usage, within class (pooled is confounded by the label)
    usage = {}
    for label, tag in ((1, "real"), (0, "generated")):
        sub = df[df.label == label]
        if len(sub) > 8 and sub[GAP].std() > 0:
            rho, pval = spearmanr(sub[GAP], sub["d_logit"])
            usage[tag] = {"spearman_gap_vs_logit": float(rho), "p": float(pval), "n": len(sub)}
    out["usage_within_class"] = usage

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    if args.markdown:
        gap_acc = suff.get("gap only", {}).get("cv_accuracy", float("nan"))
        lines = [
            "# Is `d_acc` reading ring openness?",
            "",
            f"`{Path(str(probe.get('ckpt'))).parent.parent.name}` at g_updates "
            f"{probe.get('g_updates')} — {n_real} real / {n_fake} generated rows, "
            f"{df['batch'].nunique()} batches.",
            "",
            "## Dependence (counterfactual on the trained D)",
            "",
            "| scoring | d_acc |",
            "|---|---|",
            f"| baseline | {probe.get('d_acc_baseline'):.4f} |",
            f"| terminal gap held | {probe.get('d_acc_gapblind'):.4f} |",
            f"| Rg held | {probe.get('d_acc_rgblind'):.4f} |",
            f"| both held | {probe.get('d_acc_openblind'):.4f} |",
            "",
            f"**{probe.get('verdict')}**",
            "",
            "A collapse proves dependence; a null bounds it from below, because `glob` enters",
            "only the head and the trunk may re-derive an openness proxy from `geom`/`iface`.",
            "",
            "## Sufficiency (grouped-CV logistic regression)",
            "",
            "| features | accuracy | AUC |",
            "|---|---|---|",
        ]
        for name, s in suff.items():
            lines.append(f"| {name} | {s['cv_accuracy']:.4f} | {s['cv_auc']:.4f} |")
        lines += [
            "",
            f"D itself scores {d_acc_rows:.4f} on these rows. Gap alone reaches "
            f"{gap_acc:.4f}.",
            "",
            "## Usage (within-class Spearman, gap vs D logit)",
            "",
            "| class | rho | n |",
            "|---|---|---|",
        ]
        for tag, u in usage.items():
            lines.append(f"| {tag} | {u['spearman_gap_vs_logit']:+.3f} | {u['n']} |")
        lines += [
            "",
            "## The distributions being separated",
            "",
            "| column | real median | generated median |",
            "|---|---|---|",
        ]
        for col, d in dist.items():
            lines.append(f"| `{col}` | {d['real_median']:.3f} | {d['gen_median']:.3f} |")
        Path(args.markdown).write_text("\n".join(lines) + "\n")
        logger.info(f"wrote {args.markdown}")

    logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
