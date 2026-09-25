"""Adversarial harness and OOD scorer (pre-build checks 3c and 3e).

Takes two labelled sets of feature profiles and reports how well a classifier separates
them, plus the feature importances that say on what.

Nothing synthetic exists yet, so the harness runs against real sets now.  Its job today is
to demonstrate that it CAN detect a difference it should detect; the synthetic comparison
comes later and reuses this code unchanged.

Four protocol rules, each of which exists because its absence produces a confidently wrong
number:

**Positive control first.**  CPSea cyclics against real linears is known to differ
substantially, so AUC should come back near 1.0.  An AUC of 0.5 on a comparison that
matters means nothing until the harness has shown it can detect a difference it should.
The control result gates the report rather than sitting beside it.

**Tuning and held-out features reported separately.**  An AUC of 0.5 on features that were
optimised against is not evidence of a distribution match.  `peptide_profile` partitions
the vector into the conformational descriptors a corruption sampler would draw targets for
(TUNING) and the interface/packing descriptors it would not (HELD_OUT); both AUCs are
reported at every iteration and the held-out one is the one that means something.

**A control classifier on staging features alone.**  If receptor length and fragmentation
separate the sets by themselves, the main AUC is measuring the staging and is void until
the control is at chance.  Reported every time, never on request.

**Grouped splits.**  Folds are split on `cluster_id`, not on rows.  These sets are
family-redundant, and a random row split puts near-duplicates on both sides of the fold,
which inflates every AUC here toward 1.0 for a reason that has nothing to do with the
distributions.

Runs in `.venv` (sklearn 1.9).  Neither xgboost nor lightgbm is installed, so the
gradient-boosted arm is sklearn's `HistGradientBoostingClassifier`.
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
from script_utils.peptide_profile import (  # noqa: E402
    FEATURES,
    HELDOUT_FEATURES,
    STAGING_FEATURES,
    TUNING_FEATURES,
)


def _make_clf(kind: str, seed: int, n_train: int = 10_000):
    """Classifier sized for the training fold it will see.

    `min_samples_leaf` is scaled rather than left at sklearn's default of 20.  That default
    is tuned for large data and silently produces a CONSTANT model on small folds: a split
    needs `min_samples_leaf` rows in each child, so with 32 training rows no split is legal,
    every tree is a single leaf, and the result is an AUC of exactly 0.500 with all-zero
    permutation importances.  Measured on a 40-row smoke -- where it correctly failed the
    positive control, but would have been read as "the sets are indistinguishable" in any
    stratum small enough to hit it.  The length strata are exactly that small.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=seed))
    return HistGradientBoostingClassifier(
        random_state=seed, max_iter=200, early_stopping=False,
        min_samples_leaf=max(2, min(20, int(n_train) // 10)),
    )


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str,
           n_splits: int, seed: int) -> dict:
    """Grouped, stratified cross-validated ROC AUC.

    Returns the pooled out-of-fold AUC rather than the mean of per-fold AUCs: with small
    folds a per-fold mean is dominated by whichever fold happened to be easiest, and one
    fold with a single class present cannot produce an AUC at all.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y)), "note": "only one class present"}
    n_splits = max(2, min(n_splits, int(min(np.bincount(y)))))
    if n_splits < 2:
        return {"auc": float("nan"), "n": int(len(y)), "note": "too few in minority class"}

    oof = np.full(len(y), np.nan)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = _make_clf(kind, seed, n_train=len(tr))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    ok = np.isfinite(oof)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return {"auc": float("nan"), "n": int(ok.sum()), "note": "no usable out-of-fold rows"}

    # A constant score is a DEGENERATE FIT, not a measurement of indistinguishability, and
    # it scores exactly 0.5 -- which reads as "at chance" and would be quoted as a finding.
    # Report it as unusable instead.
    if float(np.std(oof[ok])) < 1e-12:
        return {"auc": float("nan"), "n": int(ok.sum()), "n_splits": int(n_splits),
                "note": "degenerate fit: the model predicted a constant, so this is not a "
                        "measurement of separability. Usually too few rows for the tree to "
                        "make any legal split."}
    return {"auc": float(roc_auc_score(y[ok], oof[ok])), "n": int(ok.sum()),
            "n_splits": int(n_splits)}


def importances(X: np.ndarray, y: np.ndarray, names: list[str], kind: str, seed: int) -> dict:
    """Permutation importance on a held-out split.

    Permutation rather than a model-native attribute: `HistGradientBoostingClassifier` does
    not expose one, and a split-count proxy would rank a feature by how often the tree used
    it rather than by how much the answer depends on it.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    if len(np.unique(y)) < 2 or len(y) < 8:
        return {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {}
    clf = _make_clf(kind, seed, n_train=len(Xtr))
    clf.fit(Xtr, ytr)
    r = permutation_importance(clf, Xte, yte, n_repeats=10, random_state=seed,
                               scoring="roc_auc")
    return {n: float(m) for n, m in zip(names, r.importances_mean)}


def _matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def run_pair(df: pd.DataFrame, label_a: str, label_b: str, cfg: dict) -> dict:
    """Every AUC for one labelled pair: full, tuning, held-out, and the staging control."""
    acfg = cfg.get("adversary") or {}
    kind = "logistic" if acfg.get("classifier") == "logistic" else "hist_gradient_boosting"
    n_splits = int(acfg.get("n_splits", 5))
    seed = int(acfg.get("seed", 0))

    sub = df[df["label"].isin([label_a, label_b])].copy()
    if sub.empty:
        return {"pair": [label_a, label_b], "note": "no rows"}
    y = (sub["label"] == label_b).astype(int).to_numpy()
    groups = sub["cluster_id"].astype(str).fillna("na").to_numpy()

    out: dict = {"pair": [label_a, label_b], "classifier": kind,
                 "n_a": int((y == 0).sum()), "n_b": int((y == 1).sum())}

    blocks = {
        "all": list(FEATURES),
        "tuning": list(TUNING_FEATURES),
        "heldout": list(HELDOUT_FEATURES),
    }
    for name, cols in blocks.items():
        out[f"auc_{name}"] = cv_auc(_matrix(sub, cols), y, groups, kind, n_splits, seed)

    if acfg.get("control_on_staging_features", True):
        cols = [c for c in STAGING_FEATURES if c in sub.columns]
        out["auc_staging_control"] = cv_auc(_matrix(sub, cols), y, groups, kind,
                                            n_splits, seed)
        # The same control on the PRE-crop staging, which is what the number would have
        # been without the uniform crop. Reported so the crop's effect is measured rather
        # than asserted.
        pre = [f"pre_{c}" for c in STAGING_FEATURES if f"pre_{c}" in sub.columns]
        if pre:
            out["auc_staging_control_precrop"] = cv_auc(_matrix(sub, pre), y, groups,
                                                        kind, n_splits, seed)

    out["importances"] = importances(_matrix(sub, list(FEATURES)), y, list(FEATURES),
                                     kind, seed)

    if acfg.get("stratify_by_peptide_length", True) and "peptide_length" in sub.columns:
        edges = [float(e) for e in acfg.get("length_bins", [0, 8, 11, 14, 100])]
        strata = []
        for lo, hi in zip(edges, edges[1:]):
            m = (sub["peptide_length"] > lo) & (sub["peptide_length"] <= hi)
            g = sub[m]
            if len(g) < 8:
                strata.append({"bin": f"({lo:g},{hi:g}]", "n": int(len(g)),
                               "note": "too few rows"})
                continue
            yy = (g["label"] == label_b).astype(int).to_numpy()
            strata.append({
                "bin": f"({lo:g},{hi:g}]", "n": int(len(g)),
                "n_a": int((yy == 0).sum()), "n_b": int((yy == 1).sum()),
                "auc_heldout": cv_auc(_matrix(g, list(HELDOUT_FEATURES)), yy,
                                      g["cluster_id"].astype(str).fillna("na").to_numpy(),
                                      kind, n_splits, seed),
            })
        out["by_length"] = strata
    return out


def fit_ood(df: pd.DataFrame, cfg: dict) -> dict:
    """Pre-build check 3e: a per-input OOD score against the CPSea-side distribution.

    Fitted on the CPSea side ONLY.  The score answers "how far is this input from the
    distribution the generator was trained on", so fitting it on the union of the sets
    would blunt exactly the signal it exists to give.

    Returned as the fitted parameters plus a scored column on every row, so an LNR input
    can later carry an OOD score alongside its prediction.
    """
    from sklearn.neighbors import KernelDensity

    ocfg = cfg.get("ood") or {}
    fit_on = ocfg.get("fit_on", "cpsea")
    cols = list(FEATURES)
    ref = df[df["label"] == fit_on]
    if len(ref) < 10:
        return {"note": f"too few {fit_on} rows to fit ({len(ref)})"}

    Xr = _matrix(ref, cols)
    mu, sd = np.nanmean(Xr, axis=0), np.nanstd(Xr, axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)

    def z(M):
        return np.nan_to_num((M - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    # Scott's rule, so the bandwidth follows the sample size and dimension rather than
    # being a tuned constant that would quietly set how strict the score is.
    n, d = Xr.shape
    bw = float(n ** (-1.0 / (d + 4)))
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(z(Xr))

    scores = kde.score_samples(z(_matrix(df, cols)))
    ref_scores = kde.score_samples(z(Xr))
    # Reported as a percentile against the reference set: a raw log-density is unreadable
    # on its own, while "this input sits below 3% of CPSea" is directly actionable.
    pct = np.searchsorted(np.sort(ref_scores), scores) / max(1, len(ref_scores))

    return {
        "fit_on": fit_on, "n_fit": int(n), "bandwidth": bw, "features": cols,
        "mean": mu.tolist(), "std": sd.tolist(),
        "ref_logdensity_q": {str(q): float(np.percentile(ref_scores, q))
                             for q in (1, 5, 25, 50, 75, 95, 99)},
        "per_row": {"example_id": df["example_id"].tolist(),
                    "label": df["label"].tolist(),
                    "logdensity": scores.tolist(),
                    "pct_vs_fit_set": pct.tolist()},
        "by_label_median_pct": {
            str(lab): float(np.median(pct[(df["label"] == lab).to_numpy()]))
            for lab in sorted(df["label"].unique())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", required=True, help="profile_rows.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", default="",
                    help="Space-separated a:b pairs. Empty = the positive control only.")
    args = ap.parse_args()

    cfg = precheck_config.load(args.config)
    acfg = cfg.get("adversary") or {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.rows)

    pc = acfg.get("positive_control") or {}
    ctrl_a, ctrl_b = pc.get("label_a", "cpsea"), pc.get("label_b", "pepbench")
    min_auc = float(pc.get("min_auc", 0.90))

    results = {"positive_control": run_pair(df, ctrl_a, ctrl_b, cfg), "pairs": []}
    ctrl_auc = results["positive_control"].get("auc_all", {}).get("auc", float("nan"))
    passed = bool(np.isfinite(ctrl_auc) and ctrl_auc >= min_auc)
    results["positive_control"]["min_auc"] = min_auc
    results["positive_control"]["passed"] = passed

    # The control GATES the rest rather than sitting beside it. If the harness cannot
    # detect a difference it should detect, every other AUC it produces is uninterpretable,
    # so they are still computed but carried with the failure attached.
    for spec in (args.pairs.split() if args.pairs else []):
        if ":" not in spec:
            raise SystemExit(f"--pairs entries look like a:b, got {spec!r}")
        a, b = spec.split(":", 1)
        r = run_pair(df, a, b, cfg)
        r["positive_control_passed"] = passed
        results["pairs"].append(r)

    results["ood"] = fit_ood(df, cfg)
    (out_dir / "adversary_results.json").write_text(json.dumps(results, indent=2, default=float))

    ood = results["ood"]
    if "per_row" in ood:
        pd.DataFrame(ood.pop("per_row")).to_csv(out_dir / "ood_scores.csv", index=False)

    print(f"positive control {ctrl_a} vs {ctrl_b}: "
          f"AUC(all)={ctrl_auc:.3f} threshold={min_auc} -> "
          f"{'PASS' if passed else 'FAIL'}")
    for key in ("auc_tuning", "auc_heldout", "auc_staging_control",
                "auc_staging_control_precrop"):
        v = results["positive_control"].get(key, {})
        if v:
            print(f"  {key:30s} {v.get('auc', float('nan')):.3f}  (n={v.get('n')})")
    print(f"wrote {out_dir}/adversary_results.json")


if __name__ == "__main__":
    main()
