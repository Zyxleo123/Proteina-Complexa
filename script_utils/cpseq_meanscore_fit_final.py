#!/usr/bin/env python3
"""Fit the shipping proxy on all 20 targets and save it as a usable artifact.

The sweep's verdict (`cpseq_meanscore_report.py`) is that the winning predictor is a
**ridge on composition-delta + ProteinMPNN features**, not the 22.6k-param mutation-effect
network, which it beats at every aggregator and every loss. This fits that ridge on all 20
targets and writes the coefficients, the standardisation statistics and the feature spec,
so the search can score candidates without this repo's training path.

Reported alongside: the in-sample rho (which is NOT the expected performance -- the honest
number is the leave-one-target-out rho from the report) and the per-feature standardised
coefficients, so the model can be sanity-checked rather than trusted.

Cost note, measured: the ProteinMPNN features take **0.46 s/candidate** (933 s for 2031 on
CPU), against ~238 s for the Boltz-2 fold + Rosetta refinement the proxy exists to avoid.
The expensive autoregressive feature is 0.19 % of a fold, so affordability is not a
consideration -- include it.

Scope limit, stated because it is easy to over-claim: every training candidate is a 1-7
substitution variant of ONE reference peptide per target. The model interpolates inside a
small mutational ball around a reference, which is the search-triage use case. It is not a
general peptide-dG predictor, and it needs a native complex for the target to compute the
MPNN features at all.

Energies are Rosetta ref2015 REU, NOT measured binding free energies. Lower is better.

Usage:
    python script_utils/cpseq_meanscore_fit_final.py \
        --dataset .../dataset_v3.json \
        --extra-features .../mpnn_feats_bb_v3.json .../mpnn_feats_ar_v3.json \
        --out .../proxy_model_v3.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_meanscore_baseline import AA, CHARGE, KD  # noqa: E402
from cpseq_meanscore_scorer import (  # noqa: E402
    aggregate_label,
    extra_matrix,
    load_extra_features,
)

ALPHAS = np.logspace(-2, 4, 25)
COMP_NAMES = [f"dcomp_{a}" for a in AA] + ["d_kd_mean", "d_charge", "n_mutations"]


def comp_delta(r) -> list[float]:
    s, o = r["sequence"], r["original_sequence"]
    d = [(s.count(a) - o.count(a)) / len(s) for a in AA]
    return d + [
        st.mean(KD.get(c, 0.0) for c in s) - st.mean(KD.get(c, 0.0) for c in o),
        sum(CHARGE.get(c, 0.0) for c in s) - sum(CHARGE.get(c, 0.0) for c in o),
        float(r.get("n_mutations") or 0),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--extra-features", nargs="*", default=[], type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--agg", default="trimmed20")
    args = ap.parse_args()

    payload = json.loads(args.dataset.read_text())
    rows = [r for r in payload["rows"] if r.get("arm") != "reference"]
    rows = [r for r in rows if r.get("original_sequence")
            and len(r["sequence"]) == len(r["original_sequence"])
            and any(a != b for a, b in zip(r["sequence"], r["original_sequence"]))]

    names: list[str] = []
    merged = {}
    if args.extra_features:
        merged, names, dropped, have = load_extra_features(args.extra_features, rows)
        rows = [r for r, h in zip(rows, have) if h]
        if dropped:
            print(f"dropped (constant or absent): {dropped}")

    parents = [r["parent_id"] for r in rows]
    pa = np.asarray(parents)
    targets = sorted(set(parents))
    y = np.array([aggregate_label(r["fold_dG"], args.agg) for r in rows])
    Xc = np.array([comp_delta(r) for r in rows], dtype=float)
    feat_names = list(COMP_NAMES)
    X = Xc
    if names:
        X = np.hstack([Xc, extra_matrix(merged, names, rows, parents)])
        feat_names += [f"{n}__target_centered" for n in names]

    # Per-target centring of the label: the model must learn the WITHIN-target contrast.
    # Between-target label SD is 12.3 REU against 3.76 within, so without this the fit is
    # mostly a target-identity readout that cannot generalise to a held-out target.
    yt = y.copy()
    offsets = {}
    for p in targets:
        m = pa == p
        offsets[p] = float(yt[m].mean())
        yt[m] -= offsets[p]

    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    model = RidgeCV(alphas=ALPHAS)
    model.fit((X - mu) / sd, yt)
    pred = model.predict((X - mu) / sd)

    in_sample = []
    for p in targets:
        m = pa == p
        if m.sum() >= 4 and len(set(pred[m])) > 1:
            rr = spearmanr(y[m], pred[m]).statistic
            if rr == rr:
                in_sample.append(rr)

    print(f"dataset     {args.dataset}")
    print(f"rows        {len(rows)} over {len(targets)} targets")
    print(f"label       {args.agg}, per-target centred (lower REU = better)")
    print(f"features    {len(feat_names)}  ({'with' if names else 'without'} MPNN)")
    print(f"alpha       {model.alpha_:.4g}")
    print(f"in-sample within-target rho  {st.mean(in_sample):.3f}   "
          f"<-- NOT the expected performance")
    print(f"honest expectation: LOTO rho 0.257 for an unseen target, 0.287 within a "
          f"target\n            whose journal already holds scored sequences.\n")

    # Per-target centring makes each `*_delta` column IDENTICAL to its raw column: the
    # reference value is constant within a target, so subtracting it is absorbed by the
    # centring. The ridge then splits one effect across two columns, so reading a single
    # coefficient understates it by half. Detected and reported rather than silently
    # deduplicated, because dropping a duplicated column changes the effective ridge
    # penalty on that direction and would move every number already measured.
    dupes = []
    Xs = (X - mu) / sd
    for i in range(Xs.shape[1]):
        for j in range(i + 1, Xs.shape[1]):
            if np.std(Xs[:, i]) > 0 and np.std(Xs[:, j]) > 0:
                if abs(np.corrcoef(Xs[:, i], Xs[:, j])[0, 1]) > 0.9999:
                    dupes.append((feat_names[i], feat_names[j]))
    if dupes:
        print("\nCOLLINEAR PAIRS (r > 0.9999) -- their coefficients SUM to one effect:")
        for a, b in dupes:
            print(f"  {a}  ==  {b}")
        print("  Effective feature count is lower than the nominal one; a future run "
              "should\n  drop the redundant column, which also restores the intended "
              "ridge penalty.")

    coef = dict(zip(feat_names, model.coef_.tolist()))
    top = sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:14]
    print("largest standardised coefficients (sign: negative = lowers predicted dG = good)")
    for n, c in top:
        print(f"  {n:34s} {c:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "kind": "ridge",
        "label_agg": args.agg,
        "label_units": "Rosetta ref2015 REU, per-target centred; lower is better",
        "feature_names": feat_names,
        "extra_feature_names": names,
        "mean": mu.tolist(), "scale": sd.tolist(),
        "coef": model.coef_.tolist(), "intercept": float(model.intercept_),
        "alpha": float(model.alpha_),
        "n_rows": len(rows), "n_targets": len(targets),
        "targets": targets,
        "collinear_pairs": [list(d) for d in dupes],
        "n_effective_features": len(feat_names) - len(dupes),
        "in_sample_within_target_rho": st.mean(in_sample),
        "expected_loto_rho": 0.257,
        "expected_within_target_rho": 0.287,
        "label_ceiling_sqrt_reliability": 0.896,
        "apply": ("standardise x with mean/scale, dot with coef, add intercept; "
                  "the result is a per-target CENTRED dG, so it ranks candidates within "
                  "one target and is NOT comparable across targets"),
        "scope": ("trained only on 1-7 substitution variants of one reference peptide "
                  "per target; requires a native complex for the target to compute the "
                  "ProteinMPNN features"),
    }, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
