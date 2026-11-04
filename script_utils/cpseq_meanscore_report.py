#!/usr/bin/env python3
"""Collect every trained scorer config and say which one, if any, earned its place.

Reads `scorer_<agg>_<loss>.json` from the sweep plus `reliability_*.json`, and prints one
table. The verdict rule is written down here rather than decided while reading the table:

* The network has to beat the **best linear baseline** under the identical
  leave-one-target-out loop. The baselines are in each scorer JSON, measured in the same
  run, so this is not a cross-run comparison.
* It has to beat it by more than the **seed spread** of the network itself. A gap smaller
  than the spread across training seeds is not a result.
* Everything is reported as a fraction of `sqrt(reliability)`, the score a perfect
  predictor of the latent mean would get against a ~10-fold label.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    files = sorted(args.dir.glob("scorer_*.json"))
    files = [f for f in files if "smoke" not in f.name]
    if not files:
        print(f"GATE: no scorer_*.json under {args.dir} -- nothing to report.")
        print("      (If stage 2 ran, check its logs; if it never ran, check the chain.)")
        return

    rel_path = args.dir / "reliability_v3.json"
    rel = json.loads(rel_path.read_text()) if rel_path.exists() else None

    print("=" * 94)
    print("MEAN-dG PROXY SCORER -- sweep verdict")
    print("=" * 94)

    if rel:
        sw = rel["sweep"]
        order = sorted(sw, key=lambda n: -sw[n]["reliability_median"])
        print("\nTARGET-SCORE STATISTIC (which aggregate over the ~10 folds to regress on)")
        print(f"  {'aggregator':18s} {'rank_rel':>9s} {'95% CI':>16s} "
              f"{'ceiling':>8s} {'value_rel':>10s}")
        for n in order:
            s = sw[n]
            lo, hi = s["reliability_ci"]
            print(f"  {n:18s} {s['reliability_median']:9.3f}   "
                  f"[{lo:5.3f},{hi:5.3f}] {s['ceiling']:8.3f} "
                  f"{s.get('reliability_values', float('nan')):10.3f}")
        print(f"  recommended: {rel['recommended_aggregator']}")
        if rel.get("label_sd_2k_folds_REU") is not None:
            print(f"  label noise: +/-{rel['label_sd_2k_folds_REU']:.2f} REU on a "
                  f"{2*rel['k_per_half']}-fold label vs "
                  f"{rel['within_target_signal_sd_REU']:.2f} REU of within-target signal")
        curve = rel.get("reliability_vs_folds", {})
        best_agg = rel["recommended_aggregator"]
        if best_agg in curve:
            pts = sorted((int(k), v["reliability"]) for k, v in curve[best_agg].items())
            print("  reliability vs folds: "
                  + "  ".join(f"n={k}:{v:.3f}" for k, v in pts))
        if rel.get("sb_extrapolation"):
            ex = rel["sb_extrapolation"]
            print("  if more folds were bought: "
                  + "  ".join(f"n={k}:{float(v):.3f}" for k, v in sorted(
                      ex.items(), key=lambda kv: int(kv[0]))))

    # ---- what is each pretrained feature worth, on its own? -------------------------
    # One scalar per candidate, scored by the same within-target metric. This is the
    # direct answer to "do pretrained weights help here", independent of any model.
    feat_rho = {}
    ds_path = args.dir / "dataset_v3.json"
    feat_files = sorted(args.dir.glob("mpnn_feats_*.json")) + \
        sorted(args.dir.glob("mlm_feats_*.json"))
    if ds_path.exists() and feat_files:
        import collections

        from scipy.stats import spearmanr
        ds = json.loads(ds_path.read_text())

        def agg_trim20(vals):
            s = sorted(min(v, 0.0) for v in vals)
            c = int(len(s) * 0.20)
            keep = s[c: len(s) - c] if len(s) - 2 * c >= 1 else s
            return st.mean(keep)

        labels, parents = {}, {}
        for r in ds["rows"]:
            if r.get("arm") == "reference":
                continue
            k = f"{r['parent_id']}|{r['sequence']}"
            labels[k] = agg_trim20(r["fold_dG"])
            parents[k] = r["parent_id"]

        print("\nPRETRAINED FEATURE VALUE (each scalar alone, within-target rho)")
        print("  sign convention: a log-likelihood is higher-is-better while dG is")
        print("  lower-is-better, so |rho| is the signal and a NEGATIVE rho agrees.")
        print(f"\n  {'feature':24s} {'n_tgt':>5s} {'rho':>7s} {'|rho|':>6s} {'source'}")
        for ff in feat_files:
            blob = json.loads(ff.read_text())
            feats = blob.get("features") or {}
            if not feats:
                continue
            for name in blob.get("feature_names", []):
                by_t = collections.defaultdict(list)
                for k, v in feats.items():
                    val = v.get(name)
                    if k in labels and isinstance(val, (int, float)) and val == val:
                        by_t[parents[k]].append((labels[k], val))
                rhos = []
                for vals in by_t.values():
                    if len(vals) < 4:
                        continue
                    a = [x for x, _ in vals]
                    b = [y for _, y in vals]
                    if len(set(b)) < 2:
                        continue
                    rr = spearmanr(a, b).statistic
                    if rr == rr:
                        rhos.append(rr)
                if rhos:
                    m = st.mean(rhos)
                    feat_rho[name] = {"rho": m, "n_targets": len(rhos),
                                      "source": ff.name}
                    print(f"  {name:24s} {len(rhos):5d} {m:7.3f} {abs(m):6.3f} "
                          f"{ff.name}")
        esm_path = args.dir / "esm_probe_v3.json"
        if esm_path.exists():
            print(f"\n  (frozen ESM-2 probe results are in {esm_path.name}; it fits a "
                  f"ridge on\n   embeddings rather than exposing one scalar, so it is "
                  f"not in this table.)")

    rows = []
    for f in files:
        d = json.loads(f.read_text())
        loto = d.get("loto") or {}
        within = d.get("within_target") or {}
        base = d.get("linear_baselines") or {}
        best_base = max(base.items(), key=lambda kv: kv[1]["rho"]) if base else (None, {"rho": float("nan")})
        per_t = loto.get("per_target") or {}
        seed_spread = float("nan")
        if per_t:
            spreads = [max(v["rho_per_seed"]) - min(v["rho_per_seed"])
                       for v in per_t.values() if v.get("rho_per_seed")]
            seed_spread = st.median(spreads) if spreads else float("nan")
        # The arm is the filename suffix the train job's MEANSCORE_TAG set, because the
        # sequence-only and feature-augmented sweeps are the same configs and would
        # otherwise be indistinguishable rows.
        stem = f.stem
        arm = "seq-only"
        for agg_name in ("trimmed20", "mean_clip0", "median_clip0", "hodges_lehmann"):
            if agg_name in stem:
                tail = stem.split(agg_name, 1)[1]
                parts = tail.split("_")
                if len(parts) > 2 and parts[2]:
                    arm = parts[2]
                break
        rows.append({
            "file": f.name,
            "arm": arm,
            "extra_features": d.get("extra_features") or [],
            "agg": d.get("label_agg"),
            "loss": (d.get("config") or {}).get("loss"),
            "ceiling": (d.get("ceiling") or {}).get("sqrt_reliability", float("nan")),
            "loto": loto.get("mean_rho_ensemble", float("nan")),
            "loto_single": loto.get("mean_rho_single_seed", float("nan")),
            "within": within.get("rho", float("nan")),
            "best_base_name": best_base[0],
            "best_base": best_base[1]["rho"],
            "seed_spread": seed_spread,
        })

    rows.sort(key=lambda r: -(r["loto"] if r["loto"] == r["loto"] else -9))
    ceil = rows[0]["ceiling"]
    print(f"\nSCORER SWEEP   (ceiling = {ceil:.3f}; all rho are within-target, "
          f"averaged over targets)")
    print(f"  {'arm':9s} {'agg':11s} {'loss':5s} {'LOTO':>7s} {'%ceil':>6s} "
          f"{'1seed':>7s} {'WITHIN':>7s} {'%ceil':>6s} {'bestbase':>9s} {'gap':>7s}")
    for r in rows:
        gap = r["loto"] - r["best_base"]
        print(f"  {str(r['arm']):9s} {str(r['agg']):11s} {str(r['loss']):5s} "
              f"{r['loto']:7.3f} {100*r['loto']/ceil:6.1f} {r['loto_single']:7.3f} "
              f"{r['within']:7.3f} {100*r['within']/ceil:6.1f} "
              f"{r['best_base']:9.3f} {gap:+7.3f}")

    # The paired arm comparison is the point of running both sweeps, so it is computed
    # rather than left for the reader: same (agg, loss), with and without the features.
    paired = {}
    for r in rows:
        paired.setdefault((r["agg"], r["loss"]), {})[r["arm"]] = r
    deltas = [(k, v["seq-only"]["loto"], v[a]["loto"], a)
              for k, v in paired.items() for a in v if a != "seq-only"
              and "seq-only" in v]
    if deltas:
        print("\nPAIRED ARM DELTA (same agg+loss, features added)")
        print(f"  {'agg':11s} {'loss':5s} {'arm':9s} {'seq-only':>9s} "
              f"{'augmented':>10s} {'delta':>7s}")
        for (agg_n, loss_n), a, b, arm in sorted(deltas):
            print(f"  {str(agg_n):11s} {str(loss_n):5s} {arm:9s} {a:9.3f} "
                  f"{b:10.3f} {b - a:+7.3f}")
        mean_d = st.mean(b - a for _k, a, b, _arm in deltas)
        print(f"  mean delta over {len(deltas)} pairs: {mean_d:+.3f}")

    best = rows[0]
    print(f"\nbest linear baseline in-run: {best['best_base_name']} "
          f"rho {best['best_base']:.3f}")
    print(f"median seed spread per target: {best['seed_spread']:.3f}")

    gap = best["loto"] - best["best_base"]
    print("\nVERDICT")
    if gap != gap:
        verdict = "INCONCLUSIVE -- missing numbers"
        print("  INCONCLUSIVE: a required number is missing.")
    elif gap > best["seed_spread"]:
        verdict = "network beats the baseline by more than its own seed spread"
        print(f"  The network ({best['loto']:.3f}) beats the best linear baseline "
              f"({best['best_base']:.3f}) by {gap:+.3f},")
        print(f"  which exceeds its median seed spread ({best['seed_spread']:.3f}). "
              f"The architecture is earning something.")
    elif gap > 0:
        verdict = "network ahead but inside its own seed spread -- not a result"
        print(f"  The network is ahead by {gap:+.3f}, but its median seed spread is "
              f"{best['seed_spread']:.3f}.")
        print(f"  That gap is inside the noise of its own training. Not a result: a "
              f"{best['best_base_name']} ridge is the honest scorer so far.")
    else:
        verdict = "network does NOT beat the linear baseline"
        print(f"  The network does NOT beat the linear baseline ({gap:+.3f}).")
        print(f"  On this data a {best['best_base_name']} ridge is the scorer. The "
              f"mutation-effect architecture is not what is limiting it.")

    print(f"\n  Deployment note: LOTO {best['loto']:.3f} is the number for a target never "
          f"scored;")
    print(f"  WITHIN {best['within']:.3f} is the number for a target whose journal "
          f"already holds scored")
    print(f"  sequences, which is the setting a search actually runs in.")

    if args.json:
        args.json.write_text(json.dumps(
            {"verdict": verdict, "ceiling": ceil, "rows": rows,
             "pretrained_feature_rho": feat_rho}, indent=1, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
