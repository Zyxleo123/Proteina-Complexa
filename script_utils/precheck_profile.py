"""Reference feature distributions over LNR, PepBench and CPSea cyclics (pre-build check 3b).

Two sub-commands:

  profile    shard a staged set, compute `peptide_profile.profile()` per complex, write JSONL.
  reference  aggregate the shards into the serialized reference-distribution artifact that
             the section-5 corruption sampler will later draw target feature values from.

**LNR is reference-only.**  It is development data that has already influenced this
repository's preprocessing and sampling decisions, so it is profiled and reported but never
used to tune anything.  The artifact marks it so downstream code cannot reach for it by
accident.

**Every set is cropped with the same operation.**  PepBench receptors are whole chains and
CPSea receptors are pocket crops, and four of the profile features are sensitive to that
(README_POSE_DECOY_INVENTORY 2.5).  Cropping only PepBench would swap one asymmetry for
another, so `profile.crop_radius_A` is applied uniformly and the staging vector is recorded
both before and after so the effect is visible rather than assumed.

Runs in `.venv`.  CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script_utils import precheck_config  # noqa: E402
from script_utils.peptide_profile import (  # noqa: E402
    FEATURES,
    STAGING_FEATURES,
    crop_receptor,
    load_complex,
    profile,
    staging_profile,
)

QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def select_rows(meta_path: str, source_filter: str, max_examples: int, seed: int) -> pd.DataFrame:
    """Cluster-unique sample from a staged metadata parquet.

    Cluster-unique because these sets are family-redundant: sampling rows uniformly would
    let one large family dominate a distribution that is then quoted as the set's.

    `cpsea_train` is 2.44M rows over 555k clusters, so the obvious implementation --
    shuffle everything, then drop duplicate clusters -- costs minutes of wall clock and
    several GB per shard, paid again by every shard in the array.  Instead the cluster list
    is drawn down FIRST and only the surviving rows are materialised.  The chosen row
    within a cluster is picked by a seeded hash rather than by file order, so the selection
    is unbiased and reproducible without a global shuffle.
    """
    want = ["example_id", "path", "cluster_id", "peptide_length", "receptor_length",
            "cyclization_type", "dataset_source", "pdb_id", "nc_gap_angstrom"]
    have = set(pq.ParquetFile(meta_path).schema.names)
    df = pq.read_table(meta_path, columns=[c for c in want if c in have]).to_pandas()

    if source_filter and "dataset_source" in df.columns:
        df = df[df["dataset_source"] == source_filter]
    if df.empty:
        raise SystemExit(f"{meta_path}: no rows after source_filter={source_filter!r}")

    rng = np.random.default_rng(seed)
    if "cluster_id" in df.columns:
        clusters = df["cluster_id"].dropna().unique()
        if max_examples and len(clusters) > max_examples:
            keep = set(rng.choice(clusters, size=max_examples, replace=False).tolist())
            df = df[df["cluster_id"].isin(keep)]
        key = pd.util.hash_pandas_object(
            df["example_id"].astype(str) + f"|{seed}", index=False)
        df = (df.assign(_k=key.to_numpy())
                .sort_values("_k")
                .drop_duplicates(subset="cluster_id", keep="first")
                .drop(columns="_k"))

    if max_examples and len(df) > max_examples:
        df = df.iloc[rng.permutation(len(df))[:max_examples]]
    return df.reset_index(drop=True)


def run_profile(args, cfg: dict) -> None:
    pcfg = cfg["profile"]
    sets = pcfg["sets"]
    if args.set not in sets:
        raise SystemExit(f"unknown set {args.set!r}; config has {sorted(sets)}")
    scfg = sets[args.set]
    radius = float((cfg.get("adversary") or {}).get("pocket_crop_radius_A", 0.0) or 0.0)

    df = select_rows(scfg["metadata"], scfg.get("source_filter", ""),
                     int(scfg.get("max_examples", 0)), int(pcfg.get("seed", 0)))
    df = df.iloc[args.shard::args.n_shards]
    if args.limit:
        df = df.head(args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["example_id"])
                except Exception:
                    raise SystemExit(f"{out}: unparseable row while resuming; delete and rerun")

    t0 = time.perf_counter()
    n_ok = n_skip = 0
    with out.open("a") as fh:
        for _, r in df.iterrows():
            ex = str(r["example_id"])
            if ex in done:
                continue
            cx = load_complex(r["path"], example_id=ex)
            if cx is None:
                n_skip += 1
                continue
            pre = staging_profile(cx)
            if radius > 0:
                cx = crop_receptor(cx, radius)
            try:
                feats = profile(cx)
            except Exception as exc:                      # one bad record costs that record
                print(f"  skip {ex}: {type(exc).__name__}: {exc}", flush=True)
                n_skip += 1
                continue
            row = {"example_id": ex, "label": args.set, "path": str(r["path"]),
                   "cluster_id": r.get("cluster_id"),
                   "crop_radius_A": radius}
            row.update(feats)
            row.update(staging_profile(cx))
            # Staging BEFORE the crop, kept so the report can show what the crop removed
            # rather than asserting that it worked.
            row.update({f"pre_{k}": v for k, v in pre.items()})
            fh.write(json.dumps(row, default=float) + "\n")
            n_ok += 1
            if n_ok % 50 == 0:
                fh.flush()
                print(f"  [{n_ok}] {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"wrote {out}: {n_ok} profiled, {n_skip} skipped, {time.perf_counter() - t0:.0f}s")


def run_reference(args, cfg: dict) -> None:
    """Aggregate shards into the serialized reference-distribution artifact."""
    in_dir = Path(args.profile_dir)
    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no profile shards under {in_dir}")

    rows, bad = [], []
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                bad.append(f"{f.name}:{n}: {type(exc).__name__}: {exc}")
    if bad:
        raise SystemExit(
            "REFUSING to aggregate: {} unparseable row(s). Skipping them would turn lost "
            "data into a smaller-but-plausible distribution.\n  {}".format(
                len(bad), "\n  ".join(bad[:10])))

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "profile_rows.parquet", index=False)

    ref: dict = {
        "schema": 1,
        "features": list(FEATURES),
        "staging_features": list(STAGING_FEATURES),
        "quantiles": list(QUANTILES),
        "crop_radius_A": float(df["crop_radius_A"].iloc[0]) if len(df) else None,
        "raw_rows": str((out_dir / "profile_rows.parquet").resolve()),
        "sets": {},
        # LNR is development data that has already shaped this repo's preprocessing and
        # sampling decisions. Marked so a downstream sampler cannot draw targets from it.
        "reference_only": ["lnr"],
    }
    for label, g in df.groupby("label"):
        entry = {"n": int(len(g)), "features": {}, "staging": {}}
        for feat in FEATURES:
            v = pd.to_numeric(g[feat], errors="coerce").dropna().to_numpy()
            if v.size == 0:
                entry["features"][feat] = {"n": 0}
                continue
            entry["features"][feat] = {
                "n": int(v.size),
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "min": float(v.min()),
                "max": float(v.max()),
                "q": {str(q): float(np.percentile(v, q)) for q in QUANTILES},
            }
        for feat in STAGING_FEATURES:
            if feat not in g:
                continue
            v = pd.to_numeric(g[feat], errors="coerce").dropna().to_numpy()
            if v.size:
                entry["staging"][feat] = {
                    "median": float(np.median(v)),
                    "q": {str(q): float(np.percentile(v, q)) for q in QUANTILES},
                }
        ref["sets"][label] = entry

    (out_dir / "reference_distributions.json").write_text(json.dumps(ref, indent=2))
    print(f"wrote {out_dir}/reference_distributions.json "
          f"({', '.join(f'{k}={v['n']}' for k, v in ref['sets'].items())})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile")
    p.add_argument("--config", required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)

    r = sub.add_parser("reference")
    r.add_argument("--config", required=True)
    r.add_argument("--profile-dir", required=True)
    r.add_argument("--out-dir", required=True)

    args = ap.parse_args()
    cfg = precheck_config.load(args.config)
    (run_profile if args.cmd == "profile" else run_reference)(args, cfg)


if __name__ == "__main__":
    main()
