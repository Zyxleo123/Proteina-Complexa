"""Aggregate + plot the AE round-trip arms. CPU only -- reads saved rows, never the model.

Answers the go/no-go question: do LINEAR LNR peptides reconstruct through the CPSea
autoencoder as well as the cyclic CPSea peptides it was finetuned on?

Also quantifies the one confound that comparison has. The LNR arm is raw crystal
structures; the CPSea arm is relaxed models. Relaxed models have near-ideal bond geometry,
crystal structures do not, so part of any reconstruction gap could be input geometry rather
than linear-vs-cyclic. This script measures backbone-geometry ideality per arm directly from
the input PDBs, so the gap can be discounted rather than over-read.

Figures are drawn from the JSONL rows alone, so they re-render without re-running the GPU job.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HEADLINE = "ae/atom_rmse_A"
SEQ = "ae/seq_acc"

# Engh & Huber ideal backbone bond lengths (A).
IDEAL = {"N-CA": 1.458, "CA-C": 1.525, "C-N": 1.329}


def load_rows(paths: list[Path]) -> pd.DataFrame:
    """Loads JSONL rows, REFUSING corrupt lines rather than skipping them.

    Concurrent appends to a shared file have silently NUL-corrupted rows on this cluster
    before. A summary computed over the survivors of an unnoticed corruption is worse than
    no summary, so any unparseable line is a hard failure naming the file and line number.
    """
    records = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"FATAL: results file missing: {path}")
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            if "\x00" in line:
                raise SystemExit(f"FATAL: NUL byte in {path}:{lineno} -- file is corrupt, do not trust it.")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"FATAL: unparseable row {path}:{lineno}: {exc}")
    if not records:
        raise SystemExit("FATAL: no rows loaded.")
    return pd.DataFrame(records)


def backbone_geometry_deviation(pdb_path: Path, chain: str = "B") -> dict[str, float] | None:
    """RMS deviation of backbone bond lengths from ideal, over one chain of a PDB.

    A relaxed model sits near 0; a crystal structure does not. This is the confound probe,
    not a quality judgement.
    """
    residues: list[dict[str, tuple[float, float, float]]] = []
    current_key = None
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        key = line[22:27]
        name = line[12:16].strip()
        if name not in ("N", "CA", "C"):
            continue
        if key != current_key:
            residues.append({})
            current_key = key
        residues[-1][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    if len(residues) < 2:
        return None

    devs: dict[str, list[float]] = defaultdict(list)
    for i, res in enumerate(residues):
        if "N" in res and "CA" in res:
            devs["N-CA"].append(math.dist(res["N"], res["CA"]) - IDEAL["N-CA"])
        if "CA" in res and "C" in res:
            devs["CA-C"].append(math.dist(res["CA"], res["C"]) - IDEAL["CA-C"])
        if i + 1 < len(residues) and "C" in res and "N" in residues[i + 1]:
            d = math.dist(res["C"], residues[i + 1]["N"])
            if d < 2.0:  # only real peptide bonds; a chain break is not a geometry defect
                devs["C-N"].append(d - IDEAL["C-N"])
    out = {}
    all_dev: list[float] = []
    for bond, values in devs.items():
        if values:
            out[f"bond_rmsd_{bond}"] = float(np.sqrt(np.mean(np.square(values))))
            all_dev.extend(values)
    out["bond_rmsd_all"] = float(np.sqrt(np.mean(np.square(all_dev)))) if all_dev else float("nan")
    return out


def add_geometry_probe(df: pd.DataFrame, metadata_files: dict[str, Path]) -> pd.DataFrame:
    """Joins per-example backbone-ideality onto the rows, via each arm's metadata parquet."""
    lookup: dict[str, str] = {}
    for arm, meta_path in metadata_files.items():
        if not meta_path or not Path(meta_path).exists():
            print(f"  [geometry probe] skipping arm {arm}: metadata not found at {meta_path}")
            continue
        meta = pd.read_parquet(meta_path)
        lookup.update(dict(zip(meta["example_id"].astype(str), meta["path"].astype(str))))

    values = []
    for example_id in df["example_id"].astype(str):
        pdb = lookup.get(example_id)
        geom = backbone_geometry_deviation(Path(pdb)) if pdb and Path(pdb).exists() else None
        values.append(geom.get("bond_rmsd_all", float("nan")) if geom else float("nan"))
    df = df.copy()
    df["input_bond_rmsd_A"] = values
    return df


def describe(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
            "p10": float(s.quantile(0.10)), "p90": float(s.quantile(0.90)),
            "min": float(s.min()), "max": float(s.max())}


def make_figure(df: pd.DataFrame, out_png: Path) -> None:
    arms = sorted(df["arm"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    data = [df.loc[df.arm == a, HEADLINE].dropna().values for a in arms]
    ax.boxplot(data, tick_labels=arms, showfliers=True)
    ax.set_ylabel("all-atom RMSE (A)")
    ax.set_title("AE round-trip reconstruction error\n(lower = AE represents this input well)")
    ax.tick_params(axis="x", rotation=12)

    ax = axes[1]
    for arm in arms:
        sub = df[df.arm == arm]
        ax.scatter(sub["peptide_length"], sub[HEADLINE], s=18, alpha=0.7, label=arm)
    ax.set_xlabel("peptide length (residues)")
    ax.set_ylabel("all-atom RMSE (A)")
    ax.set_title("Reconstruction error vs length")
    ax.legend(fontsize=8)

    ax = axes[2]
    for arm in arms:
        sub = df[df.arm == arm]
        ax.scatter(sub["input_bond_rmsd_A"], sub[HEADLINE], s=18, alpha=0.7, label=arm)
    ax.set_xlabel("input backbone bond RMSD from ideal (A)")
    ax.set_ylabel("all-atom RMSE (A)")
    ax.set_title("Confound probe:\ncrystal-vs-relaxed input geometry")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"figure: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True, type=Path, help="JSONL files, one per arm.")
    ap.add_argument("--lnr-metadata", type=Path, default=None)
    ap.add_argument("--cpsea-metadata", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(args.results)
    n_failed = int((df.get("status", "ok") == "failed").sum()) if "status" in df else 0
    if n_failed:
        print(f"WARNING: {n_failed} examples failed to score:")
        for _, r in df[df.status == "failed"].iterrows():
            print(f"    {r['example_id']}: {r.get('error')}")
    df = df[df.get("status", "ok") == "ok"].copy()
    if HEADLINE not in df.columns:
        raise SystemExit(f"FATAL: results have no {HEADLINE} column. Columns: {sorted(df.columns)}")

    df = add_geometry_probe(df, {"lnr": args.lnr_metadata, "cpsea": args.cpsea_metadata})

    summary: dict = {"n_failed_to_score": n_failed, "arms": {}}
    for arm in sorted(df["arm"].unique()):
        sub = df[df.arm == arm]
        summary["arms"][arm] = {
            "n": int(len(sub)),
            "atom_rmse_A": describe(sub[HEADLINE]),
            "seq_acc": describe(sub[SEQ]) if SEQ in sub else {},
            "coord_mae_A": describe(sub["ae/coord_mae_A"]) if "ae/coord_mae_A" in sub else {},
            "input_bond_rmsd_A": describe(sub["input_bond_rmsd_A"]),
            "by_length_bucket": {
                str(bucket): describe(grp[HEADLINE])
                for bucket, grp in sub.groupby(pd.cut(sub["peptide_length"], [4, 8, 12, 16]), observed=True)
            },
        }

    arms = sorted(df["arm"].unique())
    if len(arms) == 2:
        a, b = arms
        va, vb = df.loc[df.arm == a, HEADLINE].dropna(), df.loc[df.arm == b, HEADLINE].dropna()
        summary["comparison"] = {
            "arms": [a, b],
            "median_atom_rmse_A": [float(va.median()), float(vb.median())],
            "median_ratio": float(vb.median() / va.median()) if va.median() else None,
            "note": ("Interpret alongside input_bond_rmsd_A: if the arms differ there, part of "
                     "any reconstruction gap is crystal-vs-relaxed input geometry, not linear-vs-cyclic."),
        }

    (args.out_dir / "ae_roundtrip_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    df.to_csv(args.out_dir / "ae_roundtrip_rows.csv", index=False)
    make_figure(df, args.out_dir / "ae_roundtrip.png")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
