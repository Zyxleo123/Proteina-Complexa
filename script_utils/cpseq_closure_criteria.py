#!/usr/bin/env python3
"""Measure the published cyclization-closure criteria side by side on one set of structures.

Four criteria, applied identically to experimental crystal macrocycles, native CPSea
entries, and this project's CP generators, so the numbers are comparable:

| criterion | quantity | window |
|---|---|---|
| CPSea broad screen | terminal Cb-Cb | 3-8 A |
| CPSea chemistry assignment | terminal Cb-Cb | disulfide [3,4.5), head-to-tail [4.5,6), isopeptide [6,8] |
| MuCO | terminal N-C (after minimisation) | 1.33 +/- 0.10 A |
| this project (frozen) | terminal N-C | 1.1-1.6 A |

Why all four rather than one. CPSea's Cb screen is a **pre-refinement chemistry
assignment**: it decides which bond to build, then CPSea modifies the terminal residues,
adds the covalent bond and minimises. So passing it does not state that a backbone ring is
closed -- and this project has twice documented that `cyc_cb_window_success` over the broad
3-8 A range saturates (`README_ANCHOR_GRAFT.md`, `README_M15_CEILING_AUDIT.md`). MuCO's
criterion is the opposite: it checks the actual bonding atoms after minimisation, and is
tighter than this project's own frozen window. Reporting both directions is the only way a
reader can tell "the anchors are positioned for a ring" from "the ring is closed".

Note that CPSea's head-to-tail band is **4.5-6 A**, not the whole 3-8 A screen. A sample
at Cb 3.2 A passes the broad screen while being assigned DISULFIDE chemistry by the same
code -- which, for a head-to-tail project, is a miss, not a pass.

Cb is the first side-chain carbon; glycine gets the standard virtual Cb built from N, CA
and C. The head-to-tail bond atoms are N of the first residue and C of the last.

Usage:
    python script_utils/cpseq_closure_criteria.py --n-cpsea 150 \
        --out /data/user_data/yixiz/cpseq/search_pilot/tables/closure_criteria.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.geometry import read_chain, sequence_of  # noqa: E402

# CPSea's released postprocessing bands, on the terminal Cb-Cb distance.
CPSEA_BROAD = (3.0, 8.0)
CPSEA_BANDS = (
    ("disulfide", 3.0, 4.5),
    ("head_tail", 4.5, 6.0),
    ("isopeptide", 6.0, 8.0),
)
# MuCO: the actual bonding atoms after minimisation.
MUCO_NC = (1.33 - 0.10, 1.33 + 0.10)
MUCO_SS = (2.05 - 0.10, 2.05 + 0.10)
# This project's frozen mainchain bond window.
FROZEN_NC = (1.1, 1.6)


def virtual_cb(res: dict) -> np.ndarray:
    """Standard virtual Cb from N, CA, C. Used for glycine and any missing Cb."""
    N, CA, C = res["N"], res["CA"], res["C"]
    b, c = CA - N, C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def measure(pdb_path: Path, chain: str) -> dict | None:
    """Terminal Cb-Cb and head-to-tail N-C for one structure."""
    try:
        residues = read_chain(pdb_path, chain)
    except Exception as exc:  # noqa: BLE001
        return {"status": f"read_failed:{type(exc).__name__}"}
    if len(residues) < 2:
        return {"status": "too_few_residues"}
    (n0, first), (nL, last) = residues[0], residues[-1]

    out: dict = {
        "status": "ok",
        "length": len(residues),
        "sequence": sequence_of(residues),
        "first_res": n0,
        "last_res": nL,
    }
    try:
        cb0 = first["CB"] if "CB" in first else virtual_cb(first)
        cbL = last["CB"] if "CB" in last else virtual_cb(last)
        out["cb_source"] = (
            f"{'real' if 'CB' in first else 'virtual'}/"
            f"{'real' if 'CB' in last else 'virtual'}"
        )
        out["cb_cb_A"] = round(float(np.linalg.norm(cb0 - cbL)), 3)
    except KeyError as exc:
        out["cb_cb_A"] = None
        out["cb_source"] = f"missing:{exc}"

    # Head-to-tail bond: N of the first residue to C of the last.
    try:
        out["nc_A"] = round(float(np.linalg.norm(first["N"] - last["C"])), 3)
    except KeyError as exc:
        out["nc_A"] = None
        out["nc_missing"] = str(exc)

    # Disulfide bond: SG of the first terminal CYS to SG of the last. Measured whenever
    # both termini carry an SG, so a head-to-tail entry that happens to have terminal
    # cysteines is still reported rather than silently skipped.
    if "SG" in first and "SG" in last:
        out["ss_A"] = round(float(np.linalg.norm(first["SG"] - last["SG"])), 3)
    else:
        out["ss_A"] = None
    return out


def verdicts(m: dict) -> dict:
    d, nc, ss = m.get("cb_cb_A"), m.get("nc_A"), m.get("ss_A")
    band = None
    if d is not None:
        for name, lo, hi in CPSEA_BANDS:
            if lo <= d < hi or (name == "isopeptide" and lo <= d <= hi):
                band = name
                break
    return {
        "cpsea_broad_pass": None if d is None else bool(CPSEA_BROAD[0] <= d <= CPSEA_BROAD[1]),
        "cpsea_band": band,
        "cpsea_head_tail_band_pass": band == "head_tail",
        "muco_nc_pass": None if nc is None else bool(MUCO_NC[0] <= nc <= MUCO_NC[1]),
        "frozen_nc_pass": None if nc is None else bool(FROZEN_NC[0] <= nc <= FROZEN_NC[1]),
        # MuCO's disulfide criterion, on the actual bonding atoms.
        "muco_ss_pass": None if ss is None else bool(MUCO_SS[0] <= ss <= MUCO_SS[1]),
    }


def summarise(name: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    def frac(key):
        vals = [r["verdicts"][key] for r in ok if r["verdicts"].get(key) is not None]
        return (round(sum(vals) / len(vals), 3), len(vals)) if vals else (None, 0)
    cb = [r["cb_cb_A"] for r in ok if r.get("cb_cb_A") is not None]
    nc = [r["nc_A"] for r in ok if r.get("nc_A") is not None]
    ss = [r["ss_A"] for r in ok if r.get("ss_A") is not None]
    bands: dict[str, int] = {}
    for r in ok:
        b = r["verdicts"].get("cpsea_band") or "outside_3_8"
        bands[b] = bands.get(b, 0) + 1
    muco_ss, n_ss = frac("muco_ss_pass")
    broad, n_broad = frac("cpsea_broad_pass")
    ht, _ = frac("cpsea_head_tail_band_pass")
    muco, n_muco = frac("muco_nc_pass")
    frozen, _ = frac("frozen_nc_pass")
    return {
        "source": name,
        "n": len(rows),
        "n_measured": len(ok),
        "cb_cb_median_A": round(float(np.median(cb)), 3) if cb else None,
        "nc_median_A": round(float(np.median(nc)), 3) if nc else None,
        "cpsea_broad_3_8_pass": broad,
        "cpsea_band_counts": bands,
        "cpsea_head_tail_4p5_6_pass": ht,
        "muco_nc_1p33_pm_0p10_pass": muco,
        "frozen_nc_1p1_1p6_pass": frozen,
        "n_with_nc": n_muco,
        "ss_median_A": round(float(np.median(ss)), 3) if ss else None,
        "muco_ss_2p05_pm_0p10_pass": muco_ss,
        "n_with_ss": n_ss,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cpsea", type=int, default=150,
                    help="native CPSea head_tail structures to sample (keep modest)")
    ap.add_argument("--n-cpsea-disulfide", type=int, default=100,
                    help="native CPSea disulfide structures to sample; 0 to skip. These "
                         "carry the MuCO S-S criterion and CPSea's own disulfide Cb band.")
    ap.add_argument("--cpsea-metadata",
                    default="/data/user_data/yixiz/CPSea/CPSea_full/CPSea/preprocessed/metadata/cpsea_val.parquet")
    ap.add_argument("--controls", default="/data/user_data/yixiz/cpseq/controls/controls.json")
    ap.add_argument("--backbones-root",
                    default="/data/user_data/yixiz/cpseq/search_pilot/backbones")
    ap.add_argument("--dump", action="append", default=[], metavar="LABEL=PATH",
                    help="extra rosetta_dump directory, labelled. Repeatable. Use this to "
                         "measure a model in ITS OWN conditioning mode: an unconditional "
                         "model's unconditional dump, a sequence-conditioned model's "
                         "seq-conditioned dump. Measuring a seq-cond model unconditionally "
                         "scores it off-mode and is not a fair comparison.")
    ap.add_argument("--peptide-chain", default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    per_source: dict[str, list[dict]] = {}

    # ---- experimental crystal macrocycles -------------------------------------------
    ctrl_path = Path(args.controls)
    if ctrl_path.exists():
        rows = []
        for c in json.loads(ctrl_path.read_text()):
            if c.get("chemistry") != "mainchain":
                continue
            m = measure(Path(c["structure_path"]), c.get("peptide_chain", "B"))
            if m:
                m |= {"id": c.get("pdb_id"), "verdicts": verdicts(m)}
                rows.append(m)
        per_source["experimental_crystal_mainchain"] = rows

    # ---- native CPSea head_tail ------------------------------------------------------
    meta = Path(args.cpsea_metadata)
    if meta.exists():
        df = pd.read_parquet(meta)
        ht = df[df.cyclization_type == "head_tail"]
        # Cluster-unique so the sample is not many copies of one macrocycle.
        if "cluster_id" in ht.columns:
            ht = ht.drop_duplicates("cluster_id")
        ht = ht.sample(n=min(args.n_cpsea, len(ht)), random_state=args.seed)
        rows = []
        for _, r in ht.iterrows():
            p = Path(r["path"])
            if not p.exists():
                continue
            m = measure(p, str(r.get("binder_chain_id", "B")))
            if m:
                m |= {"id": r["example_id"], "verdicts": verdicts(m)}
                rows.append(m)
        per_source["native_cpsea_head_tail"] = rows

        if args.n_cpsea_disulfide:
            ds = df[df.cyclization_type == "disulfide"]
            if "cluster_id" in ds.columns:
                ds = ds.drop_duplicates("cluster_id")
            ds = ds.sample(n=min(args.n_cpsea_disulfide, len(ds)), random_state=args.seed)
            drows = []
            for _, r in ds.iterrows():
                q = Path(r["path"])
                if not q.exists():
                    continue
                m = measure(q, str(r.get("binder_chain_id", "B")))
                if m:
                    m |= {"id": r["example_id"], "verdicts": verdicts(m)}
                    drows.append(m)
            per_source["native_cpsea_disulfide"] = drows

    # ---- this project's generators ---------------------------------------------------
    root = Path(args.backbones_root)
    for gen_dir in sorted(root.glob("gen_*")):
        pdbs = sorted(
            p for p in gen_dir.glob("rosetta_dump/step_*/*.pdb")
            if not p.name.endswith(("_refined.pdb", "_isolated_relaxed.pdb"))
        )
        if not pdbs:
            continue
        rows = []
        for p in pdbs:
            m = measure(p, "B")
            if m:
                m |= {"id": p.stem, "verdicts": verdicts(m)}
                rows.append(m)
        per_source[gen_dir.name] = rows

    # ---- explicitly labelled dumps, each in its own mode ----------------------------
    for spec in args.dump:
        if "=" not in spec:
            print(f"[criteria] --dump needs LABEL=PATH, got {spec!r}", file=sys.stderr)
            return 2
        label, path = spec.split("=", 1)
        d = Path(path)
        pdbs = sorted(
            q for q in (d.glob("step_*/*.pdb") if d.is_dir() else [])
            if not q.name.endswith(("_refined.pdb", "_isolated_relaxed.pdb"))
        )
        if not pdbs:
            print(f"[criteria] no structures under {d} for {label!r}")
            continue
        rows = []
        for q in pdbs:
            m = measure(q, args.peptide_chain)
            if m:
                m |= {"id": q.stem, "verdicts": verdicts(m)}
                rows.append(m)
        per_source[label] = rows

    summaries = [summarise(k, v) for k, v in per_source.items() if v]

    hdr = (f"{'source':34s} {'n':>4s} {'CB med':>7s} {'N-C med':>8s} "
           f"{'broad3-8':>9s} {'HT4.5-6':>8s} {'MuCO':>6s} {'frozen':>7s} "
           f"{'S-S med':>8s} {'MuCO_SS':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        print(f"{s['source']:34s} {s['n_measured']:4d} "
              f"{str(s['cb_cb_median_A']):>7s} {str(s['nc_median_A']):>8s} "
              f"{str(s['cpsea_broad_3_8_pass']):>9s} "
              f"{str(s['cpsea_head_tail_4p5_6_pass']):>8s} "
              f"{str(s['muco_nc_1p33_pm_0p10_pass']):>6s} "
              f"{str(s['frozen_nc_1p1_1p6_pass']):>7s} "
              f"{str(s['ss_median_A']):>8s} "
              f"{str(s['muco_ss_2p05_pm_0p10_pass']):>8s}")
    print("\nCPSea chemistry-band assignment (on terminal Cb-Cb):")
    for s in summaries:
        print(f"  {s['source']:34s} {s['cpsea_band_counts']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "criteria": {
                "cpsea_broad_cb_A": list(CPSEA_BROAD),
                "cpsea_bands_cb_A": [[n, lo, hi] for n, lo, hi in CPSEA_BANDS],
                "muco_nc_A": list(MUCO_NC),
                "muco_ss_A": list(MUCO_SS),
                "frozen_nc_A": list(FROZEN_NC),
                "note": "CPSea's Cb bands are a PRE-refinement chemistry assignment, not "
                        "a statement that a ring is closed; MuCO checks the bonding atoms "
                        "after minimisation.",
            },
            "summaries": summaries,
            "per_structure": per_source,
        }, indent=2, default=str))
        print(f"\n[criteria] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
