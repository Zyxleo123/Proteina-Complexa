"""ARM B -- re-stage the LNR targets with CPSea's receptor convention (pocket crop).

The defect this fixes
---------------------
CPSea's preprocessed PDBs store the receptor ALREADY spatially cropped to the binding
pocket, with the original residue numbering preserved. Measured on cpsea_val:

    chain A: 144 CA numbered 1..307 with 16 numbering GAPS
    max over kept receptor residues of min-distance-to-binder: 17-20 A (p95 ~16.8)

Those numbering gaps are not cosmetic. `SegmentAwareResidueFeaturesTransform` turns them
into `effective_chain_id` / `pos_in_segment`, and the denoiser consumes them through
`chain_idx_pair` and `rel_seq_sep`. Every CPSea training example therefore presents the
receptor as ~13-18 disjoint fragments.

The LNR staging (scripts/build_lnr_metadata.py) faithfully preserved each source PDB's own
numbering -- but LNR ships COMPLETE receptor chains, which are naturally contiguous. So the
LNR targets reach the model as ~1 fragment with `pos_in_segment` running past 800, a
receptor representation the model never saw in training. `CroppingTransform2` does not
repair this: it trims a little (279 -> 246) but cannot invent the missing gaps.

What this does
--------------
Rewrites each staged LNR PDB keeping only receptor residues within `--radius` Angstrom
(heavy-atom to heavy-atom) of the peptide, and PRESERVES the original residue numbers so
the dropped residues show up as numbering gaps -- exactly CPSea's convention. The peptide
chain is copied through untouched.

Coordinates are never transformed, so the bound pose is preserved bit-for-bit and the
re-staged set differs from the original in exactly one respect: which receptor residues are
present. That makes the Arm B vs original-uncond comparison single-variable.
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_pdb(path: Path):
    """-> (header_lines, {chain: OrderedDict[resseq -> [line, ...]]}) for ATOM records."""
    header, chains = [], {}
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith(("REMARK", "HEADER", "TITLE", "CRYST")):
            header.append(line)
            continue
        if not line.startswith("ATOM"):
            continue
        ch = line[21]
        resseq = int(line[22:26])
        chains.setdefault(ch, OrderedDict()).setdefault(resseq, []).append(line)
    return header, chains


def heavy_coords(lines):
    """Heavy-atom xyz for one residue. Hydrogens are excluded so the radius means the same
    thing regardless of whether a source file carries them."""
    out = []
    for l in lines:
        elem = l[76:78].strip() or l[12:16].strip()[:1]
        if elem == "H":
            continue
        out.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.asarray(out, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-metadata", required=True, help="Existing LNR staged parquet.")
    ap.add_argument("--out-dir", required=True, help="Directory for re-staged PDBs + metadata.")
    ap.add_argument("--radius", type=float, default=18.0,
                    help="Heavy-atom crop radius in Angstrom. Default 18.0, measured from "
                         "cpsea_val (max kept-residue min-distance 17-20 A).")
    ap.add_argument("--pdb-dir", default=None,
                    help="Read source PDBs from HERE (by basename) instead of the metadata's "
                         "own `path`. Used by Arm C to crop the RELAXED complexes while "
                         "reusing the original metadata rows unchanged.")
    ap.add_argument("--receptor-chain", default="A")
    ap.add_argument("--peptide-chain", default="B")
    ap.add_argument("--min-receptor-residues", type=int, default=30,
                    help="Matches the training crop's target_min_length. Examples whose pocket "
                         "falls below this are DROPPED and recorded, never silently padded.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pdb_out = out_dir / "pdbs"
    pdb_out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(args.in_metadata)
    rows, dropped = [], []
    seg_before, seg_after, kept_frac = [], [], []

    for _, r in meta.iterrows():
        src = Path(str(r["path"]))
        if args.pdb_dir:
            src = Path(args.pdb_dir) / src.name
        if not src.is_file():
            dropped.append((r["example_id"], f"source pdb missing: {src}"))
            continue
        header, chains = parse_pdb(src)
        rec = chains.get(args.receptor_chain)
        pep = chains.get(args.peptide_chain)
        if not rec or not pep:
            dropped.append((r["example_id"], "missing receptor or peptide chain"))
            continue

        pep_xyz = np.concatenate([heavy_coords(v) for v in pep.values()], axis=0)
        keep = []
        for resseq, lines in rec.items():
            R = heavy_coords(lines)
            if R.size == 0:
                continue
            d = np.sqrt(((R[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1)).min()
            if d <= args.radius:
                keep.append(resseq)
        if len(keep) < args.min_receptor_residues:
            dropped.append((r["example_id"], f"pocket too small ({len(keep)} residues)"))
            continue

        nums_before = sorted(rec.keys())
        seg_before.append(sum(1 for a, b in zip(nums_before, nums_before[1:]) if b != a + 1) + 1)
        seg_after.append(sum(1 for a, b in zip(keep, keep[1:]) if b != a + 1) + 1)
        kept_frac.append(len(keep) / len(rec))

        dst = pdb_out / src.name
        with dst.open("w") as fh:
            fh.writelines(header)
            fh.write(f"REMARK 999 ARM B: receptor cropped to {args.radius:.1f} A heavy-atom "
                     f"pocket; ORIGINAL residue numbering preserved (gaps are meaningful)\n")
            # Receptor first, then peptide -- chain order the CPSea loader expects.
            for resseq in keep:
                fh.writelines(rec[resseq])
            for resseq in pep:
                fh.writelines(pep[resseq])

        row = r.to_dict()
        row["path"] = str(dst.resolve())
        row["receptor_length"] = len(keep)
        rows.append(row)

    out_meta = out_dir / "metadata"
    out_meta.mkdir(parents=True, exist_ok=True)
    dest = out_meta / "lnr_test_pocket.parquet"
    pd.DataFrame(rows).to_parquet(dest, index=False)

    if dropped:
        adf = pd.DataFrame(dropped, columns=["example_id", "reason"])
        adf.to_csv(out_dir / "dropped.csv", index=False)

    print(f"re-staged {len(rows)} of {len(meta)} examples -> {dest}")
    if dropped:
        print(f"dropped {len(dropped)} (see {out_dir/'dropped.csv'}):")
        for eid, why in dropped[:10]:
            print(f"    {eid}: {why}")
    if seg_after:
        print(f"\nreceptor contiguous segments  before: median "
              f"{np.median(seg_before):.1f}   after: median {np.median(seg_after):.1f}")
        print(f"receptor residues kept: median {np.median(kept_frac)*100:.0f}%")
        print("CPSea val reference: median 13-18 segments")


if __name__ == "__main__":
    main()
