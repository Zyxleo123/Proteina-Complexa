#!/usr/bin/env python3
"""Audit that CPSea macrocycles really do close between the binder termini.

Three questions, answered against the data rather than against the docstrings
that currently assert this (`cyclization/mask.py`, `pair_feats.py`):

  RAW  -- in the untouched `CPSea_pdb/*.pdb.gz`, does the CONECT-recorded ring
          bond connect the first and last *standard* residue of the peptide
          chain? (Caps ACE/NME and hydrogens are ignored, exactly as the
          preprocessing drops them, so "first residue" means the same thing on
          both sides.)
  PRE  -- does the preprocessed PDB still carry that bond, between the same two
          residues, with both anchor atoms present and at a physically sane
          distance? (i.e. preprocessing neither dropped nor scrambled the
          cyclization atoms.)
  BOTH -- do the two agree, and does the label the training transform would read
          come out as (0, L_b - 1)?

Sampling: the full set is ~2.4M structures, so by default this audits a random
stratified sample per split x metadata cyclization_type. `--all` scans
everything (slow; use `--workers`).

Examples
--------
    python script_utils/audit_cpsea_cyclization_endpoints.py --n-per-stratum 2000
    python script_utils/audit_cpsea_cyclization_endpoints.py --splits val test --all
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Kept in sync with script_utils/preprocess_cpsea.py -- "first binder residue" must mean
# the same thing on the raw and the preprocessed side or the comparison is meaningless.
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
CAP_RESNAMES = {"ACE", "NME"}

RAW_BINDER_CHAIN = "L"  # renamed to "B" by preprocess_cpsea.CHAIN_RENAME
PRE_BINDER_CHAIN = "B"

# Anchor-atom distance sanity windows (Angstrom), deliberately WIDER than the
# acceptance windows in eval/cyclic_reconstruction_metrics.py: this is asking
# "did preprocessing corrupt the geometry", not "is the ring closed well".
SANE_BOND_A = (0.9, 2.6)


def _open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def parse_pdb(path: str):
    """(atoms_by_serial, conect_pairs). Atoms carry name/resname/chain/resseq/xyz."""
    atoms: dict[int, dict] = {}
    pairs: set[tuple[int, int]] = set()
    with _open_text(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    serial = int(line[6:11])
                    atoms[serial] = {
                        "name": line[12:16].strip(),
                        "resname": line[17:20].strip(),
                        "chain": line[21],
                        "resseq": int(line[22:26]),
                        "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                    }
                except ValueError:
                    continue
            elif line.startswith("CONECT"):
                c = line.rstrip("\n")
                serials = [int(c[i : i + 5]) for i in range(6, len(c), 5) if c[i : i + 5].strip()]
                if len(serials) < 2:
                    continue
                for partner in serials[1:]:
                    if partner != serials[0]:
                        pairs.add((min(serials[0], partner), max(serials[0], partner)))
    return atoms, sorted(pairs)


def binder_residue_order(atoms: dict[int, dict], chain: str) -> list[int]:
    """Binder-local index -> resSeq, standard residues only, ascending resSeq.

    Ascending order matches `preprocess_cpsea`'s output sort key
    (chain, resseq, atom_name), so local indices agree across raw/preprocessed.
    """
    resseqs = {a["resseq"] for a in atoms.values() if a["chain"] == chain and a["resname"] in STANDARD_AA}
    return sorted(resseqs)


def _dist(p, q) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def _classify(name_a, res_a, name_b, res_b) -> str | None:
    """Ring chemistry of one bonded atom pair, or None if it is not a ring bond."""
    names, ress = {name_a, name_b}, (res_a, res_b)
    if names == {"SG"} and ress == ("CYS", "CYS"):
        return "disulfide"
    if names == {"N", "C"}:
        return "mainchain"
    lys_acid = (
        (name_a == "NZ" and res_a == "LYS" and name_b in {"CG", "CD", "C"} and res_b in {"ASP", "GLU", "ASN", "GLN"})
        or (name_b == "NZ" and res_b == "LYS" and name_a in {"CG", "CD", "C"} and res_a in {"ASP", "GLU", "ASN", "GLN"})
    )
    if lys_acid:
        return "isopeptide"
    return None


def ring_bonds(atoms, pairs, chain, order) -> list[dict]:
    """Every inter-residue CONECT bond inside the binder chain, classified."""
    local = {rs: k for k, rs in enumerate(order)}
    out = []
    for a, b in pairs:
        aa, bb = atoms.get(a), atoms.get(b)
        if aa is None or bb is None:
            continue
        if aa["chain"] != chain or bb["chain"] != chain:
            continue
        if aa["resseq"] not in local or bb["resseq"] not in local:
            continue
        i, j = local[aa["resseq"]], local[bb["resseq"]]
        if i == j:
            continue
        chem = _classify(aa["name"], aa["resname"], bb["name"], bb["resname"])
        if chem is None:
            continue
        # Adjacent backbone N-C is the ordinary peptide bond, not a ring closure.
        if chem == "mainchain" and abs(i - j) == 1:
            continue
        lo, hi = (i, j) if i < j else (j, i)
        out.append({"i": lo, "j": hi, "chem": chem, "dist_A": _dist(aa["xyz"], bb["xyz"])})
    return out


def audit_one(job) -> dict:
    example_id, raw_path, pre_path, peptide_length, meta_hint = job
    r: dict = {"example_id": example_id, "hint": meta_hint, "peptide_length": int(peptide_length)}

    # ---- RAW ----
    try:
        atoms, pairs = parse_pdb(raw_path)
        order = binder_residue_order(atoms, RAW_BINDER_CHAIN)
        bonds = ring_bonds(atoms, pairs, RAW_BINDER_CHAIN, order)
        r["raw_len"] = len(order)
        r["raw_n_ring_bonds"] = len(bonds)
        if len(bonds) == 1:
            b = bonds[0]
            r.update(raw_i=b["i"], raw_j=b["j"], raw_chem=b["chem"], raw_dist_A=b["dist_A"])
            r["raw_terminal"] = (b["i"] == 0) and (b["j"] == len(order) - 1)
        r["raw_status"] = "ok" if len(bonds) == 1 else f"n_ring_bonds={len(bonds)}"
    except Exception as e:  # noqa: BLE001 - an unreadable file is a finding, not a crash
        r["raw_status"] = f"error:{type(e).__name__}"

    # ---- PREPROCESSED ----
    try:
        atoms, pairs = parse_pdb(pre_path)
        order = binder_residue_order(atoms, PRE_BINDER_CHAIN)
        bonds = ring_bonds(atoms, pairs, PRE_BINDER_CHAIN, order)
        r["pre_len"] = len(order)
        r["pre_n_ring_bonds"] = len(bonds)
        if len(bonds) == 1:
            b = bonds[0]
            r.update(pre_i=b["i"], pre_j=b["j"], pre_chem=b["chem"], pre_dist_A=b["dist_A"])
            r["pre_terminal"] = (b["i"] == 0) and (b["j"] == len(order) - 1)
            r["pre_dist_sane"] = SANE_BOND_A[0] <= b["dist_A"] <= SANE_BOND_A[1]
        r["pre_status"] = "ok" if len(bonds) == 1 else f"n_ring_bonds={len(bonds)}"
    except Exception as e:  # noqa: BLE001
        r["pre_status"] = f"error:{type(e).__name__}"

    # ---- AGREEMENT ----
    if r.get("raw_status") == "ok" and r.get("pre_status") == "ok":
        r["agree"] = (
            r["raw_i"] == r["pre_i"] and r["raw_j"] == r["pre_j"] and r["raw_chem"] == r["pre_chem"]
        )
        # Coordinates must survive preprocessing untouched.
        r["coords_preserved"] = abs(r["raw_dist_A"] - r["pre_dist_A"]) < 1e-3
    r["len_matches_metadata"] = r.get("pre_len") == int(peptide_length)
    return r


def build_jobs(df: pd.DataFrame) -> list:
    return [
        (row.example_id, row.source_path, row.path, row.peptide_length, row.cyclization_type)
        for row in df.itertuples()
    ]


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n{'=' * 78}\nAudited {n} structures\n{'=' * 78}")

    def rate(key, want=True):
        vals = [r[key] for r in rows if key in r]
        good = sum(1 for v in vals if v == want)
        miss = n - len(vals)
        pct = 100.0 * good / len(vals) if vals else float("nan")
        return f"{good}/{len(vals)} ({pct:.4f}%)" + (f"  [{miss} not evaluable]" if miss else "")

    print(f"RAW  single ring bond found : {sum(1 for r in rows if r.get('raw_status') == 'ok')}/{n}")
    print(f"RAW  bond is (0, L-1)       : {rate('raw_terminal')}")
    print(f"PRE  single ring bond found : {sum(1 for r in rows if r.get('pre_status') == 'ok')}/{n}")
    print(f"PRE  bond is (0, L-1)       : {rate('pre_terminal')}")
    print(f"PRE  anchor dist in {SANE_BOND_A}A : {rate('pre_dist_sane')}")
    print(f"RAW/PRE same (i, j, chem)   : {rate('agree')}")
    print(f"RAW/PRE coords identical    : {rate('coords_preserved')}")
    print(f"PRE  len == metadata length : {rate('len_matches_metadata')}")

    print("\nchemistry (preprocessed) x metadata hint:")
    xt = Counter((r.get("pre_chem", "-"), r["hint"]) for r in rows)
    for (chem, hint), c in sorted(xt.items()):
        print(f"  {chem:<12} x {hint:<10} {c}")

    print("\nnon-ok statuses:")
    bad = Counter(r["raw_status"] for r in rows if r.get("raw_status") != "ok")
    bad.update(Counter("pre:" + r["pre_status"] for r in rows if r.get("pre_status") != "ok"))
    print("  none" if not bad else "\n".join(f"  {k}: {v}" for k, v in bad.most_common()))

    viol = [
        r for r in rows
        if r.get("raw_terminal") is False
        or r.get("pre_terminal") is False
        or r.get("agree") is False
        or r.get("pre_dist_sane") is False
        or r.get("coords_preserved") is False
    ]
    print(f"\nviolations: {len(viol)}")
    for r in viol[:20]:
        print(f"  {r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("CPSEA_FULL_DATA_PATH", "/zfsauton/scratch/yixiz/CPSea/CPSea_full/CPSea"))
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--n-per-stratum", type=int, default=1000, help="per split x cyclization_type")
    p.add_argument("--all", action="store_true", help="audit every structure (slow)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--shard", type=int, default=0, help="This task's shard index (for a slurm array full scan).")
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    meta_dir = Path(args.data_root) / "preprocessed" / "metadata"
    frames = []
    for split in args.splits:
        f = meta_dir / f"cpsea_{split}.parquet"
        if not f.exists():
            print(f"skip missing {f}")
            continue
        df = pd.read_parquet(f)
        df["split"] = split
        if not args.all:
            df = (
                df.groupby("cyclization_type", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.n_per_stratum), random_state=args.seed))
            )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if args.num_shards > 1:
        df = df.iloc[args.shard :: args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(df)} structures")
    jobs = build_jobs(df)
    random.Random(args.seed).shuffle(jobs)
    print(f"auditing {len(jobs)} structures with {args.workers} workers ...")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(audit_one, jobs, chunksize=8), 1):
            rows.append(r)
            if k % 5000 == 0:
                print(f"  ... {k}/{len(jobs)}", flush=True)

    summarize(rows)
    if args.out_csv:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
