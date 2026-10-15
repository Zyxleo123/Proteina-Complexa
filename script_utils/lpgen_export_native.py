#!/usr/bin/env python3
"""Export the common LP manifest into the per-complex layout PPFlow and POTFlow read.

Both arms want a directory per complex holding a peptide file and a receptor file; they
differ only in the filenames and in how they decide membership. One exporter therefore
serves both, and because the structures come from ``lpgen_manifest.parquet`` the two arms
are guaranteed to see the same examples -- which is the whole point of the manifest.

LAYOUT

    {out_dir}/{split}/{example_id}_{chain}/peptide.pdb
                                          /pocket.pdb
                                          /peptide_repaired.pdb   (--ppflow-aliases)
                                          /receptor_repaired.pdb  (--ppflow-aliases)
    {out_dir}/{split}_names.pt      a torch-loadable list of dir names (PPFlow)
    {out_dir}/names.txt             POTFlow's blocklist, deliberately empty
    {out_dir}/lpgen_export.summary.json

ONE DIRECTORY PER SPLIT, which is what makes membership manifest-driven. POTFlow's loader
takes its dataset to BE the directory listing (``all_pdbs = os.listdir(self.structure_dir)``)
and keys its LMDB cache on a separate ``name`` field, so pointing a split at its own
``structure_dir`` makes ``os.listdir`` return exactly the manifest's rows -- no loader
patch, and no way for a stale cache name to quietly select the wrong set. PPFlow instead
filters one ``data_dir`` through ``include_pdb_path``, a ``torch.load``-ed list of names,
so it gets ``{split}_names.pt`` and uses its own native mechanism.

THE DIRECTORY NAME ENDS IN THE PEPTIDE CHAIN ID, because POTFlow recovers the chain with
``chain_id = data['id'][-1]`` -- the last character of the directory name. That is why
PepMerge's entries are named ``1a07_C``. Our peptides are chain ``B``, so a directory named
anything else silently hands POTFlow the wrong chain letter.

THE POCKET IS A 10.0 A HEAVY-ATOM CROP, which is not a guess. Measured against PepMerge
(the dataset POTFlow's released configuration points at): across sampled entries, the
farthest residue *included* in ``pocket.pdb`` sits 9.37-9.98 A from the nearest peptide
heavy atom, and the nearest residue *excluded* from it sits 10.01-10.53 A away. The
threshold is exactly 10.0 A, heavy atom to heavy atom, whole residues. Feeding the full
receptor instead would be both a distribution mismatch against that pocket and a needless
cost -- our receptors run to 1,339 residues against a ~30% pocket.

NO FoldX REPAIR IS APPLIED. PPFlow's native filenames are ``*_repaired.pdb`` because
upstream runs FoldX RepairPDB first. ``--ppflow-aliases`` writes our unrepaired structures
under those names so the loader finds them; that is a deliberate, recorded substitution,
not a claim that the structures were repaired.

Run (see ``scripts/lpgen_export.sbatch``):

    .venv/bin/python script_utils/lpgen_export_native.py \\
        --manifest $ZFS/LPData/preprocessed/metadata_lpgen/lpgen_manifest.parquet \\
        --out-dir  $ZFS/LPData/preprocessed/lpgen_native
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

#: Heavy-atom cutoff defining the receptor pocket. Measured from PepMerge; see the module
#: docstring. Whole residues: a residue with ANY heavy atom inside the cutoff is kept
#: entirely, which is what makes the output parseable as residues rather than fragments.
POCKET_CUTOFF_A = 10.0

#: Minimum pocket residues for the complex to be usable. A peptide with no receptor
#: residue within the cutoff is not a bound complex; training on it would teach the
#: target-conditioning trunk to ignore an empty target.
MIN_POCKET_RESIDUES = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits", default="train,val",
                   help="Comma-separated splits to export. 'heldout' is reserved for the "
                        "later CP-sequence comparison and is NOT exported by default.")
    p.add_argument("--pocket-cutoff-A", type=float, default=POCKET_CUTOFF_A)
    p.add_argument("--ppflow-aliases", action="store_true",
                   help="Also write peptide_repaired.pdb / receptor_repaired.pdb, the "
                        "filenames PPFlow's loader opens. No FoldX repair is applied.")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=-1, help="Per-split cap for smoke runs.")
    p.add_argument("--clean", action="store_true",
                   help="Remove an existing split directory first. Without it, a rerun "
                        "that now exports FEWER rows would leave the extra directories "
                        "behind, and os.listdir membership would still include them.")
    return p.parse_args()


# ------------------------------------------------------------------------ PDB handling
def _is_heavy(line: str) -> bool:
    el = line[76:78].strip()
    if el:
        return el != "H"
    return not line[12:16].strip().startswith("H")


def read_atoms(pdb_path: str) -> dict:
    """Group ATOM lines by (chain, resseq+icode), keeping the lines verbatim.

    The lines are passed through unchanged apart from the serial number on write, so
    coordinates, occupancies, b-factors and element columns cannot be corrupted by
    reformatting -- the usual way a hand-rolled PDB writer goes wrong.
    """
    res: dict = {}
    order: list = []
    for line in open(pdb_path, "r"):
        if not line.startswith("ATOM"):
            continue
        key = (line[21], line[22:27])
        if key not in res:
            res[key] = []
            order.append(key)
        res[key].append(line)
    return {"res": res, "order": order}


def write_pdb(path: Path, keys: list, res: dict) -> int:
    """Write the given residues, renumbering atom serials. Returns the atom count.

    TER and END are written to the column spec (record name padded to 6, TER carrying the
    final residue's name/chain/number). Bare ``TER\\n`` is not merely untidy: a parser
    matches ``line[0:6]`` against ``"TER   "``, so it is silently an unrecognized record.
    That matters here because POTFlow runs mkdssp on ``peptide.pdb``, DSSP is strict about
    chain termination, and any DSSP failure is swallowed by a bare ``except:`` upstream and
    turns the peptide into an all-helix label with no log line.
    """
    serial = 1
    with open(path, "w") as fh:
        last = None
        for key in keys:
            for line in res[key]:
                fh.write(f"ATOM  {serial:>5d}{line[11:]}")
                serial += 1
                last = line
        if last is not None:
            # COLUMNS: 1-6 "TER   ", 7-11 serial, 18-20 resName, 22 chainID, 23-27 resSeq+iCode
            fh.write(f"TER   {serial:>5d}      {last[17:20]} {last[21]}{last[22:27]}\n")
            serial += 1
        fh.write("END   \n")
    return serial - 1


def export_one(task: tuple) -> dict:
    """Write one complex's directory. Returns a status record, never raises."""
    (example_id, path, pep_chain, split, out_root, cutoff, aliases) = task
    rec = {"example_id": example_id, "status": "ok", "reason": "",
           "dir_name": "", "n_pocket_res": 0, "n_pep_res": 0, "n_rec_res": 0}
    try:
        parsed = read_atoms(path)
    except OSError as exc:
        rec["status"] = "unreadable"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        return rec

    res, order = parsed["res"], parsed["order"]
    pep_keys = [k for k in order if k[0] == pep_chain]
    rec_keys = [k for k in order if k[0] != pep_chain]
    rec["n_pep_res"], rec["n_rec_res"] = len(pep_keys), len(rec_keys)
    if not pep_keys:
        rec["status"] = "no_peptide_chain"
        rec["reason"] = f"chain {pep_chain!r} absent"
        return rec
    if not rec_keys:
        rec["status"] = "no_receptor"
        rec["reason"] = "only the peptide chain is present"
        return rec

    pep_xyz = np.array(
        [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
         for k in pep_keys for l in res[k] if _is_heavy(l)]
    )
    if pep_xyz.size == 0:
        rec["status"] = "no_peptide_heavy_atoms"
        rec["reason"] = "peptide chain has no heavy atoms"
        return rec

    # Whole-residue crop: a residue is in if ANY of its heavy atoms is within the cutoff.
    pocket_keys = []
    for k in rec_keys:
        xyz = np.array(
            [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
             for l in res[k] if _is_heavy(l)]
        )
        if xyz.size == 0:
            continue
        if np.min(np.linalg.norm(xyz[:, None, :] - pep_xyz[None, :, :], axis=-1)) <= cutoff:
            pocket_keys.append(k)

    rec["n_pocket_res"] = len(pocket_keys)
    if len(pocket_keys) < MIN_POCKET_RESIDUES:
        rec["status"] = "pocket_too_small"
        rec["reason"] = f"{len(pocket_keys)} residues within {cutoff} A (min {MIN_POCKET_RESIDUES})"
        return rec

    # POTFlow reads the peptide chain id as the LAST CHARACTER of the directory name.
    dir_name = f"{example_id}_{pep_chain}"
    rec["dir_name"] = dir_name
    d = Path(out_root) / split / dir_name
    d.mkdir(parents=True, exist_ok=True)
    write_pdb(d / "peptide.pdb", pep_keys, res)
    write_pdb(d / "pocket.pdb", pocket_keys, res)
    if aliases:
        # PPFlow's loader opens these names. Copies, not symlinks: the upstream loader
        # may be pointed at a different root, and a relative symlink would dangle.
        shutil.copyfile(d / "peptide.pdb", d / "peptide_repaired.pdb")
        shutil.copyfile(d / "pocket.pdb", d / "receptor_repaired.pdb")
    return rec


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_parquet(args.manifest)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = set(splits) - set(man.split.unique())
    if unknown:
        raise SystemExit(f"FATAL: --splits names {sorted(unknown)}, manifest has "
                         f"{sorted(man.split.unique())}")

    records: dict = {}
    for split in splits:
        rows = man[man.split == split].sort_values("example_id", kind="mergesort")
        if args.limit > 0:
            rows = rows.head(args.limit)
        split_dir = out_dir / split
        if args.clean and split_dir.exists():
            logger.warning(f"--clean: removing {split_dir}")
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        tasks = [
            (r.example_id, r.path, r.peptide_chain_id, split,
             str(out_dir), args.pocket_cutoff_A, args.ppflow_aliases)
            for r in rows.itertuples()
        ]
        logger.info(f"[{split}] exporting {len(tasks):,} complexes with {args.workers} workers")
        with Pool(args.workers) as pool:
            out = pool.map(export_one, tasks, chunksize=64)
        records[split] = out
        status = Counter(r["status"] for r in out)
        logger.info(f"[{split}] {dict(status)}")

        ok = [r for r in out if r["status"] == "ok"]
        # PPFlow's native membership mechanism: include_pdb_path is torch.load()-ed.
        torch.save([r["dir_name"] for r in ok], out_dir / f"{split}_names.pt")

        # Cross-check that the directory listing -- which IS POTFlow's dataset -- matches
        # what we think we wrote. A stale directory from an earlier, larger run would
        # otherwise silently enlarge the training set.
        on_disk = {p.name for p in split_dir.iterdir() if p.is_dir()}
        expected = {r["dir_name"] for r in ok}
        if on_disk != expected:
            extra, missing = sorted(on_disk - expected), sorted(expected - on_disk)
            raise SystemExit(
                f"FATAL: {split} directory listing disagrees with the export: "
                f"{len(extra)} extra (e.g. {extra[:3]}), {len(missing)} missing "
                f"(e.g. {missing[:3]}). Rerun with --clean."
            )

    # POTFlow imports a module-level blocklist read from a hardcoded absolute path in the
    # original author's home directory. Ours is deliberately EMPTY: membership comes from
    # the per-split structure_dir, so nothing should be filtered a second time here.
    (out_dir / "names.txt").write_text("")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "out_dir": str(out_dir),
        "pocket_cutoff_A": args.pocket_cutoff_A,
        "pocket_cutoff_provenance":
            "measured against PepMerge: included pocket residues reach 9.37-9.98 A, "
            "excluded receptor residues start at 10.01-10.53 A",
        "min_pocket_residues": MIN_POCKET_RESIDUES,
        "ppflow_aliases": args.ppflow_aliases,
        "foldx_repair_applied": False,
        "names_txt": "empty by design; POTFlow's blocklist is replaced by per-split dirs",
        "splits": {
            s: {
                "exported": sum(1 for r in v if r["status"] == "ok"),
                "failed": sum(1 for r in v if r["status"] != "ok"),
                "by_status": dict(Counter(r["status"] for r in v)),
                "pocket_residues": {
                    "min": int(min((r["n_pocket_res"] for r in v if r["status"] == "ok"), default=0)),
                    "median": float(np.median([r["n_pocket_res"] for r in v if r["status"] == "ok"]))
                    if any(r["status"] == "ok" for r in v) else 0.0,
                    "max": int(max((r["n_pocket_res"] for r in v if r["status"] == "ok"), default=0)),
                },
                "receptor_residues_median": float(
                    np.median([r["n_rec_res"] for r in v if r["status"] == "ok"])
                ) if any(r["status"] == "ok" for r in v) else 0.0,
                "names_pt": str(out_dir / f"{s}_names.pt"),
            }
            for s, v in records.items()
        },
        "smoke_limit": args.limit if args.limit > 0 else None,
    }
    (out_dir / "lpgen_export.summary.json").write_text(json.dumps(summary, indent=2))

    failures = [r for v in records.values() for r in v if r["status"] != "ok"]
    pd.DataFrame(failures, columns=["example_id", "status", "reason", "dir_name",
                                    "n_pocket_res", "n_pep_res", "n_rec_res"]).to_csv(
        out_dir / "lpgen_export_failures.csv", index=False)

    for s, v in summary["splits"].items():
        logger.info(f"  {s:8s} exported {v['exported']:>7,}  failed {v['failed']:>5,}  "
                    f"pocket med {v['pocket_residues']['median']:.0f} res "
                    f"(receptor med {v['receptor_residues_median']:.0f})")
    logger.info(f"summary -> {out_dir / 'lpgen_export.summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
