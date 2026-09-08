"""Stage the MEET linear peptide-pocket benchmark into CPSea-compatible inputs.

Why this exists
---------------
The de novo (unconditional) generation baseline scored ~45-52% ring closure on the LNR
targets (evaluation_results/uncondgen_20260902_151235) against ~90% natively. One candidate
explanation is that LNR's crystal receptors are OOD for a model trained on AFDB-derived
CPSea pockets. MEET is the control for that: it starts from the same 8.64M AFDB domains
CPSea used and extracts linear peptide-pocket complexes with a CPSea-like filter stack, so
running the SAME de novo corner on MEET targets isolates "target OOD" from everything else.

Input format (MEET zenodo release, e.g. valid.tar.gz)
-----------------------------------------------------
    index.txt   one row per entry: <id>\t<byte_start>\t<byte_end>\t<props json>
    data.bin    concatenated gzip members; [byte_start:byte_end] is one gzipped JSON
                [id, [[chain, [block, ...]], ...], bonds, {}]
                block = [resname, [[atom_name, [x,y,z], element, serial, {...}], ...],
                         [resnum, icode], {props}]
    index.npy   (n, 4) offsets -- not needed, index.txt carries the same byte range

Chain "R" is the pocket (already spatially cropped, with residue-numbering GAPS -- the same
convention CPSea's preprocessed receptors use, see scripts/restage_lnr_pocket.py). Chain "L"
is the peptide, flanked by artificial ACE / NME caps that carry SMILES-style residue names
("CC=O" / "CN") and record their identity in the block props as `original_name`. Those two
blocks are NOT amino acids and are stripped here; every reported peptide length is the
number of real residues.

Output matches scripts/build_lnr_metadata.py exactly (same columns, same geometry
definitions) so MEET and LNR rows can be pooled and binned by N-C gap and length.

Usage:
    python scripts/build_meet_metadata.py \
        --meet-root /zfsauton/scratch/yixiz/LPData/MEET_valid \
        --out-dir CPSea_data/meet_staged --split valid
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lnr_metadata import (  # noqa: E402  -- identical definitions on purpose
    BACKBONE,
    BINDER_MAX,
    BINDER_MIN,
    STANDARD_AA,
    TARGET_MIN,
    virtual_cb,
)

# MEET writes the terminal caps with SMILES-ish residue names and keeps the real PDB name in
# the block props. Match on BOTH so a rename upstream cannot silently smuggle a cap through
# as a residue (it would not be in STANDARD_AA either, which is the third line of defence).
CAP_NAMES = {"ACE", "NME", "NHE", "NH2"}
CAP_RESNAMES = {"CC=O", "CN", "N"}


def read_index(meet_root: Path) -> list[tuple[str, int, int, dict]]:
    entries = []
    for ln in (meet_root / "index.txt").read_text().splitlines():
        if not ln.strip():
            continue
        eid, start, end, props = ln.split("\t", 3)
        entries.append((eid, int(start), int(end), json.loads(props)))
    return entries


def load_entry(fh, start: int, end: int):
    fh.seek(start)
    return json.loads(gzip.decompress(fh.read(end - start)))


def is_cap(block) -> bool:
    resname, _, _, props = block[0], block[1], block[2], (block[3] if len(block) > 3 else {})
    return (str(props.get("original_name", "")).upper() in CAP_NAMES
            or resname in CAP_RESNAMES)


def blocks_to_residues(blocks, drop_caps: bool):
    """-> (OrderedDict[(resseq, icode, resname) -> {atom: xyz}], flags)."""
    residues, flags = OrderedDict(), {"n_caps": 0, "nonstd": set(), "dup_keys": 0}
    for block in blocks:
        if drop_caps and is_cap(block):
            flags["n_caps"] += 1
            continue
        resname = block[0]
        if resname not in STANDARD_AA:
            flags["nonstd"].add(resname)
            continue
        resnum, icode = block[2][0], (block[2][1] or " ")
        key = (int(resnum), icode if icode.strip() else " ", resname)
        if key in residues:
            flags["dup_keys"] += 1
            continue
        atoms = OrderedDict()
        for atom in block[1]:
            name, xyz = atom[0], tuple(float(v) for v in atom[1])
            element = (atom[2] if len(atom) > 2 else name[:1]) or name[:1]
            if str(element).strip().upper() == "H":
                continue
            atoms.setdefault(name, (xyz, str(element).strip()))
        residues[key] = atoms
    return residues, flags


def xyz_only(residue: dict) -> dict:
    return {k: v[0] for k, v in residue.items()}


def as_lnr_res(residue: dict) -> dict:
    """virtual_cb() reads build_lnr_metadata's (line, xyz) atom tuples and takes [1]."""
    return {k: (None, v) for k, v in residue.items()}


def heavy_xyz(residues: OrderedDict) -> list[tuple[float, float, float]]:
    return [xyz for atoms in residues.values() for xyz, _ in atoms.values()]


def write_staged_pdb(out_path: Path, receptor: OrderedDict, peptide: OrderedDict,
                     example_id: str) -> None:
    """chain A = receptor pocket, chain B = peptide. Residue numbers are copied from MEET so
    the pocket's numbering GAPS survive -- SegmentAwareResidueFeaturesTransform reads them as
    the receptor's fragment structure, which is the whole point of using a cropped pocket."""
    lines = [
        f"REMARK 999 MEET STAGED example_id={example_id}\n",
        "REMARK 999 CHAIN MAP: pocket(R)->A, peptide(L)->B\n",
        "REMARK 999 REMOVED: ACE/NME caps, hydrogens, non-standard residues\n",
        "REMARK 999 COORDINATES UNCHANGED (chain id rewritten only)\n",
    ]
    serial = 1
    for chain_id, residues in (("A", receptor), ("B", peptide)):
        for (resseq, icode, resname), atoms in residues.items():
            for name, (xyz, element) in atoms.items():
                aname = f" {name:<3s}" if len(name) < 4 else name
                lines.append(
                    f"ATOM  {serial:5d} {aname}{' '}{resname:>3s} {chain_id}"
                    f"{resseq:4d}{icode}   "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
                )
                serial += 1
        lines.append(f"TER   {serial:5d}\n")
        serial += 1
    lines.append("END\n")
    out_path.write_text("".join(lines))


def crop_pocket(receptor: OrderedDict, peptide: OrderedDict, radius: float) -> OrderedDict:
    pep = heavy_xyz(peptide)
    kept = OrderedDict()
    for key, atoms in receptor.items():
        if any(math.dist(a, p) <= radius for a, _ in atoms.values() for p in pep):
            kept[key] = atoms
    return kept


def n_segments(residues: OrderedDict) -> int:
    nums = sorted(k[0] for k in residues)
    return 1 + sum(1 for a, b in zip(nums, nums[1:]) if b != a + 1) if nums else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meet-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--split", default="meet_valid", help="Value for the `split` column and the parquet stem.")
    ap.add_argument("--pocket-chain", default="R")
    ap.add_argument("--peptide-chain", default="L")
    ap.add_argument("--pocket-radius", type=float, default=0.0,
                    help="If >0, additionally crop the MEET pocket to this heavy-atom radius "
                         "around the peptide (residue numbers preserved, so gaps widen). "
                         "0 = keep MEET's own pocket as shipped.")
    ap.add_argument("--keep-caps", action="store_true",
                    help="Do NOT strip ACE/NME. Diagnostic only -- the caps are not amino "
                         "acids and the CPSea loader would reject them.")
    args = ap.parse_args()

    pdb_out = args.out_dir / "pdbs"
    pdb_out.mkdir(parents=True, exist_ok=True)

    entries = read_index(args.meet_root)
    rows, audit = [], []

    with open(args.meet_root / "data.bin", "rb") as fh:
        for eid, start, end, props in entries:
            record = {"meet_id": eid, "kept": False, "reason": ""}
            try:
                entry = load_entry(fh, start, end)
            except Exception as exc:  # a corrupt member must be reported, never skipped quietly
                record["reason"] = f"decode_failed: {exc}"
                audit.append(record)
                continue

            chains = {c[0]: c[1] for c in entry[1]}
            if args.pocket_chain not in chains or args.peptide_chain not in chains:
                record["reason"] = f"missing_chain (have {sorted(chains)})"
                audit.append(record)
                continue

            peptide, pflags = blocks_to_residues(chains[args.peptide_chain],
                                                drop_caps=not args.keep_caps)
            receptor, rflags = blocks_to_residues(chains[args.pocket_chain], drop_caps=False)
            L, R_full = len(peptide), len(receptor)
            record.update(peptide_length_raw=len(chains[args.peptide_chain]),
                          n_caps_stripped=pflags["n_caps"],
                          peptide_length=L, receptor_length_shipped=R_full,
                          peptide_nonstd=";".join(sorted(pflags["nonstd"])),
                          receptor_nonstd=";".join(sorted(rflags["nonstd"])))

            if not (BINDER_MIN <= L <= BINDER_MAX):
                record["reason"] = f"peptide_length_{L}_outside_{BINDER_MIN}_{BINDER_MAX}"
                audit.append(record)
                continue

            if args.pocket_radius > 0:
                receptor = crop_pocket(receptor, peptide, args.pocket_radius)
            R = len(receptor)
            record["receptor_length"] = R
            if R < TARGET_MIN:
                record["reason"] = f"receptor_length_{R}_below_{TARGET_MIN}"
                audit.append(record)
                continue

            missing = [k for k, atoms in peptide.items() if not set(BACKBONE) <= set(atoms)]
            if missing:
                record["reason"] = f"incomplete_backbone_{len(missing)}_residues"
                audit.append(record)
                continue

            keys = list(peptide)
            pep_xyz = {k: xyz_only(v) for k, v in peptide.items()}
            # Identical definitions to build_lnr_metadata.py so the two sets bin together.
            n_first, c_last = pep_xyz[keys[0]]["N"], pep_xyz[keys[-1]]["C"]
            nc_dist = math.dist(n_first, c_last)

            cbs = [virtual_cb(as_lnr_res(pep_xyz[k])) for k in keys]
            pair_dists = [(math.dist(cbs[i], cbs[j]), i, j)
                          for i in range(L) for j in range(i + 3, L) if cbs[i] and cbs[j]]
            if pair_dists:
                best_ss = min(pair_dists, key=lambda t: abs(t[0] - 4.0))
                best_iso = min(pair_dists, key=lambda t: abs(t[0] - 6.5))
            else:
                best_ss = best_iso = (float("nan"), -1, -1)

            sgs = [(i, pep_xyz[k]["SG"]) for i, k in enumerate(keys)
                   if k[2] == "CYS" and "SG" in pep_xyz[k]]
            existing_ss = min((math.dist(a[1], b[1]) for i, a in enumerate(sgs) for b in sgs[i + 1:]),
                              default=float("nan"))

            example_id = f"MEET_{eid}"
            out_pdb = pdb_out / f"{example_id}.pdb"
            write_staged_pdb(out_pdb, receptor, peptide, example_id)

            rows.append({
                # --- columns the CPSea loader consumes ---
                "example_id": example_id,
                "path": str(out_pdb.resolve()),
                "binder_chain_id": "B",
                # MEET ids are <AFDB accession>_<domain>_cluster<k>_ligand<n>; the AFDB
                # accession is the only thing that groups redundant rows.
                "cluster_id": eid.split("_cluster")[0],
                "split": args.split,
                "peptide_length": L,
                "receptor_length": R,
                # Linear by construction, as in the LNR staging.
                "cyclization_type": "other",
                "source_path": str((args.meet_root / "data.bin").resolve()),
                # --- stratification columns (ignored by the loader) ---
                "arm": f"meet_{args.split}",
                "pdb_id": eid,
                "nc_gap_angstrom": nc_dist,
                "best_ss_cb_dist": best_ss[0], "best_ss_i": best_ss[1], "best_ss_j": best_ss[2],
                "best_iso_cb_dist": best_iso[0], "best_iso_i": best_iso[1], "best_iso_j": best_iso[2],
                "n_cys": len(sgs),
                "min_sg_sg": existing_ss,
                "has_existing_disulfide": bool(existing_ss == existing_ss and existing_ss < 2.5),
                "n_receptor_segments": n_segments(receptor),
                "peptide_sequence": "".join(k[2] for k in keys),
            })
            record.update(kept=True, reason="ok")
            audit.append(record)

    if not rows:
        raise SystemExit("No MEET entries survived filtering -- refusing to write empty metadata.")

    df = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit)
    meta_dir = args.out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    stem = args.split if args.pocket_radius <= 0 else f"{args.split}_pocket{int(args.pocket_radius)}"
    meta_path = meta_dir / f"{stem}.parquet"
    df.to_parquet(meta_path, index=False)
    audit_df.to_csv(args.out_dir / f"{stem}_staging_audit.csv", index=False)

    summary = {
        "entries_in_index": len(entries),
        "kept": len(df),
        "dropped": len(entries) - len(df),
        "pocket_radius": args.pocket_radius,
        "drop_reasons": audit_df.query("~kept")["reason"].value_counts().to_dict(),
        "peptide_length": {"min": int(df.peptide_length.min()),
                           "median": float(df.peptide_length.median()),
                           "max": int(df.peptide_length.max())},
        "receptor_length": {"min": int(df.receptor_length.min()),
                            "median": float(df.receptor_length.median()),
                            "max": int(df.receptor_length.max())},
        "n_receptor_segments": {"min": int(df.n_receptor_segments.min()),
                                "median": float(df.n_receptor_segments.median()),
                                "max": int(df.n_receptor_segments.max())},
        "nc_gap_angstrom": {"min": float(df.nc_gap_angstrom.min()),
                            "median": float(df.nc_gap_angstrom.median()),
                            "max": float(df.nc_gap_angstrom.max())},
        "nc_gap_bins": {"<=10A": int((df.nc_gap_angstrom <= 10).sum()),
                        "10-15A": int(((df.nc_gap_angstrom > 10) & (df.nc_gap_angstrom <= 15)).sum()),
                        ">15A": int((df.nc_gap_angstrom > 15).sum())},
        "n_with_ss_ready_pair_3.4_4.8A": int(((df.best_ss_cb_dist >= 3.4) & (df.best_ss_cb_dist <= 4.8)).sum()),
        "n_with_existing_disulfide": int(df.has_existing_disulfide.sum()),
    }
    (args.out_dir / f"{stem}_staging_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nmetadata: {meta_path}\naudit:    {args.out_dir / f'{stem}_staging_audit.csv'}")


if __name__ == "__main__":
    main()
