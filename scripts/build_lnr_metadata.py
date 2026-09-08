"""Stage the LNR linear peptide-receptor benchmark into CPSea-compatible inputs.

Why this exists
---------------
The AE round-trip go/no-go (see scripts/eval_ae_roundtrip.py) has to feed LNR through
*exactly* the CPSea loader path, otherwise a reconstruction gap could be a data-plumbing
artifact rather than a statement about linear peptides. The CPSea `StructureDataset` reads
a metadata parquet with a `path` column and expects preprocessed PDBs following the
convention in $CPSEA_DATA_PATH/preprocessed/processed/:

    chain A = receptor/target, chain B = binder, ATOM records only, no hydrogens,
    no waters/ligands, standard amino acids only.

LNR ships raw crystal PDBs plus a `test.txt` of (pdb_id, receptor_chain, peptide_chain).
This script rewrites each into that convention and emits the metadata parquet, plus an
audit CSV recording every entry that was dropped and why.

Coordinates are NEVER transformed -- only chain IDs (and, if insertion codes are present,
residue numbers) are rewritten, so the bound pose is preserved bit-for-bit.

Usage:
    python scripts/build_lnr_metadata.py --lnr-root $ZFS/LNR --out-dir CPSea_data/lnr_staged
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import pandas as pd

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
BACKBONE = ("N", "CA", "C", "O")

# CPSea's CroppingTransform2 (configs/dataset/unified/cpsea_peptide.yaml) sets
# binder_min_length/binder_max_length 5/16 and target_min_length 30. Applying the same
# bounds here means nothing is silently dropped later inside the loader.
BINDER_MIN, BINDER_MAX = 5, 16
TARGET_MIN = 30


def parse_pdb_chains(path: Path, chains: set[str]) -> tuple[dict, dict]:
    """Returns (residues, flags).

    residues: {chain: OrderedDict[(resseq, icode, resname) -> {atom_name: (line, xyz)}]}
    Only the first MODEL, only ATOM records, only standard amino acids, no hydrogens,
    altloc restricted to blank/'A'.
    """
    residues = {c: OrderedDict() for c in chains}
    flags = {"has_icode": set(), "n_altloc_dropped": 0, "n_hydrogen_dropped": 0,
             "n_hetatm_dropped": 0, "nonstd_resnames": set(), "multi_model": False}

    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL":
                flags["multi_model"] = True
                break
            if rec == "HETATM":
                ch = line[21]
                if ch in chains:
                    rn = line[17:20].strip()
                    flags["n_hetatm_dropped"] += 1
                    if rn != "HOH":
                        flags["nonstd_resnames"].add(rn)
                continue
            if rec != "ATOM  ":
                continue
            ch = line[21]
            if ch not in chains:
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                flags["n_altloc_dropped"] += 1
                continue
            element = line[76:78].strip()
            atom_name = line[12:16].strip()
            # Hydrogens: trust the element column, fall back to the PDB name convention
            # (leading digit then H, e.g. "1HB ") when the column is blank.
            if element == "H" or (not element and atom_name[:1].isdigit() and "H" in atom_name[:2]):
                flags["n_hydrogen_dropped"] += 1
                continue
            resname = line[17:20].strip()
            if resname not in STANDARD_AA:
                flags["nonstd_resnames"].add(resname)
                continue
            resseq, icode = line[22:26].strip(), line[26]
            if icode != " ":
                flags["has_icode"].add(ch)
            key = (resseq, icode, resname)
            residues[ch].setdefault(key, {})
            # First occurrence wins (altloc A after blank is the same atom).
            residues[ch][key].setdefault(
                atom_name,
                (line, (float(line[30:38]), float(line[38:46]), float(line[46:54]))),
            )
    return residues, flags


def virtual_cb(res: dict) -> tuple[float, float, float] | None:
    """Ideal CB position from N/CA/C (the standard -0.58273431/0.56802827/-0.54067466
    combination), used so GLY still gets a CB-CB entry in the crosslink-feasibility screen."""
    if "CB" in res:
        return res["CB"][1]
    if not all(a in res for a in ("N", "CA", "C")):
        return None
    n, ca, c = (res[a][1] for a in ("N", "CA", "C"))
    b = tuple(ca[i] - n[i] for i in range(3))
    cc = tuple(c[i] - ca[i] for i in range(3))
    a = (b[1] * cc[2] - b[2] * cc[1], b[2] * cc[0] - b[0] * cc[2], b[0] * cc[1] - b[1] * cc[0])
    return tuple(-0.58273431 * a[i] + 0.56802827 * b[i] - 0.54067466 * cc[i] + ca[i] for i in range(3))


def write_staged_pdb(out_path: Path, receptor: OrderedDict, peptide: OrderedDict,
                     renumber_receptor: bool, renumber_peptide: bool, example_id: str) -> None:
    """Writes chain A (receptor) + chain B (peptide). Columns other than chain id and
    residue number are copied verbatim, so element/occupancy/B-factor survive intact."""
    lines = [
        f"REMARK 999 LNR STAGED example_id={example_id}\n",
        "REMARK 999 CHAIN MAP: receptor->A, peptide->B\n",
        "REMARK 999 REMOVED: hydrogens, waters, ligands, non-standard residues, altloc!=A\n",
        "REMARK 999 COORDINATES UNCHANGED (chain id / residue number rewritten only)\n",
    ]
    serial = 1
    for chain_id, residues, renumber in (("A", receptor, renumber_receptor),
                                         ("B", peptide, renumber_peptide)):
        for idx, (key, atoms) in enumerate(residues.items(), start=1):
            resseq = idx if renumber else int(key[0])
            for _, (line, _) in atoms.items():
                new = (line[:6] + f"{serial:5d}" + line[11:21] + chain_id
                       + f"{resseq:4d}" + " " + line[27:])
                lines.append(new)
                serial += 1
        lines.append(f"TER   {serial:5d}\n")
        serial += 1
    lines.append("END\n")
    out_path.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lnr-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--list-file", default="test.txt")
    args = ap.parse_args()

    pdb_out = args.out_dir / "pdbs"
    pdb_out.mkdir(parents=True, exist_ok=True)

    rows, audit = [], []
    entries = [ln.split() for ln in (args.lnr_root / args.list_file).read_text().splitlines() if ln.strip()]

    for fields in entries:
        pdb_id, rec_chain, pep_chain = fields[0], fields[1], fields[2]
        src = args.lnr_root / "pdbs" / f"{pdb_id}.pdb"
        record = {"pdb_id": pdb_id, "receptor_chain": rec_chain, "peptide_chain": pep_chain,
                  "kept": False, "reason": ""}

        if not src.exists():
            record["reason"] = "pdb_missing"
            audit.append(record)
            continue
        if rec_chain == pep_chain:
            record["reason"] = "receptor_and_peptide_same_chain"
            audit.append(record)
            continue

        residues, flags = parse_pdb_chains(src, {rec_chain, pep_chain})
        peptide, receptor = residues[pep_chain], residues[rec_chain]
        L, R = len(peptide), len(receptor)
        record.update(peptide_length=L, receptor_length=R,
                      dropped_hetatm=flags["n_hetatm_dropped"],
                      dropped_altloc=flags["n_altloc_dropped"],
                      nonstd_seen=";".join(sorted(flags["nonstd_resnames"])),
                      multi_model=flags["multi_model"])

        if not (BINDER_MIN <= L <= BINDER_MAX):
            record["reason"] = f"peptide_length_{L}_outside_{BINDER_MIN}_{BINDER_MAX}"
            audit.append(record)
            continue
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
        # Head-to-tail gap: how far a mainchain closure would have to travel (ideal ~1.33 A).
        n_first, c_last = peptide[keys[0]]["N"][1], peptide[keys[-1]]["C"][1]
        nc_dist = math.dist(n_first, c_last)

        # Internal crosslink screen: disulfide/isopeptide pick their own (i, j), so the
        # relevant number is the BEST available pair, not the termini. |i-j| >= 3 excludes
        # pairs too close in sequence to bridge.
        cbs = [virtual_cb(peptide[k]) for k in keys]
        pair_dists = [(math.dist(cbs[i], cbs[j]), i, j)
                      for i in range(L) for j in range(i + 3, L) if cbs[i] and cbs[j]]
        if pair_dists:
            best_ss = min(pair_dists, key=lambda t: abs(t[0] - 4.0))
            best_iso = min(pair_dists, key=lambda t: abs(t[0] - 6.5))
        else:
            best_ss = best_iso = (float("nan"), -1, -1)

        sgs = [(i, peptide[k]["SG"][1]) for i, k in enumerate(keys)
               if k[2] == "CYS" and "SG" in peptide[k]]
        existing_ss = min((math.dist(a[1], b[1]) for i, a in enumerate(sgs) for b in sgs[i + 1:]),
                          default=float("nan"))

        example_id = f"LNR_{pdb_id}_{rec_chain}_{pep_chain}"
        out_pdb = pdb_out / f"{example_id}.pdb"
        write_staged_pdb(out_pdb, receptor, peptide,
                         renumber_receptor=rec_chain in flags["has_icode"],
                         renumber_peptide=pep_chain in flags["has_icode"],
                         example_id=example_id)

        rows.append({
            # --- columns the CPSea loader consumes (schema must match cpsea_val.parquet) ---
            "example_id": example_id,
            "path": str(out_pdb.resolve()),
            "binder_chain_id": "B",
            "cluster_id": pdb_id,
            "split": "lnr",
            "peptide_length": L,
            "receptor_length": R,
            # LNR peptides are linear by construction. "other" is the value CPSea already
            # uses for rows without a resolved linkage, so no downstream branch is surprised.
            "cyclization_type": "other",
            "source_path": str(src.resolve()),
            # --- LNR-specific stratification columns (ignored by the loader) ---
            "arm": "lnr_linear_crystal",
            "pdb_id": pdb_id,
            "nc_gap_angstrom": nc_dist,
            "best_ss_cb_dist": best_ss[0], "best_ss_i": best_ss[1], "best_ss_j": best_ss[2],
            "best_iso_cb_dist": best_iso[0], "best_iso_i": best_iso[1], "best_iso_j": best_iso[2],
            "n_cys": len(sgs),
            "min_sg_sg": existing_ss,
            "has_existing_disulfide": bool(existing_ss == existing_ss and existing_ss < 2.5),
        })
        record.update(kept=True, reason="ok")
        audit.append(record)

    if not rows:
        raise SystemExit("No LNR entries survived filtering -- refusing to write an empty metadata file.")

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = args.out_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    df.to_parquet(meta_dir / "lnr_test.parquet", index=False)
    pd.DataFrame(audit).to_csv(args.out_dir / "lnr_staging_audit.csv", index=False)

    summary = {
        "entries_in_list": len(entries),
        "kept": len(df),
        "dropped": len(entries) - len(df),
        "drop_reasons": pd.DataFrame(audit).query("~kept")["reason"].value_counts().to_dict(),
        "peptide_length": {"min": int(df.peptide_length.min()), "median": float(df.peptide_length.median()),
                           "max": int(df.peptide_length.max())},
        "nc_gap_angstrom": {"min": float(df.nc_gap_angstrom.min()),
                            "median": float(df.nc_gap_angstrom.median()),
                            "max": float(df.nc_gap_angstrom.max())},
        "n_with_ss_ready_pair_3.4_4.8A": int(((df.best_ss_cb_dist >= 3.4) & (df.best_ss_cb_dist <= 4.8)).sum()),
        "n_with_existing_disulfide": int(df.has_existing_disulfide.sum()),
    }
    (args.out_dir / "lnr_staging_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nmetadata: {meta_dir / 'lnr_test.parquet'}\naudit:    {args.out_dir / 'lnr_staging_audit.csv'}")


if __name__ == "__main__":
    main()
