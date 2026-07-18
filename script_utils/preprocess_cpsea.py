#!/usr/bin/env python3
"""Convert CPSea cyclic peptide–receptor PDBs into Proteina training format.

Each transformation is audited: intended removals (H, ACE/NME caps) are logged
separately from unintended information loss. Cyclization CONECT records are
remapped to surviving heavy-atom serials and written to output PDBs (ignored by
Proteina training loaders).

Output layout (default under ``$CPSEA_DATA_PATH/preprocessed``):

  {out_dir}/processed/{train,val,test}/*.pdb
  {out_dir}/metadata/cpsea_{train,val,test}.parquet
  {out_dir}/manifest/run_config.json
  {out_dir}/manifest/preprocess_stats.json
  {out_dir}/manifest/per_structure_audit.csv
  {out_dir}/manifest/rejected.csv

Set ``CPSEA_DATA_PATH`` in ``.env`` (default: ``/zfsauton/scratch/yixiz/CPSea/CPSea_PDB``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
CAP_RESNAMES = {"ACE", "NME"}
CHAIN_RENAME = {"R": "A", "L": "B"}  # receptor → target A, peptide → binder B
INTENDED_REMOVAL_REASONS = frozenset({"hydrogen", "cap_hetatm", "non_standard_residue", "unknown_chain"})


@dataclass
class AtomRecord:
    serial: int
    record_type: str  # ATOM or HETATM
    atom_name: str
    resname: str
    chain_id: str
    resseq: int
    x: float
    y: float
    z: float
    line: str

    @property
    def coord_key(self) -> tuple[str, int, str, str]:
        return (self.chain_id, self.resseq, self.resname.strip(), self.atom_name.strip())

    @property
    def coord(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class StructureAudit:
    example_id: str
    source_path: str
    status: str = "ok"
    reject_reason: str = ""
    input_atoms: int = 0
    input_conect_pairs: int = 0
    input_chains: str = ""
    removed_hydrogen: int = 0
    removed_cap: int = 0
    removed_non_standard: int = 0
    removed_unknown_chain: int = 0
    output_atoms: int = 0
    output_chains: str = ""
    conect_pairs_input: int = 0
    conect_pairs_kept: int = 0
    conect_pairs_dropped_removed_atom: int = 0
    conect_pairs_dropped_duplicate: int = 0
    cyclization_type: str = ""
    peptide_length: int = 0
    receptor_length: int = 0
    coord_max_delta_A: float = 0.0
    coord_mismatch_count: int = 0
    output_path: str = ""
    split: str = ""
    cluster_id: str = ""
    intended_removals: dict = field(default_factory=dict)
    conect_drop_samples: list = field(default_factory=list)


def _open_text(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".pdb.gz"):
        return gzip.open(path, "rt")
    return open(path)


def parse_pdb_lines(lines: list[str]) -> tuple[list[AtomRecord], list[tuple[int, int]]]:
    atoms: list[AtomRecord] = []
    conect_pairs: list[tuple[int, int]] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            atoms.append(
                AtomRecord(
                    serial=int(line[6:11]),
                    record_type=line[:6].strip(),
                    atom_name=line[12:16],
                    resname=line[17:20],
                    chain_id=line[21],
                    resseq=int(line[22:26]),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    line=line.rstrip("\n"),
                )
            )
        elif line.startswith("CONECT"):
            serials = [int(line[i : i + 5]) for i in range(6, len(line.rstrip()), 5) if line[i : i + 5].strip()]
            if not serials:
                continue
            center = serials[0]
            for partner in serials[1:]:
                pair = (min(center, partner), max(center, partner))
                conect_pairs.append(pair)
    # dedupe conect while preserving count for audit
    unique_pairs = sorted(set(conect_pairs))
    return atoms, unique_pairs


def classify_removal(atom: AtomRecord) -> str | None:
    elem = atom.atom_name.strip()[0] if atom.atom_name.strip() else ""
    if elem == "H" or atom.atom_name.strip().startswith("H"):
        return "hydrogen"
    if atom.resname.strip() in CAP_RESNAMES:
        return "cap_hetatm"
    if atom.chain_id not in CHAIN_RENAME:
        return "unknown_chain"
    if atom.record_type == "ATOM" and atom.resname.strip() in STANDARD_AA:
        return None
    if atom.record_type == "HETATM":
        return "non_standard_residue"
    if atom.resname.strip() not in STANDARD_AA:
        return "non_standard_residue"
    return None


def remap_chain(atom: AtomRecord) -> AtomRecord:
    new_chain = CHAIN_RENAME[atom.chain_id]
    old_line = atom.line
    new_line = old_line[:21] + new_chain + old_line[22:]
    return AtomRecord(
        serial=atom.serial,
        record_type=atom.record_type,
        atom_name=atom.atom_name,
        resname=atom.resname,
        chain_id=new_chain,
        resseq=atom.resseq,
        x=atom.x,
        y=atom.y,
        z=atom.z,
        line=new_line,
    )


def count_residues(atoms: list[AtomRecord], chain_id: str) -> int:
    return len({a.resseq for a in atoms if a.chain_id == chain_id and a.atom_name.strip() == "CA"})


def infer_cyclization_type(atoms: list[AtomRecord], conect_pairs: list[tuple[int, int]], serial_to_atom: dict[int, AtomRecord]) -> str:
    for a, b in conect_pairs:
        atom_a = serial_to_atom.get(a)
        atom_b = serial_to_atom.get(b)
        if atom_a is None or atom_b is None:
            continue
        if atom_a.chain_id != "B" or atom_b.chain_id != "B":
            continue
        names = {atom_a.atom_name.strip(), atom_b.atom_name.strip()}
        if "SG" in names and atom_a.resname.strip() == "CYS" and atom_b.resname.strip() == "CYS":
            return "disulfide"
        backbone = {"N", "C", "CA"}
        if names <= backbone or ("N" in names and "C" in names):
            return "head_tail"
    return "other"


def verify_coordinate_preservation(
    original: list[AtomRecord],
    cleaned: list[AtomRecord],
) -> tuple[float, int]:
    orig_map = {}
    for a in original:
        reason = classify_removal(a)
        if reason is not None:
            continue
        remapped = remap_chain(a)
        orig_map[remapped.coord_key] = remapped.coord

    max_delta = 0.0
    mismatches = 0
    for a in cleaned:
        key = a.coord_key
        if key not in orig_map:
            mismatches += 1
            continue
        ox, oy, oz = orig_map[key]
        delta = ((a.x - ox) ** 2 + (a.y - oy) ** 2 + (a.z - oz) ** 2) ** 0.5
        max_delta = max(max_delta, delta)
        if delta > 1e-3:
            mismatches += 1
    return max_delta, mismatches


def format_atom_line(atom: AtomRecord, new_serial: int) -> str:
    return (
        f"{atom.record_type:<6}{new_serial:>5} {atom.atom_name}{atom.resname:>4} "
        f"{atom.chain_id}{atom.resseq:>4}    "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{atom.line[54:].rstrip()}\n"
    )


def write_pdb(
    path: Path,
    atoms: list[AtomRecord],
    conect_pairs: list[tuple[int, int]],
    example_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial_map = {old.serial: i + 1 for i, old in enumerate(atoms)}
    with open(path, "w") as f:
        f.write(f"REMARK 999 CPSEA PREPROCESS example_id={example_id}\n")
        f.write("REMARK 999 CHAIN MAP: R->A (target), L->B (binder)\n")
        f.write("REMARK 999 REMOVED (intended): hydrogens, ACE/NME caps\n")
        f.write("REMARK 999 CONECT: cyclization bonds remapped; ignored by Proteina training\n")
        prev_chain = None
        for atom in atoms:
            if prev_chain is not None and atom.chain_id != prev_chain:
                f.write("TER\n")
            prev_chain = atom.chain_id
            f.write(format_atom_line(atom, serial_map[atom.serial]))
        f.write("TER\n")
        for a, b in conect_pairs:
            na, nb = serial_map[a], serial_map[b]
            f.write(f"CONECT{na:5d}{nb:5d}\n")
            f.write(f"CONECT{nb:5d}{na:5d}\n")
        f.write("END\n")


def process_structure(
    source_path: Path,
    example_id: str,
    peptide_min: int,
    peptide_max: int,
) -> tuple[list[AtomRecord], list[tuple[int, int]], StructureAudit]:
    """Process one CPSea structure with full per-step audit."""
    audit = StructureAudit(example_id=example_id, source_path=str(source_path.resolve()))
    with _open_text(source_path) as f:
        lines = f.readlines()

    atoms, conect_pairs = parse_pdb_lines(lines)
    audit.input_atoms = len(atoms)
    audit.input_conect_pairs = len(conect_pairs)
    audit.input_chains = ",".join(sorted({a.chain_id for a in atoms}))
    audit.conect_pairs_input = len(conect_pairs)

    kept: list[AtomRecord] = []
    removal_counts: Counter = Counter()
    old_serial_to_new: dict[int, int] = {}

    for atom in atoms:
        reason = classify_removal(atom)
        if reason is not None:
            removal_counts[reason] += 1
            continue
        remapped = remap_chain(atom)
        kept.append(remapped)

    audit.removed_hydrogen = removal_counts["hydrogen"]
    audit.removed_cap = removal_counts["cap_hetatm"]
    audit.removed_non_standard = removal_counts["non_standard_residue"]
    audit.removed_unknown_chain = removal_counts["unknown_chain"]
    audit.intended_removals = dict(removal_counts)

    chain_ids = sorted({a.chain_id for a in kept})
    if chain_ids != ["A", "B"]:
        audit.status = "rejected"
        audit.reject_reason = f"expected_chains_A_B_got_{chain_ids}"
        return [], [], audit

    peptide_len = count_residues(kept, "B")
    receptor_len = count_residues(kept, "A")
    audit.peptide_length = peptide_len
    audit.receptor_length = receptor_len

    if not (peptide_min <= peptide_len <= peptide_max):
        audit.status = "rejected"
        audit.reject_reason = f"peptide_length_{peptide_len}_outside_{peptide_min}_{peptide_max}"
        return [], [], audit

    kept.sort(key=lambda a: (a.chain_id, a.resseq, a.atom_name.strip()))

    # Map original serial → new serial via coord_key
    orig_kept_by_key = {}
    for atom in atoms:
        if classify_removal(atom) is not None:
            continue
        remapped = remap_chain(atom)
        orig_kept_by_key[remapped.coord_key] = atom.serial

    rebuilt: list[AtomRecord] = []
    for i, atom in enumerate(kept):
        rebuilt.append(
            AtomRecord(
                serial=i + 1,
                record_type=atom.record_type,
                atom_name=atom.atom_name,
                resname=atom.resname,
                chain_id=atom.chain_id,
                resseq=atom.resseq,
                x=atom.x,
                y=atom.y,
                z=atom.z,
                line=atom.line,
            )
        )

    old_to_new: dict[int, int] = {}
    for new_atom in rebuilt:
        old_serial = orig_kept_by_key.get(new_atom.coord_key)
        if old_serial is not None:
            old_to_new[old_serial] = new_atom.serial

    kept_pairs: list[tuple[int, int]] = []
    dropped_removed = 0
    dropped_dup = 0
    seen: set[tuple[int, int]] = set()
    drop_samples: list[str] = []
    for a, b in conect_pairs:
        if a not in old_to_new or b not in old_to_new:
            dropped_removed += 1
            if len(drop_samples) < 5:
                atom_a = next((x for x in atoms if x.serial == a), None)
                atom_b = next((x for x in atoms if x.serial == b), None)
                ra = classify_removal(atom_a) if atom_a else "missing"
                rb = classify_removal(atom_b) if atom_b else "missing"
                drop_samples.append(f"{a}-{b}:reasons={ra},{rb}")
            continue
        na, nb = old_to_new[a], old_to_new[b]
        pair = (min(na, nb), max(na, nb))
        if pair in seen:
            dropped_dup += 1
            continue
        seen.add(pair)
        kept_pairs.append(pair)

    audit.conect_pairs_kept = len(kept_pairs)
    audit.conect_pairs_dropped_removed_atom = dropped_removed
    audit.conect_pairs_dropped_duplicate = dropped_dup
    audit.conect_drop_samples = drop_samples

    serial_to_atom = {a.serial: a for a in rebuilt}
    audit.cyclization_type = infer_cyclization_type(rebuilt, kept_pairs, serial_to_atom)

    max_delta, mismatches = verify_coordinate_preservation(atoms, rebuilt)
    audit.coord_max_delta_A = max_delta
    audit.coord_mismatch_count = mismatches
    if mismatches > 0 or max_delta > 1e-3:
        audit.status = "rejected"
        audit.reject_reason = f"coordinate_mismatch_delta_{max_delta:.6f}_count_{mismatches}"
        return [], [], audit

    audit.output_atoms = len(rebuilt)
    audit.output_chains = ",".join(chain_ids)
    return rebuilt, kept_pairs, audit


def assign_cluster_splits(
    cluster_ids: list[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, str]:
    unique = sorted(set(cluster_ids))
    rng = pd.Series(unique).sample(frac=1.0, random_state=seed).tolist()
    n = len(rng)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    split_map = {}
    for i, cid in enumerate(rng):
        if i < n_train:
            split_map[cid] = "train"
        elif i < n_train + n_val:
            split_map[cid] = "val"
        else:
            split_map[cid] = "test"
    return split_map


def load_cluster_map(cluster_tsv: Path | None) -> dict[str, str]:
    if cluster_tsv is None or not cluster_tsv.exists():
        return {}
    df = pd.read_csv(cluster_tsv, sep="\t")
    return dict(zip(df["Cluster_Member"], df["Cluster_Center"], strict=False))


def _scan_pdb_root(pdb_root: Path) -> dict[str, Path]:
    """One directory scan; avoids millions of per-file stat() calls on NFS."""
    available: dict[str, Path] = {}
    with os.scandir(pdb_root) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            if name.endswith(".pdb.gz"):
                available[name[: -len(".pdb.gz")]] = Path(entry.path)
            elif name.endswith(".pdb"):
                stem = name[: -len(".pdb")]
                available.setdefault(stem, Path(entry.path))
    return available


def discover_inputs(pdb_root: Path, index_file: Path | None) -> list[tuple[str, Path]]:
    logger.info(f"Scanning {pdb_root} for input structures...")
    available = _scan_pdb_root(pdb_root)
    logger.info(f"Found {len(available)} PDB files under {pdb_root}")

    if index_file and index_file.exists():
        stems = [line.strip() for line in index_file.read_text().splitlines() if line.strip()]
        items = [(stem, available[stem]) for stem in stems if stem in available]
        if items:
            missing = len(stems) - len(items)
            if missing:
                logger.warning(f"Index lists {len(stems)} IDs; {missing} have no PDB on disk")
            return items
        logger.warning(
            f"Index {index_file} matched 0 files under {pdb_root}; falling back to scan"
        )
    return list(available.items())


def audit_to_row(audit: StructureAudit) -> dict:
    return {
        "example_id": audit.example_id,
        "status": audit.status,
        "reject_reason": audit.reject_reason,
        "source_path": audit.source_path,
        "output_path": audit.output_path,
        "split": audit.split,
        "cluster_id": audit.cluster_id,
        "input_atoms": audit.input_atoms,
        "output_atoms": audit.output_atoms,
        "input_chains": audit.input_chains,
        "output_chains": audit.output_chains,
        "removed_hydrogen": audit.removed_hydrogen,
        "removed_cap": audit.removed_cap,
        "removed_non_standard": audit.removed_non_standard,
        "removed_unknown_chain": audit.removed_unknown_chain,
        "conect_pairs_input": audit.conect_pairs_input,
        "conect_pairs_kept": audit.conect_pairs_kept,
        "conect_pairs_dropped_removed_atom": audit.conect_pairs_dropped_removed_atom,
        "conect_pairs_dropped_duplicate": audit.conect_pairs_dropped_duplicate,
        "cyclization_type": audit.cyclization_type,
        "peptide_length": audit.peptide_length,
        "receptor_length": audit.receptor_length,
        "coord_max_delta_A": audit.coord_max_delta_A,
        "coord_mismatch_count": audit.coord_mismatch_count,
        "conect_drop_samples": "|".join(audit.conect_drop_samples[:5]),
    }


def verify_output_pdb(output_path: Path, audit: StructureAudit) -> dict:
    """Round-trip verify written PDB matches audit counts."""
    with open(output_path) as f:
        lines = f.readlines()
    atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
    conect_lines = [l for l in lines if l.startswith("CONECT")]
    on_disk_pairs = len(conect_lines) // 2  # symmetric CONECT
    return {
        "example_id": audit.example_id,
        "on_disk_atoms": len(atom_lines),
        "on_disk_conect_pairs": on_disk_pairs,
        "atoms_match": len(atom_lines) == audit.output_atoms,
        "conect_match": on_disk_pairs == audit.conect_pairs_kept,
    }


def process_one(job: tuple[str, Path, str, str, int, int, Path]) -> tuple[StructureAudit, dict | None, dict | None]:
    """Worker: process, write, and verify a single structure. Runs in a subprocess."""
    stem, source_path, cluster_id, split, peptide_min, peptide_max, out_dir = job
    atoms, conect_pairs, audit = process_structure(source_path, stem, peptide_min, peptide_max)
    audit.cluster_id = cluster_id
    audit.split = split

    if audit.status != "ok":
        logger.warning(f"REJECTED {stem}: {audit.reject_reason}")
        return audit, None, None

    out_path = out_dir / "processed" / audit.split / f"{stem}.pdb"
    write_pdb(out_path, atoms, conect_pairs, stem)
    audit.output_path = str(out_path.resolve())

    v = verify_output_pdb(out_path, audit)
    if not v["atoms_match"] or not v["conect_match"]:
        logger.error(
            f"OUTPUT VERIFY FAIL {stem}: atoms={v['on_disk_atoms']}/{audit.output_atoms} "
            f"conect={v['on_disk_conect_pairs']}/{audit.conect_pairs_kept}"
        )
        audit.status = "rejected"
        audit.reject_reason = "output_verify_failed"
        return audit, None, v

    logger.info(
        f"OK {stem} | split={audit.split} | atoms {audit.input_atoms}->{audit.output_atoms} "
        f"(−H:{audit.removed_hydrogen} −cap:{audit.removed_cap}) | "
        f"CONECT {audit.conect_pairs_input}->{audit.conect_pairs_kept} "
        f"(dropped_removed:{audit.conect_pairs_dropped_removed_atom}) | "
        f"peptide={audit.peptide_length} cyclization={audit.cyclization_type} | "
        f"coord_delta={audit.coord_max_delta_A:.2e}Å"
    )
    metadata_row = {
        "example_id": stem,
        "path": audit.output_path,
        "binder_chain_id": "B",
        "cluster_id": cluster_id,
        "split": audit.split,
        "peptide_length": audit.peptide_length,
        "receptor_length": audit.receptor_length,
        "cyclization_type": audit.cyclization_type,
        "source_path": audit.source_path,
        "conect_pairs_kept": audit.conect_pairs_kept,
    }
    return audit, metadata_row, v


def default_cpsea_root() -> Path:
    return Path(os.environ.get("CPSEA_DATA_PATH", "/zfsauton/scratch/yixiz/CPSea/CPSea_PDB"))


def main():
    cpsea_root = default_cpsea_root()
    parser = argparse.ArgumentParser(description="Preprocess CPSea PDBs for Proteina training.")
    parser.add_argument(
        "--pdb-root",
        type=Path,
        default=cpsea_root / "CPSea_PDB_pdb",
        help=f"Directory of raw .pdb.gz files (default: $CPSEA_DATA_PATH/CPSea_PDB_pdb = {cpsea_root / 'CPSea_PDB_pdb'})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=cpsea_root / "preprocessed",
        help=f"Preprocessed output root (default: $CPSEA_DATA_PATH/preprocessed = {cpsea_root / 'preprocessed'})",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=cpsea_root / "CPSea_PDB_index.txt",
        help="CPSea index file (default: $CPSEA_DATA_PATH/CPSea_PDB_index.txt)",
    )
    parser.add_argument(
        "--cluster-tsv",
        type=Path,
        default=cpsea_root / "CPSea_PDB_properties" / "CPSea_PDB_Cluster.tsv",
        help="Foldseek cluster TSV for reproducible splits",
    )
    parser.add_argument("--peptide-min-length", type=int, default=5)
    parser.add_argument("--peptide-max-length", type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cluster splits (reproducible)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N structures (smoke tests)")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes (default: cpu_count - 1). Use 1 to disable multiprocessing.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.out_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpsea_data_path": str(cpsea_root.resolve()),
        "pdb_root": str(args.pdb_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "index": str(args.index.resolve()) if args.index else None,
        "cluster_tsv": str(args.cluster_tsv.resolve()) if args.cluster_tsv else None,
        "peptide_min_length": args.peptide_min_length,
        "peptide_max_length": args.peptide_max_length,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "limit": args.limit,
        "chain_rename": CHAIN_RENAME,
        "intended_removals": sorted(INTENDED_REMOVAL_REASONS),
    }
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    run_config["config_hash"] = config_hash
    (manifest_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    transformation_policy = {
        "preserved": [
            "heavy_atom_coordinates (verified per atom: max delta <= 1e-3 Å)",
            "standard_amino_acid_ATOM_records on chains R and L",
            "cyclization_CONECT where both endpoint atoms survive filtering",
            "residue_numbers_and_atom_names",
        ],
        "intended_removals": {
            "hydrogen": "All hydrogen atoms (elem H or atom name starting with H)",
            "cap_hetatm": "ACE/NME cap HETATM on peptide chain L",
            "non_standard_residue": "Non-standard residues (not in 20 AA)",
            "unknown_chain": "Atoms not on chain R or L",
        },
        "intended_transforms": {
            "chain_rename": CHAIN_RENAME,
            "atom_serial_renumber": "Sequential 1..N after filtering",
            "conect_remap": "Drop pairs touching removed atoms; renumber surviving pairs",
            "chain_order": "A (target) before B (binder)",
        },
        "not_preserved": [
            "CONECT involving removed H or ACE/NME atoms (logged as dropped_removed_atom)",
            "duplicate symmetric CONECT pairs (logged as dropped_duplicate)",
            "hydrogen coordinates",
            "cap group geometry",
        ],
        "training_loader_note": "Proteina load_any() ignores CONECT; bonds not used in atom37 encoding",
    }
    (manifest_dir / "transformation_policy.json").write_text(json.dumps(transformation_policy, indent=2))

    cluster_map = load_cluster_map(args.cluster_tsv if args.cluster_tsv.exists() else None)
    inputs = discover_inputs(args.pdb_root, args.index if args.index and args.index.exists() else None)
    if not inputs:
        logger.error(f"No input structures found under {args.pdb_root}")
        return 1
    if args.limit:
        inputs = inputs[: args.limit]

    logger.info(f"Processing {len(inputs)} structures from {args.pdb_root}")
    logger.info(f"Run config hash: {config_hash} (see {manifest_dir / 'run_config.json'})")

    # Assign splits by cluster
    all_cluster_ids = [cluster_map.get(stem, stem) for stem, _ in inputs]
    split_map = assign_cluster_splits(all_cluster_ids, args.train_frac, args.val_frac, args.seed)

    jobs = [
        (
            stem,
            source_path,
            cluster_map.get(stem, stem),
            split_map.get(cluster_map.get(stem, stem), "train"),
            args.peptide_min_length,
            args.peptide_max_length,
            args.out_dir,
        )
        for stem, source_path in inputs
    ]

    audits: list[StructureAudit] = []
    metadata_rows: list[dict] = []
    verify_rows: list[dict] = []

    logger.info(f"Using {args.workers} worker process(es)")
    if args.workers <= 1:
        results = (process_one(job) for job in jobs)
    else:
        chunksize = max(1, len(jobs) // (args.workers * 20))
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            results = list(executor.map(process_one, jobs, chunksize=chunksize))

    for audit, metadata_row, v in results:
        audits.append(audit)
        if v is not None:
            verify_rows.append(v)
        if metadata_row is not None:
            metadata_row["config_hash"] = config_hash
            metadata_rows.append(metadata_row)

    # Write audit CSVs
    audit_rows = [audit_to_row(a) for a in audits]
    pd.DataFrame(audit_rows).to_csv(manifest_dir / "per_structure_audit.csv", index=False)
    rejected = [r for r in audit_rows if r["status"] != "ok"]
    pd.DataFrame(rejected).to_csv(manifest_dir / "rejected.csv", index=False)
    pd.DataFrame(verify_rows).to_csv(manifest_dir / "output_verify.csv", index=False)

    ok_audits = [a for a in audits if a.status == "ok"]
    stats = {
        "config_hash": config_hash,
        "n_input": len(inputs),
        "n_ok": len(ok_audits),
        "n_rejected": len(audits) - len(ok_audits),
        "reject_reasons": dict(Counter(a.reject_reason for a in audits if a.status != "ok")),
        "intended_removals_total": {
            "hydrogen": sum(a.removed_hydrogen for a in ok_audits),
            "cap_hetatm": sum(a.removed_cap for a in ok_audits),
            "non_standard_residue": sum(a.removed_non_standard for a in ok_audits),
        },
        "conect_summary": {
            "pairs_input_total": sum(a.conect_pairs_input for a in ok_audits),
            "pairs_kept_total": sum(a.conect_pairs_kept for a in ok_audits),
            "pairs_dropped_removed_atom_total": sum(a.conect_pairs_dropped_removed_atom for a in ok_audits),
            "structures_with_kept_conect": sum(1 for a in ok_audits if a.conect_pairs_kept > 0),
            "structures_zero_kept_conect": sum(1 for a in ok_audits if a.conect_pairs_kept == 0),
        },
        "cyclization_types": dict(Counter(a.cyclization_type for a in ok_audits)),
        "split_counts": dict(Counter(r["split"] for r in metadata_rows)),
        "coord_max_delta_A_overall": max((a.coord_max_delta_A for a in ok_audits), default=0.0),
        "output_verify_failures": sum(1 for v in verify_rows if not (v["atoms_match"] and v["conect_match"])),
    }
    (manifest_dir / "preprocess_stats.json").write_text(json.dumps(stats, indent=2))

    # Parquet per split
    meta_dir = args.out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_df = pd.DataFrame(metadata_rows)
    if len(meta_df) == 0:
        logger.error("No structures passed preprocessing; see manifest/rejected.csv")
        return 1
    for split in ("train", "val", "test"):
        split_df = meta_df[meta_df["split"] == split]
        if len(split_df) > 0:
            out_parquet = meta_dir / f"cpsea_{split}.parquet"
            split_df.to_parquet(out_parquet, index=False)
            logger.info(f"Wrote {out_parquet} ({len(split_df)} rows)")

    logger.info("=" * 60)
    logger.info(f"Preprocess complete: {stats['n_ok']}/{stats['n_input']} OK")
    logger.info(f"Intended removals (totals): {stats['intended_removals_total']}")
    logger.info(f"CONECT summary: {stats['conect_summary']}")
    logger.info(f"Manifest: {manifest_dir}")
    logger.info(f"Stats: {manifest_dir / 'preprocess_stats.json'}")

    if stats["n_rejected"] > 0:
        logger.warning(f"Rejected {stats['n_rejected']} structures — see {manifest_dir / 'rejected.csv'}")

    return 0 if stats["n_ok"] > 0 else 1


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    raise SystemExit(main())
