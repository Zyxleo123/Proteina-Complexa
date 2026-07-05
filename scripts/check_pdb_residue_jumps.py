#!/usr/bin/env python3
"""
Check residue-number jumps and physical peptide-bond breaks in PDB / PDB.GZ files.

Definitions
-----------
NUMBERING_ONLY:
    Residue numbers jump, but the previous residue C atom and next residue N atom
    are close enough to be a normal peptide bond.

REAL_BREAK:
    The previous residue C atom and next residue N atom are missing or farther
    than --cn-cutoff, optionally accompanied by a residue-number jump.

The script also reports separate counts for the binder chain, default chain B.

Examples
--------
Single file:
    python scripts/check_pdb_residue_jumps.py complex.pdb

Directory:
    python scripts/check_pdb_residue_jumps.py preprocessed_sample100/processed

Directory, gzipped raw PDBs:
    python scripts/check_pdb_residue_jumps.py CPSea_sample_100 --pattern "*.pdb.gz"

Only print summary:
    python scripts/check_pdb_residue_jumps.py CPSea_sample_100 --pattern "*.pdb.gz" --quiet
"""

from __future__ import annotations

import argparse
import gzip
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BACKBONE_ATOMS_FOR_BREAK_CHECK = {"N", "CA", "C"}


@dataclass
class Residue:
    chain: str
    resi: int
    icode: str
    resn: str
    atoms: dict[str, tuple[float, float, float]]

    @property
    def label(self) -> str:
        insertion = self.icode.strip()
        return f"{self.resn} {self.chain}{self.resi}{insertion}"


@dataclass
class BreakEvent:
    chain: str
    status: str
    prev: Residue
    curr: Residue
    numbering_gap: bool
    c_n_distance: float | None
    ca_ca_distance: float | None
    missing_atoms: str | None

    @property
    def is_binder_chain(self) -> bool:
        return False


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def iter_pdb_paths(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    return sorted(p for p in path.rglob(pattern) if p.is_file())


def parse_pdb(path: Path) -> dict[str, OrderedDict[tuple[int, str, str], Residue]]:
    """
    Parse ATOM/HETATM records into ordered residues per chain.

    Residue key is (residue number, insertion code, residue name). This keeps
    insertion codes distinct while preserving file order.
    """
    chains: dict[str, OrderedDict[tuple[int, str, str], Residue]] = defaultdict(OrderedDict)

    with open_text(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom = line[12:16].strip()
            if atom not in BACKBONE_ATOMS_FOR_BREAK_CHECK and atom != "O":
                # O is not needed for C-N break detection, but keeping it is harmless
                # if later you want to extend the script.
                pass

            resn = line[17:20].strip()
            chain = line[21].strip() or "_"

            try:
                resi = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            icode = line[26].strip()
            key = (resi, icode, resn)

            if key not in chains[chain]:
                chains[chain][key] = Residue(
                    chain=chain,
                    resi=resi,
                    icode=icode,
                    resn=resn,
                    atoms={},
                )

            chains[chain][key].atoms[atom] = (x, y, z)

    return chains


def classify_adjacent_residues(
    prev: Residue,
    curr: Residue,
    cn_cutoff: float,
) -> BreakEvent | None:
    """
    Return a BreakEvent if residue numbering jumps or the peptide bond is physically broken.
    Return None for normal consecutive residues with a normal C-N bond.
    """
    numbering_gap = curr.resi != prev.resi + 1

    missing = []
    if "C" not in prev.atoms:
        missing.append(f"{prev.chain}{prev.resi}.C")
    if "N" not in curr.atoms:
        missing.append(f"{curr.chain}{curr.resi}.N")

    c_n_distance = None
    ca_ca_distance = None

    if "CA" in prev.atoms and "CA" in curr.atoms:
        ca_ca_distance = distance(prev.atoms["CA"], curr.atoms["CA"])

    if missing:
        # Missing C/N makes the physical peptide bond check unavailable.
        # Treat it as a real break if it is also a numbering gap; otherwise report it
        # as REAL_BREAK as a conservative data-quality issue.
        status = "REAL_BREAK"
        return BreakEvent(
            chain=prev.chain,
            status=status,
            prev=prev,
            curr=curr,
            numbering_gap=numbering_gap,
            c_n_distance=c_n_distance,
            ca_ca_distance=ca_ca_distance,
            missing_atoms=", ".join(missing),
        )

    c_n_distance = distance(prev.atoms["C"], curr.atoms["N"])
    physical_break = c_n_distance > cn_cutoff

    if not numbering_gap and not physical_break:
        return None

    status = "REAL_BREAK" if physical_break else "NUMBERING_ONLY"
    return BreakEvent(
        chain=prev.chain,
        status=status,
        prev=prev,
        curr=curr,
        numbering_gap=numbering_gap,
        c_n_distance=c_n_distance,
        ca_ca_distance=ca_ca_distance,
        missing_atoms=None,
    )


def find_break_events(
    chains: dict[str, OrderedDict[tuple[int, str, str], Residue]],
    cn_cutoff: float,
) -> list[BreakEvent]:
    events: list[BreakEvent] = []

    for chain, residues_od in chains.items():
        residues = list(residues_od.values())
        for prev, curr in zip(residues[:-1], residues[1:]):
            event = classify_adjacent_residues(prev, curr, cn_cutoff=cn_cutoff)
            if event is not None:
                events.append(event)

    return events


def format_event(event: BreakEvent) -> str:
    if event.c_n_distance is None:
        cn_str = "C-N=NA"
    else:
        cn_str = f"C-N={event.c_n_distance:.2f} Å"

    ca_str = ""
    if event.ca_ca_distance is not None:
        ca_str = f", CA-CA={event.ca_ca_distance:.2f} Å"

    missing_str = ""
    if event.missing_atoms:
        missing_str = f", missing={event.missing_atoms}"

    numbering_str = "numbering_gap=yes" if event.numbering_gap else "numbering_gap=no"

    return (
        f"{event.status}: {event.prev.label} -> {event.curr.label}, "
        f"{cn_str}{ca_str}, {numbering_str}{missing_str}"
    )


def summarize_events(events: Iterable[BreakEvent], binder_chain: str) -> dict[str, int]:
    counts = Counter()
    for event in events:
        counts[event.status] += 1
        counts[f"{event.status}_chain_{event.chain}"] += 1
        if event.chain == binder_chain:
            counts[f"{event.status}_BINDER"] += 1
    return dict(counts)


def analyze_file(path: Path, cn_cutoff: float, binder_chain: str) -> tuple[list[BreakEvent], dict[str, int]]:
    chains = parse_pdb(path)
    events = find_break_events(chains, cn_cutoff=cn_cutoff)
    counts = summarize_events(events, binder_chain=binder_chain)
    return events, counts


def print_file_report(path: Path, events: list[BreakEvent], binder_chain: str) -> None:
    print(f"\n## {path}")

    if not events:
        print("Clean: no numbering jumps or physical C-N breaks detected.")
        return

    events_by_chain: dict[str, list[BreakEvent]] = defaultdict(list)
    for event in events:
        events_by_chain[event.chain].append(event)

    for chain in sorted(events_by_chain):
        suffix = " (binder)" if chain == binder_chain else ""
        print(f"\nChain {chain}{suffix}")
        for event in events_by_chain[chain]:
            print(format_event(event))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check residue-number jumps and physical peptide-bond breaks in PDB/PDB.GZ files."
    )
    parser.add_argument("input", type=Path, help="PDB/PDB.GZ file or directory to scan.")
    parser.add_argument(
        "--pattern",
        default="*.pdb*",
        help='Glob pattern used when input is a directory. Default: "*.pdb*"',
    )
    parser.add_argument(
        "--cn-cutoff",
        type=float,
        default=2.0,
        help="C_i to N_next distance cutoff in Å for REAL_BREAK. Default: 2.0",
    )
    parser.add_argument(
        "--binder-chain",
        default="B",
        help='Binder chain ID to count separately. Default: "B"',
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file event details and print only summary tables.",
    )
    args = parser.parse_args()

    paths = iter_pdb_paths(args.input, args.pattern)

    global_counts = Counter()
    files_with_issues = 0
    files_clean = 0

    print(f"Scanning {len(paths)} file(s)")
    print(f"Binder chain counted separately: {args.binder_chain}")

    for path in paths:
        events, counts = analyze_file(path, cn_cutoff=args.cn_cutoff, binder_chain=args.binder_chain)

        if events:
            files_with_issues += 1
        else:
            files_clean += 1

        global_counts["REAL_BREAK"] += counts.get("REAL_BREAK", 0)
        global_counts["NUMBERING_ONLY"] += counts.get("NUMBERING_ONLY", 0)
        global_counts["BINDER_REAL_BREAK"] += counts.get("REAL_BREAK_BINDER", 0)
        global_counts["BINDER_NUMBERING_ONLY"] += counts.get("NUMBERING_ONLY_BINDER", 0)

        if not args.quiet:
            print_file_report(path, events, binder_chain=args.binder_chain)

            file_real = counts.get("REAL_BREAK", 0)
            file_numbering = counts.get("NUMBERING_ONLY", 0)
            file_b_real = counts.get("REAL_BREAK_BINDER", 0)
            file_b_numbering = counts.get("NUMBERING_ONLY_BINDER", 0)
            print(
                "\nFile summary: "
                f"REAL_BREAK={file_real}, "
                f"NUMBERING_ONLY={file_numbering}, "
                f"{args.binder_chain}_REAL_BREAK={file_b_real}, "
                f"{args.binder_chain}_NUMBERING_ONLY={file_b_numbering}"
            )

    print("\n## Summary")
    print(
        "| Files | With issues | Clean | REAL_BREAK | NUMBERING_ONLY | "
        f"{args.binder_chain}_REAL_BREAK | {args.binder_chain}_NUMBERING_ONLY |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {len(paths)} | {files_with_issues} | {files_clean} | "
        f"{global_counts['REAL_BREAK']} | {global_counts['NUMBERING_ONLY']} | "
        f"{global_counts['BINDER_REAL_BREAK']} | {global_counts['BINDER_NUMBERING_ONLY']} |"
    )


if __name__ == "__main__":
    main()
