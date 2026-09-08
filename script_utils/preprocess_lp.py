#!/usr/bin/env python3
"""Convert PepBench / ProtFrag linear peptide-protein complexes into Proteina training format.

This is the linear-peptide (LP) counterpart of ``preprocess_cpsea.py``. It writes the
*same* on-disk layout and the *same* metadata schema, so a mixed metadata file can simply
concatenate the two (see ``build_mixed_metadata.py``).

Why these two datasets. LNR is a 93-target *test* benchmark with no training split; the
training resources that go with it are PepBench train (4,157 experimental peptide-protein
complexes) and ProtFrag (70,498 synthetic peptide-like fragments cut out of protein
contexts). Both are distributed from the PepBench Zenodo record. They are here to expose
the flow model's *target-conditioning* trunk to experimental PDB receptors of the kind LNR
contains -- the receptor is never encoded by the VAE, so no amount of VAE training fixes
target OOD.

Differences from the CPSea preprocessor, all of them consequences of the data:

  * **Chain mapping is per entry, not global.** CPSea is uniformly ``R``/``L``; PepBench
    names the receptor and peptide chains per complex in its index file (ProtFrag happens
    to be ``R``/``L`` throughout but is read the same way, from its own index). Getting
    this backwards would train the model to generate the *receptor*, so the mapping is
    read from the index and the resulting chain set is asserted to be exactly ``{A, B}``.
  * **No CONECT records.** These peptides are linear. Every row is written with
    ``cyclization_type="linear"``, which ``CyclizationLabelTransform`` turns into the
    ``LINEAR`` conditioning token rather than ``UNSPECIFIED`` -- see that transform's
    ``linear_type_names``. Labeling them UNSPECIFIED instead would quietly redefine the
    classifier-free-guidance null as "linear peptide".
  * **LNR holdout.** LNR is the held-out test set, so any training entry that is the same
    structure as, or a close homolog of, an LNR receptor is leakage. Two filters run:
    exact PDB-ID match, and receptor sequence identity above ``--lnr-identity-threshold``
    (k-mer prefilter, then pairwise alignment on the survivors). Dropped entries are
    written to ``manifest/lnr_excluded.csv`` with the reason and the matched LNR target,
    so the filter is auditable rather than a number in a log line. The filter runs
    *after* the PDBs are written (it needs the cleaned receptor sequence), so excluded
    entries leave a file on disk but never enter the metadata parquet -- and the metadata
    is the only thing training reads.

Output layout (default under ``$LP_DATA_PATH/preprocessed``):

  {out_dir}/processed/{train,val}/*.pdb
  {out_dir}/metadata/lp_{train,val}.parquet
  {out_dir}/manifest/run_config.json
  {out_dir}/manifest/preprocess_stats.json
  {out_dir}/manifest/per_structure_audit.csv
  {out_dir}/manifest/rejected.csv
  {out_dir}/manifest/lnr_excluded.csv

Splits come from the datasets themselves, not from re-clustering: PepBench ships
``train.txt`` / ``valid.txt``, and ProtFrag ships a single ``all.txt`` that is training
data in its entirety (CP-Composer and PepGLAD both use it that way).

Run (see ``scripts/preprocess_lp.sbatch``):

    python -m script_utils.preprocess_lp \\
        --pepbench-root $ZFS/LPData/PepBench/train_valid \\
        --protfrag-root $ZFS/LPData/ProtFrag \\
        --lnr-root      $ZFS/LNR \\
        --out-dir       $ZFS/LPData/preprocessed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

# Reuse the CPSea primitives that are genuinely format-generic. The chain remapping and
# the cyclization inference are NOT reused: those encode CPSea-specific assumptions.
from script_utils.preprocess_cpsea import (
    STANDARD_AA,
    AtomRecord,
    format_atom_line,
    parse_pdb_lines,
)

# Three-letter to one-letter, for the receptor sequences the LNR filter compares.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

TARGET_CHAIN = "A"  # receptor
BINDER_CHAIN = "B"  # peptide

# The metadata value that makes CyclizationLabelTransform emit the LINEAR token. Must
# stay inside `CyclizationLabelTransform.LINEAR_TYPE_NAMES`; asserted in the tests.
LINEAR_TYPE_NAME = "linear"


@dataclass
class LPAudit:
    example_id: str
    source_path: str
    dataset_source: str
    status: str = "ok"
    reject_reason: str = ""
    input_atoms: int = 0
    input_chains: str = ""
    receptor_chain_in: str = ""
    peptide_chain_in: str = ""
    removed_hydrogen: int = 0
    removed_non_standard: int = 0
    removed_other_chain: int = 0
    output_atoms: int = 0
    output_chains: str = ""
    peptide_length: int = 0
    receptor_length: int = 0
    coord_max_delta_A: float = 0.0
    output_path: str = ""
    split: str = ""
    receptor_seq: str = ""
    intended_removals: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Index files
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexEntry:
    stem: str
    receptor_chain: str
    peptide_chain: str
    dataset_source: str
    split: str


def read_index(path: Path, dataset_source: str, split: str) -> list[IndexEntry]:
    """Parse a PepBench-format index: ``<stem>\\t<receptor_chain>\\t<peptide_chain>\\t...``.

    LNR's ``test.txt``, PepBench's ``train.txt``/``valid.txt`` and ProtFrag's ``all.txt``
    all share this format.
    """
    entries: list[IndexEntry] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"{path}:{lineno}: expected at least 3 tab-separated fields, got {parts!r}")
        stem, rec, pep = parts[0], parts[1].strip(), parts[2].strip()
        if not rec or not pep:
            raise ValueError(f"{path}:{lineno}: empty chain id in {parts!r}")
        if rec == pep:
            raise ValueError(f"{path}:{lineno}: receptor and peptide chain are both {rec!r}")
        entries.append(IndexEntry(stem, rec, pep, dataset_source, split))
    return entries


def pdb_id_from_stem(stem: str) -> str | None:
    """Extract the 4-character PDB accession from a dataset stem.

    PepBench: ``A_B_pdb2isq``  -> ``2isq``
    ProtFrag: ``pdb2rsx_124_131`` -> ``2rsx``
    LNR:      ``1bjr``          -> ``1bjr``

    Returns None when no accession can be read, which the caller treats as "cannot
    prove it is safe by ID" -- the sequence filter still applies.
    """
    for token in stem.split("_"):
        if token.startswith("pdb") and len(token) >= 7:
            return token[3:7].lower()
    if len(stem) == 4:
        return stem.lower()
    return None


# --------------------------------------------------------------------------------------
# Structure processing
# --------------------------------------------------------------------------------------


def _classify_removal(atom: AtomRecord, keep_chains: dict[str, str]) -> str | None:
    """Why this atom is dropped, or None to keep it.

    Mirrors the CPSea rules (hydrogens and non-standard residues out) but takes the
    chains to keep as an argument, since LP chain ids vary per entry.
    """
    name = atom.atom_name.strip()
    if name.startswith("H") or (name and name[0] == "H"):
        return "hydrogen"
    if atom.chain_id not in keep_chains:
        return "other_chain"
    if atom.resname.strip() not in STANDARD_AA:
        return "non_standard_residue"
    if atom.record_type == "HETATM":
        return "non_standard_residue"
    return None


def _remap(atom: AtomRecord, new_chain: str) -> AtomRecord:
    line = atom.line[:21] + new_chain + atom.line[22:]
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
        line=line,
    )


def _chain_residues(atoms: list[AtomRecord], chain_id: str) -> list[tuple[int, str]]:
    """Ordered (resseq, resname) for residues of `chain_id` that have a CA."""
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for a in atoms:
        if a.chain_id == chain_id and a.atom_name.strip() == "CA" and a.resseq not in seen:
            seen.add(a.resseq)
            out.append((a.resseq, a.resname.strip()))
    return out


def _sequence(atoms: list[AtomRecord], chain_id: str) -> str:
    return "".join(THREE_TO_ONE.get(rn, "X") for _, rn in _chain_residues(atoms, chain_id))


def write_pdb(path: Path, atoms: list[AtomRecord], example_id: str, dataset_source: str) -> None:
    """Write the cleaned complex. No CONECT block: these peptides are linear."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"REMARK 999 LP PREPROCESS example_id={example_id} source={dataset_source}\n")
        f.write(f"REMARK 999 CHAIN MAP: receptor->{TARGET_CHAIN} (target), peptide->{BINDER_CHAIN} (binder)\n")
        f.write("REMARK 999 REMOVED (intended): hydrogens, non-standard residues, other chains\n")
        f.write("REMARK 999 LINEAR peptide: no cyclization, no CONECT records\n")
        prev_chain = None
        for i, atom in enumerate(atoms):
            if prev_chain is not None and atom.chain_id != prev_chain:
                f.write("TER\n")
            prev_chain = atom.chain_id
            f.write(format_atom_line(atom, i + 1))
        f.write("TER\nEND\n")


def process_structure(
    source_path: Path,
    entry: IndexEntry,
    peptide_min: int,
    peptide_max: int,
    receptor_min: int,
) -> tuple[list[AtomRecord], LPAudit]:
    audit = LPAudit(
        example_id=entry.stem,
        source_path=str(source_path),
        dataset_source=entry.dataset_source,
        split=entry.split,
        receptor_chain_in=entry.receptor_chain,
        peptide_chain_in=entry.peptide_chain,
    )
    if not source_path.exists():
        audit.status = "rejected"
        audit.reject_reason = "missing_source_pdb"
        return [], audit

    atoms, _conect = parse_pdb_lines(source_path.read_text().splitlines(keepends=True))
    audit.input_atoms = len(atoms)
    audit.input_chains = ",".join(sorted({a.chain_id for a in atoms}))

    keep_chains = {entry.receptor_chain: TARGET_CHAIN, entry.peptide_chain: BINDER_CHAIN}
    kept: list[AtomRecord] = []
    removals: Counter = Counter()
    for atom in atoms:
        reason = _classify_removal(atom, keep_chains)
        if reason is not None:
            removals[reason] += 1
            continue
        kept.append(_remap(atom, keep_chains[atom.chain_id]))

    audit.removed_hydrogen = removals["hydrogen"]
    audit.removed_non_standard = removals["non_standard_residue"]
    audit.removed_other_chain = removals["other_chain"]
    audit.intended_removals = dict(removals)

    chains = sorted({a.chain_id for a in kept})
    if chains != [TARGET_CHAIN, BINDER_CHAIN]:
        audit.status = "rejected"
        audit.reject_reason = f"expected_chains_A_B_got_{chains}"
        return [], audit

    # Sort so the target chain is written first, matching CPSea's output ordering.
    kept.sort(key=lambda a: (a.chain_id != TARGET_CHAIN, a.resseq, a.serial))

    audit.peptide_length = len(_chain_residues(kept, BINDER_CHAIN))
    audit.receptor_length = len(_chain_residues(kept, TARGET_CHAIN))
    audit.receptor_seq = _sequence(kept, TARGET_CHAIN)

    if not (peptide_min <= audit.peptide_length <= peptide_max):
        audit.status = "rejected"
        audit.reject_reason = f"peptide_length_{audit.peptide_length}_outside_{peptide_min}_{peptide_max}"
        return [], audit
    if audit.receptor_length < receptor_min:
        audit.status = "rejected"
        audit.reject_reason = f"receptor_length_{audit.receptor_length}_below_{receptor_min}"
        return [], audit

    # Coordinates must survive untouched; only atom selection and the chain letter change.
    max_delta = 0.0
    by_serial = {a.serial: a for a in atoms}
    for a in kept:
        o = by_serial[a.serial]
        max_delta = max(max_delta, abs(a.x - o.x) + abs(a.y - o.y) + abs(a.z - o.z))
    audit.coord_max_delta_A = max_delta
    if max_delta > 1e-6:
        audit.status = "rejected"
        audit.reject_reason = f"coordinates_altered_{max_delta:.3e}A"
        return [], audit

    audit.output_atoms = len(kept)
    audit.output_chains = ",".join(chains)
    return kept, audit


def process_one(job: tuple[IndexEntry, Path, int, int, int, Path]) -> tuple[LPAudit, dict | None]:
    entry, source_path, peptide_min, peptide_max, receptor_min, out_dir = job
    atoms, audit = process_structure(source_path, entry, peptide_min, peptide_max, receptor_min)
    if audit.status != "ok":
        return audit, None

    out_path = out_dir / "processed" / entry.split / f"{entry.dataset_source}__{entry.stem}.pdb"
    example_id = f"{entry.dataset_source}__{entry.stem}"
    write_pdb(out_path, atoms, example_id, entry.dataset_source)
    audit.output_path = str(out_path.resolve())
    audit.example_id = example_id

    row = {
        "example_id": example_id,
        "path": audit.output_path,
        "binder_chain_id": BINDER_CHAIN,
        # No structural clustering is available for these sets (no foldseek/mmseqs in the
        # env), and the splits are the datasets' own, so cluster_id is the PDB accession.
        # That is enough to keep the same crystal structure from straddling train/val.
        "cluster_id": pdb_id_from_stem(entry.stem) or example_id,
        "split": entry.split,
        "peptide_length": audit.peptide_length,
        "receptor_length": audit.receptor_length,
        "cyclization_type": LINEAR_TYPE_NAME,
        "source_path": audit.source_path,
        "conect_pairs_kept": 0,
        "dataset_source": entry.dataset_source,
    }
    return audit, row


# --------------------------------------------------------------------------------------
# LNR holdout
# --------------------------------------------------------------------------------------


def _kmers(seq: str, k: int) -> set[str]:
    return {seq[i : i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else set()


def _identity(a: str, b: str) -> float:
    """Sequence identity of the optimal global alignment, over the shorter sequence.

    Normalizing by the shorter sequence (rather than alignment length) is deliberate: a
    200-residue construct that fully contains a 100-residue LNR receptor is leakage, and
    dividing by the 200-length alignment would score it ~50% and let it through.
    """
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import SubstitutionMatrix, align_optimal

    try:
        sa, sb = ProteinSequence(a), ProteinSequence(b)
    except Exception:  # noqa: BLE001 - non-standard letters; fall back to a k-mer estimate
        ka, kb = _kmers(a, 4), _kmers(b, 4)
        return len(ka & kb) / max(1, min(len(ka), len(kb)))
    matrix = SubstitutionMatrix.std_protein_matrix()
    alignment = align_optimal(sa, sb, matrix, gap_penalty=(-10, -1), terminal_penalty=False, max_number=1)[0]
    trace = alignment.trace
    # Index the plain strings, not the ProteinSequence objects: biotite indexing yields
    # symbol codes, and an accidental code-vs-letter comparison would silently score 0.
    matches = sum(
        1
        for i in range(trace.shape[0])
        if trace[i, 0] >= 0 and trace[i, 1] >= 0 and a[trace[i, 0]] == b[trace[i, 1]]
    )
    return matches / max(1, min(len(a), len(b)))


def build_lnr_filter(
    lnr_root: Path,
    kmer_size: int,
    containment_prefilter: float,
) -> tuple[set[str], list[tuple[str, str]], set[str]]:
    """Load the LNR test set: its PDB ids, its (id, receptor sequence) pairs, and kmers.

    Returns:
        (pdb_ids, [(lnr_id, receptor_seq)], union of all LNR receptor k-mers)
    """
    index = lnr_root / "test.txt"
    entries = read_index(index, dataset_source="lnr", split="test")
    ids: set[str] = set()
    seqs: list[tuple[str, str]] = []
    all_kmers: set[str] = set()
    for e in entries:
        pdb_id = pdb_id_from_stem(e.stem)
        if pdb_id:
            ids.add(pdb_id)
        pdb_path = lnr_root / "pdbs" / f"{e.stem}.pdb"
        if not pdb_path.exists():
            logger.warning(f"LNR entry {e.stem} has no PDB at {pdb_path}; ID filter still applies")
            continue
        atoms, _ = parse_pdb_lines(pdb_path.read_text().splitlines(keepends=True))
        seq = _sequence(atoms, e.receptor_chain)
        if len(seq) < kmer_size:
            logger.warning(f"LNR entry {e.stem} receptor chain {e.receptor_chain} has {len(seq)} residues; skipped")
            continue
        seqs.append((e.stem, seq))
        all_kmers |= _kmers(seq, kmer_size)
    logger.info(
        f"LNR holdout: {len(ids)} PDB ids, {len(seqs)} receptor sequences, "
        f"{len(all_kmers)} distinct {kmer_size}-mers (prefilter containment >= {containment_prefilter})"
    )
    return ids, seqs, all_kmers


def lnr_exclusions(
    rows: list[dict],
    audits: dict[str, LPAudit],
    lnr_ids: set[str],
    lnr_seqs: list[tuple[str, str]],
    lnr_kmers: set[str],
    kmer_size: int,
    containment_prefilter: float,
    identity_threshold: float,
) -> dict[str, tuple[str, str, float]]:
    """example_id -> (reason, matched LNR target, score) for every row that must be dropped.

    Two passes. The ID pass is exact and free. The sequence pass aligns only candidates
    whose receptor shares enough k-mers with the LNR pool to *possibly* clear the identity
    threshold -- an alignment per (candidate, LNR target) pair over 46k candidates would
    not finish, and a candidate sharing almost no k-mers with any LNR receptor cannot be
    40% identical to one.
    """
    excluded: dict[str, tuple[str, str, float]] = {}
    lnr_kmer_by_target = {name: _kmers(seq, kmer_size) for name, seq in lnr_seqs}

    n_prefiltered = 0
    for row in rows:
        eid = row["example_id"]
        if row["cluster_id"] in lnr_ids:
            excluded[eid] = ("pdb_id_match", row["cluster_id"], 1.0)
            continue
        seq = audits[eid].receptor_seq
        cand = _kmers(seq, kmer_size)
        if not cand:
            continue
        if len(cand & lnr_kmers) / len(cand) < containment_prefilter:
            continue
        n_prefiltered += 1
        best_name, best_id = "", 0.0
        for name, lseq in lnr_seqs:
            # Per-target prefilter too: no point aligning against an LNR receptor this
            # candidate shares nothing with just because it matched some other one.
            lk = lnr_kmer_by_target[name]
            if not lk or len(cand & lk) / min(len(cand), len(lk)) < containment_prefilter:
                continue
            ident = _identity(seq, lseq)
            if ident > best_id:
                best_id, best_name = ident, name
        if best_id >= identity_threshold:
            excluded[eid] = ("sequence_identity", best_name, best_id)

    logger.info(
        f"LNR holdout: {len(excluded)} excluded "
        f"({sum(1 for v in excluded.values() if v[0] == 'pdb_id_match')} by PDB id, "
        f"{sum(1 for v in excluded.values() if v[0] == 'sequence_identity')} by sequence identity); "
        f"{n_prefiltered} candidates survived the k-mer prefilter and were aligned"
    )
    return excluded


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def collect_inputs(args) -> list[tuple[IndexEntry, Path]]:
    jobs: list[tuple[IndexEntry, Path]] = []
    if args.pepbench_root:
        root = args.pepbench_root
        for fname, split in (("train.txt", "train"), ("valid.txt", "val")):
            index = root / fname
            if not index.exists():
                raise FileNotFoundError(f"PepBench index not found: {index}")
            for e in read_index(index, "pepbench", split):
                jobs.append((e, root / "pdbs" / f"{e.stem}.pdb"))
    if args.protfrag_root:
        root = args.protfrag_root
        index = root / "all.txt"
        if not index.exists():
            raise FileNotFoundError(f"ProtFrag index not found: {index}")
        # ProtFrag is training data in its entirety; it ships no validation split.
        for e in read_index(index, "protfrag", "train"):
            jobs.append((e, root / "pdbs" / f"{e.stem}.pdb"))
    if not jobs:
        raise SystemExit("Nothing to do: pass --pepbench-root and/or --protfrag-root")
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Preprocess PepBench/ProtFrag linear peptide complexes.")
    parser.add_argument("--pepbench-root", type=Path, default=None, help="dir with train.txt, valid.txt, pdbs/")
    parser.add_argument("--protfrag-root", type=Path, default=None, help="dir with all.txt, pdbs/")
    parser.add_argument("--lnr-root", type=Path, default=None, help="dir with test.txt, pdbs/ (held-out benchmark)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--peptide-min-length", type=int, default=5)
    parser.add_argument("--peptide-max-length", type=int, default=16)
    parser.add_argument("--receptor-min-length", type=int, default=30, help="matches CroppingTransform2.target_min_length")
    parser.add_argument("--lnr-identity-threshold", type=float, default=0.40)
    parser.add_argument("--lnr-kmer-size", type=int, default=6)
    parser.add_argument("--lnr-containment-prefilter", type=float, default=0.20)
    parser.add_argument("--skip-lnr-filter", action="store_true", help="do not filter against LNR (leaks the benchmark)")
    parser.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N structures PER dataset source (smoke tests). Per-source, not "
        "overall: a global cap would take them in listing order and yield PepBench only, so a "
        "smoke run would build a mix with no ProtFrag in it and fail the training preflight.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    out_dir: Path = args.out_dir
    (out_dir / "manifest").mkdir(parents=True, exist_ok=True)

    jobs = collect_inputs(args)
    if args.limit:
        per_source: Counter = Counter()
        capped = []
        for e, path in jobs:
            if per_source[e.dataset_source] >= args.limit:
                continue
            per_source[e.dataset_source] += 1
            capped.append((e, path))
        jobs = capped
        logger.info(f"--limit {args.limit} per source -> {dict(per_source)}")
    logger.info(f"Discovered {len(jobs)} input structures")

    run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_config["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    work = [
        (e, p, args.peptide_min_length, args.peptide_max_length, args.receptor_min_length, out_dir)
        for e, p in jobs
    ]
    audits: list[LPAudit] = []
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.nproc) as pool:
        for n, (audit, row) in enumerate(pool.map(process_one, work, chunksize=32), start=1):
            audits.append(audit)
            if row is not None:
                rows.append(row)
            if n % 2000 == 0:
                logger.info(f"  processed {n}/{len(work)} ({len(rows)} kept)")

    audit_by_id = {a.example_id: a for a in audits if a.status == "ok"}

    excluded: dict[str, tuple[str, str, float]] = {}
    if args.lnr_root and not args.skip_lnr_filter:
        lnr_ids, lnr_seqs, lnr_kmers = build_lnr_filter(
            args.lnr_root, args.lnr_kmer_size, args.lnr_containment_prefilter
        )
        excluded = lnr_exclusions(
            rows, audit_by_id, lnr_ids, lnr_seqs, lnr_kmers,
            args.lnr_kmer_size, args.lnr_containment_prefilter, args.lnr_identity_threshold,
        )
        pd.DataFrame(
            [
                {
                    "example_id": eid,
                    "dataset_source": audit_by_id[eid].dataset_source,
                    "reason": reason,
                    "matched_lnr_target": match,
                    "score": score,
                }
                for eid, (reason, match, score) in sorted(excluded.items())
            ]
        ).to_csv(out_dir / "manifest" / "lnr_excluded.csv", index=False)
        rows = [r for r in rows if r["example_id"] not in excluded]
    elif args.skip_lnr_filter:
        logger.warning("LNR filter SKIPPED: any LNR result from a model trained on this data is leaked.")
    else:
        logger.warning("No --lnr-root given: LNR filter did not run.")

    pd.DataFrame([asdict(a) for a in audits]).to_csv(out_dir / "manifest" / "per_structure_audit.csv", index=False)
    rejected = [asdict(a) for a in audits if a.status != "ok"]
    pd.DataFrame(rejected).to_csv(out_dir / "manifest" / "rejected.csv", index=False)

    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_df = pd.DataFrame(rows)
    split_counts: dict[str, int] = {}
    for split in ("train", "val"):
        part = meta_df[meta_df["split"] == split] if len(meta_df) else meta_df
        split_counts[split] = len(part)
        if len(part):
            path = meta_dir / f"lp_{split}.parquet"
            part.reset_index(drop=True).to_parquet(path, index=False)
            logger.info(f"Wrote {path} ({len(part)} rows)")

    stats = {
        "n_input": len(jobs),
        "n_ok": len(audit_by_id),
        "n_rejected": len(rejected),
        "n_lnr_excluded": len(excluded),
        "n_written": len(rows),
        "split_counts": split_counts,
        "source_counts": dict(Counter(r["dataset_source"] for r in rows)),
        "reject_reasons": dict(Counter(a["reject_reason"] for a in rejected).most_common(20)),
        "peptide_length_hist": dict(sorted(Counter(r["peptide_length"] for r in rows).items())),
    }
    (out_dir / "manifest" / "preprocess_stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "manifest" / "run_config.json").write_text(json.dumps(run_config, indent=2))
    logger.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
