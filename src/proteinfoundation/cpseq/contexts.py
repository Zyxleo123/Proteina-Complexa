"""Parent contexts: an LP reference sequence plus the target it was solved against.

A "context" is one parent of the dataset: a receptor (one or more chains, as deposited),
a linear peptide bound to it, that peptide's sequence, and the pocket the peptide sits
in. Every candidate sequence for that parent inherits the *same* target conditioning --
same chains, same crop, same target-only template, same pocket definition -- so a score
difference between two candidates is a sequence effect and not a quietly different
target.

Sources
-------
`LNR` (Zenodo 13373108, staged at `$ZFS/LNR`): 93 experimentally determined linear
peptide-receptor complexes, listed in `test.txt` as `pdb_id receptor_chain peptide_chain`.
**Development data**, as `preprocess_lp.py` already treats it (it holds LNR out of the
training pool at 0.40 sequence identity).

Pocket definition
-----------------
Defined **once per parent**, from the parent's own bound LP pose, and then reused
unchanged for every mutant of that parent. It is a *localisation* -- which receptor
residues line the site -- and nothing in it restrains individual peptide residues to
their LP contacts. Contact retention is not an objective of this dataset, so the pocket
exists only to tell each predictor where on the receptor to look, and the same pocket
goes to all of them.

Deriving it from the parent LP pose means the pocket is not derived from any candidate's
own (unknown) pose, so it cannot leak an answer. It does mean the pocket is as good as
the parent's crystal structure, which is the intended trade.

Family split
------------
Receptors are clustered by **sequence identity** with mmseqs (already installed; see
`.env`'s `MMSEQS_EXEC`), and the split is over *clusters*, not over parents. Everything
derived from one parent -- all nine candidates, every chemistry subset, both controls
sharing a receptor -- lands in the same part, because a mutant sharing a part boundary
with its own reference would make a scorer look like it generalises when it has merely
memorised the target.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from proteinfoundation.cpseq.candidates import THREE_TO_ONE

BACKBONE = ("N", "CA", "C", "O")
STANDARD_AA = frozenset(THREE_TO_ONE)

# Contact definition for the pocket: any receptor residue with a heavy atom within this
# distance of any peptide heavy atom. 8.0 A is wide enough to include the second shell
# that shapes the site without pulling in most of the protein; it is a localisation
# radius, not a contact-retention restraint.
POCKET_RADIUS_A = 8.0

# The CP work in this project runs on 5-16mers (CPSea's curation rule, and what
# `configs/dataset/unified/cpsea_peptide.yaml` sets as binder_min/max_length). An LP
# reference outside that range cannot be turned into a comparable ring here.
PEPTIDE_MIN_LEN, PEPTIDE_MAX_LEN = 5, 16


@dataclass
class ParentContext:
    parent_id: str
    source: str
    pdb_id: str
    structure_path: str
    receptor_chains: tuple[str, ...]
    peptide_chain: str
    reference_sequence: str
    peptide_length: int
    # 0-based indices into the receptor's own residue list, per chain, for the pocket.
    pocket_residues: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # PDB-numbered pocket residues, which is what the predictors' inputs actually name.
    pocket_resseq: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # Min heavy-atom distance from each pocket residue to the peptide, aligned with the
    # two above. Ranks the pocket so a small, satisfiable CONDITIONING set can be drawn
    # from it; the full set stays descriptive.
    pocket_min_dist: dict[str, tuple[float, ...]] = field(default_factory=dict)
    receptor_sequences: dict[str, str] = field(default_factory=dict)
    family_id: str | None = None
    split: str | None = None
    is_control: bool = False
    note: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["receptor_chains"] = ",".join(self.receptor_chains)
        d["pocket_residues"] = json.dumps({k: list(v) for k, v in self.pocket_residues.items()})
        d["pocket_resseq"] = json.dumps({k: list(v) for k, v in self.pocket_resseq.items()})
        d["pocket_min_dist"] = json.dumps({k: list(v) for k, v in self.pocket_min_dist.items()})
        d["receptor_sequences"] = json.dumps(self.receptor_sequences)
        d["n_pocket_residues"] = sum(len(v) for v in self.pocket_residues.values())
        return d


def _open_maybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def parse_structure(path: Path) -> dict[str, "OrderedDict"]:
    """{chain -> OrderedDict[(resseq, icode) -> {atom_name: (x, y, z)}]}.

    First MODEL only, ATOM records only, standard amino acids, no hydrogens, altloc
    blank or 'A'. Coordinates are read, never transformed.
    """
    chains: dict[str, OrderedDict] = {}
    resnames: dict[str, dict] = {}
    with _open_maybe_gz(path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            resname = line[17:20].strip()
            if resname not in STANDARD_AA:
                continue
            element = line[76:78].strip()
            atom = line[12:16].strip()
            if element == "H" or atom.startswith("H"):
                continue
            chain = line[21]
            key = (int(line[22:26]), line[26])
            ch = chains.setdefault(chain, OrderedDict())
            rn = resnames.setdefault(chain, {})
            ch.setdefault(key, {})[atom] = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
            rn[key] = resname
    return {"atoms": chains, "resnames": resnames}


def chain_sequence(parsed: dict, chain: str) -> str:
    rn = parsed["resnames"].get(chain, {})
    return "".join(THREE_TO_ONE[rn[k]] for k in parsed["atoms"].get(chain, {}) if k in rn)


def pocket_for(
    parsed: dict,
    receptor_chains: tuple[str, ...],
    peptide_chain: str,
    radius_A: float = POCKET_RADIUS_A,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], dict[str, tuple[float, ...]]]:
    """Receptor residues lining the peptide site.

    Returns (0-based index per chain, PDB resSeq per chain, min heavy-atom distance to
    the peptide per chain), the three aligned element-for-element.
    """
    pep = parsed["atoms"].get(peptide_chain, {})
    pep_xyz = np.array([xyz for res in pep.values() for xyz in res.values()], dtype=np.float64)
    if pep_xyz.size == 0:
        raise ValueError(f"peptide chain {peptide_chain!r} has no heavy atoms")

    local: dict[str, tuple[int, ...]] = {}
    resseq: dict[str, tuple[int, ...]] = {}
    mindist: dict[str, tuple[float, ...]] = {}
    r2 = radius_A * radius_A
    for chain in receptor_chains:
        keys = list(parsed["atoms"].get(chain, {}))
        hits_local, hits_resseq, hits_d = [], [], []
        for idx, key in enumerate(keys):
            xyz = np.array(list(parsed["atoms"][chain][key].values()), dtype=np.float64)
            d2 = ((xyz[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1)
            dmin2 = float(d2.min())
            if dmin2 <= r2:
                hits_local.append(idx)
                hits_resseq.append(key[0])
                # Kept because the *conditioning* pocket has to be a small, satisfiable
                # set of genuine contacts, which needs a ranking -- see
                # `targets.pocket_contacts_for`.
                hits_d.append(round(float(np.sqrt(dmin2)), 3))
        local[chain] = tuple(hits_local)
        resseq[chain] = tuple(hits_resseq)
        mindist[chain] = tuple(hits_d)
    return local, resseq, mindist


def contacting_chains(
    parsed: dict, peptide_chain: str, radius_A: float = 5.0
) -> tuple[str, ...]:
    """Every chain with a heavy atom near the peptide, peptide excluded.

    `README_TARGET_DISTRIBUTION_MATCHING.md` records four LNR cases where a contacting
    chain was discarded. Dropping a chain that touches the peptide changes the pocket, so
    the full contacting set is computed and kept rather than trusting `test.txt`'s single
    receptor chain.
    """
    pep = parsed["atoms"].get(peptide_chain, {})
    pep_xyz = np.array([xyz for res in pep.values() for xyz in res.values()], dtype=np.float64)
    if pep_xyz.size == 0:
        return ()
    out, r2 = [], radius_A * radius_A
    for chain, residues in parsed["atoms"].items():
        if chain == peptide_chain:
            continue
        xyz = np.array([x for res in residues.values() for x in res.values()], dtype=np.float64)
        if xyz.size == 0:
            continue
        # Chunked to keep the pairwise block small for large receptors.
        hit = False
        for s in range(0, len(xyz), 4096):
            blk = xyz[s : s + 4096]
            if (((blk[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1) <= r2).any():
                hit = True
                break
        if hit:
            out.append(chain)
    return tuple(sorted(out))


def load_lnr_contexts(
    lnr_root: Path,
    list_file: str = "test.txt",
    *,
    min_len: int = PEPTIDE_MIN_LEN,
    max_len: int = PEPTIDE_MAX_LEN,
) -> tuple[list[ParentContext], list[dict]]:
    """Parse LNR into parent contexts. Returns (contexts, rejected_rows)."""
    lnr_root = Path(lnr_root)
    contexts: list[ParentContext] = []
    rejected: list[dict] = []

    for raw in (lnr_root / list_file).read_text().splitlines():
        parts = raw.split()
        if len(parts) < 3:
            continue
        pdb_id, rec_chain, pep_chain = parts[0], parts[1], parts[2]
        path = lnr_root / "pdbs" / f"{pdb_id}.pdb"
        if not path.exists():
            rejected.append({"pdb_id": pdb_id, "reason": "structure file missing"})
            continue
        parsed = parse_structure(path)
        if pep_chain not in parsed["atoms"]:
            rejected.append({"pdb_id": pdb_id, "reason": f"peptide chain {pep_chain} absent"})
            continue

        seq = chain_sequence(parsed, pep_chain)
        if not (min_len <= len(seq) <= max_len):
            rejected.append(
                {
                    "pdb_id": pdb_id,
                    "reason": f"peptide length {len(seq)} outside [{min_len},{max_len}]",
                    "peptide_length": len(seq),
                }
            )
            continue

        rec_chains = contacting_chains(parsed, pep_chain)
        note = ""
        if rec_chain not in rec_chains:
            note = f"test.txt names receptor chain {rec_chain} but it does not contact the peptide"
        extra = tuple(c for c in rec_chains if c != rec_chain)
        if extra:
            note = (note + "; " if note else "") + (
                f"kept {len(extra)} further contacting chain(s) {','.join(extra)} that "
                "test.txt does not list"
            )
        if not rec_chains:
            rejected.append({"pdb_id": pdb_id, "reason": "no receptor chain contacts the peptide"})
            continue

        local, resseq, mindist = pocket_for(parsed, rec_chains, pep_chain)
        contexts.append(
            ParentContext(
                parent_id=f"lnr_{pdb_id}",
                source="LNR",
                pdb_id=pdb_id,
                structure_path=str(path),
                receptor_chains=rec_chains,
                peptide_chain=pep_chain,
                reference_sequence=seq,
                peptide_length=len(seq),
                pocket_residues=local,
                pocket_resseq=resseq,
                pocket_min_dist=mindist,
                receptor_sequences={c: chain_sequence(parsed, c) for c in rec_chains},
                note=note,
            )
        )
    return contexts, rejected


def _pairwise_identity_clusters(
    names: list[str], seqs: list[str], min_seq_id: float, coverage: float
) -> dict[str, str]:
    """Single-linkage clusters at `min_seq_id`, using Biopython's aligner.

    Fallback for when mmseqs is not installed -- which it is not in this environment,
    despite `.env` exporting `MMSEQS_EXEC`: that path holds no binary (neither does
    `FOLDSEEK_EXEC`). Rather than skip the family split, which is the one thing that has
    to exist *before* variants are generated, receptors are compared directly.

    Global alignment, BLOSUM62, identity normalised by the **shorter** sequence, and a
    coverage floor on the alignment length, so a short peptide-binding domain matching
    inside a large multidomain receptor still clusters with it. O(n^2) alignments, which
    is nothing at this scale (63 receptors) and would need mmseqs at thousands.

    Single linkage is deliberate: it is the conservative choice for a *split*, because it
    merges on any evidence of relatedness and therefore errs toward keeping possibly
    related targets on the same side.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    # LOCAL, not global. A global alignment of a 50-mer against a 500-residue receptor
    # spreads chance matches across the whole long sequence; normalising those by the
    # shorter sequence then clears 0.3 for unrelated pairs, and under single linkage the
    # first such false edge chains every family into one. Measured: global+shorter-norm
    # put all 63 LNR receptors in a single family. Local alignment with identity over the
    # aligned columns is the conventional definition and the one mmseqs reports.
    aligner.mode = "local"

    n = len(seqs)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a in range(n):
        for b in range(a + 1, n):
            short = min(len(seqs[a]), len(seqs[b]))
            if short == 0:
                continue
            aln = aligner.align(seqs[a], seqs[b])[0]
            ta, tb = aln[0], aln[1]
            matches = sum(1 for x, y in zip(ta, tb) if x == y and x != "-")
            aligned = sum(1 for x, y in zip(ta, tb) if x != "-" and y != "-")
            if aligned == 0:
                continue
            # identity over the aligned columns; coverage of the shorter sequence.
            if matches / aligned >= min_seq_id and aligned / short >= coverage:
                union(a, b)

    # Name each family after its lowest-indexed member, so the id is stable under
    # reordering of the input.
    return {names[i]: f"fam_{names[find(i)]}" for i in range(n)}


def assign_families(
    contexts: list[ParentContext],
    work_dir: Path,
    mmseqs_exec: str | None = None,
    *,
    min_seq_id: float = 0.3,
    coverage: float = 0.5,
) -> dict[str, str]:
    """Cluster receptors by sequence identity; returns {parent_id -> family_id}.

    One parent's family is the cluster of its **longest** receptor chain. 0.3 identity at
    0.5 coverage is a conventional remote-homology threshold; it is recorded with the
    split so the boundary is reproducible rather than implicit.

    Parents whose receptors cluster together get one family, so two deposits of the same
    protein cannot land on opposite sides of the split.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    names, seqs = [], []
    for ctx in contexts:
        longest = max(ctx.receptor_sequences.items(), key=lambda kv: len(kv[1]))[1]
        names.append(ctx.parent_id)
        seqs.append(longest)
    fasta = work_dir / "receptors.fasta"
    with open(fasta, "w") as fh:
        for name, seq in zip(names, seqs):
            fh.write(f">{name}\n{seq}\n")

    if not mmseqs_exec or not Path(mmseqs_exec).exists():
        logger_msg = (
            f"mmseqs not available at {mmseqs_exec!r}; clustering receptors with the "
            "built-in pairwise-identity fallback"
        )
        print(f"[cpseq.contexts] {logger_msg}")
        (work_dir / "clustering_method.json").write_text(
            json.dumps(
                {
                    "method": "biopython_pairwise_single_linkage",
                    "reason": f"mmseqs binary absent at {mmseqs_exec!r}",
                    "min_seq_id": min_seq_id,
                    "coverage": coverage,
                    "identity_normalised_by": "shorter_sequence",
                    "linkage": "single",
                },
                indent=2,
            )
        )
        return _pairwise_identity_clusters(names, seqs, min_seq_id, coverage)

    (work_dir / "clustering_method.json").write_text(
        json.dumps(
            {
                "method": "mmseqs_easy_cluster",
                "mmseqs_exec": str(mmseqs_exec),
                "min_seq_id": min_seq_id,
                "coverage": coverage,
                "cov_mode": 0,
            },
            indent=2,
        )
    )
    out_prefix = work_dir / "rec_clu"
    cmd = [
        mmseqs_exec, "easy-cluster", str(fasta), str(out_prefix), str(work_dir / "tmp"),
        "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
        "-v", "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    rep_of: dict[str, str] = {}
    for line in (out_prefix.parent / f"{out_prefix.name}_cluster.tsv").read_text().splitlines():
        rep, member = line.split("\t")[:2]
        rep_of[member] = rep
    return {ctx.parent_id: f"fam_{rep_of.get(ctx.parent_id, ctx.parent_id)}" for ctx in contexts}


def split_by_family(
    contexts: list[ParentContext],
    families: dict[str, str],
    *,
    dev_fraction: float = 1.0,
    seed: int = 42,
) -> dict[str, str]:
    """{parent_id -> split}. LNR is development data, so it goes to `dev` whole.

    `dev_fraction=1.0` is the honest default for this tranche: LNR has already influenced
    this repository's preprocessing and sampling decisions, so holding part of it out
    would produce a test set that is not independent. A real held-out set needs target
    families from a source that has not touched the pipeline; until one is staged, the
    missing coverage is reported rather than faked.
    """
    by_family: dict[str, list[str]] = defaultdict(list)
    for ctx in contexts:
        by_family[families[ctx.parent_id]].append(ctx.parent_id)

    if dev_fraction >= 1.0:
        return {pid: "dev" for pids in by_family.values() for pid in pids}

    rng = np.random.default_rng(seed)
    fams = sorted(by_family)
    rng.shuffle(fams)
    n_dev = int(round(dev_fraction * len(fams)))
    out = {}
    for k, fam in enumerate(fams):
        part = "dev" if k < n_dev else "heldout"
        for pid in by_family[fam]:
            out[pid] = part
    return out
