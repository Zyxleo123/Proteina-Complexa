#!/usr/bin/env python3
"""Build the ONE common LP manifest that every generator arm trains against.

Four arms (PPFlow, POTFlow, PepGLAD fixed-sequence, RFdiffusion3) each have their own
native loader. Each one gets a thin adapter, but all four read *this* file, so that a
difference between arms is a difference in model and objective -- never a difference in
which examples they happened to see. The alternative, letting each upstream repo use its
own index/cache/split files, is how three arms end up trained on three subsets and the
comparison stops meaning anything.

WHAT IS ALREADY DONE AND NOT REDONE HERE. ``preprocess_lp.py`` already converted PepBench
and ProtFrag into one on-disk layout: chain ``A`` = receptor, chain ``B`` = peptide, no
hydrogens, no non-standard residues, no HETATM, no CONECT, and the LNR benchmark filtered
out (161 entries, by PDB id and by receptor-sequence identity). This script does not touch
structures. It selects rows, adds the fields that preprocessing did not record, and fixes
the split.

THE FOUR THINGS IT ADDS.

1.  **Peptide sequence.** The preprocessing audit recorded ``receptor_seq`` (it needed one
    for the LNR identity filter) but never the peptide sequence, and the metadata has only
    ``peptide_length``. PepGLAD's fixed-sequence mode conditions on the sequence and RFD3
    reveals a fraction of it, so both arms are unimplementable without this column. It is
    read back out of chain ``B`` of the written PDB -- the structure the arms actually
    train on -- rather than from the upstream source file, so a sequence that disagrees
    with the coordinates is a detected error instead of a silent one.

2.  **A leak-free, family-level split.** The inherited split comes from PepBench's own
    ``train.txt``/``valid.txt``, and it leaks: ProtFrag fragments were cut from the same
    PDB entries as 4 of the 80 PepBench validation complexes (families ``2m14``, ``2yvc``,
    ``7xbd``), so those receptors appear in training as fragments. Assignment here is a
    pure function of the family key, which makes that class of leak unrepresentable --
    a parent and all of its fragment derivatives share a ``cluster_id`` and therefore
    always land on the same side.

3.  **Reserved families.** Families that ``cp2lp_build_splits.py`` holds out for CP->LP
    evaluation are marked ``heldout`` and are not trained on and not validated on. They
    are reserved for the later CP-sequence comparison. This reuses ``family_of`` and
    ``in_eval_split`` at the same ``(eval_frac=0.15, seed=0)``, so the reserved set here
    is *identical* to the CP->LP eval set rather than merely similar -- the hash is
    deterministic across processes and stable as rows are added.

4.  **Measured topology.** ``preprocess_lp.py`` wrote every LP row
    ``cyclization_type="linear"`` with ``conect_pairs_kept=0`` -- because PepBench and
    ProtFrag ship no CONECT records at all. That label is therefore an assumption
    inherited from the source rather than a measurement, and a cyclic peptide passes it
    unchallenged. Measured here from coordinates: 77 of the 45,627 peptides carry an
    intra-peptide disulfide at ~2.03 A, several between the first and last residue --
    CP topology wearing an LP label. They are excluded by default (``--keep-crosslinked``
    readmits them) and the count is recorded either way. It matters more than 0.17%
    suggests: all 77 are PepBench, which carries 24% of each batch off 7.5% of the rows,
    so they would be drawn at roughly 3x their row share while training an *LP* generator.

SOURCE SAMPLING WEIGHTS ARE NOT INVENTED HERE. The project already configures them, in
``training_cpsea_lpmix_from_v4cfg.yaml``: ``pepbench: 0.06``, ``protfrag: 0.19`` inside a
25% linear budget, which renormalised over the LP-only pool is 24% / 76%. That split was
chosen ~sqrt-proportionally to pool size, between the natural ratio (which buries PepBench
at 7.5% of rows) and an even one (which would redraw its 3.4k rows ~91x an epoch). It is
the default here. ``--pepbench-frac`` overrides it; passing 0.8 gives the inverse ratio and
makes PepBench repeat ~150x an epoch against ProtFrag's ~2x, so it is a deliberate choice
and is recorded as one in the summary.

Weights are emitted as a per-row ``sample_weight`` over train rows, normalised to sum to 1,
so an arm's sampler can consume them directly and every arm draws the same mixture.

Outputs (``--out-dir``):

  ``lpgen_manifest.parquet``        every accepted row, with split/sequence/chains/weight
  ``lpgen_val_panel.parquet``       the ~32-complex validation panel (PepBench, from val)
  ``lpgen_manifest.summary.json``   counts, weights, and the leak/disjointness gates
  ``lpgen_excluded.csv``            every dropped row with its reason

Run (see ``scripts/lpgen_manifest.sbatch``):

    .venv/bin/python script_utils/lpgen_manifest.py \\
        --lp-train $ZFS/LPData/preprocessed/metadata/lp_train.parquet \\
        --lp-val   $ZFS/LPData/preprocessed/metadata/lp_val.parquet \\
        --out-dir  $ZFS/LPData/preprocessed/metadata_lpgen
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openfold.np.residue_constants import restype_3to1  # noqa: E402
from proteinfoundation.cp2lp.data import family_of, in_eval_split  # noqa: E402

#: ``cyclization_type`` values that count as a linear peptide. Anything else is a cyclic or
#: crosslinked peptide and is excluded from this pool -- and *recorded*, because "the filter
#: found nothing" and "the filter never ran" have to be distinguishable afterwards.
LINEAR_TYPES = frozenset({"linear"})

#: Backbone atoms every peptide residue must carry for an arm to be able to build a frame.
REQUIRED_BACKBONE = ("N", "CA", "C")

#: SG-SG separation at or below which two cysteines are bonded. A real disulfide is
#: ~2.05 A; 2.5 A is loose enough to catch a slightly strained crystal one and tight
#: enough that two merely adjacent cysteines do not register.
DISULFIDE_MAX_A = 2.5
#: Below this, the "bond" is shorter than any real S-S and the structure is anomalous.
#: Recorded under its own reason rather than quietly counted as a disulfide.
DISULFIDE_MIN_A = 1.8
#: N(first)-C(last) separation at or below which the backbone is closed head-to-tail.
#: A peptide bond is ~1.33 A.
MAINCHAIN_CLOSURE_MAX_A = 1.8

#: The split that is reserved for the later CP-sequence comparison.
HELDOUT = "heldout"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lp-train", required=True, help="lp_train.parquet from preprocess_lp.py")
    p.add_argument("--lp-val", required=True, help="lp_val.parquet from preprocess_lp.py")
    p.add_argument("--out-dir", required=True)

    # These two MUST match cp2lp_build_splits.py, or the reserved set stops being the
    # CP->LP eval set and the later comparison is contaminated.
    p.add_argument("--reserve-frac", type=float, default=0.15,
                   help="Family fraction reserved for the later CP->LP comparison. Must "
                        "match cp2lp_build_splits.py --eval-frac.")
    p.add_argument("--reserve-seed", type=int, default=0,
                   help="Must match cp2lp_build_splits.py --seed.")

    p.add_argument("--val-frac", type=float, default=0.10,
                   help="Family fraction of the REMAINING (non-reserved) pool used for this "
                        "milestone's validation.")
    p.add_argument("--val-seed", type=int, default=7,
                   help="Distinct from --reserve-seed so val is a fresh slice of the "
                        "trainable pool rather than a re-draw of the reserved one.")

    p.add_argument("--panel-size", type=int, default=32,
                   help="Validation panel size, spread over length bins and distinct families.")
    p.add_argument("--panel-sources", default="pepbench",
                   help="Sources eligible for the panel. PepBench by default: ProtFrag "
                        "fragments come from monomeric contexts, so their 'interface' is "
                        "intramolecular packing and they are not a binding check.")

    p.add_argument("--pepbench-frac", type=float, default=None,
                   help="PepBench share of each batch (ProtFrag gets the rest). Default "
                        "reuses the configured lp-mix weights renormalised to LP-only "
                        "(0.06/0.25 = 0.24). Pass 0.8 for the inverse, binder-focused ratio.")

    p.add_argument("--keep-crosslinked", action="store_true",
                   help="Keep peptides measured to be cyclic (intra-peptide disulfide or a "
                        "closed mainchain) in the LP pool. Off by default: they are CP "
                        "topology wearing an LP label, and the inherited metadata cannot "
                        "see them because the sources ship no CONECT records.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=-1,
                   help="Process only the first N rows (smoke runs). -1 = all.")
    return p.parse_args()


# --------------------------------------------------------------------- structure reading
def read_chains(pdb_path: str) -> dict:
    """Return per-chain ordered residue lists and the atom names seen for each residue.

    A hand-rolled column parse rather than a PDB library: these files were written by our
    own preprocessor, they are plain ATOM records, and the parse has to agree with how the
    arms' loaders read them (fixed columns), not with a library's error recovery.
    """
    chains: dict[str, list] = {}
    atoms: dict[tuple, set] = {}
    #: Atoms seen per residue. `atoms` is a SET, so it collapses duplicate atom names and
    #: cannot reveal a residue-key collision; the count can (n_atoms > len(name set)).
    n_atoms: dict[tuple, int] = {}
    coords: dict[tuple, tuple] = {}
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resname = line[17:20].strip()
            # Residue identity is (number, insertion code): a bare number collides across
            # insertion codes and would merge two residues into one.
            key = (chain, line[22:26].strip(), line[26])
            if key not in atoms:
                atoms[key] = set()
                n_atoms[key] = 0
                chains.setdefault(chain, []).append((key, resname))
            name = line[12:16].strip()
            atoms[key].add(name)
            n_atoms[key] += 1
            # Only the atoms the topology checks need, so a 3,000-atom receptor does not
            # cost 3,000 coordinate triples per complex across 45k files.
            if name in ("SG", "N", "C"):
                coords[(key, name)] = (
                    float(line[30:38]), float(line[38:46]), float(line[46:54]),
                )
    return {"chains": chains, "atoms": atoms, "n_atoms": n_atoms, "coords": coords}


def _dist(a: tuple, b: tuple) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def peptide_topology(residues: list, coords: dict) -> dict:
    """Measure whether the peptide is actually cyclic, from coordinates.

    The inherited metadata cannot answer this. ``preprocess_lp.py`` wrote every LP row
    ``cyclization_type="linear"`` with ``conect_pairs_kept=0`` because PepBench and
    ProtFrag ship no CONECT records at all -- so that label is an assumption inherited
    from the source, not a measurement, and a disulfide-stapled macrocycle passes it
    unchallenged. Measured on the real pool: 77 of 45,627 peptides carry an intra-peptide
    SG-SG bond at ~2.03 A, several of them between the first and last residue, which is
    CP topology wearing an LP label.
    """
    sg = [(k, coords[(k, "SG")]) for k, _ in residues if (k, "SG") in coords]
    bonds, anomalous = [], []
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            d = _dist(sg[i][1], sg[j][1])
            if d <= DISULFIDE_MAX_A:
                pair = (sg[i][0][1] + sg[i][0][2].strip(), sg[j][0][1] + sg[j][0][2].strip(),
                        round(d, 3))
                (anomalous if d < DISULFIDE_MIN_A else bonds).append(pair)

    # Head-to-tail: the N of the first residue bonded to the C of the last.
    closure = None
    if len(residues) >= 2:
        n_first = coords.get((residues[0][0], "N"))
        c_last = coords.get((residues[-1][0], "C"))
        if n_first is not None and c_last is not None:
            d = _dist(n_first, c_last)
            if d <= MAINCHAIN_CLOSURE_MAX_A:
                closure = round(d, 3)
    return {"disulfides": bonds, "anomalous_sg": anomalous, "mainchain_closure_A": closure}


def extract_one(task: tuple) -> dict:
    """Read one complex and return the fields the manifest needs, or a failure reason.

    Returns a dict always -- never raises and never drops a row silently. The plan's
    requirement is that adapter failures are *recorded*, so a row that cannot be read
    comes back with ``status != 'ok'`` and lands in lpgen_excluded.csv.
    """
    example_id, path, peptide_chain, peptide_length = task
    out = {
        "example_id": example_id,
        "status": "ok",
        "reason": "",
        "peptide_seq": "",
        "receptor_chain_ids": "",
        "n_peptide_res": 0,
        "n_noncanonical": 0,
        "n_missing_backbone": 0,
        "n_disulfide": 0,
        "disulfide_pairs": "",
        "mainchain_closure_A": float("nan"),
        "n_collided_res_peptide": 0,
        "n_collided_res_receptor": 0,
    }
    try:
        parsed = read_chains(path)
    except OSError as exc:
        out["status"] = "unreadable"
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out

    chains, atoms, coords = parsed["chains"], parsed["atoms"], parsed["coords"]
    n_atoms = parsed["n_atoms"]
    if peptide_chain not in chains:
        out["status"] = "no_peptide_chain"
        out["reason"] = f"chain {peptide_chain!r} absent; chains={sorted(chains)}"
        return out

    residues = chains[peptide_chain]
    out["n_peptide_res"] = len(residues)
    rec_keys_for_collision_check = [
        k for c, lst in chains.items() if c != peptide_chain for k, _ in lst
    ]
    receptor = sorted(c for c in chains if c != peptide_chain)
    out["receptor_chain_ids"] = ",".join(receptor)
    if not receptor:
        out["status"] = "no_receptor_chain"
        out["reason"] = f"only chain {peptide_chain!r} present"
        return out

    # RESIDUE-KEY COLLISIONS. Two genuinely different residues sharing one
    # (chain, resseq, icode) key. Measured on the full pool: 1,806 of 45,540 structures
    # (3.97%), every one of them ProtFrag, none PepBench -- consistent with
    # preprocess_lp.py merging a multi-chain receptor into a single chain `A` without
    # renumbering, so two source chains both carrying residue 82 land on one key.
    #
    # Two distinct harms, which is why this is a hard exclusion rather than a warning:
    #   * atomworks refuses the file outright ("Duplicate atom names detected in the same
    #     residue -- cannot infer struct_conn"), so RFD3 would silently drop ~4% of the
    #     training set through the dataloader's fallback retries;
    #   * 359 structures (0.79%) collide inside the PEPTIDE chain, and since this parser
    #     and preprocess_lp.py both key on the same tuple, the collided group counts as ONE
    #     residue -- so peptide_length and peptide_seq are understated and AGREE with each
    #     other, which is exactly why the length check below cannot catch it.
    collided = {"peptide": 0, "receptor": 0}
    for key in [k for k, _ in residues] + rec_keys_for_collision_check:
        if n_atoms[key] > len(atoms[key]):
            collided["peptide" if key[0] == peptide_chain else "receptor"] += 1
    out["n_collided_res_peptide"] = collided["peptide"]
    out["n_collided_res_receptor"] = collided["receptor"]

    seq, noncanon, missing = [], 0, 0
    for key, resname in residues:
        one = restype_3to1.get(resname)
        if one is None:
            noncanon += 1
            # 'X' keeps the sequence length equal to the residue count, so a length
            # mismatch below means a real count disagreement and not a dropped letter.
            one = "X"
        if not set(REQUIRED_BACKBONE).issubset(atoms[key]):
            missing += 1
        seq.append(one)

    out["peptide_seq"] = "".join(seq)
    out["n_noncanonical"] = noncanon
    out["n_missing_backbone"] = missing

    topo = peptide_topology(residues, coords)
    out["n_disulfide"] = len(topo["disulfides"])
    out["disulfide_pairs"] = ";".join(f"{a}-{b}@{d}" for a, b, d in topo["disulfides"])
    if topo["mainchain_closure_A"] is not None:
        out["mainchain_closure_A"] = topo["mainchain_closure_A"]

    if collided["peptide"] or collided["receptor"]:
        out["status"] = "residue_key_collision"
        out["reason"] = (
            f"{collided['peptide']} peptide + {collided['receptor']} receptor residue key(s) "
            f"hold more than one residue"
        )
    elif noncanon:
        out["status"] = "noncanonical_residue"
        out["reason"] = f"{noncanon} non-canonical residue(s) in peptide"
    elif missing:
        out["status"] = "missing_backbone"
        out["reason"] = f"{missing} peptide residue(s) lack one of {REQUIRED_BACKBONE}"
    elif len(residues) != int(peptide_length):
        # The metadata's length and the structure disagree. Training on this row would
        # silently pad or truncate, so it is an error rather than a warning.
        out["status"] = "length_mismatch"
        out["reason"] = f"metadata peptide_length={peptide_length}, chain has {len(residues)}"
    elif topo["anomalous_sg"]:
        # Shorter than any real S-S. Not called a disulfide and not trained on either.
        out["status"] = "anomalous_sg_contact"
        out["reason"] = f"SG-SG below {DISULFIDE_MIN_A} A: {topo['anomalous_sg']}"
    elif topo["disulfides"]:
        out["status"] = "crosslinked_disulfide"
        out["reason"] = f"{len(topo['disulfides'])} intra-peptide disulfide(s): {out['disulfide_pairs']}"
    elif topo["mainchain_closure_A"] is not None:
        out["status"] = "mainchain_cyclic"
        out["reason"] = f"N(first)-C(last) = {topo['mainchain_closure_A']} A"
    return out


# ------------------------------------------------------------------------------- splitting
def assign_split(family: str, reserve_frac: float, reserve_seed: int,
                 val_frac: float, val_seed: int) -> str:
    """Family -> one of ``heldout`` / ``val`` / ``train``.

    A pure function of the family key, which is what makes the parent/fragment leak
    unrepresentable: every row sharing a ``cluster_id`` gets the same answer, in every
    process and on every rerun.

    ``heldout`` is tested first and wins, so widening ``--val-frac`` can never eat into
    the families reserved for the CP->LP comparison.
    """
    if in_eval_split(family, reserve_frac, reserve_seed):
        return HELDOUT
    if in_eval_split(family, val_frac, val_seed):
        return "val"
    return "train"


def build_panel(val_df: pd.DataFrame, size: int, sources: list) -> pd.DataFrame:
    """Pick ~``size`` validation complexes spread over length and distinct families.

    One per family, walking length bins round-robin, so the panel cannot be eight copies
    of the same receptor at the same length. Deterministic: sorted, no sampling.
    """
    pool = val_df[val_df.dataset_source.isin(sources)].copy()
    if pool.empty:
        return pool
    pool = pool.sort_values(["peptide_length", "cluster_id", "example_id"], kind="mergesort")

    by_len: dict[int, list] = {}
    for _, row in pool.iterrows():
        by_len.setdefault(int(row.peptide_length), []).append(row)

    picked, seen_families = [], set()
    lengths = sorted(by_len)
    # Round-robin over length bins: fills the length axis first and only then takes a
    # second example from a bin, which is what "spanning lengths" has to mean.
    while len(picked) < size:
        progressed = False
        for L in lengths:
            if len(picked) >= size:
                break
            while by_len[L]:
                row = by_len[L].pop(0)
                if row.cluster_id in seen_families:
                    continue
                picked.append(row)
                seen_families.add(row.cluster_id)
                progressed = True
                break
        if not progressed:
            break
    return pd.DataFrame(picked)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.lp_train)
    val = pd.read_parquet(args.lp_val)
    df = pd.concat([train, val], ignore_index=True)
    logger.info(f"read {len(train):,} train + {len(val):,} val = {len(df):,} rows")

    if df.example_id.duplicated().any():
        dupes = df.example_id[df.example_id.duplicated()].unique()[:5]
        raise SystemExit(f"FATAL: duplicate example_id across the two inputs, e.g. {list(dupes)}")

    df["inherited_split"] = df["split"]
    if args.limit > 0:
        # Stratified by source, NOT a plain head: example_ids sort with every `pepbench__*`
        # before every `protfrag__*`, so an alphabetical head of 2,000 contains zero
        # ProtFrag rows and the smoke run then exercises neither the mixture weights nor
        # the parent/fragment families where the inherited leak actually lives.
        df = df.sort_values("example_id", kind="mergesort")
        shares = df.dataset_source.value_counts(normalize=True)
        keep_ids = []
        for src, share in shares.items():
            n_src = max(1, int(round(args.limit * share)))
            keep_ids.extend(df[df.dataset_source == src].example_id.head(n_src))
        df = df[df.example_id.isin(set(keep_ids))].head(args.limit).copy()
        logger.warning(
            f"--limit {args.limit}: SMOKE RUN, not a complete manifest "
            f"({dict(df.dataset_source.value_counts())})"
        )

    excluded = []

    # ---- 1. LP-only. Nothing is expected to go here (preprocess_lp wrote every row
    # ---- linear with zero CONECT pairs) and the count is recorded so that stays checkable.
    cyc = ~df.cyclization_type.isin(LINEAR_TYPES)
    conect = df.conect_pairs_kept.astype(int) > 0
    nonlinear = cyc | conect
    for _, row in df[nonlinear].iterrows():
        excluded.append({
            "example_id": row.example_id,
            "dataset_source": row.dataset_source,
            "stage": "linear_filter",
            "reason": (f"cyclization_type={row.cyclization_type!r}"
                       if row.cyclization_type not in LINEAR_TYPES
                       else f"conect_pairs_kept={row.conect_pairs_kept}"),
        })
    df = df[~nonlinear].copy()
    logger.info(f"linear filter: kept {len(df):,}, excluded {int(nonlinear.sum()):,}")

    # ---- 2. Sequences and chain roles, read back out of the structures.
    tasks = list(zip(df.example_id, df.path, df.binder_chain_id, df.peptide_length))
    logger.info(f"reading {len(tasks):,} structures with {args.workers} workers")
    with Pool(args.workers) as pool:
        records = pool.map(extract_one, tasks, chunksize=256)
    ext = pd.DataFrame.from_records(records).set_index("example_id")
    logger.info(f"structure read status: {dict(Counter(ext.status))}")

    df = df.join(ext, on="example_id")

    # The measured-cyclic statuses are a POOL decision, not a read failure: the structure
    # is fine, it is just not a linear peptide. --keep-crosslinked readmits them, and the
    # summary records which way it went either way.
    CYCLIC_STATUSES = ("crosslinked_disulfide", "mainchain_cyclic")
    measured_cyclic = df.status.isin(CYCLIC_STATUSES)
    logger.info(
        f"measured peptide topology: {int(measured_cyclic.sum()):,} cyclic "
        f"({dict(Counter(df.status[measured_cyclic]))}), "
        f"{int((df.n_disulfide > 0).sum()):,} with an intra-peptide disulfide"
    )
    if args.keep_crosslinked and measured_cyclic.any():
        logger.warning(
            f"--keep-crosslinked: readmitting {int(measured_cyclic.sum()):,} MEASURED-CYCLIC "
            "peptides into the LP pool"
        )
        df.loc[measured_cyclic, "status"] = "ok"

    bad = df.status != "ok"
    for _, row in df[bad].iterrows():
        excluded.append({
            "example_id": row.example_id,
            "dataset_source": row.dataset_source,
            "stage": ("topology" if row.status in CYCLIC_STATUSES
                      else "residue_collision" if row.status == "residue_key_collision"
                      else "structure_read"),
            "reason": f"{row.status}: {row.reason}",
        })
    df = df[~bad].copy()
    logger.info(f"structure filter: kept {len(df):,}, excluded {int(bad.sum()):,}")
    if df.empty:
        raise SystemExit("FATAL: every row failed the structure read -- check --lp-train paths")

    # ---- 2b. AtomWorks pn_unit_iid columns.
    #
    # RFD3's binder condition gates on `query_pn_unit_iids`, which GenericDFParser fills
    # from named manifest columns -- and its docstring is explicit that those columns hold
    # "the CIFUtils pn_unit_iid(s)". Those are NOT bare chain ids: measured on 237 loaded
    # examples, atomworks reports `A_1` / `B_1` (chain plus transformation index, which is
    # `1` for every one of our single-assembly PDBs). Handing it `A` / `B` instead composes
    # and loads fine, then fails deep in the spatial crop with "No crop center found!",
    # because `is_query_pn_unit` matches nothing.
    PN_UNIT_SUFFIX = "_1"
    multi = df.receptor_chain_ids.astype(str).str.contains(",")
    if multi.any():
        # Would need one pn_unit per receptor chain; our pool is uniformly single-chain A.
        logger.warning(
            f"{int(multi.sum())} row(s) have multiple receptor chains; their "
            f"receptor_pn_unit_iid is only the first chain"
        )
    df["receptor_pn_unit_iid"] = (
        df.receptor_chain_ids.astype(str).str.split(",").str[0] + PN_UNIT_SUFFIX
    )
    df["peptide_pn_unit_iid"] = df.binder_chain_id.astype(str) + PN_UNIT_SUFFIX

    # ---- 2c. An example id in AtomWorks' own format, for RFD3 only.
    #
    # RFD3's logging callbacks call `atomworks.ml.example_id.parse_example_id`, which
    # demands exactly four brace-delimited fields --
    #   {[dataset_names]}{pdb_id}{assembly_id}{[query_pn_unit_iids]}
    # (see `generate_example_id` in that module) -- and raises
    # `Invalid example ID format ... 0 matches found (expected 4)` on anything else. Our
    # own ids are not in that shape, and the failure lands in `on_train_batch_start`, i.e.
    # after the model and the data path are already working.
    #
    # A SECOND column rather than a rename: `example_id` is this project's primary key, used
    # by the splits, the native export's directory names and every join. Here the dataset
    # name carries the source, which is what makes RFD3's dataset-sampling-ratio logging
    # report the PepBench/ProtFrag mixture, and the `pdb_id` slot carries our full unique id
    # (the upstream docstring allows "any group-level identifier") because `cluster_id`
    # alone repeats across rows and the id must stay unique.
    #
    # Both list fields are passed through `eval()` upstream, so they must be valid Python
    # list literals -- hence repr(), not manual string building.
    df["rfd3_example_id"] = [
        "{%s}{%s}{%s}{%s}"
        % (repr(["lpgen", str(src)]), eid, "1", repr([str(rpn), str(ppn)]))
        for src, eid, rpn, ppn in zip(
            df.dataset_source, df.example_id,
            df.receptor_pn_unit_iid, df.peptide_pn_unit_iid,
        )
    ]

    # ---- 3. Family-level split.
    df["family_id"] = [family_of(r) for r in df.to_dict("records")]
    df["split"] = [
        assign_split(f, args.reserve_frac, args.reserve_seed, args.val_frac, args.val_seed)
        for f in df.family_id
    ]

    # The gate the whole file exists for. Disjointness is true by construction, so a
    # failure here means assign_split stopped being a pure function of the family.
    fams = {s: set(g.family_id) for s, g in df.groupby("split")}
    for a, b in (("train", "val"), ("train", HELDOUT), ("val", HELDOUT)):
        shared = fams.get(a, set()) & fams.get(b, set())
        if shared:
            raise SystemExit(f"FATAL: {len(shared)} families in both {a} and {b}: {sorted(shared)[:5]}")

    # Did reassignment actually repair the inherited leak? Reported, not assumed.
    inherited_val = set(df[df.inherited_split == "val"].family_id)
    inherited_train = set(df[df.inherited_split == "train"].family_id)
    repaired = sorted(inherited_val & inherited_train)

    # ---- 4. Per-row sampling weight over train rows.
    pep_frac = args.pepbench_frac
    weights_source = "cli --pepbench-frac"
    if pep_frac is None:
        # Reuse the configured lp-mix weights, renormalised over the LP-only pool.
        pep_frac = 0.06 / 0.25
        weights_source = "training_cpsea_lpmix_from_v4cfg.yaml source_fractions, renormalised"
    target = {"pepbench": pep_frac, "protfrag": 1.0 - pep_frac}

    df["sample_weight"] = 0.0
    tr = df.split == "train"
    counts = df[tr].dataset_source.value_counts().to_dict()
    for src, frac in target.items():
        n = counts.get(src, 0)
        if n == 0:
            if frac > 0:
                logger.warning(f"source {src!r} has {frac:.0%} of the batch but 0 train rows")
            continue
        # Even within a source, so the per-row weight is the source share spread over its
        # rows. An unnamed source keeps weight 0 and is sampled never -- loudly, above.
        df.loc[tr & (df.dataset_source == src), "sample_weight"] = frac / n
    unnamed = set(df[tr].dataset_source) - set(target)
    if unnamed:
        logger.warning(f"train sources with NO configured fraction, weight 0: {sorted(unnamed)}")

    # ---- 5. Write.
    keep = [
        "example_id", "family_id", "dataset_source", "split", "inherited_split",
        "path", "receptor_chain_ids", "binder_chain_id", "peptide_seq",
        "receptor_pn_unit_iid", "peptide_pn_unit_iid", "rfd3_example_id",
        "peptide_length", "receptor_length", "cyclization_type", "sample_weight",
        "n_disulfide", "mainchain_closure_A", "cluster_id", "source_path",
    ]
    manifest = df[keep].sort_values("example_id", kind="mergesort").reset_index(drop=True)
    manifest = manifest.rename(columns={"binder_chain_id": "peptide_chain_id"})
    man_path = out_dir / "lpgen_manifest.parquet"
    manifest.to_parquet(man_path, index=False)

    panel = build_panel(
        df[df.split == "val"], args.panel_size,
        [s.strip() for s in args.panel_sources.split(",") if s.strip()],
    )
    panel_path = out_dir / "lpgen_val_panel.parquet"
    if not panel.empty:
        panel = panel[keep].rename(columns={"binder_chain_id": "peptide_chain_id"})
        panel = panel.sort_values("example_id", kind="mergesort").reset_index(drop=True)
        panel.to_parquet(panel_path, index=False)
    else:
        logger.warning("validation panel is EMPTY -- no val rows from --panel-sources")

    exc_path = out_dir / "lpgen_excluded.csv"
    pd.DataFrame(excluded, columns=["example_id", "dataset_source", "stage", "reason"]).to_csv(
        exc_path, index=False
    )

    by_split = {
        s: {
            "rows": int(len(g)),
            "families": int(g.family_id.nunique()),
            "by_source": g.dataset_source.value_counts().to_dict(),
            "peptide_length": {
                "min": int(g.peptide_length.min()), "max": int(g.peptide_length.max()),
                "median": float(g.peptide_length.median()),
            },
        }
        for s, g in manifest.groupby("split")
    }
    # What the sampler will actually draw, which is the number to report -- not the
    # row counts, which are the natural ratio and differ by ~an order of magnitude.
    tr_man = manifest[manifest.split == "train"]
    realised = (tr_man.groupby("dataset_source").sample_weight.sum() / tr_man.sample_weight.sum()
                ).to_dict() if tr_man.sample_weight.sum() > 0 else {}

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"lp_train": args.lp_train, "lp_val": args.lp_val},
        "outputs": {
            "manifest": str(man_path),
            "val_panel": str(panel_path) if not panel.empty else None,
            "excluded": str(exc_path),
        },
        "split_params": {
            "reserve_frac": args.reserve_frac, "reserve_seed": args.reserve_seed,
            "val_frac": args.val_frac, "val_seed": args.val_seed,
            "note": "reserve_* MUST match cp2lp_build_splits.py so `heldout` IS the CP->LP eval set",
        },
        "rows_in": int(len(train) + len(val)),
        "rows_accepted": int(len(manifest)),
        "rows_excluded": len(excluded),
        "excluded_by_stage": dict(Counter(e["stage"] for e in excluded)),
        "residue_key_collisions": {
            "n_excluded": sum(1 for e in excluded if e["stage"] == "residue_collision"),
            "why": "two different residues sharing one (chain, resseq, icode) key: atomworks "
                   "refuses the file, and this parser (like preprocess_lp.py) would count the "
                   "group as ONE residue, understating peptide_length and peptide_seq in "
                   "mutual agreement so the length check cannot see it",
            "likely_cause": "preprocess_lp.py merges a multi-chain receptor into chain A "
                            "without renumbering residues",
        },
        "topology": {
            "measured_from_coordinates": True,
            "why": "preprocess_lp.py wrote every LP row cyclization_type='linear' with "
                   "conect_pairs_kept=0 because PepBench and ProtFrag ship no CONECT "
                   "records; that label is inherited from the source, not measured, so a "
                   "disulfide-stapled macrocycle passes it unchallenged.",
            "disulfide_window_A": [DISULFIDE_MIN_A, DISULFIDE_MAX_A],
            "mainchain_closure_max_A": MAINCHAIN_CLOSURE_MAX_A,
            "n_excluded_cyclic": sum(1 for e in excluded if e["stage"] == "topology"),
            "kept_crosslinked": bool(args.keep_crosslinked),
            "accepted_with_disulfide": int((manifest.n_disulfide > 0).sum()),
        },
        "excluded_by_reason": dict(Counter(e["reason"].split(":")[0] for e in excluded)),
        "by_split": by_split,
        "families_total": int(manifest.family_id.nunique()),
        "family_disjointness_checked": True,
        "inherited_split_leak_repaired": {
            "n_families": len(repaired),
            "families": repaired,
            "note": "families the INHERITED PepBench/ProtFrag split put on both sides; the "
                    "family-level reassignment puts each on exactly one.",
        },
        "source_fractions": {
            "requested": target,
            "source": weights_source,
            "realised_by_weight": realised,
        },
        "val_panel": {
            "size": int(len(panel)),
            "requested": args.panel_size,
            "families": int(panel.family_id.nunique()) if not panel.empty else 0,
            "lengths": sorted(panel.peptide_length.unique().tolist()) if not panel.empty else [],
        },
        "peptide_length_range": [int(manifest.peptide_length.min()),
                                 int(manifest.peptide_length.max())],
        "smoke_limit": args.limit if args.limit > 0 else None,
    }
    sum_path = out_dir / "lpgen_manifest.summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))

    logger.info(f"wrote {man_path} ({len(manifest):,} rows)")
    for s, v in sorted(by_split.items()):
        logger.info(f"  {s:8s} {v['rows']:>7,} rows  {v['families']:>6,} families  {v['by_source']}")
    logger.info(f"realised batch mixture: { {k: round(v, 4) for k, v in realised.items()} }")
    logger.info(f"val panel: {len(panel)} complexes -> {panel_path}")
    logger.info(f"inherited-split leak repaired for {len(repaired)} families: {repaired}")
    logger.info(f"summary -> {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
