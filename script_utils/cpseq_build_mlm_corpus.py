"""Extract the (pocket sequence, cyclic-peptide sequence, chemistry) corpus for the
masked-LM sequence proposer.

WHY A CORPUS FILE AND NOT A DATASET CLASS. The flow stack reads CPSea by parsing a PDB
per example inside the dataloader, which is affordable there because a step costs a
forward through a 160M trunk plus an autoencoder. A sequence model's step is three orders
of magnitude cheaper, so PDB parsing would become the whole cost of training. Everything
the proposer needs is a few hundred bytes per example, so it is extracted once, on CPU,
and training reads JSONL.

WHAT A ROW CARRIES, and why each field is here rather than derived later:

  pep_seq       The binder sequence, one letter per residue, in chain order. The thing
                being modelled.
  pocket_seq    Receptor residues with any heavy atom within `--pocket-radius` of any
                binder heavy atom, in CHAIN order, with `GAP_CHAR` between residues that
                are not consecutive in the receptor numbering. The gap markers are not
                decoration: project memory measured a contiguous full receptor against a
                fragmented pocket crop at +22 to +34 points of ring closure, so a model
                that cannot tell a real neighbour from a spliced one is being handed a
                receptor that does not exist.
  cyc_type      Resolved by `infer_cyclization_label`, NOT by the metadata string. The
                metadata column cannot separate disulfide from isopeptide and lumps every
                isopeptide into "other" (see `cyclization/parse_labels.py`), which is 57%
                of CPSea. Training a chemistry-conditioned model on that column would
                teach it that the majority chemistry is "unknown".
  anchor_i/j    The cyclization bond's binder-local endpoints, from the same parser. The
                mask schedule needs them: an anchor is the position whose identity the
                chemistry constrains, so "mask the anchors" is a distinct query from
                "mask two residues somewhere", and project memory records isopeptide
                failure as ~50% anchor IDENTITY rather than geometry.
  cluster_id    Foldseek cluster, carried through so a split can be checked and a holdout
                can be applied by cluster rather than by example.

ONE OUTPUT FILE PER SHARD. Concurrent `O_APPEND` writes to one file on this filesystem
were measured to NUL-corrupt rows (88 of 144 lost), so a shard never shares its output.
The aggregate is whatever `cpseq_mlm_data.py` globs, and it refuses a corrupt line rather
than skipping it.

RESUMABLE because the array task will be preempted: a shard whose `.done` marker exists
is skipped, and the marker is written only after the JSONL is closed and re-read.

Usage (one shard):
    python script_utils/cpseq_build_mlm_corpus.py \
        --metadata /path/cpsea_train.parquet --split train \
        --out-dir $SCRATCH/cpseq_mlm_corpus --shard 0 --num-shards 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# The repo is an editable install, but a batch script runs from the spool directory, so
# `src` is added explicitly rather than relying on the cwd.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from proteinfoundation.cyclization.constants import (  # noqa: E402
    COND_TYPE_TO_NAME,
    NO_CYCLIZATION_INDEX,
    UNSPECIFIED,
)
from proteinfoundation.cyclization.parse_labels import infer_cyclization_label  # noqa: E402

# Separates non-consecutive runs of pocket residues. Not a residue, and deliberately not
# one of ESM-2's own `-`/`.` tokens: the tokenizer maps those to learned embeddings whose
# pretraining meaning is "alignment gap", which is a different claim from "the receptor
# continues but those residues are far from the binder".
GAP_CHAR = "/"

# Backbone + side-chain heavy atoms only. Hydrogens are already stripped by
# `preprocess_cpsea.py`, but a stray one would shrink every distance and silently widen
# the pocket, so the filter is explicit.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_pdb_residues(pdb_path: str) -> dict[str, list[dict]]:
    """Single pass over a preprocessed CPSea PDB.

    Returns `{chain_id: [residue, ...]}` in file order, each residue
    `{"resseq", "resname", "coords": [(x, y, z), ...]}`. Only the fields the pocket
    cutoff and the sequence need, so this stays a few hundred bytes per residue instead
    of materialising an AtomArray.

    A residue with no recognised three-letter code is kept with an empty `resname` so the
    caller can reject the whole example rather than silently emit a shorter sequence than
    the structure has -- a sequence one residue short of its own structure would misplace
    every anchor index downstream.
    """
    chains: dict[str, list[dict]] = {}
    current: dict[tuple[str, int], dict] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain_id = line[21]
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            if atom_name.startswith("H") or atom_name in ("OXT",):
                continue
            key = (chain_id, resseq)
            res = current.get(key)
            if res is None:
                res = {"resseq": resseq, "resname": resname, "coords": []}
                current[key] = res
                chains.setdefault(chain_id, []).append(res)
            res["coords"].append((x, y, z))
    return chains


def _residue_sequence(residues: list[dict]) -> str | None:
    """One-letter sequence, or None if any residue is a non-canonical code."""
    out = []
    for r in residues:
        one = _THREE_TO_ONE.get(r["resname"])
        if one is None:
            return None
        out.append(one)
    return "".join(out)


def extract_pocket(
    receptor: list[dict], binder: list[dict], radius: float
) -> tuple[str, list[int]]:
    """Pocket sequence (chain order, `GAP_CHAR` between non-consecutive runs) and the
    receptor `resseq` of each kept residue.

    The cutoff is heavy-atom to heavy-atom, matching `CroppingTransform2`'s
    `interface_threshold` of 8.0 A rather than inventing a second definition of contact.

    VECTORISED, and not as a micro-optimisation: the pure-Python triple loop this
    replaces was measured at 0.41 s per example, which is 278 core-hours over CPSea's
    2.44M rows and would have made the corpus build cost more than training on it. A
    receptor is ~750 atoms and a binder ~50, so the full pairwise block is ~37k distances
    -- small enough to compute densely and reduce per residue.
    """
    import numpy as np

    binder_xyz = np.asarray(
        [c for res in binder for c in res["coords"]], dtype=np.float32
    )
    rec_xyz = np.asarray(
        [c for res in receptor for c in res["coords"]], dtype=np.float32
    )
    if binder_xyz.size == 0 or rec_xyz.size == 0:
        return "", []
    # Which residue each receptor atom belongs to, so a per-atom hit reduces to a
    # per-residue hit without a second pass over the structure.
    owner = np.repeat(
        np.arange(len(receptor), dtype=np.int32),
        [len(res["coords"]) for res in receptor],
    )
    d2 = ((rec_xyz[:, None, :] - binder_xyz[None, :, :]) ** 2).sum(-1)
    atom_hit = (d2 <= radius * radius).any(axis=1)
    res_hit = np.zeros(len(receptor), dtype=bool)
    np.logical_or.at(res_hit, owner, atom_hit)
    kept: list[dict] = [res for res, h in zip(receptor, res_hit) if h]

    pieces: list[str] = []
    resids: list[int] = []
    prev_resseq: int | None = None
    for res in kept:
        one = _THREE_TO_ONE.get(res["resname"])
        if one is None:
            continue
        if prev_resseq is not None and res["resseq"] != prev_resseq + 1:
            pieces.append(GAP_CHAR)
        pieces.append(one)
        resids.append(res["resseq"])
        prev_resseq = res["resseq"]
    return "".join(pieces), resids


def build_row(
    rec: dict, pocket_radius: float, binder_chain_id: str
) -> tuple[dict | None, str]:
    """One corpus row, or `(None, reason)`. The reason is counted, never discarded: a
    rejection rate that moves between runs is the first sign the data root changed.
    """
    pdb_path = rec["path"]
    if not os.path.exists(pdb_path):
        return None, "pdb_missing"
    try:
        chains = parse_pdb_residues(pdb_path)
    except Exception as e:  # noqa: BLE001 - one bad file must not kill a shard
        return None, f"parse_error:{type(e).__name__}"

    binder = chains.get(binder_chain_id)
    if not binder:
        return None, "no_binder_chain"
    receptor = [r for cid, rs in chains.items() if cid != binder_chain_id for r in rs]
    if not receptor:
        return None, "no_receptor"

    pep_seq = _residue_sequence(binder)
    if pep_seq is None:
        return None, "noncanonical_binder_residue"
    if len(pep_seq) != int(rec["peptide_length"]):
        # The metadata length and the structure disagree, so one of them is describing a
        # different molecule. Anchor indices are binder-local, so guessing which to trust
        # would silently shift them.
        return None, "peptide_length_mismatch"

    pocket_seq, pocket_resids = extract_pocket(receptor, binder, pocket_radius)
    if not pocket_resids:
        return None, "empty_pocket"

    label = infer_cyclization_label(
        pdb_path=pdb_path,
        residue_pdb_idx=[r["resseq"] for r in binder],
        binder_length_hint=int(rec["peptide_length"]),
        binder_chain_id=binder_chain_id,
        cyclization_type_hint=rec.get("cyclization_type"),
    )
    cyc_type = int(label["type"]) if label["has_cyclization"] else UNSPECIFIED
    if cyc_type == NO_CYCLIZATION_INDEX:
        cyc_type = UNSPECIFIED

    return {
        "example_id": rec["example_id"],
        "cluster_id": rec.get("cluster_id"),
        "split": rec.get("split"),
        "pep_seq": pep_seq,
        "pocket_seq": pocket_seq,
        "n_pocket_res": len(pocket_resids),
        "pocket_resids": pocket_resids,
        "cyc_type": cyc_type,
        "cyc_type_name": COND_TYPE_TO_NAME.get(cyc_type, "unspecified"),
        "cyc_anchor_i": int(label["i"]) if label["has_cyclization"] else -1,
        "cyc_anchor_j": int(label["j"]) if label["has_cyclization"] else -1,
        "cyc_reason": label["reason"],
        "receptor_length": int(rec.get("receptor_length") or 0),
    }, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True, help="CPSea split parquet")
    ap.add_argument("--split", required=True, help="recorded on every row, for provenance")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--pocket-radius", type=float, default=8.0)
    ap.add_argument("--binder-chain-id", default="B")
    ap.add_argument("--limit", type=int, default=0, help="first N rows of the shard; smoke only")
    ap.add_argument("--force", action="store_true", help="rebuild even if .done exists")
    ap.add_argument(
        "--workers", type=int, default=8,
        help="threads reading structures concurrently. This job is I/O BOUND, not CPU "
             "bound: measured cold, 42.4 s for 200 rows spent only 7.5 s in userspace, "
             "so 85%% was NFS wait on a directory holding 2.4M files. Threads (not "
             "processes) because file reads and the numpy distance block both release the "
             "GIL, so the waits overlap without paying fork or pickling costs. 1 disables.",
    )
    args = ap.parse_args()

    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_shard{args.shard:04d}of{args.num_shards:04d}"
    out_path = out_dir / f"{stem}.jsonl"
    done_path = out_dir / f"{stem}.done"
    if done_path.exists() and not args.force:
        print(f"[skip] {done_path} exists")
        return 0

    cols = [
        "example_id", "path", "binder_chain_id", "cluster_id", "split",
        "peptide_length", "receptor_length", "cyclization_type",
    ]
    df = pd.read_parquet(args.metadata, columns=cols)
    n_total = len(df)
    # Contiguous blocks, not strided: rows of one cluster sit together in the parquet, so
    # a block keeps a cluster inside one shard and makes a per-shard count readable.
    per = math.ceil(n_total / args.num_shards)
    lo, hi = args.shard * per, min((args.shard + 1) * per, n_total)
    block = df.iloc[lo:hi]
    if args.limit:
        block = block.iloc[: args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] rows {lo}..{hi} of {n_total} ({len(block)} to do)")

    reasons: dict[str, int] = {}
    cyc_counts: dict[str, int] = {}
    n_ok = 0
    t0 = time.time()
    records = block.to_dict("records")

    def _one(rec):
        chain = rec.get("binder_chain_id") or args.binder_chain_id
        return build_row(rec, args.pocket_radius, chain)

    tmp_path = out_path.with_suffix(".jsonl.partial")
    with open(tmp_path, "w") as out:
        # `map` preserves input order, so the shard's output is deterministic regardless
        # of which thread finished first -- a corpus whose row order changed between runs
        # would make a resumed or re-run shard non-reproducible.
        if args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=args.workers)
            results = pool.map(_one, records)
        else:
            pool = None
            results = map(_one, records)
        try:
            for n_seen, (row, reason) in enumerate(results, start=1):
                reasons[reason] = reasons.get(reason, 0) + 1
                if row is not None:
                    row["split"] = args.split
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    n_ok += 1
                    cyc_counts[row["cyc_type_name"]] = (
                        cyc_counts.get(row["cyc_type_name"], 0) + 1
                    )
                if n_seen % 2000 == 0:
                    # Flushed: a block-buffered progress line makes a slow job on a busy
                    # node indistinguishable from a hung one, which cost real time here.
                    rate = n_seen / max(time.time() - t0, 1e-9)
                    print(f"  {n_seen}/{len(records)} ok={n_ok} {rate:.1f} rows/s",
                          flush=True)
                    out.flush()
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
    os.replace(tmp_path, out_path)

    # Re-read before claiming the shard is done, so a truncated write is caught here
    # rather than by the trainer three jobs later.
    n_lines = 0
    with open(out_path) as f:
        for line in f:
            json.loads(line)
            n_lines += 1
    if n_lines != n_ok:
        print(f"[FAIL] wrote {n_ok} rows but re-read {n_lines}", file=sys.stderr)
        return 1

    stats = {
        "shard": args.shard, "num_shards": args.num_shards, "split": args.split,
        "metadata": args.metadata, "rows_in_block": len(block), "rows_written": n_ok,
        "pocket_radius": args.pocket_radius, "reasons": reasons,
        "cyc_type_counts": cyc_counts, "seconds": round(time.time() - t0, 1),
    }
    (out_dir / f"{stem}.stats.json").write_text(json.dumps(stats, indent=2))
    done_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
