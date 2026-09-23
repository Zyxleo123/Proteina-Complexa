"""Deliverable E on CPSea_full -- within-domain competitor pairs, with the spatial check.

Two segments of the same domain make a competitor pair only if they bind DIFFERENT places
on the receptor.  Sequence disjointness is necessary but not sufficient: two segments can
be disjoint in sequence and still occupy the same surface, in which case they are
near-duplicates and the arm they support does not remove the "return this pocket's
canonical answer" shortcut it exists to remove.

The Milestone-1 figure of 2,395 multi-segment domains was measured on CPSea_PDB and does
not transfer, because the build set is CPSea_full.  This recomputes it there and adds the
spatial gate.

  index    exact counts over the whole 2.44M-row index (parquet only, no structures), plus
           a capped uniform sample of candidate pairs for the spatial stage.
  spatial  per-shard: load both segments, compare receptor contact sets and interface
           positions.

Frames: two segments of one domain live in separate PDBs with separately cropped
receptors, so nothing guarantees a shared coordinate frame.  Contact-set Jaccard is taken
over receptor resSeq and is frame-free.  Centroid separation needs a frame, so the two
files are superposed on the CA atoms of their SHARED receptor residues first; the fit RMSD
is reported so a bad superposition cannot pass silently as a small separation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_lnr_metadata import parse_pdb_chains  # noqa: E402
from script_utils import m15_config  # noqa: E402

# `{PDB}_{chain}_{start}_{end}` with one or more `_relaxed` suffixes.  CPSea_PDB carries
# one, CPSea_full two -- matching only the single form would drop every CPSea_full row.
ID_RE = re.compile(r"^(?P<domain>.+?)_(?P<start>\d+)_(?P<end>\d+)(?:_relaxed)+$")


def parse_example_id(example_id: str) -> tuple[str, int, int] | None:
    m = ID_RE.match(example_id)
    if not m:
        return None
    return m.group("domain"), int(m.group("start")), int(m.group("end"))


# ------------------------------------------------------------------------------ index
def index(args) -> None:
    cfg = m15_config.load(args.config)["deliverable_e"]
    meta = m15_config.resolve(cfg["metadata"])
    min_sep = int(cfg["min_sequence_separation"])

    df = pq.read_table(meta, columns=["example_id", "path", "cluster_id",
                                      "peptide_length", "cyclization_type"]).to_pandas()
    parsed = [parse_example_id(e) for e in df["example_id"]]
    ok = np.array([p is not None for p in parsed])
    df = df[ok].reset_index(drop=True)
    df["domain"] = [p[0] for p, k in zip(parsed, ok) if k]
    df["start"] = [p[1] for p, k in zip(parsed, ok) if k]
    df["end"] = [p[2] for p, k in zip(parsed, ok) if k]

    counts = df.groupby("domain").size()
    multi = counts[counts >= 2]

    rng = np.random.default_rng(int(cfg["seed"]))
    pair_rows = []
    n_pairs = n_disjoint = n_disjoint_sep = 0
    for domain, grp in df[df["domain"].isin(multi.index)].groupby("domain", sort=False):
        segs = grp.sort_values("start").to_dict("records")
        for x in range(len(segs)):
            for y in range(x + 1, len(segs)):
                s1, s2 = segs[x], segs[y]
                n_pairs += 1
                # Half-open [start, end): overlapping windows of one site (1A0P_A_182_196
                # against 1A0P_A_183_196) are near-duplicates, not competitors.
                disjoint = s1["end"] < s2["start"] or s2["end"] < s1["start"]
                if not disjoint:
                    continue
                n_disjoint += 1
                gap = (s2["start"] - s1["end"]) if s1["end"] < s2["start"] else (s1["start"] - s2["end"])
                if gap < min_sep:
                    continue
                n_disjoint_sep += 1
                pair_rows.append({
                    "domain": domain,
                    "example_id_a": s1["example_id"], "path_a": s1["path"],
                    "example_id_b": s2["example_id"], "path_b": s2["path"],
                    "start_a": s1["start"], "end_a": s1["end"],
                    "start_b": s2["start"], "end_b": s2["end"],
                    "seq_gap": int(gap),
                    "cyclization_type_a": s1["cyclization_type"],
                    "cyclization_type_b": s2["cyclization_type"],
                })

    summary = {
        "metadata": str(meta),
        "n_rows": int(len(df)),
        "n_unparsed_ids": int((~ok).sum()),
        "n_domains": int(len(counts)),
        "n_multi_segment_domains": int(len(multi)),
        "n_within_domain_pairs": int(n_pairs),
        "n_sequence_disjoint_pairs": int(n_disjoint),
        f"n_disjoint_and_ge{min_sep}_apart": int(n_disjoint_sep),
        "n_domains_with_a_disjoint_pair": int(pd.DataFrame(pair_rows)["domain"].nunique())
        if pair_rows else 0,
        "min_sequence_separation": min_sep,
    }

    pairs = pd.DataFrame(pair_rows)
    cap = int(cfg["spatial_sample_cap"])
    summary["spatial_sample_cap"] = cap
    if len(pairs) > cap:
        # Uniform, so the surviving fraction measured on the sample estimates the fraction
        # over the whole population with a binomial interval the report can state.
        pairs = pairs.iloc[np.sort(rng.choice(len(pairs), cap, replace=False))]
    summary["n_pairs_sampled_for_spatial"] = int(len(pairs))
    summary["spatial_sampling_fraction"] = (
        float(len(pairs)) / n_disjoint_sep if n_disjoint_sep else float("nan"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "deliverable_e_index.json").write_text(json.dumps(summary, indent=2))
    pairs.to_parquet(args.out_dir / "deliverable_e_candidate_pairs.parquet", index=False)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------- spatial
def receptor_and_peptide(path: Path) -> dict | None:
    residues, _ = parse_pdb_chains(path, {"A", "B"})
    rec, pep = residues["A"], residues["B"]
    rec_ca = {int(k[0]): np.array(rec[k]["CA"][1]) for k in rec if "CA" in rec[k]}
    pep_ca = np.array([pep[k]["CA"][1] for k in pep if "CA" in pep[k]], dtype=float)
    if len(rec_ca) < 3 or len(pep_ca) < 2:
        return None
    return {"rec_ca": rec_ca, "pep_ca": pep_ca}


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation+translation taking P onto Q, and the resulting RMSD."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - R @ pc
    rmsd = float(np.sqrt((((R @ P.T).T + t - Q) ** 2).sum(-1).mean()))
    return R, t, rmsd


def contact_resseq(rec_ca: dict[int, np.ndarray], pep_ca: np.ndarray, cutoff: float) -> set[int]:
    keys = list(rec_ca)
    coords = np.array([rec_ca[k] for k in keys])
    d = np.linalg.norm(coords[:, None, :] - pep_ca[None, :, :], axis=-1)
    return {keys[i] for i in np.where(d.min(axis=1) < cutoff)[0]}


def spatial(args) -> None:
    full = m15_config.load(args.config)
    cfg, geo = full["deliverable_e"], full["geometry"]
    cutoff = float(geo["contact_cutoff_A"])

    pairs = pd.read_parquet(args.pairs)
    shard = pairs.iloc[args.shard::args.num_shards]
    print(f"[spatial] shard {args.shard}/{args.num_shards}: {len(shard)} of {len(pairs)} pairs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict | None] = {}
    n_ok = 0
    with args.out.open("w") as fh:
        for rec in shard.itertuples(index=False):
            row = {"domain": rec.domain, "example_id_a": rec.example_id_a,
                   "example_id_b": rec.example_id_b, "seq_gap": int(rec.seq_gap)}
            try:
                for tag, path in (("a", rec.path_a), ("b", rec.path_b)):
                    if path not in cache:
                        cache[path] = receptor_and_peptide(Path(path))
                sa, sb = cache[rec.path_a], cache[rec.path_b]
            except Exception as exc:  # noqa: BLE001
                row.update(status="unreadable", error=f"{type(exc).__name__}: {exc}")
                fh.write(json.dumps(row) + "\n")
                continue
            if sa is None or sb is None:
                row["status"] = "incomplete"
                fh.write(json.dumps(row) + "\n")
                continue

            ca_set = contact_resseq(sa["rec_ca"], sa["pep_ca"], cutoff)
            cb_set = contact_resseq(sb["rec_ca"], sb["pep_ca"], cutoff)
            inter, union = len(ca_set & cb_set), len(ca_set | cb_set)
            row.update({
                "n_contacts_a": len(ca_set), "n_contacts_b": len(cb_set),
                "contact_jaccard": (inter / union) if union else float("nan"),
                "n_shared_contact_residues": inter,
            })

            shared = sorted(set(sa["rec_ca"]) & set(sb["rec_ca"]))
            if len(shared) >= 3:
                P = np.array([sb["rec_ca"][k] for k in shared])
                Q = np.array([sa["rec_ca"][k] for k in shared])
                R, t, fit = kabsch(P, Q)
                pep_b_in_a = (R @ sb["pep_ca"].T).T + t
                row["superpose_rmsd_A"] = fit
                row["n_shared_receptor_residues"] = len(shared)
                row["centroid_separation_A"] = float(
                    np.linalg.norm(sa["pep_ca"].mean(0) - pep_b_in_a.mean(0)))
                row["min_peptide_peptide_ca_A"] = float(
                    np.linalg.norm(sa["pep_ca"][:, None, :] - pep_b_in_a[None, :, :],
                                   axis=-1).min())
            else:
                # No shared receptor residues means no common frame; the Jaccard is still
                # valid (it is 0 by construction) but the separation is not measurable.
                row["superpose_rmsd_A"] = float("nan")
                row["n_shared_receptor_residues"] = len(shared)
                row["centroid_separation_A"] = float("nan")
                row["min_peptide_peptide_ca_A"] = float("nan")
            row["status"] = "ok"
            n_ok += 1
            fh.write(json.dumps(row) + "\n")
    print(f"[spatial] {n_ok} scored -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("index")
    i.add_argument("--config", type=Path, required=True)
    i.add_argument("--out-dir", type=Path, required=True)
    i.set_defaults(func=index)

    s = sub.add_parser("spatial")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--pairs", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.set_defaults(func=spatial)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
