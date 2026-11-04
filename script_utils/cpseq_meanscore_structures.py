#!/usr/bin/env python3
"""Resolve each mean-score target to a native complex, and say which chain is the peptide.

The learned scorer in `cpseq_meanscore_scorer.py` sees no structure at all, and the
feature baselines say that is the binding constraint: post-fold readouts reach
within-target rho 0.781 against a ~0.80 ceiling, so once a structure exists the Rosetta dG
is nearly determined and the whole difficulty is sequence -> structure. To inject
structural context *without* paying for a fold, every candidate has to be scored against
its target's reference backbone -- which first requires knowing, per target, which file
holds the complex and which chain inside it is the peptide.

That mapping is derived, not configured. For each target the dataset already carries the
reference peptide sequence (`original_sequence`, identical across all of a target's rows),
so this script reads every chain of the candidate complex and keeps the chain whose
sequence **equals** it. An exact match is self-validating: a wrong file or a wrong chain
cannot produce one. Targets that only match approximately are reported with their best
alignment and `exact: false` rather than silently accepted, because a near-match usually
means a different construct, a different chain pair, or a numbering offset -- all of which
would put the substitutions at the wrong positions and quietly corrupt every feature
downstream.

Two structure sources, because the 20 targets come from two cohorts:
  * `pepbench_<A>_<B>_pdb<id>`  -> LPData/PepBench/train_valid/pdbs/<A>_<B>_pdb<id>.pdb
    The chain pair is in the filename, so the peptide chain is known a priori and the
    sequence match is a check rather than a search.
  * `lnr_<id>`                  -> LNR/pdbs/<id>.pdb
    No chain pair in the name, so the peptide chain is found by the sequence match.

Only backbone geometry is used downstream (ProteinMPNN reads N, CA, C, O), so alternate
conformations and side-chain completeness do not matter here; chain identity and residue
order do.

Usage:
    python script_utils/cpseq_meanscore_structures.py \
        --dataset /zfsauton/scratch/yixiz/cpseq/meanscore/dataset_v3.json \
        --json    /zfsauton/scratch/yixiz/cpseq/meanscore/structures_v3.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}

PEPBENCH_DIR = Path("/zfsauton/scratch/yixiz/LPData/PepBench/train_valid/pdbs")
LNR_DIR = Path("/zfsauton/scratch/yixiz/LNR/pdbs")


def read_chains(pdb: Path) -> dict[str, list[tuple]]:
    """chain -> ordered [(resseq, icode, resname)] from ATOM records, first altloc only.

    Keyed on (resseq, icode) so insertion codes do not collapse two residues into one.
    """
    chains: dict[str, list[tuple]] = collections.defaultdict(list)
    seen: dict[str, set] = collections.defaultdict(set)
    with pdb.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            ch = line[21]
            resname = line[17:20].strip()
            key = (line[22:26].strip(), line[26])
            if key in seen[ch]:
                continue
            seen[ch].add(key)
            chains[ch].append((key[0], key[1], resname))
    return dict(chains)


def chain_seq(res: list[tuple]) -> str:
    return "".join(THREE_TO_ONE.get(r[2], "X") for r in res)


def best_window(hay: str, needle: str) -> tuple[int, int]:
    """(offset, n_matches) of the best equal-length window of `hay` against `needle`."""
    if not needle or len(hay) < len(needle):
        return (-1, 0)
    best = (-1, -1)
    for i in range(len(hay) - len(needle) + 1):
        m = sum(1 for a, b in zip(hay[i : i + len(needle)], needle) if a == b)
        if m > best[1]:
            best = (i, m)
    return best


def candidate_paths(target: str) -> list[Path]:
    if target.startswith("pepbench_"):
        stem = target[len("pepbench_") :]
        return [PEPBENCH_DIR / f"{stem}.pdb"]
    if target.startswith("lnr_"):
        stem = target[len("lnr_") :]
        return [LNR_DIR / f"{stem}.pdb"]
    return []


def resolve(target: str, ref_seq: str) -> dict:
    out = {
        "target": target,
        "reference_sequence": ref_seq,
        "reference_length": len(ref_seq),
        "pdb": None,
        "peptide_chain": None,
        "receptor_chains": [],
        "exact": False,
        "note": None,
    }
    paths = [p for p in candidate_paths(target) if p.exists()]
    if not paths:
        out["note"] = "no candidate pdb on disk"
        return out
    pdb = paths[0]
    out["pdb"] = str(pdb)
    chains = read_chains(pdb)
    if not chains:
        out["note"] = "no CA atoms parsed"
        return out

    seqs = {c: chain_seq(r) for c, r in chains.items()}
    out["chain_lengths"] = {c: len(s) for c, s in seqs.items()}

    # Exact whole-chain match is the strongest evidence and is tried first.
    exact = [c for c, s in seqs.items() if s == ref_seq]
    if len(exact) == 1:
        out.update(peptide_chain=exact[0], exact=True,
                   receptor_chains=sorted(c for c in seqs if c != exact[0]))
        return out
    if len(exact) > 1:
        # Prefer the chain the filename names, if the cohort encodes one.
        named = target[len("pepbench_"):].split("_")[1] if target.startswith("pepbench_") else None
        pick = named if named in exact else sorted(exact)[0]
        out.update(peptide_chain=pick, exact=True,
                   receptor_chains=sorted(c for c in seqs if c != pick),
                   note=f"reference matches {len(exact)} chains exactly ({sorted(exact)}); "
                        f"took {pick}")
        return out

    # No exact chain. Report the best window so the failure is diagnosable rather than
    # a bare "not found": an offset with near-full identity means a numbering/construct
    # difference, a low identity means the wrong file.
    scored = []
    for c, s in seqs.items():
        off, m = best_window(s, ref_seq)
        scored.append((m / max(1, len(ref_seq)), c, off, m, len(s)))
    scored.sort(reverse=True)
    ident, c, off, m, ln = scored[0]
    out.update(
        peptide_chain=c,
        exact=False,
        receptor_chains=sorted(x for x in seqs if x != c),
        note=(f"NO exact chain match. best: chain {c} len {ln}, window offset {off}, "
              f"{m}/{len(ref_seq)} identical ({ident:.2f}). "
              f"chain seqs: { {k: v[:40] for k, v in seqs.items()} }"),
    )
    out["best_identity"] = ident
    out["best_offset"] = off
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.dataset.read_text())
    refs: dict[str, str] = {}
    for r in payload["rows"]:
        if r.get("original_sequence"):
            refs.setdefault(r["parent_id"], r["original_sequence"])

    print(f"dataset   {args.dataset}")
    print(f"targets   {len(refs)}\n")
    print(f"{'target':24s} {'pep':>4s} {'len':>4s} {'recv':>5s} {'exact':>6s}  pdb")
    results = {}
    for t in sorted(refs):
        res = resolve(t, refs[t])
        results[t] = res
        print(f"{t:24s} {str(res['peptide_chain']):>4s} {res['reference_length']:4d} "
              f"{len(res['receptor_chains']):5d} {str(res['exact']):>6s}  "
              f"{Path(res['pdb']).name if res['pdb'] else '-'}")
        if res["note"]:
            print(f"    NOTE: {res['note'][:300]}")

    n_exact = sum(1 for v in results.values() if v["exact"])
    print(f"\nexact peptide-chain matches: {n_exact}/{len(results)}")
    if n_exact < len(results):
        print("  Targets without an exact match are NOT usable for structure features")
        print("  until resolved -- a wrong chain puts every substitution at the wrong")
        print("  position. They are kept in the manifest with exact=false so the")
        print("  feature builder can skip them explicitly.")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"dataset": str(args.dataset), "n_targets": len(results),
         "n_exact": n_exact, "targets": results}, indent=1))
    print(f"\nwrote     {args.json}")


if __name__ == "__main__":
    main()
