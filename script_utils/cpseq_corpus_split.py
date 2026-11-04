"""Split an eligible corpus list into N cluster shares, grouped by receptor sequence.

Why receptor-grouped. The expensive part of staging is the ColabFold MSA fetch, and
MSAs are keyed by receptor *sequence hash*, not by complex. If two complexes that share a
receptor land on different clusters, that receptor's MSA is fetched twice (autonlab and
babel have separate filesystems and separate caches). So we keep every complex that
shares a receptor on one side: the receptor hash sets of the two shares are then disjoint
and nothing is fetched twice.

A complex can list more than one receptor chain, so "shares a receptor" is transitive:
A--h1--B and B--h2--C must all co-locate. We union-find complexes over shared hashes into
connected components, then greedily bin-pack components (largest first, into the smaller
bin) so the shares are balanced on COMPLEX COUNT.

Eligibility is NOT re-derived here -- the input list is the inventory's `eligible.txt`,
already filtered by the real prepare filter. We only read structures to recover each
complex's receptor sequence hash, using the SAME `load_lnr_contexts` + `sequence_key` the
inventory used, so the hashes agree by construction with `receptor_msas_to_fetch.json`.

Usage:
    python script_utils/cpseq_corpus_split.py \
        --root <corpus_root> --eligible <inventory>/eligible.txt \
        --source PepBench --out-dir <split_out> --names autonlab babel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.contexts import (  # noqa: E402
    PEPTIDE_MAX_LEN,
    PEPTIDE_MIN_LEN,
    load_lnr_contexts,
)
from proteinfoundation.cpseq.targets import sequence_key  # noqa: E402


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="corpus root holding pdbs/ (the inventory's INV_ROOT)")
    ap.add_argument("--eligible", type=Path, required=True,
                    help="the inventory's eligible.txt (corpus 4-field format)")
    ap.add_argument("--source", default="PepBench")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--names", nargs="+", default=["autonlab", "babel"],
                    help="one label per share; order is deterministic")
    ap.add_argument("--min-len", type=int, default=PEPTIDE_MIN_LEN)
    ap.add_argument("--max-len", type=int, default=PEPTIDE_MAX_LEN)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = a.out_dir / "split_summary.json"
    if summary_path.exists() and not a.force:
        print(f"[split] already done -> {summary_path} (use --force to redo)")
        print(summary_path.read_text())
        return 0

    # Original lines keyed by example id (first field), so each share is written in the
    # corpus's own 4-field format, not reconstructed.
    lines = [l for l in a.eligible.read_text().splitlines() if len(l.split()) >= 3]
    line_of = {l.split()[0]: l for l in lines}
    print(f"[split] {len(lines)} eligible entries from {a.eligible}")

    # Re-run the real loader to recover receptor hashes, via the inventory's work-dir
    # pattern: a private list.txt beside a symlink to the corpus pdbs/.
    work = a.out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "list.txt").write_text("\n".join(lines) + "\n")
    pdbs = work / "pdbs"
    if not pdbs.exists():
        pdbs.symlink_to(a.root.resolve() / "pdbs")

    contexts, rejected = load_lnr_contexts(
        work, "list.txt", min_len=a.min_len, max_len=a.max_len,
        source=a.source, id_prefix=a.source.lower(),
    )
    if rejected:
        # eligible.txt should pass the filter cleanly; a rejection here means the input
        # was not an eligible list or the filter drifted. Surface it, do not hide it.
        print(f"[split] WARNING: {len(rejected)} of {len(lines)} inputs were rejected by "
              f"the loader -- the input may not be an eligible list. First few:")
        for r in rejected[:5]:
            print(f"   {r}")

    # example -> set of receptor hashes; union examples that share any hash.
    uf = UnionFind()
    ex_hashes: dict[str, set[str]] = {}
    hash_owner: dict[str, str] = {}  # a representative example per hash
    missing = 0
    for c in contexts:
        ex = c.pdb_id
        if ex not in line_of:
            missing += 1
            continue
        uf.find(ex)  # register even a hash-less example as its own component
        hs = {sequence_key(seq) for seq in c.receptor_sequences.values()}
        ex_hashes[ex] = hs
        for h in hs:
            if h in hash_owner:
                uf.union(hash_owner[h], ex)
            else:
                hash_owner[h] = ex
    if missing:
        print(f"[split] WARNING: {missing} loaded contexts had no matching eligible line")

    # Components: representative root -> member examples.
    comps: dict[str, list[str]] = {}
    for ex in ex_hashes:
        comps.setdefault(uf.find(ex), []).append(ex)
    total_hashes = len(hash_owner)
    print(f"[split] {len(ex_hashes)} complexes, {total_hashes} unique receptor hashes, "
          f"{len(comps)} connected components "
          f"(largest {max((len(v) for v in comps.values()), default=0)} complexes)")

    # Greedy balanced bin-packing on complex count: largest component into smallest bin.
    k = len(a.names)
    bins: list[list[str]] = [[] for _ in range(k)]
    for _, members in sorted(comps.items(), key=lambda kv: -len(kv[1])):
        j = min(range(k), key=lambda i: len(bins[i]))
        bins[j].extend(members)

    # Write one list file per share, sorted for a stable diff, and tally disjointness.
    shares = []
    hashset_per_bin = []
    for name, members in zip(a.names, bins):
        members_sorted = sorted(members)
        out = a.out_dir / f"split_{name}.txt"
        out.write_text("\n".join(line_of[ex] for ex in members_sorted) + "\n")
        hs: set[str] = set()
        for ex in members:
            hs |= ex_hashes[ex]
        hashset_per_bin.append(hs)
        shares.append({"name": name, "file": str(out),
                       "n_complexes": len(members), "n_receptor_hashes": len(hs)})
        print(f"[split]   {name}: {len(members)} complexes, {len(hs)} receptor MSAs -> {out}")

    # Disjointness is the whole point: assert no receptor hash spans two shares.
    overlap = 0
    for i in range(k):
        for j in range(i + 1, k):
            overlap += len(hashset_per_bin[i] & hashset_per_bin[j])

    summary = {
        "root": str(a.root),
        "eligible": str(a.eligible),
        "source": a.source,
        "peptide_length_window": [a.min_len, a.max_len],
        "n_complexes_total": len(ex_hashes),
        "n_receptor_hashes_total": total_hashes,
        "n_components": len(comps),
        "largest_component": max((len(v) for v in comps.values()), default=0),
        "shares": shares,
        "receptor_hash_overlap_between_shares": overlap,
        "disjoint": overlap == 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if overlap:
        print(f"\nFATAL: {overlap} receptor hashes appear in more than one share -- the "
              f"grouping is broken, the MSA fetch would double-count.")
        return 1
    if sum(s["n_complexes"] for s in shares) != len(line_of):
        print(f"\nFATAL: shares cover {sum(s['n_complexes'] for s in shares)} of "
              f"{len(line_of)} eligible complexes -- some complex was dropped.")
        return 1
    print(f"\nwrote split_summary.json and {k} share lists under {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
