"""Count how much of a corpus split is eligible, through the REAL prepare filter.

Parsing alone answers "how many survive, and why do the rest not" -- no templates, no
MSA, no network, no GPU. Those answers are what size the expensive stages, so they come
first.

It calls `load_lnr_contexts`, the same function `cpseq_prepare_targets.py` calls, so the
verdict here is the verdict there. Re-deriving the filter would let the two drift, and
this would then describe a filter nobody runs.

Sharding: the loader reads `<root>/<list_file>` and `<root>/pdbs/`, so each shard gets a
private dir holding its slice of the list beside a symlink to the real `pdbs/`. The
corpus dir is never written to. A shard whose JSON exists is skipped, so a time limit
means resubmitting, not restarting.

Usage:
    python script_utils/cpseq_corpus_inventory.py --root <corpus> --list-file train.txt \
        --source PepBench --out-dir <out> --shard-index 0 --shard-count 20
    python script_utils/cpseq_corpus_inventory.py --root <corpus> --list-file train.txt \
        --source PepBench --out-dir <out> --merge --msa-cache-dirs <dir> [<dir> ...]
"""

from __future__ import annotations

import argparse
import collections
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


def entries(root: Path, list_file: str) -> list[str]:
    """The split's lines, keeping only what the loader keeps (>=3 whitespace fields)."""
    return [l for l in (root / list_file).read_text().splitlines() if len(l.split()) >= 3]


def kind(reason: str) -> str:
    """Collapse a reason to its kind: the filter has four, not one per peptide length."""
    for probe, name in (
        ("outside", "peptide length outside window"),
        ("absent", "peptide chain absent"),
        ("structure file missing", "structure file missing"),
        ("no receptor chain contacts", "no receptor chain contacts the peptide"),
    ):
        if probe in reason:
            return name
    return reason


def run_shard(a: argparse.Namespace) -> int:
    out = a.out_dir / f"shard{a.shard_index:04d}.json"
    if out.exists() and not a.force:
        print(f"[inventory] shard {a.shard_index} done already -> {out}")
        return 0

    all_lines = entries(a.root, a.list_file)
    mine = all_lines[a.shard_index :: a.shard_count]
    print(f"[inventory] {a.list_file}: {len(all_lines)} entries; "
          f"shard {a.shard_index}/{a.shard_count} takes {len(mine)}")

    root = a.out_dir / "shards" / f"shard{a.shard_index:04d}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "list.txt").write_text("\n".join(mine) + "\n")
    if not (root / "pdbs").exists():
        (root / "pdbs").symlink_to(a.root.resolve() / "pdbs")

    contexts, rejected = load_lnr_contexts(
        root, "list.txt", min_len=a.min_len, max_len=a.max_len,
        source=a.source, id_prefix=a.source.lower(),
    )

    # Receptor MSAs are keyed by sequence hash, so the number of DISTINCT keys is the
    # fetch count -- not the number of targets.
    receptors: dict[str, int] = {}
    eligible = []
    for c in contexts:
        for seq in c.receptor_sequences.values():
            receptors[sequence_key(seq)] = len(seq)
        eligible.append({"parent_id": c.parent_id, "pdb_id": c.pdb_id,
                         "peptide_length": c.peptide_length})

    out.write_text(json.dumps({
        "shard_index": a.shard_index, "n_entries": len(mine),
        "n_eligible": len(eligible), "n_rejected": len(rejected),
        "eligible": eligible, "rejected": rejected,
        "receptor_seq_lengths": receptors,
    }, indent=2))
    print(f"[inventory] shard {a.shard_index}: {len(eligible)} eligible, "
          f"{len(rejected)} rejected, {len(receptors)} unique receptor(s)")
    return 0


def merge(a: argparse.Namespace) -> int:
    shards = sorted(a.out_dir.glob("shard*.json"))
    if not shards:
        print(f"FATAL: no shard outputs under {a.out_dir}")
        return 1

    seen = n_elig = 0
    rejected: list[dict] = []
    eligible: list[dict] = []
    receptors: dict[str, int] = {}
    for f in shards:
        d = json.loads(f.read_text())
        seen += d["n_entries"]
        n_elig += d["n_eligible"]
        rejected += d["rejected"]
        eligible += d["eligible"]
        receptors.update(d["receptor_seq_lengths"])

    expected = len(entries(a.root, a.list_file))
    cached = {p.stem for d in a.msa_cache_dirs for p in Path(d).glob("*.a3m")}

    summary = {
        "list_file": a.list_file, "source": a.source,
        "peptide_length_window": [a.min_len, a.max_len],
        "n_shards_found": len(shards),
        "n_entries_in_split": expected,
        "n_entries_seen": seen,
        "complete": seen == expected,
        "n_eligible": n_elig,
        "n_rejected": len(rejected),
        "rejection_kinds": dict(collections.Counter(kind(r["reason"]) for r in rejected)),
        "n_unique_receptor_sequences": len(receptors),
        "n_receptor_msas_cached": len(set(receptors) & cached),
        "n_receptor_msas_to_fetch": len(set(receptors) - cached),
    }
    (a.out_dir / "inventory_summary.json").write_text(json.dumps(summary, indent=2))
    (a.out_dir / "receptor_msas_to_fetch.json").write_text(
        json.dumps(sorted(set(receptors) - cached), indent=2))

    # The eligible split in the corpus's own format, so staging is pointed at this
    # instead of re-deriving eligibility and possibly disagreeing with it.
    keep = {e["pdb_id"] for e in eligible}
    kept = [l for l in entries(a.root, a.list_file) if l.split()[0] in keep]
    (a.out_dir / "eligible.txt").write_text("\n".join(kept) + "\n")

    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        print(f"\nWARNING: saw {seen} of {expected} entries -- shards MISSING, counts are "
              f"partial. Resubmit the array before quoting any of this.")
        return 1
    print(f"\nwrote inventory_summary.json, receptor_msas_to_fetch.json, "
          f"eligible.txt ({len(kept)} lines) under {a.out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--list-file", default="train.txt")
    ap.add_argument("--source", default="PepBench")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--min-len", type=int, default=PEPTIDE_MIN_LEN)
    ap.add_argument("--max-len", type=int, default=PEPTIDE_MAX_LEN)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--msa-cache-dirs", nargs="*", type=Path, default=[])
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    return merge(a) if a.merge else run_shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
