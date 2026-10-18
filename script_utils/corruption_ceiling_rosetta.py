#!/usr/bin/env python
"""Rosetta interface dG for the corruption-ceiling complexes, as a CPU sidecar.

One row per dumped complex, keyed by `pdb_tag`, so the report can join it back onto the
ladder rows and read dG against each example's own native rung. Kept off the OpenMM job
because it needs a different interpreter and parallelises differently.

dG_separated is not comparable across targets -- a bigger interface scores lower whatever
its quality -- so nothing here compares two examples. The report only ever uses the
per-example DELTA against that example's native rung.

The complexes are OPEN peptides mid-corruption, so no cyclization bond is declared
(`cyclization_type=None`): declaring one that isn't there is what makes FastRelax drag the
gap shut and report a dG for a structure that was never scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.evaluation.rosetta_energy import (  # noqa: E402
    compute_rosetta_interface_metrics_single,
    is_pyrosetta_available,
)

DG_COL = "binder_rosetta_dG_separated"


def load_rows(paths):
    """Ladder rows. A corrupt line is fatal: the aggregate would still compute without it."""
    rows = []
    for p in paths:
        for lineno, line in enumerate(Path(p).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(f"FATAL: corrupt row {p}:{lineno}. Delete and rerun that shard.")
    return rows


def already_scored(out_path: Path) -> set[str]:
    done = set()
    if not out_path.exists():
        return done
    for lineno, line in enumerate(out_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["pdb_tag"])
        except (json.JSONDecodeError, KeyError):
            raise SystemExit(f"FATAL: corrupt row {out_path}:{lineno}. Delete and rerun.")
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", nargs="+", required=True, help="ladder_shard*.jsonl")
    ap.add_argument("--out", required=True, help="Output JSONL for THIS shard (appended).")
    ap.add_argument("--binder-chain", default="B", help="write_projected_pdb's convention.")
    ap.add_argument("--target-chains", default="A")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    if not is_pyrosetta_available():
        # A gate, not a failure: exiting non-zero here strands the report job in
        # DependencyNeverSatisfied with nothing on record explaining why.
        print("GATE: PyRosetta is not importable in this interpreter. No dG rows written;")
        print("GATE: exiting 0 so the report still runs on the geometry alone.")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_scored(out_path)

    todo = [r for r in load_rows(args.ladder)
            if r.get("complex_pdb") and r.get("pdb_tag") and r.get("status") != "failed"]
    todo = [r for r in todo if r["pdb_tag"] not in done]
    todo = todo[args.shard_index::args.shard_count]
    print(f"[rosetta] shard {args.shard_index}/{args.shard_count}: {len(todo)} to score "
          f"({len(done)} already done)", flush=True)

    n = 0
    with out_path.open("a") as fh:
        for r in todo:
            pdb = Path(r["complex_pdb"])
            if not pdb.is_file():
                print(f"  MISSING {pdb}", flush=True)
                continue
            m = compute_rosetta_interface_metrics_single(
                str(pdb), args.binder_chain, args.target_chains, cyclization_type=None)
            fh.write(json.dumps({
                "pdb_tag": r["pdb_tag"], "example_id": r["example_id"],
                "rung_index": r.get("rung_index"), "target_gap_A": r.get("target_gap_A"),
                "is_native_rung": r.get("is_native_rung"), **m,
            }) + "\n")
            fh.flush()
            n += 1
            print(f"  {r['pdb_tag']} {DG_COL}={m.get(DG_COL)}", flush=True)

    print(f"[rosetta] scored {n} complexes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
