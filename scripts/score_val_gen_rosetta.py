#!/usr/bin/env python
"""CPU sidecar: score Rosetta interface dG on the complexes val_generation dumped to disk.

Pairs with `eval/val_gen_dump.py` (the GPU training loop writes complex PDBs + manifest rows) and
`scripts/score_val_gen_rosetta.sbatch` (the CPU job that runs this). Never import this into the
training process -- Rosetta relax is seconds of CPU per complex and would stall the GPU.

For every `step_*/manifest_rank*.jsonl` row not already scored, run
`compute_rosetta_interface_metrics_single` and append one row to the output JSONL. Resumable: a row
already present in the output (keyed by step+pdb) is skipped, so re-running after a time-out only
scores what is new. wandb can then read the output JSONL and plot `dG_separated` against `step`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Repo `src` on path when run as a bare script (the sbatch also sets PYTHONPATH).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from proteinfoundation.evaluation.rosetta_energy import (  # noqa: E402
    compute_rosetta_interface_metrics_single,
    is_pyrosetta_available,
)


def _already_scored(out_path: str) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((int(r["step"]), r["pdb"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                # Refuse to silently treat a corrupt line as "not done" AND as "done": skip it for
                # the resume set but leave it in the file for a human to see (parallel-append rules).
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True, help="Root dir containing step_*/ dumps.")
    ap.add_argument("--out", required=True, help="Output JSONL (one row per scored complex).")
    ap.add_argument("--max-per-step", type=int, default=0, help="Cap complexes scored per step (0=all).")
    args = ap.parse_args()

    if not is_pyrosetta_available():
        # A gate-failure, not an error: exit 0 so an afterok chain is not stranded (CLAUDE.md rule).
        print("PyRosetta unavailable; nothing scored. Exiting 0.", flush=True)
        return 0

    done = _already_scored(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    manifests = sorted(glob.glob(os.path.join(args.dump_dir, "step_*", "manifest_rank*.jsonl")))
    if not manifests:
        print(f"No manifests under {args.dump_dir}; nothing to score. Exiting 0.", flush=True)
        return 0

    n_new = 0
    with open(args.out, "a") as out_f:
        for man in manifests:
            step_dir = os.path.dirname(man)
            per_step = 0
            with open(man) as mf:
                rows = [json.loads(l) for l in mf if l.strip()]
            for row in rows:
                key = (int(row["step"]), row["pdb"])
                if key in done:
                    continue
                if args.max_per_step and per_step >= args.max_per_step:
                    break
                pdb_path = os.path.join(step_dir, row["pdb"])
                if not os.path.exists(pdb_path):
                    continue
                metrics = compute_rosetta_interface_metrics_single(
                    pdb_path=pdb_path,
                    binder_chain=row["binder_chain"],
                    target_chains=row["target_chains"],
                    cyclization_type=row.get("cyclization_type"),
                    cyclization_i=row.get("cyclization_i"),
                    cyclization_j=row.get("cyclization_j"),
                )
                carry = ("step", "epoch", "rank", "bid", "pdb", "cyclization_type",
                         "cyclization_i", "cyclization_j")
                out_row = {k: row[k] for k in carry if k in row}
                out_row.update(metrics)
                out_f.write(json.dumps(out_row) + "\n")
                out_f.flush()
                done.add(key)
                per_step += 1
                n_new += 1

    print(f"Scored {n_new} new complexes -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
