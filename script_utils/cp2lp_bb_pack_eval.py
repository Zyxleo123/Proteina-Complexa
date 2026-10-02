"""CPU stage of the checkpoint evaluation: pack every dumped backbone and score it all-atom.

Reads the npz dumps and the per-attempt JSONL that `cp2lp_bb_evaluate.py` wrote, runs the
shared packing adapter over each attempt with the same fixed seed budget in every arm, and
emits ONE manifest row per attempt carrying every outcome including the failures.

THIS IS THE JOB THAT DECIDES WHETHER THE RESULT IS HOLISTIC. Until it has run, the arms are
scored on backbone geometry alone, which is a different and much easier question than "is the
completed linear peptide any good". A report built without it must say backbone-only.

CPU, and resumable: each attempt's row is appended as it finishes and an already-finished
attempt is skipped, so hitting a time limit is a resubmit rather than a restart. One output
file PER JOB, never shared -- concurrent O_APPEND to one file was measured to NUL-corrupt rows
here (88 of 144 lost), so the aggregator reads a directory of files and REFUSES corrupt lines
rather than skipping them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attempts", required=True, help="JSONL from cp2lp_bb_evaluate.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-name", default=None, help="defaults to packed__<job-id-or-pid>.jsonl")
    p.add_argument("--pack-seeds", type=int, default=3, help="IDENTICAL in every arm")
    p.add_argument("--source-pdb-root", default=None,
                   help="optional directory of <example_id>.pdb source complexes, for chi "
                        "seeding and the source contact map")
    p.add_argument("--keep-pdbs", action="store_true", help="retain every packed PDB")
    p.add_argument("--no-minimize", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="stop after N attempts (smoke)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from proteinfoundation.cp2lp import packing as pk

    if not pk.is_available():
        logger.error(
            "PyRosetta is unavailable in this environment, so no attempt can be completed to "
            "all-atom. Any report built from this run MUST be labelled backbone-only."
        )
        return 1

    attempts = [json.loads(l) for l in Path(args.attempts).read_text().splitlines() if l.strip()]
    if not attempts:
        print(f"GATE: {args.attempts} has no rows -- the generate stage produced nothing.")
        return 0
    logger.info(f"{len(attempts)} attempts to pack from {args.attempts}")

    out = Path(args.out_dir)
    (out / "packed").mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    out_file = out / (args.out_name or f"packed__{job}.jsonl")

    # RESUMABLE. Every row already written by ANY job in this directory is considered done, so
    # a resubmit does not redo work and two jobs splitting the panel do not collide.
    done: set[str] = set()
    for f in sorted(out.glob("packed__*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["tag"])
            except Exception:
                # A corrupt line is REFUSED, not skipped: a silently dropped row makes the
                # denominators wrong and the comparison unattributable.
                logger.error(f"corrupt row in {f} -- refusing to proceed: {line[:120]!r}")
                return 1
    if done:
        logger.info(f"{len(done)} attempts already packed; skipping those")

    import pyrosetta

    from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init

    _ensure_pyrosetta_init()
    scorefxn = pyrosetta.create_score_function("ref2015")

    n_done = n_valid = 0
    t0 = time.time()
    with open(out_file, "a") as fh:
        for k, a in enumerate(attempts):
            tag = f"{a['arm']}__{a['ckpt_tag']}__{a['example_id']}__s{a['seed']}"
            if tag in done:
                continue
            if args.limit and n_done >= args.limit:
                break
            try:
                z = np.load(a["npz"])
            except Exception as exc:
                fh.write(json.dumps({
                    "tag": tag, **{x: a[x] for x in ("arm", "ckpt_tag", "example_id", "seed")},
                    "setup_error": f"npz unreadable: {type(exc).__name__}: {exc}",
                    "selected_seed": -1, "selection_basis": "dump_unreadable",
                }) + "\n")
                fh.flush()
                continue

            src_pdb = None
            if args.source_pdb_root:
                cand = Path(args.source_pdb_root) / f"{a['example_id']}.pdb"
                if cand.exists():
                    src_pdb = str(cand)

            bb = torch.from_numpy(z["gen_bb"])
            n = bb.shape[0]
            tmask = torch.from_numpy(z["target_mask"]).bool()
            row = pk.pack_candidate(
                bb=bb,
                aatype=z["pep_aatype"],
                pep_mask=torch.ones(n, dtype=torch.bool),
                target_atom37_ang=z["target_atom37_ang"][tmask.numpy()],
                target_aatype=z["target_aatype"][tmask.numpy()],
                work_dir=out / "packed" / a["arm"],
                tag=tag,
                source_pdb=src_pdb,
                n_seeds=int(args.pack_seeds),
                minimize=not args.no_minimize,
                scorefxn=scorefxn,
                keep_pdbs=bool(args.keep_pdbs),
            )
            # The RAW backbone measurements travel with the packed ones, so a report can bin
            # by edit size or terminal gap without joining two files.
            row.update({x: a[x] for x in a if x not in ("npz",)})
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            n_done += 1
            n_valid += int(row.get("n_valid", 0) > 0)
            if n_done % 10 == 0:
                logger.info(
                    f"  {n_done} packed ({n_valid} with a valid candidate), "
                    f"{(time.time() - t0) / max(n_done, 1):.1f}s each"
                )

    print(json.dumps({
        "attempts_seen": len(attempts), "packed_this_job": n_done,
        "with_valid_candidate": n_valid, "out": str(out_file),
        "cpu_seconds": time.time() - t0,
        "seconds_per_attempt": (time.time() - t0) / max(n_done, 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
