"""Live progress of a de novo / SDEdit run, straight off the shard JSONLs.

Safe on a login node: it only counts lines in files the GPU jobs append to. The sampler
flushes after every edit and completes each target's whole (chemistry x seed) block before
moving on, so a prefix is chemistry-balanced -- a mid-run read is a real preview, not a
biased one.

Usage:
    python scripts/uncond_progress.py evaluation_results/uncondgen_meet_<STAMP> \
        [--metadata CPSea_data/meet_staged_pocket18/metadata/meet_valid_pocket18.parquet]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--metadata", type=Path, default=None,
                    help="If given, reports progress as a fraction of the targets that were "
                         "actually requested, instead of only what has been written so far.")
    ap.add_argument("--per-shard", action="store_true")
    args = ap.parse_args()

    shards = sorted(args.results_dir.glob("edits_shard*.jsonl"))
    if not shards:
        raise SystemExit(f"no edits_shard*.jsonl under {args.results_dir} -- the array has "
                         f"not written anything yet (check slurm_logs/).")

    per_target: Counter[str] = Counter()
    per_type: Counter[str] = Counter()
    ok = failed = corrupt = 0
    for sh in shards:
        n_sh = 0
        for line in sh.read_text().splitlines():
            if not line.strip():
                continue
            if "\x00" in line:
                corrupt += 1
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                corrupt += 1
                continue
            n_sh += 1
            if r.get("status") == "failed":
                failed += 1
                continue
            ok += 1
            per_target[r.get("example_id")] += 1
            per_type[r.get("cyc_type")] += 1
        if args.per_shard:
            print(f"  {sh.name}: {n_sh} rows")

    full = max(per_target.values()) if per_target else 0
    complete = sum(1 for v in per_target.values() if v == full)
    print(f"{ok} ok edits, {failed} failed, {corrupt} corrupt, across {len(shards)} shard(s)")
    print(f"targets touched: {len(per_target)}  "
          f"({complete} complete at {full} edits each, {len(per_target) - complete} in flight)")
    print("edits per requested chemistry: " + ", ".join(f"{k}={v}" for k, v in sorted(per_type.items())))

    if args.metadata:
        import pandas as pd
        want = len(pd.read_parquet(args.metadata))
        print(f"progress: {complete}/{want} targets complete "
              f"({100.0 * complete / want:.1f}%), {ok}/{want * full if full else 0} edits"
              if full else f"progress: 0/{want} targets complete")
    if corrupt:
        raise SystemExit(f"FATAL: {corrupt} corrupt line(s) -- shards must never share an "
                         f"output path. Do not summarise this run until it is explained.")


if __name__ == "__main__":
    main()
