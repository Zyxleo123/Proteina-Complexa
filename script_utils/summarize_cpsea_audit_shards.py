#!/usr/bin/env python3
"""Merge the per-shard CSVs from a full `audit_cpsea_cyclization_endpoints.py` scan.

    python script_utils/summarize_cpsea_audit_shards.py slurm_logs/cyc_audit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "slurm_logs/cyc_audit")
    files = sorted(d.glob("shard_*.csv"))
    if not files:
        print(f"no shard_*.csv under {d}")
        return 1
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    n = len(df)
    print(f"merged {len(files)} shards -> {n} structures\n")

    def rate(col, want=True):
        if col not in df:
            return "column absent"
        s = df[col]
        ev = s.notna().sum()
        good = int((s == want).sum())
        return f"{good}/{ev} ({100.0 * good / ev:.6f}%)" + (f"  [{n - ev} not evaluable]" if n - ev else "")

    for col, label in [
        ("raw_terminal", "RAW  bond is (0, L-1)"),
        ("pre_terminal", "PRE  bond is (0, L-1)"),
        ("pre_dist_sane", "PRE  anchor dist sane"),
        ("agree", "RAW/PRE same (i,j,chem)"),
        ("coords_preserved", "RAW/PRE coords identical"),
        ("len_matches_metadata", "PRE  len == metadata"),
    ]:
        print(f"{label:<26}: {rate(col)}")

    print("\nstatuses:")
    print(df["raw_status"].value_counts().to_string())
    print(df["pre_status"].value_counts().to_string())

    print("\nchemistry x metadata hint:")
    print(pd.crosstab(df.get("pre_chem"), df["hint"]).to_string())

    viol = df[
        (df.get("raw_terminal") == False)  # noqa: E712 - NaN must not count as a violation
        | (df.get("pre_terminal") == False)  # noqa: E712
        | (df.get("agree") == False)  # noqa: E712
        | (df.get("pre_dist_sane") == False)  # noqa: E712
        | (df.get("coords_preserved") == False)  # noqa: E712
    ]
    print(f"\nviolations: {len(viol)}")
    if len(viol):
        out = d / "violations.csv"
        viol.to_csv(out, index=False)
        print(viol.head(20).to_string())
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
