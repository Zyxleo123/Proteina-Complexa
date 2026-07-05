#!/usr/bin/env python3
"""Fast batch runner for check_pdb_residue_jumps using find(1) for file listing."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pdb_residue_jumps import analyze_file, print_file_report  # noqa: E402


def iter_pdbs(root: Path, pattern: str):
    if pattern == "*.pdb.gz":
        name_args = ["-name", "*.pdb.gz"]
    elif pattern == "*.pdb":
        name_args = ["-name", "*.pdb"]
    else:
        name_args = ["(", "-name", "*.pdb", "-o", "-name", "*.pdb.gz", ")"]

    cmd = ["find", str(root), "-type", "f", *name_args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        path = line.strip()
        if path:
            yield Path(path)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"find failed with exit code {proc.returncode}")


def count_pdbs(root: Path, pattern: str) -> int:
    return sum(1 for _ in iter_pdbs(root, pattern))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--pattern", default="*.pdb")
    parser.add_argument("--cn-cutoff", type=float, default=2.0)
    parser.add_argument("--binder-chain", default="B")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50000)
    args = parser.parse_args()

    t0 = time.time()
    if args.input.is_file():
        total = 1
        paths = [args.input]
    else:
        total = count_pdbs(args.input, args.pattern)
        paths = iter_pdbs(args.input, args.pattern)
    print(f"Scanning {total} file(s)", flush=True)
    print(f"Binder chain counted separately: {args.binder_chain}")

    global_counts = Counter()
    files_with_issues = 0
    files_clean = 0

    for i, path in enumerate(paths, start=1):
        events, counts = analyze_file(path, cn_cutoff=args.cn_cutoff, binder_chain=args.binder_chain)
        if events:
            files_with_issues += 1
        else:
            files_clean += 1

        global_counts["REAL_BREAK"] += counts.get("REAL_BREAK", 0)
        global_counts["NUMBERING_ONLY"] += counts.get("NUMBERING_ONLY", 0)
        global_counts["BINDER_REAL_BREAK"] += counts.get("REAL_BREAK_BINDER", 0)
        global_counts["BINDER_NUMBERING_ONLY"] += counts.get("NUMBERING_ONLY_BINDER", 0)

        if not args.quiet:
            print_file_report(path, events, binder_chain=args.binder_chain)
            print(
                "\nFile summary: "
                f"REAL_BREAK={counts.get('REAL_BREAK', 0)}, "
                f"NUMBERING_ONLY={counts.get('NUMBERING_ONLY', 0)}, "
                f"{args.binder_chain}_REAL_BREAK={counts.get('REAL_BREAK_BINDER', 0)}, "
                f"{args.binder_chain}_NUMBERING_ONLY={counts.get('NUMBERING_ONLY_BINDER', 0)}"
            )

        if args.progress_every and i % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate if rate else 0
            print(
                f"[progress] {i}/{total} files "
                f"({100 * i / total:.1f}%), "
                f"REAL_BREAK={global_counts['REAL_BREAK']}, "
                f"NUMBERING_ONLY={global_counts['NUMBERING_ONLY']}, "
                f"elapsed={elapsed / 60:.1f}m, eta={eta / 60:.1f}m",
                flush=True,
            )

    print("\n## Summary")
    print(
        "| Files | With issues | Clean | REAL_BREAK | NUMBERING_ONLY | "
        f"{args.binder_chain}_REAL_BREAK | {args.binder_chain}_NUMBERING_ONLY |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| {total} | {files_with_issues} | {files_clean} | "
        f"{global_counts['REAL_BREAK']} | {global_counts['NUMBERING_ONLY']} | "
        f"{global_counts['BINDER_REAL_BREAK']} | {global_counts['BINDER_NUMBERING_ONLY']} |"
    )
    print(f"\nTotal elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
