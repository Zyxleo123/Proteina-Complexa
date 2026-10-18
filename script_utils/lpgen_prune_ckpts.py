#!/usr/bin/env python3
"""Bound the disk a long RFD3 run spends on checkpoints.

WHY THIS EXISTS. foundry's trainer writes `epoch-NNNN.ckpt` and never prunes --
`FabricTrainer.save_checkpoint` (upstream/foundry/src/foundry/trainers/fabric.py:815)
has no `save_top_k` equivalent, so the only built-in lever is the save cadence. At
2.69 GB per RFD3 checkpoint, `checkpoint_every_n_epochs: 2` and ~1.25 h/epoch, an
unattended fine-tune burns ~26 GB/day and grows without bound. lp_finetune_v1 reached
41 GB in 16 checkpoints before anyone looked.

WHAT IS SAFE TO DELETE. A checkpoint is needed for exactly two things: resuming, and
re-sampling from a chosen epoch. Resuming needs only the newest. Re-sampling wants a
sparse ladder, not every second epoch. Nothing measured is lost either way -- the
per-epoch numbers live in `val_metrics/validation_output_all_epochs.csv` and the sampled
structures in `val_structures/`, both of which this script never touches.

TWO TRAPS THIS SCRIPT IS BUILT AROUND.

1. Resume resolves a DIRECTORY by `sorted(ckpt_load_dir.iterdir())[-1]`
   (fabric.py:948), with NO `*.ckpt` filter. Any file in `ckpt/` sorting after
   `epoch-NNNN.ckpt` is loaded as "the latest checkpoint" -- and `epoch-` starts with
   'e', so a stray `last.ckpt`, `prune.log` or `tmp.x` wins. Hence: this script writes
   NOTHING into the checkpoint directory, and warns if something else has.

2. A checkpoint being written right now is a partial file. Deleting by "keep newest"
   alone could reap a complete checkpoint in favour of a half-written one, so anything
   younger than --min-age-seconds is left alone and never counts as the newest.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

CKPT_RE = re.compile(r"^epoch-(\d+)\.ckpt$")
GB = 1024**3


def find_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path, int, float]]:
    """Return [(epoch, path, size_bytes, age_seconds)] for epoch-NNNN.ckpt only."""
    now = time.time()
    found = []
    for p in ckpt_dir.iterdir():
        m = CKPT_RE.match(p.name)
        if not m or not p.is_file():
            continue
        st = p.stat()
        found.append((int(m.group(1)), p, st.st_size, now - st.st_mtime))
    return sorted(found)


def warn_about_stray_files(ckpt_dir: Path, ckpt_names: set[str]) -> list[str]:
    """Flag files that would hijack `sorted(iterdir())[-1]` resume resolution.

    Only names sorting ABOVE the highest epoch-NNNN.ckpt can win, so that is what we
    report -- a `README` ('R' < 'e') is harmless and not worth shouting about.
    """
    if not ckpt_names:
        return []
    highest = max(ckpt_names)
    return sorted(
        p.name
        for p in ckpt_dir.iterdir()
        if p.name not in ckpt_names and p.name > highest
    )


def select_deletions(
    ckpts: list[tuple[int, Path, int, float]],
    keep_newest: int,
    keep_every: int,
    min_age_seconds: float,
) -> tuple[list[tuple[int, Path, int]], list[tuple[int, Path, int, str]]]:
    """Split checkpoints into (delete, keep-with-reason).

    Settled = old enough to be sure it is not mid-write. Only settled checkpoints are
    eligible for deletion AND for counting as "newest", so an in-flight write can never
    displace the complete checkpoint it is about to replace.
    """
    settled = [c for c in ckpts if c[3] >= min_age_seconds]
    newest_epochs = {c[0] for c in settled[-keep_newest:]} if keep_newest > 0 else set()

    delete, keep = [], []
    for epoch, path, size, age in ckpts:
        if age < min_age_seconds:
            keep.append((epoch, path, size, f"too new ({age:.0f}s, possibly mid-write)"))
        elif epoch in newest_epochs:
            keep.append((epoch, path, size, "newest (resume point)"))
        elif keep_every > 0 and epoch % keep_every == 0:
            keep.append((epoch, path, size, f"milestone (epoch %% {keep_every} == 0)"))
        else:
            delete.append((epoch, path, size))
    return delete, keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument(
        "--keep-newest",
        type=int,
        default=1,
        help="Keep this many newest checkpoints; the resume point (default: 1).",
    )
    ap.add_argument(
        "--keep-every",
        type=int,
        default=50,
        help="Also keep every Nth epoch as a milestone; 0 disables (default: 50).",
    )
    ap.add_argument(
        "--min-age-seconds",
        type=float,
        default=300.0,
        help="Never touch, or count as newest, a checkpoint younger than this "
        "(default: 300). An RFD3 checkpoint takes tens of seconds to write.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted and exit without deleting.",
    )
    args = ap.parse_args()

    tag = "[prune:dry-run]" if args.dry_run else "[prune]"

    if not args.ckpt_dir.is_dir():
        # A gate, not a failure: the train job calls this before the first checkpoint
        # exists, and a non-zero exit there would strand the run for no reason.
        print(f"{tag} no checkpoint directory yet: {args.ckpt_dir}")
        return 0

    if args.keep_newest < 1:
        print(
            f"{tag} REFUSING --keep-newest={args.keep_newest}: that would delete the "
            "resume point and force a restart from the released weights.",
            file=sys.stderr,
        )
        return 2

    ckpts = find_checkpoints(args.ckpt_dir)
    if not ckpts:
        print(f"{tag} no epoch-NNNN.ckpt files in {args.ckpt_dir}")
        return 0

    for name in warn_about_stray_files(args.ckpt_dir, {p.name for _, p, _, _ in ckpts}):
        print(
            f"{tag} WARNING: {name!r} sorts after the newest checkpoint. foundry "
            "resolves a ckpt DIRECTORY with sorted(iterdir())[-1] and does not filter "
            "for *.ckpt, so this file would be loaded INSTEAD of the checkpoint. "
            "Move it out of the checkpoint directory.",
            file=sys.stderr,
        )

    delete, keep = select_deletions(
        ckpts, args.keep_newest, args.keep_every, args.min_age_seconds
    )

    print(f"{tag} {args.ckpt_dir}")
    for epoch, path, size, reason in keep:
        print(f"{tag}   KEEP   {path.name}  {size / GB:5.2f} GB  {reason}")

    freed = 0
    for epoch, path, size in delete:
        if args.dry_run:
            print(f"{tag}   DELETE {path.name}  {size / GB:5.2f} GB")
            freed += size
            continue
        try:
            path.unlink()
        except OSError as exc:
            # One unlinkable file must not abort the sweep; the rest still frees space.
            print(f"{tag}   FAILED {path.name}: {exc}", file=sys.stderr)
            continue
        print(f"{tag}   DELETE {path.name}  {size / GB:5.2f} GB")
        freed += size

    verb = "would free" if args.dry_run else "freed"
    print(
        f"{tag} {len(keep)} kept, {len(delete)} deleted, {verb} {freed / GB:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
