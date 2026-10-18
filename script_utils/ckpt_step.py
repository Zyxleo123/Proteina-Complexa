"""Print the training step stored in a Lightning checkpoint, without loading it.

Picking "the latest checkpoint" by filename is wrong here. Lightning writes
``last.ckpt`` and then ``last-v1.ckpt``, ``last-v2.ckpt`` ... on requeue, but
the suffix order does not track time: in ``cpsea_seqcond_v2`` the bare
``last.ckpt`` (step 2400) is newer than ``last-v1.ckpt`` (step 1500), and only
``last-v3.ckpt`` holds the furthest state at step 3000. Sorting by name picks
step 2400; sorting by mtime disagrees with itself across a checkpoint and its
EMA twin.

``torch.load`` would answer this, but a CPSea checkpoint is ~2.9 GB and the
step is a scalar sitting in the zip's pickle header. So read just that: open
the ``.ckpt`` as the zip archive it is, pull out ``data.pkl`` (a few hundred
KB), disassemble it and read the integer that follows the key.

Usage:
    python script_utils/ckpt_step.py <ckpt> [<ckpt> ...]   # "<step>\t<path>"
    python script_utils/ckpt_step.py --max <ckpt> ...      # best path only
"""

from __future__ import annotations

import argparse
import io
import pickletools
import re
import sys
import zipfile


def read_meta(path: str) -> dict[str, int | None]:
    """Return {'epoch': int|None, 'global_step': int|None} for a checkpoint."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("data.pkl")]
        if not names:
            raise ValueError("no data.pkl inside; not a Lightning checkpoint")
        raw = z.read(names[0])

    out = io.StringIO()
    pickletools.dis(io.BytesIO(raw), out=out)
    text = out.getvalue()

    meta: dict[str, int | None] = {}
    for key in ("epoch", "global_step"):
        # The keys sit at the top of the pickle as BINUNICODE followed by a
        # BINPUT and then the integer opcode, whose width varies with the
        # value (BININT1 / BININT2 / BININT).
        i = text.find(f"BINUNICODE '{key}'")
        if i < 0:
            meta[key] = None
            continue
        m = re.search(r"BININT\d?\s+(\d+)", text[i : i + 200])
        meta[key] = int(m.group(1)) if m else None
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument(
        "--max",
        action="store_true",
        help="print only the path with the highest global_step",
    )
    args = ap.parse_args(argv)

    rows: list[tuple[int, str]] = []
    for p in args.ckpt:
        try:
            step = read_meta(p).get("global_step")
        except Exception as exc:  # unreadable or truncated
            print(f"ERR\t{p}\t{type(exc).__name__}", file=sys.stderr)
            continue
        if step is None:
            print(f"ERR\t{p}\tno global_step", file=sys.stderr)
            continue
        rows.append((step, p))

    if not rows:
        return 1

    if args.max:
        # Ties broken by name so the choice is reproducible across runs.
        print(max(rows, key=lambda r: (r[0], r[1]))[1])
    else:
        for step, p in sorted(rows):
            print(f"{step}\t{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
