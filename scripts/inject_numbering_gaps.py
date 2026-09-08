"""Chop long receptor segments by RENUMBERING ONLY -- not one atom moves or disappears.

Why this exists
---------------
`compute_segment_info` starts a new segment when EITHER the PDB numbering jumps OR the
C_{i-1}->N_i distance exceeds 2 A (segment_utils.py:121). Measured on the staged sets, the
segment COUNT is set entirely by the physical term -- erasing or adding numbering has no
effect on it, because a spatial crop leaves real backbone gaps.

What is still mismatched after the 18 A pocket crop (Arm B) is not the count but the RUN
LENGTH: median longest contiguous receptor run 51 residues, vs 28-31 on CPSea val. So the
denoiser sees `pos_in_segment` values it never saw in training even though the fragment
count now matches.

There are two ways to shorten those runs:
  * physically, by cropping tighter (14 A reproduces CPSea's segments/max_run/tgt_len all
    at once) -- but that also deletes real receptor atoms, so it moves three things;
  * by numbering alone, which is what this script does: it inserts numbering jumps inside
    runs longer than `--max-seg-len`, so the featurizer splits them, while the coordinates,
    the atom set and every other record stay byte-identical.

Running both separates "the model wants short segments as a FEATURE" from "the model wants
a smaller receptor". Note the honest caveat: this writes numbering that disagrees with the
geometry (residues 1.3 A apart are told they are in different fragments), which is a
featurization probe, not a physically meaningful structure. Do not stage designs from it.

Only columns 23-26 (resSeq) and 27 (iCode) of receptor ATOM/HETATM lines are touched. The
binder chain is passed through untouched, and so is every non-coordinate record.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

CN_BREAK_CUTOFF = 2.0  # matches DEFAULT_CN_BREAK_CUTOFF in segment_utils.py


def residue_key(line: str):
    return (int(line[22:26]), line[26])


def read_residues(path: Path, binder_chain: str):
    """-> OrderedDict[(chain, resseq, icode) -> [line, ...]] for receptor coordinate lines."""
    res = OrderedDict()
    for line in path.read_text().splitlines(keepends=True):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[21] == binder_chain:
            continue
        res.setdefault((line[21],) + residue_key(line), []).append(line)
    return res


def backbone_atom(lines, name):
    for l in lines:
        if l[12:16].strip() == name:
            return np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return None


def is_broken(prev_lines, curr_lines) -> bool:
    """True iff the featurizer would call this a physical break."""
    c, n = backbone_atom(prev_lines, "C"), backbone_atom(curr_lines, "N")
    if c is None or n is None:
        return True
    return float(np.linalg.norm(c - n)) > CN_BREAK_CUTOFF


def run_lengths(keys, res):
    """Segment run lengths under the real rule (physical break OR numbering jump)."""
    runs, run = [], 1
    for i in range(1, len(keys)):
        same_chain = keys[i][0] == keys[i - 1][0]
        num_jump = keys[i][1] != keys[i - 1][1] + 1
        if (not same_chain) or num_jump or is_broken(res[keys[i - 1]], res[keys[i]]):
            runs.append(run)
            run = 1
        else:
            run += 1
    runs.append(run)
    return runs


def plan_numbering(keys, res, max_seg_len: int, gap: int):
    """-> {old key -> new resseq}. Existing breaks are preserved; runs longer than
    max_seg_len get extra numbering jumps inserted at regular intervals."""
    mapping, run, offset = {}, 0, 0
    prev_out = None
    for i, k in enumerate(keys):
        if i == 0:
            new = k[1]
            run = 1
        else:
            same_chain = k[0] == keys[i - 1][0]
            broke = (not same_chain) or k[1] != keys[i - 1][1] + 1 or is_broken(res[keys[i - 1]], res[k])
            if not same_chain:
                offset, prev_out, run = 0, None, 1
                new = k[1]
            elif broke:
                # Keep the existing break. Numbering must stay strictly increasing, so if the
                # source jump has been swallowed by an earlier injected offset, grow the
                # offset instead of emitting a number that would collide.
                run = 1
                if k[1] + offset <= prev_out:
                    offset += prev_out + gap - (k[1] + offset)
                new = k[1] + offset
            elif run >= max_seg_len:
                # Force a new segment purely through numbering.
                offset += gap
                run = 1
                new = k[1] + offset
            else:
                run += 1
                new = k[1] + offset
        mapping[k] = new
        prev_out = new
    return mapping


def rewrite(path: Path, dst: Path, binder_chain: str, mapping, note: str) -> int:
    n_touched = 0
    out = []
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")) and line[21] != binder_chain:
            k = (line[21],) + residue_key(line)
            new = mapping.get(k)
            if new is not None:
                line = f"{line[:22]}{new:4d} {line[27:]}"
                n_touched += 1
        out.append(line)
    out.insert(0, note)
    dst.write_text("".join(out))
    return n_touched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-metadata", required=True, help="Staged parquet whose `path` column points at the PDBs.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-name", default="lnr_test_gapped.parquet")
    ap.add_argument("--max-seg-len", type=int, default=31,
                    help="Longest receptor run to leave intact. Default 31 = CPSea val's "
                         "median max_pos_in_segment.")
    ap.add_argument("--gap", type=int, default=30, help="Size of each injected numbering jump.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pdb_out = out_dir / "pdbs"
    pdb_out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(args.in_metadata)
    rows, before, after, seg_before, seg_after = [], [], [], [], []
    for _, r in meta.iterrows():
        src = Path(str(r["path"]))
        binder = str(r["binder_chain_id"])
        res = read_residues(src, binder)
        keys = list(res)
        if not keys:
            raise SystemExit(f"no receptor residues in {src}")

        runs0 = run_lengths(keys, res)
        mapping = plan_numbering(keys, res, args.max_seg_len, args.gap)
        dst = pdb_out / src.name
        note = (f"REMARK 999 GAP-MORE ARM: receptor RENUMBERED ONLY to cap segment runs at "
                f"{args.max_seg_len}; coordinates and atom set identical to source\n")
        rewrite(src, dst, binder, mapping, note)

        res2 = read_residues(dst, binder)
        runs1 = run_lengths(list(res2), res2)
        if sum(runs0) != sum(runs1):
            raise SystemExit(f"residue count changed for {src.name}: {sum(runs0)} -> {sum(runs1)}")
        if max(runs1) > args.max_seg_len:
            raise SystemExit(f"{src.name}: run of {max(runs1)} survived (cap {args.max_seg_len})")
        before.append(max(runs0)); after.append(max(runs1))
        seg_before.append(len(runs0)); seg_after.append(len(runs1))

        row = r.to_dict()
        row["path"] = str(dst.resolve())
        rows.append(row)

    out_meta = out_dir / "metadata"
    out_meta.mkdir(parents=True, exist_ok=True)
    dest = out_meta / args.out_name
    pd.DataFrame(rows).to_parquet(dest, index=False)
    print(f"renumbered {len(rows)} examples -> {dest}")
    print(f"max run       median {np.median(before):.1f} -> {np.median(after):.1f}")
    print(f"segments      median {np.median(seg_before):.1f} -> {np.median(seg_after):.1f}")
    print("CPSea val reference: segments ~14, max run ~31")


if __name__ == "__main__":
    main()
