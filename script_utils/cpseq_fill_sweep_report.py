#!/usr/bin/env python3
"""The filler sweep's readout: best-of-n at MATCHED n, not best-of-whatever-you-drew.

Why matched n is the whole point
--------------------------------
A best-of-N search spends only the tail, so the figure of merit is the best draw a
complex produced, not the median one. That makes the readout sensitive to N -- and the
arms here do NOT draw equal N from the same masks, for a reason intrinsic to what is
being compared:

  * A random filler picks residues uniformly, so a no-op needs every masked position to
    redraw the native residue. At mask sizes 1-5 that essentially never happens, and the
    arm keeps 12 draws from 12 masks.
  * ProteinMPNN near argmax (the locked T=0.2) reproduces the native residue often, and
    the no-op screen then drops the draw. Measured on this cohort: 8 of 12 kept at
    T=0.2 against 11 of 12 at T=1.0.

So the low-temperature arm is penalised twice by a raw comparison -- once for a narrow
tail, which is the real effect, and once for having fewer draws to take a minimum over,
which is bookkeeping. Re-reading the ladder pilot this way showed the second term is not
small: the raw gap between `llm_mask_mpnn_fill` (-6.56) and `llm_mask_random_fill`
(-10.04) is 3.48 REU, but at a matched n=6 it is 1.91 and at n=4 it is 1.19. The finding
survives; its size did not.

The same trap is about to be sprung by temperature itself, because raising T raises yield
as well as diversity. Crediting temperature with draws it merely retained would be the
same error in a new place, which is why this report exists rather than a median table.

Method
------
Per (arm, complex), collect every draw's median interface dG over its valid folds, minus
that complex's own reference median (per-complex normalisation, so interface size cannot
drive the comparison). Then for each n, estimate median best-of-n by subsampling n draws
without replacement and taking the minimum, and report the median across complexes.

Complexes with fewer than n surviving draws are DROPPED at that n, and the count is
printed next to every cell: a row whose complex count has collapsed is a different
cohort, not a deeper search, and the n=4 / n=6 rows are usually the trustworthy ones.

Run:  python script_utils/cpseq_fill_sweep_report.py --out-dir <run dir> [--arms a,b,c]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_GATE = "v3-cis-amide-muco-windows"
ENERGY_KEY = "refined_dG_separated"
DEFAULT_ARMS = (
    "mpnn_fill_t02", "mpnn_fill_t05", "mpnn_fill_t08", "mpnn_fill_t10",
    "shared_mask_random_fill",
)


def load_draws(out_dir: Path, gate: str, arms: tuple[str, ...]):
    """-> (draws[(arm, pid)] = [normalised dG, ...], n_reference_complexes)."""
    ref_seq: dict[str, str] = {}
    by_seq: dict[tuple, list] = collections.defaultdict(list)
    for journal in sorted((out_dir / "state").glob("*/outcomes.jsonl")):
        pid = journal.parent.name
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stage") != "ladder" or row.get("technical_failure"):
                continue
            if row.get("scoring_version") != gate:
                continue
            if row.get("arm") == "reference":
                ref_seq[pid] = row["sequence"]
            by_seq[(row.get("arm"), pid, row["sequence"])].append(row)

    ref_energy: dict[str, float] = {}
    for (arm, pid, _seq), rows in by_seq.items():
        if arm != "reference":
            continue
        vals = [r[ENERGY_KEY] for r in rows
                if r.get("valid") and r.get(ENERGY_KEY) is not None]
        if vals:
            ref_energy[pid] = st.median(vals)

    draws: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for (arm, pid, seq), rows in by_seq.items():
        if arm not in arms or pid not in ref_energy or pid not in ref_seq:
            continue
        if seq == ref_seq[pid]:
            continue
        vals = [r[ENERGY_KEY] for r in rows
                if r.get("valid") and r.get(ENERGY_KEY) is not None]
        if vals:
            draws[(arm, pid)].append(st.median(vals) - ref_energy[pid])
    return draws, len(ref_energy)


def best_of_n(vals: list[float], n: int, rng: random.Random, trials: int) -> float | None:
    """Median over `trials` subsamples of the best (most negative) of n draws.

    Without replacement, because a real best-of-n search spends n DISTINCT draws; drawing
    with replacement would let one lucky candidate be spent twice and would flatter every
    arm whose tail is thin -- the exact arms under test.
    """
    if len(vals) < n:
        return None
    if len(vals) == n:
        return min(vals)
    return st.median([min(rng.sample(vals, n)) for _ in range(trials)])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--gate", default=DEFAULT_GATE)
    ap.add_argument("--ns", default="2,4,6,8,10,12",
                    help="matched draw counts to report best-of-n at")
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    arms = tuple(a for a in args.arms.split(",") if a)
    ns = [int(x) for x in args.ns.split(",") if x]
    draws, n_ref = load_draws(args.out_dir, args.gate, arms)
    if not draws:
        print(f"No draws for {arms} under gate {args.gate} in {args.out_dir}.")
        print("Nothing folded yet, or the arm labels do not match. Not an error.")
        return 0

    present = [a for a in arms if any(k[0] == a for k in draws)]
    missing = [a for a in arms if a not in present]
    print(f"gate {args.gate}   {n_ref} complex(es) with a reference baseline")
    if missing:
        print(f"NOT YET FOLDED (absent from the journal): {', '.join(missing)}")

    print("\nYield -- draws that survived to at least one valid fold")
    print(f"  {'arm':<26} {'cplx':>5} {'median N':>9} {'min':>4} {'max':>4} {'total':>6}")
    for a in present:
        counts = [len(v) for (arm, _p), v in draws.items() if arm == a]
        print(f"  {a:<26} {len(counts):>5} {st.median(counts):>9.1f} "
              f"{min(counts):>4} {max(counts):>4} {sum(counts):>6}")

    print("\nMedian best-of-n across complexes, at MATCHED n "
          "(subsampled without replacement)")
    print("  a cell's (c) is how many complexes still had n draws; a collapsing c "
          "means\n  that row is a different cohort, not a deeper search")
    width = 22
    print("   n |" + "".join(f"{a[:width - 2]:>{width}}" for a in present))
    for n in ns:
        cells = []
        for a in present:
            rng = random.Random(args.seed + n)
            vals = [best_of_n(v, n, rng, args.trials)
                    for (arm, _p), v in sorted(draws.items()) if arm == a]
            vals = [v for v in vals if v is not None]
            cells.append(f"{st.median(vals):+7.2f} (c={len(vals):>2})" if vals
                         else f"{'-':>7}        ")
        print(f"  {n:>2} |" + "".join(f"{c:>{width}}" for c in cells))

    print("\nUnmatched, for contrast only -- every arm spends all the draws it has.")
    print("  This is the number that overstated the filler effect in the pilot.")
    for a in present:
        vals = [min(v) for (arm, _p), v in draws.items() if arm == a and v]
        counts = [len(v) for (arm, _p), v in draws.items() if arm == a and v]
        print(f"  {a:<26} median best {st.median(vals):+7.2f}   mean N={st.fmean(counts):.1f}")

    print("\nMedian TYPICAL draw -- the moment the filler was tuned for, and NOT the one")
    print("  a best-of-N search spends. Reported so a filler that wins here and loses")
    print("  above is visible as such rather than as a contradiction.")
    for a in present:
        pooled = [x for (arm, _p), v in draws.items() if arm == a for x in v]
        print(f"  {a:<26} median draw {st.median(pooled):+7.2f}   (n={len(pooled)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
