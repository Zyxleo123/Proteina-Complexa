"""Re-select the top-N designs of a completed run by closure gate + Rosetta interface energy.

The generation pipeline picks its final top-N by the *search* reward (AF2 ``i_pae`` and, once
wired, the closure reward). For a macrocycle that is the wrong instrument to *rank* by: AF2
refolds the binder linearly, so its interface confidence never sees the ring, and the closure
reward says nothing about binding. Rosetta ``dG_separated`` -- which is computed on the
generated complex and so *does* see the ring -- is the trustworthy ranking signal, but it only
exists after the evaluate/analyze step has run, i.e. after selection already happened.

This re-selection closes that gap. It reads the post-analysis RAW results CSV (which carries
both ``generated_binder_cyc_bond_closed`` and ``generated_binder_rosetta_dG_separated``),
drops designs whose ring did not close, and ranks the survivors by Rosetta dG (lower = tighter).
An open macrocycle is a failed design, so it never wins regardless of its Rosetta score.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

CLOSED_COL = "generated_binder_cyc_bond_closed"
ROSETTA_COL = "generated_binder_rosetta_dG_separated"


def reselect(
    df: pd.DataFrame,
    n_best: int,
    rosetta_col: str = ROSETTA_COL,
    closed_col: str = CLOSED_COL,
    gate_closed: bool = True,
) -> pd.DataFrame:
    """Return the top-``n_best`` rows: closed rings only, ranked by lowest Rosetta dG.

    Args:
        df: One row per generated design (a RAW analysis results frame).
        n_best: Number of designs to keep.
        rosetta_col: Column to rank by, ascending (lower dG = tighter binding = better).
        closed_col: Boolean-ish column; rows that are not truthy here are dropped when gating.
        gate_closed: If True, drop designs whose ring did not close before ranking. If False,
            rank every design by Rosetta alone (closure ignored).

    Returns:
        A frame of at most ``n_best`` rows, sorted best-first, with a ``reselect_rank`` column.
        Rows with a missing Rosetta score are dropped (they cannot be ranked).
    """
    if rosetta_col not in df.columns:
        raise KeyError(f"Ranking column {rosetta_col!r} not in results (has this run been analyzed?)")

    pool = df.copy()
    n_total = len(pool)

    n_open = 0
    if gate_closed:
        if closed_col not in pool.columns:
            raise KeyError(f"Gate column {closed_col!r} not in results; pass gate_closed=False to skip the gate")
        # .eq(True): NaN and any non-True value count as an open ring, no fillna downcast.
        closed_mask = pool[closed_col].eq(True)
        n_open = int((~closed_mask).sum())
        pool = pool[closed_mask]

    n_no_score = int(pool[rosetta_col].isna().sum())
    pool = pool.dropna(subset=[rosetta_col])

    pool = pool.sort_values(rosetta_col, ascending=True).head(n_best).reset_index(drop=True)
    pool.insert(0, "reselect_rank", range(1, len(pool) + 1))

    print(
        f"  reselect: {n_total} designs -> gated {n_open} open ring(s), "
        f"{n_no_score} missing Rosetta -> kept {len(pool)} (target {n_best})"
    )
    return pool


def _find_raw_csv(results_dir: str) -> str:
    hits = glob.glob(os.path.join(results_dir, "RAW_protein_binder_results_*combined.csv"))
    if not hits:
        raise FileNotFoundError(f"No RAW_protein_binder_results_*combined.csv in {results_dir}")
    return sorted(hits)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="A search_binder_local_pipeline_* results directory")
    ap.add_argument("--n-best", type=int, required=True, help="Designs to keep")
    ap.add_argument("--rosetta-col", default=ROSETTA_COL)
    ap.add_argument("--closed-col", default=CLOSED_COL)
    ap.add_argument("--no-gate", action="store_true", help="Rank by Rosetta only; ignore ring closure")
    ap.add_argument("--out", default=None, help="Output CSV (default: <results_dir>/reselected_top{N}.csv)")
    args = ap.parse_args()

    csv = _find_raw_csv(args.results_dir)
    df = pd.read_csv(csv)
    top = reselect(
        df,
        n_best=args.n_best,
        rosetta_col=args.rosetta_col,
        closed_col=args.closed_col,
        gate_closed=not args.no_gate,
    )
    out = args.out or os.path.join(args.results_dir, f"reselected_top{args.n_best}.csv")
    top.to_csv(out, index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
