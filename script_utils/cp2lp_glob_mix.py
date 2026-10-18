"""Build the HELD-OUT two-role parquet the glob-dependence probe needs.

The probe scores the discriminator on length-matched real/generated pairs, so one batch has
to contain both roles. None of the existing splits does: `cp2lp_eval.parquet` is
118,140 cyclic rows and nothing else, `cp2lp_eval_real_lp.parquet` is 536 linear PepBench
rows and nothing else, and only `cp2lp_train.parquet` mixes them -- which is the split the
discriminator was trained on, so a sufficiency regression fitted there would be measuring
memorisation. This concatenates the two eval splits (identical schemas, receptor families
disjoint from train by construction in `cp2lp_build_splits.py`) into one mixed held-out
parquet.

THE ROW MIX IS THE POINT, not a convenience. 536 linear against 118,140 cyclic is 0.45%
linear, so at any realistic batch size almost every batch would be single-role, the probe
would skip it, and the run would burn a GPU hour to report `no scorable step`. The cyclic
side is therefore SUBSAMPLED to hit a target linear fraction. All 536 real LPs are kept --
they are the scarce class and the discriminator's real pool is PepBench only
(`cp2lp.real_lp_sources: [pepbench]`), so ProtFrag rows would not reach the pool anyway.

The cyclic subsample is stratified by peptide length. Length matching is exact
(`d_match_frac` 1.0), so a cyclic pool whose lengths do not cover the real LPs' lengths
would starve the reservoir and quietly shrink the measurement to whatever lengths happened
to overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

LINEAR = "linear"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cp-parquet", required=True, help="held-out cyclic rows (cp2lp_eval)")
    p.add_argument("--lp-parquet", required=True, help="held-out real LP rows (cp2lp_eval_real_lp)")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--linear-frac",
        type=float,
        default=0.25,
        help="target fraction of linear rows; the cyclic side is subsampled to hit it",
    )
    p.add_argument("--length-col", default="peptide_length")
    p.add_argument(
        "--no-stratify",
        action="store_true",
        help=(
            "sample cyclic rows uniformly instead of matching the real LPs' length "
            "histogram. The stratified default biases the cyclic pool toward short "
            "peptides (median 9), which is a confound for any edit-magnitude reading."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cp = pd.read_parquet(args.cp_parquet)
    lp = pd.read_parquet(args.lp_parquet)

    if "cyclization_type" not in cp.columns:
        logger.error("no `cyclization_type` column; cannot verify the roles")
        return 1
    n_lin_in_cp = int((cp["cyclization_type"] == LINEAR).sum())
    if n_lin_in_cp:
        logger.warning(f"{n_lin_in_cp} linear rows already in the cyclic split; dropping them")
        cp = cp[cp["cyclization_type"] != LINEAR]
    lp = lp[lp["cyclization_type"] == LINEAR]
    if lp.empty:
        logger.error(f"no linear rows in {args.lp_parquet}")
        return 1

    n_lp = len(lp)
    n_cp_target = max(1, int(round(n_lp * (1.0 - args.linear_frac) / args.linear_frac)))
    n_cp_target = min(n_cp_target, len(cp))

    length_col = args.length_col if args.length_col in cp.columns else None
    if args.no_stratify:
        length_col = None
        logger.info("--no-stratify: sampling cyclic rows uniformly over length")
    if length_col is None:
        logger.warning(
            f"no `{args.length_col}` column; subsampling cyclic rows uniformly. Length "
            "coverage is then unchecked and the reservoir may starve."
        )
        cp_keep = cp.sample(n=n_cp_target, random_state=args.seed)
    else:
        # Stratify on the real LPs' own length histogram, so every length the reservoir will
        # be asked to match has cyclic rows behind it.
        want = lp[length_col].value_counts(normalize=True)
        parts = []
        rng = np.random.default_rng(args.seed)
        for length, frac in want.items():
            pool = cp[cp[length_col] == length]
            k = int(round(frac * n_cp_target))
            if pool.empty or k == 0:
                continue
            parts.append(pool.sample(n=min(k, len(pool)), random_state=int(rng.integers(1 << 31))))
        cp_keep = (
            pd.concat(parts, ignore_index=True)
            if parts
            else cp.sample(n=n_cp_target, random_state=args.seed)
        )
        covered = set(cp_keep[length_col].unique())
        missing = sorted(set(lp[length_col].unique()) - covered)
        if missing:
            logger.warning(
                f"{len(missing)} real-LP lengths have no cyclic rows: {missing[:10]}. Those "
                "reals can never be matched; the probe will measure the rest."
            )

    mixed = pd.concat([cp_keep, lp], ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mixed.to_parquet(args.out, index=False)
    got = float((mixed["cyclization_type"] == LINEAR).mean())
    logger.info(
        f"wrote {args.out}: {len(mixed)} rows, {len(cp_keep)} cyclic + {n_lp} linear "
        f"({got:.1%} linear, target {args.linear_frac:.1%})"
    )
    if length_col is not None:
        logger.info(
            f"length range cyclic {int(cp_keep[length_col].min())}-{int(cp_keep[length_col].max())}, "
            f"linear {int(lp[length_col].min())}-{int(lp[length_col].max())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
