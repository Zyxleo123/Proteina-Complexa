"""Build the CP->LP train/eval metadata with receptor families kept apart.

Three outputs from one mixed metadata frame:

``cp2lp_train.parquet``
    training families. CPSea head-to-tail cyclic peptides (the generator inputs) plus the
    linear rows (the discriminator's real pool and the flow-matching anchor).

``cp2lp_eval.parquet``
    eval families, CPSea head-to-tail only. These are the source CPs that triplets get
    generated for -- never seen in training, by receptor family.

``cp2lp_eval_real_lp.parquet``
    eval families, PepBench only. The report's real-side reference, measured by the same
    code that measures the generated peptides.

WHY HEAD-TO-TAIL ONLY on the CP side. It is the chemistry whose linear precursor is
unambiguous: delete the backbone bond between residue L-1 and residue 0 and what is left
is exactly the same sequence as an open chain. A disulfide or isopeptide macrocycle
linearises by breaking a SIDE-CHAIN bond, which leaves the backbone already linear and the
"LP" differing from the CP only in two side chains -- a different and much easier problem
that would quietly dominate the statistics if it were mixed in.

WHY FAMILIES AND NOT ROWS. CPSea clusters are receptor-level, and two rows in one cluster
are effectively the same binding site. Splitting by row would put near-duplicates on both
sides and make "held-out" mean nothing. The assignment is a hash of the family id, so it
is identical across processes and reruns, and stable when the metadata gains rows.

Run:
    .venv/bin/python script_utils/cp2lp_build_splits.py \\
        --mixed $LPDATA/metadata_mixed/mixed_train.parquet \\
        --out-dir $LPDATA/metadata_mixed --eval-frac 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cp2lp.data import family_of, in_eval_split  # noqa: E402

#: CPSea's label for a head-to-tail (backbone) macrocycle.
HEAD_TAIL = "head_tail"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mixed", required=True, help="mixed_train.parquet from the LP-mixing preprocess.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--real-lp-sources", default="pepbench",
                   help="Comma-separated sources counted as REAL bound LPs (the discriminator's pool).")
    p.add_argument("--flow-lp-sources", default="pepbench,protfrag",
                   help="Sources kept for the real-LP flow-matching anchor. ProtFrag fragments come "
                        "from monomeric contexts, so they anchor LP GEOMETRY but are not bound-peptide "
                        "examples -- which is why the two lists differ by default.")
    p.add_argument("--max-cp-train", type=int, default=-1,
                   help="Optional cap on CPSea training rows (-1 = all). The pool is 776k; a cap makes "
                        "an epoch a meaningful unit for a pilot.")
    p.add_argument("--prefix", default="cp2lp")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.mixed)
    logger.info(f"read {len(df):,} rows from {args.mixed}")

    real_sources = [s.strip() for s in args.real_lp_sources.split(",") if s.strip()]
    flow_sources = [s.strip() for s in args.flow_lp_sources.split(",") if s.strip()]
    missing = [s for s in set(real_sources + flow_sources) if s not in set(df["dataset_source"].unique())]
    if missing:
        raise SystemExit(f"sources {missing} are not in the metadata; have {sorted(df.dataset_source.unique())}")

    cp = df[(df.dataset_source == "cpsea") & (df.cyclization_type == HEAD_TAIL)].copy()
    lp = df[df.dataset_source.isin(flow_sources)].copy()
    logger.info(f"head-to-tail CPs: {len(cp):,}   linear rows: {len(lp):,} ({flow_sources})")
    if cp.empty or lp.empty:
        raise SystemExit("one side of the split is empty -- check --mixed and the source names.")

    def assign(frame: pd.DataFrame) -> pd.Series:
        fams = frame.apply(lambda r: family_of(r.to_dict()), axis=1)
        return fams, fams.map(lambda f: in_eval_split(f, args.eval_frac, args.seed))

    cp_fams, cp_eval = assign(cp)
    lp_fams, lp_eval = assign(lp)

    # Families are assigned by the same hash on both sides, so a family that appears in
    # both CPSea and PepBench lands on the same side of the split rather than straddling it.
    leak = set(cp_fams[~cp_eval]) & set(cp_fams[cp_eval])
    assert not leak, f"CP family leak: {len(leak)}"
    leak = set(lp_fams[~lp_eval]) & set(lp_fams[lp_eval])
    assert not leak, f"LP family leak: {len(leak)}"
    cross = set(cp_fams[~cp_eval]) & set(cp_fams[cp_eval]) | set(lp_fams[~lp_eval]) & set(cp_fams[cp_eval])
    assert not cross, f"cross-kind family leak: {len(cross)}"

    cp_train, cp_eval_df = cp[~cp_eval], cp[cp_eval]
    lp_train = lp[~lp_eval]
    lp_eval_real = lp[lp_eval & lp.dataset_source.isin(real_sources)]

    if args.max_cp_train > 0 and len(cp_train) > args.max_cp_train:
        cp_train = cp_train.sample(n=args.max_cp_train, random_state=args.seed)
        logger.info(f"capped CPSea training rows to {len(cp_train):,}")

    train = pd.concat([cp_train, lp_train], ignore_index=True)
    paths = {
        "train": out_dir / f"{args.prefix}_train.parquet",
        "eval": out_dir / f"{args.prefix}_eval.parquet",
        "eval_real_lp": out_dir / f"{args.prefix}_eval_real_lp.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    cp_eval_df.to_parquet(paths["eval"], index=False)
    lp_eval_real.to_parquet(paths["eval_real_lp"], index=False)

    manifest = {
        "source": str(args.mixed),
        "eval_frac": args.eval_frac,
        "seed": args.seed,
        "cyclization_type": HEAD_TAIL,
        "real_lp_sources": real_sources,
        "flow_lp_sources": flow_sources,
        "counts": {
            "train_total": len(train),
            "train_cp": len(cp_train),
            "train_lp": len(lp_train),
            "train_lp_by_source": lp_train.dataset_source.value_counts().to_dict(),
            "eval_cp": len(cp_eval_df),
            "eval_real_lp": len(lp_eval_real),
        },
        "families": {
            "train_cp": int(cp_fams[~cp_eval].nunique()),
            "eval_cp": int(cp_fams[cp_eval].nunique()),
            "train_lp": int(lp_fams[~lp_eval].nunique()),
            "eval_lp": int(lp_fams[lp_eval].nunique()),
        },
        "length_range": {
            "cp": [int(cp.peptide_length.min()), int(cp.peptide_length.max())],
            "lp": [int(lp.peptide_length.min()), int(lp.peptide_length.max())],
        },
        "paths": {k: str(v) for k, v in paths.items()},
    }
    (out_dir / f"{args.prefix}_split_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(json.dumps(manifest["counts"], indent=2))

    # The discriminator matches real to generated BY LENGTH. A generated length with no
    # real counterpart can never be scored, so the overlap is a hard precondition, not a
    # nice-to-have -- report it here rather than discovering it as a silent zero later.
    cp_lengths = set(cp_train.peptide_length.unique())
    lp_lengths = set(lp_train[lp_train.dataset_source.isin(real_sources)].peptide_length.unique())
    uncovered = sorted(cp_lengths - lp_lengths)
    if uncovered:
        logger.warning(
            f"peptide lengths present in CP training rows but absent from the real-LP pool "
            f"{real_sources}: {uncovered}. Generated samples at these lengths get no "
            f"discriminator gradient (watch cp2lp_d_match_frac)."
        )
    else:
        logger.info(f"every CP training length {sorted(cp_lengths)} has a real-LP counterpart.")

    logger.info(f"wrote {paths['train']}, {paths['eval']}, {paths['eval_real_lp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
