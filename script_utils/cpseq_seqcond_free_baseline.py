#!/usr/bin/env python3
"""The bar `val_gen_seqcond/seq/free_recovery` has to clear on the LNR tranche.

`free_recovery` is the model's agreement with the native residue at positions it was NOT
shown. On its own that number means nothing: CPSea cyclises terminally and the residue
distribution is skewed, so a lookup table that has never seen a pocket already scores well
above chance. README_CPSEQ_MLM_PROPOSER.md records a sibling arm landing at 9.77% against
a 9.7% always-guess-leucine prior, which read as learning until the prior was computed.

Two priors, both fitted on CPSea and both evaluated on the LNR peptides:

  marginal    always guess the commonest residue.
  positional  argmax of P(residue | position, length), the parameter-free table. This is
              the one that matters, and it is the bar the training run itself reports
              (`val/baseline_positional_acc`).

Evaluated over ALL positions of each LNR peptide, not a sampled subset. The eval hides a
uniform random subset, so the all-positions rate is the expectation of the hidden-position
rate and needs no matching draw -- which also keeps this independent of the eval's seed.

A fallback matters here: `lnr_1elr` is a 5-mer and CPSea has 2 of those in a 4200-row
shard, so P(. | position, length=5) is fitted on almost nothing. Cells under
`--min-count` fall back to the length-marginal and then to the global marginal, and the
fallback rate is reported, because a prior that is mostly fallback is a marginal wearing
a hat.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def fit(rows: list[str], min_count: int):
    by_pos_len: dict[tuple[int, int], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    by_len: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    overall: collections.Counter = collections.Counter()
    for seq in rows:
        L = len(seq)
        for i, a in enumerate(seq):
            if a not in AA20:
                continue
            by_pos_len[(i, L)][a] += 1
            by_len[L][a] += 1
            overall[a] += 1

    def predict(i: int, L: int) -> tuple[str, str]:
        cell = by_pos_len.get((i, L))
        if cell and sum(cell.values()) >= min_count:
            return cell.most_common(1)[0][0], "positional"
        cell = by_len.get(L)
        if cell and sum(cell.values()) >= min_count:
            return cell.most_common(1)[0][0], "length_marginal"
        return overall.most_common(1)[0][0], "global_marginal"

    return predict, overall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpsea-shard", required=True,
                    help="a corpus shard, used only to FIT the priors")
    ap.add_argument("--lnr-parquet", required=True,
                    help="the tranche the priors are EVALUATED on")
    ap.add_argument("--parents-csv",
                    default="/data/user_data/yixiz/cpseq/tranche1/manifest/parents.csv",
                    help="fallback source of reference sequences, joined on cluster_id")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd

    train_seqs = [
        json.loads(line)["pep_seq"] for line in Path(args.cpsea_shard).read_text().splitlines()
    ]
    predict, overall = fit(train_seqs, args.min_count)
    marginal_residue = overall.most_common(1)[0][0]
    print(f"priors fitted on {len(train_seqs)} CPSea peptides; "
          f"commonest residue {marginal_residue}")

    df = pd.read_parquet(args.lnr_parquet)
    if "pep_seq" in df.columns:
        seqs = {r["example_id"]: r["pep_seq"] for _, r in df.iterrows()}
    else:
        # The staging parquet carries structure paths, not sequences. Join to the
        # tranche manifest on `cluster_id` ('2v8x' -> 'lnr_2v8x'), which is the same
        # `reference_sequence` the search treats as the original -- so the prior is
        # scored against exactly the peptide the model is asked to recover. A missing
        # parent is fatal: silently scoring 11 of 12 would move the bar.
        import csv as _csv

        manifest = {
            r["parent_id"]: r["reference_sequence"]
            for r in _csv.DictReader(open(args.parents_csv))
        }
        seqs = {}
        missing = []
        for _, r in df.iterrows():
            key = f"lnr_{r['cluster_id']}"
            if key not in manifest:
                missing.append(key)
                continue
            seqs[r["example_id"]] = manifest[key]
        if missing:
            raise SystemExit(
                f"no reference sequence in {args.parents_csv} for {missing}; the bar "
                f"would be computed over a different set than the model was run on."
            )
        print(f"sequences joined from {args.parents_csv} on cluster_id")

    pos_hit = marg_hit = total = 0
    sources: collections.Counter = collections.Counter()
    per_parent = []
    for pid, seq in seqs.items():
        ph = mh = 0
        for i, a in enumerate(seq):
            if a not in AA20:
                continue
            pred, src = predict(i, len(seq))
            sources[src] += 1
            ph += pred == a
            mh += marginal_residue == a
            total += 1
        pos_hit += ph
        marg_hit += mh
        per_parent.append({
            "example_id": pid, "length": len(seq),
            "positional_hits": ph, "marginal_hits": mh,
        })
        print(f"  {pid:18s} L={len(seq):2d}  positional {ph:2d}/{len(seq)}  "
              f"marginal {mh:2d}/{len(seq)}")

    out = {
        "n_positions": total,
        "positional_argmax_acc": round(pos_hit / total, 4),
        "marginal_argmax_acc": round(marg_hit / total, 4),
        "marginal_residue": marginal_residue,
        "prior_source_counts": dict(sources),
        "fallback_frac": round(
            1 - sources["positional"] / max(total, 1), 4
        ),
        "fitted_on": args.cpsea_shard,
        "evaluated_on": args.lnr_parquet,
        "min_count": args.min_count,
        "per_parent": per_parent,
        "note": (
            "Evaluated over ALL positions; the eval hides a uniform random subset, so "
            "this is the expectation of the hidden-position rate. free_recovery must "
            "clear positional_argmax_acc, not marginal_argmax_acc."
        ),
    }
    print(f"\npositional prior {out['positional_argmax_acc']:.4f}  "
          f"marginal prior {out['marginal_argmax_acc']:.4f}  "
          f"over {total} positions")
    print(f"prior cells by source: {dict(sources)} "
          f"(fallback {out['fallback_frac']:.2%})")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
