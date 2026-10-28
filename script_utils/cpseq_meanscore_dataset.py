#!/usr/bin/env python3
"""Build a mean-score regression table from ONE run directory, read off folds on disk.

The label this emits is the per-sequence mean of `refined_dG_separated` over that
sequence's seeds, with each fold clipped at 0 REU first. Every part of that sentence was
forced by a measurement, and `cpseq_meanscore_reliability.py` reproduces all of them:

1. **One run directory, one scoring version.** Pooling runs pools scoring protocols, and
   v3/v4/v5 refine the junction differently, so a pooled label is a mixture of protocols
   rather than a quantity. `--out-dir` takes one run and `--scoring-version` one version.

2. **Clip each fold at 0 REU before averaging.** 0.38 % of folds come back from
   refinement at up to +30,351 REU, and 10 of those pass the validity checks, so the
   validity gate does not remove them. The unclipped mean has split-half reliability
   **-0.02**: it is a readout of which sequence happened to draw a blow-up. Clipping
   lifts that to **0.83**. The median scores 0.81 and is the project convention for
   reporting energy (README_CPSEQ_PEPBENCH_SCALEUP sec. 2); `--agg median` selects it.

3. **The mean, not the minimum.** A minimum over more folds is lower whether or not the
   sequences differ, which is what withdrew the budget ladder's energy column.

4. **Sequences with few folds are kept but flagged.** `n_folds` and `label_se` are
   emitted per row so a consumer can weight or filter; `--min-folds` drops the rest.

The per-fold values are written out alongside the label. A Gaussian-NLL model that
supervises individual folds needs them, and so does any later variance head.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_meanscore_dataset.py \
        --out-dir /data/user_data/yixiz/cpseq/pepbench_corpus_pot \
        --json /data/user_data/yixiz/cpseq/meanscore/dataset_v3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

STAGE = "ladder"
DRAW_ARM = "random_control"
REFERENCE_ARM = "reference"
# The energy field the label is built from. `refined_dG_separated` is the post-refinement
# interface energy, which is the quantity the screening objective already ranks by.
ENERGY_FIELD = "refined_dG_separated"
# Folds above this are refinement blow-ups, not weak binders. See rule 2 in the docstring.
CLIP_AT = 0.0

# Carried through to each row unchanged. These identify the target and the edit, and are
# constant across a sequence's folds, so the first fold decides them.
SEQ_KEYS = (
    "parent_id",
    "pdb_id",
    "family_id",
    "split",
    "arm",
    "sequence",
    "original_sequence",
    "peptide_length",
    "n_mutations",
    "mutation_fraction",
    "mutated_positions",
    "identity_to_original",
    "chemistry",
    "proposal_kind",
)
# Averaged over the same folds as the label, as auxiliary regression targets. Boltz-2's
# own confidence is in here because it is free at generation time, which makes it both a
# candidate feature and the baseline any learned scorer has to beat.
AUX_FIELDS = (
    "refined_dSASA_int",
    "refined_dG_separated_per_1000A2",
    "refined_delta_unsatHbonds",
    "refined_hbonds_int",
    "refined_sc_value",
    "refined_packstat",
    "local_deformation_energy_proxy",
)


def load_folds(out_dir: Path, scoring_version: str, stage: str):
    """Group scored folds by (parent_id, sequence). Returns dict -> list of outcome rows."""
    grouped: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    skipped = collections.Counter()
    state = out_dir / "state"
    if not state.is_dir():
        raise SystemExit(f"no state/ directory under {out_dir}")
    for path in sorted(state.glob("*/outcomes.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("scoring_version") != scoring_version:
                    skipped["other_scoring_version"] += 1
                    continue
                if stage and row.get("stage") != stage:
                    skipped["other_stage"] += 1
                    continue
                if row.get("status") != "scored":
                    skipped[f"status_{row.get('status')}"] += 1
                    continue
                if row.get(ENERGY_FIELD) is None:
                    skipped["no_energy"] += 1
                    continue
                grouped[(row["parent_id"], row["sequence"])].append(row)
    return grouped, skipped


def aggregate(folds: list[dict], agg: str) -> dict:
    """One row: the label, its standard error, and the per-fold values behind it."""
    folds = sorted(folds, key=lambda r: r.get("prediction_seed") or 0)
    raw = [float(r[ENERGY_FIELD]) for r in folds]
    clipped = [min(v, CLIP_AT) for v in raw]
    valid = [bool(r.get("valid")) for r in folds]
    n = len(clipped)

    label = st.mean(clipped) if agg == "mean" else st.median(clipped)
    # SE of the mean. For the median this understates the spread slightly, but it is the
    # same monotone function of the fold scatter and is only used for weighting.
    se = (st.stdev(clipped) / n**0.5) if n > 1 else None

    row = {k: folds[0].get(k) for k in SEQ_KEYS}
    row.update(
        n_folds=n,
        n_valid_folds=sum(valid),
        valid_rate=sum(valid) / n,
        label_agg=agg,
        label=label,
        label_clipped_mean=st.mean(clipped),
        label_clipped_median=st.median(clipped),
        label_sd=st.stdev(clipped) if n > 1 else None,
        label_se=se,
        n_folds_clipped=sum(1 for a, b in zip(raw, clipped) if a != b),
        seeds=[r.get("prediction_seed") for r in folds],
        fold_dG=raw,
        fold_dG_clipped=clipped,
        fold_valid=valid,
    )
    for field in AUX_FIELDS:
        vals = [float(r[field]) for r in folds if r.get(field) is not None]
        row[f"mean_{field}"] = st.mean(vals) if vals else None
    # Confidence lives in a nested dict on the score record but is flattened onto the
    # outcome rows; tolerate either shape.
    for field in ("iptm", "complex_plddt", "complex_iplddt", "confidence_score"):
        vals = [
            float(r[field]) for r in folds if isinstance(r.get(field), (int, float))
        ]
        row[f"mean_{field}"] = st.mean(vals) if vals else None
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="ONE run directory")
    ap.add_argument("--scoring-version", default="v3-cis-amide-muco-windows")
    ap.add_argument("--stage", default=STAGE, help="'' to accept every stage")
    ap.add_argument("--agg", choices=("mean", "median"), default="mean")
    ap.add_argument("--min-folds", type=int, default=8)
    ap.add_argument("--arms", default=f"{DRAW_ARM},{REFERENCE_ARM}")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()

    arms = {a for a in args.arms.split(",") if a}
    grouped, skipped = load_folds(args.out_dir, args.scoring_version, args.stage)

    rows, dropped = [], collections.Counter()
    for (parent, _seq), folds in sorted(grouped.items()):
        if folds[0].get("arm") not in arms:
            dropped["other_arm"] += 1
            continue
        if len(folds) < args.min_folds:
            dropped[f"under_{args.min_folds}_folds"] += 1
            continue
        rows.append(aggregate(folds, args.agg))

    targets = sorted({r["parent_id"] for r in rows})
    per_target = collections.Counter(r["parent_id"] for r in rows)
    by_arm = collections.Counter(r["arm"] for r in rows)

    payload = {
        "source_out_dir": str(args.out_dir),
        "scoring_version": args.scoring_version,
        "stage": args.stage,
        "energy_field": ENERGY_FIELD,
        "energy_units": "Rosetta ref2015 REU (NOT a measured binding free energy)",
        "clip_at": CLIP_AT,
        "label": f"per-sequence {args.agg} of {ENERGY_FIELD} clipped at {CLIP_AT} REU",
        "min_folds": args.min_folds,
        "n_rows": len(rows),
        "n_targets": len(targets),
        "n_folds_total": sum(r["n_folds"] for r in rows),
        "rows_by_arm": dict(by_arm),
        "sequences_per_target": dict(collections.Counter(per_target.values())),
        "skipped_folds": dict(skipped),
        "dropped_sequences": dict(dropped),
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1))

    print(f"source      {args.out_dir}")
    print(f"scoring     {args.scoring_version}  stage={args.stage or 'any'}")
    print(f"label       {payload['label']}")
    print(f"rows        {len(rows)} sequences over {len(targets)} targets, {payload['n_folds_total']} folds")
    print(f"by arm      {dict(by_arm)}")
    print(f"dropped     {dict(dropped)}")
    print(f"skipped     {dict(skipped)}")
    clipped_rows = sum(1 for r in rows if r["n_folds_clipped"])
    print(f"clipping    {sum(r['n_folds_clipped'] for r in rows)} folds in {clipped_rows} sequences hit the {CLIP_AT} REU clip")
    print(f"wrote       {args.json}")


if __name__ == "__main__":
    main()
