"""Build the parent/candidate tables for the sequence -> CP outcome dataset.

Order matters here and is enforced by the script's structure: parent contexts are parsed,
receptors are clustered into families, the **split is assigned**, and only then are
candidate sequences generated. Generating variants first and splitting afterwards is how
a mutant ends up on the other side of a boundary from its own reference.

Outputs (under `--out-dir`):

    manifest/
      parents.csv              one row per parent context, with family + split
      candidates.csv           one row per candidate sequence
      capability_table.csv     (tool, chemistry) -> supported / linear_only / unsupported
      chemistry_spec.json      atom-level bonds, residue requirements, terminal states
      rejected_parents.csv     every LNR entry dropped, with the measured reason
      build_config.json        resolved settings + config hash, for reproducibility
      clustering_method.json   which clustering path ran and at what thresholds

Nothing here needs a GPU or a predictor; it is the dataset skeleton every arm consumes.

Usage:
    python script_utils/cpseq_build_dataset.py \
        --lnr-root /data/user_data/yixiz/LNR \
        --out-dir /data/user_data/yixiz/cpseq/tranche1 \
        --n-parents 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import chemistry as chem  # noqa: E402
from proteinfoundation.cpseq.candidates import (  # noqa: E402
    DEFAULT_BANDS,
    DEFAULT_VARIANTS_PER_BAND,
    build_candidates,
    identity_to_reference,
)
from proteinfoundation.cpseq.contexts import (  # noqa: E402
    POCKET_RADIUS_A,
    assign_families,
    load_lnr_contexts,
    split_by_family,
)


def select_diverse_parents(contexts, families, n: int, seed: int = 42):
    """`n` parents, at most one per receptor family, spread over peptide length.

    Length is the axis the ring chemistry is most sensitive to (a 5-mer head-to-tail ring
    is a strained molecule; a 16-mer is not), so the selection walks the length-sorted
    list in even strides rather than sampling uniformly, and never takes two parents from
    one family -- two deposits of the same receptor would spend a slot without adding a
    target.
    """
    seen_family: set[str] = set()
    pool = []
    for ctx in sorted(contexts, key=lambda c: (c.peptide_length, c.parent_id)):
        fam = families[ctx.parent_id]
        if fam in seen_family:
            continue
        seen_family.add(fam)
        pool.append(ctx)
    if n >= len(pool):
        return pool
    # Even strides through the length-sorted pool.
    idx = [round(k * (len(pool) - 1) / (n - 1)) for k in range(n)] if n > 1 else [0]
    out, used = [], set()
    for i in idx:
        while i in used:
            i = (i + 1) % len(pool)
        used.add(i)
        out.append(pool[i])
    return sorted(out, key=lambda c: (c.peptide_length, c.parent_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lnr-root", type=Path, default=Path("/data/user_data/yixiz/LNR"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-parents", type=int, default=12)
    ap.add_argument(
        "--chemistries",
        default="",
        help="comma-separated; default is every chemistry all predictors AND the scorer "
        "support (the cross-model set), plus any chemistry supported by a subset, which "
        "is emitted as its own labeled subset",
    )
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mmseqs-exec", default=None)
    args = ap.parse_args()

    out = args.out_dir
    man = out / "manifest"
    man.mkdir(parents=True, exist_ok=True)

    chem.assert_matches_project_constants()

    # --- chemistry selection -------------------------------------------------------
    cross_model = chem.common_chemistries()  # predictors + scorer
    if args.chemistries:
        chemistries = [c.strip() for c in args.chemistries.split(",") if c.strip()]
    else:
        # The cross-model set, plus chemistries that at least one predictor and the
        # scorer support -- those become separately labeled subsets, never pooled into
        # the model comparison.
        extra = [
            name
            for name in chem.CHEMISTRIES
            if name not in cross_model
            and chem.capability("rosetta", name).status == chem.SUPPORTED
            and any(
                chem.capability(p, name).status == chem.SUPPORTED for p in chem.PREDICTORS
            )
        ]
        chemistries = cross_model + extra
    print(f"[cpseq] cross-model chemistry set: {cross_model}")
    print(f"[cpseq] building chemistries:      {chemistries}")

    # --- parents, families, split (BEFORE variants) ---------------------------------
    contexts, rejected = load_lnr_contexts(args.lnr_root)
    print(f"[cpseq] LNR: {len(contexts)} parent contexts, {len(rejected)} rejected")

    families = assign_families(
        contexts, man, args.mmseqs_exec, min_seq_id=args.min_seq_id, coverage=args.coverage
    )
    splits = split_by_family(contexts, families, seed=args.seed)
    for ctx in contexts:
        ctx.family_id = families[ctx.parent_id]
        ctx.split = splits[ctx.parent_id]
    n_fam = len(set(families.values()))
    print(f"[cpseq] {n_fam} receptor families; split sizes "
          f"{pd.Series(list(splits.values())).value_counts().to_dict()}")

    selected = select_diverse_parents(contexts, families, args.n_parents, args.seed)
    sel_ids = {c.parent_id for c in selected}
    for ctx in contexts:
        ctx.is_control = False
    print(f"[cpseq] selected {len(selected)} parents for the tranche: "
          f"lengths {[c.peptide_length for c in selected]}")

    # --- candidates ------------------------------------------------------------------
    rows = []
    for ctx in selected:
        for chemistry in chemistries:
            cands = build_candidates(
                ctx.parent_id,
                ctx.reference_sequence,
                chemistry,
                bands=DEFAULT_BANDS,
                variants_per_band=DEFAULT_VARIANTS_PER_BAND,
            )
            for cand in cands:
                row = cand.to_row()
                row["identity_to_reference"] = round(identity_to_reference(cand), 6)
                row["family_id"] = ctx.family_id
                row["split"] = ctx.split
                rows.append(row)
    cand_df = pd.DataFrame(rows)

    # --- write -----------------------------------------------------------------------
    parents_df = pd.DataFrame(
        [dict(c.to_row(), selected_for_tranche=c.parent_id in sel_ids) for c in contexts]
    )
    parents_df.to_csv(man / "parents.csv", index=False)
    cand_df.to_csv(man / "candidates.csv", index=False)
    pd.DataFrame(rejected).to_csv(man / "rejected_parents.csv", index=False)
    pd.DataFrame(chem.capability_table_rows()).to_csv(man / "capability_table.csv", index=False)
    (man / "chemistry_spec.json").write_text(
        json.dumps({k: v.to_dict() for k, v in chem.CHEMISTRIES.items()}, indent=2)
    )

    cfg = {
        "lnr_root": str(args.lnr_root),
        "n_parents_requested": args.n_parents,
        "n_parents_selected": len(selected),
        "chemistries_built": chemistries,
        "cross_model_chemistries": cross_model,
        "bands": list(DEFAULT_BANDS),
        "variants_per_band": DEFAULT_VARIANTS_PER_BAND,
        "pocket_radius_A": POCKET_RADIUS_A,
        "family_min_seq_id": args.min_seq_id,
        "family_coverage": args.coverage,
        "seed": args.seed,
        "n_candidates": len(cand_df),
    }
    cfg["config_hash"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:16]
    (man / "build_config.json").write_text(json.dumps(cfg, indent=2))

    # --- summary ----------------------------------------------------------------------
    print(f"\n[cpseq] wrote {len(cand_df)} candidates over {len(selected)} parents "
          f"x {len(chemistries)} chemistry/ies -> {man}")
    print(f"[cpseq] config_hash={cfg['config_hash']}")
    per_chem = cand_df.groupby("chemistry").agg(
        candidates=("candidate_id", "count"),
        parents=("parent_id", "nunique"),
        min_identity=("identity_to_reference", "min"),
        max_mut_frac=("mutation_fraction", "max"),
    )
    print("\n" + per_chem.to_string())
    if (cand_df["identity_to_reference"] < 0.5 - 1e-9).any():
        raise SystemExit("FATAL: a candidate fell below 50% identity to its reference")
    print("\n[cpseq] identity floor >= 0.50 holds for every candidate")


if __name__ == "__main__":
    main()
