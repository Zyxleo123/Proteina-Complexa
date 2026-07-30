"""Select a balanced, cluster-deduplicated CPSea evaluation target set.

Why this exists
---------------
The project's targets dict carries 5 CPSea targets, none of them natively disulfide.
That made the disulfide failure mode (the weakest of the three chemistries) impossible
to study on-native, and it made per-target confidence intervals hopeless: with n=5 the
Phase 0 joint-success CI was [4.0%, 30.5%].

The held-out CPSea test split has 645 targets in 296 sequence clusters, including 79
native disulfides. This script turns that into an evaluation set.

Three design choices worth stating:

1. **Cluster deduplication.** One target per ``cluster_id``. Targets in a cluster are
   homologous, so treating them as separate observations would inflate n without adding
   information and would make the bootstrap CI (which resamples targets as the unit of
   independence) too tight.

2. **Chemistry comes from parsing, not from the metadata hint.** The metadata
   ``cyclization_type`` has only three values and lumps isopeptide together with every
   other chemistry under ``"other"`` -- `parse_labels.py` is explicit that the hint is
   "never used to assign i/j/type". So each candidate is parsed with the same
   `infer_cyclization_label` the training pipeline uses, and targets whose parsed type
   disagrees with the hint, or that yield no label at all, are reported and dropped
   rather than silently mislabelled.

3. **Balanced strata.** Equal targets per chemistry, so the per-linkage comparison has
   comparable power in each arm. Pooling the natural distribution would leave disulfide
   swamped -- exactly the situation that made the recorded runs useless for it (n=21).

Every selected target carries its native Rosetta dG from
``CPSea_PDB_Affinity.tsv``, which supplies the per-target quality bar without any new
Rosetta run.

Usage:
    python script_utils/select_cpsea_eval_targets.py --per-type 32 \
        --out-yaml configs/targets/cpsea_eval_targets.yaml \
        --out-csv calibration_native/cpsea_eval_targets.csv
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from proteinfoundation.cyclization.constants import (  # noqa: E402
    CYCLIZATION_TYPE_TO_NAME,
    NO_CYCLIZATION_INDEX,
)
from proteinfoundation.cyclization.parse_labels import (  # noqa: E402
    infer_cyclization_label,
    read_atom_conect_records,
)

logger = logging.getLogger(__name__)

DATA_ROOT = "/zfsauton/scratch/yixiz/CPSea/CPSea_PDB"
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def binder_residue_order(atoms: dict, chain: str) -> list[int]:
    """Binder-local index -> resSeq, standard residues only, ascending resSeq.

    Ascending order matches `preprocess_cpsea`'s output sort key, so local indices
    agree with what the model sees.
    """
    resseqs = {a["resseq"] for a in atoms.values() if a["chain_id"] == chain and a["resname"] in STANDARD_AA}
    return sorted(resseqs)


def parse_one(job: tuple) -> dict:
    example_id, path, binder_chain, peptide_length, hint = job
    out = {"example_id": example_id, "parsed_type": None, "parse_reason": None}
    try:
        atoms, _ = read_atom_conect_records(path)
        order = binder_residue_order(atoms, binder_chain)
        label = infer_cyclization_label(
            pdb_path=path,
            residue_pdb_idx=order,
            binder_length_hint=peptide_length,
            binder_chain_id=binder_chain,
            cyclization_type_hint=hint,
        )
        out["parse_reason"] = label.get("reason")
        type_idx = int(label.get("type", NO_CYCLIZATION_INDEX))
        if label.get("has_cyclization") and type_idx != NO_CYCLIZATION_INDEX:
            out["parsed_type"] = CYCLIZATION_TYPE_TO_NAME.get(type_idx)
        out["parsed_i"] = label.get("i")
        out["parsed_j"] = label.get("j")
    except Exception as e:  # noqa: BLE001 - one unreadable structure must not kill the sweep
        out["parse_reason"] = f"error:{type(e).__name__}: {e}"
    return out


def build(args) -> pd.DataFrame:
    meta_path = os.path.join(args.data_root, "preprocessed", "metadata", f"cpsea_{args.split}.parquet")
    meta = pd.read_parquet(meta_path)
    logger.info("Loaded %d rows from %s", len(meta), meta_path)

    affinity_path = os.path.join(args.data_root, f"{os.path.basename(args.data_root)}_properties", f"{os.path.basename(args.data_root)}_Affinity.tsv")
    if os.path.exists(affinity_path):
        aff = pd.read_csv(affinity_path, sep="\t")
        meta = meta.merge(aff[["id", "rosetta_dG"]], left_on="example_id", right_on="id", how="left")
        logger.info("Joined native dG for %d/%d targets", int(meta["rosetta_dG"].notna().sum()), len(meta))
    else:
        logger.warning("No affinity table at %s; targets will have no native dG bar.", affinity_path)
        meta["rosetta_dG"] = pd.NA

    meta = meta[
        meta["peptide_length"].between(args.min_len, args.max_len)
        & meta["receptor_length"].between(args.min_receptor, args.max_receptor)
    ]
    logger.info("After length filters: %d", len(meta))

    # Parse EVERY candidate, not just one arbitrary member per cluster. Clusters are mixed
    # across chemistries (the per-type cluster counts sum to more than the cluster total),
    # so picking a representative before knowing its chemistry makes the balance of the
    # final set a lottery -- and it is the rarest arm, disulfide, that loses. Parsing all
    # of them lets the representative be chosen to favour the scarce chemistry.
    jobs = [
        (r.example_id, r.path, r.binder_chain_id, int(r.peptide_length), r.cyclization_type)
        for r in meta.itertuples()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parsed = pd.DataFrame(list(pool.map(parse_one, jobs, chunksize=8)))
    reps = meta.merge(parsed, on="example_id", how="left")

    unparsed = int(reps["parsed_type"].isna().sum())
    if unparsed:
        logger.warning("Dropping %d/%d targets with no parseable cyclization label", unparsed, len(reps))
    reps = reps[reps["parsed_type"].notna()]

    # Disagreements between the coarse hint and the parsed chemistry are expected for
    # "other" (which has no dedicated hint) but not for head_tail/disulfide.
    hint_map = {"head_tail": "mainchain", "disulfide": "disulfide"}
    checkable = reps[reps["cyclization_type"].isin(hint_map)]
    mismatched = checkable[checkable["parsed_type"] != checkable["cyclization_type"].map(hint_map)]
    if len(mismatched):
        logger.warning(
            "%d targets where the metadata hint and the parsed chemistry disagree; dropping them "
            "(a target whose own label is ambiguous cannot support a per-linkage claim)",
            len(mismatched),
        )
        reps = reps.drop(index=mismatched.index)

    logger.info("Parsed chemistry (all candidates):\n%s", reps["parsed_type"].value_counts().to_string())

    # Now deduplicate: one target per cluster, preferring the globally scarcest chemistry
    # present in that cluster. Homologous targets are not independent observations, so the
    # cluster is the unit; which member represents it is free, and spending that freedom on
    # the under-represented arm costs nothing and materially improves its power.
    rarity = reps["parsed_type"].value_counts()
    reps = reps.assign(_rarity=reps["parsed_type"].map(rarity))
    before = len(reps)
    reps = (
        reps.sort_values(["_rarity", "example_id"])
        .groupby("cluster_id", as_index=False)
        .first()
        .drop(columns=["_rarity"])
    )
    logger.info("Cluster-deduplicated %d -> %d representatives (rarest chemistry preferred)", before, len(reps))
    logger.info("Parsed chemistry (cluster representatives):\n%s", reps["parsed_type"].value_counts().to_string())

    selected = (
        reps.sort_values("example_id")
        .groupby("parsed_type", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), args.per_type), random_state=args.seed), include_groups=True)
        .reset_index(drop=True)
    )
    return selected


def to_targets_yaml(df: pd.DataFrame, prefix: str) -> str:
    lines = [
        "# CPSea evaluation targets -- AUTO-GENERATED by script_utils/select_cpsea_eval_targets.py",
        "# Cluster-deduplicated, balanced across parsed cyclization chemistry.",
        "# `cyclization_type` here is the PARSED chemistry (from CONECT records), not the",
        "# coarse metadata hint, which lumps isopeptide into 'other'.",
        "# `native_rosetta_dG` is the per-target quality bar from CPSea_PDB_Affinity.tsv.",
        "",
        "target_dict_cfg:",
    ]
    for row in df.itertuples():
        dg = getattr(row, "rosetta_dG", None)
        lines += [
            f"  {prefix}{row.example_id}:",
            "    source: cpsea",
            f"    target_filename: {row.example_id}",
            f"    target_path: {row.path}",
            '    target_input: "A"',
            "    hotspot_residues: []",
            f"    binder_length: [{int(row.peptide_length)}, {int(row.peptide_length)}]",
            "    pdb_id: null",
            f"    cyclization_type: {row.parsed_type}",
            f"    native_rosetta_dG: {round(float(dg), 3) if pd.notna(dg) else 'null'}",
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--per-type", type=int, default=32)
    p.add_argument("--min-len", type=int, default=6)
    p.add_argument("--max-len", type=int, default=16)
    p.add_argument("--min-receptor", type=int, default=40)
    p.add_argument("--max-receptor", type=int, default=400)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="cpsea_eval_")
    p.add_argument("--out-yaml", default="configs/targets/cpsea_eval_targets.yaml")
    p.add_argument("--out-csv", default="calibration_native/cpsea_eval_targets.csv")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = build(args)
    if df.empty:
        logger.error("No targets selected.")
        return 1

    print("\n=== SELECTED EVALUATION TARGETS ===")
    print(df.groupby("parsed_type").agg(n=("example_id", "size"), median_len=("peptide_length", "median"), median_native_dG=("rosetta_dG", "median")).to_string())
    print(f"\ntotal: {len(df)} targets across {df['cluster_id'].nunique()} clusters")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        keep = ["example_id", "path", "binder_chain_id", "cluster_id", "peptide_length", "receptor_length", "cyclization_type", "parsed_type", "parsed_i", "parsed_j", "rosetta_dG"]
        df[[c for c in keep if c in df.columns]].to_csv(args.out_csv, index=False)
        logger.info("Wrote %s", args.out_csv)

    if args.out_yaml:
        os.makedirs(os.path.dirname(args.out_yaml) or ".", exist_ok=True)
        with open(args.out_yaml, "w") as handle:
            handle.write(to_targets_yaml(df, args.prefix))
        logger.info("Wrote %s", args.out_yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
