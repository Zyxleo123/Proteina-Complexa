"""Stage 3: sequence-level summaries, same-target comparisons, cross-predictor agreement.

Reads the score rows and writes joined tables. Four rules this module exists to enforce,
each of which is easy to break by accident and invisible afterwards:

**Technical completion and validity are different numbers.** A predictor that crashed on
half its tasks and produced perfect rings on the rest does not have a 100 % validity
rate. `n_attempted`, `n_completed` and `n_valid` are all reported, and the validity
fraction is stated against completed predictions with the completion rate beside it.

**Best-of-k is a statistic about k.** An arm that can only produce 2 samples
(afcycdesign -- see its adapter) must not have its best-of-2 compared against another
arm's best-of-4. Both are computed, each labelled with its own k, and `best_of_k` carries
the actual sample count the maximum was taken over.

**Predictor confidences are not pooled.** pLDDT, ptm, i_ptm and Boltz's confidences mean
different things on different scales; they stay in per-predictor columns and are never
averaged across arms or used as affinity labels.

**No per-sequence predictor cherry-picking.** Taking each sequence's most favourable
predictor and presenting the result as a label would manufacture a bias. Per-predictor
summaries stay separate, and the cross-predictor table reports agreement, not a winner.

Component scores stay separate columns. No weighted sum is invented here -- that is a
modelling decision for whoever trains on this, and baking in weights now would hide the
trade-offs they need to see.

An all-invalid sequence gets a row saying so, with `n_valid = 0` and the failure reasons
counted. It is not dropped, and it is certainly not given a favourable score.

Usage:
    python script_utils/cpseq_summarize.py --dataset-dir /data/user_data/yixiz/cpseq/tranche1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.scorer import receptor_aligned_peptide_rmsd  # noqa: E402

# Component score columns kept separate on purpose. Order is presentation only.
COMPONENTS = [
    "dG_separated",
    "dSASA_int",
    "dG_separated_per_1000A2",
    "delta_unsatHbonds",
    "hbonds_int",
    "sc_value",
    "packstat",
]


def load_rows(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((dataset_dir / "scores").glob("rows_*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001, S112
                continue
    if not rows:
        return pd.DataFrame()

    flat = []
    for r in rows:
        ros = r.get("rosetta") or {}
        raw, ref = ros.get("raw") or {}, ros.get("refined") or {}
        geo = r.get("geometry") or {}
        clo = geo.get("closure") or {}
        cl = geo.get("clashes") or {}
        deform = ros.get("deformation") or {}
        disp = ros.get("refinement_displacement") or {}
        conf = r.get("predictor_confidence") or {}
        out = {
            "task_id": r.get("task_id"),
            "candidate_id": r.get("candidate_id"),
            "parent_id": r.get("parent_id"),
            "family_id": r.get("family_id"),
            "split": r.get("split"),
            "chemistry": r.get("chemistry"),
            "predictor": r.get("predictor"),
            "predictor_revision": r.get("predictor_revision"),
            "input_hash": r.get("input_hash"),
            "sequence": r.get("sequence_requested"),
            "peptide_length": r.get("peptide_length"),
            "n_mutations": r.get("n_mutations"),
            "mutation_fraction": r.get("mutation_fraction"),
            "is_reference": r.get("is_reference"),
            "sample_index": r.get("sample_index"),
            "prediction_status": r.get("prediction_status"),
            "scoring_status": r.get("scoring_status"),
            "valid": bool(r.get("valid")),
            "validity_reasons": "; ".join(r.get("validity_reasons") or []),
            "sequence_matches": geo.get("sequence_matches"),
            "raw_pdb_path": r.get("raw_pdb_path"),
            "refined_pdb_path": ros.get("refined_pdb_path"),
            "pocket_conditioning": r.get("pocket_conditioning"),
            # closure geometry
            "closure_status": clo.get("status"),
            "closure_bond_A": clo.get("bond_length_A"),
            "closure_bond_dev_A": clo.get("bond_length_dev_A"),
            "closure_angle_i_dev_deg": clo.get("angle_i_dev_deg"),
            "closure_angle_j_dev_deg": clo.get("angle_j_dev_deg"),
            "closure_torsion_dev_deg": clo.get("dihedral_dev_deg"),
            "n_intra_clashes": cl.get("n_intra_clashes"),
            "n_inter_clashes": cl.get("n_inter_clashes"),
            "all_L": (geo.get("chirality") or {}).get("all_L"),
            # deformation proxy
            "deformation_status": deform.get("status"),
            "deformation_energy": deform.get("deformation_energy"),
            # refinement displacement
            "refine_binder_ca_rmsd_A": disp.get("binder_ca_rmsd_A"),
            "refine_receptor_ca_rmsd_A": disp.get("receptor_ca_rmsd_A"),
        }
        for c in COMPONENTS:
            out[f"raw_{c}"] = raw.get(c)
            out[f"refined_{c}"] = ref.get(c)
        # Per-predictor confidence columns, never pooled across predictors.
        for k, v in conf.items():
            out[f"conf_{k}"] = v
        flat.append(out)
    return pd.DataFrame(flat)


def _spread(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(float(np.mean(vals)), 4),
        "median": round(float(np.median(vals)), 4),
        "std": round(float(np.std(vals, ddof=1)), 4) if vals.size > 1 else None,
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
    }


def sequence_summaries(df: pd.DataFrame, score_col: str = "refined_dG_separated") -> pd.DataFrame:
    """One row per (candidate, predictor, chemistry). Attempts, completion, validity, scores."""
    out = []
    for (cand, pred, chem), g in df.groupby(["candidate_id", "predictor", "chemistry"], dropna=False):
        attempted = len(g)
        completed = int((g["prediction_status"] == "ok").sum())
        scored = int((g["scoring_status"] == "ok").sum())
        valid = g[g["valid"]]
        n_valid = len(valid)

        vals = pd.to_numeric(valid[score_col], errors="coerce").dropna().to_numpy()
        stats = _spread(vals)
        # Lower (more negative) dG is better, so "best" is the minimum.
        best = round(float(vals.min()), 4) if vals.size else None

        row = {
            "candidate_id": cand,
            "predictor": pred,
            "chemistry": chem,
            "parent_id": g["parent_id"].iloc[0],
            "family_id": g["family_id"].iloc[0],
            "split": g["split"].iloc[0],
            "sequence": g["sequence"].iloc[0],
            "peptide_length": g["peptide_length"].iloc[0],
            "n_mutations": g["n_mutations"].iloc[0],
            "mutation_fraction": g["mutation_fraction"].iloc[0],
            "is_reference": g["is_reference"].iloc[0],
            # Technical completion, kept separate from validity.
            "n_attempted": attempted,
            "n_completed": completed,
            "n_scored": scored,
            "n_valid": n_valid,
            "completion_rate": round(completed / attempted, 4) if attempted else None,
            # Validity fraction is AMONG COMPLETED predictions, not among attempts.
            "validity_fraction_of_completed": round(n_valid / completed, 4) if completed else None,
            "exact_sequence_rate_of_completed": (
                round(float(g.loc[g["prediction_status"] == "ok", "sequence_matches"].mean()), 4)
                if completed else None
            ),
            # best_of_k carries its own k, so a 2-sample arm is never read as best-of-4.
            "best_of_k": int(vals.size),
            f"best_valid_{score_col}": best,
            f"mean_valid_{score_col}": stats["mean"],
            f"median_valid_{score_col}": stats["median"],
            f"std_valid_{score_col}": stats["std"],
            f"range_valid_{score_col}": (
                round(stats["max"] - stats["min"], 4) if stats["max"] is not None else None
            ),
        }
        # An all-invalid sequence gets an explicit outcome, not a dropped row and
        # certainly not an invented favourable score.
        if n_valid == 0:
            reasons = Counter()
            for r in g["validity_reasons"]:
                for part in str(r).split("; "):
                    if part and part != "nan":
                        reasons[part.split(":")[0]] += 1
            row["outcome"] = "all_samples_invalid" if completed else "no_completed_prediction"
            row["top_failure_reasons"] = json.dumps(dict(reasons.most_common(4)))
        else:
            row["outcome"] = "has_valid_samples"
            row["top_failure_reasons"] = None

        # Component medians over valid samples, separate columns, no weighted sum.
        for c in COMPONENTS:
            cv = pd.to_numeric(valid[f"refined_{c}"], errors="coerce").dropna().to_numpy()
            row[f"median_valid_refined_{c}"] = (
                round(float(np.median(cv)), 4) if cv.size else None
            )
        dv = pd.to_numeric(valid["deformation_energy"], errors="coerce").dropna().to_numpy()
        row["median_valid_deformation_energy"] = round(float(np.median(dv)), 4) if dv.size else None
        out.append(row)
    return pd.DataFrame(out)


def same_target_comparisons(seq_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Pairwise candidate comparisons within one (parent, predictor, chemistry).

    This is the shape a ranker trains on: two candidate sequences against the same target
    under the same conditioning, with the score difference and both validity outcomes.
    Pairs where either side has no valid sample are kept, flagged, and given a null
    delta -- a sequence that produced nothing is informative, and dropping it would
    bias the training set toward sequences that happened to work.
    """
    col = f"best_valid_{score_col}"
    rows = []
    for (parent, pred, chem), g in seq_df.groupby(["parent_id", "predictor", "chemistry"]):
        g = g.sort_values("candidate_id")
        for a, b in combinations(g.itertuples(index=False), 2):
            va, vb = getattr(a, col), getattr(b, col)
            both = va is not None and vb is not None and not (pd.isna(va) or pd.isna(vb))
            rows.append({
                "parent_id": parent,
                "predictor": pred,
                "chemistry": chem,
                "candidate_a": a.candidate_id,
                "candidate_b": b.candidate_id,
                "sequence_a": a.sequence,
                "sequence_b": b.sequence,
                "n_mutations_a": a.n_mutations,
                "n_mutations_b": b.n_mutations,
                "is_reference_a": a.is_reference,
                "is_reference_b": b.is_reference,
                "n_valid_a": a.n_valid,
                "n_valid_b": b.n_valid,
                "best_of_k_a": a.best_of_k,
                "best_of_k_b": b.best_of_k,
                f"best_{score_col}_a": va,
                f"best_{score_col}_b": vb,
                "delta": round(float(va) - float(vb), 4) if both else None,
                "comparable": bool(both),
                "reason_not_comparable": None if both else (
                    f"a:{a.outcome}" if not both and (va is None or pd.isna(va)) else f"b:{b.outcome}"
                ),
            })
    return pd.DataFrame(rows)


def cross_predictor_agreement(df: pd.DataFrame, max_pairs_per_candidate: int = 2) -> pd.DataFrame:
    """Receptor-aligned peptide RMSD between predictors on the same candidate.

    The receptor defines the frame. Aligning on the peptide instead would minimise the
    very quantity being measured and report two different docking solutions as agreeing.
    """
    rows = []
    usable = df[(df["scoring_status"] == "ok") & df["raw_pdb_path"].notna()]
    for cand, g in usable.groupby("candidate_id"):
        preds = sorted(g["predictor"].unique())
        if len(preds) < 2:
            continue
        for pa, pb in combinations(preds, 2):
            ga = g[g["predictor"] == pa].sort_values("sample_index")
            gb = g[g["predictor"] == pb].sort_values("sample_index")
            n = min(len(ga), len(gb), max_pairs_per_candidate)
            for k in range(n):
                ra, rb = ga.iloc[k], gb.iloc[k]
                res = receptor_aligned_peptide_rmsd(
                    ra["raw_pdb_path"], rb["raw_pdb_path"],
                    binder_chain="B", receptor_chains=["A"],
                )
                rows.append({
                    "candidate_id": cand,
                    "parent_id": ra["parent_id"],
                    "chemistry": ra["chemistry"],
                    "predictor_a": pa,
                    "predictor_b": pb,
                    "sample_a": int(ra["sample_index"]),
                    "sample_b": int(rb["sample_index"]),
                    "valid_a": bool(ra["valid"]),
                    "valid_b": bool(rb["valid"]),
                    **{k2: v for k2, v in res.items()},
                })
    return pd.DataFrame(rows)


def within_predictor_variability(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Sample-to-sample spread within one (candidate, predictor) -- the model's own noise."""
    rows = []
    for (cand, pred), g in df.groupby(["candidate_id", "predictor"]):
        vals = pd.to_numeric(g[score_col], errors="coerce").dropna().to_numpy()
        s = _spread(vals)
        rows.append({
            "candidate_id": cand, "predictor": pred, "n_samples": len(g),
            "n_with_score": int(vals.size),
            **{f"{score_col}_{k}": v for k, v in s.items()},
            "closure_bond_dev_spread_A": round(
                float(pd.to_numeric(g["closure_bond_dev_A"], errors="coerce").dropna().std(ddof=1)), 4
            ) if g["closure_bond_dev_A"].notna().sum() > 1 else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--score-col", default="refined_dG_separated")
    ap.add_argument("--skip-agreement", action="store_true")
    args = ap.parse_args()

    df = load_rows(args.dataset_dir)
    if df.empty:
        print("[summarize] no score rows found")
        return

    out = args.dataset_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "samples.csv", index=False)
    print(f"[summarize] {len(df)} sample rows -> {out / 'samples.csv'}")

    seq = sequence_summaries(df, args.score_col)
    seq.to_csv(out / "sequence_summaries.csv", index=False)
    print(f"[summarize] {len(seq)} sequence summaries -> {out / 'sequence_summaries.csv'}")

    cmp_df = same_target_comparisons(seq, args.score_col)
    cmp_df.to_csv(out / "same_target_comparisons.csv", index=False)
    print(f"[summarize] {len(cmp_df)} same-target pairs "
          f"({int(cmp_df['comparable'].sum()) if len(cmp_df) else 0} comparable) "
          f"-> {out / 'same_target_comparisons.csv'}")

    var = within_predictor_variability(df, args.score_col)
    var.to_csv(out / "within_predictor_variability.csv", index=False)

    if not args.skip_agreement:
        agree = cross_predictor_agreement(df)
        agree.to_csv(out / "cross_predictor_agreement.csv", index=False)
        print(f"[summarize] {len(agree)} cross-predictor pose comparisons "
              f"-> {out / 'cross_predictor_agreement.csv'}")

    # Per-predictor x chemistry headline, completion kept apart from validity.
    print("\n=== per predictor x chemistry ===")
    for (pred, chem), g in seq.groupby(["predictor", "chemistry"]):
        att, comp, val = g["n_attempted"].sum(), g["n_completed"].sum(), g["n_valid"].sum()
        best = pd.to_numeric(g[f"best_valid_{args.score_col}"], errors="coerce").dropna()
        print(f"  {pred:12s} {chem:11s} candidates={len(g):4d} "
              f"samples attempted={att:4d} completed={comp:4d} valid={val:4d} "
              f"completion={comp / att if att else 0:.2%} "
              f"validity_of_completed={val / comp if comp else 0:.2%} "
              f"median best {args.score_col}={best.median() if len(best) else float('nan'):.2f}")


if __name__ == "__main__":
    main()
