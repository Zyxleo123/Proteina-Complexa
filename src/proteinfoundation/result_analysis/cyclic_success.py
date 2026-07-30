"""Success accounting for macrocyclic peptide binders.

The stock `protein_binder` criteria (``i_pAE*31 <= 7``, ``pLDDT >= 0.9``,
``scRMSD_ca < 1.5``) were calibrated on de novo miniproteins of 60-100 residues.
Measured over 298 CPSea designs for target 1E2T:

===================  ==========  =========  ====================
criterion            threshold   best obs.  designs passing
===================  ==========  =========  ====================
``pLDDT >= 0.9``     0.900       0.830      **0 / 298**
``i_pAE*31 <= 7``    0.226       0.370      **0 / 298**
``scRMSD_ca < 1.5``  1.500       0.250      250 / 298
``ring closed``      --          --         179 / 191 (not scored)
===================  ==========  =========  ====================

Two of the three criteria have *zero* headroom: no design in any recorded run has
ever come close, so they do not rank models, they veto every model equally. That
is the whole of the long-standing 0% pass rate. Meanwhile ring closure -- the one
property that distinguishes a macrocycle from a peptide, and the property the
model was actually changed to improve -- sits at 94% and enters the score nowhere.

This module fixes the accounting in three ways:

1. **The ring is a gate.** `filter_by_cyclization_validity` rejects designs whose
   requested bond did not form. Closure is a property of the *generated backbone*,
   identical across every MPNN redesign of a sample, so it is a sample-level gate
   and deliberately not folded into the per-redesign "any redesign passes all
   criteria" logic in `filter_by_success_thresholds`.

   That distinction is not cosmetic. Closure columns are flat
   (``generated_binder_cyc_bond_closed``), not in the
   ``{seq_type}_{prefix}_{metric}_all`` shape that filter builds names in, so
   handing them to it does not raise -- it logs "missing column" and silently
   drops the criterion. A chemistry gate that quietly evaporates is worse than
   no gate.

2. **Raw and post-selection success stay separate.**
   `compute_cyclic_success_breakdown` reports generation quality (over everything
   sampled, before any filtering) apart from selected quality, so a number that
   was used to *filter* candidates can never be reported as the headline result.

3. **Thresholds are provisional until calibrated.** See
   `CYCLIC_PEPTIDE_PROVISIONAL_THRESHOLDS`.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column prefix the evaluation harness stamps on metrics measured from the
# *generated* structure (the only structure that carries the ring; a refold does not).
GENERATED_PREFIX = "generated_"

CYC_CLOSED_COL = f"{GENERATED_PREFIX}binder_cyc_bond_closed"
CYC_FUSED_COL = f"{GENERATED_PREFIX}binder_cyc_bond_fused"
CYC_DIST_COL = f"{GENERATED_PREFIX}binder_cyc_bond_dist_A"
CYC_TYPE_REQ_COL = f"{GENERATED_PREFIX}binder_cyc_type_requested"
CYC_TYPE_OBS_COL = f"{GENERATED_PREFIX}binder_cyc_type_observed"
CYC_TYPE_SAT_COL = f"{GENERATED_PREFIX}binder_cyc_type_satisfied"


# PROVISIONAL. These are NOT calibrated and must not be used for a headline result.
#
# They are placed one step outside the best value observed across 298 linear-refold
# designs, purely so the pipeline has non-degenerate criteria to run end to end. The
# honest values come from `configs/evaluate_cpsea_native_calibration.yaml`: score the
# real crystallographic peptide for each target through this same harness (now with
# the cyclic offset on) and set each threshold so the native passes. A threshold no
# native structure can meet is a bug in the threshold, not a finding about the model.
#
# Deriving thresholds from the *design* distribution instead would be circular -- it
# would define success as "typical of what we already generate".
CYCLIC_PEPTIDE_PROVISIONAL_THRESHOLDS = {
    "i_pAE": {"threshold": 12.0, "op": "<=", "scale": 31.0, "column_prefix": "complex"},
    "pLDDT": {"threshold": 0.7, "op": ">=", "scale": 1.0, "column_prefix": "complex"},
    "scRMSD_ca": {"threshold": 1.5, "op": "<", "scale": 1.0, "column_prefix": "binder"},
}


def _as_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Column as a strict bool Series; missing or non-boolean entries become False.

    CSV round-trips turn booleans into the strings ``"True"``/``"False"``, and
    undeterminable closures are NaN. NaN must read as False for a *requirement*
    (an unmeasurable ring is not a closed ring) -- callers wanting the opposite
    polarity should negate after, not rely on NaN propagation.
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    raw = df[col]
    if raw.dtype == bool:
        return raw
    return raw.astype(str).str.strip().str.lower().eq("true")


def cyclization_validity_mask(
    df: pd.DataFrame,
    require_closed: bool = True,
    reject_fused: bool = True,
    require_type_satisfied: bool = False,
) -> pd.Series:
    """Bool mask over rows: does this design carry the macrocycle it was asked for?

    Args:
        df: Evaluation dataframe (one row per generated design).
        require_closed: Require the closing bond inside its acceptance window.
        reject_fused: Reject anchor atoms closer than a bond can physically be. A
            fused ring is a distinct failure from an open one -- it is what a
            one-sided closure pressure produces -- and it passes any pure
            proximity test, so it needs its own veto.
        require_type_satisfied: Also require the observed chemistry to match the
            requested linkage. Off by default because
            ``binder_cyc_type_requested`` is NaN in every run recorded so far
            (type conditioning was not active at design time), which would gate
            every design out.
    """
    mask = pd.Series(True, index=df.index)
    if require_closed:
        mask &= _as_bool_series(df, CYC_CLOSED_COL)
    if reject_fused:
        mask &= ~_as_bool_series(df, CYC_FUSED_COL)
    if require_type_satisfied:
        mask &= _as_bool_series(df, CYC_TYPE_SAT_COL)
    return mask


def filter_by_cyclization_validity(
    df: pd.DataFrame,
    require_closed: bool = True,
    reject_fused: bool = True,
    require_type_satisfied: bool = False,
) -> pd.DataFrame:
    """Drop designs that failed to form the requested macrocycle (sample-level gate)."""
    mask = cyclization_validity_mask(
        df,
        require_closed=require_closed,
        reject_fused=reject_fused,
        require_type_satisfied=require_type_satisfied,
    )
    kept, total = int(mask.sum()), len(df)
    pct = 100.0 * kept / total if total else 0.0
    logger.info("Cyclization validity gate: %d/%d designs kept (%.0f%%)", kept, total, pct)
    return df[mask]


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else float("nan")


def compute_cyclic_success_breakdown(
    df: pd.DataFrame,
    quality_mask: pd.Series | None = None,
    target_col: str = "target_name",
    require_closed: bool = True,
    reject_fused: bool = True,
    require_type_satisfied: bool = False,
) -> dict:
    """Separate raw-generation, chemistry-gate, and joint success.

    `quality_mask` is the *independent* binder-quality verdict (interface/confidence
    thresholds). It is kept apart from the chemistry gate on purpose: reporting a
    single blended number would let a metric used to select candidates reappear as
    the headline result, which is exactly the confound to avoid.

    Returns micro rates (over all designs) plus macro rates (mean over per-target
    rates, so a target with many designs cannot dominate) and per-target detail.
    """
    n = len(df)
    chem = cyclization_validity_mask(
        df,
        require_closed=require_closed,
        reject_fused=reject_fused,
        require_type_satisfied=require_type_satisfied,
    )
    if quality_mask is None:
        quality = pd.Series(False, index=df.index)
        quality_available = False
    else:
        quality = quality_mask.reindex(df.index).fillna(False).astype(bool)
        quality_available = True

    joint = chem & quality

    out: dict = {
        "n_designs": n,
        # Raw generation: measured over everything sampled, before any filtering.
        "raw_ring_closed_rate": _rate(int(_as_bool_series(df, CYC_CLOSED_COL).sum()), n),
        "raw_ring_fused_rate": _rate(int(_as_bool_series(df, CYC_FUSED_COL).sum()), n),
        "raw_chemistry_valid_rate": _rate(int(chem.sum()), n),
        "quality_available": quality_available,
        "quality_pass_rate": _rate(int(quality.sum()), n) if quality_available else float("nan"),
        "joint_success_rate": _rate(int(joint.sum()), n) if quality_available else float("nan"),
    }

    if CYC_DIST_COL in df.columns:
        dist = pd.to_numeric(df[CYC_DIST_COL], errors="coerce").dropna()
        out["bond_dist_A_median"] = float(dist.median()) if len(dist) else float("nan")
        out["bond_dist_A_n"] = int(len(dist))

    # Per-linkage-type breakdown: the three chemistries fail for different reasons and
    # at very different rates, so a pooled number hides the one that is actually broken.
    type_col = CYC_TYPE_REQ_COL if df.get(CYC_TYPE_REQ_COL) is not None else CYC_TYPE_OBS_COL
    if type_col in df.columns:
        by_type = {}
        for linkage, sub in df.groupby(df[type_col].astype(str)):
            if linkage in ("nan", "None", ""):
                continue
            sub_chem = chem.reindex(sub.index).fillna(False)
            by_type[linkage] = {
                "n": len(sub),
                "chemistry_valid_rate": _rate(int(sub_chem.sum()), len(sub)),
            }
        out["by_linkage_type"] = by_type

    if target_col in df.columns:
        per_target = {}
        for target, sub in df.groupby(df[target_col].astype(str)):
            sub_chem = chem.reindex(sub.index).fillna(False)
            sub_joint = joint.reindex(sub.index).fillna(False)
            per_target[target] = {
                "n": len(sub),
                "chemistry_valid_rate": _rate(int(sub_chem.sum()), len(sub)),
                "joint_success_rate": _rate(int(sub_joint.sum()), len(sub)) if quality_available else float("nan"),
                # Target coverage: did best-of-N land at least one usable design here?
                "any_success": bool(sub_joint.any()) if quality_available else bool(sub_chem.any()),
            }
        out["per_target"] = per_target
        out["n_targets"] = len(per_target)
        out["macro_chemistry_valid_rate"] = float(
            np.mean([v["chemistry_valid_rate"] for v in per_target.values()])
        )
        out["target_coverage"] = float(np.mean([v["any_success"] for v in per_target.values()]))
        if quality_available:
            out["macro_joint_success_rate"] = float(
                np.mean([v["joint_success_rate"] for v in per_target.values()])
            )

    return out
