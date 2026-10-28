"""Frozen, chemistry-specific geometry tolerances, and the verdict that applies them.

Separating this from `geometry.py` is deliberate: that module *measures*, this one
*judges*, and the thresholds are data that get written to disk with the tranche so a
later reader can see exactly what bar a structure was held to.

Where the numbers come from
---------------------------
Two sources, and which is which is recorded per field:

**Standard chemistry.** The closing-bond length windows are the project's existing ones
(`eval.cyclic_reconstruction_metrics`: mainchain 1.1-1.6 A, disulfide 1.8-2.3 A,
isopeptide 1.1-1.6 A), reused rather than restated so a "closed" ring means the same
thing here as it does in training.

**The control structures.** There were no existing angle or torsion tolerances, so they
are derived the way `script_utils/derive_cyclic_thresholds.py` derives its thresholds:
place the bar at the **worst experimental control**, loosened by a margin, so every real
macrocycle passes. The point of a tolerance derived this way is to stop the criteria
vetoing everything, not to be selective -- selectivity comes from the metrics with real
headroom (interface dG, deformation, clashes).

**The control set is small**: 4 mainchain and 2 disulfide verified complexes. A bar set
from 2 structures is a weak bar, and the saved JSON says so in `n_controls` so nobody
mistakes it for a distribution.

Pass/fail is never inferred from a missing value. A structure with no measurable closure
is `invalid` with reason `unmeasurable`, which is distinct from `invalid` with reason
`bond_length_out_of_window`: the first is an absent measurement, the second is a
measurement that failed. Neither is success.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    DISULFIDE_SG_BOND_WINDOW_A,
    ISOPEPTIDE_N_C_BOND_WINDOW_A,
    MAINCHAIN_CN_BOND_WINDOW_A,
)

# Below this the anchor atoms are interpenetrating rather than bonded -- no covalent bond
# is anywhere near this short. Reused from `evaluation.cyclization_pdb_metrics`, where it
# exists because a closure term with no lower bound can fuse two atoms together, and that
# failure is invisible to a one-sided proximity test.
FUSED_BOND_DIST_A = 1.0

BOND_WINDOWS_A = {
    "mainchain": MAINCHAIN_CN_BOND_WINDOW_A,
    "disulfide": DISULFIDE_SG_BOND_WINDOW_A,
    "isopeptide": ISOPEPTIDE_N_C_BOND_WINDOW_A,
}

# --------------------------------------------------------------------------------------
# Adopted structural-quality requirements (2026-10-22)
# --------------------------------------------------------------------------------------
# Closure windows, as requested: MuCO's published closure-distance windows, which inspect
# the actual closing atoms rather than a nearby proxy. These are TIGHTER than the project's
# previous 1.1-1.6 A mainchain window, and the difference is not academic:
#
#   7K2F, an experimental crystal macrocycle control, closes at 1.554 A and therefore
#   FAILS this window while passing the previous one.
#
# It is recorded here rather than left to be rediscovered, because the acceptance test for
# this gate is "every retained native reference passes, with every original failure
# explained". 7K2F is that one explained failure: it is a genuine deposited structure with
# an unusually long closing bond, it is the same control MuCO's own +-0.10 A window
# rejects, and it is retained in the control set as a known out-of-window reference rather
# than deleted. The other 3 mainchain controls and all 150 native CPSea head-to-tail
# entries measured pass this window.
ADOPTED_BOND_WINDOWS_A = {
    "mainchain": (1.23, 1.43),   # closing amide
    "disulfide": (1.95, 2.15),   # closing disulfide
    "isopeptide": (1.23, 1.43),  # also an amide bond
}
ADOPTED_BOND_WINDOW_SOURCE = (
    "MuCO published closure-distance windows (amide 1.23-1.43 A, disulfide 1.95-2.15 A); "
    "adopted 2026-10-22. Tighter than the prior 1.1-1.6 A project standard; 7K2F "
    "(1.554 A) is a known retained control that fails it -- see module comment."
)

# Amide planarity. An amide bond is planar with two real arrangements, cis and trans, and
# a bond within this tolerance of EITHER is correctly formed; `chemistry.AtomBond`
# declares which arrangements each chemistry permits and `geometry.closure_geometry`
# measures against the nearest, recording which it matched.
#
# This replaces a tolerance derived as "worst of 4 experimental controls + 5% margin".
# That derivation was sound in procedure and wrong in reference set: all four mainchain
# controls happen to be trans (dihedrals 177.0, 173.0, 172.0, 179.2), so it could only
# ever produce a trans-only window, and it produced 8.44 deg. Measured against 150
# cluster-unique native CPSea head-to-tail macrocycles -- the corpus these models are
# trained on -- that window admits 27/150 while cis-or-trans at 30 deg admits 117/150,
# and 75/150 of those natives sit nearer cis than trans with a median deviation from
# trans of 90.3 deg. A gate calibrated on 4 structures was rejecting 82% of the
# chemistry it exists to measure.
# Planarity is an AMIDE criterion and applies only to chemistries whose closing bond is
# an amide. A disulfide is not planar: its CB-SG-SG-CB dihedral sits near +-90 deg with a
# genuinely broad spread, so a 30 deg planarity bar around +-90 is a category error, not a
# strict standard. Measured: it fails 1SLD, a deposited crystal structure, at 52.82 deg.
# Disulfides therefore keep the project's existing control-derived procedure, which is
# chemistry-appropriate because it makes no planarity claim.
ADOPTED_TORSION_TOL_DEG = 30.0
AMIDE_CHEMISTRIES = ("mainchain", "isopeptide")
ADOPTED_TORSION_SOURCE = (
    "planarity within 30 deg of the nearest arrangement the chemistry permits (cis or "
    "trans for an amide), following the established twisted-peptide classification; "
    "adopted 2026-10-22, replacing a trans-only 8.44 deg window derived from 4 "
    "all-trans controls that admitted 27/150 native CPSea head-to-tail macrocycles."
)

# Fractional slack away from the worst control, so a bar sitting exactly on an observed
# value does not fail on floating-point noise or a re-run. Same default as
# `derive_cyclic_thresholds.py`.
DEFAULT_MARGIN = 0.05

# Fallbacks used only where no control of that chemistry exists. Generous on purpose: an
# undersupported chemistry should not be judged by an invented-precise bar.
FALLBACK_ANGLE_TOL_DEG = 25.0
FALLBACK_TORSION_TOL_DEG = 60.0

# Clash limits. Both are shared across chemistries because a steric overlap is a steric
# overlap regardless of how the ring closes.
MAX_INTRA_CLASHES = 0
MAX_INTER_CLASHES = 0


@dataclass
class ChemistryTolerance:
    chemistry: str
    bond_length_min_A: float
    bond_length_max_A: float
    bond_length_source: str
    angle_tol_deg: float
    angle_tol_source: str
    torsion_tol_deg: float
    torsion_tol_source: str
    n_controls: int
    max_intra_clashes: int = MAX_INTRA_CLASHES
    max_inter_clashes: int = MAX_INTER_CLASHES
    # The worst closure-torsion deviation among this chemistry's experimental controls,
    # kept for audit. It is NOT the bar -- `torsion_tol_deg` is a declared standard -- but
    # it is what a control-derived bar would have been, so the two stay comparable.
    worst_control_torsion_dev_deg: float | None = None
    fused_bond_dist_A: float = FUSED_BOND_DIST_A
    require_all_L: bool = True
    require_backbone_complete: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def derive_tolerances(
    control_closures: dict[str, list[dict]], *, margin: float = DEFAULT_MARGIN
) -> dict[str, ChemistryTolerance]:
    """Build the frozen tolerance set.

    `control_closures` maps chemistry -> list of `geometry.closure_geometry` records from
    **experimental** control structures. Bond-length windows ignore it (they come from
    standard chemistry); angle and torsion tolerances are the worst control deviation
    plus the margin, or the fallback where that chemistry has no control.
    """
    out: dict[str, ChemistryTolerance] = {}
    for chemistry, (lo, hi) in ADOPTED_BOND_WINDOWS_A.items():
        records = [
            r for r in control_closures.get(chemistry, []) if r.get("status") == "ok"
        ]
        angle_devs, tor_devs = [], []
        for r in records:
            for key in ("angle_i_dev_deg", "angle_j_dev_deg"):
                if r.get(key) is not None:
                    angle_devs.append(float(r[key]))
            if r.get("dihedral_dev_deg") is not None:
                tor_devs.append(float(r["dihedral_dev_deg"]))

        if angle_devs:
            angle_tol = round(max(angle_devs) * (1.0 + margin), 2)
            angle_src = f"worst of {len(records)} experimental controls + {margin:.0%} margin"
        else:
            angle_tol, angle_src = FALLBACK_ANGLE_TOL_DEG, "fallback (no control of this chemistry)"

        # Declared standard, NOT derived from the controls. Deriving it from this control
        # set is what produced a trans-only window; the worst control deviation is still
        # reported below so the derivation remains auditable against the adopted bar.
        worst_control_torsion = round(max(tor_devs), 2) if tor_devs else None
        if chemistry in AMIDE_CHEMISTRIES:
            tor_tol, tor_src = ADOPTED_TORSION_TOL_DEG, ADOPTED_TORSION_SOURCE
        elif tor_devs:
            # Not a planarity criterion; the existing worst-control procedure stands.
            tor_tol = round(max(tor_devs) * (1.0 + margin), 2)
            tor_src = (f"worst of {len(records)} experimental controls + {margin:.0%} "
                       f"margin (non-amide closure; planarity does not apply)")
        else:
            tor_tol = FALLBACK_TORSION_TOL_DEG
            tor_src = "fallback (no control of this chemistry; planarity does not apply)"

        out[chemistry] = ChemistryTolerance(
            chemistry=chemistry,
            bond_length_min_A=lo,
            bond_length_max_A=hi,
            bond_length_source=ADOPTED_BOND_WINDOW_SOURCE,
            angle_tol_deg=angle_tol,
            angle_tol_source=angle_src,
            torsion_tol_deg=tor_tol,
            torsion_tol_source=tor_src,
            worst_control_torsion_dev_deg=worst_control_torsion,
            n_controls=len(records),
        )
    return out


def freeze(tolerances: dict[str, ChemistryTolerance], out_path: Path, *, note: str = "") -> None:
    """Write the tolerance set with provenance. Written once per tranche, then read-only."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(
            {
                "frozen_on": date.today().isoformat(),
                "note": note
                or (
                    "Frozen for the first tranche. Bond-length windows are the project "
                    "standard; angle and torsion tolerances are set at the worst "
                    "experimental control plus a margin, so every real macrocycle "
                    "passes. Derived from a SMALL control set -- see n_controls."
                ),
                "margin": DEFAULT_MARGIN,
                "clash_cutoff_note": (
                    "max_intra/inter_clashes are counts of heavy-atom pairs below the "
                    "2.2 A cutoff in geometry.clashes, excluding bonded and 1-3 "
                    "neighbours and, for a head-to-tail ring, its own closing bond."
                ),
                "tolerances": {k: v.to_dict() for k, v in tolerances.items()},
            },
            indent=2,
        )
    )


def load(path: Path) -> dict[str, ChemistryTolerance]:
    data = json.loads(Path(path).read_text())["tolerances"]
    return {k: ChemistryTolerance(**v) for k, v in data.items()}


def verdict(validity: dict, tol: ChemistryTolerance) -> dict:
    """Apply `tol` to a `geometry.validity_report`. Returns a verdict with every reason.

    `valid` requires all of: exact sequence, complete backbone, a measurable closing bond
    inside the window and not fused, both closure angles and the torsion inside tolerance,
    all-L stereochemistry, and no severe clashes. Each failure is listed, so a structure
    that misses on two counts does not look like it missed on one.
    """
    reasons: list[str] = []
    checks: dict[str, bool | None] = {}

    seq_ok = bool(validity.get("sequence_matches"))
    checks["sequence_exact"] = seq_ok
    if not seq_ok:
        reasons.append(
            f"sequence mismatch: requested {validity.get('sequence_requested')!r}, "
            f"got {validity.get('sequence_observed')!r}"
        )

    if tol.require_backbone_complete:
        bb_ok = bool(validity.get("atoms", {}).get("backbone_complete"))
        checks["backbone_complete"] = bb_ok
        if not bb_ok:
            n = validity.get("atoms", {}).get("n_residues_missing_atoms")
            reasons.append(f"incomplete backbone: {n} residue(s) missing atoms")

    closure = validity.get("closure", {})
    if closure.get("status") != "ok":
        # An unmeasurable ring is NOT a closed ring, and it is not the same failure as a
        # measured bond that is out of window.
        checks["closure_measurable"] = False
        checks["closure_in_window"] = None
        reasons.append(f"closure unmeasurable: {closure.get('status')}")
    else:
        checks["closure_measurable"] = True
        d = closure.get("bond_length_A")
        in_win = d is not None and tol.bond_length_min_A <= d <= tol.bond_length_max_A
        checks["closure_in_window"] = bool(in_win)
        if not in_win:
            reasons.append(
                f"closing bond {d} A outside [{tol.bond_length_min_A}, "
                f"{tol.bond_length_max_A}] for {tol.chemistry}"
            )
        fused = d is not None and d < tol.fused_bond_dist_A
        checks["not_fused"] = not fused
        if fused:
            reasons.append(f"anchor atoms fused at {d} A (< {tol.fused_bond_dist_A} A)")

        for key, label in (("angle_i_dev_deg", "angle_i"), ("angle_j_dev_deg", "angle_j")):
            dev = closure.get(key)
            ok = dev is not None and dev <= tol.angle_tol_deg
            checks[f"{label}_in_tol"] = bool(ok)
            if not ok:
                reasons.append(
                    f"{label} deviates {dev} deg (tolerance {tol.angle_tol_deg})"
                )
        tdev = closure.get("dihedral_dev_deg")
        tok = tdev is not None and tdev <= tol.torsion_tol_deg
        checks["torsion_in_tol"] = bool(tok)
        if not tok:
            reasons.append(f"closure torsion deviates {tdev} deg (tolerance {tol.torsion_tol_deg})")

    if tol.require_all_L:
        chir = validity.get("chirality", {})
        l_ok = bool(chir.get("all_L"))
        checks["all_L"] = l_ok
        if not l_ok:
            reasons.append(f"{chir.get('n_d_amino_acids')} D-amino acid(s) present")

    cl = validity.get("clashes", {})
    intra_ok = int(cl.get("n_intra_clashes", 0)) <= tol.max_intra_clashes
    inter_ok = int(cl.get("n_inter_clashes", 0)) <= tol.max_inter_clashes
    checks["intra_clashes_ok"] = intra_ok
    checks["inter_clashes_ok"] = inter_ok
    if not intra_ok:
        reasons.append(
            f"{cl.get('n_intra_clashes')} intra-peptide clash(es) "
            f"(max {tol.max_intra_clashes}, min dist {cl.get('min_intra_dist_A')} A)"
        )
    if not inter_ok:
        reasons.append(
            f"{cl.get('n_inter_clashes')} peptide-receptor clash(es) "
            f"(max {tol.max_inter_clashes}, min dist {cl.get('min_inter_dist_A')} A)"
        )

    return {
        "valid": all(v for v in checks.values() if v is not None) and not reasons,
        "checks": checks,
        "reasons": reasons,
        "chemistry": tol.chemistry,
        "tolerance_applied": tol.to_dict(),
    }
