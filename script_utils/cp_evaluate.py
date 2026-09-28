"""Turn a candidate pose into a contract row, and score the paired LP baseline.

Split from `cp_search` so the grading path can be imported without the search engines --
`score_sdedit_contract.py` grades structures this repo produced months ago and has no use for
GenKIC.

The paired baseline is the part that is easy to get wrong. The contract asks for
"CP score minus original-LP score using identical receptor preparation, scoring, and
relaxation settings", and `refine_lp_baseline` exists to make "identical" literal: same score
function, same coordinate restraints, same packable set, same relax repeats, differing only in
that no ring is declared. Scoring an unrelaxed crystal LP against a relaxed CP would credit
cyclization with the relaxation's own energy gain, which this repo has measured at hundreds of
REU (+773.6 raw vs -42.8 relaxed on one CPSea design) -- two orders of magnitude larger than
the +5 REU the contract is trying to resolve.
"""

from __future__ import annotations

import statistics

import cp_contract as C
import cp_search as S
import cp_topology as T


def _packable_selector(target):
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector, NeighborhoodResidueSelector, OrResidueSelector)

    binder_sel = ChainSelector(target.binder_chain)
    near_sel = NeighborhoodResidueSelector(binder_sel, 6.0, True)
    return OrResidueSelector(binder_sel, near_sel)


def _add_coord_restraints(pose, target, coord_sd: float) -> None:
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc

    anchor = AtomID(1, pose.fold_tree().root())
    for p in target.binder_pos:
        if not pose.residue(p).has("CA"):
            continue
        pose.add_constraint(CoordinateConstraint(
            AtomID(pose.residue(p).atom_index("CA"), p), anchor,
            target.pose.residue(p).xyz("CA"), HarmonicFunc(0.0, coord_sd)))


def _pack_and_relax(pose, target, sf, relax_repeats: int) -> None:
    import pyrosetta.rosetta as R

    packable = _packable_selector(target)
    tf = R.core.pack.task.TaskFactory()
    tf.push_back(R.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(R.core.pack.task.operation.RestrictToRepacking())
    # flip=True turns the selection into "everything NOT packable", which is what gets frozen.
    tf.push_back(R.core.pack.task.operation.OperateOnResidueSubset(
        R.core.pack.task.operation.PreventRepackingRLT(), packable, True))
    packer = R.protocols.minimization_packing.PackRotamersMover(sf)
    packer.task_factory(tf)
    packer.apply(pose)

    mm = R.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)
    for p in target.binder_pos:
        mm.set_bb(p, True)
        mm.set_chi(p, True)
    mask = packable.apply(pose)
    for p in target.receptor_pos:
        if mask[p]:
            mm.set_chi(p, True)
    relax = R.protocols.relax.FastRelax(sf, int(relax_repeats))
    relax.set_movemap(mm)
    relax.apply(pose)


def refine_candidate(pose, target, i_pos, j_pos, chem, atoms,
                     relax_repeats: int = 1, coord_sd: float = 0.5) -> None:
    """Pack permitted side chains and relax, in place, holding the declared ring.

    Ring constraints are (re-)added here because ref2015 carries zero weight on the
    constraint terms: without `constrained_score_function` the minimizer ignores them and
    `fa_rep` is free to push the declared bond back open, handing back a "cyclic" structure
    whose ring has quietly reopened. Design is off -- sequence is fixed within a search, so
    the packer may only choose rotamers.
    """
    # The engines can leave the PDBInfo obsolete or absent (GenKIC rebuilds the pose), and
    # everything below addresses residues by chain: the packable ChainSelector here, and
    # InterfaceAnalyzerMover's interface string later. Repaired once, at the choke point.
    if not T.ensure_pdb_info(pose, target.pose):
        raise RuntimeError(
            f"cannot rebuild a valid PDBInfo before relax (pose {pose.size()} residues vs "
            f"reference {target.pose.size()}); refusing to relax a pose whose chains cannot "
            "be resolved")
    pose.remove_constraints()
    T.add_ring_constraints(pose, i_pos, j_pos, chem, atoms)
    _add_coord_restraints(pose, target, coord_sd)
    sf = S.constrained_score_function(cst_weight=5.0, coord_weight=1.0)
    _pack_and_relax(pose, target, sf, relax_repeats)
    pose.remove_constraints()


def refine_lp_baseline(target, relax_repeats: int = 1, coord_sd: float = 0.5):
    """Relax the ORIGINAL LP down the identical pack/relax path, with no ring declared."""
    import pyrosetta

    pose = pyrosetta.Pose(target.pose)
    _add_coord_restraints(pose, target, coord_sd)
    sf = S.constrained_score_function(cst_weight=5.0, coord_weight=1.0)
    _pack_and_relax(pose, target, sf, relax_repeats)
    pose.remove_constraints()
    return pose


def interface_metrics_or_nan(pose, target) -> dict:
    """`C.interface_metrics`, but a Rosetta failure yields NaN instead of killing the batch."""
    try:
        return C.interface_metrics(pose, target.binder_chain, target.receptor_chains)
    except Exception as exc:                                       # noqa: BLE001
        return {"dG_separated": float("nan"), "interface_error": str(exc)[:200]}


def _median(xs):
    good = [x for x in xs if x == x]
    return statistics.median(good) if good else float("nan")


def _spread(xs):
    good = [x for x in xs if x == x]
    return (max(good) - min(good)) if len(good) > 1 else 0.0


def lp_baseline_metrics(target, repeats: int = 1, relax_repeats: int = 1,
                        seed_base: int = 9_000_000) -> dict:
    """Interface dG and peptide internal energy for the relaxed original LP.

    `repeats` independent relaxations. The median feeds the paired delta and the spread is
    reported next to it, so a delta smaller than the baseline's own run-to-run noise is
    visible as such rather than read as a result.
    """
    import pyrosetta

    plain = pyrosetta.create_score_function("ref2015")
    dgs, internals, totals = [], [], []
    for r in range(max(1, repeats)):
        S.set_rng_seed(seed_base + r)
        pose = refine_lp_baseline(target, relax_repeats=relax_repeats)
        dgs.append(interface_metrics_or_nan(pose, target).get("dG_separated", float("nan")))
        internals.append(C.peptide_internal_energy(pose, target.binder_pos, plain))
        totals.append(float(plain(pose)))
    return {
        "lp_n_repeats": int(max(1, repeats)),
        "lp_dG_separated_median": round(_median(dgs), 3),
        "lp_dG_separated_spread": round(_spread(dgs), 3),
        "lp_dG_separated_all": [round(x, 3) for x in dgs],
        "lp_peptide_internal_median": round(_median(internals), 3),
        "lp_peptide_internal_spread": round(_spread(internals), 3),
        "lp_total_score_median": round(_median(totals), 3),
    }


def _anchors_match(pose, i_pos: int, j_pos: int, spec) -> bool:
    return (pose.residue(i_pos).name3() == spec.aa_i
            and pose.residue(j_pos).name3() == spec.aa_j)


def evaluate_candidate(
    target,
    pose,
    spec,
    atoms: dict,
    window: dict,
    i_pos: int,
    j_pos: int,
    *,
    stage: str,
    protected_local=(),
    lp_baseline: dict | None = None,
    with_energy: bool = True,
) -> dict:
    """Every contract metric for one candidate pose, flat.

    `stage` is `"pre_relax"` or `"post_relax"`. Acceptance applies to the post-relax row; the
    pre-relax row is kept beside it so a candidate that only qualified through relaxation is
    identifiable rather than indistinguishable from one that was always good.

    Energy is optional because it dominates cost: the search stage grades thousands of
    candidates on geometry and sequence alone, and only a ranked shortlist pays for
    `InterfaceAnalyzerMover` plus repeated relaxation.
    """
    import pyrosetta

    chem = T.chemistry(spec.chemistry)
    bp = target.binder_pos
    released = set(window["released_local"])
    row: dict = {
        "example_id": target.example_id,
        "stage": stage,
        "chemistry": spec.chemistry,
        "anchor_aa_i": spec.aa_i,
        "anchor_aa_j": spec.aa_j,
        "k_nterm": spec.k_nterm,
        "m_cterm": spec.m_cterm,
        "n_released": window["n_released"],
        "released_local": window["released_local"],
        "extra_mutations": {str(k): v for k, v in sorted(spec.extra_mutations.items())},
        "spec_label": spec.label(),
        "cyc_i_local": 0,
        "cyc_j_local": target.length - 1,
        "peptide_length": target.length,
    }

    row.update(T.ring_bond_geometry(pose, i_pos, j_pos, chem, atoms))
    row["requested_chemistry"] = spec.chemistry
    row["requested_chemistry_satisfied"] = bool(
        row["bond_is_declared"] and _anchors_match(pose, i_pos, j_pos, spec))

    row.update(C.sequence_metrics(
        target.orig_sequence, C.peptide_sequence(pose, bp), protected_local))

    residual = C.receptor_frame_residual(pose, target.pose, target.receptor_pos)
    if residual == residual and residual > C.RECEPTOR_FRAME_TOL_A:
        residual = C.align_on_receptor(pose, target.pose, target.receptor_pos)
        row["receptor_realigned"] = True
    else:
        row["receptor_realigned"] = False
    row["receptor_frame_residual_A"] = round(residual, 4)
    row["ca_rmsd_A"] = round(C.ca_rmsd(pose, target.pose, bp), 3)
    iface_pos = target.interface_pose_positions()
    row["interface_ca_rmsd_A"] = round(C.ca_rmsd(pose, target.pose, iface_pos), 3)
    row["n_interface_residues"] = len(iface_pos)
    dev = C.per_residue_ca_deviation(pose, target.pose, bp)
    row["per_residue_ca_dev_A"] = dev
    row["interior_max_ca_dev_A"] = round(
        max((d for li, d in enumerate(dev) if li not in released), default=0.0), 3)

    heavy = C.heavy_atom_contacts(pose, bp, target.receptor_pos)
    ca = C.ca_contacts(pose, bp, target.receptor_pos)
    row["n_contacts_orig_heavy45"] = len(target.orig_contacts_heavy)
    row["n_contacts_cand_heavy45"] = len(heavy)
    row["contact_retention_heavy45"] = round(C.retention(target.orig_contacts_heavy, heavy), 4)
    row["contact_retention_ca10"] = round(C.retention(target.orig_contacts_ca, ca), 4)

    row.update(C.clash_metrics(pose, bp, target.receptor_pos, ring_pair=(i_pos, j_pos)))
    row.update(C.stereochemistry_metrics(pose, target.pose, bp))

    if with_energy:
        plain = pyrosetta.create_score_function("ref2015")
        for key, value in interface_metrics_or_nan(pose, target).items():
            row[f"cp_{key}"] = round(value, 3) if isinstance(value, float) else value
        row["cp_peptide_internal"] = round(
            C.peptide_internal_energy(pose, bp, plain, ring={"chem": chem, "atoms": atoms}), 3)
        row["cp_total_score"] = round(float(plain(pose)), 3)
        if lp_baseline:
            row.update(lp_baseline)
            dg_cp = row.get("cp_dG_separated", float("nan"))
            dg_lp = lp_baseline.get("lp_dG_separated_median", float("nan"))
            row["ddg_interface_reu"] = (round(dg_cp - dg_lp, 3)
                                        if dg_cp == dg_cp and dg_lp == dg_lp else float("nan"))
            i_cp = row.get("cp_peptide_internal", float("nan"))
            i_lp = lp_baseline.get("lp_peptide_internal_median", float("nan"))
            row["d_peptide_internal_reu"] = (round(i_cp - i_lp, 3)
                                             if i_cp == i_cp and i_lp == i_lp else float("nan"))

    row.update(C.accept(row))
    row.update(C.tier_flags(row))
    return row


def shortlist_key(row: dict) -> tuple:
    """Ranking key for the search stage: contract-relevant quality, best first.

    Energy is deliberately absent: it is not computed at this stage, and ranking on a cheap
    proxy is what `README_LP2CP_PRECHECK` measured as uninformative (contact retention vs
    interface dG, r^2 0.004, residual SD 19.44 REU against a dG SD of 19.47). This key decides
    who gets *measured*, not who wins.
    """
    def f(key, default):
        v = row.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return default if v != v else v

    gates = (
        bool(row.get("bond_ok")),
        bool(row.get("clashes_ok")),
        bool(row.get("stereochem_ok")),
        f("ca_rmsd_A", 99.0) <= C.MAX_CA_RMSD_A,
        f("interface_ca_rmsd_A", 99.0) <= C.MAX_INTERFACE_CA_RMSD_A,
        f("contact_retention_heavy45", 0.0) >= C.MIN_CONTACT_RETENTION,
    )
    return (
        -sum(gates),
        -f("contact_retention_heavy45", 0.0),
        f("interface_ca_rmsd_A", 99.0),
        f("ca_rmsd_A", 99.0),
        f("n_substitutions", 99),
    )
