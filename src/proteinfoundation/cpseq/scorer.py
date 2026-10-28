"""One shared chemistry-aware evaluator, applied identically to every predictor's output.

The whole point of this module is that it does not know which predictor produced the
structure it is given. Same target preparation, same scorefunction, same peptide
topology, same protonation policy (the shared PyRosetta init flags), same target
flexibility, same finite refinement budget. Anything predictor-specific -- a cyclic
offset, a pocket constraint, a diffusion sample count -- belongs in the prediction
record, not here, and the two are kept in separate files so they cannot be confused.

What it computes, per structure
-------------------------------
**Raw and refined, both kept.** The raw prediction is scored before anything touches it,
then refined under a fixed budget and scored again, and both numbers survive into the
dataset along with how far refinement actually moved the structure. A refined dG with no
raw companion hides whether the model produced a plausible pose or whether the relax
rescued it.

**Interface energetics.** Rosetta `InterfaceAnalyzerMover`: `dG_separated`, `dSASA_int`,
and `dG_separated/dSASA_int * 100`. The normalised form is **null**, never zero or
infinity, when `dSASA_int` is not positive -- a peptide that does not touch the receptor
has no interface to normalise by, and 0 would read as a neutral interface while a
division would read as a spectacular one. Raw energy and raw area are both kept so the
ratio can always be recomputed.

These are Rosetta proxies. They are not measured binding free energies, and nothing here
should be presented as one.

**Interface quality.** Buried unsatisfied polar groups (`delta_unsatHbonds`), interface
hydrogen bonds (`hbonds_int`), and the shape complementarity / packing terms the current
evaluator exposes (`sc_value`, `packstat`) -- the exact InterfeceAnalyzer definitions,
preserved rather than reimplemented.

**Peptide deformation proxy.** The bound cyclic peptide is extracted with its ring and
terminal chemistry intact, scored, then relaxed *without the receptor* under the same
finite budget, and the difference is reported. This is a **local deformation-energy
proxy**: the free reference is a local minimum reached from the bound conformation, not a
globally optimised unbound state, and it is not a thermodynamic free energy. A stronger
reference would need an independently sampled free-CP ensemble, which is a separate
calculation on a predefined subset.

**Restraints.** Refinement restrains to the input coordinates, because the job is to
score the geometry the model produced, not to search for a better one nearby. Those
restraints are recorded, and the reported energies are read with a scorefunction whose
constraint weights are zero, so no restraint term is ever counted as physical energy.

Closure chemistry stays active throughout: the ring bond is declared on the complex
before refinement, on the separated state, and on the isolated peptide. A macrocycle
scored as a linear peptide that happens to sit in a ring is the instrument problem this
whole module exists to avoid.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from proteinfoundation.cpseq.chemistry import CHEMISTRIES, SUPPORTED, capability

# Shared refinement budget. FastRelax with a single repeat and coordinate restraints is
# what `evaluation.rosetta_energy` already uses; keeping the same number means a dG from
# this module and one from the existing design pipeline mean the same thing.
RELAX_REPEATS = 1
SCOREFUNCTION = "ref2015"

# Rosetta terms pulled from InterfaceAnalyzerMover, reusing the existing tuple so the two
# code paths cannot drift.
from proteinfoundation.evaluation.rosetta_energy import (  # noqa: E402
    ROSETTA_INTERFACE_SCORE_KEYS,
    _ensure_pyrosetta_init,
)


def _init_flags_record() -> dict:
    """The shared PyRosetta init, recorded so the protonation/typing policy is explicit."""
    return {
        "scorefunction": SCOREFUNCTION,
        "relax_repeats": RELAX_REPEATS,
        "relax_restraint": (
            "constrain_relax_to_start_coords + closure AtomPairConstraint "
            "(harmonic, sd 0.05 A, on both the complex and the isolated peptide)"
        ),
        # The restraints are carried by a separate scorefunction; every reported energy
        # is read with plain ref2015, whose constraint weights are zero.
        "constraint_weights_in_reported_energy": 0.0,
        "init_flags": (
            "-mute all -use_input_sc -ignore_unrecognized_res -ignore_zero_occupancy false "
            "-load_PDB_components false -relax:default_repeats 2 -no_fconfig "
            "-use_terminal_residues true -in:file:silent_struct_type binary"
        ),
        "notes": (
            "Protonation is Rosetta's default for ref2015 (no explicit protonation "
            "sampling); hydrogens are rebuilt by Rosetta on load. Identical for every "
            "predictor."
        ),
    }


def _seqpos_of_chain(pose, chain: str) -> list[int]:
    info = pose.pdb_info()
    return [i for i in range(1, pose.size() + 1) if info.chain(i) == chain]


def _declare_ring(pose, chemistry: str, seqpos_i: int, seqpos_j: int) -> dict:
    """Declare the closing bond on `pose` at explicit pose positions.

    Separate from `rosetta_energy._declare_cyclic_bond` because that one resolves
    positions through `PDBInfo`, which an extracted single-chain subpose does not
    reliably carry. Here the caller has already resolved them.
    """
    from pyrosetta.rosetta.core.chemical import VariantType
    from pyrosetta.rosetta.core.pose import remove_variant_type_from_pose_residue

    if chemistry == "mainchain":
        # A closing amide needs free N/C connection points, and several variants cap
        # them. The plain terminus variants are the common case, but a CHEMICAL cap is
        # different in kind: an acetylated N-terminus or an amidated/methylated
        # C-terminus is a real blocking group, and a head-to-tail ring through it is
        # not a molecule. Measured on LNR 1elr, whose peptide Rosetta types as
        # MET:AcetylatedNtermProteinFull -- declaring the bond raised
        # "doesnt have connection at N" and killed the process with no traceback.
        #
        # So caps are removed, but their removal is REPORTED: stripping an acetyl group
        # changes what molecule was scored, and a caller comparing against an
        # experimental control needs to know that happened.
        removed = []
        for pos, variants in (
            (seqpos_i, ("LOWER_TERMINUS_VARIANT", "ACETYLATED_NTERMINUS_VARIANT",
                        "ACETYLATED_NTERMINUS_CONNECTION_VARIANT", "METHYLATED_NTERM_VARIANT",
                        "N_ACETYLATION", "ACETYLATION")),
            (seqpos_j, ("UPPER_TERMINUS_VARIANT", "METHYLATED_CTERMINUS_VARIANT",
                        "DIMETHYLATED_CTERMINUS_VARIANT", "CTERM_AMIDATION")),
        ):
            for vname in variants:
                vt = getattr(VariantType, vname, None)
                if vt is None:
                    continue
                if pose.residue(pos).has_variant_type(vt):
                    remove_variant_type_from_pose_residue(pose, vt, pos)
                    removed.append({"seqpos": pos, "variant": vname})
        try:
            pose.conformation().declare_chemical_bond(seqpos_i, "N", seqpos_j, "C")
        except RuntimeError as exc:
            return {
                "declared": False,
                "how": None,
                "reason": f"declare_chemical_bond refused: {exc}",
                "variants_removed": removed,
                "residue_types": [pose.residue(seqpos_i).name(), pose.residue(seqpos_j).name()],
            }
        return {
            "declared": True,
            "how": "declare_chemical_bond(N,C) after stripping terminus/cap variants",
            "variants_removed": removed,
            "chemical_cap_removed": any(
                "ACETYL" in r["variant"] or "METHYL" in r["variant"] or "AMIDATION" in r["variant"]
                for r in removed
            ),
        }

    if chemistry == "disulfide":
        ok = pose.conformation().add_disulfide_bond(seqpos_i, seqpos_j)
        return {
            "declared": bool(ok),
            "how": "add_disulfide_bond",
            "reason": None if ok else (
                f"add_disulfide_bond refused for {seqpos_i}/{seqpos_j} "
                f"({pose.residue(seqpos_i).name3()}/{pose.residue(seqpos_j).name3()})"
            ),
        }

    if chemistry == "isopeptide":
        # Not supported, and not faked. An arbitrary LYS-NZ <-> acid side-chain bond
        # needs a residue-connection patch plain amino-acid types do not carry.
        return {
            "declared": False,
            "how": None,
            "reason": "isopeptide bond cannot be declared in Rosetta (no connection patch)",
        }

    return {"declared": False, "how": None, "reason": f"unknown chemistry {chemistry!r}"}


def _nearest_permitted_dihedral(observed_deg: float, spec) -> float:
    """The arrangement around the closing bond that this structure is already in.

    The torsion restraint targets the arrangement the PREDICTOR produced, not the
    chemistry's nominal one. An amide is planar with two real arrangements, cis and
    trans, and 75 of 150 native CPSea macrocycles sit nearer cis -- so restraining every
    ring to trans would drag half of them through 180 degrees and call the result
    refinement. A disulfide is symmetric at +-90 for the same reason. This is the same
    "nearest permitted arrangement" rule `geometry.closure_geometry` judges by, so the
    restraint holds exactly what the gate will measure.
    """
    alts = list(spec.bond.dihedral_alternatives_deg or (spec.bond.dihedral_deg,))
    if spec.bond.dihedral_symmetric:
        alts += [-a for a in alts]

    def circular_gap(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    return min(alts, key=lambda ref: circular_gap(observed_deg, ref))


def _add_closure_constraint(pose, chemistry: str, seqpos_i: int, seqpos_j: int) -> dict:
    """Pin the closing JUNCTION -- bond, both angles and the torsion -- during relax.

    `declare_chemical_bond` updates the pose's connectivity but gives the minimiser no
    term that holds those two atoms together, and `constrain_relax_to_start_coords`
    turned out to be too weak to stand in for one. Measured, before this existed: a
    head-to-tail ring that Boltz closed to a median 0.010 A of reference came out of
    refinement at 1.149 A -- i.e. refinement OPENED 319 of 319 rings it was supposed to
    preserve, and the refined interface energies were computed on a broken macrocycle.
    (Disulfides went the other way: Rosetta's own disulfide term pulled them from 0.494 A
    to 0.023 A, closing 144 rings the predictor had failed to close. That is a refiner
    artefact too, and the raw/refined split is what makes both visible.)

    Why the angles and the torsion, and not only the bond
    ------------------------------------------------------
    An AtomPairConstraint alone fixes a distance and leaves the rest of the junction
    free, so the minimiser pays for its repulsive term by bending the closure open while
    holding the two atoms the right distance apart. Measured on 24 folds of three LNR
    complexes, before this existed:

    | quantity            | pre-relax          | post-relax           | tolerance   |
    |---------------------|--------------------|----------------------|-------------|
    | closure angle dev   | median 0.99 deg    | median 16.63 deg     | 7.54 deg    |
    | closure torsion dev | median 2.35 deg    | median 15.23 deg     | 8.44 deg    |
    | closure bond length | 1.33 A             | 1.33 A               | 1.23-1.43 A |
    | angles over tol     | **0 of 48**        | **37 of 48**         |             |

    The bond -- the one restrained quantity -- is held perfectly, and every violation is
    manufactured by the relax: Boltz-2's junction geometry was clean in 48 of 48
    measurements. On `lnr_5vao` the predictor produced 4 valid folds and refinement broke
    all 4. So this is not a gate masking a real defect; it is the scorer being stopped
    from creating one.

    The restraints are artificial, so they are scored with a SEPARATE scorefunction and
    never contribute to a reported energy; `restraints` in the output records what was
    applied and at what reference values.
    """
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring.constraints import (
        AngleConstraint,
        AtomPairConstraint,
        DihedralConstraint,
    )
    from pyrosetta.rosetta.core.scoring.func import CircularHarmonicFunc, HarmonicFunc
    from pyrosetta.rosetta.numeric import dihedral_degrees

    spec = CHEMISTRIES[chemistry]
    atom_i, atom_j = spec.bond.atom_i, spec.bond.atom_j
    outer_i, outer_j = spec.bond.outer_i, spec.bond.outer_j
    if chemistry == "isopeptide":
        return {"applied": False, "reason": "isopeptide is not scored at all"}
    try:
        res_i, res_j = pose.residue(seqpos_i), pose.residue(seqpos_j)
        need = ((res_i, atom_i), (res_j, atom_j), (res_i, outer_i), (res_j, outer_j))
        missing = [a for r, a in need if not r.has(a)]
        if missing:
            return {"applied": False, "reason": f"missing {','.join(sorted(set(missing)))}"}

        id_bi = AtomID(res_i.atom_index(atom_i), seqpos_i)
        id_bj = AtomID(res_j.atom_index(atom_j), seqpos_j)
        id_oi = AtomID(res_i.atom_index(outer_i), seqpos_i)
        id_oj = AtomID(res_j.atom_index(outer_j), seqpos_j)

        pose.add_constraint(AtomPairConstraint(
            id_bi, id_bj,
            # sd 0.05 A: tight enough to hold a covalent bond, loose enough not to
            # dominate the local geometry the relax is otherwise free to fix.
            HarmonicFunc(spec.bond.length_A, 0.05),
        ))

        # Angles, in radians. sd 0.05 rad (2.9 deg) is the angular analogue of the
        # bond's 0.05 A: a deviation at the gate's 7.54 deg tolerance then costs about
        # 7 REU, enough to hold the junction against the repulsive term without
        # flattening the side-chain repacking the relax exists to do.
        ang_sd = 0.05
        # angle_i is at endpoint i (outer_i - bond_i - bond_j), angle_j at endpoint j --
        # the same two angles, in the same order, that `closure_geometry` measures.
        pose.add_constraint(AngleConstraint(
            id_oi, id_bi, id_bj,
            HarmonicFunc(math.radians(spec.bond.angle_i_deg), ang_sd),
        ))
        pose.add_constraint(AngleConstraint(
            id_bi, id_bj, id_oj,
            HarmonicFunc(math.radians(spec.bond.angle_j_deg), ang_sd),
        ))

        # Torsion, targeted at the arrangement this structure is already in. Circular,
        # because a torsion wraps and a plain harmonic would punish -179 for being 358
        # degrees from +179.
        xyz = pose.xyz
        observed = float(dihedral_degrees(xyz(id_oi), xyz(id_bi), xyz(id_bj), xyz(id_oj)))
        target = _nearest_permitted_dihedral(observed, spec)
        tors_sd = 0.10  # 5.7 deg: looser than the angles, as the gate's window is wider
        pose.add_constraint(DihedralConstraint(
            id_oi, id_bi, id_bj, id_oj,
            CircularHarmonicFunc(math.radians(target), tors_sd),
        ))

        return {
            "applied": True,
            "type": "AtomPair + 2x Angle + Dihedral",
            "atoms": f"{outer_i}-{atom_i}({seqpos_i})-{atom_j}({seqpos_j})-{outer_j}",
            "target_length_A": spec.bond.length_A,
            "harmonic_sd_A": 0.05,
            "target_angle_i_deg": spec.bond.angle_i_deg,
            "target_angle_j_deg": spec.bond.angle_j_deg,
            "harmonic_sd_rad": ang_sd,
            "observed_dihedral_deg": round(observed, 2),
            "target_dihedral_deg": target,
            "dihedral_sd_rad": tors_sd,
            "excluded_from_reported_energy": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {"applied": False, "reason": f"{type(exc).__name__}: {exc}"}


def _ca_coords(pose, seqpos: list[int]) -> np.ndarray:
    out = []
    for i in seqpos:
        res = pose.residue(i)
        if res.has("CA"):
            xyz = res.xyz("CA")
            out.append([xyz.x, xyz.y, xyz.z])
    return np.asarray(out, dtype=np.float64)


def _rmsd(a: np.ndarray, b: np.ndarray) -> float | None:
    """Plain (already-superposed) RMSD. Used only on before/after of the SAME pose."""
    if a.size == 0 or a.shape != b.shape:
        return None
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def _normalised_dg(dg: float | None, dsasa: float | None) -> float | None:
    """dG_separated / dSASA_int * 100, or None where the area makes it undefined.

    Explicitly None rather than 0 or inf: a peptide that makes no interface has no area
    to normalise by, 0 would read as a neutral interface, and a division by a tiny area
    would read as a spectacular one.
    """
    if dg is None or dsasa is None:
        return None
    if not np.isfinite(dg) or not np.isfinite(dsasa) or dsasa <= 0.0:
        return None
    return round(float(dg) / float(dsasa) * 100.0, 6)


def score_structure(
    pdb_path: str | Path,
    *,
    binder_chain: str,
    receptor_chains: list[str],
    chemistry: str,
    requested_sequence: str,
    refined_out_path: str | Path | None = None,
    isolated_out_path: str | Path | None = None,
    deformation_cache: dict | None = None,
) -> dict:
    """Score one predicted complex. Returns a record; raises only on programmer error.

    A Rosetta failure, an undeclarable ring or a missing atom produces a record whose
    `status` says so. None of those is successful closure, and none of them is an
    experimental nonbinder label either.
    """
    pdb_path = Path(pdb_path)
    rec: dict[str, Any] = {
        "pdb_path": str(pdb_path),
        "chemistry": chemistry,
        "binder_chain": binder_chain,
        "receptor_chains": list(receptor_chains),
        "protocol": _init_flags_record(),
        "status": "unscored",
    }

    scorer_cap = capability("rosetta", chemistry)
    rec["scorer_chemistry_status"] = scorer_cap.status
    if scorer_cap.status != SUPPORTED:
        # The structure may be a perfectly good prediction; our scorer simply cannot
        # represent its ring. That is a scoring gap, recorded as one.
        rec["status"] = "scoring_chemistry_unsupported"
        rec["reason"] = scorer_cap.note
        return rec

    try:
        _ensure_pyrosetta_init()
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "pyrosetta_unavailable"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        return rec

    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "pose_load_failed"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        return rec

    binder_pos = _seqpos_of_chain(pose, binder_chain)
    rec["binder_length"] = len(binder_pos)
    if len(binder_pos) < 2:
        rec["status"] = "binder_chain_missing"
        rec["reason"] = f"chain {binder_chain!r} has {len(binder_pos)} residues in the pose"
        return rec

    observed_seq = "".join(pose.residue(i).name1() for i in binder_pos)
    rec["sequence_requested"] = requested_sequence
    rec["sequence_observed"] = observed_seq
    rec["sequence_matches"] = observed_seq == requested_sequence

    # --- ring declaration on the complex -------------------------------------------
    seqpos_i, seqpos_j = binder_pos[0], binder_pos[-1]
    ring = _declare_ring(pose, chemistry, seqpos_i, seqpos_j)
    rec["ring_declared_complex"] = ring
    if not ring["declared"]:
        rec["status"] = "ring_not_declared"
        rec["reason"] = ring.get("reason")
        return rec

    sf = pyrosetta.create_score_function(SCOREFUNCTION)
    interface = "".join([binder_chain]) + "_" + "".join(receptor_chains)

    def _make_mover():
        """InterfaceAnalyzerMover across PyRosetta API versions.

        PyRosetta 2026.39 -- the build installed here -- dropped the plain-string
        constructor in favour of a `DockingPartners` object, so `InterfaceAnalyzerMover
        ("B_A")` raises a 27-overload TypeError. `evaluation.rosetta_energy` still passes
        a string, which means the repo's existing Rosetta path does not run on this
        version either. Try the typed factory first and fall back to the string form so
        this works on both.
        """
        try:
            from pyrosetta.rosetta.core.pose import DockingPartners

            return InterfaceAnalyzerMover(
                DockingPartners.docking_partners_from_string(interface)
            )
        except (ImportError, AttributeError, TypeError):
            return InterfaceAnalyzerMover(interface)

    def analyze(p, *, with_packstat: bool) -> dict:
        mover = _make_mover()
        mover.set_pack_separated(True)
        # packstat must be asked for: left off it returns a flat 0.0 for every
        # structure -- an uncomputed default, not a value, which would read dataset-wide
        # as a perfectly unpacked interface. It is computed on BOTH poses: measured at
        # 1.5s vs 2.2s without it on a 220-residue complex, i.e. free. (It was briefly
        # restricted to the refined pose on the assumption that a Monte-Carlo SASA
        # calculation would dominate; timing said otherwise.)
        mover.set_compute_packstat(bool(with_packstat))
        mover.set_compute_interface_sc(True)
        mover.apply(p)
        out = {k: float(p.scores[k]) for k in ROSETTA_INTERFACE_SCORE_KEYS if k in p.scores}
        if not with_packstat:
            out.pop("packstat", None)
        return out

    try:
        raw_pose = pyrosetta.Pose(pose)
        raw_scores = analyze(pyrosetta.Pose(pose), with_packstat=True)
        rec["raw"] = {
            "total_score": round(float(sf(raw_pose)), 3),
            **{k: round(v, 3) for k, v in raw_scores.items()},
        }
        rec["raw"]["dG_separated_per_1000A2"] = _normalised_dg(
            raw_scores.get("dG_separated"), raw_scores.get("dSASA_int")
        )
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "raw_scoring_failed"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        return rec

    # --- shared finite refinement ---------------------------------------------------
    ca_before_binder = _ca_coords(pose, binder_pos)
    receptor_pos = [p for ch in receptor_chains for p in _seqpos_of_chain(pose, ch)]
    ca_before_rec = _ca_coords(pose, receptor_pos)
    try:
        # Two scorefunctions, deliberately. `sf_relax` carries the restraint weights the
        # refinement needs; `sf` (plain ref2015, zero constraint weight) is the only one
        # used for any reported energy, so no artificial restraint term can leak into a
        # physical proxy.
        from pyrosetta.rosetta.core.scoring import ScoreType

        sf_relax = pyrosetta.create_score_function(SCOREFUNCTION)
        # All three restraint terms, or the angle and torsion constraints would be
        # added to the pose and then scored at weight zero -- present in the record and
        # absent from the minimiser, which is the failure mode this fix exists to end.
        sf_relax.set_weight(ScoreType.atom_pair_constraint, 1.0)
        sf_relax.set_weight(ScoreType.angle_constraint, 1.0)
        sf_relax.set_weight(ScoreType.dihedral_constraint, 1.0)
        rec["restraints"] = _add_closure_constraint(pose, chemistry, seqpos_i, seqpos_j)

        relax = FastRelax(sf_relax, RELAX_REPEATS)
        # Restrain to the input coordinates: scoring the geometry the model produced,
        # not searching for a better one nearby. Recorded in `protocol`.
        relax.constrain_relax_to_start_coords(True)
        relax.apply(pose)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "refinement_failed"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        return rec

    try:
        ref_scores = analyze(pyrosetta.Pose(pose), with_packstat=True)
        rec["refined"] = {
            # `sf` carries zero constraint weights, so the restraint term used during
            # relax contributes nothing to this reported energy.
            "total_score": round(float(sf(pose)), 3),
            **{k: round(v, 3) for k, v in ref_scores.items()},
        }
        rec["refined"]["dG_separated_per_1000A2"] = _normalised_dg(
            ref_scores.get("dG_separated"), ref_scores.get("dSASA_int")
        )
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "refined_scoring_failed"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        return rec

    rec["refinement_displacement"] = {
        "binder_ca_rmsd_A": _rmsd(ca_before_binder, _ca_coords(pose, binder_pos)),
        "receptor_ca_rmsd_A": _rmsd(ca_before_rec, _ca_coords(pose, receptor_pos)),
        "total_score_delta": round(
            rec["refined"]["total_score"] - rec["raw"]["total_score"], 3
        ),
    }

    if refined_out_path:
        Path(refined_out_path).parent.mkdir(parents=True, exist_ok=True)
        pose.dump_pdb(str(refined_out_path))
        rec["refined_pdb_path"] = str(refined_out_path)

    # --- peptide deformation proxy ---------------------------------------------------
    rec["deformation"] = _deformation_proxy(
        pose=pose,
        binder_chain=binder_chain,
        chemistry=chemistry,
        sequence=observed_seq,
        sf=sf,
        isolated_out_path=isolated_out_path,
        cache=deformation_cache,
    )

    rec["status"] = "ok"
    return rec


def _deformation_proxy(
    *,
    pose,
    binder_chain: str,
    chemistry: str,
    sequence: str,
    sf,
    isolated_out_path: str | Path | None,
    cache: dict | None,
) -> dict:
    """Bound-conformation vs locally-relaxed-free intramolecular energy of the peptide.

    A **local** proxy by construction: the free reference is relaxed from the bound
    conformation, so it is the nearest free minimum, not the global unbound state. Named
    that way in the output so it cannot be read as a thermodynamic quantity.
    """
    import pyrosetta

    out: dict[str, Any] = {
        "label": "local_deformation_energy_proxy",
        "is_thermodynamic_free_energy": False,
        "free_reference": "relaxed from the bound conformation, same finite budget",
        "status": "unscored",
    }
    try:
        chain_idx = None
        info = pose.pdb_info()
        for i in range(1, pose.size() + 1):
            if info.chain(i) == binder_chain:
                chain_idx = pose.chain(i)
                break
        if chain_idx is None:
            out["status"] = "binder_chain_not_found"
            return out

        peptide = pose.split_by_chain(chain_idx)
        n = peptide.size()
        if n < 2:
            out["status"] = "peptide_too_short"
            return out

        # Re-declare the SAME ring on the isolated peptide. Splitting a chain out of a
        # pose does not carry the declared bond with it, and scoring the extracted
        # macrocycle as a linear peptide would make every ring look strain-free.
        ring = _declare_ring(peptide, chemistry, 1, n)
        out["ring_declared_isolated"] = ring
        if not ring["declared"]:
            out["status"] = "ring_not_declared_isolated"
            out["reason"] = ring.get("reason")
            return out

        e_bound = float(sf(peptide))
        out["bound_conformation_energy"] = round(e_bound, 3)

        coord_key = hashlib.sha256(
            _ca_coords(peptide, list(range(1, n + 1))).round(3).tobytes()
        ).hexdigest()[:16]
        cache_key = f"{sequence}|{chemistry}|{coord_key}"
        out["cache_key"] = cache_key

        if cache is not None and cache_key in cache:
            e_free = cache[cache_key]["free_energy"]
            out["cache_hit"] = True
        else:
            out["cache_hit"] = False
            free = pyrosetta.Pose(peptide)
            from pyrosetta.rosetta.core.scoring import ScoreType
            from pyrosetta.rosetta.protocols.relax import FastRelax

            # No coordinate restraints here -- the free reference is meant to move. But
            # the RING must still be held, or the "free cyclic peptide" reference is a
            # linear peptide and the deformation proxy compares two different molecules.
            out["free_restraints"] = _add_closure_constraint(free, chemistry, 1, n)
            sf_free = pyrosetta.create_score_function(SCOREFUNCTION)
            sf_free.set_weight(ScoreType.atom_pair_constraint, 1.0)
            sf_free.set_weight(ScoreType.angle_constraint, 1.0)
            sf_free.set_weight(ScoreType.dihedral_constraint, 1.0)
            relax = FastRelax(sf_free, RELAX_REPEATS)
            relax.apply(free)
            e_free = float(sf(free))
            if isolated_out_path:
                Path(isolated_out_path).parent.mkdir(parents=True, exist_ok=True)
                free.dump_pdb(str(isolated_out_path))
                out["isolated_relaxed_pdb_path"] = str(isolated_out_path)
            if cache is not None:
                cache[cache_key] = {"free_energy": e_free, "sequence": sequence,
                                    "chemistry": chemistry}

        out["free_relaxed_energy"] = round(float(e_free), 3)
        out["deformation_energy"] = round(e_bound - float(e_free), 3)
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "deformation_failed"
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def receptor_aligned_peptide_rmsd(
    pdb_a: str | Path,
    pdb_b: str | Path,
    *,
    binder_chain: str,
    receptor_chains: list[str],
) -> dict:
    """Peptide CA-RMSD between two structures after superposing on the RECEPTOR.

    The receptor is what defines the frame. Superposing on the peptide instead would
    minimise exactly the quantity being measured and report two completely different
    docking solutions as being in agreement -- it would hide docking disagreement, which
    is the thing worth knowing when two predictors differ.
    """
    from proteinfoundation.cpseq.geometry import read_chain

    def coords(path, chains):
        out = []
        for ch in chains:
            for _, atoms in read_chain(path, ch):
                if "CA" in atoms:
                    out.append(atoms["CA"])
        return np.asarray(out, dtype=np.float64)

    rec_a, rec_b = coords(pdb_a, receptor_chains), coords(pdb_b, receptor_chains)
    pep_a, pep_b = coords(pdb_a, [binder_chain]), coords(pdb_b, [binder_chain])
    if rec_a.shape != rec_b.shape or rec_a.size == 0:
        return {"status": "receptor_mismatch", "n_receptor_a": len(rec_a), "n_receptor_b": len(rec_b)}
    if pep_a.shape != pep_b.shape or pep_a.size == 0:
        return {"status": "peptide_mismatch", "n_peptide_a": len(pep_a), "n_peptide_b": len(pep_b)}

    # Kabsch on the receptor only; the transform is then applied to the peptide.
    ca, cb = rec_a.mean(0), rec_b.mean(0)
    cov = (rec_a - ca).T @ (rec_b - cb)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    pep_b_moved = (pep_b - cb) @ rot.T + ca

    return {
        "status": "ok",
        "receptor_ca_rmsd_after_alignment_A": round(
            float(np.sqrt((((rec_b - cb) @ rot.T + ca - rec_a) ** 2).sum(1).mean())), 3
        ),
        "peptide_ca_rmsd_receptor_aligned_A": round(
            float(np.sqrt(((pep_a - pep_b_moved) ** 2).sum(1).mean())), 3
        ),
        "n_receptor_ca": int(len(rec_a)),
        "n_peptide_ca": int(len(pep_a)),
        "alignment_frame": "receptor",
    }
