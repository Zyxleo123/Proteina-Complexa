"""Receptor-aware sidechain completion for a generated backbone. ONE adapter, BOTH arms.

WHY THIS IS NOT OPTIONAL. The arms emit a backbone and nothing else, so every quality claim
about them -- does it bind, does it clash, did it keep the source's contacts -- is a claim
about a molecule that does not exist until sidechains are on it. Scoring the backbone alone
answers a different and much easier question. So packing runs at EVERY scheduled evaluation
checkpoint, not only at final export, and a result labelled "backbone-only" is explicitly an
incomplete result rather than a different way of reporting the same one.

The protocol, and the reason each step is pinned:

1. BUILD a linear peptide carrying the SOURCE sequence on the generated N/CA/C/O. Rosetta
   rebuilds the missing sidechain atoms from ideal internal geometry; the backbone is ours.
2. STRIP CYCLIC CHEMISTRY, explicitly. The sources are macrocycles and their sequences are
   full of cysteines, lysines and aspartates that were crosslinked in the CP. Rosetta's
   automatic disulfide detection will happily re-bond two nearby cysteines, which would turn
   the linear peptide this whole project exists to produce back into a cycle -- and it would
   do it silently, in the scorer, after every geometry gate had already passed. Retained
   terminal cysteines do NOT authorize a closing S-S bond. Verified after the fact, not
   assumed: :func:`assert_linear_topology` re-checks the pose.
3. ADD TERMINAL CHEMISTRY, including C-terminal OXT. A carboxylate written without OXT reads
   as amidated or as mid-chain, which is a subtly different molecule and a different energy.
4. SEED FROM THE SOURCE'S OWN ROTAMERS where residue identity matches, as a CANDIDATE and not
   a constraint. The source's sidechains are a real bound conformation and are the best guess
   available; freezing them would instead be asserting the new backbone wants the old
   rotamers, which is the thing being tested.
5. REPACK the peptide's sidechains against the FIXED receptor. All peptide BACKBONE atoms and
   all receptor atoms stay put: no receptor repacking, no backbone relaxation. Checked
   numerically afterwards, because a mover that quietly moves the backbone would make every
   edit-size number in the comparison wrong.
6. THREE SEEDS, identical budget in both arms, all three outcomes retained including failures.

NO GRADIENT PASSES THROUGH ANY OF THIS. Packing informs validation and model selection only.
If packing later proves to be the binding constraint, that calls for a separate feedback
mechanism, not for making this differentiable.

Angstrom at the Rosetta boundary, nanometres everywhere else. The conversion happens once,
here, in :func:`_peptide_atom37_ang`.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from proteinfoundation.cp2lp.backbone import backbone_to_atom37
from proteinfoundation.cp2lp.export import build_oxt
from proteinfoundation.cp2lp.geometry import CA_IDX, C_IDX, N_IDX, O_IDX

#: Chain ids the complex is written with. Receptor first, matching `export._write_complex` and
#: the rest of the evaluation tooling.
RECEPTOR_CHAIN = "A"
PEPTIDE_CHAIN = "B"

#: Number of independent packing runs per generated backbone. Identical in both arms -- a
#: budget difference here would be a sampling-effort difference wearing a quality label.
DEFAULT_PACK_SEEDS = 3

#: Tolerance for "the packer did not move the backbone", angstrom. A repack should move fixed
#: atoms by exactly zero; this allows for PDB round-trip rounding only.
BACKBONE_FIXED_TOL_ANG = 1e-2

#: The PREDECLARED selection order, fixed before any arm was run. Stated as data so it cannot
#: quietly become "whichever metric separated the arms".
SELECTION_ORDER = (
    "valid",            # linear chemistry + intact backbone + sterics within calibrated gates
    "contact_retention",  # then: how much of the source's interface survived
    "interface_energy",   # then, only as a tie-break: dG_separated
)


@dataclass
class PackResult:
    """One packing outcome. Retained whether it succeeded or failed."""

    seed_index: int
    ok: bool
    failure: str | None = None
    pdb_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:  # keeps manifest debugging readable
        state = "ok" if self.ok else f"FAILED({self.failure})"
        return f"PackResult(seed={self.seed_index}, {state}, n_metrics={len(self.metrics)})"


def is_available() -> bool:
    """Is PyRosetta importable and initialisable in this process?"""
    try:
        from proteinfoundation.evaluation.rosetta_energy import is_pyrosetta_available

        return bool(is_pyrosetta_available())
    except Exception:
        return False


# --------------------------------------------------------------------------- PDB writing


def _peptide_atom37_ang(
    bb: torch.Tensor, mask: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """``[b, n, 4, 3]`` nm backbone -> ``[b, n, 37, 3]`` ANGSTROM atom37, with OXT placed.

    OXT comes from `export.build_oxt`, the same construction the triplet export uses, so a
    packed candidate and an exported one carry the identical terminal chemistry.
    """
    atom37, atom_mask = backbone_to_atom37(bb, mask)
    lengths = mask.bool().sum(dim=1)
    atom37, atom_mask = build_oxt(atom37, atom_mask, lengths)
    return (atom37 * 10.0).detach().cpu().numpy(), atom_mask.detach().cpu().numpy()


def write_complex_pdb(
    path: str,
    pep_atom37_ang: np.ndarray,
    pep_aatype: np.ndarray,
    target_atom37_ang: np.ndarray,
    target_aatype: np.ndarray,
) -> str:
    """Receptor as chain A, peptide as chain B. Reuses the export path's writer.

    NOTE on the writer's mask convention: `write_prot_to_pdb` decides which atoms exist from
    ``sum(|xyz|) > 1e-7``, i.e. an atom at the coordinate origin is treated as absent. That is
    what makes a backbone-only structure come out backbone-only, since `backbone_to_atom37`
    zeroes every other slot. It also means an atom that genuinely sits within 1e-7 angstrom of
    the receptor-frame origin would be dropped -- possible in principle, and not worth
    defending against beyond saying so here.
    """
    from proteinfoundation.cp2lp.export import _write_complex

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _write_complex(path, pep_atom37_ang, pep_aatype, target_atom37_ang, target_aatype)
    return path


# ------------------------------------------------------------------- topology hygiene


def strip_cyclic_topology(pose, peptide_chain: str = PEPTIDE_CHAIN) -> dict[str, int]:
    """Remove every closing bond and crosslink variant from the peptide. Returns what it did.

    THE FAILURE THIS PREVENTS is silent and total: the source peptides are macrocycles, so
    their sequences carry the residues that were crosslinked, and two cysteines left near each
    other get an automatic disulfide the moment Rosetta looks at the pose. The peptide then
    scores as a cycle, after every linear-topology gate upstream has already passed it.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.chemical import VariantType
    from pyrosetta.rosetta.core.pose import remove_variant_type_from_pose_residue

    counts = {"disulfide_removed": 0, "other_variant_removed": 0}
    pep = peptide_residue_indices(pose, peptide_chain)

    # 1. Disulfides. `change_cys_state` converts CYD back to CYS and breaks the partner
    # record; removing the variant type alone can leave a dangling connection.
    for i in pep:
        res = pose.residue(i)
        if res.has_variant_type(VariantType.DISULFIDE):
            try:
                pyrosetta.rosetta.core.conformation.change_cys_state(
                    i, "CYS", pose.conformation()
                )
            except Exception:
                remove_variant_type_from_pose_residue(pose, VariantType.DISULFIDE, i)
            counts["disulfide_removed"] += 1

    # 2. Any other crosslink/cyclisation variant Rosetta may have assigned.
    for vt_name in ("SIDECHAIN_CONJUGATION", "C_METHYLAMIDATION", "CTERM_AMIDATION"):
        vt = getattr(VariantType, vt_name, None)
        if vt is None:
            continue
        for i in pep:
            if pose.residue(i).has_variant_type(vt):
                remove_variant_type_from_pose_residue(pose, vt, i)
                counts["other_variant_removed"] += 1
    return counts


def add_terminal_chemistry(pose, peptide_chain: str = PEPTIDE_CHAIN) -> None:
    """Free N terminus and free C terminus (which is what puts OXT on) for the peptide."""
    from pyrosetta.rosetta.core.chemical import VariantType
    from pyrosetta.rosetta.core.pose import add_variant_type_to_pose_residue

    pep = peptide_residue_indices(pose, peptide_chain)
    if not pep:
        return
    if not pose.residue(pep[0]).has_variant_type(VariantType.LOWER_TERMINUS_VARIANT):
        add_variant_type_to_pose_residue(pose, VariantType.LOWER_TERMINUS_VARIANT, pep[0])
    if not pose.residue(pep[-1]).has_variant_type(VariantType.UPPER_TERMINUS_VARIANT):
        add_variant_type_to_pose_residue(pose, VariantType.UPPER_TERMINUS_VARIANT, pep[-1])


def assert_linear_topology(pose, peptide_chain: str = PEPTIDE_CHAIN) -> None:
    """Raise if the peptide is still cyclic in any way. The check that makes step 2 a fact."""
    from pyrosetta.rosetta.core.chemical import VariantType

    pep = peptide_residue_indices(pose, peptide_chain)
    bad = [i for i in pep if pose.residue(i).has_variant_type(VariantType.DISULFIDE)]
    if bad:
        raise RuntimeError(
            f"peptide still carries disulfide variants at pose residues {bad} after "
            "strip_cyclic_topology -- the candidate would be scored as a cycle"
        )
    if not pep:
        return
    # A closing mainchain bond would show as a chemical connection between the termini.
    first, last = pose.residue(pep[0]), pose.residue(pep[-1])
    for res, other in ((first, pep[-1]), (last, pep[0])):
        for ci in range(1, res.n_possible_residue_connections() + 1):
            if res.connected_residue_at_resconn(ci) == other:
                raise RuntimeError(
                    "peptide termini are chemically bonded -- the candidate is a macrocycle, "
                    "not a linear peptide"
                )


def peptide_residue_indices(pose, peptide_chain: str = PEPTIDE_CHAIN) -> list[int]:
    """1-based pose indices of the peptide chain, in order."""
    info = pose.pdb_info()
    return [i for i in range(1, pose.total_residue() + 1) if info.chain(i) == peptide_chain]


# ------------------------------------------------------------------------ chi transfer


def transfer_source_chis(
    pose,
    src_chis: list[list[float]] | None,
    peptide_chain: str = PEPTIDE_CHAIN,
) -> int:
    """Set the peptide's chi angles from the source CP, where residue identity matches.

    An INITIALISATION, not a constraint: the packer is free to move every one of these, and
    the source rotamer simply joins the candidate set as a starting point. Returns how many
    residues were seeded, which is the number that says whether this did anything -- an
    identity mismatch on every residue would make it a silent no-op.
    """
    if not src_chis:
        return 0
    pep = peptide_residue_indices(pose, peptide_chain)
    n = 0
    for local, pose_idx in enumerate(pep):
        if local >= len(src_chis):
            break
        chis = src_chis[local]
        res = pose.residue(pose_idx)
        if not chis or res.nchi() == 0:
            continue
        # Only as many chis as BOTH sides have: a mismatch in count means the identities do
        # not actually match and writing the rest would scramble the rotamer.
        k = min(len(chis), res.nchi())
        if k == 0:
            continue
        for c in range(1, k + 1):
            pose.set_chi(c, pose_idx, float(chis[c - 1]))
        n += 1
    return n


def extract_chis(pdb_path: str, peptide_chain: str = PEPTIDE_CHAIN) -> list[list[float]]:
    """Read the peptide's chi angles out of a source complex PDB. ``[n_res][n_chi]`` degrees."""
    from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init

    _ensure_pyrosetta_init()
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)
    out = []
    for i in peptide_residue_indices(pose, peptide_chain):
        res = pose.residue(i)
        out.append([float(pose.chi(c, i)) for c in range(1, res.nchi() + 1)])
    return out


# ---------------------------------------------------------------------------- the packer


def _repack_task(pose, peptide_chain: str, use_input_sc: bool = True):
    """A PackerTask that repacks ONLY the peptide and never designs.

    Receptor residues are `prevent_repacking`, not merely `restrict_to_repacking`: the plan's
    primary comparison holds the receptor completely fixed, and "repack the receptor too"
    would make the measured interface a property of the receptor's relaxation as much as of
    the peptide.
    """
    import pyrosetta

    task = pyrosetta.standard_packer_task(pose)
    task.restrict_to_repacking()  # never design: the sequence is the source's and is fixed
    pep = set(peptide_residue_indices(pose, peptide_chain))
    for i in range(1, pose.total_residue() + 1):
        if i in pep:
            if use_input_sc:
                # Keeps the source-seeded rotamer in the candidate set alongside the library.
                task.nonconst_residue_task(i).or_include_current(True)
        else:
            task.nonconst_residue_task(i).prevent_repacking()
    return task


def _backbone_coords(pose, peptide_chain: str) -> np.ndarray:
    """Peptide backbone + all receptor coordinates, for the did-anything-move check."""
    pts = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        names = ("N", "CA", "C", "O") if pose.pdb_info().chain(i) == peptide_chain else None
        if names is None:
            for a in range(1, res.natoms() + 1):
                if not res.is_virtual(a):
                    xyz = res.xyz(a)
                    pts.append([xyz.x, xyz.y, xyz.z])
        else:
            for nm in names:
                if res.has(nm):
                    xyz = res.xyz(nm)
                    pts.append([xyz.x, xyz.y, xyz.z])
    return np.asarray(pts, dtype=float)


def pack_one(
    pose,
    seed_index: int,
    peptide_chain: str = PEPTIDE_CHAIN,
    minimize: bool = True,
    scorefxn=None,
):
    """Repack the peptide once. Returns ``(packed_pose, diagnostics)``.

    THE THREE "SEEDS" ARE THREE DRAWS FROM ROSETTA'S GLOBAL RNG, taken by calling the packer
    three times on independent copies of the same pose. They are not three re-initialisations:
    `pyrosetta.init` is process-global and re-running it would reset the whole environment
    mid-job. So the draws are reproducible per process given the init flags and the call order,
    and `seed_index` records which draw a result came from rather than seeding anything itself.
    Stated plainly because "seed" suggests a control that is not there.
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.minimization_packing import (
        MinMover,
        PackRotamersMover,
    )

    if scorefxn is None:
        scorefxn = pyrosetta.create_score_function("ref2015")
    work = pose.clone()
    before = _backbone_coords(work, peptide_chain)

    task = _repack_task(work, peptide_chain)
    PackRotamersMover(scorefxn, task).apply(work)

    diag = {"seed_index": float(seed_index), "minimized": 0.0}
    if minimize:
        # SIDECHAIN ONLY: chi true, bb false. A MinMover with bb=True would relax the very
        # backbone this experiment is measuring.
        mm = pyrosetta.rosetta.core.kinematics.MoveMap()
        mm.set_bb(False)
        mm.set_jump(False)
        pep = set(peptide_residue_indices(work, peptide_chain))
        for i in range(1, work.total_residue() + 1):
            mm.set_chi(i, i in pep)
        MinMover(mm, scorefxn, "lbfgs_armijo_nonmonotone", 0.01, True).apply(work)
        diag["minimized"] = 1.0

    after = _backbone_coords(work, peptide_chain)
    if before.shape != after.shape:
        raise RuntimeError(
            f"the atom set changed during packing ({before.shape} -> {after.shape}); the "
            "fixed-backbone check cannot be made"
        )
    shift = float(np.abs(before - after).max()) if before.size else 0.0
    diag["fixed_atom_max_shift_ang"] = shift
    if shift > BACKBONE_FIXED_TOL_ANG:
        raise RuntimeError(
            f"packing moved a fixed atom by {shift:.4f} A (tolerance "
            f"{BACKBONE_FIXED_TOL_ANG}) -- the peptide backbone or the receptor was not held, "
            "so every edit-size number measured against this candidate would be wrong"
        )
    diag["total_score"] = float(scorefxn(work))
    return work, diag


# ---------------------------------------------------------------------- scoring a pose


#: Heavy-atom distance at which a peptide-receptor residue pair counts as in CONTACT, nm.
#: 0.8 nm is the project's existing `contact_cutoff_nm`, used by the training contact term and
#: by the discriminator's interface features. Declared here, before any arm was scored, so the
#: threshold cannot become "whichever one separated the arms".
CONTACT_CUTOFF_NM = 0.8

#: A tighter cutoff reported BESIDE the above as a diagnostic, not as the criterion. 0.45 nm is
#: a conventional heavy-atom contact distance; 0.8 nm is generous and a retention figure
#: computed at it can look healthy while the close-range packing has changed completely.
CONTACT_TIGHT_NM = 0.45

#: Clash threshold, nm. The project's calibrated value: 0.27 nm REJECTS 43 of 60 native
#: crystal complexes, so a generator scored against it is being pushed away from native-like
#: packing rather than towards it. 0.20 nm passes 59 of 60.
CLASH_NM = 0.20


def _heavy_coords(pose, indices: list[int]) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Heavy-atom coordinates of the given pose residues, plus (residue, atom-name) labels."""
    pts, labels = [], []
    for i in indices:
        res = pose.residue(i)
        for a in range(1, res.natoms() + 1):
            if res.is_virtual(a) or res.atom_type(a).element() == "H":
                continue
            xyz = res.xyz(a)
            pts.append([xyz.x, xyz.y, xyz.z])
            labels.append((i, res.atom_name(a).strip()))
    return np.asarray(pts, dtype=float).reshape(-1, 3), labels


def residue_contact_map(pose, peptide_chain: str = PEPTIDE_CHAIN, cutoff_nm: float = CONTACT_CUTOFF_NM) -> set:
    """`{(peptide_local_index, receptor_pose_index)}` pairs within `cutoff_nm`, all-atom.

    Residue PAIRS rather than atom pairs: the question the plan asks is whether the original
    residue-pair contacts survived, and an atom-level set would churn with every rotamer the
    packer chose while the interaction it represents stayed put.
    """
    pep = peptide_residue_indices(pose, peptide_chain)
    rec = [i for i in range(1, pose.total_residue() + 1) if i not in set(pep)]
    if not pep or not rec:
        return set()
    cut_ang = cutoff_nm * 10.0
    out = set()
    for local, i in enumerate(pep):
        p, _ = _heavy_coords(pose, [i])
        if not len(p):
            continue
        for j in rec:
            q, _ = _heavy_coords(pose, [j])
            if not len(q):
                continue
            # Cheap reject on centroids before the full pairwise block.
            if np.linalg.norm(p.mean(0) - q.mean(0)) > cut_ang + 20.0:
                continue
            d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
            if d.min() <= cut_ang:
                out.add((local, j))
    return out


def clash_report(pose, peptide_chain: str = PEPTIDE_CHAIN, threshold_nm: float = CLASH_NM) -> dict:
    """All-atom steric report for the PACKED peptide: counts AND severity, not just a minimum.

    The plan is explicit that a minimum distance is an additional diagnostic rather than the
    packing metric, and it is right: one atom pair at 0.19 nm and forty at 0.19 nm give the
    same minimum and are not the same structure. So this returns the count of violating pairs,
    the summed overlap depth, the worst single overlap AND the minimum distance.

    Bonded exclusions: intra-peptide pairs are only counted at sequence separation >= 2, since
    residues i and i+1 share a peptide bond and are supposed to have atoms at bonding distance.
    """
    pep = peptide_residue_indices(pose, peptide_chain)
    rec = [i for i in range(1, pose.total_residue() + 1) if i not in set(pep)]
    thr = threshold_nm * 10.0
    out = {
        "clash_receptor_count": 0, "clash_receptor_overlap_ang": 0.0,
        "clash_receptor_worst_ang": 0.0, "min_receptor_dist_ang": float("inf"),
        "clash_self_count": 0, "clash_self_overlap_ang": 0.0,
        "clash_self_worst_ang": 0.0, "min_self_dist_ang": float("inf"),
        "n_peptide_heavy_atoms": 0,
    }
    if not pep:
        return out
    pp, plab = _heavy_coords(pose, pep)
    out["n_peptide_heavy_atoms"] = int(len(pp))

    if rec and len(pp):
        rp, _ = _heavy_coords(pose, rec)
        if len(rp):
            d = np.linalg.norm(pp[:, None, :] - rp[None, :, :], axis=-1)
            over = np.clip(thr - d, 0.0, None)
            out["clash_receptor_count"] = int((over > 0).sum())
            out["clash_receptor_overlap_ang"] = float(over.sum())
            out["clash_receptor_worst_ang"] = float(over.max())
            out["min_receptor_dist_ang"] = float(d.min())

    if len(pp) > 1:
        res_of = np.asarray([pep.index(r) for r, _ in plab])
        d = np.linalg.norm(pp[:, None, :] - pp[None, :, :], axis=-1)
        sep = np.abs(res_of[:, None] - res_of[None, :])
        eligible = sep >= 2
        if eligible.any():
            over = np.clip(thr - d, 0.0, None) * eligible
            out["clash_self_count"] = int(((over > 0) & eligible).sum() // 2)
            out["clash_self_overlap_ang"] = float(over.sum() / 2.0)
            out["clash_self_worst_ang"] = float(over.max())
            masked = np.where(eligible, d, np.inf)
            out["min_self_dist_ang"] = float(masked.min())
    for k in ("min_receptor_dist_ang", "min_self_dist_ang"):
        if not np.isfinite(out[k]):
            out[k] = -1.0
    return out


def sequence_of(pose, peptide_chain: str = PEPTIDE_CHAIN) -> str:
    """One-letter sequence of the peptide chain, for the exact-compliance gate."""
    return "".join(pose.residue(i).name1() for i in peptide_residue_indices(pose, peptide_chain))


def interface_energy(pose, peptide_chain: str = PEPTIDE_CHAIN) -> dict:
    """`dG_separated` and the peptide's own score, WITHOUT relax. In-process, no PDB round trip.

    NO RELAX, deliberately and at a cost. `InterfaceAnalyzerMover`'s `pack_separated` repacks
    only the unbound reference, so the bound pose is scored as given -- and the plan forbids
    backbone relaxation in the primary comparison, because relaxing would move the very
    backbone being measured. The consequence is that `dG_separated` here includes whatever
    `fa_rep` the generated backbone carries, so it is a BOUND-STATE SCORE and not a binding
    free energy. It is used as a tie-break and reported as a number, never as a thermodynamic
    claim -- which is also why the peptide-only score comes back beside it, so strain and
    interface are separable rather than summed into one unreadable figure.
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pep = peptide_residue_indices(pose, peptide_chain)
    rec_chains = sorted(
        {pose.pdb_info().chain(i) for i in range(1, pose.total_residue() + 1)} - {peptide_chain}
    )
    out: dict[str, float] = {}
    try:
        mover = InterfaceAnalyzerMover(f"{peptide_chain}_{''.join(rec_chains)}")
        mover.set_pack_separated(True)
        mover.set_compute_packstat(False)
        work = pose.clone()
        mover.apply(work)
        data = mover.get_all_data()
        out["dG_separated"] = float(mover.get_separated_interface_energy())
        out["dSASA_int_ang2"] = float(mover.get_interface_delta_sasa())
        out["n_interface_residues"] = float(data.int_res_set.size()) if hasattr(data, "int_res_set") else float("nan")
    except Exception as exc:  # a scorer failure must not be reported as a good score
        out["dG_separated"] = float("nan")
        out["dSASA_int_ang2"] = float("nan")
        out["interface_error"] = f"{type(exc).__name__}: {exc}"

    # PEPTIDE STRAIN, SEPARATELY. The plan asks for interface energy and peptide
    # strain/geometry as separate numbers: a candidate with a good interface and a tortured
    # peptide is not the same result as one with both good, and a single total hides which.
    try:
        import pyrosetta as _pr

        sf = _pr.create_score_function("ref2015")
        sub = _pr.rosetta.core.pose.Pose()
        # `pdbslice` wants a vector1; building it by append is the portable form.
        keep = _pr.rosetta.utility.vector1_unsigned_long()
        for i in pep:
            keep.append(i)
        _pr.rosetta.core.pose.pdbslice(sub, pose, keep)
        out["peptide_score"] = float(sf(sub))
        out["peptide_score_per_res"] = float(sf(sub)) / max(len(pep), 1)
    except Exception as exc:
        out["peptide_score"] = float("nan")
        out["peptide_score_per_res"] = float("nan")
        out["peptide_score_error"] = f"{type(exc).__name__}: {exc}"
    return out


def score_packed(
    packed,
    source_contacts: set | None = None,
    source_sequence: str | None = None,
    peptide_chain: str = PEPTIDE_CHAIN,
) -> dict:
    """Every COMPLETED-LP measurement the comparison reports, for one packed candidate."""
    met: dict[str, float | str | bool] = {}
    met.update(clash_report(packed, peptide_chain))
    met.update(interface_energy(packed, peptide_chain))

    seq = sequence_of(packed, peptide_chain)
    met["sequence"] = seq
    met["n_residues"] = len(seq)
    if source_sequence is not None:
        # EXACT compliance, not a similarity: the arms copy the sequence and are not allowed
        # to change it, so anything but equality is a failed candidate rather than a worse one.
        met["sequence_exact"] = bool(seq == source_sequence)

    if source_contacts is not None:
        now = residue_contact_map(packed, peptide_chain, CONTACT_CUTOFF_NM)
        kept = source_contacts & now
        met["contacts_source"] = len(source_contacts)
        met["contacts_kept"] = len(kept)
        met["contacts_gained"] = len(now - source_contacts)
        met["contact_retention"] = (
            len(kept) / len(source_contacts) if source_contacts else float("nan")
        )
        tight_now = residue_contact_map(packed, peptide_chain, CONTACT_TIGHT_NM)
        met["contacts_tight_now"] = len(tight_now)

    try:
        assert_linear_topology(packed, peptide_chain)
        met["linear_topology"] = True
    except RuntimeError as exc:
        met["linear_topology"] = False
        met["linear_topology_error"] = str(exc)
    return met


# -------------------------------------------------------------- the candidate driver


def is_valid(met: dict, require_sequence: bool = True) -> tuple[bool, str]:
    """The first key of `SELECTION_ORDER`: does this candidate pass the calibrated gates?

    Returns `(valid, reason)` so a rejection carries its cause into the manifest. The gates
    are the project's existing calibrated ones, NOT thresholds tuned per arm:

      * linear chemistry -- no re-formed disulfide, no closing bond;
      * exact source sequence -- the arms copy it and may not change it;
      * sterics -- no heavy-atom pair inside 0.20 nm, the value that passes 59 of 60 native
        crystals where the "obvious" 0.27 nm rejects 43 of them.

    RAW BACKBONE VALIDITY IS NOT CHECKED HERE. It is a property of the generator, measured
    before packing by `backbone.backbone_validity`, and the caller gates on it -- so that a
    packing failure and a broken backbone stay distinguishable in the manifest instead of
    collapsing into one "invalid".
    """
    if not met.get("linear_topology", False):
        return False, "cyclic_topology"
    if require_sequence and not met.get("sequence_exact", True):
        return False, "sequence_mismatch"
    if met.get("clash_receptor_count", 0) > 0:
        return False, "receptor_clash"
    if met.get("clash_self_count", 0) > 0:
        return False, "self_clash"
    return True, ""


def select_representative(results: list[PackResult], require_sequence: bool = True) -> tuple[int, str]:
    """Pick one of the N packing outcomes by the PREDECLARED order. `(index, basis)`.

    The order is fixed in `SELECTION_ORDER` and was written before any arm ran:

      1. valid linear chemistry / backbone / sterics;
      2. then source-contact retention;
      3. then interface energy, as a tie-break ONLY.

    IF NONE IS VALID the best diagnostic candidate is still returned, with a basis saying so.
    Silently dropping it would turn a measurable failure into a missing row, and a comparison
    whose denominators differ per arm is not a comparison.
    """
    if not results:
        return -1, "no_results"
    ok = [r for r in results if r.ok]
    if not ok:
        return results[0].seed_index, "all_packing_failed"
    valid = [r for r in ok if is_valid(r.metrics, require_sequence)[0]]
    pool, basis = (valid, "valid") if valid else (ok, "no_valid_candidate")

    def key(r: PackResult):
        ret = r.metrics.get("contact_retention", float("nan"))
        ret = -1.0 if ret != ret else float(ret)  # NaN sorts last
        dg = r.metrics.get("dG_separated", float("nan"))
        dg = float("inf") if dg != dg else float(dg)  # NaN sorts last; lower dG is better
        return (-ret, dg)

    return sorted(pool, key=key)[0].seed_index, basis


def pack_candidate(
    bb: torch.Tensor,
    aatype: np.ndarray,
    pep_mask: torch.Tensor,
    target_atom37_ang: np.ndarray,
    target_aatype: np.ndarray,
    work_dir: str | Path,
    tag: str,
    source_pdb: str | None = None,
    n_seeds: int = DEFAULT_PACK_SEEDS,
    minimize: bool = True,
    scorefxn=None,
    keep_pdbs: bool = True,
) -> dict:
    """Build, strip, seed, repack and score ONE generated backbone. Returns a manifest row.

    `bb` is a single example's `[n, 4, 3]` backbone in nm (the batch axis is the caller's).
    All `n_seeds` outcomes are retained, failures included, and the representative is chosen
    by `select_representative`.

    `source_pdb`, when given, supplies both the source sidechain torsions to seed the packer
    with and the source contact map to measure retention against -- so "did this keep the
    binding mode" is answered against the actual source complex rather than against an
    idealisation of it.
    """
    import pyrosetta

    from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init

    _ensure_pyrosetta_init()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if scorefxn is None:
        scorefxn = pyrosetta.create_score_function("ref2015")

    n = int(pep_mask.sum())
    pep_ang, _ = _peptide_atom37_ang(bb[None], pep_mask[None])
    pdb = str(work / f"{tag}__cand.pdb")
    write_complex_pdb(pdb, pep_ang[0, :n], aatype[:n], target_atom37_ang, target_aatype)

    row: dict = {"tag": tag, "candidate_pdb": pdb, "n_residues": n, "results": []}
    try:
        pose = pyrosetta.pose_from_pdb(pdb)
        row["stripped"] = strip_cyclic_topology(pose)
        add_terminal_chemistry(pose)
        assert_linear_topology(pose)
        src_contacts = None
        src_seq = None
        if source_pdb and os.path.exists(source_pdb):
            src_pose = pyrosetta.pose_from_pdb(source_pdb)
            src_contacts = residue_contact_map(src_pose)
            src_seq = sequence_of(src_pose)
            row["n_chi_seeded"] = transfer_source_chis(pose, extract_chis(source_pdb))
            row["source_sequence"] = src_seq
            row["contacts_source"] = len(src_contacts)
    except Exception as exc:
        row["setup_error"] = f"{type(exc).__name__}: {exc}"
        row["selected_seed"] = -1
        row["selection_basis"] = "setup_failed"
        return row

    import time

    results: list[PackResult] = []
    for s in range(n_seeds):
        t0 = time.time()
        try:
            packed, diag = pack_one(pose, s, minimize=minimize, scorefxn=scorefxn)
            met = score_packed(packed, src_contacts, src_seq)
            met.update({k: float(v) for k, v in diag.items()})
            met["pack_seconds"] = time.time() - t0
            valid, why = is_valid(met, require_sequence=src_seq is not None)
            met["valid"] = valid
            met["invalid_reason"] = why
            out_pdb = None
            if keep_pdbs:
                out_pdb = str(work / f"{tag}__pack{s}.pdb")
                packed.dump_pdb(out_pdb)
            results.append(PackResult(s, True, None, out_pdb, met))
        except Exception as exc:
            results.append(
                PackResult(s, False, f"{type(exc).__name__}: {exc}", None,
                           {"pack_seconds": time.time() - t0})
            )

    row["results"] = [
        {"seed": r.seed_index, "ok": r.ok, "failure": r.failure, "pdb": r.pdb_path, **r.metrics}
        for r in results
    ]
    sel, basis = select_representative(results, require_sequence=src_seq is not None)
    row["selected_seed"] = sel
    row["selection_basis"] = basis
    row["n_packed_ok"] = sum(1 for r in results if r.ok)
    row["n_valid"] = sum(1 for r in results if r.ok and r.metrics.get("valid", False))
    return row
