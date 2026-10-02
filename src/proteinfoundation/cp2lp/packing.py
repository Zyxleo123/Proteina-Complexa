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
