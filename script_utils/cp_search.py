"""Search engines that turn a bound linear peptide into a bound macrocycle.

Two engines, chosen by chemistry, because the two chemistries pose different problems:

**GenKIC (`mainchain`).** Head-to-tail cyclization is a genuine kinematic closure: the
backbone must form a covalent ring, and generalized kinematic closure solves exactly that.
Loop residues are given in ring order -- the C-terminal window, across the new bond, into the
N-terminal window -- so the residues flanking the window become GenKIC's anchors and are held
*exactly* fixed. Measured on `LNR_2qos_C_A`: closing at k=m=3 leaves the peptide interior at
0.000 A and the receptor at 0.000 A displacement, with the C--N bond at 1.328 A.

**Free-terminus anneal (`disulfide`, `isopeptide`).** A side-chain bridge leaves both backbone
termini free, so there is no backbone ring to close -- the requirement is that two side-chain
atoms meet at bonding geometry. GenKIC cannot be used: it builds its loop's atom chain from
polymer connections only, so a bond through CB/SG or CE/NZ is not on the chain it searches
(it rejects the closure dihedral as "not found in the chain of atoms to be closed" regardless
of how the atoms are named). Instead the bond is declared, harmonic ring constraints are
added, the released terminal torsions are perturbed, and a constrained minimization pulls the
bridge to ideal geometry. The achieved geometry is then *measured* against the chemistry's
window rather than assumed -- a candidate whose bridge did not actually close is rejected.

Both engines preserve the peptide interior kinematically, not by restraint. GenKIC does it
through its anchors; the anneal engine does it with a fold tree re-rooted at an interior
peptide residue so that terminal torsions propagate outward, toward the termini, and never
into the interior. Verified by measurement in both cases (interior and receptor max
displacement 0.0000 A), which is why `released_residues` is recorded per candidate: the set of
residues whose coordinates were allowed to change is exact, not approximate.

The receptor is never moved by either engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cp_contract as C
import cp_topology as T


@dataclass
class Target:
    """A staged LP complex plus everything derived from it that the search needs."""

    example_id: str
    pdb_path: str
    binder_chain: str
    pose: object                       # the ORIGINAL LP pose; never mutated
    binder_pos: list[int]
    receptor_pos: list[int]
    receptor_chains: list[str]
    orig_sequence: str
    orig_contacts_heavy: set
    orig_contacts_ca: set
    interface_local: list[int]

    @property
    def length(self) -> int:
        return len(self.binder_pos)

    @property
    def native_aa_i(self) -> str:
        return self.pose.residue(self.binder_pos[0]).name3()

    @property
    def native_aa_j(self) -> str:
        return self.pose.residue(self.binder_pos[-1]).name3()

    def interface_pose_positions(self) -> list[int]:
        return [self.binder_pos[li] for li in self.interface_local]


def load_target(example_id: str, pdb_path: str, binder_chain: str) -> Target:
    """Load a staged complex and precompute the original LP's reference metrics."""
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(pdb_path)
    binder_pos = T.binder_positions(pose, binder_chain)
    receptor_pos = T.receptor_positions(pose, binder_chain)
    if not receptor_pos:
        raise ValueError(f"{example_id}: no receptor chain besides {binder_chain!r}")
    heavy = C.heavy_atom_contacts(pose, binder_pos, receptor_pos)
    return Target(
        example_id=example_id,
        pdb_path=pdb_path,
        binder_chain=binder_chain,
        pose=pose,
        binder_pos=binder_pos,
        receptor_pos=receptor_pos,
        receptor_chains=T.receptor_chain_ids(pose, binder_chain),
        orig_sequence=C.peptide_sequence(pose, binder_pos),
        orig_contacts_heavy=heavy,
        orig_contacts_ca=C.ca_contacts(pose, binder_pos, receptor_pos),
        interface_local=C.interface_local_indices(heavy),
    )


@dataclass
class SearchSpec:
    """One point of the search: chemistry, anchor identities, window, extra mutations.

    `extra_mutations` are binder-local index -> three-letter code, applied on top of the
    anchor identities. They exist so a budget of 2 or 3 can be spent on something other than
    the anchors; every one of them counts against the mutation budget, and none of them may
    touch a protected residue.
    """

    chemistry: str
    aa_i: str
    aa_j: str
    k_nterm: int
    m_cterm: int
    extra_mutations: dict[int, str] = field(default_factory=dict)

    def label(self) -> str:
        extra = "".join(f"_{k}{v}" for k, v in sorted(self.extra_mutations.items()))
        return f"{self.chemistry}_{self.aa_i}{self.aa_j}_k{self.k_nterm}m{self.m_cterm}{extra}"


def reroot_outward(pose, binder_pos: list[int], anchor_local: int) -> bool:
    """Re-root the fold tree so peptide torsions propagate outward to the termini.

    The default fold tree enters the peptide at its first residue, so perturbing residue 1
    swings the entire peptide and the bound pose is destroyed. Rooting at an interior residue
    and growing edges in both directions confines each terminal window's motion to itself.
    Returns the tree's own validity check.
    """
    import pyrosetta.rosetta as R

    anchor = binder_pos[anchor_local]
    first_b, last_b = binder_pos[0], binder_pos[-1]
    ft = R.core.kinematics.FoldTree()
    ft.add_edge(1, first_b - 1, -1)           # the receptor, in pose order
    ft.add_edge(1, anchor, 1)                 # jump into the peptide's interior anchor
    if anchor < last_b:
        ft.add_edge(anchor, last_b, -1)       # outward, toward the C-terminus
    if anchor > first_b:
        ft.add_edge(anchor, first_b, -1)      # outward, toward the N-terminus
    ok = bool(ft.check_fold_tree())
    if ok:
        pose.fold_tree(ft)
    return ok


def add_search_coord_constraints(pose, target, released_pose, sd: float = 1.0) -> int:
    """Restrain the RELEASED residues' CA atoms to their original positions, for the search.

    This is the lever on pose preservation, and it was missing. `randomize_backbone_by_rama_prepro`
    resamples released torsions with no preference for where the residues started, and
    `genkic_close` was handed a score function with the constraint terms at ZERO weight -- so
    the lowest-energy selector chose among closures purely on ref2015 and was indifferent to
    whether the peptide had swung across the interface. Measured on the first real run: 209 of
    222 scored candidates were already beyond the 2 A whole-peptide bar BEFORE relaxation, and
    relax then pulled them back by only 0.2 A. The pose failure is made in the search and
    relaxation cannot undo it.

    Only the released residues are restrained: the interior is already held exactly by the
    engine's kinematics, so restraining it would add a force with nothing to do.

    `sd` trades closure against fidelity, and the trade is real rather than free -- a target
    whose termini start 45 A apart cannot close while holding every released CA within 1 A, so
    a tight `sd` will make some targets unreachable instead of reachable-but-wrong. That is the
    honest outcome and it is why `sd` is a swept knob, not a constant.

    Returns:
        The number of constraints added.
    """
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc

    anchor = AtomID(1, pose.fold_tree().root())
    n = 0
    for p in released_pose:
        if not (pose.residue(p).has("CA") and target.pose.residue(p).has("CA")):
            continue
        pose.add_constraint(CoordinateConstraint(
            AtomID(pose.residue(p).atom_index("CA"), p), anchor,
            target.pose.residue(p).xyz("CA"), HarmonicFunc(0.0, sd)))
        n += 1
    return n


def set_rng_seed(seed: int) -> None:
    """Seed both RNGs a search attempt draws from, so a batch is reproducible on resume.

    Rosetta's generator drives GenKIC and the packer; Python's drives the anneal engine's
    torsion perturbations. Seeding only one leaves half the search unreproducible, which
    would make a resumed batch quietly diverge from the one it is continuing.
    """
    import random

    import pyrosetta.rosetta as R

    R.numeric.random.rg().set_seed("mt19937", int(seed))
    random.seed(int(seed))


def prepare(target: Target, spec: SearchSpec) -> dict[str, object]:
    """Apply the spec's identities and declare its ring bond on a fresh copy of the LP.

    Returns the prepared pose plus the resolved ring atoms and released-residue record.
    Mutations are applied BEFORE the bond is declared, because the declaration depends on the
    anchors' identities (and, for isopeptide, patches their residue types).
    """
    import pyrosetta

    chem = T.chemistry(spec.chemistry)
    pose = pyrosetta.Pose(target.pose)
    bp = target.binder_pos
    i_pos, j_pos = bp[0], bp[-1]

    T.mutate_residue(pose, i_pos, spec.aa_i)
    T.mutate_residue(pose, j_pos, spec.aa_j)
    for local, name3 in spec.extra_mutations.items():
        T.mutate_residue(pose, bp[local], name3)

    atoms = T.resolve_ring_atoms(chem, spec.aa_i, spec.aa_j)
    window = T.released_residues(bp, spec.k_nterm, spec.m_cterm)
    info = T.declare_ring_bond(pose, i_pos, j_pos, chem, atoms)
    # Mutating residues and changing variant types both obsolete the PDBInfo, and every
    # chemistry here does at least one of them. Repair it now, while the reference is in hand.
    if not T.ensure_pdb_info(pose, target.pose):
        raise ValueError(
            f"{spec.label()}: could not rebuild a valid PDBInfo after declaring the ring "
            f"(pose {pose.size()} residues vs reference {target.pose.size()})")
    return {"pose": pose, "atoms": atoms, "window": window, "declare": info,
            "i_pos": i_pos, "j_pos": j_pos, "chem": chem}


def constrained_score_function(
    cst_weight: float = 5.0, coord_weight: float = 0.0
) -> object:
    """ref2015 with the constraint terms switched on.

    ref2015 ships with zero weight on every constraint term, so constraints added to a pose
    are silently ignored during minimization unless they are weighted here. That is the
    failure mode where a relax quietly pulls a declared ring back open.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType

    sf = pyrosetta.create_score_function("ref2015")
    sf.set_weight(ScoreType.atom_pair_constraint, cst_weight)
    sf.set_weight(ScoreType.angle_constraint, cst_weight)
    sf.set_weight(ScoreType.dihedral_constraint, cst_weight)
    if coord_weight:
        sf.set_weight(ScoreType.coordinate_constraint, coord_weight)
    return sf


# --------------------------------------------------------------------------------------
# engine 1: GenKIC (head-to-tail)
# --------------------------------------------------------------------------------------

def genkic_close(
    pose,
    window: dict,
    i_pos: int,
    j_pos: int,
    chem: T.RingChemistry,
    atoms: dict[str, str],
    score_function,
    attempts: int = 500,
    rama_cutoff: float = 2.0,
    min_solutions: int = 8,
) -> bool:
    """One GenKIC call: close the ring through `chem`'s bond, or report failure.

    Returns True when a solution was found and applied to `pose`. Loop residues are supplied
    in ring order so GenKIC's auto-assigned anchors are the residues flanking the window,
    which it holds exactly fixed.

    `min_solutions` is the quality lever, and it is why this arm is not a first-hit search.
    GenKIC stops once it has collected `min_solutions` closures, then `lowest_energy_selector`
    returns the best of them; at `min_solutions=1` it stops at the FIRST closure and the
    selector has nothing to choose from, which is exactly the "stop at the first closed
    structure" behaviour the contract rules out. `attempts` is the upper bound on tries, so an
    easy window costs little and a hard one is capped.
    """
    import pyrosetta.rosetta as R

    loop = window["loop_order"]
    # GenKIC needs three DISTINCT pivots, strictly ordered along the loop. A 2-residue loop
    # (the ladder's k=m=1 rung) makes `loop[len//2]` the last residue, so pivots 2 and 3
    # coincide and Rosetta aborts the PROCESS with "The third pivot cannot be before the
    # second pivot" -- a C++ UtilityExitException that no Python try/except around
    # `gk.apply` can catch, because it fires in `set_pivot_atoms` validation. Measured on
    # the first real complex. Refusing the rung here costs nothing real: the same window is
    # also far too short to close, and the caller records it as an unsearched rung rather
    # than a failed one.
    if len(loop) < 3:
        return False
    middle = max(1, min(len(loop) // 2, len(loop) - 2))
    gk = R.protocols.generalized_kinematic_closure.GeneralizedKIC()
    for r in loop:
        gk.add_loop_residue(r)
    gk.set_pivot_atoms(loop[0], "CA", loop[middle], "CA", loop[-1], "CA")
    # (prev_j, atom_j, atom_i, next_i) must be true bonded neighbours along the ring.
    gk.close_bond(
        j_pos, atoms["atom_j"], i_pos, atoms["atom_i"],
        j_pos, atoms["prev_j"], i_pos, atoms["next_i"],
        chem.ideal_len, chem.ideal_ang1, chem.ideal_ang2, chem.ideal_tor,
        False, False,
    )
    gk.add_perturber("randomize_backbone_by_rama_prepro")
    for r in loop:
        gk.add_residue_to_perturber_residue_list(r)
    gk.add_filter("loop_bump_check")
    gk.add_filter("rama_prepro_check")
    gk.set_filter_resnum(loop[middle])
    gk.set_filter_rama_cutoff_energy(rama_cutoff)
    gk.set_selector_type("lowest_energy_selector")
    gk.set_selector_scorefunction(score_function)
    gk.set_closure_attempts(int(attempts))
    gk.set_min_solution_count(max(1, int(min_solutions)))
    gk.set_dont_fail_if_no_solution_found(True)

    # GenKIC can also drop the pose's PDBInfo. It is repaired downstream rather than here,
    # by `cp_topology.ensure_pdb_info` against the original LP pose -- this function has no
    # reference pose to rebuild from, and the declaration in `prepare` has usually obsoleted
    # the PDBInfo before GenKIC ever runs.
    try:
        gk.apply(pose)
    except RuntimeError:
        return False
    return bool(gk.last_run_successful())


# --------------------------------------------------------------------------------------
# engine 2: free-terminus anneal (side-chain bridges)
# --------------------------------------------------------------------------------------

def bridge_anneal(
    pose,
    target: Target,
    window: dict,
    i_pos: int,
    j_pos: int,
    chem: T.RingChemistry,
    atoms: dict[str, str],
    perturb_deg: float = 45.0,
    min_rounds: int = 2,
    coord_sd: float = 1.0,
) -> bool:
    """Perturb the released terminal torsions, then minimize under the ring constraints.

    The two termini are free, so nothing needs solving kinematically; what has to happen is
    that the bridging atoms reach bonding geometry. The harmonic ring constraints supply that
    pull and the minimizer does the work. Returns True if the bridge ended inside the
    chemistry's geometry window -- measured, never assumed.
    """
    import random

    import pyrosetta
    import pyrosetta.rosetta as R

    bp = target.binder_pos
    anchor_local = _interior_anchor_local(target, window)
    if anchor_local is None:
        return False
    if not reroot_outward(pose, bp, anchor_local):
        return False

    released = set(window["released_pose"])
    # Perturb only the released backbone; with the outward tree this cannot touch the interior.
    for p in window["released_pose"]:
        pose.set_phi(p, pose.phi(p) + random.uniform(-perturb_deg, perturb_deg))
        pose.set_psi(p, pose.psi(p) + random.uniform(-perturb_deg, perturb_deg))

    T.add_ring_constraints(pose, i_pos, j_pos, chem, atoms)
    # Same pose-preservation restraint the GenKIC path needs: without it the minimizer is free
    # to drag the released termini anywhere that satisfies the ring, and nothing later recovers
    # the original placement.
    if coord_sd and coord_sd > 0:
        add_search_coord_constraints(pose, target, window["released_pose"], sd=coord_sd)
    sf = constrained_score_function(cst_weight=10.0,
                                    coord_weight=1.0 if coord_sd and coord_sd > 0 else 0.0)

    mm = R.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)
    for p in released:
        mm.set_bb(p, True)
    # Bridge side chains must be free to rotate, and so must any peptide side chain whose
    # residue moved; receptor side chains stay fixed here and are repacked later.
    for p in set(released) | {i_pos, j_pos}:
        mm.set_chi(p, True)

    minimizer = R.protocols.minimization_packing.MinMover()
    minimizer.movemap(mm)
    minimizer.score_function(sf)
    minimizer.min_type("lbfgs_armijo_nonmonotone")
    minimizer.tolerance(1e-4)
    for _ in range(max(1, min_rounds)):
        minimizer.apply(pose)

    geom = T.ring_bond_geometry(pose, i_pos, j_pos, chem, atoms)
    return bool(geom["bond_ok"])


def _interior_anchor_local(target: Target, window: dict) -> int | None:
    """Pick the interior residue to root the fold tree at: the middle of the fixed span."""
    L = target.length
    lo, hi = window["k_nterm"], L - window["m_cterm"] - 1
    if lo > hi:
        return None
    return (lo + hi) // 2


