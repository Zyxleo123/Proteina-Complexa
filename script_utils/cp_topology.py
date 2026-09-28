"""Ring topology for cyclic peptides in Rosetta: declare it, constrain it, measure it.

This is the layer that makes a macrocycle a macrocycle *to Rosetta* rather than a linear
peptide that happens to sit in a ring-shaped pose. Three chemistries, all bonded between
the peptide's two termini (CPSea is exclusively terminal-bonded -- see
`docs/README_LP2CP_ROUTE_DECISION.md` section 1 -- and this work keeps that placement fixed):

    mainchain    head-to-tail amide,  C(j) -- N(i)
    disulfide    Cys/Cys bridge,      SG(j) -- SG(i)
    isopeptide   Lys/Asp|Glu amide,   CG|CD(j) -- NZ(i)     (or with i/j swapped)

Why this module exists rather than reusing `evaluation/rosetta_energy._declare_cyclic_bond`
-------------------------------------------------------------------------------------------
Two defects in that function make it unusable here, both measured:

1. **Isopeptide was never declared at all.** It logged a warning and returned False, so the
   bond was absent from the pose and the reported dG described an open topology. The fix is
   Rosetta's own `SidechainConjugation` patch (database
   `chemical/residue_type_sets/fa_standard/patches/SidechainConjugation.txt`), which opens a
   real `CONNECT` at LYS:NZ / ASP:CG / GLU:CD, deletes the extra protons (3 HZ -> 1 amide HZ)
   and the second carboxyl oxygen (OD1+OD2 -> one amide OD), and retypes the atoms as `Nbb`
   and `CObb`. Verified on a real complex: `is_bonded` becomes True and ref2015 stays sane
   (-198 vs -244 for the unmodified pose) instead of exploding. Declaring NZ--CG *without*
   the patch leaves NZ carrying three protons, i.e. a five-bonded nitrogen.

2. **Endpoint mapping assumed binder residues are numbered 1..L.** It maps a binder-local
   index with `pdb2pose(chain, i_local + 1)`. The LNR staged complexes preserve each source
   crystal's own numbering -- `LNR_2qos_C_A` chain B runs **158..168** -- so that lookup
   returns 0 and the call raises. `binder_positions` here maps by chain order instead, which
   is correct for any numbering.

Both are corrected in `rosetta_energy.py` as well, so the repo has one behaviour.

Geometry conventions
--------------------
`close_bond`-style geometry is defined over four atoms, (prev_j, aj, ai, next_i), walking the
ring from the C-terminal anchor across the new bond into the N-terminal anchor. `prev_j` and
`next_i` must be *true bonded neighbours* of the bridging atoms: naming CA for a side-chain
bridge (where CB sits between CA and SG) makes GenKIC reject the dihedral with "not found in
the chain of atoms to be closed". Measured the hard way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

# Bond-geometry acceptance windows. Deliberately generous on the angles (a relaxed structure
# is not an ideal one) and tight on the length, because the length is what says "bonded".
AMIDE_LEN, AMIDE_LEN_TOL = 1.328, 0.10
SS_LEN, SS_LEN_TOL = 2.030, 0.20
ANGLE_TOL_DEG = 15.0


@dataclass(frozen=True)
class RingChemistry:
    """One terminal cyclization chemistry.

    Attributes:
        name: `"mainchain"`, `"disulfide"` or `"isopeptide"`.
        atom_i: bridging atom on the N-terminal anchor (residue `i`, binder-local 0).
        atom_j: bridging atom on the C-terminal anchor (residue `j`, binder-local L-1).
        prev_j: atom bonded to `atom_j` on the ring-interior side.
        next_i: atom bonded to `atom_i` on the ring-interior side.
        ideal_len, len_tol: bond length and its acceptance half-window, Angstrom.
        ideal_ang1: angle (prev_j, atom_j, atom_i), degrees.
        ideal_ang2: angle (atom_j, atom_i, next_i), degrees.
        ideal_tor: dihedral (prev_j, atom_j, atom_i, next_i), degrees.
        anchor_options: allowed (aa_i, aa_j) identity pairs. `None` means "any residue is
            already competent", which is true only for the head-to-tail amide -- every
            amino acid has a backbone N and C. A non-empty list means the chemistry is
            *chemically impossible* at mutation budget 0 unless the original LP already
            carries one of these pairs at its termini.
        genkic_closable: whether GenKIC can close this bond. False for the side-chain
            bridges: GenKIC builds its loop's atom chain from polymer (backbone)
            connections only, so a bond through CB/SG or CE/NZ is not on the chain it
            searches. Measured -- it raises "the dihedral angle specified was not found in
            the chain of atoms to be closed" no matter how the atoms are named. Those
            chemistries use the free-terminus engine instead (see `cp_engines`).
    """

    name: str
    atom_i: str
    atom_j: str
    prev_j: str
    next_i: str
    ideal_len: float
    len_tol: float
    ideal_ang1: float
    ideal_ang2: float
    ideal_tor: float
    genkic_closable: bool
    anchor_options: tuple[tuple[str, str], ...] | None = None
    # Side-chain torsions that must be free for the bridge to reach, per residue name3.
    bridge_chis: dict[str, int] = field(default_factory=dict)


MAINCHAIN = RingChemistry(
    name="mainchain",
    atom_i="N", atom_j="C", prev_j="CA", next_i="CA",
    ideal_len=AMIDE_LEN, len_tol=AMIDE_LEN_TOL,
    ideal_ang1=116.2, ideal_ang2=121.7, ideal_tor=180.0,
    genkic_closable=True,
    anchor_options=None,
)

DISULFIDE = RingChemistry(
    name="disulfide",
    atom_i="SG", atom_j="SG", prev_j="CB", next_i="CB",
    ideal_len=SS_LEN, len_tol=SS_LEN_TOL,
    ideal_ang1=103.0, ideal_ang2=103.0, ideal_tor=-85.0,
    genkic_closable=False,
    anchor_options=(("CYS", "CYS"),),
    bridge_chis={"CYS": 1},
)

ISOPEPTIDE = RingChemistry(
    name="isopeptide",
    # atom_i / atom_j are resolved per anchor pair by `isopeptide_atoms`, because either
    # terminus may carry the lysine. These defaults describe LYS at i, ASP at j.
    atom_i="NZ", atom_j="CG", prev_j="CB", next_i="CE",
    ideal_len=AMIDE_LEN, len_tol=AMIDE_LEN_TOL,
    ideal_ang1=116.2, ideal_ang2=121.7, ideal_tor=180.0,
    genkic_closable=False,
    anchor_options=(("LYS", "ASP"), ("LYS", "GLU"), ("ASP", "LYS"), ("GLU", "LYS")),
    bridge_chis={"LYS": 4, "ASP": 2, "GLU": 3},
)

CHEMISTRIES: dict[str, RingChemistry] = {
    c.name: c for c in (MAINCHAIN, DISULFIDE, ISOPEPTIDE)
}

# Which atom is the amine and which the acyl carbon, per residue type.
_ISO_AMINE = {"LYS": ("NZ", "CE")}          # (bridging atom, its ring-interior neighbour)
_ISO_ACYL = {"ASP": ("CG", "CB"), "GLU": ("CD", "CG")}


def isopeptide_atoms(aa_i: str, aa_j: str) -> dict[str, str]:
    """Resolve the four ring atoms for an isopeptide bond between anchors `aa_i`/`aa_j`.

    Either terminus may be the lysine; the other must be Asp or Glu. Returns the same keys
    `RingChemistry` uses, so callers can override the chemistry's defaults per anchor pair.

    Raises:
        ValueError: the pair is not one amine plus one acyl donor.
    """
    if aa_i in _ISO_AMINE and aa_j in _ISO_ACYL:
        atom_i, next_i = _ISO_AMINE[aa_i]
        atom_j, prev_j = _ISO_ACYL[aa_j]
    elif aa_i in _ISO_ACYL and aa_j in _ISO_AMINE:
        atom_i, next_i = _ISO_ACYL[aa_i]
        atom_j, prev_j = _ISO_AMINE[aa_j]
    else:
        raise ValueError(
            f"isopeptide needs one LYS and one ASP/GLU anchor, got aa_i={aa_i!r} aa_j={aa_j!r}"
        )
    return {"atom_i": atom_i, "atom_j": atom_j, "prev_j": prev_j, "next_i": next_i}


def resolve_ring_atoms(chem: RingChemistry, aa_i: str, aa_j: str) -> dict[str, str]:
    """The four ring atoms for this chemistry at these anchor identities."""
    if chem.name == "isopeptide":
        return isopeptide_atoms(aa_i, aa_j)
    return {"atom_i": chem.atom_i, "atom_j": chem.atom_j,
            "prev_j": chem.prev_j, "next_i": chem.next_i}


# --------------------------------------------------------------------------------------
# pose helpers
# --------------------------------------------------------------------------------------

def binder_positions(pose, binder_chain: str) -> list[int]:
    """Pose sequence positions of the binder chain, in order.

    Maps by chain membership and pose order, NOT by PDB residue number: the staged LNR
    complexes keep each crystal's own numbering (chain B of `LNR_2qos_C_A` runs 158..168),
    so binder-local index `k` is `binder_positions(...)[k]` and emphatically not
    `pdb2pose(chain, k + 1)`.
    """
    pdb_info = pose.pdb_info()
    out = [i for i in range(1, pose.size() + 1) if pdb_info.chain(i) == binder_chain]
    if not out:
        raise ValueError(
            f"no residues on chain {binder_chain!r}; chains present: "
            f"{sorted({pdb_info.chain(i) for i in range(1, pose.size() + 1)})}"
        )
    return out


def receptor_positions(pose, binder_chain: str) -> list[int]:
    """Pose sequence positions of everything that is not the binder chain."""
    pdb_info = pose.pdb_info()
    return [i for i in range(1, pose.size() + 1) if pdb_info.chain(i) != binder_chain]


def receptor_chain_ids(pose, binder_chain: str) -> list[str]:
    pdb_info = pose.pdb_info()
    seen: list[str] = []
    for i in range(1, pose.size() + 1):
        c = pdb_info.chain(i)
        if c != binder_chain and c not in seen:
            seen.append(c)
    return seen


def sidechain_connection_id(pose, pos: int, atom: str) -> int:
    """Residue-connection id at `pos` whose connect atom is `atom`.

    Looked up rather than assumed: the side-chain connection's index depends on how many
    polymer connections the residue still has, so a terminal residue carrying its
    Nterm/Cterm variant numbers it differently from an internal one.

    Raises:
        ValueError: no connection on this residue uses `atom`.
    """
    residue = pose.residue(pos)
    target = residue.atom_index(atom)
    for conn_id in range(1, residue.type().n_possible_residue_connections() + 1):
        if residue.residue_connect_atom_index(conn_id) == target:
            return conn_id
    raise ValueError(
        f"residue {pos} ({residue.name()}) has no residue connection at atom {atom!r}; "
        f"connections use atoms "
        f"{[residue.atom_name(residue.residue_connect_atom_index(c)).strip() for c in range(1, residue.type().n_possible_residue_connections() + 1)]}"
    )


def ensure_pdb_info(pose, ref_pose) -> bool:
    """Guarantee `pose` carries a valid PDBInfo, rebuilding it from `ref_pose` if not.

    Changing a residue's variant types -- which every chemistry here does, whether by
    stripping the terminus variants for a head-to-tail amide or adding
    `SidechainConjugation` for an isopeptide -- marks the pose's PDBInfo obsolete. GenKIC
    then rebuilds the pose and can drop it entirely. Anything that addresses residues by
    CHAIN afterwards breaks: `ChainSelector` raises "received a pose without a valid
    PDBInfo", and `InterfaceAnalyzerMover`'s "B_A" interface string cannot be resolved
    either. Measured on the first real complex, where it killed the relax of every single
    mainchain candidate -- the primary chemistry produced no scored results at all, while
    the side-chain chemistries happened to survive.

    Rebuilt rather than un-obsoleted: the flag is set because Rosetta cannot vouch for the
    mapping, so the honest repair is to re-impose the reference's chains and numbering. That
    is sound precisely because no operation here changes the residue COUNT or their order,
    which is asserted rather than assumed.

    Returns:
        True if the pose ends with a valid PDBInfo.
    """
    import pyrosetta.rosetta.core.pose as _pose

    info = pose.pdb_info()
    if info is not None and not info.obsolete():
        return True
    ref_info = ref_pose.pdb_info()
    if ref_info is None or pose.size() != ref_pose.size():
        return False
    fresh = _pose.PDBInfo(pose)
    for i in range(1, pose.size() + 1):
        fresh.chain(i, ref_info.chain(i))
        fresh.number(i, ref_info.number(i))
    fresh.obsolete(False)
    pose.pdb_info(fresh)
    return True


def mutate_residue(pose, pos: int, name3: str) -> None:
    """Mutate `pos` to `name3`, keeping the backbone. No-op if already that identity."""
    import pyrosetta.rosetta as R

    if pose.residue(pos).name3() == name3:
        return
    mover = R.protocols.simple_moves.MutateResidue()
    mover.set_target(pos)
    mover.set_res_name(name3)
    mover.apply(pose)


def declare_ring_bond(
    pose,
    i_pos: int,
    j_pos: int,
    chem: RingChemistry,
    atoms: dict[str, str] | None = None,
) -> dict[str, object]:
    """Declare `chem`'s closing bond between pose residues `i_pos` and `j_pos`, in place.

    Assumes the anchors already carry the required identities (call `mutate_residue` first);
    this function does not mutate. Every chemistry ends with a real covalent connection --
    `pose.residue(i_pos).is_bonded(j_pos)` is True -- so there is no "declared=False" path
    the way `rosetta_energy._declare_cyclic_bond` had for isopeptide.

    Returns:
        Dict with `declared` (bool), `atoms` (the four ring atom names) and `residue_types`
        (the post-declaration full type names, so a log can show e.g.
        `LYS:SidechainConjugation`).

    Raises:
        ValueError: anchors carry the wrong identity for the chemistry, or Rosetta refused
            the bond.
    """
    import pyrosetta.rosetta as R
    from pyrosetta.rosetta.core.chemical import VariantType
    from pyrosetta.rosetta.core.pose import (
        add_variant_type_to_pose_residue,
        remove_variant_type_from_pose_residue,
    )

    aa_i, aa_j = pose.residue(i_pos).name3(), pose.residue(j_pos).name3()
    atoms = atoms or resolve_ring_atoms(chem, aa_i, aa_j)

    if chem.name == "mainchain":
        # A closing amide needs open N and C connections. The terminus variants cap those
        # with extra protons/oxygens and no open bond site, so they come off first -- the
        # same thing PeptideCyclizeMover does internally.
        remove_variant_type_from_pose_residue(pose, VariantType.LOWER_TERMINUS_VARIANT, i_pos)
        remove_variant_type_from_pose_residue(pose, VariantType.UPPER_TERMINUS_VARIANT, j_pos)
        declare = R.protocols.simple_moves.DeclareBond()
        # add_termini=False, run_kic=False, kic_res1=0, kic_res2=0, rebuild_fold_tree=True.
        # The fold-tree rebuild is what lets GenKIC traverse the new bond.
        declare.set(j_pos, "C", i_pos, "N", False, False, 0, 0, True)
        declare.apply(pose)

    elif chem.name == "disulfide":
        if {aa_i, aa_j} != {"CYS"}:
            raise ValueError(f"disulfide needs CYS at both anchors, got {aa_i}/{aa_j}")
        if not pose.conformation().add_disulfide_bond(i_pos, j_pos):
            raise ValueError(
                f"add_disulfide_bond refused pose residues {i_pos}/{j_pos} "
                f"({aa_i}/{aa_j}) -- SG-SG is "
                f"{(pose.residue(i_pos).xyz('SG') - pose.residue(j_pos).xyz('SG')).norm():.2f} A"
            )

    elif chem.name == "isopeptide":
        # Open a real side-chain CONNECT on both anchors before declaring the bond, or NZ
        # keeps three protons and CG keeps two oxygens and the "bond" is a chemical fiction.
        for pos in (i_pos, j_pos):
            add_variant_type_to_pose_residue(pose, VariantType.SIDECHAIN_CONJUGATION, pos)
        declare = R.protocols.simple_moves.DeclareBond()
        declare.set(j_pos, atoms["atom_j"], i_pos, atoms["atom_i"], False, False, 0, 0, True)
        declare.apply(pose)
        # The patch's new atoms (the amide HZ, the single OD/OE, the V1 virtual) have ICOORs
        # defined relative to the connection, so they sit at stale positions until rebuilt.
        # The connection id is looked up from the bridging atom rather than assumed to be 3:
        # a terminal residue that still carries its Nterm/Cterm variant has one fewer polymer
        # connection, which shifts the side-chain connection's index.
        for pos, atom in ((i_pos, atoms["atom_i"]), (j_pos, atoms["atom_j"])):
            conn_id = sidechain_connection_id(pose, pos, atom)
            pose.conformation().rebuild_residue_connection_dependent_atoms(pos, conn_id)

    else:
        raise ValueError(f"unknown chemistry {chem.name!r}")

    if not pose.residue(i_pos).is_bonded(j_pos):
        raise ValueError(
            f"{chem.name}: declared the bond but residue {i_pos} still reports not bonded "
            f"to {j_pos} -- refusing to score an open topology as closed"
        )
    return {
        "declared": True,
        "atoms": atoms,
        "residue_types": (pose.residue(i_pos).name(), pose.residue(j_pos).name()),
    }


# --------------------------------------------------------------------------------------
# geometry measurement
# --------------------------------------------------------------------------------------

def _angle_deg(a, b, c) -> float:
    ba = [a[k] - b[k] for k in range(3)]
    bc = [c[k] - b[k] for k in range(3)]
    nba = math.sqrt(sum(x * x for x in ba)) or 1e-9
    nbc = math.sqrt(sum(x * x for x in bc)) or 1e-9
    cos = sum(ba[k] * bc[k] for k in range(3)) / (nba * nbc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _dihedral_deg(p0, p1, p2, p3) -> float:
    b0 = [p0[k] - p1[k] for k in range(3)]
    b1 = [p2[k] - p1[k] for k in range(3)]
    b2 = [p3[k] - p2[k] for k in range(3)]
    n1 = math.sqrt(sum(x * x for x in b1)) or 1e-9
    b1 = [x / n1 for x in b1]
    v = [b0[k] - sum(b0[m] * b1[m] for m in range(3)) * b1[k] for k in range(3)]
    w = [b2[k] - sum(b2[m] * b1[m] for m in range(3)) * b1[k] for k in range(3)]
    x = sum(v[k] * w[k] for k in range(3))
    cross = [b1[1] * v[2] - b1[2] * v[1],
             b1[2] * v[0] - b1[0] * v[2],
             b1[0] * v[1] - b1[1] * v[0]]
    y = sum(cross[k] * w[k] for k in range(3))
    return math.degrees(math.atan2(y, x))


def _xyz(pose, pos: int, atom: str) -> tuple[float, float, float]:
    v = pose.residue(pos).xyz(atom)
    return (v[0], v[1], v[2])


def ring_bond_geometry(
    pose, i_pos: int, j_pos: int, chem: RingChemistry, atoms: dict[str, str] | None = None
) -> dict[str, object]:
    """Measure the closing bond: length, both angles, the dihedral, and whether it is bonded.

    `bond_ok` is the contract's geometry gate -- length inside the chemistry's window AND
    both angles within `ANGLE_TOL_DEG`. The dihedral is reported but not gated: a cis amide
    (`|torsion|` near 0) is strained rather than wrong, and gating it would silently discard
    real macrocycles, so it is surfaced as `torsion_deg` and `amide_is_cis` for the reader.
    """
    atoms = atoms or resolve_ring_atoms(
        chem, pose.residue(i_pos).name3(), pose.residue(j_pos).name3()
    )
    ai, aj = atoms["atom_i"], atoms["atom_j"]

    # The residues at the anchors may not carry these atoms at all -- asking for a disulfide
    # where the pose decoded MET leaves no SG to measure, and Rosetta's response to
    # `residue.xyz("SG")` is to abort the process ("ResidueType MET:NtermProteinFull does not
    # have an atom SG"), not to raise something catchable at the call site.
    #
    # That case is a graded FAILURE, not an ungradable candidate: everything else in the
    # contract (sequence, contacts, pose RMSD, clashes) is still perfectly measurable, and the
    # candidate simply does not express the requested chemistry. Returning NaN geometry with
    # `bond_ok` False keeps it in the table. Raising here instead cost 318 of 800 rows on the
    # first real SDEdit run -- and they were precisely the abstaining candidates, the ones whose
    # removal from the denominator would most flatter that arm.
    missing = sorted({
        name for pos, name in ((i_pos, ai), (i_pos, atoms["next_i"]),
                               (j_pos, aj), (j_pos, atoms["prev_j"]))
        if not pose.residue(pos).has(name)
    })
    if missing:
        return {
            "bond_atoms": f"{ai}-{aj}",
            "bond_length_A": float("nan"),
            "bond_angle1_deg": float("nan"),
            "bond_angle2_deg": float("nan"),
            "bond_torsion_deg": float("nan"),
            "bond_is_declared": False,
            "bond_length_ok": False,
            "bond_angles_ok": False,
            "bond_ok": False,
            "amide_is_cis": False,
            "bond_atoms_missing": missing,
            "bond_missing_reason": (
                f"{chem.name} needs {ai}/{aj}; residue {i_pos} is "
                f"{pose.residue(i_pos).name3()} and residue {j_pos} is "
                f"{pose.residue(j_pos).name3()}, which lack {missing}"),
        }

    length = math.dist(_xyz(pose, i_pos, ai), _xyz(pose, j_pos, aj))
    ang1 = _angle_deg(_xyz(pose, j_pos, atoms["prev_j"]), _xyz(pose, j_pos, aj),
                      _xyz(pose, i_pos, ai))
    ang2 = _angle_deg(_xyz(pose, j_pos, aj), _xyz(pose, i_pos, ai),
                      _xyz(pose, i_pos, atoms["next_i"]))
    tor = _dihedral_deg(_xyz(pose, j_pos, atoms["prev_j"]), _xyz(pose, j_pos, aj),
                        _xyz(pose, i_pos, ai), _xyz(pose, i_pos, atoms["next_i"]))
    len_ok = abs(length - chem.ideal_len) <= chem.len_tol
    ang_ok = (abs(ang1 - chem.ideal_ang1) <= ANGLE_TOL_DEG
              and abs(ang2 - chem.ideal_ang2) <= ANGLE_TOL_DEG)
    return {
        "bond_atoms": f"{ai}-{aj}",
        "bond_length_A": round(length, 3),
        "bond_angle1_deg": round(ang1, 1),
        "bond_angle2_deg": round(ang2, 1),
        "bond_torsion_deg": round(tor, 1),
        "bond_is_declared": bool(pose.residue(i_pos).is_bonded(j_pos)),
        "bond_length_ok": bool(len_ok),
        "bond_angles_ok": bool(ang_ok),
        "bond_ok": bool(len_ok and ang_ok),
        "amide_is_cis": bool(chem.name != "disulfide" and abs(tor) < 90.0),
    }


def add_ring_constraints(
    pose,
    i_pos: int,
    j_pos: int,
    chem: RingChemistry,
    atoms: dict[str, str] | None = None,
    len_sd: float = 0.02,
    ang_sd_deg: float = 5.0,
    tor_sd_deg: float = 10.0,
) -> int:
    """Add harmonic constraints holding the closing bond at ideal geometry.

    Rosetta's minimizer does not otherwise know the declared bond should *stay* at bonding
    distance -- `fa_rep` will happily push the two atoms apart during relax, which would
    hand back a "cyclic" structure whose ring had quietly reopened. These constraints are
    what keep the declared ring declared. Returns the number of constraints added.

    The dihedral is constrained only for the amide chemistries, where planarity is real.
    A disulfide's CB-SG-SG-CB dihedral is genuinely bimodal (+/-85 deg), so pinning it to one
    value would bias the search; its length and angles are constrained and the dihedral is
    left to the score function.
    """
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring.constraints import (
        AngleConstraint, AtomPairConstraint, DihedralConstraint)
    from pyrosetta.rosetta.core.scoring.func import HarmonicFunc

    atoms = atoms or resolve_ring_atoms(
        chem, pose.residue(i_pos).name3(), pose.residue(j_pos).name3()
    )

    def aid(pos: int, atom: str) -> AtomID:
        return AtomID(pose.residue(pos).atom_index(atom), pos)

    a_i, a_j = aid(i_pos, atoms["atom_i"]), aid(j_pos, atoms["atom_j"])
    a_prev, a_next = aid(j_pos, atoms["prev_j"]), aid(i_pos, atoms["next_i"])

    n = 0
    pose.add_constraint(AtomPairConstraint(a_i, a_j, HarmonicFunc(chem.ideal_len, len_sd)))
    n += 1
    rad = math.radians
    pose.add_constraint(AngleConstraint(
        a_prev, a_j, a_i, HarmonicFunc(rad(chem.ideal_ang1), rad(ang_sd_deg))))
    pose.add_constraint(AngleConstraint(
        a_j, a_i, a_next, HarmonicFunc(rad(chem.ideal_ang2), rad(ang_sd_deg))))
    n += 2
    if chem.name != "disulfide":
        pose.add_constraint(DihedralConstraint(
            a_prev, a_j, a_i, a_next, HarmonicFunc(rad(chem.ideal_tor), rad(tor_sd_deg))))
        n += 1
    return n


# --------------------------------------------------------------------------------------
# anchor identity / mutation budget bookkeeping
# --------------------------------------------------------------------------------------

def anchor_mutation_cost(
    native_aa_i: str, native_aa_j: str, chem: RingChemistry
) -> list[dict[str, object]]:
    """Every allowed anchor assignment for `chem`, with what it costs in mutations.

    Returns one row per option, each with `aa_i`, `aa_j` and `n_anchor_mutations`, sorted
    cheapest first. For `mainchain` the single row costs 0 -- every amino acid already has a
    backbone N and C, so head-to-tail needs no anchor mutation at all.

    This is the function that makes "chemically impossible at this budget" explicit rather
    than a mysteriously empty result: a terminal disulfide costs 2 mutations unless the LP
    already ends in Cys on both sides, so budgets 0 and 1 are *empty by chemistry*, not by
    search failure. Stage 0 measured exactly that on 8/8 cases.
    """
    if chem.anchor_options is None:
        return [{"aa_i": native_aa_i, "aa_j": native_aa_j, "n_anchor_mutations": 0}]
    rows = []
    for aa_i, aa_j in chem.anchor_options:
        cost = int(aa_i != native_aa_i) + int(aa_j != native_aa_j)
        rows.append({"aa_i": aa_i, "aa_j": aa_j, "n_anchor_mutations": cost})
    rows.sort(key=lambda r: (r["n_anchor_mutations"], r["aa_i"], r["aa_j"]))
    return rows


def min_anchor_cost(native_aa_i: str, native_aa_j: str, chem: RingChemistry) -> int:
    """Cheapest anchor cost for this chemistry on this LP, in mutations."""
    return min(r["n_anchor_mutations"]
               for r in anchor_mutation_cost(native_aa_i, native_aa_j, chem))


def feasible_at_budget(
    native_aa_i: str, native_aa_j: str, chem: RingChemistry, budget: int
) -> bool:
    """Is `chem` reachable at all within `budget` total mutations on this LP?"""
    return min_anchor_cost(native_aa_i, native_aa_j, chem) <= budget


def released_residues(
    binder_pos: list[int], k_nterm: int, m_cterm: int
) -> dict[str, object]:
    """Which residues a (k, m) window releases, recorded both ways.

    `loop_order` is the ring-order traversal GenKIC needs: the C-terminal window first, then
    across the closing bond into the N-terminal window. `anchor_lower`/`anchor_upper` are the
    flanking residues GenKIC holds fixed -- they must exist and be distinct, which is what
    `k + m <= L - 2` enforces.
    """
    L = len(binder_pos)
    if k_nterm < 1 or m_cterm < 1:
        raise ValueError(f"window must release >=1 residue per terminus, got k={k_nterm} m={m_cterm}")
    if k_nterm + m_cterm > L - 2:
        raise ValueError(
            f"window k={k_nterm} m={m_cterm} leaves no fixed anchor on a {L}-residue peptide "
            f"(need k + m <= L - 2 = {L - 2})"
        )
    c_side = [binder_pos[L - m_cterm + t] for t in range(m_cterm)]
    n_side = [binder_pos[t] for t in range(k_nterm)]
    return {
        "k_nterm": k_nterm,
        "m_cterm": m_cterm,
        "n_released": k_nterm + m_cterm,
        "released_local": list(range(k_nterm)) + list(range(L - m_cterm, L)),
        "released_pose": n_side + c_side,
        "loop_order": c_side + n_side,
        "anchor_lower": binder_pos[L - m_cterm - 1],
        "anchor_upper": binder_pos[k_nterm],
    }


def window_ladder(L: int, max_released: int | None = None) -> list[tuple[int, int]]:
    """The (k, m) expansion ladder, smallest movable region first.

    Ordered by total released residues then by asymmetry, so the search reaches a closure
    with the *least* of the peptide disturbed before it tries releasing more. GenKIC needs
    roughly six loop residues to close reliably -- measured on `LNR_2qos_C_A`, where k=m=2
    found no solution in 2000 attempts while k=m=3 closed in 0.1 s -- but the small windows
    are kept in the ladder because they are cheap to try and, when they do close, they are
    the best-retaining answers.
    """
    cap = L - 2 if max_released is None else min(max_released, L - 2)
    out: list[tuple[int, int]] = []
    for total in range(2, cap + 1):
        for k in range(1, total):
            m = total - k
            if k + m <= L - 2:
                out.append((k, m))
    out.sort(key=lambda km: (km[0] + km[1], abs(km[0] - km[1]), -km[0]))
    return out


def chemistry(name: str) -> RingChemistry:
    if name not in CHEMISTRIES:
        raise ValueError(f"unknown chemistry {name!r}; have {sorted(CHEMISTRIES)}")
    return CHEMISTRIES[name]


def all_chemistry_names() -> Iterable[str]:
    return CHEMISTRIES.keys()
