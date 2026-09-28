"""The acceptance contract for a cyclized peptide: does this candidate actually qualify?

One module so that every arm -- Rosetta/GenKIC, SDEdit, anything later -- is graded by the
same code rather than by three similar-looking reimplementations. `evaluate_candidate`
returns a flat row; `accept` turns a row into a verdict plus the list of reasons it failed.

The contract (fixed for the experiment; see `docs/README_GENKIC_LP2CP.md`)
------------------------------------------------------------------------
* the requested terminal chemistry, an explicitly declared covalent bond, and bond geometry
  inside the chemistry's window (`cp_topology.ring_bond_geometry`). CB proximity is not
  closure and is not accepted as a substitute.
* sequence: >= 80% identity to the ORIGINAL LP **and** <= 2 substitutions, anchor mutations
  included. Explicitly protected residues unchanged.
* contacts: >= 80% of the original peptide-receptor residue pairs retained, a contact being
  any heavy-atom pair within 4.5 A. Terminal contacts are counted like every other -- the
  termini are where the edit happens, so exempting them would grade the arm on the part it
  did not touch.
* pose: whole-peptide CA RMSD <= 2 A and interface-residue CA RMSD <= 1 A, interface residues
  taken from the ORIGINAL LP's heavy-atom contacts, measured in the receptor frame with no
  separate peptide alignment.
* energy: paired interface degradation <= +5 REU (CP minus LP, identical preparation,
  scoring and relaxation). The stricter <= 0 result is reported alongside, never instead.
* no severe clashes, no stereochemical defects.

Applied to the FINAL RELAXED candidate. Pre-relax values are recorded next to post-relax
ones under a `pre_` prefix so a candidate that only became acceptable through relaxation is
visible as such.

REU are Rosetta energy units. They are not kcal/mol and are not free energies.

Two traps this module is built around
-------------------------------------
1. **"Measured in the receptor frame" requires checking the frame.** For a candidate built
   by moving a peptide inside a fixed receptor, the frames are identical and a direct RMSD is
   the right measurement. For a candidate that arrived as a separately written complex, they
   may not be -- so `align_on_receptor` superimposes on the RECEPTOR only, never on the
   peptide, and the residual is reported. Aligning on the peptide would hide exactly the
   rigid-body drift the metric is meant to catch.
2. **Contacts are residue PAIRS, not counts.** Retention is `|kept & original| / |original|`,
   so a candidate that sheds three native contacts and forms three new ones scores 0.75 on a
   4-contact interface rather than 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

# --- contract constants (fixed for the experiment) ------------------------------------
HEAVY_CONTACT_CUTOFF_A = 4.5      # primary, heavy-atom
CA_CONTACT_CUTOFF_A = 10.0        # legacy CA-CA metric, kept for historical comparison
                                  # (scripts/sdedit_cyclize.py CONTACT_CUTOFF_NM = 1.0 nm)
MIN_SEQ_IDENTITY = 0.80
MAX_SUBSTITUTIONS = 2
MIN_CONTACT_RETENTION = 0.80
MAX_CA_RMSD_A = 2.0
MAX_INTERFACE_CA_RMSD_A = 1.0
MAX_DDG_REU = 5.0
STRICT_DDG_REU = 0.0
SEVERE_CLASH_FACTOR = 0.75        # heavy-atom pair closer than this * sum of vdW radii
RECEPTOR_FRAME_TOL_A = 0.05       # above this, align on the receptor before measuring

# Reporting grids. These do NOT move the primary bar; they show how the result behaves as
# the bar moves, which is the difference between a sensitivity analysis and a relabelling.
REPORT_MUTATION_BUDGETS = (0, 1, 2, 3)
REPORT_IDENTITY_TIERS = (0.90, 0.80, 0.70)

_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90, "H": 1.10}

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Ideal backbone bond lengths, for the stereochemistry guard.
_IDEAL_BB = {("N", "CA"): 1.458, ("CA", "C"): 1.524, ("C", "O"): 1.231}


@dataclass(frozen=True)
class ContractThresholds:
    """The acceptance bar. Frozen during an experiment; reported alongside every result."""

    min_seq_identity: float = MIN_SEQ_IDENTITY
    max_substitutions: int = MAX_SUBSTITUTIONS
    min_contact_retention: float = MIN_CONTACT_RETENTION
    max_ca_rmsd_A: float = MAX_CA_RMSD_A
    max_interface_ca_rmsd_A: float = MAX_INTERFACE_CA_RMSD_A
    max_ddg_reu: float = MAX_DDG_REU
    heavy_contact_cutoff_A: float = HEAVY_CONTACT_CUTOFF_A

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_THRESHOLDS = ContractThresholds()


# --------------------------------------------------------------------------------------
# sequence
# --------------------------------------------------------------------------------------

def one_letter(name3: str) -> str:
    """Single-letter code, `X` for anything non-canonical (patched types keep their parent)."""
    return _THREE_TO_ONE.get(name3.upper(), "X")


def peptide_sequence(pose, binder_pos: Sequence[int]) -> str:
    return "".join(one_letter(pose.residue(p).name3()) for p in binder_pos)


def sequence_metrics(
    original: str, candidate: str, protected_local: Iterable[int] = ()
) -> dict[str, object]:
    """Identity, substitution count and protected-residue compliance against the ORIGINAL LP.

    Every difference counts, anchor mutations included -- the anchors are the point of the
    edit, and a budget that exempted them would not bound anything.

    Raises:
        ValueError: the two sequences differ in length. Peptide length is held fixed for the
            primary comparison, so a length change is a bug rather than a candidate.
    """
    if len(original) != len(candidate):
        raise ValueError(
            f"peptide length changed ({len(original)} -> {len(candidate)}); length is held "
            "fixed for the primary comparison"
        )
    L = len(original)
    mutated = [k for k in range(L) if original[k] != candidate[k]]
    protected = sorted(set(protected_local))
    violated = [k for k in protected if k in mutated]
    return {
        "peptide_length": L,
        "orig_sequence": original,
        "cand_sequence": candidate,
        "n_substitutions": len(mutated),
        "mutated_local": mutated,
        "seq_identity": round((L - len(mutated)) / L, 4) if L else float("nan"),
        "protected_local": protected,
        "protected_violations": violated,
        "protected_ok": not violated,
    }


# --------------------------------------------------------------------------------------
# contacts
# --------------------------------------------------------------------------------------

def _heavy_atoms(pose, pos: int) -> list[tuple[float, float, float]]:
    res = pose.residue(pos)
    out = []
    for k in range(1, res.natoms() + 1):
        if res.atom_type(k).element().strip().upper() == "H":
            continue
        if res.atom_type(k).is_virtual():
            continue
        v = res.xyz(k)
        out.append((v[0], v[1], v[2]))
    return out


def _ca(pose, pos: int) -> tuple[float, float, float]:
    v = pose.residue(pos).xyz("CA")
    return (v[0], v[1], v[2])


def heavy_atom_contacts(
    pose,
    binder_pos: Sequence[int],
    receptor_pos: Sequence[int],
    cutoff: float = HEAVY_CONTACT_CUTOFF_A,
) -> set[tuple[int, int]]:
    """Residue pairs `(binder_local_index, receptor_pose_position)` in heavy-atom contact.

    The binder side is a LOCAL index so the set stays comparable across poses; the receptor
    side is a pose position, which is stable because the receptor is never renumbered.

    Receptor residues are pre-screened by CA distance (cutoff + 12 A covers any side chain
    pair) before the heavy-atom pass, which keeps this cheap on the large receptors in the
    set -- `LNR_2np0_A_B` has 1289.
    """
    pep_ca = [_ca(pose, p) for p in binder_pos]
    screen = cutoff + 12.0
    near = [r for r in receptor_pos
            if any(math.dist(_ca(pose, r), c) <= screen for c in pep_ca)]
    pep_heavy = [_heavy_atoms(pose, p) for p in binder_pos]
    rec_heavy = {r: _heavy_atoms(pose, r) for r in near}
    out: set[tuple[int, int]] = set()
    for li, atoms_p in enumerate(pep_heavy):
        for r in near:
            for ap in atoms_p:
                hit = False
                for ar in rec_heavy[r]:
                    if math.dist(ap, ar) <= cutoff:
                        hit = True
                        break
                if hit:
                    out.add((li, r))
                    break
    return out


def ca_contacts(
    pose,
    binder_pos: Sequence[int],
    receptor_pos: Sequence[int],
    cutoff: float = CA_CONTACT_CUTOFF_A,
) -> set[tuple[int, int]]:
    """Legacy CA-CA contact set, reproducing `scripts/sdedit_cyclize.py`'s 1.0 nm definition.

    Kept only so new numbers can be read against the historical `contact_retention` column.
    It is a coarse neighbour count, not an interface measurement -- `README_LP2CP_PRECHECK`
    measured r^2 0.004 against Rosetta interface dG -- so the heavy-atom set above is primary.
    """
    out: set[tuple[int, int]] = set()
    for li, p in enumerate(binder_pos):
        cp = _ca(pose, p)
        for r in receptor_pos:
            if math.dist(cp, _ca(pose, r)) <= cutoff:
                out.add((li, r))
    return out


def retention(original: set[tuple[int, int]], candidate: set[tuple[int, int]]) -> float:
    """Fraction of the ORIGINAL contact pairs still present. NaN on an empty original set."""
    if not original:
        return float("nan")
    return len(original & candidate) / len(original)


def interface_local_indices(original_contacts: set[tuple[int, int]]) -> list[int]:
    """Binder-local indices that touch the receptor in the ORIGINAL LP."""
    return sorted({li for li, _ in original_contacts})


# --------------------------------------------------------------------------------------
# geometry: RMSD in the receptor frame
# --------------------------------------------------------------------------------------

def receptor_frame_residual(pose_a, pose_b, receptor_pos: Sequence[int]) -> float:
    """Max receptor CA displacement between two poses; 0 means one shared frame."""
    if not receptor_pos:
        return float("nan")
    return max(math.dist(_ca(pose_a, r), _ca(pose_b, r)) for r in receptor_pos)


def align_on_receptor(mobile_pose, ref_pose, receptor_pos: Sequence[int]) -> float:
    """Superimpose `mobile_pose` onto `ref_pose` using RECEPTOR CA atoms only, in place.

    Never uses peptide atoms: the peptide's displacement is the measurement, so including it
    in the fit would absorb the very drift being measured. Returns the post-fit receptor
    residual.
    """
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.core.scoring import superimpose_pose
    from pyrosetta.rosetta.std import map_core_id_AtomID_core_id_AtomID

    amap = map_core_id_AtomID_core_id_AtomID()
    for r in receptor_pos:
        amap[AtomID(mobile_pose.residue(r).atom_index("CA"), r)] = AtomID(
            ref_pose.residue(r).atom_index("CA"), r)
    superimpose_pose(mobile_pose, ref_pose, amap)
    return receptor_frame_residual(mobile_pose, ref_pose, receptor_pos)


def ca_rmsd(pose_a, pose_b, positions: Sequence[int]) -> float:
    """Direct CA RMSD over `positions`, with no superposition of its own."""
    if not positions:
        return float("nan")
    sq = [math.dist(_ca(pose_a, p), _ca(pose_b, p)) ** 2 for p in positions]
    return math.sqrt(sum(sq) / len(sq))


def per_residue_ca_deviation(pose_a, pose_b, positions: Sequence[int]) -> list[float]:
    return [round(math.dist(_ca(pose_a, p), _ca(pose_b, p)), 3) for p in positions]


# --------------------------------------------------------------------------------------
# clashes and stereochemistry
# --------------------------------------------------------------------------------------

def _element(pose, pos: int, k: int) -> str:
    return pose.residue(pos).atom_type(k).element().strip().upper()


def clash_metrics(
    pose,
    binder_pos: Sequence[int],
    receptor_pos: Sequence[int],
    ring_pair: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Severe heavy-atom overlaps involving the peptide.

    A pair counts as a severe clash when its distance is below
    `SEVERE_CLASH_FACTOR * (vdW_i + vdW_j)`. Excluded: atoms in the same residue, in
    sequence-adjacent peptide residues, and in the two ring-bonded anchors -- all of which
    are genuinely bonded or 1-3 related and would otherwise register as permanent clashes.
    """
    exempt: set[frozenset[int]] = set()
    for a, b in zip(binder_pos, binder_pos[1:]):
        exempt.add(frozenset((a, b)))
    if ring_pair:
        exempt.add(frozenset(ring_pair))

    pep = [(p, k) for p in binder_pos
           for k in range(1, pose.residue(p).natoms() + 1)
           if _element(pose, p, k) != "H" and not pose.residue(p).atom_type(k).is_virtual()]
    pep_ca = [_ca(pose, p) for p in binder_pos]
    near_rec = [r for r in receptor_pos
                if any(math.dist(_ca(pose, r), c) <= 20.0 for c in pep_ca)]
    partners = [(r, k) for r in near_rec
                for k in range(1, pose.residue(r).natoms() + 1)
                if _element(pose, r, k) != "H" and not pose.residue(r).atom_type(k).is_virtual()]

    worst = float("inf")
    n_clash = 0
    n_intra = 0
    # peptide vs receptor
    for (p, kp) in pep:
        vp = pose.residue(p).xyz(kp)
        ep = _VDW.get(_element(pose, p, kp), 1.7)
        for (r, kr) in partners:
            vr = pose.residue(r).xyz(kr)
            d = (vp - vr).norm()
            lim = SEVERE_CLASH_FACTOR * (ep + _VDW.get(_element(pose, r, kr), 1.7))
            if d < lim:
                n_clash += 1
            worst = min(worst, d / (ep + _VDW.get(_element(pose, r, kr), 1.7)))
    # peptide internal
    for a in range(len(pep)):
        pa, ka = pep[a]
        va = pose.residue(pa).xyz(ka)
        ea = _VDW.get(_element(pose, pa, ka), 1.7)
        for b in range(a + 1, len(pep)):
            pb, kb = pep[b]
            if pa == pb or frozenset((pa, pb)) in exempt:
                continue
            d = (va - pose.residue(pb).xyz(kb)).norm()
            eb = _VDW.get(_element(pose, pb, kb), 1.7)
            if d < SEVERE_CLASH_FACTOR * (ea + eb):
                n_intra += 1
    return {
        "n_severe_clashes_interface": n_clash,
        "n_severe_clashes_intrapeptide": n_intra,
        "worst_overlap_ratio": round(worst, 3) if worst != float("inf") else float("nan"),
        "clashes_ok": bool(n_clash == 0 and n_intra == 0),
    }


def _dihedral(p0, p1, p2, p3) -> float:
    b0 = [p0[k] - p1[k] for k in range(3)]
    b1 = [p2[k] - p1[k] for k in range(3)]
    b2 = [p3[k] - p2[k] for k in range(3)]
    n1 = math.sqrt(sum(x * x for x in b1)) or 1e-9
    b1 = [x / n1 for x in b1]
    v = [b0[k] - sum(b0[m] * b1[m] for m in range(3)) * b1[k] for k in range(3)]
    w = [b2[k] - sum(b2[m] * b1[m] for m in range(3)) * b1[k] for k in range(3)]
    x = sum(v[k] * w[k] for k in range(3))
    cross = [b1[1] * v[2] - b1[2] * v[1], b1[2] * v[0] - b1[0] * v[2],
             b1[0] * v[1] - b1[1] * v[0]]
    return math.degrees(math.atan2(sum(cross[k] * w[k] for k in range(3)), x))


def _atom_xyz(pose, pos: int, name: str):
    v = pose.residue(pos).xyz(name)
    return (v[0], v[1], v[2])


def stereochemistry_metrics(
    pose, ref_pose, binder_pos: Sequence[int]
) -> dict[str, object]:
    """Chirality and backbone-geometry guards on the peptide.

    Chirality is checked against the ORIGINAL LP rather than an absolute constant: the input
    crystal peptide is all-L by construction, so the reference sign of the CA improper
    dihedral (N, CA, C, CB) is read off it and any candidate residue whose sign differs is a
    flipped centre. Self-calibrating, so it cannot be wrong about the convention.

    Backbone-geometry deviations are measured directly rather than inferred from a
    `cart_bonded` score, so the number means Angstroms.
    """
    flipped = []
    for li, p in enumerate(binder_pos):
        res, ref = pose.residue(p), ref_pose.residue(binder_pos[li])
        if not (res.has("CB") and ref.has("CB")):
            continue                                   # glycine
        d_cand = _dihedral(_atom_xyz(pose, p, "N"), _atom_xyz(pose, p, "CA"),
                           _atom_xyz(pose, p, "C"), _atom_xyz(pose, p, "CB"))
        d_ref = _dihedral(_atom_xyz(ref_pose, binder_pos[li], "N"),
                          _atom_xyz(ref_pose, binder_pos[li], "CA"),
                          _atom_xyz(ref_pose, binder_pos[li], "C"),
                          _atom_xyz(ref_pose, binder_pos[li], "CB"))
        if d_cand * d_ref < 0:
            flipped.append(li)

    max_bond_dev = 0.0
    for p in binder_pos:
        res = pose.residue(p)
        for (a, b), ideal in _IDEAL_BB.items():
            if res.has(a) and res.has(b):
                d = abs((res.xyz(a) - res.xyz(b)).norm() - ideal)
                max_bond_dev = max(max_bond_dev, d)

    # omega across every peptide bond inside the binder, including the ring bond when the
    # anchors are sequence-adjacent in the ring.
    omegas = []
    for a, b in zip(binder_pos, binder_pos[1:]):
        if not (pose.residue(a).has("CA") and pose.residue(b).has("N")):
            continue
        omegas.append(_dihedral(_atom_xyz(pose, a, "CA"), _atom_xyz(pose, a, "C"),
                                _atom_xyz(pose, b, "N"), _atom_xyz(pose, b, "CA")))
    worst_omega = max((min(abs(abs(o) - 180.0), abs(o)) for o in omegas), default=float("nan"))
    return {
        "n_chirality_flipped": len(flipped),
        "chirality_flipped_local": flipped,
        "max_bb_bond_deviation_A": round(max_bond_dev, 3),
        "worst_omega_deviation_deg": round(worst_omega, 1) if omegas else float("nan"),
        "stereochem_ok": bool(not flipped and max_bond_dev <= 0.05),
    }


# --------------------------------------------------------------------------------------
# energies
# --------------------------------------------------------------------------------------

def peptide_internal_energy(
    pose, binder_pos: Sequence[int], score_function, ring: dict | None = None
) -> float:
    """ref2015 total score of the peptide alone, ring intact.

    Reported as strain: `ddG` says whether the interface survived, this says whether the
    peptide paid for it internally.

    `pdbslice` carries residue TYPES (including patches and disulfide variants) but does NOT
    carry the inter-residue connections between them. For a disulfide that is fatal rather
    than cosmetic: both CYS arrive still flagged as disulfide-bonded, their partner mapping
    is gone, and ref2015's `FullatomDisulfideEnergyContainer` aborts the process with
    "Could not find disulfide partner for residue 1" -- a C++ UtilityExitException, not a
    catchable Python error at the point it matters. Measured on the first real complex.

    So the ring is re-declared on the slice. In the subpose the anchors are residues 1 and
    `len(binder_pos)` by construction (pdbslice preserves order), which is the mapping used
    here. `ring` carries the chemistry and its four atom names; without it the slice is
    scored open and the returned strain describes a linear peptide, so the caller passes it
    for every chemistry.
    """
    import pyrosetta
    from pyrosetta.rosetta.utility import vector1_unsigned_long

    keep = vector1_unsigned_long()
    for p in binder_pos:
        keep.append(p)
    sub = pyrosetta.Pose()
    pyrosetta.rosetta.core.pose.pdbslice(sub, pose, keep)

    if ring is not None:
        import cp_topology as _T

        i_sub, j_sub = 1, sub.size()
        # The slice arrives with the anchors still carrying their variants but no connection
        # between them. Strip the disulfide flag before re-forming, or add_disulfide_bond
        # refuses a residue it already believes is bonded.
        if ring["chem"].name == "disulfide":
            from pyrosetta.rosetta.core.chemical import VariantType
            from pyrosetta.rosetta.core.pose import remove_variant_type_from_pose_residue

            for pos in (i_sub, j_sub):
                remove_variant_type_from_pose_residue(sub, VariantType.DISULFIDE, pos)
        try:
            _T.declare_ring_bond(sub, i_sub, j_sub, ring["chem"], ring["atoms"])
        except (ValueError, RuntimeError):
            # An un-re-declarable slice is reported as a missing measurement, never as a
            # number: a silently open ring would understate strain by exactly the term the
            # metric exists to expose.
            return float("nan")
    return float(score_function(sub))


def interface_metrics(pose, binder_chain: str, receptor_chains: Sequence[str]) -> dict[str, float]:
    """`InterfaceAnalyzerMover` terms on an already-prepared pose.

    Takes a pose rather than a path so the caller controls preparation -- the paired
    comparison depends on the CP and the LP going through identical steps, and re-reading a
    dumped PDB would silently drop the declared ring bond (no PDB writer here emits LINK
    records, and Rosetta would retype the anchors as ordinary free termini).
    """
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    interface = f"{binder_chain}_{''.join(receptor_chains)}"
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    work = pose.clone()
    mover.apply(work)
    keys = ("dG_separated", "dSASA_int", "delta_unsatHbonds", "sc_value", "hbonds_int",
            "packstat", "nres_int")
    return {k: float(work.scores[k]) for k in keys if k in work.scores}


# --------------------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------------------

def accept(row: dict, thresholds: ContractThresholds = DEFAULT_THRESHOLDS) -> dict[str, object]:
    """Apply the contract to one evaluated candidate row.

    Returns `accepted` plus `fail_reasons`, a list naming every criterion that failed rather
    than only the first -- "which bar is binding" is the question the search needs answered.
    A missing metric (NaN) fails its criterion; it is never treated as a pass.
    """
    reasons: list[str] = []

    def num(key):
        v = row.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if v != v else v

    if not row.get("requested_chemistry_satisfied", False):
        reasons.append("wrong_or_missing_chemistry")
    if not row.get("bond_is_declared", False):
        reasons.append("bond_not_declared")
    if not row.get("bond_ok", False):
        reasons.append("bond_geometry")

    ident = num("seq_identity")
    if ident is None or ident < thresholds.min_seq_identity:
        reasons.append("seq_identity")
    nsub = row.get("n_substitutions")
    if nsub is None or int(nsub) > thresholds.max_substitutions:
        reasons.append("n_substitutions")
    if not row.get("protected_ok", True):
        reasons.append("protected_residue_mutated")

    ret = num("contact_retention_heavy45")
    if ret is None or ret < thresholds.min_contact_retention:
        reasons.append("contact_retention")

    rmsd = num("ca_rmsd_A")
    if rmsd is None or rmsd > thresholds.max_ca_rmsd_A:
        reasons.append("ca_rmsd")
    irmsd = num("interface_ca_rmsd_A")
    if irmsd is None or irmsd > thresholds.max_interface_ca_rmsd_A:
        reasons.append("interface_ca_rmsd")

    ddg = num("ddg_interface_reu")
    if ddg is None or ddg > thresholds.max_ddg_reu:
        reasons.append("ddg_interface")

    if not row.get("clashes_ok", False):
        reasons.append("severe_clash")
    if not row.get("stereochem_ok", False):
        reasons.append("stereochemistry")

    return {
        "accepted": not reasons,
        "fail_reasons": reasons,
        "accepted_strict_ddg": bool(
            not reasons and ddg is not None and ddg <= STRICT_DDG_REU),
    }


def tier_flags(row: dict, thresholds: ContractThresholds = DEFAULT_THRESHOLDS) -> dict[str, bool]:
    """Acceptance under each reporting budget/identity tier, primary bar left untouched.

    `pass_budget{b}` relaxes only the substitution cap; `pass_identity{t}` relaxes only the
    identity floor. Everything else -- chemistry, bond, contacts, RMSD, energy, clashes --
    holds at the primary value, so a weaker tier is never a primary success under a new name.
    """
    out: dict[str, bool] = {}
    base = accept(row, thresholds)
    non_seq = [r for r in base["fail_reasons"] if r not in ("n_substitutions", "seq_identity")]
    nsub = row.get("n_substitutions")
    ident = row.get("seq_identity")
    for b in REPORT_MUTATION_BUDGETS:
        ok = not non_seq and nsub is not None and int(nsub) <= b
        ok = ok and (ident is not None and float(ident) >= thresholds.min_seq_identity)
        out[f"pass_budget{b}"] = bool(ok)
    for t in REPORT_IDENTITY_TIERS:
        ok = not non_seq and ident is not None and float(ident) >= t
        ok = ok and (nsub is not None and int(nsub) <= thresholds.max_substitutions)
        out[f"pass_identity{int(t * 100)}"] = bool(ok)
    return out
