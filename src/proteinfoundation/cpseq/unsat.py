"""Which polar atoms the interface buries without giving them a hydrogen-bond partner.

Why this exists
---------------
`README_CPSEQ_LLM_SEARCH.md` §2 shows the proposer a per-position table of the
interactions a candidate *makes*. It has no column for the interactions a candidate
*owes*. A buried lysine whose NZ points into a hydrophobic wall renders as
`hydrophobic:A:Leu104`, indistinguishable from a good anchor, so the one defect that
large-but-poorly-matched interfaces are made of is invisible to the model that would
have to fix it.

The physics, in one paragraph: a nitrogen or oxygen on a protein surface in water is
hydrogen-bonded to water for free. Binding squeezes that water out. A polar atom left in
the dry zone with no replacement partner has paid the desolvation cost and received
nothing, and each one is a drag on affinity. Counting them per atom -- not as a single
total -- is what turns "this interface is worse than it looks" into "position 7's NZ is
buried and partnerless, and A:Asp112 OD1 is 4.9 A away".

What this is NOT
----------------
It is **not** Rosetta's `delta_unsatHbonds`, the number `scorer.py` already records and
`README_CPSEQ_LLM_SEARCH.md` reports. That one comes from `InterfaceAnalyzerMover`, uses
explicit hydrogens placed by Rosetta and an energy cutoff, and arrives as one integer
with no atom attribution. This module reimplements the *concept* geometrically on
heavy atoms, because the attribution is the whole point and Rosetta does not hand it
over. The two therefore measure the same idea by different means and will not agree atom
for atom; `cpseq_unsat_probe.py` measures how closely they agree at all, and nothing
should be shown to a proposer or written into a report until that agreement is known.

Like `interactions.py`, this scores nothing and gates nothing. It describes.

Heavy atoms only, and what that costs
-------------------------------------
Boltz-2 writes no hydrogens, so donor geometry cannot be checked against an H position.
Satisfaction is therefore a heavy-atom criterion: a complementary donor/acceptor within
`HBOND_HEAVY_MAX_A`. Two consequences, both deliberate:

- It is **permissive about satisfaction**, which makes the unsat count conservative. An
  atom this module calls unsatisfied has no candidate partner at all, not merely a badly
  angled one. That is the right direction of error for a measurement whose output is
  "this is a defect, change it".
- Donor-donor and acceptor-acceptor pairs are excluded by the typing tables below, so
  two backbone carbonyls 3.4 A apart do not satisfy each other.

Three geometric artifacts are excluded explicitly, because each would otherwise report a
covalent bond as a hydrogen bond:

- Sequence-adjacent backbone N/O. `N(i)` and `O(i-1)` both hang off `C(i-1)` at ~2.25 A.
- The head-to-tail closing amide of a cyclic peptide, which makes residue 1 and residue
  L adjacent in exactly the same way. The peptide chain's adjacency is therefore treated
  as cyclic, which is safe for a linear peptide too: it only ever removes a 1<->L pair
  that a linear peptide's termini are too far apart to have formed.
- Anything closer than `HBOND_HEAVY_MIN_A`, which no real hydrogen bond reaches.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from proteinfoundation.cpseq.interactions import (
    THREE_TO_ONE,
    _pick_peptide_chain,
    _res_label,
    resolve_pdb,
)

# Burial takes TWO conditions, and the first version of this module had only one.
#
# With an absolute-area rule alone, the only buried polar atoms on a helical peptide are
# backbone N and O already satisfied by the peptide's own i,i+4 network -- measured on
# `lnr_2v8x`/`RIIYDRKFLMECRN`, all 16 of them -- so the interface-caused count came out 0
# while Rosetta reported 5. The side chains that actually face the receptor sit at
# partial burial and were never examined.
#
# Adding a desolvation condition alone is wrong in the other direction. That same fold's
# `B:ARG13:NH1` loses 26 A^2 on binding and keeps 24 A^2, so it still reaches water and
# pays no desolvation penalty; calling it unsatisfied would invent a defect.
#
# So an atom is a defect when the interface took its water away (`dSASA`) AND left it
# little (`sasa_complex`). Both thresholds are provisional until `cpseq_unsat_probe.py`
# reports which setting tracks Rosetta's number; nothing should be shown to a proposer on
# the defaults alone.
BURIED_SASA_MAX_A2 = 2.5
DESOLVATED_MIN_A2 = 2.0
# The grid the probe scores. Kept here rather than in the probe so the thresholds a run
# was judged on and the thresholds the renderer uses cannot drift apart.
THRESHOLD_GRID: tuple[tuple[float, float], ...] = (
    (0.5, 1.0), (0.5, 2.0), (1.0, 1.0), (1.0, 2.0), (2.5, 2.0),
    (2.5, 5.0), (5.0, 2.0), (5.0, 5.0), (5.0, 10.0),
)
# Heavy-atom donor/acceptor separation counted as a hydrogen bond. 3.5 A is the usual
# N/O...N/O cutoff when no hydrogens are present.
HBOND_HEAVY_MAX_A = 3.5
# Below this, the pair is not a hydrogen bond but a covalent or geminal arrangement.
HBOND_HEAVY_MIN_A = 2.4
# How far to look for a complementary partner an unsatisfied atom could *reach* if the
# position were substituted. Purely a hint for the prompt; it enters no count.
REACHABLE_PARTNER_MAX_A = 6.5

# Protonation and naming variants collapse onto the canonical residue they are a form
# of, so the typing tables below need one entry each. `interactions.THREE_TO_ONE`
# already enumerates the variants PyRosetta's relax leaves behind.
_CANONICAL_RESNAME = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG", "TYM": "TYR",
    "MSE": "MET",
}

# Side-chain hydrogen-bond donors: heavy atoms carrying a donatable hydrogen.
_SIDECHAIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "TRP": {"NE1"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}

# Side-chain acceptors. His ND1/NE2 and the hydroxyls appear in both tables: which role a
# histidine nitrogen or a hydroxyl oxygen plays depends on a tautomer and a rotamer that
# a hydrogen-free structure does not determine, and admitting both is the permissive
# choice this module's docstring commits to.
_SIDECHAIN_ACCEPTORS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}

_BACKBONE_N = {"N"}
_BACKBONE_O = {"O", "OXT"}


def _canonical(resname: str) -> str:
    r = resname.strip().upper()
    return _CANONICAL_RESNAME.get(r, r)


def atom_roles(resname: str, atom_name: str) -> tuple[bool, bool]:
    """`(is_donor, is_acceptor)` for one heavy atom, by residue and PDB atom name."""
    res = _canonical(resname)
    name = atom_name.strip().upper()
    # Proline's backbone nitrogen is tertiary and carries no hydrogen to donate.
    is_donor = (name in _BACKBONE_N and res != "PRO") or (
        name in _SIDECHAIN_DONORS.get(res, ())
    )
    is_acceptor = name in _BACKBONE_O or name in _SIDECHAIN_ACCEPTORS.get(res, ())
    return is_donor, is_acceptor


# --------------------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------------------


@dataclass
class PolarAtom:
    """One polar heavy atom, with everything needed to decide and explain its state."""

    chain: str
    resname: str
    resnum: int
    atom: str
    pos: int | None  # 1-based peptide position; None for a receptor atom
    is_donor: bool
    is_acceptor: bool
    is_backbone: bool
    index: int = -1  # row in the coordinate array
    sasa_complex: float | None = None
    sasa_alone: float | None = None
    n_partners_complex: int = 0
    n_partners_alone: int = 0
    partners_complex: list[str] = field(default_factory=list)
    reachable: list[dict] = field(default_factory=list)

    @property
    def dsasa(self) -> float | None:
        """Area lost on binding. None when either state is missing, never 0.0, so that
        "not measured" cannot be read as "lost nothing"."""
        if self.sasa_complex is None or self.sasa_alone is None:
            return None
        return max(0.0, self.sasa_alone - self.sasa_complex)

    def buried_complex_at(self, buried_max: float, desolv_min: float) -> bool:
        """Desolvated by the interface and left with little area, both required."""
        if self.sasa_complex is None:
            return False
        ds = self.dsasa
        return self.sasa_complex <= buried_max and ds is not None and ds >= desolv_min

    def buried_alone_at(self, buried_max: float) -> bool:
        """Buried in the unbound partner. There is no desolvation condition here: the
        question this answers is whether the atom was ALREADY enclosed before binding,
        which is an absolute property of the unbound structure."""
        return self.sasa_alone is not None and self.sasa_alone <= buried_max

    def unsat_complex_at(self, buried_max: float, desolv_min: float) -> bool:
        return (
            self.buried_complex_at(buried_max, desolv_min)
            and self.n_partners_complex == 0
        )

    def unsat_alone_at(self, buried_max: float) -> bool:
        return self.buried_alone_at(buried_max) and self.n_partners_alone == 0

    # Convenience at the module defaults, for the renderer and the cached record.
    @property
    def buried_complex(self) -> bool:
        return self.buried_complex_at(BURIED_SASA_MAX_A2, DESOLVATED_MIN_A2)

    @property
    def buried_alone(self) -> bool:
        return self.buried_alone_at(BURIED_SASA_MAX_A2)

    @property
    def unsat_complex(self) -> bool:
        return self.unsat_complex_at(BURIED_SASA_MAX_A2, DESOLVATED_MIN_A2)

    @property
    def unsat_alone(self) -> bool:
        return self.unsat_alone_at(BURIED_SASA_MAX_A2)

    @property
    def fixable_by_substitution(self) -> bool:
        """A side-chain atom on the peptide. The distinction decides what the whole idea
        is worth: a buried backbone amide and a receptor-side atom are both real defects
        and neither can be repaired by changing the peptide's sequence, so a proposer
        told about them has been handed an impossible instruction."""
        return self.pos is not None and not self.is_backbone

    def label(self) -> str:
        return f"{_res_label(self.chain, self.resname, self.resnum)}:{self.atom}"

    def to_dict(self) -> dict:
        return {
            "label": self.label(),
            "chain": self.chain,
            "resname": self.resname,
            "resnum": self.resnum,
            "atom": self.atom,
            "pos": self.pos,
            "role": ("donor" if self.is_donor else "")
            + ("/" if self.is_donor and self.is_acceptor else "")
            + ("acceptor" if self.is_acceptor else ""),
            "backbone": self.is_backbone,
            "sasa_complex_A2": (
                None if self.sasa_complex is None else round(self.sasa_complex, 2)
            ),
            "sasa_alone_A2": (
                None if self.sasa_alone is None else round(self.sasa_alone, 2)
            ),
            "dsasa_A2": None if self.dsasa is None else round(self.dsasa, 2),
            "fixable_by_substitution": self.fixable_by_substitution,
            "n_partners_complex": self.n_partners_complex,
            "n_partners_alone": self.n_partners_alone,
            "partners_complex": self.partners_complex,
            "unsat_complex": self.unsat_complex,
            "unsat_alone": self.unsat_alone,
            "newly_unsat": self.unsat_complex and not self.unsat_alone,
            "reachable": self.reachable,
        }


# --------------------------------------------------------------------------------------
# Hydrogen-bond partner counting
# --------------------------------------------------------------------------------------


def _adjacency_excluded(
    a: PolarAtom, b: PolarAtom, cyclic_chain: str | None, chain_length: dict[str, int]
) -> bool:
    """True when a backbone N/O pair is covalently adjacent rather than hydrogen bonded.

    Only backbone N against backbone O is affected: a side-chain atom on residue `i+1` is
    free to hydrogen bond residue `i`'s backbone, and excluding those would discard real
    interactions.
    """
    if a.chain != b.chain:
        return False
    a_bb = a.atom in _BACKBONE_N or a.atom in _BACKBONE_O
    b_bb = b.atom in _BACKBONE_N or b.atom in _BACKBONE_O
    if not (a_bb and b_bb):
        return False
    gap = abs(a.resnum - b.resnum)
    if gap == 1:
        return True
    # The closing amide of a head-to-tail cycle makes residue 1 and residue L adjacent.
    if cyclic_chain is not None and a.chain == cyclic_chain:
        n = chain_length.get(a.chain, 0)
        if n > 2 and gap == n - 1:
            return True
    return False


def _count_partners(
    atoms: list[PolarAtom],
    xyz: np.ndarray,
    *,
    subset: np.ndarray | None,
    cyclic_chain: str | None,
    chain_length: dict[str, int],
    record_labels: bool,
) -> tuple[list[int], list[list[str]]]:
    """Complementary hydrogen-bond partners per atom, within `subset` only.

    `subset` is a boolean mask selecting which atoms exist in this state -- the whole
    complex, or one partner alone. Passing a mask rather than rebuilding the structure
    keeps the separated state exactly coincident with the complex, so a difference
    between the two states can only come from the partner's absence and never from a
    re-parse or a different atom order.
    """
    n = len(atoms)
    counts = [0] * n
    labels: list[list[str]] = [[] for _ in range(n)]
    if n == 0:
        return counts, labels
    live = np.ones(n, dtype=bool) if subset is None else subset
    idx_live = np.flatnonzero(live)
    if idx_live.size == 0:
        return counts, labels

    from scipy.spatial import cKDTree

    tree = cKDTree(xyz[idx_live])
    pairs = tree.query_pairs(HBOND_HEAVY_MAX_A, output_type="ndarray")
    for i_local, j_local in pairs:
        i, j = int(idx_live[i_local]), int(idx_live[j_local])
        a, b = atoms[i], atoms[j]
        if a.chain == b.chain and a.resnum == b.resnum:
            continue  # two polar atoms of one residue cannot hydrogen bond each other
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        if d < HBOND_HEAVY_MIN_A:
            continue
        if _adjacency_excluded(a, b, cyclic_chain, chain_length):
            continue
        # Complementary roles, in either direction.
        if not ((a.is_donor and b.is_acceptor) or (a.is_acceptor and b.is_donor)):
            continue
        counts[i] += 1
        counts[j] += 1
        if record_labels:
            labels[i].append(f"{b.label()} {d:.2f}A")
            labels[j].append(f"{a.label()} {d:.2f}A")
    return counts, labels


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def unsat_report(
    pdb_path: str | os.PathLike,
    *,
    peptide_sequence: str | None = None,
    peptide_chain: str | None = None,
    structure_stage: str = "raw",
    treat_peptide_as_cyclic: bool = True,
) -> dict:
    """Buried unsatisfied polar atoms in one complex, per atom and per peptide position.

    Pass the **raw** predicted complex, for the reason `interactions.summarize_complex`
    documents: relax moves atoms to relieve strain and repairs part of the defect a
    proposer needs to see. `structure_stage` records which was passed and changes nothing.

    The counts come in three flavours, and the difference between them matters:

    - `unsat_complex` -- buried and partnerless in the assembled complex.
    - `unsat_alone` -- the same, with each partner evaluated by itself. An atom buried in
      the peptide's own core and unsatisfied there was not the interface's doing.
    - `newly_unsat` -- unsatisfied in the complex and **not** already so alone. This is
      the interface's own contribution, and the quantity comparable in spirit to
      Rosetta's `delta_unsatHbonds`.
    """
    import shutil

    requested = str(pdb_path)
    resolved, tmpdir = resolve_pdb(requested)
    try:
        return _unsat_resolved(
            requested=requested,
            pdb_path=resolved,
            structure_stage=structure_stage,
            peptide_sequence=peptide_sequence,
            peptide_chain=peptide_chain,
            treat_peptide_as_cyclic=treat_peptide_as_cyclic,
        )
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _unsat_resolved(
    *,
    requested: str,
    pdb_path: str | None,
    structure_stage: str,
    peptide_sequence: str | None,
    peptide_chain: str | None,
    treat_peptide_as_cyclic: bool,
) -> dict:
    from Bio.PDB import PDBParser

    out: dict = {
        "source_pdb": requested,
        "structure_stage": structure_stage,
        "ok": False,
        "error": None,
        "peptide_chain": peptide_chain,
        "chain_choice_reason": None,
        "peptide_sequence_expected": peptide_sequence,
        "peptide_sequence_observed": None,
        "sequence_matches": None,
        "criteria": {
            "buried_sasa_max_A2": BURIED_SASA_MAX_A2,
            "desolvated_min_A2": DESOLVATED_MIN_A2,
            "hbond_heavy_min_A": HBOND_HEAVY_MIN_A,
            "hbond_heavy_max_A": HBOND_HEAVY_MAX_A,
            "hydrogens_used": False,
            "thresholds_provisional": True,
        },
        "counts": {},
        "grid": [],
        "atoms": [],
        "by_position": {},
    }
    if pdb_path is None:
        out["error"] = "pdb not found (neither the path nor a .gz beside it)"
        return out
    if pdb_path != requested:
        out["decompressed_from_gz"] = True

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PDBParser(QUIET=True).get_structure("cx", pdb_path)[0]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"parse failed: {type(exc).__name__}: {exc}"
        return out

    if peptide_chain is None:
        peptide_chain, reason = _pick_peptide_chain(model, peptide_sequence)
        out["chain_choice_reason"] = reason
    else:
        out["chain_choice_reason"] = "caller-specified"
    if peptide_chain is None or peptide_chain not in model:
        out["error"] = f"peptide chain {peptide_chain!r} not in structure"
        return out
    out["peptide_chain"] = peptide_chain

    pep_residues = [r for r in model[peptide_chain] if r.id[0] == " "]
    if not pep_residues:
        out["error"] = f"chain {peptide_chain} has no standard residues"
        return out
    observed = "".join(
        THREE_TO_ONE.get(r.get_resname().strip().upper(), "X") for r in pep_residues
    )
    out["peptide_sequence_observed"] = observed
    if peptide_sequence is not None:
        out["sequence_matches"] = observed == peptide_sequence
    pos_of_resnum = {r.id[1]: i for i, r in enumerate(pep_residues, start=1)}

    # --- per-atom SASA, complex and separated ----------------------------------------
    # Computed at atom level on the assembled complex and then on each partner alone. The
    # separated state is produced by deleting the other chain from a copy, which is the
    # same "separate the partners, do not move them" convention InterfaceAnalyzer uses:
    # no repacking, no minimisation, so the only difference is the partner's absence.
    try:
        import copy

        from Bio.PDB.SASA import ShrakeRupley

        sr = ShrakeRupley()
        sr.compute(model, level="A")
        sasa_complex = {
            (ch.id, res.id[1], at.get_name().strip().upper()): float(at.sasa)
            for ch in model
            for res in ch
            if res.id[0] == " "
            for at in res
            if at.element != "H"
        }

        sasa_alone: dict[tuple[str, int, str], float] = {}
        for ch_id in [c.id for c in model]:
            side = copy.deepcopy(model[ch_id])
            sr.compute(side, level="A")
            for res in side:
                if res.id[0] != " ":
                    continue
                for at in res:
                    if at.element == "H":
                        continue
                    sasa_alone[(ch_id, res.id[1], at.get_name().strip().upper())] = float(
                        at.sasa
                    )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"sasa failed: {type(exc).__name__}: {exc}"
        return out

    # --- the polar atom inventory -----------------------------------------------------
    atoms: list[PolarAtom] = []
    coords: list[np.ndarray] = []
    chain_length: dict[str, int] = {}
    for ch in model:
        n_std = 0
        for res in ch:
            if res.id[0] != " ":
                continue  # waters and hetero groups: Boltz writes none, and a crystal
                # water left in a receptor is not a partner the peptide can rely on
            n_std += 1
            resname = res.get_resname().strip()
            for at in res:
                el = (at.element or "").strip().upper()
                if el not in ("N", "O"):
                    continue
                name = at.get_name().strip().upper()
                is_d, is_a = atom_roles(resname, name)
                if not (is_d or is_a):
                    continue  # a polar atom with no assignable role, e.g. an unknown
                    # residue's nitrogen: counting it would invent a defect
                key = (ch.id, res.id[1], name)
                atoms.append(
                    PolarAtom(
                        chain=ch.id,
                        resname=resname,
                        resnum=res.id[1],
                        atom=name,
                        pos=pos_of_resnum.get(res.id[1]) if ch.id == peptide_chain else None,
                        is_donor=is_d,
                        is_acceptor=is_a,
                        is_backbone=name in _BACKBONE_N or name in _BACKBONE_O,
                        index=len(atoms),
                        sasa_complex=sasa_complex.get(key),
                        sasa_alone=sasa_alone.get(key),
                    )
                )
                coords.append(np.asarray(at.coord, dtype=np.float64))
        chain_length[ch.id] = n_std

    if not atoms:
        out["error"] = "no typed polar atoms found"
        return out
    xyz = np.vstack(coords)
    is_pep = np.array([a.chain == peptide_chain for a in atoms], dtype=bool)
    if not is_pep.any() or is_pep.all():
        out["error"] = "structure has no receptor chain outside the peptide"
        return out

    cyclic = peptide_chain if treat_peptide_as_cyclic else None

    # Partners in the complex, then within each side alone.
    cx_counts, cx_labels = _count_partners(
        atoms, xyz, subset=None, cyclic_chain=cyclic,
        chain_length=chain_length, record_labels=True,
    )
    pep_counts, _ = _count_partners(
        atoms, xyz, subset=is_pep, cyclic_chain=cyclic,
        chain_length=chain_length, record_labels=False,
    )
    rec_counts, _ = _count_partners(
        atoms, xyz, subset=~is_pep, cyclic_chain=cyclic,
        chain_length=chain_length, record_labels=False,
    )
    for i, a in enumerate(atoms):
        a.n_partners_complex = cx_counts[i]
        a.partners_complex = sorted(cx_labels[i])
        a.n_partners_alone = pep_counts[i] if is_pep[i] else rec_counts[i]

    # --- the actionable hint: a complementary partner within reach --------------------
    # For each newly unsatisfied PEPTIDE atom, the nearest receptor polar atoms of
    # complementary role that it is not already bonded to. This is what makes the record
    # a proposal rather than a complaint: "nothing here" and "an acceptor 4.9 A away" ask
    # for different substitutions.
    from scipy.spatial import cKDTree

    rec_idx = np.flatnonzero(~is_pep)
    rec_tree = cKDTree(xyz[rec_idx])
    for a in atoms:
        if a.pos is None or not (a.unsat_complex and not a.unsat_alone):
            continue
        near = rec_tree.query_ball_point(xyz[a.index], REACHABLE_PARTNER_MAX_A)
        hits = []
        for j_local in near:
            b = atoms[int(rec_idx[j_local])]
            if not ((a.is_donor and b.is_acceptor) or (a.is_acceptor and b.is_donor)):
                continue
            d = float(np.linalg.norm(xyz[a.index] - xyz[b.index]))
            if d <= HBOND_HEAVY_MAX_A:
                continue  # already a partner, or excluded above for a stated reason
            hits.append({"atom": b.label(), "dist_A": round(d, 2)})
        a.reachable = sorted(hits, key=lambda h: h["dist_A"])[:3]

    # --- counts -----------------------------------------------------------------------
    def _tally(sel: list[PolarAtom], bmax: float, dmin: float) -> dict:
        uc = [a.unsat_complex_at(bmax, dmin) for a in sel]
        ua = [a.unsat_alone_at(bmax) for a in sel]
        return {
            "polar_atoms": len(sel),
            "buried_complex": sum(a.buried_complex_at(bmax, dmin) for a in sel),
            "unsat_complex": sum(uc),
            "unsat_alone": sum(ua),
            # The interface's own contribution, by set difference rather than by
            # subtracting totals: an atom the interface SATISFIES would otherwise cancel
            # one it strands, and those are different events.
            "newly_unsat": sum(c and not a for c, a in zip(uc, ua)),
            "newly_satisfied": sum(a and not c for c, a in zip(uc, ua)),
        }

    pep_atoms = [a for i, a in enumerate(atoms) if is_pep[i]]
    rec_atoms_l = [a for i, a in enumerate(atoms) if not is_pep[i]]
    pep_sc = [a for a in pep_atoms if a.fixable_by_substitution]
    pep_bb = [a for a in pep_atoms if not a.fixable_by_substitution]

    def _grid_entry(bmax: float, dmin: float) -> dict:
        return {
            "buried_sasa_max_A2": bmax,
            "desolvated_min_A2": dmin,
            "total": _tally(atoms, bmax, dmin),
            "peptide": _tally(pep_atoms, bmax, dmin),
            "peptide_sidechain": _tally(pep_sc, bmax, dmin),
            "peptide_backbone": _tally(pep_bb, bmax, dmin),
            "receptor": _tally(rec_atoms_l, bmax, dmin),
        }

    # Counted across the whole grid, because which thresholds reproduce Rosetta's number
    # is an open question and recomputing the structure for each one would cost 1.5 s a
    # fold. The SASA pass and the partner search are already done; thresholding is free.
    out["grid"] = [_grid_entry(b, d) for b, d in THRESHOLD_GRID]
    out["counts"] = _grid_entry(BURIED_SASA_MAX_A2, DESOLVATED_MIN_A2)
    # The one number to compare against `raw_delta_unsatHbonds`, named so it cannot be
    # mistaken for it, plus the split that decides what acting on it could ever buy.
    out["counts"]["geom_newly_unsat_total"] = out["counts"]["total"]["newly_unsat"]
    out["counts"]["geom_newly_unsat_peptide_sidechain"] = out["counts"][
        "peptide_sidechain"
    ]["newly_unsat"]

    # Only the offending atoms are kept: the full polar inventory is thousands of rows
    # per complex and the cached record has to stay prompt-sized.
    out["atoms"] = [
        a.to_dict() for a in atoms if a.unsat_complex or (a.pos is not None and a.unsat_alone)
    ]
    by_pos: dict[int, list[dict]] = {}
    for a in atoms:
        if a.pos is not None and a.unsat_complex:
            by_pos.setdefault(a.pos, []).append(
                {
                    "atom": a.atom,
                    "backbone": a.is_backbone,
                    "sasa_complex_A2": (
                        None if a.sasa_complex is None else round(a.sasa_complex, 2)
                    ),
                    "newly_unsat": a.unsat_complex and not a.unsat_alone,
                    "reachable": a.reachable,
                }
            )
    out["by_position"] = {str(k): v for k, v in sorted(by_pos.items())}
    out["ok"] = True
    return out


# --------------------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------------------


def render_unsat(report: dict, *, max_positions: int = 16) -> str:
    """The unsatisfied atoms as lines a prompt can carry, one per peptide position.

    Side-chain defects are listed before backbone ones because a substitution can fix a
    side chain and cannot fix a backbone amide -- a model told to repair a buried
    backbone N has been handed an impossible instruction.
    """
    if not report.get("ok"):
        return f"[unsat] unavailable: {report.get('error') or 'unknown error'}"
    counts = report.get("counts") or {}
    pep = counts.get("peptide") or {}
    lines = [
        f"[buried unsatisfied polar atoms, geometric] peptide={pep.get('unsat_complex', 0)} "
        f"(interface-caused {pep.get('newly_unsat', 0)}) "
        f"receptor_side={(counts.get('receptor') or {}).get('newly_unsat', 0)}"
    ]
    by_pos = report.get("by_position") or {}
    if not by_pos:
        lines.append("  none: every buried polar peptide atom has a partner.")
        return "\n".join(lines)
    shown = 0
    for pos_s, entries in sorted(by_pos.items(), key=lambda kv: int(kv[0])):
        if shown >= max_positions:
            lines.append(f"  (+{len(by_pos) - shown} more positions)")
            break
        ordered = sorted(entries, key=lambda e: (e["backbone"], e["atom"]))
        parts = []
        for e in ordered:
            tag = e["atom"] + ("(bb)" if e["backbone"] else "")
            if e.get("reachable"):
                r = e["reachable"][0]
                tag += f" -> {r['atom']} {r['dist_A']}A"
            elif not e["newly_unsat"]:
                tag += " (also unsat unbound)"
            parts.append(tag)
        lines.append(f"  pos {int(pos_s):>3}  " + "; ".join(parts))
        shown += 1
    return "\n".join(lines)
