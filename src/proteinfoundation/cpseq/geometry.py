"""Validity and closure geometry, measured on a predicted complex PDB.

What "valid" has to mean here
-----------------------------
A macrocycle can be wrong in ways a single distance cannot see. Two anchor atoms sitting
at exactly 1.33 A can still point in a chemically impossible direction, because distance
constrains neither the angles at either end nor the torsion about the bond. So closure is
measured as length **and** both angles **and** the torsion, per chemistry, against the
same reference values `cyclization.linkage_geometry` supervises training with.

Separately from closure, a structure can be invalid as a *peptide*: a D-amino acid where
an L was requested, a backbone bond at an impossible length, or atoms interpenetrating.
Those are measured too, because a ring that closes perfectly around a mirror-image
residue is not the molecule that was asked for.

Every measurement here is reference-free: it scores the structure on its own chemistry,
never against a native, so it applies identically to a candidate with no known answer and
to an experimental control.

Tolerances
----------
Nothing in this module decides pass/fail. It returns measured values; `tolerances.py`
holds the frozen windows and applies them. That split exists so the tranche's thresholds
can be stated, saved and audited as data rather than being buried in the measurement
code.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from proteinfoundation.cpseq.chemistry import CHEMISTRIES, isopeptide_acid_atom

BACKBONE = ("N", "CA", "C")

# Heavy-atom clash cutoff. 2.2 A is well below any non-bonded contact distance for
# carbon/nitrogen/oxygen pairs (the sum of two carbon vdW radii is ~3.4 A), so a pair
# this close is interpenetration, not a tight contact. Bonded and 1-3 neighbours are
# excluded by construction -- a real C-N bond at 1.33 A is not a clash.
CLASH_CUTOFF_A = 2.2

# Engh & Huber backbone targets, for reporting deviation rather than for pass/fail.
BACKBONE_REFERENCE = {
    "N_CA": 1.458,
    "CA_C": 1.525,
    "C_N": 1.329,
    "omega_trans_deg": 180.0,
}

# Improper N-CA-C-CB for an L-amino acid, measured on this project's control structures
# (420 residues, four LNR crystal chains): mean +122.9 deg, range +113.2 to +132.5.
# See `ca_chirality`, whose sign convention this fixes.
L_AMINO_ACID_IMPROPER_DEG = 122.9


def read_chain(
    pdb_path: str | Path, chain: str, *, standard_only: bool = True
) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Ordered [(resname, {atom: xyz})] for one chain. First model, heavy atoms, altloc ' '/'A'.

    `standard_only` drops anything that is not one of the 20 canonical amino acids. It
    defaults on because experimental PDBs put waters, ions and ligands on the same chain
    id as the peptide: reading 1elr chain B without this filter returns 19 "residues" and
    a sequence of `XMEEVDXXXXXXXXXXXXX` for what is a 5-mer. This version of the dataset
    is canonical-residue only, so a non-standard residue is out of scope rather than
    something to silently coerce.
    """
    from proteinfoundation.cpseq.candidates import THREE_TO_ONE

    residues: dict[tuple[int, str], tuple[str, dict[str, np.ndarray]]] = {}
    order: list[tuple[int, str]] = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")) or line[21] != chain:
                continue
            if line[16] not in (" ", "A"):
                continue
            if standard_only and line[17:20].strip() not in THREE_TO_ONE:
                continue
            atom = line[12:16].strip()
            element = line[76:78].strip()
            if element == "H" or (not element and atom.startswith("H")):
                continue
            key = (int(line[22:26]), line[26])
            if key not in residues:
                residues[key] = (line[17:20].strip(), {})
                order.append(key)
            residues[key][1][atom] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return [residues[k] for k in order]


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _dihedral_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Signed torsion a-b-c-d in degrees, in (-180, 180]."""
    b0, b1, b2 = a - b, c - b, d - c
    n1 = np.cross(b0, b1)
    n2 = np.cross(-b1, b2)
    if np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8:
        return float("nan")
    m1 = np.cross(n1, b1 / np.linalg.norm(b1))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return math.degrees(math.atan2(y, x))


def _closure_atoms(
    residues: list[tuple[str, dict[str, np.ndarray]]], chemistry: str
) -> tuple[dict[str, np.ndarray | None], str]:
    """The four/six atoms the closure geometry is measured on, plus a status string."""
    (name_i, atoms_i), (name_j, atoms_j) = residues[0], residues[-1]
    spec = CHEMISTRIES[chemistry]

    if chemistry == "mainchain":
        # Bond is C(last) -> N(first); angles are CA(last)-C(last)-N(first) and
        # C(last)-N(first)-CA(first); torsion is CA(last)-C(last)-N(first)-CA(first),
        # i.e. the ring's own omega.
        need = {
            "bond_i": atoms_i.get("N"),
            "bond_j": atoms_j.get("C"),
            "angle_i_outer": atoms_i.get("CA"),
            "angle_j_outer": atoms_j.get("CA"),
        }
        missing = [k for k, v in need.items() if v is None]
        return need, ("ok" if not missing else f"missing_atoms:{','.join(missing)}")

    if chemistry == "disulfide":
        if name_i != "CYS" or name_j != "CYS":
            return {}, f"wrong_residues:{name_i}/{name_j}_not_CYS"
        need = {
            "bond_i": atoms_i.get("SG"),
            "bond_j": atoms_j.get("SG"),
            "angle_i_outer": atoms_i.get("CB"),
            "angle_j_outer": atoms_j.get("CB"),
        }
        missing = [k for k, v in need.items() if v is None]
        return need, ("ok" if not missing else f"missing_atoms:{','.join(missing)}")

    if chemistry == "isopeptide":
        # Either orientation; the acid's bonding carbon depends on its identity, and
        # GLU/GLN carry both a CG and a CD, so a first-found search over (CG, CD) would
        # resolve to the wrong atom for those two.
        if name_i == "LYS" and name_j in {"ASP", "ASN", "GLU", "GLN"}:
            lys_atoms, acid_atoms, acid_name = atoms_i, atoms_j, name_j
            flip = False
        elif name_j == "LYS" and name_i in {"ASP", "ASN", "GLU", "GLN"}:
            lys_atoms, acid_atoms, acid_name = atoms_j, atoms_i, name_i
            flip = True
        else:
            return {}, f"wrong_residues:{name_i}/{name_j}_not_LYS+acid"
        cz = isopeptide_acid_atom(acid_name)
        outer = {"ASP": "CB", "ASN": "CB", "GLU": "CG", "GLN": "CG"}[acid_name]
        need = {
            "bond_i": lys_atoms.get("NZ"),
            "bond_j": acid_atoms.get(cz),
            "angle_i_outer": lys_atoms.get("CE"),
            "angle_j_outer": acid_atoms.get(outer),
        }
        missing = [k for k, v in need.items() if v is None]
        status = "ok" if not missing else f"missing_atoms:{','.join(missing)}"
        need["_flipped"] = flip  # type: ignore[assignment]
        return need, status

    return {}, f"unknown_chemistry:{chemistry}"


def closure_geometry(
    pdb_path: str | Path, binder_chain: str, chemistry: str
) -> dict:
    """Measured closure length, both angles and the torsion, with the reference values.

    Returns `status != "ok"` when the atoms the chemistry needs are absent or the
    terminal residues are the wrong identity. A missing atom is **not** a closed ring --
    it is an unmeasurable one, and it is reported as such rather than as a NaN that a
    mean would quietly drop.
    """
    spec = CHEMISTRIES[chemistry]
    out: dict = {
        "chemistry": chemistry,
        "status": "unmeasured",
        "bond_length_A": None,
        "angle_i_deg": None,
        "angle_j_deg": None,
        "dihedral_deg": None,
        "ref_bond_length_A": spec.bond.length_A,
        "ref_angle_i_deg": spec.bond.angle_i_deg,
        "ref_angle_j_deg": spec.bond.angle_j_deg,
        "ref_dihedral_deg": spec.bond.dihedral_deg,
        "dihedral_symmetric": spec.bond.dihedral_symmetric,
    }
    residues = read_chain(pdb_path, binder_chain)
    out["n_residues"] = len(residues)
    if len(residues) < 2:
        out["status"] = "too_few_residues"
        return out

    need, status = _closure_atoms(residues, chemistry)
    out["status"] = status
    if status != "ok":
        return out

    bi, bj = need["bond_i"], need["bond_j"]
    oi, oj = need["angle_i_outer"], need["angle_j_outer"]
    out["bond_length_A"] = round(float(np.linalg.norm(bi - bj)), 3)
    # angle_i is at endpoint i (outer_i - bond_i - bond_j); angle_j is at endpoint j.
    out["angle_i_deg"] = round(_angle_deg(oi, bi, bj), 2)
    out["angle_j_deg"] = round(_angle_deg(bi, bj, oj), 2)
    out["dihedral_deg"] = round(_dihedral_deg(oi, bi, bj, oj), 2)

    # Deviations, with the symmetric fold applied where both handednesses are real.
    out["bond_length_dev_A"] = round(abs(out["bond_length_A"] - spec.bond.length_A), 3)
    out["angle_i_dev_deg"] = round(abs(out["angle_i_deg"] - spec.bond.angle_i_deg), 2)
    out["angle_j_dev_deg"] = round(abs(out["angle_j_deg"] - spec.bond.angle_j_deg), 2)
    d_obs, d_ref = out["dihedral_deg"], spec.bond.dihedral_deg
    if spec.bond.dihedral_symmetric:
        dev = min(_circ_dev(d_obs, d_ref), _circ_dev(d_obs, -d_ref))
    else:
        dev = _circ_dev(d_obs, d_ref)
    out["dihedral_dev_deg"] = round(dev, 2)
    return out


def _circ_dev(obs: float, ref: float) -> float:
    """Smallest absolute angular difference in degrees, across the +/-180 seam."""
    d = (obs - ref + 180.0) % 360.0 - 180.0
    return abs(d)


def ca_chirality(residues: list[tuple[str, dict[str, np.ndarray]]]) -> dict:
    """L/D assignment per residue from the N-CA-C-CB improper torsion.

    Sign convention **calibrated on this project's own control structures**, not assumed:
    420 residues from four LNR crystal chains -- all L by construction -- give this
    improper at +122.9 deg (median +122.7, range +113.2 to +132.5, 100% positive). So
    with the atom order (N, CA, C, CB), L is POSITIVE and D is negative. Getting this
    backwards is silent and total: the first version of this function labelled every
    residue of a crystal structure a D-amino acid.

    GLY has no CB and is achiral, so it is excluded rather than counted as either.
    """
    per_res, n_d = [], 0
    for idx, (name, atoms) in enumerate(residues):
        if name == "GLY" or not all(a in atoms for a in ("N", "CA", "C", "CB")):
            per_res.append({"index": idx, "resname": name, "improper_deg": None, "config": "n/a"})
            continue
        t = _dihedral_deg(atoms["N"], atoms["CA"], atoms["C"], atoms["CB"])
        cfg = "L" if t > 0 else "D"
        if cfg == "D":
            n_d += 1
        per_res.append(
            {"index": idx, "resname": name, "improper_deg": round(t, 2), "config": cfg}
        )
    return {
        "n_residues_assigned": sum(1 for r in per_res if r["config"] in ("L", "D")),
        "n_d_amino_acids": n_d,
        "all_L": n_d == 0,
        "per_residue": per_res,
    }


def backbone_geometry(
    residues: list[tuple[str, dict[str, np.ndarray]]], *, cyclic_mainchain: bool
) -> dict:
    """In-chain backbone bond lengths and omega torsions, with worst deviations.

    `cyclic_mainchain` adds the ring's own closing peptide bond to the in-chain set, so a
    head-to-tail macrocycle is measured as the N-residue ring it is rather than as a
    linear chain with one unexamined junction.
    """
    n = len(residues)
    nca, cac, cn, omegas = [], [], [], []
    for i, (_, atoms) in enumerate(residues):
        if all(a in atoms for a in ("N", "CA")):
            nca.append(float(np.linalg.norm(atoms["N"] - atoms["CA"])))
        if all(a in atoms for a in ("CA", "C")):
            cac.append(float(np.linalg.norm(atoms["CA"] - atoms["C"])))

    pairs = [(i, i + 1) for i in range(n - 1)]
    if cyclic_mainchain and n >= 2:
        pairs.append((n - 1, 0))
    for i, j in pairs:
        ai, aj = residues[i][1], residues[j][1]
        if "C" in ai and "N" in aj:
            cn.append(float(np.linalg.norm(ai["C"] - aj["N"])))
        if all(k in ai for k in ("CA", "C")) and all(k in aj for k in ("N", "CA")):
            omegas.append(_dihedral_deg(ai["CA"], ai["C"], aj["N"], aj["CA"]))

    def _stats(vals, ref):
        if not vals:
            return {"n": 0, "mean": None, "max_dev": None}
        arr = np.asarray(vals, dtype=float)
        return {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 3),
            "max_dev": round(float(np.abs(arr - ref).max()), 3),
        }

    om_dev = [_circ_dev(o, 180.0) for o in omegas if not math.isnan(o)]
    return {
        "N_CA": _stats(nca, BACKBONE_REFERENCE["N_CA"]),
        "CA_C": _stats(cac, BACKBONE_REFERENCE["CA_C"]),
        "C_N": _stats(cn, BACKBONE_REFERENCE["C_N"]),
        "n_peptide_bonds": len(cn),
        "omega_max_dev_from_trans_deg": round(max(om_dev), 2) if om_dev else None,
        # cis peptide bonds are rare but real (proline); counted, not flagged as invalid.
        "n_cis_peptide_bonds": sum(1 for o in omegas if not math.isnan(o) and abs(o) < 90.0),
        "includes_ring_closure_bond": bool(cyclic_mainchain and len(cn) == len(pairs)),
    }


def clashes(
    binder: list[tuple[str, dict[str, np.ndarray]]],
    receptor: list[tuple[str, dict[str, np.ndarray]]] | None,
    *,
    cutoff_A: float = CLASH_CUTOFF_A,
    cyclic_mainchain: bool = False,
) -> dict:
    """Severe heavy-atom overlaps, intra-peptide and peptide-receptor.

    Intra-peptide pairs within the same residue or in sequence-adjacent residues are
    excluded: those are bonded or 1-3 related, and their short distances are chemistry,
    not clash. For a head-to-tail ring the first/last residue pair is adjacent too, so it
    is excluded as well -- otherwise every correctly closed macrocycle would report its
    own closing bond as a clash.
    """
    n = len(binder)
    adjacency: set[tuple[int, int]] = {(i, i) for i in range(n)}
    for i in range(n - 1):
        adjacency.add((i, i + 1))
    if cyclic_mainchain and n >= 2:
        adjacency.add((0, n - 1))

    intra = []
    for i in range(n):
        for j in range(i, n):
            if (i, j) in adjacency or (j, i) in adjacency:
                continue
            for ai, xi in binder[i][1].items():
                for aj, xj in binder[j][1].items():
                    d = float(np.linalg.norm(xi - xj))
                    if d < cutoff_A:
                        intra.append({"res_i": i, "atom_i": ai, "res_j": j, "atom_j": aj,
                                      "dist_A": round(d, 3)})

    inter = []
    if receptor:
        rec_xyz, rec_meta = [], []
        for ridx, (_, atoms) in enumerate(receptor):
            for a, x in atoms.items():
                rec_xyz.append(x)
                rec_meta.append((ridx, a))
        rec_arr = np.asarray(rec_xyz)
        for i in range(n):
            for ai, xi in binder[i][1].items():
                d = np.linalg.norm(rec_arr - xi, axis=1)
                for k in np.where(d < cutoff_A)[0]:
                    inter.append({
                        "binder_res": i, "binder_atom": ai,
                        "receptor_res": rec_meta[k][0], "receptor_atom": rec_meta[k][1],
                        "dist_A": round(float(d[k]), 3),
                    })

    return {
        "cutoff_A": cutoff_A,
        "n_intra_clashes": len(intra),
        "n_inter_clashes": len(inter),
        "min_intra_dist_A": round(min((c["dist_A"] for c in intra), default=float("nan")), 3)
        if intra else None,
        "min_inter_dist_A": round(min((c["dist_A"] for c in inter), default=float("nan")), 3)
        if inter else None,
        # Capped: a badly clashing pose can produce thousands, and the counts above are
        # the signal. The examples are for diagnosis.
        "intra_examples": intra[:20],
        "inter_examples": inter[:20],
    }


def required_atoms_present(
    residues: list[tuple[str, dict[str, np.ndarray]]]
) -> dict:
    """Backbone completeness, and which residues are missing which backbone atoms."""
    missing = []
    for idx, (name, atoms) in enumerate(residues):
        absent = [a for a in BACKBONE if a not in atoms]
        if name != "GLY" and "CB" not in atoms:
            absent.append("CB")
        if absent:
            missing.append({"index": idx, "resname": name, "missing": absent})
    return {
        "n_residues": len(residues),
        "n_residues_missing_atoms": len(missing),
        "backbone_complete": not missing,
        "missing_detail": missing[:20],
    }


def sequence_of(residues: list[tuple[str, dict[str, np.ndarray]]]) -> str:
    from proteinfoundation.cpseq.candidates import THREE_TO_ONE

    return "".join(THREE_TO_ONE.get(name, "X") for name, _ in residues)


def validity_report(
    pdb_path: str | Path,
    *,
    binder_chain: str,
    receptor_chains: list[str],
    chemistry: str,
    requested_sequence: str,
) -> dict:
    """Everything in this module, for one structure, in one record."""
    binder = read_chain(pdb_path, binder_chain)
    receptor: list[tuple[str, dict[str, np.ndarray]]] = []
    for ch in receptor_chains:
        receptor.extend(read_chain(pdb_path, ch))

    cyclic_mainchain = chemistry == "mainchain"
    got_seq = sequence_of(binder)
    return {
        "pdb_path": str(pdb_path),
        "binder_chain": binder_chain,
        "receptor_chains": list(receptor_chains),
        "sequence_requested": requested_sequence,
        "sequence_observed": got_seq,
        "sequence_matches": got_seq == requested_sequence,
        "atoms": required_atoms_present(binder),
        "closure": closure_geometry(pdb_path, binder_chain, chemistry),
        "chirality": ca_chirality(binder),
        "backbone": backbone_geometry(binder, cyclic_mainchain=cyclic_mainchain),
        "clashes": clashes(binder, receptor, cyclic_mainchain=cyclic_mainchain),
    }
