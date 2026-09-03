"""Stage-wise "soft closure" projection: pull a bound LINEAR peptide's termini together
under a real force field BEFORE any ring bond exists, so SDEdit gets an input whose gap it
can actually span.

Why
---
The LNR SDEdit sweep closed mainchain rings as a step function of the INPUT head-to-tail
gap: 60% of peptides in the smallest gap bin closed, 4% in the largest, and the worst cases
(LNR_1jrr_A_P at 45.6 A, LNR_4x3h_A_B at 26.4 A, LNR_3cvl_A_B at 15.4 A) never closed once
in 20 grid points. Closure there is all-or-nothing -- there are no near-misses to nudge --
so no amount of extra noise or guidance in latent space helps. The gap is a property of the
INPUT, and the fix has to be applied to the input.

This script does that fix as a physical projection, not a generative one:

  1. Build the complex in amber14 + OBC2 implicit solvent (PDBFixer prepares it), with the
     receptor's atoms FROZEN (zero mass) so the pocket cannot deform to meet the peptide.
  2. Add a harmonic `pull` restraint between the two atoms the chosen chemistry would
     actually bond (see ATOMS_FOR_TYPE), and walk its equilibrium length r0 down a ladder
     (20 -> 15 -> 12 -> 9 -> 6 -> 3 A), minimizing at every rung. A single jump to bond
     length yanks the peptide through the receptor; a ladder lets the force field relax the
     backbone between pulls.
  3. Hold the binding pose with FLAT-BOTTOM restraints on one representative heavy-atom
     contact per interface residue -- free movement inside +-`tol`, quadratic outside. One
     pair per residue is the point: restraining every interacting atom makes the ring
     geometrically unreachable, which is the failure mode this whole staging exists to avoid.

No bond is formed here. The output is a still-linear peptide in a pre-closed pose; the flow
model does the chemistry afterwards (see scripts/sdedit_cyclize.py), re-encoding from the
projected PDB so the latent it edits matches the geometry it was given.

Minimization alone can stall behind a torsional barrier, so each input is projected
`--n-projections` times: replica 0 from the crystal pose, the rest from small random
backbone-torsion perturbations. Replicas are filtered on closure, contact retention, clash
and chain geometry -- which makes the accepted count a FEASIBILITY ORACLE. Zero accepted
replicas is the informative answer that this pose cannot be minimally cyclized at these
termini.

Outputs (all under --out-dir):
    pdbs/<example_id>__proj<k>.pdb    projected complex, staged-PDB convention
    projections.jsonl                one row per replica, accepted or not
    stages/<example_id>__proj<k>.npz  per-rung peptide coordinates, for the animation
    metadata/projected.parquet       accepted replicas, ready for scripts/sdedit_cyclize.py

CPU only -- no GPU, no torch, no model. Runs in the isolated .venv_openmm (OpenMM is not
installable alongside the training environment's pins).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from openmm import CustomBondForce, LocalEnergyMinimizer, Platform, unit
import openmm
import openmm.app as app
from pdbfixer import PDBFixer

# ---------------------------------------------------------------------------------------
# Chemistry: which two atoms a closure of each type would actually bond, and at what length.
# Endpoints follow the convention the flow model is conditioned on and that
# scripts/build_lnr_metadata.py measures: residue 0 and residue L-1 of the binder.
# ---------------------------------------------------------------------------------------
ATOMS_FOR_TYPE = {
    # type       -> (atom on FIRST residue, atom on LAST residue, ideal bond length A,
    #                required first resname(s), required last resname(s))
    "mainchain":  ("N",  "C",  1.33, None,            None),
    "disulfide":  ("SG", "SG", 2.05, ("CYS",),        ("CYS",)),
    "isopeptide": ("NZ", "CG", 1.33, ("LYS",),        ("ASP", "ASN")),
}
# Residues to mutate into when the terminal identity does not already support the chemistry.
MUTATE_TO = {"disulfide": ("CYS", "CYS"), "isopeptide": ("LYS", "ASP")}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# Contact definition used by scripts/sdedit_cyclize.py for `contact_retention`: peptide CA
# to receptor CA within 10 A. Retention here uses the SAME definition so the projection's
# number and the edit's number are directly comparable rather than merely similar.
CA_CONTACT_CUTOFF_A = 10.0
# Heavy-atom cutoff for CHOOSING which pairs to restrain -- a much tighter, physical contact.
HEAVY_CONTACT_CUTOFF_A = 4.5

KJ_PER_KCAL = 4.184


def _chain_atoms(chain):
    return [a for a in chain.atoms()]


def _atom_index(residue, name):
    for a in residue.atoms():
        if a.name == name:
            return a.index
    return None


def _is_heavy(atom):
    return atom.element is not None and atom.element.symbol != "H"


def _pdb_atom_name(name: str) -> str:
    """PDB columns 13-16. Names up to 3 characters start in column 14; 4-character names
    start in column 13. Left-justifying everything shifts the element and breaks parsers."""
    return f"{name:<4.4s}" if len(name) >= 4 else f" {name:<3.3s}"


# ---------------------------------------------------------------------------------------
# Input parsing: we keep the ORIGINAL staged PDB text around, because the receptor is frozen
# and its lines can be copied through byte-for-byte (B-factors and all), and because the
# peptide's original atom SET is what we write back -- atoms PDBFixer adds are needed for the
# energy but must not silently enrich the structure handed to the model.
# ---------------------------------------------------------------------------------------
def read_staged_pdb(path: Path):
    """Returns (header_lines, {chain: OrderedDict[resseq -> {atom_name: line}]}, order)."""
    header, chains, order = [], {}, []
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("REMARK"):
            header.append(line)
            continue
        if not line.startswith("ATOM  "):
            continue
        ch = line[21]
        resseq = int(line[22:26])
        name = line[12:16].strip()
        if ch not in chains:
            chains[ch] = OrderedDict()
            order.append(ch)
        chains[ch].setdefault(resseq, OrderedDict())[name] = line
    return header, chains, order


def coords_of(chain_dict):
    """{resseq: {atom: (x,y,z)}} in Angstrom."""
    return {
        rs: {n: (float(l[30:38]), float(l[38:46]), float(l[46:54])) for n, l in atoms.items()}
        for rs, atoms in chain_dict.items()
    }


# ---------------------------------------------------------------------------------------
# Backbone torsion perturbation -- the escape hatch from a minimizer that is stuck behind a
# barrier. Rotating about phi/psi moves every downstream atom rigidly, which is a large,
# physically sensible move; random Cartesian jitter would just be minimized straight back.
# ---------------------------------------------------------------------------------------
def perturb_backbone_torsions(xyz, pep_atom_indices_by_res, sigma_deg, rng):
    """Rotates about each residue's phi (N-CA) and psi (CA-C) axis by N(0, sigma_deg),
    carrying all atoms C-terminal to the axis. `xyz` is modified in place (nm)."""
    n_res = len(pep_atom_indices_by_res)
    for ri in range(n_res):
        res = pep_atom_indices_by_res[ri]
        for axis_a, axis_b, downstream_names in (
            ("N", "CA", None),   # phi: everything after CA in this residue and beyond
            ("CA", "C", None),   # psi: everything after C in this residue and beyond
        ):
            ia, ib = res.get(axis_a), res.get(axis_b)
            if ia is None or ib is None:
                continue
            ang = math.radians(rng.gauss(0.0, sigma_deg))
            if abs(ang) < 1e-9:
                continue
            p0, p1 = xyz[ia], xyz[ib]
            axis = p1 - p0
            nrm = np.linalg.norm(axis)
            if nrm < 1e-9:
                continue
            axis = axis / nrm
            # Atoms carried: in THIS residue, those after the axis in backbone order; in every
            # later residue, all atoms. Sidechain atoms of this residue follow CA, so they move
            # with psi but not with phi -- the standard convention.
            moving = []
            if axis_a == "N":       # phi rotates the sidechain + C/O + all later residues
                moving += [i for n, i in res.items() if n not in ("N", "H", "CA", "HA")]
            else:                    # psi rotates only O + all later residues
                moving += [i for n, i in res.items() if n in ("O", "OXT")]
            for rj in range(ri + 1, n_res):
                moving += list(pep_atom_indices_by_res[rj].values())
            if not moving:
                continue
            idx = np.asarray(sorted(set(moving)), dtype=int)
            v = xyz[idx] - p0
            c, s = math.cos(ang), math.sin(ang)
            xyz[idx] = (
                v * c + np.cross(axis, v) * s + axis[None, :] * (v @ axis)[:, None] * (1.0 - c) + p0
            )


# ---------------------------------------------------------------------------------------
# Quality gates applied to every replica.
# ---------------------------------------------------------------------------------------
def geometry_report(pep_xyz_A, pep_heavy_idx, pep_ca_idx, rec_heavy_A, rec_ca_A,
                    pep_ca_ref_A, ref_contacts):
    """Everything the accept/reject decision needs, in Angstrom throughout."""
    out = {}
    ca = pep_xyz_A[pep_ca_idx]
    steps = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    out["ca_step_min_A"] = float(steps.min())
    out["ca_step_max_A"] = float(steps.max())

    heavy = pep_xyz_A[pep_heavy_idx]
    out["min_pep_rec_heavy_A"] = (
        float(np.linalg.norm(heavy[:, None, :] - rec_heavy_A[None, :, :], axis=-1).min())
        if len(rec_heavy_A) else float("inf")
    )

    d = np.linalg.norm(ca[:, None, :] - rec_ca_A[None, :, :], axis=-1)
    now = {(int(i), int(j)) for i, j in zip(*np.where(d < CA_CONTACT_CUTOFF_A))}
    out["n_contacts_input"] = len(ref_contacts)
    out["n_contacts_projected"] = len(now)
    out["contact_retention"] = (
        float(len(ref_contacts & now) / len(ref_contacts)) if ref_contacts else float("nan")
    )
    out["ca_rmsd_to_input_A"] = float(np.sqrt(((ca - pep_ca_ref_A) ** 2).sum(-1).mean()))
    out["ca_max_dev_to_input_A"] = float(np.linalg.norm(ca - pep_ca_ref_A, axis=-1).max())
    return out


def terminal_cb_A(pep_xyz_A, res_atoms):
    """CB-CB of the two terminal residues; CA substitutes for glycine, which has none."""
    def pick(r):
        return pep_xyz_A[r["CB"]] if "CB" in r else pep_xyz_A[r["CA"]]
    return float(np.linalg.norm(pick(res_atoms[0]) - pick(res_atoms[-1])))


# ---------------------------------------------------------------------------------------
# System preparation.
# ---------------------------------------------------------------------------------------
def prepare_complex(pdb_path: Path, mutations: list[str] | None, pep_chain: str):
    """PDBFixer-prepared complex. `mutations` are PDBFixer specs ("ALA-3-CYS") on `pep_chain`.

    `missingResidues` is cleared on purpose: a staged LNR PDB carries no SEQRES, so anything
    PDBFixer "finds" there would be invented loop, and a rebuilt receptor loop would silently
    change the pocket this projection is supposed to hold fixed.
    """
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    if mutations:
        fixer.applyMutations(mutations, pep_chain)
        # applyMutations leaves the new sidechain's atoms missing; they are built below.
        fixer.findMissingResidues()
        fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    n_added = sum(len(v) for v in fixer.missingAtoms.values()) + len(fixer.missingTerminals)
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return fixer, n_added


def index_maps(topology):
    """(receptor_chain, peptide_chain, per-residue {atom_name: index} for the peptide)."""
    chains = list(topology.chains())
    if len(chains) != 2:
        raise RuntimeError(f"expected exactly 2 chains (receptor, peptide), got {len(chains)}")
    rec, pep = chains[0], chains[1]
    pep_res_atoms = [{a.name: a.index for a in r.atoms()} for r in pep.residues()]
    return rec, pep, pep_res_atoms


def pick_contact_pairs(xyz_A, pep_res_heavy, rec_heavy_idx, max_per_residue):
    """One (or two) representative peptide-receptor heavy-atom pairs per interface residue.

    Deliberately sparse. Restraining every interacting atom pins the peptide as a rigid body
    and the ring becomes unreachable -- the closure then fails for a reason that has nothing
    to do with whether the pose could tolerate cyclization.
    """
    rec_xyz = xyz_A[rec_heavy_idx]
    pairs = []
    for res_heavy in pep_res_heavy:
        if not res_heavy:
            continue
        pep_xyz = xyz_A[res_heavy]
        d = np.linalg.norm(pep_xyz[:, None, :] - rec_xyz[None, :, :], axis=-1)
        if d.min() > HEAVY_CONTACT_CUTOFF_A:
            continue
        flat = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
        used_pep = set()
        for pi, ri in flat:
            if len(used_pep) >= max_per_residue:
                break
            if pi in used_pep:
                continue
            if d[pi, ri] > HEAVY_CONTACT_CUTOFF_A:
                break
            used_pep.add(int(pi))
            pairs.append((int(res_heavy[pi]), int(rec_heavy_idx[ri]), float(d[pi, ri])))
    return pairs


def build_forces(system, xyz_A, pep_idx_a, pep_idx_b, contact_pairs,
                 k_pull_kj_nm2, k_contact_kj_nm2, tol_A):
    """Adds the pull restraint and the flat-bottom contact restraints. Returns the pull force."""
    pull = CustomBondForce("0.5*k_pull*(r-r0)^2")
    pull.addGlobalParameter("k_pull", float(k_pull_kj_nm2))
    pull.addGlobalParameter("r0", float(np.linalg.norm(xyz_A[pep_idx_a] - xyz_A[pep_idx_b]) * 0.1))
    pull.addBond(pep_idx_a, pep_idx_b)
    system.addForce(pull)

    if contact_pairs:
        # Two-sided flat bottom: penalise the contact drifting apart AND collapsing, so the
        # interface is held in place rather than merely prevented from unbinding.
        contacts = CustomBondForce("0.5*k_contact*max(0, abs(r-r_ref)-tol)^2")
        contacts.addGlobalParameter("k_contact", float(k_contact_kj_nm2))
        contacts.addGlobalParameter("tol", float(tol_A) * 0.1)
        contacts.addPerBondParameter("r_ref")
        for pa, ra, d_A in contact_pairs:
            contacts.addBond(pa, ra, [float(d_A) * 0.1])
        system.addForce(contacts)
    return pull


# ---------------------------------------------------------------------------------------
# One example is prepared ONCE. PDBFixer's hydrogen build and OpenMM's createSystem depend
# only on the topology, and the contact restraints are chosen off the crystal pose, so all
# of that is identical for every replica -- doing it per replica cost ~63 s of the ~65 s a
# replica takes. Replicas differ only in their STARTING COORDINATES, which is a
# `context.setPositions` away.
# ---------------------------------------------------------------------------------------
class ExampleContext:
    """Force field, frozen receptor, restraints and index vectors for one input peptide."""

    def __init__(self, fixer, cyc_type, args, ref, forcefield, platform):
        self.topology = fixer.topology
        self.base_xyz_nm = np.asarray(fixer.positions.value_in_unit(unit.nanometer), dtype=np.float64)
        self.ref = ref
        rec, pep, self.pep_res_atoms = index_maps(self.topology)

        first_res, last_res = list(pep.residues())[0], list(pep.residues())[-1]
        a_name, b_name, self.bond_len_A, _, _ = ATOMS_FOR_TYPE[cyc_type]
        self.ia, self.ib = _atom_index(first_res, a_name), _atom_index(last_res, b_name)
        if self.ia is None or self.ib is None:
            raise RuntimeError(
                f"{cyc_type} needs {a_name} on {first_res.name}{first_res.id} and {b_name} on "
                f"{last_res.name}{last_res.id}; one is absent even after atom rebuilding"
            )
        self.pull_atoms = (f"{first_res.name}{first_res.id}:{a_name}-"
                           f"{last_res.name}{last_res.id}:{b_name}")

        rec_atoms = _chain_atoms(rec)
        self.rec_heavy_idx = np.asarray([a.index for a in rec_atoms if _is_heavy(a)], dtype=int)
        self.pep_res_heavy = [[a.index for a in r.atoms() if _is_heavy(a)] for r in pep.residues()]
        self.pep_heavy_idx = np.asarray([i for r in self.pep_res_heavy for i in r], dtype=int)
        self.pep_ca_idx = np.asarray([ra["CA"] for ra in self.pep_res_atoms], dtype=int)

        system = forcefield.createSystem(
            self.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            # No constraints: the receptor is frozen with zero masses, and OpenMM refuses a
            # constraint that touches a massless particle. Minimization needs none anyway.
            constraints=None,
        )
        for a in rec_atoms:
            system.setParticleMass(a.index, 0.0)

        # Chosen and measured on the CRYSTAL pose, never a perturbed one: r_ref is the
        # original LP contact distance, so a replica's perturbation is something the
        # restraints pull back OUT of, not a new reference they lock in.
        self.contact_pairs = pick_contact_pairs(
            ref["xyz_A_ref"], self.pep_res_heavy, self.rec_heavy_idx, args.contacts_per_residue)
        build_forces(system, ref["xyz_A_ref"], self.ia, self.ib, self.contact_pairs,
                     args.k_pull, args.k_contact, args.contact_tol_A)

        self.integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)  # never stepped
        self.context = openmm.Context(system, self.integrator, platform)


def project_once(ec: ExampleContext, cyc_type, ladder_A, args, replica):
    """Runs the staged pull for one replica. Returns (row, peptide-heavy coords per rung)."""
    xyz_nm = ec.base_xyz_nm.copy()
    if replica > 0 and args.torsion_sigma_deg > 0:
        rng = random.Random(args.seed * 10_000 + replica)
        perturb_backbone_torsions(xyz_nm, ec.pep_res_atoms, args.torsion_sigma_deg, rng)
    xyz_A = xyz_nm * 10.0

    ec.context.setPositions(xyz_nm * unit.nanometer)
    d_start_A = float(np.linalg.norm(xyz_A[ec.ia] - xyz_A[ec.ib]))
    # Only descend the ladder from below the current separation; rungs above it would push
    # the termini APART before pulling them together.
    rungs = [r for r in ladder_A if r < d_start_A] + [ec.bond_len_A + args.stop_bond_margin_A]

    trace, stage_rows = [xyz_A[ec.pep_heavy_idx].copy()], []
    xyz_now = xyz_A
    for rung in rungs:
        ec.context.setParameter("r0", float(rung) * 0.1)
        LocalEnergyMinimizer.minimize(ec.context, args.tolerance, args.max_iterations)
        st = ec.context.getState(getPositions=True, getEnergy=True)
        xyz_now = np.asarray(st.getPositions().value_in_unit(unit.nanometer)) * 10.0
        d = float(np.linalg.norm(xyz_now[ec.ia] - xyz_now[ec.ib]))
        cb = terminal_cb_A(xyz_now, ec.pep_res_atoms)
        trace.append(xyz_now[ec.pep_heavy_idx].copy())
        stage_rows.append({
            "r0_A": float(rung), "pull_dist_A": d, "terminal_cb_A": cb,
            "energy_kj": float(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)),
        })
        if cb < args.stop_cb_A and d < args.stop_pull_A:
            break

    geo = geometry_report(xyz_now, ec.pep_heavy_idx, ec.pep_ca_idx,
                          ec.ref["xyz_A_ref"][ec.rec_heavy_idx], ec.ref["rec_ca_A"],
                          ec.ref["pep_ca_A"], ec.ref["contacts"])
    row = {
        "replica": replica,
        "cyc_type": cyc_type,
        "pull_atoms": ec.pull_atoms,
        "n_contact_restraints": len(ec.contact_pairs),
        "pull_dist_start_A": d_start_A,
        "pull_dist_final_A": stage_rows[-1]["pull_dist_A"] if stage_rows else d_start_A,
        "terminal_cb_start_A": ec.ref["terminal_cb_A"],
        "terminal_cb_final_A": stage_rows[-1]["terminal_cb_A"] if stage_rows else ec.ref["terminal_cb_A"],
        "n_stages_run": len(stage_rows),
        "stages": stage_rows,
        **geo,
    }
    # The oracle. Every clause is a way the projection can be "successful" but useless:
    # a closed ring that has left the pocket, or crushed into it, or broken the chain.
    row["pass_closure"] = bool(row["terminal_cb_final_A"] < args.accept_cb_A)
    row["pass_retention"] = bool(
        row["contact_retention"] >= args.accept_retention or math.isnan(row["contact_retention"])
    )
    row["pass_clash"] = bool(row["min_pep_rec_heavy_A"] >= args.accept_min_heavy_A)
    row["pass_chain"] = bool(
        row["ca_step_min_A"] >= args.accept_ca_step[0] and row["ca_step_max_A"] <= args.accept_ca_step[1]
    )
    row["accepted"] = bool(
        row["pass_closure"] and row["pass_retention"] and row["pass_clash"] and row["pass_chain"]
    )
    return row, np.stack(trace)


# ---------------------------------------------------------------------------------------
# Writing the projected complex back out in the staged-PDB convention the CPSea loader wants
# (chain A receptor, chain B binder, heavy atoms only, standard residues only).
# ---------------------------------------------------------------------------------------
def write_projected_pdb(out_path, header, orig_chains, topology, xyz_A, example_id,
                        keep_added_atoms):
    """Receptor lines are copied byte-for-byte -- its atoms were frozen, so re-emitting them
    from the minimized coordinates could only introduce float round-trip noise."""
    _, pep_chain, _ = index_maps(topology)
    orig_pep = orig_chains.get("B", {})

    lines = list(header) + [
        f"REMARK 999 SOFT-CLOSURE PROJECTION example_id={example_id}\n",
        "REMARK 999 receptor (chain A) FROZEN and copied verbatim; chain B minimized\n",
        "REMARK 999 NO ring bond formed here -- peptide is still linear\n",
    ]
    serial = 1
    for rs, atoms in orig_chains.get("A", {}).items():
        for name, line in atoms.items():
            lines.append(line[:6] + f"{serial:5d}" + line[11:])
            serial += 1
    lines.append(f"TER   {serial:5d}\n")
    serial += 1

    n_written = 0
    for res in pep_chain.residues():
        if res.name not in STANDARD_AA:
            continue
        try:
            resseq = int(res.id)
        except ValueError:
            resseq = res.index + 1
        orig_atoms = orig_pep.get(resseq, {})
        for atom in res.atoms():
            if not _is_heavy(atom):
                continue
            if not keep_added_atoms and orig_atoms and atom.name not in orig_atoms:
                continue
            src = orig_atoms.get(atom.name)
            occ_b = src[54:66] if src else "  1.00  0.00"
            x, y, z = xyz_A[atom.index]
            lines.append(
                "ATOM  " + f"{serial:5d}" + " " + _pdb_atom_name(atom.name) + " "
                + f"{res.name:>3.3s}" + " B" + f"{resseq:4d}" + "    "
                + f"{x:8.3f}{y:8.3f}{z:8.3f}" + f"{occ_b:12.12s}" + " " * 10
                + f"{atom.element.symbol:>2.2s}" + "  \n"
            )
            serial += 1
            n_written += 1
    lines.append(f"TER   {serial:5d}\n")
    lines.append("END\n")
    out_path.write_text("".join(lines))
    return n_written


def _names_for(topology, indices):
    by_idx = {a.index: a.name for a in topology.atoms()}
    return [by_idx[int(i)] for i in indices]


def _resids_for(topology, indices):
    """Peptide-local residue ORDER (0..L-1) per heavy atom, so the renderer can group atoms
    into residues without re-parsing the topology."""
    order = {r.index: k for k, r in enumerate(list(topology.chains())[1].residues())}
    by_idx = {a.index: order[a.residue.index] for a in topology.atoms() if a.residue.index in order}
    return [by_idx[int(i)] for i in indices]


def reference_state(fixer):
    """Input-pose quantities every replica of this peptide is scored against."""
    topology = fixer.topology
    rec, pep, pep_res_atoms = index_maps(topology)
    xyz_A = np.asarray(fixer.positions.value_in_unit(unit.nanometer)) * 10.0
    rec_ca = np.asarray([a.index for a in rec.atoms() if a.name == "CA"], dtype=int)
    pep_ca = np.asarray([ra["CA"] for ra in pep_res_atoms], dtype=int)
    d = np.linalg.norm(xyz_A[pep_ca][:, None, :] - xyz_A[rec_ca][None, :, :], axis=-1)
    contacts = {(int(i), int(j)) for i, j in zip(*np.where(d < CA_CONTACT_CUTOFF_A))}
    return {
        "xyz_A_ref": xyz_A,
        "rec_ca_A": xyz_A[rec_ca],
        "pep_ca_A": xyz_A[pep_ca],
        "contacts": contacts,
        "terminal_cb_A": terminal_cb_A(xyz_A, pep_res_atoms),
        "n_residues": len(pep_res_atoms),
    }


def mutations_for(cyc_type, fixer_topology):
    """PDBFixer mutation specs needed so the terminal residues can carry the linkage."""
    if cyc_type not in MUTATE_TO:
        return []
    _, pep, _ = index_maps(fixer_topology)
    residues = list(pep.residues())
    want_first, want_last = MUTATE_TO[cyc_type]
    _, _, _, ok_first, ok_last = ATOMS_FOR_TYPE[cyc_type]
    specs = []
    if residues[0].name not in ok_first:
        specs.append(f"{residues[0].name}-{residues[0].id}-{want_first}")
    if residues[-1].name not in ok_last:
        specs.append(f"{residues[-1].name}-{residues[-1].id}-{want_last}")
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="LNR staged metadata parquet.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--examples", nargs="+", default=None,
                    help="example_id list. Default: every row of --metadata.")
    ap.add_argument("--cyc-type", default="mainchain", choices=sorted(ATOMS_FOR_TYPE))
    ap.add_argument("--n-projections", type=int, default=16,
                    help="Replicas per input. Replica 0 is the unperturbed crystal pose.")
    ap.add_argument("--torsion-sigma-deg", type=float, default=12.0,
                    help="Per-torsion backbone perturbation for replicas > 0. 0 disables.")
    ap.add_argument("--seed", type=int, default=0)
    # --- restraints ---
    ap.add_argument("--k-pull", type=float, default=2000.0, help="kJ/mol/nm^2")
    ap.add_argument("--k-contact", type=float, default=500.0, help="kJ/mol/nm^2")
    ap.add_argument("--contact-tol-A", type=float, default=0.75)
    ap.add_argument("--contacts-per-residue", type=int, default=1, choices=(1, 2))
    ap.add_argument("--ladder-A", nargs="+", type=float, default=[20, 15, 12, 9, 6, 3])
    # --- minimizer ---
    ap.add_argument("--tolerance", type=float, default=10.0, help="kJ/mol/nm")
    ap.add_argument("--max-iterations", type=int, default=1000)
    # --- early stop ---
    ap.add_argument("--stop-cb-A", type=float, default=9.0,
                    help="Stop descending once terminal CB is inside this AND the pulled pair "
                         "is inside --stop-pull-A. The generator finishes the closure.")
    ap.add_argument("--stop-pull-A", type=float, default=6.0)
    ap.add_argument("--stop-bond-margin-A", type=float, default=1.0,
                    help="Final rung sits this far above the ideal bond length -- the last "
                         "angstrom is the model's job, not the minimizer's.")
    # --- acceptance ---
    ap.add_argument("--accept-cb-A", type=float, default=10.0)
    ap.add_argument("--accept-retention", type=float, default=0.8)
    ap.add_argument("--accept-min-heavy-A", type=float, default=2.2)
    ap.add_argument("--accept-ca-step", nargs=2, type=float, default=[3.4, 4.2])
    ap.add_argument("--keep-per-example", type=int, default=4,
                    help="Accepted replicas carried into the metadata parquet, best first.")
    ap.add_argument("--keep-added-atoms", action="store_true",
                    help="Also emit heavy atoms PDBFixer built that the input lacked. Off by "
                         "default so the projection is the only difference from the input.")
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve inputs, write the control metadata and print the work plan, "
                         "then stop before building a force field. Catches selection and "
                         "plumbing errors without paying for a minimization.")
    args = ap.parse_args()

    import pandas as pd

    meta = pd.read_parquet(args.metadata)
    if args.examples:
        missing = sorted(set(args.examples) - set(meta["example_id"]))
        if missing:
            raise SystemExit(f"FATAL: example_id(s) not in {args.metadata}: {missing}")
        meta = meta[meta["example_id"].isin(args.examples)].copy()
    if meta.empty:
        raise SystemExit("FATAL: no input rows selected.")

    out_dir = Path(args.out_dir)
    pdb_dir, stage_dir, meta_dir = out_dir / "pdbs", out_dir / "stages", out_dir / "metadata"
    for d in (pdb_dir, stage_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "projections.jsonl"

    # The matched CONTROL: the same peptides, unprojected, so the downstream SDEdit arm can
    # be run on both at the SAME grid and seeds. Without it "closure improved" is confounded
    # with the original sweep having used a different grid and a single seed. Written before
    # any minimization, so the control still exists if every projection is rejected.
    control_path = meta_dir / "control.parquet"
    meta.to_parquet(control_path, index=False)
    print(f"control metadata (unprojected, same peptides): {control_path}", flush=True)

    print(f"{len(meta)} input peptide(s) x {args.n_projections} replica(s) "
          f"= {len(meta) * args.n_projections} projections, type={args.cyc_type}", flush=True)
    for _, r in meta.iterrows():
        print(f"  {r['example_id']:20s} L={int(r['peptide_length']):2d} "
              f"nc_gap={float(r['nc_gap_angstrom']):6.2f} A", flush=True)
    if args.dry_run:
        print("DRY RUN OK", flush=True)
        return

    forcefield = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    platform = Platform.getPlatformByName(args.platform)

    done = set()
    if rows_path.exists():
        for lineno, line in enumerate(rows_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["run_key"])
            except (json.JSONDecodeError, KeyError):
                raise SystemExit(f"FATAL: corrupt row {rows_path}:{lineno}. Delete and rerun.")
    if done:
        print(f"resuming: {len(done)} projections already done", flush=True)

    t_wall = time.time()
    with rows_path.open("a") as fh:
        for _, mrow in meta.iterrows():
            example_id = str(mrow["example_id"])
            src = Path(str(mrow["path"]))
            if not src.is_file():
                raise SystemExit(f"FATAL: staged PDB missing for {example_id}: {src}")
            header, orig_chains, _ = read_staged_pdb(src)

            # Prepared ONCE per example. Every replica shares this topology, this frozen
            # receptor and these contact restraints; only the starting coordinates differ.
            t_prep = time.time()
            probe, n_added = prepare_complex(src, None, "B")
            muts = mutations_for(args.cyc_type, probe.topology)
            if muts:
                print(f"  {example_id}: mutating termini for {args.cyc_type}: {muts}", flush=True)
                fixer, n_added = prepare_complex(src, muts, "B")
            else:
                fixer = probe
            ref = reference_state(fixer)
            try:
                ec = ExampleContext(fixer, args.cyc_type, args, ref, forcefield, platform)
            except Exception as exc:  # noqa: BLE001
                for k in range(args.n_projections):
                    row = {"run_key": f"{example_id}|{args.cyc_type}|{k}", "example_id": example_id,
                           "replica": k, "cyc_type": args.cyc_type, "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}"}
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"  FAIL {example_id} (setup): {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"  {example_id}: prepared in {time.time() - t_prep:.0f}s "
                  f"({ec.topology.getNumAtoms()} atoms, {len(ec.contact_pairs)} contact restraints, "
                  f"{len(ec.rec_heavy_idx)} receptor heavy atoms frozen)", flush=True)

            for k in range(args.n_projections):
                run_key = f"{example_id}|{args.cyc_type}|{k}"
                if run_key in done:
                    continue
                t0 = time.time()
                try:
                    row, trace = project_once(ec, args.cyc_type, args.ladder_A, args, k)
                except Exception as exc:  # noqa: BLE001
                    row = {"run_key": run_key, "example_id": example_id, "replica": k,
                           "cyc_type": args.cyc_type, "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}"}
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"  FAIL {run_key}: {row['error']}", flush=True)
                    continue

                tag = f"{example_id}__proj{k:02d}"
                # `trace` holds peptide heavy atoms only; the PDB writer indexes the full
                # system, so scatter the last frame back through the same index vector.
                xyz_final = np.zeros((ec.topology.getNumAtoms(), 3))
                xyz_final[ec.pep_heavy_idx] = trace[-1]
                n_pep_atoms = write_projected_pdb(
                    pdb_dir / f"{tag}.pdb", header, orig_chains, ec.topology, xyz_final,
                    tag, args.keep_added_atoms)
                np.savez_compressed(
                    stage_dir / f"{tag}.npz",
                    pep_heavy_xyz=trace.astype(np.float32),
                    pep_heavy_idx=ec.pep_heavy_idx.astype(np.int32),
                    pep_heavy_names=np.asarray(_names_for(ec.topology, ec.pep_heavy_idx)),
                    pep_heavy_resid=np.asarray(_resids_for(ec.topology, ec.pep_heavy_idx),
                                               dtype=np.int32),
                    pep_resnames=np.asarray(
                        [r.name for r in list(ec.topology.chains())[1].residues()]),
                    pep_ca_ref=ref["pep_ca_A"].astype(np.float32),
                    # Full crystal-pose heavy atoms, on the SAME index vector as the trace, so
                    # the "before" panel is the real input even for a perturbed replica (whose
                    # trace[0] is the perturbed pose, not the crystal one).
                    pep_heavy_ref=ref["xyz_A_ref"][ec.pep_heavy_idx].astype(np.float32),
                    rec_ca=ref["rec_ca_A"].astype(np.float32),
                    meta_json=json.dumps({k2: v for k2, v in row.items() if k2 != "stages"}),
                    stages_json=json.dumps(row["stages"]),
                )
                row.update(run_key=run_key, example_id=example_id, status="ok", tag=tag,
                           projected_pdb=str((pdb_dir / f"{tag}.pdb").resolve()),
                           n_peptide_atoms=n_pep_atoms, n_atoms_added_by_fixer=n_added,
                           mutations=";".join(muts), seconds=round(time.time() - t0, 1),
                           input_nc_gap_A=float(mrow["nc_gap_angstrom"]),
                           peptide_length=int(mrow["peptide_length"]),
                           binder_chain_id="B", cluster_id=str(mrow.get("cluster_id", example_id)))
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"  {tag}: pull {row['pull_dist_start_A']:6.2f} -> "
                      f"{row['pull_dist_final_A']:5.2f} A  CB {row['terminal_cb_final_A']:5.2f} A  "
                      f"ret {row['contact_retention']:.2f}  clash {row['min_pep_rec_heavy_A']:.2f} A  "
                      f"{'ACCEPT' if row['accepted'] else 'reject'}  ({row['seconds']:.0f}s)",
                      flush=True)

    write_outputs(out_dir, rows_path, meta, args)
    print(f"done in {time.time() - t_wall:.0f}s", flush=True)


def write_outputs(out_dir: Path, rows_path: Path, meta, args) -> None:
    """Feasibility summary + the metadata parquet scripts/sdedit_cyclize.py will consume."""
    import pandas as pd

    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise SystemExit(f"FATAL: every projection failed -- see {rows_path}")

    summary = {"n_inputs": int(meta.shape[0]), "n_projections": len(rows),
               "n_failed": len(rows) - len(ok), "cyc_type": args.cyc_type, "per_example": {}}
    keep = []
    for example_id, grp in pd.DataFrame(ok).groupby("example_id"):
        acc = grp[grp["accepted"]].sort_values(
            ["terminal_cb_final_A", "contact_retention"], ascending=[True, False])
        summary["per_example"][example_id] = {
            "n_replicas": int(len(grp)),
            "n_accepted": int(len(acc)),
            "input_nc_gap_A": float(grp["input_nc_gap_A"].iloc[0]),
            "best_terminal_cb_A": float(grp["terminal_cb_final_A"].min()),
            "best_pull_dist_A": float(grp["pull_dist_final_A"].min()),
            "best_contact_retention": float(grp["contact_retention"].max()),
            # The oracle's verdict. "infeasible" is a RESULT, not an error: it says this pose
            # cannot be minimally cyclized at these termini under these tolerances.
            "verdict": "feasible" if len(acc) else "infeasible",
            "fail_counts": {c: int((~grp[c]).sum())
                            for c in ("pass_closure", "pass_retention", "pass_clash", "pass_chain")},
        }
        keep.append(acc.head(args.keep_per_example))

    kept = pd.concat(keep) if keep else pd.DataFrame()
    (out_dir / "projection_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2), flush=True)

    if kept.empty:
        print("\nNo replica passed every gate -- writing NO metadata parquet. The SDEdit arm "
              "downstream must not run on rejected projections.", flush=True)
        return

    out = pd.DataFrame({
        # Columns the CPSea loader consumes; schema matches CPSea_data/lnr_staged.
        "example_id": kept["tag"],
        "path": kept["projected_pdb"],
        "binder_chain_id": "B",
        "cluster_id": kept["cluster_id"],
        "split": "lnr_projected",
        "peptide_length": kept["peptide_length"].astype(int),
        "cyclization_type": "other",   # still linear: no ring bond was formed
        # Provenance + the projection's own numbers, for joining against the pre-projection run.
        "arm": f"soft_closure_{args.cyc_type}",
        "parent_example_id": kept["example_id"],
        "replica": kept["replica"].astype(int),
        "nc_gap_angstrom": kept["pull_dist_final_A"],
        "input_nc_gap_angstrom": kept["input_nc_gap_A"],
        "terminal_cb_A": kept["terminal_cb_final_A"],
        "projection_contact_retention": kept["contact_retention"],
        "projection_ca_rmsd_A": kept["ca_rmsd_to_input_A"],
    })
    path = out_dir / "metadata" / "projected.parquet"
    out.to_parquet(path, index=False)
    print(f"\nmetadata for the SDEdit arm: {path}  ({len(out)} projected inputs)", flush=True)


if __name__ == "__main__":
    main()
