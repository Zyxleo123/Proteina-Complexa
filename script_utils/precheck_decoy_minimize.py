"""Frozen-pocket minimization of perturbed poses (pre-build check 1, stage B).

Reads the stage-A manifest, minimizes every pose with the RECEPTOR FROZEN, and writes the
minimized complex alongside the input.  The before/after pair is what makes the collapse
rate measurable: how much of an initial displacement a local minimization simply undoes.

## Why one system per complex, not one per pose

`scripts/soft_closure_project.py` records that building the force field, the frozen
receptor and the index vectors costs about 63 s of a 65 s replica.  Every pose of a complex
shares a topology and differs only by a rigid transform of the peptide, so the system is
built ONCE from the native and then re-posed for each of the 16 poses using the transform
the manifest carries.  Re-running pdbfixer per pose would multiply the job by sixteen for
no difference in the answer.

The transform is re-applied to the HYDROGENATED coordinates rather than reading stage A's
pose PDBs, because pdbfixer adds hydrogens whose count and order need not match across
separately-fixed files.  Applying the same rigid motion to the same fixed structure keeps
every pose of a complex in one atom ordering by construction.

## The restraint is a HOLD, not a PULL

`soft_closure_project.py`'s restraint drags termini together down a ladder.  This one does
the opposite job: the decoy starts closed and must stay closed, so the ring bond is held at
its input distance.  Without it a force field that never saw the ring bond -- CONECT records
do not survive into `createSystem` -- would happily open the macrocycle during minimization,
and the interface score would then be measuring a broken peptide.

Runs in `.venv_openmm` (openmm 8.6 / pdbfixer).  No mdtraj, no sklearn, no PyRosetta here:
the two environments are mutually exclusive by design, so this stage writes PDBs and the
scoring stage reads them back in `.venv`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

try:
    import openmm
    from openmm import app, unit
    from openmm.app import PDBFile
    from openmm.openmm import LocalEnergyMinimizer
    from pdbfixer import PDBFixer
except ImportError as exc:                               # pragma: no cover - env guard
    raise SystemExit(
        f"{exc}. This stage runs in .venv_openmm, not .venv -- the two are mutually "
        f"exclusive (OpenMM is not installable alongside the training environment's pins)."
    )


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation; duplicated from pose_decoys because that module imports mdtraj,
    which .venv_openmm does not have.  Nine lines is cheaper than a shared dependency that
    would drag the whole feature stack into this environment."""
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    th = np.radians(angle_deg)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _is_heavy(atom) -> bool:
    return atom.element is not None and atom.element.symbol != "H"


class ComplexContext:
    """Force field, frozen receptor and ring restraint for one complex.  Built once."""

    def __init__(self, native_pdb: str, ring: tuple[int, int] | None, ring_len_A: float,
                 mcfg: dict):
        fixer = PDBFixer(filename=native_pdb)
        fixer.findMissingResidues()
        # Deliberately NOT rebuilding missing residues: the receptor is a pocket crop, so
        # every numbering gap looks like a missing loop and rebuilding them would invent
        # the excised protein back into the pocket.
        fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        self.topology = fixer.topology
        self.positions_nm = np.asarray(
            fixer.positions.value_in_unit(unit.nanometer), dtype=np.float64)

        chains = list(self.topology.chains())
        if len(chains) < 2:
            raise RuntimeError(f"{native_pdb}: expected receptor + peptide chains, "
                               f"found {len(chains)}")
        rec_chain, pep_chain = chains[0], chains[1]
        self.rec_atoms = [a.index for a in rec_chain.atoms()]
        self.pep_atoms = np.array([a.index for a in pep_chain.atoms()], dtype=int)
        self.pep_heavy = np.array([a.index for a in pep_chain.atoms() if _is_heavy(a)],
                                  dtype=int)

        forcefield = app.ForceField(*mcfg["forcefield"])
        system = forcefield.createSystem(
            self.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=float(mcfg["nonbonded_cutoff_nm"]) * unit.nanometer,
            # No constraints: OpenMM refuses a constraint touching a massless particle, and
            # the receptor is frozen by zeroing masses. Minimization needs none anyway.
            constraints=None,
        )
        # Frozen pocket: mass 0 keeps the receptor contributing to the energy while making
        # it immovable, which is what "frozen-pocket minimization" means here.
        for i in self.rec_atoms:
            system.setParticleMass(i, 0.0)

        self.ring = None
        if ring is not None and ring[0] >= 0 and ring[1] >= 0:
            # Indices came from the UNFIXED file; hydrogens shift them. Re-find the pair by
            # matching (residue index, atom name) rather than trusting the raw index.
            self.ring = self._map_ring(native_pdb, ring)
            if self.ring is not None:
                force = openmm.HarmonicBondForce()
                force.addBond(int(self.ring[0]), int(self.ring[1]),
                              float(ring_len_A) * 0.1 * unit.nanometer,
                              float(mcfg["ring_hold_k_kj_mol_nm2"])
                              * unit.kilojoule_per_mole / unit.nanometer ** 2)
                system.addForce(force)

        platform = openmm.Platform.getPlatformByName(mcfg.get("platform", "CPU"))
        self.integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)  # never stepped
        self.context = openmm.Context(system, self.integrator, platform)
        self.tolerance = float(mcfg["tolerance_kj_mol_nm"])
        self.max_iterations = int(mcfg["max_iterations"])

    def _map_ring(self, native_pdb: str, ring: tuple[int, int]):
        """Map raw-file atom indices onto the hydrogenated topology by (residue, name)."""
        raw = PDBFile(native_pdb)
        raw_atoms = list(raw.topology.atoms())
        try:
            want = [(raw_atoms[i].residue.index, raw_atoms[i].name) for i in ring]
        except IndexError:
            return None
        lookup = {(a.residue.index, a.name): a.index for a in self.topology.atoms()}
        mapped = [lookup.get(w) for w in want]
        return None if any(m is None for m in mapped) else (mapped[0], mapped[1])

    def minimize_pose(self, rot: np.ndarray, trans_A: np.ndarray, centre_A: np.ndarray):
        """Apply the rigid transform to the peptide, minimize, return (start, end) in A."""
        xyz_A = self.positions_nm * 10.0
        posed = xyz_A.copy()
        posed[self.pep_atoms] = ((xyz_A[self.pep_atoms] - centre_A) @ rot.T
                                 + centre_A + trans_A)

        self.context.setPositions(posed * 0.1 * unit.nanometer)
        e0 = self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        LocalEnergyMinimizer.minimize(self.context, self.tolerance, self.max_iterations)
        st = self.context.getState(getPositions=True, getEnergy=True)
        out_A = np.asarray(st.getPositions().value_in_unit(unit.nanometer)) * 10.0
        e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return posed, out_A, e0, e1

    def write(self, xyz_A: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            PDBFile.writeFile(self.topology, (xyz_A * 0.1).tolist() * unit.nanometer, fh,
                              keepIds=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--minimize-config", required=True,
                    help="JSON of decoy_calibration.minimize, written by the submitter. "
                         "The YAML loader lives behind imports .venv_openmm lacks.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mcfg = json.loads(Path(args.minimize_config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_dir = out_dir / "minimized"

    rows = []
    for f in sorted(Path(args.manifest_dir).glob("manifest_shard*.jsonl")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"{f}:{n}: unparseable manifest row ({exc}). Refusing to "
                                 f"continue -- a skipped row is silently missing data.")
    if not rows:
        raise SystemExit(f"no manifest rows under {args.manifest_dir}")

    by_example: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        by_example.setdefault(r["example_id"], []).append(r)

    examples = list(by_example)[args.shard::args.n_shards]
    if args.limit:
        examples = examples[:args.limit]

    out_path = out_dir / f"minimized_shard{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    done.add((d["example_id"], d["pose_index"]))
                except Exception:
                    raise SystemExit(f"{out_path}: unparseable row while resuming; "
                                     "delete it and rerun")

    t0 = time.perf_counter()
    n_pose = n_fail = 0
    with out_path.open("a") as fh:
        for ex in examples:
            poses = sorted(by_example[ex], key=lambda r: r["pose_index"])
            if all((ex, p["pose_index"]) in done for p in poses):
                continue
            first = poses[0]
            try:
                ctx = ComplexContext(
                    first["native_pdb"],
                    (int(first.get("ring_atom_i", -1)), int(first.get("ring_atom_j", -1))),
                    float(first.get("ring_bond_A", 1.33) or 1.33),
                    mcfg)
            except Exception as exc:
                print(f"  skip {ex}: system build failed: {type(exc).__name__}: {exc}",
                      flush=True)
                n_fail += 1
                continue

            centre = np.array([first["centre_x_A"], first["centre_y_A"], first["centre_z_A"]])
            for p in poses:
                if (ex, p["pose_index"]) in done:
                    continue
                rot = rotation_matrix(
                    np.array([p["rot_axis_x"], p["rot_axis_y"], p["rot_axis_z"]]),
                    float(p["rotation_deg"]))
                trans = np.array([p["tx_A"], p["ty_A"], p["tz_A"]])
                try:
                    start_A, end_A, e0, e1 = ctx.minimize_pose(rot, trans, centre)
                except Exception as exc:
                    print(f"  {ex} pose {p['pose_index']}: minimize failed: {exc}",
                          flush=True)
                    n_fail += 1
                    continue

                tag = f"{ex}__pose{int(p['pose_index']):02d}"
                pre = min_dir / f"{tag}__pre.pdb"
                post = min_dir / f"{tag}__post.pdb"
                ctx.write(start_A, pre)
                ctx.write(end_A, post)

                # Peptide heavy-atom displacement caused by the minimization itself: how
                # much of the perturbation the minimizer simply undid.
                d = float(np.sqrt(((end_A[ctx.pep_heavy] - start_A[ctx.pep_heavy]) ** 2)
                                  .sum(-1).mean()))
                fh.write(json.dumps({
                    **{k: p[k] for k in p if k != "pose_pdb"},
                    "pre_pdb": str(pre), "post_pdb": str(post),
                    "energy_start_kj": e0, "energy_end_kj": e1,
                    "minimization_rmsd_A": d,
                    "ring_held": bool(ctx.ring is not None),
                }, default=float) + "\n")
                # Flushed per POSE, not per complex: the resume key is
                # (example_id, pose_index), so buffering a whole complex would throw away
                # every pose of it on a wall-clock kill despite the finer key.
                fh.flush()
                n_pose += 1
            print(f"  {ex}: {len(poses)} poses ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"wrote {out_path}: {n_pose} poses minimized, {n_fail} failed, "
          f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
