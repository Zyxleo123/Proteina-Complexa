"""ARM C -- relax the LNR complexes so their geometry matches CPSea's training distribution.

Why this exists
---------------
The pinned model was trained on CPSea, which is 100% AlphaFold models, 100% energy-relaxed:

    cpsea_train  n= 2444209  AF-model=100.0%  relaxed=100.0%
    cpsea_val    n=  134394  AF-model=100.0%  relaxed=100.0%

It has never seen an experimental structure. LNR is 100% raw crystal. After Arm B fixed the
receptor CROP (see scripts/restage_lnr_pocket.py), a residual closure gap remained, and the
two competing explanations for it -- forced-vs-native cyclization type, and peptide length --
were both measured and excluded. Structure provenance is the axis left standing: idealized
bond geometry, rotamer distributions, crystal packing artifacts and disorder all differ
between a relaxed AF model and a raw crystal structure.

The protocol
------------
CPSea's own relaxation is not recoverable from here -- the `_relaxed_relaxed` suffix is
already present in the upstream source_path, so it happened before this repo saw the data.
Since the structures being matched ARE AlphaFold models, this replicates AlphaFold's own
relaxation (alphafold/relax/amber_minimize.py): amber99sb, harmonic position restraints on
every non-hydrogen atom at stiffness 10 kcal/mol/A^2, L-BFGS to a 10 kJ/mol/nm tolerance.

Restrained minimization is the point, not a compromise: it idealizes bond lengths/angles and
relieves clashes WITHOUT moving the fold, so the binding pose -- the thing the whole
experiment conditions on -- is preserved.

Numbering is preserved bit-exactly
----------------------------------
Relaxed coordinates are transplanted back into the ORIGINAL ATOM records, matched on
(chain, resseq, atom name). Chain IDs, residue numbers and atom order are therefore
untouched. This matters more than it sounds: the numbering GAPS are what encode the receptor
as disjoint segments, and rewriting them would silently undo Arm B's fix. Atoms pdbfixer
adds (missing sidechain atoms, hydrogens) are dropped; an original atom that goes missing is
an error, not a silent omission.

Relax the FULL complex, then crop. Cropping first would minimize severed chain ends into
the vacuum they were cut from.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# AlphaFold's amber_minimize defaults.
STIFFNESS_KCAL_MOL_A2 = 10.0
# UNITS TRAP: AlphaFold passes `tolerance=2.39 kcal/mol` (an ENERGY) because its OpenMM
# interpreted it that way. OpenMM 8 takes a FORCE tolerance in kJ/mol/nm and rejects energy
# units outright. AF's 2.39 kcal/mol is numerically 10 kJ/mol, which is exactly OpenMM's own
# default force tolerance -- so 10.0 kJ/mol/nm both satisfies the modern API and preserves
# the number AF actually used.
TOLERANCE_KJ_MOL_NM = 10.0
MAX_ITERATIONS = 0  # 0 = run to convergence


def relax_one(src: Path, dst: Path, stiffness: float, tolerance: float) -> dict:
    """Restrained Amber minimization; returns per-structure diagnostics."""
    import openmm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(src))
    fixer.findMissingResidues()
    # Do NOT build in residues absent from the crystal: inventing unresolved loops would
    # change what the receptor IS, not just how relaxed it is.
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    ff = app.ForceField("amber99sb.xml")
    system = ff.createSystem(fixer.topology, constraints=app.HBonds)

    # Harmonic restraint on every heavy atom -- AF's formulation.
    force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness * unit.kilocalories_per_mole / unit.angstroms**2)
    for p in ("x0", "y0", "z0"):
        force.addPerParticleParameter(p)
    n_restrained = 0
    for atom in fixer.topology.atoms():
        if atom.element is not None and atom.element.symbol != "H":
            force.addParticle(atom.index, fixer.positions[atom.index].value_in_unit(unit.nanometers))
            n_restrained += 1
    system.addForce(force)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = openmm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(fixer.topology, system, integrator, platform)
    sim.context.setPositions(fixer.positions)

    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalories_per_mole)
    sim.minimizeEnergy(
        tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=MAX_ITERATIONS)
    state = sim.context.getState(getPositions=True, getEnergy=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    pos = state.getPositions().value_in_unit(unit.angstroms)

    # (chain, resseq, atomname) -> relaxed xyz
    relaxed = {}
    for atom in fixer.topology.atoms():
        res = atom.residue
        relaxed[(res.chain.id, int(res.id), atom.name.strip())] = pos[atom.index]

    # Transplant into the ORIGINAL records so numbering/chains/atom order survive verbatim.
    out_lines, n_written, missing, max_shift = [], 0, [], 0.0
    for line in src.read_text().splitlines(keepends=True):
        if not line.startswith("ATOM"):
            out_lines.append(line)
            continue
        ch = line[21]
        resseq = int(line[22:26])
        name = line[12:16].strip()
        key = (ch, resseq, name)
        if key not in relaxed:
            missing.append(key)
            continue
        x, y, z = relaxed[key]
        ox, oy, oz = float(line[30:38]), float(line[38:46]), float(line[46:54])
        max_shift = max(max_shift, ((x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2) ** 0.5)
        out_lines.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        n_written += 1

    if missing:
        raise RuntimeError(f"{len(missing)} original atoms absent after fixing, "
                           f"e.g. {missing[:3]}")

    dst.write_text("".join(out_lines))
    return {"n_restrained": n_restrained, "n_atoms_written": n_written,
            "energy_before_kcal": e0, "energy_after_kcal": e1, "max_shift_A": max_shift}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-metadata", required=True,
                    help="LNR staged parquet -- the FULL complexes, before any pocket crop.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stiffness", type=float, default=STIFFNESS_KCAL_MOL_A2)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_KJ_MOL_NM,
                    help="Minimizer FORCE tolerance in kJ/mol/nm (OpenMM 8 semantics).")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0,{args.shard_count})")

    out_dir = Path(args.out_dir)
    pdb_out = out_dir / "pdbs"
    pdb_out.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(args.in_metadata)
    if args.limit:
        meta = meta.head(args.limit)
    mine = meta.iloc[args.shard_index::args.shard_count]
    print(f"{len(meta)} complexes; shard {args.shard_index}/{args.shard_count} takes {len(mine)}",
          flush=True)

    # Per-shard log: concurrent O_APPEND to one file has NUL-corrupted rows on this cluster.
    rec_path = log_dir / f"relax_shard{args.shard_index}.jsonl"
    import json
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            # Only SUCCESSES count as done. Treating a failed row as done would skip it
            # forever on resume, silently shrinking the set instead of retrying it.
            if rec.get("status") == "ok":
                done.add(rec["example_id"])
        print(f"resuming: {len(done)} already relaxed", flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    with rec_path.open("a") as fh:
        for _, r in mine.iterrows():
            eid = str(r["example_id"])
            if eid in done:
                continue
            src = Path(str(r["path"]))
            dst = pdb_out / src.name
            try:
                info = relax_one(src, dst, args.stiffness, args.tolerance)
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                fh.write(json.dumps({"example_id": eid, "status": "failed",
                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                fh.flush()
                print(f"  FAIL {eid}: {type(exc).__name__}: {exc}", flush=True)
                continue
            info.update(example_id=eid, status="ok", path=str(dst.resolve()))
            fh.write(json.dumps(info) + "\n")
            fh.flush()
            n_ok += 1
            print(f"  {eid}: dE {info['energy_before_kcal']:.0f} -> "
                  f"{info['energy_after_kcal']:.0f} kcal/mol, max shift "
                  f"{info['max_shift_A']:.2f} A", flush=True)

    print(f"done: {n_ok} relaxed, {n_fail} failed, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
