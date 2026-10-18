"""Isolate the `Particle coordinate is NaN` failure before it scales into the decoy build.

`AF-A0A537RZW2-F1_0_130_140_relaxed_relaxed` raised OpenMMException inside the settling
minimize during the replica-timing run. The timing harness caught it per complex, so it cost
one row there -- but the decoy build runs that same minimizer on every replica of every
complex, where a 5% silent loss rate biases the decoy distribution with nothing to flag it.

This answers the three questions a skip-or-fix policy needs and nothing else:

  1. Is the INPUT already non-finite?  If PDBFixer or the staged PDB hands over a NaN, this
     is a data problem and the minimizer is innocent.  (The doubled `_relaxed_relaxed`
     suffix on the failing id makes a degenerate twice-relaxed geometry the first suspect.)
  2. If the input is clean, WHICH step diverges?  A bare `except OpenMMException` cannot say,
     because the exception surfaces from whichever minimize happened to be running.  So the
     coordinates are checked after every single minimize.
  3. Is it deterministic and is it specific to this complex?  A NaN that moves with the seed
     is a sampling problem; one that reproduces exactly is geometry.  Control complexes from
     the same fixture say whether the policy needs to be general or targeted.

Emits one JSON row per (complex, replica) plus a verdict block, so the policy is chosen from
a record rather than from a single traceback.  Runs in `.venv_openmm`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from soft_closure_project import (  # noqa: E402
    ExampleContext, LocalEnergyMinimizer, Platform, perturb_backbone_torsions,
    prepare_complex, reference_state, unit,
)

# The complex that failed in m15_20260922_184436/timing. Overridable, but recorded here so
# the reproducer stays runnable after the fixture seed or metadata changes underneath it.
DEFAULT_TARGET = "AF-A0A537RZW2-F1_0_130_140_relaxed_relaxed"


def finite_report(xyz: np.ndarray, tag: str) -> dict:
    """Where the non-finite values are, not just whether there are any.

    An atom index is actionable (it names a residue); a bare boolean is not. `max_abs` is
    carried because a coordinate blowing up to 1e30 before going NaN means the minimizer
    diverged, while a NaN with sane neighbours means it arrived that way.
    """
    bad = ~np.isfinite(xyz).all(axis=1)
    return {
        "tag": tag,
        "n_atoms": int(xyz.shape[0]),
        "n_nonfinite_atoms": int(bad.sum()),
        "first_nonfinite_atom": int(np.argmax(bad)) if bad.any() else -1,
        "max_abs_coord_nm": (float(np.abs(xyz[np.isfinite(xyz)]).max())
                             if np.isfinite(xyz).any() else float("nan")),
        "clean": bool(not bad.any()),
    }


def probe_one(rec, args, forcefield, platform) -> dict:
    """Walk one complex's settling path, checking coordinates after every minimize."""
    row: dict = {"example_id": str(rec.example_id), "path": str(rec.path),
                 "cyclization_type": str(rec.cyclization_type),
                 "peptide_length": int(rec.peptide_length),
                 "receptor_length": int(rec.receptor_length),
                 "stages": [], "status": "ok", "error": None,
                 "first_bad_stage": None}

    try:
        fixer, n_added, _ = prepare_complex(Path(rec.path), None, args.pep_chain)
        row["n_added_atoms"] = int(n_added)
        ref = reference_state(fixer)
        ctx_args = SimpleNamespace(contacts_per_residue=args.contacts_per_residue,
                                   k_pull=args.k_pull, k_contact=args.k_contact,
                                   contact_tol_A=args.contact_tol_A)
        ec = ExampleContext(fixer, "mainchain", ctx_args, ref, forcefield, platform)

        # Question 1: is the input clean, before any minimization touches it?
        st0 = finite_report(ec.base_xyz_nm, "input:base_xyz_nm")
        row["stages"].append(st0)
        row["input_clean"] = st0["clean"]
        if not st0["clean"]:
            # Nothing downstream is interpretable if the input is already broken, and
            # OpenMM will raise on setPositions anyway. Stop here with the finding.
            row["status"] = "input_nonfinite"
            row["first_bad_stage"] = st0["tag"]
            return row

        base_A = ec.base_xyz_nm * 10.0
        d_native_A = float(np.linalg.norm(base_A[ec.ia] - base_A[ec.ib]))
        row["native_bond_gap_A"] = d_native_A
        ec.context.setParameter("r0", d_native_A * 0.1)

        for k in range(args.n_replicas):
            xyz_nm = ec.base_xyz_nm.copy()
            if args.torsion_sigma_deg > 0:
                perturb_backbone_torsions(xyz_nm, ec.pep_res_atoms, args.torsion_sigma_deg,
                                          random.Random(args.seed * 10_000 + k))
            # Question 2, first half: did the PERTURBATION produce the NaN? The torsion
            # rebuild does trigonometry per residue, so a degenerate bond geometry there
            # yields NaN before the minimizer is ever called.
            rp = finite_report(xyz_nm, f"replica{k}:after_perturb")
            row["stages"].append(rp)
            if not rp["clean"]:
                row["status"] = "perturb_nonfinite"
                row["first_bad_stage"] = rp["tag"]
                return row

            ec.context.setPositions(xyz_nm * unit.nanometer)
            try:
                LocalEnergyMinimizer.minimize(ec.context, args.tolerance, args.max_iterations)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "minimize_raised"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["first_bad_stage"] = f"replica{k}:minimize"
                return row

            # Question 2, second half: OpenMM does not always raise -- it can return
            # positions that are already NaN and fail on the NEXT call, which is what makes
            # an outer except unable to name the diverging step.
            xyz_out = np.asarray(ec.context.getState(getPositions=True)
                                 .getPositions().value_in_unit(unit.nanometer))
            rm = finite_report(xyz_out, f"replica{k}:after_minimize")
            row["stages"].append(rm)
            if not rm["clean"]:
                row["status"] = "minimize_nonfinite"
                row["first_bad_stage"] = rm["tag"]
                return row

        row["hold_phase_clean"] = True

        # The LADDER. `m15_replica_timing.py` runs this after the replica loop whenever
        # `time_ladder` is true (it is, in m15.yaml), and main()'s try/except wraps both -- so
        # a NaN attributed to "the timing run" may well come from here rather than from the
        # hold above. The two are very different deformations: the hold keeps r0 at the native
        # separation, while the ladder drags the termini to 1.33 A, which is where a backbone
        # that cannot reach gets crushed. Probing only the hold would clear a path that was
        # never the suspect.
        if args.with_ladder:
            ec.context.setPositions(ec.base_xyz_nm * unit.nanometer)
            for ri, rung in enumerate(np.linspace(d_native_A, 1.33, args.ladder_rungs)):
                ec.context.setParameter("r0", float(rung) * 0.1)
                try:
                    LocalEnergyMinimizer.minimize(ec.context, args.tolerance,
                                                  args.max_iterations)
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "ladder_minimize_raised"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["first_bad_stage"] = f"ladder_rung{ri}(r0={rung:.2f}A):minimize"
                    return row
                xyz_l = np.asarray(ec.context.getState(getPositions=True)
                                   .getPositions().value_in_unit(unit.nanometer))
                rl = finite_report(xyz_l, f"ladder_rung{ri}(r0={rung:.2f}A):after_minimize")
                row["stages"].append(rl)
                if not rl["clean"]:
                    row["status"] = "ladder_nonfinite"
                    row["first_bad_stage"] = rl["tag"]
                    return row

    except Exception as exc:  # noqa: BLE001
        row["status"] = "setup_raised"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=12)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", default=DEFAULT_TARGET,
                    help="example_id to isolate; substring match is accepted")
    ap.add_argument("--n-controls", type=int, default=3,
                    help="clean complexes run alongside, to tell 'this complex' from 'this code'")
    ap.add_argument("--n-replicas", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=2,
                    help="times the target is re-run, to test determinism")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pep-chain", default="B")
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--contacts-per-residue", type=int, default=4)
    ap.add_argument("--k-pull", type=float, default=200.0)
    ap.add_argument("--k-contact", type=float, default=20.0)
    ap.add_argument("--contact-tol-A", type=float, default=1.0)
    ap.add_argument("--torsion-sigma-deg", type=float, default=10.0)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--max-iterations", type=int, default=0)
    # Default ON: the ladder is the phase the timing run's try/except also covered, so it is
    # the prime suspect. `--no-ladder` isolates the hold path on its own.
    ap.add_argument("--with-ladder", dest="with_ladder", action="store_true", default=True)
    ap.add_argument("--no-ladder", dest="with_ladder", action="store_false")
    ap.add_argument("--ladder-rungs", type=int, default=6)
    args = ap.parse_args()

    from openmm import app as _app  # noqa: N813
    forcefield = _app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    platform = Platform.getPlatformByName(args.platform)

    df = pq.read_table(args.metadata).to_pandas()
    hit = df[df["example_id"].astype(str).str.contains(args.target, regex=False)]
    if hit.empty:
        # A gate, not a crash: the fixture may have been regenerated. Say so and exit 0 so
        # nothing downstream is stranded on a dependency that will never be satisfied.
        print(f"GATE: target '{args.target}' not in {args.metadata}. Nothing to reproduce.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"status": "target_absent",
                                        "target": args.target}) + "\n")
        return

    target_rec = list(hit.iloc[:1].itertuples(index=False))[0]
    rest = df[~df["example_id"].astype(str).str.contains(args.target, regex=False)]
    rng = np.random.default_rng(args.seed)
    n_ctl = min(args.n_controls, len(rest))
    ctl_recs = list(rest.iloc[np.sort(rng.choice(len(rest), n_ctl, replace=False))]
                    .itertuples(index=False)) if n_ctl else []

    print(f"[nan-repro] target : {target_rec.example_id}")
    print(f"[nan-repro] controls: {[str(r.example_id) for r in ctl_recs]}")
    print(f"[nan-repro] {args.repeats} repeat(s) x {args.n_replicas} replica(s)\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w") as fh:
        for rep in range(args.repeats):
            r = probe_one(target_rec, args, forcefield, platform)
            r.update({"role": "target", "repeat": rep})
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"  target repeat {rep}: {r['status']}"
                  f"  first_bad={r.get('first_bad_stage')}")
        for rec in ctl_recs:
            r = probe_one(rec, args, forcefield, platform)
            r.update({"role": "control", "repeat": 0})
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"  control {rec.example_id}: {r['status']}")

    tgt = [r for r in rows if r["role"] == "target"]
    ctl = [r for r in rows if r["role"] == "control"]
    statuses = sorted({r["status"] for r in tgt})
    # Which PHASE failed decides who is affected, so it is part of the verdict rather than
    # something to be read back out of the stage tags. The decoy build settles (hold); only
    # the closure projection walks the ladder.
    ladder_only = bool(statuses) and all(s.startswith("ladder_") for s in statuses if s != "ok")
    verdict = {
        "target": str(target_rec.example_id),
        "target_statuses": statuses,
        "deterministic": len(statuses) == 1,
        "input_was_nonfinite": any(r.get("input_clean") is False for r in tgt),
        "first_bad_stages": sorted({str(r.get("first_bad_stage")) for r in tgt}),
        "n_controls": len(ctl),
        "n_controls_failed": sum(1 for r in ctl if r["status"] != "ok"),
        "reproduced": any(r["status"] != "ok" for r in tgt),
        "ladder_enabled": bool(args.with_ladder),
        "hold_phase_clean": all(r.get("hold_phase_clean") for r in tgt),
        "failure_is_ladder_only": ladder_only and any(r["status"] != "ok" for r in tgt),
    }
    # The policy follows mechanically from the verdict, so state it here rather than
    # leaving it to be re-derived from the rows by whoever reads this next.
    if not verdict["reproduced"]:
        verdict["policy"] = (
            "did NOT reproduce" + ("" if args.with_ladder else
                                   " -- but the LADDER was disabled, so the phase the timing "
                                   "run also covered was never exercised; re-run with "
                                   "--with-ladder before concluding anything") +
            ". If it does not reproduce with the ladder either, the failure is seed- or "
            "environment-dependent; treat as a replica-level skip with a recorded reason and "
            "do not special-case this complex")
    elif verdict["failure_is_ladder_only"]:
        verdict["policy"] = (
            "LADDER ONLY -- the hold phase is clean across every replica, and the divergence "
            "appears while pulling the termini toward bond distance. The DECOY BUILD settles "
            "(holds r0 at the native separation) and is therefore NOT affected; the CLOSURE "
            "PROJECTION path is. Do not gate the decoy build on this. Fix or guard "
            "soft_closure_project.project_once, where the ladder lives")
    elif verdict["input_was_nonfinite"]:
        verdict["policy"] = ("FIX upstream -- the staged/relaxed input is already "
                             "non-finite; the minimizer is innocent and filtering inputs "
                             "removes the whole class")
    elif verdict["n_controls_failed"]:
        verdict["policy"] = ("GENERAL code problem -- controls fail too; fix the settling "
                             "step before building any decoys")
    else:
        verdict["policy"] = ("SKIP at replica granularity -- specific to this complex's "
                             "geometry; discard non-finite replicas with a recorded reason, "
                             "fail the complex only if all replicas go non-finite, and "
                             "report both counts in the run summary")
    args.out.with_suffix(".verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
