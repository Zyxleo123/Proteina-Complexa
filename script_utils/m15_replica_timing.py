"""Milestone 1.5 / section 2 -- measure the replica economics before sizing the pilot.

The whole decoy budget rests on one ratio: how much of a replica's cost is per-EXAMPLE
context setup (force field, PDBFixer hydrogen build, system construction, restraint
selection) versus per-REPLICA marginal work (set coordinates, minimize).  The
`ExampleContext` docstring in scripts/soft_closure_project.py claims ~63 s of a ~65 s
replica is setup.  If that holds, decoys are nearly free and complexes are expensive, and
the sampling shape should be wide on replicas and narrow on complexes.

This measures it rather than trusting it, on the 20-complex AFDB fixture.

Two deliberate choices:

  * The marginal unit measured is ONE restrained minimization, not the 6-rung pull ladder
    `project_once` runs.  A decoy is perturbed and settled once; quoting the ladder's cost
    would overstate the marginal by roughly the number of rungs and make replicas look
    expensive.  The ladder is timed too, for comparability with the 65 s figure.
  * Every fixture complex is timed on the mainchain restraint (N of residue 0 to C of
    residue L-1) with r0 pinned at the NATIVE distance, so the restraint holds rather than
    pulls.  That is the decoy-settle force, and it is also the only chemistry whose atoms
    are guaranteed present.  Cost does not depend on which two atoms carry the bond term:
    the minimizer works over the whole system either way.

Runs in `.venv_openmm`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from soft_closure_project import (  # noqa: E402
    ExampleContext, LocalEnergyMinimizer, Platform, geometry_report, perturb_backbone_torsions,
    prepare_complex, reference_state, app, unit,
)


def pick_fixture(metadata: Path, n: int, seed: int) -> list:
    """`n` complexes, spread across cyclization types so the timing is not one chemistry."""
    df = pq.read_table(metadata).to_pandas()
    rng = np.random.default_rng(seed)
    per_type = max(1, n // max(1, df["cyclization_type"].nunique()))
    picks = []
    for _, grp in df.groupby("cyclization_type"):
        take = min(per_type, len(grp))
        picks.append(grp.iloc[np.sort(rng.choice(len(grp), take, replace=False))])
    out = __import__("pandas").concat(picks)
    if len(out) < n:                      # top up from whatever is left
        rest = df[~df["example_id"].isin(out["example_id"])]
        if len(rest):
            extra = min(n - len(out), len(rest))
            out = __import__("pandas").concat(
                [out, rest.iloc[np.sort(rng.choice(len(rest), extra, replace=False))]])
    return list(out.iloc[:n].itertuples(index=False))


def time_one(rec, args, forcefield, platform) -> dict:
    """Setup cost and per-replica marginal cost for one complex."""
    row = {"example_id": rec.example_id, "path": str(rec.path),
           "cyclization_type": str(rec.cyclization_type),
           "peptide_length": int(rec.peptide_length),
           "receptor_length": int(rec.receptor_length)}

    t0 = time.perf_counter()
    fixer, n_added = prepare_complex(Path(rec.path), None, args.pep_chain)
    t_prepare = time.perf_counter() - t0

    ref = reference_state(fixer)
    ctx_args = SimpleNamespace(contacts_per_residue=args.contacts_per_residue,
                               k_pull=args.k_pull, k_contact=args.k_contact,
                               contact_tol_A=args.contact_tol_A)
    t0 = time.perf_counter()
    ec = ExampleContext(fixer, "mainchain", ctx_args, ref, forcefield, platform)
    t_context = time.perf_counter() - t0

    base_A = ec.base_xyz_nm * 10.0
    d_native_A = float(np.linalg.norm(base_A[ec.ia] - base_A[ec.ib]))
    # Hold, not pull: r0 at the native separation is exactly the ring-closure HOLD restraint
    # the decoy-settling step needs (gap-list item 4), obtained from the existing force.
    ec.context.setParameter("r0", d_native_A * 0.1)

    replica_s, gap_A = [], []
    import random as _random
    for k in range(args.n_replicas):
        xyz_nm = ec.base_xyz_nm.copy()
        t0 = time.perf_counter()
        if args.torsion_sigma_deg > 0:
            perturb_backbone_torsions(xyz_nm, ec.pep_res_atoms, args.torsion_sigma_deg,
                                      _random.Random(args.seed * 10_000 + k))
        ec.context.setPositions(xyz_nm * unit.nanometer)
        LocalEnergyMinimizer.minimize(ec.context, args.tolerance, args.max_iterations)
        st = ec.context.getState(getPositions=True)
        replica_s.append(time.perf_counter() - t0)
        xyz_A = np.asarray(st.getPositions().value_in_unit(unit.nanometer)) * 10.0
        gap_A.append(float(np.linalg.norm(xyz_A[ec.ia] - xyz_A[ec.ib])))

    # The ladder, for comparability with the ~65 s figure in the ExampleContext docstring.
    t_ladder = float("nan")
    if args.time_ladder:
        ec.context.setPositions(ec.base_xyz_nm * unit.nanometer)
        t0 = time.perf_counter()
        for rung in np.linspace(d_native_A, 1.33, args.ladder_rungs):
            ec.context.setParameter("r0", float(rung) * 0.1)
            LocalEnergyMinimizer.minimize(ec.context, args.tolerance, args.max_iterations)
        t_ladder = time.perf_counter() - t0

    marginal = statistics.median(replica_s)
    setup = t_prepare + t_context
    row.update({
        "n_atoms": int(ec.base_xyz_nm.shape[0]),
        "n_added_atoms": int(n_added),
        "n_contact_restraints": int(len(ec.contact_pairs)),
        "t_prepare_s": t_prepare,
        "t_context_s": t_context,
        "t_setup_s": setup,
        "t_replica_median_s": marginal,
        "t_replica_mean_s": statistics.fmean(replica_s),
        "t_replica_min_s": min(replica_s),
        "t_replica_max_s": max(replica_s),
        "t_replica_first_s": replica_s[0],
        "t_ladder_s": t_ladder,
        "setup_over_marginal": setup / marginal if marginal > 0 else float("nan"),
        "native_bond_gap_A": d_native_A,
        "settled_gap_median_A": float(np.median(gap_A)),
        # Wall-clock per decoy at three shapes, which is the number the budget actually needs.
        "s_per_decoy_at_8": (setup + 8 * marginal) / 8,
        "s_per_decoy_at_30": (setup + 30 * marginal) / 30,
        "s_per_decoy_at_100": (setup + 100 * marginal) / 100,
    })
    del ec
    return row


def write_summary(rows: list[dict], args) -> dict:
    """Median over whatever has completed so far. Safe to call after every complex."""
    med = lambda k: statistics.median([r[k] for r in rows])  # noqa: E731
    summary = {
        "n_complexes": len(rows),
        "n_complexes_requested": args.n_complexes,
        "partial": len(rows) < args.n_complexes,
        "n_replicas_each": args.n_replicas,
        "setup_median_s": med("t_setup_s"),
        "marginal_median_s": med("t_replica_median_s"),
        "ratio_median": med("setup_over_marginal"),
        "s_per_decoy_at_8": med("s_per_decoy_at_8"),
        "s_per_decoy_at_30": med("s_per_decoy_at_30"),
        "s_per_decoy_at_100": med("s_per_decoy_at_100"),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-complexes", type=int, default=20)
    ap.add_argument("--n-replicas", type=int, default=32)
    ap.add_argument("--pep-chain", default="B")
    ap.add_argument("--torsion-sigma-deg", type=float, default=8.0)
    # Defaults mirror scripts/soft_closure_project.py exactly. A bare invocation must
    # measure the reference protocol, not a differently-parameterised one that happens to
    # share a name.
    ap.add_argument("--contacts-per-residue", type=int, default=1)
    ap.add_argument("--k-pull", type=float, default=2000.0)
    ap.add_argument("--k-contact", type=float, default=500.0)
    ap.add_argument("--contact-tol-A", type=float, default=0.75)
    ap.add_argument("--tolerance", type=float, default=10.0)
    ap.add_argument("--max-iterations", type=int, default=1000)
    ap.add_argument("--ladder-rungs", type=int, default=6)
    ap.add_argument("--time-ladder", action="store_true")
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    forcefield = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    platform = Platform.getPlatformByName(args.platform)
    fixture = pick_fixture(args.metadata, args.n_complexes, args.seed)
    print(f"[timing] {len(fixture)} complexes x {args.n_replicas} replicas on {args.platform}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w") as fh:
        for rec in fixture:  # summary is rewritten after EVERY complex, see below
            try:
                row = time_one(rec, args, forcefield, platform)
            except Exception as exc:  # noqa: BLE001
                row = {"example_id": rec.example_id, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
            else:
                row["status"] = "ok"
                rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"  {row['example_id']}: setup {row.get('t_setup_s', float('nan')):.1f}s  "
                  f"marginal {row.get('t_replica_median_s', float('nan')):.2f}s  "
                  f"ratio {row.get('setup_over_marginal', float('nan')):.3f}x", flush=True)
            # Rewritten after EVERY complex. This job is minutes per complex, so a Slurm
            # time limit is a normal outcome, not an error -- and a summary written only
            # at the end means a wall-clock kill throws away every complex that DID
            # finish, and the report then silently omits replica economics entirely.
            if rows:
                write_summary(rows, args)

    if not rows:
        raise SystemExit("every fixture complex failed -- no timing to report")
    print(json.dumps(write_summary(rows, args), indent=2))


if __name__ == "__main__":
    main()
