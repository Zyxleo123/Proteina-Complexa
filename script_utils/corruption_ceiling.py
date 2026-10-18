#!/usr/bin/env python
"""How far can a CPSea complex's termini be pulled apart before the interface breaks?

This is the gating measurement for the synthetic-decyclization route. The measured facts
that make it gating:

  * CPSea fragments were ALREADY closed before cyclization -- pre-cyclization terminal
    CA-CA median 6.95 A, max 10.71 A over n=400 AFDB parents, median closure travel 0.76 A
    (disulfide -0.09 A). So the corpus contains no example of a large LP->CP move.
  * Real linear binders sit at 16.3 A (LNR) / 21.6 A (PepBench) median terminal gap.

Synthetic corruption is therefore the only possible source of transport signal, and it has
to manufacture 100% of the travel. If corruption cannot reach ~16 A while keeping the
interface, the route has no training signal and no real-data fallback -- which is worth
knowing before a writer, a filter and a training loop get built on top of it.

What this does, per complex: freeze the receptor, put a harmonic restraint on the terminal
CA pair, and walk its equilibrium length UP a ladder, minimizing at each rung. It is the
existing soft-closure projector run backwards, reusing its force field, frozen receptor,
contact-pair selection and geometry report so the two directions stay comparable.

Contact restraints default to OFF (`--k-contact 0`): the question is what the interface
DOES under opening, so retention is an outcome here, not a constraint. Turning them on
gives the other arm -- whether a target gap is reachable while the interface is held.

Reads the ceiling off two signals, per rung:
  * achieved terminal CA gap vs the rung it was asked for -- saturation means the pocket
    will not let the termini separate further;
  * contact_retention / min heavy distance / CA-step sanity, and Rosetta interface dG,
    which a separate CPU job adds from the dumped complexes.

Rows are JSONL, one file per shard (concurrent O_APPEND to a shared file NUL-corrupts rows
on this filesystem). Resume is at EXAMPLE granularity because the ladder is cumulative --
each rung starts from the previous rung's relaxed state, so a half-finished ladder cannot
be continued, only redone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# Imported for reuse, not reimplemented: same force field, same frozen receptor, same
# contact-pair choice and the same geometry report as the closure direction.
from soft_closure_project import (  # noqa: E402
    ATOMS_FOR_TYPE,
    ExampleContext,
    geometry_report,
    prepare_complex,
    read_staged_pdb,
    terminal_bonds,
    reference_state,
    terminal_cb_A,
    write_projected_pdb,
)

import openmm  # noqa: E402
from openmm import LocalEnergyMinimizer, Platform, unit  # noqa: E402
from openmm import app  # noqa: E402

# CPSea's metadata labels vs the chemistry names the projector uses. `other` is ISOPEPTIDE
# in CPSea_Basic.tsv (checked: the three counts line up exactly with the metadata's).
CYC_LABEL = {"head_tail": "mainchain", "disulfide": "disulfide", "other": "isopeptide"}
PULL_KEY = "ca_termini"


def interior_ca_rmsd_A(ca_now, ca_ref, n_term):
    """CA RMSD with `n_term` residues excluded at EACH end.

    Terminal travel is the edit being asked for, so billing it as damage inverts the
    conclusion -- that mistake is on record. The interior is what should have been preserved.
    """
    if n_term <= 0 or len(ca_now) <= 2 * n_term + 1:
        return float("nan")
    a, b = ca_now[n_term:-n_term], ca_ref[n_term:-n_term]
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def open_ladder(ec, args, header, orig_chains, example_id, cyc_label, pdb_dir,
                ring_opened, removed_bonds):
    """Walk the terminal CA restraint UP the ladder. Returns the per-rung rows."""
    ec.context.setPositions(ec.base_xyz_nm * unit.nanometer)
    xyz_A = ec.base_xyz_nm * 10.0
    d0 = float(np.linalg.norm(xyz_A[ec.ia] - xyz_A[ec.ib]))
    ca_ref = ec.ref["pep_ca_A"]

    # Three kinds of rung, and the middle one is the point:
    #   native   -- the untouched input pose, not minimized and not restrained
    #   baseline -- minimized at the NATIVE separation, i.e. everything the pulled rungs get
    #               EXCEPT the pulling
    #   pull     -- minimized at a target separation above the native one
    # dG must be read against `baseline`, not `native`. Measured on the first smoke: the
    # isopeptide complex showed a flat +6 to +11 REU penalty at EVERY rung, including one
    # 0.25 A from its native separation -- which is not a cost of opening, it is the constant
    # offset between the amber14+obc2 minimum and the unminimized pose Rosetta was handed.
    # Comparing a minimized structure to an unminimized reference bills that offset as damage.
    #
    # Only rungs ABOVE the current separation open anything; a rung below would pull the
    # termini together, which is the other experiment.
    plan = ([("native", None), ("baseline", d0)]
            + [("pull", float(r)) for r in sorted(args.ladder_A) if r > d0])
    rows = []
    for k, (kind, rung) in enumerate(plan):
        limit_reason, n_rounds = None, None
        if rung is None:
            xyz_now, energy = xyz_A, float("nan")
        else:
            ec.context.setParameter("r0", float(rung) * 0.1)
            # Minimize REPEATEDLY at a fixed r0 until the separation stops changing. One
            # bounded minimize() per rung made the achieved gap a function of the iteration
            # budget: rungs came out in near-identical pairs and then jumped, and the
            # resulting "shortfall" recorded how tired the minimizer was, not how much the
            # pocket resisted. Those two have to be told apart, which is what limit_reason
            # is for -- only `stalled` is evidence about the interface.
            prev = float(np.linalg.norm(xyz_now[ec.ia] - xyz_now[ec.ib]))
            limit_reason, n_rounds = "rounds_exhausted", 0
            for n_rounds in range(1, args.pull_rounds + 1):
                LocalEnergyMinimizer.minimize(ec.context, args.tolerance, args.max_iterations)
                st = ec.context.getState(getPositions=True, getEnergy=True)
                xyz_now = np.asarray(st.getPositions().value_in_unit(unit.nanometer)) * 10.0
                energy = float(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                d_now = float(np.linalg.norm(xyz_now[ec.ia] - xyz_now[ec.ib]))
                if abs(d_now - float(rung)) <= args.converge_tol_A:
                    limit_reason = "reached"
                    break
                if abs(d_now - prev) < args.stall_eps_A:
                    limit_reason = "stalled"
                    break
                prev = d_now

        achieved = float(np.linalg.norm(xyz_now[ec.ia] - xyz_now[ec.ib]))
        geo = geometry_report(xyz_now, ec.pep_heavy_idx, ec.pep_ca_idx,
                             ec.ref["xyz_A_ref"][ec.rec_heavy_idx], ec.ref["rec_ca_A"],
                             ca_ref, ec.ref["contacts"])
        suffix = {"native": "native", "baseline": "base"}.get(kind, f"{rung:g}"
                                                               if rung is not None else "?")
        tag = f"{example_id}_{cyc_label}_gap{suffix}"
        pdb_path = pdb_dir / f"{tag}__complex.pdb"
        xyz_full = np.zeros((ec.topology.getNumAtoms(), 3))
        xyz_full[ec.pep_heavy_idx] = xyz_now[ec.pep_heavy_idx]
        write_projected_pdb(pdb_path, header, orig_chains, ec.topology, xyz_full,
                            tag, args.keep_added_atoms)
        rows.append({
            "run_key": f"{example_id}|{k}",
            "example_id": example_id,
            # `cyc_type` / `pdb_tag` / `complex_pdb` are the column names
            # scripts/score_sdedit_rosetta.py keys on, so it scores these unchanged.
            "cyc_type": cyc_label,
            "pdb_tag": tag,
            "complex_pdb": str(pdb_path),
            "rung_index": k,
            "rung_kind": kind,
            "target_gap_A": None if rung is None else float(rung),
            "is_native_rung": kind == "native",
            # The paired reference for dG: same minimization, no pulling.
            "is_baseline_rung": kind == "baseline",
            "terminal_ca_start_A": d0,
            "terminal_ca_A": achieved,
            # Saturation is the ceiling signal: the pocket refusing further separation
            # shows up as shortfall growing while the ask keeps rising.
            "shortfall_A": None if rung is None else float(rung) - achieved,
            # `reached` = the rung was achieved. `stalled` = it stopped moving short of the
            # ask, which is the only shortfall that says anything about the interface.
            # `rounds_exhausted` = still moving when the round budget ran out: an instrument
            # limit, and the report must refuse to read it as a ceiling.
            "limit_reason": limit_reason,
            "n_rounds": n_rounds,
            "ring_opened": ring_opened,
            "removed_ring_bonds": removed_bonds,
            "terminal_cb_A": terminal_cb_A(xyz_now, ec.pep_res_atoms),
            "interior_ca_rmsd_A": interior_ca_rmsd_A(xyz_now[ec.pep_ca_idx], ca_ref,
                                                     args.exclude_termini),
            "energy_kj": energy,
            "k_contact": float(args.k_contact),
            "n_contact_restraints": len(ec.contact_pairs),
            "ladder_complete": k == len(plan) - 1,
            **geo,
        })
    return rows


def select_examples(metadata, per_type, shard, num_shards, seed):
    """Stratified by chemistry, then sharded. Deterministic in `seed`."""
    cols = ["example_id", "path", "cyclization_type"]
    df = pd.read_parquet(metadata, columns=cols)
    df = df[df["cyclization_type"].isin(CYC_LABEL)]
    # An explicit loop, not groupby.apply: pandas 3 excludes the grouping column from the
    # frame handed to apply, so the result comes back WITHOUT cyclization_type and the
    # caller KeyErrors on it much later, inside the first example.
    parts = [g.sample(n=min(per_type, len(g)), random_state=seed)
             for _, g in df.groupby("cyclization_type")]
    picked = pd.concat(parts).sort_values("example_id").reset_index(drop=True)
    return picked.iloc[shard::num_shards]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="CPSea preprocessed metadata parquet.")
    ap.add_argument("--out", required=True, help="Output JSONL for THIS shard (appended).")
    ap.add_argument("--pdb-dir", required=True, help="Where the per-rung complexes are dumped.")
    ap.add_argument("--per-type", type=int, default=20, help="Complexes per chemistry.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ladder-A", nargs="+", type=float,
                    default=[8, 10, 12, 14, 16, 18, 20, 24],
                    help="Ascending target terminal CA-CA separations. 16 A is the LNR "
                         "median and 21.6 A the PepBench median -- the range that matters.")
    ap.add_argument("--exclude-termini", type=int, default=2,
                    help="Residues per end left out of interior_ca_rmsd_A.")
    # --- restraints. Defaults reproduce the closure run's force constants. ---
    ap.add_argument("--k-pull", type=float, default=2000.0, help="kJ/mol/nm^2")
    ap.add_argument("--k-contact", type=float, default=0.0,
                    help="0 (default) lets the interface break so retention is an OUTCOME. "
                         "Non-zero holds the interface and asks if the gap is reachable.")
    ap.add_argument("--contact-tol-A", type=float, default=0.75)
    ap.add_argument("--contacts-per-residue", type=int, default=1, choices=(1, 2))
    ap.add_argument("--torsion-sigma-deg", type=float, default=0.0)
    ap.add_argument("--tolerance", type=float, default=10.0, help="kJ/mol/nm")
    ap.add_argument("--max-iterations", type=int, default=2000)
    # Rounds of minimization per rung, at fixed r0. Repeated rounds converge without a
    # stiffer spring, which matters: a stiff pull yanks the peptide through the receptor,
    # which is the failure the ladder exists to avoid.
    ap.add_argument("--pull-rounds", type=int, default=12)
    ap.add_argument("--converge-tol-A", type=float, default=0.5,
                    help="Within this of the asked separation counts as reached.")
    ap.add_argument("--stall-eps-A", type=float, default=0.05,
                    help="A round moving less than this means the pull has stopped winning.")
    ap.add_argument("--require-ring-open", action="store_true",
                    help="Fail an example whose ring could not be opened, instead of "
                         "recording ring_opened=false and carrying on.")
    ap.add_argument("--keep-added-atoms", action="store_true")
    ap.add_argument("--platform", default="CPU")
    args = ap.parse_args()

    out_path, pdb_dir = Path(args.out), Path(args.pdb_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Resume at example granularity: the ladder is cumulative, so an example is either
    # fully done (it wrote a ladder_complete row) or must be redone from the start.
    done = set()
    if out_path.exists():
        for lineno, line in enumerate(out_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # Refuse, never skip: a corrupt row still lets the aggregate compute, which
                # is worse than a missing one.
                raise SystemExit(f"FATAL: corrupt row {out_path}:{lineno}. Delete and rerun.")
            if r.get("ladder_complete") or r.get("status") == "failed":
                done.add(r["example_id"])
    if done:
        print(f"resuming: {len(done)} examples already complete", flush=True)

    sel = select_examples(args.metadata, args.per_type, args.shard, args.num_shards, args.seed)
    print(f"[ceiling] shard {args.shard}/{args.num_shards}: {len(sel)} complexes, "
          f"ladder {args.ladder_A}, k_contact={args.k_contact}", flush=True)

    forcefield = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    platform = Platform.getPlatformByName(args.platform)
    n_ok = 0
    with out_path.open("a") as fh:
        for _, mrow in sel.iterrows():
            example_id = str(mrow["example_id"])
            if example_id in done:
                continue
            cyc_label = CYC_LABEL[str(mrow["cyclization_type"])]
            src = Path(str(mrow["path"]))
            t0 = time.time()
            try:
                if not src.is_file():
                    raise FileNotFoundError(f"staged PDB missing: {src}")
                header, orig_chains, _ = read_staged_pdb(src)
                # No terminal mutation: pulling on CA needs no particular residue, which is
                # why this experiment uses the CA pair rather than the chemistry's anchors.
                # break_ring=True is the decyclization. Without it OpenMM re-creates the
                # terminal SG-SG on its own and a "disulfide" peptide stays cyclic, so the
                # pull measures a closed macrocycle stretching (~8 A) instead of an opening.
                fixer, _, removed_bonds = prepare_complex(src, None, "B", break_ring=True)
                # Verify rather than trust: confirm no first<->last bond survived. A silently
                # still-closed ring is the exact failure this flag exists to prevent, and it
                # looks like a real ceiling in the output.
                still_closed = terminal_bonds(fixer.topology, "B")
                ring_opened = not still_closed
                if still_closed and args.require_ring_open:
                    raise RuntimeError(
                        f"ring still closed after break_ring: "
                        f"{[f'{a.name}-{b.name}' for _, a, b in still_closed]}")
                ref = reference_state(fixer)
                ec = ExampleContext(fixer, PULL_KEY, args, ref, forcefield, platform)
                rows = open_ladder(ec, args, header, orig_chains, example_id,
                                   cyc_label, pdb_dir, ring_opened, removed_bonds)
            except Exception as exc:  # noqa: BLE001
                rows = [{"run_key": f"{example_id}|setup", "example_id": example_id,
                         "cyc_type": cyc_label, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"}]
                print(f"  FAIL {example_id}: {rows[0]['error']}", flush=True)
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if rows and rows[-1].get("ladder_complete"):
                n_ok += 1
                last = rows[-1]
                print(f"  {example_id} [{cyc_label}] {len(rows)} rungs in "
                      f"{time.time() - t0:.0f}s  start {rows[0]['terminal_ca_A']:.1f} A -> "
                      f"{last['terminal_ca_A']:.1f} A (asked {last['target_gap_A']:g}), "
                      f"retention {last['contact_retention']:.2f} [{last['limit_reason']}, "
                      f"ring_opened={last['ring_opened']}]", flush=True)

    n_attempted = sum(1 for _, m in sel.iterrows() if str(m["example_id"]) not in done)
    print(f"[ceiling] done: {n_ok} ladders -> {out_path}")
    if n_attempted and n_ok == 0:
        print(f"FATAL: all {n_attempted} examples failed -- writing no usable rows. Exiting "
              f"non-zero so the dependent stages do not run against an empty directory.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
