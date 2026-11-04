#!/usr/bin/env python3
"""Score AF3-folded cyclic-peptide complexes with the shared evaluator. Emits dG rows.

This is the CPU half of the AF3 arm. It takes AF3 model mmCIFs, rebuilds each one as a
peptide-chain PDB (`adapters.af3.write_complex_pdb`, whose atom mapping is asserted on
names AND bond lengths before use), applies the frozen acceptance gate, and runs the same
Rosetta protocol every other arm in this project runs.

Nothing here is AF3-specific after the conversion. That is deliberate and it is the only
thing that makes an AF3 number comparable to a Boltz-2 one: same scorefunction (ref2015),
same relax budget, same tolerances, same protonation policy, same chemistry. The predictor
only decides which coordinates go in.

Energies are **Rosetta ref2015 REU**, recorded as such on every row. They are not measured
binding free energies and nothing here converts them into one.

A native/crystal structure can be scored in the same call (`--native`). That is worth
doing on every control: a dG is only interpretable next to the dG the same protocol gives
the deposited complex, and this project has already been bitten by an evaluator whose
absolute numbers looked fine and whose native check it had never run.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import time
from pathlib import Path

# PyRosetta is a C++ extension and a fault inside it exits with a bare "Segmentation
# fault (core dumped)" and no Python stack, which says nothing about which structure or
# which call. This prints a Python-level traceback at the faulting frame.
faulthandler.enable()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import tolerances as tolmod  # noqa: E402
from proteinfoundation.cpseq.adapters.af3 import (  # noqa: E402
    AF3_REV,
    DEFAULT_REPRESENTATION,
    REPRESENTATIONS,
    AF3MappingError,
    write_complex_pdb,
)
from proteinfoundation.cpseq.evaluate import (  # noqa: E402
    ENERGY_UNITS,
    SCORING_PROTOCOL_VERSION,
    scoring_behaviour,
)
from proteinfoundation.cpseq.geometry import validity_report  # noqa: E402
from proteinfoundation.cpseq.scorer import score_structure  # noqa: E402
from proteinfoundation.cpseq.smiles import cyclic_peptide  # noqa: E402

PEPTIDE_CHAIN = "B"


def score_one(
    *,
    label: str,
    pdb_path: Path,
    sequence: str,
    receptor_chains: list[str],
    chemistry: str,
    tol,
    behaviour: dict,
    scoring_version: str,
    source: str,
    peptide_chain: str = PEPTIDE_CHAIN,
) -> dict:
    """One scored complex. Returns a row; records a failure rather than raising."""
    t0 = time.time()
    row: dict = {
        "label": label,
        "source": source,
        "sequence": sequence,
        "chemistry": chemistry,
        "pdb_path": str(pdb_path),
        "peptide_chain": peptide_chain,
        "scoring_version": scoring_version,
        "validity_stage": behaviour["validity_stage"],
        "relax_restraint": behaviour["relax_restraint"],
        "energy_units": ENERGY_UNITS,
    }
    stem = pdb_path.with_suffix("")
    try:
        raw = validity_report(
            pdb_path, binder_chain=peptide_chain, receptor_chains=receptor_chains,
            chemistry=chemistry, requested_sequence=sequence,
        )
        raw_verdict = tolmod.verdict(raw, tol)
        sc = score_structure(
            pdb_path, binder_chain=peptide_chain, receptor_chains=receptor_chains,
            chemistry=chemistry, requested_sequence=sequence,
            refined_out_path=f"{stem}_refined.pdb",
            isolated_out_path=f"{stem}_isolated_relaxed.pdb",
            relax_restraint=behaviour["relax_restraint"],
        )
    except Exception as exc:  # noqa: BLE001
        row |= {"status": "scoring_failed", "error": f"{type(exc).__name__}: {exc}",
                "valid": False, "wall_time_s": round(time.time() - t0, 2)}
        return row

    # Under v3 the label comes from the RAW prediction and the energy from the relaxed
    # structure. That pairing is v3's known defect, not an accident of this script, and
    # it is why both verdicts are carried on the row.
    post_verdict = None
    refined = Path(f"{stem}_refined.pdb")
    if refined.exists():
        try:
            post = validity_report(
                refined, binder_chain=peptide_chain, receptor_chains=receptor_chains,
                chemistry=chemistry, requested_sequence=sequence,
            )
            post_verdict = tolmod.verdict(post, tol)
        except Exception:  # noqa: BLE001
            post_verdict = None

    chosen = raw_verdict if behaviour["validity_stage"] == "prerelax" else (
        post_verdict or {"valid": False, "reasons": ["no refined structure"]})

    # The energies are NESTED under `raw` and `refined`; they are not top-level keys.
    # Reading them from the top level returned None on every fold while `status` said
    # "ok" -- a zero-information row rather than an error, which is the shape of bug that
    # silently empties a results table.
    #
    # `refined` is the reported energy (README_CPSEQ_MPNN_FILL_SETTINGS sec. 2), and a
    # candidate's value is the median of it over that candidate's valid folds (sec. 3).
    # The RAW figure is carried only to show what the relax relieved; it is not a binding
    # proxy and must never be quoted as one. Measured on 5TU6: raw +759 REU against
    # -30.5 refined for the same AF3 structure, and the deposited CRYSTAL is +55 raw
    # against -27.5 refined. An unrelaxed interface energy is uninterpretable even for a
    # crystal, so the two are kept under names that cannot be mistaken for each other.
    raw_block = sc.get("raw") or {}
    refined_block = sc.get("refined") or {}
    displacement = sc.get("refinement_displacement") or {}

    row |= {
        "status": sc.get("status", "ok"),
        "scoring_reason": sc.get("reason"),
        "valid": bool(chosen.get("valid")),
        "reasons": chosen.get("reasons", []),
        "prerelax_valid": bool(raw_verdict.get("valid")),
        "prerelax_reasons": raw_verdict.get("reasons", []),
        "postrelax_valid": (None if post_verdict is None
                            else bool(post_verdict.get("valid"))),
        "raw_closure_bond_A": raw.get("closure", {}).get("bond_length_A"),
        "raw_dihedral_deg": raw.get("closure", {}).get("dihedral_deg"),
        "raw_dihedral_arrangement": raw.get("closure", {}).get("dihedral_arrangement"),
        # --- the reported energy -------------------------------------------------
        "refined_dG_separated": refined_block.get("dG_separated"),
        # dG per 1000 A^2 of buried interface. Carried because a larger interface buys a
        # better absolute dG for free: on 5TU6 the AF3 fold beat the native on absolute
        # dG (-30.5 vs -25.4) while LOSING on this (-2.26 vs -2.49), having buried ~330
        # A^2 more surface.
        "refined_dG_per_1000A2": refined_block.get("dG_separated_per_1000A2"),
        "refined_dSASA_int": refined_block.get("dSASA_int"),
        "refined_total_score": refined_block.get("total_score"),
        "refined_packstat": refined_block.get("packstat"),
        "refined_sc_value": refined_block.get("sc_value"),
        # --- pre-relax, for provenance only, NOT a binding number ----------------
        "unrelaxed_dG_separated": raw_block.get("dG_separated"),
        "unrelaxed_dSASA_int": raw_block.get("dSASA_int"),
        "unrelaxed_total_score": raw_block.get("total_score"),
        # How far the restrained relax moved each side. `receptor_ca_rmsd_A` is the
        # target binding-site displacement: AF3 predicts the complex jointly and a
        # template is a feature rather than a restraint, so the receptor is genuinely
        # free to move, and how far it moved is part of reading the result.
        "binder_ca_rmsd_A": displacement.get("binder_ca_rmsd_A"),
        "receptor_ca_rmsd_A": displacement.get("receptor_ca_rmsd_A"),
        "relax_total_score_delta": displacement.get("total_score_delta"),
        "refined_pdb_path": sc.get("refined_pdb_path"),
        "wall_time_s": round(time.time() - t0, 2),
    }
    return row


def _score_worker(task: dict) -> dict:
    """Score one structure in its own process. Returns the row.

    A separate process per fold, not threads: the work is inside PyRosetta's C++ and
    holds the GIL, so threads would serialise. Each worker loads the tolerances and
    initialises PyRosetta itself -- `_ensure_pyrosetta_init` is per-process state, and
    inheriting a half-initialised Rosetta across a fork is a way to get a segfault with
    no traceback.
    """
    import faulthandler
    faulthandler.enable()
    from proteinfoundation.cpseq import tolerances as _tol

    tol = _tol.load(Path(task["tolerances"]))[task["chemistry"]]
    row = score_one(
        label=task["label"], pdb_path=Path(task["pdb_path"]), sequence=task["sequence"],
        receptor_chains=task["receptor_chains"], chemistry=task["chemistry"], tol=tol,
        behaviour=task["behaviour"], scoring_version=task["scoring_version"],
        source=task["source"], peptide_chain=task["peptide_chain"],
    )
    for key in ("af3_closure_bond_A",):
        if key in task:
            row[key] = task[key]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif", action="append", default=[],
                    help="repeatable. `<label>=<path to AF3 model.cif>`")
    ap.add_argument("--sequence", required=True,
                    help="the peptide sequence that was folded")
    ap.add_argument("--receptor-chains", default="A",
                    help="comma-separated receptor chain ids in the output (default A)")
    ap.add_argument("--chemistry", default="mainchain")
    ap.add_argument("--require-closure", action="store_true",
                    help="skip relax+score for folds whose RAW closure bond is outside "
                         "the amide window, journaling them as `open_ring_unscored` with "
                         "the measured distance. Not an optimisation: restraining a bond "
                         "that is not there drags the complex apart (receptor CA moved "
                         "28-65 A, energies to +49,500 REU) and under v5 the forced "
                         "closure is then certified valid -- 12/19 on folds that were "
                         "0/20 closed. The fold is still REPORTED, never dropped.")
    ap.add_argument("--closure-window", default="1.23,1.43",
                    help="amide closure window in Angstrom for --require-closure")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel relax+score processes. Rosetta is the expensive half "
                         "(~110 s/fold measured) and each fold is independent, so this "
                         "is the main lever on wall time. Keep it at or below the task's "
                         "--cpus-per-task; PyRosetta is single-threaded per process.")
    ap.add_argument("--representation", default=DEFAULT_REPRESENTATION,
                    choices=list(REPRESENTATIONS),
                    help="must match the representation the folds were PRODUCED with; "
                         "the converter reads the output differently for each")
    ap.add_argument("--native", default=None,
                    help="optional deposited/crystal PDB of the same complex, scored by "
                         "the identical protocol as the reference the dG is read against")
    ap.add_argument("--native-peptide-chain", default="B",
                    help="peptide chain id in the native file, if not B")
    ap.add_argument("--native-receptor-chains", default=None)
    ap.add_argument("--tolerances", type=Path, required=True)
    ap.add_argument("--scoring-version", default=SCORING_PROTOCOL_VERSION)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="where converted PDBs and relaxed structures are written")
    args = ap.parse_args()

    behaviour = scoring_behaviour(args.scoring_version)
    tol_set = tolmod.load(args.tolerances)
    tol = tol_set[args.chemistry]
    receptor_chains = [c.strip() for c in args.receptor_chains.split(",") if c.strip()]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== cpseq_af3_score ===")
    print(f"scoring version   {args.scoring_version}")
    print(f"  validity judged on the {behaviour['validity_stage']} structure")
    print(f"  relax restrains  {behaviour['relax_restraint']}")
    print(f"tolerances        {args.tolerances}")
    print(f"  {args.chemistry}: bond [{tol.bond_length_min_A}, {tol.bond_length_max_A}] A, "
          f"planarity {tol.torsion_tol_deg} deg, angles {tol.angle_tol_deg} deg")
    print(f"af3 revision      {AF3_REV}")
    print(f"sequence          {args.sequence} (L={len(args.sequence)})")
    print(f"energy units      {ENERGY_UNITS}")

    peptide = cyclic_peptide(args.sequence, chemistry=args.chemistry)
    # Append-as-you-go, and resume on the label. Writing only at the end means one crash
    # discards every finished row: the native 5TU6 crystal segfaults PyRosetta (its
    # HOH/MG/DST hetero groups make Rosetta build a third chain against a 2-partner
    # interface spec), and that took a completed AF3 row with it. Rosetta minutes are too
    # expensive to recompute because a later structure failed.
    rows: list[dict] = []
    tasks: list[dict] = []
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["label"])
            except Exception:  # noqa: BLE001, S112
                continue
        if done:
            print(f"[score] resuming: {len(done)} row(s) already in {args.out} "
                  f"({sorted(done)})")
    journal = args.out.open("a")

    def emit(row: dict) -> None:
        journal.write(json.dumps(row) + "\n")
        journal.flush()
        rows.append(row)

    for spec in args.cif:
        if "=" not in spec:
            print(f"[score] SKIP malformed --cif {spec!r}; want <label>=<path>")
            continue
        label, path = spec.split("=", 1)
        if label in done:
            print(f"[score] {label}: already journaled, skipping (Rosetta minutes)")
            continue
        cif = Path(path)
        if not cif.exists():
            print(f"[score] SKIP {label}: {cif} missing")
            continue
        converted = args.work_dir / f"{label}.pdb"
        try:
            conv = write_complex_pdb(cif_path=cif, peptide=peptide, out_path=converted,
                                     representation=args.representation)
        except AF3MappingError as exc:
            print(f"[score] {label}: MAPPING REFUSED -- {exc}")
            emit({"label": label, "source": "af3", "status": "mapping_refused",
                  "error": str(exc), "valid": False})
            continue
        print(f"[score] {label}: converted, AF3 closure {conv['af3_closure_bond_A']} A, "
              f"{conv['n_peptide_atoms']} peptide / {conv['n_receptor_atoms']} receptor atoms")
        if args.require_closure:
            lo, hi = (float(x) for x in args.closure_window.split(","))
            gap = conv["af3_closure_bond_A"]
            if not (lo <= gap <= hi):
                print(f"[score] {label}: ring OPEN at {gap} A (window {lo}-{hi}); "
                      f"journaled unscored -- relaxing it would force the ring shut and "
                      f"the energy would describe the forced structure")
                emit({"label": label, "source": "af3",
                      "status": "open_ring_unscored", "valid": False,
                      "sequence": args.sequence,
                      "representation": args.representation,
                      "raw_closure_bond_A": gap,
                      "closure_window_A": [lo, hi],
                      "scoring_version": args.scoring_version,
                      "reasons": [f"raw closure {gap} A outside {lo}-{hi} A"]})
                continue
        tasks.append({
            "label": label, "pdb_path": str(converted), "sequence": args.sequence,
            "receptor_chains": conv["receptor_chains"], "chemistry": args.chemistry,
            "behaviour": behaviour, "scoring_version": args.scoring_version,
            "source": "af3", "peptide_chain": PEPTIDE_CHAIN,
            "tolerances": str(args.tolerances),
            "af3_closure_bond_A": conv["af3_closure_bond_A"],
        })

    if args.native and "native" in done:
        print("[score] native: already journaled, skipping")
    elif args.native:
        native = Path(args.native)
        nat_rec = [c.strip() for c in
                   (args.native_receptor_chains or args.receptor_chains).split(",")
                   if c.strip()]
        tasks.append({
            "label": "native", "pdb_path": str(native), "sequence": args.sequence,
            "receptor_chains": nat_rec, "chemistry": args.chemistry,
            "behaviour": behaviour, "scoring_version": args.scoring_version,
            "source": "crystal",
            # The deposited file uses its own chain ids; only the converted AF3 output is
            # normalised to chain B.
            "peptide_chain": args.native_peptide_chain,
            "tolerances": str(args.tolerances),
        })
        print(f"[score] native: queued "
              f"refined_dG={row.get('refined_dG_separated')} REU")

    if tasks:
        print(f"\n[score] relax+score {len(tasks)} structure(s) on {args.workers} worker(s)")
        if args.workers <= 1:
            for task in tasks:
                row = _score_worker(task)
                emit(row)
                print(f"[score] {row['label']}: valid={row.get('valid')} "
                      f"refined_dG={row.get('refined_dG_separated')} REU "
                      f"({row.get('wall_time_s')}s)")
        else:
            # `spawn`, not the default fork: a forked child inherits whatever Rosetta
            # state the parent has, and a half-initialised Rosetta is how this pipeline
            # already produced one bare segfault with no traceback. The parent is the
            # ONLY writer -- rows come back through imap_unordered and are journaled
            # here, so concurrent O_APPEND (which NUL-corrupts rows on this filesystem)
            # never happens.
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=args.workers) as pool:
                for row in pool.imap_unordered(_score_worker, tasks):
                    emit(row)
                    print(f"[score] {row['label']}: valid={row.get('valid')} "
                          f"refined_dG={row.get('refined_dG_separated')} REU "
                          f"({row.get('wall_time_s')}s)", flush=True)

    journal.close()

    def num(value, width: int, places: int = 2) -> str:
        return (f"{value:{width}.{places}f}" if isinstance(value, (int, float))
                else f"{'-':>{width}}")

    # `refined dG` first and widest, because it is the reported quantity. The unrelaxed
    # column is last and labelled `unrelaxed`, never `dG`: it is provenance, and on these
    # structures it runs +55 to +760 REU where the refined figure is around -25.
    print(f"\n{'label':16s} {'source':8s} {'valid':6s} {'bond A':>7s} "
          f"{'refined dG':>11s} {'/1000A2':>8s} {'dSASA':>8s} "
          f"{'recCA A':>8s} {'unrelaxed':>10s}  reasons")
    for row in rows:
        print(f"{row.get('label','?'):16s} {row.get('source','?'):8s} "
              f"{str(row.get('valid')):6s} "
              f"{num(row.get('raw_closure_bond_A'), 7, 3)} "
              f"{num(row.get('refined_dG_separated'), 11)} "
              f"{num(row.get('refined_dG_per_1000A2'), 8)} "
              f"{num(row.get('refined_dSASA_int'), 8, 1)} "
              f"{num(row.get('receptor_ca_rmsd_A'), 8)} "
              f"{num(row.get('unrelaxed_dG_separated'), 10, 1)}"
              f"  {','.join(row.get('reasons') or [])[:50]}")
    print(f"\n[score] rows -> {args.out}")
    print("[score] `refined dG` is the reported energy; `unrelaxed` is provenance only.")
    print("[score] energies are Rosetta ref2015 REU, not binding free energies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
