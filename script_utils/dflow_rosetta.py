"""Paired Rosetta interface scoring for LP -> CP: the CP candidate minus its ORIGINAL LP.

Paired, because the absolute number is not the claim
---------------------------------------------------
`dG_separated` for a generated complex is not comparable across targets -- it scales with
interface size, and the receptors here range over ~90 to ~220 residues. The contract asks a
paired question: does cyclizing THIS peptide cost more than +5 REU against THIS peptide's own
linear original. So every candidate is scored alongside its own `input_complex_pdb` with
identical preparation, identical relaxation and identical score function, and the reported
quantity is the difference. REU are Rosetta Energy Units and are not kcal/mol.

Why this does its own scoring instead of calling `pyrosetta_interface_energy`
----------------------------------------------------------------------------
That function returns scores and discards the pose, so it cannot answer two things the
contract asks for:

    did the ring survive relaxation?   the anchor-atom distance is measured on the pose
                                       BEFORE and AFTER FastRelax
    peptide internal energy / strain   the binder alone is scored, bound geometry and
                                       relaxed, which is where a forced ring shows up

Its topology handling is reused verbatim (`_declare_cyclic_bond`, `_ensure_pyrosetta_init`,
`ROSETTA_INTERFACE_SCORE_KEYS`), so the interface number means the same thing it does
everywhere else in this repo.

The isopeptide topology gap, stated rather than papered over
-----------------------------------------------------------
`_declare_cyclic_bond` declares a mainchain amide (`declare_chemical_bond` after stripping
the terminus variants) and a disulfide (`add_disulfide_bond`), but for ISOPEPTIDE it returns
False: no residue-connection patch is wired up, so Rosetta scores a topology whose closing
bond it does not know about. That is not fixed here -- a half-correct side-chain conjugation
patch would produce numbers that look fine and are wrong.

What is done instead is to measure the consequence. `FastRelax` runs with
`constrain_relax_to_start_coords(True)`, so every atom is restrained to the input geometry
and an undeclared ring should not fly apart; `ring_dist_before_A` / `ring_dist_after_A` say
whether it actually did. Every row carries `topology_declared`, and the report keeps
isopeptide's energy tier separate from the other two rather than pooling a measurement whose
topology is known to be incomplete.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from proteinfoundation.evaluation.rosetta_energy import (  # noqa: E402
    ROSETTA_INTERFACE_SCORE_KEYS,
    _declare_cyclic_bond,
    _ensure_pyrosetta_init,
)

# Anchor atoms per chemistry, for the ring-survival measurement. Same choices as
# `scripts/dflow_pdb.ring_atom_names` and `per_sample_requested_bond_distance`.
_ACID_ATOM = {"ASP": "CG", "ASN": "CG", "GLU": "CD", "GLN": "CD"}


def _ring_atoms(pose, binder_chain: str, cyc_type: str, i_local: int, j_local: int):
    """(seqpos_i, atom_i, seqpos_j, atom_j) for the closing bond, or None if undefinable."""
    info = pose.pdb_info()
    si = info.pdb2pose(binder_chain, i_local + 1)
    sj = info.pdb2pose(binder_chain, j_local + 1)
    if si == 0 or sj == 0:
        return None
    ni, nj = pose.residue(si).name3(), pose.residue(sj).name3()
    if cyc_type == "mainchain":
        return si, "N", sj, "C"
    if cyc_type == "disulfide":
        return (si, "SG", sj, "SG") if (ni == "CYS" and nj == "CYS") else None
    if cyc_type == "isopeptide":
        if ni == "LYS" and nj in _ACID_ATOM:
            return si, "NZ", sj, _ACID_ATOM[nj]
        if nj == "LYS" and ni in _ACID_ATOM:
            return si, _ACID_ATOM[ni], sj, "NZ"
        return None
    return None


def _ring_dist(pose, spec) -> float:
    if spec is None:
        return math.nan
    si, ai, sj, aj = spec
    try:
        return float(pose.residue(si).xyz(ai).distance(pose.residue(sj).xyz(aj)))
    except Exception:  # noqa: BLE001
        return math.nan


def score_complex(
    pdb_path: str,
    binder_chain: str,
    target_chains: list[str],
    *,
    cyc_type: str | None = None,
    cyc_i: int | None = None,
    cyc_j: int | None = None,
    relax: bool = True,
) -> dict:
    """Interface + internal-energy terms for one complex, with ring survival measured.

    `cyc_type=None` is the LINEAR baseline path: no bond is declared, which is correct --
    the original LP has no ring. The ring columns come back NaN there by construction, not
    by failure.
    """
    _ensure_pyrosetta_init()
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.relax import FastRelax

    out: dict = {"pdb": str(pdb_path), "relaxed": bool(relax)}
    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    sf = pyrosetta.create_score_function("ref2015")

    ring = None
    if cyc_type is not None:
        if cyc_i is None or cyc_j is None:
            # Not defaulted to the termini: a silently-wrong endpoint would be scored as a
            # ring that opened, which reads as a model failure rather than a caller bug.
            raise ValueError(
                f"cyc_type={cyc_type!r} needs explicit cyc_i/cyc_j (got {cyc_i}, {cyc_j})"
            )
        ring = _ring_atoms(pose, binder_chain, cyc_type, int(cyc_i), int(cyc_j))
    out["ring_atoms_resolved"] = ring is not None

    declared = None
    if cyc_type is not None:
        try:
            declared = bool(
                _declare_cyclic_bond(pose, binder_chain, cyc_type, int(cyc_i), int(cyc_j))
            )
        except (ValueError, RuntimeError) as exc:
            declared = False
            out["declare_error"] = f"{type(exc).__name__}: {exc}"
    out["topology_declared"] = declared

    def analyze(p) -> dict:
        mover = InterfaceAnalyzerMover("".join([binder_chain]) + "_" + "".join(target_chains))
        mover.set_pack_separated(True)
        mover.apply(p)
        return {k: float(p.scores[k]) for k in ROSETTA_INTERFACE_SCORE_KEYS if k in p.scores}

    # The binder alone, in its BOUND geometry: this is the internal energy a forced ring
    # shows up in, and it must be taken before relax touches anything.
    def binder_only(p):
        import pyrosetta as pr
        from pyrosetta.rosetta.protocols.grafting import return_region

        info = p.pdb_info()
        pos = [s for s in range(1, p.size() + 1) if info.chain(s) == binder_chain]
        if not pos:
            return math.nan
        try:
            sub = return_region(pr.Pose(p), min(pos), max(pos))
            return float(sf(sub))
        except Exception:  # noqa: BLE001
            return math.nan

    out["ring_dist_before_A"] = _ring_dist(pose, ring)
    out["total_score_norelax"] = float(sf(pose))
    out["binder_internal_norelax"] = binder_only(pose)
    out["dG_separated_norelax"] = analyze(pyrosetta.Pose(pose)).get("dG_separated", math.nan)

    if relax:
        fast_relax = FastRelax(sf, 1)
        # Restrained to the input coordinates: this scores the geometry the model produced,
        # it does not search for a better one nearby. It is also what keeps an undeclared
        # isopeptide ring from simply opening during relaxation.
        fast_relax.constrain_relax_to_start_coords(True)
        fast_relax.apply(pose)
        out["total_score_relaxed"] = float(sf(pose))
        out["binder_internal_relaxed"] = binder_only(pose)
    out["ring_dist_after_A"] = _ring_dist(pose, ring)

    out.update(analyze(pose))
    if "dG_separated" not in out:
        raise KeyError(f"dG_separated missing for {pdb_path}")
    return out


def _load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Refuse, never skip: a concurrently-appended shard file has been
                # NUL-corrupted on this cluster before, and silently dropping the bad lines
                # would shrink the denominator without saying so.
                raise SystemExit(f"FATAL: corrupt row {path}:{lineno}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True,
                    help="GPU-stage JSONL file(s) carrying `complex_pdb`.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=1,
                    help="Relaxation/scoring repeats per structure. FastRelax is stochastic, "
                         "so a single number carries unknown spread; the contract asks for 3 "
                         "on shortlisted candidates, with MATCHING repeats on the LP baseline.")
    ap.add_argument("--only-passing", action="store_true",
                    help="Score only candidates that already passed every structural gate. "
                         "The energy tier is a conjunction with them, so scoring a candidate "
                         "that failed pose or sequence buys nothing.")
    ap.add_argument("--max-per-example", type=int, default=0,
                    help="Cap candidates per (example, chemistry); 0 = no cap. Ranked by the "
                         "optimizer's own objective, so the cap keeps the best rather than "
                         "whichever restart happened to finish first.")
    ap.add_argument("--stage", default="opt", choices=["opt", "init", "both"],
                    help="Which stage's candidates to score. `both` is what makes the "
                         "paired did-optimization-help comparison possible on energy too.")
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args()

    rows = _load_rows([Path(p) for p in args.rows])
    cands = [r for r in rows if r.get("status") == "ok" and r.get("complex_pdb")]
    if args.stage != "both":
        cands = [r for r in cands if r.get("stage") == args.stage]
    if args.only_passing:
        cands = [r for r in cands if r.get("gate_joint_structural")]
    if args.max_per_example:
        by: dict = {}
        for r in cands:
            by.setdefault((r["example_id"], r["cyc_type_name"], r.get("stage")), []).append(r)
        cands = []
        for key, group in by.items():
            group.sort(key=lambda r: r.get("objective", float("inf")))
            cands.extend(group[: args.max_per_example])
    cands.sort(key=lambda r: str(r.get("run_key")))
    mine = cands[args.shard_index :: args.shard_count]
    print(f"{len(rows)} rows -> {len(cands)} candidates; shard {args.shard_index}/"
          f"{args.shard_count} takes {len(mine)}", flush=True)
    if not mine:
        # Exit 0: an empty shard is a legitimate outcome of sharding a small candidate set,
        # and a non-zero exit would strand the report job on DependencyNeverSatisfied.
        print("nothing to score in this shard", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).touch()
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["score_key"])
    if done:
        print(f"resuming: {len(done)} scored", flush=True)

    # The LP baseline is one file per example, shared by every candidate of that example, so
    # it is scored once per (example, repeat) and reused. With --repeats 3 that is 3 LP
    # scorings, which is the matching baseline the contract asks for.
    lp_cache: dict = {}
    t0 = time.time()
    n = 0
    with out_path.open("a") as fh:
        for r in mine:
            chains = [c for c in str(r.get("target_chains", "A")).split(",") if c]
            bchain = str(r.get("binder_chain", "B"))
            cyc = str(r["cyc_type_name"])
            ci, cj = int(r.get("cyc_i", 0)), int(r.get("cyc_j", r.get("peptide_length", 1) - 1))
            for rep in range(int(args.repeats)):
                key = f"{r['run_key']}|rep{rep}"
                if key in done:
                    continue
                row = {"score_key": key, "run_key": r["run_key"], "repeat": rep,
                       "example_id": r["example_id"], "cyc_type_name": cyc,
                       "stage": r.get("stage"), "restart": r.get("restart"),
                       "init_mode": r.get("init_mode"), "objective_tag": r.get("objective_tag"),
                       "arm": r.get("arm", "dflow")}
                try:
                    cp = score_complex(r["complex_pdb"], bchain, chains, cyc_type=cyc,
                                       cyc_i=ci, cyc_j=cj, relax=not args.no_relax)
                except Exception as exc:  # noqa: BLE001
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"  FAIL {key}: {type(exc).__name__}: {exc}", flush=True)
                    continue

                lp_key = (r.get("input_complex_pdb"), rep)
                if lp_key not in lp_cache:
                    try:
                        lp_cache[lp_key] = score_complex(
                            r["input_complex_pdb"], bchain, chains,
                            cyc_type=None, relax=not args.no_relax)
                    except Exception as exc:  # noqa: BLE001
                        lp_cache[lp_key] = {"error": f"{type(exc).__name__}: {exc}"}
                lp = lp_cache[lp_key]

                row.update({f"cp/{k}": v for k, v in cp.items()})
                row.update({f"lp/{k}": v for k, v in lp.items()})
                if "dG_separated" in lp:
                    row["ddg_REU"] = cp["dG_separated"] - lp["dG_separated"]
                    row["ddg_norelax_REU"] = (
                        cp.get("dG_separated_norelax", math.nan)
                        - lp.get("dG_separated_norelax", math.nan)
                    )
                    row["d_binder_internal_relaxed"] = (
                        cp.get("binder_internal_relaxed", math.nan)
                        - lp.get("binder_internal_relaxed", math.nan)
                    )
                    row["status"] = "ok"
                else:
                    row["status"] = "lp_failed"
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                if n % 10 == 0:
                    print(f"  {n} scorings ({time.time() - t0:.0f}s)", flush=True)
                if args.max_seconds and (time.time() - t0) > args.max_seconds:
                    print(f"  budget reached after {n} scorings; resubmit to continue",
                          flush=True)
                    print(f"done: {n} scorings, {time.time() - t0:.0f}s", flush=True)
                    return
    print(f"done: {n} scorings, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
