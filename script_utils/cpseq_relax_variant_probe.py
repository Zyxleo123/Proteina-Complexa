#!/usr/bin/env python3
"""Three relax protocols, judged on whether they can tell sequences apart.

The question is NOT which protocol agrees with the current one. `cpseq_shell_relax_probe`
already answered that (Spearman 0.65, shift +4.3 REU correlated with dG) and the answer
only matters if full relax is ground truth -- which it is not. Full relax deforms the
receptor 0.64-0.72 A internally, i.e. it rebuilds the target we are supposed to be
holding fixed.

The property a scoring protocol actually needs is DISCRIMINATION: the same sequence
folded at different seeds should land in the same place, and different sequences should
land in different places. Measured here as the ratio

    between-sequence sd  /  within-sequence (seed-to-seed) sd

A protocol that halves the spread between candidates but also halves the noise is no
worse; one that shifts every number by 5 REU and leaves the ratio alone costs nothing at
all, because every comparison this project makes is between candidates of the same
target. This is why the shift-based rejection of the shell variant does not settle the
binder-only question.

The three variants:

  full          what the scorer does today: every residue's backbone and side chains
                free, restrained to the input coordinates.
  binder_only   only the binder moves. The receptor is frozen outright -- no backbone
                minimisation and no side-chain repacking. Physically the strong claim:
                a real receptor does adapt its side chains to a new binder, so this
                should penalise exactly the designs that need accommodation.
  repack_shell  the middle, and standard Rosetta interface practice: the receptor's
                BACKBONE is frozen everywhere, so the target keeps its deposited fold,
                while receptor side chains inside the interface may repack and minimise.
                The binder is fully free.
  capped200     every degree of freedom stays free -- backbone and side chains, both
                chains -- and only the MINIMISER'S ITERATION BUDGET is capped, at 200.
                This is BindCraft's choice, and it is a different kind of economy from
                the three above: they remove degrees of freedom, which biases whatever
                those degrees of freedom were relieving, while this one truncates the
                converging tail and biases nothing systematically. BindCraft's own notes
                record that disabling backbone movement saves ~30 % and worsens metrics
                on average -- which is the same conclusion this probe reached for
                `repack_shell` (split-half rho 0.29 -> 0.05), arrived at independently.

Each variant is run on the SAME raw prediction, so the comparison is paired per
structure. CPU only; nothing is predicted.

Usage:
    python script_utils/cpseq_relax_variant_probe.py --out-dir $ZFS/cpseq/es_pilot_lnr \
        --sequences 8 --jobs 2 --out relax_variants.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCOREFUNCTION = "ref2015"
VARIANTS = ("full", "capped200", "binder_only", "repack_shell")


def _one(task: tuple) -> dict:
    pdb_path, shell_A, variants = task
    out: dict = {"pdb": pdb_path, "status": "unscored"}
    try:
        import pyrosetta
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            PreventRepacking,
            RestrictToRepacking,
        )
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax

        pyrosetta.init(
            "-mute all -ignore_unrecognized_res -load_PDB_components false", silent=True
        )
        pose = pyrosetta.pose_from_pdb(pdb_path)
        sf = pyrosetta.create_score_function(SCOREFUNCTION)
        n = pose.total_residue()
        chains = pose.split_by_chain()
        sizes = [(chains[i].total_residue(), i) for i in range(1, len(chains) + 1)]
        bchain = min(sizes)[1]
        binder = [i for i in range(1, n + 1) if pose.chain(i) == bchain]
        if len(binder) < 2 or len(binder) == n:
            out["status"] = "no_two_chain_interface"
            return out
        out |= {"n_residues": n, "n_binder": len(binder)}

        shell = set(binder)
        for i in range(1, n + 1):
            if i in shell:
                continue
            xi = pose.residue(i).nbr_atom_xyz()
            if any(xi.distance(pose.residue(j).nbr_atom_xyz()) <= shell_A for j in binder):
                shell.add(i)
        out["n_shell"] = len(shell)
        bset = set(binder)

        def dG(p) -> float:
            # Once per pose: the mover repacks the separated state and mutates what it
            # is given, so a second call reports a different number.
            m = InterfaceAnalyzerMover(1)
            m.set_pack_separated(True)
            m.set_compute_packstat(True)
            m.set_compute_interface_sc(True)
            m.apply(p)
            return float(p.scores["dG_separated"])

        def run(kind: str) -> tuple[float, float]:
            p = pyrosetta.Pose(pose)
            r = FastRelax(sf, 1)
            if kind == "capped200":
                # Nothing is frozen; the minimiser simply stops after 200 iterations.
                r.max_iter(200)
            elif kind != "full":
                mm = MoveMap()
                mm.set_bb(False)
                mm.set_chi(False)
                for i in binder:          # the binder is free in both restricted modes
                    mm.set_bb(i, True)
                    mm.set_chi(i, True)
                if kind == "repack_shell":
                    # Receptor backbone stays frozen; its interface side chains may move.
                    for i in shell - bset:
                        mm.set_chi(i, True)
                r.set_movemap(mm)

                # A MoveMap governs MINIMISATION only. Without a task factory the packer
                # still repacks the whole receptor, so `binder_only` would not be
                # binder-only -- it would be "receptor side chains repack but cannot then
                # be minimised", which is nobody's protocol.
                tf = TaskFactory()
                tf.push_back(RestrictToRepacking())      # never design
                prevent = PreventRepacking()
                allowed = bset if kind == "binder_only" else shell
                for i in range(1, n + 1):
                    if i not in allowed:
                        prevent.include_residue(i)
                tf.push_back(prevent)
                r.set_task_factory(tf)
            r.constrain_relax_to_start_coords(True)
            t = time.time()
            r.apply(p)
            return time.time() - t, dG(p)

        for v in variants:
            secs, val = run(v)
            out[f"{v}_s"] = round(secs, 1)
            out[f"{v}_dG"] = round(val, 3)
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def pick(out_dir: Path, n_seq: int, version: str | None) -> list[dict]:
    """Sequences with every seed folded, so within-sequence spread is measurable."""
    cohort = json.loads((out_dir / "cohort.json").read_text())
    picked = []
    for pid in cohort["cohort"]:
        p = out_dir / "state" / pid / "outcomes.jsonl"
        if not p.exists():
            continue
        by_seq: dict[str, dict] = defaultdict(dict)
        for line in p.read_text().splitlines():
            try:
                o = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            if o.get("stage") != "ladder" or o.get("technical_failure"):
                continue
            if version and o.get("scoring_version") != version:
                continue
            if not o.get("raw_pdb_path") or not Path(o["raw_pdb_path"]).exists():
                continue
            # One entry per seed; a re-scored fold appears twice and the structure is
            # the same file either way.
            by_seq[o["sequence"]][o["prediction_seed"]] = o
        depth = max((len(v) for v in by_seq.values()), default=0)
        full = {s: v for s, v in by_seq.items() if len(v) == depth and depth > 1}
        # The reference first, then the rest in journal order, so every complex
        # contributes its baseline.
        ref = next((s for s, v in full.items()
                    if any(r.get("arm") == "reference" for r in v.values())), None)
        order = ([ref] if ref else []) + [s for s in full if s != ref]
        for s in order[:n_seq]:
            for seed, row in sorted(full[s].items()):
                picked.append({"parent_id": pid, "sequence": s, "seed": seed,
                               "arm": row.get("arm"), "pdb": row["raw_pdb_path"]})
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", action="append", required=True)
    ap.add_argument("--sequences", type=int, default=8, help="sequences per complex")
    ap.add_argument("--complexes", type=int, default=2, help="complexes per run dir")
    ap.add_argument("--shell", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--scoring-version", default="v5-junction-restrained-relax")
    ap.add_argument("--variants", default="",
                    help="comma separated subset of " + ",".join(VARIANTS))
    ap.add_argument("--out", default="relax_variants.json")
    args = ap.parse_args()

    chosen = tuple(v.strip() for v in args.variants.split(",") if v.strip()) or VARIANTS
    bad = [v for v in chosen if v not in VARIANTS]
    if bad:
        print(f"[probe] unknown variant(s): {bad}")
        return 1

    items = []
    for d in args.out_dir:
        got = pick(Path(d), args.sequences, args.scoring_version or None)
        keep_pids = sorted({g["parent_id"] for g in got})[: args.complexes]
        got = [g for g in got if g["parent_id"] in keep_pids]
        print(f"[probe] {Path(d).name}: {len(got)} structure(s) over "
              f"{len(keep_pids)} complex(es)")
        items += got
    if not items:
        print("[probe] nothing to score")
        return 1

    print(f"[probe] {len(items)} structure(s) x {len(chosen)} variant(s), "
          f"{args.jobs} worker(s)")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = ex.map(_one, [(i["pdb"], args.shell, chosen) for i in items])
        for i, (meta, r) in enumerate(zip(items, results), 1):
            r |= {k: meta[k] for k in ("parent_id", "sequence", "seed", "arm")}
            rows.append(r)
            if r["status"] == "ok":
                print(f"  [{i}/{len(items)}] {meta['parent_id'][:16]:16s} "
                      + "  ".join(f"{v}={r[f'{v}_dG']:+8.2f} ({r[f'{v}_s']:5.1f}s)"
                                  for v in chosen))
            else:
                print(f"  [{i}/{len(items)}] {r['status']}")

    ok = [r for r in rows if r["status"] == "ok"]
    summary: dict = {"n": len(rows), "n_ok": len(ok), "shell_A": args.shell,
                     "wall_time_s": round(time.time() - t0, 1), "variants": {}}
    for v in chosen:
        by_seq: dict[tuple, list[float]] = defaultdict(list)
        for r in ok:
            by_seq[(r["parent_id"], r["sequence"])].append(r[f"{v}_dG"])
        # Within-sequence spread is pooled ACROSS complexes; between-sequence spread is
        # computed per complex and then pooled, because two targets having different
        # interface energies is not discrimination between candidates.
        within = [st.stdev(v_) for v_ in by_seq.values() if len(v_) > 1]
        per_parent: dict[str, list[float]] = defaultdict(list)
        for (pid, _s), v_ in by_seq.items():
            per_parent[pid].append(st.fmean(v_))
        between = [st.stdev(m) for m in per_parent.values() if len(m) > 1]
        w = st.fmean(within) if within else float("nan")
        b = st.fmean(between) if between else float("nan")
        summary["variants"][v] = {
            "n_sequences": len(by_seq),
            "within_sequence_sd": round(w, 3),
            "between_sequence_sd": round(b, 3),
            "discrimination": round(b / w, 3) if w else None,
            "median_s": round(st.median([r[f"{v}_s"] for r in ok]), 1),
            "median_dG": round(st.median([r[f"{v}_dG"] for r in ok]), 2),
        }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n=== discrimination: between-sequence sd / within-sequence (seed) sd ===")
    print(f"  {'variant':14s} {'within':>8s} {'between':>8s} {'ratio':>7s} "
          f"{'median s':>9s} {'median dG':>10s}")
    for v in chosen:
        s = summary["variants"][v]
        print(f"  {v:14s} {s['within_sequence_sd']:8.2f} {s['between_sequence_sd']:8.2f} "
              f"{str(s['discrimination']):>7s} {s['median_s']:9.1f} {s['median_dG']:10.2f}")
    print("\nHigher ratio = better able to rank candidates through the seed noise.")
    print(f"written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
