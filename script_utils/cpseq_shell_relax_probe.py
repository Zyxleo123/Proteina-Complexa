#!/usr/bin/env python3
"""Does restricting the relax to an interface shell change the number it reports?

The scorer's FastRelax moves every residue in the complex. Profiled on a 422-residue
complex that is 71-105 s of a ~74-105 s scoring step -- about 68 % of the whole pipeline
-- and nearly all of it is spent re-relaxing a receptor that is IDENTICAL for every
candidate of a target. A 10 A shell around the binder ran 8.7x faster on that one
structure for a +1.33 REU shift in `dG_separated`, which is inside the measured
relax-to-relax noise (median 1.18, max 6.40 REU).

One structure is an anecdote. This pairs full-relax against shell-relax on a sample, from
the SAME raw prediction, and reports the shift's distribution. The decision rule stated
in advance, so it cannot be fitted to the answer:

- adopt if the median |shift| is within the relax replicate noise (~1.2 REU) and the
  shift does not correlate with dG -- a constant offset is harmless to a comparison
  between candidates, a dG-dependent one is not, because it would change the RANKING
  rather than the scale;
- reject if it reorders candidates, which is what `rank_correlation` below measures.

Runs on CPU only, from retained raw structures. No GPU, no prediction.

Usage:
    python script_utils/cpseq_shell_relax_probe.py --out-dir $ZFS/cpseq/es_pilot_lnr \
        --n 25 --shell 10.0 --jobs 8 --out shell_probe_lnr.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCOREFUNCTION = "ref2015"


def _one(args: tuple) -> dict:
    """Full relax and shell relax of one structure, in a subprocess.

    PyRosetta is initialised per process: it holds global state, and a shared instance
    across a pool is how a parallel Rosetta script corrupts its own results.
    """
    pdb_path, shell_A = args
    out: dict = {"pdb": pdb_path, "shell_A": shell_A, "status": "unscored"}
    try:
        import pyrosetta
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax

        pyrosetta.init(
            "-mute all -ignore_unrecognized_res -load_PDB_components false", silent=True
        )
        pose = pyrosetta.pose_from_pdb(pdb_path)
        sf = pyrosetta.create_score_function(SCOREFUNCTION)
        n = pose.total_residue()
        out["n_residues"] = n

        chains = pose.split_by_chain()
        sizes = [(chains[i].total_residue(), i) for i in range(1, len(chains) + 1)]
        bchain = min(sizes)[1]
        binder = [i for i in range(1, n + 1) if pose.chain(i) == bchain]
        out["n_binder"] = len(binder)
        if len(binder) < 2 or len(binder) == n:
            out["status"] = "no_two_chain_interface"
            return out

        def dG(p) -> float:
            # Called ONCE per pose: the mover repacks the separated state and mutates the
            # pose it is given, so a second call on the same pose returns a different
            # number. A probe that called it twice is how the first version of this
            # measurement reported deltas that were not deltas.
            m = InterfaceAnalyzerMover(1)
            m.set_pack_separated(True)
            m.set_compute_packstat(True)
            m.set_compute_interface_sc(True)
            m.apply(p)
            return float(p.scores["dG_separated"])

        t = time.time()
        pf = pyrosetta.Pose(pose)
        r = FastRelax(sf, 1)
        r.constrain_relax_to_start_coords(True)
        r.apply(pf)
        out["full_s"] = round(time.time() - t, 1)
        out["full_dG"] = round(dG(pf), 3)

        keep = set(binder)
        for i in range(1, n + 1):
            if i in keep:
                continue
            xi = pose.residue(i).nbr_atom_xyz()
            if any(xi.distance(pose.residue(j).nbr_atom_xyz()) <= shell_A for j in binder):
                keep.add(i)
        out["n_movable"] = len(keep)

        mm = MoveMap()
        mm.set_bb(False)
        mm.set_chi(False)
        for i in keep:
            mm.set_bb(i, True)
            mm.set_chi(i, True)
        t = time.time()
        ps = pyrosetta.Pose(pose)
        r = FastRelax(sf, 1)
        r.set_movemap(mm)
        r.constrain_relax_to_start_coords(True)
        r.apply(ps)
        out["shell_s"] = round(time.time() - t, 1)
        out["shell_dG"] = round(dG(ps), 3)

        out["shift"] = round(out["shell_dG"] - out["full_dG"], 3)
        out["speedup"] = round(out["full_s"] / max(out["shell_s"], 1e-6), 2)
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Spearman rho, without scipy. The decisive number: ranking, not scale."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", action="append", required=True,
                    help="repeatable: a run directory whose retained raw structures to sample")
    ap.add_argument("--n", type=int, default=25, help="structures per run directory")
    ap.add_argument("--shell", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="shell_relax_probe.json")
    args = ap.parse_args()

    tasks = []
    for d in args.out_dir:
        pat = str(Path(d) / "predictions" / "*" / "*" / "boltz" / "**" / "*_model_*.pdb")
        found = [
            p for p in sorted(glob.glob(pat, recursive=True))
            if not p.endswith(("_refined.pdb", "_isolated_relaxed.pdb"))
        ]
        # Evenly spaced through the listing rather than the first N, so the sample is not
        # all one target's candidates.
        if len(found) > args.n:
            step = len(found) / args.n
            found = [found[int(i * step)] for i in range(args.n)]
        print(f"[probe] {d}: {len(found)} structure(s)")
        tasks += [(p, args.shell) for p in found]

    if not tasks:
        print("[probe] no retained raw structures found -- was this run at `full` retention?")
        return 1

    print(f"[probe] {len(tasks)} structure(s), shell {args.shell} A, {args.jobs} job(s)")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(_one, tasks), 1):
            rows.append(r)
            if r["status"] == "ok":
                print(f"  [{i}/{len(tasks)}] {r['n_residues']:4d} res "
                      f"({r['n_movable']:3d} movable)  full {r['full_s']:6.1f}s "
                      f"{r['full_dG']:+8.2f}  shell {r['shell_s']:5.1f}s {r['shell_dG']:+8.2f}  "
                      f"shift {r['shift']:+6.2f}  {r['speedup']:5.2f}x")
            else:
                print(f"  [{i}/{len(tasks)}] {r['status']}")

    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"n": len(rows), "n_ok": len(ok), "shell_A": args.shell,
               "wall_time_s": round(time.time() - t0, 1)}
    if ok:
        sh = [r["shift"] for r in ok]
        sp = [r["speedup"] for r in ok]
        summary |= {
            "shift_median": round(st.median(sh), 3),
            "shift_mean": round(st.fmean(sh), 3),
            "shift_sd": round(st.stdev(sh), 3) if len(sh) > 1 else None,
            "shift_abs_median": round(st.median([abs(x) for x in sh]), 3),
            "shift_min": min(sh), "shift_max": max(sh),
            "speedup_median": round(st.median(sp), 2),
            "full_s_median": round(st.median([r["full_s"] for r in ok]), 1),
            "shell_s_median": round(st.median([r["shell_s"] for r in ok]), 1),
            "rank_correlation_full_vs_shell": round(
                rank_correlation([r["full_dG"] for r in ok], [r["shell_dG"] for r in ok]), 4),
            # A shift that grows with dG changes the ranking; a constant one does not.
            "shift_vs_full_dG_correlation": round(
                rank_correlation([r["full_dG"] for r in ok], sh), 4),
        }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwritten to {Path(args.out).resolve()}")
    if ok:
        print("\nDecision rule, stated before the run:")
        print(f"  median |shift| {summary['shift_abs_median']} REU vs relax replicate "
              f"noise ~1.2 REU")
        print(f"  Spearman rho (full vs shell ranking) {summary['rank_correlation_full_vs_shell']}"
              f" -- near 1.0 means candidates are ordered the same way")
        print(f"  shift-vs-dG rho {summary['shift_vs_full_dG_correlation']}"
              f" -- near 0 means a constant offset, not a ranking change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
