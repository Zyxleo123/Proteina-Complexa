"""Measure the scoring half of a cpseq fold, and the two levers that cut it.

The companion to `cpseq_fold_cost_probe.py`, which measures the GPU half. Together they
account for a run's wall time: on the 22,642 v3 folds of `llm_ladder`, 37.8 s median of
Boltz-2 prediction against 29.6 s median of Rosetta scoring, so scoring is 54 % of that
run's 327 job-hours and is the half nobody has profiled.

Three questions, one job, because they need the same structures.

COST. Where the 29.6 s goes. A fold's scoring is: `validity_report` on the structure as
predicted (cheap -- the file's own measurement is 1.5 s without packstat, 2.2 s with),
then `score_structure`, which is an InterfaceAnalyzer pass on the raw pose and then TWO
FastRelax runs -- the restrained complex, and the free peptide inside the deformation
proxy. Only the relaxes should be expensive. This measures each separately instead of
assuming it.

LEVER 1: THE SKIP. Under a prerelax-gated protocol -- v3 is one,
`scoring_behaviour('v3-cis-amide-muco-windows')['validity_stage'] == 'prerelax'` -- the
verdict is read off the structure the predictor produced, so a fold that fails it is
already decided and the refined energy can never reach the gate or
`best_valid_dG_separated`. 60.9 % of `llm_ladder`'s v3 folds are invalid and 58.2 % of
its scoring hours went to relaxing them. `score_structure(skip_relax=True)` is the new
path; this times it against the full one on the SAME structures and checks that the
verdict it returns is identical, which is the only thing that makes the skip admissible.

LEVER 2: THE POOL. FastRelax is single-threaded, and the shard axis is the PARENT list,
so a task holding one complex scores strictly serially however many CPUs it was given.
This measures throughput at 1, 2, 4 and 8 worker processes to find where it stops
scaling -- PyRosetta is ~1.5 GB per worker on a 422-residue pose, so memory, not cores,
may be the binding constraint.

Read-only on the run directory. Structures are copied out and decompressed in the work
directory, and every output the scorer writes is redirected there, so a production
prediction tree is never touched -- which also means this can run against a live run.

Results go to JSONL so the summary and any figure can be redrawn without re-scoring.

Run as a job, never inline: PyRosetta's working set is well above the login shell's cap
and the failure is a bare SIGKILL with no traceback.
"""

from __future__ import annotations

import argparse
import faulthandler
import gzip
import json
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

# PyRosetta is a C++ extension and does not always fail as a Python exception: Rosetta's
# own `utility_exit` terminates the process, and a segfault in the scorer takes the
# interpreter with it. Either way `try/except` never runs and the log ends mid-arm with
# no traceback -- which is exactly how run 95740 died after 16 minutes, having printed
# nothing but its banner. faulthandler turns both into a stack trace on stderr.
faulthandler.enable()

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.cpseq import tolerances as tolmod  # noqa: E402
from proteinfoundation.cpseq.evaluate import (  # noqa: E402
    BINDER_CHAIN,
    scoring_behaviour,
)
from proteinfoundation.cpseq.geometry import validity_report  # noqa: E402
from proteinfoundation.cpseq.scorer import score_structure  # noqa: E402


def _receptor_chains(pdb: Path) -> list[str]:
    """Every chain in the file except the binder.

    Read from the structure rather than from a manifest so the probe needs no cohort,
    no target context and no dataset directory -- it can be pointed at any predicted
    complex this project has written.
    """
    seen: list[str] = []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            ch = line[21]
            if ch != BINDER_CHAIN and ch not in seen:
                seen.append(ch)
    return seen


def collect_folds(run_dir: Path, n: int, seed: int) -> list[dict]:
    """Sample folds from a run's journals, spread across its complexes.

    Spread matters: receptor size drives both the relax cost and the memory, so a sample
    drawn from one complex would measure that complex rather than the run. Takes the same
    number from each parent, in journal order, then shuffles.
    """
    per_parent: dict[str, list[dict]] = {}
    for jf in sorted((run_dir / "state").glob("*/outcomes.jsonl")):
        rows = []
        for ln in jf.read_text(errors="replace").splitlines():
            ln = ln.strip()
            if not ln or ln[0] != "{":
                continue
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if r.get("status") != "scored" or not r.get("raw_pdb_path"):
                continue
            rows.append(r)
        if rows:
            per_parent[jf.parent.name] = rows

    if not per_parent:
        return []
    take = max(1, n // len(per_parent))
    picked: list[dict] = []
    for _pid, rows in sorted(per_parent.items()):
        picked.extend(rows[:take])
    random.Random(seed).shuffle(picked)
    return picked[:n]


def stage_structure(row: dict, work: Path) -> Path | None:
    """Copy one fold's raw structure into the work dir, decompressing if needed.

    `rescorable` retention gzips the raw complex, so the path the journal recorded does
    not exist until it is restored. Restoring it IN PLACE would write into the run's
    prediction tree; this copies instead, which is what makes the probe safe to run
    against a live run.
    """
    src = Path(row["raw_pdb_path"])
    dest = work / f"{src.stem}.pdb"
    if dest.exists():
        return dest
    if src.exists():
        shutil.copy2(src, dest)
        return dest
    gz = Path(str(src) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rb") as fh, open(dest, "wb") as out:
            shutil.copyfileobj(fh, out)
        return dest
    return None


def score_job(args: tuple) -> dict:
    """One fold's scoring, timed. Module level so the pool can pickle it."""
    pdb_s, chemistry, sequence, recv, tol, skip_relax, outdir_s = args
    pdb = Path(pdb_s)
    stem = str(Path(outdir_s) / pdb.stem)
    out: dict = {"pdb": pdb.name, "skip_relax": skip_relax}

    t0 = time.time()
    val = validity_report(
        pdb, binder_chain=BINDER_CHAIN, receptor_chains=recv,
        chemistry=chemistry, requested_sequence=sequence,
    )
    out["validity_s"] = round(time.time() - t0, 3)
    verd = tolmod.verdict(val, tol)
    out["prerelax_valid"] = bool(verd["valid"])

    t1 = time.time()
    sc = score_structure(
        pdb, binder_chain=BINDER_CHAIN, receptor_chains=recv,
        chemistry=chemistry, requested_sequence=sequence,
        refined_out_path=f"{stem}_refined.pdb",
        isolated_out_path=f"{stem}_isolated_relaxed.pdb",
        skip_relax=skip_relax,
    )
    out["score_s"] = round(time.time() - t1, 3)
    out["total_s"] = round(time.time() - t0, 3)
    out["scoring_status"] = sc.get("status")
    out["relax_skipped"] = bool(sc.get("relax_skipped"))
    # The raw analysis must survive a skip -- that is the whole reason the guard sits
    # inside `score_structure` rather than around it.
    out["has_raw"] = bool(sc.get("raw"))
    out["has_refined"] = bool(sc.get("refined"))
    out["raw_dG"] = (sc.get("raw") or {}).get("dG_separated")
    out["refined_dG"] = (sc.get("refined") or {}).get("dG_separated")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="a cpseq run directory with state/*/outcomes.jsonl and "
                         "predictions/ on THIS cluster. Read-only.")
    ap.add_argument("--work", required=True, help="scratch dir for staged structures")
    ap.add_argument("--out", required=True, help="results JSONL")
    ap.add_argument("--n", type=int, default=24,
                    help="folds to measure per configuration")
    ap.add_argument("--chemistry", default="mainchain")
    ap.add_argument("--scoring-version", default="v3-cis-amide-muco-windows")
    ap.add_argument("--tolerances", default="")
    ap.add_argument("--workers", default="1,2,4,8",
                    help="comma-separated worker counts for the pool scaling arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    behaviour = scoring_behaviour(args.scoring_version)
    print(f"=== cpseq_score_cost_probe ===")
    print(f"run_dir:   {run_dir}")
    print(f"scoring:   {args.scoring_version}  stage={behaviour['validity_stage']}")
    if behaviour["validity_stage"] != "prerelax":
        print("NOTE: this protocol grades the RELAXED complex, so the skip arm below is")
        print("      inert by construction -- it is measured anyway, as the control.")

    tol_path = Path(args.tolerances) if args.tolerances else None
    if tol_path is None:
        cohort = json.loads((run_dir / "cohort.json").read_text())
        tol_path = Path(cohort["dataset_dir"]) / "manifest" / "geometry_tolerances.json"
    tol_all = tolmod.load(tol_path)
    tol = tol_all.get(args.chemistry)
    if tol is None:
        print(f"FATAL: no tolerance for {args.chemistry} in {tol_path}")
        return 1
    print(f"tolerances:{tol_path}")

    rows = collect_folds(run_dir, args.n, args.seed)
    print(f"sampled:   {len(rows)} fold(s) from "
          f"{len({r['parent_id'] for r in rows})} complex(es)")
    if not rows:
        print("GATE: no scored folds with a structure path -- nothing to measure.")
        return 0

    jobs: list[tuple] = []
    staged = 0
    for r in rows:
        pdb = stage_structure(r, work)
        if pdb is None:
            continue
        recv = _receptor_chains(pdb)
        if not recv:
            continue
        staged += 1
        jobs.append((str(pdb), args.chemistry, r["sequence"], recv, tol))
    print(f"staged:    {staged} structure(s) into {work}")
    if not jobs:
        print("GATE: nothing could be staged (structures missing and no .gz either).")
        return 0

    results: list[dict] = []
    fh = open(out_path, "a")

    def emit(rec: dict) -> None:
        results.append(rec)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    # --- arm 1: full vs skipped, serial, same structures -----------------------------
    #
    # Every fold is announced BEFORE it is scored and the line is flushed. A hard exit
    # inside Rosetta leaves no traceback even with faulthandler if the process is killed
    # outright, so the last "scoring" line in the log is what identifies the structure
    # that did it. Without this the only information a crash left was its timestamp.
    for skip in (False, True):
        odir = work / ("out_skip" if skip else "out_full")
        odir.mkdir(parents=True, exist_ok=True)
        label = "skip_relax" if skip else "full"
        t0 = time.time()
        ok = 0
        for i, j in enumerate(jobs, start=1):
            print(f"  [{label}] {i}/{len(jobs)} scoring {Path(j[0]).name} "
                  f"(L={len(j[2])}, receptor chains={''.join(j[3])})", flush=True)
            try:
                rec = score_job(j + (skip, str(odir)))
            except Exception as exc:  # noqa: BLE001
                # One unscoreable structure must not cost the whole arm. Recorded as a
                # row so it is visible in the results rather than only in the log.
                print(f"      FAILED: {type(exc).__name__}: {exc}", flush=True)
                emit({"arm": label, "workers": 1, "pdb": Path(j[0]).name,
                      "skip_relax": skip, "error": f"{type(exc).__name__}: {exc}"})
                continue
            rec |= {"arm": label, "workers": 1}
            emit(rec)
            ok += 1
            print(f"      {rec['scoring_status']}  validity={rec['validity_s']}s "
                  f"score={rec['score_s']}s  prerelax_valid={rec['prerelax_valid']}",
                  flush=True)
        el = time.time() - t0
        print(f"[arm1] {label:10s} {ok}/{len(jobs)} fold(s) in {el:7.1f}s "
              f"=> {el / max(1, ok):6.2f}s/fold")

    # --- arm 1c: THE POSITIONAL CONTROL ----------------------------------------------
    #
    # `full` again, at the END of the serial section. It changes nothing, so it must
    # reproduce the first `full` arm -- and if it does not, the skip comparison above is
    # void, because then arm order and not the relax is what moved the number.
    #
    # This is not belt-and-braces. Job 95856 measured 177 s vs 47 s for two arms that
    # differed in NOTHING but position (cold page cache on a 2.2 GB checkpoint, CUDA
    # JIT), and that 3.8× had already been written up as a 5× algorithmic win before the
    # control caught it. docs/README_CPSEQ_BENCHMARK_PRACTICE.md is the standing rule.
    odir = work / "out_full_control"
    odir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = 0
    for i, j in enumerate(jobs, start=1):
        print(f"  [full_control] {i}/{len(jobs)} scoring {Path(j[0]).name}", flush=True)
        try:
            rec = score_job(j + (False, str(odir)))
        except Exception as exc:  # noqa: BLE001
            print(f"      FAILED: {type(exc).__name__}: {exc}", flush=True)
            emit({"arm": "full_control", "workers": 1, "pdb": Path(j[0]).name,
                  "skip_relax": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rec |= {"arm": "full_control", "workers": 1}
        emit(rec)
        ok += 1
    el_ctrl = time.time() - t0
    print(f"[arm1c] full_control {ok}/{len(jobs)} fold(s) in {el_ctrl:7.1f}s "
          f"=> {el_ctrl / max(1, ok):6.2f}s/fold")

    # --- arm 2: pool scaling, full scoring ------------------------------------------
    #
    # Worker counts are run in the order given AND THEN REVERSED, because the confound
    # here is monotone with the independent variable: measured only ascending, every
    # larger pool runs warmer than the one before it and warm-up is added straight into
    # the apparent scaling curve. Running both directions makes the positional component
    # visible as the gap between the two passes at the same worker count.
    ws = [int(x) for x in args.workers.split(",") if x.strip() and int(x) > 1]
    schedule = [(w, "up") for w in ws] + [(w, "down") for w in reversed(ws)]
    pool_times: dict[tuple[int, str], float] = {}
    for w, direction in schedule:
        odir = work / f"out_pool{w}_{direction}"
        odir.mkdir(parents=True, exist_ok=True)
        payload = [j + (False, str(odir)) for j in jobs]
        t0 = time.time()
        n = 0
        try:
            with ProcessPoolExecutor(max_workers=w) as ex:
                for rec in ex.map(score_job, payload):
                    rec |= {"arm": "pool", "workers": w, "direction": direction}
                    emit(rec)
                    n += 1
        except BrokenProcessPool as exc:
            # A worker that hard-exits inside Rosetta takes the pool with it, and the
            # surviving results are whatever was already yielded. Reported as a result
            # for this worker count rather than ending the probe: it is the answer for
            # how far the pool scales, which is the question.
            print(f"      POOL BROKEN at w={w} after {n} fold(s): {exc}", flush=True)
            emit({"arm": "pool", "workers": w, "direction": direction,
                  "error": f"BrokenProcessPool: {exc}", "completed": n})
        el = time.time() - t0
        pool_times[(w, direction)] = el
        print(f"[arm2] pool w={w:<2d} {direction:<4s} {n}/{len(jobs)} fold(s) in "
              f"{el:7.1f}s => {el / max(1, n):6.2f}s/fold", flush=True)

    fh.close()

    # --- the summary ----------------------------------------------------------------
    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[len(xs) // 2], 2) if xs else None

    # Error rows carry no timings, so they are excluded from every statistic and counted
    # separately. Letting them through would make a crashed fold read as a fast one.
    full = [r for r in results if r["arm"] == "full" and "error" not in r]
    skip = [r for r in results if r["arm"] == "skip_relax" and "error" not in r]
    ctrl = [r for r in results if r["arm"] == "full_control" and "error" not in r]
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"NOTE: {len(errs)} row(s) errored and are excluded from the statistics:")
        for r in errs[:5]:
            print(f"  {r.get('arm')} w={r.get('workers')} {r.get('pdb','-')}: "
                  f"{r['error'][:120]}")
    if not full:
        print("GATE: no fold scored successfully -- nothing to summarise.")
        return 0
    print()
    print("=== summary ===")
    # --- the control verdict, FIRST, because it decides whether the rest is readable ---
    print("--- positional control (docs/README_CPSEQ_BENCHMARK_PRACTICE.md) ---")
    if ctrl:
        f_med = med(r["score_s"] for r in full) or 0.0
        c_med = med(r["score_s"] for r in ctrl) or 0.0
        drift = abs(c_med - f_med) / f_med if f_med else 0.0
        print(f"full (first):  {f_med} s/fold      full_control (last): {c_med} s/fold")
        print(f"positional drift: {drift:.1%}")
        if drift > 0.15:
            print("*** CONTROL FAILED: the same configuration changed by more than 15% ***")
            print("*** on position alone. Every ratio below is contaminated and must   ***")
            print("*** NOT be quoted. Re-run with the arms serialised, and check no    ***")
            print("*** other job shared this node.                                     ***")
        else:
            print("control OK: position accounts for <15%, so the ratios below are "
                  "warm-vs-warm and quotable.")
    else:
        print("NO CONTROL ARM RAN -- no ratio from this run may be quoted.")
    print()

    print(f"validity_report (prerelax):   {med(r['validity_s'] for r in full)} s")
    print(f"score_structure full:         {med(r['score_s'] for r in full)} s")
    print(f"score_structure skip_relax:   {med(r['score_s'] for r in skip)} s")
    inval = [r for r in full if not r["prerelax_valid"]]
    print(f"prerelax invalid:             {len(inval)}/{len(full)}")
    if full and skip:
        f_med = med(r["score_s"] for r in full) or 0
        s_med = med(r["score_s"] for r in skip) or 0
        if f_med:
            frac = len(inval) / len(full)
            saved = frac * (f_med - s_med) / f_med
            print(f"relax share of score_structure: "
                  f"{(f_med - s_med) / f_med:.1%}")
            print(f"=> skip saves {saved:.1%} of scoring on THIS sample "
                  f"(invalid fraction {frac:.1%})")

    # Identical verdicts are what makes the skip admissible; a difference is a bug.
    by_pdb_full = {r["pdb"]: r for r in full}
    mismatch = [r["pdb"] for r in skip
                if by_pdb_full.get(r["pdb"], {}).get("prerelax_valid")
                != r["prerelax_valid"]]
    print(f"prerelax verdict mismatches between arms: {len(mismatch)} "
          f"(must be 0){'  ' + ','.join(mismatch[:5]) if mismatch else ''}")
    kept_raw = sum(1 for r in skip if r["has_raw"])
    print(f"skipped rows that KEPT the raw analysis: {kept_raw}/{len(skip)} "
          f"(must be all -- the raw interface energy is not relax-dependent)")
    lost_ref = sum(1 for r in skip if r["has_refined"])
    print(f"skipped rows carrying a refined energy:  {lost_ref}/{len(skip)} "
          f"(must be 0)")

    # --- pool scaling, both directions ------------------------------------------------
    # The gap between the up and down pass at the SAME worker count is the positional
    # component. A scaling curve is only as trustworthy as that gap is small.
    if pool_times:
        print()
        print("--- pool scaling (ascending vs descending; gap = positional effect) ---")
        serial = (med(r["score_s"] for r in full) or 0.0) * len(jobs)
        for w in sorted({w for w, _d in pool_times}):
            up = pool_times.get((w, "up"))
            dn = pool_times.get((w, "down"))
            gap = (abs(up - dn) / max(up, dn)) if (up and dn) else None
            spd = (serial / min(up, dn)) if (up and dn and min(up, dn)) else None
            print(f"  w={w:<3d} up={up or float('nan'):7.1f}s "
                  f"down={dn or float('nan'):7.1f}s "
                  f"gap={'n/a' if gap is None else f'{gap:.1%}':>6s} "
                  f"speedup_vs_serial={'n/a' if spd is None else f'{spd:.2f}x'}")
        print("  (speedup is against the serial `full` arm on the same structures;")
        print("   a gap above ~15% means that worker count's number is positional, not")
        print("   a scaling result.)")

    print()
    print(f"rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
