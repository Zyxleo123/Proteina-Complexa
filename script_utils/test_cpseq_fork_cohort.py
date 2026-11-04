#!/usr/bin/env python3
"""Checks for the cohort fork and the paired potentials comparison. No GPU, no folds.

The cases here are the ones where a plausible implementation gives a confidently wrong
answer rather than an error -- above all the journal copy, which produces a complete,
successful, entirely meaningless run.

Run:  python script_utils/test_cpseq_fork_cohort.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fork = _load(HERE / "cpseq_fork_cohort.py", "fork_mod")
cmp_mod = _load(HERE / "cpseq_potentials_compare.py", "cmp_mod")

GATE = "v3-cis-amide-muco-windows"


def build_src(root: Path, *, use_potentials: bool = False, parents=("a", "b", "c")):
    """A minimal but structurally real source run: cohort, journals, proposals."""
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    tol = root / "manifest" / "geometry_tolerances.json"
    tol.write_text(json.dumps({"mainchain": {"n_c_max": 1.5}}))
    cohort = {
        "kind": "seed_ladder",
        "cohort": [{"parent_id": p} for p in parents],
        "parent_sources": {p: str(root) for p in parents},
        "dataset_dir": str(root),
        "n_draws": 64,
        "seeds": list(range(10)),
        "budget_rungs": [1, 2, 4, 8, 64],
        "use_potentials": use_potentials,
        "scoring_version": GATE,
        "geometry_tolerances_sha256": fork.gate_sha256(tol),
    }
    (root / "cohort.json").write_text(json.dumps(cohort, indent=2))
    for p in parents:
        d = root / "state" / p
        d.mkdir(parents=True, exist_ok=True)
        (d / "outcomes.jsonl").write_text(json.dumps({
            "parent_id": p, "stage": "ladder", "arm": "reference",
            "sequence": "AAA", "seed_rank": 0, "valid": True,
            "scoring_version": GATE, "refined_dG_separated": -20.0,
        }) + "\n")
    for p in parents:
        d = root / "mask_proposals" / "random_mask_mpnn_fill" / "round1"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{p}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
            {"sequence": f"S{p}1", "accepted": True},
            {"sequence": f"S{p}2", "accepted": True},
            {"sequence": f"S{p}2", "accepted": True},     # duplicate -> one draw
            {"sequence": f"S{p}X", "accepted": False},    # rejected -> not a draw
        ]))
    return cohort


def run_fork(src: Path, dest: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(HERE / "cpseq_fork_cohort.py"),
         "--src", str(src), "--dest", str(dest), "--use-potentials", "on", *extra],
        capture_output=True, text=True, cwd=str(REPO))


def test_gate_hash_matches_the_authoritative_one() -> None:
    print("\n[the inlined gate hash agrees with cpseq_search's]")
    try:
        sys.path.insert(0, str(HERE))
        from cpseq_search import gate_sha256 as authoritative  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"  skip (cannot import cpseq_search: {type(exc).__name__}) -- "
              f"run this in the project venv to exercise it")
        return
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "geometry_tolerances.json"
        # Provenance keys present, so the exclusion rule is actually exercised.
        p.write_text(json.dumps({
            "mainchain": {"n_c_max": 1.5, "planarity_deg": 8.44},
            "disulfide": {"ss_max": 2.3},
            "copied_from": "/some/other/corpus",
            "copy_rationale": "frozen for the v3 gate",
            "frozen_on": "2026-10-25",
        }))
        check("inlined gate_sha256 == cpseq_search.gate_sha256",
              fork.gate_sha256(p) == authoritative(p),
              f"{fork.gate_sha256(p)[:16]} vs {authoritative(p)[:16]}")


def test_journals_are_not_copied() -> None:
    print("\n[the journals are NOT carried into the fork]")
    # This is the whole point. A fork that copied state/ would find every fold already
    # paid for, do no work, exit 0, and the report would then print the SOURCE run's
    # numbers under the fork's name -- a silent null dressed as a measurement.
    with tempfile.TemporaryDirectory() as td:
        src, dest = Path(td) / "src", Path(td) / "dest"
        build_src(src)
        r = run_fork(src, dest)
        check("fork succeeds", r.returncode == 0, r.stdout + r.stderr)
        check("dest has NO state/ directory", not (dest / "state").exists())
        check("dest has no outcome rows at all",
              list(dest.rglob("outcomes.jsonl")) == [])
        check("the log says why the journal was left behind",
              "not copied: state/" in r.stdout)
        check("proposals ARE copied",
              (dest / "mask_proposals" / "random_mask_mpnn_fill"
               / "round1" / "a.jsonl").exists())
        check("the copied proposals are byte-identical",
              (dest / "mask_proposals" / "random_mask_mpnn_fill"
               / "round1" / "a.jsonl").read_bytes()
              == (src / "mask_proposals" / "random_mask_mpnn_fill"
                  / "round1" / "a.jsonl").read_bytes())


def test_only_use_potentials_changes() -> None:
    print("\n[exactly one key differs between the two cohorts]")
    with tempfile.TemporaryDirectory() as td:
        src, dest = Path(td) / "src", Path(td) / "dest"
        build_src(src)
        run_fork(src, dest)
        a = json.loads((src / "cohort.json").read_text())
        b = json.loads((dest / "cohort.json").read_text())
        differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        check("only use_potentials and the provenance stanza differ",
              differing == {"use_potentials", "forked_from"}, str(differing))
        check("use_potentials flipped on", b["use_potentials"] is True)
        for key in ("cohort", "seeds", "n_draws", "budget_rungs",
                    "scoring_version", "geometry_tolerances_sha256"):
            check(f"{key} carried over verbatim", a[key] == b[key])
        check("provenance records what changed",
              b["forked_from"]["changed"]["use_potentials"] == [False, True])


def test_collapse_fork() -> None:
    print("\n[--collapse writes the collapse triple and keeps its refusals]")
    with tempfile.TemporaryDirectory() as td:
        # Source at potentials-on so collapse is the ONLY changed setting -- a
        # collapse-only fork must be allowed although use_potentials does not move.
        src = Path(td) / "src"
        build_src(src, use_potentials=True)   # seeds = range(10), depth 10 > 4

        # Depth 10 collapse without --chunk-fix is refused, same as ladder-cohort.
        r = run_fork(src, Path(td) / "dno", "--collapse")
        check("refuses --collapse above depth 4 without --chunk-fix",
              r.returncode == 1 and "needs --chunk-fix" in r.stdout, r.stdout[-300:])

        # With the fix (and a batch size) it proceeds, although potentials did not change.
        dest = Path(td) / "dcol"
        r = run_fork(src, dest, "--collapse", "--chunk-fix",
                     "--max-parallel-samples", "15")
        check("collapse-only fork proceeds (potentials unchanged)",
              r.returncode == 0, r.stdout[-400:])
        if (dest / "cohort.json").exists():
            c = json.loads((dest / "cohort.json").read_text())
            check("collapse_seeds set true", c.get("collapse_seeds") is True)
            check("chunk_fix set true", c.get("chunk_fix") is True)
            check("max_parallel_samples recorded", c.get("max_parallel_samples") == 15)
            check("no diffusion_samples key written (fold side derives it)",
                  "diffusion_samples" not in c)
            check("provenance records the collapse change",
                  c["forked_from"]["changed"]["collapse_seeds"] == [False, True])
            check("not flagged a replication", c["forked_from"]["replication"] is False)
            check("journals still not copied", not (dest / "state").exists())

        # A plain potentials fork of a pre-collapse source writes NO collapse keys, so
        # the byte-for-byte guarantee the other tests rely on is preserved.
        src2 = Path(td) / "src2"
        build_src(src2)   # potentials off, no collapse keys
        d2 = Path(td) / "d2"
        r = run_fork(src2, d2)   # flips potentials only
        check("plain fork succeeds", r.returncode == 0, r.stdout[-300:])
        if (d2 / "cohort.json").exists():
            c2 = json.loads((d2 / "cohort.json").read_text())
            check("plain fork adds no collapse_seeds key",
                  "collapse_seeds" not in c2)


def test_refuses_the_dangerous_cases() -> None:
    print("\n[refusals]")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        build_src(src)

        r = run_fork(src, src)
        check("refuses src == dest", r.returncode == 1 and "REFUSING" in r.stdout)

        # A source that already has potentials on: forking it changes nothing, and the
        # comparison downstream would be a run against itself.
        src2 = Path(td) / "src2"
        build_src(src2, use_potentials=True)
        r = run_fork(src2, Path(td) / "d2")
        check("refuses a fork that changes nothing",
              r.returncode == 1 and "already has use_potentials" in r.stdout)

        # A destination carrying journal rows. Without --resume this must refuse: those
        # rows were folded under some other setting and the journal key cannot tell.
        d3 = Path(td) / "d3"
        build_src(d3)
        r = run_fork(src, d3)
        check("refuses a destination that already holds journal rows",
              r.returncode == 1 and "journal row" in r.stdout, r.stdout[-400:])

        # ... and with --resume it must STILL refuse, because that destination's cohort
        # says potentials are off while this call asks for on.
        r = run_fork(src, d3, "--resume")
        check("--resume still refuses a cohort that disagrees about the setting",
              r.returncode == 1 and "already exists with use_potentials" in r.stdout,
              r.stdout[-400:])

        # The gate hash no longer matches what the cohort recorded.
        src4 = Path(td) / "src4"
        build_src(src4)
        (src4 / "manifest" / "geometry_tolerances.json").write_text(
            json.dumps({"tolerances": {"mainchain": {"n_c_max": 1.9}}}))
        r = run_fork(src4, Path(td) / "d4")
        check("refuses when the gate hash does not match",
              r.returncode == 1 and "does not match" in r.stdout, r.stdout[-300:])
        check("prints the thresholds-only hash too, so the mismatch can be diagnosed",
              "thresholds-only" in r.stdout and "on disk now" in r.stdout)
        check("names both the prose case and the numbers case",
              "only the PROSE moved" in r.stdout and "the NUMBERS moved" in r.stdout)

        # --rehash-gate is the escape hatch, and it must not be silent.
        r = run_fork(src4, Path(td) / "d4b", "--rehash-gate")
        check("--rehash-gate without a reason is refused",
              r.returncode == 1 and "needs --rehash-reason" in r.stdout)
        d4c = Path(td) / "d4c"
        r = run_fork(src4, d4c, "--rehash-gate",
                     "--rehash-reason", "numbers equal the audit's adopted set")
        check("--rehash-gate with a reason proceeds", r.returncode == 0,
              r.stdout[-300:])
        if (d4c / "cohort.json").exists():
            got = json.loads((d4c / "cohort.json").read_text())
            check("the override is recorded on the forked cohort",
                  got.get("gate_rehash", {}).get("reason")
                  == "numbers equal the audit's adopted set")
            check("the forked cohort records the hash it will actually be scored under",
                  got["geometry_tolerances_sha256"]
                  == fork.gate_sha256(src4 / "manifest"
                                      / "geometry_tolerances.json"))
            check("and keeps the source's hash alongside it",
                  got["gate_rehash"]["source_cohort_hash"]
                  == json.loads((src4 / "cohort.json").read_text())
                      ["geometry_tolerances_sha256"])

        # A prose edit moves the authoritative hash but not the thresholds hash. That
        # distinction is the whole reason both are printed.
        src4d = Path(td) / "src4d"
        build_src(src4d)
        tolp = src4d / "manifest" / "geometry_tolerances.json"
        before = fork.thresholds_sha256(tolp)
        doc = json.loads(tolp.read_text())
        doc["note"] = "reworded; no threshold touched"
        tolp.write_text(json.dumps(doc))
        check("a prose edit leaves the thresholds hash alone",
              fork.thresholds_sha256(tolp) == before)
        check("...but does change the authoritative gate hash",
              fork.gate_sha256(tolp) != json.loads(
                  (src4d / "cohort.json").read_text())["geometry_tolerances_sha256"])

        # A source cohort predating the `scoring_version` key. Without an explicit
        # version the fork's folds would be scored under the module default, the v3
        # report would find no rows for the arm, and the whole GPU spend would be
        # unreadable. The failure is invisible until the report comes back empty.
        src7 = Path(td) / "src7"
        build_src(src7)
        c7 = json.loads((src7 / "cohort.json").read_text())
        del c7["scoring_version"]
        c7["scoring_protocol_version_for_energies"] = GATE
        (src7 / "cohort.json").write_text(json.dumps(c7))
        r = run_fork(src7, Path(td) / "d7")
        check("refuses a cohort with no scoring_version",
              r.returncode == 1 and "carries no `scoring_version`" in r.stdout,
              r.stdout[-300:])
        check("...and names the version to pass",
              GATE in r.stdout and "--scoring-version" in r.stdout)
        d7b = Path(td) / "d7b"
        r = run_fork(src7, d7b, "--scoring-version", GATE)
        check("proceeds once the version is given explicitly", r.returncode == 0,
              r.stdout[-300:])
        if (d7b / "cohort.json").exists():
            check("the forked cohort carries the version",
                  json.loads((d7b / "cohort.json").read_text())
                  .get("scoring_version") == GATE)

        r = run_fork(src7, Path(td) / "d7c", "--scoring-version", "v9-not-a-protocol")
        check("refuses an unknown scoring protocol",
              r.returncode == 1, r.stdout[-200:])

        # --remap must verify the corpus it is pointed at actually exists.
        src6 = Path(td) / "src6"
        build_src(src6)
        r = run_fork(src6, Path(td) / "d6", "--remap", f"{src6}=/nope/missing")
        check("refuses a remap onto a path that does not exist",
              r.returncode == 1 and "do not exist" in r.stdout, r.stdout[-300:])

        # Thresholds missing entirely.
        src5 = Path(td) / "src5"
        build_src(src5)
        (src5 / "manifest" / "geometry_tolerances.json").unlink()
        r = run_fork(src5, Path(td) / "d5")
        check("refuses when the thresholds file is missing",
              r.returncode == 1 and "cannot be verified" in r.stdout)


def test_dry_run_writes_nothing() -> None:
    print("\n[--dry-run touches nothing]")
    with tempfile.TemporaryDirectory() as td:
        src, dest = Path(td) / "src", Path(td) / "dest"
        build_src(src)
        r = run_fork(src, dest, "--dry-run")
        check("succeeds", r.returncode == 0)
        check("destination not created", not dest.exists())


def test_inventory_counts_draws_like_the_loader() -> None:
    print("\n[the inventory counts DRAWS, not proposal lines]")
    # The loaders drop rejected rows and treat a sequence proposed twice as ONE draw.
    # An inventory that counted lines would promise draw depth the fold job cannot
    # deliver, and the mismatch would only surface as a thinner arm after the GPU spend.
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        build_src(src, parents=("a",))
        inv = fork.count_proposals(src, ["a"])
        got = inv["random_mask_mpnn_fill"]["per_parent"]["a"]
        check("4 lines (1 dup, 1 rejected) count as 2 draws", got == 2, f"got {got}")


def test_compare_pairs_and_separates_the_enlarged_set() -> None:
    print("\n[the paired comparison separates the effect from the complex set]")
    # The scenario the plan predicts: potentials barely move the arm per complex, but
    # give two complexes a usable baseline. An unpaired reading of the two aggregates
    # says the arm collapsed; the paired reading says nothing happened.
    def report(rows, n_complexes, no_base):
        return {"out_dir": "x", "draw_arm": "llm_search", "scoring_version": GATE,
                "rungs": [{"B": "all", "n_complexes": n_complexes,
                           "n_no_baseline": no_base, "n_unserved": 0,
                           "energy_unmeasurable": 0, "mean_draws": 10.0,
                           "energy_median_delta": None, "energy_wins": 0,
                           "energy_losses": 0,
                           "draws": "all", "directions": 2, "min_ref_valid": 3,
                           "energy_key": "refined_dG_separated",
                           "per_complex": rows}]}

    off_rows = [
        {"parent_id": "p1", "status": "ok", "energy_delta": -8.0},
        {"parent_id": "p2", "status": "ok", "energy_delta": -7.0},
        {"parent_id": "g1", "status": "no_baseline"},
        {"parent_id": "g2", "status": "no_baseline"},
    ]
    on_rows = [
        {"parent_id": "p1", "status": "ok", "energy_delta": -7.9},
        {"parent_id": "p2", "status": "ok", "energy_delta": -7.1},
        {"parent_id": "g1", "status": "ok", "energy_delta": -0.5},
        {"parent_id": "g2", "status": "ok", "energy_delta": -0.3},
    ]
    with tempfile.TemporaryDirectory() as td:
        o, n, j = Path(td) / "o.json", Path(td) / "n.json", Path(td) / "c.json"
        o.write_text(json.dumps(report(off_rows, 2, 2)))
        n.write_text(json.dumps(report(on_rows, 4, 0)))
        r = subprocess.run(
            [sys.executable, str(HERE / "cpseq_potentials_compare.py"),
             "--off", str(o), "--on", str(n), "--json", str(j)],
            capture_output=True, text=True, cwd=str(REPO))
        check("compare succeeds", r.returncode == 0, r.stdout + r.stderr)
        got = json.loads(j.read_text())
        check("pairs only the complexes BOTH runs scored",
              got["fixed_set"] == ["p1", "p2"], str(got["fixed_set"]))
        check("the paired change is ~0, not the aggregate's 7 REU",
              abs(got["paired_change_median"]) < 0.2,
              str(got["paired_change_median"]))
        check("names the gained complexes", got["gained"] == ["g1", "g2"])
        check("the enlarged set is reported separately",
              got["enlarged_n"] == 4 and got["fixed_n"] == 2)
        # The headline trap: medians over the two FULL sets differ by ~3.5 REU purely
        # because of which complexes are in them.
        check("the enlarged median really is far from the fixed one",
              abs(got["enlarged_median_on"] - got["fixed_median_on"]) > 3.0,
              f"{got['enlarged_median_on']} vs {got['fixed_median_on']}")


def test_compare_refuses_mismatched_runs() -> None:
    print("\n[the comparison refuses what it cannot pair]")
    base = {"out_dir": "x", "draw_arm": "llm_search", "scoring_version": GATE,
            "rungs": [{"B": "all", "n_complexes": 1, "n_no_baseline": 0,
                       "n_unserved": 0, "energy_unmeasurable": 0, "mean_draws": 1.0,
                       "energy_median_delta": -1.0, "energy_wins": 1,
                       "energy_losses": 0,
                       "draws": "all", "directions": 2, "min_ref_valid": 3,
                       "energy_key": "refined_dG_separated",
                       "per_complex": [{"parent_id": "p1", "status": "ok",
                                        "energy_delta": -1.0}]}]}

    def run(off, on):
        with tempfile.TemporaryDirectory() as td:
            o, n = Path(td) / "o.json", Path(td) / "n.json"
            o.write_text(json.dumps(off))
            n.write_text(json.dumps(on))
            return subprocess.run(
                [sys.executable, str(HERE / "cpseq_potentials_compare.py"),
                 "--off", str(o), "--on", str(n)],
                capture_output=True, text=True, cwd=str(REPO))

    import copy
    other_arm = copy.deepcopy(base)
    other_arm["draw_arm"] = "random_mask_mpnn_fill"
    r = run(base, other_arm)
    check("refuses two different ARMS", r.returncode == 1 and "draw_arm" in r.stdout)

    other_gate = copy.deepcopy(base)
    other_gate["scoring_version"] = "v5-something"
    r = run(base, other_gate)
    check("refuses two different GATES",
          r.returncode == 1 and "scoring_version" in r.stdout)

    other_est = copy.deepcopy(base)
    other_est["rungs"][0]["min_ref_valid"] = 0
    r = run(base, other_est)
    check("refuses two different ESTIMATORS",
          r.returncode == 1 and "different estimator" in r.stdout)

    no_pc = copy.deepcopy(base)
    del no_pc["rungs"][0]["per_complex"]
    r = run(base, no_pc)
    check("explains a JSON written before per_complex existed",
          r.returncode != 0 and "per_complex" in (r.stdout + r.stderr))

    many = copy.deepcopy(base)
    many["rungs"].append(copy.deepcopy(base["rungs"][0]))
    r = run(base, many)
    check("refuses a budget-rung report",
          r.returncode != 0 and "rungs" in (r.stdout + r.stderr))


def main() -> int:
    test_gate_hash_matches_the_authoritative_one()
    test_journals_are_not_copied()
    test_only_use_potentials_changes()
    test_collapse_fork()
    test_refuses_the_dangerous_cases()
    test_dry_run_writes_nothing()
    test_inventory_counts_draws_like_the_loader()
    test_compare_pairs_and_separates_the_enlarged_set()
    test_compare_refuses_mismatched_runs()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
