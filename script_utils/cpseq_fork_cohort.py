#!/usr/bin/env python3
"""Fork a ladder run into a sibling run that changes ONE inference setting.

Why this exists
---------------
`use_potentials` is a property of the COHORT, not a per-run flag: `cpseq_search.py`
builds its `Protocol` from `cohort.get("use_potentials")` so that two shards cannot
disagree about the predictor settings and write folds nothing downstream would notice
were incomparable. So "re-fold the same sequences with potentials on" is not a flag on
the fold job -- it is a second cohort.

Re-running `ladder-cohort` would not do it. That command SAMPLES its complexes from the
corpora by a recorded rule, so a second invocation is a second sample, and the point of
this check is that the complex set, the seeds, the rungs and the draws are the ones that
produced the number being tested. This script therefore DERIVES the sibling cohort from
the original file rather than rebuilding it: everything is copied verbatim and exactly
one key is rewritten.

The trap this script exists to prevent
--------------------------------------
The journal key is `(sequence, prediction_seed, stage, arm)` -- see `Journal` in
`cpseq_search.py`, and the comment on `use_potentials` in the cohort record. It carries
NO inference settings. Copy `state/` into the sibling directory and every fold reads as
already paid for: the fold job finds nothing to do, exits successfully, and the report
then prints the ORIGINAL run's potentials-off numbers under the new directory's name.
There is no error and no warning anywhere in that sequence. It is a silent null result
that looks like a measurement.

So this script copies the PROPOSALS and refuses to copy the journals, and it refuses to
write into a destination that already has journal rows unless told to resume.

(The prediction and score caches are safe either way -- `use_potentials` is in
`Boltz2Config.to_dict()` and therefore in `input_hash`, so a potentials-on fold can
never collide with a potentials-off one. The journal is the only settings-blind key,
which is precisely why it is the one that has to start empty.)

What is copied, and what is not
-------------------------------
    cohort.json        copied, with `use_potentials` rewritten and provenance added
    llm_proposals/     copied verbatim      -- arm D's (`llm_search`) draws
    mask_proposals/    copied verbatim      -- arms E/F/G's draws
    state/             NOT copied           -- the journals; see above
    predictions/       NOT copied           -- rebuilt, keyed by the new settings
    scores/            NOT copied           -- likewise

Nothing in the SOURCE directory is written, opened for writing, or moved.

Usage:
    python script_utils/cpseq_fork_cohort.py \
        --src /data/user_data/yixiz/cpseq/llm_ladder \
        --dest /data/user_data/yixiz/cpseq/llm_ladder_pot \
        --use-potentials on
    # add --dry-run to print the plan and touch nothing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The proposal trees, and the per-arm subdirectory layout of each. Kept as data rather
# than hardcoded in the copy loop so that adding an arm is a one-line change here and
# cannot be half-done.
PROPOSAL_TREES = (
    # (relative path, has a per-arm level under it)
    ("llm_proposals", False),
    ("mask_proposals", True),
)
# Directories that must NOT be carried across, with the reason each one is refused.
# Printed, not silently skipped: someone reading the log should be able to see that the
# journals were left behind on purpose.
NEVER_COPY = {
    "state": "the journal; its key carries no inference settings, so copying it would "
             "retire every fold as already paid for and the run would measure nothing",
    "predictions": "keyed by input_hash, which includes the inference settings; the "
                   "new settings produce new hashes, so there is nothing to reuse",
    "scores": "keyed by (scoring_version, input_hash); same reason as predictions",
}


# Provenance keys are excluded from the hash: a copied tolerance file records where it
# came from, so two corpora can hold identical numbers in byte-different files.
GATE_PROVENANCE_KEYS = ("copied_from", "copy_rationale", "frozen_on")


def gate_sha256(path: Path) -> str:
    """Content hash of the acceptance thresholds, ignoring their provenance keys.

    A deliberate copy of `cpseq_search.gate_sha256` rather than an import of it. That
    module pulls in pandas and the whole scoring stack, which is a heavy dependency for
    four lines of hashing and makes this script unable to run anywhere the full
    environment is not installed -- including the quick local check that the fork plan
    is sane before a job is ever queued.

    `test_cpseq_fork_cohort.py` asserts this function agrees with the authoritative one
    whenever that one is importable, so the copy cannot drift unnoticed.
    """
    import hashlib

    doc = json.loads(Path(path).read_text())
    thresholds = {k: v for k, v in doc.items() if k not in GATE_PROVENANCE_KEYS}
    return hashlib.sha256(json.dumps(thresholds, sort_keys=True).encode()).hexdigest()


def thresholds_sha256(path: Path) -> str:
    """Hash of the `tolerances` subtree alone -- the NUMBERS that decide pass/fail.

    A diagnostic, never a substitute for `gate_sha256`. The authoritative hash covers
    every key except the three provenance ones, which includes the free-text `note` and
    `clash_cutoff_note` and the `margin`. So re-wording a note changes the authoritative
    hash while no threshold moved, and the refusal then reports "the acceptance
    thresholds have changed" about a prose edit.

    Both are printed on a mismatch so the two cases can be told apart. The refusal stands
    either way -- `ladder` checks the authoritative hash and will refuse too, so a fork
    that proceeded would only move the failure into the GPU job -- but "only the prose
    moved" is a five-minute fix to the recorded hash, and "the numbers moved" means the
    run being compared against was scored under a different gate and the comparison is
    not available at all.
    """
    import hashlib

    doc = json.loads(Path(path).read_text())
    return hashlib.sha256(
        json.dumps(doc.get("tolerances", {}), sort_keys=True).encode()).hexdigest()


def gate_hash(dataset_dir: Path) -> str | None:
    tol = dataset_dir / "manifest" / "geometry_tolerances.json"
    if not tol.exists():
        return None
    return gate_sha256(tol)


def journal_rows(root: Path) -> int:
    """How many outcome rows the destination already holds."""
    n = 0
    for f in (root / "state").glob("*/outcomes.jsonl"):
        n += sum(1 for line in f.read_text().splitlines() if line.strip())
    return n


def count_proposals(src: Path, parents: list[str]) -> dict:
    """Per arm, how many draws each parent has on disk, from the proposal files alone.

    This is the inventory that decides whether the check is even runnable: a parent with
    no proposal file contributes no draws, and an arm whose proposals never reached full
    draw depth in the source run will not reach it in the fork either -- the fork copies
    the proposals, it does not make new ones.
    """
    out: dict[str, dict[str, int]] = {}
    for rel, per_arm in PROPOSAL_TREES:
        base = src / rel
        if not base.is_dir():
            continue
        arms = sorted(p.name for p in base.iterdir() if p.is_dir()) if per_arm \
            else ["llm_search"]
        for arm in arms:
            adir = base / arm if per_arm else base
            counts: dict[str, int] = {}
            seqs: dict[str, set] = {}
            for rnd in sorted(adir.glob("round*")):
                for pf in sorted(rnd.glob("*.jsonl")):
                    pid = pf.stem
                    seqs.setdefault(pid, set())
                    for line in pf.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Mirror the loaders: only ACCEPTED proposals become draws, and a
                        # sequence proposed twice is ONE draw. Counting raw lines would
                        # overstate the arm's draw depth.
                        if not r.get("sequence"):
                            continue
                        if rel == "mask_proposals" and not r.get("accepted", True):
                            continue
                        seqs[pid].add(r["sequence"])
            for pid, s in seqs.items():
                counts[pid] = len(s)
            if counts:
                out[arm] = {"rel": rel, "per_parent": counts,
                            "rounds": len(list(adir.glob("round*"))),
                            "missing_parents": sorted(set(parents) - set(counts))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="the finished run to fork. Read-only; never written to.")
    ap.add_argument("--dest", required=True,
                    help="the sibling run directory to create. Must not already hold "
                         "journal rows (see --resume).")
    ap.add_argument("--use-potentials", choices=("on", "off"), required=True,
                    help="the value to write into the sibling cohort. Required and "
                         "explicit: the whole purpose of the fork is this one key, so "
                         "it is not something to leave to a default.")
    ap.add_argument("--collapse", action="store_true",
                    help="set `collapse_seeds` on the forked cohort: buy its fold depth "
                         "from ONE invocation's diffusion samples off a single trunk "
                         "(README_CPSEQ_FOLD_PIPELINE_STANDARD sec. 2) instead of one "
                         "invocation per seed. Like `use_potentials`, this is a COHORT "
                         "property and the two regimes key APART in the journal, so a "
                         "collapsed fork must have its own --dest. The fold side reads "
                         "this back and sets diffusion_samples to the seed depth itself; "
                         "nothing else here changes. When the source is already "
                         "collapsed, pass this flag off to fork back to per-seed.")
    ap.add_argument("--chunk-fix", action="store_true",
                    help="record `chunk_fix` on the forked cohort: the fold job shadows "
                         "the denoiser chunk-count correction in front of Boltz-2. "
                         "REQUIRED with --collapse above seed depth 4 (the same refusal "
                         "ladder-cohort makes), because upstream computes the chunk "
                         "count as `multiplicity % max_parallel_samples + 1` where it "
                         "means ceil(/), so the batch size is otherwise an accident of "
                         "the remainder.")
    ap.add_argument("--max-parallel-samples", type=int, default=None,
                    help="record the denoiser batch size on the forked cohort (boltz's "
                         "own default when unset). With --collapse this caps peak GPU "
                         "memory, since the per-input denoiser multiplicity grows to "
                         "3*depth; 15 was measured to fit a depth-10 collapse in 32 GB.")
    ap.add_argument("--same-setting-ok", action="store_true",
                    help="permit a fork whose `use_potentials` equals the source's. "
                         "For a REPLICATION fork, where the thing being changed is the "
                         "machine rather than the setting: re-folding the source's own "
                         "sequences at its own setting on different hardware gives a "
                         "same-hardware baseline, which is the only way a cross-machine "
                         "comparison has one variable. Normally refused, because a fork "
                         "that changes nothing is two directories holding the same "
                         "experiment.")
    ap.add_argument("--resume", action="store_true",
                    help="allow a destination that already has journal rows, for the "
                         "normal case of re-running this after a partial fold. It still "
                         "refuses if the existing cohort disagrees about the setting.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the inventory, write nothing.")
    ap.add_argument("--json", default=None,
                    help="write the inventory here (default <dest>/fork_report.json).")
    ap.add_argument("--remap", action="append", default=[], metavar="OLD=NEW",
                    help="repeatable. Rewrite a path prefix in `dataset_dir` and "
                         "`parent_sources`. Needed when the source run was made on a "
                         "different machine: the cohort records absolute corpus paths, "
                         "and a fork folded elsewhere must point at the same corpora "
                         "where they actually are. It rewrites WHERE the corpus is "
                         "read from, never WHICH corpus -- the remapped paths are "
                         "checked to exist and the gate hash is still verified against "
                         "whatever is found there.")
    ap.add_argument("--rehash-gate", action="store_true",
                    help="record the gate hash of the tolerance file actually on disk "
                         "instead of the source cohort's, allowing the fork to proceed "
                         "when the two differ. Requires --rehash-reason. Use ONLY when "
                         "the threshold NUMBERS have been shown equal and the hash "
                         "differs over prose (the hash covers `note`, "
                         "`clash_cutoff_note` and `margin` as well as the thresholds, "
                         "so a reworded note moves it). Both hashes and the reason are "
                         "recorded on the forked cohort.")
    ap.add_argument("--scoring-version", default="",
                    help="set `scoring_version` on the forked cohort. REQUIRED when the "
                         "source cohort does not carry one: that key postdates some "
                         "runs, and `cohort_scoring_version` then falls back to the "
                         "module default (currently v4-postrelax-validity), so every "
                         "new fold would be scored under a protocol the comparison "
                         "cannot read -- the v3 report would find zero rows for the arm "
                         "after the whole GPU spend. The source's "
                         "`scoring_protocol_version_for_energies` records what its "
                         "energies were actually computed under and is the value to "
                         "pass.")
    ap.add_argument("--rehash-reason", default="",
                    help="the evidence that the thresholds themselves are unchanged. "
                         "Recorded verbatim on the forked cohort, because a future "
                         "reader of these numbers needs to know this override happened "
                         "and on what grounds.")
    args = ap.parse_args()

    if args.rehash_gate and not args.rehash_reason.strip():
        print("[fork] REFUSING: --rehash-gate needs --rehash-reason. Overriding the "
              "gate check without recording why is how a run ends up scored under "
              "thresholds nobody can reconstruct.")
        return 1

    src, dest = Path(args.src), Path(args.dest)
    want = args.use_potentials == "on"

    if src.resolve() == dest.resolve():
        print("[fork] REFUSING: --src and --dest are the same directory. A fork that "
              "overwrote its source would destroy the run the comparison is against.")
        return 1

    cohort_path = src / "cohort.json"
    if not cohort_path.exists():
        print(f"[fork] FATAL: no cohort at {cohort_path}")
        print("       Is /data mounted on this node? It is not mounted on the login "
              "node, so this script has to run inside a job.")
        return 1
    cohort = json.loads(cohort_path.read_text())

    had = bool(cohort.get("use_potentials", False))
    parents = [e["parent_id"] if isinstance(e, dict) else e for e in cohort["cohort"]]

    # Collapse is the second forkable cohort setting. The source's current value, the
    # requested one, and the seed depth that decides whether --chunk-fix is mandatory.
    src_collapse = bool(cohort.get("collapse_seeds", False))
    want_collapse = bool(args.collapse)
    depth = len(cohort.get("seeds", []))
    # Mirror ladder-cohort's refusal exactly: collapse above depth 4 without the chunk
    # fix lets upstream's `multiplicity % max_parallel_samples + 1` set the batch size by
    # remainder. A property of the whole run, so it is caught here, not per shard.
    if want_collapse and depth > 4 and not args.chunk_fix:
        print(f"[fork] REFUSING: --collapse at seed depth {depth} needs --chunk-fix. "
              f"Upstream computes the denoiser chunk count as "
              f"`multiplicity % max_parallel_samples + 1` where it means ceil(/), so at "
              f"this depth the batch size is an accident of the remainder, not a number "
              f"anyone chose. Pass --chunk-fix (and --max-parallel-samples).")
        return 1

    # Path remapping, applied before anything reads `dataset_dir`, so the gate check
    # below verifies the corpus the fork will ACTUALLY be folded against.
    remaps: list[tuple[str, str]] = []
    for spec in args.remap:
        if "=" not in spec:
            print(f"[fork] FATAL: --remap wants OLD=NEW, got {spec!r}")
            return 1
        old, new = spec.split("=", 1)
        remaps.append((old, new))

    def remap(p: str) -> str:
        for old, new in remaps:
            if p.startswith(old):
                return new + p[len(old):]
        return p

    if remaps:
        cohort["dataset_dir"] = remap(cohort["dataset_dir"])
        cohort["parent_sources"] = {k: remap(v)
                                    for k, v in cohort["parent_sources"].items()}
        missing = sorted({v for v in cohort["parent_sources"].values()
                          if not Path(v).is_dir()})
        print(f"[fork] remapped corpus paths: "
              f"{sorted(set(cohort['parent_sources'].values()))}")
        if missing:
            print(f"[fork] REFUSING: remapped corpus path(s) do not exist: {missing}")
            print(f"    A fork whose parents point at a missing corpus fails per "
                  f"complex inside the fold job ('no target context -- skipped'), "
                  f"which looks like a thin arm rather than a broken path.")
            return 1

    print("=== cpseq_fork_cohort ===")
    print(f"src:   {src}")
    print(f"dest:  {dest}")
    print(f"use_potentials: {had} -> {want}")
    print(f"parents: {len(parents)}")
    print(f"seeds:   {cohort.get('seeds')}")
    print(f"n_draws: {cohort.get('n_draws')}  rungs={cohort.get('budget_rungs')}")
    print(f"gate:    {cohort.get('scoring_version')}")
    print(f"tolerances_sha256: {str(cohort.get('geometry_tolerances_sha256'))[:16]}")

    # A cohort with no `scoring_version` would have its new folds scored under the module
    # default, which is not necessarily the protocol the source run's numbers are in.
    # Refused rather than defaulted: the failure is invisible until the report comes back
    # with no rows, by which time the folds are paid for.
    src_sv = cohort.get("scoring_version")
    energies_sv = cohort.get("scoring_protocol_version_for_energies")
    if not src_sv and not args.scoring_version:
        print(f"[fork] REFUSING: the source cohort carries no `scoring_version`, so "
              f"every fold of the fork would be\n    scored under the module default "
              f"instead of the protocol the source's numbers are in.")
        print(f"    The source records scoring_protocol_version_for_energies="
              f"{energies_sv!r}.")
        print(f"    Pass --scoring-version {energies_sv or '<version>'} to set it "
              f"explicitly.")
        return 1
    if args.scoring_version and src_sv and args.scoring_version != src_sv:
        print(f"[fork] REFUSING: --scoring-version {args.scoring_version!r} disagrees "
              f"with the source cohort's {src_sv!r}. Scoring the fork under a different "
              f"protocol is a second changed variable.")
        return 1
    if args.scoring_version and energies_sv and args.scoring_version != energies_sv:
        print(f"[fork] WARNING: --scoring-version {args.scoring_version!r} is not the "
              f"source's scoring_protocol_version_for_energies {energies_sv!r}. That is "
              f"the protocol its energies were computed under; make sure this is "
              f"deliberate.")

    # A fork must change SOMETHING -- either potentials or collapse. A collapse-only fork
    # keeps use_potentials equal to the source's and is still a real experiment, so the
    # "nothing changed" refusal fires only when BOTH settings match.
    potentials_changed = had != want
    collapse_changed = src_collapse != want_collapse
    if not potentials_changed and not collapse_changed and not args.same_setting_ok:
        print(f"[fork] REFUSING: the source cohort already has use_potentials={had} and "
              f"collapse_seeds={src_collapse}, and this call changes neither. A fork "
              f"that changes nothing is two directories holding the same experiment, "
              f"and the comparison it feeds would be a run against itself.")
        print(f"    Pass --collapse to change the fold regime, or --same-setting-ok for "
              f"a REPLICATION on different hardware.")
        return 1
    if not potentials_changed and not collapse_changed:
        print(f"[fork] REPLICATION fork: use_potentials stays {had}, collapse_seeds "
              f"stays {src_collapse}. The variable being changed is the machine.")
    elif collapse_changed:
        print(f"[fork] collapse_seeds: {src_collapse} -> {want_collapse}"
              + (f"  (chunk_fix={args.chunk_fix}, "
                 f"max_parallel_samples={args.max_parallel_samples})"
                 if want_collapse else ""))

    # The gate must still be the one the source cohort recorded, or the fork's folds are
    # scored against different thresholds than the numbers being compared to. `ladder`
    # makes this same check and refuses; making it HERE means finding out now rather
    # than after the fold job has queued.
    ddir = Path(cohort["dataset_dir"])
    recorded = cohort.get("geometry_tolerances_sha256")
    actual = gate_hash(ddir)
    if recorded and actual is None:
        print(f"[fork] REFUSING: the source cohort records a threshold hash but "
              f"{ddir / 'manifest' / 'geometry_tolerances.json'} is missing, so the "
              f"gate cannot be verified. `ladder` refuses to fold in this state too, so "
              f"forking now would only move the failure into the fold job.")
        return 1
    gate_rehash = None
    if recorded and actual and recorded != actual and args.rehash_gate:
        tol_file = ddir / "manifest" / "geometry_tolerances.json"
        gate_rehash = {
            "source_cohort_hash": recorded,
            "recorded_instead": actual,
            "thresholds_sha256": thresholds_sha256(tol_file),
            "tolerances_file": str(tol_file),
            "reason": args.rehash_reason.strip(),
        }
        print(f"[fork] GATE REHASH (--rehash-gate)")
        print(f"    source cohort recorded {recorded[:24]}")
        print(f"    recording instead      {actual[:24]}")
        print(f"    thresholds-only        {gate_rehash['thresholds_sha256'][:24]}")
        print(f"    reason: {gate_rehash['reason']}")
        print(f"    This is an override. The fork's folds will be SCORED under "
              f"{tol_file},\n    and the comparison is only valid if that file's "
              f"thresholds are the ones the source\n    run was scored under. Both "
              f"hashes and this reason are recorded on the forked cohort.")
        recorded = actual
    if recorded and actual and recorded != actual:
        tol_file = ddir / "manifest" / "geometry_tolerances.json"
        print(f"[fork] REFUSING: the gate hash recorded by the source cohort does not "
              f"match {tol_file}.")
        print(f"    cohort recorded {recorded[:24]}")
        print(f"    on disk now     {actual[:24]}")
        print(f"    thresholds-only {thresholds_sha256(tol_file)[:24]}  "
              f"(the `tolerances` subtree)")
        print(f"    The recorded hash covers every key but the three provenance ones, so "
              f"it also covers `note`,\n    `clash_cutoff_note` and `margin`. Two cases, "
              f"and they need different fixes:")
        print(f"      * only the PROSE moved -- the numbers are the gate the comparison "
              f"wants. Re-record the\n        hash on the cohort (or restore the file's "
              f"earlier text) and re-run; nothing was mis-scored.")
        print(f"      * the NUMBERS moved -- the run being compared against was scored "
              f"under different\n        thresholds, so a like-for-like comparison is "
              f"not available until one side is re-scored.")
        print(f"    Either way `ladder` checks this same hash and would refuse, so "
              f"forking now would only move\n    the failure into the GPU job.")
        return 1
    if recorded and actual and gate_rehash is None:
        print(f"gate verified: thresholds unchanged since the source cohort was built")
    elif gate_rehash is not None:
        # Do not say "verified" after an override. The hash now matches only because it
        # was just rewritten, and a log line claiming verification is exactly what a
        # later reader would take at face value.
        print(f"gate: NOT verified against the source cohort -- overridden above")

    inventory = count_proposals(src, parents)
    print("\n--- draws available from the proposals on disk ---")
    if not inventory:
        print("  (none -- no llm_proposals/ or mask_proposals/ in the source)")
    for arm, info in sorted(inventory.items()):
        pp = info["per_parent"]
        vals = sorted(pp.values())
        med = vals[len(vals) // 2] if vals else 0
        print(f"  {arm:<26} {info['rel']:<15} rounds={info['rounds']} "
              f"parents={len(pp)}/{len(parents)} draws/parent "
              f"min={vals[0] if vals else 0} med={med} max={vals[-1] if vals else 0}")
        if info["missing_parents"]:
            print(f"      no proposal file for: {', '.join(info['missing_parents'])}")

    # A destination with journal rows is the dangerous case, so it is checked before
    # anything is written and named for what it is.
    if dest.exists():
        existing = journal_rows(dest)
        if existing and not args.resume:
            print(f"\n[fork] REFUSING: {dest} already holds {existing} journal row(s).")
            print("    Those rows were folded under whatever settings their cohort had "
                  "at the time. Adding more under a changed setting would mix two "
                  "predictors in one directory, and the journal key cannot tell them "
                  "apart. Either pass --resume (if this is the same experiment "
                  "continuing) or fork into a directory that does not exist yet.")
            return 1
        dcohort = dest / "cohort.json"
        if dcohort.exists():
            prev = json.loads(dcohort.read_text())
            if bool(prev.get("use_potentials", False)) != want:
                print(f"\n[fork] REFUSING: {dcohort} already exists with "
                      f"use_potentials={prev.get('use_potentials')}, and this call asks "
                      f"for {want}. Rewriting it would re-label folds already on disk "
                      f"as having been made under settings they were not made under.")
                return 1
        if existing:
            print(f"\n[fork] resuming into a destination with {existing} journal row(s)")

    for name, why in sorted(NEVER_COPY.items()):
        if (src / name).exists():
            print(f"  not copied: {name}/ -- {why}")

    if args.dry_run:
        print("\n[fork] --dry-run: nothing written")
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    copied = {}
    for rel, _per_arm in PROPOSAL_TREES:
        s = src / rel
        if not s.is_dir():
            continue
        d = dest / rel
        # dirs_exist_ok so a re-run is idempotent rather than an error. The proposals are
        # the one thing that must be byte-identical to the source, so they are COPIED
        # rather than symlinked: a symlink would make the fork's draws depend on the
        # source directory still existing and still saying the same thing.
        shutil.copytree(s, d, dirs_exist_ok=True)
        n = sum(1 for _ in d.rglob("*.jsonl"))
        copied[rel] = n
        print(f"  copied {rel}/  ({n} proposal file(s))")

    forked = dict(cohort)
    forked["use_potentials"] = want
    # The collapse triple, written exactly as ladder-cohort records it. The fold side
    # reads `collapse_seeds` and sets diffusion_samples to the seed depth itself, so no
    # diffusion_samples key is written here (the cohort never stored one). Only written
    # when the fork actually involves collapse (requested now, or already on the source),
    # so a plain potentials fork of a pre-collapse cohort stays byte-for-byte as before.
    if want_collapse or "collapse_seeds" in cohort:
        forked["collapse_seeds"] = want_collapse
        forked["chunk_fix"] = bool(args.chunk_fix) if want_collapse else False
        forked["max_parallel_samples"] = (
            int(args.max_parallel_samples)
            if (want_collapse and args.max_parallel_samples) else None
        )
    if gate_rehash is not None:
        forked["geometry_tolerances_sha256"] = gate_rehash["recorded_instead"]
        forked["gate_rehash"] = gate_rehash
    if args.scoring_version:
        forked["scoring_version"] = args.scoring_version
        # `cohort_scoring_version` validates this name and raises on an unknown one, so
        # a typo is caught here rather than per shard.
        try:
            from cpseq_search import scoring_behaviour  # noqa: PLC0415

            forked["scoring_behaviour"] = scoring_behaviour(args.scoring_version)
        except ImportError:
            print("[fork] note: cpseq_search not importable here, so "
                  "`scoring_behaviour` was not filled in; the fold job derives it.")
        except Exception as exc:  # noqa: BLE001
            print(f"[fork] REFUSING: --scoring-version {args.scoring_version!r} is not "
                  f"a known protocol ({exc})")
            return 1
        print(f"  scoring_version set to {args.scoring_version}")
    # Provenance, so the sibling cohort can always answer "what am I a fork of, and what
    # was changed" without reference to this script or to a doc.
    forked["forked_from"] = {
        "src": str(src),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "changed": {"use_potentials": [had, want],
                    "collapse_seeds": [src_collapse, want_collapse]},
        "replication": had == want and src_collapse == want_collapse,
        "note": (
            "Derived from the source cohort by cpseq_fork_cohort.py, not rebuilt by "
            "ladder-cohort: the complexes, seeds, draw count and rungs are the source's "
            "verbatim, so the only difference between the two runs is the key named in "
            "`changed`. Proposals were copied so the SEQUENCES are identical by "
            "construction rather than merely reproducible from proposal_seed. The "
            "journals were deliberately NOT copied -- their key carries no inference "
            "settings, so a copy would have retired every fold as already paid for."
        ),
    }
    (dest / "cohort.json").write_text(json.dumps(forked, indent=2))
    print(f"  wrote {dest / 'cohort.json'}  (use_potentials={want})")

    report = {
        "src": str(src), "dest": str(dest),
        "use_potentials": {"src": had, "dest": want},
        "collapse_seeds": {"src": src_collapse, "dest": want_collapse},
        "chunk_fix": forked.get("chunk_fix", False),
        "max_parallel_samples": forked.get("max_parallel_samples"),
        "parents": parents,
        "seeds": cohort.get("seeds"),
        "n_draws": cohort.get("n_draws"),
        "scoring_version": cohort.get("scoring_version"),
        "geometry_tolerances_sha256": recorded,
        "gate_verified": bool(recorded and actual and recorded == actual),
        "gate_rehash": gate_rehash,
        "remaps": [f"{o}={n}" for o, n in remaps],
        "proposal_files_copied": copied,
        "inventory": inventory,
    }
    jpath = Path(args.json) if args.json else dest / "fork_report.json"
    jpath.write_text(json.dumps(report, indent=2))
    print(f"  wrote {jpath}")

    print("\n[fork] OK. The fold jobs now run against --out-dir "
          f"{dest} with the SAME arm flags as the source run; every fold is new work "
          "because the destination's journal is empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
