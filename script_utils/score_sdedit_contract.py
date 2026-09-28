"""Grade existing SDEdit outputs under the SAME contract as the GenKIC arm.

Why this can be done at all, and done fairly
--------------------------------------------
Each SDEdit `edits_shard*.jsonl` row carries `complex_pdb` (the generated peptide WITH its
receptor, written at sample time) and `input_complex_pdb` (the ORIGINAL bound linear peptide
in the IDENTICAL cropped receptor). That second path is the paired baseline the contract asks
for -- same receptor preparation, same crop, same frame -- so no receptor mapping, no
re-cropping and no reconstruction is needed. Both poses are loaded the same way, and the
receptor residues correspond one-to-one by pose order.

This matters because a peptide-only PDB could NOT be graded: the contract needs
peptide-receptor heavy-atom contacts, interface-residue RMSD in the receptor frame, and a
paired interface dG, and none of those exist without the receptor. This repo has measured
that a peptide saved alone can never be re-united with its receptor (two loads of one example
differ by ~2.7 A), so runs that wrote only `pdbs/` are ungradable in principle, not merely
inconvenient. `--edits` therefore points at a run that wrote `complexes/`.

What is deliberately NOT done
-----------------------------
* **Abstaining rows are not filtered out.** `requested_type_satisfied == 0` means the sampled
  sequence carries no competent anchor pair for the requested chemistry. Those rows are graded
  and FAIL on chemistry, because dropping them would compute SDEdit's success rate over only
  the attempts where it had already half-succeeded. They are counted and reported.
  (Note also that abstained rows carry NaN in the run's own metric columns, and NaN is truthy
  in Python -- a naive filter scores a 100%-abstention cell as 100% success.)
* **No re-generation and no re-selection.** These are the structures the arm produced.
* **The whole peptide counts as released.** SDEdit perturbs the entire peptide, so there is no
  "preserved interior" to report and `interior_max_ca_dev_A` is not meaningful here; the
  released set is recorded as every residue, which is the truth.

Two-phase, like the GenKIC arm: every row is graded cheaply on geometry/sequence/contacts/pose,
then a ranked shortlist per (example, chemistry) pays for packing, relaxation and interface
scoring against the paired LP baseline. Acceptance applies to the final relaxed candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cp_contract as C            # noqa: E402
import cp_evaluate as E            # noqa: E402
import cp_search as S              # noqa: E402
import cp_topology as T            # noqa: E402


@dataclass
class SDEditSpec:
    """A graded SDEdit candidate's identity, shaped like `cp_search.SearchSpec`.

    `k_nterm`/`m_cterm` are -1 rather than a window: SDEdit does not release a terminal
    window, it resamples the whole peptide. Writing 0 there would claim a preserved interior
    that does not exist.
    """

    chemistry: str
    aa_i: str
    aa_j: str
    pdb_tag: str
    t_ca: float
    t_lat: float
    seed: int
    k_nterm: int = -1
    m_cterm: int = -1
    extra_mutations: dict[int, str] = field(default_factory=dict)

    def label(self) -> str:
        return self.pdb_tag


def whole_peptide_window(binder_pos: list[int]) -> dict:
    """The released-residue record for an arm that resamples the entire peptide."""
    L = len(binder_pos)
    return {
        "k_nterm": -1, "m_cterm": -1, "n_released": L,
        "released_local": list(range(L)),
        "released_pose": list(binder_pos),
        "loop_order": [], "anchor_lower": None, "anchor_upper": None,
    }


def anchors_competent(chem: T.RingChemistry, aa_i: str, aa_j: str) -> bool:
    """Can `chem`'s bond exist at these two residue identities at all?"""
    if chem.anchor_options is None:
        return True                      # head-to-tail needs only a backbone N and C
    return (aa_i, aa_j) in chem.anchor_options


def load_edits(edits: list[Path]) -> list[dict]:
    """Every edits row. A corrupt line is fatal, never skipped."""
    rows: list[dict] = []
    for path in edits:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"FATAL: corrupt edits row {path}:{lineno} ({exc})")
    return rows


def done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["row_key"])
        except json.JSONDecodeError:
            raise SystemExit(
                f"FATAL: corrupt row {path}:{lineno}. Two writers probably shared this file.")
        except KeyError:
            continue
    return done


def grade_example(example_id: str, rows: list[dict], args, emit) -> dict[str, int]:
    """Grade every SDEdit candidate for one example, then relax+score a shortlist."""
    import pyrosetta

    tally = {"graded": 0, "declared": 0, "abstained": 0, "shortlisted": 0, "accepted": 0,
             "skipped": 0}

    inputs = {r.get("input_complex_pdb") for r in rows if r.get("input_complex_pdb")}
    if len(inputs) != 1:
        emit({"row_key": f"{example_id}|ambiguous_input", "status": "ambiguous_input",
              "example_id": example_id, "error": f"{len(inputs)} distinct input complexes"})
        return tally
    input_pdb = inputs.pop()
    binder_chain = rows[0].get("binder_chain", "B")
    if not Path(input_pdb).is_file():
        emit({"row_key": f"{example_id}|missing_input", "status": "missing_input",
              "example_id": example_id, "error": f"input complex not found: {input_pdb}"})
        return tally

    try:
        target = S.load_target(example_id, input_pdb, binder_chain)
    except Exception as exc:                                          # noqa: BLE001
        emit({"row_key": f"{example_id}|load_failed", "status": "load_failed",
              "example_id": example_id, "error": f"{type(exc).__name__}: {exc}"[:300]})
        return tally

    pool: dict[str, list] = defaultdict(list)
    for r in rows:
        cpx = r.get("complex_pdb")
        tag = r.get("pdb_tag", "?")
        chem_name = r.get("cyc_type")
        if not cpx or not Path(cpx).is_file():
            emit({"row_key": f"{example_id}|{tag}|missing", "status": "missing_complex",
                  "example_id": example_id, "pdb_tag": tag, "chemistry": chem_name,
                  "error": f"complex not found: {cpx}"})
            tally["skipped"] += 1
            continue
        if chem_name not in T.CHEMISTRIES:
            emit({"row_key": f"{example_id}|{tag}|badchem", "status": "unknown_chemistry",
                  "example_id": example_id, "pdb_tag": tag, "chemistry": chem_name})
            tally["skipped"] += 1
            continue
        chem = T.chemistry(chem_name)

        try:
            pose = pyrosetta.pose_from_pdb(cpx)
            bp = T.binder_positions(pose, binder_chain)
        except Exception as exc:                                      # noqa: BLE001
            emit({"row_key": f"{example_id}|{tag}|loadfail", "status": "candidate_load_failed",
                  "example_id": example_id, "pdb_tag": tag, "chemistry": chem_name,
                  "error": f"{type(exc).__name__}: {exc}"[:300]})
            tally["skipped"] += 1
            continue

        if len(bp) != target.length:
            # Peptide length is held fixed for the primary comparison, so this is a data
            # problem rather than a candidate. Recorded, not silently dropped.
            emit({"row_key": f"{example_id}|{tag}|len", "status": "length_mismatch",
                  "example_id": example_id, "pdb_tag": tag, "chemistry": chem_name,
                  "error": f"candidate L={len(bp)} vs reference L={target.length}"})
            tally["skipped"] += 1
            continue

        i_pos, j_pos = bp[0], bp[-1]
        aa_i, aa_j = pose.residue(i_pos).name3(), pose.residue(j_pos).name3()
        spec = SDEditSpec(chemistry=chem_name, aa_i=aa_i, aa_j=aa_j, pdb_tag=tag,
                          t_ca=float(r.get("t_ca_start", float("nan"))),
                          t_lat=float(r.get("t_lat_start", float("nan"))),
                          seed=int(r.get("seed", -1)))
        window = whole_peptide_window(bp)
        atoms = None
        declared = False
        declare_error = ""
        if anchors_competent(chem, aa_i, aa_j):
            try:
                atoms = T.resolve_ring_atoms(chem, aa_i, aa_j)
                T.declare_ring_bond(pose, i_pos, j_pos, chem, atoms)
                declared = True
            except Exception as exc:                                  # noqa: BLE001
                # A disulfide whose SG atoms are far apart is REFUSED by Rosetta; that is a
                # true negative for this candidate, not an error in the grading.
                declare_error = f"{type(exc).__name__}: {exc}"[:200]
        if atoms is None:
            atoms = T.resolve_ring_atoms(chem, *(chem.anchor_options[0]
                                                 if chem.anchor_options else (aa_i, aa_j)))

        try:
            row = E.evaluate_candidate(
                target, pose, spec, atoms, window, i_pos, j_pos,
                stage="pre_relax", protected_local=args.protected_list, with_energy=False)
        except Exception as exc:                                      # noqa: BLE001
            emit({"row_key": f"{example_id}|{tag}|gradefail", "status": "grade_failed",
                  "example_id": example_id, "pdb_tag": tag, "chemistry": chem_name,
                  "error": f"{type(exc).__name__}: {exc}"[:300]})
            tally["skipped"] += 1
            continue

        row.update({
            "status": "ok", "arm": "sdedit", "engine": "sdedit",
            "pdb_tag": tag, "t_ca_start": spec.t_ca, "t_lat_start": spec.t_lat,
            "seed": spec.seed, "source_complex_pdb": cpx,
            "anchors_competent": bool(anchors_competent(chem, aa_i, aa_j)),
            # The run's own abstention flag, carried through so the two definitions can be
            # compared rather than conflated.
            "sdedit_requested_type_satisfied": r.get("requested_type_satisfied"),
            "sdedit_contact_retention_legacy": r.get("contact_retention"),
            "declare_error": declare_error,
            # SDEdit resamples the whole peptide, so there is no mutation BUDGET to report --
            # the substitution count against the original LP is the measurement.
            "mutation_budget": None,
        })
        row["row_key"] = f"{example_id}|{tag}|pre"
        emit(row)
        tally["graded"] += 1
        if declared:
            tally["declared"] += 1
        if not row.get("requested_chemistry_satisfied"):
            tally["abstained"] += 1

        # Only a candidate with a real declared bond can ever be accepted, so only those are
        # worth the cost of relaxation. Everything else already has its verdict.
        if declared:
            pool[chem_name].append((E.shortlist_key(row), row, pose, atoms, window,
                                    i_pos, j_pos, spec, chem))

    if not any(pool.values()):
        return tally

    try:
        lp_baseline = E.lp_baseline_metrics(target, repeats=args.lp_repeats,
                                            relax_repeats=args.relax_repeats)
    except Exception as exc:                                          # noqa: BLE001
        emit({"row_key": f"{example_id}|lp_baseline_failed", "status": "lp_baseline_failed",
              "example_id": example_id, "error": f"{type(exc).__name__}: {exc}"[:300]})
        return tally
    emit({"row_key": f"{example_id}|lp_baseline", "status": "lp_baseline", "arm": "sdedit",
          "example_id": example_id, "peptide_length": target.length,
          "orig_sequence": target.orig_sequence,
          "n_contacts_orig_heavy45": len(target.orig_contacts_heavy),
          "n_interface_residues": len(target.interface_local), **lp_baseline})

    for chem_name, entries in pool.items():
        entries.sort(key=lambda e: e[0])
        for rank, (_, pre_row, pose, atoms, window, i_pos, j_pos, spec, chem) in enumerate(
                entries[: args.shortlist]):
            # Seeded from the tag, not Python's hash(): PYTHONHASHSEED randomizes str hashing
            # per process, so a resumed shard would relax from different seeds than the run it
            # is continuing and its repeats would not be reproducible.
            seed_base = (int.from_bytes(spec.pdb_tag.encode(), "little", signed=False)
                         % 1_000_000) + 500_000
            try:
                pose, cp_stats = E.refine_candidate_repeats(
                    pose, target, i_pos, j_pos, chem, atoms,
                    repeats=args.cp_repeats, relax_repeats=args.relax_repeats,
                    seed_base=seed_base)
            except Exception as exc:                                  # noqa: BLE001
                emit({"row_key": f"{example_id}|{spec.pdb_tag}|relaxfail",
                      "status": "relax_failed", "example_id": example_id,
                      "pdb_tag": spec.pdb_tag, "chemistry": chem_name,
                      "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            try:
                post = E.evaluate_candidate(
                    target, pose, spec, atoms, window, i_pos, j_pos,
                    stage="post_relax", protected_local=args.protected_list,
                    lp_baseline=lp_baseline, with_energy=True)
            except Exception as exc:                                  # noqa: BLE001
                emit({"row_key": f"{example_id}|{spec.pdb_tag}|scorefail",
                      "status": "score_failed", "example_id": example_id,
                      "pdb_tag": spec.pdb_tag, "chemistry": chem_name,
                      "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            post.update(cp_stats)
            post.update({"status": "ok", "arm": "sdedit", "engine": "sdedit",
                         "pdb_tag": spec.pdb_tag, "t_ca_start": spec.t_ca,
                         "t_lat_start": spec.t_lat, "seed": spec.seed,
                         "shortlist_rank": rank, "mutation_budget": None,
                         "source_complex_pdb": pre_row.get("source_complex_pdb")})
            for key in ("bond_length_A", "bond_ok", "ca_rmsd_A", "interface_ca_rmsd_A",
                        "contact_retention_heavy45", "clashes_ok",
                        "n_severe_clashes_interface"):
                post[f"pre_{key}"] = pre_row.get(key)
            post["row_key"] = f"{example_id}|{spec.pdb_tag}|post"
            tally["shortlisted"] += 1
            if post.get("accepted"):
                tally["accepted"] += 1
            if args.pdb_dir and (post.get("accepted") or rank < args.dump_top):
                out = Path(args.pdb_dir)
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"{spec.pdb_tag}__relaxed.pdb"
                pose.dump_pdb(str(path))
                post["candidate_pdb"] = str(path)
            emit(post)
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edits", nargs="+", required=True,
                    help="edits_shard*.jsonl files, or a run directory containing them.")
    ap.add_argument("--out", required=True, help="Output JSONL for THIS shard (appended).")
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--examples", nargs="+", default=None)
    ap.add_argument("--chemistries", nargs="+", default=list(T.CHEMISTRIES))
    ap.add_argument("--protected", default=None,
                    help="0-based local indices that must not be mutated, e.g. '3,5-7'.")
    ap.add_argument("--shortlist", type=int, default=8,
                    help="Candidates per (example, chemistry) that get relaxed and scored.")
    ap.add_argument("--dump-top", type=int, default=2)
    ap.add_argument("--relax-repeats", type=int, default=1,
                    help="FastRelax cycles within one trajectory (not independent samples).")
    ap.add_argument("--cp-repeats", type=int, default=1,
                    help="Independent relaxation/scoring runs per shortlisted candidate.")
    ap.add_argument("--lp-repeats", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--time-budget-s", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0, {args.shard_count})")

    files: list[Path] = []
    for spec in args.edits:
        p = Path(spec)
        if p.is_dir():
            files.extend(sorted(p.glob("edits_shard*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"FATAL: --edits path not found: {spec}")
    if not files:
        raise SystemExit(f"FATAL: no edits_shard*.jsonl found under {args.edits}")

    rows = load_edits(files)
    rows = [r for r in rows if r.get("cyc_type") in set(args.chemistries)]
    if args.examples:
        want = set(args.examples)
        rows = [r for r in rows if str(r.get("example_id")) in want]
    by_example: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_example[str(r.get("example_id"))].append(r)

    examples = sorted(by_example)
    mine = examples[args.shard_index::args.shard_count]
    print(f"{len(files)} edits file(s), {len(rows)} rows, {len(examples)} examples; "
          f"shard {args.shard_index}/{args.shard_count} takes {len(mine)}", flush=True)

    if args.dry_run:
        for ex in mine:
            chems = sorted({r.get("cyc_type") for r in by_example[ex]})
            print(f"  {ex}  {len(by_example[ex]):5d} candidates  {chems}", flush=True)
        print("DRY RUN OK: ran no Rosetta.", flush=True)
        return 0

    # Parsed once, up front: a bad --protected spec should fail before Rosetta starts, not
    # per example. Length is unknown here, so an index is validated against each example's
    # own length inside the grader via cp_contract's protected bookkeeping.
    args.protected_list = []
    if args.protected:
        for part in args.protected.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                args.protected_list.extend(range(int(lo), int(hi) + 1))
            else:
                args.protected_list.append(int(part))
        args.protected_list = sorted(set(args.protected_list))

    from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init
    _ensure_pyrosetta_init()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out_path)
    if done:
        print(f"resuming: {len(done)} rows already written", flush=True)

    t_start = time.time()
    totals = {"graded": 0, "declared": 0, "abstained": 0, "shortlisted": 0, "accepted": 0,
              "skipped": 0}
    with out_path.open("a") as fh:
        def emit(row: dict) -> None:
            key = row.get("row_key")
            if key and key in done:
                return
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            if key:
                done.add(key)

        for ex in mine:
            if args.time_budget_s and (time.time() - t_start) > args.time_budget_s:
                print(f"time budget reached; stopping before {ex}. Resubmit to continue.",
                      flush=True)
                break
            t0 = time.time()
            tally = grade_example(ex, by_example[ex], args, emit)
            for k, v in tally.items():
                totals[k] += v
            print(f"  {ex} graded={tally['graded']} declared={tally['declared']} "
                  f"abstained={tally['abstained']} shortlisted={tally['shortlisted']} "
                  f"accepted={tally['accepted']} skipped={tally['skipped']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"done in {time.time() - t_start:.0f}s: {totals} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
