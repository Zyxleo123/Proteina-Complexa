"""Rosetta search for a cyclic version of a bound linear peptide that keeps its pose.

What this does
--------------
For each staged LP complex and each requested chemistry, holds the receptor fixed, releases a
small window of residues around the two termini, and searches for closed conformations through
the intended terminal bond -- GenKIC for head-to-tail, a constrained free-terminus anneal for
the side-chain bridges (see `script_utils/cp_search`). It expands the window step by step and
records exactly which residues each rung released. Sequence identities are fixed within a
search; the required anchor identities are enumerated, and any further mutations come from an
explicit budget with every substitution counted against the ORIGINAL LP.

Quality, not yield. The search does not stop at the first closed structure: it keeps searching
to improve retention and energy, ranks what it finds by the contract's own criteria
(`cp_evaluate.shortlist_key`), and only a shortlist pays for packing, relaxation and interface
scoring. Acceptance is applied to the FINAL RELAXED candidate, with the pre-relax row recorded
beside it.

Cost model
----------
Closure attempts are cheap (GenKIC: ~2000 attempts in well under a second on an 11-mer);
relaxation is not (seconds to minutes per complex). So the run is two-phase inside one job:

    phase 1   search        `--attempts-per-call` x `--calls` per (chemistry, window, anchors)
    phase 2   shortlist     `--shortlist` best candidates per (chemistry, window) get packed,
                            relaxed and interface-scored against a paired LP baseline

Resumability
------------
One JSONL per shard, appended and flushed per row, with a `row_key` that is unique per
(example, spec, seed, stage). On restart the file is read and finished keys are skipped, so
hitting a time limit means resubmitting rather than restarting. Never share an output file
between concurrent jobs: concurrent O_APPEND has NUL-corrupted rows on this cluster (88 of 144
lost), so each shard writes its own file and the aggregator refuses corrupt lines instead of
skipping them.

Chemically impossible combinations are written out as rows with `status="infeasible_budget"`
rather than omitted, so coverage reporting can distinguish "the search failed" from "this
chemistry cannot exist at this budget" -- a terminal disulfide needs Cys at both termini and so
costs 2 mutations, which makes budgets 0 and 1 empty by chemistry on almost every LP.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "script_utils"))

import cp_contract as C            # noqa: E402
import cp_evaluate as E           # noqa: E402
import cp_search as S             # noqa: E402
import cp_topology as T           # noqa: E402

CHEMISTRIES = ("mainchain", "disulfide", "isopeptide")


def parse_protected(spec: str | None, length: int) -> list[int]:
    """Parse `--protected` (0-based local indices, commas and ranges) into a sorted list.

    Protected residues may not be mutated by anything, anchors included. An index outside the
    peptide is a FATAL error rather than a silent no-op: a typo here would quietly remove the
    protection the caller asked for.
    """
    if not spec:
        return []
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            rng = range(int(lo), int(hi) + 1)
        else:
            rng = [int(part)]
        for k in rng:
            if not 0 <= k < length:
                raise SystemExit(
                    f"FATAL: --protected index {k} outside the peptide (length {length})")
            out.add(k)
    return sorted(out)


def done_keys(path: Path) -> set[str]:
    """Row keys already written. Refuses a corrupt line rather than skipping it."""
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
                f"FATAL: corrupt row {path}:{lineno}. Two writers probably shared this file; "
                "do not trust this run.")
        except KeyError:
            continue
    return done


def specs_for(target: S.Target, chem_name: str, budget: int, protected: list[int],
              max_released: int | None) -> list[S.SearchSpec]:
    """Every (anchors, window) point to search for one chemistry at one mutation budget.

    Anchor options come from `cp_topology.anchor_mutation_cost`, cheapest first, filtered to
    those affordable within `budget` and those that do not touch a protected residue. The
    window ladder is ordered smallest-first so the best-retaining answers are found before the
    search starts releasing more of the peptide.
    """
    chem = T.chemistry(chem_name)
    out: list[S.SearchSpec] = []
    ladder = T.window_ladder(target.length, max_released=max_released)
    for opt in T.anchor_mutation_cost(target.native_aa_i, target.native_aa_j, chem):
        if opt["n_anchor_mutations"] > budget:
            continue
        # An anchor mutation at a protected terminus is not allowed.
        if 0 in protected and opt["aa_i"] != target.native_aa_i:
            continue
        if (target.length - 1) in protected and opt["aa_j"] != target.native_aa_j:
            continue
        for (k, m) in ladder:
            out.append(S.SearchSpec(chemistry=chem_name, aa_i=opt["aa_i"], aa_j=opt["aa_j"],
                                    k_nterm=k, m_cterm=m))
    return out


def run_target(target: S.Target, args, emit) -> dict[str, int]:
    """Search one target across the requested chemistries. Returns a small tally."""
    protected = parse_protected(args.protected, target.length)
    tally = {"searched": 0, "closed": 0, "shortlisted": 0, "accepted": 0, "infeasible": 0}
    lp_baseline: dict | None = None

    for chem_name in args.chemistries:
        chem = T.chemistry(chem_name)
        min_cost = T.min_anchor_cost(target.native_aa_i, target.native_aa_j, chem)
        # A larger budget only widens the search if it admits anchor options a smaller one could
        # not afford. Head-to-tail costs 0 anchor mutations, so budgets 0/1/2 enumerate the
        # IDENTICAL spec set and searching each would triple the work for literally the same
        # candidates. The duplicate budget is recorded as such -- it still counts as covered in
        # the tier grid, it simply is not re-searched.
        #
        # Note what this makes explicit: this arm spends the budget on ANCHOR identities only.
        # `SearchSpec.extra_mutations` exists and is counted, but no additional non-anchor
        # mutations are enumerated, so a budget above a chemistry's anchor cost buys nothing
        # here. That is a limitation of the search, not a property of the chemistry.
        searched_signatures: dict[frozenset, int] = {}
        for budget in args.budgets:
            if min_cost > budget:
                key = f"{target.example_id}|{chem_name}|budget{budget}|infeasible"
                emit({
                    "row_key": key, "status": "infeasible_budget",
                    "example_id": target.example_id, "chemistry": chem_name,
                    "mutation_budget": budget,
                    "min_anchor_mutations": min_cost,
                    "native_anchor_aa_i": target.native_aa_i,
                    "native_anchor_aa_j": target.native_aa_j,
                    "peptide_length": target.length,
                    "reason": (f"{chem_name} needs {min_cost} anchor mutation(s) on "
                               f"{target.native_aa_i}/{target.native_aa_j} termini; "
                               f"budget {budget} cannot express it"),
                })
                tally["infeasible"] += 1
                continue

            specs = specs_for(target, chem_name, budget, protected, args.max_released)
            if not specs:
                continue
            signature = frozenset(s.label() for s in specs)
            if signature in searched_signatures:
                prior = searched_signatures[signature]
                emit({
                    "row_key": f"{target.example_id}|{chem_name}|budget{budget}|duplicate",
                    "status": "duplicate_budget", "example_id": target.example_id,
                    "chemistry": chem_name, "mutation_budget": budget,
                    "same_as_budget": prior, "n_specs": len(specs),
                    "reason": (f"budget {budget} admits exactly the same {len(specs)} specs as "
                               f"budget {prior} ({chem_name} needs {min_cost} anchor "
                               f"mutation(s), and no non-anchor mutations are enumerated), so "
                               f"searching it again would reproduce identical candidates"),
                })
                continue
            searched_signatures[signature] = budget
            pool: list[tuple[tuple, dict, object, dict, dict, int, int, S.SearchSpec]] = []

            for spec in specs:
                for call in range(args.calls):
                    seed = args.seed + 1000 * call + 7 * (spec.k_nterm * 13 + spec.m_cterm)
                    S.set_rng_seed(seed)
                    try:
                        prep = S.prepare(target, spec)
                    except ValueError as exc:
                        emit({"row_key": f"{target.example_id}|{spec.label()}|prep{call}",
                              "status": "prepare_failed", "example_id": target.example_id,
                              "chemistry": chem_name, "spec_label": spec.label(),
                              "error": str(exc)[:300]})
                        continue
                    pose, atoms, window = prep["pose"], prep["atoms"], prep["window"]
                    i_pos, j_pos = prep["i_pos"], prep["j_pos"]
                    tally["searched"] += 1

                    if chem.genkic_closable:
                        ok = S.genkic_close(pose, window, i_pos, j_pos, chem, atoms,
                                            S.constrained_score_function(cst_weight=0.0),
                                            attempts=args.attempts_per_call,
                                            min_solutions=args.min_solutions)
                    else:
                        ok = S.bridge_anneal(pose, target, window, i_pos, j_pos, chem, atoms,
                                             perturb_deg=args.perturb_deg)
                    if not ok:
                        continue
                    tally["closed"] += 1

                    row = E.evaluate_candidate(
                        target, pose, spec, atoms, window, i_pos, j_pos,
                        stage="pre_relax", protected_local=protected, with_energy=False)
                    row.update({"mutation_budget": budget, "seed": seed, "call": call,
                                "status": "ok", "engine":
                                    "genkic" if chem.genkic_closable else "bridge_anneal"})
                    row["row_key"] = (f"{target.example_id}|{spec.label()}|b{budget}"
                                      f"|s{seed}|pre")
                    emit(row)
                    pool.append((E.shortlist_key(row), row, pose, atoms, window,
                                 i_pos, j_pos, spec))

            # ---- phase 2: relax and energy-score the best of this (chemistry, budget) -----
            if not pool:
                continue
            pool.sort(key=lambda entry: entry[0])
            chosen = pool[: args.shortlist]
            if lp_baseline is None:
                # The paired baseline is what makes ddG a PAIRED comparison; without it every
                # candidate for this target would be scored against nothing. So a failure here
                # is recorded and the target abandoned, rather than letting the shortlist emit
                # rows whose ddG is silently NaN and whose acceptance therefore silently fails.
                try:
                    lp_baseline = E.lp_baseline_metrics(
                        target, repeats=args.lp_repeats, relax_repeats=args.relax_repeats)
                except Exception as exc:                                   # noqa: BLE001
                    emit({"row_key": f"{target.example_id}|lp_baseline_failed",
                          "status": "lp_baseline_failed", "example_id": target.example_id,
                          "error": f"{type(exc).__name__}: {exc}"[:300]})
                    return tally
                emit({"row_key": f"{target.example_id}|lp_baseline", "status": "lp_baseline",
                      "example_id": target.example_id, "peptide_length": target.length,
                      "orig_sequence": target.orig_sequence,
                      "n_contacts_orig_heavy45": len(target.orig_contacts_heavy),
                      "n_interface_residues": len(target.interface_local),
                      **lp_baseline})

            for rank, (_, pre_row, pose, atoms, window, i_pos, j_pos, spec) in enumerate(chosen):
                S.set_rng_seed(int(pre_row["seed"]) + 500_000)
                # Both the relax and the scoring can fail on a single candidate -- Rosetta
                # surfaces some C++ UtilityExitExceptions as Python RuntimeError, and
                # InterfaceAnalyzerMover has its own failure modes. Catching broadly here is
                # what keeps one bad candidate from costing the whole TARGET: the shard is
                # resumable per row_key, and a target that died mid-way would otherwise be
                # re-searched from scratch on every resubmission. The failure is written out
                # as a row, so it stays visible instead of thinning the table silently.
                try:
                    pose, cp_stats = E.refine_candidate_repeats(
                        pose, target, i_pos, j_pos, chem, atoms,
                        repeats=args.cp_repeats, relax_repeats=args.relax_repeats,
                        seed_base=int(pre_row["seed"]) + 500_000)
                except Exception as exc:                                   # noqa: BLE001
                    emit({"row_key": pre_row["row_key"].replace("|pre", "|relaxfail"),
                          "status": "relax_failed", "example_id": target.example_id,
                          "chemistry": chem_name, "spec_label": spec.label(),
                          "error": f"{type(exc).__name__}: {exc}"[:300]})
                    continue
                try:
                    post = E.evaluate_candidate(
                        target, pose, spec, atoms, window, i_pos, j_pos,
                        stage="post_relax", protected_local=protected,
                        lp_baseline=lp_baseline, with_energy=True)
                    post.update(cp_stats)
                except Exception as exc:                                   # noqa: BLE001
                    emit({"row_key": pre_row["row_key"].replace("|pre", "|scorefail"),
                          "status": "score_failed", "example_id": target.example_id,
                          "chemistry": chem_name, "spec_label": spec.label(),
                          "error": f"{type(exc).__name__}: {exc}"[:300]})
                    continue
                post.update({"mutation_budget": budget, "seed": pre_row["seed"],
                             "call": pre_row["call"], "status": "ok",
                             "shortlist_rank": rank,
                             "engine": pre_row["engine"]})
                # carry the pre-relax values so "only acceptable after relax" is visible
                for key in ("bond_length_A", "bond_ok", "ca_rmsd_A", "interface_ca_rmsd_A",
                            "contact_retention_heavy45", "clashes_ok",
                            "n_severe_clashes_interface"):
                    post[f"pre_{key}"] = pre_row.get(key)
                post["row_key"] = pre_row["row_key"].replace("|pre", "|post")
                tally["shortlisted"] += 1
                if post.get("accepted"):
                    tally["accepted"] += 1
                if args.pdb_dir and (post.get("accepted") or rank < args.dump_top):
                    out = Path(args.pdb_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    path = out / (f"{target.example_id}__{spec.label()}__b{budget}"
                                  f"__s{pre_row['seed']}.pdb")
                    pose.dump_pdb(str(path))
                    post["candidate_pdb"] = str(path)
                emit(post)
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="Staged LP parquet (e.g. lnr_staged).")
    ap.add_argument("--out", required=True, help="Output JSONL for THIS shard (appended).")
    ap.add_argument("--pdb-dir", default=None, help="Where to write candidate structures.")
    ap.add_argument("--examples", nargs="+", default=None,
                    help="Restrict to these example_ids. An id absent from the metadata is a "
                         "FATAL error, not a silent empty run.")
    ap.add_argument("--chemistries", nargs="+", default=list(CHEMISTRIES),
                    choices=CHEMISTRIES)
    ap.add_argument("--budgets", nargs="+", type=int, default=[2],
                    help="Mutation budgets to search (total substitutions vs the original LP).")
    ap.add_argument("--protected", default=None,
                    help="0-based local indices that must not be mutated, e.g. '3,5-7'.")
    ap.add_argument("--attempts-per-call", type=int, default=2000,
                    help="GenKIC closure attempts per call.")
    ap.add_argument("--min-solutions", type=int, default=8,
                    help="Closures GenKIC collects per call before the lowest-energy selector "
                         "picks one. At 1 it stops at the FIRST closure and the selector has "
                         "nothing to choose from -- this is the quality-over-yield lever.")
    ap.add_argument("--calls", type=int, default=4,
                    help="Independent search calls per (anchors, window). Total closure "
                         "attempts per chemistry is roughly calls * attempts * |ladder|.")
    ap.add_argument("--max-released", type=int, default=None,
                    help="Cap on released residues; default is the peptide's own limit (L-2).")
    ap.add_argument("--perturb-deg", type=float, default=45.0,
                    help="Torsion perturbation amplitude for the bridge-anneal engine.")
    ap.add_argument("--shortlist", type=int, default=8,
                    help="Candidates per (chemistry, budget) that get relaxed and scored.")
    ap.add_argument("--dump-top", type=int, default=2,
                    help="Also write structures for this many top-ranked non-accepted rows.")
    ap.add_argument("--relax-repeats", type=int, default=1,
                    help="FastRelax cycles WITHIN one relaxation trajectory (not independent "
                         "samples of it -- see --cp-repeats).")
    ap.add_argument("--cp-repeats", type=int, default=1,
                    help="Independent relaxation/scoring runs per shortlisted CANDIDATE. The "
                         "contract asks for 3; the median feeds the paired delta and the "
                         "spread says whether that delta exceeds relax noise.")
    ap.add_argument("--lp-repeats", type=int, default=1,
                    help="Independent LP baseline relaxations (median feeds the paired delta).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--time-budget-s", type=float, default=0.0,
                    help="Stop starting new targets after this many seconds (0 = no limit). "
                         "Resume by resubmitting; finished rows are skipped.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve targets and print the search plan without running Rosetta.")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0, {args.shard_count})")

    import pandas as pd

    meta = pd.read_parquet(args.metadata)
    if args.examples:
        want, have = set(args.examples), set(meta["example_id"].astype(str))
        missing = want - have
        if missing:
            raise SystemExit(f"FATAL: --examples not in metadata: {sorted(missing)}")
        meta = meta[meta["example_id"].astype(str).isin(want)]
    meta = meta.sort_values("example_id").reset_index(drop=True)
    rows = meta.iloc[args.shard_index::args.shard_count]
    print(f"{len(meta)} targets; shard {args.shard_index}/{args.shard_count} takes {len(rows)}",
          flush=True)

    # State the search effort explicitly, per LP and per chemistry, rather than leaving it
    # implicit in four interacting knobs. `attempts` is an upper bound: GenKIC stops once it has
    # `min_solutions` closures, so an easy window spends a fraction of the budget and a hard one
    # is capped at it. The printed number is therefore the ceiling, not the expected cost.
    for _, m in rows.head(1).iterrows():
        L = int(m["peptide_length"])
        n_rungs = len(T.window_ladder(L, max_released=args.max_released))
        per_anchor = args.attempts_per_call * args.calls * n_rungs
        print(f"search effort ceiling (L={L} example): {n_rungs} window rungs x {args.calls} "
              f"calls x {args.attempts_per_call} attempts = {per_anchor:,} closure attempts per "
              f"anchor option per budget; GenKIC collects {args.min_solutions} closures per call "
              f"before selecting.", flush=True)

    if args.dry_run:
        for _, m in rows.iterrows():
            L = int(m["peptide_length"])
            n_rungs = len(T.window_ladder(L, max_released=args.max_released))
            print(f"  {m['example_id']}  L={L}  chain={m['binder_chain_id']}"
                  f"  nc_gap={m.get('nc_gap_angstrom', float('nan')):.1f}"
                  f"  rungs={n_rungs}"
                  f"  ceiling={args.attempts_per_call * args.calls * n_rungs:,}/anchor/budget"
                  f"  chemistries={args.chemistries} budgets={args.budgets}", flush=True)
        print("DRY RUN OK: ran no Rosetta.", flush=True)
        return 0

    from proteinfoundation.evaluation.rosetta_energy import _ensure_pyrosetta_init
    _ensure_pyrosetta_init()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out_path)
    if done:
        print(f"resuming: {len(done)} rows already written", flush=True)

    t_start = time.time()
    totals = {"searched": 0, "closed": 0, "shortlisted": 0, "accepted": 0, "infeasible": 0}
    with out_path.open("a") as fh:
        def emit(row: dict) -> None:
            key = row.get("row_key")
            if key and key in done:
                return
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            if key:
                done.add(key)

        for _, m in rows.iterrows():
            if args.time_budget_s and (time.time() - t_start) > args.time_budget_s:
                print(f"time budget reached; stopping before {m['example_id']}. "
                      "Resubmit to continue.", flush=True)
                break
            example_id = str(m["example_id"])
            pdb_path = str(m["path"])
            if not Path(pdb_path).is_file():
                print(f"  SKIP {example_id}: staged PDB missing ({pdb_path})", flush=True)
                continue
            t0 = time.time()
            try:
                target = S.load_target(example_id, pdb_path, str(m["binder_chain_id"]))
            except Exception as exc:                                  # noqa: BLE001
                emit({"row_key": f"{example_id}|load_failed", "status": "load_failed",
                      "example_id": example_id, "error": str(exc)[:300]})
                print(f"  {example_id}: LOAD FAILED {exc}", flush=True)
                continue
            tally = run_target(target, args, emit)
            for k, v in tally.items():
                totals[k] += v
            print(f"  {example_id} L={target.length} seq={target.orig_sequence} "
                  f"searched={tally['searched']} closed={tally['closed']} "
                  f"shortlisted={tally['shortlisted']} accepted={tally['accepted']} "
                  f"infeasible={tally['infeasible']} ({time.time() - t0:.0f}s)", flush=True)

    print(f"done in {time.time() - t_start:.0f}s: {totals} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
