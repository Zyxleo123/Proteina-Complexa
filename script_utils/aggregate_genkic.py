"""Aggregate the GenKIC search rows into the arm's results tables.

Reads every `rows_*.jsonl` shard, grades nothing (the rows already carry their contract
verdicts), and reports:

    per-target      the best ACCEPTED candidate per LP, and the best near-miss for the rest
    unresolved      every LP with no accepted candidate, with the bar that bound it
    coverage        accepted / attempted / chemically-infeasible, per chemistry and budget
    tiers           mutation budgets 0/1/2/3 and identity 90/80/70, reported side by side
    effort          quality against search effort, so "more attempts" can be judged
    relax           pre- vs post-relax metrics, and shortlist repeat variability

Three reporting rules this module enforces, because each one is a way a result can look
better than it is:

1. **The denominator is every target in the metadata**, not every target with rows. A shard
   that died, a target that never loaded and a target the search genuinely failed all have
   to appear, or "accepted / targets-with-results" silently becomes the success rate.
2. **A weaker tier is never a primary success.** `pass_budget3` and `pass_identity70` are
   reported in their own columns next to the primary bar, never merged into it.
3. **Chemically infeasible is not a failure.** A terminal disulfide needs Cys at both
   termini, so budgets 0 and 1 cannot express it on almost any LP. Those cells are counted
   and labelled separately from "searched and did not close", and they stay visible in the
   coverage table rather than being dropped from the denominator.

A corrupt JSONL line is FATAL, never skipped: concurrent O_APPEND has NUL-corrupted rows on
this cluster (88 of 144 lost on one run), and a silent skip would turn that data loss into a
quietly smaller table.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PRIMARY_BUDGET_TIERS = (0, 1, 2, 3)
PRIMARY_IDENTITY_TIERS = (90, 80, 70)
CHEMISTRIES = ("mainchain", "disulfide", "isopeptide")


def load_rows(rows_dir: Path) -> list[dict]:
    """Every row from every shard. A corrupt line stops the aggregation."""
    # Search shards write rows_<n>.jsonl; the smoke writes a single rows.jsonl (either in the
    # directory given, or in a subdirectory of the run), so the same report can be cut over a
    # smoke and over a full run without a second code path.
    files = (sorted(rows_dir.glob("rows_*.jsonl"))
             + sorted(rows_dir.glob("rows.jsonl"))
             + sorted(rows_dir.glob("*/rows.jsonl")))
    if not files:
        raise SystemExit(
            f"FATAL: no rows_*.jsonl, rows.jsonl or */rows.jsonl under {rows_dir}")
    out: list[dict] = []
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"FATAL: corrupt row {path}:{lineno} ({exc}). Two writers probably shared "
                    "this file; do not trust this run. Re-run the affected shard."
                )
    print(f"loaded {len(out)} rows from {len(files)} shard file(s)")
    return out


def _num(row: dict, key: str, default=float("nan")) -> float:
    try:
        v = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return v


def _fmt(v, nd=3, width=0) -> str:
    if v is None or (isinstance(v, float) and v != v):
        s = "--"
    elif isinstance(v, float):
        s = f"{v:.{nd}f}"
    else:
        s = str(v)
    return s.rjust(width) if width else s


def quality_key(row: dict) -> tuple:
    """Rank accepted candidates: energy first, then pose fidelity, then fewest mutations.

    Unlike the search-stage `shortlist_key`, energy IS available here and IS the point --
    among candidates that all satisfy the contract, the best one is the one that costs the
    interface least.
    """
    return (
        _num(row, "ddg_interface_reu", 1e9),
        _num(row, "interface_ca_rmsd_A", 1e9),
        -_num(row, "contact_retention_heavy45", 0.0),
        _num(row, "n_substitutions", 99),
        _num(row, "ca_rmsd_A", 1e9),
    )


def nearmiss_key(row: dict) -> tuple:
    """Rank non-accepted candidates by how close they came: fewest failed criteria first."""
    return (
        len(row.get("fail_reasons") or []),
        _num(row, "ddg_interface_reu", 1e9),
        -_num(row, "contact_retention_heavy45", 0.0),
    )


def report(rows: list[dict], all_targets: list[str], out_dir: Path, sdedit_rows: list[dict]) -> None:
    post = [r for r in rows if r.get("stage") == "post_relax" and r.get("status") == "ok"]
    pre = [r for r in rows if r.get("stage") == "pre_relax" and r.get("status") == "ok"]
    infeasible = [r for r in rows if r.get("status") == "infeasible_budget"]
    broken = [r for r in rows if r.get("status") in
              ("load_failed", "prepare_failed", "relax_failed", "score_failed",
               "lp_baseline_failed")]

    by_target: dict[str, list[dict]] = defaultdict(list)
    for r in post:
        by_target[r["example_id"]].append(r)

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    # ------------------------------------------------------------------ run shape
    out("=" * 100)
    out("GenKIC LP -> CP search: results")
    out("=" * 100)
    out(f"targets in metadata          : {len(all_targets)}")
    out(f"targets with searched rows    : {len(by_target)}")
    out(f"pre-relax candidates          : {len(pre)}")
    out(f"post-relax scored candidates  : {len(post)}")
    out(f"chemically infeasible cells   : {len(infeasible)}")
    if broken:
        out(f"rows reporting a crash        : {len(broken)}  <-- investigate, not a result")
        for r in broken[:10]:
            out(f"    {r.get('status'):16s} {r.get('example_id')} {r.get('error', '')[:80]}")
    silent = [t for t in all_targets if t not in by_target]
    if silent:
        out(f"targets with NO rows at all   : {len(silent)}")
        out(f"    {', '.join(silent[:12])}{' ...' if len(silent) > 12 else ''}")
        out("    (a shard that died, or a target the search never reached -- these count as")
        out("     unresolved in every table below, they are not dropped from the denominator)")
    out()

    # ------------------------------------------------------------ primary headline
    accepted = [r for r in post if r.get("accepted")]
    accepted_strict = [r for r in post if r.get("accepted_strict_ddg")]
    targets_accepted = sorted({r["example_id"] for r in accepted})
    targets_strict = sorted({r["example_id"] for r in accepted_strict})
    n_all = len(all_targets)
    out("-" * 100)
    out("PRIMARY RESULT (contract as fixed: ddG <= +5 REU, >=80% identity, <=2 subs,")
    out("                >=80% contact retention, CA RMSD <=2 A, interface <=1 A)")
    out("-" * 100)
    out(f"  accepted candidates          : {len(accepted)}")
    out(f"  LPs with >=1 accepted        : {len(targets_accepted)} / {n_all} "
        f"({100.0 * len(targets_accepted) / n_all:.1f}% of all staged targets)")
    out(f"  LPs under the strict bar     : {len(targets_strict)} / {n_all} "
        f"({100.0 * len(targets_strict) / n_all:.1f}%)   [ddG <= 0 REU]")
    # How much of each peptide is interface decides how strict the 1 A interface bar really
    # is. Where every residue contacts the receptor, the interface bar (<=1 A) applies to the
    # whole peptide and therefore SUBSUMES the 2 A whole-peptide bar rather than complementing
    # it -- the binding criterion is then twice as strict as the table's headline suggests.
    fracs = [(_num(r, "n_interface_residues") / _num(r, "peptide_length"))
             for r in post
             if _num(r, "peptide_length") == _num(r, "peptide_length")
             and _num(r, "peptide_length") > 0]
    fracs = [f for f in fracs if f == f]
    if fracs:
        n_full = sum(1 for f in fracs if f >= 0.999)
        out(f"  interface residues / peptide : median {statistics.median(fracs):.2f} "
            f"({n_full}/{len(fracs)} candidates have EVERY residue at the interface)")
        if n_full:
            out("     where that fraction is 1.0 the <=1 A interface bar covers the whole")
            out("     peptide and subsumes the <=2 A whole-peptide bar -- the effective")
            out("     requirement there is 1 A on all residues, not 2 A.")
    out()

    # ------------------------------------------------------------ per-target table
    out("-" * 100)
    out("BEST CANDIDATE PER LP  (accepted first; otherwise the closest near-miss)")
    out("-" * 100)
    out(f"  {'example_id':22s} {'chem':10s} {'anch':7s} {'k/m':5s} {'bond':>5s} "
        f"{'ret':>5s} {'rmsd':>5s} {'irms':>5s} {'ddG':>8s} {'sub':>3s}  verdict")
    per_target_rows = []
    for tid in all_targets:
        cands = by_target.get(tid, [])
        if not cands:
            out(f"  {tid:22s} {'--':10s} {'--':7s} {'--':5s} {'--':>5s} {'--':>5s} "
                f"{'--':>5s} {'--':>5s} {'--':>8s} {'--':>3s}  NO ROWS")
            per_target_rows.append({"example_id": tid, "verdict": "no_rows"})
            continue
        ok = [r for r in cands if r.get("accepted")]
        best = min(ok, key=quality_key) if ok else min(cands, key=nearmiss_key)
        verdict = "ACCEPTED" if ok else "failed: " + ",".join(best.get("fail_reasons") or [])
        if ok and best.get("accepted_strict_ddg"):
            verdict = "ACCEPTED (strict)"
        out(f"  {tid:22s} {best.get('chemistry', '--'):10s} "
            f"{str(best.get('anchor_aa_i', ''))[:3]}/{str(best.get('anchor_aa_j', ''))[:3]:3s} "
            f"{best.get('k_nterm', '?')}/{best.get('m_cterm', '?')}  "
            f"{_fmt(_num(best, 'bond_length_A'), 2, 5)} "
            f"{_fmt(_num(best, 'contact_retention_heavy45'), 2, 5)} "
            f"{_fmt(_num(best, 'ca_rmsd_A'), 2, 5)} "
            f"{_fmt(_num(best, 'interface_ca_rmsd_A'), 2, 5)} "
            f"{_fmt(_num(best, 'ddg_interface_reu'), 2, 8)} "
            f"{_fmt(best.get('n_substitutions'), 0, 3)}  {verdict}")
        per_target_rows.append({
            "example_id": tid,
            "verdict": "accepted" if ok else "failed",
            "n_candidates": len(cands),
            "n_accepted": len(ok),
            **{k: best.get(k) for k in (
                "chemistry", "anchor_aa_i", "anchor_aa_j", "k_nterm", "m_cterm", "n_released",
                "released_local", "spec_label", "mutation_budget", "seed",
                "bond_length_A", "bond_ok", "bond_torsion_deg", "amide_is_cis",
                "contact_retention_heavy45", "contact_retention_ca10",
                "ca_rmsd_A", "interface_ca_rmsd_A", "interior_max_ca_dev_A",
                "n_substitutions", "seq_identity", "cand_sequence", "orig_sequence",
                "ddg_interface_reu", "cp_dG_separated", "lp_dG_separated_median",
                "lp_dG_separated_spread", "d_peptide_internal_reu", "cp_peptide_internal",
                "n_severe_clashes_interface", "n_chirality_flipped",
                "accepted", "accepted_strict_ddg", "fail_reasons", "candidate_pdb")},
        })
    out()

    # ------------------------------------------------------------------ unresolved
    unresolved = [r for r in per_target_rows if r["verdict"] != "accepted"]
    out("-" * 100)
    out(f"UNRESOLVED LPs: {len(unresolved)} / {n_all}")
    out("-" * 100)
    bound_by: Counter = Counter()
    for r in unresolved:
        if r["verdict"] == "no_rows":
            bound_by["no_rows"] += 1
            continue
        for reason in (r.get("fail_reasons") or ["unknown"]):
            bound_by[reason] += 1
    out("  which bar bound them (a candidate can fail several, all are counted):")
    for reason, n in bound_by.most_common():
        out(f"    {reason:28s} {n:4d}")
    out()

    # -------------------------------------------------------------------- coverage
    out("-" * 100)
    out("COVERAGE BY CHEMISTRY x MUTATION BUDGET")
    out("  searched = a closure was attempted;  infeasible = the chemistry cannot exist at")
    out("  that budget on that LP (counted, never dropped from the denominator)")
    out("-" * 100)
    dup = [r for r in rows if r.get("status") == "duplicate_budget"]
    budgets = sorted({int(r["mutation_budget"]) for r in post + infeasible + dup
                      if r.get("mutation_budget") is not None})
    out(f"  {'chemistry':12s} {'budget':>6s} {'targets':>8s} {'infeas':>7s} "
        f"{'cands':>6s} {'accept':>7s} {'LPs_ok':>7s}  note")
    for chem in CHEMISTRIES:
        for b in budgets:
            cands = [r for r in post if r.get("chemistry") == chem
                     and int(r.get("mutation_budget", -1)) == b]
            infs = [r for r in infeasible if r.get("chemistry") == chem
                    and int(r.get("mutation_budget", -1)) == b]
            dups = [r for r in dup if r.get("chemistry") == chem
                    and int(r.get("mutation_budget", -1)) == b]
            acc = [r for r in cands if r.get("accepted")]
            note = ""
            if dups and not cands:
                # NOT missing coverage: this budget admits the same specs as a cheaper one, so
                # its results are the cheaper budget's results. Reading the empty row as a gap
                # would understate coverage exactly where the chemistry is free.
                same = sorted({str(r.get("same_as_budget")) for r in dups})
                note = (f"same specs as budget {','.join(same)} -- not re-searched; "
                        f"its results carry over ({len(dups)} LPs)")
            out(f"  {chem:12s} {b:6d} {len({r['example_id'] for r in cands}):8d} "
                f"{len({r['example_id'] for r in infs}):7d} {len(cands):6d} {len(acc):7d} "
                f"{len({r['example_id'] for r in acc}):7d}  {note}")
    out("  A candidate found at budget b also satisfies every larger budget, which the tier")
    out("  grid below accounts for by counting substitutions rather than by re-searching.")
    out()

    # ----------------------------------------------------------------------- tiers
    out("-" * 100)
    out("TIER GRID (sensitivity, NOT a relabelling: every non-sequence criterion stays at")
    out("           its primary value in every cell)")
    out("-" * 100)
    out(f"  {'tier':22s} {'candidates':>11s} {'LPs':>5s}  note")
    prim_c, prim_t = len(accepted), len(targets_accepted)
    out(f"  {'PRIMARY (<=2, >=80%)':22s} {prim_c:11d} {prim_t:5d}  the contract")
    for b in PRIMARY_BUDGET_TIERS:
        hits = [r for r in post if r.get(f"pass_budget{b}")]
        flag = "  <-- primary budget" if b == 2 else ""
        out(f"  {'budget <= ' + str(b):22s} {len(hits):11d} "
            f"{len({r['example_id'] for r in hits}):5d}{flag}")
    for t in PRIMARY_IDENTITY_TIERS:
        hits = [r for r in post if r.get(f"pass_identity{t}")]
        flag = "  <-- primary identity" if t == 80 else ""
        out(f"  {'identity >= ' + str(t) + '%':22s} {len(hits):11d} "
            f"{len({r['example_id'] for r in hits}):5d}{flag}")
    out()

    # ---------------------------------------------------------------- pre vs post
    out("-" * 100)
    out("RELAXATION: what it changed (post-relax value minus pre-relax, medians over scored")
    out("            candidates). Acceptance is applied POST-relax.")
    out("-" * 100)
    for key in ("bond_length_A", "contact_retention_heavy45", "ca_rmsd_A",
                "interface_ca_rmsd_A"):
        deltas = [_num(r, key) - _num(r, f"pre_{key}") for r in post
                  if _num(r, key) == _num(r, key) and _num(r, f"pre_{key}") == _num(r, f"pre_{key}")]
        if deltas:
            out(f"  {key:28s} median delta {statistics.median(deltas):+8.3f}   "
                f"(n={len(deltas)}, range {min(deltas):+.3f} .. {max(deltas):+.3f})")
    only_after = [r for r in post if r.get("accepted") and not r.get("pre_bond_ok")]
    out(f"  accepted candidates whose ring geometry only became acceptable through relax: "
        f"{len(only_after)}")
    # The paired delta is only meaningful against the noise of the two things being subtracted,
    # so BOTH spreads are reported. A |ddG| inside them is not a measured degradation.
    for label, key in (("LP baseline", "lp_dG_separated_spread"),
                       ("CP candidate", "cp_dG_separated_spread")):
        spreads = [_num(r, key) for r in post]
        spreads = [s for s in spreads if s == s]
        if spreads:
            out(f"  {label} dG spread across independent relax repeats: "
                f"median {statistics.median(spreads):.2f} REU (max {max(spreads):.2f})")
    reps = [_num(r, "cp_n_relax_scored") for r in post]
    reps = [x for x in reps if x == x]
    if reps and max(reps) <= 1:
        out("  NOTE: only ONE relax repeat per candidate in this run, so the CP spread is 0 by")
        out("        construction and no ddG can be tested against relax noise. The contract")
        out("        asks for 3 -- relaunch with CP_REPEATS=3 before quoting a ddG as real.")
    noisy = [r for r in post
             if _num(r, "ddg_interface_reu") == _num(r, "ddg_interface_reu")
             and _num(r, "cp_dG_separated_spread", 0.0) + _num(r, "lp_dG_separated_spread", 0.0)
             >= abs(_num(r, "ddg_interface_reu", 0.0))]
    if post:
        out(f"  candidates whose |ddG| does NOT exceed the combined repeat spread: "
            f"{len(noisy)}/{len(post)} -- for these the energy criterion is not resolvable")
    out()

    # ---------------------------------------------------------------------- effort
    out("-" * 100)
    out("QUALITY vs SEARCH EFFORT (by released-residue count: the ladder's own expansion)")
    out("-" * 100)
    out(f"  {'released':>8s} {'cands':>6s} {'accept':>7s} {'med_ret':>8s} {'med_irms':>9s} "
        f"{'med_ddG':>8s}")
    for n_rel in sorted({int(r["n_released"]) for r in post if r.get("n_released") is not None}):
        cands = [r for r in post if int(r.get("n_released", -1)) == n_rel]
        acc = [r for r in cands if r.get("accepted")]

        def med(key):
            vals = [_num(r, key) for r in cands]
            vals = [v for v in vals if v == v]
            return statistics.median(vals) if vals else float("nan")

        out(f"  {n_rel:8d} {len(cands):6d} {len(acc):7d} "
            f"{_fmt(med('contact_retention_heavy45'), 3, 8)} "
            f"{_fmt(med('interface_ca_rmsd_A'), 3, 9)} {_fmt(med('ddg_interface_reu'), 2, 8)}")
    out("  (the ladder is ordered smallest-window-first, so a result at a LOW released count")
    out("   is strictly better: less of the peptide was disturbed to get it)")
    out()

    # ------------------------------------------------------------ SDEdit comparison
    out("-" * 100)
    out("COMPARISON: SDEdit outputs under the SAME contract")
    out("-" * 100)
    if not sdedit_rows:
        out("  nothing graded -- pass --sdedit-rows to include the SDEdit arm.")
        out("  This cell is deliberately empty rather than absent: a one-sided table would")
        out("  read as a comparison that had been made.")
    else:
        # Filter by STAGE as well as status: the SDEdit arm emits a cheap pre-relax row for
        # every candidate and a post-relax row only for the shortlist, so counting both
        # together would inflate its candidate count by roughly the shortlist ratio and
        # compare its unrelaxed rows against the GenKIC arm's relaxed ones.
        s_post = [r for r in sdedit_rows
                  if r.get("status") == "ok" and r.get("stage") == "post_relax"]
        s_pre = [r for r in sdedit_rows
                 if r.get("status") == "ok" and r.get("stage") == "pre_relax"]
        s_targets = {r["example_id"] for r in s_post}
        n_abstain = sum(1 for r in s_pre if not r.get("requested_chemistry_satisfied"))
        out(f"  SDEdit candidates graded (pre-relax) : {len(s_pre)}")
        out(f"    of which the requested chemistry was not even expressible: {n_abstain}"
            f"{f' ({100.0 * n_abstain / len(s_pre):.1f}%)' if s_pre else ''}")
        out("    (abstention: the sampled sequence carries no competent anchor pair. These")
        out("     are graded and FAIL on chemistry, never dropped -- excluding them would")
        out("     compute the success rate over only the attempts that half-succeeded.)")
        out(f"  SDEdit candidates relaxed+scored     : {len(s_post)}")
        out()
        out(f"  {'arm':12s} {'LPs graded':>11s} {'cands':>6s} {'accepted':>9s} {'LPs_ok':>7s} "
            f"{'med_ret':>8s} {'med_ddG':>8s}")

        def med_of(rs, key):
            vals = [_num(r, key) for r in rs]
            vals = [v for v in vals if v == v]
            return statistics.median(vals) if vals else float("nan")

        for name, rs in (("genkic", post), ("sdedit", s_post)):
            acc = [r for r in rs if r.get("accepted")]
            out(f"  {name:12s} {len({r['example_id'] for r in rs}):11d} {len(rs):6d} "
                f"{len(acc):9d} {len({r['example_id'] for r in acc}):7d} "
                f"{_fmt(med_of(rs, 'contact_retention_heavy45'), 3, 8)} "
                f"{_fmt(med_of(rs, 'ddg_interface_reu'), 2, 8)}")
        shared = sorted(s_targets & set(by_target))
        out(f"  LPs graded by BOTH arms: {len(shared)} -- the only fair head-to-head subset")
        if shared:
            g_ok = {t for t in shared if any(r.get("accepted") for r in by_target[t])}
            s_ok = {t for t in shared
                    if any(r.get("accepted") for r in s_post if r["example_id"] == t)}
            out(f"    accepted by genkic only : {len(g_ok - s_ok)}")
            out(f"    accepted by sdedit only : {len(s_ok - g_ok)}")
            out(f"    accepted by both        : {len(g_ok & s_ok)}")
            out(f"    accepted by neither     : {len(set(shared) - g_ok - s_ok)}")
    out()

    # ------------------------------------------------------------------- artefacts
    out("-" * 100)
    pdbs = [r.get("candidate_pdb") for r in post if r.get("candidate_pdb")]
    out(f"representative structures written: {len(pdbs)}")
    acc_pdbs = [r.get("candidate_pdb") for r in accepted if r.get("candidate_pdb")]
    for p in acc_pdbs[:15]:
        out(f"  {p}")
    if len(acc_pdbs) > 15:
        out(f"  ... and {len(acc_pdbs) - 15} more")
    out("=" * 100)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.txt").write_text("\n".join(lines) + "\n")
    with (out_dir / "per_target.jsonl").open("w") as fh:
        for r in per_target_rows:
            fh.write(json.dumps(r, default=str) + "\n")
    summary = {
        "n_targets_metadata": n_all,
        "n_targets_with_rows": len(by_target),
        "n_post_relax_candidates": len(post),
        "n_accepted": len(accepted),
        "n_targets_accepted": len(targets_accepted),
        "n_targets_accepted_strict": len(targets_strict),
        "n_infeasible_cells": len(infeasible),
        "n_crash_rows": len(broken),
        "targets_accepted": targets_accepted,
        "targets_unresolved": [r["example_id"] for r in unresolved],
        "bound_by": dict(bound_by),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_dir}/results.txt, per_target.jsonl, summary.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-dir", required=True, help="Directory of rows_*.jsonl shards.")
    ap.add_argument("--metadata", required=True,
                    help="Staged LP parquet, for the FULL target denominator.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sdedit-rows", default=None,
                    help="SDEdit candidates graded under the same contract: a JSONL file, or "
                         "a directory of rows_*.jsonl shards.")
    args = ap.parse_args()

    import pandas as pd

    meta = pd.read_parquet(args.metadata)
    all_targets = sorted(meta["example_id"].astype(str))

    rows = load_rows(Path(args.rows_dir))
    sdedit_rows: list[dict] = []
    if args.sdedit_rows:
        p = Path(args.sdedit_rows)
        files = sorted(p.glob("rows_*.jsonl")) if p.is_dir() else ([p] if p.is_file() else [])
        if not files:
            print(f"WARNING: --sdedit-rows given but no rows found at {p}; the comparison "
                  "section will report that it had nothing to grade.")
        for path in files:
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    sdedit_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    raise SystemExit(f"FATAL: corrupt SDEdit row {path}:{lineno}")
        print(f"loaded {len(sdedit_rows)} SDEdit rows from {len(files)} file(s)")

    report(rows, all_targets, Path(args.out_dir), sdedit_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
