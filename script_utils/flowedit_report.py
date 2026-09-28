#!/usr/bin/env python3
"""Aggregate FlowEdit / SDEdit candidates into the tables the contract asks for. CPU only.

Reads only saved artefacts, so every number re-renders without re-sampling or re-scoring.

What it reports, and what it keeps apart
----------------------------------------
PRIMARY successes are candidates that pass every fixed gate INCLUDING the paired relaxed
Rosetta delta, since acceptance requires the relaxed paired score. Those are the headline.

Kept strictly separate, because they are not the bar:
  * mutation budgets 0/1/2/3 and identity thresholds 90/80/70% -- reporting sweeps;
  * geometry-and-sequence-only passes, which have no dG yet;
  * undeclared-isopeptide rows (Asn/Gln acceptors Rosetta cannot declare here), which are
    excluded from claimed isopeptide successes rather than counted;
  * the AE round-trip floor, which says how much of the substitution budget was spent
    before any editing happened.

Coverage versus search effort is reported as pass@k: with k candidates per (LP, chemistry),
what fraction of LPs have at least one accepted candidate. A per-candidate rate and a
per-LP coverage answer different questions and the document states both.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(paths, what: str) -> list[dict]:
    """Read JSONL rows, REFUSING corrupt lines. A skipped line is silent data loss."""
    rows = []
    for p in paths:
        p = Path(p)
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"FATAL: corrupt {what} row {p}:{lineno} ({exc}). "
                                 "Re-run the shard that wrote it; do not filter it out.")
    return rows


def med(vals):
    ok = [v for v in vals if v is not None and v == v]
    return statistics.median(ok) if ok else float("nan")


def iqr(vals):
    ok = sorted(v for v in vals if v is not None and v == v)
    if len(ok) < 4:
        return float("nan")
    q1 = ok[len(ok) // 4]
    q3 = ok[(3 * len(ok)) // 4]
    return q3 - q1


def fmt(v, nd=3):
    if v is None or v != v:
        return "n/a"
    return f"{v:.{nd}f}"


def pct(n, d):
    return "n/a" if not d else f"{100.0 * n / d:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--rosetta", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    cand = load(args.candidates, "candidate")
    ros = load(args.rosetta, "rosetta") if args.rosetta else []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in cand if r.get("status") == "ok"]
    failed = [r for r in cand if r.get("status") != "ok"]
    ros_by_key = {r["run_key"]: r for r in ros if r.get("status") == "ok"}

    # Join: the Rosetta verdict supersedes the candidate's (which had no dG).
    for r in ok:
        j = ros_by_key.get(r.get("run_key"))
        r["_scored"] = bool(j)
        if j:
            r["dG_delta_REU"] = j.get("dG_delta_REU")
            r["pass_ddg"] = j.get("pass_ddg")
            r["ddg_non_worsening"] = j.get("ddg_non_worsening")
            r["accepted"] = j.get("accepted")
            r["excluded_undeclared_isopeptide"] = j.get("excluded_undeclared_isopeptide")
            r["ring_bond_post_median_A"] = j.get("ring_bond_post_median_A")
            r["cp_dG_spread"] = j.get("cp_dG_spread")
            r["lp_dG_spread"] = j.get("lp_dG_spread")
        else:
            r["accepted"] = False

    methods = sorted({r.get("method") for r in ok if r.get("method")})
    chems = sorted({r.get("cyc_type") for r in ok if r.get("cyc_type")})
    lines: list[str] = []
    W = lines.append

    W(f"# FlowEdit LP->CP results{' -- ' + args.run_tag if args.run_tag else ''}")
    W("")
    W(f"Candidates: {len(cand)} rows ({len(ok)} ok, {len(failed)} failed). "
      f"Rosetta-scored: {len(ros_by_key)}.")
    W("")
    if not ros_by_key:
        W("> **No Rosetta scores present.** Acceptance requires the paired relaxed "
          "interface score, so NOTHING below is an accepted candidate. The geometry and "
          "sequence gates are reported on their own, and are not the contract's bar.")
        W("")

    # ---------------- primary table ----------------
    W("## Primary: accepted candidates (all gates, including paired relaxed dG)")
    W("")
    W("| method | chemistry | candidates | geom+seq pass | scored | accepted | "
      "accepted LPs | dG<=0 |")
    W("|---|---|---|---|---|---|---|---|")
    for m in methods:
        for c in chems:
            g = [r for r in ok if r.get("method") == m and r.get("cyc_type") == c]
            if not g:
                continue
            gs = [r for r in g if r.get("pass_geometry_and_sequence")]
            sc = [r for r in g if r.get("_scored")]
            acc = [r for r in g if r.get("accepted")]
            lps = {r.get("example_id") for r in g}
            acc_lps = {r.get("example_id") for r in acc}
            nw = [r for r in sc if r.get("ddg_non_worsening")]
            W(f"| {m} | {c} | {len(g)} | {len(gs)} ({pct(len(gs), len(g))}) | {len(sc)} | "
              f"{len(acc)} ({pct(len(acc), len(g))}) | {len(acc_lps)}/{len(lps)} | "
              f"{len(nw)} ({pct(len(nw), len(sc))}) |")
    W("")

    # ---------------- gate attrition ----------------
    W("## Where candidates are lost")
    W("")
    W("Each column is that gate's pass rate among ok candidates, so the biggest gap is the "
      "binding constraint. These are not independent -- a candidate can fail several.")
    W("")
    gates = ["pass_ring", "pass_clash", "pass_identity", "pass_substitutions",
             "pass_protected", "pass_contacts", "pass_rmsd", "pass_interface_rmsd"]
    W("| method | chemistry | n | " + " | ".join(g.replace("pass_", "") for g in gates) + " |")
    W("|---|---|---|" + "---|" * len(gates))
    for m in methods:
        for c in chems:
            g = [r for r in ok if r.get("method") == m and r.get("cyc_type") == c]
            if not g:
                continue
            cells = [pct(sum(1 for r in g if r.get(k)), len(g)) for k in gates]
            W(f"| {m} | {c} | {len(g)} | " + " | ".join(cells) + " |")
    W("")

    # ---------------- quality distributions ----------------
    W("## Quality of what was produced (medians, IQR in brackets)")
    W("")
    W("| method | chemistry | retention | CA RMSD A | iface CA RMSD A | subs | "
      "bond A | dG delta REU |")
    W("|---|---|---|---|---|---|---|---|")
    for m in methods:
        for c in chems:
            g = [r for r in ok if r.get("method") == m and r.get("cyc_type") == c]
            if not g:
                continue
            def col(k, nd=3):
                v = [r.get(k) for r in g]
                return f"{fmt(med(v), nd)} [{fmt(iqr(v), nd)}]"
            W(f"| {m} | {c} | {col('contact_retention')} | {col('ca_rmsd_A', 2)} | "
              f"{col('interface_ca_rmsd_A', 2)} | {col('n_substitutions', 1)} | "
              f"{col('bond_dist_A', 2)} | {col('dG_delta_REU', 2)} |")
    W("")

    # ---------------- the AE floor ----------------
    W("## The autoencoder floor (not a FlowEdit result)")
    W("")
    W("Decoding an UNEDITED latent already differs from the crystal. If the AE alone spends "
      "the substitution budget, no sampler can pass the sequence gate, and attributing that "
      "to the editing method would be a misattribution.")
    W("")
    rt_subs = [r.get("ae_rt_n_substitutions") for r in ok]
    rt_over = [r for r in ok if r.get("ae_rt_consumes_budget")]
    W(f"- Median AE round-trip substitutions: **{fmt(med(rt_subs), 1)}** "
      f"(IQR {fmt(iqr(rt_subs), 1)})")
    W(f"- Candidates whose AE round trip ALONE exceeds the 2-substitution budget: "
      f"**{len(rt_over)}/{len(ok)}** ({pct(len(rt_over), len(ok))})")
    W(f"- Median AE round-trip CA RMSD: {fmt(med([r.get('ae_rt_ca_rmsd_A') for r in ok]), 3)} A")
    W("")

    # ---------------- reporting-only sweeps ----------------
    W("## Reporting sweeps (NOT the primary bar)")
    W("")
    W("The fixed thresholds above do not move. These say how the picture changes if the "
      "sequence constraint were set elsewhere; they exclude the dG gate unless stated.")
    W("")
    budget_cols = [f"pass_budget_{k}" for k in (0, 1, 2, 3)]
    ident_cols = [f"pass_identity_{k}" for k in (90, 80, 70)]
    W("| method | chemistry | n | " + " | ".join(
        ["budget 0", "budget 1", "budget 2", "budget 3", "id>=90%", "id>=80%", "id>=70%"]) + " |")
    W("|---|---|---|" + "---|" * 7)
    for m in methods:
        for c in chems:
            g = [r for r in ok if r.get("method") == m and r.get("cyc_type") == c]
            if not g:
                continue
            cells = [pct(sum(1 for r in g if r.get(k)), len(g))
                     for k in budget_cols + ident_cols]
            W(f"| {m} | {c} | {len(g)} | " + " | ".join(cells) + " |")
    W("")

    # ---------------- coverage vs effort ----------------
    W("## Quality-qualified coverage versus search effort")
    W("")
    W("pass@k: fraction of LPs with at least one ACCEPTED candidate within the first k "
      "candidates per (LP, chemistry). A per-candidate success rate and per-LP coverage "
      "answer different questions.")
    W("")
    W("| method | chemistry | LPs | " + " | ".join(f"pass@{k}" for k in (1, 2, 4, 8, 16, 32)) + " |")
    W("|---|---|---|" + "---|" * 6)
    for m in methods:
        for c in chems:
            grp: dict[str, list[dict]] = defaultdict(list)
            for r in ok:
                if r.get("method") == m and r.get("cyc_type") == c:
                    grp[r.get("example_id")].append(r)
            if not grp:
                continue
            cells = []
            for k in (1, 2, 4, 8, 16, 32):
                hit = 0
                for _lp, rs in grp.items():
                    rs2 = sorted(rs, key=lambda r: (r.get("seed") is None, r.get("seed")))
                    if any(r.get("accepted") for r in rs2[:k]):
                        hit += 1
                cells.append(pct(hit, len(grp)))
            W(f"| {m} | {c} | {len(grp)} | " + " | ".join(cells) + " |")
    W("")

    # ---------------- best per LP ----------------
    W("## Best accepted candidate per LP")
    W("")
    acc = [r for r in ok if r.get("accepted")]
    if not acc:
        W("None. No candidate passed every gate including the paired relaxed dG.")
    else:
        W("Ranked within each LP by dG delta (lower is better), ties by retention.")
        W("")
        W("| LP | method | chemistry | dG delta REU | retention | CA RMSD A | subs | bond A |")
        W("|---|---|---|---|---|---|---|---|")
        by_lp: dict[str, list[dict]] = defaultdict(list)
        for r in acc:
            by_lp[r.get("example_id")].append(r)
        for lp in sorted(by_lp):
            b = sorted(by_lp[lp], key=lambda r: (r.get("dG_delta_REU", 9e9),
                                                 -(r.get("contact_retention") or 0)))[0]
            W(f"| {lp} | {b.get('method')} | {b.get('cyc_type')} | "
              f"{fmt(b.get('dG_delta_REU'), 2)} | {fmt(b.get('contact_retention'))} | "
              f"{fmt(b.get('ca_rmsd_A'), 2)} | {b.get('n_substitutions')} | "
              f"{fmt(b.get('bond_dist_A'), 2)} |")
    W("")

    # ---------------- unresolved ----------------
    all_lps = sorted({r.get("example_id") for r in ok})
    acc_lps = {r.get("example_id") for r in acc}
    unresolved = [lp for lp in all_lps if lp not in acc_lps]
    W("## Unresolved LPs")
    W("")
    W(f"{len(unresolved)}/{len(all_lps)} LPs have no accepted candidate.")
    if unresolved:
        W("")
        W("| LP | best retention | best dG delta | geom+seq passes | scored |")
        W("|---|---|---|---|---|")
        for lp in unresolved:
            g = [r for r in ok if r.get("example_id") == lp]
            gs = sum(1 for r in g if r.get("pass_geometry_and_sequence"))
            sc = sum(1 for r in g if r.get("_scored"))
            best_ret = max((r.get("contact_retention") or 0) for r in g)
            dgs = [r.get("dG_delta_REU") for r in g if r.get("dG_delta_REU") is not None]
            W(f"| {lp} | {fmt(best_ret)} | {fmt(min(dgs) if dgs else None, 2)} | "
              f"{gs}/{len(g)} | {sc} |")
    W("")

    # ---------------- isopeptide exclusions ----------------
    excl = [r for r in ok if r.get("excluded_undeclared_isopeptide")]
    if excl:
        W("## Excluded: isopeptides Rosetta could not declare")
        W("")
        W(f"{len(excl)} isopeptide candidates have Asn/Gln acceptors, for which this "
          "Rosetta build has no SidechainConjugation variant. Their rings cannot be "
          "declared, so they are excluded from accepted isopeptide successes rather than "
          "scored as if closed. Their geometry is still reported above.")
        W("")

    # ---------------- failures ----------------
    if failed:
        W("## Failed candidates")
        W("")
        kinds: dict[str, int] = defaultdict(int)
        for r in failed:
            kinds[str(r.get("error", "?")).split(":")[0]] += 1
        W(f"{len(failed)} candidates failed to generate:")
        W("")
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
            W(f"- `{k}`: {v}")
        W("")

    doc = "\n".join(lines) + "\n"
    (out_dir / "RESULTS.md").write_text(doc)

    summary = {
        "run_tag": args.run_tag,
        "n_candidates": len(cand), "n_ok": len(ok), "n_failed": len(failed),
        "n_rosetta_scored": len(ros_by_key),
        "n_accepted": len(acc),
        "lps_total": len(all_lps), "lps_accepted": len(acc_lps),
        "by_method_chem": {
            f"{m}|{c}": {
                "n": sum(1 for r in ok if r.get("method") == m and r.get("cyc_type") == c),
                "accepted": sum(1 for r in acc if r.get("method") == m and r.get("cyc_type") == c),
            }
            for m in methods for c in chems
        },
        "ae_roundtrip_median_subs": med(rt_subs),
        "ae_roundtrip_over_budget": len(rt_over),
    }
    # NaN is not valid JSON. Python's encoder emits it happily and strict parsers then
    # reject the whole file, so absent values become null.
    def json_safe(o):
        if isinstance(o, float) and o != o:
            return None
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [json_safe(v) for v in o]
        return o

    (out_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_dir / 'RESULTS.md'} and {out_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
