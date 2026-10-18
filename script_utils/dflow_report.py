"""Join structure, sequence and energy into the tables the LP -> CP contract actually asks for.

Everything here reads the RE-SCORED rows, never an arm's own columns, so D-Flow, SDEdit and
guided SDEdit are compared through one instrument (`script_utils/dflow_rescore.py`). An arm's
native columns were produced by different code with a different contact definition and no
clash, sequence-budget or interface-RMSD test, and pooling them would be a comparison of
scoring functions wearing the costume of a comparison of methods.

The headline number is a CONJUNCTION
------------------------------------
`gate_joint_structural` is ring + sequence + contacts + pose + clash, computed from
coordinates. It is deliberately NOT the headline: the contract's energy clause needs a
relaxation that only the Rosetta sidecar can run. `gate_joint_full` is that conjunction AND
the paired interface score, and it is the number to quote. Both are reported, together with
which clause bound, because a single pass/fail hides the reason and the reason is the result.

Three things this refuses to do
-------------------------------
1. Pool `init` with `opt`. Every D-Flow restart is scored twice -- the same source state
   before and after optimization, same seed, same schedule. Without that pairing a headline
   cannot distinguish "the optimizer worked" from "16 restarts found a good sample", which is
   the first thing to ask of a method whose selling point is optimization. The paired table
   is computed on restarts where BOTH stages scored, so it is a real paired statistic.

2. Pool isopeptide energy with the other two chemistries. `_declare_cyclic_bond` returns
   False for isopeptide -- no residue-connection patch is wired up -- so Rosetta scores a
   topology whose closing bond it does not know about. Those rows are reported in their own
   tier and flagged, rather than averaged into a number that would look fine and be wrong.

3. Rank by sample count. `quality vs compute` divides by GPU seconds actually spent, because
   a method that wins per-sample while costing 40x per sample has not won.

Nondominated candidates are kept across sequence, pose and energy rather than collapsed into
a weighted score: the three are not commensurable, and a scalarization would silently encode
a preference the contract does not state.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "script_utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dflow_accept import pareto_front  # noqa: E402
from dflow_losses import CONTRACT, IDENTITY_TIERS, MUTATION_BUDGETS  # noqa: E402

# Structural gates, in the order they are reported. `gate_ring_declared` is deliberately
# absent: the older SDEdit runs were written before the CONECT writer existed, so gating on
# it would make the baselines look worse for a reason about the writer, not the structure.
STRUCT_GATES = [
    "gate_ring_closed",
    "gate_identity",
    "gate_substitutions",
    "gate_protected",
    "gate_contacts",
    "gate_ca_rmsd",
    "gate_iface_ca_rmsd",
    "gate_clash",
]


def _finite(v) -> bool:
    """True only for a real number. NaN is truthy in Python, and an abstained row carries
    NaN rather than None -- a naive `if v:` filter scores a 100%-abstention cell as a 100%
    pass, which has happened on this project."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _read_jsonl(paths: list[Path]) -> list[dict]:
    """Read rows, REFUSING corrupt lines rather than skipping them.

    Concurrent O_APPEND to one file has NUL-corrupted rows on this cluster and 88 of 144 were
    lost silently. A skipped bad line turns data loss into a smaller denominator, which is
    invisible in the output; a refusal is not.
    """
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            if "\0" in line:
                raise SystemExit(
                    f"FATAL: NUL byte in {p}:{n}. This is the concurrent-append corruption "
                    "seen on this cluster; the shard files must not share an output path. "
                    "Refusing to report on partial data."
                )
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"FATAL: unparseable JSON at {p}:{n}: {exc}") from exc
    return rows


def _rate(rows: list[dict], key: str) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if bool(r.get(key)))
    return {"n": n, "k": k, "rate": (k / n) if n else None}


def attach_energy(cands: list[dict], ros: list[dict]) -> dict:
    """Median paired ddG per run_key, plus the spread across relaxation repeats.

    Median, not mean: FastRelax is stochastic and a single failed trajectory is an outlier,
    not a signal. The spread is reported alongside because a candidate whose three repeats
    straddle the +5 REU threshold has not met it in any meaningful sense.
    """
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in ros:
        if r.get("status") == "ok" and _finite(r.get("ddg_REU")):
            by_key[str(r.get("run_key"))].append(r)

    n_attached = 0
    for c in cands:
        g = by_key.get(str(c.get("run_key")), [])
        if not g:
            c["ddg_REU"] = None
            c["gate_energy"] = False
            c["energy_scored"] = False
            continue
        vals = [float(r["ddg_REU"]) for r in g]
        c["ddg_REU"] = st.median(vals)
        c["ddg_repeats"] = len(vals)
        c["ddg_min_REU"] = min(vals)
        c["ddg_max_REU"] = max(vals)
        c["ddg_spread_REU"] = max(vals) - min(vals)
        if len(vals) > 1:
            c["ddg_stdev_REU"] = st.stdev(vals)
        c["energy_scored"] = True
        c["gate_energy"] = c["ddg_REU"] <= CONTRACT["max_ddg_REU"]
        c["gate_energy_strict"] = c["ddg_REU"] <= 0.0
        # Whether Rosetta actually knew about the closing bond. False for every isopeptide.
        c["topology_declared"] = bool(g[0].get("cp/topology_declared"))
        c["ring_dist_before_A"] = g[0].get("cp/ring_dist_before_A")
        c["ring_dist_after_A"] = g[0].get("cp/ring_dist_after_A")
        n_attached += 1

    for c in cands:
        c["gate_joint_full"] = bool(c.get("gate_joint_structural")) and bool(c.get("gate_energy"))
    return {"candidates": len(cands), "with_energy": n_attached,
            "rosetta_rows_ok": sum(len(v) for v in by_key.values())}


def arm_table(cands: list[dict]) -> dict:
    """Per-arm pass rates: every gate on its own, then the two conjunctions."""
    out: dict = {}
    for arm in sorted({str(c.get("arm")) for c in cands}):
        rows = [c for c in cands if str(c.get("arm")) == arm]
        # D-Flow rows exist at two stages; an arm's table is reported per stage so `init` is
        # never silently averaged into `opt`.
        for stage in sorted({str(r.get("stage")) for r in rows}):
            srows = [r for r in rows if str(r.get("stage")) == stage]
            t = {g: _rate(srows, g) for g in STRUCT_GATES}
            t["gate_joint_structural"] = _rate(srows, "gate_joint_structural")
            t["gate_joint_full"] = _rate(srows, "gate_joint_full")
            t["gate_energy"] = _rate([r for r in srows if r.get("energy_scored")], "gate_energy")
            t["gate_energy_strict"] = _rate(
                [r for r in srows if r.get("energy_scored")], "gate_energy_strict")
            t["n_energy_scored"] = sum(1 for r in srows if r.get("energy_scored"))
            for b in MUTATION_BUDGETS:
                t[f"budget_{b}"] = _rate(srows, f"budget_{b}")
            for tier in IDENTITY_TIERS:
                k = f"identity_ge_{int(tier * 100)}"
                t[k] = _rate(srows, k)
            # Which clause bound, among rows that failed the structural conjunction.
            failed = [r for r in srows if not r.get("gate_joint_structural")]
            bind: dict[str, int] = defaultdict(int)
            for r in failed:
                for g in r.get("gates_failed") or []:
                    bind[g] += 1
            t["binding_clause"] = dict(sorted(bind.items(), key=lambda kv: -kv[1]))
            t["n_failed_structural"] = len(failed)
            out[f"{arm}|{stage}"] = t
    return out


def paired_init_opt(cands: list[dict]) -> dict:
    """Did optimization move the restart it started from?

    Paired on (arm, example, chemistry, objective, restart): the same source state scored
    before and after. Restarts missing either stage are excluded rather than filled in, so
    this is a genuine paired statistic and not two marginal rates subtracted.
    """
    by: dict[tuple, dict] = defaultdict(dict)
    for c in cands:
        if str(c.get("stage")) not in ("init", "opt"):
            continue
        key = (str(c.get("arm")), str(c.get("example_id")), str(c.get("cyc_type_name")),
               str(c.get("objective_tag")), str(c.get("restart")))
        by[key][str(c["stage"])] = c

    pairs = [(v["init"], v["opt"]) for v in by.values() if "init" in v and "opt" in v]
    if not pairs:
        return {"n_pairs": 0, "note": "no restart scored at both stages"}

    def _delta(field: str) -> dict:
        d = [float(o[field]) - float(i[field])
             for i, o in pairs if _finite(i.get(field)) and _finite(o.get(field))]
        if not d:
            return {"n": 0}
        return {"n": len(d), "median": st.median(d),
                "mean": sum(d) / len(d), "improved": sum(1 for x in d if x < 0)}

    res = {
        "n_pairs": len(pairs),
        "ring_closed": {
            "init": sum(1 for i, _ in pairs if i.get("gate_ring_closed")),
            "opt": sum(1 for _, o in pairs if o.get("gate_ring_closed")),
            # McNemar cells: the only informative ones for a paired binary outcome.
            "opened_by_opt": sum(1 for i, o in pairs
                                 if not i.get("gate_ring_closed") and o.get("gate_ring_closed")),
            "lost_by_opt": sum(1 for i, o in pairs
                               if i.get("gate_ring_closed") and not o.get("gate_ring_closed")),
        },
        "joint_structural": {
            "init": sum(1 for i, _ in pairs if i.get("gate_joint_structural")),
            "opt": sum(1 for _, o in pairs if o.get("gate_joint_structural")),
            "gained_by_opt": sum(1 for i, o in pairs
                                 if not i.get("gate_joint_structural")
                                 and o.get("gate_joint_structural")),
            "lost_by_opt": sum(1 for i, o in pairs
                               if i.get("gate_joint_structural")
                               and not o.get("gate_joint_structural")),
        },
        "joint_full": {
            "init": sum(1 for i, _ in pairs if i.get("gate_joint_full")),
            "opt": sum(1 for _, o in pairs if o.get("gate_joint_full")),
        },
        # Negative deltas are improvements for all three.
        "delta_ca_rmsd_A": _delta("ca_rmsd_A"),
        "delta_iface_ca_rmsd_A": _delta("iface_ca_rmsd_A"),
        "delta_n_substitutions": _delta("n_substitutions"),
        "delta_ddg_REU": _delta("ddg_REU"),
    }
    ret = [float(o["contact_retention"]) - float(i["contact_retention"])
           for i, o in pairs
           if _finite(i.get("contact_retention")) and _finite(o.get("contact_retention"))]
    if ret:
        res["delta_contact_retention"] = {
            "n": len(ret), "median": st.median(ret),
            "improved": sum(1 for x in ret if x > 0)}
    return res


def coverage(cands: list[dict], gate: str) -> dict:
    """Per-LP coverage: how many input peptides have at least one candidate that qualifies.

    This is the number that answers "can the method do this at all", and it is reported per
    arm because an arm that qualifies 3 candidates on one easy peptide has covered one LP.
    """
    out: dict = {}
    for arm in sorted({str(c.get("arm")) for c in cands}):
        rows = [c for c in cands if str(c.get("arm")) == arm and str(c.get("stage")) != "init"]
        ex = {str(r.get("example_id")) for r in rows}
        hit = {str(r.get("example_id")) for r in rows if r.get(gate)}
        by_chem: dict = {}
        for chem in sorted({str(r.get("cyc_type_name")) for r in rows}):
            crow = [r for r in rows if str(r.get("cyc_type_name")) == chem]
            cex = {str(r.get("example_id")) for r in crow}
            chit = {str(r.get("example_id")) for r in crow if r.get(gate)}
            by_chem[chem] = {"examples": len(cex), "covered": len(chit),
                             "rate": len(chit) / len(cex) if cex else None,
                             "candidates": len(crow),
                             "qualified": sum(1 for r in crow if r.get(gate))}
        out[arm] = {"examples": len(ex), "covered": len(hit),
                    "rate": len(hit) / len(ex) if ex else None,
                    "unresolved": sorted(ex - hit), "by_chemistry": by_chem}
    return out


def best_per_lp(cands: list[dict], gate: str) -> list[dict]:
    """One row per (arm, example, chemistry): the best candidate, and what it was.

    "Best" ranks qualified candidates ahead of unqualified ones, then by paired energy, then
    by pose. Unqualified examples still appear, carrying the clause that bound them -- an
    unresolved case reported as an absence is indistinguishable from one never attempted.
    """
    by: dict[tuple, list[dict]] = defaultdict(list)
    for c in cands:
        if str(c.get("stage")) == "init":
            continue
        by[(str(c.get("arm")), str(c.get("example_id")), str(c.get("cyc_type_name")))].append(c)

    out = []
    for (arm, ex, chem), rows in sorted(by.items()):
        def rank(r: dict):
            return (
                0 if r.get(gate) else 1,
                0 if r.get("gate_joint_structural") else 1,
                float(r["ddg_REU"]) if _finite(r.get("ddg_REU")) else math.inf,
                float(r["ca_rmsd_A"]) if _finite(r.get("ca_rmsd_A")) else math.inf,
            )
        b = min(rows, key=rank)
        out.append({
            "arm": arm, "example_id": ex, "cyc_type_name": chem,
            "n_candidates": len(rows),
            "n_qualified": sum(1 for r in rows if r.get(gate)),
            "run_key": b.get("run_key"), "restart": b.get("restart"),
            "qualified": bool(b.get(gate)),
            "gate_joint_structural": bool(b.get("gate_joint_structural")),
            "seq_identity": b.get("seq_identity"),
            "n_substitutions": b.get("n_substitutions"),
            "contact_retention": b.get("contact_retention"),
            "ca_rmsd_A": b.get("ca_rmsd_A"),
            "iface_ca_rmsd_A": b.get("iface_ca_rmsd_A"),
            "bond_dist_A": b.get("bond_dist_A"),
            "ddg_REU": b.get("ddg_REU"),
            "ddg_spread_REU": b.get("ddg_spread_REU"),
            "topology_declared": b.get("topology_declared"),
            "complex_pdb": b.get("complex_pdb"),
            "gates_failed": b.get("gates_failed"),
        })
    return out


def energy_by_topology(cands: list[dict]) -> dict:
    """Energy kept in two tiers by whether Rosetta was told about the ring.

    Isopeptide has no residue-connection patch, so its closing bond is invisible to the score
    function. FastRelax runs with `constrain_relax_to_start_coords(True)`, so an undeclared
    ring should not fly apart -- `ring_drift_A` is whether it actually did, and is the
    evidence for how far the undeclared tier can be trusted.
    """
    out: dict = {}
    scored = [c for c in cands if c.get("energy_scored") and str(c.get("stage")) != "init"]
    for chem in sorted({str(c.get("cyc_type_name")) for c in scored}):
        rows = [c for c in scored if str(c.get("cyc_type_name")) == chem]
        dd = [float(r["ddg_REU"]) for r in rows if _finite(r.get("ddg_REU"))]
        drift = [abs(float(r["ring_dist_after_A"]) - float(r["ring_dist_before_A"]))
                 for r in rows
                 if _finite(r.get("ring_dist_after_A")) and _finite(r.get("ring_dist_before_A"))]
        out[chem] = {
            "n": len(rows),
            "topology_declared": bool(rows[0].get("topology_declared")) if rows else None,
            "ddg_median_REU": st.median(dd) if dd else None,
            "ddg_min_REU": min(dd) if dd else None,
            "pass_plus5": sum(1 for r in rows if r.get("gate_energy")),
            "pass_strict0": sum(1 for r in rows if r.get("gate_energy_strict")),
            "ring_drift_median_A": st.median(drift) if drift else None,
            "ring_drift_max_A": max(drift) if drift else None,
        }
    return out


def compute_table(cands: list[dict], cost: list[dict]) -> dict:
    """Quality against GPU seconds actually spent, not against sample count."""
    out: dict = {}
    for arm in sorted({str(c.get("arm")) for c in cands}):
        rows = [c for c in cands if str(c.get("arm")) == arm]
        opt = [r for r in rows if str(r.get("stage")) != "init"]
        secs = [float(r["opt_seconds"]) for r in rows
                if str(r.get("stage")) == "opt" and _finite(r.get("opt_seconds"))]
        total = sum(secs)
        nq_s = sum(1 for r in opt if r.get("gate_joint_structural"))
        nq_f = sum(1 for r in opt if r.get("gate_joint_full"))
        out[arm] = {
            "candidates": len(opt),
            "restarts_timed": len(secs),
            "gpu_seconds_total": round(total, 1) if secs else None,
            "gpu_seconds_median_per_restart": round(st.median(secs), 1) if secs else None,
            "qualified_structural": nq_s,
            "qualified_full": nq_f,
            "gpu_seconds_per_qualified_structural": round(total / nq_s, 1) if (secs and nq_s) else None,
            "gpu_seconds_per_qualified_full": round(total / nq_f, 1) if (secs and nq_f) else None,
            # Absent for the baselines, which did not record per-restart timing. Stated as
            # missing rather than imputed.
            "timing_available": bool(secs),
        }
    if cost:
        out["_cost_summary_from_logs"] = cost
    return out


def nondominated(cands: list[dict]) -> list[dict]:
    """Nondominated across sequence, pose and energy, among structurally-qualified rows.

    Kept as a front rather than scalarized: identity, RMSD and REU are not commensurable, and
    a weighted sum would encode a preference the contract does not state.
    """
    rows = [c for c in cands
            if str(c.get("stage")) != "init" and c.get("gate_joint_structural")
            and _finite(c.get("ddg_REU"))]
    if not rows:
        return []
    idx = pareto_front(rows, [("seq_identity", "max"), ("ca_rmsd_A", "min"), ("ddg_REU", "min")])
    return [{k: rows[i].get(k) for k in
             ("arm", "example_id", "cyc_type_name", "run_key", "seq_identity",
              "n_substitutions", "ca_rmsd_A", "iface_ca_rmsd_A", "contact_retention",
              "ddg_REU", "gate_joint_full", "complex_pdb")}
            for i in idx]


def _fmt(v, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def _rate_str(d: dict) -> str:
    if not d or not d.get("n"):
        return "-"
    return f"{d['k']}/{d['n']} ({100 * d['rate']:.1f}%)"


def _cell(k: str) -> str:
    """Arm keys are `arm|stage`, and a raw `|` splits a markdown cell and shifts every column
    to its right -- backticks do not protect it. The JSON keys stay as they are."""
    return str(k).replace("|", " / ")


def markdown(rep: dict) -> str:
    L: list[str] = []
    A = L.append
    A(f"# D-Flow LP -> CP results ({rep['run_id']})")
    A("")
    A(f"Generated {rep['generated']} from `{rep['results_dir']}`.")
    A("")
    A("`gate_joint_full` is the headline: ring + sequence + contacts + pose + clash **and** the")
    A(f"paired Rosetta interface score within +{CONTRACT['max_ddg_REU']:.0f} REU of the peptide's own")
    A("linear original. REU are Rosetta Energy Units and are not kcal/mol. Every arm below was")
    A("scored by one instrument (`dflow_rescore.py`), never by its own columns.")
    A("")

    cal = rep.get("calibration")
    A("## Instrument check")
    A("")
    if cal:
        A("Native crystal peptides scored against themselves. Identity, substitutions, contact")
        A("retention and RMSD are arithmetic and must be perfect; the ring gate is EXPECTED to")
        A("fail, because these inputs are linear — that failure is the positive control.")
        A("")
        A("```")
        A(json.dumps(cal, indent=2)[:4000])
        A("```")
    else:
        A("**Not available** — `calibration.json` was not found. The pass rates below have not")
        A("been checked against a known answer.")
    A("")

    A("## Per-arm gates")
    A("")
    A("| arm / stage | n | ring | identity | subs | contacts | CA RMSD | iface RMSD | clash | joint struct | joint FULL |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, t in rep["arms"].items():
        A(f"| `{_cell(k)}` | {t['gate_ring_closed']['n']} | {_rate_str(t['gate_ring_closed'])} | "
          f"{_rate_str(t['gate_identity'])} | {_rate_str(t['gate_substitutions'])} | "
          f"{_rate_str(t['gate_contacts'])} | {_rate_str(t['gate_ca_rmsd'])} | "
          f"{_rate_str(t['gate_iface_ca_rmsd'])} | {_rate_str(t['gate_clash'])} | "
          f"**{_rate_str(t['gate_joint_structural'])}** | **{_rate_str(t['gate_joint_full'])}** |")
    A("")
    A("Energy tiers, among rows Rosetta actually scored:")
    A("")
    A("| arm / stage | scored | <= +5 REU | <= 0 REU (strict) |")
    A("|---|---|---|---|")
    for k, t in rep["arms"].items():
        A(f"| `{_cell(k)}` | {t['n_energy_scored']} | {_rate_str(t['gate_energy'])} | "
          f"{_rate_str(t['gate_energy_strict'])} |")
    A("")

    A("## Reported tiers (never a substitute for the gates above)")
    A("")
    A("| arm / stage | budget 0 | budget 1 | budget 2 | budget 3 | id >=90% | id >=80% | id >=70% |")
    A("|---|---|---|---|---|---|---|---|")
    for k, t in rep["arms"].items():
        A(f"| `{_cell(k)}` | " + " | ".join(
            _rate_str(t[f"budget_{b}"]) for b in MUTATION_BUDGETS) + " | " + " | ".join(
            _rate_str(t[f"identity_ge_{int(x * 100)}"]) for x in IDENTITY_TIERS) + " |")
    A("")

    A("## Did optimization work? (paired init vs opt, same source state)")
    A("")
    p = rep["paired"]
    if not p.get("n_pairs"):
        A(f"No paired restarts: {p.get('note', 'none')}.")
    else:
        A(f"{p['n_pairs']} restarts scored at both stages.")
        A("")
        A("| outcome | init | opt | gained | lost |")
        A("|---|---|---|---|---|")
        r = p["ring_closed"]
        A(f"| ring closed | {r['init']} | {r['opt']} | {r['opened_by_opt']} | {r['lost_by_opt']} |")
        j = p["joint_structural"]
        A(f"| joint structural | {j['init']} | {j['opt']} | {j['gained_by_opt']} | {j['lost_by_opt']} |")
        A(f"| joint FULL | {p['joint_full']['init']} | {p['joint_full']['opt']} | - | - |")
        A("")
        A("Paired deltas (opt minus init; negative is better for all four):")
        A("")
        A("| quantity | n | median | improved |")
        A("|---|---|---|---|")
        for key, lbl in (("delta_ca_rmsd_A", "CA RMSD (A)"),
                         ("delta_iface_ca_rmsd_A", "interface CA RMSD (A)"),
                         ("delta_n_substitutions", "substitutions"),
                         ("delta_ddg_REU", "paired ddG (REU)")):
            d = p.get(key, {})
            if d.get("n"):
                A(f"| {lbl} | {d['n']} | {_fmt(d['median'])} | {d['improved']}/{d['n']} |")
        d = p.get("delta_contact_retention", {})
        if d.get("n"):
            A(f"| contact retention (higher better) | {d['n']} | {_fmt(d['median'])} | {d['improved']}/{d['n']} |")
    A("")

    A("## Coverage: LPs with at least one qualifying candidate")
    A("")
    A("| arm | examples | covered (joint FULL) | rate | covered (structural only) |")
    A("|---|---|---|---|---|")
    cs = rep["coverage_structural"]
    for arm, c in rep["coverage_full"].items():
        s = cs.get(arm, {})
        A(f"| `{arm}` | {c['examples']} | {c['covered']} | "
          f"{_fmt(c['rate'])} | {s.get('covered', '-')} |")
    A("")
    for arm, c in rep["coverage_full"].items():
        if c["unresolved"]:
            A(f"Unresolved under `{arm}` ({len(c['unresolved'])}): "
              + ", ".join(f"`{e}`" for e in c["unresolved"][:40])
              + (" ..." if len(c["unresolved"]) > 40 else ""))
    A("")

    A("## Energy by chemistry, split on whether Rosetta knew about the ring")
    A("")
    A("| chemistry | n | topology declared | median ddG (REU) | best | <= +5 | <= 0 | ring drift median / max (A) |")
    A("|---|---|---|---|---|---|---|---|")
    for chem, e in rep["energy_by_topology"].items():
        A(f"| {chem} | {e['n']} | {_fmt(e['topology_declared'])} | {_fmt(e['ddg_median_REU'], 2)} | "
          f"{_fmt(e['ddg_min_REU'], 2)} | {e['pass_plus5']} | {e['pass_strict0']} | "
          f"{_fmt(e['ring_drift_median_A'], 2)} / {_fmt(e['ring_drift_max_A'], 2)} |")
    A("")
    A("Isopeptide's closing bond is **not** declared to Rosetta (`_declare_cyclic_bond` has no")
    A("residue-connection patch for it), so its energy column describes a topology the score")
    A("function does not know about. Ring drift under coordinate-constrained FastRelax is the")
    A("evidence for how far that tier can be read; it is not pooled with the other two.")
    A("")

    A("## Quality against compute")
    A("")
    A("| arm | candidates | restarts timed | GPU s total | GPU s / restart | qualified (full) | GPU s / qualified |")
    A("|---|---|---|---|---|---|---|")
    for arm, c in rep["compute"].items():
        if arm.startswith("_"):
            continue
        A(f"| `{arm}` | {c['candidates']} | {c['restarts_timed']} | {_fmt(c['gpu_seconds_total'], 0)} | "
          f"{_fmt(c['gpu_seconds_median_per_restart'], 0)} | {c['qualified_full']} | "
          f"{_fmt(c['gpu_seconds_per_qualified_full'], 0)} |")
    A("")
    A("Baseline arms did not record per-restart timing, so their cost cells are `-` rather than")
    A("an imputed number.")
    A("")

    nd = rep["nondominated"]
    A(f"## Nondominated candidates ({len(nd)}) across sequence, pose and energy")
    A("")
    if nd:
        A("| arm | example | chem | identity | subs | CA RMSD | contacts | ddG (REU) | full |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in nd[:60]:
            A(f"| `{r['arm']}` | {r['example_id']} | {r['cyc_type_name']} | "
              f"{_fmt(r['seq_identity'])} | {r['n_substitutions']} | {_fmt(r['ca_rmsd_A'], 2)} | "
              f"{_fmt(r['contact_retention'])} | {_fmt(r['ddg_REU'], 2)} | "
              f"{_fmt(r['gate_joint_full'])} |")
    else:
        A("None: no structurally-qualified candidate also carries a paired energy score.")
    A("")
    A(f"Full per-candidate rows: `{rep['results_dir']}/report/best_per_lp.jsonl` and")
    A("`rescored/rescored_*.jsonl`.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescored", nargs="+", required=True,
                    help="rescored_*.jsonl from dflow_rescore.py, every arm.")
    ap.add_argument("--rosetta", nargs="*", default=[],
                    help="rosetta_shard*.jsonl from dflow_rosetta.py.")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--cost-log", nargs="*", default=[],
                    help="Optional slurm .out files; COST_SUMMARY lines are extracted.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--results-dir", default="")
    args = ap.parse_args()

    cands = _read_jsonl([Path(p) for p in args.rescored])
    if not cands:
        raise SystemExit("FATAL: no rescored rows; nothing to report.")
    ros = _read_jsonl([Path(p) for p in args.rosetta])

    cost = []
    for p in args.cost_log:
        for line in Path(p).read_text().splitlines():
            if line.startswith("COST_SUMMARY "):
                cost.append(json.loads(line[len("COST_SUMMARY "):]))

    join = attach_energy(cands, ros)
    cal = None
    if args.calibration and Path(args.calibration).exists():
        cal = json.loads(Path(args.calibration).read_text())

    rep = {
        "run_id": args.run_id or "dflow",
        "results_dir": args.results_dir or str(Path(args.out_dir).parent),
        "generated": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "contract": CONTRACT,
        "join": join,
        "calibration": cal,
        "arms": arm_table(cands),
        "paired": paired_init_opt(cands),
        "coverage_full": coverage(cands, "gate_joint_full"),
        "coverage_structural": coverage(cands, "gate_joint_structural"),
        "energy_by_topology": energy_by_topology(cands),
        "compute": compute_table(cands, cost),
        "nondominated": nondominated(cands),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    best = best_per_lp(cands, "gate_joint_full")
    with (out / "best_per_lp.jsonl").open("w") as fh:
        for r in best:
            fh.write(json.dumps(r, default=str) + "\n")
    (out / "RESULTS.md").write_text(markdown(rep))

    print(f"candidates {join['candidates']}, with energy {join['with_energy']}")
    for k, t in rep["arms"].items():
        print(f"  {k:28s} joint struct {_rate_str(t['gate_joint_structural'])}"
              f"  joint FULL {_rate_str(t['gate_joint_full'])}")
    print(f"wrote {out}/RESULTS.md, report.json, best_per_lp.jsonl")


if __name__ == "__main__":
    main()
