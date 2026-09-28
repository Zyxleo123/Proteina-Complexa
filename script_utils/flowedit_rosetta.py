#!/usr/bin/env python3
"""Paired Rosetta interface scoring for LP->CP candidates. CPU only, no GPU, no model.

What is paired, and why that is the whole point
-----------------------------------------------
`dG_separated` is not comparable across targets -- a 357-residue receptor and a 294-residue
one have different interface scales -- so an absolute value says little. What is comparable
is the SAME complex before and after the edit:

    before   the input bound LINEAR peptide (frame["input_pdb"], written at sampling time)
    after    the edited cyclic peptide in that same receptor

and the paired delta, CP minus LP, is what "did cyclization cost binding?" means. The
contract's bar is dG_delta <= +5 REU, with <= 0 also reported. REU are Rosetta Energy Units,
not kcal/mol, and are not converted to one.

"Identical preparation, scoring and relaxation settings" -- with one intended difference
-----------------------------------------------------------------------------------------
Both sides go through the same `compute_rosetta_interface_metrics_single`, same scorefunction,
same coordinate-constrained FastRelax. The one difference is that the CP side declares its
closing bond and the LP side has none to declare, because it is linear. That is the
manipulation, not a confound, and it is stated rather than hidden.

Repeats
-------
FastRelax is stochastic, so a single score is a draw, not a value. Every shortlisted
candidate is scored `--repeats` times AND its LP baseline is scored the same number of times,
so the delta's spread reflects both sides. Medians are reported with the full per-repeat
list, so variability is auditable rather than asserted.

The LP baseline is cached per input complex: all candidates of one example share it, and
re-relaxing it per candidate would burn the budget on a number that does not change (while
also making the delta noisier than it needs to be by resampling the same quantity).

Shortlist
---------
Rosetta is the expensive step, so it runs on a shortlist chosen by the cheap gates. The
shortlist is spread ACROSS THE RETENTION FRONTIER rather than taken from the top: the
question is what binding costs as retention varies, and a top-retention-only sample cannot
answer it. Within each (method, example, chemistry) group candidates are binned by contact
retention and sampled round-robin from the bins.

Resumable, and shardable by candidate. Each shard writes its own JSONL: concurrent O_APPEND
has corrupted rows on this filesystem, and the aggregator refuses corrupt lines rather than
skipping them.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from proteinfoundation.evaluation.rosetta_energy import (  # noqa: E402
    compute_rosetta_interface_metrics_single,
    is_pyrosetta_available,
)

DG = "binder_rosetta_dG_separated"
DECLARED = "binder_rosetta_cyclic_topology_declared"
RING_POST = "binder_rosetta_ring_bond_distance_after_relax"
RING_PRE = "binder_rosetta_ring_bond_distance_norelax"


def load_rows(paths: list[Path]) -> list[dict]:
    """Read candidate JSONL rows, REFUSING corrupt lines rather than skipping them.

    A skipped line is silent data loss: concurrent appends on this filesystem have produced
    NUL-corrupted rows before, and a summary computed over the survivors looks clean.
    """
    rows = []
    for p in paths:
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"FATAL: corrupt row {p}:{lineno} ({exc}). This is data loss, not noise. "
                    "Re-run the shard that wrote this file; do not filter the line out."
                )
    return rows


def binder_endpoints_from_pdb(pdb: str, chain: str) -> tuple[int, int, str, str]:
    """(i_local, j_local, first_resname, last_resname) from the file's own numbering.

    `_declare_cyclic_bond` maps binder-local k to PDB resSeq k+1, so the local indices are
    derived from the real resids rather than assumed to be 0..L-1 -- a complex written with
    original residue numbering would otherwise fail endpoint mapping outright.
    """
    seen: dict[int, str] = {}
    for line in open(pdb):
        if line.startswith("ATOM") and line[21] == chain:
            seen[int(line[22:26])] = line[17:20].strip()
    if not seen:
        raise ValueError(f"no ATOM records for chain {chain!r} in {pdb}")
    ks = sorted(seen)
    return ks[0] - 1, ks[-1] - 1, seen[ks[0]], seen[ks[-1]]


def score_once(pdb: str, binder: str, targets: str, cyc_type: str | None) -> dict:
    """One scoring draw. `cyc_type=None` for the linear baseline, which has no ring."""
    kw = {}
    if cyc_type is not None:
        i, j, _ri, _rj = binder_endpoints_from_pdb(pdb, binder)
        kw = {"cyclization_type": cyc_type, "cyclization_i": i, "cyclization_j": j}
    return compute_rosetta_interface_metrics_single(pdb, binder, targets, **kw)


def score_repeats(pdb: str, binder: str, targets: str, cyc_type: str | None,
                  repeats: int) -> dict:
    """`repeats` independent draws. Returns medians, the raw list, and the spread."""
    dgs, declared, ring_pre, ring_post, raws = [], [], [], [], []
    for _ in range(repeats):
        m = score_once(pdb, binder, targets, cyc_type)
        dgs.append(m.get(DG, float("nan")))
        declared.append(m.get(DECLARED, float("nan")))
        ring_pre.append(m.get(RING_PRE, float("nan")))
        ring_post.append(m.get(RING_POST, float("nan")))
        raws.append(m)
    ok = [v for v in dgs if v == v]
    return {
        "dG_median": statistics.median(ok) if ok else float("nan"),
        "dG_all": dgs,
        "dG_min": min(ok) if ok else float("nan"),
        "dG_max": max(ok) if ok else float("nan"),
        "dG_spread": (max(ok) - min(ok)) if len(ok) > 1 else 0.0,
        "dG_n_ok": len(ok),
        "topology_declared": declared[0] if declared else float("nan"),
        "ring_bond_pre_median": statistics.median([v for v in ring_pre if v == v])
        if any(v == v for v in ring_pre) else float("nan"),
        "ring_bond_post_median": statistics.median([v for v in ring_post if v == v])
        if any(v == v for v in ring_post) else float("nan"),
        "first_raw": raws[0] if raws else {},
    }


def eligible(row: dict) -> bool:
    """Cheap gates, applied before spending Rosetta time on a candidate."""
    return (
        row.get("status") == "ok"
        and bool(row.get("pass_geometry_and_sequence"))
        and row.get("complex_pdb")
        and row.get("input_complex_pdb")
    )


def shortlist(rows: list[dict], per_group: int, bins: int = 4) -> list[dict]:
    """Spread across the retention frontier within each (method, example, chemistry).

    Binning by retention and sampling round-robin means the shortlist covers the range of
    poses the sampler produced. Taking the top by retention would answer "what does the most
    conservative edit cost?" and nothing about the trade-off.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("method"), r.get("example_id"), r.get("cyc_type"))].append(r)

    out = []
    for _key, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        grp = [g for g in grp if g.get("contact_retention") == g.get("contact_retention")]
        if not grp:
            continue
        grp.sort(key=lambda r: r["contact_retention"])
        buckets: list[list[dict]] = [[] for _ in range(bins)]
        lo = grp[0]["contact_retention"]
        hi = grp[-1]["contact_retention"]
        span = (hi - lo) or 1.0
        for g in grp:
            b = min(bins - 1, int((g["contact_retention"] - lo) / span * bins))
            buckets[b].append(g)
        # Highest retention first inside each bucket, then round-robin across buckets.
        for b in buckets:
            b.sort(key=lambda r: -r["contact_retention"])
        picked, k = [], 0
        while len(picked) < min(per_group, len(grp)):
            progressed = False
            for b in buckets:
                if k < len(b) and len(picked) < min(per_group, len(grp)):
                    picked.append(b[k])
                    progressed = True
            if not progressed:
                break
            k += 1
        out.extend(picked)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="Candidate JSONL files (globs already expanded by the shell).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--per-group", type=int, default=8,
                    help="Shortlist size per (method, example, chemistry).")
    ap.add_argument("--bins", type=int, default=4, help="Retention bins for the spread.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"FATAL: shard-index {args.shard_index} outside [0, {args.shard_count})")

    paths = [Path(p) for p in args.candidates]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} candidate file(s) missing, e.g. {missing[0]}")
    rows = load_rows(paths)
    ok_rows = [r for r in rows if eligible(r)]
    short = shortlist(ok_rows, args.per_group, args.bins)
    mine = short[args.shard_index :: args.shard_count]

    n_lp = len({r["input_complex_pdb"] for r in mine})
    print(f"{len(rows)} candidate rows -> {len(ok_rows)} eligible -> {len(short)} shortlisted "
          f"-> {len(mine)} in this shard", flush=True)
    print(f"cost: {len(mine)} CP x {args.repeats} + {n_lp} LP x {args.repeats} = "
          f"{(len(mine) + n_lp) * args.repeats} relax+score calls", flush=True)
    if not ok_rows and rows:
        # Distinguish "nothing passed the gates" from "no input": a silent empty output
        # would look identical downstream.
        reasons = defaultdict(int)
        for r in rows:
            if r.get("status") != "ok":
                reasons["status != ok"] += 1
            elif not r.get("pass_geometry_and_sequence"):
                reasons["failed geometry/sequence gates"] += 1
            elif not r.get("complex_pdb"):
                reasons["no complex written"] += 1
            elif not r.get("input_complex_pdb"):
                reasons["no input complex"] += 1
        print("NOTE: no eligible candidates. Breakdown: " + json.dumps(dict(reasons)), flush=True)
    if args.dry_run:
        print("DRY RUN OK", flush=True)
        return
    if not is_pyrosetta_available():
        raise SystemExit("FATAL: PyRosetta is not importable; this stage cannot run.")

    import pyrosetta
    pyrosetta.init("-mute all")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for lineno, line in enumerate(out_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["run_key"])
            except (json.JSONDecodeError, KeyError):
                raise SystemExit(f"FATAL: corrupt row {out_path}:{lineno}. Delete and rerun.")
    if done:
        print(f"resuming: {len(done)} already scored", flush=True)

    # One LP baseline per input complex, scored once and reused: it does not depend on the
    # candidate, and re-drawing it per candidate would add noise to every delta.
    lp_cache: dict[tuple, dict] = {}
    n_ok = n_fail = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for r in mine:
            key = r["run_key"]
            if key in done:
                continue
            binder = r.get("binder_chain") or "B"
            targets = r.get("target_chains") or "A"
            cyc = r["cyc_type"]
            try:
                cp = score_repeats(r["complex_pdb"], binder, targets, cyc, args.repeats)
                lp_key = (r["input_complex_pdb"], binder, targets)
                if lp_key not in lp_cache:
                    lp_cache[lp_key] = score_repeats(
                        r["input_complex_pdb"], binder, targets, None, args.repeats)
                lp = lp_cache[lp_key]
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                fh.write(json.dumps({"run_key": key, "status": "failed",
                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                fh.flush()
                print(f"  FAIL {key}: {type(exc).__name__}: {exc}", flush=True)
                continue

            delta = cp["dG_median"] - lp["dG_median"]
            row = {
                "run_key": key, "status": "ok",
                "example_id": r.get("example_id"), "method": r.get("method"),
                "cyc_type": cyc, "seed": r.get("seed"),
                "t_ca_start": r.get("t_ca_start"), "t_lat_start": r.get("t_lat_start"),
                "edit_end_frac": r.get("edit_end_frac"), "n_avg": r.get("n_avg"),
                "repeats": args.repeats,
                "cp_dG_median": cp["dG_median"], "cp_dG_all": cp["dG_all"],
                "cp_dG_spread": cp["dG_spread"],
                "lp_dG_median": lp["dG_median"], "lp_dG_all": lp["dG_all"],
                "lp_dG_spread": lp["dG_spread"],
                "dG_delta_REU": delta,
                "pass_ddg": bool(delta == delta and delta <= 5.0),
                "ddg_non_worsening": bool(delta == delta and delta <= 0.0),
                "topology_declared": cp["topology_declared"],
                "ring_bond_pre_median_A": cp["ring_bond_pre_median"],
                "ring_bond_post_median_A": cp["ring_bond_post_median"],
                # Strain proxy: how much relaxation had to fix. A large gain means the pose
                # as generated was strained, even if the relaxed score looks fine.
                "cp_relax_gain": (cp["first_raw"].get("binder_rosetta_total_score_norelax",
                                                      float("nan"))
                                  - cp["first_raw"].get("binder_rosetta_total_score_relaxed",
                                                        float("nan"))),
                # Carried through so the report never has to re-join against the GPU rows.
                "contact_retention": r.get("contact_retention"),
                "ca_rmsd_A": r.get("ca_rmsd_A"),
                "interface_ca_rmsd_A": r.get("interface_ca_rmsd_A"),
                "n_substitutions": r.get("n_substitutions"),
                "seq_identity": r.get("seq_identity"),
                "bond_dist_A": r.get("bond_dist_A"),
                "anchor_i_res": r.get("anchor_i_res"), "anchor_j_res": r.get("anchor_j_res"),
                "ae_rt_n_substitutions": r.get("ae_rt_n_substitutions"),
                "complex_pdb": r.get("complex_pdb"),
                "input_complex_pdb": r.get("input_complex_pdb"),
            }
            # The contract's full verdict, now that dG exists. Isopeptide rows whose topology
            # could not be declared (Asn/Gln acceptors) are marked so they can be excluded
            # from claimed isopeptide successes instead of counted as scored ones.
            row["accepted"] = bool(
                r.get("pass_geometry_and_sequence") and row["pass_ddg"]
                and (cyc != "isopeptide" or row["topology_declared"] == 1.0)
            )
            row["excluded_undeclared_isopeptide"] = bool(
                cyc == "isopeptide" and row["topology_declared"] != 1.0
            )
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_ok += 1
            if n_ok % 5 == 0:
                print(f"  {n_ok} scored ({time.time() - t0:.0f}s)", flush=True)

    print(f"done: {n_ok} scored, {n_fail} failed, {time.time() - t0:.0f}s", flush=True)
    if n_ok == 0 and n_fail > 0:
        raise SystemExit(f"FATAL: all {n_fail} scorings failed -- see the errors above.")


if __name__ == "__main__":
    main()
