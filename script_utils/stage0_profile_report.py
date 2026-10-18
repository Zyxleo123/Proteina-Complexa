"""Stage 0 report: what the existing LP -> CP system costs, and whether it is measuring.

Two questions, answered from artefacts that are already on disk:

1. COST. Seconds and peak GPU memory per generated structure, CPU core-seconds per scored
   structure, and bytes written per structure -- broken out by pocket size and by whether
   guidance was on, because those are the two axes that move the numbers. The point is to
   size the Stage 1 tranche before launching it, so the table ends with an extrapolation to
   a requested structure count and a verdict against the declared GPU/CPU caps.

2. INSTRUMENT HEALTH. A cost figure for a pipeline that is not actually measuring anything is
   worse than no figure, so the same pass re-checks the preflight items: decoded-sequence
   compliance and mutation counting, receptor-frame consistency, and closure chemistry.

Two traps this file exists to not fall into
-------------------------------------------
ABSTENTION. When the cyclization head finds no candidate anchor pair it abstains, and the
per-chemistry `cyc/*_bond_success` fields come back NaN. NaN is truthy in Python, so
`if row["cyc/disulfide_bond_success"]:` scores a 100%-abstention cell as 100% closure. Every
closure figure here gates on `requested_type_satisfied` first and reports the abstention rate
beside the closure rate, never one without the other.

SETUP OVERHEAD. Model load is paid once per JOB (a 2.9 GB flow checkpoint plus a 4.1 GB AE off
ZFS), not once per structure. Folding it into a per-structure mean makes a 16-structure probe
look far more expensive per unit than the tranche it is being used to size. Generation seconds
and setup seconds are therefore kept apart all the way to the extrapolation, which adds setup
back in per job.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

CHEMISTRIES = ("mainchain", "disulfide", "isopeptide")


def _finite(x):
    """True only for a real number. Guards the NaN-is-truthy trap described in the docstring."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _mean(xs):
    xs = [x for x in xs if _finite(x)]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, nd=1):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def read_rows(results_dir: Path) -> list[dict]:
    rows = []
    files = sorted(results_dir.glob("edits_shard*.jsonl"))
    if not files:
        raise SystemExit(f"FATAL: no edits_shard*.jsonl under {results_dir}. The GPU job wrote nothing.")
    for f in files:
        for lineno, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Refuse, do not skip. Concurrent O_APPEND has NUL-corrupted rows on this
                # cluster; silently dropping them would quietly shrink the denominator of
                # every rate below and the report would still look clean.
                raise SystemExit(f"FATAL: corrupt row {f}:{lineno}. Delete the file and rerun.")
    return rows


def cost_table(rows: list[dict], cases: dict) -> dict:
    """Per-structure GPU cost, split by pocket end and guidance arm."""
    end_of = {c["example_id"]: c["pocket_end"] for c in cases.get("cases", [])}
    recept_of = {c["example_id"]: c["receptor_length"] for c in cases.get("cases", [])}

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        end = end_of.get(r.get("example_id"), "unknown")
        arm = "guided" if _finite(r.get("guidance_w")) and r["guidance_w"] > 0 else "unguided"
        groups[(end, arm)].append(r)

    out = {}
    for (end, arm), rs in sorted(groups.items()):
        out[f"{end}/{arm}"] = {
            "n": len(rs),
            "receptor_length_mean": _mean([recept_of.get(r.get("example_id")) for r in rs]),
            "gen_seconds_mean": _mean([r.get("prof/gen_seconds") for r in rs]),
            "gen_seconds_max": max(
                (r["prof/gen_seconds"] for r in rs if _finite(r.get("prof/gen_seconds"))), default=None),
            "peak_gpu_alloc_MB_max": max(
                (r["prof/peak_gpu_alloc_MB"] for r in rs if _finite(r.get("prof/peak_gpu_alloc_MB"))), default=None),
            "peak_gpu_reserved_MB_max": max(
                (r["prof/peak_gpu_reserved_MB"] for r in rs if _finite(r.get("prof/peak_gpu_reserved_MB"))), default=None),
        }
    return out


def health_table(rows: list[dict]) -> dict:
    """Is the instrument measuring? Compliance, frame, and closure -- abstention-aware."""
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]

    abstained = [r for r in ok if r.get("requested_type_satisfied") != 1]
    satisfied = [r for r in ok if r.get("requested_type_satisfied") == 1]

    closure = {}
    for chem in CHEMISTRIES:
        key = f"cyc/{chem}_bond_success"
        req = [r for r in ok if r.get("cyc_type") == chem]
        if not req:
            continue
        # Denominator is the rows that BOTH requested this chemistry and did not abstain.
        # An abstained row has NaN here and must not enter either side of the fraction.
        pool = [r for r in satisfied if r.get("cyc_type") == chem and _finite(r.get(key))]
        closure[chem] = {
            "requested": len(req),
            "scorable": len(pool),
            "abstained": len(req) - len(pool),
            "closure_rate_of_scorable": (sum(1 for r in pool if r[key] >= 1.0) / len(pool)) if pool else None,
        }

    frames = [r.get("frame_residual_A") for r in ok if _finite(r.get("frame_residual_A"))]
    return {
        "n_rows": len(rows),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "failure_reasons": sorted({str(r.get("error", "?"))[:120] for r in failed})[:5],
        "abstention_rate": (len(abstained) / len(ok)) if ok else None,
        "closure_by_chemistry": closure,
        # Mutation counting. The Stage 1 budget is defined on this column, so if it is absent
        # or constant the budget is not enforceable, and that has to surface here, not later.
        "n_substitutions": {
            "present_on_rows": sum(1 for r in ok if _finite(r.get("n_substitutions"))),
            "mean": _mean([r.get("n_substitutions") for r in ok]),
            "min": min((r["n_substitutions"] for r in ok if _finite(r.get("n_substitutions"))), default=None),
            "max": max((r["n_substitutions"] for r in ok if _finite(r.get("n_substitutions"))), default=None),
        },
        "peptide_ca_rmsd_to_input_A_mean": _mean([r.get("ca_rmsd_to_input_A") for r in ok]),
        "peptide_ca_max_dev_to_input_A_max": max(
            (r["ca_max_dev_to_input_A"] for r in ok if _finite(r.get("ca_max_dev_to_input_A"))), default=None),
        # Receptor-frame consistency: the Kabsch residual of the input peptide onto its own
        # crystal pose. Complexes are refused above the tolerance rather than written
        # misplaced, so rows WITHOUT this field had no complex written at all.
        "frame_residual_A": {
            "rows_with_complex": len(frames),
            "rows_without_complex": len(ok) - len(frames),
            "max": max(frames, default=None),
        },
    }


def output_bytes(results_dir: Path) -> dict:
    def du(p: Path):
        if not p.is_dir():
            return None, 0
        files = [f for f in p.rglob("*") if f.is_file()]
        return sum(f.stat().st_size for f in files), len(files)

    pdb_b, pdb_n = du(results_dir / "pdb")
    cpx_b, cpx_n = du(results_dir / "complexes")
    jsonl_b = sum(f.stat().st_size for f in results_dir.glob("edits_shard*.jsonl"))
    return {
        "peptide_pdb_bytes": pdb_b, "peptide_pdb_files": pdb_n,
        "complex_pdb_bytes": cpx_b, "complex_pdb_files": cpx_n,
        "edits_jsonl_bytes": jsonl_b,
        "total_bytes": (pdb_b or 0) + (cpx_b or 0) + jsonl_b,
    }


def extrapolate(cost, rosetta, nbytes, n_structures, setup_seconds,
                structures_per_job, gpu_cap_hours, cpu_cap_core_hours) -> dict:
    """Project the measured per-unit costs onto a requested tranche size.

    Setup is added once per JOB, not per structure. Parallel workers shorten elapsed time but
    do not remove total compute, so this is a total-cost projection, not a runtime prediction.
    """
    total_n = sum(v["n"] for v in cost.values())
    timed = [v for v in cost.values() if _finite(v.get("gen_seconds_mean"))]
    if not timed or not total_n:
        return {"error": "no per-structure timings; was the GPU job run with PROFILE=1?"}
    # Weighted by the measured group sizes, so an unbalanced probe is not read as balanced.
    gen_s = sum(v["gen_seconds_mean"] * v["n"] for v in timed) / sum(v["n"] for v in timed)

    n_jobs = max(1, math.ceil(n_structures / max(1, structures_per_job)))
    setup_total = (setup_seconds or 0.0) * n_jobs
    gpu_hours = (gen_s * n_structures + setup_total) / 3600.0

    per_row = rosetta.get("core_seconds_per_scored_row")
    cpu_core_hours = (per_row * n_structures / 3600.0) if _finite(per_row) else None
    per_struct_bytes = nbytes["total_bytes"] / total_n

    return {
        "n_structures": n_structures,
        "assumed_structures_per_job": structures_per_job,
        "assumed_n_jobs": n_jobs,
        "gen_seconds_per_structure": round(gen_s, 2),
        "setup_seconds_per_job": setup_seconds,
        "gpu_hours": round(gpu_hours, 2),
        "gpu_cap_hours": gpu_cap_hours,
        "gpu_within_cap": gpu_hours <= gpu_cap_hours,
        "cpu_core_hours": None if cpu_core_hours is None else round(cpu_core_hours, 1),
        "cpu_cap_core_hours": cpu_cap_core_hours,
        "cpu_within_cap": None if cpu_core_hours is None else cpu_core_hours <= cpu_cap_core_hours,
        "output_GB": round(per_struct_bytes * n_structures / 2**30, 2),
    }


def render_markdown(meta, cost, health, rosetta, nbytes, proj) -> str:
    L = []
    a = L.append
    a("# Stage 0 profile — LP → CP, existing system\n")
    a(f"- results dir: `{meta['results_dir']}`")
    a(f"- git revision: `{meta.get('git_rev', '?')}`")
    a(f"- flow checkpoint: `{meta.get('flow_ckpt', '?')}`")
    a(f"- autoencoder: `{meta.get('ae_ckpt', '?')}`")
    a(f"- metadata: `{meta.get('metadata', '?')}`")
    a(f"- sampler: nsteps={meta.get('nsteps', '?')}, t_ca={meta.get('t_ca', '?')}, "
      f"t_lat={meta.get('t_lat', '?')}, chemistry={meta.get('cyc_types', '?')}")
    a(f"- GPU: {meta.get('gpu', '?')}\n")
    # A smoke run uses a coarse sampler, and its per-structure cost is proportionally lower.
    # Quoting those numbers as the profile would understate Stage 1 by the step ratio, so the
    # report says so on its own face rather than relying on whoever reads it to check nsteps.
    nsteps = str(meta.get("nsteps", ""))
    if nsteps and not nsteps.startswith("0"):
        a(f"> **Not a production profile.** nsteps={nsteps}, but the design sampler's own "
          "value is 400. The costs below scale roughly with step count — do not size Stage 1 "
          "from this run.\n")

    a("## Cost per generated structure\n")
    a("| group | n | receptor len | gen s (mean) | gen s (max) | peak alloc MB | peak reserved MB |")
    a("|---|---|---|---|---|---|---|")
    for k, v in cost.items():
        a(f"| {k} | {v['n']} | {_fmt(v['receptor_length_mean'], 0)} | {_fmt(v['gen_seconds_mean'], 2)} | "
          f"{_fmt(v['gen_seconds_max'], 2)} | {_fmt(v['peak_gpu_alloc_MB_max'])} | "
          f"{_fmt(v['peak_gpu_reserved_MB_max'])} |")
    a("")
    a(f"Setup (model load, dataset build), paid once per job: **{_fmt(meta.get('setup_seconds'))} s**.\n")

    a("## CPU scoring (Rosetta interface dG)\n")
    if rosetta:
        a(f"- scored rows: {rosetta.get('n_scored_rows')}")
        a(f"- CPU core-seconds: {_fmt(rosetta.get('cpu_core_seconds'))} "
          f"(wall {_fmt(rosetta.get('wall_seconds'))} s on {rosetta.get('cpus_allocated')} cores)")
        a(f"- per scored row: {_fmt(rosetta.get('core_seconds_per_scored_row'), 2)} core-s")
        a(f"- max RSS: {_fmt(rosetta.get('max_rss_MB'))} MB\n")
    else:
        a("_No Rosetta cost sidecar — the CPU stage was gated or has not run._\n")

    a("## Output size\n")
    a(f"- peptide PDBs: {nbytes['peptide_pdb_files']} files, {nbytes['peptide_pdb_bytes'] or 0:,} B")
    a(f"- complex PDBs: {nbytes['complex_pdb_files']} files, {nbytes['complex_pdb_bytes'] or 0:,} B")
    a(f"- edits JSONL: {nbytes['edits_jsonl_bytes']:,} B")
    a(f"- **total: {nbytes['total_bytes']:,} B**\n")

    a("## Is the instrument measuring?\n")
    a(f"- rows: {health['n_rows']} ({health['n_ok']} ok, {health['n_failed']} failed)")
    for r in health["failure_reasons"]:
        a(f"  - `{r}`")
    a(f"- abstention rate (no anchor pair found, chemistry unscorable): "
      f"{_fmt((health['abstention_rate'] or 0) * 100)}%")
    a("")
    a("| chemistry | requested | scorable | abstained | closure of scorable |")
    a("|---|---|---|---|---|")
    for chem, v in health["closure_by_chemistry"].items():
        rate = v["closure_rate_of_scorable"]
        a(f"| {chem} | {v['requested']} | {v['scorable']} | {v['abstained']} | "
          f"{'n/a' if rate is None else f'{rate * 100:.0f}%'} |")
    a("")
    ns = health["n_substitutions"]
    a(f"- mutation counting: present on {ns['present_on_rows']}/{health['n_ok']} rows, "
      f"range {ns['min']}–{ns['max']} (mean {_fmt(ns['mean'], 2)})")
    a(f"- peptide CA RMSD to input: {_fmt(health['peptide_ca_rmsd_to_input_A_mean'], 2)} Å mean, "
      f"worst-CA {_fmt(health['peptide_ca_max_dev_to_input_A_max'], 2)} Å "
      "(all-residue, not motif-restricted — see the Stage 1 gap note)")
    fr = health["frame_residual_A"]
    a(f"- receptor-frame consistency: {fr['rows_with_complex']} rows placed "
      f"(max Kabsch residual {_fmt(fr['max'], 3)} Å), {fr['rows_without_complex']} refused\n")

    a("## Projection to the Stage 1 tranche\n")
    if "error" in proj:
        a(f"_{proj['error']}_\n")
    else:
        a(f"- structures: {proj['n_structures']:,} "
          f"(assumed {proj['assumed_n_jobs']} job(s) × {proj['assumed_structures_per_job']})")
        a(f"- GPU: **{proj['gpu_hours']} h** vs cap {proj['gpu_cap_hours']} h — "
          + ("WITHIN CAP" if proj["gpu_within_cap"]
             else "OVER CAP: cut candidates equally across arms before launching"))
        cpu = proj["cpu_core_hours"]
        a(f"- CPU: **{'n/a' if cpu is None else f'{cpu} core-h'}** vs cap "
          f"{proj['cpu_cap_core_hours']} core-h — "
          + ("n/a" if proj["cpu_within_cap"] is None
             else ("WITHIN CAP" if proj["cpu_within_cap"] else "OVER CAP")))
        a(f"- output: ~{proj['output_GB']} GB\n")
        a("These are total-compute projections at the measured per-unit rates. Parallel workers "
          "shorten elapsed time without reducing the total, and a pocket mix heavier than the "
          "profiled one moves the GPU figure.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-prefix", default=None,
                    help="Written as <prefix>.json and <prefix>.md. "
                         "Default: <results-dir>/profile/stage0_profile")
    ap.add_argument("--n-structures", type=int, default=1360,
                    help="Tranche size to project onto. 1360 is the plan's Stage 1 ceiling "
                         "(24 cases x 3 arms x 16 conformations, plus finalist re-evaluation).")
    ap.add_argument("--structures-per-job", type=int, default=240,
                    help="How many structures one GPU job does, so setup is billed once per "
                         "job rather than once per structure.")
    ap.add_argument("--gpu-cap-hours", type=float, default=24.0)
    ap.add_argument("--cpu-cap-core-hours", type=float, default=300.0)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    profile_dir = results_dir / "profile"
    out_prefix = Path(args.out_prefix) if args.out_prefix else profile_dir / "stage0_profile"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def load(p):
        return json.loads(p.read_text()) if p.is_file() else {}

    cases = load(profile_dir / "cases.json")
    meta = load(profile_dir / "run_meta.json")
    meta.setdefault("results_dir", str(results_dir))
    rosetta = load(profile_dir / "rosetta_cost.json")

    rows = read_rows(results_dir)
    cost = cost_table(rows, cases)
    health = health_table(rows)
    nbytes = output_bytes(results_dir)
    proj = extrapolate(cost, rosetta, nbytes, args.n_structures, meta.get("setup_seconds"),
                       args.structures_per_job, args.gpu_cap_hours, args.cpu_cap_core_hours)

    payload = {"meta": meta, "cases": cases, "cost_per_structure": cost,
               "instrument_health": health, "rosetta_cost": rosetta,
               "output_bytes": nbytes, "projection": proj}
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    md = render_markdown(meta, cost, health, rosetta, nbytes, proj)
    out_prefix.with_suffix(".md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {out_prefix.with_suffix('.json')}")
    print(f"wrote {out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
