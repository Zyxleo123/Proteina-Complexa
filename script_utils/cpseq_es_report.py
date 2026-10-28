#!/usr/bin/env python3
"""Read an evolutionary-search pilot: did the proposer beat its matched control?

Reports exactly what the plan specifies and nothing it does not:

- **Mean post-relax interface dG over VALID folds**, with the valid-fold count out of
  folds attempted beside it. No minimum validity ratio is imposed, so the denominator
  travels with every number -- a -40 REU mean over 1 of 6 folds and the same mean over
  6 of 6 are not the same claim.
- **Cross-fitted.** Candidates are ranked on folds `0..depth/2` (the half the ES was
  allowed to see) and reported on `depth/2..depth` (the half it never saw). The project
  measured a -4.66 REU best-of-8 advantage vanish entirely under cross-fitting, so the
  held-out column is the only one that may be read as a result.
- **Matched folds.** The per-arm fold count is printed, because an arm folded more times
  than its control shows an advantage made of folds.
- One scoring protocol. A rescore appends rows, so pooling versions would count a
  re-scored fold once per version and mix validity labels from two different scorers.

Usage:
    python script_utils/cpseq_es_report.py --out-dir $ZFS/cpseq/es_pilot_lnr
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.evaluate import SCORING_PROTOCOL_VERSION  # noqa: E402

ARM_LLM, ARM_RANDOM, ARM_REF = "llm_search", "random_control", "reference"


def load(out_dir: Path, version: str | None) -> tuple[list[dict], dict]:
    cohort = json.loads((out_dir / "cohort.json").read_text())
    rows, seen_versions = [], set()
    for pid in cohort["cohort"]:
        p = out_dir / "state" / pid / "outcomes.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                o = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            if o.get("stage") != "ladder" or o.get("technical_failure"):
                continue
            sv = o.get("scoring_version")
            if sv:
                seen_versions.add(sv)
            if version and sv and sv != version:
                continue
            rows.append(o)
    return rows, {"cohort": cohort, "versions": sorted(seen_versions)}


def agg(folds: list[dict]) -> dict:
    """Mean dG over the valid folds, and the denominator it came from."""
    dgs = [
        float(f["best_valid_dG_separated"])
        for f in folds
        if f.get("valid") and f.get("best_valid_dG_separated") is not None
    ]
    return {
        "n_folds": len(folds),
        "n_valid": sum(1 for f in folds if f.get("valid")),
        "n_valid_prerelax": sum(1 for f in folds if f.get("prerelax_valid")),
        "mean_dG": st.fmean(dgs) if dgs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scoring-version", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    want = args.scoring_version or SCORING_PROTOCOL_VERSION
    rows, meta = load(out_dir, want)
    if not rows:
        print(f"[es-report] no ladder rows at scoring version {want}")
        return 1
    depth = len(meta["cohort"].get("seeds") or []) or 1
    half = depth // 2
    print(f"[es-report] {out_dir.name}: {len(rows)} fold(s) at {want}")
    if len(meta["versions"]) > 1:
        print(f"[es-report] other versions on disk (excluded): "
              f"{[v for v in meta['versions'] if v != want]}")
    print(f"[es-report] fold depth {depth}: select on ranks 0..{half - 1}, "
          f"report on {half}..{depth - 1}\n")

    # --- fold budget per arm, the precondition for any comparison ------------------
    by_arm_folds: dict[str, int] = defaultdict(int)
    by_arm_seqs: dict[str, set] = defaultdict(set)
    for o in rows:
        by_arm_folds[o.get("arm")] += 1
        by_arm_seqs[o.get("arm")].add(o.get("sequence"))
    print("=== fold budget (a mismatch here invalidates everything below) ===")
    for arm in sorted(by_arm_folds):
        print(f"  {arm:16s} {by_arm_folds[arm]:4d} fold(s) over "
              f"{len(by_arm_seqs[arm]):3d} sequence(s)")

    # --- per complex x arm, cross-fitted -------------------------------------------
    per: dict[tuple, list[dict]] = defaultdict(list)
    for o in rows:
        rank = o.get("seed_rank")
        if rank is None:
            continue
        side = "select" if rank < half else "measure"
        per[(o.get("parent_id"), o.get("arm"), side)].append(o)

    parents = sorted({o.get("parent_id") for o in rows})
    print("\n=== best candidate per arm, chosen on SELECT, reported on MEASURE ===")
    print("   (mean dG over valid folds; valid/attempted in brackets)")
    hdr = f"  {'complex':22s} {'arm':15s} {'chosen on select':>22s} {'held-out measure':>22s}"
    print(hdr)
    deltas = []
    for pid in parents:
        ref_m = agg(per.get((pid, ARM_REF, "measure"), []))
        for arm in (ARM_LLM, ARM_RANDOM):
            sel = per.get((pid, arm, "select"), [])
            mea = per.get((pid, arm, "measure"), [])
            # Rank sequences on the select half only, then report that sequence's
            # held-out folds. This is the whole point: picking on the measure half
            # would make the held-out number a function of the folds it holds out.
            by_seq_sel: dict[str, list[dict]] = defaultdict(list)
            for o in sel:
                by_seq_sel[o["sequence"]].append(o)
            ranked = [
                (s, a) for s, fs in by_seq_sel.items()
                if (a := agg(fs))["mean_dG"] is not None
            ]
            if not ranked:
                print(f"  {pid:22s} {arm:15s} {'no valid candidate':>22s}")
                continue
            ranked.sort(key=lambda t: t[1]["mean_dG"])
            best_seq, best_sel = ranked[0]
            held = agg([o for o in mea if o["sequence"] == best_seq])
            s_txt = f"{best_sel['mean_dG']:+.2f} [{best_sel['n_valid']}/{best_sel['n_folds']}]"
            m_txt = (f"{held['mean_dG']:+.2f} [{held['n_valid']}/{held['n_folds']}]"
                     if held["mean_dG"] is not None
                     else f"invalid [0/{held['n_folds']}]")
            print(f"  {pid:22s} {arm:15s} {s_txt:>22s} {m_txt:>22s}   {best_seq}")
            if held["mean_dG"] is not None and ref_m["mean_dG"] is not None:
                deltas.append((pid, arm, held["mean_dG"] - ref_m["mean_dG"]))
        r_txt = (f"{ref_m['mean_dG']:+.2f} [{ref_m['n_valid']}/{ref_m['n_folds']}]"
                 if ref_m["mean_dG"] is not None else f"invalid [0/{ref_m['n_folds']}]")
        print(f"  {pid:22s} {'reference':15s} {'':>22s} {r_txt:>22s}")

    # --- did the generations actually improve? -------------------------------------
    print("\n=== LLM arm by generation (select half, the signal the ES optimised) ===")
    gens: dict[int, list[dict]] = defaultdict(list)
    for o in rows:
        if o.get("arm") != ARM_LLM or (o.get("seed_rank") or 0) >= half:
            continue
        gens[int(o.get("round_index") or 0)].append(o)
    for g in sorted(gens):
        a = agg(gens[g])
        best = min(
            (float(o["best_valid_dG_separated"]) for o in gens[g]
             if o.get("valid") and o.get("best_valid_dG_separated") is not None),
            default=None,
        )
        mean = f"{a['mean_dG']:+.2f}" if a["mean_dG"] is not None else "n/a"
        bst = f"{best:+.2f}" if best is not None else "n/a"
        print(f"  generation {g}: {a['n_folds']:3d} fold(s), "
              f"{a['n_valid']:3d} valid post-relax, {a['n_valid_prerelax']:3d} pre-relax, "
              f"mean {mean} REU, best {bst} REU")

    # --- ancestry: did offspring descend from the elite that was meant to make them?
    print("\n=== ancestry (elite -> offspring, the thing a leaderboard prompt cannot show) ===")
    anc = defaultdict(set)
    for o in rows:
        if o.get("arm") == ARM_LLM and o.get("ancestor_sequence"):
            anc[o["ancestor_sequence"]].add(o["sequence"])
    n_with = sum(1 for o in rows if o.get("arm") == ARM_LLM and o.get("ancestor_sequence"))
    n_llm = sum(1 for o in rows if o.get("arm") == ARM_LLM)
    print(f"  {n_with}/{n_llm} LLM fold(s) name an ancestor; "
          f"{len(anc)} distinct elite(s) produced offspring")

    print("\n=== held-out delta vs the unmutated reference (negative = better) ===")
    for arm in (ARM_LLM, ARM_RANDOM):
        ds = [d for _, a, d in deltas if a == arm]
        if ds:
            print(f"  {arm:16s} median {st.median(ds):+.2f} REU over "
                  f"{len(ds)} complex(es)   per-complex {[round(d, 1) for d in ds]}")
        else:
            print(f"  {arm:16s} no complex with both a valid held-out candidate and "
                  f"a valid reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
