#!/usr/bin/env python3
"""Which rule for ranking candidates survives being held out?

The search ranks a candidate by the **mean interface energy over its folds that passed
the validity checks**, with no minimum on how many passed. That was defensible when a
third of folds failed and the passing ratio carried information. It is not obviously
right now: under the current checks (`v5-junction-restrained-relax`) 997 of 1000 folds
pass, so the mean is taken over essentially every fold and the ratio says nothing -- while
the seed-to-seed energy spread stays at 4.3 units median. Ranking on a two-fold mean
against that spread is ranking on noise, whatever the passing ratio does.

So this compares ranking rules the only way that means anything: each rule scores every
candidate using the SELECTION half of its folds, and is then judged on the MEASUREMENT
half it never saw. A rule that picks a better candidate on held-out folds is better, and
no rule gets to be judged on the folds it was computed from.

Seven rules, and the two interesting ones are last:

- `mean_valid` -- what the search does today.
- `median_valid` -- the same, robust to one lucky fold.
- `min_ratio_half` -- disqualify candidates passing under half their folds, then rank by
  mean energy. This is the rule the critique of the current objective implies.
- `pass_then_energy` -- most passing folds first, energy as the tie-break. §12 item 4.
- `penalized_mean` -- a failed fold is charged the worst energy seen on that target
  rather than dropped, so a candidate cannot improve its score by failing.
- `mean_per_area` -- §13's area-normalised energy, which reversed the buried-area trade.
- `shrunk_mean` -- the candidate's mean pulled toward the target's average by how little
  evidence it rests on, with the pull estimated from the data (pooled within-candidate
  variance against between-candidate variance). This is the rule that takes the seed
  spread seriously instead of hoping it averages out, and nothing in the project has
  tried it.

Reads the journal only. Nothing is folded, nothing is re-scored.

Usage:
    .venv/bin/python script_utils/cpseq_rank_rules.py \
        --run-dir /data/user_data/yixiz/cpseq/es_pilot_lnr \
        --run-dir /data/user_data/yixiz/cpseq/es_pilot_pep \
        --scoring-version v5-junction-restrained-relax \
        --out /data/user_data/yixiz/cpseq/rank_rules_v5.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_SCORING_VERSION = "v5-junction-restrained-relax"

RULES = (
    "mean_valid",
    "median_valid",
    "min_ratio_half",
    "pass_then_energy",
    "penalized_mean",
    "mean_per_area",
    "shrunk_mean",
)


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def load_rows(run_dir: Path, scoring_version: str | None) -> tuple[dict, dict]:
    """Newest row per (sequence, rank, arm, stage), by target. §14.3's fix."""
    newest: dict[str, dict[tuple, dict]] = defaultdict(dict)
    census: dict[str, int] = defaultdict(int)
    for path in sorted((run_dir / "state").glob("*/outcomes.jsonl")):
        target = path.parent.name
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            census[str(r.get("scoring_version"))] += 1
            if r.get("technical_failure"):
                continue
            sv = r.get("scoring_version")
            if scoring_version is not None and sv is not None and sv != scoring_version:
                continue
            key = (r.get("sequence"), r.get("seed_rank"), r.get("arm"), r.get("stage"))
            newest[target][key] = r
    return {t: list(d.values()) for t, d in newest.items()}, dict(census)


def split_half(rows: list[dict]) -> tuple[int, int]:
    ranks = [r.get("seed_rank") for r in rows if r.get("seed_rank") is not None]
    depth = (max(ranks) + 1) if ranks else 0
    return depth, depth // 2


def half_stats(folds: list[dict]) -> dict:
    """One sequence's standing over one half of its folds."""
    dg = [f["best_valid_dG_separated"] for f in folds
          if f.get("valid") and f.get("best_valid_dG_separated") is not None]
    area = [f["refined_dG_separated_per_1000A2"] for f in folds
            if f.get("valid") and f.get("refined_dG_separated_per_1000A2") is not None]
    return {
        "n": len(folds),
        "n_valid": sum(1 for f in folds if f.get("valid")),
        "energies": dg,
        "areas": area,
        "mean": st.fmean(dg) if dg else None,
        "median": st.median(dg) if dg else None,
        "mean_area": st.fmean(area) if area else None,
    }


# --------------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------------
#
# Every rule returns a score where LOWER IS BETTER, and None when it refuses to rank the
# candidate at all. A refusal is not a bad score: a rule that disqualifies a candidate is
# saying "do not pick this", and scoring it as +inf would let it win a target where
# everything was disqualified.


def shrinkage_context(cands: dict[str, dict]) -> dict:
    """Empirical-Bayes pieces for one target, from the selection half only.

    `within` is the pooled seed-to-seed variance of a single fold, measured from the
    candidates that have at least two valid folds. `between` is how much the candidates
    genuinely differ, which is the observed spread of their means minus the part of that
    spread the seed noise alone would produce. If `between` comes out at or below zero,
    the candidates are indistinguishable on this half and the rule shrinks all the way to
    the target mean -- which is the honest answer, not a failure.
    """
    within_parts: list[float] = []
    for c in cands.values():
        e = c["sel"]["energies"]
        if len(e) >= 2:
            within_parts.append(st.variance(e))
    within = st.fmean(within_parts) if within_parts else None
    means = [c["sel"]["mean"] for c in cands.values() if c["sel"]["mean"] is not None]
    ns = [len(c["sel"]["energies"]) for c in cands.values() if c["sel"]["energies"]]
    grand = st.fmean(means) if means else None
    if within is None or len(means) < 3:
        return {"grand": grand, "within": within, "between": None}
    observed = st.variance(means)
    n_bar = st.fmean(ns) if ns else 1.0
    between = max(0.0, observed - (within / max(1.0, n_bar)))
    return {"grand": grand, "within": within, "between": between}


def score(rule: str, c: dict, ctx: dict) -> float | None:
    sel = c["sel"]
    if rule == "mean_valid":
        return sel["mean"]
    if rule == "median_valid":
        return sel["median"]
    if rule == "min_ratio_half":
        if sel["n"] and sel["n_valid"] / sel["n"] < 0.5:
            return None
        return sel["mean"]
    if rule == "pass_then_energy":
        if sel["mean"] is None:
            return None
        # One scalar, so the ordering is explicit: each additional failed fold costs more
        # than any energy difference seen on this cohort (max spread ~22 units).
        return -1000.0 * sel["n_valid"] + sel["mean"]
    if rule == "penalized_mean":
        if not sel["energies"]:
            return None
        worst = ctx.get("worst")
        if worst is None:
            return sel["mean"]
        n_fail = sel["n"] - sel["n_valid"]
        return (sum(sel["energies"]) + n_fail * worst) / max(1, sel["n"])
    if rule == "mean_per_area":
        return sel["mean_area"]
    if rule == "shrunk_mean":
        m, n = sel["mean"], len(sel["energies"])
        if m is None or not n:
            return None
        grand, within, between = ctx.get("grand"), ctx.get("within"), ctx.get("between")
        if grand is None or within is None or between is None:
            return m
        w = between / (between + within / n) if (between + within / n) > 0 else 0.0
        return grand + w * (m - grand)
    raise ValueError(rule)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _sign_test(deltas: list[float]) -> dict:
    neg = sum(1 for d in deltas if d < 0)
    pos = sum(1 for d in deltas if d > 0)
    n = neg + pos
    if n == 0:
        return {"n": 0, "n_better": 0, "n_worse": 0, "p": None}
    k = min(neg, pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {"n": n, "n_better": neg, "n_worse": pos, "p": round(min(1.0, 2 * tail), 4)}


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(st.fmean(xs), 3) if xs else None


def build_targets(runs: list[tuple[Path, dict]], min_candidates: int) -> list[dict]:
    """One entry per target: its candidates, each split into the two halves of its folds."""
    out: list[dict] = []
    for run_dir, by_target in runs:
        for target, rows in sorted(by_target.items()):
            depth, mid = split_half(rows)
            if mid < 1:
                continue
            by_seq: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                if r.get("arm") == "reference":
                    continue  # the baseline is not a candidate the search may pick
                by_seq[r["sequence"]].append(r)
            cands: dict[str, dict] = {}
            for seq, folds in by_seq.items():
                sel = [f for f in folds
                       if f.get("seed_rank") is not None and f["seed_rank"] < mid]
                mea = [f for f in folds
                       if f.get("seed_rank") is not None and f["seed_rank"] >= mid]
                if not sel or not mea:
                    continue
                cands[seq] = {"sel": half_stats(sel), "mea": half_stats(mea)}
            # A rule can only be compared where there is a choice to make.
            if len(cands) < min_candidates:
                continue
            worst = max(
                (e for c in cands.values() for e in c["sel"]["energies"]), default=None
            )
            ctx = shrinkage_context(cands)
            ctx["worst"] = worst
            out.append({"run": run_dir.name, "target": target, "depth": depth,
                        "mid": mid, "cands": cands, "ctx": ctx})
    return out


def evaluate(targets: list[dict]) -> dict:
    per_rule: dict[str, dict] = {r: {"picks": [], "rho": [], "no_valid": 0,
                                     "abstained": 0, "true_best": 0} for r in RULES}
    for t in targets:
        cands, ctx = t["cands"], t["ctx"]
        truth = {s: c["mea"]["mean"] for s, c in cands.items()}
        scorable = [s for s, v in truth.items() if v is not None]
        best_true = min(scorable, key=lambda s: truth[s]) if scorable else None
        for rule in RULES:
            scored = {s: score(rule, c, ctx) for s, c in cands.items()}
            live = {s: v for s, v in scored.items() if v is not None}
            if not live:
                per_rule[rule]["abstained"] += 1
                continue
            pick = min(live, key=lambda s: live[s])
            if truth.get(pick) is None:
                # The rule chose a candidate with no passing fold in the held-out half.
                per_rule[rule]["no_valid"] += 1
            else:
                per_rule[rule]["picks"].append(
                    {"target": t["target"], "run": t["run"], "sequence": pick,
                     "held_out": truth[pick],
                     "best_possible": truth[best_true] if best_true else None,
                     "regret": truth[pick] - truth[best_true] if best_true else None,
                     "n_candidates": len(cands)}
                )
                if pick == best_true:
                    per_rule[rule]["true_best"] += 1
            common = [s for s in live if truth.get(s) is not None]
            if len(common) >= 3:
                rho = _spearman([live[s] for s in common], [truth[s] for s in common])
                if rho is not None:
                    per_rule[rule]["rho"].append(rho)

    base = {p["target"] + "|" + p["run"]: p for p in per_rule["mean_valid"]["picks"]}
    out: dict = {
        "n_targets": len(targets),
        "mean_candidates_per_target": _mean([float(len(t["cands"])) for t in targets]),
        "mean_selection_folds": _mean(
            [float(t["mid"]) for t in targets]
        ),
        "note": (
            "Every rule scores candidates on the SELECTION half and is judged on the "
            "MEASUREMENT half. `held_out_energy` is the energy of the candidate the rule "
            "picked, on folds it never saw; lower is better. `regret` is how much worse "
            "that pick is than the best available candidate. `rho` is the within-target "
            "rank correlation between the rule's score and the held-out truth. "
            "`picked_unusable` counts targets where the rule's pick had no passing fold "
            "in the held-out half at all."
        ),
        "rules": {},
    }
    for rule in RULES:
        d = per_rule[rule]
        paired = []
        for p in d["picks"]:
            b = base.get(p["target"] + "|" + p["run"])
            if b is not None:
                paired.append(p["held_out"] - b["held_out"])
        out["rules"][rule] = {
            "n_targets_ranked": len(d["picks"]),
            "abstained_on": d["abstained"],
            "picked_unusable": d["no_valid"],
            "held_out_energy": _mean([p["held_out"] for p in d["picks"]]),
            "regret": _mean([p["regret"] for p in d["picks"]]),
            "picked_the_true_best": d["true_best"],
            "rho": _mean(d["rho"]),
            "vs_mean_valid": {"mean_delta": _mean(paired),
                              "sign_test": _sign_test(paired)},
        }
    out["picks"] = {r: per_rule[r]["picks"] for r in RULES}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scoring-version", default=DEFAULT_SCORING_VERSION)
    ap.add_argument("--min-candidates", type=int, default=4,
                    help="targets with fewer candidates in both halves are skipped: "
                         "there is no ranking decision to evaluate")
    args = ap.parse_args()

    sv = None if args.scoring_version == "all" else args.scoring_version
    runs: list[tuple[Path, dict]] = []
    for rd in args.run_dir:
        by_target, census = load_rows(rd, sv)
        if len(census) > 1:
            print(f"NOTE {rd.name} spans checking protocols: {census}; filtered to {sv!r}")
        if by_target:
            runs.append((rd, by_target))
    if not runs:
        print("no usable rows", file=sys.stderr)
        return 2

    targets = build_targets(runs, args.min_candidates)
    if not targets:
        print("no target has candidates folded in both halves", file=sys.stderr)
        return 2
    rep = evaluate(targets)
    rep["scoring_version"] = args.scoring_version
    rep["runs"] = [str(r[0]) for r in runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))

    print(f"\n{rep['n_targets']} target(s), "
          f"{rep['mean_candidates_per_target']} candidate(s) each, "
          f"{rep['mean_selection_folds']} selection fold(s)\n")
    print(f"{rep['note']}\n")
    print(f"{'rule':>18} {'ranked':>7} {'held_out':>9} {'regret':>7} {'best':>5} "
          f"{'unusable':>9} {'rho':>7} {'vs_cur':>7} {'sign_p':>7}")
    for rule, d in rep["rules"].items():
        v = d["vs_mean_valid"]
        print(f"{rule:>18} {d['n_targets_ranked']:>7} "
              f"{str(d['held_out_energy']):>9} {str(d['regret']):>7} "
              f"{d['picked_the_true_best']:>5} {d['picked_unusable']:>9} "
              f"{str(d['rho']):>7} {str(v['mean_delta']):>7} "
              f"{str(v['sign_test']['p']):>7}")
    print(f"\nreport: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
