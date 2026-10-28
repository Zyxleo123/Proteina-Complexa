#!/usr/bin/env python3
"""The draw ladder, read from folds already on disk. Safe to run against a live run.

Each random draw and the unchanged cyclized peptide were folded at the same ten seeds.
This turns those folds into a ladder over B, the number of draws a run is credited with,
and reports every number at matched fold counts.

Three rules, each of which exists because breaking it produced a result that was later
withdrawn (README_CPSEQ_PEPBENCH_SCALEUP sec. 3):

1. **A draw's outcome is a rate over its folds, never one fold.** A single fold of a
   sequence that passes 44 % of the time is a coin flip, and a table of coin flips has
   structure that looks like signal.

2. **The unchanged peptide gets the same number of folds as the design it is compared
   against.** Best-of-B against best-of-1 measures B, not the sequences. Every
   comparison here states the fold count on both sides.

3. **Energy is summarised by the MEDIAN over folds, never the minimum.** A minimum over
   more folds is lower whether or not the sequences differ: folding noise alone
   (sd 3.67 REU) reproduced the entire "energy improves with budget" column that was
   withdrawn. The median does not move with fold count.

The rungs are nested prefixes of one draw stream, so B=8 is literally the first 8 draws
of the 64. Nothing is subsampled and nothing is simulated.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_ladder_report.py --out-dir $ZFS/cpseq/seed_ladder
    python script_utils/cpseq_ladder_report.py --out-dir $ZFS/cpseq/seed_ladder \
        --depth 10 --json ladder_report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

REFERENCE_ARM = "reference"
# The arm whose draws this report reads. `--arm` selects it, so one run directory can
# hold more than one proposal mechanism and be reported once per mechanism against the
# SAME reference folds. The reference is written under its own arm and is shared, which
# is what makes an arm-to-arm comparison paired: both sides are measured against the
# identical folds of the unchanged peptide rather than against two independent foldings
# of it.
DRAW_ARM = "random_control"
STAGE = "ladder"
# The field a candidate is ranked and scored by. `best_valid_dG_separated` is the
# screening objective `evaluate.py` defines, so it is the default and every existing
# number in the reports was produced with it.
#
# It is settable because the choice of objective is itself a finding. Measured across 480
# folds of this run (`cpseq_unsat_probe.py`), WITHIN parent: buried area vs this
# objective rho -0.464 (15/17 parents negative), while Rosetta's refined buried-unsat
# count vs it is only +0.204. Dividing by buried area -- `refined_dG_separated_per_1000A2`,
# already recorded on every row -- reverses that: area falls to +0.255 and unsat rises to
# +0.493 (16/17 parents). So the two objectives reward measurably different interfaces,
# and which one a comparison used has to be a parameter rather than a line of code.
#
# Caveat carried wherever a non-default key is used: a row's `refined_*` fields all come
# from the one representative sample, and that sample was chosen by `refined_dG_separated`
# (`cpseq_search.py:571`). An alternative key therefore re-ranks CANDIDATES on folds whose
# per-fold representative was still picked by the default, which is a different question
# from having run the search under the alternative objective from the start.
ENERGY_KEY = "best_valid_dG_separated"


def load(out_dir: Path) -> dict[str, list[dict]]:
    """Ladder rows per complex, technical failures dropped.

    A technical failure is a fold that did not happen -- a predictor crash or timeout --
    not a structure that failed the checks. Counting it as a failed fold would put
    infrastructure in the denominator of a pass rate.
    """
    per: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted((out_dir / "state").glob("*/outcomes.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # A row still being appended to is not a corrupt run.
                continue
            if r.get("stage") != STAGE or r.get("technical_failure"):
                continue
            per[r.get("parent_id", f.parent.name)].append(r)
    return dict(per)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p at q=0.5, ties excluded by the caller."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact Fisher p for a 2x2 table, no scipy dependency."""
    def lp(x, y, z, w):
        n = x + y + z + w
        return math.exp(
            math.lgamma(x + y + 1) + math.lgamma(z + w + 1)
            + math.lgamma(x + z + 1) + math.lgamma(y + w + 1)
            - math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(y + 1)
            - math.lgamma(z + 1) - math.lgamma(w + 1)
        )
    obs = lp(a, b, c, d)
    total = 0.0
    row1, col1, n = a + b, a + c, a + b + c + d
    for x in range(max(0, col1 - (n - row1)), min(row1, col1) + 1):
        p = lp(x, row1 - x, col1 - x, n - row1 - col1 + x)
        if p <= obs * (1 + 1e-9):
            total += p
    return min(1.0, total)


class Sequence:
    """One sequence's folds for one complex, truncatable to the first `d` of them."""

    def __init__(self, rows: list[dict], energy_key: str = ENERGY_KEY):
        self.energy_key = energy_key
        # Keyed by seed RANK, which is the row's position in the cohort's seed list.
        # Truncating by rank is what makes "the same run at fewer folds" well defined;
        # truncating by seed value would depend on dict ordering.
        #
        # The LAST row for a rank wins, not the first. The journal is append-only and
        # `Journal.has` keys on (sequence, seed, stage, arm) WITHOUT the scoring version
        # (`cpseq_search.py`), so re-scoring a fold after the gate is recalibrated appends
        # a second row rather than replacing the first. Keeping the first would mean
        # reading a retired gate's verdict and discarding the re-score that was paid for.
        #
        # Measured on `es_pilot_lnr`: the reference's four folds exist twice, 1/4 valid
        # under `v4-postrelax-validity` and 4/4 under `v5-junction-restrained-relax`,
        # while that run's LLM arm is entirely v5. First-wins read the reference at 1/4
        # and the LLM arm at 4/4 -- an arm comparison whose baseline was judged by a
        # harsher, retired gate than the arm under test.
        self.by_rank: dict[int, dict] = {}
        self.scoring_versions: set[str] = set()
        for r in rows:
            rank = r.get("seed_rank")
            if rank is None:
                continue
            self.by_rank[int(rank)] = r
            if r.get("scoring_version"):
                self.scoring_versions.add(str(r["scoring_version"]))
        self.draw_index = next(
            (r.get("draw_index") for r in rows if r.get("draw_index") is not None), None
        )

    def folds(self, depth: int) -> list[dict]:
        return [self.by_rank[k] for k in sorted(self.by_rank) if k < depth]

    def n_folds(self, depth: int) -> int:
        return len(self.folds(depth))

    def n_pass(self, depth: int) -> int:
        return sum(1 for r in self.folds(depth) if r.get("valid"))

    def pass_rate(self, depth: int) -> float | None:
        n = self.n_folds(depth)
        return self.n_pass(depth) / n if n else None

    def folds_in(self, lo: int, hi: int) -> list[dict]:
        return [self.by_rank[k] for k in sorted(self.by_rank) if lo <= k < hi]

    def n_pass_in(self, lo: int, hi: int) -> int:
        return sum(1 for r in self.folds_in(lo, hi) if r.get("valid"))

    def median_energy_in(self, lo: int, hi: int) -> float | None:
        """Median energy over the passing folds whose rank is in [lo, hi).

        Exists so that one half of a sequence's folds can CHOOSE it and the other half
        can MEASURE it. Choosing and measuring on the same folds is what makes a
        best-of-B number biased: the candidate that wins is partly the one whose noise
        happened to be favourable, and that same noise is then reported as its energy.
        """
        e = [r[self.energy_key] for r in self.folds_in(lo, hi)
             if r.get("valid") and r.get(self.energy_key) is not None]
        return st.median(e) if e else None

    def median_energy(self, depth: int) -> float | None:
        """Median energy over the folds that PASSED. None if none did.

        Over passing folds only, because the energy of a structure that failed the
        geometry checks describes a pose that was rejected. The count is reported
        alongside everywhere this is used, so a median over two folds is never read as
        one over ten.
        """
        e = [r[self.energy_key] for r in self.folds(depth)
             if r.get("valid") and r.get(self.energy_key) is not None]
        return st.median(e) if e else None


def version_census(per_parent: dict[str, list[dict]]) -> dict[str, int]:
    """How many rows each `scoring_version` wrote, across the whole run directory.

    Printed by every report, because a run holding more than one version is one whose
    folds were judged by more than one gate, and no selection rule can repair that. The
    gate was recalibrated on 2026-10-22 (`tolerances.py`: trans-only 8.44 deg replaced by
    cis-or-trans 30 deg after the old window was measured admitting 27 of 150 native
    CPSea macrocycles), which moved pass rates from ~0.37 to ~1.00 -- so a mixed
    directory can show an arm beating a baseline that was simply scored more harshly.
    """
    census: dict[str, int] = {}
    for rows in per_parent.values():
        for r in rows:
            v = str(r.get("scoring_version"))
            census[v] = census.get(v, 0) + 1
    return dict(sorted(census.items(), key=lambda kv: -kv[1]))


def warn_if_mixed(per_parent: dict[str, list[dict]], selected: str | None = None) -> bool:
    """Print the version census and say plainly when it is not a single gate.

    Returns True when the rows span more than one version AND none was selected, which is
    the case a caller must not report as a result.
    """
    census = version_census(per_parent)
    if selected:
        print(f"scoring_version: filtered to {selected!r} "
              f"({census.get(selected, 0)} of {sum(census.values())} rows)")
        return False
    if len(census) <= 1:
        print(f"scoring_version: {', '.join(f'{k} ({v})' for k, v in census.items())}")
        return False
    print("scoring_version: MIXED -- "
          + ", ".join(f"{k} ({v})" for k, v in census.items()))
    print("  These folds were judged by more than one gate. Pass --scoring-version to "
          "pick one, or re-score the run, before reading any comparison off it.")
    return True


def build(
    per_parent: dict[str, list[dict]],
    draw_arm: str = DRAW_ARM,
    energy_key: str = ENERGY_KEY,
    scoring_version: str | None = None,
) -> dict[str, dict]:
    """Per complex: the unchanged peptide, and its draws keyed by stream position."""
    out: dict[str, dict] = {}
    for pid, all_rows in per_parent.items():
        # A sequence's stream position and arm are resolved from ALL of its rows, before
        # the version filter, and only its FOLDS are filtered. `cpseq_rescore_folds.py`
        # builds its rows with `proposal=None`, so a re-scored row carries no
        # `draw_index` and no `round_index` -- those live in the `extra` the ladder passes
        # (`cpseq_search.py:1653`). Filtering first therefore discards the only rows that
        # know where the draw sits in the stream, the draw drops out of `draws`, and a
        # prefix ladder silently reports zero complete draws for the whole arm. Measured
        # on `es_pilot_lnr`: after re-scoring its control draw 0 to v5, every complex
        # still read `random_control: 0` until this was resolved per sequence.
        by_seq_all: dict[str, list[dict]] = collections.defaultdict(list)
        for r in all_rows:
            by_seq_all[r["sequence"]].append(r)
        draw_index_of: dict[str, int] = {}
        arms_of: dict[str, set] = {}
        for seq, rs in by_seq_all.items():
            di = next(
                (r.get("draw_index") for r in rs if r.get("draw_index") is not None), None
            )
            if di is not None:
                draw_index_of[seq] = int(di)
            arms_of[seq] = {r.get("arm") for r in rs}
        ref_seq = next(
            (r["sequence"] for r in all_rows if r.get("arm") == REFERENCE_ARM), None
        )

        rows = [
            r
            for r in all_rows
            if scoring_version is None or r.get("scoring_version") == scoring_version
        ]
        if not rows:
            continue
        by_seq: dict[str, list[dict]] = collections.defaultdict(list)
        for r in rows:
            by_seq[r["sequence"]].append(r)
        draws: dict[int, Sequence] = {}
        for seq, rs in by_seq.items():
            if seq == ref_seq:
                continue
            if draw_arm not in arms_of.get(seq, set()):
                continue
            s = Sequence(rs, energy_key)
            di = s.draw_index if s.draw_index is not None else draw_index_of.get(seq)
            if di is not None:
                draws[int(di)] = s
        out[pid] = {
            "reference": Sequence(by_seq[ref_seq], energy_key) if ref_seq else None,
            "draws": draws,
            "source": rows[0].get("split"),
            "corpus": "LNR" if pid.startswith("lnr_") else "PepBench",
        }
    return out


def complete_prefix(draws: dict[int, Sequence], depth: int) -> int:
    """How many draws from the start of the stream are folded at full `depth`.

    A rung is a PREFIX, so a gap at draw 3 caps the ladder at B=2 however many later
    draws happen to be finished. Reporting the count of finished draws instead would
    quietly build a rung out of a non-prefix.
    """
    n = 0
    while draws.get(n) is not None and draws[n].n_folds(depth) >= depth:
        n += 1
    return n


def ladder_rows(state: dict[str, dict], rungs: list[int], depth: int,
                reliability: float) -> list[dict]:
    """One row per rung: coverage, matched pass rates, and the energy comparison."""
    rows = []
    for b in rungs:
        # Only complexes whose first b draws AND unchanged peptide are fully folded.
        # A complex caught mid-rung has a truncated draw set and would drag coverage
        # down for a reason that is not about the method.
        usable = [
            (pid, s) for pid, s in state.items()
            if s["reference"] is not None
            and s["reference"].n_folds(depth) >= depth
            and complete_prefix(s["draws"], depth) >= b
        ]
        if not usable:
            continue
        k_needed = math.ceil(reliability * depth)

        cov_reliable = cov_everpass = 0
        design_pass = design_folds = ref_pass = ref_folds = 0
        diffs: list[float] = []
        wins = losses = 0
        # Per-complex paired pass rates. The complex is the unit of analysis, not the
        # fold: ten folds of one sequence are repeated measures of that sequence, so a
        # test pooled over folds counts them as independent complexes and returns a
        # p-value far too small. This project has already made that error once -- the
        # pilot's pooled 2.6e-6 became p = 0.031 once the target was the unit -- so the
        # pooled number is kept in the JSON and labelled, and this is the headline.
        pass_wins = pass_losses = 0
        pass_diffs: list[float] = []
        # Cross-fitted energy: the first half of the folds picks the candidate, the
        # second half measures it, and the unchanged peptide is measured on that same
        # second half. Selection and measurement therefore share no fold, so this
        # number cannot be produced by favourable noise -- which is the one thing the
        # plain best-of-B column cannot rule out, however many folds each sequence got.
        half = depth // 2
        ho_diffs: list[float] = []
        ho_wins = ho_losses = ho_unmeasurable = 0
        for pid, s in usable:
            prefix = [s["draws"][i] for i in range(b)]
            if any(d.n_pass(depth) >= k_needed for d in prefix):
                cov_reliable += 1
            if any(d.n_pass(depth) >= 1 for d in prefix):
                cov_everpass += 1
            d_pass = sum(d.n_pass(depth) for d in prefix)
            d_folds = sum(d.n_folds(depth) for d in prefix)
            design_pass += d_pass
            design_folds += d_folds
            r_pass = s["reference"].n_pass(depth)
            r_folds = s["reference"].n_folds(depth)
            ref_pass += r_pass
            ref_folds += r_folds
            if d_folds and r_folds:
                dd = d_pass / d_folds - r_pass / r_folds
                pass_diffs.append(dd)
                if dd > 0:
                    pass_wins += 1
                elif dd < 0:
                    pass_losses += 1

            # Best draw by MEDIAN energy among those clearing the reliability bar,
            # against the unchanged peptide's median over the same number of folds.
            ok = [d for d in prefix
                  if d.n_pass(depth) >= k_needed and d.median_energy(depth) is not None]
            ref_e = s["reference"].median_energy(depth)
            if ok and ref_e is not None:
                best = min(d.median_energy(depth) for d in ok)
                diffs.append(best - ref_e)
                if best < ref_e:
                    wins += 1
                elif best > ref_e:
                    losses += 1

            # The cross-fitted version of the same comparison.
            if half >= 1:
                pick = [(d.median_energy_in(0, half), d) for d in prefix
                        if d.median_energy_in(0, half) is not None]
                ref_ho = s["reference"].median_energy_in(half, depth)
                if pick and ref_ho is not None:
                    chosen = min(pick, key=lambda t: t[0])[1]
                    got = chosen.median_energy_in(half, depth)
                    if got is None:
                        # Chosen on the first half, then passed no fold in the second.
                        # Not an energy win or loss -- it is a candidate the selection
                        # could not stand behind, and counting it either way would
                        # flatter or punish the method for a missing measurement.
                        ho_unmeasurable += 1
                    else:
                        ho_diffs.append(got - ref_ho)
                        if got < ref_ho:
                            ho_wins += 1
                        elif got > ref_ho:
                            ho_losses += 1

        rows.append({
            "B": b,
            "n_complexes": len(usable),
            "fold_depth": depth,
            "reliability_k_of_n": f"{k_needed}/{depth}",
            "coverage_reliable": cov_reliable / len(usable),
            "coverage_any_single_fold": cov_everpass / len(usable),
            "design_pass_rate": design_pass / design_folds if design_folds else None,
            "design_folds": design_folds,
            "reference_pass_rate": ref_pass / ref_folds if ref_folds else None,
            "reference_folds": ref_folds,
            # Per complex, paired: this is the headline test for the pass rate.
            "pass_rate_median_delta": st.median(pass_diffs) if pass_diffs else None,
            "pass_rate_wins": pass_wins,
            "pass_rate_losses": pass_losses,
            "pass_rate_sign_p": (
                sign_test(pass_wins, pass_losses) if (pass_wins + pass_losses) else None
            ),
            # Pooled over folds. PSEUDO-REPLICATED -- ten folds of one sequence are
            # repeated measures, so this treats one complex as ten. Kept because it is
            # what a fold-level comparison gives, never as the headline.
            "pass_rate_fisher_p_pooled_pseudoreplicated": fisher_two_sided(
                design_pass, design_folds - design_pass,
                ref_pass, ref_folds - ref_pass,
            ) if design_folds and ref_folds else None,
            "n_energy_pairs": len(diffs),
            "median_energy_delta": st.median(diffs) if diffs else None,
            "energy_wins": wins,
            "energy_losses": losses,
            "energy_sign_p": sign_test(wins, losses) if (wins + losses) else None,
            # Cross-fitted: candidate chosen on folds 0..half, measured on half..depth,
            # peptide measured on half..depth. Free of selection bias by construction.
            "heldout_n_pairs": len(ho_diffs),
            "heldout_median_energy_delta": st.median(ho_diffs) if ho_diffs else None,
            "heldout_wins": ho_wins,
            "heldout_losses": ho_losses,
            "heldout_sign_p": (
                sign_test(ho_wins, ho_losses) if (ho_wins + ho_losses) else None
            ),
            "heldout_unmeasurable": ho_unmeasurable,
            "heldout_select_folds": [0, half],
            "heldout_measure_folds": [half, depth],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--depth", type=int, default=0,
                    help="folds per sequence to read (default: the cohort's full depth)")
    ap.add_argument("--reliability", type=float, default=0.5,
                    help="a draw 'passes' when it clears the structure checks in at "
                         "least this fraction of its folds (default 0.5). 1/depth "
                         "reproduces the old one-fold definition.")
    ap.add_argument("--by-corpus", action="store_true",
                    help="also print the ladder separately for LNR and PepBench")
    ap.add_argument("--common-complexes", action="store_true",
                    help="restrict EVERY rung to the complexes that are complete at the "
                         "deepest rung, so B is the only thing that changes down the "
                         "table. Without this, a mid-flight run gives each rung a "
                         "different complex set -- the ones that finish first are the "
                         "small, fast ones -- and a trend down the column can be that "
                         "changing set rather than the budget.")
    ap.add_argument("--arm", default=DRAW_ARM,
                    help=f"which proposal arm's draws to report (default {DRAW_ARM}). "
                         f"A run directory may hold several arms; each is reported "
                         f"separately against the SAME shared reference folds, which is "
                         f"what makes the arms paired rather than independently "
                         f"baselined. Use llm_search for the LLM-proposed arm.")
    ap.add_argument("--json", default="",
                    help="written under --out-dir so the figures are redrawable without "
                         "recomputing anything. Defaults to ladder_report.json for the "
                         "random arm and ladder_report_<arm>.json for any other, so "
                         "reporting a second arm cannot overwrite the first's file.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cohort = json.loads((out_dir / "cohort.json").read_text())
    depth = args.depth or len(cohort["seeds"])
    rungs = [b for b in cohort["budget_rungs"]]
    json_name = args.json or (
        "ladder_report.json" if args.arm == DRAW_ARM
        else f"ladder_report_{args.arm}.json"
    )

    state = build(load(out_dir), draw_arm=args.arm)
    if not state:
        print(f"[ladder-report] no {STAGE} rows under {out_dir}/state")
        return 1

    # Progress first: a ladder read off a run that has not reached a rung is silent
    # about it, so say what is finished before showing any number.
    print(f"=== draw ladder, {out_dir} ===")
    print(f"draw arm: {args.arm}   (reference folds are shared across arms)")
    print(f"complexes with folds: {len(state)} of {cohort['n_parents']}  "
          f"fold depth read: {depth} of {len(cohort['seeds'])}")
    done = {pid: complete_prefix(s["draws"], depth) for pid, s in state.items()}
    ref_done = sum(1 for s in state.values()
                   if s["reference"] is not None and s["reference"].n_folds(depth) >= depth)
    print(f"unchanged peptide complete at depth {depth}: {ref_done}/{len(state)}")
    print(f"draws complete from the start of the stream: "
          f"min {min(done.values())}, median {int(st.median(list(done.values())))}, "
          f"max {max(done.values())} (of {cohort['n_draws']})")

    if args.common_complexes:
        deepest = max(rungs)
        keep = {p for p, s in state.items()
                if complete_prefix(s["draws"], depth) >= deepest
                and s["reference"] is not None
                and s["reference"].n_folds(depth) >= depth}
        if not keep:
            # Falling back silently would print a table that answers a different
            # question from the one the flag asked for.
            print(f"[ladder-report] --common-complexes: no complex is complete at "
                  f"B={deepest} yet, so there is no set to hold fixed. Drop the flag to "
                  f"see the per-rung tables, remembering that their rungs rest on "
                  f"different complexes.")
            return 1
        print(f"\n--common-complexes: every rung below is the SAME {len(keep)} "
              f"complex(es), those complete at B={deepest}. B is the only thing that "
              f"changes down the table.")
        state = {p: s for p, s in state.items() if p in keep}

    groups = [("all", state)]
    if args.by_corpus:
        for label in ("LNR", "PepBench"):
            groups.append(
                (label, {k: v for k, v in state.items() if v["corpus"] == label})
            )

    # The arm is recorded on the report itself. Two arms of one run directory produce two
    # structurally identical JSON files, and a file that did not name its arm could be
    # read as the other one's.
    report = {"out_dir": str(out_dir), "draw_arm": args.arm, "fold_depth": depth,
              "reliability": args.reliability, "groups": {}}
    for label, grp in groups:
        if not grp:
            continue
        rows = ladder_rows(grp, rungs, depth, args.reliability)
        report["groups"][label] = rows
        if not rows:
            print(f"\n--- {label}: no rung is complete yet ---")
            continue
        print(f"\n--- {label} ({len(grp)} complexes) ---")
        print("          found  found  design peptide | pass rate per complex | "
              "energy, best of B   | energy, cross-fitted")
        print("  B  cplx  often   once    pass    pass |  d(rate)  w/l   sign p | "
              " dE REU  w/l   sign p |  dE REU  w/l   sign p")
        for r in rows:
            de = "-" if r["median_energy_delta"] is None else f"{r['median_energy_delta']:+7.2f}"
            dr = ("-" if r["pass_rate_median_delta"] is None
                  else f"{r['pass_rate_median_delta']:+8.3f}")
            pp = "-" if r["pass_rate_sign_p"] is None else f"{r['pass_rate_sign_p']:.3f}"
            sp = "-" if r["energy_sign_p"] is None else f"{r['energy_sign_p']:.3f}"
            hd = ("-" if r["heldout_median_energy_delta"] is None
                  else f"{r['heldout_median_energy_delta']:+7.2f}")
            hp = "-" if r["heldout_sign_p"] is None else f"{r['heldout_sign_p']:.3f}"
            print(f"{r['B']:>3} {r['n_complexes']:>5} "
                  f"{r['coverage_reliable']:>6.3f} {r['coverage_any_single_fold']:>6.3f} "
                  f"{r['design_pass_rate']:>7.3f} {r['reference_pass_rate']:>7.3f} | "
                  f"{dr:>8} {r['pass_rate_wins']:>2}/{r['pass_rate_losses']:<2} {pp:>7} | "
                  f"{de:>7} {r['energy_wins']:>2}/{r['energy_losses']:<2} {sp:>7} | "
                  f"{hd:>7} {r['heldout_wins']:>2}/{r['heldout_losses']:<2} {hp:>7}")

    path = out_dir / json_name
    path.write_text(json.dumps(report, indent=2))
    print(f"\n[ladder-report] -> {path}")
    k = math.ceil(args.reliability * depth)
    print("\nWhat the columns mean, in order:")
    print(f"  B           candidates tried per complex: the first B of the {cohort['n_draws']} "
          f"drawn")
    print("  cplx        complexes whose first B candidates AND unchanged peptide are "
          f"all folded {depth} times")
    print(f"  found often fraction of complexes with a candidate that passed the "
          f"structure checks in >= {k} of its {depth} folds")
    print("  found once  the same fraction under the old rule -- passed in at least 1 "
          "fold. The gap between these two columns is what one fold per candidate was "
          "buying.")
    print(f"  design/peptide pass   per-fold pass rate, candidates vs the unchanged "
          f"peptide. Both sides are single folds, so the counts are comparable.")
    print("  pass rate per complex  the same comparison done WITHIN each complex and "
          "then counted across complexes, which is the headline: ten folds of one "
          "sequence are repeated measures, so a test pooled over folds counts one "
          "complex as ten and returns a p-value far too small. The pooled Fisher p is "
          "in the JSON, labelled.")
    print("  median dE   per complex, best candidate's MEDIAN energy over its passing "
          "folds minus the peptide's median over its own. Negative = candidate better. "
          "Medians, not minima: a minimum over more folds is lower whatever the "
          "sequences are.")
    print("  design w/l  complexes where that difference favours the candidate / the "
          "peptide, with an exact sign test.")
    print(f"  cross-fitted  the same energy comparison with the candidate CHOSEN on "
          f"folds 1-{depth // 2} and MEASURED on folds {depth // 2 + 1}-{depth}, "
          f"against the peptide measured on folds {depth // 2 + 1}-{depth}. Selection "
          f"and measurement share no fold, so this column cannot be produced by "
          f"favourable noise -- it is the one the best-of-B column has to agree with "
          f"before a search advantage can be claimed.")
    print("Energies are Rosetta ref2015 REU, not measured binding free energies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
