#!/usr/bin/env python3
"""Would naming an elite's interface defects have told the proposer where to edit?

The proposal stage already folds each elite against the target and renders a per-position
interface table into that elite's own prompt (`cpseq_llm_propose.run_parent`). What it
never does is *name the defect*: the model is handed `burial 0.00  dmin 6.1  ctc 0` and
has to conclude by itself that the position is wasted. Before building a block that says
so in words, the premise has to be checked:

**an edit on a position the structure calls defective must beat an edit elsewhere.**

If it does not, a defect block is a longer prompt and nothing else. This is decidable on
folds already on disk, at zero GPU cost, which is the order §12 item 4 and §13 used.

Two tests, because they answer different halves of the question:

1. **Elite mode** (`llm_search`). Each LLM candidate carries the elite it was asked to
   edit in `ancestor_sequence`, so the positions it changed *relative to that elite* are
   known. Within each elite, offspring that touched a flagged position are compared
   against offspring that did not. This is the real question and it is weakly powered:
   the ES pilots hold 38 elites over 6 complexes.

2. **Control mode** (`random_control`). Random draws have no ancestry, but they are draws
   from the ORIGINAL, and the reference's own folds are on disk -- so the same flags can
   be computed on the reference and scored against hundreds of draws that touched them by
   chance. This cannot say whether an LLM would *act* on a defect block, only whether the
   positions it would point at are the better ones to spend an edit on. It carries most
   of the statistical power here, so it is on by default.

Three honesty constraints, each of which changes the answer:

- **Defects come from the SELECTION half of the folds, outcomes from the MEASUREMENT
  half.** Flags read off the same folds the outcome is scored on would confirm
  themselves. This is the `--feedback-folds select` guard of the live arm, applied to the
  analysis.
- **Defects are read from the RAW structure.** Relax repairs the overlaps a proposer most
  needs to see (`interactions.summarize_complex` measured 0.79 A -> 2.7 A on one
  position), so the refined copy would report a clean interface for the defect in
  question.
- **A flag is reported at three recurrence levels.** `best_row_for` hands the live prompt
  the single best-energy fold, and against a 4.3 REU median seed spread a defect seen in
  one fold of a handful may not be a property of the sequence at all. `single` is what a
  defect block built today would carry; `any` and `all` say how much of it survives
  asking the other folds. A large gap between them is itself the finding.

What is NOT claimed: nothing here measures whether an 8B model can use a defect block.
It measures whether there is anything in one worth using.

Usage:
    .venv/bin/python script_utils/cpseq_defect_probe.py \
        --run-dir /data/user_data/yixiz/cpseq/es_pilot_lnr \
        --run-dir /data/user_data/yixiz/cpseq/es_pilot_pep \
        --jobs 2 \
        --out /data/user_data/yixiz/cpseq/defect_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.interactions import summarize_complex  # noqa: E402

# The gate recalibration of 2026-10-22 means pass rates are not comparable across scoring
# protocols (§14: v3 ~0.37-0.81, v4 ~0.06-0.07, v5 ~0.98-1.00 on the same folds), and a
# re-score APPENDS a row rather than replacing one. So this reader filters to one protocol
# and keeps the NEWEST row per fold, which is the fix §14.3 applied to the other readers.
DEFAULT_SCORING_VERSION = "v5-junction-restrained-relax"

ARM_LLM = "llm_search"
ARM_CONTROL = "random_control"
ARM_REFERENCE = "reference"

# --------------------------------------------------------------------------------------
# What counts as a defect
# --------------------------------------------------------------------------------------
#
# Each class is a defect a SUBSTITUTION could plausibly repair, and each one corresponds
# to a move the system prompt already suggests ("filling a pocket that a small residue
# leaves empty", "removing a side chain that overlaps the receptor", ...). Classes a
# proposal cannot act on -- a buried backbone amide, a receptor-side atom -- are
# deliberately absent: `cpseq_unsat_probe.py` exists to measure how much of the unsat
# count is of that kind, and flagging them here would dilute every comparison below.

POLAR = set("STNQHYCW")
CHARGED = set("DEKR")
# A position with no contact at all. All three conditions, not any: a residue can read
# 0.00 burial while still sitting 3.5 A off the receptor with contacts, and that is a
# shallow contact, not an empty position.
EMPTY_BURIAL_MAX = 0.05
EMPTY_DMIN_MIN_A = 5.0
# Buried enough to pay a desolvation cost. Deliberately permissive, which makes the
# unsat/charged flags conservative rather than generous.
UNSAT_BURIAL_MIN = 0.50
CHARGED_BURIAL_MIN = 0.70
SATISFYING = {"hbond", "saltbridge", "waterbridge", "pistacking", "pication"}

DEFECT_CLASSES = ("clash", "empty", "unsat_polar", "buried_charged")


def classify_position(p: dict) -> set[str]:
    """The defect labels one position of one fold carries. Pure, so it is testable."""
    labels: set[str] = set()
    aa = str(p.get("residue") or "")
    burial = p.get("buried_fraction")
    dmin = p.get("min_dist_A")
    contacts = p.get("n_contacts") or 0
    inter = p.get("interactions") or []
    kinds = {str(i.get("type") or "").lower() for i in inter}

    if (p.get("n_clashes") or 0) >= 1:
        labels.add("clash")
    if (
        burial is not None
        and dmin is not None
        and burial <= EMPTY_BURIAL_MAX
        and dmin >= EMPTY_DMIN_MIN_A
        and contacts == 0
    ):
        labels.add("empty")
    if (
        burial is not None
        and burial >= UNSAT_BURIAL_MIN
        and aa in (POLAR | CHARGED)
        and not (kinds & SATISFYING)
    ):
        labels.add("unsat_polar")
    if (
        burial is not None
        and burial >= CHARGED_BURIAL_MIN
        and aa in CHARGED
        and "saltbridge" not in kinds
    ):
        labels.add("buried_charged")
    return labels


def flags_from_summaries(
    summaries: list[dict], *, representative: dict | None
) -> dict[str, dict[int, list[str]]]:
    """Flagged positions at three recurrence levels, as {level: {pos0: [labels]}}.

    `single` uses only `representative` -- the one fold `best_row_for` would have handed
    the prompt -- so it measures a defect block built from today's code. `any` flags a
    defect seen in at least one selection fold, `all` only one seen in every one of them.
    Positions are 0-based here; the summaries are 1-based.
    """
    counts: dict[tuple[int, str], int] = defaultdict(int)
    n = 0
    for s in summaries:
        if not s or not s.get("ok"):
            continue
        n += 1
        for p in s.get("positions") or []:
            pos0 = int(p["pos"]) - 1
            for label in classify_position(p):
                counts[(pos0, label)] += 1

    out: dict[str, dict[int, list[str]]] = {}
    for level, need in (("any", 1), ("all", n)):
        d: dict[int, list[str]] = defaultdict(list)
        if n:
            for (pos0, label), c in counts.items():
                if c >= need:
                    d[pos0].append(label)
        out[level] = {k: sorted(v) for k, v in d.items()}

    # How much of a flag is a property of the SEQUENCE rather than of one fold. Per
    # defect class: the (position, class) pairs seen in at least one selection fold, and
    # how many of those held in every one of them. A class whose flags mostly fail to
    # recur cannot be stated to a proposer as a fact about its candidate -- it is a fact
    # about one prediction.
    recur: dict[str, list[int]] = {}
    for label in DEFECT_CLASSES:
        seen = [c for (_p, lab), c in counts.items() if lab == label]
        recur[label] = [len(seen), sum(1 for c in seen if n and c >= n)]
    out["_recurrence"] = recur  # type: ignore[assignment]

    d1: dict[int, list[str]] = defaultdict(list)
    if representative and representative.get("ok"):
        for p in representative.get("positions") or []:
            labels = classify_position(p)
            if labels:
                d1[int(p["pos"]) - 1] = sorted(labels)
    out["single"] = dict(d1)
    out["_n_selection_folds"] = n  # type: ignore[assignment]
    return out


# --------------------------------------------------------------------------------------
# Statistics, in pure python so the report path has no scipy dependency
# --------------------------------------------------------------------------------------


def _sign_test(deltas: list[float]) -> dict:
    """Two-sided exact sign test. Zeros are dropped, which is the conservative choice."""
    neg = sum(1 for d in deltas if d < 0)
    pos = sum(1 for d in deltas if d > 0)
    n = neg + pos
    if n == 0:
        return {"n": 0, "n_better": 0, "n_worse": 0, "p": None}
    k = min(neg, pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {"n": n, "n_better": neg, "n_worse": pos, "p": round(min(1.0, 2 * tail), 4)}


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


def _mannwhitney(a: list[float], b: list[float]) -> dict:
    """Normal approximation with tie correction. Approximate at these n, and labelled so
    in the report rather than quoted as exact."""
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return {"n_a": n1, "n_b": n2, "u": None, "p": None}
    r = _ranks(list(a) + list(b))
    r1 = sum(r[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    tie = defaultdict(int)
    for v in list(a) + list(b):
        tie[v] += 1
    n = n1 + n2
    corr = sum(t**3 - t for t in tie.values())
    var = n1 * n2 / 12 * ((n + 1) - corr / (n * (n - 1)))
    if var <= 0:
        return {"n_a": n1, "n_b": n2, "u": u1, "p": None}
    z = (u1 - mu) / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"n_a": n1, "n_b": n2, "u": u1, "z": round(z, 3), "p": round(min(1.0, p), 4)}


def _mean(xs: list[float]) -> float | None:
    return round(st.fmean(xs), 3) if xs else None


# --------------------------------------------------------------------------------------
# Reading the journal
# --------------------------------------------------------------------------------------


def load_rows(run_dir: Path, scoring_version: str | None) -> tuple[dict, dict]:
    """Newest journal row per (sequence, rank, arm, stage), by parent, plus a census."""
    newest: dict[str, dict[tuple, dict]] = defaultdict(dict)
    census: dict[str, int] = defaultdict(int)
    for path in sorted((run_dir / "state").glob("*/outcomes.jsonl")):
        parent = path.parent.name
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            census[str(r.get("scoring_version"))] += 1
            if r.get("technical_failure"):
                continue  # a crashed fold is not evidence about a sequence
            sv = r.get("scoring_version")
            if scoring_version is not None and sv is not None and sv != scoring_version:
                continue
            key = (r.get("sequence"), r.get("seed_rank"), r.get("arm"), r.get("stage"))
            newest[parent][key] = r  # later line wins: §14.3
    return (
        {p: list(d.values()) for p, d in newest.items()},
        dict(census),
    )


def split_halves(rows: list[dict]) -> tuple[int, int]:
    """Fold depth and the first measurement rank, from the ranks actually present."""
    ranks = [r.get("seed_rank") for r in rows if r.get("seed_rank") is not None]
    depth = (max(ranks) + 1) if ranks else 0
    return depth, depth // 2


def outcome(rows: list[dict], *, min_rank: int, max_rank: int | None) -> dict:
    """One sequence's standing over a half of its folds.

    `dG` is the mean over VALID folds, which is the objective the arm optimises and the
    confirmation stage reports; `dG_norm` is §13's area-normalised version, carried
    alongside because selecting on raw dG was measured trading polar satisfaction for
    buried area. `pass_frac` is reported separately rather than folded in, because §12
    item 4 leaves the pass-count-first rule open.
    """
    sel = [
        r
        for r in rows
        if r.get("seed_rank") is not None
        and r["seed_rank"] >= min_rank
        and (max_rank is None or r["seed_rank"] < max_rank)
    ]
    dg = [r["best_valid_dG_separated"] for r in sel
          if r.get("valid") and r.get("best_valid_dG_separated") is not None]
    dgn = [r["refined_dG_separated_per_1000A2"] for r in sel
           if r.get("valid") and r.get("refined_dG_separated_per_1000A2") is not None]
    n_valid = sum(1 for r in sel if r.get("valid"))
    return {
        "n_folds": len(sel),
        "n_valid": n_valid,
        "pass_frac": (n_valid / len(sel)) if sel else None,
        "dG": st.fmean(dg) if dg else None,
        "dG_norm": st.fmean(dgn) if dgn else None,
    }


def representative_row(rows: list[dict]) -> dict | None:
    """The fold `cpseq_llm_propose.best_row_for` would have summarised for the prompt:
    the valid fold with the best energy, else the smallest violation."""
    mine = [r for r in rows if not r.get("technical_failure") and r.get("raw_pdb_path")]
    if not mine:
        return None
    valid = [r for r in mine if r.get("valid")]
    if valid:
        with_e = [r for r in valid if r.get("best_valid_dG_separated") is not None]
        if with_e:
            return min(with_e, key=lambda r: r["best_valid_dG_separated"])
        return valid[0]
    with_v = [r for r in mine if r.get("min_violation") is not None]
    return min(with_v, key=lambda r: r["min_violation"]) if with_v else mine[0]


# --------------------------------------------------------------------------------------
# Interface summaries, cached
# --------------------------------------------------------------------------------------


def _summary_task(task: tuple) -> dict:
    """Worker: one structure's interface summary. Top-level, for ProcessPoolExecutor."""
    cache_path, raw_path, sequence = task
    cache = Path(cache_path)
    if cache.exists():
        try:
            return {"cache": cache_path, "summary": json.loads(cache.read_text())}
        except Exception:  # noqa: BLE001
            pass
    try:
        s = summarize_complex(raw_path, peptide_sequence=sequence, structure_stage="raw")
    except Exception as exc:  # noqa: BLE001
        s = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(s))
    return {"cache": cache_path, "summary": s}


def summary_cache_paths(run_dir: Path, out_dir: Path, parent: str, ihash: str) -> tuple:
    """Where to look for an existing summary, and where to write a new one.

    The run's own `interfaces/<parent>/<hash>.json` is read when present: it is the same
    computation on the same structure with the same key, so recomputing it would spend
    1.2-2.6 s to reproduce a file already paid for. New summaries go to the probe's own
    directory, so this never writes into a production artifact.
    """
    existing = run_dir / "interfaces" / parent / f"{ihash}.json"
    mine = out_dir / "summaries" / parent / f"{ihash}.json"
    return existing, mine


# --------------------------------------------------------------------------------------
# The two tests
# --------------------------------------------------------------------------------------


def build_units(
    runs: list[tuple[Path, dict]],
    *,
    include_control: bool,
) -> tuple[list[dict], list[dict]]:
    """Every (parent, reference-sequence, offspring) unit the probe can score.

    Returns (units, needed_summaries). A unit names the sequence whose defects are to be
    flagged ("base"), the candidate that edited it, and the positions it changed relative
    to that base -- never relative to the original, which is what the budget is counted
    against but not what the edit acted on.
    """
    units: list[dict] = []
    needed: dict[tuple, dict] = {}

    for run_dir, by_parent in runs:
        for parent, rows in sorted(by_parent.items()):
            depth, mid = split_halves(rows)
            if depth < 2:
                continue
            by_seq: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_seq[r["sequence"]].append(r)

            # ---- elite mode: ancestry is recorded on the row ----
            pairs: list[tuple[str, str, str]] = []
            for seq, srows in by_seq.items():
                anc = next(
                    (r.get("ancestor_sequence") for r in srows
                     if r.get("arm") == ARM_LLM and r.get("ancestor_sequence")),
                    None,
                )
                if anc and anc in by_seq and anc != seq:
                    pairs.append((anc, seq, "elite"))

            # ---- control mode: the base is the original, scored on the reference arm ----
            if include_control:
                orig = next(
                    (r.get("original_sequence") for r in rows if r.get("original_sequence")),
                    None,
                )
                if orig and orig in by_seq:
                    for seq, srows in by_seq.items():
                        if seq == orig:
                            continue
                        if any(r.get("arm") == ARM_CONTROL for r in srows):
                            pairs.append((orig, seq, "control"))

            for base, cand, mode in pairs:
                brows, crows = by_seq[base], by_seq[cand]
                if len(base) != len(cand):
                    continue  # a length change is not a substitution; nothing to attribute
                edits = [i for i, (a, b) in enumerate(zip(base, cand)) if a != b]
                if not edits:
                    continue
                base_sel = [
                    r for r in brows
                    if r.get("seed_rank") is not None and r["seed_rank"] < mid
                    and r.get("raw_pdb_path")
                ]
                if not base_sel:
                    continue
                rep = representative_row(base_sel)
                for r in base_sel:
                    ih = r.get("input_hash")
                    if ih:
                        needed[(str(run_dir), parent, ih)] = {
                            "run_dir": str(run_dir),
                            "parent": parent,
                            "input_hash": ih,
                            "raw_pdb_path": r["raw_pdb_path"],
                            "sequence": base,
                        }
                units.append(
                    {
                        "run": run_dir.name,
                        "run_dir": str(run_dir),
                        "parent_id": parent,
                        "mode": mode,
                        "base_sequence": base,
                        "candidate_sequence": cand,
                        "edited_vs_base": edits,
                        "n_edits_vs_base": len(edits),
                        "depth": depth,
                        "measurement_min_rank": mid,
                        "selection_hashes": [
                            r.get("input_hash") for r in base_sel if r.get("input_hash")
                        ],
                        "representative_hash": (rep or {}).get("input_hash"),
                        "base_outcome": outcome(brows, min_rank=mid, max_rank=None),
                        "cand_outcome": outcome(crows, min_rank=mid, max_rank=None),
                    }
                )
    return units, list(needed.values())


def score_units(units: list[dict], summaries: dict[tuple, dict]) -> list[dict]:
    """Attach flags and deltas to each unit. One record per candidate, per level."""
    out: list[dict] = []
    flag_cache: dict[tuple, dict] = {}
    for u in units:
        key = (u["run_dir"], u["parent_id"], u["base_sequence"])
        if key not in flag_cache:
            sel = [
                summaries.get((u["run_dir"], u["parent_id"], h))
                for h in u["selection_hashes"]
            ]
            rep = summaries.get(
                (u["run_dir"], u["parent_id"], u.get("representative_hash"))
            )
            flag_cache[key] = flags_from_summaries(
                [s for s in sel if s], representative=rep
            )
        flags = flag_cache[key]
        L = len(u["base_sequence"])
        rec = dict(u)
        bo, co = u["base_outcome"], u["cand_outcome"]
        # Negative = the candidate improved on the sequence it edited, on folds neither
        # the flags nor any selection ever saw.
        rec["d_dG"] = (
            round(co["dG"] - bo["dG"], 3)
            if co["dG"] is not None and bo["dG"] is not None else None
        )
        rec["d_dG_norm"] = (
            round(co["dG_norm"] - bo["dG_norm"], 4)
            if co["dG_norm"] is not None and bo["dG_norm"] is not None else None
        )
        rec["d_pass"] = (
            round(co["pass_frac"] - bo["pass_frac"], 3)
            if co["pass_frac"] is not None and bo["pass_frac"] is not None else None
        )
        rec["n_selection_folds"] = flags.get("_n_selection_folds", 0)
        rec["recurrence"] = flags.get("_recurrence", {})
        rec["levels"] = {}
        for level in ("single", "any", "all"):
            fl = flags.get(level) or {}
            hit_pos = [i for i in u["edited_vs_base"] if i in fl]
            labels = sorted({lab for i in hit_pos for lab in fl[i]})
            rec["levels"][level] = {
                "n_flagged_positions": len(fl),
                "flagged_positions": sorted(fl),
                "flagged_share": round(len(fl) / L, 3) if L else None,
                "n_edits_on_flagged": len(hit_pos),
                "hit": bool(hit_pos),
                "hit_share": round(len(hit_pos) / len(u["edited_vs_base"]), 3),
                "labels_hit": labels,
            }
        out.append(rec)
    return out


def report(records: list[dict], *, metric: str = "d_dG") -> dict:
    """Aggregate. Paired within base sequence for the elite mode; within parent for the
    control, where every draw shares one base and the pairing unit has to be the
    complex."""
    rep: dict = {
        "n_records": len(records),
        "metric": metric,
        "note": (
            "Flags are computed on the SELECTION half of the base sequence's folds and "
            "outcomes on the MEASUREMENT half, so no flag is scored on a fold it was "
            "derived from. Negative deltas mean the edit improved on the sequence it "
            "edited. `single` is the one fold today's prompt would have shown."
        ),
        "modes": {},
    }
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_mode[r["mode"]].append(r)

    for mode, recs in sorted(by_mode.items()):
        usable = [r for r in recs if r.get(metric) is not None]
        m: dict = {
            "n_units": len(recs),
            "n_with_outcome": len(usable),
            "n_parents": len({r["parent_id"] for r in recs}),
            "n_base_sequences": len({(r["parent_id"], r["base_sequence"]) for r in recs}),
            "mean_selection_folds": _mean([r["n_selection_folds"] for r in recs]),
            "mean_edits_vs_base": _mean([float(r["n_edits_vs_base"]) for r in recs]),
            "levels": {},
        }
        group_key = (
            (lambda r: (r["parent_id"], r["base_sequence"]))
            if mode == "elite"
            else (lambda r: r["parent_id"])
        )
        for level in ("single", "any", "all"):
            hits = [r[metric] for r in usable if r["levels"][level]["hit"]]
            miss = [r[metric] for r in usable if not r["levels"][level]["hit"]]
            # Paired: within one base sequence (or one complex), hits against misses.
            paired: list[float] = []
            groups: dict[tuple, dict[bool, list[float]]] = defaultdict(
                lambda: {True: [], False: []}
            )
            for r in usable:
                groups[group_key(r)][r["levels"][level]["hit"]].append(r[metric])
            for g in groups.values():
                if g[True] and g[False]:
                    paired.append(st.fmean(g[True]) - st.fmean(g[False]))
            # Dose-response: does the SHARE of edits landing on flagged positions track
            # the outcome? The categorical test is uninformative when a draw edits half
            # the peptide, which the random control does by construction.
            rho_within: list[float] = []
            for g_id in {group_key(r) for r in usable}:
                sub = [r for r in usable if group_key(r) == g_id]
                if len(sub) >= 5:
                    rho = _spearman(
                        [r["levels"][level]["hit_share"] for r in sub],
                        [r[metric] for r in sub],
                    )
                    if rho is not None:
                        rho_within.append(rho)
            # Per class, against the candidates that touched no flag at all, paired
            # within the group. `unsat_polar` is the one to watch: §10 and §13 measured
            # the objective trading polar satisfaction for buried area, so it is the
            # defect the ranking number actively hides -- and therefore the only one
            # where naming it can add something a better selection rule would not.
            by_label: dict[str, dict] = {}
            for lab in DEFECT_CLASSES:
                h = [r[metric] for r in usable if lab in r["levels"][level]["labels_hit"]]
                if not h:
                    continue
                lab_paired: list[float] = []
                lab_groups: dict[tuple, dict[str, list[float]]] = defaultdict(
                    lambda: {"hit": [], "clean": []}
                )
                for r in usable:
                    lv = r["levels"][level]
                    if lab in lv["labels_hit"]:
                        lab_groups[group_key(r)]["hit"].append(r[metric])
                    elif not lv["hit"]:
                        lab_groups[group_key(r)]["clean"].append(r[metric])
                for g in lab_groups.values():
                    if g["hit"] and g["clean"]:
                        lab_paired.append(st.fmean(g["hit"]) - st.fmean(g["clean"]))
                by_label[lab] = {
                    "n": len(h),
                    "mean": _mean(h),
                    "paired_vs_unflagged": _mean(lab_paired),
                    "sign_test": _sign_test(lab_paired),
                }
            m["levels"][level] = {
                "mean_flagged_share": _mean(
                    [r["levels"][level]["flagged_share"] for r in recs
                     if r["levels"][level]["flagged_share"] is not None]
                ),
                "n_hit": len(hits),
                "n_miss": len(miss),
                "mean_hit": _mean(hits),
                "mean_miss": _mean(miss),
                "pooled_delta": (
                    round(st.fmean(hits) - st.fmean(miss), 3) if hits and miss else None
                ),
                "paired": {
                    "n_groups": len(paired),
                    "mean_delta": _mean(paired),
                    "sign_test": _sign_test(paired),
                },
                "pooled_test_approx": _mannwhitney(hits, miss),
                "spearman_hit_share_vs_metric": {
                    "n_groups": len(rho_within),
                    "mean_rho": _mean(rho_within),
                    "n_negative": sum(1 for r in rho_within if r < 0),
                },
                "by_defect_class": by_label,
            }

        # Is a flag a property of the sequence or of one prediction? Measured over
        # distinct base sequences, not over units, so a base with many offspring does not
        # vote many times. `single_reproduced` is the share of what today's prompt would
        # have been shown -- the best-energy fold's defects -- that also held in every
        # selection fold of the same sequence.
        seen_base: set[tuple] = set()
        shares: list[float] = []
        per_class: dict[str, list[int]] = {k: [0, 0] for k in DEFECT_CLASSES}
        for r in recs:
            key = (r["parent_id"], r["base_sequence"], r["run"])
            if key in seen_base:
                continue
            seen_base.add(key)
            single = set(r["levels"]["single"]["flagged_positions"])
            allf = set(r["levels"]["all"]["flagged_positions"])
            if single:
                shares.append(len(single & allf) / len(single))
            for lab, (n_any, n_all) in (r.get("recurrence") or {}).items():
                if lab in per_class:
                    per_class[lab][0] += n_any
                    per_class[lab][1] += n_all
        m["flag_reproducibility"] = {
            "n_base_sequences": len(seen_base),
            "single_reproduced_in_every_selection_fold": _mean(shares),
            "by_class_any_vs_all": {
                k: {
                    "n_flagged_once": v[0],
                    "n_flagged_in_every_fold": v[1],
                    "share": round(v[1] / v[0], 3) if v[0] else None,
                }
                for k, v in per_class.items()
            },
        }
        rep["modes"][mode] = m
    return rep


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, action="append", required=True,
                    help="an ES/ladder run directory; repeat to pool runs")
    ap.add_argument("--out", type=Path, required=True, help="per-unit JSONL")
    ap.add_argument("--scoring-version", default=DEFAULT_SCORING_VERSION,
                    help="one protocol only; 'all' pools them and is almost never right")
    ap.add_argument("--metric", default="d_dG",
                    choices=("d_dG", "d_dG_norm", "d_pass"))
    ap.add_argument("--no-control", action="store_true",
                    help="elite mode only (the control carries most of the power)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit-summaries", type=int, default=0)
    ap.add_argument("--report-only", action="store_true",
                    help="re-aggregate an existing JSONL without touching a structure")
    args = ap.parse_args()

    if args.report_only:
        recs = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
        return _emit(report(recs, metric=args.metric), args.out)

    sv = None if args.scoring_version == "all" else args.scoring_version
    runs: list[tuple[Path, dict]] = []
    for rd in args.run_dir:
        by_parent, census = load_rows(rd, sv)
        if len(census) > 1:
            print(f"NOTE {rd.name} spans scoring protocols: {census}")
            print(f"     filtered to {sv!r}; see §14 on why pooling them is not valid")
        if not by_parent:
            print(f"no usable rows under {rd}/state at {sv!r}", file=sys.stderr)
            continue
        runs.append((rd, by_parent))
    if not runs:
        return 2

    units, needed = build_units(runs, include_control=not args.no_control)
    if not units:
        print("no scorable units: no row carries `ancestor_sequence` and no reference "
              "arm was found. The LLM ladder runs predate elite breeding; the ES pilots "
              "are the runs that recorded ancestry.", file=sys.stderr)
        return 2

    tasks = []
    out_dir = args.out.parent / (args.out.stem + "_cache")
    for n in needed:
        existing, mine = summary_cache_paths(
            Path(n["run_dir"]), out_dir, n["parent"], n["input_hash"]
        )
        cache = existing if existing.exists() else mine
        tasks.append((str(cache), n["raw_pdb_path"], n["sequence"]))
    if args.limit_summaries:
        tasks = tasks[: args.limit_summaries]

    reused = sum(1 for t in tasks if Path(t[0]).exists())
    missing = sum(1 for t in tasks if not os.path.exists(t[1]))
    print(f"{len(units)} unit(s) over {len({(u['run'], u['parent_id']) for u in units})} "
          f"complex/run pair(s); {len(tasks)} base-sequence selection fold(s) to "
          f"summarise ({reused} already cached, {missing} missing a structure)")

    summaries: dict[tuple, dict] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for n, res in zip(needed, ex.map(_summary_task, tasks, chunksize=1)):
            summaries[(n["run_dir"], n["parent"], n["input_hash"])] = res["summary"]
            done += 1
            if done % 25 == 0 or done == len(tasks):
                ok = sum(1 for s in summaries.values() if s.get("ok"))
                print(f"  {done}/{len(tasks)}  ok={ok}", flush=True)

    records = score_units(units, summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return _emit(report(records, metric=args.metric), args.out)


def _emit(rep: dict, out_path: Path) -> int:
    rep_path = out_path.with_suffix(".report.json")
    rep_path.write_text(json.dumps(rep, indent=2))
    print(f"\n{rep['note']}\n")
    for mode, m in rep["modes"].items():
        print(
            f"=== {mode}: {m['n_with_outcome']}/{m['n_units']} unit(s) with an outcome, "
            f"{m['n_base_sequences']} base sequence(s), {m['n_parents']} complex(es), "
            f"{m['mean_selection_folds']} selection fold(s), "
            f"{m['mean_edits_vs_base']} edit(s) per candidate"
        )
        print(f"{'level':>7} {'flagged':>8} {'n_hit':>6} {'n_miss':>7} "
              f"{'mean_hit':>9} {'mean_miss':>10} {'pooled':>8} "
              f"{'paired':>8} {'groups':>7} {'sign_p':>7} {'rho':>7}")
        for level, d in m["levels"].items():
            sgn = d["paired"]["sign_test"]
            print(
                f"{level:>7} {str(d['mean_flagged_share']):>8} {d['n_hit']:>6} "
                f"{d['n_miss']:>7} {str(d['mean_hit']):>9} {str(d['mean_miss']):>10} "
                f"{str(d['pooled_delta']):>8} {str(d['paired']['mean_delta']):>8} "
                f"{sgn['n']:>7} {str(sgn['p']):>7} "
                f"{str(d['spearman_hit_share_vs_metric']['mean_rho']):>7}"
            )
        cls = m["levels"]["any"]["by_defect_class"]
        if cls:
            print("  by defect class, level=any, against candidates that touched no "
                  "flag at all:")
            for k, v in sorted(cls.items()):
                s = v["sign_test"]
                print(f"    {k:>15}  n={v['n']:>3}  mean={str(v['mean']):>7}  "
                      f"paired={str(v['paired_vs_unflagged']):>7}  "
                      f"groups={s['n']:>2}  p={s['p']}")
        fr = m.get("flag_reproducibility") or {}
        if fr:
            print(f"  flag reproducibility over {fr['n_base_sequences']} base "
                  f"sequence(s): {fr['single_reproduced_in_every_selection_fold']} of "
                  f"the best-energy fold's flags held in every selection fold")
            for k, v in sorted(fr["by_class_any_vs_all"].items()):
                if v["n_flagged_once"]:
                    print(f"    {k:>15}  {v['n_flagged_in_every_fold']:>4}"
                          f"/{v['n_flagged_once']:<4} recurred  share={v['share']}")
        print()
    print(f"report: {rep_path}")
    print(f"rows:   {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
