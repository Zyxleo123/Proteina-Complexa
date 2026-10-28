#!/usr/bin/env python3
"""Which blocks of the proposal prompt is the model actually reading?

The prompt is ~4,000 tokens of interface tables, scored history and rules, and no part of
it has ever been shown to change what comes back. This removes one block at a time and
measures the response.

**It does not fold anything, and that is the design, not a shortcut.** §7 could not
separate the LLM proposer from random draws at depth 10 with ~50 GPU-h: the deltas scatter
around -2 REU against a 4.3 REU median seed spread. An ablation scored on interface energy
would therefore need more folds per cell than the whole arm comparison had, times the
number of components. What *is* cheaply measurable is whether a block changes the
proposals at all -- and a block that does not change the proposals cannot be changing the
energies either, so this is the test that comes first.

Four rules, each because the obvious version produces a number that is not a measurement:

1. **The control is the real prompt, off disk.** `propose_round` writes every prompt it
   issued to `<out-dir>/llm_proposals/round<N>/prompts/*.prompt.txt`, so the unablated arm
   is the exact text the run used, not a reconstruction that might differ from it. Each
   variant is then surgery on that text, and every removal asserts it found what it was
   removing -- a prompt whose wording has since changed breaks the harness loudly instead
   of silently ablating nothing.

2. **Everything is paired and sampled the same number of times.** Same prompt, same seed
   list, same count for the control as for every variant. The proposer samples at
   temperature 0.9, so two runs of the *identical* prompt differ; without the control
   resampled at the same depth, that spread would be attributed to whichever component
   was removed.

3. **The noise floor is measured, and effects are reported in its units.** The same
   prompt at different seeds gives a baseline overlap between proposal sets. A variant
   whose overlap with the control is no lower than the control's overlap with itself has
   not been shown to change anything.

4. **Removal is confounded with length; scrambling is not.** Deleting a table also
   shortens the prompt and drops the model's attention budget somewhere else, so
   `scramble_tables` keeps every table at full size with its per-position numbers
   permuted ACROSS positions. Same shape, same residues, same marginal distribution,
   wrong assignment. If scrambled looks like the control, the table is decoration; if it
   looks like removal, the content is being read. This is the single most informative
   variant here and it costs exactly what the removal costs.

What is measured per cell: yield and every rejection reason under the production screen,
how many positions each proposal edits, and WHERE those edits land relative to the real
structure -- on positions that touch nothing, that overlap the receptor, or that are
buried anchors the prompt says to keep. The structural truth is read from the unablated
prompt's own table, so it is identical across variants by construction.

Usage:
    # free: what each block costs, no model
    .venv/bin/python script_utils/cpseq_prompt_ablation.py \
        --prompt-dir /data/user_data/yixiz/cpseq/es_pilot_pep/llm_proposals/round2/prompts \
        --prompt-dir /data/user_data/yixiz/cpseq/es_pilot_lnr/llm_proposals/round2/prompts \
        --dry-run --out /data/user_data/yixiz/cpseq/prompt_ablation.jsonl

    # the measurement (GPU, one model load)
    .venv/bin/python script_utils/cpseq_prompt_ablation.py \
        --prompt-dir ... --seeds 3 \
        --out /data/user_data/yixiz/cpseq/prompt_ablation.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq.llm_propose import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LlamaProposer,
    parse_completion,
    screen_candidates,
)

SYSTEM_MARK = "=== SYSTEM ==="
USER_MARK = "=== USER ==="

# The rendered table's row format, from `interactions.render_complex`:
#   f"  {pos:>3} {residue}  {ss:>2} {burial:6} {dmin:5} {contacts:>4} {clashes:>4}  {detail}"
ROW_RE = re.compile(
    r"^  (?P<pos>[ \d]{3}) (?P<aa>[A-Z])  (?P<ss>.{2}) (?P<burial>.{6}) "
    r"(?P<dmin>.{5}) (?P<ctc>.{4}) (?P<ovl>.{4})  (?P<detail>.*)$"
)
HEADER_LINE = "  pos aa  ss  burial  dmin  ctc  ovl  interactions / nearby receptor"


def stable_key(parts: tuple) -> int:
    """A seed derived from the cell's identity, not from `hash()`.

    Python randomises `hash()` per process unless PYTHONHASHSEED is set, so using it here
    would mean the seeds -- and therefore the completions -- differ between two runs of
    the same command, which is the one thing an ablation cannot afford.
    """
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16)


def _num(s: str) -> float | None:
    s = s.strip()
    if not s or s in {"-", "?", "n/a"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------
# Reading a saved prompt
# --------------------------------------------------------------------------------------


def read_prompt_file(path: Path) -> tuple[str, str]:
    text = path.read_text()
    if SYSTEM_MARK not in text or USER_MARK not in text:
        raise ValueError(f"{path.name}: not a saved prompt file")
    sys_part = text.split(SYSTEM_MARK, 1)[1]
    system, user = sys_part.split(USER_MARK, 1)
    return system.strip("\n"), user.strip("\n")


def blocks_of(user: str) -> list[str]:
    """The prompt is `"\\n\\n".join(parts)` in `build_user_prompt`, so the blocks are
    recoverable exactly."""
    return user.split("\n\n")


def label_of(block: str) -> str:
    """A stable name for one block, used by the variant definitions."""
    first = block.splitlines()[0] if block.splitlines() else ""
    if first.startswith("TARGET:"):
        return "target"
    if first.startswith("HOW CANDIDATES ARE JUDGED"):
        return "judging"
    if first.startswith("INTERFACE OF THE UNMUTATED REFERENCE"):
        return "reference_table"
    if first.startswith("Receptor pocket residues:"):
        return "pocket_residues"
    if first.startswith("The structure predictor was conditioned on this pocket"):
        return "pocket_note"
    if first.startswith("CANDIDATES ALREADY SCORED"):
        return "history_lines"
    if first.startswith("INTERFACE OF SCORED CANDIDATE"):
        return "scored_table"
    if first.startswith("INTERFACE OF A REJECTED CANDIDATE"):
        return "rejected_table"
    if first.startswith("DO NOT PROPOSE"):
        return "do_not_repeat"
    if first.startswith("The energies above are means"):
        return "noise_note"
    if first.startswith("SEQUENCE TO IMPROVE"):
        return "elite_block"
    if first.startswith("INTERFACE OF THE SEQUENCE TO IMPROVE"):
        return "elite_table"
    if first.startswith("TASK"):
        return "task"
    return "other"


def parse_facts(user: str) -> dict:
    """Everything the screen needs, read back out of the prompt text itself.

    Reading it from the prompt rather than from the journal keeps the harness honest
    about one thing in particular: the legality rules are applied against the same
    original, alphabet and do-not-repeat list the model was actually shown.
    """
    facts: dict = {"ok": True, "error": None}
    m = re.search(r"^ORIGINAL LINEAR PEPTIDE: ([A-Z]+)\s+\(length (\d+)\)", user, re.M)
    if not m:
        return {"ok": False, "error": "no ORIGINAL LINEAR PEPTIDE line"}
    facts["original"] = m.group(1)
    facts["length"] = int(m.group(2))
    m = re.search(r"^SUBSTITUTION BUDGET: at most (\d+) positions", user, re.M)
    facts["stated_budget"] = int(m.group(1)) if m else None
    m = re.search(r"^ALLOWED RESIDUES FOR SUBSTITUTIONS: ([A-Z]+)", user, re.M)
    if not m:
        return {"ok": False, "error": "no ALLOWED RESIDUES line"}
    facts["alphabet"] = tuple(m.group(1))
    m = re.search(r"^Propose exactly (\d+) new sequences", user, re.M)
    facts["n_requested"] = int(m.group(1)) if m else None

    # The do-not-repeat list is the `already` set the production screen is given.
    already: set[str] = set()
    for b in blocks_of(user):
        if label_of(b) == "do_not_repeat":
            for line in b.splitlines()[1:]:
                s = line.strip()
                if s and re.fullmatch(r"[A-Z]+", s):
                    already.add(s)
    facts["already"] = already

    # The sequence the model was told to edit: the elite if this is an offspring prompt,
    # else the original. Everything downstream diffs against this, never against the
    # original, because that is what the edit acted on.
    base = facts["original"]
    for b in blocks_of(user):
        if label_of(b) == "elite_block":
            lines = [l.strip() for l in b.splitlines()[1:] if l.strip()]
            if lines and re.fullmatch(r"[A-Z]+", lines[0]):
                base = lines[0]
            break
    facts["base"] = base
    facts["is_offspring_prompt"] = base != facts["original"]
    return facts


def true_table(user: str, base: str) -> dict[int, dict] | None:
    """The real per-position structure of the sequence being edited.

    Taken from the elite's own table when the prompt carries one, else from the
    reference's. Checked against `base`: a table belonging to a different sequence would
    silently mislabel every position, so a mismatch returns nothing rather than a guess.
    """
    want = ("elite_table", "reference_table")
    for kind in want:
        for b in blocks_of(user):
            if label_of(b) != kind:
                continue
            head = next((l for l in b.splitlines() if l.startswith("[")), "")
            m = re.search(r"peptide=([A-Z]+)", head)
            if not m:
                continue
            if kind == "elite_table" and m.group(1) != base:
                continue
            if kind == "reference_table" and m.group(1) != base:
                continue
            rows: dict[int, dict] = {}
            for line in b.splitlines():
                rm = ROW_RE.match(line)
                if not rm:
                    continue
                rows[int(rm.group("pos")) - 1] = {
                    "aa": rm.group("aa"),
                    "burial": _num(rm.group("burial")),
                    "dmin": _num(rm.group("dmin")),
                    "ctc": _num(rm.group("ctc")),
                    "ovl": _num(rm.group("ovl")),
                }
            if len(rows) == len(base):
                return rows
    return None


def position_class(row: dict) -> str:
    """What the structure says about one position, in the prompt's own terms.

    Priority order, so the classes partition: an overlap is the failure the prompt calls
    the most important thing to act on; `empty` is a position contributing nothing;
    `anchor` is a buried contacting position the prompt says to KEEP, and is therefore
    the class an edit should AVOID if the table is being read.
    """
    if (row.get("ovl") or 0) > 0:
        return "clash"
    if (row.get("ctc") or 0) == 0:
        return "empty"
    if (row.get("burial") or 0) >= 0.5:
        return "anchor"
    return "shallow"


# --------------------------------------------------------------------------------------
# The variants
# --------------------------------------------------------------------------------------


def drop_labels(blocks: list[str], labels: set[str]) -> tuple[list[str], int]:
    kept = [b for b in blocks if label_of(b) not in labels]
    return kept, len(blocks) - len(kept)


def strip_overlap_mentions(blocks: list[str]) -> tuple[list[str], int]:
    """Remove every trace of receptor overlap: the `ovl` column, the OVERLAP annotation
    in the detail column, the `overlaps=N` in each table head, the column legend, and the
    sentence in HOW CANDIDATES ARE JUDGED that names it the most important thing.

    All five, because removing the column alone would leave the information in the text
    beside it and the variant would be measuring nothing.
    """
    n = 0
    out: list[str] = []
    for b in blocks:
        lines = b.splitlines()
        new: list[str] = []
        for line in lines:
            rm = ROW_RE.match(line)
            if rm:
                detail = re.sub(r"^OVERLAP [\d.]+A with [^;]*; ", "", rm.group("detail"))
                if detail != rm.group("detail"):
                    n += 1
                new.append(
                    f"  {rm.group('pos')} {rm.group('aa')}  {rm.group('ss')} "
                    f"{rm.group('burial')} {rm.group('dmin')} {rm.group('ctc')}  "
                    f"{detail}"
                )
                n += 1
                continue
            if line == HEADER_LINE:
                new.append("  pos aa  ss  burial  dmin  ctc  interactions / nearby receptor")
                n += 1
                continue
            if line.startswith("[") and "overlaps=" in line:
                new.append(re.sub(r" overlaps=\S+", "", line))
                n += 1
                continue
            if "ovl=steric overlaps (any is fatal)" in line:
                new.append(re.sub(r", ovl=steric overlaps \(any is fatal\)\.?", ".", line))
                n += 1
                continue
            if "the 'ovl' column below is the single most important thing to act on" in line:
                new.append(
                    re.sub(
                        r"\nPeptide-receptor overlap is by far the most common reason "
                        r"for rejection, so the 'ovl' column below is the single most "
                        r"important thing to act on\.",
                        "",
                        line,
                    )
                )
                n += 1
                continue
            new.append(line)
        b2 = "\n".join(new)
        # The judging block holds its overlap sentence inside one long line, so it is
        # removed on the whole block rather than line by line.
        if label_of(b) == "judging" and "most important thing to act on" in b2:
            b2 = re.sub(
                r"\s*Peptide-receptor overlap is by far the most common reason for "
                r"rejection, so the 'ovl' column below is the single most important "
                r"thing to act on\.",
                "",
                b2,
            )
            n += 1
        out.append(b2)
    return out, n


def scramble_tables(blocks: list[str], *, seed: int) -> tuple[list[str], int]:
    """Permute each table's per-position measurements across its positions.

    `pos` and `aa` stay put so the table still describes the right sequence; burial,
    dmin, ctc, ovl and the interaction detail move together as a unit, so every value
    remains a real measurement of this complex, attached to the wrong residue. The length
    and the shape of the prompt are untouched.
    """
    rng = random.Random(seed)
    n = 0
    out: list[str] = []
    for b in blocks:
        lines = b.splitlines()
        idx = [i for i, l in enumerate(lines) if ROW_RE.match(l)]
        if len(idx) < 3:
            out.append(b)
            continue
        parsed = [ROW_RE.match(lines[i]) for i in idx]
        payload = [
            (m.group("ss"), m.group("burial"), m.group("dmin"), m.group("ctc"),
             m.group("ovl"), m.group("detail"))
            for m in parsed
        ]
        order = list(range(len(payload)))
        # A permutation with no fixed point where one exists, so no row keeps its own
        # measurements by accident.
        for _ in range(64):
            rng.shuffle(order)
            if all(o != i for i, o in enumerate(order)):
                break
        for slot, (i, m) in enumerate(zip(idx, parsed)):
            ss, burial, dmin, ctc, ovl, detail = payload[order[slot]]
            lines[i] = (
                f"  {m.group('pos')} {m.group('aa')}  {ss} {burial} {dmin} "
                f"{ctc} {ovl}  {detail}"
            )
            n += 1
        out.append("\n".join(lines))
    return out, n


TABLE_LABELS = {"reference_table", "scored_table", "rejected_table", "elite_table"}

VARIANTS: dict[str, dict] = {
    "full": {"what": "the production prompt, unchanged"},
    "no_judging": {"drop": {"judging"},
                   "what": "how candidates are judged, and that overlap matters most"},
    "no_reference_table": {"drop": {"reference_table", "pocket_residues"},
                           "what": "the unmutated reference's interface table"},
    "no_history_lines": {"drop": {"history_lines"},
                         "what": "the scored candidates' verdicts and energies"},
    "no_scored_tables": {"drop": {"scored_table"},
                         "what": "full interface tables for the best scored candidates"},
    "no_rejected_table": {"drop": {"rejected_table"},
                          "what": "the rejected candidate's interface table"},
    "no_elite_table": {"drop": {"elite_table"},
                       "what": "the interface of the sequence it was told to improve"},
    "no_any_table": {"drop": TABLE_LABELS | {"pocket_residues"},
                     "what": "every interface table; verdicts and energies remain"},
    "no_noise_note": {"drop": {"noise_note"},
                      "what": "the warning that a few REU is seed noise"},
    "no_do_not_repeat": {"drop": {"do_not_repeat"},
                         "what": "the explicit list of already-scored sequences"},
    "no_overlap_info": {"fn": "overlap",
                        "what": "every mention of receptor overlap, column included"},
    "scramble_tables": {"fn": "scramble",
                        "what": "tables kept at full size, measurements permuted "
                                "across positions"},
}

DEFAULT_VARIANTS = (
    "full",
    "no_reference_table",
    "no_elite_table",
    "no_any_table",
    "no_overlap_info",
    "scramble_tables",
    "no_history_lines",
    "no_judging",
)


def apply_variant(user: str, variant: str, *, scramble_seed: int = 0) -> dict:
    """Build one variant of one prompt, refusing to pretend a no-op was an ablation."""
    spec = VARIANTS[variant]
    blocks = blocks_of(user)
    if variant == "full":
        return {"user": user, "n_changed": 0, "applicable": True}
    if spec.get("fn") == "overlap":
        new, n = strip_overlap_mentions(blocks)
    elif spec.get("fn") == "scramble":
        new, n = scramble_tables(blocks, seed=scramble_seed)
    else:
        new, n = drop_labels(blocks, spec["drop"])
    return {"user": "\n\n".join(new), "n_changed": n, "applicable": n > 0}


# --------------------------------------------------------------------------------------
# Scoring one completion
# --------------------------------------------------------------------------------------


def score_completion(text: str, facts: dict, table: dict[int, dict] | None) -> dict:
    """Production parsing and screening, plus where the edits landed."""
    cands, malformed = parse_completion(text, original=facts["original"])
    accepted, rejected = screen_candidates(
        cands,
        original=facts["original"],
        alphabet=facts["alphabet"],
        already=set(facts["already"]) | {facts["original"]},
    )
    base = facts["base"]
    edits_per: list[int] = []
    placed: dict[str, int] = defaultdict(int)
    touched: set[int] = set()
    for seq, _reason in accepted:
        if len(seq) != len(base):
            continue
        e = [i for i, (a, b) in enumerate(zip(base, seq)) if a != b]
        edits_per.append(len(e))
        touched.update(e)
        for i in e:
            placed[position_class(table[i]) if table and i in table else "unknown"] += 1
    n_edits = sum(placed.values())
    return {
        "n_parsed": len(cands),
        "n_malformed": len(malformed),
        "n_accepted": len(accepted),
        "accepted": [s for s, _ in accepted],
        # Reasons are bucketed by their fixed prefix, because several carry the offending
        # residue or count in the text and would otherwise each be their own category.
        "rejections": dict(sorted(
            Counter(
                r["reason"].split(":")[0].split(" (")[0] for r in rejected
            ).items()
        )),
        "mean_edits": round(st.fmean(edits_per), 3) if edits_per else None,
        "distinct_positions": len(touched),
        "edit_placement": dict(placed),
        "edit_placement_share": (
            {k: round(v / n_edits, 3) for k, v in placed.items()} if n_edits else {}
        ),
    }


def _jaccard(a: set, b: set) -> float | None:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------------------
# A statistic with resolution, and a null to read it against
# --------------------------------------------------------------------------------------
#
# Jaccard over sets of accepted SEQUENCES turned out to be saturated: at temperature 0.9
# the control's own two samples of the identical prompt overlap by ~0.02, so there is no
# room below it for an ablation to show up. The fix is not a bigger sample of the same
# statistic but a statistic that aggregates: WHICH POSITIONS the proposals edit, pooled
# over a cell's seeds and proposals. Two draws from the same process then agree closely
# even when they share no whole sequence, which is what makes a difference visible.
#
# The null is a permutation over the cells themselves. Pool the control's and the
# variant's samples, re-split them at random, and recompute the distance. That asks
# exactly the right question -- "is the control-to-variant difference larger than
# re-labelling the same samples would produce?" -- and needs no assumption about the
# proposer's sampling distribution, which nobody has characterised.


def edit_vector(rec: dict) -> Counter:
    """Which positions one cell's accepted proposals edited, counted with multiplicity."""
    base = rec.get("base") or ""
    c: Counter = Counter()
    for seq in rec.get("score", {}).get("accepted") or []:
        if len(seq) != len(base):
            continue
        for i, (a, b) in enumerate(zip(base, seq)):
            if a != b:
                c[i] += 1
    return c


def tv_distance(a: Counter, b: Counter) -> float | None:
    """Total variation between two position distributions. 0 = identical, 1 = disjoint."""
    na, nb = sum(a.values()), sum(b.values())
    if na == 0 or nb == 0:
        return None
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def _pool(vs: list[Counter]) -> Counter:
    out: Counter = Counter()
    for v in vs:
        out.update(v)
    return out


def permutation_distance(
    ctrl: list[Counter], var: list[Counter], *, seed: int = 0, n_perm: int = 200
) -> dict:
    """Observed control-vs-variant distance, against re-labellings of the same cells."""
    obs = tv_distance(_pool(ctrl), _pool(var))
    if obs is None:
        return {"observed": None, "null_median": None, "p": None, "n_perm": 0}
    allv = list(ctrl) + list(var)
    k = len(ctrl)
    rng = random.Random(seed)
    null: list[float] = []
    for _ in range(n_perm):
        idx = list(range(len(allv)))
        rng.shuffle(idx)
        d = tv_distance(
            _pool([allv[i] for i in idx[:k]]), _pool([allv[i] for i in idx[k:]])
        )
        if d is not None:
            null.append(d)
    if not null:
        return {"observed": round(obs, 4), "null_median": None, "p": None, "n_perm": 0}
    # One-sided: an ablation can only make the distributions MORE different.
    p = (1 + sum(1 for d in null if d >= obs)) / (1 + len(null))
    return {
        "observed": round(obs, 4),
        "null_median": round(st.median(null), 4),
        "excess": round(obs - st.median(null), 4),
        "p": round(p, 4),
        "n_perm": len(null),
    }


def class_distribution(table: dict[int, dict]) -> Counter:
    """How many positions of each kind the sequence being edited actually has.

    Without this the placement shares are uninterpretable: a 24 % share of edits landing
    on buried anchors means nothing until it is set against the share of positions that
    ARE buried anchors.
    """
    return Counter(position_class(row) for row in table.values())


def _sign_test(deltas: list[float]) -> dict:
    neg = sum(1 for d in deltas if d < 0)
    pos = sum(1 for d in deltas if d > 0)
    n = neg + pos
    if n == 0:
        return {"n": 0, "n_lower": 0, "n_higher": 0, "p": None}
    k = min(neg, pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {"n": n, "n_lower": neg, "n_higher": pos, "p": round(min(1.0, 2 * tail), 4)}


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(st.fmean(xs), 3) if xs else None


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def report(records: list[dict], tables: dict[str, dict] | None = None) -> dict:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("ok"):
            cells[(r["prompt"], r["variant"])].append(r)

    prompts = sorted({k[0] for k in cells})
    variants = [v for v in VARIANTS if any(k[1] == v for k in cells)]

    # The noise floor: the same prompt and variant at two different seeds.
    floor: dict[str, list[float]] = defaultdict(list)
    for (prompt, variant), rs in cells.items():
        sets = [set(r["score"]["accepted"]) for r in rs]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                j_ = _jaccard(sets[i], sets[j])
                if j_ is not None:
                    floor[variant].append(j_)

    out: dict = {
        "n_records": len(records),
        "n_prompts": len(prompts),
        "note": (
            "Overlap is Jaccard between sets of accepted sequences. `self_overlap` is "
            "the same prompt and variant at two seeds -- the noise floor. "
            "`overlap_with_full` is the variant against the control AT THE SAME SEED. A "
            "variant whose overlap with the control is not below the control's own "
            "self-overlap has not been shown to change the proposals. Nothing here is "
            "evidence about interface energy; no candidate was folded."
        ),
        "variants": {},
    }

    base_cells = {p: cells.get((p, "full"), []) for p in prompts}
    for variant in variants:
        rs = [r for (p, v), lst in cells.items() if v == variant for r in lst]
        xs_full: list[float] = []
        paired_yield: list[float] = []
        paired_anchor: list[float] = []
        for p in prompts:
            mine = cells.get((p, variant)) or []
            ctrl = base_cells.get(p) or []
            if not mine or not ctrl:
                continue
            by_seed_ctrl = {c["seed"]: c for c in ctrl}
            for m in mine:
                c = by_seed_ctrl.get(m["seed"])
                if c is None:
                    continue
                j_ = _jaccard(set(m["score"]["accepted"]), set(c["score"]["accepted"]))
                if j_ is not None and variant != "full":
                    xs_full.append(j_)
            if variant == "full":
                continue
            my_yield = _mean([r["score"]["n_accepted"] / max(1, r["n_requested"])
                              for r in mine])
            ct_yield = _mean([r["score"]["n_accepted"] / max(1, r["n_requested"])
                              for r in ctrl])
            if my_yield is not None and ct_yield is not None:
                paired_yield.append(my_yield - ct_yield)
            my_a = _mean([r["score"]["edit_placement_share"].get("anchor", 0.0)
                          for r in mine])
            ct_a = _mean([r["score"]["edit_placement_share"].get("anchor", 0.0)
                          for r in ctrl])
            if my_a is not None and ct_a is not None:
                paired_anchor.append(my_a - ct_a)

        rej: dict[str, int] = defaultdict(int)
        for r in rs:
            for k, v in r["score"]["rejections"].items():
                rej[k] += v

        # The edit-position distribution against a permutation null, per prompt.
        perm: list[dict] = []
        for p in prompts:
            mine = cells.get((p, variant)) or []
            ctrl = base_cells.get(p) or []
            if variant == "full" or len(mine) < 2 or len(ctrl) < 2:
                continue
            d = permutation_distance(
                [edit_vector(r) for r in ctrl],
                [edit_vector(r) for r in mine],
                seed=stable_key((p, variant)) % 10000,
            )
            if d["p"] is not None:
                perm.append(d)

        # Placement against chance: the share of EDITS on each class of position over the
        # share of POSITIONS of that class. 1.0 is indistinguishable from editing at
        # random; below 1.0 on `anchor` would be the model keeping what the prompt tells
        # it to keep.
        enrich: dict[str, float | None] = {}
        if tables:
            ratios: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                t = tables.get(r["prompt"])
                if not t:
                    continue
                cd = class_distribution(t)
                n_pos = sum(cd.values())
                n_ed = sum(r["score"]["edit_placement"].values())
                if not n_pos or not n_ed:
                    continue
                for cls in ("clash", "empty", "anchor", "shallow"):
                    exp = cd.get(cls, 0) / n_pos
                    if exp > 0:
                        ratios[cls].append(
                            (r["score"]["edit_placement"].get(cls, 0) / n_ed) / exp
                        )
            enrich = {k: _mean(v) for k, v in ratios.items()}
        out["variants"][variant] = {
            "what_is_removed": VARIANTS[variant]["what"],
            "n_cells": len({(r["prompt"], r["seed"]) for r in rs}),
            "n_prompts_applicable": len({r["prompt"] for r in rs}),
            "mean_prompt_chars": _mean([float(r["prompt_chars"]) for r in rs]),
            "mean_prompt_tokens": _mean([r.get("prompt_tokens") for r in rs]),
            "mean_parsed": _mean([float(r["score"]["n_parsed"]) for r in rs]),
            "mean_malformed": _mean([float(r["score"]["n_malformed"]) for r in rs]),
            "mean_accepted": _mean([float(r["score"]["n_accepted"]) for r in rs]),
            "mean_yield": _mean([r["score"]["n_accepted"] / max(1, r["n_requested"])
                                 for r in rs]),
            "mean_edits_per_proposal": _mean([r["score"]["mean_edits"] for r in rs]),
            "mean_distinct_positions": _mean(
                [float(r["score"]["distinct_positions"]) for r in rs]
            ),
            "edit_placement_share": {
                k: _mean([r["score"]["edit_placement_share"].get(k, 0.0) for r in rs])
                for k in ("clash", "empty", "anchor", "shallow", "unknown")
            },
            "rejections": dict(rej),
            "self_overlap": _mean(floor.get(variant, [])),
            "overlap_with_full": _mean(xs_full) if variant != "full" else None,
            "placement_enrichment_vs_chance": enrich,
            "edit_distribution_vs_full": {
                "n_prompts": len(perm),
                "mean_observed": _mean([d["observed"] for d in perm]),
                "mean_null_median": _mean([d["null_median"] for d in perm]),
                "mean_excess": _mean([d["excess"] for d in perm]),
                "n_prompts_p_below_0.1": sum(1 for d in perm if d["p"] < 0.1),
                # `n_higher` is the number of prompts where the variant's edit
                # distribution sat FURTHER from the control than re-labelling the same
                # cells would put it, which is the direction an ablation predicts.
                "sign_test_excess": _sign_test([d["excess"] for d in perm]),
            },
            "paired_vs_full": {
                "yield": {"mean_delta": _mean(paired_yield),
                          "sign_test": _sign_test(paired_yield)},
                "anchor_edit_share": {"mean_delta": _mean(paired_anchor),
                                      "sign_test": _sign_test(paired_anchor)},
            },
        }
    return out


def _emit(rep: dict, out_path: Path) -> int:
    rep_path = out_path.with_suffix(".report.json")
    rep_path.write_text(json.dumps(rep, indent=2))
    print(f"\n{rep['note']}\n")
    print(f"{'variant':>20} {'cells':>5} {'tok':>6} {'yield':>6} {'malf':>5} "
          f"{'edits':>6} {'self':>6} {'vs_full':>7}  "
          f"{'TVobs':>6} {'TVnull':>6} {'excess':>7} {'p<.1':>5} {'sgn_p':>6}")
    for v, d in rep["variants"].items():
        ed = d["edit_distribution_vs_full"]
        print(
            f"{v:>20} {d['n_cells']:>5} {str(d['mean_prompt_tokens'] or '-'):>6} "
            f"{str(d['mean_yield']):>6} {str(d['mean_malformed']):>5} "
            f"{str(d['mean_edits_per_proposal']):>6} "
            f"{str(d['self_overlap']):>6} {str(d['overlap_with_full']):>7}  "
            f"{str(ed['mean_observed']):>6} {str(ed['mean_null_median']):>6} "
            f"{str(ed['mean_excess']):>7} "
            f"{ed['n_prompts_p_below_0.1']:>2}/{ed['n_prompts']:<2} "
            f"{str(ed['sign_test_excess']['p']):>6}"
        )
    if any(d.get("placement_enrichment_vs_chance") for d in rep["variants"].values()):
        print(f"\nwhere edits land, over chance for that sequence's own position mix "
              f"(1.0 = indistinguishable from editing at random)")
        print(f"{'variant':>20} {'clash':>7} {'empty':>7} {'anchor':>7} {'shallow':>8}")
        for v, d in rep["variants"].items():
            e = d.get("placement_enrichment_vs_chance") or {}
            print(f"{v:>20} {str(e.get('clash')):>7} {str(e.get('empty')):>7} "
                  f"{str(e.get('anchor')):>7} {str(e.get('shallow')):>8}")
    print(f"\nreport: {rep_path}\nrows:   {out_path}")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def collect_prompts(args) -> list[dict]:
    """The prompts to ablate, one per (run, complex, round).

    One per complex and round, so a complex with four elites does not get four times the
    vote; the first is the best-ranked elite. Shared by the generation path and
    `--report-only`, so the report's structural baseline is derived from exactly the
    prompts the run used rather than from a second, possibly different, selection.
    """
    chosen: dict[tuple, Path] = {}
    for d in args.prompt_dir:
        for p in sorted(d.glob(f"*_a{args.attempt}.prompt.txt")):
            m = re.match(r"^(?P<parent>.+?)_r(?P<round>\d+)_(?P<tag>[a-z0-9]+)_a\d+$",
                         p.stem.replace(".prompt", ""))
            if not m:
                continue
            key = (d.parent.parent.parent.name, m.group("parent"), m.group("round"))
            if key not in chosen:
                chosen[key] = p

    usable: list[dict] = []
    for key, path in sorted(chosen.items()):
        try:
            system, user = read_prompt_file(path)
        except ValueError as exc:
            print(f"skip {path.name}: {exc}")
            continue
        facts = parse_facts(user)
        if not facts.get("ok"):
            print(f"skip {path.name}: {facts['error']}")
            continue
        if args.offspring_only and not facts["is_offspring_prompt"]:
            continue
        table = true_table(user, facts["base"])
        if table is None:
            print(f"skip {path.name}: no interface table matches the sequence being "
                  f"edited, so edit placement could not be attributed")
            continue
        usable.append({"key": key, "path": path, "system": system, "user": user,
                       "facts": facts, "table": table})
        if len(usable) >= args.max_prompts:
            break
    return usable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt-dir", type=Path, action="append", required=True,
                    help="a run's llm_proposals/round<N>/prompts directory; repeatable")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                    help=f"comma-separated; available: {','.join(VARIANTS)}")
    ap.add_argument("--seeds", type=int, default=3,
                    help="samples per cell. The control gets the same number, which is "
                         "what makes the noise floor comparable to every effect.")
    ap.add_argument("--max-prompts", type=int, default=12)
    ap.add_argument("--attempt", type=int, default=0,
                    help="which saved attempt to use. 0 is the clean first ask; later "
                         "attempts carry a retry tail about the previous answer.")
    ap.add_argument("--offspring-only", action="store_true", default=True,
                    help="use prompts that name an elite to improve (the default)")
    ap.add_argument("--any-prompt", dest="offspring_only", action="store_false")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--dry-run", action="store_true",
                    help="build every variant and report what each block costs, without "
                         "loading the model")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        recs = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
        # The structural truth is not stored on the records -- it is re-derived from the
        # same prompts, by the same selection, so the placement baseline cannot drift
        # from the one the run itself used.
        tables: dict[str, dict] = {}
        for u in collect_prompts(args):
            tables[f"{u['key'][0]}/{u['key'][1]}/r{u['key'][2]}"] = u["table"]
        missing = {r["prompt"] for r in recs} - set(tables)
        if missing:
            print(f"NOTE no table re-derived for {len(missing)} prompt(s): "
                  f"{sorted(missing)[:3]}; placement is reported without them")
        return _emit(report(recs, tables), args.out)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        print(f"unknown variant(s): {unknown}", file=sys.stderr)
        return 2
    if "full" not in variants:
        print("the control `full` is required: every effect is measured against it",
              file=sys.stderr)
        return 2

    usable = collect_prompts(args)
    if not usable:
        print("no usable prompts found", file=sys.stderr)
        return 2

    print(f"{len(usable)} prompt(s), {len(variants)} variant(s), {args.seeds} seed(s) "
          f"= {len(usable) * len(variants) * args.seeds} generation(s)")
    for u in usable:
        f = u["facts"]
        print(f"  {u['key'][0]}/{u['key'][1]} r{u['key'][2]}: base={f['base']} "
              f"L={f['length']} budget={f['stated_budget']} ask={f['n_requested']} "
              f"blocks={len(blocks_of(u['user']))}")

    built: list[dict] = []
    for u in usable:
        for v in variants:
            got = apply_variant(u["user"], v,
                                scramble_seed=stable_key(u["key"]) % 9973)
            if not got["applicable"] and v != "full":
                continue
            built.append({"u": u, "variant": v, "user": got["user"],
                          "n_changed": got["n_changed"]})

    print(f"\n{'variant':>20} {'prompts':>8} {'mean chars':>11} {'vs full':>9}")
    full_chars = {b["u"]["key"]: len(b["user"]) for b in built if b["variant"] == "full"}
    for v in variants:
        mine = [b for b in built if b["variant"] == v]
        if not mine:
            print(f"{v:>20} {0:>8}   not present in any prompt")
            continue
        ch = _mean([float(len(b["user"])) for b in mine])
        delta = _mean([float(len(b["user"]) - full_chars.get(b["u"]["key"], len(b["user"])))
                       for b in mine])
        print(f"{v:>20} {len(mine):>8} {str(ch):>11} {str(delta):>9}")

    if args.dry_run:
        print("\n--dry-run: no model loaded, nothing generated")
        return 0

    proposer = LlamaProposer(
        args.model, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    proposer.load()
    print(f"\nmodel loaded in {proposer.load_seconds}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    total = len(built) * args.seeds
    done = 0
    with open(args.out, "w") as fh:
        for b in built:
            u = b["u"]
            for s in range(args.seeds):
                # The seed depends on the prompt and the sample index, never on the
                # variant: the control and every ablation are sampled at the same seeds.
                seed = (stable_key(u["key"]) % 100000) * 10 + s
                comp = proposer.complete(u["system"], b["user"], seed=seed)
                rec = {
                    "ok": True,
                    "run": u["key"][0],
                    "prompt": f"{u['key'][0]}/{u['key'][1]}/r{u['key'][2]}",
                    "parent_id": u["key"][1],
                    "variant": b["variant"],
                    "n_changed": b["n_changed"],
                    "seed": s,
                    "seed_value": seed,
                    "base": u["facts"]["base"],
                    "n_requested": u["facts"]["n_requested"] or 1,
                    "prompt_chars": len(b["user"]),
                    "prompt_tokens": comp["prompt_tokens"],
                    "completion_tokens": comp["completion_tokens"],
                    "seconds": round(comp["seconds"], 2),
                    "score": score_completion(comp["text"], u["facts"], u["table"]),
                    "completion": comp["text"][:4000],
                }
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total}  last={b['variant']} "
                          f"accepted={rec['score']['n_accepted']}/{rec['n_requested']}",
                          flush=True)

    return _emit(report(records), args.out)


if __name__ == "__main__":
    raise SystemExit(main())
