#!/usr/bin/env python3
"""Does asking for edits instead of whole sequences recover the wasted output?

The proposal stage asks the model to write out each candidate in full, then screens what
comes back. Measured on the same twelve prompts the ablation used: 1.31-1.75 lines per
answer were unusable, nearly all of them the wrong length, against five to seven
requested -- and usable output sits near half. A 16-mer written out by hand is an
opportunity to miscount, and the model takes it.

The alternative is to ask for the edits and build the sequence in code: `K7R, L9M`
instead of `VPLRARNLPPSFFGEP`. Three things follow, and only the first is about yield:

1. The whole "wrong length" failure class stops existing, because the length is no longer
   something the model has to reproduce.
2. The edit becomes the output rather than something inferred from it. §4 of the search
   doc recorded the model describing a change at position 4 while making one at position
   5; with an edit list there is nothing to disagree with.
3. Citing the source residue (`K7R`, not `7R`) lets the code check the model is editing
   the residue it believes is there. `apply_substitutions` already rejects a mismatch, so
   a confused proposal is caught rather than silently applied to whatever sits at that
   position.

What this does NOT change: every legality rule stays in code and stays identical, so the
two formats are screened by the same `screen_candidates` against the same alphabet,
budget, novelty set and original. The budget is still counted against the ORIGINAL even
though the edits are stated against the sequence being improved -- which is exactly the
bookkeeping the model was being asked to do in its head, and now is not.

Paired like the ablation: the same prompts, the same seeds, the same number of samples
per format, so the two formats differ in nothing but the format.

Usage:
    .venv/bin/python script_utils/cpseq_edit_format_probe.py \
        --prompt-dir /data/user_data/yixiz/cpseq/es_pilot_pep/llm_proposals/round2/prompts \
        --prompt-dir /data/user_data/yixiz/cpseq/es_pilot_lnr/llm_proposals/round2/prompts \
        --seeds 3 --out /data/user_data/yixiz/cpseq/edit_format.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpseq_prompt_ablation import (  # noqa: E402
    blocks_of,
    collect_prompts,
    label_of,
    stable_key,
)
from proteinfoundation.cpseq.llm_propose import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LlamaProposer,
    apply_substitutions,
    parse_completion,
    screen_candidates,
)

# --------------------------------------------------------------------------------------
# The edit-list format
# --------------------------------------------------------------------------------------

SYSTEM_EDIT_TAIL = (
    "Output format. First a short block of reasoning, at most 120 words. Then the "
    "proposals, one per line. Each proposal is a list of substitutions, NOT a sequence:\n"
    "EDIT: <original residue><position><new residue>, ... | <brief reason>\n"
    "For example `EDIT: K7R, L9M | add a salt bridge at 7 and fill the pocket at 9`.\n"
    "Positions are numbered from 1 along the sequence you were told to improve. Name the "
    "residue currently at that position before the number: if it does not match, the "
    "proposal is discarded, so copy it from the sequence rather than guessing.\n"
    "Do not write out the full sequence. The sequence is built from your edits "
    "automatically, so you never need to count letters.\n"
    "Emit exactly the number of EDIT lines requested and nothing after the last one."
)

_EDIT_LINE = re.compile(
    r"^[ \t]*(?:\d{1,2}[.)]\s*)?EDITS?\s*:\s*(?P<body>[^|\n]*?)\s*"
    r"(?:\|\s*(?P<reason>.*))?$",
    re.MULTILINE | re.IGNORECASE,
)
_SUB = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,3})([ACDEFGHIKLMNPQRSTVWY])\b")


SYSTEM_EDIT_NUMBERED_TAIL = SYSTEM_EDIT_TAIL.replace(
    "Positions are numbered from 1 along the sequence you were told to improve. Name the "
    "residue currently at that position before the number: if it does not match, the "
    "proposal is discarded, so copy it from the sequence rather than guessing.",
    "Positions are numbered from 1 along the sequence you were told to improve, which is "
    "listed for you position by position. Name the residue currently at that position "
    "before the number, taking it from that list. Do NOT take it from the original "
    "sequence: the original is only used to count the budget, and a residue copied from "
    "it will not match, which discards the proposal.",
)


def rewrite_system(system: str, *, numbered: bool = False) -> str:
    """Swap the output-format paragraph for the edit-list one.

    Anchored on the sentence that introduces it, and raises if that anchor is gone, so a
    reworded system turn fails loudly instead of being sent with two contradictory format
    instructions.
    """
    anchor = "Output format."
    if anchor not in system:
        raise ValueError("system turn has no 'Output format.' paragraph to replace")
    tail = SYSTEM_EDIT_NUMBERED_TAIL if numbered else SYSTEM_EDIT_TAIL
    return system[: system.index(anchor)] + tail


def rewrite_task(user: str, *, base: str, budget: int, n_requested: int,
                 original: str, numbered: bool = False) -> str:
    """Swap the TASK block for one that asks for edits.

    With `numbered`, three further changes, all aimed at one measured failure: the model
    naming residues the original holds rather than the sequence it was told to edit. The
    base is laid out position by position, the original is not reprinted beside the edit
    instruction, and the budget is stated as a count rather than as a comparison against
    a second sequence spelled out in full.
    """
    blocks = blocks_of(user)
    idx = [i for i, b in enumerate(blocks) if label_of(b) == "task"]
    if not idx:
        raise ValueError("no TASK block found")
    if numbered:
        blocks[idx[0]] = (
            f"TASK\nPropose exactly {n_requested} distinct sets of substitutions to "
            f"{base}, the sequence named above as the one to improve.\n"
            + numbered_base(base) + "\n"
            f"State each proposal as substitutions only -- do not write out a "
            f"sequence:\n"
            f"EDIT: <residue currently there><position><new residue>, ... | "
            f"<brief reason>\n"
            f"One to three substitutions per proposal is usually right. Each proposal "
            f"must differ from the others, and the resulting sequence must not be one "
            f"already scored.\n"
            f"Budget: the finished sequence may differ from the original at no more "
            f"than {budget} positions in total, counting the differences {base} already "
            f"carries. That count is checked in software -- state the edits you intend "
            f"and it will be enforced, so do not try to work it out from the original.\n"
            f"End with exactly {n_requested} lines of the form:\n"
            f"EDIT: <residue currently there><position><new residue>, ... | "
            f"<brief reason>"
        )
        return "\n\n".join(blocks)
    blocks[idx[0]] = (
        f"TASK\nPropose exactly {n_requested} distinct sets of substitutions to "
        f"{base}, the sequence named above as the one to improve.\n"
        f"State each as substitutions only -- do not write out a sequence. Number "
        f"positions from 1 along {base}, and name the residue that is currently there:\n"
        f"EDIT: <current residue><position><new residue>, ... | <brief reason>\n"
        f"One to three substitutions per proposal is usually right. Each proposal must "
        f"differ from the others, and the resulting sequence must not be one already "
        f"scored.\n"
        f"The budget still applies to the FINAL sequence against the original "
        f"{original}: at most {budget} positions may differ from it in total, counting "
        f"the substitutions {base} already carries together with yours. The count is "
        f"checked in software, so state the edits you intend and it will be enforced.\n"
        f"End with exactly {n_requested} lines of the form:\n"
        f"EDIT: <current residue><position><new residue>, ... | <brief reason>"
    )
    return "\n\n".join(blocks)


def numbered_base(base: str) -> str:
    """The sequence being edited, laid out position by position.

    The model has to answer with a residue attached to a number, and the prompt was
    making it count along a bare string to find one. Here the mapping is given.
    """
    # Residue before number, matching the order the EDIT line asks for (`Y1F`), so the
    # model reads `Y1` and writes `Y1F` rather than transposing a `1Y` it was shown.
    cells = "   ".join(f"{aa}{i + 1}" for i, aa in enumerate(base))
    return (
        f"THE SEQUENCE YOU ARE EDITING, POSITION BY POSITION\n  {cells}\n"
        f"Every residue you name in an EDIT line must come from THIS list. The original "
        f"sequence is used ONLY to count the budget; never take a residue name from it."
    )


_DIFFERS = re.compile(
    r"^It differs from the original at (?P<n>\d+) position\(s\): (?P<subs>.+)$",
    re.MULTILINE,
)
# Every scored candidate's one-liner carries the same notation: `(4 sub: L6S,L7Q,C9V)`.
# Twelve of those outnumber the single elite line, so they are the larger exposure.
_HIST_SUBS = re.compile(
    r"\((?P<n>\d+) sub: (?:\(none\)|[A-Z]\d{1,3}[A-Z](?:,[A-Z]\d{1,3}[A-Z])*)\)"
)


def neutralize_original_notation(user: str, *, base: str, original: str) -> str:
    """Rewrite the one line that teaches the wrong mapping.

    The elite block states how the sequence to improve differs from the original using
    exactly the notation the model is asked to emit -- `D1Y` -- but anchored on the
    ORIGINAL. Measured consequence: 26 of 58 mis-cited substitutions named the residue
    the original holds at that position, against roughly 5 % expected if the letter were
    arbitrary. The information is worth keeping; the notation is not, so it is restated in
    words. Raises if the line is absent, so a reworded prompt fails loudly rather than
    leaving the notation in place under a variant that claims to have removed it.
    """
    m = _DIFFERS.search(user)
    if not m:
        raise ValueError("elite block has no 'differs from the original' line")
    diffs = [
        f"position {i + 1} holds {base[i]} here, where the original holds {original[i]}"
        for i in range(min(len(base), len(original)))
        if base[i] != original[i]
    ]
    body = "; ".join(diffs) if diffs else "nowhere -- it is the original"
    out = user[: m.start()] + (
        f"It differs from the original at {m.group('n')} position(s): {body}."
    ) + user[m.end():]
    # The counts stay -- how far a candidate sits from the original is real information
    # the model should have. Only the notation goes, because the sequences themselves are
    # printed on the same line and the substitution list adds nothing the model cannot
    # see, while modelling exactly the mistake being measured.
    out, n_hist = _HIST_SUBS.subn(lambda mm: f"({mm.group('n')} sub)", out)
    if "sub: " in out:
        raise ValueError("original-anchored substitution notation survived the rewrite")
    return out


def parse_edit_completion(text: str, *, base: str) -> tuple[list[tuple[str, str]], list[dict]]:
    """Read edit lists and build the sequences they describe.

    Returns `(candidates, malformed)` with the same shape `parse_completion` returns, so
    both formats feed the identical screen. A line whose cited residue disagrees with
    `base` is malformed rather than silently applied -- that disagreement is the signal
    that the model has lost track of the sequence, and acting on it would edit a position
    it did not mean.
    """
    found: list[tuple[str, str]] = []
    malformed: list[dict] = []
    for m in _EDIT_LINE.finditer(text):
        body, reason = m.group("body") or "", (m.group("reason") or "").strip()
        subs = [(a, int(p), b) for a, p, b in _SUB.findall(body.upper())]
        if not subs:
            malformed.append({"text": m.group(0)[:120], "reason": "no substitution parsed"})
            continue
        if any(not (1 <= p <= len(base)) for _a, p, _b in subs):
            malformed.append({"text": m.group(0)[:120],
                              "reason": "position outside the sequence"})
            continue
        seq = apply_substitutions(base, subs)
        if seq is None:
            malformed.append({"text": m.group(0)[:120],
                              "reason": "substitution cites wrong source residue"})
            continue
        found.append((seq, reason))
    return found, malformed


# --------------------------------------------------------------------------------------
# Scoring, shared by both formats
# --------------------------------------------------------------------------------------


def score(text: str, facts: dict, *, fmt: str) -> dict:
    base, original = facts["base"], facts["original"]
    if fmt.startswith("edits"):
        cands, malformed = parse_edit_completion(text, base=base)
    else:
        cands, malformed = parse_completion(text, original=original)
    accepted, rejected = screen_candidates(
        cands, original=original, alphabet=facts["alphabet"],
        already=set(facts["already"]) | {original},
    )
    edits = []
    for seq, _r in accepted:
        if len(seq) == len(base):
            edits.append(sum(1 for a, b in zip(base, seq) if a != b))
    return {
        "n_parsed": len(cands),
        "n_malformed": len(malformed),
        "n_accepted": len(accepted),
        "accepted": [s for s, _ in accepted],
        "malformed_reasons": dict(sorted(Counter(
            m["reason"] for m in malformed
        ).items())),
        "rejections": dict(sorted(Counter(
            r["reason"].split(":")[0].split(" (")[0] for r in rejected
        ).items())),
        "mean_edits": round(st.fmean(edits), 3) if edits else None,
    }


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(st.fmean(xs), 3) if xs else None


def _sign_test(deltas: list[float]) -> dict:
    neg = sum(1 for d in deltas if d < 0)
    pos = sum(1 for d in deltas if d > 0)
    n = neg + pos
    if n == 0:
        return {"n": 0, "n_lower": 0, "n_higher": 0, "p": None}
    k = min(neg, pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {"n": n, "n_lower": neg, "n_higher": pos, "p": round(min(1.0, 2 * tail), 4)}


def report(records: list[dict]) -> dict:
    by_fmt: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_fmt[r["format"]].append(r)
    out: dict = {
        "n_records": len(records),
        "note": (
            "Both formats are screened by the same production rules. `yield` is accepted "
            "over requested. `unusable` counts lines that produced no candidate at all: "
            "for whole sequences that is almost always the wrong length, for edit lists "
            "it is a mis-cited source residue or no parsable substitution. Nothing here "
            "is evidence about interface energy; no candidate was folded."
        ),
        "formats": {},
    }
    base = {(r["prompt"], r["seed"]): r for r in by_fmt.get("sequence", [])}
    for fmt, rs in sorted(by_fmt.items()):
        paired: list[float] = []
        for r in rs:
            b = base.get((r["prompt"], r["seed"]))
            if b is not None and fmt != "sequence":
                paired.append(
                    (r["score"]["n_accepted"] / max(1, r["n_requested"]))
                    - (b["score"]["n_accepted"] / max(1, b["n_requested"]))
                )
        reasons: dict[str, int] = defaultdict(int)
        for r in rs:
            for k, v in r["score"]["malformed_reasons"].items():
                reasons[k] += v
        rej: dict[str, int] = defaultdict(int)
        for r in rs:
            for k, v in r["score"]["rejections"].items():
                rej[k] += v
        out["formats"][fmt] = {
            "n_cells": len(rs),
            "mean_requested": _mean([float(r["n_requested"]) for r in rs]),
            "mean_parsed": _mean([float(r["score"]["n_parsed"]) for r in rs]),
            "mean_unusable": _mean([float(r["score"]["n_malformed"]) for r in rs]),
            "mean_accepted": _mean([float(r["score"]["n_accepted"]) for r in rs]),
            "yield": _mean([r["score"]["n_accepted"] / max(1, r["n_requested"])
                            for r in rs]),
            "mean_edits_per_proposal": _mean([r["score"]["mean_edits"] for r in rs]),
            "mean_completion_tokens": _mean([r.get("completion_tokens") for r in rs]),
            "mean_seconds": _mean([r.get("seconds") for r in rs]),
            "unusable_reasons": dict(reasons),
            "rejections": dict(rej),
            "yield_vs_sequence_format": {
                "mean_delta": _mean(paired), "sign_test": _sign_test(paired),
            } if fmt != "sequence" else None,
        }
    return out


def _emit(rep: dict, out_path: Path) -> int:
    p = out_path.with_suffix(".report.json")
    p.write_text(json.dumps(rep, indent=2))
    print(f"\n{rep['note']}\n")
    print(f"{'format':>10} {'cells':>6} {'asked':>6} {'parsed':>7} {'unusable':>9} "
          f"{'accepted':>9} {'yield':>6} {'edits':>6} {'out_tok':>8} {'secs':>6}")
    for f, d in rep["formats"].items():
        print(f"{f:>10} {d['n_cells']:>6} {str(d['mean_requested']):>6} "
              f"{str(d['mean_parsed']):>7} {str(d['mean_unusable']):>9} "
              f"{str(d['mean_accepted']):>9} {str(d['yield']):>6} "
              f"{str(d['mean_edits_per_proposal']):>6} "
              f"{str(d['mean_completion_tokens']):>8} {str(d['mean_seconds']):>6}")
    for f, d in rep["formats"].items():
        v = d.get("yield_vs_sequence_format")
        # None when this run generated no whole-sequence cells to pair against, which is
        # the normal case for `--formats <one> --append`: the pairing is recovered by
        # re-reporting over the merged rows, not by this print.
        if v and v["mean_delta"] is not None:
            s = v["sign_test"]
            print(f"\n{f} vs whole sequences: yield {v['mean_delta']:+.3f} per request, "
                  f"paired over {s['n']} cell(s), better on {s['n_higher']}, "
                  f"worse on {s['n_lower']}, p={s['p']}")
        if d["unusable_reasons"]:
            print(f"  {f} unusable lines: " + ", ".join(
                f"{k}={v}" for k, v in sorted(d["unusable_reasons"].items(),
                                              key=lambda kv: -kv[1])))
        if d["rejections"]:
            print(f"  {f} screened out: " + ", ".join(
                f"{k}={v}" for k, v in sorted(d["rejections"].items(),
                                              key=lambda kv: -kv[1])))
    print(f"\nreport: {p}\nrows:   {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt-dir", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-prompts", type=int, default=12)
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument("--offspring-only", action="store_true", default=True)
    ap.add_argument("--any-prompt", dest="offspring_only", action="store_false")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--formats", default="sequence,edits,edits_numbered",
                    help="comma-separated subset to generate. Seeds depend only on the "
                         "prompt and sample index, never on the format, so a format "
                         "generated on its own is paired with runs already on disk and "
                         "the rows can simply be concatenated.")
    ap.add_argument("--append", action="store_true",
                    help="add to --out instead of replacing it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        recs = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
        return _emit(report(recs), args.out)

    usable = collect_prompts(args)
    if not usable:
        print("no usable prompts found", file=sys.stderr)
        return 2

    cells: list[dict] = []
    for u in usable:
        f = u["facts"]
        n_req = f["n_requested"] or 5
        try:
            sys_edit = rewrite_system(u["system"])
            user_edit = rewrite_task(
                u["user"], base=f["base"], budget=f["stated_budget"] or 0,
                n_requested=n_req, original=f["original"],
            )
        except ValueError as exc:
            print(f"skip {u['path'].name}: {exc}")
            continue
        cells.append({"u": u, "format": "sequence", "system": u["system"],
                      "user": u["user"], "n_requested": n_req})
        cells.append({"u": u, "format": "edits", "system": sys_edit,
                      "user": user_edit, "n_requested": n_req})
        try:
            user_num = rewrite_task(
                neutralize_original_notation(
                    u["user"], base=f["base"], original=f["original"]
                ),
                base=f["base"], budget=f["stated_budget"] or 0,
                n_requested=n_req, original=f["original"], numbered=True,
            )
        except ValueError as exc:
            print(f"skip numbered variant for {u['path'].name}: {exc}")
        else:
            cells.append({"u": u, "format": "edits_numbered",
                          "system": rewrite_system(u["system"], numbered=True),
                          "user": user_num, "n_requested": n_req})

    want = {x.strip() for x in args.formats.split(",") if x.strip()}
    cells = [c for c in cells if c["format"] in want]
    print(f"{len(usable)} prompt(s), format(s) {sorted(want)}, {args.seeds} seed(s) = "
          f"{len(cells) * args.seeds} generation(s)")
    if args.dry_run:
        ex = next(c for c in cells if c["format"].startswith("edits"))
        print("\n=== edit-format TASK block ===")
        print([b for b in blocks_of(ex["user"]) if label_of(b) == "task"][0])
        print("\n=== edit-format system tail ===")
        print(ex["system"][-700:])
        return 0

    proposer = LlamaProposer(args.model, temperature=args.temperature,
                             top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    proposer.load()
    print(f"model loaded in {proposer.load_seconds:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    total = len(cells) * args.seeds
    done = 0
    with open(args.out, "a" if args.append else "w") as fh:
        for c in cells:
            u = c["u"]
            for s in range(args.seeds):
                # Seeded on the prompt and sample index only, never on the format, so
                # both formats are sampled at the same seeds.
                seed = (stable_key(u["key"]) % 100000) * 10 + s
                comp = proposer.complete(c["system"], c["user"], seed=seed)
                rec = {
                    "prompt": f"{u['key'][0]}/{u['key'][1]}/r{u['key'][2]}",
                    "format": c["format"],
                    "seed": s,
                    "seed_value": seed,
                    "base": u["facts"]["base"],
                    "original": u["facts"]["original"],
                    "n_requested": c["n_requested"],
                    "prompt_tokens": comp["prompt_tokens"],
                    "completion_tokens": comp["completion_tokens"],
                    "seconds": round(comp["seconds"], 2),
                    "score": score(comp["text"], u["facts"], fmt=c["format"]),
                    "completion": comp["text"][:4000],
                }
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total} last={c['format']} "
                          f"accepted={rec['score']['n_accepted']}/{rec['n_requested']}",
                          flush=True)
    return _emit(report(records), args.out)


if __name__ == "__main__":
    raise SystemExit(main())
