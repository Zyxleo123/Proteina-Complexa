#!/usr/bin/env python3
"""Arm E, round 1 on the LLM pilot complexes: the LLM says WHERE, the MLM says WHAT.

The proposal is split in two. `meta-llama/Llama-3.1-8B-Instruct` reads the same PLIP and
Biopython interface tables arm D reads and answers with POSITIONS ONLY. The CPSea
unmasking model (`cpseq_mlm_pocket`, step 60000) then fills those positions, conditioned
on the receptor pocket and the cyclization chemistry. Neither model does the other's job.

WHY SPLIT IT. README_CPSEQ_LLM_SEARCH.md §6 records the LLM locating overlaps and empty
positions unprompted, and §4 records its stated reasons not matching its own edits. It
reads interfaces better than it writes sequences. It also cannot count: all 16 malformed
round-1 proposals were off-by-one in length. A position list cannot be miscounted, and
the residue is chosen by a model fitted on 2.44M CPSea rows instead of by a text model
with no training signal for it.

THE CONTROL IS THE POINT. `random_mask_mlm_fill` draws masks uniformly at the SAME sizes
the LLM asked for, and fills them with the same model, temperature, order and draw count.
The two arms differ in exactly one thing -- which positions are opened -- so a difference
between them is attributable to the LLM's choice and nothing else. Without it this
measures the unmasking model, which is not the question.

Nothing here is folded. These are proposals. The Boltz-2 gate is what turns one into a
result, and with a 4.3 REU median seed spread a sequence that looks good in this file has
not been measured.

Usage:
  python script_utils/cpseq_mask_pilot.py --out-dir $DATA/cpseq/mask_pilot
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.cpseq.llm_propose import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    LlamaProposer,
    propose_mask_round,
    screen_candidates,
)
from proteinfoundation.cpseq.mlm_propose import MlmProposer  # noqa: E402
from proteinfoundation.cpseq.search import (  # noqa: E402
    ARM_LLM_MASK,
    ARM_MLM_MASK,
    KIND_MLM_FILL,
    default_alphabet,
)

DEFAULT_MLM = (
    "/data/user_data/yixiz/cpseq_mlm/runs/cpseq_mlm_pocket/checkpoints/hf_step60000"
)
# The arm D probe this reuses: the same three complexes, and its cached reference
# interface summaries, so both arms are shown the identical picture of the target.
LLM_PROBE = Path("/data/user_data/yixiz/cpseq/llm_probe")
PILOT = ("lnr_1elr", "lnr_2v8x", "lnr_6seo")


def reference_summary(parent_id: str, original: str) -> dict | None:
    """The cached interface of the unmutated reference, by sequence not by filename.

    Matching on `peptide_sequence_expected` rather than trusting a hash keeps this
    correct if the cache is ever rebuilt under a different input key.
    """
    for path in sorted((LLM_PROBE / "interfaces" / parent_id).glob("*.json")):
        d = json.loads(path.read_text())
        if d.get("peptide_sequence_expected") == original and d.get("ok"):
            d["_cache_path"] = str(path)
            return d
    return None


def matched_random_masks(
    sizes: list[int], length: int, seed: int, blocked: set[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    """Uniform masks at the LLM's own mask sizes.

    Matched on size because `search.match_counts` exists for exactly this reason: a
    difference between arms must not be an artefact of one of them mutating harder. The
    LLM's mask sizes are a property of the LLM, so the control inherits them rather than
    drawing its own.
    """
    rng = random.Random(seed)
    out: list[tuple[int, ...]] = []
    seen = set(blocked)
    exhausted: list[int] = []
    for k in sizes:
        for _ in range(200):
            m = tuple(sorted(rng.sample(range(length), k)))
            if m not in seen:
                seen.add(m)
                out.append(m)
                break
        else:
            # A short arm is reported, never padded with a duplicate -- the same rule
            # arm D follows when the model cannot fill its round. But the shortfall is
            # per SIZE, not for the round: on `lnr_1elr` (L=5) there are only 5 masks of
            # size 1 and the LLM had already taken 3, while masks of size 2 were nowhere
            # near exhausted. Breaking out of the loop on the first failure dropped the
            # satisfiable sizes too and left the control at 2 masks against the LLM's 8.
            exhausted.append(k)
            continue
    if exhausted:
        print(f"    note: no novel mask left at size(s) "
              f"{sorted(set(exhausted))} for length {length}; the control is short by "
              f"{len(exhausted)} and is NOT padded", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pockets", required=True,
                    help="JSON from cpseq_build_pilot_pockets.py")
    ap.add_argument("--llm-model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--mlm-model", default=DEFAULT_MLM)
    ap.add_argument("--round", type=int, default=1, dest="round_index")
    ap.add_argument("--n-masks", type=int, default=8,
                    help="mask sets requested from the LLM per parent")
    ap.add_argument("--draws-per-mask", type=int, default=2,
                    help="MLM fills per mask; identical in both arms")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--llm-temperature", type=float, default=0.9)
    ap.add_argument("--fill-temperature", type=float, default=1.0)
    ap.add_argument("--order", default="random", choices=("random", "confidence"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--parents", default=",".join(PILOT))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (out_dir / "proposals").mkdir(parents=True, exist_ok=True)

    targets = json.loads(Path(args.pockets).read_text())
    parents = [p for p in args.parents.split(",") if p]
    alpha = default_alphabet(allow_cys=False)

    llm = LlamaProposer(args.llm_model, device=args.device,
                        temperature=args.llm_temperature)
    t0 = time.time()
    llm.load()
    print(f"llama loaded in {time.time() - t0:.1f}s", flush=True)

    # Both models are resident at once: 16.5 GiB plus 2.6 GiB. That fits the 48 GiB card
    # this is pinned to, and it avoids writing masks to disk only to read them straight
    # back. Boltz-2 is NOT in this job, which is the constraint that mattered.
    mlm = MlmProposer(args.mlm_model, device=args.device)
    t0 = time.time()
    mlm.load()
    print(f"mlm loaded in {time.time() - t0:.1f}s, "
          f"pocket_enabled={mlm.pocket_enabled}", flush=True)
    if not mlm.pocket_enabled:
        raise SystemExit("the unmasking checkpoint is the no-pocket arm; wrong model")

    all_rows: list[dict] = []
    summary: list[dict] = []
    for parent_id in parents:
        tgt = targets[parent_id]
        original = tgt["reference_sequence"]
        budget = tgt["budget"]
        ref = reference_summary(parent_id, original)
        print(f"\n=== {parent_id}  {original}  L={len(original)}  budget={budget}  "
              f"reference_interface={'cached' if ref else 'MISSING'}", flush=True)

        masks, record = propose_mask_round(
            proposer=llm,
            original=original,
            budget=budget,
            round_index=args.round_index,
            parent_id=parent_id,
            reference_summary=ref,
            history=[],
            alphabet=alpha,
            pdb_id=tgt["pdb_id"],
            max_attempts=args.max_attempts,
            prompt_dir=out_dir / "prompts",
            n_requested=args.n_masks,
        )
        record["reference_interface"] = ref.get("_cache_path") if ref else None
        (out_dir / "masks" / f"{parent_id}_r{args.round_index}.json").write_text(
            json.dumps(record, indent=2)
        )
        sizes = [len(m.mask) for m in masks]
        print(f"  LLM: {len(masks)}/{args.n_masks} legal masks, sizes {sizes}, "
              f"{len(record['malformed'])} malformed, "
              f"{len(record['rejections'])} rejected", flush=True)
        for m in masks:
            print(f"    mask {[i + 1 for i in m.mask]}  {m.reason[:88]}", flush=True)
        if not masks:
            summary.append({"parent_id": parent_id, "n_masks": 0,
                            "note": record.get("note", "")})
            continue

        control = matched_random_masks(
            sizes, len(original), seed=args.seed + abs(hash(parent_id)) % 9973,
            blocked={m.mask for m in masks},
        )
        print(f"  control: {len(control)} random masks at the same sizes", flush=True)

        for arm, mask_list in (
            (ARM_LLM_MASK, [m for m in masks]),
            (ARM_MLM_MASK, control),
        ):
            rows = mlm.fill(
                masks=mask_list,
                pocket_seq=tgt["pocket_seq"],
                cyc_type=tgt["cyc_type"],
                original=original,
                draws_per_mask=args.draws_per_mask,
                temperature=args.fill_temperature,
                order=args.order,
                seed=args.seed,
            )
            # The shared legality screen, against the ORIGINAL. A legal mask can still
            # produce an illegal sequence, and that rejection belongs to the unmasking
            # model rather than to whoever chose the positions.
            screened, rejected = screen_candidates(
                [(r["sequence"], r["mask_reason"]) for r in rows],
                original=original, alphabet=alpha, already={original},
            )
            keep = {s for s, _ in screened}
            n_noop = sum(1 for r in rows if r["noop"])
            for r in rows:
                r.update({
                    "parent_id": parent_id,
                    "arm": arm,
                    "round_index": args.round_index,
                    "proposal_kind": KIND_MLM_FILL,
                    "original": original,
                    "accepted": r["sequence"] in keep,
                })
            all_rows.extend(rows)
            print(f"  {arm:22s} {len(rows):3d} fills -> "
                  f"{len(screened):3d} accepted, {n_noop} no-op, "
                  f"{len(rejected)} rejected", flush=True)
            summary.append({
                "parent_id": parent_id,
                "arm": arm,
                "n_masks": len(mask_list),
                "mask_sizes": sizes,
                "n_fills": len(rows),
                "n_accepted": len(screened),
                "n_noop": n_noop,
                "n_rejected": len(rejected),
                "rejection_reasons": sorted({r["reason"] for r in rejected}),
            })

    path = out_dir / "proposals" / f"round{args.round_index}.jsonl"
    with path.open("w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "llm": llm.config,
        "mlm": mlm.config,
        "draws_per_mask": args.draws_per_mask,
        "fill_temperature": args.fill_temperature,
        "unmask_order": args.order,
        "seed": args.seed,
        "parents": parents,
        "pockets_source": args.pockets,
        "summary": summary,
        "folded": False,
        "note": (
            "Proposals only. Nothing here has been through the Boltz-2 gate, so no "
            "statement about design quality can be made from this file."
        ),
    }
    (out_dir / f"round{args.round_index}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {path} ({len(all_rows)} rows) and the run meta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
