"""Does the checkpoint reproduce its own reported accuracy through MY inference path?

The pilot targets gave argmax recovery of 2/35 and a mean per-position perplexity of
16.3 against 20.0 for uniform. The run's `manifest.json` reports top-1 of 0.268 over all
regimes and 0.266 in the `budget` regime, against a positional prior of 0.204. Those two
statements cannot both describe a working setup, so one of them is about something else.

This separates the two candidates by running the SAME code -- same `EsmMlmEncoder`, same
`MlmProposer`, same logit restriction -- on rows from the model's own validation corpus:

  reproduces ~0.27  -> the inference path is fine and the LNR pilot targets are simply
                       off-distribution for a model trained on CPSea.
  near chance       -> the path is wrong, and nothing measured on the pilot targets
                       means anything yet.

Two mask regimes, because they are different questions. `single` masks one position with
every other revealed, which is what the proposer does. `budget` masks k ~ U{1..L/2} at
once, which is what the training eval scored.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO = Path("/home/yixiz/Proteina-Complexa")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from proteinfoundation.cpseq.mlm_propose import AA20, MlmProposer  # noqa: E402

POCKET = "/data/user_data/yixiz/cpseq_mlm/runs/cpseq_mlm_pocket/checkpoints/hf_step60000"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n-rows", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    from cpseq_mlm_data import CorpusRow

    rows = []
    with open(args.shard) as fh:
        for line in fh:
            rows.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_rows]
    print(f"{len(rows)} val rows from {Path(args.shard).name}")

    p = MlmProposer(POCKET, device=args.device)
    p.load()

    results = {}
    for regime in ("single", "budget"):
        hit = tot = 0
        logp = 0.0
        rng = random.Random(args.seed + 1)
        for r in rows:
            seq = r["pep_seq"]
            L = len(seq)
            if L < 2:
                continue
            if regime == "single":
                masks = [[i] for i in range(L)]
            else:
                k = rng.randint(1, max(1, L // 2))
                masks = [sorted(rng.sample(range(L), k))]
            for hidden in masks:
                row = CorpusRow(
                    example_id=r["example_id"], pep_seq=seq,
                    pocket_seq=r["pocket_seq"], cyc_type=int(r["cyc_type"]),
                    cyc_anchor_i=int(r["cyc_anchor_i"]),
                    cyc_anchor_j=int(r["cyc_anchor_j"]),
                )
                enc = p._encoder.encode(row, hidden, cyc_type=int(r["cyc_type"]))
                if enc["too_long"]:
                    continue
                ids = torch.tensor([enc["input_ids"]], device=p.device)
                attn = torch.tensor([enc["attention_mask"]], device=p.device)
                with torch.no_grad():
                    logits = p._model(input_ids=ids, attention_mask=attn).logits[0]
                res = logits[:, p._residue_ids].float()
                for pos in hidden:
                    slot = res[enc["pep_start"] + pos]
                    pred = AA20[int(slot.argmax())]
                    if seq[pos] not in AA20:
                        continue
                    hit += pred == seq[pos]
                    tot += 1
                    lp = torch.log_softmax(slot, dim=-1)
                    logp += float(lp[AA20.index(seq[pos])])
        acc = hit / max(tot, 1)
        ppl = math.exp(-logp / max(tot, 1))
        results[regime] = {"top1": round(acc, 4), "n": tot, "perplexity": round(ppl, 3)}
        print(f"  {regime:7s} top-1 {acc:.4f} over {tot} positions, "
              f"perplexity {ppl:.2f}")

    manifest = json.loads(
        (Path(POCKET).parent.parent / "manifest.json").read_text()
    )
    base = manifest["baselines"]
    print("\nreported in the run's own manifest, for comparison:")
    print(f"  positional prior (all regimes) top-1 "
          f"{base['all']['positional_argmax_acc']:.4f}")
    print(f"  marginal prior   (all regimes) top-1 "
          f"{base['all']['marginal_argmax_acc']:.4f}")
    print("  model top-1 reported in README_CPSEQ_MLM_MIGRATION.md: 0.268 (all), "
          "0.266 (budget)")
    Path(args.out).write_text(json.dumps(
        {"shard": args.shard, "n_rows": len(rows), "results": results,
         "manifest_baselines": base["all"]}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
