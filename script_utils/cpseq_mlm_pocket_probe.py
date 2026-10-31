"""Does the unmasking model's pocket channel do anything on these three targets?

`val/pocket_gain` is +0.042 top-1 over all of CPSea val -- real and positive, but small,
and measured on 4,757 positions of a panel these three complexes are not in. Before the
pocket is used as the thing that makes this proposer target-aware, it has to be shown to
move anything here.

Two probes, both paired and both cheap:

  1. SCORE the native peptide under its own pocket, under each of the other parents'
     pockets, and under no pocket at all. If the channel is live the native pocket wins.
  2. FILL the same masks under its own pocket and under a swapped one, from the SAME
     seeds. Same seed and same mask means every difference in the output is the pocket.

A null result here is informative and must not be buried: it would mean this arm is a
chemistry-and-length prior rather than a pocket-conditioned proposer, and the honest
comparison for it is then the no-pocket twin, not the LLM.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/home/yixiz/Proteina-Complexa")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from proteinfoundation.cpseq.mlm_propose import MlmProposer  # noqa: E402

POCKET_CKPT = (
    "/data/user_data/yixiz/cpseq_mlm/runs/cpseq_mlm_pocket/checkpoints/hf_step60000"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pockets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tgts = json.loads(Path(args.pockets).read_text())
    pids = list(tgts)
    p = MlmProposer(POCKET_CKPT, device=args.device)
    p.load()

    out = {"checkpoint": POCKET_CKPT, "score": [], "fill": []}

    print("=== probe 1: pseudo-perplexity of the native peptide, by pocket shown ===")
    print(f"{'parent':10s} {'pocket shown':12s} {'ppl':>8s} {'logprob':>9s}")
    for pid in pids:
        t = tgts[pid]
        for other in pids + ["<none>"]:
            pocket = "" if other == "<none>" else tgts[other]["pocket_seq"]
            s = p.score(
                sequence=t["reference_sequence"],
                pocket_seq=pocket,
                cyc_type=t["cyc_type"],
            )
            tag = "own" if other == pid else ("none" if other == "<none>" else other)
            row = {
                "parent_id": pid,
                "pocket_from": other,
                "is_own": other == pid,
                "pseudo_perplexity": round(s["pseudo_perplexity"], 4),
                "pseudo_logprob": round(s["pseudo_logprob"], 4),
                "n_scored": s["n_scored"],
            }
            out["score"].append(row)
            print(
                f"{pid:10s} {tag:12s} {row['pseudo_perplexity']:8.3f} "
                f"{row['pseudo_logprob']:9.3f}"
            )

    print("\n=== probe 2: same mask, same seed, own pocket vs a swapped one ===")
    print(f"{'parent':10s} {'swap from':10s} {'identical fills':>16s} {'of':>4s}")
    for i, pid in enumerate(pids):
        t = tgts[pid]
        other = pids[(i + 1) % len(pids)]
        L = t["peptide_length"]
        # Every single position, so the comparison is not a function of which mask was
        # drawn. One position at a time is also the regime the search actually uses.
        same = 0
        total = 0
        per_pos = []
        for pos in range(L):
            kw = dict(
                original=t["reference_sequence"],
                edit_mask=[pos],
                cyc_type=t["cyc_type"],
                n=args.draws,
                temperature=1.0,
                order="random",
                seed=7,
                budget=t["budget"],
            )
            a = p.propose(pocket_seq=t["pocket_seq"], **kw)
            b = p.propose(pocket_seq=tgts[other]["pocket_seq"], **kw)
            agree = sum(1 for x, y in zip(a, b) if x.sequence == y.sequence)
            same += agree
            total += len(a)
            per_pos.append({
                "position": pos,
                "native": t["reference_sequence"][pos],
                "agree": agree,
                "n": len(a),
                "own": "".join(x.sequence[pos] for x in a),
                "swapped": "".join(y.sequence[pos] for y in b),
            })
        out["fill"].append({
            "parent_id": pid,
            "swapped_with": other,
            "identical": same,
            "total": total,
            "per_position": per_pos,
        })
        print(f"{pid:10s} {other:10s} {same:16d} {total:4d}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
