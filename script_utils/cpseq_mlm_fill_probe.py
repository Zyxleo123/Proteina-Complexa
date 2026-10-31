"""What does the unmasking model's per-position distribution actually look like?

The pocket-swap probe turned up something that matters more than the pocket. Sampling one
position at temperature 1.0, the residue drawn at draw index k was very nearly the SAME
residue at every position of a peptide -- draw 0 gave R at nine different positions of
`lnr_2v8x`, draw 1 gave K, draw 5 gave M. With a fixed seed per draw index the uniform
variate is shared, so identical draws across positions mean the per-position CDFs are
nearly identical: at temperature 1.0 the sample is coming from something close to a
position-independent residue prior.

That is consistent with the training numbers rather than contradicting them. Top-1
accuracy of 0.268 against a positional prior of 0.204 says the ARGMAX carries signal; it
says nothing about the tail, and a broad tail of similar shape everywhere is what a
temperature-1.0 sample mostly returns.

So the question this probe answers is: is the signal in the argmax, and at what
temperature does a sample still carry it? It reports, per position, the top-3 and the
entropy, and the probability mass the model puts on the native residue. One forward per
position; no sampling, so there is nothing to average over.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path("/home/yixiz/Proteina-Complexa")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script_utils"))

from proteinfoundation.cpseq.mlm_propose import AA20, MlmProposer  # noqa: E402

POCKET = "/data/user_data/yixiz/cpseq_mlm/runs/cpseq_mlm_pocket/checkpoints/hf_step60000"
NOPOCKET = (
    "/data/user_data/yixiz/cpseq_mlm/runs/cpseq_mlm_nopocket/checkpoints/hf_step60000"
)


def distribution(p: MlmProposer, seq: str, pos: int, pocket: str, cyc: int):
    """The model's residue distribution at one masked position, over the 20 canonical."""
    import torch

    from cpseq_mlm_data import CorpusRow

    row = CorpusRow(
        example_id="probe", pep_seq=seq, pocket_seq=pocket, cyc_type=cyc,
        cyc_anchor_i=-1, cyc_anchor_j=-1,
    )
    enc = p._encoder.encode(row, [pos], cyc_type=cyc)
    ids = torch.tensor([enc["input_ids"]], device=p.device)
    attn = torch.tensor([enc["attention_mask"]], device=p.device)
    with torch.no_grad():
        logits = p._model(input_ids=ids, attention_mask=attn).logits[0]
    slot = logits[enc["pep_start"] + pos, p._residue_ids].float()
    return torch.softmax(slot, dim=-1).cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pockets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tgts = json.loads(Path(args.pockets).read_text())
    pids = list(tgts)
    p = MlmProposer(POCKET, device=args.device)
    p.load()

    out = []
    for i, pid in enumerate(pids):
        t = tgts[pid]
        seq, cyc = t["reference_sequence"], t["cyc_type"]
        other = tgts[pids[(i + 1) % len(pids)]]["pocket_seq"]
        print(f"\n=== {pid}  {seq}")
        print(f"{'pos':>4} {'nat':>3} {'top-3 (own pocket)':<28} {'H':>5} "
              f"{'p(nat)':>7} {'top-1 swapped':>14}")
        for pos in range(len(seq)):
            pr = distribution(p, seq, pos, t["pocket_seq"], cyc)
            sw = distribution(p, seq, pos, other, cyc)
            order = pr.argsort(descending=True)
            top3 = " ".join(f"{AA20[int(j)]}{pr[int(j)]:.2f}" for j in order[:3])
            H = float(-(pr * (pr + 1e-12).log()).sum())
            nat = AA20.index(seq[pos]) if seq[pos] in AA20 else None
            pnat = float(pr[nat]) if nat is not None else float("nan")
            sw_top = AA20[int(sw.argmax())]
            row = {
                "parent_id": pid, "position": pos, "native": seq[pos],
                "top3": [(AA20[int(j)], round(float(pr[int(j)]), 4)) for j in order[:3]],
                "entropy_nats": round(H, 4),
                "perplexity": round(math.exp(H), 2),
                "p_native": round(pnat, 4),
                "argmax_own": AA20[int(pr.argmax())],
                "argmax_swapped": sw_top,
                "argmax_agrees_under_swap": AA20[int(pr.argmax())] == sw_top,
            }
            out.append(row)
            flag = "" if row["argmax_agrees_under_swap"] else "  <- argmax moves"
            print(f"{pos + 1:>4} {seq[pos]:>3} {top3:<28} {H:5.2f} {pnat:7.3f} "
                  f"{sw_top:>14}{flag}")

    n = len(out)
    moved = sum(1 for r in out if not r["argmax_agrees_under_swap"])
    nat_top1 = sum(1 for r in out if r["argmax_own"] == r["native"])
    mean_ppl = sum(r["perplexity"] for r in out) / n
    print(f"\nover {n} positions: mean per-position perplexity {mean_ppl:.2f} "
          f"(uniform over 20 = 20.0)")
    print(f"  argmax recovers the native residue at {nat_top1}/{n}")
    print(f"  argmax moves when the pocket is swapped at {moved}/{n}")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
