#!/usr/bin/env python3
"""Structure-aware features for every candidate, from ProteinMPNN, without folding anything.

Why this exists. The feature baselines say the learned scorer's problem is not capacity:
post-fold readouts reach within-target rho 0.781 against a ~0.80 ceiling, so once a
structure exists the Rosetta dG is nearly determined, while sequence-only features reach
0.222. The missing input is structural context -- and the scorer cannot fold, because
avoiding the ~238 s/candidate fold is the entire point of a proxy.

ProteinMPNN closes that gap for free on this particular dataset. Every candidate for a
target is an equal-length substitution variant of one reference peptide, and that
reference peptide has a *native complex* on disk (all 20/20 resolved by exact sequence
match, see `cpseq_meanscore_structures.py`). So each candidate can be threaded onto the
reference backbone and scored by a model that sees the receptor, with no prediction step.

Two scores per candidate, because they fail differently
-------------------------------------------------------
* **`bb_ll`** -- log p(aa_i | backbone) summed over peptide positions, from one
  backbone-conditioned pass per target (`conditional_probs` with the sequence hidden).
  This is additive over positions, so it is cheap (20 passes scores all 2k candidates)
  and it is exactly the functional form the scorer's own mutation-effect head assumes.
* **`ar_ll`** -- the teacher-forced autoregressive score, p(s_i | s_<i, backbone),
  averaged over several random decoding orders. This is what ProteinMPNN's `--score_only`
  reports. It sees intra-peptide sequence context that `bb_ll` cannot, at one pass per
  candidate per decoding order.

Both are also emitted as **deltas against the reference peptide** (`*_delta`), which is
the form most likely to transfer across targets: an absolute likelihood is dominated by
peptide length and by how designable that particular backbone is, both of which are
constant within a target and therefore invisible to the within-target metric.

What this is not
----------------
A likelihood is not an energy. ProteinMPNN scores "does this sequence belong on this
backbone", which is correlated with interface quality but is not interface dG, and it is
blind to the fact that the candidate's *own* backbone would differ from the reference's.
It also reads **backbone atoms only** (N, CA, C, O) and is **cyclic-blind**: its
`PositionalEncodings` one-hots a clipped LINEAR relative offset, so a head-to-tail closure
is invisible to it. That is *not* worked around here. Fixing it needs a patched featurizer
(ProteinMPNN consumes `residue_idx`, not an offset matrix, so the repo's
`evaluation/cyclic_offset.py` cannot simply be dropped in), which is the same limitation
`cpseq_mpnn_propose.py` already records. The consequence to keep in mind: these features
describe the peptide as a linear chain on the reference backbone.

Energies elsewhere in this pipeline are Rosetta ref2015 REU and are NOT measured binding
free energies; the numbers here are log-likelihoods, in nats.

Usage:
    python script_utils/cpseq_meanscore_mpnn_feats.py \
        --dataset    /zfsauton/scratch/yixiz/cpseq/meanscore/dataset_v3.json \
        --structures /zfsauton/scratch/yixiz/cpseq/meanscore/structures_v3.json \
        --json       /zfsauton/scratch/yixiz/cpseq/meanscore/mpnn_feats_v3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
MPNN_DIR = REPO / "community_models" / "ProteinMPNN"
sys.path.insert(0, str(MPNN_DIR))

from protein_mpnn_utils import (  # noqa: E402
    ProteinMPNN,
    StructureDatasetPDB,
    parse_PDB,
    tied_featurize,
)

# ProteinMPNN's fixed alphabet order.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {a: i for i, a in enumerate(MPNN_ALPHABET)}


def load_model(model_name: str, device, backbone_noise: float = 0.0):
    ckpt_path = MPNN_DIR / "vanilla_model_weights" / f"{model_name}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no ProteinMPNN weights at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # The checkpoint stores only num_edges and noise_level; hidden_dim and the layer count
    # are hardcoded in protein_mpnn_run.py, so they are mirrored here rather than guessed.
    # Keeping these identical to the runner is what makes a likelihood from this script
    # the same quantity as one from `--score_only`.
    hidden_dim, num_layers = 128, 3
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=hidden_dim,
        edge_features=hidden_dim, hidden_dim=hidden_dim,
        num_encoder_layers=num_layers, num_decoder_layers=num_layers,
        augment_eps=backbone_noise, k_neighbors=ckpt["num_edges"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt.get("num_edges"), ckpt.get("noise_level")


def featurize_target(pdb: str, peptide_chain: str, receptor_chains: list[str], device):
    """Parse the complex and return the tied_featurize tensors plus the design mask."""
    chains = [peptide_chain] + list(receptor_chains)
    pdb_dicts = parse_PDB(pdb, input_chain_list=chains)
    dataset = StructureDatasetPDB(pdb_dicts, truncate=None, max_length=20000)
    batch = [dataset[0]]
    chain_dict = {batch[0]["name"]: ([peptide_chain], list(receptor_chains))}
    out = tied_featurize(batch, device, chain_dict, None, None, None, None, None,
                         ca_only=False)
    X, S, mask, _lengths, chain_M, chain_encoding_all, _chain_list, _visible, _masked, \
        _masked_chain_length, chain_M_pos, _omit_AA_mask, residue_idx, *_rest = out
    return {
        "X": X, "S": S, "mask": mask, "chain_M": chain_M,
        "chain_M_pos": chain_M_pos, "residue_idx": residue_idx,
        "chain_encoding_all": chain_encoding_all,
    }


def design_positions(t) -> np.ndarray:
    """Indices where ProteinMPNN would design, i.e. the peptide chain's positions."""
    m = (t["chain_M"][0] * t["chain_M_pos"][0] * t["mask"][0]).detach().cpu().numpy()
    return np.where(m > 0.5)[0]


def decode_seq(S_row, idx) -> str:
    return "".join(MPNN_ALPHABET[int(S_row[i])] for i in idx)


@torch.no_grad()
def backbone_marginals(model, t, idx, n_draw: int, device) -> np.ndarray:
    """log p(aa_i | backbone) at each designed position, sequence hidden.

    `conditional_probs(..., backbone_only=True)` is ProteinMPNN's own
    `--conditional_probs_only_backbone` path, averaged over decoding orders.
    """
    acc = []
    for d in range(n_draw):
        randn = torch.randn(t["chain_M"].shape, device=device,
                            generator=torch.Generator(device=device).manual_seed(d))
        lp = model.conditional_probs(
            t["X"], t["S"], t["mask"], t["chain_M"] * t["chain_M_pos"],
            t["residue_idx"], t["chain_encoding_all"], randn, True)
        acc.append(lp[0].detach().cpu().numpy())
    return np.mean(acc, axis=0)[idx]  # (L, 21)


@torch.no_grad()
def autoregressive_ll(model, t, idx, seqs: list[str], n_draw: int, device,
                      batch_size: int) -> np.ndarray:
    """Mean teacher-forced log p(s_i | s_<i, backbone) over designed positions."""
    L_full = t["S"].shape[1]
    out = np.zeros(len(seqs))
    pos = torch.tensor(idx, device=device, dtype=torch.long)
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        B = len(chunk)
        S = t["S"].repeat(B, 1).clone()
        for bi, s in enumerate(chunk):
            S[bi, pos] = torch.tensor([AA_TO_IDX.get(c, 20) for c in s],
                                      device=device, dtype=S.dtype)
        X = t["X"].repeat(B, 1, 1, 1)
        mask = t["mask"].repeat(B, 1)
        chain_M = t["chain_M"].repeat(B, 1)
        chain_M_pos = t["chain_M_pos"].repeat(B, 1)
        residue_idx = t["residue_idx"].repeat(B, 1)
        chain_enc = t["chain_encoding_all"].repeat(B, 1)
        tot = np.zeros(B)
        for d in range(n_draw):
            randn = torch.randn((B, L_full), device=device,
                                generator=torch.Generator(device=device).manual_seed(d))
            log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_enc,
                              randn)
            lp = torch.gather(log_probs, 2, S.unsqueeze(-1)).squeeze(-1)  # (B, L_full)
            sel = lp[:, pos]
            tot += sel.mean(dim=1).detach().cpu().numpy()
        out[start : start + B] = tot / n_draw
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--structures", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--model-name", default="v_48_020")
    ap.add_argument("--n-draw-marginal", type=int, default=8)
    ap.add_argument("--n-draw-ar", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-ar", action="store_true",
                    help="only the cheap backbone marginals")
    ap.add_argument("--limit-targets", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true",
                    help="keep features already in --json and skip those targets, so a "
                         "time limit means resubmitting rather than restarting")
    args = ap.parse_args()

    device = torch.device(args.device)
    payload = json.loads(args.dataset.read_text())
    structs = json.loads(args.structures.read_text())["targets"]

    rows = [r for r in payload["rows"] if r.get("original_sequence")]
    by_target: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_target[r["parent_id"]].append(r)

    model, num_edges, noise = load_model(args.model_name, device)
    print(f"model     ProteinMPNN {args.model_name} (k={num_edges}, noise={noise})")
    print(f"targets   {len(by_target)}   device {args.device}")
    print(f"draws     marginal={args.n_draw_marginal} ar={args.n_draw_ar}"
          f"{'  (AR SKIPPED)' if args.skip_ar else ''}\n")

    feats: dict[str, dict] = {}
    qa = {"ok": [], "skipped": {}}
    done: set[str] = set()
    if args.resume and args.json.exists():
        prev = json.loads(args.json.read_text())
        feats = prev.get("features", {})
        qa = prev.get("qa", qa)
        done = set(qa.get("ok", []))
        print(f"resume    {len(feats)} rows over {len(done)} targets already on disk\n")
    targets = sorted(by_target)
    if args.limit_targets:
        targets = targets[: args.limit_targets]

    def payload() -> dict:
        return {
            "dataset": str(args.dataset),
            "structures": str(args.structures),
            "model": args.model_name,
            "n_draw_marginal": args.n_draw_marginal,
            "n_draw_ar": args.n_draw_ar,
            "skip_ar": bool(args.skip_ar),
            "feature_names": ["mpnn_bb_ll", "mpnn_bb_ll_mean", "mpnn_bb_ll_delta",
                              "mpnn_bb_worst_sub", "mpnn_ar_ll", "mpnn_ar_ll_delta"],
            "units": "natural-log likelihood (nats), NOT an energy",
            "caveats": ["backbone atoms only (N, CA, C, O)",
                        "cyclic-blind: linear clipped relative positional encoding",
                        "scored on the REFERENCE backbone, not the candidate's own"],
            "n_targets_ok": len(qa["ok"]),
            "qa": qa,
            "features": feats,
        }

    def flush() -> None:
        """Written after every target, so a time limit costs one target, not the run."""
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload(), indent=1))

    for ti, tname in enumerate(targets):
        if tname in done:
            print(f"[{ti+1:2d}/{len(targets)}] {tname:24s} already done, skipping")
            continue
        s = structs.get(tname)
        if not s or not s.get("exact"):
            qa["skipped"][tname] = "no exact peptide-chain match in the manifest"
            print(f"[{ti+1:2d}/{len(targets)}] {tname:24s} SKIP (no exact structure)")
            continue
        trows = by_target[tname]
        ref = trows[0]["original_sequence"]
        try:
            t = featurize_target(s["pdb"], s["peptide_chain"], s["receptor_chains"],
                                 device)
        except Exception as exc:  # a parse failure is per-target, not fatal
            qa["skipped"][tname] = f"featurize failed: {exc}"
            print(f"[{ti+1:2d}/{len(targets)}] {tname:24s} SKIP ({exc})")
            continue

        idx = design_positions(t)
        decoded = decode_seq(t["S"][0].cpu().numpy(), idx)
        # Self-check: the design mask must cover exactly the reference peptide. If it does
        # not, every substitution would land at the wrong position, so this target is
        # dropped rather than scored.
        if decoded != ref:
            qa["skipped"][tname] = (f"design mask decodes to {decoded!r} but the "
                                    f"reference is {ref!r}")
            print(f"[{ti+1:2d}/{len(targets)}] {tname:24s} SKIP (mask {decoded} != ref {ref})")
            continue

        marg = backbone_marginals(model, t, idx, args.n_draw_marginal, device)  # (L,21)
        seqs = [r["sequence"] for r in trows]
        bb = np.array([
            sum(marg[i, AA_TO_IDX.get(c, 20)] for i, c in enumerate(sq)) for sq in seqs
        ])
        bb_ref = sum(marg[i, AA_TO_IDX.get(c, 20)] for i, c in enumerate(ref))

        if args.skip_ar:
            ar = np.full(len(seqs), np.nan)
            ar_ref = float("nan")
        else:
            ar = autoregressive_ll(model, t, idx, seqs, args.n_draw_ar, device,
                                   args.batch_size)
            ar_ref = float(autoregressive_ll(model, t, idx, [ref], args.n_draw_ar,
                                             device, 1)[0])

        L = len(ref)
        for r, sq, b, a in zip(trows, seqs, bb, ar):
            # Worst single substituted position, which a sum can hide.
            worst = min(
                (marg[i, AA_TO_IDX.get(sq[i], 20)] for i in range(L) if sq[i] != ref[i]),
                default=0.0)
            feats[f"{tname}|{sq}"] = {
                "mpnn_bb_ll": float(b),
                "mpnn_bb_ll_mean": float(b / L),
                "mpnn_bb_ll_delta": float(b - bb_ref),
                "mpnn_bb_worst_sub": float(worst),
                "mpnn_ar_ll": float(a),
                "mpnn_ar_ll_delta": float(a - ar_ref) if a == a else float("nan"),
            }
        qa["ok"].append(tname)
        flush()
        print(f"[{ti+1:2d}/{len(targets)}] {tname:24s} n={len(trows):4d} L={L:2d} "
              f"bb_ref={bb_ref:8.2f} ar_ref={ar_ref:7.3f}")

    flush()
    print(f"\ntargets scored {len(qa['ok'])}/{len(targets)}   rows {len(feats)}")
    if qa["skipped"]:
        print(f"skipped: {json.dumps(qa['skipped'], indent=1)[:600]}")
    print(f"wrote     {args.json}")


if __name__ == "__main__":
    main()
