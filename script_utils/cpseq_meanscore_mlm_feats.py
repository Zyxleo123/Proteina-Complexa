#!/usr/bin/env python3
"""Pocket-conditioned sequence features from the project's own fine-tuned PepMLM.

The sequence-side counterpart to `cpseq_meanscore_mpnn_feats.py`. ProteinMPNN scores a
candidate against the reference *backbone*; this scores it against the *pocket sequence*,
using the CPSea unmasking model (PepMLM-650M fine-tuned for 60k steps on 2.44M CPSea rows,
`cpseq_mlm/runs/cpseq_mlm_pocket/checkpoints/hf_step60000`). Two pretrained views of the
same candidate that fail in different ways is worth more than either alone.

Everything about the input format is taken from the training code rather than rebuilt:
`parse_pdb_residues` and `extract_pocket` from `cpseq_build_mlm_corpus`, `CorpusRow` and
`EsmMlmEncoder` from `cpseq_mlm_data`. That matters more here than it sounds. The encoder
lays out `<cls> <pocket...> <eos> <cyc:TYPE> <peptide...> <eos>` with `<gap>` between
non-consecutive pocket runs, and feeding a differently-shaped pocket is not a degraded
feature but a different conditioning than the model ever saw -- the same failure
[[lnr-receptor-segmentation-mismatch]] measured at +22 to +34 points of closure. The
pocket radius is 8.0 A, the corpus default.

Three features, cheapest first
------------------------------
* **`mlm_marg_ll`** -- every peptide position masked at once, so the logits depend only on
  the pocket. Summing log p(candidate_aa_i | pocket) is additive over positions, which
  makes it one forward per *target* and the direct analogue of the ProteinMPNN backbone
  marginals. A masked LM's per-position marginals do not compose into a joint (the reason
  `mlm_propose.py` fills positions one at a time), so this is a scoring convenience, not a
  sequence probability.
* **`mlm_sub_ll`** -- only the substituted positions masked, with the rest of the
  *candidate* as context. One forward per candidate, and it sees intra-peptide context the
  marginal cannot.
* **`mlm_sub_logodds`** -- from the same forward, log p(mutant) - log p(wildtype) summed
  over substituted positions. This is the mutation-effect form, and it cancels the
  position's own baseline preference, which is pure nuisance inside a target.

Honest limits
-------------
The model's measured pocket gain is small: +0.041 accuracy over a positional-marginal
baseline (0.268 vs 0.204 at step 60000), and a pocket-shuffled control still reaches
0.226, so real pocket information is worth ~0.042 accuracy. [[cpseq-mlm-ceiling-verdict]]
put pocket sequence at ~0.20 nats. Expect a weak feature, and judge it by whether it adds
to ProteinMPNN rather than by its own rho.

These are log-likelihoods in nats. They are not energies, and the Rosetta dG they are
meant to predict is itself a ref2015 score, not a measured binding free energy.

Usage:
    python script_utils/cpseq_meanscore_mlm_feats.py \
        --dataset    /zfsauton/scratch/yixiz/cpseq/meanscore/dataset_v3.json \
        --structures /zfsauton/scratch/yixiz/cpseq/meanscore/structures_v3.json \
        --model      /zfsauton/scratch/yixiz/cpseq_mlm/runs/cpseq_mlm_pocket/checkpoints/hf_step60000 \
        --json       /zfsauton/scratch/yixiz/cpseq/meanscore/mlm_feats_v3.json
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
sys.path.insert(0, str(REPO / "script_utils"))
sys.path.insert(0, str(REPO / "src"))

from cpseq_build_mlm_corpus import (  # noqa: E402
    _residue_sequence,
    _THREE_TO_ONE,
    extract_pocket,
    parse_pdb_residues,
)
from cpseq_mlm_data import AA20, CorpusRow, EsmMlmEncoder, add_task_tokens  # noqa: E402

POCKET_RADIUS = 8.0  # the corpus default; see --pocket-radius in cpseq_build_mlm_corpus


def polymer_only(residues: list[dict]) -> list[dict]:
    """Keep canonical amino-acid residues, dropping waters, ions and ligands.

    `parse_pdb_residues` reads ATOM *and* HETATM, which is right for the preprocessed
    CPSea complexes the corpus was built from -- those carry no solvent. The native
    crystal structures used here do: 1elr chain B holds waters, which gave every residue
    an unrecognised code and made `_residue_sequence` return None for the whole chain.

    Filtering is the faithful choice rather than a workaround. The model was conditioned
    on solvent-free pockets, so leaving waters in the receptor would put heavy atoms into
    the 8 A contact set that training never had, and change which residues the pocket
    contains.
    """
    return [r for r in residues if r.get("resname") in _THREE_TO_ONE]


def load_model(model_path: Path, device, dtype: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForMaskedLM.from_pretrained(str(model_path),
                                                 dtype=getattr(torch, dtype))
    # The fine-tune added <gap> and the <cyc:*> tokens. If the saved tokenizer somehow
    # lacks them the encoder raises, which is the right outcome: a missing special token
    # would shift every peptide position by one and quietly change what is being scored.
    try:
        EsmMlmEncoder(tok, pocket_enabled=True)
    except ValueError:
        add_task_tokens(tok, model)
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def forward_logprobs(model, enc, rows_hidden, device, residue_ids, batch_size: int):
    """log-softmax over the 20 residues at each hidden position, per encoded row."""
    out = []
    for start in range(0, len(rows_hidden), batch_size):
        chunk = rows_hidden[start : start + batch_size]
        maxlen = max(len(c["input_ids"]) for c in chunk)
        ids = torch.full((len(chunk), maxlen), enc.pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, c in enumerate(chunk):
            n = len(c["input_ids"])
            ids[i, :n] = torch.tensor(c["input_ids"], dtype=torch.long)
            att[i, :n] = 1
        logits = model(input_ids=ids.to(device),
                       attention_mask=att.to(device)).logits
        # Restrict to the canonical residues before normalising: the head can emit X, B,
        # <mask> or a special token, and leaving those in the partition function makes the
        # numbers depend on vocabulary the task never uses.
        sub = logits[:, :, residue_ids]
        lp = torch.log_softmax(sub.float(), dim=-1).cpu().numpy()
        for i in range(len(chunk)):
            out.append(lp[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--structures", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--pocket-radius", type=float, default=POCKET_RADIUS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--skip-sub", action="store_true",
                    help="only the cheap pocket-only marginals (one forward per target)")
    ap.add_argument("--limit-targets", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from proteinfoundation.cyclization.constants import MAINCHAIN  # noqa: E402

    device = torch.device(args.device)
    payload = json.loads(args.dataset.read_text())
    structs = json.loads(args.structures.read_text())["targets"]

    rows = [r for r in payload["rows"] if r.get("original_sequence")]
    by_target: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_target[r["parent_id"]].append(r)

    model, tok = load_model(args.model, device, args.dtype)
    enc = EsmMlmEncoder(tok, pocket_enabled=True)
    residue_ids = enc.residue_ids()
    aa_pos = {a: i for i, a in enumerate(AA20)}
    print(f"model     {args.model}")
    print(f"device    {args.device}  dtype {args.dtype}  batch {args.batch_size}")
    print(f"targets   {len(by_target)}  pocket_radius {args.pocket_radius} A")
    print(f"mode      {'marginals only' if args.skip_sub else 'marginals + substituted'}\n")

    feats: dict[str, dict] = {}
    qa = {"ok": [], "skipped": {}}
    done: set[str] = set()
    if args.resume and args.json.exists():
        prev = json.loads(args.json.read_text())
        feats = prev.get("features", {})
        qa = prev.get("qa", qa)
        done = set(qa.get("ok", []))
        print(f"resume    {len(feats)} rows over {len(done)} targets already on disk\n")

    def payload_out() -> dict:
        return {
            "dataset": str(args.dataset),
            "structures": str(args.structures),
            "model": str(args.model),
            "pocket_radius": args.pocket_radius,
            "skip_sub": bool(args.skip_sub),
            "feature_names": ["mlm_marg_ll", "mlm_marg_ll_mean", "mlm_marg_ll_delta",
                              "mlm_marg_worst", "mlm_sub_ll", "mlm_sub_logodds"],
            "units": "natural-log likelihood (nats), NOT an energy",
            "caveats": [
                "masked-LM marginals do not compose into a joint sequence probability",
                "pocket is an 8.0 A heavy-atom crop with <gap> markers, as trained",
                "measured pocket gain is small: +0.041 acc over a positional baseline",
            ],
            "n_targets_ok": len(qa["ok"]),
            "qa": qa,
            "features": feats,
        }

    def flush() -> None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload_out(), indent=1))

    targets = sorted(by_target)
    if args.limit_targets:
        targets = targets[: args.limit_targets]

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
        L = len(ref)

        try:
            chains = parse_pdb_residues(s["pdb"])
            binder = polymer_only(chains.get(s["peptide_chain"]) or [])
            receptor = [r for cid, rs in chains.items()
                        if cid != s["peptide_chain"] for r in polymer_only(rs)]
            if not binder or not receptor:
                raise ValueError("missing binder or receptor chain")
            pep_from_struct = _residue_sequence(binder)
            # Same self-check as the MPNN path: if the structure's peptide is not the
            # reference, substitution positions would be scored at the wrong index.
            if pep_from_struct != ref:
                raise ValueError(f"structure peptide {pep_from_struct!r} != reference {ref!r}")
            pocket_seq, pocket_resids = extract_pocket(receptor, binder,
                                                       args.pocket_radius)
            if not pocket_resids:
                raise ValueError("empty pocket")
        except Exception as exc:  # noqa: BLE001 - per-target, not fatal
            qa["skipped"][tname] = f"pocket build failed: {exc}"
            print(f"[{ti+1:2d}/{len(targets)}] {tname:24s} SKIP ({exc})")
            continue

        def row_for(seq: str) -> CorpusRow:
            return CorpusRow(example_id=f"{tname}|{seq}", pep_seq=seq,
                             pocket_seq=pocket_seq, cyc_type=int(MAINCHAIN),
                             cyc_anchor_i=0, cyc_anchor_j=L - 1)

        # --- 1. pocket-only marginals: all positions hidden, so one forward per target
        enc_all = enc.encode(row_for(ref), hidden=list(range(L)))
        if enc_all.get("too_long"):
            qa["skipped"][tname] = f"encoded length {enc_all['length']} > max_length"
            print(f"[{ti+1:2d}/{len(targets)}] {tname:24s} SKIP (too long)")
            continue
        lp_all = forward_logprobs(model, enc, [enc_all], device, residue_ids, 1)[0]
        pep_start = enc_all["pep_start"]
        marg = lp_all[pep_start : pep_start + L]  # (L, 20)

        seqs = [r["sequence"] for r in trows]
        marg_ll = np.array([
            sum(marg[i, aa_pos[c]] for i, c in enumerate(sq) if c in aa_pos)
            for sq in seqs])
        marg_ref = sum(marg[i, aa_pos[c]] for i, c in enumerate(ref) if c in aa_pos)

        # --- 2. substituted positions only, with the candidate's own context
        sub_ll = np.full(len(seqs), np.nan)
        sub_lo = np.full(len(seqs), np.nan)
        if not args.skip_sub:
            encoded, keep, hid_list = [], [], []
            for i, sq in enumerate(seqs):
                hid = [p for p in range(L) if sq[p] != ref[p]]
                if not hid:
                    continue
                e = enc.encode(row_for(sq), hidden=hid)
                if e.get("too_long"):
                    continue
                encoded.append(e)
                keep.append(i)
                hid_list.append(hid)
            lps = forward_logprobs(model, enc, encoded, device, residue_ids,
                                   args.batch_size)
            for i, hid, e, lp in zip(keep, hid_list, encoded, lps):
                ps = e["pep_start"]
                sq = seqs[i]
                tot = sum(lp[ps + p, aa_pos[sq[p]]] for p in hid if sq[p] in aa_pos)
                odds = sum(lp[ps + p, aa_pos[sq[p]]] - lp[ps + p, aa_pos[ref[p]]]
                           for p in hid if sq[p] in aa_pos and ref[p] in aa_pos)
                sub_ll[i] = float(tot)
                sub_lo[i] = float(odds)

        for r, sq, m, sl, so in zip(trows, seqs, marg_ll, sub_ll, sub_lo):
            worst = min((marg[i, aa_pos[sq[i]]] for i in range(L)
                         if sq[i] != ref[i] and sq[i] in aa_pos), default=0.0)
            feats[f"{tname}|{sq}"] = {
                "mlm_marg_ll": float(m),
                "mlm_marg_ll_mean": float(m / L),
                "mlm_marg_ll_delta": float(m - marg_ref),
                "mlm_marg_worst": float(worst),
                "mlm_sub_ll": float(sl),
                "mlm_sub_logodds": float(so),
            }
        qa["ok"].append(tname)
        flush()
        print(f"[{ti+1:2d}/{len(targets)}] {tname:24s} n={len(trows):4d} L={L:2d} "
              f"pocket={len(pocket_resids):3d}res marg_ref={marg_ref:8.2f}")

    flush()
    print(f"\ntargets scored {len(qa['ok'])}/{len(targets)}   rows {len(feats)}")
    if qa["skipped"]:
        print(f"skipped: {json.dumps(qa['skipped'], indent=1)[:800]}")
    print(f"wrote     {args.json}")


if __name__ == "__main__":
    main()
