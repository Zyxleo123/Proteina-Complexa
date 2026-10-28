#!/usr/bin/env python3
"""Does a frozen ESM-2 peptide encoder beat amino-acid composition at ranking mean dG?

This probes ONE component of a proposed mean-score scorer -- the peptide sequence encoder
-- in isolation, before any of the rest is built. It embeds each cyclic peptide with
frozen ESM-2, mean-pools over residues, and fits the same ridge under the same
target-held-out CV and the same within-target metric as `cpseq_meanscore_baseline.py`.
If a 1280-dim protein language model embedding does not beat a 49-feature composition
vector here, the encoder is not what is limiting the scorer and a cross-attention module
on top of it is premature.

Two things this probe cannot do, and does not claim:

1. **It has no target information.** Every sequence is embedded alone, so the model sees
   the peptide and not the pocket it binds. That is the point: it measures the peptide
   encoder's own contribution, and the gap it leaves is the budget for a target encoder.

2. **It is frozen and mean-pooled.** A fine-tuned or attention-pooled encoder could do
   better. This is a lower bound on ESM-2, not a verdict on it.

The peptide is cyclic and ESM-2 was trained on linear chains, so the embedding does not
know the backbone closes. `--circular-pad` repeats the first `k` residues at the end to
give the model the wrap-around context; it is reported alongside the plain embedding.

Energies are Rosetta ref2015 REU and are NOT measured binding free energies.

Usage:
    python script_utils/cpseq_meanscore_esm_probe.py \
        --dataset /data/user_data/yixiz/cpseq/meanscore/dataset_v3.json \
        --json /data/user_data/yixiz/cpseq/meanscore/esm_probe_v3.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpseq_meanscore_baseline import label_ceiling, run_cv, seq_features  # noqa: E402

DEFAULT_ESM = "facebook/esm2_t33_650M_UR50D"


def embed(sequences: list[str], model_name: str, cache_dir: Path, circular_pad: int,
          device: str, batch_size: int = 64) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir)).to(device).eval()

    prepared = [s + s[:circular_pad] if circular_pad else s for s in sequences]
    out = []
    with torch.no_grad():
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            # Drop BOS/EOS: they carry no residue and shift the pooled mean by 1/L on a
            # peptide this short.
            mask[:, 0] = 0.0
            for row, seq in enumerate(chunk):
                mask[row, len(seq) + 1] = 0.0
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--esm", default=DEFAULT_ESM)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/home/yixiz/Proteina-Complexa/community_models/ckpts/ESM2"))
    ap.add_argument("--circular-pad", type=int, default=3,
                    help="residues of wrap-around context appended; 0 disables")
    ap.add_argument("--include-reference", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-seq", type=int, default=4)
    ap.add_argument("--embed-cache", type=Path,
                    help="npz to reuse embeddings from, or write them to")
    ap.add_argument("--pca", default="16,64,256",
                    help="also score ESM reduced to this many PCs; '' disables")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    payload = json.loads(args.dataset.read_text())
    rows = payload["rows"]
    if not args.include_reference:
        rows = [r for r in rows if r.get("arm") != "reference"]
    y = [r["label"] for r in rows]
    parents = [r["parent_id"] for r in rows]
    sequences = [r["sequence"] for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dataset   {args.dataset}")
    print(f"rows      {len(rows)} sequences, {len(set(parents))} targets")
    print(f"encoder   {args.esm} (frozen, mean-pooled) on {device}")

    _, ceiling, nceil = label_ceiling(rows, args.min_seq)
    print(f"ceiling   within-target rho {ceiling:.3f} ({nceil} targets)\n")

    names = sorted(seq_features(rows[0]))
    comp = [[seq_features(r)[n] for n in names] for r in rows]

    comp_arr = np.asarray(comp, dtype=float)
    cached = {}
    if args.embed_cache and args.embed_cache.exists():
        with np.load(args.embed_cache) as handle:
            cached = {k: handle[k] for k in handle.files}
        if len(cached.get("plain", [])) != len(sequences):
            cached = {}  # row count changed, so the cache is for another table
        else:
            print(f"embeddings reused from {args.embed_cache}")
    if not cached:
        cached["plain"] = embed(sequences, args.esm, args.cache_dir, 0, device)
        if args.circular_pad:
            cached["wrapped"] = embed(sequences, args.esm, args.cache_dir,
                                      args.circular_pad, device)
        if args.embed_cache:
            args.embed_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.embed_cache, **cached)
            print(f"embeddings cached to {args.embed_cache}")

    plain, wrapped = cached["plain"], cached.get("wrapped")
    variants = {"composition (49 feat)": comp_arr,
                f"ESM-2 mean-pool ({plain.shape[1]}d)": plain}
    if wrapped is not None:
        variants[f"ESM-2 + {args.circular_pad}-residue wrap"] = wrapped
        variants["ESM-2 wrap + composition"] = np.hstack([wrapped, comp_arr])
    # Ridge on 1280 dims over ~2600 training rows can lose to 49 features by overfitting
    # alone. Reducing ESM to a few PCs separates "the embedding is uninformative" from
    # "the embedding has too many dimensions for this table". The PCA is unsupervised, so
    # fitting it on all rows leaks no label.
    for k in (int(p) for p in args.pca.split(",") if p.strip()):
        if k >= min(plain.shape):
            continue
        from sklearn.decomposition import PCA
        pcs = PCA(n_components=k, random_state=0).fit_transform(
            (plain - plain.mean(0)) / (plain.std(0) + 1e-8))
        variants[f"ESM-2 -> {k} PCs"] = pcs
        variants[f"ESM-2 {k} PCs + composition"] = np.hstack([pcs, comp_arr])

    results = {}
    print(f"{'features':<34} {'within-tgt rho':>14} {'frac of ceiling':>16} {'global rho':>11}")
    for name, X in variants.items():
        rho, glob, used, _ = run_cv(X, y, parents,
                                    lambda: RidgeCV(alphas=np.logspace(-1, 6, 29)),
                                    args.folds, args.min_seq)
        frac = rho / ceiling if ceiling else float("nan")
        print(f"{name:<34} {rho:>14.3f} {frac:>16.2f} {glob:>11.3f}")
        results[name] = {"within_target_rho": rho, "frac_of_ceiling": frac,
                         "global_rho": glob, "targets_scored": used,
                         "n_features": int(X.shape[1])}

    out = {
        "dataset": str(args.dataset), "esm": args.esm,
        "label": payload["label"], "n_rows": len(rows),
        "n_targets": len(set(parents)), "ceiling_full": ceiling,
        "note": "no target information in any variant; peptide encoder measured alone",
        "results": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwrote     {args.json}")


if __name__ == "__main__":
    main()
