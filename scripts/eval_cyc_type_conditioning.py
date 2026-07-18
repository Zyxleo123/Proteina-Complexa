#!/usr/bin/env python3
"""Does the model actually honor a requested cyclization type?

Loads a type-conditioned checkpoint, generates binders for CPSea val receptors while
requesting each cyclization type in turn, and measures whether the request was honored.

The headline number is **satisfaction rate**: the fraction of generated binders whose
*decoded sequence* can chemically support the requested linkage (two CYS for a disulfide,
LYS + acid/amide for an isopeptide). That is the whole point of conditioning the denoiser
rather than only the head -- a cyclization-blind model would happily emit a binder with no
cysteine while being asked for a disulfide.

Two controls make the number interpretable:
  - `unspecified`: the unconditional baseline. Its CYS-pair rate is the *background* rate.
    If requesting `disulfide` does not beat it, the conditioning is not doing anything.
  - `mainchain` is trivially satisfiable (any binder of length >= 2 has termini), so it
    should sit at ~100% and is a sanity check, not evidence.

Usage:
    python scripts/eval_cyc_type_conditioning.py --ckpt <path/to/last-EMA.ckpt> [--n-batches 8]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from omegaconf import OmegaConf

from proteinfoundation.cyclization.constants import (
    AA_ASN,
    AA_ASP,
    AA_CYS,
    AA_GLN,
    AA_GLU,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
    UNSPECIFIED,
)
from proteinfoundation.cyclization.inference import predict_cyclization_from_clean

REQUESTS = [("unspecified", UNSPECIFIED), ("mainchain", MAINCHAIN), ("disulfide", DISULFIDE), ("isopeptide", ISOPEPTIDE)]


def sequence_supports(aa: torch.Tensor, mask: torch.Tensor, type_idx: int, allow_asn_gln: bool = True) -> torch.Tensor:
    """[B] bool: can each decoded sequence chemically support `type_idx`?

    Deliberately recomputed here from the decoded AA sequence rather than read off the
    model's own validity mask, so this is an independent check of the model's output and not
    a restatement of its own bookkeeping.
    """
    m = mask.bool()
    n_cys = ((aa == AA_CYS) & m).sum(-1)
    n_lys = ((aa == AA_LYS) & m).sum(-1)
    acid = (aa == AA_ASP) | (aa == AA_GLU)
    if allow_asn_gln:
        acid = acid | (aa == AA_ASN) | (aa == AA_GLN)
    n_acid = (acid & m).sum(-1)

    if type_idx == DISULFIDE:
        return n_cys >= 2
    if type_idx == ISOPEPTIDE:
        return (n_lys >= 1) & (n_acid >= 1)
    if type_idx == MAINCHAIN:
        return m.sum(-1) >= 2
    # UNSPECIFIED: satisfied if ANY linkage is chemically possible.
    return (m.sum(-1) >= 2) | (n_cys >= 2) | ((n_lys >= 1) & (n_acid >= 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Trained type-conditioned checkpoint (last-EMA.ckpt).")
    ap.add_argument("--n-batches", type=int, default=8, help="Val batches per requested type.")
    ap.add_argument("--nsteps", type=int, default=200, help="Flow integration steps.")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import os

    from proteinfoundation.proteina import Proteina

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_exp = OmegaConf.create(ckpt["hyper_parameters"]["cfg_exp"])
    if not cfg_exp.get("cyclization", {}).get("type_conditioning", False):
        raise SystemExit(
            "This checkpoint was NOT trained with cyclization.type_conditioning=true. "
            "Requesting a type from it is meaningless -- train with "
            "configs/example/training_cpsea_peptide_cyc_typecond.yaml first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Proteina.load_from_checkpoint(args.ckpt, map_location=device).eval().to(device)

    # Minimal inference config: single-pass generation, no search/reward.
    model.configure_inference(
        OmegaConf.create({"args": {"self_cond": True, "nsteps": args.nsteps, "guidance_w": 1.0, "ag_ratio": 0.0},
                          "model": {"bb_ca": {}, "local_latents": {}}, "n_recycle": 0}),
        nn_ag=None,
    )

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        data_cfg = compose(config_name="example/training_cpsea_peptide_cyc_typecond", overrides=["+single=true"])
    dm = instantiate(data_cfg.dataset.unified.datamodule)
    dm.setup("fit")
    batches = [b for _, b in zip(range(args.n_batches), dm.val_dataloader())]

    rows = []
    for name, type_idx in REQUESTS:
        stats: dict[str, list] = defaultdict(list)
        for batch in batches:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            bs = batch["mask"].shape[0]
            # Overwrite any dataset-supplied label: we are asking, not reading.
            batch["cyclization_type_cond"] = torch.full((bs,), type_idx, dtype=torch.long, device=device)
            model.cyclization_target_type = None if type_idx == UNSPECIFIED else type_idx

            with torch.no_grad():
                gen = model.generate(batch)
                decoded = model.autoencoder.decode(
                    z_latent=gen["local_latents"], ca_coors_nm=gen["bb_ca"], mask=batch["mask"]
                )
                # The model's own verdict, from cyclization/inference.py.
                pred = predict_cyclization_from_clean(
                    model,
                    ca=gen["bb_ca"],
                    z_latent=gen["local_latents"],
                    mask=batch["mask"],
                    cond_type=batch["cyclization_type_cond"],
                )

            aa = decoded["residue_type"]
            m = batch["mask"]
            stats["supported"] += sequence_supports(aa, m, type_idx).float().tolist()
            stats["model_satisfied"] += pred["cyclization_type_satisfied"].float().tolist()
            stats["n_cys"] += ((aa == AA_CYS) & m.bool()).sum(-1).float().tolist()
            stats["has_2cys"] += (((aa == AA_CYS) & m.bool()).sum(-1) >= 2).float().tolist()

        n = len(stats["supported"])
        rows.append(
            (name, n,
             sum(stats["supported"]) / n,
             sum(stats["model_satisfied"]) / n,
             sum(stats["has_2cys"]) / n,
             sum(stats["n_cys"]) / n)
        )

    print("\n## Did the model honor the requested cyclization type?\n")
    print("| requested    |   n | seq supports | model says sat. | >=2 CYS | mean CYS |")
    print("|--------------|----:|-------------:|----------------:|--------:|---------:|")
    for name, n, sat, model_sat, cys2, ncys in rows:
        print(f"| {name:<12} | {n:>3} | {sat:>11.1%} | {model_sat:>14.1%} | {cys2:>6.1%} | {ncys:>8.2f} |")

    base = {r[0]: r[4] for r in rows}
    print("\nRead this as: `disulfide` must beat `unspecified` on the '>=2 CYS' column.")
    print(f"  unspecified (background): {base['unspecified']:.1%}")
    print(f"  disulfide   (requested) : {base['disulfide']:.1%}")
    if base["disulfide"] > base["unspecified"] + 0.05:
        print("  => Conditioning is steering the SEQUENCE toward the requested linkage.")
    else:
        print("  => NO effect: the denoiser is ignoring the type. Check that cyclization_type_emb")
        print("     is in nn.feats_cond_seq and that the model trained long enough.")


if __name__ == "__main__":
    main()
