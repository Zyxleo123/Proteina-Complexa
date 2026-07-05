#!/usr/bin/env python3
"""Computes per-latent-dimension mean/std stats for the optional `latent_normalization` feature.

Motivation
----------
Even if aggregate latent stats look roughly Gaussian, the frozen AE's KL can be high, meaning
the per-channel scale/offset of the latent space may not be flow-friendly. This script measures
the channel-wise mean and std of the AE's local_latents over a real dataset (unmasked residues
only) and saves them so they can be plugged into a training config as:

    latent_normalization:
      enabled: true
      stats_path: /path/to/latent_stats.pt

Training then uses `z_flow = (z_ae - mean) / std`, and decoding inverts this with
`z_ae = z_flow * std + mean` (see `proteinfoundation.utils.sample_utils`). This script itself
never trains anything; it only loads a frozen AE checkpoint and a dataset, and writes a stats
file (default: torch.save'd dict with "mean"/"std" tensors of shape [latent_dim]).

Usage
-----
    source env.sh
    python script_utils/compute_latent_stats.py \\
        --config-name example/training_cpsea_peptide_smoke \\
        --split train --num-batches 200 --batch-size 16 \\
        --target sample \\
        --out <AE_STORE>/latent_stats_<data>_<target>.pt

Use `--target mean` to compute stats over the encoder posterior mean instead of a sampled
latent (matches whatever `local_latent_target` you plan to train with, though the two are
close in practice since only the *scale* differs by the posterior variance).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

CA_IDX = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="example/training_cpsea_peptide_smoke",
                   help="Hydra config to compose (defines dataset transforms + AE ckpt path).")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--num-batches", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--target", choices=["sample", "mean"], default="sample",
                   help="Compute stats over the sampled latent (matches local_latent_target=sample) "
                        "or the posterior mean (matches local_latent_target=mean).")
    p.add_argument("--metadata", default=None, help="Override train metadata parquet.")
    p.add_argument("--val-metadata", default=None, help="Override val metadata parquet.")
    p.add_argument("--out", required=True, help="Output path for the stats file (torch.save dict).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    torch.manual_seed(args.seed)

    import proteinfoundation.patches.atomworks_patches  # noqa: F401  (required for atomize_token)
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
    from proteinfoundation.train import load_data_module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[latent_stats] device={device}")

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config_name, overrides=["+single=true", "+nolog=true"])

    ae_ckpt = cfg.get("autoencoder_ckpt_path", None)
    if ae_ckpt is None:
        raise SystemExit("cfg.autoencoder_ckpt_path missing; this config does not use local_latents.")
    print(f"[latent_stats] loading AE from {ae_ckpt}")
    ae = AutoEncoder.load_from_checkpoint(ae_ckpt)
    ae = ae.to(device).eval()
    for prm in ae.parameters():
        prm.requires_grad_(False)
    latent_dim = ae.latent_dim
    print(f"[latent_stats] latent_dim={latent_dim}  target={args.target}")

    _, dm = load_data_module(cfg, is_cluster_run=False)
    dm.num_workers = args.num_workers
    dm.batch_size = args.batch_size
    dm.pin_memory = False
    if args.metadata:
        dm.metadata_file = args.metadata
    if args.val_metadata:
        dm.val_metadata_file = args.val_metadata
    dm.setup("fit")
    loader = dm.val_dataloader() if args.split == "val" else dm.train_dataloader()

    # Welford-style running moments over unmasked residues, so we don't need to hold every
    # latent in memory for large datasets.
    count = 0
    running_sum = torch.zeros(latent_dim, dtype=torch.float64)
    running_sumsq = torch.zeros(latent_dim, dtype=torch.float64)

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if n_done >= args.num_batches:
                break
            batch = move_to_device(batch, device)

            coord_mask = batch["coord_mask"].bool()  # [b, n, 37]
            res_mask = coord_mask[..., CA_IDX]  # [b, n], unmasked (real) residues only
            if "mask" not in batch:
                batch["mask"] = res_mask

            enc = ae.encode(batch)
            z = enc["z_latent"] if args.target == "sample" else enc["mean"]  # [b, n, d]
            z_valid = z[res_mask].double().cpu()  # [num_valid_residues, d]

            count += z_valid.shape[0]
            running_sum += z_valid.sum(dim=0)
            running_sumsq += (z_valid**2).sum(dim=0)

            n_done += 1
            if n_done % 20 == 0:
                print(f"[latent_stats] processed {n_done}/{args.num_batches} batches, {count} residues so far")

    if count == 0:
        raise SystemExit("No residues processed; check dataloader / split.")

    mean = (running_sum / count).float()  # [d]
    var = (running_sumsq / count).float() - mean**2  # [d], E[z^2] - E[z]^2
    var = var.clamp(min=1e-12)
    std = var.sqrt()

    print("\n================ LATENT STATS ================")
    print(f"n_residues={count}  latent_dim={latent_dim}  target={args.target}  split={args.split}")
    print(f"mean: {mean.tolist()}")
    print(f"std:  {std.tolist()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "std": std,
            "config": {
                "config_name": args.config_name,
                "split": args.split,
                "target": args.target,
                "num_batches": n_done,
                "n_residues": count,
                "ae_ckpt": str(ae_ckpt),
                "latent_dim": latent_dim,
            },
        },
        out_path,
    )
    print(f"\n[latent_stats] wrote {out_path}")
    print("Use with: latent_normalization.enabled=true latent_normalization.stats_path=" + str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
