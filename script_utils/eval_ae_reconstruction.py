#!/usr/bin/env python3
"""Standalone AE reconstruction diagnostic CLI (Part 5).

Compares any AE checkpoint (PC pretrained, CPSea-finetuned, or a run's own AE) via the same
`run_ae_reconstruction_eval` used by `Proteina.log_unified_validation_metrics`, so aggregate
metrics from two different `--ae_ckpt` runs are directly comparable (same keys, same math),
including cyclic-geometry metrics (`cyc_*`, `mainchain_cn_*`, `disulfide_*`, `isopeptide_*`).

Outputs:
  * aggregate metrics dict (mean over all processed batches), printed and written to JSON, and
    optionally logged to W&B (`--wandb_run_name`).
  * a per-sample CSV and JSONL with `sample_id`, `length`, `cyclization_type`, and every
    reconstruction metric for that individual sample.

Usage
-----
    source env.sh
    python script_utils/eval_ae_reconstruction.py \
        --config example/training_cpsea_peptide_smoke \
        --ae_ckpt /path/to/pc_pretrained_ae.ckpt \
        --split val --num_batches 20 --tag pc_pretrained \
        --wandb_run_name ae_recon_pc_pretrained

    python script_utils/eval_ae_reconstruction.py \
        --config example/training_cpsea_peptide_smoke \
        --ae_ckpt /path/to/cpsea_finetuned_ae.ckpt \
        --split val --num_batches 20 --tag cpsea_finetuned \
        --wandb_run_name ae_recon_cpsea_finetuned
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

CA_IDX = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="example/training_cpsea_peptide_smoke",
                   help="Hydra config name to compose (defines dataset transforms + AE config).")
    p.add_argument("--ae_ckpt", default=None,
                   help="Path to the AE checkpoint to evaluate; overrides cfg.autoencoder_ckpt_path.")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--num_batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--sample_posterior", action="store_true",
                   help="Decode from a sampled posterior z instead of the posterior mean.")
    p.add_argument("--wandb_run_name", default=None, help="If set, logs aggregate metrics to W&B.")
    p.add_argument("--tag", default="current", choices=["pc_pretrained", "cpsea_finetuned", "current"],
                   help="Label describing which AE checkpoint this run evaluates.")
    p.add_argument("--out", default=None,
                   help="Output dir for JSON/CSV/JSONL (default: $PROTEINA_ZFS_PATH/training_runs/diagnostics/ae_recon_eval).")
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


def slice_batch(batch: dict, idx: int) -> dict:
    """Slices a single sample out of a batch, keeping the leading batch dim of size 1
    (so all the shared batch-shaped metric functions still work unmodified)."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 1:
            out[k] = v[idx : idx + 1]
        else:
            out[k] = v
    return out


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import proteinfoundation.patches.atomworks_patches  # noqa: F401  (required for atomize_token)
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from proteinfoundation.eval.ae_reconstruction_eval import run_ae_reconstruction_eval
    from proteinfoundation.eval.cyclic_reconstruction_metrics import extract_cyclization_metadata
    from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
    from proteinfoundation.train import load_data_module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_ae_recon] device={device}")

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=["+single=true", "+nolog=true"])

    ae_ckpt = args.ae_ckpt or cfg.get("autoencoder_ckpt_path", None)
    if ae_ckpt is None:
        raise SystemExit("No AE checkpoint provided: pass --ae_ckpt or set cfg.autoencoder_ckpt_path.")
    print(f"[eval_ae_recon] loading AE from {ae_ckpt} (tag={args.tag})")
    ae = AutoEncoder.load_from_checkpoint(ae_ckpt, map_location=device)
    ae = ae.to(device).eval()
    for prm in ae.parameters():
        prm.requires_grad_(False)

    _, dm = load_data_module(cfg, is_cluster_run=False)
    dm.num_workers = args.num_workers
    dm.batch_size = args.batch_size
    dm.pin_memory = False
    dm.setup("fit")
    loader = dm.val_dataloader() if args.split == "val" else dm.train_dataloader()

    out_dir = Path(args.out) if args.out else Path(
        os.environ.get("PROTEINA_ZFS_PATH", str(REPO))) / "training_runs" / "diagnostics" / "ae_recon_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"ae_recon_eval_{args.tag}_{args.split}.csv"
    jsonl_path = out_dir / f"ae_recon_eval_{args.tag}_{args.split}.jsonl"
    json_path = out_dir / f"ae_recon_eval_{args.tag}_{args.split}.json"

    aggregate_metrics: dict[str, list[float]] = defaultdict(list)
    per_sample_rows: list[dict] = []
    sample_counter = 0
    n_done = 0

    with torch.no_grad():
        for batch in loader:
            if n_done >= args.num_batches:
                break
            batch = move_to_device(batch, device)

            batch_metrics = run_ae_reconstruction_eval(
                ae, batch, prefix="val/ae_recon", sample_posterior=args.sample_posterior
            )
            for k, v in batch_metrics.items():
                aggregate_metrics[k].append(v)

            mask = batch.get("mask", None)
            if mask is None:
                mask = batch["coord_mask"][..., CA_IDX].bool()
            cyc_meta = extract_cyclization_metadata(batch)
            b = mask.shape[0]
            for i in range(b):
                length = int(mask[i].sum().item())
                if length == 0:
                    continue
                cyc_type = int(cyc_meta["type"][i].item()) if cyc_meta is not None else -1
                sample_metrics = run_ae_reconstruction_eval(
                    ae, slice_batch(batch, i), prefix="ae_recon", sample_posterior=args.sample_posterior
                )
                row = {
                    "sample_id": sample_counter,
                    "length": length,
                    "cyclization_type": cyc_type,
                    "tag": args.tag,
                    **sample_metrics,
                }
                per_sample_rows.append(row)
                sample_counter += 1

            n_done += 1
            if n_done % 5 == 0:
                print(f"[eval_ae_recon] processed {n_done}/{args.num_batches} batches "
                      f"({sample_counter} samples)")

    if not aggregate_metrics:
        raise SystemExit("No batches processed; check dataloader / split / --num_batches.")

    aggregate = {k: float(np.nanmean(v)) for k, v in aggregate_metrics.items()}

    with open(json_path, "w") as f:
        json.dump(
            {
                "config": {
                    "config_name": args.config,
                    "ae_ckpt": str(ae_ckpt),
                    "tag": args.tag,
                    "split": args.split,
                    "num_batches": n_done,
                    "num_samples": sample_counter,
                    "sample_posterior": args.sample_posterior,
                },
                "aggregate_metrics": aggregate,
            },
            f,
            indent=2,
        )
    print(f"[eval_ae_recon] wrote {json_path}")

    if per_sample_rows:
        fieldnames = list(per_sample_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample_rows)
        print(f"[eval_ae_recon] wrote {csv_path}")

        with open(jsonl_path, "w") as f:
            for row in per_sample_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[eval_ae_recon] wrote {jsonl_path}")

    print("\n================ AE RECONSTRUCTION EVAL ================")
    print(f"tag={args.tag}  ae_ckpt={ae_ckpt}  split={args.split}  "
          f"batches={n_done}  samples={sample_counter}")
    for k in sorted(aggregate):
        print(f"  {k} = {aggregate[k]:.4f}")

    if args.wandb_run_name:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "proteina-complexa"),
            name=args.wandb_run_name,
            config={"tag": args.tag, "ae_ckpt": str(ae_ckpt), "split": args.split},
        )
        wandb.log(aggregate)
        run.finish()
        print(f"[eval_ae_recon] logged aggregate metrics to W&B run '{args.wandb_run_name}'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
