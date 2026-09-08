"""AE round-trip go/no-go: does the CPSea autoencoder faithfully encode LINEAR peptides?

Why this is the first experiment
--------------------------------
Every latent-space editing method under consideration (SDEdit-style partial noising,
D-Flow, DDIB, FlowEdit) begins by encoding the input linear peptide with the CPSea
autoencoder. That AE was finetuned on cyclic peptides. If it cannot reconstruct a linear
bound peptide, every one of those methods is dead at step 0 -- and this test costs minutes
of GPU instead of weeks.

What it measures
----------------
`run_ae_reconstruction_eval` (proteinfoundation.eval.ae_reconstruction_eval) computes
D(x_ca_gt, E(x_gt)) vs ground truth: all-atom coordinate error plus sequence recovery,
decoding from the deterministic posterior mean and the native Ca trace.

An absolute number cannot answer the go/no-go question -- 0.4 A means nothing on its own.
So this script is run on BOTH arms through the identical loader and transform stack:

    lnr_linear_crystal   the staged LNR benchmark (scripts/build_lnr_metadata.py)
    cpsea_cyclic_val     the CPSea validation split, in-distribution reference

The comparison is the result; the absolute value is not.

Determinism: GlobalRotationTransform (a training augmentation) is dropped by default so
both arms are scored in the same fixed frame and repeat runs agree. --keep-rotation
restores it.

Resumable: rows are appended to the JSONL and already-scored example_ids are skipped, so
hitting a time limit means resubmitting, not restarting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import OmegaConf, open_dict

from proteinfoundation.datasets.structure_data import structure_collate_fn
from proteinfoundation.eval.ae_reconstruction_eval import run_ae_reconstruction_eval
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.train import _resolve_datamodule_config

METRIC_PREFIX = "ae"


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def build_dataset(config_name: str, metadata_file: str, keep_rotation: bool):
    """Instantiates the CPSea val dataset with `metadata_file` swapped in.

    Uses the real training config so the transform stack is byte-identical to what the AE
    was trained against -- the whole point of staging LNR into CPSea's PDB convention.
    """
    with hydra.initialize("../configs", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=config_name)

    # CPSea configs split the datamodule across `dataset.unified.datamodule` (the real
    # definition: _target_, transforms, metadata paths) and `dataset.datamodule` (per-run
    # overrides). Reuse training's own resolver so this cannot drift from what training does.
    dm_cfg = _resolve_datamodule_config(cfg.dataset)
    if dm_cfg is None:
        raise SystemExit(f"FATAL: config {config_name!r} has no resolvable datamodule config.")
    dm_cfg = OmegaConf.create(OmegaConf.to_container(dm_cfg, resolve=True))
    with open_dict(dm_cfg):
        dm_cfg.val_metadata_file = metadata_file
        dm_cfg.batch_size = 1
        dm_cfg.num_workers = 0
        if not keep_rotation:
            dm_cfg.atom37_transforms = [
                t for t in dm_cfg.atom37_transforms
                if "GlobalRotationTransform" not in str(t.get("_target_", ""))
            ]

    datamodule = hydra.utils.instantiate(dm_cfg)
    datamodule.setup("fit")
    ae_ckpt = cfg.get("autoencoder_ckpt_path", None)
    return datamodule.val_dataset, ae_ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="example/training_cpsea_peptide_smoke",
                    help="Hydra config supplying the CPSea transform stack + AE checkpoint path.")
    ap.add_argument("--metadata", required=True, help="Parquet with the CPSea metadata schema.")
    ap.add_argument("--arm", required=True, help="Label written into every row, e.g. lnr_linear_crystal.")
    ap.add_argument("--out", required=True, help="JSONL output (appended; resumable).")
    ap.add_argument("--ae-ckpt", default=None, help="Overrides the config's autoencoder_ckpt_path.")
    ap.add_argument("--limit", type=int, default=0, help="Score at most N examples (0 = all). Smoke runs.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep-rotation", action="store_true",
                    help="Keep GlobalRotationTransform (random). Off by default for determinism.")
    args = ap.parse_args()

    L.seed_everything(args.seed, workers=True)
    torch.use_deterministic_algorithms(False)

    dataset, cfg_ae_ckpt = build_dataset(args.config_name, args.metadata, args.keep_rotation)
    ae_ckpt = args.ae_ckpt or os.environ.get("CPSEA_AE_CKPT_PATH") or cfg_ae_ckpt
    if not ae_ckpt or not Path(ae_ckpt).is_file():
        raise SystemExit(f"FATAL: autoencoder checkpoint not found: {ae_ckpt!r}")
    print(f"[arm={args.arm}] AE checkpoint: {ae_ckpt}", flush=True)
    print(f"[arm={args.arm}] {len(dataset)} examples from {args.metadata}", flush=True)

    autoencoder = AutoEncoder.load_from_checkpoint(ae_ckpt, map_location="cpu")
    autoencoder.eval().to(args.device)
    for p in autoencoder.parameters():
        p.requires_grad = False

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                # A torn line means the file cannot be trusted to say what is finished.
                raise SystemExit(
                    f"FATAL: corrupt row in {out_path}. Delete it and rerun rather than "
                    "resuming onto an unreadable record."
                )
    if done:
        print(f"[arm={args.arm}] resuming: {len(done)} already scored", flush=True)

    meta = dataset.metadata
    n_ok = n_fail = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for idx in range(len(dataset)):
            row = meta.iloc[idx]
            example_id = str(row["example_id"])
            if example_id in done:
                continue
            if args.limit and n_ok >= args.limit:
                break

            try:
                sample = dataset[idx]
                if sample is None:
                    raise RuntimeError("dataset returned None (load or transform failed)")
                batch = to_device(structure_collate_fn([sample]), args.device)
                with torch.no_grad():
                    metrics = run_ae_reconstruction_eval(
                        autoencoder, batch, prefix=METRIC_PREFIX, sample_posterior=False
                    )
            except Exception as exc:  # noqa: BLE001 - one bad structure must not kill the sweep
                n_fail += 1
                fh.write(json.dumps({"example_id": example_id, "arm": args.arm,
                                     "status": "failed", "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                fh.flush()
                print(f"  FAIL {example_id}: {type(exc).__name__}: {exc}", flush=True)
                continue

            record = {"example_id": example_id, "arm": args.arm, "status": "ok"}
            for col in ("peptide_length", "receptor_length", "cyclization_type", "cluster_id",
                        "nc_gap_angstrom", "best_ss_cb_dist", "best_iso_cb_dist", "n_cys"):
                if col in meta.columns:
                    value = row[col]
                    record[col] = value.item() if hasattr(value, "item") else value
            record.update({k: float(v) for k, v in metrics.items()})
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"  {n_ok} scored ({time.time() - t0:.0f}s)", flush=True)

    print(f"[arm={args.arm}] done: {n_ok} scored, {n_fail} failed, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
