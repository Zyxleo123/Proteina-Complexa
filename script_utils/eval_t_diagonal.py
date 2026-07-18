#!/usr/bin/env python3
"""Diagonal-vs-square `t` diagnostic: does the model only work when one modality is clean?

WHY THIS EXISTS
---------------
Training and sampling disagree about `t`:

  * TRAINING draws `t` **independently per modality**. `loss.t_distribution.shared_groups: []`
    means the sharing loop in `ProductSpaceFlowMatcher.sample_t` never executes
    (product_space_flow_matcher.py:214), so `t_bb_ca` and `t_local_latents` are independent
    draws. Training therefore covers the whole SQUARE [0,1] x [0,1].

  * SAMPLING indexes every modality off the same step counter
    (`ts[data_mode][step]`, product_space_flow_matcher.py:829), so `t_ca == t_z` at every
    step. Sampling only ever walks the DIAGONAL of that square -- a measure-zero slice of the
    training distribution.

Off the diagonal, one modality is much cleaner than the other, and the model can take a
cross-modal shortcut instead of generating: the latent decodes to full atom37, so it CONTAINS
the backbone (a clean `z` gives away the Ca trace), and a clean backbone largely gives away the
sequence. A model that learns to copy rather than to generate scores well on every
teacher-forced metric we currently log and still fails at sampling -- which is what
`val_t_*` shows: at `t_ca in [0, 0.2]` (Ca ~90% noise) `decode_predca_predz/atom_rmse_A` is
0.36 A, under a metric with NO Kabsch alignment (cyclic_reconstruction_metrics.py:135).
That is not reachable from 90% noise; the model is reading the backbone off a latent that,
in that bin, was left uniformly random and so usually near-clean (the existing bins key only
on `t["bb_ca"]`, proteina.py:662).

WHAT THIS SCRIPT DOES
---------------------
Sweeps a 2-D grid of (t_ca, t_z), forcing both times to fixed constants, and reports flow loss
and decode quality in every cell. All cells see the SAME batches and the SAME x_0 noise draw
(seeded per batch), so cells are directly comparable.

Read the output like this:

  * Compare the DIAGONAL (t_ca == t_z, the line sampling actually walks) against the
    OFF-DIAGONAL cells (where a shortcut is available).
  * If off-diagonal cells are good and the diagonal collapses -- especially near the origin,
    where sampling STARTS -- the model has learned to copy across modalities, not to generate
    from the receptor. Fix: `shared_groups: [["bb_ca", "local_latents"]]`, which makes training
    sample one `t` and share it, matching the sampler exactly.
  * If the diagonal looks like the rest of the square, this hypothesis is wrong and the
    sampling failure is elsewhere.

`seq_acc` is reported against a MAJORITY-CLASS BASELINE computed on the same residues, because
"26% recovery" is only meaningful next to what a constant predictor scores.

Usage
-----
NEEDS A GPU. Do not run this on a login node: it loads a ~3 GB flow ckpt plus a ~4 GB AE ckpt and
then runs 25 cells x num_batches forward passes. On CPU that is hours, and the `torch.load` stalls
in a native call that Ctrl-C cannot interrupt. The script refuses to run on CPU unless you pass
`--allow_cpu`.

    bash ~/slurm/cpsea/cpsea_t_diagonal.sh                # submits to a GPU node

Costs one validation pass per grid cell (default 25 cells); no training, no gradients.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

CA_IDX = 1
_T0 = time.time()


def log(msg: str) -> None:
    """Stage logging with elapsed time, flushed immediately: the expensive steps here (two multi-GB
    `torch.load`s, a 2.4M-row metadata parquet) are native calls that look like a hang, so every
    stage announces itself BEFORE it starts, not after."""
    print(f"[t_diag +{time.time() - _T0:7.1f}s] {msg}", flush=True)

# Metrics pulled out of each cell for the summary matrices. Everything else still lands in the JSON.
HEADLINE = [
    ("latent_ratio_vs_mean", "latent MSE / predict-the-mean MSE   (>1 = WORSE THAN NO MODEL)"),
    ("latent_ratio_vs_xt", "latent MSE / predict-x_t MSE        (>1 = worse than not moving)"),
    ("local_latents_mse", "latent x1 MSE (unscaled)"),
    ("latent_mse_predict_mean", "  ...vs the predict-the-mean baseline"),
    ("bb_ca_mse", "bb_ca x1 MSE (unscaled)"),
    ("decode_predca_predz/seq_acc", "seq recovery"),
    ("decode_predca_predz/atom_rmse_A", "all-atom RMSE (A, unaligned)"),
    ("decode_predca_predz/isopeptide_bond_success", "isopeptide closure"),
    ("z_norm_ratio", "||z_pred|| / ||z_gt||"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="example/training_cpsea_peptide_cyc_typecond",
                   help="Hydra config name; supplies dataset transforms, AE config and nn definition.")
    p.add_argument("--ckpt", required=True, help="Flow checkpoint to evaluate (e.g. last-EMA.ckpt).")
    p.add_argument("--ae_ckpt", default=None, help="Override cfg.autoencoder_ckpt_path.")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--t_grid", default="0.1,0.3,0.5,0.7,0.9",
                   help="Comma-separated t values; the grid is the cartesian product with itself.")
    p.add_argument("--out", default=None, help="Output dir for the JSON (default: $PROTEINA_ZFS_PATH/training_runs/diagnostics/t_diagonal).")
    p.add_argument("--tag", default="t_diagonal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_cpu", action="store_true",
                   help="Run without a GPU anyway. Expect hours, and torch.load stalls uninterruptibly.")
    return p.parse_args()


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


def load_model(cfg, ckpt: str, ae_ckpt: str | None, device):
    """Mirrors `generate.load_model`, including its decoder-missing fixup, so the AE that decodes
    here is the same one generation would use (and keeps the flow run's latent_norm_stats).

    THE AE MUST BE THE ONE THE FLOW WAS TRAINED AGAINST. `Proteina.load_from_checkpoint` restores
    `cfg_exp` from the checkpoint's hparams, so the checkpoint already knows its own AE path -- we
    only override it when the user explicitly passes `--ae_ckpt`. Passing the *composed config's*
    `autoencoder_ckpt_path` instead (which is what a naive `cfg.get(...)` does) silently swaps in a
    different AE: e.g. `ft_cpflow_fz_cpae_full` trained against `finetune_full_128/last.ckpt`, while
    `$CPSEA_AE_CKPT_PATH` points at `finetune_full_128/step-40000-EMA.ckpt`. Different encoder =>
    the flow's velocity points into a latent space that no longer matches the x_1 target, which
    inflates the latent MSE ~3x and fabricates a "bad at low t" pattern out of nothing (at high t
    x_1_pred ~= x_t, so the wrong-space velocity barely shows; at low t it is the whole prediction).
    """
    from proteinfoundation.proteina import Proteina

    if ae_ckpt is not None:
        log(f"AE OVERRIDE (--ae_ckpt): {ae_ckpt}")
        model = Proteina.load_from_checkpoint(ckpt, strict=False, autoencoder_ckpt_path=ae_ckpt)
    else:
        model = Proteina.load_from_checkpoint(ckpt, strict=False)
    ae_path = ae_ckpt or model.cfg_exp.get("autoencoder_ckpt_path")
    log(f"AE the flow was trained against: {ae_path}")

    cfg_ae = cfg.get("autoencoder_ckpt_path")
    if ae_ckpt is None and cfg_ae is not None and str(cfg_ae) != str(ae_path):
        log(f"NOTE: --config's autoencoder_ckpt_path differs and was IGNORED: {cfg_ae}")

    if getattr(model, "autoencoder", None) is not None and model.autoencoder.decoder is None:
        # The flow checkpoint stores an encoder-only AE. Swap in the full one, but carry over the
        # flow run's normalization stats -- without them decode skips `z * std + mean` and the
        # decoded structure is silently garbage (see generate.py:198).
        latent_norm_stats = getattr(model.autoencoder, "latent_norm_stats", None)
        if ae_path is None or not os.path.exists(ae_path):
            raise SystemExit(f"Flow ckpt has no decoder and autoencoder_ckpt_path is unusable: {ae_path}")
        print(f"[t_diag] flow ckpt has no decoder; loading full AE from {ae_path}")
        full_model = Proteina.load_from_checkpoint(ae_path, strict=False)
        if full_model.autoencoder is None or full_model.autoencoder.decoder is None:
            raise SystemExit(f"AE checkpoint at {ae_path} also has no decoder.")
        full_model.autoencoder.latent_norm_stats = latent_norm_stats
        model.autoencoder = full_model.autoencoder
        del full_model

    model = model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model


@torch.no_grad()
def eval_cell(model, batches, t_ca: float, t_z: float, seed: int, device) -> dict[str, float]:
    """Evaluates the model with `t` PINNED to (t_ca, t_z) for every sample in every batch.

    Replicates `Proteina.training_step` up to `compute_loss`, minus the Lightning logging (which
    needs a trainer). `fm.sample_t` is monkeypatched rather than writing `batch["t"]` directly, so
    `corrupt_batch` builds `x_t` from these times through exactly the training code path.
    """
    from proteinfoundation.eval.ae_reconstruction_eval import run_four_way_decode_eval
    from proteinfoundation.utils.sample_utils import add_clean_samples
    from proteinfoundation.utils.training_handlers import handle_batch_conditioning

    fm = model.fm
    original_sample_t = fm.sample_t
    pinned = {"bb_ca": t_ca, "local_latents": t_z}

    def pinned_sample_t(shape, device):  # noqa: ANN001 - matches fm.sample_t signature
        return {
            dm: torch.full(shape, pinned[dm], device=device, dtype=torch.float32)
            if dm in pinned
            else original_sample_t(shape=shape, device=device)[dm]
            for dm in fm.data_modes
        }

    fm.sample_t = pinned_sample_t
    acc: dict[str, list[float]] = defaultdict(list)
    try:
        for bi, raw in enumerate(batches):
            # Same x_0 noise in every cell => cells differ only by t, not by the noise draw.
            torch.manual_seed(seed + bi)
            batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in raw.items()}

            batch = add_clean_samples(
                batch,
                model.cfg_exp.product_flowmatcher,
                getattr(model, "autoencoder", None),
                local_latent_target=model.cfg_exp.get("local_latent_target", "sample"),
                detach_latent_target_for_flow=bool(model.cfg_exp.get("detach_latent_target_for_flow", False)),
            )
            batch = fm.corrupt_batch(batch)
            bs, _ = batch["mask"].shape

            batch, n_recycle = handle_batch_conditioning(
                batch, bs, model.cfg_exp.training, model.call_nn, fm
            )
            nn_out = model.call_nn(batch, n_recycle=n_recycle)

            losses = fm.compute_loss(batch=batch, nn_out=nn_out)
            for dm in ("bb_ca", "local_latents"):
                key = f"{dm}_unscaled_justlog"
                if key in losses:
                    acc[f"{dm}_mse"].append(float(losses[key].mean().item()))

            for k, v in run_four_way_decode_eval(model.autoencoder, fm, batch, nn_out).items():
                acc[k[len("val/"):]].append(float(v))

            x1_pred = fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
            mask = batch["mask"].bool()
            z_pred = x1_pred["local_latents"][mask]
            z_gt = batch["x_1"]["local_latents"][mask]
            z_t = batch["x_t"]["local_latents"][mask]

            # CALIBRATION. An x1 MSE is meaningless on its own -- at low t the Bayes-optimal
            # prediction IS blurry, so a large MSE there is expected, not a defect. Score the model
            # against the two predictions that require no model at all:
            #   predict_mean: x_1_pred = 0 (the latent mean). MSE = Var(x_1).
            #   predict_xt:   x_1_pred = x_t ("no change").
            # ratio_vs_mean > 1 means the network is WORSE THAN PREDICTING NOTHING at this t --
            # which, at the low t where sampling starts, would mean the ODE is pushed the wrong way
            # from step one and nothing downstream can recover.
            mse_model = (z_pred - z_gt).pow(2).mean()
            mse_mean = z_gt.pow(2).mean()
            mse_xt = (z_t - z_gt).pow(2).mean()
            acc["latent_mse_model"].append(float(mse_model.item()))
            acc["latent_mse_predict_mean"].append(float(mse_mean.item()))
            acc["latent_mse_predict_xt"].append(float(mse_xt.item()))
            acc["latent_ratio_vs_mean"].append(float((mse_model / mse_mean.clamp(min=1e-8)).item()))
            acc["latent_ratio_vs_xt"].append(float((mse_model / mse_xt.clamp(min=1e-8)).item()))

            # Mean-collapse check: a model regressing to the conditional mean emits a SHRUNKEN
            # latent. Ratio << 1 at low t is the signature of a blurry, off-manifold z -- which the
            # decoder (trained only on encoder outputs, KL weight 1e-5) has never seen.
            acc["z_norm_ratio"].append(
                float((z_pred.norm(dim=-1).mean() / z_gt.norm(dim=-1).mean().clamp(min=1e-6)).item())
            )
    finally:
        fm.sample_t = original_sample_t

    return {k: float(np.nanmean(v)) for k, v in acc.items() if len(v)}


@torch.no_grad()
def majority_class_seq_acc(batches) -> float:
    """Accuracy of the best constant amino-acid predictor on these residues.

    Without this, `seq_acc` is uninterpretable: 20 classes does NOT mean chance is 5%, because
    the residue distribution is far from uniform.
    """
    counts: dict[int, int] = defaultdict(int)
    for b in batches:
        mask = b["mask"].bool() if "mask" in b else b["coord_mask"][..., CA_IDX].bool()
        for tok in b["residue_type"][mask].flatten().tolist():
            counts[int(tok)] += 1
    total = sum(counts.values())
    return max(counts.values()) / total if total else float("nan")


def print_matrix(results, grid, key, label) -> None:
    print(f"\n### {label}    [{key}]")
    print("  rows = t_bb_ca (structure) | cols = t_local_latents (sequence+sidechains)")
    print("        " + "".join(f"{t:>9.2f}" for t in grid))
    for t_ca in grid:
        cells = []
        for t_z in grid:
            v = results[(t_ca, t_z)].get(key, float("nan"))
            cells.append("      nan" if np.isnan(v) else f"{v:9.3f}")
        diag_mark = ""
        print(f"  {t_ca:4.2f} |" + "".join(cells) + diag_mark)
    diag = [results[(t, t)].get(key, float("nan")) for t in grid]
    off = [results[(a, b)].get(key, float("nan")) for a in grid for b in grid if a != b]
    print(f"  diagonal (what sampling walks): " + " ".join(f"{v:.3f}" for v in diag))
    print(f"  diagonal mean {np.nanmean(diag):.3f}   |   off-diagonal mean {np.nanmean(off):.3f}")


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import proteinfoundation.patches.atomworks_patches  # noqa: F401  (required for atomize_token)
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    from proteinfoundation.train import load_data_module

    if not torch.cuda.is_available() and not args.allow_cpu:
        print(
            "ERROR: no GPU visible (torch.cuda.is_available() == False).\n"
            "  This loads a ~3 GB flow ckpt + ~4 GB AE ckpt and runs 25 cells of a 160M-param net.\n"
            "  On CPU that takes hours, and the torch.load stalls in a native call that Ctrl-C\n"
            "  cannot interrupt -- you have to kill the shell.\n"
            "  You are probably on a login node. Submit to a GPU node instead:\n"
            "      bash ~/slurm/cpsea/cpsea_t_diagonal.sh\n"
            "  Or pass --allow_cpu if you really mean it.",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = [float(x) for x in args.t_grid.split(",")]
    log(f"device={device}  grid={grid}  ({len(grid)**2} cells x {args.num_batches} batches)")

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config, overrides=["+single=true", "+nolog=true"])

    log(f"loading flow ckpt (this is the slow part, ~3 GB): {args.ckpt}")
    model = load_model(cfg, args.ckpt, args.ae_ckpt, device)
    log("model loaded and on device")

    log("building datamodule (reads the metadata parquet; can be slow on the full split)")
    _, dm = load_data_module(cfg, is_cluster_run=False)
    dm.num_workers = args.num_workers
    dm.batch_size = args.batch_size
    dm.pin_memory = False
    dm.setup("fit")
    loader = dm.val_dataloader() if args.split == "val" else dm.train_dataloader()

    # Cache the batches once: every cell must see identical data for the matrix to mean anything.
    log(f"caching {args.num_batches} batches from split={args.split}")
    batches = []
    for i, b in enumerate(loader):
        if i >= args.num_batches:
            break
        batches.append(move_to_device(b, device))
    if not batches:
        raise SystemExit("No batches loaded; check --split / --num_batches.")
    n_samples = sum(int(b["coord_mask"][..., CA_IDX].bool().any(-1).sum().item()) for b in batches)
    log(f"cached {len(batches)} batches (~{n_samples} structures)")

    baseline = majority_class_seq_acc(batches)
    log(f"majority-class seq_acc baseline on these residues: {baseline:.3f}")

    results: dict[tuple[float, float], dict[str, float]] = {}
    n_cells = len(grid) ** 2
    for ci, t_ca in enumerate(grid):
        for cj, t_z in enumerate(grid):
            results[(t_ca, t_z)] = eval_cell(model, batches, t_ca, t_z, args.seed, device)
            tag = "DIAG" if t_ca == t_z else "    "
            r = results[(t_ca, t_z)]
            log(
                f"cell {ci * len(grid) + cj + 1:2d}/{n_cells} {tag} t_ca={t_ca:.2f} t_z={t_z:.2f}  "
                f"latent_mse={r.get('local_latents_mse', float('nan')):.3f}  "
                f"bb_ca_mse={r.get('bb_ca_mse', float('nan')):.3f}  "
                f"seq_acc={r.get('decode_predca_predz/seq_acc', float('nan')):.3f}  "
                f"atom_rmse={r.get('decode_predca_predz/atom_rmse_A', float('nan')):.3f}"
            )

    for key, label in HEADLINE:
        print_matrix(results, grid, key, label)

    print(f"\n### how to read this")
    print(f"  seq_acc majority-class baseline = {baseline:.3f}. A diagonal seq_acc near that number")
    print( "  means the model cannot pick a sequence without being handed a clean modality.")
    print( "  If the off-diagonal is strong and the diagonal collapses, training's independent-t")
    print( "  draw (shared_groups: []) let the model learn cross-modal copying instead of")
    print( "  generation, and sampling -- which only walks the diagonal -- has no shortcut left.")
    print( "  Fix to test: loss.t_distribution.shared_groups: [['bb_ca', 'local_latents']]")

    out_dir = Path(args.out) if args.out else Path(
        os.environ.get("PROTEINA_ZFS_PATH", str(REPO))) / "training_runs" / "diagnostics" / "t_diagonal"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.tag}_{args.split}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "ckpt": args.ckpt,
                "config": args.config,
                "split": args.split,
                "num_batches": args.num_batches,
                "batch_size": args.batch_size,
                "grid": grid,
                "majority_class_seq_acc": baseline,
                "cells": {f"{a}_{b}": v for (a, b), v in results.items()},
            },
            f,
            indent=2,
        )
    print(f"\n[t_diag] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
