"""Generate (LP, CP, target) triplets from a trained CP->LP generator checkpoint.

Reads a training run's own ``exp_config_*.json`` so the model is rebuilt exactly as it
trained -- same network shape, same feature list, same AE -- then samples linear peptides
for a held-out set of CPSea cyclic complexes and writes scored triplets.

The sampler used here is the FULL deployed one (``--nsteps``, default 200), not the short
differentiable trajectory the GAN updates use. Those exist only so discriminator gradients
can reach the flow network within a tractable memory budget; nothing should be exported
from a 24-step rollout.

Resumable: each shard writes its own manifest and is skipped if already present, so
hitting a time limit means resubmitting rather than restarting. Per-shard files also avoid
the concurrent-append corruption that shared output files produce on this cluster.

Example:
    .venv/bin/python script_utils/cp2lp_generate.py \\
        --run-dir $STORE/cp2lp_gan_v1 \\
        --cp-parquet $LPDATA/metadata_mixed/cp2lp_eval.parquet \\
        --out-dir $STORE/cp2lp_gan_v1/triplets \\
        --n-cp 200 --samples-per-cp 4 --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Training run store dir (holds checkpoints/).")
    p.add_argument("--ckpt", default=None, help="Explicit checkpoint path; default checkpoints/last-EMA.ckpt.")
    p.add_argument("--raw-weights", action="store_true", help="Use last.ckpt instead of the EMA weights.")
    p.add_argument("--cp-parquet", required=True, help="Metadata of the CPSea complexes to edit (eval families).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-cp", type=int, default=200, help="Number of source CPs to draw. -1 for all.")
    p.add_argument("--samples-per-cp", type=int, default=4, help="LPs generated per CP per seed.")
    p.add_argument("--seeds", default="0", help="Comma-separated noise seeds; each is a separate pass.")
    p.add_argument("--nsteps", type=int, default=200, help="ODE steps for the deployed sampler.")
    p.add_argument("--batch-size", type=int, default=4, help="Source CPs per forward.")
    p.add_argument("--write-rejected", action="store_true", help="Also write PDBs for rejected samples.")
    p.add_argument(
        "--reference-mode",
        action="store_true",
        help="Treat --cp-parquet as REAL bound linear peptides: AE round-trip and measure "
        "them with the same scoring code, writing the report's real-side reference. "
        "No generation happens.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="Build the plan and exit without a GPU.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(".env")

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_files = sorted(ckpt_dir.glob("exp_config_*.json"))
    if not cfg_files:
        logger.error(f"No exp_config_*.json in {ckpt_dir}; this must be a training run's store dir.")
        return 1
    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_dir / ("last.ckpt" if args.raw_weights else "last-EMA.ckpt")
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    logger.info(
        f"ckpt={ckpt_path.name} cp_parquet={args.cp_parquet} n_cp={args.n_cp} "
        f"samples_per_cp={args.samples_per_cp} seeds={seeds} nsteps={args.nsteps}"
    )
    if args.dry_run:
        logger.info("dry-run: plan built, nothing sampled.")
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from proteinfoundation.cp2lp.export import export_triplets, measure_real_lps
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import load_data_module
    from proteinfoundation.utils.sample_utils import add_clean_samples

    cfg_exp = OmegaConf.create(json.load(cfg_files[0].open()))
    OmegaConf.set_struct(cfg_exp, False)

    # Point both metadata files at the eval set: only val_dataloader is used, and letting
    # setup() build the multi-million-row train dataset here is pure waste.
    dm = cfg_exp.dataset.unified.datamodule
    dm.metadata_file = str(args.cp_parquet)
    dm.val_metadata_file = str(args.cp_parquet)
    dm.val_filters = None
    # The weighted sampler is built from the TRAIN metadata, which has just been repointed
    # at a single-source parquet, and `source_fractions` must name exactly the sources
    # present -- so the inherited three-way training mix is invalid here and raises during
    # setup(). MEASURED (job 68907): the reference stage died with "Requested
    # ['cpsea','pepbench','protfrag'], found ['pepbench']" before loading the model. Nothing
    # on this path trains, only `val_dataloader()` is used, and an empty mix is the
    # supported "no weighted sampling" value.
    dm.source_fractions = None
    if cfg_exp.get("rollout_finetune") is not None:
        cfg_exp.rollout_finetune.enabled = False
    if cfg_exp.get("force_precision_f32"):
        torch.set_float32_matmul_precision("high")

    _, datamodule = load_data_module(cfg_exp, is_cluster_run=True)
    datamodule.setup("validate")

    model = CP2LPGenerator(cfg_exp)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    logger.info(
        f"Loaded {ckpt_path.name} (global_step={ckpt.get('global_step')}) "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        logger.warning(f"Missing keys (first 5): {list(missing)[:5]}")
    del ckpt
    device = torch.device(args.device)
    model.to(device).eval()

    loader = datamodule.val_dataloader()
    n_batches = len(loader) if args.n_cp < 0 else math.ceil(args.n_cp / max(1, args.batch_size))

    totals = {"n_samples": 0, "n_accepted": 0}
    for seed in seeds:
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for bidx, batch in enumerate(loader):
            if bidx >= n_batches:
                break
            shard_done = seed_dir / f"shard{bidx:05d}.done"
            if shard_done.exists():
                logger.info(f"seed={seed} batch={bidx}: already done, skipping.")
                continue

            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            source_ids = list(batch.get("example_id", [f"batch{bidx}_i{i}" for i in range(batch["mask"].shape[0])]))

            if args.reference_mode:
                # Real LPs: encode, decode, measure. Same round trip the discriminator's
                # real pool goes through, so the report compares peptides rather than
                # comparing one processing path against another.
                with torch.no_grad():
                    ref = add_clean_samples(batch, cfg_exp.product_flowmatcher, model.autoencoder)
                    rmask = ref["mask"].bool()
                    decoded = model._decode({dm: ref["x_1"][dm] for dm in model._data_modes}, rmask)
                    summary = measure_real_lps(
                        out_dir=str(seed_dir),
                        decoded=decoded,
                        batch=ref,
                        mask=rmask,
                        example_ids=source_ids,
                        manifest_name=f"real_reference_shard{bidx:05d}.jsonl",
                    )
                totals["n_samples"] += summary["n_real"]
                shard_done.touch()
                logger.info(f"reference batch={bidx + 1}/{n_batches}: measured {summary['n_real']}")
                continue

            result = model.generate_lp_for_cp(
                batch,
                nsteps=args.nsteps,
                samples_per_cp=args.samples_per_cp,
                seed=seed * 100003 + bidx,  # distinct per (seed, shard), reproducible
            )
            summary = export_triplets(
                out_dir=str(seed_dir),
                result=result,
                source_ids=source_ids,
                seed=seed,
                samples_per_cp=args.samples_per_cp,
                manifest_name=f"triplets_shard{bidx:05d}.jsonl",
                write_rejected=args.write_rejected,
            )
            totals["n_samples"] += summary["n_samples"]
            totals["n_accepted"] += summary["n_accepted"]
            shard_done.touch()
            logger.info(
                f"seed={seed} batch={bidx + 1}/{n_batches}: "
                f"{summary['n_accepted']}/{summary['n_samples']} accepted"
            )

    totals["acceptance_rate"] = totals["n_accepted"] / max(1, totals["n_samples"])
    with open(out_dir / "generate_summary.json", "w") as f:
        json.dump({**totals, "seeds": seeds, "ckpt": str(ckpt_path)}, f, indent=2)
    logger.info(f"CP2LP generate done: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
