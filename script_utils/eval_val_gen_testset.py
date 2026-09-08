"""Run `val_generation` (single-pass ODE, exactly as in the training validation loop) on a
stratified sample of the native CPSea **test** set, for one frozen flow checkpoint.

WHY THIS EXISTS
---------------
During training, `val_generation` scores non-teacher-forced samples on a couple of batches of
the held-out set (`cpsea_val.parquet`). This re-runs that identical procedure -- same sampler
(`design_sampling`: nsteps=400, self_cond, sampling_mode `sc`), same `n_repeat`, same metrics --
on a larger, stratified draw from `cpsea_test.parquet`, which is the same distribution. The point
is a low-variance read on sampled cyclization closure (Ca/Cb + per-linkage bond success) that
should reproduce the training curves, plus per-complex Rosetta interface dG for every sample.

Rosetta dG is NOT computed here (it is seconds of CPU per complex and would waste the GPU).
Instead this dumps every sampled complex to disk (`val_gen_dump.dump_val_gen_complexes`, wired
through `Proteina.validation_step_generate` via `val_generation.rosetta_dump_dir`), and a separate
sharded CPU job (`scripts/score_val_gen_rosetta.py`) scores them offline.

TWO THINGS THAT DECIDE WHETHER THE ANSWER MEANS ANYTHING
--------------------------------------------------------
  * EMA WEIGHTS. Validation ran on the EMA shadow, so the numbers logged during training were
    produced by `last-EMA.ckpt`. That is the default here; scoring `last.ckpt` would compare a
    model that was never validated and the closure numbers would not reproduce for reasons having
    nothing to do with the test set.

  * UNIQUE DUMP STEP PER BATCH. `dump_val_gen_complexes` names its output dir and PDBs by
    `self.global_step`, which is 0 for every batch when there is no Lightning Trainer attached --
    so every batch would dump into `step_00000000/` with the same `..._bid{i}.pdb` names and
    silently overwrite each other, leaving one batch's worth of complexes and a manifest full of
    dangling duplicate rows. We monkeypatch `Proteina.global_step` to a per-batch counter so each
    batch owns its own `step_{batch_idx:08d}/` dir. (`current_epoch` stays 0; it is only a label.)

Resumable: batches whose closure row is already on disk are skipped, and the Rosetta dump / scorer
are each independently resumable. Not a training entrypoint. GPU, one card.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="Training run dir under store/, e.g. store/cpsea_bondunroll_pin20260828. "
                        "Its checkpoints/ holds the resolved exp config + weights.")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint to score. Default: <run-dir>/checkpoints/last-EMA.ckpt.")
    p.add_argument("--raw-weights", action="store_true",
                   help="Score last.ckpt instead of last-EMA.ckpt. Off by default: validation ran "
                        "on EMA weights, so EMA is what reproduces the logged closure numbers.")
    p.add_argument("--test-parquet", required=True,
                   help="Native CPSea test metadata parquet to sample from (cpsea_test.parquet).")
    p.add_argument("--n-complexes", type=int, default=1000,
                   help="How many distinct test complexes to sample (stratified by cyclization type, "
                        "cluster-unique). Each is sampled --n-repeat times.")
    p.add_argument("--n-repeat", type=int, default=4,
                   help="Samples per complex, as in training val_generation (default 4).")
    p.add_argument("--nsteps", type=int, default=None,
                   help="ODE steps. Default: inherit the design sampler's (400) -- i.e. exactly "
                        "what val_generation used during training. Lower only to smoke-test.")
    p.add_argument("--seed", type=int, default=0,
                   help="Sampling seed. Re-seeded per batch as seed*100003+batch_idx for resumable, "
                        "reproducible noise.")
    p.add_argument("--out-dir", required=True,
                   help="Output dir. Holds the stratified subset parquet, closure_rows_*.jsonl "
                        "(one row per batch, resumable), and rosetta_dump/ (complex PDBs + manifests).")
    p.add_argument("--rosetta-dump-max", type=int, default=128,
                   help="Max complexes dumped per batch. Set >= batch_size*n_repeat so every sample "
                        "is dumped for Rosetta (batch_size 16 * n_repeat 4 = 64).")
    p.add_argument("--subset-seed", type=int, default=0, help="Seed for the stratified subset draw.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="Build the subset, print plan, no GPU.")
    return p.parse_args()


def build_stratified_subset(test_parquet: str, out_parquet: Path, n_total: int, seed: int) -> pd.DataFrame:
    """Cluster-unique, type-stratified draw of ``n_total`` rows, written once and reused (resumable).

    Stratifying by `cyclization_type` gives each linkage class (mainchain/head_tail, disulfide,
    other) solid n even though the natural mix is ~57/32/11 -- mainchain closure on a proportional
    draw would rest on too few complexes. Cluster-unique so near-duplicate crops do not inflate n.
    The result is shuffled so each batch of 16 mixes types (balanced per-batch closure metrics).
    """
    if out_parquet.exists():
        df = pd.read_parquet(out_parquet)
        logger.info(f"Reusing existing subset {out_parquet} ({len(df)} rows)")
        return df

    full = pd.read_parquet(test_parquet)
    logger.info(f"Read {len(full)} test rows from {test_parquet}")
    if "cluster_id" in full.columns:
        full = full.drop_duplicates(subset=["cluster_id"])
        logger.info(f"{len(full)} cluster-unique rows")

    types = sorted(full["cyclization_type"].dropna().unique().tolist())
    per = n_total // len(types)
    parts = []
    for t in types:
        sub = full[full["cyclization_type"] == t]
        take = min(per, len(sub))
        parts.append(sub.sample(n=take, random_state=seed))
        logger.info(f"  type={t}: available={len(sub)} took={take}")
    out = pd.concat(parts)
    # Top up to n_total from whatever is left if some type was short.
    if len(out) < n_total:
        rest = full[~full["example_id"].isin(out["example_id"])]
        extra = rest.sample(n=min(n_total - len(out), len(rest)), random_state=seed)
        out = pd.concat([out, extra])
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet)
    logger.info(f"Wrote stratified subset {out_parquet}: {len(out)} rows, "
                f"types={out['cyclization_type'].value_counts().to_dict()}")
    return out


def _done_batches(out_dir: Path) -> set[int]:
    """Batch indices already written across every closure_rows_*.jsonl (resume set)."""
    done = set()
    for f in sorted(out_dir.glob("closure_rows_*.jsonl")):
        with f.open(errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or "\x00" in line:
                    continue
                try:
                    done.add(int(json.loads(line)["batch_idx"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    if done:
        logger.info(f"Resume: {len(done)} batches already on disk")
    return done


def main() -> int:
    args = parse_args()
    load_dotenv(".env")

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = out_dir / "rosetta_dump"

    cfg_files = sorted(ckpt_dir.glob("exp_config_*.json"))
    if not cfg_files:
        logger.error(f"No exp_config_*.json in {ckpt_dir}. Must be a completed training run's store dir.")
        return 1
    cfg_file = cfg_files[0]

    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_dir / ("last.ckpt" if args.raw_weights else "last-EMA.ckpt")
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1

    subset_parquet = out_dir / f"cpsea_test_eval{args.n_complexes}.parquet"
    subset = build_stratified_subset(args.test_parquet, subset_parquet, args.n_complexes, args.subset_seed)
    n_batches = math.ceil(len(subset) / 16)  # val loader batch_size is 16 (see structure_data.py)

    done = _done_batches(out_dir)
    todo = [b for b in range(n_batches) if b not in done]
    logger.info(f"ckpt={ckpt_path.name} complexes={len(subset)} n_batches={n_batches} "
                f"n_repeat={args.n_repeat} nsteps={args.nsteps or 'inherit(400)'} to_run={len(todo)}")
    if args.dry_run:
        logger.info("dry-run: subset built, nothing sampled.")
        return 0
    if not todo:
        logger.info("Nothing to do -- every batch already has a closure row. Exiting 0.")
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.train import load_data_module

    cfg_exp = OmegaConf.create(json.load(cfg_file.open()))
    OmegaConf.set_struct(cfg_exp, False)

    # Point BOTH metadata files at the subset: we only ever call val_dataloader, and this avoids
    # StructureDataModule.setup() reading the 2.71M-row train parquet just to build an unused
    # train dataset (which OOM-risks and is pure waste here).
    dm = cfg_exp.dataset.unified.datamodule
    dm.metadata_file = str(subset_parquet)
    dm.val_metadata_file = str(subset_parquet)

    # Rebuild the model exactly as it trained; only the val-generation sampler bookkeeping moves.
    vg = cfg_exp.val_generation
    vg.enabled = True
    vg.n_batches = n_batches
    vg.n_repeat = args.n_repeat
    vg.nsteps = args.nsteps  # None => inherit design_sampling's 400, exactly as training val_gen
    vg.rosetta_dump_dir = str(dump_dir)
    vg.rosetta_dump_max = args.rosetta_dump_max
    vg.rosetta_auto_submit_every_n_vals = 0  # this driver never submits the sidecar itself
    if cfg_exp.get("rollout_finetune") is not None:
        cfg_exp.rollout_finetune.enabled = False  # a training term; must not fire in a scoring run

    if cfg_exp.get("force_precision_f32"):
        torch.set_float32_matmul_precision("high")

    _, datamodule = load_data_module(cfg_exp, is_cluster_run=True)
    datamodule.setup("validate")
    device = torch.device(args.device)

    # dump_val_gen_complexes keys its step dir + PDB names off self.global_step, which is 0 for every
    # batch with no Trainer attached -> all batches collide in step_00000000/. Make it a per-batch
    # counter so each batch owns step_{batch_idx:08d}/. Instance attribute set in the loop below.
    Proteina.global_step = property(lambda self: int(getattr(self, "_eval_gstep", 0)))

    model = Proteina(cfg_exp)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    logger.info(f"Loaded {ckpt_path} (global_step={ckpt.get('global_step')}) "
                f"missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        logger.warning(f"Unexpected keys (first 5): {list(unexpected)[:5]}")
    ckpt_step = int(ckpt.get("global_step", -1))
    del ckpt
    model.to(device).eval()

    # validation_step_generate ends in self.log_nan_safe(...), which needs a Trainer. Replace the
    # sink with a collector so the exact same method runs, unmodified, and we capture its metrics.
    collected: dict[str, float] = {}
    model.log_nan_safe = lambda key, value, bs, on_step: collected.__setitem__(key, float(value))
    model.val_gen_cfg.n_batches = n_batches  # belt-and-suspenders: honoured by the early-return guard

    tag = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    out_path = out_dir / f"closure_rows_{tag}.jsonl"
    logger.info(f"Writing closure rows to {out_path}; dumping complexes to {dump_dir}")

    loader = datamodule.val_dataloader()
    n_written, t0 = 0, time.time()
    with out_path.open("a") as fh:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break
            if batch_idx in done:
                continue
            L.seed_everything(args.seed * 100003 + batch_idx, workers=True)
            model._eval_gstep = batch_idx  # unique step_dir for this batch's dump
            batch = model.transfer_batch_to_device(batch, device, 0)
            collected.clear()
            model.validation_step_generate(batch, batch_idx)
            row = {
                "batch_idx": batch_idx,
                "ckpt_name": ckpt_path.name,
                "ckpt_global_step": ckpt_step,
                "nsteps": args.nsteps or 400,
                "seed": args.seed,
                "n_repeat": args.n_repeat,
                "metrics": dict(collected),
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()  # resumability is only real once the row is on disk
            n_written += 1
            if n_written % 5 == 0:
                logger.info(f"  {n_written}/{len(todo)} batches in {time.time()-t0:.0f}s")

    logger.info(f"Wrote {n_written} closure rows to {out_path} in {time.time()-t0:.0f}s")
    logger.info(f"Rosetta dump ready under {dump_dir} -- score it with scripts/score_val_gen_rosetta.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
