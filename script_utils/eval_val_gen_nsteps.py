"""Re-run `val_generation` on a FROZEN checkpoint at several ODE step counts.

WHY THIS EXISTS
---------------
Arm B (`rollout_finetune`, DRaFT-K) demonstrably moved the quantity it optimises -- its
internal sampled closure climbed 0.281 -> 0.500 over 589 rollout triggers -- and yet landed
within noise of the v4 baseline on every `val_gen` linkage type. The leading explanation,
written into that arm's config header before launch and never tested, is a FIDELITY GAP:
rollouts train at `nsteps=100` (a rollout costs `nsteps` forwards, so it is run coarse),
while `val_gen` scores at `nsteps=400`. If the arm learned to close rings under a coarse
integrator and that skill does not transfer to a fine one, the null is an artefact of the
measurement and not a statement about the method.

That is a question about the SAMPLER, not about training, so it does not need another
training run -- only the finished weights, re-scored at both step counts.

WHAT IT DOES
------------
Loads a stored *resolved* exp config (the JSON written next to the checkpoints, so the
model is rebuilt exactly as it trained, not as today's YAML would build it), restores the
weights, and calls `Proteina.validation_step_generate` directly over the val loader for
each (nsteps, seed) condition. One JSONL row per (arm, nsteps, seed, batch) with the full
metric dict, so the comparison can be re-aggregated without re-running any GPU work.

Two design points that decide whether the answer means anything:

  * EMA WEIGHTS BY DEFAULT. `ema.validate_original_weights=false`, so every `val_gen`
    number ever logged for these runs was produced by the EMA shadow weights, which live
    in the sibling `last-EMA.ckpt`. Scoring `last.ckpt` instead would compare against a
    model that was never validated, and the nsteps=400 control would fail to reproduce
    the logged value for reasons having nothing to do with nsteps. `--raw-weights`
    overrides, but the default is EMA and it is the default on purpose.

  * MATCHED NOISE ACROSS ARMS. The RNG is re-seeded from (seed, batch_idx) before every
    single batch, so at a given (nsteps, seed, batch) each arm integrates from the SAME x_0
    draw on the same complexes. This makes the arm comparison paired -- far tighter than the
    training-time comparison, which could only average over validations that never shared
    a noise draw. It does NOT pair across nsteps: `sampling_mode: sc` consumes noise per
    step, so 100 and 400 steps diverge in the RNG stream after the first step. Compare
    arms within an nsteps, never a single arm across nsteps at fixed seed.

Resumable: rows already in the output file are skipped, so a job that hits its wall limit
is restarted by resubmitting it.

Not a training entrypoint. GPU, one card. Aggregate with
`script_utils/summarize_val_gen_nsteps.py`, which is CPU-only and reads the JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, required=True,
                   help="Training run dir under store/, e.g. store/cpsea_rollout_draft_from_v4cfg. "
                        "The resolved exp config and checkpoints are read from its checkpoints/ subdir.")
    p.add_argument("--label", type=str, default=None,
                   help="Name for this arm in the output rows. Default: the run dir's basename.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint to score. Default: <run-dir>/checkpoints/last-EMA.ckpt (see --raw-weights).")
    p.add_argument("--raw-weights", action="store_true",
                   help="Score last.ckpt (the live weights) instead of last-EMA.ckpt. Off by default: "
                        "validation ran on EMA weights, so EMA is what reproduces the logged numbers.")
    p.add_argument("--nsteps", type=str, default="100,400",
                   help="Comma-separated ODE step counts. 400 is the control (reproduces val_gen); "
                        "100 is arm B's rollout fidelity.")
    p.add_argument("--seeds", type=str, default="0,1,2",
                   help="Comma-separated sampling seeds. Each is an independent noise draw, shared "
                        "across arms at the same (nsteps, seed).")
    p.add_argument("--n-batches", type=int, default=8,
                   help="Val batches per condition. Training used 2; more buys population, and the "
                        "val loader is unshuffled so batch i is the same complexes every time.")
    p.add_argument("--n-repeat", type=int, default=4, help="Samples per complex, as in training.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory for row files. THIS PROCESS PICKS ITS OWN FILENAME inside it: "
                        "rows_<label>_<jobid>.jsonl. Nothing else can be appending to that path, so "
                        "concurrent writers are impossible by construction rather than by convention "
                        "-- which is what actually failed twice: first three arms sharing one file, "
                        "then the submitter being run twice and pairing two jobs per arm. Resume "
                        "reads every rows_<label>_*.jsonl in the directory, so a second submission "
                        "skips work already on disk instead of redoing or corrupting it.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the conditions that would run and exit. No GPU, no model load.")
    return p.parse_args()


def _row_key(row: dict) -> tuple:
    return (row["label"], row["ckpt_name"], row["nsteps"], row["seed"], row["batch_idx"])


def _load_done_keys(out_dir: Path, label: str) -> set[tuple]:
    """Keys already on disk for this arm, across EVERY file, plus a loud corruption count.

    Corruption here is not hypothetical. The first version of this tool pointed all three
    concurrent scoring jobs at ONE shared output file. Concurrent O_APPEND is not atomic on
    this filesystem: 144 written rows came back as 56 parseable ones, the rest overwritten
    with NUL padding where two writers' offsets collided. The aggregation step skipped the
    bad lines in silence and produced a complete-looking table built on 39% of the data.

    Each job now owns its own file, so there is exactly one writer and this should always
    read zero corrupt lines. It counts them anyway, and aborts if any appear -- silently
    tolerating them is what turned data loss into a plausible wrong answer.
    """
    done, corrupt, seen = set(), 0, []
    for f in sorted(out_dir.glob(f"rows_{label}_*.jsonl")):
        seen.append(f.name)
        with f.open(errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if "\x00" in line:
                    corrupt += 1
                    continue
                try:
                    done.add(_row_key(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    corrupt += 1
    if seen:
        logger.info(f"Resume: read {len(done)} existing rows from {', '.join(seen)}")
    if corrupt:
        logger.error(f"{corrupt} unreadable line(s) in {out_dir} for label={label}. Those files "
                     f"were written by colliding writers or are damaged. Move them aside and rerun "
                     f"-- appending beside corrupt rows just buries the loss deeper.")
        raise SystemExit(2)
    return done


def main() -> int:
    args = parse_args()
    load_dotenv(".env")

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    label = args.label or run_dir.name
    nsteps_list = [int(x) for x in args.nsteps.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    cfg_files = sorted(ckpt_dir.glob("exp_config_*.json"))
    if not cfg_files:
        logger.error(f"No exp_config_*.json in {ckpt_dir}. This must be a completed training run's store dir.")
        return 1
    cfg_file = cfg_files[0]

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = ckpt_dir / ("last.ckpt" if args.raw_weights else "last-EMA.ckpt")
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _load_done_keys(out_dir, label)
    # SLURM job id when there is one, pid otherwise. Either way this path is this process's
    # alone -- the file is created by this run and never appended to by anything else.
    tag = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    out_path = out_dir / f"rows_{label}_{tag}.jsonl"
    logger.info(f"Writing to {out_path} (exclusive to this job)")

    conditions = [(ns, sd) for ns in nsteps_list for sd in seeds]
    todo = [
        (ns, sd, b)
        for ns, sd in conditions
        for b in range(args.n_batches)
        if (label, ckpt_path.name, ns, sd, b) not in done
    ]
    logger.info(f"arm={label} ckpt={ckpt_path.name} config={cfg_file.name}")
    logger.info(f"conditions={len(conditions)} batches/condition={args.n_batches} "
                f"already done={len(conditions) * args.n_batches - len(todo)} to run={len(todo)}")
    if args.dry_run:
        for ns, sd in conditions:
            pending = sum(1 for a, b, _ in todo if (a, b) == (ns, sd))
            logger.info(f"  nsteps={ns:>4} seed={sd}  pending batches={pending}")
        return 0
    if not todo:
        logger.info("Nothing to do -- every condition already has rows. Exiting 0.")
        return 0

    # Imported late: these pull in torch/CUDA-heavy modules, and --dry-run should not pay for them.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.train import load_data_module

    cfg_exp = OmegaConf.create(json.load(cfg_file.open()))
    OmegaConf.set_struct(cfg_exp, False)
    # The model must be rebuilt exactly as it trained, so nothing here touches architecture,
    # cyclization or dataset keys. Only the val-generation sampler knobs move.
    cfg_exp.val_generation.enabled = True
    cfg_exp.val_generation.n_batches = args.n_batches
    cfg_exp.val_generation.n_repeat = args.n_repeat
    # Rollout fine-tuning is a TRAINING term. Leaving it on would make `Proteina.__init__`
    # build its sampler and, worse, invite a rollout inside a scoring run.
    if cfg_exp.get("rollout_finetune") is not None:
        cfg_exp.rollout_finetune.enabled = False

    if cfg_exp.force_precision_f32:
        torch.set_float32_matmul_precision("high")

    _, datamodule = load_data_module(cfg_exp, is_cluster_run=True)
    datamodule.setup("validate")
    device = torch.device(args.device)

    model = Proteina(cfg_exp)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    logger.info(f"Loaded {ckpt_path} (global_step={ckpt.get('global_step')}) "
                f"missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        logger.warning(f"Unexpected keys (first 5): {list(unexpected)[:5]}")
    global_step = int(ckpt.get("global_step", -1))
    del ckpt

    model.to(device).eval()

    # `validation_step_generate` ends in `self.log_nan_safe(...)`, which needs a Trainer. There is
    # no Trainer here by design -- this is a scoring run, not a fit -- so the sink is replaced with
    # a collector. Same call signature, so the method itself is untouched and cannot drift from
    # what training runs.
    collected: dict[str, float] = {}
    model.log_nan_safe = lambda key, value, bs, on_step: collected.__setitem__(key, float(value))

    n_written = 0
    with out_path.open("a") as fh:
        for nsteps in nsteps_list:
            for seed in seeds:
                pending = [b for ns, sd, b in todo if ns == nsteps and sd == seed]
                if not pending:
                    continue
                model.val_gen_cfg.nsteps = nsteps
                loader = datamodule.val_dataloader()
                t0 = time.time()
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= args.n_batches:
                        break
                    if batch_idx not in pending:
                        continue  # already on disk from an earlier job
                    # Seeded PER BATCH, not per condition. Seeding once outside this loop would
                    # make batch i's noise depend on how many batches ran before it -- so a job
                    # resumed after the wall limit would score the remaining batches with
                    # different noise than a fresh run, and the resume would silently stop being
                    # a continuation. Per-batch seeding makes every row a function of
                    # (seed, batch_idx) alone: reproducible, resumable, and identical across arms.
                    L.seed_everything(seed * 100003 + batch_idx, workers=True)
                    batch = model.transfer_batch_to_device(batch, device, 0)
                    collected.clear()
                    model.validation_step_generate(batch, batch_idx)
                    row = {
                        "label": label,
                        "ckpt_name": ckpt_path.name,
                        "ckpt_global_step": global_step,
                        "nsteps": nsteps,
                        "seed": seed,
                        "batch_idx": batch_idx,
                        "n_repeat": args.n_repeat,
                        "metrics": dict(collected),
                    }
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()  # resumability is only real if the row is on disk before the wall hits
                    n_written += 1
                logger.info(f"nsteps={nsteps} seed={seed}: {len(pending)} batches in {time.time()-t0:.0f}s")

    logger.info(f"Wrote {n_written} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
