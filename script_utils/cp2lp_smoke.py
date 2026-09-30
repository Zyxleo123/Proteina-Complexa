"""Minimal preflight for the CP->LP generator: do the pieces fit together at all?

Four checks, in the order in which failure is cheapest to diagnose:

1. CONFIG. The source-CP features are actually wired into the network, and the AE / flow
   checkpoint pairing is the intended one. Printed, not merely asserted, because the
   pairing is the thing most worth eyeballing before burning a GPU-day.
2. CHECKPOINTS LOAD TOGETHER. Build the model, load the flow checkpoint into it, and
   report missing/unexpected keys. The discriminator is expected to be missing (it is new);
   anything else missing means the init is not what it claims to be.
3. ONE BATCH, FORWARD. Generator rollout -> AE decode -> discriminator, with the roles
   split as they will be in training. Prints the shapes and the loss components.
4. BACKWARD. Gradients reach the flow network from the adversarial term, and reach the
   discriminator from the hinge loss. A generator gradient of exactly zero here means the
   adversarial path is disconnected, which no amount of training will fix.

This is a preflight, not a validation framework: it runs one batch and exits.

Run (GPU):
    .venv/bin/python script_utils/cp2lp_smoke.py --config example/training_cp2lp_gan
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="example/training_cp2lp_gan")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--val-generation",
        action="store_true",
        help="Also run val_generation at a token budget. Off by default because a "
        "400-step sampler dominates the smoke; worth one run before a long pilot.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Default: the config's own batch size. A preflight that measures memory at a "
        "SMALLER batch than production measures the wrong thing -- the flow branch scales "
        "with batch size even though the rollout branch is capped by max_cp_per_step.",
    )
    p.add_argument(
        "--nsteps",
        type=int,
        default=None,
        help="Rollout steps. Default: the config's cp2lp.rollout.nsteps, so the smoke "
        "exercises the regime training will actually run. Do NOT set this very low to make "
        "the smoke fast: the sampler schedules are non-uniform, and at nsteps=6 the "
        "`log` schedule puts bb_ca's final step at t=0.988, where vf_to_score's 1/(1-t) "
        "factor amplifies a garbage endpoint into a NaN gradient -- a failure of the "
        "setting, not of the code.",
    )
    p.add_argument("--skip-ckpt", action="store_true", help="Skip loading the flow checkpoint (shape check only).")
    p.add_argument(
        "--trainer-steps",
        type=int,
        default=6,
        help="Training steps to run in the Trainer check. >1 on purpose: peak memory is "
        "NOT constant across batches (pair features are O(n_target^2) and the receptor "
        "crop runs to 256 residues), so one passing step says nothing about the envelope. "
        "Job 68347 OOMed a 48G a6000 in real training after a single-step smoke passed.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(".env")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name=args.config)
    OmegaConf.set_struct(cfg, False)

    # ---------------------------------------------------------------- 1. config
    print("=" * 78)
    print("1. CONFIG")
    print(f"   run_name            : {cfg.run_name}")
    print(f"   AE   (defines z)    : {cfg.autoencoder_ckpt_path}")
    print(f"   flow (init)         : {cfg.pretrain_ckpt_path}")
    print(f"   adversarial enabled : {cfg.cp2lp.adversarial.enabled}")
    src_seq = [f for f in cfg.nn.feats_seq if f.startswith("x_src_cp") or f == "src_cp_present"]
    src_pair = [f for f in cfg.nn.feats_pair_repr if f.startswith("x_src_cp")]
    print(f"   source-CP seq feats : {src_seq}")
    print(f"   source-CP pair feats: {src_pair}")
    if not src_seq and not src_pair:
        logger.error("No source-CP feature is wired in: the generator would be unconditional.")
        return 1
    for path in (cfg.autoencoder_ckpt_path, cfg.pretrain_ckpt_path):
        if not os.path.exists(path):
            logger.error(f"Checkpoint missing: {path}")
            return 1
    print("   both checkpoint files exist")

    if args.batch_size is not None:
        cfg.dataset.datamodule.batch_size = args.batch_size
    batch_size_used = int(cfg.dataset.datamodule.batch_size)
    # Two workers, not eight: the smoke reads a handful of batches, and worker startup
    # dominates its wall clock.
    cfg.dataset.datamodule.num_workers = 2
    cfg.dataset.unified.datamodule.num_workers = 2
    if args.nsteps is not None:
        cfg.cp2lp.rollout.nsteps = args.nsteps
    nsteps_used = int(cfg.cp2lp.rollout.nsteps)
    cfg.cp2lp.adversarial.warmup_g_steps = 0   # exercise the adversarial path on step 0
    # Steps 3-4 run a reduced rollout: they are per-term diagnostics and hold several
    # graphs at once. Step 5 restores the CONFIGURED values, so the Trainer step is a real
    # memory preflight for the training job rather than a smaller thing that proves nothing
    # about whether the pilot fits on its card.
    real_max_cp = int(cfg.cp2lp.max_cp_per_step)
    real_samples = int(cfg.cp2lp.samples_per_cp)
    cfg.cp2lp.max_cp_per_step = 2
    cfg.cp2lp.samples_per_cp = 2
    if cfg.get("rollout_finetune") is not None:
        cfg.rollout_finetune.enabled = False
    if cfg.get("force_precision_f32"):
        torch.set_float32_matmul_precision("high")

    from proteinfoundation.cp2lp.data import split_roles
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import load_data_module

    # ---------------------------------------------------------------- 2. checkpoints
    print("=" * 78)
    print("2. CHECKPOINTS LOAD TOGETHER")
    model = CP2LPGenerator(cfg)
    print(f"   model built; AE loaded from autoencoder_ckpt_path")
    if not args.skip_ckpt:
        from proteinfoundation.train import _splice_pretrained_weights

        ckpt = torch.load(cfg.pretrain_ckpt_path, map_location="cpu", weights_only=False)
        spliced = _splice_pretrained_weights(model.state_dict(), ckpt["state_dict"])
        missing, unexpected = model.load_state_dict(spliced, strict=False)
        d_missing = [k for k in missing if k.startswith("discriminator.")]
        # `_splice_pretrained_weights` skips `autoencoder.*` ON PURPOSE: the flow
        # checkpoint bundles its own AE, and letting it through would overwrite the AE
        # already loaded from `autoencoder_ckpt_path` -- silently swapping the latent space
        # the whole arm depends on. So these "missing" keys are the guard working.
        ae_missing = [k for k in missing if k.startswith("autoencoder.")]
        other_missing = [k for k in missing if not k.startswith(("discriminator.", "autoencoder."))]
        print(f"   flow ckpt global_step : {ckpt.get('global_step')}")
        print(f"   discriminator keys new: {len(d_missing)} (expected -- it is new)")
        print(f"   autoencoder keys held : {len(ae_missing)} (expected -- AE comes from autoencoder_ckpt_path)")
        print(f"   OTHER missing keys    : {len(other_missing)}  <- the number that matters")
        if other_missing:
            print(f"     first 8: {other_missing[:8]}")
        print(f"   unexpected keys       : {len(unexpected)}")
        if other_missing:
            logger.warning(
                "Some flow-network weights were NOT initialised from the checkpoint. Expected "
                "only for the newly appended source-CP feature columns; anything else means "
                "the init is not the run it claims to be."
            )
        del ckpt, spliced

    device = torch.device(args.device)
    model.to(device)
    model.train()

    # ---------------------------------------------------------------- 3. one batch
    print("=" * 78)
    print("3. ONE BATCH: generator -> AE -> discriminator")
    _, datamodule = load_data_module(cfg, is_cluster_run=True)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    from proteinfoundation.utils.sample_utils import add_clean_samples

    probe = add_clean_samples(dict(batch), cfg.product_flowmatcher, model.autoencoder)
    cyclic, linear = split_roles(probe)
    print(f"   batch                 : {probe['mask'].shape[0]} rows, n={probe['mask'].shape[1]}")
    print(f"   cyclic (generator in) : {int(cyclic.sum())}")
    print(f"   linear (real LP)      : {int(linear.sum())}")
    if "dataset_source" in probe:
        from collections import Counter

        print(f"   dataset_source        : {dict(Counter(list(probe['dataset_source'])))}")
    else:
        print("   dataset_source        : NOT LOADED (real pool will not be restricted)")
    if int(cyclic.sum()) == 0 or int(linear.sum()) == 0:
        logger.error("A batch needs BOTH kinds; check dataset.unified.datamodule.source_fractions.")
        return 1

    # Run the pieces directly rather than through `training_step`: manual optimization
    # needs a Trainer, and calling the internals is what isolates a break to the rollout,
    # the decode or the discriminator instead of to Lightning plumbing. The Trainer path
    # is exercised separately in step 5.
    from proteinfoundation.cp2lp.data import slice_batch
    from proteinfoundation.cp2lp.discriminator import hinge_d_loss, hinge_g_loss
    from proteinfoundation.cp2lp.geometry import residue_lengths

    bs = probe["mask"].shape[0]
    cp_batch = slice_batch(probe, cyclic, bs)
    n_cp = cp_batch["mask"].shape[0]
    keep = min(n_cp, cfg.cp2lp.max_cp_per_step)
    cp_batch = slice_batch(cp_batch, torch.arange(keep, device=device), n_cp)

    gen_batch, decoded, _ = model._generate_from_cp(cp_batch)
    print(f"   rollout out  coors_nm : {tuple(decoded['coors_nm'].shape)}  (nsteps={nsteps_used})")
    print(f"   rollout out  seq_logit: {tuple(decoded['seq_logits'].shape)}")
    print(f"   requires_grad         : {decoded['coors_nm'].requires_grad}")

    aux, aux_metrics = model._generator_losses(gen_batch, decoded)
    print(f"   generator aux loss    : {float(aux):.4f}")
    for k in ("loss_contact", "loss_sequence", "loss_geometry", "loss_clash", "loss_open"):
        if k in aux_metrics:
            print(f"     {k:<16}: {float(aux_metrics[k]):.4f}")
    for k in ("contact_retention_frac", "seq_identity", "terminal_gap_nm", "src_terminal_gap_nm"):
        if k in aux_metrics:
            print(f"     {k:<22}: {float(aux_metrics[k]):.4f}")

    fake_feats = model._featurize(decoded, gen_batch, gen_batch["mask"].bool())
    fake_logits = model.discriminator(fake_feats)
    print(f"   discriminator logits  : {tuple(fake_logits.shape)}  mean={float(fake_logits.mean()):+.4f}")

    # Real side, through the identical decode path.
    lp_batch = slice_batch(probe, linear, bs)
    lp_mask = lp_batch["mask"].bool()
    with torch.no_grad():
        real_decoded = model._decode({dm: lp_batch["x_1"][dm].detach() for dm in model._data_modes}, lp_mask)
        real_feats = model._featurize(real_decoded, lp_batch, lp_mask)
    print(f"   real LP lengths       : {residue_lengths(lp_mask).tolist()}")
    print(f"   fake LP lengths       : {residue_lengths(gen_batch['mask'].bool()).tolist()}")

    # ---------------------------------------------------------------- 4. backward
    print("=" * 78)
    print("4. GRADIENT REACHES THE WEIGHTS")
    # Per-term attribution BEFORE the combined backward: when the combined gradient turns
    # out to be NaN, the only thing that localises it is knowing which term produced it.
    # `allow_unused` distinguishes "this term does not reach the weights at all" (None)
    # from "it reaches them and the value is bad".
    nn_params = [p for p in model.nn.parameters() if p.requires_grad]
    terms = {"contact": None, "sequence": None, "geometry": None, "clash": None, "open": None}
    per_term = model._generator_loss_terms(gen_batch, decoded) if hasattr(model, "_generator_loss_terms") else None
    if per_term:
        for name, term in per_term.items():
            if not torch.is_tensor(term) or not term.requires_grad:
                terms[name] = "no-graph"
                continue
            g = torch.autograd.grad(term, nn_params, retain_graph=True, allow_unused=True)
            g = [x for x in g if x is not None]
            if not g:
                terms[name] = "unused"
            else:
                mx = max(float(x.abs().max()) for x in g)
                nan = any(bool(torch.isnan(x).any()) for x in g)
                inf = any(bool(torch.isinf(x).any()) for x in g)
                terms[name] = f"max|g|={mx:.3e}{' NaN' if nan else ''}{' Inf' if inf else ''}"
        for name, verdict in terms.items():
            print(f"     d(loss_{name})/d(nn) : {verdict}")

    adv = hinge_g_loss(fake_logits)
    g_total = aux + adv
    g_total.backward(retain_graph=True)

    def grad_census(params, label):
        """Zero, NaN and Inf are three different failures; `nan > 0` is False, so a check
        that only asks 'is it nonzero' reports a NaN gradient as a dead one."""
        grads = [p.grad for p in params if p.grad is not None]
        n_nan = sum(int(torch.isnan(g).any()) for g in grads)
        n_inf = sum(int(torch.isinf(g).any()) for g in grads)
        finite = [g for g in grads if torch.isfinite(g).all()]
        n_nonzero = sum(int(g.abs().sum() > 0) for g in finite)
        mx = max((float(g.abs().max()) for g in finite), default=0.0)
        print(f"   {label}: {len(grads)} with grad | finite-nonzero {n_nonzero} | NaN {n_nan} | Inf {n_inf} | max|g| {mx:.3e}")
        return len(grads), n_nonzero, n_nan, n_inf

    _, n_flow, nan_flow, inf_flow = grad_census(nn_params, "flow network ")
    ae_trainable = [p for p in model.autoencoder.parameters() if p.requires_grad]
    print(f"   AE trainable params (expect 0, it is frozen): {len(ae_trainable)}")

    model.zero_grad(set_to_none=True)
    d_loss = hinge_d_loss(model.discriminator(real_feats), model.discriminator({k: v.detach() for k, v in fake_feats.items()}))
    d_loss.backward()
    print(f"   discriminator loss                          : {float(d_loss):.4f}")
    _, n_d, nan_d, _ = grad_census(list(model.discriminator.parameters()), "discriminator")
    model.zero_grad(set_to_none=True)

    if nan_flow or inf_flow:
        logger.error(
            f"Flow-network gradient is non-finite (NaN in {nan_flow} params, Inf in {inf_flow}). "
            "Read the per-term attribution above: the term whose own gradient is NaN is the "
            "one to fix. A very small --nsteps makes the rollout endpoint garbage and is a "
            "common way to manufacture this, so retry at the configured nsteps before "
            "changing a loss."
        )
        return 1
    if n_flow == 0:
        logger.error(
            "No gradient reached the flow network. The rollout is not differentiable back "
            "to the weights -- check cp2lp.rollout.grad_steps and that the losses are "
            "computed on the DECODED sample rather than on a detached copy."
        )
        return 1
    if n_d == 0:
        logger.error("No gradient reached the discriminator; the hinge loss is disconnected.")
        return 1

    # ---------------------------------------------------------------- 5. lightning wiring
    print("=" * 78)
    print("5. ONE TRAINER STEP (manual optimization, two optimizers)")

    # Steps 3-4 held graphs alive on purpose (`retain_graph=True`, so each loss term could
    # be differentiated separately). Every one of those is still resident here, and the
    # Trainer step below allocates its own rollout on top -- which OOMs a 24 GB card on the
    # smoke's own diagnostics rather than on anything training would do. Rebinding to None
    # is what actually drops them: `del locals()[...]` is a no-op inside a function.
    import gc

    decoded = gen_batch = fake_feats = real_feats = real_decoded = None
    aux = adv = g_total = d_loss = fake_logits = cp_batch = probe = lp_batch = None
    per_term = nn_params = feats = None
    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"   freed diagnostics; CUDA reserved now {torch.cuda.memory_reserved() / 2**30:.2f} GiB")

    # `val_generation` integrates a 400-step sampler several times over; that is a
    # sampling-quality check, not a wiring check, and it would dominate the smoke.
    # Turning it off leaves the validation LOSS path -- the one that crashed -- intact.
    if cfg.get("val_generation") is not None:
        if args.val_generation:
            # Exercise the generation path too, at a token budget. It is parent code, but
            # this subclass adds feature columns that a GENERATION batch must also satisfy
            # -- and a generation batch carries different keys than a training one.
            cfg.val_generation.enabled = True
            cfg.val_generation.n_batches = 1
            cfg.val_generation.n_repeat = 1
            cfg.val_generation.nsteps = 20
            cfg.val_generation.rosetta_auto_submit_every_n_vals = 0
            print("   val_generation ENABLED for this check (1 batch, 1 repeat, 20 steps)")
        else:
            cfg.val_generation.enabled = False

    model.cp2lp_max_cp = real_max_cp
    model.cp2lp_samples_per_cp = real_samples
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"   restored configured rollout load: max_cp_per_step={real_max_cp} x samples_per_cp={real_samples}")

    import lightning as L

    # Mirror `train.py`'s Trainer arguments for the settings Lightning REFUSES under
    # manual optimization. The first version of this check built a bare Trainer, passed,
    # and the real job then died instantly on `Trainer(gradient_clip_val=1.0)` -- a
    # preflight that does not construct the Trainer the way production does is not a
    # preflight for production. These must stay neutral here; the module clips and
    # accumulates itself.
    assert model.automatic_optimization is False, "CP2LPGenerator must use manual optimization"
    trainer_clip_val = None
    trainer_clip_algorithm = None

    # `accumulate_grad_batches=1` for this check, and NOT because accumulation is
    # suspect. Under manual optimization Lightning's `global_step` advances only when the
    # module calls `optimizer.step()`, which `training_step` does at
    # `(batch_idx + 1) % accum == 0`. With the configured accum of 8 and
    # `limit_train_batches` capping the epoch at one batch, the optimizer never steps,
    # `max_steps` is never reached, and the Trainer loops epochs forever -- it looks like
    # a hang while the dataloader happily churns. One batch per step makes the stopping
    # condition reachable, which is all this check needs.
    model.cfg_exp.opt.accumulate_grad_batches = 1

    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        max_steps=args.trainer_steps,
        limit_train_batches=args.trainer_steps,
        # Validation is RUN, not skipped. The first version of this check set
        # limit_val_batches=0 and num_sanity_val_steps=0, so it never exercised
        # `validation_step -> validation_step_data -> training_step(batch, -1)` -- and the
        # GAN arm then died at its first real validation, 76 minutes in. A sanity pass
        # runs that path BEFORE training, which is the cheapest place to catch it.
        limit_val_batches=1,
        num_sanity_val_steps=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=trainer_clip_val,
        gradient_clip_algorithm=trainer_clip_algorithm,
        accumulate_grad_batches=1,
    )
    print(f"   module-side clip: {model.grad_clip_val} ({model.grad_clip_algorithm}); Trainer-side: off")
    trainer.fit(model, datamodule=datamodule)
    print(f"   completed {trainer.global_step} optimizer step(s) over {args.trainer_steps} batches")
    print("   a validation pass ran (sanity check) -- training_step(batch, -1) survived")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        headroom = 100.0 * (1.0 - peak / total)
        print(f"   PEAK allocated {peak:.1f} GiB of {total:.1f} GiB  ({headroom:.0f}% headroom)")
        print(f"   at batch_size={batch_size_used}, max_cp_per_step={real_max_cp} x "
              f"samples_per_cp={real_samples}, grad_steps={model.cp2lp_grad_steps}, nsteps={nsteps_used}")
        if headroom < 15.0:
            logger.warning(
                f"Only {headroom:.0f}% headroom over {args.trainer_steps} batches. Peak memory varies "
                "with receptor size, so a batch of large receptors will OOM in real training. "
                "Lower cp2lp.max_cp_per_step (then rollout.grad_steps) before launching."
            )

    print("=" * 78)
    print("SMOKE PASSED -- the pieces fit together. Start the pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
