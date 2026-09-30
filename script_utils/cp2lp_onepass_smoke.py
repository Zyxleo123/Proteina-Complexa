"""Compact preflight for the ONE-PASS CP -> LP editor. One small batch, then exit.

Not a validation framework and not a quality gate -- the pilot is allowed to start with
poor metrics. These checks exist because each one covers a failure that TRAINS FINE and
answers a different question than the one asked:

1. CONFIG AND CHECKPOINTS. The arm switch is exclusive, the residual heads and the noise
   feature are wired, the AE/flow pairing is the intended one, and the fork loads.
2. ONE G AND D UPDATE. Finite outputs, finite losses, finite gradients; the AE frozen;
   live gradient into the editor. A zero-initialised residual head blocks trunk gradient on
   the FIRST step by construction -- expected, and checked again after the head has moved,
   which is the version of the check that means something.
3. EXACTLY ONE EDITOR-TRUNK CALL per generated batch, in training AND in export, counted
   with a forward hook. This is the claim the whole mode rests on and the easiest one to
   lose: a self-conditioning pass, a recycle, a CFG unconditional branch and an ODE step
   all look like ordinary code. AE and discriminator calls do not pass through the trunk
   and so do not count.
4. FRAME SENSITIVITY, ARM EXCLUSIVITY, TRAIN/EXPORT AGREEMENT, and that the source and
   receptor tensors were not mutated.

The frame check is a RELATIVE measurement, deliberately. The trunk consumes absolute CA
coordinates through a Linear and is rotation-AUGMENTED, not rotation-equivariant, so "the
residual rotates as a vector" is not a property this architecture has and a test asserting
it would be wrong. What is checkable is that the editor's residual is no more
frame-sensitive than the pretrained velocity head it sits beside, measured on the same
pair of forwards -- which is what catches the failure that matters here: injecting
untransformed random Cartesian noise would make the output move under rotation
*independently of the input*, far beyond the baseline.

Run (GPU):
    .venv/bin/python script_utils/cp2lp_onepass_smoke.py --config example/training_cp2lp_onepass_gan
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="example/training_cp2lp_onepass_gan")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--max-cp",
        type=int,
        default=2,
        help="Source CPs in the smoke batch. 2 with 2 noise seeds each is the plan's "
        "minimum: it is what makes the within-CP diversity and the train/export agreement "
        "check exercisable at all.",
    )
    p.add_argument(
        "--largest-crop",
        action="store_true",
        default=True,
        help="Keep the largest receptor in the drawn batch rather than the first. Peak "
        "memory is O(n_target^2) through the pair features, so a memory check on a small "
        "receptor measures the wrong thing.",
    )
    p.add_argument("--skip-ckpt", action="store_true", help="Shapes only; do not load the init.")
    p.add_argument(
        "--init-ckpt",
        default=None,
        help="Override `pretrain_ckpt_path`. The smoke runs BEFORE the common fork exists "
        "-- it is what gates on whether the fork is worth producing -- so it has to be "
        "able to load the lpmix init instead. Which checkpoint it loads is irrelevant to "
        "every check here, as long as the AE/flow pairing is valid, because these are "
        "wiring checks and not quality checks.",
    )
    p.add_argument(
        "--frame-tol",
        type=float,
        default=1.5,
        help="How many times the BASE velocity head's frame sensitivity the editor's "
        "residual may show. A relative bound; see the module docstring for why there is no "
        "absolute one.",
    )
    return p.parse_args()


class _CaptureLogger(Logger):
    """An in-memory Lightning logger, so the ADDON MAGNITUDES survive the smoke.

    Without a logger, every `self.log(...)` call in `training_step` warns and discards its
    value -- which is exactly the wrong thing to lose here. `ar1.lambda: 10` and
    `nce.weight: 0.1` are PILOT values, and the question they beg is whether the weighted
    term is commensurate with the hinge loss it is added to. A term 100x too large silently
    becomes the objective; one 100x too small is inert and the arm measures nothing. Both
    look like a passing smoke.

    Writes nothing: no files, no wandb run, nothing to clean up. It keeps the last value of
    every key, which is all the preflight needs.

    SUBCLASSES `Logger` rather than duck-typing it. A plain object with `log_metrics` and
    `log_hyperparams` looks sufficient and is not -- Lightning also calls `log_graph`
    during `_log_hyperparams`, and `trainer.fit` dies with an `AttributeError` several
    seconds in. Subclassing inherits every no-op default, so the surface cannot drift out
    from under this again on a Lightning upgrade.
    """

    def __init__(self):
        super().__init__()
        self.metrics: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "cp2lp_smoke_capture"

    @property
    def version(self) -> str:
        return "0"

    @property
    def save_dir(self) -> None:
        return None

    @rank_zero_only
    def log_metrics(self, metrics, step=None) -> None:
        for k, v in metrics.items():
            try:
                self.metrics[k] = float(v)
            except (TypeError, ValueError):
                pass

    @rank_zero_only
    def log_hyperparams(self, *args, **kwargs) -> None:
        pass


#: What to print per arm, and against what. The comparison that matters is the weighted
#: addon term against `loss_d` (for the D-side addons) or `loss_g_total` (for the G-side
#: one) -- an absolute value says nothing about whether it will dominate.
ADDON_REPORT = {
    "ar1": (
        ["ar1_penalty", "loss_d_ar1_weighted", "ar1_logit_shift_abs", "ar1_mean_sigma",
         "ar1_n_perturbed_cols", "scales_frozen", "scales_n_examples"],
        "loss_d_ar1_weighted",
        "loss_d",
    ),
    "noise": (
        ["dgan_p", "dgan_eligible_levels", "dgan_max_level_std", "dgan_level_real_mean",
         "dgan_sign_real", "d_acc", "d_acc_clean", "scales_frozen"],
        None,
        None,
    ),
    "nce": (
        ["nce_loss", "loss_nce_weighted", "nce_n_scored", "nce_positions_per_example",
         "nce_n_skipped_examples"],
        "loss_nce_weighted",
        "loss_g_total",
    ),
}


def _report_addon(addon: str, captured: dict[str, float]) -> None:
    """Prints this arm's addon diagnostics and the ratio that decides if it is scaled sanely."""
    if addon not in ADDON_REPORT:
        print("   addon          : none -- nothing extra to report")
        return
    keys, num, den = ADDON_REPORT[addon]

    def get(name):
        for k in (f"train/cp2lp_{name}_step", f"train/cp2lp_{name}", f"train/cp2lp_{name}_epoch"):
            if k in captured:
                return captured[k]
        return None

    print(f"   addon `{addon}` diagnostics (last logged value):")
    for name in keys:
        v = get(name)
        print(f"     {name:<26}: {'-' if v is None else f'{v:.6g}'}")
    if num and den:
        a, b = get(num), get(den)
        if a is None or b is None or b == 0:
            print(f"     ratio {num}/{den}: not computable from this run")
        else:
            r = abs(a) / abs(b)
            label = "DOMINATES" if r > 10 else "very small" if r < 1e-3 else "commensurate"
            print(f"     |{num}| / |{den}|  = {r:.4g}   <- {label} AT INITIALISATION")
            if addon == "ar1":
                # Do not let a step-6 number be read as a verdict on the addon. An R1-type
                # penalty measures how SHARP the discriminator is, and a freshly
                # initialised D is nearly constant in its inputs -- there is nothing for it
                # to penalise yet, so a tiny value here is the expected reading and not
                # evidence the weight is wrong. It should grow by orders of magnitude as D
                # sharpens (v1's D reached logits of +-1.5-1.8 within a few thousand steps).
                print("     aR1 is QUADRATIC in D's logit shift, and D is barely trained here,")
                print("     so a tiny ratio at step 6 is EXPECTED and is not a verdict on lambda.")
                print("     RE-READ `loss_d_ar1_weighted / loss_d` at the g=1300 milestone: if it")
                print("     is still below ~1e-3 there, arm A is inert and its result would be a")
                print("     statement about the weight, not about approximate R1.")
            elif r > 10:
                print("     A term this large IS the objective now; the arm would measure the")
                print("     weight rather than the addon.")
            elif r < 1e-3:
                print("     Check this again at the g=1300 milestone before trusting the arm.")
            print("     (a PILOT weight -- if it must change, change it for this arm only and")
            print("      before that arm starts, or the comparison stops being matched)")


def _largest_receptor_index(batch: dict) -> int:
    tm = batch.get("seq_target_mask", None)
    if tm is None:
        return 0
    return int(tm.bool().sum(dim=-1).argmax())


def main() -> int:  # noqa: C901 -- a preflight is a checklist; splitting it hides the order
    args = parse_args()
    load_dotenv(".env")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name=args.config)
    OmegaConf.set_struct(cfg, False)
    if args.init_ckpt:
        cfg.pretrain_ckpt_path = args.init_ckpt

    print("=" * 78)
    print("1. CONFIG AND CHECKPOINTS")
    print(f"   run_name        : {cfg.run_name}")
    print(f"   mode            : {cfg.cp2lp.mode}")
    print(f"   addon           : {cfg.cp2lp.addon}")
    print(f"   adversarial     : {cfg.cp2lp.adversarial.enabled}")
    print(f"   AE (defines z)  : {cfg.autoencoder_ckpt_path}")
    print(f"   init            : {cfg.pretrain_ckpt_path}"
          f"{'  (OVERRIDDEN -- the fork does not exist yet)' if args.init_ckpt else '  (the common fork)'}")
    print(f"   residual heads  : {cfg.nn.get('enable_editor_residual')}")
    print(f"   feats_seq[-1]   : {cfg.nn.feats_seq[-1]}")
    print(f"   real_lp_flow_w  : {cfg.cp2lp.real_lp_flow_weight}")
    print(f"   max_g_updates   : {cfg.cp2lp.max_g_updates}")
    if str(cfg.cp2lp.mode) != "onepass":
        logger.error("This smoke is for cp2lp.mode=onepass; use cp2lp_smoke.py for the rollout arms.")
        return 1
    for path in (cfg.autoencoder_ckpt_path, cfg.pretrain_ckpt_path):
        if not os.path.exists(path):
            logger.error(f"Checkpoint missing: {path}")
            if "fork" in str(path):
                logger.error("  Run the warmup and pin it: scripts/pin_cp2lp_fork.sh")
            return 1
    print("   both checkpoint files exist")

    # ---- CHECK: the arm switch is exclusive and self-consistent.
    from proteinfoundation.cp2lp.addons import ADDONS, describe_arm, validate_addon

    try:
        addon = validate_addon(cfg.cp2lp.addon, bool(cfg.cp2lp.adversarial.enabled))
    except ValueError as exc:
        logger.error(f"Arm exclusivity check FAILED: {exc}")
        return 1
    print(f"   arm             : {describe_arm(addon, bool(cfg.cp2lp.adversarial.enabled))}")
    for bad in ("both", "ar1+noise", "nce,ar1"):
        try:
            validate_addon(bad, True)
        except ValueError:
            continue
        logger.error(f"arm exclusivity is not enforced: {bad!r} was accepted; legal set is {ADDONS}")
        return 1
    print(f"   exclusivity     : any addon outside {ADDONS} is rejected")

    cfg.dataset.datamodule.num_workers = 2
    cfg.dataset.unified.datamodule.num_workers = 2
    cfg.cp2lp.adversarial.warmup_g_steps = 0  # exercise the adversarial path on step 0
    cfg.cp2lp.max_cp_per_step = int(args.max_cp)
    cfg.cp2lp.samples_per_cp = 2
    # The scales must be available in a one-batch smoke, or the two noise-related addons
    # would both no-op and the smoke would pass without exercising them.
    cfg.cp2lp.feature_scales.min_examples = 1
    cfg.cp2lp.max_g_updates = 0  # never stop inside the smoke
    cfg.cp2lp.eval_checkpoint_g_updates = []
    if cfg.get("rollout_finetune") is not None:
        cfg.rollout_finetune.enabled = False
    if cfg.get("val_generation") is not None:
        cfg.val_generation.enabled = False
    if cfg.get("force_precision_f32"):
        torch.set_float32_matmul_precision("high")

    from proteinfoundation.cp2lp.data import slice_batch, split_roles
    from proteinfoundation.cp2lp.editor import (
        assert_inputs_unmutated,
        count_trunk_calls,
        rot_trans_consistency,
        sample_editor_noise,
        snapshot_inputs,
    )
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import _splice_pretrained_weights, load_data_module

    model = CP2LPGenerator(cfg)
    if not args.skip_ckpt:
        ckpt = torch.load(cfg.pretrain_ckpt_path, map_location="cpu", weights_only=False)
        spliced = _splice_pretrained_weights(model.state_dict(), ckpt["state_dict"])
        missing, unexpected = model.load_state_dict(spliced, strict=False)
        # Keys that are SUPPOSED to be absent from a flow checkpoint, so that "OTHER
        # missing" means what its label says. `nn.concat_pair_factory` is in
        # `_splice_pretrained_weights`' own `skip_prefixes`; `g_updates` and the editor
        # heads are new here; D / AE / addon state is new or deliberately withheld.
        expected_new = (
            "discriminator.",
            "autoencoder.",
            "feature_scales.",
            "dgan.",
            "nce.",
            "nn.concat_pair_factory",
            "nn.editor_",
            "g_updates",
        )
        other = [k for k in missing if not k.startswith(expected_new)]
        print(f"   init global_step: {ckpt.get('global_step')}")
        print(f"   init g_updates  : {int(model.g_updates)}  <- the arms inherit the fork's count")
        print(f"   new keys        : {len(missing) - len(other)} (D / AE / addon / editor heads / "
              f"splice-skipped -- all expected)")
        print(f"   OTHER missing   : {len(other)}  <- the number that matters")
        if other:
            print(f"     first 8: {other[:8]}")
        print(f"   unexpected keys : {len(unexpected)}")
        del ckpt, spliced
        # The noise adapter, exactly where train.py applies it.
        span = model.init_editor_noise_adapter()
        print(f"   noise adapter   : columns {span} of init_repr_factory.linear_out re-initialised")
        if span is None:
            logger.error("The noise adapter did not initialise: `editor_noise_seq` is not in feats_seq.")
            return 1

    device = torch.device(args.device)
    model.to(device)
    model.train()

    # ---------------------------------------------------------------- 2. one batch
    print("=" * 78)
    print("2. ONE SMALL BATCH: editor -> AE -> discriminator")
    _, datamodule = load_data_module(cfg, is_cluster_run=True)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    from proteinfoundation.utils.sample_utils import add_clean_samples

    probe = add_clean_samples(dict(batch), cfg.product_flowmatcher, model.autoencoder)
    cyclic, linear = split_roles(probe)
    bs = probe["mask"].shape[0]
    print(f"   batch           : {bs} rows, n={probe['mask'].shape[1]}")
    print(f"   cyclic / linear : {int(cyclic.sum())} / {int(linear.sum())}")
    if int(cyclic.sum()) == 0 or int(linear.sum()) == 0:
        logger.error("A batch needs BOTH kinds; check dataset.unified.datamodule.source_fractions.")
        return 1

    cp_batch = slice_batch(probe, cyclic, bs)
    n_cp = cp_batch["mask"].shape[0]
    if args.largest_crop and n_cp > args.max_cp:
        # Largest receptor first, so the memory number in step 5 is an envelope rather than
        # one lucky draw. Pair features are O(n_target^2) and the crop runs to 256 residues.
        order = probe["seq_target_mask"][cyclic].bool().sum(dim=-1).argsort(descending=True)
        keep_idx = order[: args.max_cp]
    else:
        keep_idx = torch.arange(min(n_cp, args.max_cp), device=device)
    cp_batch = slice_batch(cp_batch, keep_idx.to(device), n_cp)
    n_keep = cp_batch["mask"].shape[0]
    tgt_res = int(cp_batch["seq_target_mask"].bool().sum(dim=-1).max())
    print(f"   source CPs kept : {n_keep} (largest receptor in batch: {tgt_res} residues)")
    idx = torch.arange(n_keep, device=device).repeat_interleave(2)
    cp_batch = slice_batch(cp_batch, idx, n_keep)
    print(f"   with 2 seeds/CP : {cp_batch['mask'].shape[0]} generations")

    # ---- CHECK: exactly one trunk call, and the inputs are not mutated.
    guard = snapshot_inputs(cp_batch)
    with count_trunk_calls(model.nn) as calls:
        gen_batch, decoded, extras = model.generate_onepass(cp_batch)
    print(f"   trunk calls     : {calls[0]}  (must be 1)")
    if calls[0] != 1:
        logger.error(
            f"{calls[0]} editor-trunk calls for one generated batch. A one-pass editor must "
            "call the network exactly once -- look for a recycle (n_recycle), a "
            "self-conditioning pass, a CFG unconditional branch or a sampler step."
        )
        return 1
    mutated = assert_inputs_unmutated(gen_batch, guard)
    print(f"   inputs mutated  : {mutated or 'none'}")
    if mutated:
        logger.error(f"The editor mutated read-only inputs: {mutated}")
        return 1
    print(f"   decoded coors   : {tuple(decoded['coors_nm'].shape)}  requires_grad={decoded['coors_nm'].requires_grad}")
    print(f"   residual        : dCA rms {float(extras['metrics']['delta_ca_rms_nm']):.3e} nm, "
          f"dz rms {float(extras['metrics']['delta_latent_rms']):.3e}")
    if float(extras["metrics"]["delta_ca_rms_nm"]) != 0.0:
        logger.warning(
            "The residual is nonzero on the first forward. Expected ONLY if the init is a "
            "fork that has already trained; from a fresh zero-initialised head it should be "
            "exactly 0, i.e. the identity edit."
        )

    aux, aux_metrics = model._generator_losses(gen_batch, decoded)
    print(f"   generator aux   : {float(aux):.4f}")
    for k in ("loss_contact", "loss_sequence", "loss_geometry", "loss_clash", "loss_open"):
        if k in aux_metrics:
            print(f"     {k:<16}: {float(aux_metrics[k]):.4f}")
    for k in ("contact_retention_frac", "seq_identity", "terminal_gap_nm", "src_terminal_gap_nm"):
        if k in aux_metrics:
            print(f"     {k:<22}: {float(aux_metrics[k]):.4f}")
    if not torch.isfinite(aux):
        logger.error("The generator's non-adversarial loss is not finite.")
        return 1

    # ---------------------------------------------------------------- 3. gradients
    print("=" * 78)
    print("3. GRADIENTS")
    nn_params = [p for p in model.nn.parameters() if p.requires_grad]
    head = [p for p in model.nn.editor_ca_residual.parameters()] + [
        p for p in model.nn.editor_latents_residual.parameters()
    ]
    aux.backward(retain_graph=True)

    def census(params, label):
        grads = [p.grad for p in params if p.grad is not None]
        n_nan = sum(int(torch.isnan(g).any()) for g in grads)
        n_inf = sum(int(torch.isinf(g).any()) for g in grads)
        finite = [g for g in grads if torch.isfinite(g).all()]
        nz = sum(int(g.abs().sum() > 0) for g in finite)
        mx = max((float(g.abs().max()) for g in finite), default=0.0)
        print(f"   {label:<16}: {len(grads)} with grad | nonzero {nz} | NaN {n_nan} | Inf {n_inf} | max|g| {mx:.3e}")
        return nz, n_nan, n_inf

    nz_head, nan_h, inf_h = census(head, "residual heads")
    nz_trunk, nan_t, inf_t = census([p for p in nn_params if not any(p is q for q in head)], "trunk (rest)")
    ae_trainable = [p for p in model.autoencoder.parameters() if p.requires_grad]
    print(f"   AE trainable    : {len(ae_trainable)} (must be 0 -- the AE is frozen in every arm)")
    if ae_trainable:
        logger.error("Autoencoder parameters are trainable; the AE must stay frozen.")
        return 1
    if nan_h or inf_h or nan_t or inf_t:
        logger.error("Non-finite gradient in the editor.")
        return 1
    if nz_head == 0:
        logger.error("No gradient reached the residual heads -- the loss is not connected to the edit.")
        return 1
    if nz_trunk == 0:
        print("   trunk gradient is zero, which is EXPECTED from a zero-initialised head; "
              "re-checking after one head update")
        with torch.no_grad():
            for p in head:
                p.add_(torch.randn_like(p) * 1e-3)
        model.zero_grad(set_to_none=True)
        _, dec2, _ = model.generate_onepass(cp_batch)
        aux2, _ = model._generator_losses(gen_batch, dec2)
        aux2.backward()
        nz_trunk2, _, _ = census([p for p in nn_params if not any(p is q for q in head)], "trunk (after)")
        if nz_trunk2 == 0:
            logger.error(
                "The trunk still receives no gradient after the residual head moved. The "
                "edit is disconnected from the network, not merely zero-initialised."
            )
            return 1
    model.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------- 4. structure
    print("=" * 78)
    print("4. FRAME SENSITIVITY, TRAIN/EXPORT AGREEMENT")

    # Fixed INVARIANT noise, reused across both frames: any difference is the trunk's
    # coordinate handling and not a different random draw.
    ref = cp_batch["x_1"][model._data_modes[0]]
    z = sample_editor_noise(
        cp_batch["mask"].shape[0], cp_batch["mask"].shape[1], model.onepass_noise_channels, ref.device, ref.dtype
    )
    gb, _, _ = model.generate_onepass(cp_batch, noise=z)
    frame = rot_trans_consistency(model, gb, model._data_modes, seed=0)
    for k in sorted(frame):
        print(f"   {k:<26}: {frame[k]:.4e}" if k != "noise_unchanged" else f"   {k:<26}: {frame[k]:.0f}")
    if frame.get("noise_unchanged", 1.0) != 1.0:
        logger.error("The 'invariant' noise changed under a rigid motion; it is not invariant.")
        return 1
    ed, base = frame.get("editor_delta_rel_dev"), frame.get("base_velocity_rel_dev")
    if ed is not None and base is not None:
        ratio = ed / max(base, 1e-12)
        print(f"   editor / base frame sensitivity: {ratio:.3f}x (tolerance {args.frame_tol}x)")
        if ratio > args.frame_tol:
            logger.error(
                f"The editor's residual is {ratio:.2f}x more frame-sensitive than the "
                "pretrained velocity head on the same forwards. The trunk is only "
                "rotation-AUGMENTED so neither is exactly equivariant, but the editor "
                "should not be worse -- suspect a non-invariant input reaching the noise "
                "block, or a point-valued head being read as a displacement."
            )
            return 1

    # ---- CHECK: train and export agree given identical weights, noise and no dropout.
    model.eval()
    with torch.no_grad():
        _, dec_train, _ = model.generate_onepass(cp_batch, noise=z)
        with count_trunk_calls(model.nn) as ecalls:
            # nsteps=200 on purpose: the export path must IGNORE it, not honour it.
            exported = model.generate_lp_for_cp(dict(cp_batch), nsteps=200, samples_per_cp=1)
    model.train()
    print(f"   export trunk calls: {ecalls[0]}  (must be 1 -- nsteps=200 must be IGNORED)")
    if ecalls[0] != 1:
        logger.error(
            f"Export made {ecalls[0]} trunk calls. It must go through `generate_onepass` and "
            "must never fall through to the 200-step sampler."
        )
        return 1
    print("   (export draws its own noise, so its coordinates differ from the fixed-noise "
          "train pass by design; what is checked is the CALL COUNT and the code path)")
    print(f"   export decoded  : {tuple(exported['decoded']['coors_nm'].shape)}")

    # ---------------------------------------------------------------- 5. one G/D update
    print("=" * 78)
    print("5. ONE TRAINER STEP (manual optimization, two optimizers, this arm's addon live)")
    import gc

    # Rebinding to None is what actually drops these: `del locals()[...]` is a no-op
    # inside a function, and the Trainer step below allocates on top of whatever is
    # still resident.
    decoded = gen_batch = extras = aux = z = gb = exported = dec_train = None  # noqa: F841
    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    import lightning as L

    assert model.automatic_optimization is False, "CP2LPGenerator must use manual optimization"
    # accum 1 so the optimizer actually steps inside a handful of batches; with the
    # configured 8 and a capped epoch the optimizer never steps, `max_steps` is never
    # reached and the Trainer loops epochs forever -- it looks like a hang.
    model.cfg_exp.opt.accumulate_grad_batches = 1
    capture = _CaptureLogger()
    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        max_steps=6,
        limit_train_batches=6,
        # Validation IS run: `training_step(batch, -1)` delegates to the parent, and that
        # path is what killed the first GAN arm 76 minutes in. A sanity pass runs it before
        # training, which is the cheapest place to catch it.
        limit_val_batches=1,
        num_sanity_val_steps=1,
        enable_checkpointing=False,
        # A capturing logger, not `False`: see `_CaptureLogger` -- `logger=False` discards
        # every addon metric, which is the one class of number this step exists to show.
        logger=capture,
        # 1, not the default 50. The smoke runs 6 steps, so at the default Lightning
        # buffers every metric and flushes none of them -- the capture comes back empty and
        # the addon report prints dashes, which reads as "the addon is inert" when it is
        # only unflushed.
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
        accumulate_grad_batches=1,
    )
    trainer.fit(model, datamodule=datamodule)
    print(f"   global_step={trainer.global_step}  g_updates={int(model.g_updates)}")
    for name in ("loss_g_total", "loss_d", "d_acc", "d_match_frac", "grad_norm_preclip_g"):
        for k in (f"train/cp2lp_{name}_step", f"train/cp2lp_{name}"):
            if k in capture.metrics:
                print(f"     {name:<26}: {capture.metrics[k]:.6g}")
                break
    _report_addon(addon, capture.metrics)
    print("   NOTE: global_step counts G+D optimizer steps under manual optimization. "
          "g_updates is the comparable one.")
    if model.adv_enabled and trainer.global_step <= int(model.g_updates):
        logger.warning(
            "global_step did not exceed g_updates in an adversarial arm; the discriminator "
            "may not be stepping (check cp2lp_d_match_frac)."
        )
    print("   a validation pass ran (sanity check) -- training_step(batch, -1) survived")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        headroom = 100.0 * (1.0 - peak / total)
        print(f"   PEAK allocated {peak:.1f} GiB of {total:.1f} GiB ({headroom:.0f}% headroom) "
              f"at max_cp={args.max_cp} x 2 seeds, largest receptor {tgt_res} residues")
        if headroom < 15.0:
            logger.warning(
                f"Only {headroom:.0f}% headroom. Peak memory is O(n_target^2) and varies with "
                "receptor size. Lower cp2lp.max_cp_per_step -- AND APPLY THE SAME CHANGE TO "
                "ALL FIVE ARMS BEFORE LAUNCH, or the comparison stops being single-variable."
            )

    print("=" * 78)
    print("SMOKE PASSED -- the one-pass editor is wired correctly. Fix nothing else; start the pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
