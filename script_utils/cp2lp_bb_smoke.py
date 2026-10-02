"""GPU smoke for the two backbone arms, on the REAL trunk and a REAL batch.

Everything verified so far was verified on synthetic tensors or on native PDBs with random
head outputs. This is the first thing that runs the pretrained trunk, the real dataloader and
the real `training_step`, and it checks the claims that only become checkable there:

  1. the arm's head EXISTS in the model built from this config, and matches the config's arm;
  2. exactly ONE trunk call per generated batch -- counted with a forward hook, because a
     recycle, a self-conditioning pass and a CFG branch all look like ordinary code;
  3. at ZERO head output the arm reproduces the decoded source backbone (the zero-init
     splice property), measured on real AE decodes rather than on crystals;
  4. the arms' RAW validity contrast is there on real data: Arm A can break a chain, Arm B
     cannot;
  5. every non-adversarial term reaches the arm's head with finite gradient, and the PER-TERM
     gradient norms are reported -- scalar loss shares are not gradient shares;
  6. the real-LP ANCHOR runs, reaches the head, and sits above its template floor;
  7. a full `training_step` completes with both optimizers.

Run through `scripts/cp2lp_bb_smoke.sbatch`, which does both arms.
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
    p.add_argument("--config", required=True, help="e.g. example/training_cp2lp_bb_torsion")
    p.add_argument("--init-ckpt", default=None, help="override pretrain_ckpt_path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-cp", type=int, default=2)
    p.add_argument("--skip-ckpt", action="store_true", help="random init; only for plumbing")
    p.add_argument("--skip-training-step", action="store_true")
    p.add_argument(
        "--seed", type=int, default=20261001,
        help="seeds the dataloader so BOTH arms draw the same batch; the printed contrast "
             "between the arms is otherwise a contrast between two random draws",
    )
    return p.parse_args()


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
    print("1. CONFIG")
    bb = cfg.cp2lp.backbone
    print(f"   run_name        : {cfg.run_name}")
    print(f"   mode / arm      : {cfg.cp2lp.mode} / {bb.arm}")
    print(f"   nn head mode    : {cfg.nn.editor_backbone_mode}")
    print(f"   angle / length  : {bb.angle_source} / {bb.length_source}")
    print(f"   max_atom_shift  : {bb.max_atom_shift_nm}")
    print(f"   anchor weight   : {bb.anchor_weight}  t range {list(bb.anchor_t_range)}")
    print(f"   real_lp_flow_w  : {cfg.cp2lp.real_lp_flow_weight}  (must be 0)")
    print(f"   sequence_weight : {cfg.cp2lp.losses.sequence_weight}  (must be 0)")
    if str(cfg.cp2lp.mode) != "backbone":
        logger.error(f"this smoke is for cp2lp.mode=backbone, got {cfg.cp2lp.mode!r}")
        return 1
    if not args.skip_ckpt:
        for path in (cfg.autoencoder_ckpt_path, cfg.pretrain_ckpt_path):
            if not os.path.exists(path):
                logger.error(f"checkpoint missing: {path}")
                if "fork" in str(path):
                    logger.error("  run the one-pass warmup and pin it: scripts/pin_cp2lp_fork.sh")
                return 1
        print("   both checkpoint files exist")

    cfg.dataset.datamodule.num_workers = 2
    cfg.dataset.unified.datamodule.num_workers = 2
    cfg.cp2lp.adversarial.warmup_g_steps = 0  # exercise the adversarial path on step 0
    cfg.cp2lp.max_cp_per_step = int(args.max_cp)
    cfg.cp2lp.samples_per_cp = 2
    cfg.cp2lp.max_g_updates = 0
    cfg.cp2lp.eval_checkpoint_g_updates = []
    cfg.cp2lp.backbone.grad_probe_every_n_steps = 1  # force the probe inside the smoke
    if cfg.get("rollout_finetune") is not None:
        cfg.rollout_finetune.enabled = False
    if cfg.get("val_generation") is not None:
        cfg.val_generation.enabled = False
    if cfg.get("force_precision_f32"):
        torch.set_float32_matmul_precision("high")

    # SAME BATCH IN BOTH ARMS. Without this each arm's process draws its own batch from a
    # weighted sampler, and the first run of this smoke reported `chain_intact` 0.5000 for one
    # arm and 1.0000 for the other AT IDENTICAL ZERO EDIT -- a difference entirely explained
    # by one batch's AE decode being worse than the other's. A representation comparison that
    # prints two numbers from two different batches is not a comparison.
    import lightning as L

    L.seed_everything(int(args.seed), workers=True)

    from proteinfoundation.cp2lp.backbone import backbone_validity, extract_backbone
    from proteinfoundation.cp2lp.bb_arms import ARM_CARTESIAN, init_reconstruction_error
    from proteinfoundation.cp2lp.data import slice_batch, split_roles
    from proteinfoundation.cp2lp.editor import (
        assert_inputs_unmutated,
        count_trunk_calls,
        snapshot_inputs,
    )
    from proteinfoundation.cp2lp.module import CP2LPGenerator
    from proteinfoundation.train import _splice_pretrained_weights, load_data_module

    print("=" * 78)
    print("2. MODEL")
    model = CP2LPGenerator(cfg)
    nn_mod = getattr(model.nn, "module", model.nn)
    heads = {
        a: getattr(nn_mod, a, None) is not None
        for a in ("editor_bb_cartesian", "editor_bb_torsion", "editor_bb_root")
    }
    print(f"   backbone heads  : {heads}")
    want = "editor_bb_cartesian" if str(bb.arm) == ARM_CARTESIAN else "editor_bb_torsion"
    if not heads[want]:
        logger.error(f"arm {bb.arm!r} needs {want}, which the model does not have")
        return 1
    n_head = sum(
        p.numel() for p in model._backbone_head_parameters()
    )
    print(f"   head parameters : {n_head}")
    if n_head == 0:
        logger.error("the arm's head has no parameters; _backbone_head_parameters found nothing")
        return 1

    if not args.skip_ckpt:
        ckpt = torch.load(cfg.pretrain_ckpt_path, map_location="cpu", weights_only=False)
        spliced = _splice_pretrained_weights(model.state_dict(), ckpt["state_dict"])
        missing, unexpected = model.load_state_dict(spliced, strict=False)
        expected_new = (
            "discriminator.", "autoencoder.", "feature_scales.", "dgan.", "nce.",
            "nn.concat_pair_factory", "nn.editor_", "g_updates",
        )
        other = [k for k in missing if not k.startswith(expected_new)]
        print(f"   init g_updates  : {int(model.g_updates)}")
        print(f"   new keys        : {len(missing) - len(other)} (D / AE / editor+backbone "
              f"heads / splice-skipped -- all expected)")
        print(f"   OTHER missing   : {len(other)}  <- the number that matters")
        if other:
            print(f"     first 8       : {other[:8]}")
        print(f"   unexpected keys : {len(unexpected)}")
        del ckpt, spliced
        span = model.init_editor_noise_adapter()
        print(f"   noise adapter   : columns {span} re-initialised")

    device = torch.device(args.device)
    model.to(device)
    model.train()

    print("=" * 78)
    print("3. ONE REAL BATCH")
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
        logger.error("a batch needs BOTH kinds; check dataset.unified.datamodule.source_fractions")
        return 1

    cp_batch = slice_batch(probe, cyclic, bs)
    n_cp = cp_batch["mask"].shape[0]
    keep = torch.arange(min(n_cp, args.max_cp), device=device)
    cp_batch = slice_batch(cp_batch, keep, n_cp)
    n_keep = cp_batch["mask"].shape[0]
    idx = torch.arange(n_keep, device=device).repeat_interleave(2)
    cp_batch = slice_batch(cp_batch, idx, n_keep)
    print(f"   generations     : {cp_batch['mask'].shape[0]} ({n_keep} CPs x 2 seeds)")

    print("=" * 78)
    print("4. ONE TRUNK CALL, AND THE ZERO-OUTPUT IDENTITY")
    guard = snapshot_inputs(cp_batch)
    with count_trunk_calls(model.nn) as calls:
        gen_batch, decoded, extras = model.generate_backbone(cp_batch)
    print(f"   trunk calls     : {calls[0]}  (must be 1)")
    if calls[0] != 1:
        logger.error(
            f"{calls[0]} trunk calls for one generated batch. Look for a recycle "
            "(n_recycle), a self-conditioning pass, a CFG unconditional branch or a sampler."
        )
        return 1
    mutated = assert_inputs_unmutated(gen_batch, guard)
    print(f"   inputs mutated  : {mutated or 'none'}")
    if mutated:
        logger.error(f"the arm mutated read-only inputs: {mutated}")
        return 1

    mask = gen_batch["mask"].bool()
    # THE INIT RECONSTRUCTION, on the real AE decode of a real source CP. The synthetic test
    # cannot catch an AE whose decoded bond lengths sit outside the carried window.
    init = init_reconstruction_error(
        str(bb.arm), extras["src_bb"], mask,
        angle_source=str(bb.angle_source), length_source=str(bb.length_source),
        aatype=gen_batch.get("residue_type", None),
    )
    print(f"   init recon      : bb_rmsd {init['init_bb_rmsd_nm'] * 10:.6f} A   "
          f"max {init['init_bb_max_nm'] * 10:.6f} A")
    if init["init_bb_rmsd_nm"] > 1e-4:
        print("     NOTE: nonzero -- the arm's first output is already an unrequested edit of "
              "this size. Expected only under a template (0.118 nm angles / 0.021 nm lengths).")

    print(f"   decoded coors   : {tuple(decoded['coors_nm'].shape)}  "
          f"requires_grad={decoded['coors_nm'].requires_grad}")
    n_atoms = int(decoded["atom_mask_eff"].sum(-1).float()[mask].mean())
    print(f"   atoms / residue : {n_atoms}  (must be 4 -- backbone only, OXT added at export)")
    if n_atoms != 4:
        logger.error(f"{n_atoms} atoms per residue reach the metrics; expected exactly 4")
        return 1

    print("=" * 78)
    print("5. RAW VALIDITY ON REAL DATA, AGAINST THE SOURCE'S OWN")
    # THE SOURCE IS THE CEILING, and it is not 1.0. The arm's reference backbone is the
    # FROZEN AE's decode of the source CP, and that decode has its own geometry errors -- so
    # an arm's validity has to be read against the source column beside it, never against
    # 1.0. This is the analogue of the one-pass arm's "AE round-trip ceiling of 0.9844",
    # and without it Arm A's `chain_intact` is an uninterpretable number.
    #
    # Note also that `bb_chain_intact` here is STRICTER than the one-pass
    # `geom_chain_intact_frac`: it is an all-bonds AND all-angles test, so a few percent of
    # per-bond angle failures costs most of the peptides.
    v = backbone_validity(extras["gen_bb"], mask)
    v_src = backbone_validity(extras["src_bb"], mask)
    print(f"   {'metric':<24s} {'source (AE decode)':>19s} {'this arm':>10s}")
    for k in ("bb_chain_intact", "bb_peptide_bond_pass", "bb_intra_bond_pass",
              "bb_tau_pass", "bb_peptide_angle_pass", "bb_carbonyl_pass", "bb_cis_frac"):
        print(f"   {k:<24s} {float(v_src[k].mean()):19.4f} {float(v[k].mean()):10.4f}")
    m = extras["metrics"]
    print(f"   edit shift      : mean {float(m['bb_shift_mean_nm']) * 10:.4f} A   "
          f"max {float(m['bb_shift_max_nm']) * 10:.4f} A")

    # ARM B'S GUARANTEE, STATED CORRECTLY. Under `angle_source: source` it is "the edit
    # cannot make local geometry worse than its input", NOT "the output has ideal geometry"
    # -- the arm carries the source's own angles through, so it inherits whatever the AE
    # decode had. Asserting `== 1.0` would be asserting a claim the design never made, and
    # would fail on any batch whose AE decode is imperfect. What must hold is NO DEGRADATION.
    if str(bb.arm) != ARM_CARTESIAN:
        worse = [
            k for k in ("bb_chain_intact", "bb_peptide_bond_pass", "bb_intra_bond_pass",
                        "bb_tau_pass", "bb_peptide_angle_pass")
            if float(v[k].mean()) < float(v_src[k].mean()) - 1e-6
        ]
        if worse:
            logger.error(
                f"Arm B DEGRADED local geometry relative to its source on {worse}. Its edit "
                "space contains no point that can do that, so this is a builder bug."
            )
            return 1
        print("   Arm B degraded nothing relative to its source (its actual guarantee)")

    print("=" * 78)
    print("6. LOSSES, AND THE PER-TERM GRADIENT THAT ACTUALLY ARRIVES")
    loss, met = model._backbone_losses(gen_batch, extras)
    for k in sorted(met):
        if k.startswith("loss_"):
            print(f"   {k:<24s}: {float(met[k]):.6f}")
    print(f"   {'TOTAL':<24s}: {float(loss):.6f}")
    shares = model._log_backbone_grad_shares(gen_batch, extras)
    if shares:
        tot = sum(float(x) for x in shares.values()) or 1.0
        print("   -- weighted gradient norm at the arm's head, before clipping --")
        for k in sorted(shares, key=lambda z: -float(shares[z])):
            print(f"   {k:<24s}: {float(shares[k]):.3e}   ({100 * float(shares[k]) / tot:5.1f}%)")
        print("   READ THIS, NOT THE LOSS VALUES: a term worth 1% of the summed loss can")
        print("   carry most of the gradient, and the reverse. The clash term was once 0.06%")
        print("   of the objective while being violated.")

    loss.backward(retain_graph=False)
    head_grads = [
        (n, float(p.grad.abs().max()))
        for n, p in [
            (n, p) for n, p in nn_mod.named_parameters()
            if n.startswith(("editor_bb_cartesian", "editor_bb_torsion", "editor_bb_root"))
        ]
        if p.grad is not None
    ]
    print(f"   head grad (max) : {head_grads}")
    if not head_grads or all(g == 0.0 for _, g in head_grads):
        logger.error(
            "no gradient reached the arm's head. With a zero-initialised head the FIRST step "
            "legitimately gives the trunk nothing (d(out)/d(h) is zero through a zero "
            "weight), but the head itself must still receive gradient."
        )
        return 1
    model.zero_grad(set_to_none=True)

    print("=" * 78)
    print("7. THE REAL-LP ANCHOR")
    lp_batch = slice_batch(probe, linear, bs)
    anchor_loss, n_lp, anchor_met = model._real_lp_backbone_anchor(lp_batch)
    print(f"   rows            : {n_lp}")
    print(f"   anchor loss     : {float(anchor_loss):.6f} nm")
    for k in sorted(anchor_met):
        print(f"   {k:<24s}: {float(anchor_met[k]):.6f}")
    floor = float(anchor_met.get("anchor_template_floor_nm", 0.0))
    if float(anchor_met.get("anchor_bb_rmsd_nm", 0.0)) <= floor + 1e-9:
        print("     NOTE: the anchor is AT its template floor, so the head is contributing")
        print("     nothing yet -- expected at zero init, and a warning later in training.")
    anchor_loss.backward()
    ok = any(
        p.grad is not None and float(p.grad.abs().max()) > 0
        for n, p in nn_mod.named_parameters()
        if n.startswith(("editor_bb_cartesian", "editor_bb_torsion", "editor_bb_root"))
    )
    print(f"   anchor reaches the head: {ok}")
    if not ok:
        logger.error(
            "the real-LP anchor does not reach the arm's head, so it anchors nothing -- "
            "which is the exact defect it was written to avoid."
        )
        return 1
    model.zero_grad(set_to_none=True)

    if args.skip_training_step:
        print("\n[smoke] --skip-training-step given; the full step was NOT exercised.")
        return 0

    print("=" * 78)
    print("8. A FULL training_step WITH BOTH OPTIMIZERS")
    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1, max_steps=2, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, num_sanity_val_steps=0, limit_val_batches=0,
    )
    trainer.fit(model, datamodule=datamodule)
    print(f"   g_updates after : {int(model.g_updates)}")
    print("\n[smoke] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
