"""``CP2LPGenerator`` -- the adversarially-trained CP -> LP generator.

A ``Proteina`` subclass, so the flow network, the frozen mixed AE, the samplers and the
checkpoint plumbing are all inherited rather than reimplemented. What it adds is a second
training branch and a second optimizer.

TWO GENERATOR MODES, SELECTED BY ``cp2lp.mode``.

``onepass`` (the direct editor, and the one the five-arm comparison uses)
    One network evaluation per generated batch. The encoded source CP is fed through the
    ordinary coordinate/latent input channels, a pair of zero-initialised residual heads
    emit an edit displacement, the sum is decoded by the frozen AE, and that is the
    sample. ``t`` is pinned at a configured constant and is an architecture input, not a
    sampling coordinate: no ODE, no velocity-to-clean conversion, no CFG, no
    self-conditioning pass, and the same function serves training and export. See
    :mod:`proteinfoundation.cp2lp.editor`. The real-LP flow-matching term is OFF in this
    mode -- it supervises a velocity field the residual heads are not connected to -- and
    nothing replaces it; real bound LPs remain the discriminator's real class.

``rollout`` (legacy, still the default, unchanged)
    The original generator: a short differentiable ODE trajectory with DRaFT-K gradients
    and export from a 200-step sampler. Kept so the three measured generations in
    ``docs/old_docs/README_CP2LP.md`` stay reproducible. It is not part of the five-arm comparison
    -- the generator objective differs too, so it is historical context rather than a
    clean architecture ablation.

THREE ADD-ONS, AT MOST ONE AT A TIME, selected by ``cp2lp.addon`` and all default-off.
``ar1`` and ``noise`` are discriminator-side; ``nce`` is generator-side. Every one of them
is measured against the plain one-pass GAN, never against another addon. See
:mod:`proteinfoundation.cp2lp.addons`.

ONE OPTIMIZER STEP, THREE TERMS (rollout mode; one-pass drops the first).

*real-LP flow matching* -- ordinary conditional flow matching on PepBench/ProtFrag rows
with the source-CP condition explicitly absent. This is the anchor: it is the only term
computed on real data, and it is what keeps the shared trunk producing LP geometry rather
than whatever the discriminator is currently rewarding. Dropping it turns the run into a
pure GAN on a pretrained initialisation, which is the configuration most likely to
collapse.

*CP-conditioned generation* -- CPSea rows become generator inputs. The source CP is
encoded once and frozen into ``x_src_cp``; the evolving state carries the LINEAR topology
request; a short differentiable rollout integrates the deployed sampler from t=0 with
gradients kept through the last ``grad_steps`` network evaluations (DRaFT-K, the same
mechanism the cyclization rollout fine-tune already uses). The endpoint is decoded by the
frozen AE and scored by the adversarial and reconstruction-side losses.

*discriminator* -- hinge loss on matched real/fake pairs, stepped on its own optimizer.

MANUAL OPTIMIZATION. Two optimizers means ``automatic_optimization = False``, so gradient
accumulation, the LR scheduler step and the ``skip_nan_grad`` guard are all driven
explicitly below; Lightning does none of it in manual mode. ``accumulate_grad_batches`` is
read from the same ``opt`` block as the rest of the project so the effective batch size
stays comparable with the CPSea arms.

WHAT THE INITIALISATION HAS TO BE. The generator starts from a CPSea flow checkpoint that
was trained *against the mixed LP/CP AE*. Loading a checkpoint trained against a different
autoencoder and then decoding with this one is not a small inconsistency: ``local_latents``
is defined by the AE, so the flow network's output would be interpreted in a latent space
it was never fit to and the decoded geometry is meaningless from step zero. See the config
header for the specific pairing.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import partial

import torch
from loguru import logger

from proteinfoundation.cp2lp.addons import (
    AddonRandomStreams,
    FeatureNoiseController,
    FeatureScaleCache,
    LocalRetentionNCE,
    approx_r1_penalty,
    describe_arm,
    feature_widths,
    validate_addon,
)
from proteinfoundation.cp2lp.conditioning import (
    SRC_CP_KEY,
    attach_source_cp_condition,
    drop_source_cp_condition,
    request_linear_topology,
    source_cp_terminal_gap_nm,
    warn_if_condition_unused,
)
from proteinfoundation.cp2lp.editor import (
    DEFAULT_EDITOR_T,
    EDITOR_NOISE_KEY,
    DEFAULT_NOISE_CHANNELS,
    apply_residuals,
    build_editor_batch,
    make_noise_generator,
    sample_editor_noise,
)
from openfold.np.residue_constants import restype_num

from proteinfoundation.cp2lp.backbone import (
    DEFAULT_ANGLE_SOURCE,
    DEFAULT_LENGTH_SOURCE,
    backbone_to_atom37,
    backbone_validity,
    extract_backbone,
)
from proteinfoundation.cp2lp.bb_arms import (
    ARM_CARTESIAN,
    ARMS,
    DEFAULT_CARTESIAN_SCALE,
    DEFAULT_TORSION_SCALE,
    apply_backbone_arm,
    init_reconstruction_error,
)
from proteinfoundation.cp2lp.bb_losses import (
    NON_INTERFACE_WEIGHT,
    backbone_preservation_loss,
    backbone_reconstruction_loss,
    DEFAULT_PRESERVE_TERMINAL_EXEMPT_K,
    DEFAULT_STERIC_DEPTH_GAIN,
    backbone_steric_loss,
    open_backbone_geometry_loss,
    source_interface_weights,
)
from proteinfoundation.cp2lp.data import RealLPFeatureReservoir, slice_batch, split_roles
from proteinfoundation.cp2lp.discriminator import (
    PeptideInterfaceDiscriminator,
    discriminator_accuracy,
    hinge_d_loss,
    hinge_g_loss,
)
from proteinfoundation.cp2lp.geometry import residue_lengths, terminal_gap_nm
from proteinfoundation.cp2lp.losses import (
    clash_loss,
    contact_retention_loss,
    open_chain_geometry_loss,
    pairwise_ca_rmsd_nm,
    sequence_retention_loss,
    sequence_weight_scale,
    terminal_opening_reward,
)
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.sample_utils import add_clean_samples
from proteinfoundation.utils.training_handlers import handle_batch_conditioning


class CP2LPGenerator(Proteina):
    """Flow generator that turns a bound cyclic peptide into a bound linear one."""

    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None):
        super().__init__(cfg_exp, store_dir=store_dir, autoencoder_ckpt_path=autoencoder_ckpt_path)
        self.automatic_optimization = False
        self._init_cp2lp(cfg_exp)

    # ------------------------------------------------------------------ setup

    def _init_cp2lp(self, cfg_exp):
        cfg = cfg_exp.get("cp2lp", None) or {}
        self.cp2lp_cfg = cfg
        self.cp2lp_enabled = bool(cfg.get("enabled", False))
        if not self.cp2lp_enabled:
            self.discriminator = None
            return

        if getattr(self, "autoencoder", None) is None:
            raise ValueError(
                "cp2lp.enabled=true requires the local_latents autoencoder: the adversarial "
                "and contact losses are defined on DECODED all-atom geometry, which does not "
                "exist without it."
            )
        if "local_latents" not in cfg_exp.product_flowmatcher:
            raise ValueError("cp2lp.enabled=true requires the 'local_latents' data mode.")

        self.cp2lp_samples_per_cp = int(cfg.get("samples_per_cp", 2))
        self.cp2lp_max_cp = int(cfg.get("max_cp_per_step", 4))

        # `rollout` is the ORIGINAL generator: a short differentiable ODE trajectory,
        # DRaFT-K gradients, export from a 200-step sampler. Kept, unchanged, and still the
        # default, so the three measured generations in docs/old_docs/README_CP2LP.md remain
        # reproducible. `onepass` is the direct editor -- one network evaluation, a residual
        # edit of the encoded source CP, no sampler anywhere. The two modes share the
        # losses, the discriminator and the export gate and nothing else.
        self.cp2lp_mode = str(cfg.get("mode", "rollout")).lower()
        if self.cp2lp_mode not in ("rollout", "onepass", "backbone"):
            raise ValueError(
                f"cp2lp.mode must be 'rollout', 'onepass' or 'backbone', got {self.cp2lp_mode!r}"
            )

        # `backbone` is the two REPRESENTATION ARMS. It shares the one-pass editor's batch
        # construction, noise and single trunk call, and differs in what the trunk's output
        # becomes: a full N/CA/C/O backbone emitted by the arm itself, with the frozen AE
        # decoder OFF the output path entirely. No sequence is predicted (it is the source's),
        # no sidechain latent is decoded, and backbone validity is therefore a property of the
        # generator rather than of the AE round trip.
        bb = cfg.get("backbone", None) or {}
        self.bb_arm = str(bb.get("arm", "")).lower()
        if self.cp2lp_mode == "backbone":
            if self.bb_arm not in ARMS:
                raise ValueError(
                    f"cp2lp.mode=backbone requires cp2lp.backbone.arm in {ARMS}, got "
                    f"{self.bb_arm!r}. The arm IS the experiment; there is no default."
                )
            self.bb_angle_source = str(bb.get("angle_source", DEFAULT_ANGLE_SOURCE))
            self.bb_length_source = str(bb.get("length_source", DEFAULT_LENGTH_SOURCE))
            # Separate defaults because the two heads emit different QUANTITIES in different
            # units -- nm of displacement against radians of rotation -- so one shared number
            # would not be one shared budget. The shared, comparable budget is
            # `max_atom_shift_nm`, enforced on actual atom movement in both arms.
            default_scale = (
                DEFAULT_CARTESIAN_SCALE if self.bb_arm == ARM_CARTESIAN else DEFAULT_TORSION_SCALE
            )
            self.bb_residual_scale = float(bb.get("residual_scale", default_scale))
            shift = bb.get("max_atom_shift_nm", None)
            self.bb_max_shift_nm = None if shift in (None, "null", "") else float(shift)
            self.bb_anchor_weight = float(bb.get("anchor_weight", 1.0))
            self.bb_anchor_t_range = tuple(bb.get("anchor_t_range", (0.2, 0.8)))
            self.bb_grad_probe_every = int(bb.get("grad_probe_every_n_steps", 0))

        op = cfg.get("onepass", None) or {}
        self.onepass_t = float(op.get("t_fixed", DEFAULT_EDITOR_T))
        self.onepass_noise_channels = int(op.get("noise_channels", DEFAULT_NOISE_CHANNELS))
        self.onepass_noise_adapter_std = float(op.get("noise_adapter_std", 0.02))
        # OFF BY DEFAULT = the historical behaviour. `init_noise_adapter_` re-draws the noise
        # columns on every `pretrain_ckpt_path` load, and the fork checkpoint every arm starts
        # from already carries TRAINED columns (rms 1.98e-02), so every existing arm re-drew
        # them. Flipping that globally would have changed the initialisation of the controls
        # these arms are compared against. A true WARM START from a trained arm sets this to
        # true; see `configs/example/training_cp2lp_bb_release_warm.yaml`.
        self.onepass_preserve_loaded_noise = bool(op.get("preserve_loaded_noise_adapter", False))

        roll = cfg.get("rollout", None) or {}
        self.cp2lp_nsteps = int(roll.get("nsteps", 24))
        self.cp2lp_grad_steps = int(roll.get("grad_steps", 2))
        self.cp2lp_self_cond = roll.get("self_cond", None)

        adv = cfg.get("adversarial", None) or {}
        self.adv_enabled = bool(adv.get("enabled", True))
        self.adv_weight = float(adv.get("weight", 1.0))
        self.adv_d_lr = float(adv.get("d_lr", 2.0e-4))
        # At least one: `_discriminator_step` reads the loop's last logits after it, so a
        # configured 0 would be a NameError rather than the "skip the discriminator" it
        # looks like. Turning the adversary off is `adversarial.enabled: false`.
        self.adv_d_steps = max(1, int(adv.get("d_steps_per_g", 1)))
        self.adv_warmup_steps = int(adv.get("warmup_g_steps", 200))

        # ONE addon at a time, and only alongside an adversary. Validated here rather than
        # checked at the point of use so a contradictory config dies at startup instead of
        # producing an arm whose result cannot be attributed to anything.
        self.cp2lp_addon = validate_addon(cfg.get("addon", "none"), self.adv_enabled)

        ar1 = cfg.get("ar1", None) or {}
        # lambda 10, not the paper's 100. The paper's D is a large image/video critic with a
        # logit scale of its own; this one is three transformer layers over <=16 residues,
        # and lambda*penalty has to stay commensurate with a hinge loss whose terms are
        # O(1). 10 is a pilot value -- `loss_d_ar1_weighted` is logged next to `loss_d` so
        # the ratio can be read off a run rather than guessed at again.
        self.ar1_lambda = float(ar1.get("lambda", 10.0))
        self.ar1_sigma_rel = float(ar1.get("sigma_rel", 0.01))

        dgan = cfg.get("dgan", None) or {}
        self.dgan_noise_level_dim = int(dgan.get("noise_level_dim", 16))

        nce = cfg.get("nce", None) or {}
        self.nce_weight = float(nce.get("weight", 0.1))

        lw = cfg.get("losses", None) or {}
        self.w_contact = float(lw.get("contact_weight", 1.0))
        self.w_sequence = float(lw.get("sequence_weight", 0.5))
        self.w_geometry = float(lw.get("geometry_weight", 0.5))
        self.w_clash = float(lw.get("clash_weight", 0.5))
        self.w_open = float(lw.get("open_weight", 0.2))
        self.contact_cutoff_nm = float(lw.get("contact_cutoff_nm", 0.8))
        # Width of the soft-contact sigmoid, and therefore the ONLY range over which a lost
        # contact can be pulled back. `sigmoid((cutoff - d)/s)` loses an e-fold of slope
        # every `s`, so at s=0.1 a contact that drifted to 1.2 nm has 14x less gradient than
        # one at the 0.8 nm boundary, at 1.5 nm 275x, at 2.0 nm ~4e4x. The term could
        # therefore DEFEND marginal contacts but never RECOVER drifted ones, which is a
        # one-way ratchet under any pressure that pushes the peptide off the receptor.
        self.contact_sharpness_nm = float(lw.get("contact_sharpness_nm", 0.1))
        self.clash_threshold_nm = float(lw.get("clash_threshold_nm", 0.20))
        self.min_open_gap_nm = float(lw.get("min_open_gap_nm", 0.45))
        self.w_real_lp_flow = float(cfg.get("real_lp_flow_weight", 1.0))

        # Backbone-arm weights. Separate keys from the decoded path's, because they act on
        # different quantities: `preserve_weight` bills nm of backbone displacement where
        # `contact_weight` billed a soft contact count, and reusing one key for both would
        # make a config silently mean something else depending on the mode.
        self.w_bb_preserve = float(lw.get("preserve_weight", 1.0))
        self.w_bb_steric = float(lw.get("steric_weight", 1.0))
        # SHAPE, not size. The release arms ended at `steric_min_inter_nm` 0.0705 (cold) and
        # 0.206 (warm) against real bound LPs' 0.262 -- inside a covalent bond length -- with
        # `steric_weight` already the largest active weight and its gradient norm at 0.78 of a
        # total 2.42. A linear hinge sets an equilibrium rather than enforcing a floor, so the
        # knob that can move this is the one that makes DEEP overlap disproportionately
        # expensive. 0.0 is the historical hinge, bit-for-bit. See `_overlap_penalty`.
        self.bb_steric_depth_gain = float(
            lw.get("steric_depth_gain", DEFAULT_STERIC_DEPTH_GAIN)
        )
        # Residues per end that `preserve` does NOT charge for. 0 is historical. Non-zero is
        # what separates "keep the interface" from "do not open the ring": the preservation
        # term bills terminal travel, and terminal travel is how a macrocycle opens.
        self.bb_preserve_exempt_k = int(
            lw.get("preserve_terminal_exempt_k", DEFAULT_PRESERVE_TERMINAL_EXEMPT_K)
        )
        self.w_bb_geometry = float(lw.get("bb_geometry_weight", 1.0))
        self.w_bb_open = float(lw.get("open_weight", 0.2))
        self.bb_non_interface_weight = float(lw.get("non_interface_weight", NON_INTERFACE_WEIGHT))

        # Retire the sequence term once it has converged, so the clipped gradient budget
        # goes to the terms that have not. See `sequence_weight_scale`.
        sdec = lw.get("sequence_decay", None) or {}
        self.seq_decay_enabled = bool(sdec.get("enabled", False))
        self.seq_decay_start = float(sdec.get("start_exact_match", 0.9))
        self.seq_decay_floor = float(sdec.get("floor", 0.05))
        self.seq_decay_ema_beta = float(sdec.get("ema_beta", 0.99))
        # A PLAIN FLOAT, deliberately not a buffer: adding a buffer changes the state_dict
        # and would break loading the pinned init checkpoint (and every checkpoint these
        # arms have already written). The cost is that a requeue resets it to 0 and the
        # sequence weight returns to full for the few hundred steps the EMA needs to
        # re-warm, which is harmless and self-correcting.
        self._seq_exact_ema = 0.0

        dcfg = cfg.get("discriminator", None) or {}
        self.discriminator = PeptideInterfaceDiscriminator(
            hidden_dim=int(dcfg.get("hidden_dim", 192)),
            nlayers=int(dcfg.get("nlayers", 3)),
            nheads=int(dcfg.get("nheads", 6)),
            k_pocket=int(dcfg.get("k_pocket", 8)),
            contact_cutoff_nm=self.contact_cutoff_nm,
            pocket_identity=str(dcfg.get("pocket_identity", "chem")),
            shortcut_probe=bool(dcfg.get("shortcut_probe", True)),
            dropout=float(dcfg.get("dropout", 0.1)),
            # Only the Diffusion-GAN arm gives D a noise-level input. Every other arm
            # builds the SAME discriminator as the common fork, bit for bit, which is what
            # makes "fork all five arms from one checkpoint" mean anything.
            noise_level_dim=(self.dgan_noise_level_dim if self.cp2lp_addon == "noise" else 0),
            # BACKBONE MODE REMOVES THREE FEATURE GROUPS from D, each a shortcut the
            # generator could not close: the peptide sequence (fixed, and drawn from a
            # different population than the real LPs), sidechain-derived features (nothing on
            # this path produces sidechains inside the gradient), and exact intra-residue bond
            # deviations (Arm B's are template-exact, a crystal's are not, so one column
            # separates the classes perfectly). See discriminator.py.
            backbone_only=(self.cp2lp_mode == "backbone"),
        )

        # ---- addon state. Built only for the arm that uses it, so a checkpoint carries no
        # dormant addon buffers and the four other arms' state dicts stay identical in shape.
        self.feature_scales = None
        self.dgan = None
        self.nce = None
        if self.cp2lp_addon in ("ar1", "noise"):
            sc = cfg.get("feature_scales", None) or {}
            self.feature_scales = FeatureScaleCache(
                widths=feature_widths(self.discriminator),
                min_examples=int(sc.get("min_examples", 128)),
                std_floor=float(sc.get("std_floor", 1e-3)),
            )
        if self.cp2lp_addon == "noise":
            self.dgan = FeatureNoiseController(
                max_std=float(dgan.get("max_std", 0.2)),
                n_levels=int(dgan.get("n_levels", 32)),
                pool_size=int(dgan.get("pool_size", 64)),
                target_sign=float(dgan.get("target_sign", 0.6)),
                adapt_rate_examples=float(dgan.get("adapt_rate_examples", 5000.0)),
                adapt_every_n_steps=int(dgan.get("adapt_every_n_steps", 4)),
                p_init=float(dgan.get("p_init", 0.0)),
                p_max=float(dgan.get("p_max", 1.0)),
            )
        if self.cp2lp_addon == "nce":
            self.nce = LocalRetentionNCE(
                k_neighbors=int(nce.get("k_neighbors", 8)),
                temperature=float(nce.get("temperature", 0.07)),
                proj_dim=int(nce.get("proj_dim", 64)),
                use_projection=bool(nce.get("use_projection", True)),
                max_positions=int(nce.get("max_positions", 16)),
                exclude_terminal=int(nce.get("exclude_terminal", 2)),
                contact_cutoff_nm=self.contact_cutoff_nm,
            )

        # Lazily created, on the right device, at first use. Deliberately NOT buffers: a
        # torch.Generator has no state_dict, so a requeue restarts these streams. That is
        # acceptable for noise and it is worth writing down -- it means a resumed run is
        # not bit-identical to an uninterrupted one, while two arms launched together and
        # never requeued are.
        self._editor_noise_gen = None
        self._addon_streams = None

        self.real_lp_sources = [str(s) for s in (cfg.get("real_lp_sources", None) or [])]

        # Clipping the Trainer cannot do for us under manual optimization. Defaults match
        # what `train.py` applies to every other arm, so this is the same safeguard in a
        # different place rather than a new knob with a new value.
        self.grad_clip_val = float(cfg.get("grad_clip_val", 1.0))
        self.grad_clip_algorithm = str(cfg.get("grad_clip_algorithm", "norm"))

        # GENERATOR-UPDATE ACCOUNTING, and it is not a convenience.
        #
        # `trainer.global_step` under manual optimization is the number of `optimizer.step()`
        # calls SUMMED OVER OPTIMIZERS (lightning/pytorch/loops/optimization/manual.py
        # increments on every lightning optimizer's `_on_after_step`). This arm steps two
        # optimizers, so an adversarial arm advances `global_step` by 2 per accumulation
        # boundary -- or by 1 when the length-matched reservoir could not pair anything and
        # the D step was skipped -- while the contact-only arm always advances by 1.
        #
        # So `global_step` is NOT comparable across these arms, and a budget or a warmup
        # expressed in it means a different amount of generator training in each. Every
        # budget, milestone and warmup below is therefore counted in G updates. A buffer,
        # so a requeue resumes the count instead of restarting the schedule.
        self.register_buffer("g_updates", torch.zeros((), dtype=torch.long), persistent=True)
        self.max_g_updates = int(cfg.get("max_g_updates", 0) or 0)
        self.eval_ckpt_g_updates = sorted(
            int(s) for s in (cfg.get("eval_checkpoint_g_updates", None) or [])
        )
        self._milestones_written: set[int] = set()

        rcfg = cfg.get("reservoir", None) or {}
        self.reservoir = RealLPFeatureReservoir(
            capacity_per_length=int(rcfg.get("capacity_per_length", 64)),
            seed=int(cfg_exp.get("seed", 0)),
        )
        self.reservoir_slack = int(rcfg.get("max_length_slack", 0))

        warn_if_condition_unused(list(cfg_exp.nn.feats_seq), list(cfg_exp.nn.feats_pair_repr))
        # A THREE-WAY branch, each mode naming itself. Written out rather than nested because
        # the first attempt at adding `backbone` here split the original if/else and left
        # `onepass` logging "LEGACY ROLLOUT" while `backbone` logged "ONE-PASS editor" -- a
        # log line is exactly the kind of thing nobody re-reads after a refactor, and a run
        # mislabelled in its own log is unattributable later.
        if self.cp2lp_mode == "backbone":
            self._validate_backbone_wiring(cfg_exp)
            logger.info(
                f"CP2LP BACKBONE arm={self.bb_arm}: angles={self.bb_angle_source}, "
                f"lengths={self.bb_length_source}, residual_scale={self.bb_residual_scale}, "
                f"max_atom_shift_nm={self.bb_max_shift_nm}, anchor_weight={self.bb_anchor_weight}, "
                f"t_fixed={self.onepass_t}, noise_channels={self.onepass_noise_channels}, "
                f"samples_per_cp={self.cp2lp_samples_per_cp}, max_cp_per_step={self.cp2lp_max_cp}, "
                f"adversarial={self.adv_enabled} (weight={self.adv_weight}, "
                f"warmup={self.adv_warmup_steps}). The frozen AE decoder is OFF the output "
                f"path: this arm emits N/CA/C/O itself and predicts no sequence."
            )
        elif self.cp2lp_mode == "onepass":
            self._validate_onepass_wiring(cfg_exp)
            logger.info(
                f"CP2LP ONE-PASS editor: arm={describe_arm(self.cp2lp_addon, self.adv_enabled)}, "
                f"t_fixed={self.onepass_t}, noise_channels={self.onepass_noise_channels}, "
                f"samples_per_cp={self.cp2lp_samples_per_cp}, max_cp_per_step={self.cp2lp_max_cp}, "
                f"adversarial={self.adv_enabled} (weight={self.adv_weight}, warmup={self.adv_warmup_steps})"
            )
        else:
            logger.info(
                f"CP2LP generator enabled (LEGACY ROLLOUT mode): samples_per_cp={self.cp2lp_samples_per_cp}, "
                f"max_cp_per_step={self.cp2lp_max_cp}, rollout nsteps={self.cp2lp_nsteps} "
                f"grad_steps={self.cp2lp_grad_steps}, adversarial={self.adv_enabled} "
                f"(weight={self.adv_weight}, warmup={self.adv_warmup_steps})"
            )
        if not self.adv_enabled:
            logger.info("CP2LP: ADVERSARIAL TERM OFF -- this is the contact-loss-only ablation arm.")

    def _validate_onepass_wiring(self, cfg_exp) -> None:
        """Hard checks that only matter in one-pass mode, all of them silent failures.

        Each of these produces a run that trains, logs falling losses and answers a
        different question than the one asked:

        * no residual heads -> ``apply_residuals`` raises, which is the good case, but it
          raises on the first batch of a queued GPU job rather than at config time;
        * no noise feature -> a deterministic editor, one LP per CP for ever, and the
          within-parent diversity column of the comparison table is structurally zero;
        * a nonzero real-LP flow weight -> the branch this mode disables would still be
          adding a velocity objective the editor's readouts are not even connected to.
        """
        if not bool(cfg_exp.nn.get("enable_editor_residual", False)):
            raise ValueError(
                "cp2lp.mode=onepass requires `nn.enable_editor_residual: true`. Without it "
                "the trunk emits only a flow VELOCITY, and reading that as an edit "
                "displacement would start the editor at x_src + v(x_src, t) -- an arbitrary "
                "jump, not the identity edit the zero-initialised residual heads give."
            )
        feats = list(cfg_exp.nn.feats_seq)
        if "editor_noise_seq" not in feats:
            raise ValueError(
                "cp2lp.mode=onepass requires `editor_noise_seq` in nn.feats_seq (append it "
                "LAST). One network evaluation with no random input can only ever produce "
                "one LP per source CP."
            )
        if feats[-1] != "editor_noise_seq":
            logger.warning(
                f"`editor_noise_seq` is at position {feats.index('editor_noise_seq')} of "
                f"{len(feats)} in nn.feats_seq, not last. Features are concatenated in list "
                "order and `_splice_pretrained_weights` copies the LEADING columns, so "
                "anything after it has its pretrained weights re-bound to the wrong inputs."
            )
        if int(cfg_exp.nn.get("editor_noise_channels", DEFAULT_NOISE_CHANNELS)) != self.onepass_noise_channels:
            raise ValueError(
                f"nn.editor_noise_channels != cp2lp.onepass.noise_channels "
                f"({cfg_exp.nn.get('editor_noise_channels')} vs {self.onepass_noise_channels}); "
                "the feature block width and the sampler must agree."
            )
        if self.w_real_lp_flow != 0.0:
            raise ValueError(
                "cp2lp.mode=onepass requires `cp2lp.real_lp_flow_weight: 0.0`. The flow "
                "objective supervises a velocity field; the editor's residual heads are a "
                "different readout and the term does not reach them. Real bound LPs remain "
                "the discriminator's real class -- that path is unaffected and stays on."
            )

    @property
    def _data_modes(self) -> list[str]:
        return list(self.cfg_exp.product_flowmatcher.keys())

    # ------------------------------------------------------------------ optimizers

    def configure_optimizers(self):
        """Generator optimizer (the parent's, verbatim) plus a discriminator optimizer.

        The parent builds its optimizer from ``[p for p in self.parameters() if
        p.requires_grad]``, which would sweep the discriminator into the generator's
        optimizer. Rather than duplicate sixty lines of warmup/param-group logic that
        would then drift, the discriminator is temporarily marked non-trainable across
        the ``super()`` call -- the parent's selection then excludes it by construction,
        and nothing about its behaviour has to be re-derived here.
        """
        if not self.cp2lp_enabled or self.discriminator is None:
            return super().configure_optimizers()

        d_params = list(self.discriminator.parameters())
        saved = [p.requires_grad for p in d_params]
        for p in d_params:
            p.requires_grad_(False)
        try:
            g_cfg = super().configure_optimizers()
        finally:
            for p, flag in zip(d_params, saved):
                p.requires_grad_(flag)

        d_opt = torch.optim.Adam(d_params, lr=self.adv_d_lr, betas=(0.0, 0.99))

        if isinstance(g_cfg, dict):
            return [g_cfg, {"optimizer": d_opt}]
        return [g_cfg, d_opt]

    @contextmanager
    def _frozen_discriminator(self):
        """Makes the discriminator a fixed function for the duration of a forward pass.

        Restores the previous flags on the way out, including on an exception, so a raise
        inside the generator branch cannot leave the discriminator permanently untrainable
        -- which would look like a discriminator that simply stopped learning.
        """
        params = list(self.discriminator.parameters())
        saved = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, flag in zip(params, saved):
                p.requires_grad_(flag)

    def _unwrap_optimizers(self):
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            return opts, None
        return opts[0], (opts[1] if len(opts) > 1 else None)

    # ------------------------------------------------------------------ branches

    def _real_lp_flow_loss(self, lp_batch: dict) -> tuple[torch.Tensor, int]:
        """Plain conditional flow-matching loss on real bound linear peptides.

        The source-CP condition is dropped, not merely left unset, so the network sees the
        explicit "absent" flag and this stays a trained input mode rather than an
        out-of-distribution one.
        """
        drop_source_cp_condition(lp_batch, self._data_modes)
        lp_batch = self.fm.corrupt_batch(lp_batch)
        bs, _ = lp_batch["mask"].shape
        lp_batch, n_recycle = handle_batch_conditioning(
            lp_batch, bs, self.cfg_exp.training, self.call_nn, self.fm
        )
        nn_out = self.call_nn(lp_batch, n_recycle=n_recycle)
        losses = self.fm.compute_loss(batch=lp_batch, nn_out=nn_out)
        loss = sum(torch.mean(losses[k]) for k in losses if "_justlog" not in k)
        return loss, bs

    def _real_pool_index(self, lp_batch: dict) -> torch.Tensor | None:
        """[b] bool mask of linear rows that count as REAL BOUND LPs, or None for "all".

        Returns None -- meaning "use every linear row" -- both when no restriction is
        configured and when ``dataset_source`` was not loaded. The second case is worth
        knowing about, so it warns once: a silent fallback here would put ProtFrag
        fragments into the discriminator's real pool without anything saying so.
        """
        if not self.real_lp_sources:
            return None
        src = lp_batch.get("dataset_source", None)
        if src is None:
            if not getattr(self, "_warned_no_source_column", False):
                self._warned_no_source_column = True
                logger.warning(
                    "cp2lp.real_lp_sources is set but `dataset_source` is not in the batch -- "
                    "add it to dataset.unified.datamodule.columns_to_load. Falling back to "
                    "every linear row in the discriminator's real pool."
                )
            return None
        return torch.tensor(
            [str(s) in self.real_lp_sources for s in src],
            dtype=torch.bool,
            device=lp_batch["mask"].device,
        )

    def _decode(self, samples: dict, mask: torch.Tensor) -> dict:
        """Decodes a sampled/encoded state to all-atom nm coordinates, keeping gradient."""
        decoded = self.autoencoder.decode(
            z_latent=samples["local_latents"],
            ca_coors_nm=samples["bb_ca"],
            mask=mask,
        )
        decoded["atom_mask_eff"] = decoded["atom_mask"].bool() & mask[..., None]
        return decoded

    def _featurize(self, decoded: dict, batch: dict, mask: torch.Tensor) -> dict:
        """Builds discriminator features from a decoded peptide plus its receptor.

        Sequence enters as a softmax over the decoder's logits rather than as an argmax:
        the generator's gradient reaches its own sequence prediction only through the soft
        path, and a hard one-hot would make the sequence invisible to the adversarial term.
        """
        # A BACKBONE-ONLY DISCRIMINATOR HAS NO `res_emb`, so there is no width to pad the
        # sequence block to and no block to build. Caught by the GPU smoke rather than by
        # review: the generator path worked, the adversarial path worked, and the crash came
        # from the REAL-LP branch, which featurizes a decoded crystal peptide and is the one
        # call site that never runs in a generator-only test.
        if getattr(self.discriminator, "res_emb", None) is None:
            probs = None
        else:
            probs = self._padded_seq_probs(decoded["seq_logits"])
        return self.discriminator.build_features(
            pep_atom37=decoded["coors_nm"],
            pep_atom_mask=decoded["atom_mask_eff"],
            pep_mask=mask,
            pep_seq_probs=probs,
            target_atom37=batch["x_target"],
            target_atom_mask=batch["target_mask"],
            target_mask=batch["seq_target_mask"].bool(),
            target_aatype=batch.get("seq_target", None),
        )

    def _padded_seq_probs(self, seq_logits: torch.Tensor) -> torch.Tensor:
        """Softmax of the decoder's logits, widened or trimmed to D's class count."""
        probs = torch.softmax(seq_logits, dim=-1)
        n_classes = self.discriminator.res_emb.in_features
        if probs.shape[-1] < n_classes:
            probs = torch.nn.functional.pad(probs, (0, n_classes - probs.shape[-1]))
        elif probs.shape[-1] > n_classes:
            probs = probs[..., :n_classes]
        return probs

    def _generate_from_cp(self, cp_batch: dict) -> tuple[dict, dict, dict]:
        """Runs the short differentiable rollout on a CP-conditioned batch.

        Returns ``(gen_batch, decoded, samples)``. ``gen_batch`` is the batch the sampler
        actually saw, which the caller needs for the receptor tensors and the source CP.
        """
        bs = cp_batch["mask"].shape[0]

        # x_1 here is the ENCODED SOURCE CP -- freeze it as the condition before the
        # sampler state keys are removed.
        attach_source_cp_condition(cp_batch, data_modes=self._data_modes)

        # Everything derived from the training interpolant has to go: the rollout sets its
        # own state, and stale ground-truth-derived tensors left in the batch would be read
        # as if they described the peptide being generated.
        gen_batch = {
            k: v for k, v in cp_batch.items() if k not in ("x_1", "x_0", "x_t", "t", "x_sc", "x_recycle")
        }
        gen_batch["mask"] = cp_batch["mask"].bool()
        request_linear_topology(gen_batch, bs=bs)

        design = self._rollout_ft_sampler() if self.cp2lp_cfg.get("use_rollout_sampler", False) else None
        if design is None:
            design = self.val_gen_cfg.get("design_sampling", None) if self.val_gen_cfg else None
        if design is None:
            raise ValueError(
                "CP2LP needs a `design_sampling` block (sampler `args` + `model`). Add "
                "`- /pipeline/model_sampling@val_generation.design_sampling` to the config."
            )
        sampler_args, sampling_model_args = design.args, design.model
        self_cond = self.cp2lp_self_cond
        self_cond = bool(sampler_args.self_cond if self_cond is None else self_cond)

        n_samples, n = gen_batch["mask"].shape
        samples = self.fm.full_simulation_draft(
            batch=gen_batch,
            predict_for_sampling=partial(self.predict_for_sampling, n_recycle=int(design.get("n_recycle", 0))),
            nsteps=self.cp2lp_nsteps,
            nsamples=n_samples,
            n=n,
            self_cond=self_cond,
            sampling_model_args=sampling_model_args,
            device=gen_batch["mask"].device,
            grad_steps=self.cp2lp_grad_steps,
            guidance_w=float(sampler_args.get("guidance_w", 1.0)),
            ag_ratio=0.0,
        )
        decoded = self._decode(samples, gen_batch["mask"])
        return gen_batch, decoded, samples

    # ------------------------------------------------------- one-pass direct editor

    def init_editor_noise_adapter(self) -> tuple[int, int] | None:
        """Small nonzero input weights for the noise block. Call AFTER a pretrained splice.

        Not done in ``__init__``: the splice runs later and would overwrite it with the
        checkpoint's (zero) columns for an appended feature. ``train.py`` calls this
        immediately after loading ``pretrain_ckpt_path``, and a RESUME does not call it at
        all -- resuming loads the arm's own full state dict, where the columns already
        carry whatever they learned.
        """
        # BACKBONE MODE NEEDS THIS TOO, for exactly the reason one-pass does: the backbone
        # heads are zero-initialised as well, so with the noise columns also at zero the
        # trunk receives no gradient on step one and the noise only becomes readable once
        # the heads have moved. The arm would learn a deterministic edit first and discover
        # its random input later -- and `within-source diversity`, which the comparison
        # reports, would be structurally zero over that whole phase. Found by the GPU smoke,
        # which printed `columns None re-initialised` where the one-pass smoke treats None
        # as a hard error.
        if self.cp2lp_mode not in ("onepass", "backbone"):
            return None
        from proteinfoundation.cp2lp.editor import init_noise_adapter_

        return init_noise_adapter_(
            self.nn.init_repr_factory,
            std=self.onepass_noise_adapter_std,
            preserve_loaded=self.onepass_preserve_loaded_noise,
        )

    def _noise_generator(self) -> torch.Generator:
        if self._editor_noise_gen is None:
            self._editor_noise_gen = make_noise_generator(int(self.cfg_exp.get("seed", 0)), self.device)
        return self._editor_noise_gen

    def _streams(self) -> AddonRandomStreams:
        if self._addon_streams is None:
            self._addon_streams = AddonRandomStreams(int(self.cfg_exp.get("seed", 0)), self.device)
        return self._addon_streams

    def generate_onepass(
        self,
        cp_batch: dict,
        noise: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[dict, dict, dict]:
        """ONE network evaluation: encoded source CP in, edited-and-decoded LP out.

        The single entry point for both training and export. Sharing it is not tidiness:
        the previous generation trained on a 24-step trajectory and exported from a
        200-step one, and every training-side geometry number was therefore uninformative
        about what would be exported (``geom_chain_intact_frac`` read 0 by construction in
        the logs while the export gate used a different sampler). With one function there
        is one number.

        Args:
            cp_batch: a CPSea batch after ``add_clean_samples``.
            noise: ``[b, n, c]`` to reuse an exact draw -- the rotation check and the
                train/export agreement check both need the same noise twice.
            deterministic: omit the noise entirely. Only the frame-sensitivity measurement
                wants this; a deterministic editor is not a usable generator.

        Returns:
            ``(gen_batch, decoded, extras)``. ``extras`` carries the edited latent state
            and the residual magnitudes -- a residual norm pinned at zero is what a head
            that never left its zero init looks like, and it is otherwise indistinguishable
            from a generator that has decided not to edit.
        """
        bs, n = cp_batch["mask"].shape
        ref = cp_batch["x_1"][self._data_modes[0]]
        if noise is None and not deterministic:
            noise = sample_editor_noise(
                bs, n, self.onepass_noise_channels, ref.device, ref.dtype, self._noise_generator()
            )
        gen_batch = build_editor_batch(
            cp_batch, data_modes=self._data_modes, t_fixed=self.onepass_t, noise=noise
        )

        # ONE call. `n_recycle=0` is what makes that literally true: a recycle is another
        # full forward, and `cp2lp_onepass_smoke.py` counts trunk calls with a forward hook
        # rather than trusting this line.
        nn_out = self.call_nn(gen_batch, n_recycle=0)
        edited, deltas = apply_residuals(gen_batch["x_t"], nn_out, gen_batch["mask"], self._data_modes)
        decoded = self._decode(edited, gen_batch["mask"])

        mask = gen_batch["mask"].bool()
        denom = mask.sum().clamp(min=1)
        extras = {"edited": edited, "deltas": deltas}
        with torch.no_grad():
            extras["metrics"] = {
                "delta_ca_rms_nm": torch.sqrt((deltas["bb_ca"] ** 2).sum() / (denom * 3)),
                "delta_latent_rms": torch.sqrt(
                    (deltas["local_latents"] ** 2).sum() / (denom * deltas["local_latents"].shape[-1])
                ),
            }
        return gen_batch, decoded, extras

    # ====================================================================== backbone arms

    def _validate_backbone_wiring(self, cfg_exp) -> None:
        """Hard checks for the two representation arms. Each failure is otherwise silent.

        * a head/arm mismatch -> `apply_backbone_arm` raises, but on the first batch of a
          queued GPU job rather than at config time;
        * no noise feature -> one LP per CP for ever, and the within-source diversity column
          of the comparison is structurally zero;
        * a nonzero real-LP FLOW weight -> the velocity objective does not reach the new
          heads at all. The plan is explicit that a legacy flow loss on a discarded output
          head is not an effective anchor, so the anchor here is
          `_real_lp_backbone_anchor`, weighted by `cp2lp.backbone.anchor_weight`.
        """
        want = {ARM_CARTESIAN: "cartesian", "torsion": "torsion"}[self.bb_arm]
        got = str(cfg_exp.nn.get("editor_backbone_mode", "none"))
        if got != want:
            raise ValueError(
                f"cp2lp.backbone.arm={self.bb_arm!r} needs `nn.editor_backbone_mode: {want}`, "
                f"got {got!r}. The arm's output head has to exist before the arm can run."
            )
        feats = list(cfg_exp.nn.feats_seq)
        if "editor_noise_seq" not in feats:
            raise ValueError(
                "cp2lp.mode=backbone requires `editor_noise_seq` in nn.feats_seq (append it "
                "LAST). One network evaluation with no random input produces one LP per CP."
            )
        if feats[-1] != "editor_noise_seq":
            logger.warning(
                f"`editor_noise_seq` is at position {feats.index('editor_noise_seq')} of "
                f"{len(feats)} in nn.feats_seq, not last. Features concatenate in list order "
                "and `_splice_pretrained_weights` copies LEADING columns, so anything after "
                "it has its pretrained weights re-bound to the wrong inputs."
            )
        if self.w_real_lp_flow != 0.0:
            raise ValueError(
                "cp2lp.mode=backbone requires `cp2lp.real_lp_flow_weight: 0.0`. The flow "
                "objective supervises a velocity head that is not on this output path, so it "
                "would train a readout nothing reads. The real-LP anchor for these arms is "
                "cp2lp.backbone.anchor_weight, which supervises the backbone head itself."
            )
        if self.w_sequence != 0.0:
            raise ValueError(
                "cp2lp.mode=backbone requires `cp2lp.losses.sequence_weight: 0.0`. No sequence "
                "is predicted on this path -- the source's is copied exactly -- so a sequence "
                "retention term has nothing to act on."
            )

    def _source_backbone(self, gen_batch: dict, mask: torch.Tensor) -> torch.Tensor:
        """The source CP's backbone, `[b, n, 4, 3]` nm, by decoding the source condition.

        DETACHED. It is the arm's reference point and the preservation term's target, not a
        quantity to optimise: letting gradient into it would let the generator reduce the
        preservation loss by moving the target instead of by preserving anything.
        """
        with torch.no_grad():
            src_dec = self._decode(gen_batch[SRC_CP_KEY], mask)
            bb, _ = extract_backbone(src_dec["coors_nm"], src_dec["atom_mask_eff"])
        return bb.detach()

    def generate_backbone(
        self,
        cp_batch: dict,
        noise: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[dict, dict, dict]:
        """ONE network evaluation: encoded source CP in, a full N/CA/C/O backbone out.

        Returns `(gen_batch, decoded, extras)`, where `decoded` is shaped exactly like the
        frozen AE decoder's output -- `coors_nm`, `atom_mask_eff`, `seq_logits` -- so every
        downstream consumer (`_featurize`, the export path, the metrics) works unchanged.

        THE AE DECODER IS NOT CALLED FOR THE OUTPUT. `coors_nm` here is the arm's own
        backbone placed into the shared atom37 container with exactly four slots unmasked;
        `backbone_to_atom37` is what enforces that, as a mask rather than a convention. The
        AE is still used to decode the SOURCE (the arm needs a reference backbone) and to
        decode real LPs for the discriminator's real class -- both inputs, neither output.

        `seq_logits` is a ONE-HOT OF THE SOURCE SEQUENCE, not a prediction. The sequence is
        copied exactly and is known, so a one-hot is the truth rather than a stand-in; it is
        provided because `_featurize` reads the key, and a backbone-only discriminator drops
        it again on the way in.
        """
        bs, n = cp_batch["mask"].shape
        ref = cp_batch["x_1"][self._data_modes[0]]
        if noise is None and not deterministic:
            noise = sample_editor_noise(
                bs, n, self.onepass_noise_channels, ref.device, ref.dtype, self._noise_generator()
            )
        gen_batch = build_editor_batch(
            cp_batch, data_modes=self._data_modes, t_fixed=self.onepass_t, noise=noise
        )
        mask = gen_batch["mask"].bool()
        src_bb = self._source_backbone(gen_batch, mask)

        # ONE call. `n_recycle=0` makes that literally true; the smoke counts trunk calls
        # with a forward hook rather than trusting this line.
        nn_out = self.call_nn(gen_batch, n_recycle=0)
        gen_bb, diag = apply_backbone_arm(
            self.bb_arm,
            src_bb,
            nn_out,
            mask,
            aatype=gen_batch.get("residue_type", None),
            scale=self.bb_residual_scale,
            max_shift_nm=self.bb_max_shift_nm,
            **(
                {"angle_source": self.bb_angle_source, "length_source": self.bb_length_source}
                if self.bb_arm != ARM_CARTESIAN
                else {}
            ),
        )

        atom37, atom_mask = backbone_to_atom37(gen_bb, mask)
        decoded = {
            "coors_nm": atom37,
            "atom_mask": atom_mask,
            "atom_mask_eff": atom_mask & mask[..., None],
            "seq_logits": self._source_seq_logits(gen_batch, mask),
        }
        extras = {"gen_bb": gen_bb, "src_bb": src_bb, "metrics": {}}
        with torch.no_grad():
            v = backbone_validity(gen_bb, mask)
            extras["metrics"] = {
                **dict(diag),
                **{k: x.mean() for k, x in v.items()},
            }
        return gen_batch, decoded, extras

    def _source_seq_logits(self, gen_batch: dict, mask: torch.Tensor) -> torch.Tensor:
        """`[b, n, 21]` one-hot of the source sequence. The truth, not a prediction."""
        aatype = gen_batch.get("residue_type", None)
        b, n = mask.shape
        if aatype is None:
            return torch.zeros(b, n, restype_num + 1, device=mask.device)
        oh = torch.nn.functional.one_hot(
            aatype.long().clamp(0, restype_num), restype_num + 1
        ).to(torch.float32)
        # Large but finite, so a softmax of it is a near-one-hot rather than exactly one --
        # an exact one-hot would put a hard zero into any log the metrics take.
        return (oh * 10.0) * mask[..., None].to(oh.dtype)

    def _backbone_loss_terms(
        self, gen_batch: dict, extras: dict
    ) -> dict[str, torch.Tensor]:
        """The backbone arms' non-adversarial terms, UNWEIGHTED and separate.

        Separate because scalar loss shares are NOT gradient shares: a term worth 1% of the
        summed loss can carry most of the gradient reaching a head, and the only way to know
        is to differentiate each one on its own. `_log_backbone_grad_shares` does exactly
        that on a schedule.
        """
        mask = gen_batch["mask"].bool()
        gen_bb, src_bb = extras["gen_bb"], extras["src_bb"]
        w, _ = source_interface_weights(
            self._src_atom37(gen_batch, mask), self._src_atom_mask(gen_batch, mask), mask,
            gen_batch["x_target"], gen_batch["target_mask"], gen_batch["seq_target_mask"].bool(),
            cutoff_nm=self.contact_cutoff_nm, non_interface=self.bb_non_interface_weight,
            terminal_exempt_k=self.bb_preserve_exempt_k,
        )
        atom37, atom_mask = backbone_to_atom37(gen_bb, mask)
        out = {
            "preserve": backbone_preservation_loss(gen_bb, src_bb, mask, w)[0],
            "steric": backbone_steric_loss(
                gen_bb, mask, gen_batch["x_target"], gen_batch["target_mask"],
                gen_batch["seq_target_mask"].bool(), threshold_nm=self.clash_threshold_nm,
                depth_gain=self.bb_steric_depth_gain,
            )[0],
            "open": terminal_opening_reward(atom37, mask, self.min_open_gap_nm)[0],
        }
        # ARM A ONLY. Arm B's bond geometry is correct by construction, so for it this term
        # is a constant whose gradient is numerical noise -- including it would add noise to
        # one arm's gradient budget and not the other's, which is a difference between the
        # arms that is not the representation.
        if self.bb_arm == ARM_CARTESIAN:
            out["geometry"] = open_backbone_geometry_loss(gen_bb, mask)[0]
        return out

    def _src_atom37(self, gen_batch: dict, mask: torch.Tensor) -> torch.Tensor:
        """Source CP ALL-ATOM coordinates, for deriving the interface weights.

        All-atom on purpose: the binding mode lives in the sidechains, and the generated
        backbone has none to read it from. Cached per batch because the preservation term
        and the loss-term probe would otherwise decode the source twice.
        """
        if getattr(self, "_src_dec_cache", None) is None:
            with torch.no_grad():
                self._src_dec_cache = self._decode(gen_batch[SRC_CP_KEY], mask)
        return self._src_dec_cache["coors_nm"]

    def _src_atom_mask(self, gen_batch: dict, mask: torch.Tensor) -> torch.Tensor:
        self._src_atom37(gen_batch, mask)
        return self._src_dec_cache["atom_mask_eff"]

    def _backbone_losses(self, gen_batch: dict, extras: dict) -> tuple[torch.Tensor, dict]:
        """Weighted sum of the backbone arms' non-adversarial terms, plus diagnostics."""
        mask = gen_batch["mask"].bool()
        gen_bb, src_bb = extras["gen_bb"], extras["src_bb"]
        self._src_dec_cache = None  # one decode per batch, released at the end

        w, iface_met = source_interface_weights(
            self._src_atom37(gen_batch, mask), self._src_atom_mask(gen_batch, mask), mask,
            gen_batch["x_target"], gen_batch["target_mask"], gen_batch["seq_target_mask"].bool(),
            cutoff_nm=self.contact_cutoff_nm, non_interface=self.bb_non_interface_weight,
            terminal_exempt_k=self.bb_preserve_exempt_k,
        )
        pres, pres_met = backbone_preservation_loss(gen_bb, src_bb, mask, w)
        ster, ster_met = backbone_steric_loss(
            gen_bb, mask, gen_batch["x_target"], gen_batch["target_mask"],
            gen_batch["seq_target_mask"].bool(), threshold_nm=self.clash_threshold_nm,
            depth_gain=self.bb_steric_depth_gain,
        )
        atom37, _ = backbone_to_atom37(gen_bb, mask)
        opening, open_met = terminal_opening_reward(atom37, mask, self.min_open_gap_nm)

        total = (
            self.w_bb_preserve * pres + self.w_bb_steric * ster + self.w_bb_open * opening
        )
        metrics = {
            "loss_bb_preserve": pres.detach(), "loss_bb_steric": ster.detach(),
            "loss_bb_open": opening.detach(),
            **iface_met, **pres_met, **ster_met, **open_met,
        }
        if self.bb_arm == ARM_CARTESIAN:
            geom, geom_met = open_backbone_geometry_loss(gen_bb, mask)
            total = total + self.w_bb_geometry * geom
            metrics["loss_bb_geometry"] = geom.detach()
            metrics.update(geom_met)

        if gen_bb.shape[0] > 1:
            metrics["seed_diversity_nm"] = pairwise_ca_rmsd_nm(atom37, mask)
        self._src_dec_cache = None
        return total, metrics

    def _real_lp_backbone_anchor(self, lp_batch: dict) -> tuple[torch.Tensor, int, dict]:
        """Denoise a corrupted REAL LP back to its own backbone, THROUGH the arm's head.

        WHY THIS REPLACES THE FLOW LOSS. A legacy flow term supervises a velocity readout
        that is not on this output path, so it would train a head nothing reads and anchor
        nothing. This instead corrupts a real bound LP with the SAME interpolant, decodes the
        corrupted state to get a starting backbone, and asks the arm to put it back -- which
        reaches the new head and the shared trunk, and does so with clean LP geometry as the
        target.

        THE SOURCE CP CONDITION IS DROPPED, not left unset, so the network sees the explicit
        "absent" flag and this stays a trained input mode rather than an out-of-distribution
        one. Both arms get the same corruption draws and the same Cartesian criterion, Arm B
        evaluated through its builder -- so the anchor is one requirement reached two ways
        rather than two different objectives.
        """
        drop_source_cp_condition(lp_batch, self._data_modes)
        mask = lp_batch["mask"].bool()
        bs = mask.shape[0]

        with torch.no_grad():
            clean = {
                dm: lp_batch["x_1"][dm].detach() * mask[..., None].to(lp_batch["x_1"][dm].dtype)
                for dm in self._data_modes
            }
            true_bb, _ = extract_backbone(*(lambda d: (d["coors_nm"], d["atom_mask_eff"]))(
                self._decode(clean, mask)
            ))

        lp_batch = self.fm.corrupt_batch(lp_batch)
        # The interpolant draws t over the whole schedule; the arms only ever run at one
        # fixed t, so the anchor is restricted to a band around it. Training the head to
        # denoise from t=0.02 would be fitting a regime the generator never visits.
        lo, hi = self.bb_anchor_t_range
        for dm in self._data_modes:
            lp_batch["t"][dm] = lp_batch["t"][dm].clamp(lo, hi)

        with torch.no_grad():
            noisy_bb, _ = extract_backbone(*(lambda d: (d["coors_nm"], d["atom_mask_eff"]))(
                self._decode({dm: lp_batch["x_t"][dm] for dm in self._data_modes}, mask)
            ))

        if EDITOR_NOISE_KEY not in lp_batch:
            ref = lp_batch["x_t"][self._data_modes[0]]
            lp_batch[EDITOR_NOISE_KEY] = sample_editor_noise(
                bs, mask.shape[1], self.onepass_noise_channels, ref.device, ref.dtype,
                self._noise_generator(),
            )
        request_linear_topology(lp_batch, bs=bs)
        nn_out = self.call_nn(lp_batch, n_recycle=0)
        pred_bb, _ = apply_backbone_arm(
            self.bb_arm, noisy_bb.detach(), nn_out, mask,
            aatype=lp_batch.get("residue_type", None),
            scale=self.bb_residual_scale, max_shift_nm=None,
            **(
                {"angle_source": self.bb_angle_source, "length_source": self.bb_length_source}
                if self.bb_arm != ARM_CARTESIAN
                else {}
            ),
        )
        loss, met = backbone_reconstruction_loss(pred_bb, true_bb.detach(), mask)
        with torch.no_grad():
            # The FLOOR: how close the arm could get at zero head output. An anchor loss at
            # or below it says the head is doing nothing and the number is template drift.
            floor = init_reconstruction_error(
                self.bb_arm, true_bb, mask,
                angle_source=getattr(self, "bb_angle_source", DEFAULT_ANGLE_SOURCE),
                length_source=getattr(self, "bb_length_source", DEFAULT_LENGTH_SOURCE),
                aatype=lp_batch.get("residue_type", None),
            )
            met["anchor_template_floor_nm"] = torch.tensor(float(floor["init_bb_rmsd_nm"]))
        return loss, bs, met

    @torch.enable_grad()
    def _log_backbone_grad_shares(self, gen_batch: dict, extras: dict) -> dict:
        """Per-term gradient norm at the arm's HEAD, before clipping. Weighted, not raw.

        SCALAR LOSS SHARES ARE NOT GRADIENT SHARES, and this project has already been bitten
        by exactly that: the clash term was 0.06% of the objective while being violated, and
        a loss-value table said nothing about it. This differentiates each weighted term on
        its own and reports the norm that actually arrives, which is the quantity that
        decides whether a term is doing anything.

        Measured on a schedule rather than every step: it costs one backward per term.
        """
        terms = self._backbone_loss_terms(gen_batch, extras)
        weights = {
            "preserve": self.w_bb_preserve, "steric": self.w_bb_steric,
            "open": self.w_bb_open, "geometry": self.w_bb_geometry,
        }
        return self._grad_norms_at_head(
            {k: (v, weights.get(k, 1.0)) for k, v in terms.items()}
        )

    @torch.enable_grad()
    def _grad_norms_at_head(
        self, weighted: dict[str, tuple[torch.Tensor, float]]
    ) -> dict[str, torch.Tensor]:
        """`{name: (unweighted_term, weight)}` -> `{gradnorm/<name>: norm}` at the arm's head.

        Factored out so the terms computed OUTSIDE `_backbone_loss_terms` -- the adversarial
        term and the real-LP anchor -- can be measured against the same parameters, by the same
        arithmetic, and so be compared with `steric` on the same axis. That comparison is the
        whole point: steric's 0.78 only means something next to what is opposing it, and until
        now nothing logged the opposing terms at all.
        """
        params = [p for p in self._backbone_head_parameters() if p.requires_grad]
        out: dict[str, torch.Tensor] = {}
        if not params:
            return out
        for name, (term, weight) in weighted.items():
            if term is None or not torch.is_tensor(term) or not term.requires_grad:
                out[f"gradnorm/{name}"] = torch.zeros(())
                continue
            # retain_graph because the caller still has to run the real backward on this graph,
            # and because the next term in this loop needs it too.
            g = torch.autograd.grad(
                weight * term, params, retain_graph=True, allow_unused=True
            )
            out[f"gradnorm/{name}"] = torch.sqrt(
                sum((x * x).sum() for x in g if x is not None) + 1e-20
            ).detach()
        return out

    def _backbone_head_parameters(self):
        """The arm's own output head. Where the per-term gradient is measured."""
        nn_mod = getattr(self.nn, "module", self.nn)
        for attr in ("editor_bb_cartesian", "editor_bb_torsion", "editor_bb_root"):
            head = getattr(nn_mod, attr, None)
            if head is not None:
                yield from head.parameters()

    def _generate(self, cp_batch: dict) -> tuple[dict, dict, dict]:
        """Mode dispatch. ``(gen_batch, decoded, extras)``; ``extras`` is mode-specific."""
        if self.cp2lp_mode == "backbone":
            return self.generate_backbone(cp_batch)
        if self.cp2lp_mode == "onepass":
            return self.generate_onepass(cp_batch)
        gen_batch, decoded, samples = self._generate_from_cp(cp_batch)
        return gen_batch, decoded, {"samples": samples, "metrics": {}}

    def _generator_loss_terms(self, gen_batch: dict, decoded: dict) -> dict[str, torch.Tensor]:
        """The generator's non-adversarial terms, UNWEIGHTED and separate.

        Exists for gradient attribution: when the summed gradient comes back non-finite,
        the only thing that localises it is differentiating each term on its own. Kept
        beside `_generator_losses` rather than inside it because the training path wants
        one number and the diagnostic wants five.
        """
        mask = gen_batch["mask"].bool()
        src_cp = self._decode(gen_batch[SRC_CP_KEY], mask)
        target = dict(
            target_atom37=gen_batch["x_target"],
            target_atom_mask=gen_batch["target_mask"],
            target_mask=gen_batch["seq_target_mask"].bool(),
        )
        contact, _ = contact_retention_loss(
            gen_atom37=decoded["coors_nm"], gen_atom_mask=decoded["atom_mask_eff"],
            src_atom37=src_cp["coors_nm"].detach(), src_atom_mask=src_cp["atom_mask_eff"].detach(),
            pep_mask=mask, cutoff_nm=self.contact_cutoff_nm,
            sharpness_nm=self.contact_sharpness_nm, **target,
        )
        seq, _ = sequence_retention_loss(decoded["seq_logits"], gen_batch["residue_type"], mask)
        geom, _ = open_chain_geometry_loss(decoded["coors_nm"], decoded["atom_mask_eff"], mask)
        clash, _ = clash_loss(
            atom37=decoded["coors_nm"], atom_mask=decoded["atom_mask_eff"], pep_mask=mask,
            threshold_nm=self.clash_threshold_nm, **target,
        )
        opening, _ = terminal_opening_reward(decoded["coors_nm"], mask, self.min_open_gap_nm)
        return {"contact": contact, "sequence": seq, "geometry": geom, "clash": clash, "open": opening}

    def _generator_losses(self, gen_batch: dict, decoded: dict) -> tuple[torch.Tensor, dict]:
        """Non-adversarial generator terms plus their diagnostics."""
        mask = gen_batch["mask"].bool()
        metrics: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=mask.device)

        src_cp = self._decode(gen_batch[SRC_CP_KEY], mask)

        contact, m = contact_retention_loss(
            gen_atom37=decoded["coors_nm"],
            gen_atom_mask=decoded["atom_mask_eff"],
            src_atom37=src_cp["coors_nm"].detach(),
            src_atom_mask=src_cp["atom_mask_eff"].detach(),
            pep_mask=mask,
            target_atom37=gen_batch["x_target"],
            target_atom_mask=gen_batch["target_mask"],
            target_mask=gen_batch["seq_target_mask"].bool(),
            cutoff_nm=self.contact_cutoff_nm,
            sharpness_nm=self.contact_sharpness_nm,
        )
        total = total + self.w_contact * contact
        metrics["loss_contact"] = contact.detach()
        metrics.update(m)

        seq_loss, m = sequence_retention_loss(
            seq_logits=decoded["seq_logits"],
            target_aatype=gen_batch["residue_type"],
            pep_mask=mask,
        )
        # The EMA is updated from THIS batch's exact match before the scale is read, so the
        # weight tracks convergence without a step of lag. `m["seq_exact_match"]` is already
        # detached (computed under no_grad in the loss).
        seq_scale = 1.0
        if self.seq_decay_enabled:
            self._seq_exact_ema = (
                self.seq_decay_ema_beta * self._seq_exact_ema
                + (1.0 - self.seq_decay_ema_beta) * float(m["seq_exact_match"])
            )
            seq_scale = sequence_weight_scale(
                self._seq_exact_ema, start=self.seq_decay_start, floor=self.seq_decay_floor
            )
        total = total + (self.w_sequence * seq_scale) * seq_loss
        metrics["loss_sequence"] = seq_loss.detach()
        metrics["seq_exact_ema"] = torch.tensor(self._seq_exact_ema)
        metrics["seq_weight_scale"] = torch.tensor(seq_scale)
        metrics.update(m)

        geom, m = open_chain_geometry_loss(decoded["coors_nm"], decoded["atom_mask_eff"], mask)
        total = total + self.w_geometry * geom
        metrics["loss_geometry"] = geom.detach()
        metrics.update(m)

        clash, m = clash_loss(
            atom37=decoded["coors_nm"],
            atom_mask=decoded["atom_mask_eff"],
            pep_mask=mask,
            target_atom37=gen_batch["x_target"],
            target_atom_mask=gen_batch["target_mask"],
            target_mask=gen_batch["seq_target_mask"].bool(),
            threshold_nm=self.clash_threshold_nm,
        )
        total = total + self.w_clash * clash
        metrics["loss_clash"] = clash.detach()
        metrics.update(m)

        open_loss, m = terminal_opening_reward(decoded["coors_nm"], mask, self.min_open_gap_nm)
        total = total + self.w_open * open_loss
        metrics["loss_open"] = open_loss.detach()
        metrics.update(m)

        with torch.no_grad():
            metrics["src_terminal_gap_nm"] = source_cp_terminal_gap_nm(gen_batch).mean()
            if decoded["coors_nm"].shape[0] > 1:
                metrics["seed_diversity_nm"] = pairwise_ca_rmsd_nm(decoded["coors_nm"], mask)

        return total, metrics

    # ------------------------------------------------------------------ training

    def training_step(self, batch: dict, batch_idx: int):
        if not self.cp2lp_enabled:
            return super().training_step(batch, batch_idx)

        # `Proteina.validation_step_data` calls this with batch_idx=-1 to score a
        # VALIDATION batch (proteina.py: `val_step = batch_idx == -1`). Everything below
        # this line assumes a training batch, and on a validation batch all of it is
        # wrong: the pass runs under `no_grad` so no loss carries a graph, and
        # `(-1 + 1) % accum == 0` is True for every accum, so the optimizer and the LR
        # scheduler would be stepped from inside validation. That is what killed the GAN
        # arm at its first validation -- `manual_backward` on the graph-less
        # discriminator loss raised "element 0 of tensors does not require grad".
        #
        # Delegating to the parent is the right handling, not just the safe one: it
        # computes exactly the validation losses every other CPSea arm reports, so
        # `validation_loss/*` stays comparable across arms. The CP->LP quantities are
        # training-side diagnostics, and generated-sample quality is measured by
        # `val_generation` and the export gate rather than here.
        if batch_idx == -1:
            return super().training_step(batch, batch_idx)

        accum = max(1, int(self.cfg_exp.opt.get("accumulate_grad_batches", 1) or 1))
        opt_g, opt_d = self._unwrap_optimizers()
        log_prefix = "train"

        batch = add_clean_samples(
            batch,
            self.cfg_exp.product_flowmatcher,
            getattr(self, "autoencoder", None),
            local_latent_target=self.cfg_exp.get("local_latent_target", "sample"),
            detach_latent_target_for_flow=bool(self.cfg_exp.get("detach_latent_target_for_flow", False)),
        )
        bs = batch["mask"].shape[0]
        cyclic_idx, linear_idx = split_roles(batch)
        metrics: dict[str, torch.Tensor] = {}

        # -------------------------------------------------- real LP: flow + D real pool
        g_loss = torch.zeros((), device=batch["mask"].device)
        # Terms the gradient probe can only reach from here: the anchor is computed in the
        # real-LP branch above the generation block, and the adversarial term below it, so
        # neither is in scope inside `_backbone_loss_terms`. Collected as they are built and
        # differentiated together just before the real backward, while the graph is still
        # alive. CLEARED EVERY STEP: differentiating last step's tensor would hit a freed
        # graph, which is a crash on a schedule -- i.e. hours into a run, not at smoke time.
        self._probe_extra_terms: dict[str, tuple[torch.Tensor, float]] = {}
        real_feats_live = None
        if linear_idx.any():
            lp_batch = slice_batch(batch, linear_idx, bs)
            # THE FLOW BRANCH IS OFF IN ONE-PASS MODE. It supervises a velocity field; the
            # editor's readouts are residual heads, so the term does not reach them, and
            # nothing has been substituted for it -- no identity loss, no distillation, no
            # reconstruction. Real bound LPs still arrive here and still go into the
            # discriminator's real pool below, which is the role they have in this mode.
            if self.cp2lp_mode == "rollout":
                flow_loss, n_lp = self._real_lp_flow_loss(lp_batch)
                g_loss = g_loss + self.w_real_lp_flow * flow_loss
                metrics["loss_real_lp_flow"] = flow_loss.detach()
                metrics["n_real_lp"] = torch.tensor(float(n_lp))
            elif self.cp2lp_mode == "backbone" and self.bb_anchor_weight != 0.0:
                # THE ANCHOR THAT ACTUALLY SUPERVISES THE NEW HEAD. The flow branch above
                # trains a velocity readout that is not on this output path; this denoises a
                # corrupted real LP back to its own backbone THROUGH the arm, so the gradient
                # reaches the head and the shared trunk. Same corruption draws and the same
                # Cartesian criterion in both arms, Arm B through its builder.
                #
                # `slice_batch` copies, but `_real_lp_backbone_anchor` calls `corrupt_batch`
                # and `drop_source_cp_condition`, which mutate -- so it runs AFTER the real
                # pool below has taken its clean snapshot. Ordering matters here and the
                # reservoir push is what would silently change.
                anchor_batch = lp_batch
                self._pending_bb_anchor = anchor_batch
                metrics["n_real_lp"] = torch.tensor(float(lp_batch["mask"].shape[0]))
            else:
                metrics["n_real_lp"] = torch.tensor(float(lp_batch["mask"].shape[0]))

            # The real pool the discriminator sees is the AE ROUND TRIP of a real LP, not
            # its crystal coordinates -- otherwise "was this decoded?" separates the two
            # classes perfectly and the generator learns nothing about LP realness.
            with torch.no_grad():
                lp_mask = lp_batch["mask"].bool()
                # Masked explicitly. In rollout mode `corrupt_batch` has already applied
                # the mask to `x_1` as a side effect; one-pass mode never calls it, and
                # relying on a side effect for one mode and not the other is how the two
                # real pools would quietly stop being the same pool.
                lp_clean = {
                    dm: lp_batch["x_1"][dm].detach() * lp_mask[..., None].to(lp_batch["x_1"][dm].dtype)
                    for dm in self._data_modes
                }
                real_decoded = self._decode(lp_clean, lp_mask)
                real_feats_live = self._featurize(real_decoded, lp_batch, lp_mask)
                pool = self._real_pool_index(lp_batch)
                if self.feature_scales is not None and not self.feature_scales.ready:
                    # Fed from the REAL rows of the TRAINING stream only -- the eval
                    # families never reach this branch. Both noise-related addons define
                    # their amplitude in units of these scales, so estimating them on
                    # anything held out would put evaluation data inside the objective.
                    keep_sc = real_feats_live if pool is None else {
                        k: v[pool.to(v.device)] for k, v in real_feats_live.items()
                    }
                    if int(keep_sc["mask"].shape[0]) > 0:
                        self.feature_scales.update(keep_sc)
                if pool is not None:
                    keep = pool.nonzero(as_tuple=True)[0]
                    if keep.numel():
                        self.reservoir.push_batch(
                            {k: v[keep] for k, v in real_feats_live.items()},
                            residue_lengths(lp_mask[keep]),
                        )
                    metrics["n_real_pool"] = torch.tensor(float(keep.numel()))
                else:
                    self.reservoir.push_batch(real_feats_live, residue_lengths(lp_mask))
                metrics["reservoir_size"] = torch.tensor(float(len(self.reservoir)))

            # Now that the clean snapshot is in the reservoir, the anchor may corrupt.
            pending = getattr(self, "_pending_bb_anchor", None)
            if pending is not None:
                self._pending_bb_anchor = None
                anchor_loss, _, anchor_met = self._real_lp_backbone_anchor(pending)
                self._probe_extra_terms["anchor"] = (anchor_loss, self.bb_anchor_weight)
                g_loss = g_loss + self.bb_anchor_weight * anchor_loss
                metrics["loss_bb_anchor"] = anchor_loss.detach()
                metrics.update(anchor_met)

        # -------------------------------------------------- CP -> LP generation
        fake_feats = None
        if cyclic_idx.any():
            cp_batch = slice_batch(batch, cyclic_idx, bs)
            n_cp = cp_batch["mask"].shape[0]
            keep = min(n_cp, self.cp2lp_max_cp)
            cp_batch = slice_batch(cp_batch, torch.arange(keep, device=batch["mask"].device), n_cp)

            # Several LPs per CP from different noise draws. `repeat_interleave` keeps the
            # copies of one CP adjacent, which is what makes the diversity metric below a
            # within-CP quantity rather than a between-CP one.
            if self.cp2lp_samples_per_cp > 1:
                idx = torch.arange(keep, device=batch["mask"].device).repeat_interleave(self.cp2lp_samples_per_cp)
                cp_batch = slice_batch(cp_batch, idx, keep)

            gen_batch, decoded, extras = self._generate(cp_batch)
            if self.cp2lp_mode == "backbone":
                aux_loss, aux_metrics = self._backbone_losses(gen_batch, extras)
                if (
                    self.bb_grad_probe_every > 0
                    and int(self.g_updates) % self.bb_grad_probe_every == 0
                ):
                    aux_metrics.update(self._log_backbone_grad_shares(gen_batch, extras))
            else:
                aux_loss, aux_metrics = self._generator_losses(gen_batch, decoded)
            g_loss = g_loss + aux_loss
            metrics.update(aux_metrics)
            metrics.update(extras.get("metrics", {}))
            metrics["n_generated"] = torch.tensor(float(gen_batch["mask"].shape[0]))

            # ---- arm C addon: local contrastive retention. Generator-side, and the only
            # addon that changes the generator's objective rather than the discriminator's.
            if self.nce is not None:
                mask_g = gen_batch["mask"].bool()
                src_dec = self._decode(gen_batch[SRC_CP_KEY], mask_g)
                nce_loss, nce_metrics = self.nce(
                    gen_atom37=decoded["coors_nm"],
                    src_atom37=src_dec["coors_nm"].detach(),
                    src_atom_mask=src_dec["atom_mask_eff"].detach(),
                    pep_mask=mask_g,
                    target_atom37=gen_batch["x_target"],
                    target_atom_mask=gen_batch["target_mask"],
                    target_mask=gen_batch["seq_target_mask"].bool(),
                )
                g_loss = g_loss + self.nce_weight * nce_loss
                metrics.update(nce_metrics)
                metrics["loss_nce_weighted"] = (self.nce_weight * nce_loss).detach()

            if self.adv_enabled:
                fake_feats = self._featurize(decoded, gen_batch, gen_batch["mask"].bool())
                # G UPDATES in one-pass mode, `global_step` in the legacy one. The legacy
                # arms were trained and measured against `global_step`, and silently
                # doubling their effective warmup would make a rerun differ from the runs
                # docs/old_docs/README_CP2LP.md reports. New mode, corrected counter.
                progress = (
                    int(self.g_updates)
                    if self.cp2lp_mode in ("onepass", "backbone")
                    else int(self.global_step)
                )
                if progress >= self.adv_warmup_steps:
                    # The discriminator is a FIXED FUNCTION during the generator's step.
                    # Without this the generator's `-D(fake)` term backprops into D's own
                    # parameters and is still sitting there when the D optimizer steps --
                    # training the discriminator to RAISE its score on fakes, i.e. the
                    # exact opposite of its objective. Freezing during the forward is what
                    # matters: requires_grad is read at graph-construction time, and
                    # gradient w.r.t. D's *inputs* (which is what the generator needs)
                    # flows regardless.
                    with self._frozen_discriminator():
                        # Arm B scores G through the SAME corruption D is trained under.
                        # Scoring G on clean features while D only ever sees corrupted ones
                        # would hand the generator gradient from a function D is not being
                        # fit to, which is a different (and unadvertised) algorithm.
                        # Gradient is preserved through the corruption -- it is an affine
                        # map of the features, so it passes gradient to the generator.
                        scored, level = self._maybe_corrupt(fake_feats)
                        adv = hinge_g_loss(self.discriminator(scored, noise_level=level))
                    self._probe_extra_terms["adversarial"] = (adv, self.adv_weight)
                    g_loss = g_loss + self.adv_weight * adv
                    metrics["loss_adv_g"] = adv.detach()

        # -------------------------------------------------- generator step
        # A batch can in principle carry no differentiable term (every row filtered out of
        # both branches). `manual_backward` on a graph-less tensor raises, so check rather
        # than let an edge batch kill a two-day run.
        # The opposing half of the gradient budget, on the SAME schedule and the same
        # parameters as the backbone terms. Here rather than beside them because `adv` does not
        # exist yet at that point, and the comparison is worthless if the two halves are
        # measured at different moments on different graphs.
        if (
            self.cp2lp_mode == "backbone"
            and self.bb_grad_probe_every > 0
            and int(self.g_updates) % self.bb_grad_probe_every == 0
            and self._probe_extra_terms
        ):
            metrics.update(self._grad_norms_at_head(self._probe_extra_terms))
        self._probe_extra_terms = {}

        if g_loss.requires_grad:
            self.manual_backward(g_loss / accum)
        else:
            logger.warning(f"CP2LP: step {self.global_step} produced no differentiable generator loss; skipping backward.")
        if (batch_idx + 1) % accum == 0:
            self._step_optimizer(opt_g, tag="g")
            self.g_updates += 1
            sch = self.lr_schedulers()
            if sch is not None:
                (sch[0] if isinstance(sch, (list, tuple)) else sch).step()
            self._maybe_save_milestone()
            self._maybe_stop()
        metrics["g_updates"] = self.g_updates.detach().float()

        # -------------------------------------------------- discriminator step
        if self.adv_enabled and fake_feats is not None and opt_d is not None:
            d_metrics = self._discriminator_step(fake_feats, opt_d, accum, batch_idx)
            metrics.update(d_metrics)

        metrics["loss_g_total"] = g_loss.detach()
        self._log_cp2lp(metrics, log_prefix, bs)
        return g_loss.detach()

    def _maybe_corrupt(self, feats: dict) -> tuple[dict, torch.Tensor | None]:
        """Arm B's corruption, or a pass-through. ``(features_to_score, noise_level)``.

        Returns ``(feats, None)`` in every other arm, and the ``None`` is what a
        ``noise_level_dim=0`` discriminator requires -- passing a zero level to a D that
        has no such input raises rather than being silently ignored, which is the
        behaviour that keeps the arms from drifting into each other.

        Levels are drawn per example from the shared pool, i.i.d. for real and for fake.
        The corruption is skipped entirely until the feature scales are frozen: before
        that the standardisation would divide by an estimate made from a handful of
        examples, and the amplitude would mean something different every step.
        """
        if self.dgan is None:
            return feats, None
        b = feats["mask"].shape[0]
        device = feats["mask"].device
        if self.feature_scales is None or not self.feature_scales.ready:
            return feats, torch.zeros(b, device=device)
        streams = self._streams()
        level = self.dgan.sample_levels(b, device, cpu_generator=streams.cpu)
        corrupted = self.dgan.corrupt(
            feats,
            scales=self.feature_scales,
            levels=level,
            col_masks=self.discriminator.continuous_column_masks(),
            generator=streams.dev,
        )
        return corrupted, level

    def _discriminator_step(self, fake_feats: dict, opt_d, accum: int, batch_idx: int) -> dict:
        """Hinge update on LENGTH-MATCHED real/fake pairs.

        Unmatched fakes are dropped rather than paired against a real of another length:
        peptide length is trivially readable from the feature block, so an unmatched batch
        would hand the discriminator a free label. When the reservoir cannot match
        anything the step is skipped and ``d_match_frac`` records it, which is the number
        to check first if the adversarial term looks inert.
        """
        device = fake_feats["mask"].device
        fake_lengths = residue_lengths(fake_feats["mask"])
        real_feats, matched = self.reservoir.draw_matched(
            fake_lengths.cpu(), device, max_length_slack=self.reservoir_slack
        )
        out = {"d_match_frac": torch.tensor(float(matched.float().mean()))}
        if real_feats is None:
            return out

        keep = matched.to(device).nonzero(as_tuple=True)[0]
        fake_det = {k: v[keep].detach() for k, v in fake_feats.items()}

        for _ in range(self.adv_d_steps):
            real_scored, real_level = self._maybe_corrupt(real_feats)
            fake_scored, fake_level = self._maybe_corrupt(fake_det)
            real_logits = self.discriminator(real_scored, noise_level=real_level)
            fake_logits = self.discriminator(fake_scored, noise_level=fake_level)
            d_loss = hinge_d_loss(real_logits, fake_logits)

            # The shortcut probe reads CLEAN features in every arm, including arm B. It is
            # a cross-arm diagnostic, and a probe fed corrupted features in one arm only
            # would not be comparable with the same number from the other four. It also
            # stays detached from the trunk, so this choice does not shape anything.
            probe_loss = None
            probe_real = self.discriminator.probe_forward(real_feats)
            if probe_real is not None:
                probe_fake = self.discriminator.probe_forward(fake_det)
                probe_loss = hinge_d_loss(probe_real, probe_fake)
                d_loss = d_loss + probe_loss

            # ---- arm A addon: approximate R1, on REAL features only, D-side only.
            if self.cp2lp_addon == "ar1":
                penalty, ar1_metrics = approx_r1_penalty(
                    self.discriminator,
                    real_feats,
                    scales=self.feature_scales,
                    sigma_rel=self.ar1_sigma_rel,
                    generator=self._streams().dev,
                )
                d_loss = d_loss + self.ar1_lambda * penalty
                out.update(ar1_metrics)
                out["loss_d_ar1_weighted"] = (self.ar1_lambda * penalty).detach()

            # Belt and braces beside the val_step guard at the top of `training_step`:
            # `d_loss` reaches the weights ONLY through the discriminator's parameters
            # (both feature sets are detached), so any context that drops that graph --
            # an enclosing `no_grad`, or the discriminator left frozen -- makes this
            # backward raise rather than no-op. Failing loudly once is better than a
            # silent skip, so this warns instead of passing quietly.
            if not d_loss.requires_grad:
                logger.warning(
                    f"CP2LP: discriminator loss has no graph at step {self.global_step}; "
                    "skipping the D update. Expected only if the discriminator is frozen "
                    "or this is running under no_grad."
                )
                break
            self.manual_backward(d_loss / accum)
            if (batch_idx + 1) % accum == 0:
                self._step_optimizer(opt_d, tag="d")

        out["loss_d"] = d_loss.detach()
        out["d_acc"] = torch.tensor(discriminator_accuracy(real_logits, fake_logits))
        out["d_real_logit"] = real_logits.detach().mean()
        out["d_fake_logit"] = fake_logits.detach().mean()

        # ---- arm B: controller state, and the CLEAN diagnostics beside the corrupted ones.
        # `d_acc` in this arm is accuracy on CORRUPTED features and falls as the controller
        # raises the noise -- which is the controller working, not the discriminator losing.
        # `d_acc_clean` is the number comparable with the other four arms, and it is also
        # the guard against the failure mode where raising the noise becomes a way to stop
        # D objecting to a chemically broken peptide.
        if self.dgan is not None:
            # The controller's statistic is the sign of the CORRUPTED real logits, as in
            # the reference (its `run_D` diffuses before scoring, and `Loss/signs/real` is
            # read off that output). Feeding it the clean logits would close the loop on a
            # quantity the corruption does not affect, and the noise level would stop
            # responding to the difficulty it is supposed to be controlling.
            self.dgan.observe(real_logits)
            out.update(self.dgan.maybe_adapt(int(self.global_step)))
            out["dgan_level_real_mean"] = (
                real_level.detach().mean() if real_level is not None else torch.tensor(0.0)
            )
            with torch.no_grad():
                was = self.discriminator.training
                self.discriminator.eval()
                try:
                    zero = torch.zeros(real_feats["mask"].shape[0], device=real_feats["mask"].device)
                    zero_f = torch.zeros(fake_det["mask"].shape[0], device=fake_det["mask"].device)
                    clean_real = self.discriminator(real_feats, noise_level=zero)
                    clean_fake = self.discriminator(fake_det, noise_level=zero_f)
                finally:
                    if was:
                        self.discriminator.train()
            out["d_acc_clean"] = torch.tensor(discriminator_accuracy(clean_real, clean_fake))
            out["d_real_logit_clean"] = clean_real.mean()
            out["d_fake_logit_clean"] = clean_fake.mean()
        if self.feature_scales is not None:
            out.update({k: torch.tensor(v) for k, v in self.feature_scales.summary().items()})
        if probe_real is not None:
            # The shortcut number. Near 0.5 means the discriminator is judging peptides;
            # climbing towards 1.0 means it can tell the datasets apart from the receptor
            # alone and the generator's adversarial gradient is not about LP quality.
            out["d_shortcut_probe_acc"] = torch.tensor(discriminator_accuracy(probe_real, probe_fake))
            out["loss_d_probe"] = probe_loss.detach()
        return out

    def _maybe_stop(self) -> None:
        """Ends the run at the configured GENERATOR-update budget, not at a global_step.

        Asked of the Trainer rather than imposed with ``Trainer(max_steps=...)`` because
        ``max_steps`` reads ``global_step``, which double-counts the adversarial arms. A
        budget set that way would give the GAN arms half the generator training of the
        control at the same nominal number -- which is exactly the confound this counter
        exists to remove.
        """
        if self.max_g_updates and int(self.g_updates) >= self.max_g_updates:
            # A WARM START THAT BEGINS AT ITS BUDGET IS A CONFIG ERROR, NOT A FINISHED RUN,
            # and it has to fail loudly. `g_updates` is a persistent buffer, so it travels
            # through `pretrain_ckpt_path` even though the optimizer states do not -- a warm
            # start from a milestone inherits that milestone's count. MEASURED: job 72084
            # loaded the torsion arm's g=10300 milestone against a budget of 10300, logged
            # "reached the budget" after ONE step, and EXITED 0 -- which in this project's
            # Slurm chains releases every `afterok` dependent behind a stage that did no
            # work. `global_step` distinguishes the two cases: a genuine resume has taken
            # thousands of steps, a fresh process that trips this has taken ~none.
            if int(self.global_step) <= 1:
                raise RuntimeError(
                    f"CP2LP: this run starts AT OR ABOVE its budget -- g_updates="
                    f"{int(self.g_updates)} loaded from the checkpoint, max_g_updates="
                    f"{self.max_g_updates}, global_step={int(self.global_step)}. Nothing "
                    "would be trained. `g_updates` is a persistent buffer and therefore "
                    "travels through `pretrain_ckpt_path`, so a warm start inherits the "
                    "source milestone's count: set `cp2lp.max_g_updates` to that count PLUS "
                    "the updates you want, and move `cp2lp.eval_checkpoint_g_updates` with "
                    "it (it is matched by equality, not threshold). See "
                    "configs/example/training_cp2lp_bb_release_warm.yaml for the arithmetic."
                )
            if self.trainer is not None and not self.trainer.should_stop:
                logger.info(
                    f"CP2LP: reached the budget of {self.max_g_updates} generator updates "
                    f"(global_step={int(self.global_step)}); stopping."
                )
                self.trainer.should_stop = True

    def _maybe_save_milestone(self) -> None:
        """Writes a RETAINED checkpoint at each configured G-update milestone.

        The rolling ``last.ckpt`` is overwritten every ``last_ckpt_every_n_steps``, so
        "evaluate at 1k, 5k and 10k updates" needs checkpoints that survive. Two reasons
        this is done here rather than with another ``ModelCheckpoint``:

        * ``every_n_train_steps`` triggers on ``global_step``, which is not the same
          quantity in an adversarial arm as in the control -- the milestones would land at
          different amounts of generator training in different arms, and the comparison
          would silently stop being matched;
        * the file naming then carries the number the comparison is indexed by.

        Both a raw and an ``-EMA`` copy, matching the convention ``cp2lp_generate.py``
        already expects, and produced through the EMA callback's own swap so the EMA file
        holds EMA weights in ``state_dict`` rather than buried in the optimizer state.
        """
        n = int(self.g_updates)
        if n not in self.eval_ckpt_g_updates or n in self._milestones_written or self.trainer is None:
            return
        self._milestones_written.add(n)
        try:
            import os

            from proteinfoundation.utils.ema_callback import EMA

            ckpt_dir = getattr(self.trainer.checkpoint_callback, "dirpath", None)
            if not ckpt_dir:
                logger.warning(f"CP2LP: no checkpoint dirpath; skipping the g={n} milestone.")
                return
            raw = os.path.join(ckpt_dir, f"g{n:06d}.ckpt")
            self.trainer.save_checkpoint(raw)
            ema_cb = next((c for c in self.trainer.callbacks if isinstance(c, EMA)), None)
            if ema_cb is not None:
                with ema_cb.save_ema_model(self.trainer):
                    self.trainer.save_checkpoint(os.path.join(ckpt_dir, f"g{n:06d}-EMA.ckpt"))
            logger.info(f"CP2LP: wrote the g={n} milestone checkpoint(s) to {ckpt_dir}")
        except Exception as exc:  # a failed milestone must not kill a two-day run
            logger.warning(f"CP2LP: could not write the g={n} milestone checkpoint: {exc!r}")

    def _step_optimizer(self, opt, tag: str) -> None:
        """Clips, NaN-guards and steps one optimizer.

        Manual optimization means Lightning drives NONE of this: it refuses to own
        gradient clipping or accumulation for a module that calls `optimizer.step()`
        itself, so `train.py` turns both off Trainer-side and they live here instead. The
        clip is the project's own (norm, 1.0) applied per optimizer -- the generator and
        the discriminator are clipped separately, which is what you want, since a shared
        norm would let a large discriminator gradient shrink the generator's.

        `skip_nan_grad` likewise has no Lightning implementation in play, and this arm has
        already produced one all-NaN gradient (padded-residue `atan2`), so the check is
        not theoretical: without it a single bad batch silently poisons every weight.
        """
        if bool(self.cfg_exp.opt.get("skip_nan_grad", False)):
            bad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for group in opt.param_groups
                for p in group["params"]
            )
            if bad:
                logger.warning(f"CP2LP: non-finite gradient in the {tag} optimizer at step {self.global_step}; skipping.")
                opt.zero_grad(set_to_none=True)
                return
        # PRE-clip norm, logged because the clip makes the loss terms compete for a fixed
        # budget instead of summing freely: once this exceeds `grad_clip_val` every term's
        # effective step is scaled by `clip / norm`, so the largest term silently throttles
        # the smallest. Diagnosing that on the first two arms meant reconstructing it from a
        # step-0 smoke and epoch means, because nothing logged it. Free to compute here --
        # the clip walks the same parameters anyway.
        with torch.no_grad():
            sq = torch.zeros((), device=self.device)
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        sq = sq + p.grad.detach().pow(2).sum()
            self._log_cp2lp({f"grad_norm_preclip_{tag}": sq.sqrt()}, "train", 1)
        if self.grad_clip_val and self.grad_clip_val > 0:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.grad_clip_val,
                gradient_clip_algorithm=self.grad_clip_algorithm,
            )
        opt.step()
        opt.zero_grad(set_to_none=True)

    def _log_cp2lp(self, metrics: dict, log_prefix: str, bs: int) -> None:
        for name, value in metrics.items():
            v = float(value)
            is_nan = math.isnan(v)
            self.log(
                f"{log_prefix}/cp2lp_{name}",
                0.0 if is_nan else v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=0 if is_nan else bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def generate_lp_for_cp(
        self,
        batch: dict,
        nsteps: int | None = None,
        samples_per_cp: int = 1,
        seed: int | None = None,
    ) -> dict:
        """Full-sampler CP -> LP generation for export and evaluation.

        Distinct from the training rollout in two ways that matter: it integrates the
        deployed number of steps rather than the short differentiable trajectory, and it
        carries no gradient. ``seed`` makes a draw reproducible, which the export records
        alongside the sample so any triplet can be regenerated.
        """
        if seed is not None:
            torch.manual_seed(seed)
        bs = batch["mask"].shape[0]
        batch = add_clean_samples(
            batch,
            self.cfg_exp.product_flowmatcher,
            self.autoencoder,
            local_latent_target=self.cfg_exp.get("local_latent_target", "sample"),
        )
        if samples_per_cp > 1:
            idx = torch.arange(bs, device=batch["mask"].device).repeat_interleave(samples_per_cp)
            batch = slice_batch(batch, idx, bs)

        if self.cp2lp_mode == "onepass":
            # EXPORT AND TRAINING GO THROUGH THE SAME FUNCTION. There is no sampler to fall
            # back to here, and `nsteps` has no meaning -- it is accepted and ignored only
            # because `cp2lp_generate.py` passes its default to both modes. Ignored loudly:
            # a silently honoured `--nsteps 200` would imply an integration that does not
            # happen, and the 24-vs-200-step discrepancy that made every training-side
            # geometry number uninformative in the rollout arms is exactly what this mode
            # removes.
            if nsteps is not None and not getattr(self, "_warned_nsteps_ignored", False):
                self._warned_nsteps_ignored = True
                logger.warning(
                    f"cp2lp.mode=onepass ignores nsteps={nsteps}: the editor is ONE network "
                    "evaluation, not an ODE. Nothing on this path integrates."
                )
            gen_batch, decoded, _ = self.generate_onepass(batch)
            samples = None
        else:
            saved_nsteps = self.cp2lp_nsteps
            saved_grad = self.cp2lp_grad_steps
            try:
                self.cp2lp_nsteps = int(nsteps) if nsteps is not None else saved_nsteps
                self.cp2lp_grad_steps = 1
                gen_batch, decoded, samples = self._generate_from_cp(batch)
            finally:
                self.cp2lp_nsteps = saved_nsteps
                self.cp2lp_grad_steps = saved_grad

        mask = gen_batch["mask"].bool()
        src_cp = self._decode(gen_batch[SRC_CP_KEY], mask)
        return {
            "gen_batch": gen_batch,
            "decoded": decoded,
            "src_decoded": src_cp,
            "samples": samples,
            "mask": mask,
            "terminal_gap_nm": terminal_gap_nm(decoded["coors_nm"], mask),
            "src_terminal_gap_nm": terminal_gap_nm(src_cp["coors_nm"], mask),
        }
