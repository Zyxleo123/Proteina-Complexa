"""``CP2LPGenerator`` -- the adversarially-trained CP -> LP generator.

A ``Proteina`` subclass, so the flow network, the frozen mixed AE, the samplers and the
checkpoint plumbing are all inherited rather than reimplemented. What it adds is a second
training branch and a second optimizer.

ONE OPTIMIZER STEP, THREE TERMS.

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

from proteinfoundation.cp2lp.conditioning import (
    SRC_CP_KEY,
    attach_source_cp_condition,
    drop_source_cp_condition,
    request_linear_topology,
    source_cp_terminal_gap_nm,
    warn_if_condition_unused,
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
        )

        self.real_lp_sources = [str(s) for s in (cfg.get("real_lp_sources", None) or [])]

        # Clipping the Trainer cannot do for us under manual optimization. Defaults match
        # what `train.py` applies to every other arm, so this is the same safeguard in a
        # different place rather than a new knob with a new value.
        self.grad_clip_val = float(cfg.get("grad_clip_val", 1.0))
        self.grad_clip_algorithm = str(cfg.get("grad_clip_algorithm", "norm"))

        rcfg = cfg.get("reservoir", None) or {}
        self.reservoir = RealLPFeatureReservoir(
            capacity_per_length=int(rcfg.get("capacity_per_length", 64)),
            seed=int(cfg_exp.get("seed", 0)),
        )
        self.reservoir_slack = int(rcfg.get("max_length_slack", 0))

        warn_if_condition_unused(list(cfg_exp.nn.feats_seq), list(cfg_exp.nn.feats_pair_repr))
        logger.info(
            f"CP2LP generator enabled: samples_per_cp={self.cp2lp_samples_per_cp}, "
            f"max_cp_per_step={self.cp2lp_max_cp}, rollout nsteps={self.cp2lp_nsteps} "
            f"grad_steps={self.cp2lp_grad_steps}, adversarial={self.adv_enabled} "
            f"(weight={self.adv_weight}, warmup={self.adv_warmup_steps})"
        )
        if not self.adv_enabled:
            logger.info("CP2LP: ADVERSARIAL TERM OFF -- this is the contact-loss-only ablation arm.")

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
        probs = torch.softmax(decoded["seq_logits"], dim=-1)
        n_classes = self.discriminator.res_emb.in_features
        if probs.shape[-1] < n_classes:
            probs = torch.nn.functional.pad(probs, (0, n_classes - probs.shape[-1]))
        elif probs.shape[-1] > n_classes:
            probs = probs[..., :n_classes]
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
        real_feats_live = None
        if linear_idx.any():
            lp_batch = slice_batch(batch, linear_idx, bs)
            flow_loss, n_lp = self._real_lp_flow_loss(lp_batch)
            g_loss = g_loss + self.w_real_lp_flow * flow_loss
            metrics["loss_real_lp_flow"] = flow_loss.detach()
            metrics["n_real_lp"] = torch.tensor(float(n_lp))

            # The real pool the discriminator sees is the AE ROUND TRIP of a real LP, not
            # its crystal coordinates -- otherwise "was this decoded?" separates the two
            # classes perfectly and the generator learns nothing about LP realness.
            with torch.no_grad():
                lp_clean = {dm: lp_batch["x_1"][dm].detach() for dm in self._data_modes}
                lp_mask = lp_batch["mask"].bool()
                real_decoded = self._decode(lp_clean, lp_mask)
                real_feats_live = self._featurize(real_decoded, lp_batch, lp_mask)
                pool = self._real_pool_index(lp_batch)
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

            gen_batch, decoded, _ = self._generate_from_cp(cp_batch)
            aux_loss, aux_metrics = self._generator_losses(gen_batch, decoded)
            g_loss = g_loss + aux_loss
            metrics.update(aux_metrics)
            metrics["n_generated"] = torch.tensor(float(gen_batch["mask"].shape[0]))

            if self.adv_enabled:
                fake_feats = self._featurize(decoded, gen_batch, gen_batch["mask"].bool())
                if int(self.global_step) >= self.adv_warmup_steps:
                    # The discriminator is a FIXED FUNCTION during the generator's step.
                    # Without this the generator's `-D(fake)` term backprops into D's own
                    # parameters and is still sitting there when the D optimizer steps --
                    # training the discriminator to RAISE its score on fakes, i.e. the
                    # exact opposite of its objective. Freezing during the forward is what
                    # matters: requires_grad is read at graph-construction time, and
                    # gradient w.r.t. D's *inputs* (which is what the generator needs)
                    # flows regardless.
                    with self._frozen_discriminator():
                        adv = hinge_g_loss(self.discriminator(fake_feats))
                    g_loss = g_loss + self.adv_weight * adv
                    metrics["loss_adv_g"] = adv.detach()

        # -------------------------------------------------- generator step
        # A batch can in principle carry no differentiable term (every row filtered out of
        # both branches). `manual_backward` on a graph-less tensor raises, so check rather
        # than let an edge batch kill a two-day run.
        if g_loss.requires_grad:
            self.manual_backward(g_loss / accum)
        else:
            logger.warning(f"CP2LP: step {self.global_step} produced no differentiable generator loss; skipping backward.")
        if (batch_idx + 1) % accum == 0:
            self._step_optimizer(opt_g, tag="g")
            sch = self.lr_schedulers()
            if sch is not None:
                (sch[0] if isinstance(sch, (list, tuple)) else sch).step()

        # -------------------------------------------------- discriminator step
        if self.adv_enabled and fake_feats is not None and opt_d is not None:
            d_metrics = self._discriminator_step(fake_feats, opt_d, accum, batch_idx)
            metrics.update(d_metrics)

        metrics["loss_g_total"] = g_loss.detach()
        self._log_cp2lp(metrics, log_prefix, bs)
        return g_loss.detach()

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
            real_logits = self.discriminator(real_feats)
            fake_logits = self.discriminator(fake_det)
            d_loss = hinge_d_loss(real_logits, fake_logits)

            probe_loss = None
            probe_real = self.discriminator.probe_forward(real_feats)
            if probe_real is not None:
                probe_fake = self.discriminator.probe_forward(fake_det)
                probe_loss = hinge_d_loss(probe_real, probe_fake)
                d_loss = d_loss + probe_loss

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
        if probe_real is not None:
            # The shortcut number. Near 0.5 means the discriminator is judging peptides;
            # climbing towards 1.0 means it can tell the datasets apart from the receptor
            # alone and the generator's adversarial gradient is not about LP quality.
            out["d_shortcut_probe_acc"] = torch.tensor(discriminator_accuracy(probe_real, probe_fake))
            out["loss_d_probe"] = probe_loss.detach()
        return out

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
