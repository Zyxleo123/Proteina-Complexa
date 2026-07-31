import copy
import math
import os
import random
from functools import partial
from typing import Literal

import lightning as L
import torch
from jaxtyping import Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf
from torchmetrics import MeanMetric

from proteinfoundation.eval.ae_reconstruction_eval import (
    run_ae_reconstruction_eval,
    run_four_way_decode_eval,
)
from proteinfoundation.cyclization.inference import attach_cyclization_prediction
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    CYCLIC_METRIC_COUNT_FOR_SUFFIX,
    NM_TO_ANG,
    cyclic_geometry_metrics,
)
from proteinfoundation.eval.cyclic_reconstruction_metrics import CYCLIC_COUNT_SUFFIXES
from proteinfoundation.eval.sampled_binder_metrics import (
    CYCLIC_PRED_ONLY_SUFFIXES,
    atom37_mask_from_aatype,
    sampled_binder_metrics,
)
from proteinfoundation.eval.surface_metrics import batch_surface_agreement_from_ca
from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from proteinfoundation.logging.metric_schema import finalize_metrics, init_metric_dict
from proteinfoundation.nn.genie2 import Genie2Denoiser
from proteinfoundation.utils.file_utils import create_dir as _create_dir
from proteinfoundation.utils.sample_utils import add_clean_samples, sample_formatting
from proteinfoundation.utils.training_handlers import handle_batch_conditioning
from proteinfoundation.utils.validation_utils import (
    clean_validation_files,
    get_pdb_novelty_metric,
    get_structural_metrics,
)

# Architecture selection: v1 is the default.
# Set USE_V2_COMPLEXA_ARCH=True in .env to use v2 (ligand / AME models).
_USE_V2 = os.getenv("USE_V2_COMPLEXA_ARCH", "False") == "True"
if _USE_V2:
    from proteinfoundation.nn.local_latents_transformer_v2 import (
        LocalLatentsTransformer,
    )
else:
    from proteinfoundation.nn.local_latents_transformer import (
        LocalLatentsTransformer,
    )

from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.rewards.reward_utils import compute_reward_from_samples, initialize_reward_model
from proteinfoundation.search.search_factory import instantiate_refinement, instantiate_search
from proteinfoundation.search.search_utils import (
    append_samples,
    clone_sample_dict,
    combine_lookahead_and_final,
    expand_hotspot_mask,
)
from proteinfoundation.utils.fold_utils import extract_cath_code_from_batch
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

create_dir = rank_zero_only(_create_dir)


def _index_pytree(obj, idx: torch.Tensor, batch_size: int):
    """Recursively indexes every batch-first tensor in a (possibly nested) dict along dim 0.

    Used to build a "sub-batch"/"sub-nn_out" restricted to the samples in one `t` bin, so
    diagnostics that are otherwise only computed once per whole batch (e.g. the four-way
    decode eval) can be re-run per bin. Only tensors whose leading dim equals `batch_size`
    are indexed; everything else (scalars, strings, already batch-size-independent tensors)
    is passed through unchanged.
    """
    if torch.is_tensor(obj):
        if obj.dim() >= 1 and obj.shape[0] == batch_size:
            return obj[idx]
        return obj
    if isinstance(obj, dict):
        return {k: _index_pytree(v, idx, batch_size) for k, v in obj.items()}
    return obj


def _cyclic_metric_weight(key: str, metrics: dict[str, float], default: int) -> float:
    """Epoch-aggregation weight for one logged metric.

    Cyclic metrics are averages over only the examples of their cyclization family (see
    `cyclic_geometry_metrics`), so weighting them by the full batch size overweights
    batches that happened to contain few -- or, worse, credits a batch that contained
    none. Weight each by its family's `n_valid_*` count instead, which
    `cyclic_geometry_metrics` emits under the same prefix. Everything else keeps the
    batch size.
    """
    prefix, _, suffix = key.rpartition("/")
    count_suffix = CYCLIC_METRIC_COUNT_FOR_SUFFIX.get(suffix)
    if count_suffix is None:
        return default
    return float(metrics.get(f"{prefix}/{count_suffix}", default))


class Proteina(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None):
        super().__init__()
        if not os.environ.get("USE_V2_COMPLEXA_ARCH"):
            logger.info("USE_V2_COMPLEXA_ARCH not set, using v1 architecture (default)")
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only for inference runs
        self.validation_output_lens = {}
        self.validation_output_data = []
        # True only when train.py wires a GenDataset val loader before the structure val loader.
        self._val_has_lens_dataloader = False
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.val_path_tmp = os.path.join(self.store_dir, "val_samples")
        # create_dir(self.val_path_tmp)

        if "local_latents" in cfg_exp.product_flowmatcher:
            if autoencoder_ckpt_path is not None:
                # Allow adding new keys
                logger.info(f"Manually setting autoencoder_ckpt_path to {autoencoder_ckpt_path}")
                OmegaConf.set_struct(cfg_exp, False)
                # Update the configuration with the new key-value pair
                cfg_exp.autoencoder_ckpt_path = autoencoder_ckpt_path
                # Re-enable struct mode if needed
                OmegaConf.set_struct(cfg_exp, True)

            # `freeze_autoencoder=false` unfreezes the AE for *joint* training: the flow loss
            # then backprops into the encoder AND we add the AE's own recon+KL objective (see
            # training_step) so the latent space is anchored instead of collapsing to a
            # trivially-predictable target. Default keeps the historical frozen behaviour.
            self.freeze_autoencoder = bool(cfg_exp.get("freeze_autoencoder", True))
            self.ae_reg_weight = float(cfg_exp.get("ae_reg_weight", 1.0))
            self.autoencoder, self.latent_dim = self.load_autoencoder(
                cfg_exp, freeze_params=self.freeze_autoencoder
            )
            if self.autoencoder is not None:
                logger.info(
                    f"Autoencoder loaded (freeze={self.freeze_autoencoder}, "
                    f"ae_reg_weight={self.ae_reg_weight}). "
                    + ("Frozen (latents are a fixed target)." if self.freeze_autoencoder
                       else "JOINT training: encoder+decoder are trainable and regularized by recon+KL.")
                )
                # Optional channel-wise latent normalization (default disabled, identical to
                # historical behavior). Stats are precomputed offline (see
                # `script_utils/compute_latent_stats.py`) and stashed as a plain attribute on
                # the autoencoder instance so `get_clean_sample`/`format_sample_local_latents`
                # (which only receive the autoencoder, not the full cfg_exp) can pick them up
                # transparently for both the flow target (normalize) and decoding (unnormalize).
                latent_norm_cfg = cfg_exp.get("latent_normalization", None) or {}
                self.autoencoder.latent_norm_stats = None
                if bool(latent_norm_cfg.get("enabled", False)):
                    stats_path = latent_norm_cfg.get("stats_path", None)
                    if not stats_path:
                        raise ValueError("latent_normalization.enabled=true requires latent_normalization.stats_path")
                    stats = torch.load(stats_path, map_location="cpu")
                    norm_mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
                    norm_std = torch.as_tensor(stats["std"], dtype=torch.float32)
                    assert norm_mean.shape[-1] == self.latent_dim and norm_std.shape[-1] == self.latent_dim, (
                        f"latent_normalization stats dim {norm_mean.shape[-1]} != latent_dim {self.latent_dim}"
                    )
                    self.autoencoder.latent_norm_stats = {"mean": norm_mean, "std": norm_std}
                    logger.info(f"Loaded latent normalization stats from {stats_path}")
            # Add right latent dimensionality in the config file, needed to instantiate the flow matcher below
            if self.autoencoder is not None:
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim

            # AE reconstruction diagnostic (Part 2) and four-way decode diagnostic (Part 3):
            # cyclic-geometry-aware reconstruction metrics logged during validation to isolate
            # whether low latent/bb_ca loss corresponds to good decoded (cyclic) geometry, and
            # whether failures come from the decoder, predicted Ca, predicted latent, or both.
            # Default ON whenever an autoencoder is configured (opt-out via
            # `++reconstruction_eval_enabled=false` / `++four_way_decode_eval_enabled=false`);
            # both are no-ops (never called) when `self.autoencoder is None`.
            self.reconstruction_eval_enabled = bool(cfg_exp.get("reconstruction_eval_enabled", True))
            self.four_way_decode_eval_enabled = bool(cfg_exp.get("four_way_decode_eval_enabled", True))
            self._ae_source_logged = False

        # SAMPLING validation (`validation_step_generate`). Everything above is teacher-forced --
        # `x_t` is built from the true `x_1`, so the model is handed most of the answer and can look
        # healthy while its ODE-integrated samples are garbage. This is the only metric group that
        # scores what the model actually generates. Off by default (it costs an ODE integration).
        self.val_gen_cfg = cfg_exp.get("val_generation", None)
        self.val_gen_enabled = bool(self.val_gen_cfg.get("enabled", False)) if self.val_gen_cfg else False
        self._val_gen_sampler_warned = False

        # Optional aux: pull predicted binder CA toward the conditioned surface patch.
        # Useful as a gate-opening lever when zero-init keeps ∂L/∂Δ = g ≈ 0.
        surf_loss_cfg = cfg_exp.get("surface_loss", None) or {}
        self.surface_loss_enabled = bool(surf_loss_cfg.get("enabled", False))
        self.surface_loss_weight = float(surf_loss_cfg.get("loss_weight", 0.1))
        self.surface_loss_t_lower = float(surf_loss_cfg.get("t_lower_lim", 0.5))

        # NaN-safe epoch aggregation for metrics that are legitimately NaN on some steps (empty
        # t-bins, absent cyclization chemistries, ...). See `log_nan_safe` for why `self.log`'s
        # `reduce_fx` cannot do this. Populated lazily, keyed by sanitized metric name.
        self._nan_mean_metrics = torch.nn.ModuleDict()

        self.fm = ProductSpaceFlowMatcher(cfg_exp)

        # Neural network
        if cfg_exp.nn.name == "ca_af3":
            self.nn = ProteinTransformerAF3(**cfg_exp.nn)
        # elif cfg_exp.nn.name == "ca_af3_int":
        #     self.nn = ProteinTransformerAF3Int(**cfg_exp.nn)
        elif cfg_exp.nn.name == "local_latents_transformer":
            self.nn = LocalLatentsTransformer(**cfg_exp.nn, latent_dim=self.latent_dim)
        # elif cfg_exp.nn.name == "local_latents_transformer_int":
        #     self.nn = LocalLatentsTransformerInt(
        #         **cfg_exp.nn, latent_dim=self.latent_dim
        #     )
        elif cfg_exp.nn.name == "ca_genie2":
            self.nn = Genie2Denoiser(**cfg_exp.nn)
        else:
            raise OSError(f"Wrong nn selected for CAFlow {cfg_exp.nn.name}")

        # Scaling laws stuff
        self.nflops = 0
        self.nsamples_processed = 0
        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        # For autoguidance, overridden in `self.configure_inference`
        self.nn_ag = None

        self._init_cyclization_head(cfg_exp)
        self._init_cyclization_bond_loss(cfg_exp)

    def _init_cyclization_head(self, cfg_exp):
        """Optionally builds the CPSea cyclization-linkage prediction head.

        This is a classifier-only auxiliary head (see `proteinfoundation.cyclization`):
        it predicts the single cyclization edge `(i, j, type)` of a cyclic peptide
        binder. Entirely opt-in via `cyclization.enabled` and isolated from the rest
        of the model -- disabled by default, so it never changes existing behavior.
        """
        cyclization_cfg = cfg_exp.get("cyclization", None) or {}
        self.cyclization_enabled = bool(cyclization_cfg.get("enabled", False))
        self.cyclization_loss_weight = float(cyclization_cfg.get("loss_weight", 0.1))
        self.cyclization_detach_link_inputs = bool(cyclization_cfg.get("detach_link_inputs", True))
        self.cyclization_force_gold_valid = bool(cyclization_cfg.get("force_gold_valid", True))
        self.cyclization_allow_asn_gln_isopeptide = bool(cyclization_cfg.get("allow_asn_gln_isopeptide", True))
        # CPSea cyclizes between the termini in 520/520 sampled binders, all three types. Holding
        # every type to that pair (as MAINCHAIN already is) removes the inference-time escape hatch
        # that let disulfide/isopeptide argmax onto whichever pair was already closest to a bond.
        self.cyclization_terminal_only = bool(cyclization_cfg.get("terminal_only", False))
        self.cyclization_type_conditioning = bool(cyclization_cfg.get("type_conditioning", False))
        self.cyclization_target_type = self._resolve_cyclization_target_type(cyclization_cfg.get("target_type", None))
        self.cyclization_head = None
        if not self.cyclization_enabled:
            return

        if "local_latents" not in cfg_exp.product_flowmatcher or getattr(self, "autoencoder", None) is None:
            raise ValueError(
                "cyclization.enabled=true requires the 'local_latents' data mode with an "
                "autoencoder (decoded AA probabilities are used as head input)."
            )

        from proteinfoundation.cyclization import CyclizationLinkHead

        hidden_dim = int(cyclization_cfg.get("hidden_dim", 256))
        single_dim = self.latent_dim + 20  # predicted clean local latent + decoded AA probabilities
        self.cyclization_head = CyclizationLinkHead(single_dim=single_dim, hidden_dim=hidden_dim)
        logger.info(
            f"Cyclization link head enabled (single_dim={single_dim}, hidden_dim={hidden_dim}, "
            f"loss_weight={self.cyclization_loss_weight}, detach_link_inputs={self.cyclization_detach_link_inputs}, "
            f"type_conditioning={self.cyclization_type_conditioning}, target_type={self.cyclization_target_type})"
        )

        if self.cyclization_type_conditioning:
            self._warn_on_vacuous_type_conditioning(cfg_exp)

    def _init_cyclization_bond_loss(self, cfg_exp):
        """Optionally enables the two-sided closing-bond distance loss.

        Complementary to, and independent of, the linkage head: the head cannot
        *perceive* whether the bond closes (it is a classifier over endpoints), and
        nothing else in the objective says the bond *length* matters. Opt-in via
        `cyclization.bond_loss.enabled`; off by default, so existing runs are
        unchanged.
        """
        bond_cfg = (cfg_exp.get("cyclization", None) or {}).get("bond_loss", None) or {}
        self.cyclization_bond_loss_enabled = bool(bond_cfg.get("enabled", False))
        self.cyclization_bond_loss_weight = float(bond_cfg.get("loss_weight", 0.05))
        self.cyclization_bond_loss_t_lower = float(bond_cfg.get("t_lower_lim", 0.5))
        if not self.cyclization_bond_loss_enabled:
            return

        if "local_latents" not in cfg_exp.product_flowmatcher or getattr(self, "autoencoder", None) is None:
            raise ValueError(
                "cyclization.bond_loss.enabled=true requires the 'local_latents' data mode with an "
                "autoencoder: the bond distance only exists once the predicted latents are decoded "
                "to all-atom coordinates."
            )
        logger.info(
            f"Cyclization bond loss enabled (loss_weight={self.cyclization_bond_loss_weight}, "
            f"t_lower_lim={self.cyclization_bond_loss_t_lower})"
        )

        # Angle/dihedral terms ride on the SAME decoded structure as the distance term.
        # Decoding is the expensive part of this loss, so computing them here rather than
        # in a second pass makes the extra geometry supervision nearly free. They are
        # nested under bond_loss for that reason: without the decode there is nothing to
        # measure, so they cannot be enabled independently of it.
        geom_cfg = bond_cfg.get("geometry", None) or {}
        self.cyclization_geometry_enabled = bool(geom_cfg.get("enabled", False))
        self.cyclization_geometry_weight = float(geom_cfg.get("loss_weight", 0.05))
        self.cyclization_geometry_w_angle = float(geom_cfg.get("w_angle", 1.0))
        self.cyclization_geometry_w_dihedral = float(geom_cfg.get("w_dihedral", 1.0))
        if self.cyclization_geometry_enabled:
            logger.info(
                f"Cyclization linkage-geometry loss enabled (loss_weight={self.cyclization_geometry_weight}, "
                f"w_angle={self.cyclization_geometry_w_angle}, w_dihedral={self.cyclization_geometry_w_dihedral})"
            )

    def _warn_on_vacuous_type_conditioning(self, cfg_exp):
        """Warns about the two ways type conditioning silently degrades into a no-op."""
        # 1. Without `cyclization_type_emb` in the denoiser's conditioning features, the
        #    requested type reaches the head but never the flow model -- so generation is
        #    still cyclization-blind and can emit a sequence that cannot support the request.
        feats_cond_seq = list(cfg_exp.nn.get("feats_cond_seq", []) or [])
        if "cyclization_type_emb" not in feats_cond_seq:
            logger.warning(
                "cyclization.type_conditioning=true but 'cyclization_type_emb' is not in "
                f"nn.feats_cond_seq ({feats_cond_seq}). The requested type will condition the "
                "linkage head but NOT the denoiser, so the model may generate sequences that "
                "cannot support it. Use configs/nn/local_latents_score_nn_640M_binder_cyc.yaml."
            )

        # 2. Every CPSea row carries a real type, so dropout is the only source of UNSPECIFIED
        #    rows -- at 0.0 the null token is never trained and the head loses its cross-type
        #    negatives, so it stops being able to discriminate types at all.
        dropout = float(cfg_exp.get("training", {}).get("cyclization_type_dropout_rate", 0.0) or 0.0)
        if dropout <= 0:
            logger.warning(
                "cyclization.type_conditioning=true but training.cyclization_type_dropout_rate=0. "
                "The UNSPECIFIED token will never be trained and the head will see no cross-type "
                "negatives. Set it to ~0.15 unless you deliberately want a conditional-only model."
            )

    @staticmethod
    def _resolve_cyclization_target_type(target_type):
        """Resolves the inference-time `cyclization.target_type` config value to an int index.

        Accepts a name ("disulfide"), an int, or None/"unspecified" meaning "no type
        requested" (which reproduces the unconditional joint behavior).
        """
        from proteinfoundation.cyclization.constants import NAME_TO_CYCLIZATION_TYPE

        if target_type is None:
            return None
        if isinstance(target_type, str):
            key = target_type.strip().lower()
            if key in ("", "null", "none", "unspecified", "any"):
                return None
            if key not in NAME_TO_CYCLIZATION_TYPE:
                raise ValueError(
                    f"Unknown cyclization.target_type '{target_type}'. "
                    f"Expected one of {sorted(NAME_TO_CYCLIZATION_TYPE)} or null."
                )
            return NAME_TO_CYCLIZATION_TYPE[key]
        return int(target_type)

    def load_autoencoder(self, cfg_exp, freeze_params=True):
        """Loads autoencoder, if required."""
        if "autoencoder_ckpt_path" in cfg_exp:  # for new runs trained with refactored codebase
            ae_ckp_path = cfg_exp.autoencoder_ckpt_path
        elif (
            "autoencoder_ckpt_path" in cfg_exp.product_flowmatcher.local_latents
        ):  # for old runs trained with old codebase
            ae_ckp_path = cfg_exp.product_flowmatcher.local_latents.autoencoder_ckpt_path
        else:
            raise ValueError("No autoencoder checkpoint path provided")

        if ae_ckp_path is None:
            return None, None

        # Load and freeze parameters
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    def _surface_param_ids(self) -> set[int]:
        """Parameter ids belonging to the optional surface encoder / cross-attn stack."""
        nn = getattr(self, "nn", None)
        if nn is None or not getattr(nn, "enable_surface", False):
            return set()
        ids: set[int] = set()
        for mod_name in ("surface_encoder", "binder_surface_pair_feats", "surface_cross_layers"):
            mod = getattr(nn, mod_name, None)
            if mod is None:
                continue
            ids.update(id(p) for p in mod.parameters())
        return ids

    def configure_optimizers(self):
        base_lr = self.cfg_exp.opt.lr
        ae_lr = self.cfg_exp.get("ae_lr", None)
        ae_lr_scale = self.cfg_exp.get("ae_lr_scale", None)
        surface_lr_scale = self.cfg_exp.get("surface_lr_scale", None)

        ae = getattr(self, "autoencoder", None)
        use_separate_ae_lr = (
            ae is not None
            and not getattr(self, "freeze_autoencoder", True)
            and (ae_lr is not None or ae_lr_scale is not None)
        )
        surface_ids = self._surface_param_ids()
        use_surface_lr = surface_lr_scale is not None and len(surface_ids) > 0
        trainable = [p for p in self.parameters() if p.requires_grad]

        if use_separate_ae_lr or use_surface_lr:
            # Optional dedicated LRs for AE and/or surface stack. Default (both off) keeps the
            # historical single-optimizer, single-lr behavior exactly.
            ae_param_ids = {id(p) for p in ae.parameters()} if use_separate_ae_lr else set()
            ae_params = [p for p in trainable if id(p) in ae_param_ids]
            surface_params = [
                p for p in trainable if id(p) in surface_ids and id(p) not in ae_param_ids
            ]
            other_params = [
                p for p in trainable if id(p) not in ae_param_ids and id(p) not in surface_ids
            ]
            groups: list[dict] = []
            if other_params:
                groups.append({"params": other_params, "lr": base_lr})
            if surface_params:
                if use_surface_lr:
                    surf_lr = base_lr * float(surface_lr_scale)
                    logger.info(
                        f"Using separate surface optimizer LR: {surf_lr} "
                        f"(scale={surface_lr_scale} × base_lr={base_lr}, n_params={len(surface_params)})"
                    )
                    groups.append({"params": surface_params, "lr": surf_lr})
                elif groups:
                    groups[0]["params"].extend(surface_params)
                else:
                    groups.append({"params": surface_params, "lr": base_lr})
            if use_separate_ae_lr and ae_params:
                resolved_ae_lr = (
                    float(ae_lr) if ae_lr is not None else base_lr * float(ae_lr_scale)
                )
                logger.info(
                    f"Using separate AE optimizer LR: {resolved_ae_lr} (flow/other params LR: {base_lr})"
                )
                groups.append({"params": ae_params, "lr": resolved_ae_lr})
            optimizer = torch.optim.Adam(groups)
        else:
            optimizer = torch.optim.Adam(trainable, lr=base_lr)

        # Optional linear LR warmup (0 -> lr over warmup_steps, then constant). Helps when
        # finetuning pretrained weights onto an out-of-distribution data regime (e.g. short
        # cyclic peptides). Backward compatible: warmup_steps<=0 keeps the bare Adam optimizer.
        # When separate param groups are used, the same multiplier is applied to every group,
        # preserving the base_lr/ae_lr (and surface) ratio throughout warmup.
        warmup_steps = int(self.cfg_exp.opt.get("warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return optimizer

        def _warmup(step: int) -> float:
            return min(1.0, (step + 1) / warmup_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except (KeyError, AttributeError):
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def call_nn(
        self,
        batch: dict[str, torch.Tensor],
        n_recycle: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Calls NN with recycling. Should this be here or in the NN? Possibly better here,
        in case we want to recycle using decoder for some approach, etc, and this is akin
        to self conditioning, also here.
        Also, if we want to recycle clean sample predictions... Then we'd need this here,
        as the nn does not know about relations between v, x1, ...
        """
        # First call
        nn_out = self.nn(batch)

        # Recycle n_recycle times detaching gradients and updating input
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
            batch["x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
            nn_out = self.nn(batch)

        # Final prediction
        return nn_out

    def predict_for_sampling(
        self,
        batch: dict,
        mode: Literal["full", "ucond"],
        n_recycle: int = 0,
    ) -> tuple[dict[str, torch.Tensor] | float | None]:
        """
        This function predicts clean samples for multiple models:
        x_pred, the 'original' model, if mode == full
        x_pred_ucond, the unconditional model, , if mode == ucond

        TODO: Need to update to include autoguidance again

        These predictions will later be used to sample with guidance and autoguidance.

        Args:
            batch: Dict
            mode: str

        Returns:
            x_pred (tensor) for the requested mode
        """
        if mode == "full":
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
        elif mode == "ucond":
            assert "cath_code" in batch, "Only support CFG when cath_code is provided"
            uncond_batch = batch.copy()
            uncond_batch.pop("cath_code")
            nn_out = self.call_nn(uncond_batch, n_recycle=n_recycle)
        else:
            raise OSError(f"Wrong {mode} passed to `predict_for_sampling`")

        return nn_out

    def skip_forward_pass(self, batch: dict, batch_idx: int):
        """
        Skips the forward pass and returns 0.
        """
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def training_step(self, batch: dict, batch_idx: int):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"

        batch = add_clean_samples(
            batch,
            self.cfg_exp.product_flowmatcher,
            getattr(self, "autoencoder", None),
            local_latent_target=self.cfg_exp.get("local_latent_target", "sample"),
            detach_latent_target_for_flow=bool(self.cfg_exp.get("detach_latent_target_for_flow", False)),
        )

        if self.cfg_exp.get("log_latent_diagnostics", False):
            self.log_latent_diagnostics(batch=batch, log_prefix=log_prefix)

        # Corrupt the batch
        batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t, mask
        bs, n = batch["mask"].shape

        # Handle conditioning variables (safe config getters; missing keys default to disabled)
        batch, n_recycle = handle_batch_conditioning(
            batch,
            bs,
            self.cfg_exp.training,
            self.call_nn,
            self.fm,
        )

        nn_out = self.call_nn(batch, n_recycle=n_recycle)
        losses = self.fm.compute_loss(
            batch=batch,
            nn_out=nn_out,
        )  # Dict[str, Tensor w.batch shape [*]]

        self.log_losses(bs=bs, losses=losses, log_prefix=log_prefix, batch=batch)
        train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])
        self.log_surface_gates(bs=bs, log_prefix=log_prefix)

        cyclization_metrics = {}
        cyclization_loss_value = None
        if self.cyclization_enabled:
            cyclization_loss, cyclization_metrics = self.compute_cyclization_loss(
                batch, nn_out, log_prefix=log_prefix, bs=bs
            )
            cyclization_loss_value = float(cyclization_loss.detach().item())
            train_loss = train_loss + self.cyclization_loss_weight * cyclization_loss

        if getattr(self, "cyclization_bond_loss_enabled", False):
            bond_loss, _ = self.compute_cyclization_bond_loss(batch, nn_out, log_prefix=log_prefix, bs=bs)
            train_loss = train_loss + self.cyclization_bond_loss_weight * bond_loss

        if getattr(self, "surface_loss_enabled", False):
            surf_loss = self.compute_surface_attraction_loss(batch, nn_out, log_prefix=log_prefix, bs=bs)
            train_loss = train_loss + self.surface_loss_weight * surf_loss

        # Joint AE training: anchor the (now trainable) autoencoder with its own recon+KL
        # objective so the encoder cannot collapse the flow target `z` into something trivial.
        # The flow loss above already backprops into the encoder via x_1["local_latents"].
        if (not getattr(self, "freeze_autoencoder", True)) and getattr(self, "autoencoder", None) is not None:
            ae_reg, ae_losses = self.autoencoder.compute_ae_reg_losses(batch)
            train_loss = train_loss + self.ae_reg_weight * ae_reg
            for k, v in ae_losses.items():
                self.log(
                    f"{log_prefix}/ae_{k}",
                    torch.mean(v),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:  # Don't log these for val step
            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_flops(bs, n)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()

        if val_step:
            self.log_unified_validation_metrics(
                batch=batch,
                nn_out=nn_out,
                losses=losses,
                cyclization_metrics=cyclization_metrics,
                cyclization_loss_value=cyclization_loss_value,
                bs=bs,
            )

        return train_loss

    def log_losses(
        self,
        bs: int,
        losses: dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        batch: dict,
    ):
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k

            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

            if k in self.fm.data_modes and self.cfg_exp.get("log_t_binned_losses", False):
                # Prefer pre-1/(1-t)^2 x1 MSE so mid-t bins are readable without undoing
                # the time weight by hand. Falls back to the scaled loss if unscaled is absent.
                unscaled_key = f"{k}_unscaled_justlog"
                self.log_t_binned_loss(
                    bs=bs,
                    data_mode=k,
                    per_sample_loss=losses.get(unscaled_key, losses[k]),
                    batch=batch,
                    log_prefix=log_prefix,
                )

            if self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0) > 0.0:
                # Log also for folding and inverse folding iters
                # divides by p_aux to account for the fact that for most steps loss will be just zero
                # (since need to sync across devices, etc, need to handle logging carefully)
                p_aux = self.cfg_exp.training["p_folding_n_inv_folding_iters"] / 2
                loss = torch.mean(losses[k])  # [b]

                f_inv_fold = batch["use_ca_coors_nm_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_invfold_ca_iter/loss_{log_name}",
                    loss * f_inv_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                f_fold = batch["use_residue_type_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_fold_iter/loss_{log_name}",
                    loss * f_fold,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

    def compute_surface_attraction_loss(
        self,
        batch: dict,
        nn_out: dict,
        log_prefix: str,
        bs: int,
    ) -> torch.Tensor:
        """Symmetric soft-Chamfer between a predicted surface-facing proxy and surface points (nm).

        Cα sits on the backbone, inside the molecule, not on its molecular surface -- pulling
        it directly onto ``surface_xyz`` would drag the whole backbone (including the
        binder's exposed-away residues) onto the interface. Cβ points from the backbone
        toward the side chain/solvent and is a much closer proxy for "the atom facing the
        surface." Since the predicted structure's real Cβ requires decoding through the
        (frozen) autoencoder, this decodes ``x_1_pred`` to full Atom37 and reconstructs a
        virtual Cβ from the decoded N/CA/C backbone frame (same formula used for glycine
        elsewhere in this codebase, see ``eval.cyclic_reconstruction_metrics.get_cb_position``)
        rather than comparing Cα.

        Applied only for samples with ``t_bb_ca >= t_lower_lim`` (prediction is noise below that).
        Gives the zero-init gate a nonzero dL/dg even when the FM loss alone leaves g shut.
        """
        if "surface_xyz" not in batch or "surface_mask" not in batch or getattr(self, "autoencoder", None) is None:
            z = torch.zeros((), device=self.device, dtype=torch.float32)
            self.log(
                f"{log_prefix}/surface_attraction",
                z,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            return z

        from proteinfoundation.eval.cyclic_reconstruction_metrics import (
            CA_IDX,
            C_IDX,
            N_IDX,
            virtual_cb_from_backbone,
        )

        x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
        binder_mask = batch["mask"].bool() if "mask" in batch else batch["coord_mask"][..., 1].bool()
        decoded = self.autoencoder.decode(
            z_latent=x_1_pred["local_latents"],
            ca_coors_nm=x_1_pred["bb_ca"],
            mask=binder_mask,
        )
        decoded_coors = decoded["coors_nm"]
        decoded_mask = decoded["atom_mask"].bool()
        backbone_ok = decoded_mask[..., N_IDX] & decoded_mask[..., CA_IDX] & decoded_mask[..., C_IDX]
        cb = virtual_cb_from_backbone(
            decoded_coors[..., N_IDX, :], decoded_coors[..., CA_IDX, :], decoded_coors[..., C_IDX, :]
        )  # [B, N, 3] nm, surface-facing proxy

        surf = batch["surface_xyz"]
        smask = batch["surface_mask"].bool()

        cb_mask = binder_mask & backbone_ok
        dist = (cb[:, :, None, :] - surf[:, None, :, :]).norm(dim=-1)  # [B, N, M]
        dist = dist.masked_fill(~smask[:, None, :], 1e6)
        dist = dist.masked_fill(~cb_mask[:, :, None], 1e6)
        d_bs = dist.min(dim=-1).values  # [B, N]
        d_sb = dist.min(dim=1).values  # [B, M]

        per_b = (d_bs * cb_mask).sum(-1) / cb_mask.sum(-1).clamp_min(1)
        per_s = (d_sb * smask).sum(-1) / smask.sum(-1).clamp_min(1)
        per = 0.5 * (per_b + per_s)

        t = batch["t"]["bb_ca"]
        if t.dim() > 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        active = (t >= self.surface_loss_t_lower).to(dtype=per.dtype)
        # If no sample is active, return 0 (no spurious grad) but still log the raw Chamfer.
        denom = active.sum().clamp_min(1.0)
        loss = (per * active).sum() / denom

        self.log(
            f"{log_prefix}/surface_attraction",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/surface_attraction_raw",
            per.mean().detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return loss

    def log_surface_gates(self, bs: int, log_prefix: str) -> None:
        """Log zero-init surface cross-attn gates (`g` in ``h ← h + g·Δ``).

        If these stay ~0, surface conditioning is inert and all arms look like baseline.
        """
        nn = getattr(self, "nn", None)
        layers = getattr(nn, "surface_cross_layers", None) if nn is not None else None
        if not layers:
            return
        abs_vals = []
        for name, layer in layers.items():
            g = float(layer.gate.detach().float().item())
            abs_vals.append(abs(g))
            self.log(
                f"{log_prefix}/surface_gate/layer_{name}",
                g,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )
        self.log(
            f"{log_prefix}/surface_gate/abs_mean",
            float(sum(abs_vals) / len(abs_vals)),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nan_safe(self, key: str, value, bs: int, on_step: bool) -> None:
        """`self.log` for metrics whose per-step value is legitimately NaN sometimes.

        Several diagnostics here are NaN by design on some steps: a t-bin with no samples in it, a
        cyclization chemistry absent from the batch, an interface metric with no target. NaN (not 0)
        is the honest value -- 0 would silently drag the average toward "perfect".

        But NaN cannot be aggregated with `self.log`:
          * `reduce_fx=torch.nanmean` raises MisconfigurationException -- Lightning only accepts
            `reduce_fx` in {min, max, mean, sum} (logger_connector/result.py:_parse_reduce_fx).
          * plain `mean` is worse than useless: a single NaN step poisons the whole epoch average.
            With 5 t-bins at batch 6, some bin is empty in ~26% of steps, so essentially EVERY epoch
            value would come out NaN.

        Lightning's own error message gives the fix ("log a `torchmetrics.Metric` instance
        instead"). `MeanMetric(nan_strategy="ignore")` *is* nanmean, and it performs its own
        distributed sync -- hence `sync_dist=False`, or the value would be reduced twice.

        Metrics are created lazily. That is safe under DDP: they carry buffers, not parameters, and
        the key set is data-independent (fixed t-bins, fixed metric schema), so every rank ends up
        creating the same ones in the same order. `metric_attribute` must be passed explicitly --
        Lightning otherwise scans `named_modules()` to find the Metric it is being handed, and a
        metric added to the ModuleDict *during* the step is not in the snapshot it searches, so it
        raises "Could not find the LightningModule attribute for the torchmetrics.Metric logged".
        """
        safe = key.replace("/", "__").replace(".", "_")
        if safe not in self._nan_mean_metrics:
            self._nan_mean_metrics[safe] = MeanMetric(nan_strategy="ignore").to(self.device)
        metric = self._nan_mean_metrics[safe]
        metric(torch.as_tensor(value, dtype=torch.float32, device=self.device))
        self.log(
            key,
            metric,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=False,  # MeanMetric syncs itself; sync_dist=True would double-reduce.
            add_dataloader_idx=False,
            metric_attribute=f"_nan_mean_metrics.{safe}",
        )

    def log_t_binned_loss(
        self,
        bs: int,
        data_mode: str,
        per_sample_loss: Float[torch.Tensor, "b"],
        batch: dict,
        log_prefix: str,
    ):
        """Logs `per_sample_loss` for `data_mode` split into time bins (opt-in diagnostic).

        Expects pre-scale (unweighted) x1 MSE when available via `*_unscaled_justlog`, so bin
        curves are comparable across t without the `1/(1-t)^2` reweight. Default off
        (`log_t_binned_losses=false`), no effect on training.

        Empty bins log NaN (not 0) so WandB step curves and epoch `nanmean` aggregation are not
        dragged toward zero by missing bins. Still calls `self.log` every step for every bin key
        so DDP always sees a fixed key set (skipping keys would risk a cross-rank hang).
        """
        t_bins = list(self.cfg_exp.get("t_bins", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        t_vals = batch["t"][data_mode].detach()
        loss_vals = per_sample_loss.detach()
        for lo, hi in zip(t_bins[:-1], t_bins[1:]):
            is_last_bin = hi >= t_bins[-1]
            bin_mask = (t_vals >= lo) & (t_vals <= hi if is_last_bin else t_vals < hi)
            count = bin_mask.sum()
            if count > 0:
                bin_loss = (loss_vals * bin_mask).sum() / count
            else:
                bin_loss = torch.tensor(float("nan"), device=loss_vals.device, dtype=loss_vals.dtype)
            self.log_nan_safe(
                f"{log_prefix}/loss_{data_mode}_t_{lo}_{hi}", bin_loss, bs=bs, on_step=True
            )

    @torch.no_grad()
    def log_t_binned_val_metrics(
        self,
        batch: dict,
        nn_out: dict,
        bs: int,
        template_keys: list[str],
    ) -> None:
        """Re-runs the AE-recon / four-way-decode diagnostics restricted to each `t_bins` slice
        of `batch["t"]["bb_ca"]`, so e.g. isopeptide-bond success rate for `decode_predca_predz`
        can be read per noise level instead of averaged across the whole batch (which, under a
        skewed t-distribution like `bb_ca`'s `mix_unif_beta`, mixes very-different-difficulty
        samples into one number). Opt-in via `log_t_binned_val_metrics=true` (default off).

        Logs under `val_t_{lo}_{hi}/...` (parallel to `val/...`).

        Same empty-bin convention as `log_t_binned_loss` (NaN when a bin has no samples in
        the step, so averages are not dragged toward 0), and for the same reason (DDP's
        `sync_dist` needs the same set of logged keys every step across ranks). A bin that
        rarely gets samples (e.g. bb_ca's `[0, 0.2)` under a beta skewed toward t=1) will show
        gaps / NaNs rather than false zeros — check population before reading "low loss".

        `template_keys` fixes the logged schema: pass the real `val/ae_recon*` /
        `val/decode_*` / `val/flow_clean/*` keys produced for the full batch (computed by the
        caller before this runs), so every bin logs exactly that key set every step.
        """
        if not template_keys:
            return

        t_bins = list(self.cfg_exp.get("t_bins", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        t_vals = batch["t"]["bb_ca"].detach()
        b = t_vals.shape[0]

        sample_posterior = bool(self.cfg_exp.get("eval", {}).get("reconstruction", {}).get("sample_posterior", False))

        for lo, hi in zip(t_bins[:-1], t_bins[1:]):
            is_last_bin = hi >= t_bins[-1]
            bin_mask = (t_vals >= lo) & (t_vals <= hi if is_last_bin else t_vals < hi)
            idx = bin_mask.nonzero(as_tuple=True)[0]
            bin_prefix = f"val_t_{lo}_{hi}"

            if idx.numel() == 0:
                bin_metrics = {bin_prefix + k[len("val") :]: float("nan") for k in template_keys}
            else:
                sub_batch = _index_pytree(batch, idx, b)
                sub_nn_out = _index_pytree(nn_out, idx, b)

                raw_metrics = {}
                if getattr(self, "reconstruction_eval_enabled", False):
                    raw_metrics.update(
                        run_ae_reconstruction_eval(
                            self.autoencoder,
                            sub_batch,
                            prefix=f"{bin_prefix}/ae_recon",
                            sample_posterior=sample_posterior,
                        )
                    )
                if getattr(self, "four_way_decode_eval_enabled", False):
                    # run_four_way_decode_eval hardcodes its own "val/..." prefixes; rekey them
                    # under bin_prefix instead (it doesn't take a prefix argument).
                    for k, v in run_four_way_decode_eval(self.autoencoder, self.fm, sub_batch, sub_nn_out).items():
                        raw_metrics[bin_prefix + k[len("val") :]] = v

                # `template_keys` are namespaced "val/...". raw_metrics is namespaced under
                # `bin_prefix` already (see rekeying above / the explicit prefix passed to
                # run_ae_reconstruction_eval). Backfill anything missing (shouldn't normally
                # happen) and drop unexpected keys so every step logs exactly `template_keys`,
                # renamed under this bin's prefix.
                bin_metrics = {
                    bin_prefix + k[len("val") :]: float(raw_metrics.get(bin_prefix + k[len("val") :], float("nan")))
                    for k in template_keys
                }

            for k, v in bin_metrics.items():
                self.log_nan_safe(k, v, bs=bs, on_step=True)

    def log_latent_diagnostics(self, batch: dict, log_prefix: str):
        """Opt-in diagnostics for the local_latents AE target (`log_latent_diagnostics=true`).

        Must be called after `add_clean_samples` (needs `batch["local_latents_encoder_output"]`,
        `batch["local_latents_raw_target"]`, and `batch["x_1"]["local_latents"]`) and before
        `corrupt_batch` overwrites `batch["mask"]` (harmless either way here since corrupt_batch
        sets mask to the same value, but the encoder output / raw target keys are only present
        pre-corruption). No-op if local_latents is not a configured data mode.

        Logs (all masked over valid residues):
          - latent target mean/std/min/max (what the flow actually regresses to; if latent
            normalization is enabled this is the *normalized* target, and the raw
            pre-normalization target is also logged as `latent_target_raw_*`)
          - encoder posterior mean std, posterior scale (exp(log_scale)) mean/std
          - KL before weighting (raw, unweighted) and after weighting (using the AE's own
            configured KL weight, i.e. `autoencoder.cfg_ae.loss.kl.weight`)
        """
        if "local_latents" not in getattr(self.fm, "data_modes", []):
            return
        enc_out = batch.get("local_latents_encoder_output")
        if enc_out is None:
            return

        # Same mask resolution as `ProductSpaceFlowMatcher.process_batch`: this runs before
        # `corrupt_batch` sets `batch["mask"]`, and some datamodules (e.g. the original
        # PDB-monomer/genie2 pipeline) only populate `mask_dict` at this point.
        if "mask_dict" in batch:
            mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
        else:
            mask = batch["mask"].bool()
        bs = mask.shape[0]

        def _log_stats(v: torch.Tensor, name: str):
            vals = v[mask]
            if vals.numel() == 0:
                return
            for stat_name, stat_val in (
                ("mean", vals.mean()),
                ("std", vals.std()),
                ("min", vals.min()),
                ("max", vals.max()),
            ):
                self.log(
                    f"{log_prefix}/latent_diag_{name}_{stat_name}",
                    stat_val,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

        _log_stats(batch["x_1"]["local_latents"], "target")
        if getattr(self.autoencoder, "latent_norm_stats", None) is not None:
            _log_stats(batch["local_latents_raw_target"], "target_raw")

        _log_stats(enc_out["mean"], "posterior_mean")
        _log_stats(torch.exp(enc_out["log_scale"]), "posterior_scale")

        kl_raw = self.autoencoder.compute_kl_penalty(
            mean=enc_out["mean"], log_scale=enc_out["log_scale"], mask=mask, w=1.0
        )["kl_now_justlog"]
        kl_weight = float(self.autoencoder.cfg_ae.loss.kl.weight)
        for name, val in (
            ("kl_before_weighting", torch.mean(kl_raw)),
            ("kl_after_weighting", torch.mean(kl_raw) * kl_weight),
        ):
            self.log(
                f"{log_prefix}/latent_diag_{name}",
                val,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def compute_cyclization_loss(
        self,
        batch: dict,
        nn_out: dict,
        log_prefix: str,
        bs: int,
    ) -> tuple[torch.Tensor, dict]:
        """Computes the CPSea cyclization-linkage CE loss for one training batch.

        Safe no-op (returns zero loss) if the batch lacks cyclization fields
        (e.g. a non-CPSea dataset), so mixed training never breaks. Always
        uses the *ground-truth* AA sequence to build the validity mask; at
        inference time `predict_cyclization` uses the predicted AA sequence.

        When `cyclization.type_conditioning` is on, the requested type narrows the
        validity mask to that type's slice, which turns the existing global softmax
        into exactly `p(i, j | type)` -- restricting a softmax's support and
        renormalizing *is* conditioning, so no change to the head or the loss itself
        is needed. Rows dropped to UNSPECIFIED keep the full candidate set and so
        still train the unconditional joint `p(i, j, type)`.
        """
        from proteinfoundation.cyclization.constants import NUM_CYCLIZATION_TYPES
        from proteinfoundation.cyclization.loss import cyclization_link_loss
        from proteinfoundation.cyclization.mask import build_cyclization_validity_mask

        device = batch["mask"].device
        required_keys = ("has_cyclization", "cyclization_i", "cyclization_j", "cyclization_type", "residue_type")
        if any(k not in batch for k in required_keys):
            return torch.zeros((), device=device), {}

        binder_mask = batch["mask"]
        has_cyclization = batch["has_cyclization"].to(device=device)
        gold_i = batch["cyclization_i"].to(device=device).long()
        gold_j = batch["cyclization_j"].to(device=device).long()
        gold_type = batch["cyclization_type"].to(device=device).long()

        x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
        ca_for_link = x_1_pred["bb_ca"]
        z1_pred = x_1_pred["local_latents"]

        detach = self.cyclization_detach_link_inputs
        decode_ctx = torch.no_grad() if detach else torch.enable_grad()
        with decode_ctx:
            decoded = self.autoencoder.decode(z_latent=z1_pred, ca_coors_nm=ca_for_link, mask=binder_mask)
        aa_probs = decoded["seq_logits"].softmax(dim=-1)

        single_for_link = torch.cat([z1_pred, aa_probs], dim=-1)
        if detach:
            single_for_link = single_for_link.detach()
            ca_for_link = ca_for_link.detach()

        link_logits = self.cyclization_head(single_for_link, ca_for_link)

        cond_type = None
        if self.cyclization_type_conditioning and "cyclization_type_cond" in batch:
            cond_type = batch["cyclization_type_cond"].to(device=device).long()

        gt_aa = batch["residue_type"].to(device=device).long()
        valid_mask = build_cyclization_validity_mask(
            aa=gt_aa,
            binder_mask=binder_mask,
            gold_i=gold_i.clamp(min=0),
            gold_j=gold_j.clamp(min=0),
            gold_type=gold_type.clamp(min=0, max=NUM_CYCLIZATION_TYPES - 1),
            force_gold_valid=self.cyclization_force_gold_valid,
            allow_asn_gln_isopeptide=self.cyclization_allow_asn_gln_isopeptide,
            cond_type=cond_type,
            terminal_only=self.cyclization_terminal_only,
        )

        loss, metrics = cyclization_link_loss(
            link_logits=link_logits,
            valid_mask=valid_mask,
            gold_i=gold_i,
            gold_j=gold_j,
            gold_type=gold_type,
            has_cyclization=has_cyclization,
        )

        self.log(
            f"{log_prefix}/loss_cyclization",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        for name, value in metrics.items():
            self.log(
                f"{log_prefix}/cyclization_{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        return loss, metrics

    def _bond_loss_t_weight(self, batch: dict) -> torch.Tensor:
        """[b] weight in [0, 1] that switches the bond loss off at low flow time.

        The bond distance is read off the *predicted clean structure*, which is
        meaningless at high noise -- penalising it there would be supervising the
        model's own noise. Weight ramps linearly from `t_lower_lim` to 1.

        Takes the MIN over modalities rather than `bb_ca` alone: an anchor atom needs
        both a clean Ca trace and clean local latents to be placed, so a sample that
        is clean in one modality and noisy in the other still has no usable bond.
        """
        ts = [batch["t"][dm] for dm in self.fm.data_modes if dm in batch["t"]]
        t = ts[0] if len(ts) == 1 else torch.stack(ts, dim=0).min(dim=0).values  # [b]
        lo = self.cyclization_bond_loss_t_lower
        return torch.clamp((t - lo) / max(1.0 - lo, 1e-6), min=0.0, max=1.0)

    def compute_cyclization_bond_loss(
        self,
        batch: dict,
        nn_out: dict,
        log_prefix: str,
        bs: int,
    ) -> tuple[torch.Tensor, dict]:
        """Computes the two-sided closing-bond distance loss for one training batch.

        Safe no-op (zero loss) if the batch lacks cyclization fields, so mixed
        training never breaks.

        Unlike the linkage head's decode, this one runs **with gradients**: the whole
        point is for the penalty to reach the flow model's predicted latents and Ca
        trace. The autoencoder being frozen does not prevent that -- frozen params
        still pass gradient through to their inputs.

        Chemistry (`seq_tokens`) and atom presence come from the *decoded prediction*,
        not the ground truth, so they agree with the coordinates they gate: an SG-SG
        distance is only supervised where the model actually put two cysteines. The
        endpoints and type, by contrast, are the requested/ground-truth cyclization.
        """
        from proteinfoundation.cyclization.bond_loss import cyclization_bond_loss
        from proteinfoundation.eval.cyclic_reconstruction_metrics import extract_cyclization_metadata

        device = batch["mask"].device
        cyclization_metadata = extract_cyclization_metadata(batch)
        if cyclization_metadata is None:
            return torch.zeros((), device=device), {}

        binder_mask = batch["mask"]
        x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
        decoded = self.autoencoder.decode(
            z_latent=x_1_pred["local_latents"],
            ca_coors_nm=x_1_pred["bb_ca"],
            mask=binder_mask,
        )

        decoded_mask = decoded["atom_mask"].bool() & binder_mask.bool()[..., None]
        decoded_seq = decoded["residue_type"].long()
        t_weight = self._bond_loss_t_weight(batch)

        loss, metrics = cyclization_bond_loss(
            pred_atom37=decoded["coors_nm"],
            atom37_mask=decoded_mask,
            seq_tokens=decoded_seq,
            cyclization_metadata=cyclization_metadata,
            t_weight=t_weight,
        )

        if getattr(self, "cyclization_geometry_enabled", False):
            from proteinfoundation.cyclization.linkage_geometry import linkage_geometry_loss

            geom_loss, geom_metrics = linkage_geometry_loss(
                pred_atom37=decoded["coors_nm"],
                atom37_mask=decoded_mask,
                seq_tokens=decoded_seq,
                cyclization_metadata=cyclization_metadata,
                w_angle=self.cyclization_geometry_w_angle,
                w_dihedral=self.cyclization_geometry_w_dihedral,
                t_weight=t_weight,
            )
            loss = loss + self.cyclization_geometry_weight * geom_loss
            metrics.update({f"geom_{k}": v for k, v in geom_metrics.items()})

        self.log(
            f"{log_prefix}/loss_cyclization_bond",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        for name, value in metrics.items():
            # NaN-by-design when the batch holds no supervisable bond. Log the key on
            # every rank regardless (DDP reduces over a shared key set) but with zero
            # weight, so one inapplicable batch cannot poison the epoch mean.
            v = float(value)
            is_nan = math.isnan(v)
            self.log(
                f"{log_prefix}/cyclization_bond_{name}",
                0.0 if is_nan else v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=0 if is_nan else bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        return loss, metrics

    @torch.no_grad()
    def log_unified_validation_metrics(
        self,
        batch: dict,
        nn_out: dict,
        losses: dict[str, torch.Tensor],
        cyclization_metrics: dict,
        cyclization_loss_value: float | None,
        bs: int,
    ) -> None:
        """Logs the uniform `val/...` metric schema (see `proteinfoundation.logging.metric_schema`).

        Purely additive: every metric already logged elsewhere in `training_step` (the
        `validation_loss/...` keys) is untouched. This method only adds new keys under the
        `val/...` namespace so that every run -- regardless of AE source, frozen/joint AE,
        cyclization head on/off, or whether the reconstruction/four-way-decode diagnostics are
        enabled -- logs the exact same set of validation metric keys (unavailable ones as NaN,
        per `finalize_metrics`), which is what makes cross-run W&B comparisons meaningful.

        No-op if no autoencoder is configured (non-`local_latents` pipelines), since none of the
        AE/cyclic-geometry diagnostics apply there and their absence there is not part of the
        comparison this schema is meant to support.
        """
        if getattr(self, "autoencoder", None) is None:
            return

        metrics = init_metric_dict()

        if "local_latents" in losses:
            metrics["val/loss/latent"] = float(torch.mean(losses["local_latents"]).item())
        if "bb_ca" in losses:
            metrics["val/loss/bb_ca"] = float(torch.mean(losses["bb_ca"]).item())
        if cyclization_metrics:
            metrics["val/cyclization/acc"] = float(cyclization_metrics.get("top1_exact_acc", float("nan")))
        if cyclization_loss_value is not None:
            metrics["val/cyclization/ce"] = cyclization_loss_value

        if getattr(self, "reconstruction_eval_enabled", False):
            sample_posterior = bool(
                self.cfg_exp.get("eval", {}).get("reconstruction", {}).get("sample_posterior", False)
            )
            metrics.update(
                run_ae_reconstruction_eval(
                    self.autoencoder, batch, prefix="val/ae_recon", sample_posterior=sample_posterior
                )
            )

        if getattr(self, "four_way_decode_eval_enabled", False):
            metrics.update(run_four_way_decode_eval(self.autoencoder, self.fm, batch, nn_out))

        if self.cfg_exp.get("log_t_binned_val_metrics", False):
            binned_template_keys = [
                k
                for k in metrics
                if k.startswith("val/ae_recon") or k.startswith("val/decode_") or k.startswith("val/flow_clean/")
            ]
            self.log_t_binned_val_metrics(
                batch=batch,
                nn_out=nn_out,
                bs=bs,
                template_keys=binned_template_keys,
            )

        metrics = finalize_metrics(metrics)

        for k, v in metrics.items():
            v = float(v)
            # A metric is NaN when this batch held no example of its kind -- most batches
            # contain no disulfide-cyclized binder (~10% of CPSea), and cyclic metrics are
            # NaN-by-design when inapplicable. Lightning accumulates `value * batch_size`
            # into the epoch mean, so a single NaN makes the whole epoch NaN: that is why
            # disulfide_bond_success and mainchain_cn_bond_success have always read NaN at
            # epoch level while their per-step values were fine.
            #
            # Give the inapplicable batch zero weight rather than dropping the key. The key
            # must still be logged on every rank and every step -- under sync_dist/DDP the
            # ranks reduce over a shared key set, and a rank that skips a key desynchronises
            # the collective. Weight, not membership, is what varies.
            weight = _cyclic_metric_weight(k, metrics, default=bs)
            is_nan = math.isnan(v)
            self.log(
                k,
                0.0 if is_nan else v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=0 if is_nan else weight,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        if not self._ae_source_logged:
            self._ae_source_logged = True
            try:
                self.logger.experiment.config.update(
                    {
                        "ae_source": self.cfg_exp.get("ae_source", "unknown"),
                        "ae_frozen": bool(getattr(self, "freeze_autoencoder", True)),
                        "has_cyclization_head": bool(getattr(self, "cyclization_enabled", False)),
                        "reconstruction_eval_enabled": bool(getattr(self, "reconstruction_eval_enabled", False)),
                        "four_way_decode_eval_enabled": bool(getattr(self, "four_way_decode_eval_enabled", False)),
                    },
                    allow_val_change=True,
                )
            except Exception:
                # Best-effort only (e.g. no WandbLogger attached, smoke tests with log_wandb=False).
                pass

    def apply_cyclization_type_conditioning(self, batch: dict, bs: int) -> dict:
        """Stamps the requested cyclization type onto a generation batch, in place.

        No-op unless `cyclization.type_conditioning` is on, so the unconditional
        generation path is untouched. A batch that already carries a per-sample
        `cyclization_type_cond` (e.g. a CPSea val batch) keeps it -- the config's
        `target_type` only supplies a value where the data has none.
        """
        if not getattr(self, "cyclization_type_conditioning", False):
            return batch
        if "cyclization_type_cond" in batch:
            return batch

        from proteinfoundation.cyclization.constants import UNSPECIFIED

        type_idx = self.cyclization_target_type
        if type_idx is None:
            type_idx = UNSPECIFIED
        batch["cyclization_type_cond"] = torch.full(
            (bs,), int(type_idx), dtype=torch.long, device=batch["mask"].device
        )
        return batch

    @torch.no_grad()
    def predict_cyclization(self, batch: dict) -> dict:
        """Runs a single generation pass and predicts the cyclization edge for each sample.

        Uses the *predicted* AA sequence (never ground truth) to build the validity
        mask. Standalone utility usable for CPSea binder-cyclization inspection;
        see `single_pass_generation.py` for the wired-in `predict_step` path.

        The requested type (if any) comes from `cyclization.target_type`.

        Returns:
            Dict with `pred_cyclization_i/j/type/confidence` tensors, each [B], plus
            `cyclization_type_satisfied` ([B] bool) when type conditioning is on.
        """
        if self.cyclization_head is None:
            raise RuntimeError("Cyclization head is not enabled (set cyclization.enabled=true)")

        from proteinfoundation.cyclization.inference import predict_cyclization_from_clean

        # `generate` stamps `cyclization_type_cond` into the batch, so read it back from
        # there rather than re-deriving it -- the head must see exactly what the denoiser saw.
        gen_samples = self.generate(batch)
        cond_type = batch.get("cyclization_type_cond") if self.cyclization_type_conditioning else None
        return predict_cyclization_from_clean(
            self,
            ca=gen_samples["bb_ca"],
            z_latent=gen_samples["local_latents"],
            mask=batch["mask"],
            cond_type=cond_type,
        )

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor):
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self):
        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line but ok, easy to compare # params

    def update_n_log_nsamples_processed(self, b: int):
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_flops(self, b: int, n: int):
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(b, n)  # nn should implement this function if we want to see nflops
        except Exception:
            nflops_step = None

        if nflops_step is not None:
            self.nflops = (
                self.nflops + nflops_step * self.trainer.world_size
            )  # Times number of processes so it logs sum across devices
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        """Validation step dispatching to length-based generation or data-based loss.

        Args:
            batch: batch from dataset
            batch_idx: batch index (unused)
            dataloader_idx: With dual val loaders: 0 = GenDataset (generation), 1 = val loss.
                With a single val loader (CPSea / StructureDataModule), only val loss runs.
        """
        if self._val_has_lens_dataloader and dataloader_idx == 0:
            self.validation_step_lens(batch, batch_idx)
        elif self._val_has_lens_dataloader and dataloader_idx != 1:
            raise OSError(f"Validation dataloader with index {dataloader_idx} not recognized")
        else:
            self.validation_step_data(batch, batch_idx)

    def validation_step_data(self, batch: dict, batch_idx: int):
        """Evaluates the training loss on validation data."""
        # Sampling validation runs FIRST, on the pristine batch: `training_step` -> `corrupt_batch`
        # writes x_1/x_0/x_t/t/mask into `batch` in place, and we want the generator to see the
        # batch exactly as the design pipeline would.
        if self.val_gen_enabled:
            self.validation_step_generate(batch, batch_idx)
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(loss.item())

    @torch.no_grad()
    def validation_step_generate(self, batch: dict, batch_idx: int) -> None:
        """Integrates the ODE from t=0 on a real val complex and scores what comes out.

        This is the only validation signal here that is NOT teacher-forced. Every other logged
        metric (flow loss, t-binned loss, `run_four_way_decode_eval`) builds `x_t` by interpolating
        the *true* `x_1` with noise, so the model is always handed a real interpolant and most of
        the answer with it; `decode_gtca_*` goes further and hands the decoder the native Ca trace.
        Sampling never sees a true interpolant -- it consumes its own accumulated output. A model
        can therefore sit at a healthy loss, post 100% teacher-forced sequence recovery, and still
        emit garbage when integrated, and *no* teacher-forced metric can reveal that. Hence this.

        Cheap on purpose -- single-pass: one ODE integration, no search, no lookahead, no AF2, no
        MPNN, no folding. Scores reference-free geometry (see `eval/sampled_binder_metrics.py`)
        plus cyclization bond closure on the sampled structure.

        This is only worth its GPU time if it samples the way the DESIGN pipeline samples --
        otherwise it predicts the behaviour of a sampler nobody runs. Point
        `val_generation.design_sampling` at `configs/pipeline/model_sampling.yaml` (via a
        Hydra default, so the two cannot drift) and this uses design's exact schedules,
        `sampling_mode`, noise scales and centering.

        The fallback (`generation.model["ode"]`) is NOT equivalent and is warned about: the
        `generation` group is Proteina's *unconditional monomer* validation config, whose
        `ode` block sets `bb_ca.center_every_step=True`. Forcing the sample's CoM to zero is
        right for a monomer centred on itself and **wrong for a binder**: CPSea centres on the
        TARGET (`CenteringTransform(center_mode="target")`, `zero_com_noise=False`), so the
        binder legitimately sits off-origin at its binding site, and zero-CoM-ing it every
        step drags it to the receptor's centre -- off the training manifold -- at every step.

        Config (`val_generation` block, all optional):
            enabled     bool, default False.
            n_batches   how many val batches to sample (default 2). Each costs an ODE integration.
            n_repeat    samples per complex (default 4). >1 enables the `div/*` mode-collapse
                        readout -- the spread across repeats of the SAME target.
            design_sampling  the design pipeline's sampling config (`args` + `model`). Strongly
                        preferred; see above.
            nsteps      ODE steps. Default: the design sampler's. Lower only to make a smoke
                        run cheap -- it is itself a design-alignment knob.
            self_cond   default: the design sampler's.
        """
        vg = self.val_gen_cfg
        if batch_idx >= int(vg.get("n_batches", 2)):
            return
        if "local_latents" not in self.cfg_exp.product_flowmatcher or self.autoencoder is None:
            return

        design = vg.get("design_sampling", None)
        if design is not None:
            sampler_args, sampling_model_args = design.args, design.model
            n_recycle = int(design.get("n_recycle", 0))
        else:
            sampler_args, sampling_model_args = self.cfg_exp.generation.args, self.cfg_exp.generation.model["ode"]
            n_recycle = 0
            if not self._val_gen_sampler_warned:
                self._val_gen_sampler_warned = True
                logger.warning(
                    "val_generation has no `design_sampling` block, falling back to "
                    "generation.model['ode'] -- the UNCONDITIONAL MONOMER sampler. For a binder "
                    "model this is not merely untuned, it is wrong: bb_ca.center_every_step=True "
                    "zero-CoMs the binder every ODE step, but CPSea centres on the TARGET so the "
                    "binder belongs off-origin. Expect blown-up geometry that says nothing about "
                    "the model. Add `- /pipeline/model_sampling@val_generation.design_sampling` "
                    "to the config's defaults list."
                )

        n_repeat = max(1, int(vg.get("n_repeat", 4)))
        nsteps = vg.get("nsteps", None)
        nsteps = int(sampler_args.nsteps if nsteps is None else nsteps)
        self_cond = vg.get("self_cond", None)
        self_cond = bool(sampler_args.self_cond if self_cond is None else self_cond)

        # CPSea batches carry no `mask` before `corrupt_batch`; `full_simulation` would then fall
        # back to `torch.ones(...)` (product_space_flow_matcher.py:785) and denoise PADDING as if it
        # were real residues. Resolve it explicitly, the same way `AutoEncoder.encode` does.
        mask = batch["mask"].bool() if "mask" in batch else batch["coord_mask"][..., 1].bool()

        # repeat_interleave (not repeat): `sampled_binder_metrics` div/* assumes repeats of one
        # complex are CONTIGUOUS, and will otherwise silently compare different complexes.
        gen_batch = {
            k: (v.repeat_interleave(n_repeat, dim=0) if torch.is_tensor(v) and v.dim() >= 1 else v)
            for k, v in batch.items()
            # x_1/x_0/x_t/t are stale interpolation state if this batch was already corrupted;
            # full_simulation sets its own. Drop them so nothing downstream reads the wrong x_t.
            if k not in ("x_1", "x_0", "x_t", "t", "x_sc")
        }
        gen_mask = mask.repeat_interleave(n_repeat, dim=0)
        gen_batch["mask"] = gen_mask
        bs, n = gen_mask.shape

        self.apply_cyclization_type_conditioning(gen_batch, bs=bs)

        gen_samples = self.fm.full_simulation(
            batch=gen_batch,
            predict_for_sampling=partial(self.predict_for_sampling, n_recycle=n_recycle),
            nsteps=nsteps,
            nsamples=bs,
            n=n,
            self_cond=self_cond,
            sampling_model_args=sampling_model_args,
            device=self.device,
            guidance_w=float(sampler_args.get("guidance_w", 1.0)),
            # Autoguidance stays off regardless of what design requests: it needs a second
            # ("bad model") checkpoint that validation has no way to load.
            ag_ratio=0.0,
        )

        sample_prots = sample_formatting(
            x=gen_samples,
            extra_info={"mask": gen_mask},
            ret_mode="coors37_n_aatype",
            data_modes=list(self.cfg_exp.product_flowmatcher),
            autoencoder=self.autoencoder,
        )
        # UNITS: `sample_formatting(ret_mode="coors37_n_aatype")` returns ANGSTROM -- it applies
        # `nm_to_ang` internally (sample_utils.format_sample_local_latents) because its other
        # consumers write PDBs, which are in Angstrom. Everything below wants NANOMETERS:
        # `sampled_binder_metrics` documents nm and compares against CA_CA_NM=0.3835 (and is handed
        # `coords_nm`/`x_target`, which really are nm), and `cyclic_geometry_metrics` multiplies its
        # input by NM_TO_ANG itself. Feeding Angstrom through pinned geom/ca_ca_viol_frac at 1.0,
        # inflated every cyc/*_dist_A tenfold, and silently compared Angstrom against nm in the
        # place/* and iface/* groups. Convert exactly once, here, at the boundary.
        aatype = sample_prots["residue_type"]
        coors = sample_prots["coors"] / NM_TO_ANG  # [b, n, 37, 3], nm

        gt_coors = gen_batch.get("coords_nm")
        metrics = sampled_binder_metrics(
            coors=coors,
            aatype=aatype,
            mask=gen_mask,
            gt_coors=gt_coors,
            x_target=gen_batch.get("x_target"),
            target_mask=gen_batch.get("target_mask"),
            target_hotspot_mask=gen_batch.get("target_hotspot_mask"),
            n_repeat=n_repeat,
            prefix="val_gen",
        )

        # Surface agreement on the ODE sample (oracle / shuffle arms). Teacher-forced FM loss
        # cannot see whether the integrated binder lands on the conditioned patch; this can.
        if "surface_xyz" in gen_batch and "surface_mask" in gen_batch:
            metrics.update(
                batch_surface_agreement_from_ca(
                    ca_nm=coors[:, :, 1, :],
                    binder_mask=gen_mask,
                    surface_xyz_nm=gen_batch["surface_xyz"],
                    surface_mask=gen_batch["surface_mask"],
                    surface_normals=gen_batch.get("surface_normals"),
                    prefix="val_gen/surface",
                )
            )

        # Cyclization closure on the SAMPLED structure: the model's own predicted (i, j, type) and
        # its own sequence's chemistry. This is the number that reads ~45% teacher-forced and has
        # never been measured on an actual sample.
        cyc = attach_cyclization_prediction(self, gen_samples, gen_batch, dict(sample_prots))
        if "pred_cyclization_i" in cyc:
            meta = {
                "i": cyc["pred_cyclization_i"].long(),
                "j": cyc["pred_cyclization_j"].long(),
                "type": cyc["pred_cyclization_type"].long(),
                "has_cyclization": torch.ones_like(cyc["pred_cyclization_i"], dtype=torch.bool),
            }
            raw = cyclic_geometry_metrics(
                pred_atom37=coors,
                gt_atom37=coors,  # only the *_gt_A / *_abs_err_A keys read this; we drop those below
                atom37_mask=atom37_mask_from_aatype(aatype) & gen_mask[..., None],
                seq_tokens=aatype.long(),
                cyclization_metadata=meta,
                prefix="val_gen/cyc",
            )
            metrics.update({f"val_gen/cyc/{s}": raw[f"val_gen/cyc/{s}"] for s in CYCLIC_PRED_ONLY_SUFFIXES})
            if "cyclization_type_satisfied" in cyc:
                metrics["val_gen/cyc/type_satisfied"] = float(
                    cyc["cyclization_type_satisfied"].float().mean().item()
                )
        else:
            # Counts are a tally, not a measurement: "no examples" is 0, never NaN (a NaN count
            # would be indistinguishable from an unmeasured rate). Rates stay NaN.
            for s in CYCLIC_PRED_ONLY_SUFFIXES:
                metrics[f"val_gen/cyc/{s}"] = 0.0 if s in CYCLIC_COUNT_SUFFIXES else float("nan")
            metrics["val_gen/cyc/type_satisfied"] = float("nan")

        # NaN (not 0) for inapplicable metrics -- e.g. isopeptide closure in a batch with no
        # isopeptides. 0 would read as "nothing closes", which is a different claim from "not
        # measurable here". `log_nan_safe` aggregates over the epoch while skipping the NaNs, and
        # logs the same key set every step so DDP never sees a rank-dependent schema.
        for k, v in metrics.items():
            self.log_nan_safe(k, float(v), bs=bs, on_step=False)

    def validation_step_lens(self, batch: dict, batch_idx: int):
        """
        Generates samples and saves samples in the self.validatoin_output list. Each sample is stored as the atom37
        coordinates, of shape [n, 37, 3]. The variable self.validation_output is thus a list of tensors of shape
        [n, 37, 3].

        Args:
            batch: data batch, contains no data, but the info of the samples to generate (nsamples, nres, dt).

        Returns:
            Nothing, just stores the samples in the list.
        """
        sampling_args = copy.deepcopy(self.cfg_exp.generation.args)
        cath_code = (
            extract_cath_code_from_batch(batch) if sampling_args.fold_cond else None
        )  # When using unconditional model, don't use cath_code
        del sampling_args.fold_cond
        del sampling_args.ag_ckpt_path
        assert sampling_args.ag_ratio == 0.0, "Should turn off autoguidance for validation"

        with torch.no_grad():
            for val_mode in self.cfg_exp.generation.model:
                # fn_predict_for_sampling = partial(
                #     self.predict_for_sampling, n_recycle=0
                # )
                fn_predict_for_sampling = self.predict_for_sampling
                gen_samples = self.fm.full_simulation(
                    predict_for_sampling=fn_predict_for_sampling,
                    batch=batch,
                    nsteps=400,
                    nsamples=batch["nsamples"],
                    n=batch["nres"],
                    self_cond=False,
                    sampling_model_args=self.cfg_exp.generation.model[val_mode],
                    device=self.device,
                )
                # Dict with the data_modes as keys, and values with batch shape b

                # Format the generated samples back to proteins
                sample_prots = sample_formatting(
                    x=gen_samples,
                    extra_info={"mask": batch["mask"]},
                    ret_mode="coors37_n_aatype",
                    data_modes=list(self.cfg_exp.product_flowmatcher),
                    autoencoder=getattr(self, "autoencoder", None),
                )
                # Dict with keys `coors` (a37), `residue_type`, and `mask`,
                # shapes [b, n, 37, 3], [b, n], [b, n]

                generation_list = []
                for i in range(sample_prots["coors"].shape[0]):
                    generation_list.append(
                        (sample_prots["coors"][i], sample_prots["residue_type"][i])
                    )  # Tuple (coors [n, 37, 3], aatype [n])

                if val_mode not in self.validation_output_lens:
                    self.validation_output_lens[val_mode] = []
                self.validation_output_lens[val_mode] += generation_list

    def on_validation_epoch_end(self):
        """Process validation results at epoch end."""
        self.on_validation_epoch_end_data()
        # TODO: Re-enable length-based generation validation once refactored.
        # Disabled because it is expensive (generates full PDB samples + runs
        # structural metrics every val epoch) and currently under rework for
        # the new group-based collation.  Training loss validation still runs.
        # self.on_validation_epoch_end_lens()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def on_validation_epoch_end_lens(self):
        """
        Generates PDB files from produced samples and computes metrics.
        It does this for all sampling modes considered.
        """
        # Save structures to pdb files
        for val_mode in self.validation_output_lens:
            paths = []
            len_tracker = {}
            for i, (coors_atom37, residue_type) in enumerate(self.validation_output_lens[val_mode]):
                n = coors_atom37.shape[-3]
                len_tracker[n] = len_tracker.get(n, 0)
                len_tracker[n] += 1
                name = (
                    f"epoch_{self.current_epoch}_bid_n_{n}_num_{len_tracker[n]}_rank_{self.global_rank}_{val_mode}.pdb"
                )
                full_path = os.path.join(self.val_path_tmp, name)
                if not os.path.exists(self.val_path_tmp):
                    create_dir(self.val_path_tmp)
                try:
                    write_prot_to_pdb(
                        prot_pos=coors_atom37.float().detach().cpu().numpy(),
                        aatype=residue_type.detach().cpu().numpy(),
                        file_path=full_path,
                        overwrite=True,
                        no_indexing=True,
                    )
                    paths.append(full_path)
                except Exception as e:
                    logger.error(
                        f"[Global rank: {self.global_rank}]: Failed to write protein to PDB on validation: {e}"
                    )

            # Subset of paths for non new metrics
            paths_subset = paths.copy()
            random.shuffle(paths_subset)
            paths_subset = paths_subset[:40]

            # Compute metrics
            try:
                structural_results = get_structural_metrics(paths_subset, val_mode)
                for log_key, value in structural_results.items():
                    self.log(
                        log_key,
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        batch_size=1,
                        sync_dist=True,
                    )
            except Exception as e:
                logger.warning(f"[Global rank: {self.global_rank}]: Failed to get structural metrics: {e}")
            if self.cfg_exp.generation.metric.compute_novelty_pdb and self.global_step > 5000:
                try:
                    novelty_results = get_pdb_novelty_metric(
                        paths_subset,
                        val_mode,
                        self.cfg_exp.hardware.ncpus_per_task_train_,
                    )
                    for log_key, value in novelty_results.items():
                        self.log(
                            log_key,
                            value,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            logger=True,
                            batch_size=1,
                            sync_dist=True,
                        )
                except Exception as e:
                    logger.warning(f"[Global rank: {self.global_rank}]: Failed to get pdb novelty metric: {e}")

            # Clean up
            clean_validation_files(paths)
        self.validation_output_lens = {}

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag

    def generate(self, batch: dict) -> dict:
        """
        Runs a single generation pass with the current configuration.

        Args:
            batch: Data batch containing generation parameters. Must contain 'mask' tensor.

        Returns:
            gen_samples: Dictionary with generated samples for each data mode
        """
        self_cond = self.inf_cfg.args.self_cond
        nsteps = self.inf_cfg.args.nsteps
        guidance_w = self.inf_cfg.args.get("guidance_w", 1.0)
        ag_ratio = self.inf_cfg.args.get("ag_ratio", 0.0)

        fn_predict_for_sampling = partial(self.predict_for_sampling, n_recycle=self.inf_cfg.get("n_recycle", 0))

        # Derive nsamples and n from mask shape
        mask = batch["mask"]
        nsamples, n = mask.shape

        # Design-task batches carry no cyclization label, so the requested type has to be
        # stamped in here for `cyclization_type_emb` to read. Done in `generate` rather than
        # in a single search algorithm so every search path (single-pass, beam, MCTS) is
        # conditioned identically.
        self.apply_cyclization_type_conditioning(batch, bs=nsamples)

        gen_samples = self.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=nsteps,
            nsamples=nsamples,
            n=n,
            self_cond=self_cond,
            sampling_model_args=self.inf_cfg.model,
            device=self.device,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
        )

        return gen_samples

    # ------------------------------------------------------------------
    # predict_step helpers
    # ------------------------------------------------------------------

    def _get_search_instance(self):
        """Return cached search instance, re-creating only if algorithm changed."""
        search_algorithm = getattr(self.inf_cfg, "search", {}).get("algorithm", "single-pass")
        if (
            not hasattr(self, "_search_instance")
            or self._search_instance is None
            or getattr(self, "_search_algorithm", None) != search_algorithm
        ):
            self._search_instance = instantiate_search(self, self.inf_cfg, search_algorithm)
            self._search_algorithm = search_algorithm
        return self._search_instance

    def _refinement_enabled(self) -> bool:
        ref_cfg = getattr(self.inf_cfg, "refinement", {})
        return bool(ref_cfg.get("algorithm", None)) if ref_cfg else False

    def _apply_refinement(self, final_prots, lookahead_prots):
        """Refine samples and optionally save pre-refinement copies.

        Only call when ``_refinement_enabled()`` is True.

        Returns (final_prots, lookahead_prots, unrefined_final, unrefined_lookahead).

        NOTE: SequenceHallucination.refine() accepts target_hotspot_mask but
        the original hardcoded hotspot=None.  When hotspot-guided refinement
        is needed, pass expand_hotspot_mask(...) here.
        """
        ref_cfg = self.inf_cfg.refinement
        algorithm = ref_cfg.get("algorithm")

        if not hasattr(self, "_refinement_instance") or self._refinement_instance is None:
            self._refinement_instance = instantiate_refinement(self, self.inf_cfg, algorithm)

        refine_targets = ref_cfg.get("refine_targets", "final")
        save_pre = ref_cfg.get("save_pre_refinement", "none")

        unrefined_final = clone_sample_dict(final_prots) if save_pre in ("final", "all") else None
        unrefined_lookahead = None
        if save_pre == "all" and refine_targets == "all" and lookahead_prots is not None:
            unrefined_lookahead = clone_sample_dict(lookahead_prots)

        final_prots = self._refinement_instance.refine(final_prots)
        if refine_targets == "all" and lookahead_prots is not None:
            lookahead_prots = self._refinement_instance.refine(lookahead_prots)

        return final_prots, lookahead_prots, unrefined_final, unrefined_lookahead

    def _append_unrefined_samples(
        self,
        sample_prots,
        unrefined_final,
        unrefined_lookahead,
        expanded_hotspot_mask,
        ligand,
    ):
        """Score and append pre-refinement copies to the output dict."""
        if unrefined_final is not None:
            unrefined_final_rewards = None
            if self.reward_model is not None:
                unrefined_final_rewards = compute_reward_from_samples(
                    self.reward_model,
                    unrefined_final,
                    expanded_hotspot_mask,
                    ligand,
                )
            append_samples(
                sample_prots,
                unrefined_final,
                unrefined_final_rewards,
                "final_unrefined",
            )
        if unrefined_lookahead is not None:
            append_samples(
                sample_prots,
                unrefined_lookahead,
                unrefined_lookahead.get("rewards"),
                "lookahead_unrefined",
            )

    # ------------------------------------------------------------------
    # predict_step
    # ------------------------------------------------------------------

    @torch.inference_mode(mode=True)
    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """Run search → (optional) refinement → reward scoring → combine.

        Args:
            batch: Must contain ``mask`` tensor [batch_size, n_residues].

        Returns:
            Dict with ``coors``, ``residue_type``, ``mask``, ``rewards``,
            ``sample_type``, and optionally ``chain_index`` / ``metadata_tag``.
        """
        if "mask" not in batch:
            raise ValueError("Batch must contain 'mask' tensor")

        if not hasattr(self, "reward_model") or self.reward_model is None:
            self.reward_model = initialize_reward_model(self.inf_cfg)

        # ---- Search ----
        search_result = self._get_search_instance().search(batch)
        final_prots = search_result["final"]
        lookahead_prots = search_result.get("lookahead")

        # ---- Refinement (only when configured) ----
        unrefined_final = None
        unrefined_lookahead = None
        if self._refinement_enabled():
            final_prots, lookahead_prots, unrefined_final, unrefined_lookahead = self._apply_refinement(
                final_prots, lookahead_prots
            )

        # ---- Score finals ----
        # Lookaheads are scored during search (tile layout, modulo correct).
        # Finals are scored here.  All search algorithms output finals in
        # grouped layout (all beams/replicas for sample 0 first, then 1, …).
        #
        # BUG FIX: the original passed raw [nsamples] hotspot_mask and used
        # i % nsamples which is wrong for grouped layout.
        # expand_hotspot_mask uses repeat_interleave to match.
        nsamples = batch["mask"].shape[0]
        hotspot_mask = batch.get("target_hotspot_mask")
        ligand = getattr(self, "ligand", None)
        expanded_hotspot_mask = expand_hotspot_mask(
            hotspot_mask,
            final_prots["coors"].shape[0],
            nsamples,
        )

        final_rewards = None
        if self.reward_model is not None:
            final_rewards = compute_reward_from_samples(
                self.reward_model,
                final_prots,
                expanded_hotspot_mask,
                ligand,
            )

        # ---- Combine output ----
        sample_prots = combine_lookahead_and_final(
            lookahead=lookahead_prots,
            final=final_prots,
            final_rewards=final_rewards,
        )

        if unrefined_final is not None or unrefined_lookahead is not None:
            self._append_unrefined_samples(
                sample_prots,
                unrefined_final,
                unrefined_lookahead,
                expanded_hotspot_mask,
                ligand,
            )

        return sample_prots
