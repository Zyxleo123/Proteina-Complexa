# Apply atomworks patches early - before any imports that use atomworks/biotite
import json
import os
import re
import sys
import uuid
from pathlib import Path

import hydra
import lightning as L
import loralib as lora
import torch
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

import proteinfoundation.patches.atomworks_patches  # noqa: F401
from proteinfoundation.proteina import Proteina
from proteinfoundation.utils.ema_callback import EMA, EmaModelCheckpoint
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
from proteinfoundation.utils.fold_utils import transform_global_percentage_to_mask_dropout
from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import (
    GradAndWeightAnalysisCallback,
    LogEpochTimeCallback,
    LogSetpTimeCallback,
    SkipNanGradCallback,
)


@rank_zero_only
def log_info(msg: str):
    logger.info(msg)


@rank_zero_only
def create_dir(ckpt_path_store: str, parents: bool = True, exist_ok: bool = True):
    Path(ckpt_path_store).mkdir(parents=parents, exist_ok=exist_ok)


def _wandb_run_id_path(ckpt_path_store: str) -> str:
    return os.path.join(ckpt_path_store, "wandb_run_id.txt")


@rank_zero_only
def _save_wandb_run_id(ckpt_path_store: str, wandb_id: str) -> None:
    create_dir(ckpt_path_store, parents=True, exist_ok=True)
    path = _wandb_run_id_path(ckpt_path_store)
    Path(path).write_text(wandb_id.strip() + "\n")
    log_info(f"Saved WandB run id to {path}")


def _recover_wandb_run_id_from_logs(run_name: str) -> str | None:
    """Backfill wandb id for runs started before we persisted wandb_run_id.txt."""
    log_dirs: list[Path] = []
    zfs = os.environ.get("PROTEINA_ZFS_PATH", "")
    if zfs:
        log_dirs.append(Path(zfs) / "training_runs" / "logs" / "training")
    cpsea_log = os.environ.get("CPSEA_LOG_DIR", "")
    if cpsea_log:
        log_dirs.append(Path(cpsea_log))

    pattern = re.compile(rf"runs/{re.escape(run_name)}-[a-f0-9]{{8}}")
    for log_dir in log_dirs:
        if not log_dir.is_dir():
            continue
        logs = sorted(log_dir.glob(f"{run_name}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for log_path in logs[:5]:
            try:
                chunk = log_path.read_text(errors="ignore")[:1_000_000]
            except OSError:
                continue
            if m := pattern.search(chunk):
                return m.group(0).split("/")[-1]
    return None


def _load_wandb_run_id(ckpt_path_store: str, run_name: str) -> str | None:
    path = _wandb_run_id_path(ckpt_path_store)
    if os.path.isfile(path):
        return Path(path).read_text().strip()
    recovered = _recover_wandb_run_id_from_logs(run_name)
    if recovered:
        log_info(f"Recovered WandB run id from training logs: {recovered}")
        _save_wandb_run_id(ckpt_path_store, recovered)
    return recovered


def check_cluster() -> bool:
    """Verifies whether this is running on the cluster."""
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    is_cluster_run = slurm_job_id is not None
    if is_cluster_run:
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        )  # Send to stdout
    log_info(f"Is cluster run: {is_cluster_run}")
    log_info(f"SLURM job id: {slurm_job_id}")
    return is_cluster_run


# TODO: This is no longer used
def handle_cath_conditioning(cfg_exp) -> None:
    """Sets up dropout ratio for CATH conditioning based on global percentage."""
    if cfg_exp.training.get("fold_label_sample_ratio") is not None:
        log_info("Setting fold label dropout rate based on fold_label_sample_ratio")
        (
            cfg_exp.training.mask_T_prob,
            cfg_exp.training.mask_A_prob,
            cfg_exp.training.mask_C_prob,
        ) = transform_global_percentage_to_mask_dropout(cfg_exp.training.fold_label_sample_ratio)
        log_info(
            "Set mask_T_prob: %.3f, mask_A_prob: %.3f, mask_C_prob: %.3f"
            % (
                cfg_exp.training.mask_T_prob,
                cfg_exp.training.mask_A_prob,
                cfg_exp.training.mask_C_prob,
            )
        )
    return cfg_exp


def get_run_dirs(cfg_exp) -> tuple[str, str, str]:
    """Get root directory for run and directory to store checkpoints."""
    run_name = cfg_exp.run_name
    log_info(f"Job name: {run_name}")
    root_run = os.path.join(".", "store", run_name)  # Everything stored in ./store/<run_id>
    log_info(f"Root run: {root_run}")

    ckpt_path_store = os.path.join(root_run, "checkpoints")  # Checkpoints in ./store/run_id/checkpoints/<ckpt-file>
    log_info(f"Checkpoints directory: {ckpt_path_store}")
    return run_name, root_run, ckpt_path_store


def _fetch_resume_ckpt_name(cfg_exp, ckpt_path_store: str) -> str | None:
    """Return last.ckpt name to resume from, or None for a fresh start from pretrain_ckpt_path."""
    training_cfg = cfg_exp.get("training", {})
    resume_from_last = True if training_cfg is None else training_cfg.get("resume_from_last", True)
    if not resume_from_last:
        log_info("training.resume_from_last=false — ignoring existing last.ckpt")
        return None
    return fetch_last_ckpt(ckpt_path_store)


def initialize_callbacks(cfg_exp) -> list:
    """Initializes general training callbacks."""
    callbacks = [SeedCallback()]  # , UnusedParametersCallback()]  # Different devices will be assigend different seeds

    # Gradient and weight stats thoughout training, possibly skip updates with nan in grad
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    if cfg_exp.opt.skip_nan_grad:
        callbacks.append(SkipNanGradCallback())

    callbacks.append(LogEpochTimeCallback())
    callbacks.append(LogSetpTimeCallback())

    log_info(f"Using EMA with decay {cfg_exp.ema.decay}")
    callbacks.append(EMA(**cfg_exp.ema))
    return callbacks


def get_training_precision(cfg_exp, is_cluster_run: bool) -> str:
    """Gets and sets correct training precision."""
    precision = "32"
    if not cfg_exp.force_precision_f32:
        log_info("Using mixed precision")
        torch.set_float32_matmul_precision("medium")
        if is_cluster_run:
            precision = "bf16-mixed"
        else:
            precision = "16"
    else:
        torch.set_float32_matmul_precision("high")
    return precision


def _resolve_datamodule_config(cfg_data):
    """Resolve Lightning datamodule config, merging unified base with overrides.

    CPSea and other unified dataset configs place the full datamodule under
    ``dataset.unified.datamodule`` while training configs override a subset at
    ``dataset.datamodule``.  Hydra keeps those as siblings; this helper merges
    them so ``_target_`` and transforms are not lost.
    """
    override_cfg = cfg_data.get("datamodule")
    unified_cfg = cfg_data.get("unified", {}).get("datamodule") if hasattr(cfg_data, "get") else None

    if override_cfg is None and unified_cfg is None:
        return None
    if override_cfg is None:
        return unified_cfg
    if unified_cfg is not None and override_cfg.get("_target_") is None:
        return OmegaConf.merge(unified_cfg, override_cfg)
    return override_cfg


def load_data_module(cfg_exp, is_cluster_run: bool) -> tuple:
    """Loads data config and creates corresponding datamodule.

    Supports two patterns:
    1. Unified datamodule (Lightning): config has 'datamodule' key
    2. Atomworks config: config has 'train' key with atomworks dataset definitions
    """
    num_cpus = cfg_exp.hardware.ncpus_per_task_train_
    log_info(f"Number of CPUs per task used (will be used for number dataloader number of workers): {num_cpus}")
    cfg_data = cfg_exp.dataset

    dm_cfg = _resolve_datamodule_config(cfg_data)

    # Check for unified/Lightning datamodule pattern
    if dm_cfg is not None:
        # Overwrite number of workers
        if hasattr(dm_cfg, "num_workers"):
            dm_cfg.num_workers = num_cpus
        log_info(f"Data config {cfg_data}")

        # Instantiate the datamodule
        datamodule = hydra.utils.instantiate(dm_cfg)

        # Add validation loader if supported
        if hasattr(datamodule, "add_validation_dataloader") and hasattr(cfg_exp, "generation"):
            cfg_exp_val_data = cfg_exp.generation.dataloader
            n_replicas = cfg_exp.hardware.ngpus_per_node_ * cfg_exp.hardware.nnodes_
            datamodule.add_validation_dataloader(cfg_exp_val_data, n_replicas=n_replicas)

        return cfg_data, datamodule

    # Check for atomworks train config pattern
    elif hasattr(cfg_data, "train"):
        from proteinfoundation.datasets.atomworks_utils import (
            recursively_instantiate_datasets_and_samplers,
            simple_dataloader,
        )

        log_info(f"Data config {cfg_data}")

        # Instantiate datasets and samplers
        dataset_and_sampler = recursively_instantiate_datasets_and_samplers(cfg_data.train)

        # Create dataloader
        train_loader = simple_dataloader(
            dataset=dataset_and_sampler["dataset"],
            loader_cfg=cfg_exp.dataloader["train"],
        )

        # Return a simple namespace-like object that has train_dataloader
        class AtomworksDataModule:
            def __init__(self, train_loader):
                self._train_loader = train_loader

            def train_dataloader(self):
                return self._train_loader

            def val_dataloader(self):
                return None

        datamodule = AtomworksDataModule(train_loader)
        return cfg_data, datamodule

    else:
        raise ValueError(
            "Dataset config must have either 'datamodule' key (for unified/Lightning pattern) "
            "or 'train' key (for atomworks pattern). "
            f"Found keys: {list(cfg_data.keys())}"
        )


def _splice_tensor_to_model_shape(model_tensor: torch.Tensor, ckpt_tensor: torch.Tensor) -> torch.Tensor:
    """Copy checkpoint values into the leading overlap of ``model_tensor``; zero-fill the rest."""
    if model_tensor.shape == ckpt_tensor.shape:
        return ckpt_tensor
    if model_tensor.ndim != ckpt_tensor.ndim:
        raise ValueError(
            f"rank mismatch: model {tuple(model_tensor.shape)} vs ckpt {tuple(ckpt_tensor.shape)}"
        )
    out = torch.zeros_like(model_tensor)
    overlap_slices = tuple(slice(0, min(ms, cs)) for ms, cs in zip(model_tensor.shape, ckpt_tensor.shape))
    out[overlap_slices] = ckpt_tensor[overlap_slices]
    return out


def _splice_pretrained_weights(
    model_state_dict: dict,
    ckpt_state_dict: dict,
    skip_prefixes: tuple[str, ...] = ("nn.concat_pair_factory", "autoencoder"),
) -> dict:
    """Splice pre-trained weights into a model with potentially different dimensions.

    For keys with shape mismatches, copies the overlapping leading subtensor from the
    checkpoint and zero-initializes any extra capacity (wider layers, longer pair repr,
    etc.). Skips keys matching ``skip_prefixes``. Keys absent from the checkpoint keep
    the model's random initialization via ``load_state_dict(strict=False)``.

    ``autoencoder`` is skipped because a flow ``pretrain_ckpt_path`` (e.g. complexa.ckpt)
    bundles its own ``autoencoder.*`` weights, which would otherwise OVERWRITE the AE that
    ``Proteina.__init__`` already loaded from ``autoencoder_ckpt_path`` -- silently replacing
    a task-finetuned AE (e.g. the CPSea cyclic-peptide AE) with the pretrained one. Freezing
    does not protect against this: it only stops gradients, not ``load_state_dict``. The AE
    must always come from ``autoencoder_ckpt_path``; the flow checkpoint never touches it.
    This is independent of ``freeze_autoencoder`` (joint training still starts the AE from
    ``autoencoder_ckpt_path`` and then trains it).

    Args:
        model_state_dict: Current model state dict.
        ckpt_state_dict: Pre-trained checkpoint state dict.
        skip_prefixes: Key prefixes to skip entirely (all weights under prefix).

    Returns:
        Spliced state dict ready for model.load_state_dict(strict=False).
    """

    state_dict = {}
    n_spliced = 0
    n_skipped_shape = 0
    for k, v in ckpt_state_dict.items():
        if any(k.startswith(prefix) for prefix in skip_prefixes):
            continue
        if k not in model_state_dict:
            continue
        model_v = model_state_dict[k]
        if model_v.shape == v.shape:
            state_dict[k] = v
        elif model_v.ndim == v.ndim:
            state_dict[k] = _splice_tensor_to_model_shape(model_v, v)
            n_spliced += 1
        else:
            n_skipped_shape += 1
    if n_spliced:
        log_info(f"Spliced {n_spliced} checkpoint tensors with shape mismatches (overlap copy + zero pad).")
    if n_skipped_shape:
        log_info(f"Skipped {n_skipped_shape} checkpoint tensors with incompatible rank.")
    return state_dict


def get_model_n_ckpt_resume(cfg_exp, ckpt_path_store: str) -> tuple[Proteina, str | None]:
    """Loads model and checkpoint for training.

    Handles pre-trained checkpoint loading (with weight splicing for shape mismatches),
    LoRA layer replacement, and training resumption from last checkpoint.
    """
    model = Proteina(cfg_exp)

    # get last ckpt if needs to resume training from there
    last_ckpt_name = _fetch_resume_ckpt_name(cfg_exp, ckpt_path_store)
    last_ckpt_path = os.path.join(ckpt_path_store, last_ckpt_name) if last_ckpt_name is not None else None
    log_info(f"Last checkpoint: {last_ckpt_path}")

    # If LoRA is turned on, replace Linear with LoRA layers
    if cfg_exp.get("lora") and cfg_exp.lora.get("r"):
        replace_lora_layers(
            model,
            cfg_exp.lora.r,
            cfg_exp.lora.lora_alpha,
            cfg_exp.lora.lora_dropout,
            cfg_exp.lora.get("exclude_keys", ()),
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)

    # If this is the first run for fine-tuning, load pre-trained checkpoint and don't load optimizer states
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    if last_ckpt_path is None and pretrain_ckpt_path is not None:
        if not os.path.exists(pretrain_ckpt_path):
            raise FileNotFoundError(f"Pre-trained checkpoint not found: {pretrain_ckpt_path}")
        log_info(f"Loading from pre-trained checkpoint path {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _splice_pretrained_weights(model.state_dict(), ckpt["state_dict"])
        model.load_state_dict(state_dict, strict=False)

    # If not resuming from `last` ckpt training set seed
    if last_ckpt_path is None:
        log_info(f"Seeding everything to seed {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed)

    # # OPTIMIZATION: Remove decoder from autoencoder during training (only encoder needed)
    # if model.autoencoder is not None:
    #     log_info(
    #         "Removing autoencoder decoder during training to save memory (encoder only needed)"
    #     )
    #     del model.autoencoder.decoder
    #     model.autoencoder.decoder = None
    #     # Force garbage collection to free memory immediately
    #     import gc

    #     gc.collect()
    #     torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return model, last_ckpt_path


def setup_ckpt(cfg_exp, ckpt_path_store: str) -> list:
    """Creates checkpointing callbacks and directory to store checkpoints."""
    save_extra = bool(cfg_exp.log.get("save_extra_checkpoints", False))
    args_ckpt_last = {
        "dirpath": ckpt_path_store,
        "save_weights_only": False,
        "filename": "ignore" if save_extra else "last",
        "every_n_train_steps": cfg_exp.log.last_ckpt_every_n_steps,
        "save_last": save_extra,
    }
    checkpoint_callback_last = EmaModelCheckpoint(**args_ckpt_last)
    callbacks = [checkpoint_callback_last]
    if save_extra:
        args_ckpt = {
            "dirpath": ckpt_path_store,
            "save_last": False,
            "save_weights_only": False,
            "filename": "chk_{epoch:08d}_{step:012d}",
            "every_n_train_steps": cfg_exp.log.checkpoint_every_n_steps,
            "monitor": "train_loss",
            "save_top_k": 10000,
            "mode": "min",
        }
        callbacks.insert(0, EmaModelCheckpoint(**args_ckpt))

    create_dir(ckpt_path_store, parents=True, exist_ok=True)
    return callbacks


@rank_zero_only
def store_config_files(cfg_exp, cfg_data, run_name: str, ckpt_path_store: str) -> tuple[str, str]:
    """Write resolved exp/data configs to the checkpoint dir (local only)."""

    def _write(cfg, cfg_path: str) -> None:
        with open(cfg_path, "w") as f:
            cfg_aux = OmegaConf.to_container(cfg, resolve=True)
            json.dump(cfg_aux, f, indent=4, sort_keys=True)

    cfg_exp_file = os.path.join(ckpt_path_store, f"exp_config_{run_name}.json")
    cfg_data_file = os.path.join(ckpt_path_store, f"data_config_{run_name}.json")
    _write(cfg_exp, cfg_exp_file)
    _write(cfg_data, cfg_data_file)
    return cfg_exp_file, cfg_data_file


class LogConfigArtifactsCallback(Callback):
    """Upload config JSONs to WandB after the run is live (on_train_start).

    store_config_files() runs before trainer.fit(), but WandB is not reliably initialized
    until training starts; logging artifacts too early often silently no-ops.
    """

    def __init__(self, cfg_exp_file: str, cfg_data_file: str, run_name: str):
        self.cfg_exp_file = cfg_exp_file
        self.cfg_data_file = cfg_data_file
        self.run_name = run_name

    def on_train_start(self, trainer, pl_module) -> None:
        loggers = []
        if getattr(trainer, "loggers", None):
            loggers = list(trainer.loggers)
        elif trainer.logger is not None:
            loggers = [trainer.logger]

        for lg in loggers:
            if not isinstance(lg, WandbLogger):
                continue
            run = lg.experiment
            for label, path in (("exp", self.cfg_exp_file), ("data", self.cfg_data_file)):
                artifact = wandb.Artifact(f"config_{label}_{self.run_name}", type="config")
                artifact.add_file(path)
                run.log_artifact(artifact)
            log_info(f"WandB config artifacts logged for run id={run.id} url={run.url}")
            return


@rank_zero_only
def store_n_log_configs(cfg_exp, cfg_data, run_name: str, ckpt_path_store: str, wandb_logger) -> tuple[str, str]:
    """Stores config files locally. WandB upload is deferred to LogConfigArtifactsCallback."""
    del wandb_logger  # kept for call-site compatibility
    return store_config_files(cfg_exp, cfg_data, run_name, ckpt_path_store)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training_local_latents",
)
def main(cfg_exp) -> None:
    load_dotenv()
    log_info(f"Name of config being used: {HydraConfig.get().job.config_name}")

    is_cluster_run = check_cluster()
    nolog = cfg_exp.get("nolog", False)  # To use do `python proteinfoundation/train.py +nolog=true`
    single = cfg_exp.get("single", False)
    show_prog_bar = cfg_exp.get("show_prog_bar", False)
    if not is_cluster_run or single:
        # Rewrite number of GPUs and nodes for local runs or if single flag is used
        cfg_exp.hardware.ngpus_per_node_ = 1
        cfg_exp.hardware.nnodes_ = 1
        cfg_exp.run_name = cfg_exp.run_name + "_local"
    log_info(f"Exp config {cfg_exp}")

    run_name, root_run, ckpt_path_store = get_run_dirs(cfg_exp)
    callbacks = initialize_callbacks(cfg_exp)

    # Decide WandB resume before creating the logger. A fixed `id=run_name` with the default
    # resume="allow" re-attaches to any prior run with that id in WANDB_DIR — even when there
    # is no last.ckpt and we are loading complexa.ckpt at step 0. Only resume the WandB run
    # when we are actually resuming training from last.ckpt.
    last_ckpt_name = _fetch_resume_ckpt_name(cfg_exp, ckpt_path_store)
    resuming_training = last_ckpt_name is not None
    log_info(f"WandB resume={resuming_training} (last ckpt: {last_ckpt_name})")

    # logger
    wandb_logger = None
    if cfg_exp.log.log_wandb and not nolog:
        wandb_project = cfg_exp.log.wandb_project
        wandb_entity = cfg_exp.log.get("wandb_entity", None)
        # `log.wandb_force_new_run=true` keeps the training resume (optimizer state, global step)
        # but logs into a fresh WandB run instead of re-attaching to the interrupted one. The new
        # id is persisted, so later auto-resumes follow the new run rather than the abandoned one.
        force_new_wandb_run = cfg_exp.log.get("wandb_force_new_run", False)
        if resuming_training and force_new_wandb_run:
            log_info("log.wandb_force_new_run=true — resuming training into a NEW WandB run")
        if resuming_training and not force_new_wandb_run:
            wandb_id = _load_wandb_run_id(ckpt_path_store, run_name)
            if not wandb_id:
                raise RuntimeError(
                    f"Resuming training from last.ckpt but no WandB run id found for {run_name!r}. "
                    f"Expected {_wandb_run_id_path(ckpt_path_store)} or a wandb URL in training logs. "
                    f"Create the file manually, e.g.:\n"
                    f'  echo "{run_name}-<8-char-suffix>" > {_wandb_run_id_path(ckpt_path_store)}'
                )
            logger.info(
                f"Using WandB logger (resume): project={wandb_project}, "
                f"id={wandb_id}, name={run_name}, entity={wandb_entity}"
            )
            wandb_logger = WandbLogger(
                project=wandb_project,
                id=wandb_id,
                name=run_name,
                entity=wandb_entity,
                resume="must",
            )
        else:
            wandb_id = f"{run_name}-{uuid.uuid4().hex[:8]}"
            logger.info(
                f"Using WandB logger (new run): project={wandb_project}, "
                f"id={wandb_id}, name={run_name}, entity={wandb_entity}"
            )
            wandb_logger = WandbLogger(
                project=wandb_project,
                id=wandb_id,
                name=run_name,
                entity=wandb_entity,
                resume="never",
            )
            _save_wandb_run_id(ckpt_path_store, wandb_id)

    Trainer = L.Trainer

    cfg_data, datamodule = load_data_module(cfg_exp, is_cluster_run)

    cfg_exp_file, cfg_data_file = None, None
    # checkpoints
    if cfg_exp.log.checkpoint:  # and not nolog:
        ckpt_callbacks = setup_ckpt(cfg_exp, ckpt_path_store)
        callbacks += ckpt_callbacks
        cfg_exp_file, cfg_data_file = store_n_log_configs(
            cfg_exp, cfg_data, run_name, ckpt_path_store, wandb_logger
        )
    if wandb_logger is not None and cfg_exp_file is not None:
        callbacks.append(LogConfigArtifactsCallback(cfg_exp_file, cfg_data_file, run_name))

    # Train
    plugins = [SLURMEnvironment(auto_requeue=True)] if is_cluster_run else []
    # show_prog_bar = args.show_prog_bar or not is_cluster_run
    show_prog_bar = show_prog_bar or not is_cluster_run
    trainer = Trainer(
        max_epochs=cfg_exp.opt.max_epochs,
        accelerator=cfg_exp.hardware.accelerator,
        devices=cfg_exp.hardware.ngpus_per_node_,  # This is number of gpus per node, not total
        num_nodes=cfg_exp.hardware.nnodes_,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg_exp.log.log_every_n_steps,
        default_root_dir=root_run,
        check_val_every_n_epoch=None,  # Leave like this
        val_check_interval=cfg_exp.opt.val_check_interval,
        strategy=cfg_exp.opt.dist_strategy,
        enable_progress_bar=show_prog_bar,
        plugins=plugins,
        limit_val_batches=100,
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=0,
        precision=get_training_precision(cfg_exp, is_cluster_run),
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        limit_train_batches=cfg_exp.opt.get("limit_train_batches", None),
    )
    # Create model, warm-up or last ckpt
    model, resume_ckpt_path = get_model_n_ckpt_resume(cfg_exp, ckpt_path_store)
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
