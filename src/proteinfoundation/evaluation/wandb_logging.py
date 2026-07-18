"""Log evaluation summary metrics to Weights & Biases."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "n_valid": float(len(values)),
    }


def _collect_wandb_payload(
    all_results: dict[str, pd.DataFrame],
    evaluation_time: float,
    job_id: int,
    nsamples: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "eval/job_id": job_id,
        "eval/nsamples": nsamples,
        "eval/evaluation_time_s": evaluation_time,
    }

    for result_name, df in all_results.items():
        if df is None or df.empty:
            continue
        payload[f"eval/{result_name}/n_rows"] = len(df)

        for col in df.columns:
            if not any(
                token in col
                for token in (
                    "rosetta",
                    "interface",
                    "tmol",
                    "i_pAE",
                    "ipAE",
                    "pLDDT",
                    "scRMSD",
                    "ipsae",
                    "designability",
                    "codesignability",
                )
            ):
                continue
            summary = _numeric_summary(df[col])
            for stat_name, stat_value in summary.items():
                if stat_value is None or (isinstance(stat_value, float) and math.isnan(stat_value)):
                    continue
                payload[f"eval/{result_name}/{col}/{stat_name}"] = stat_value

    return payload


def log_evaluation_results_to_wandb(
    all_results: dict[str, pd.DataFrame],
    cfg: DictConfig,
    job_id: int,
    evaluation_time: float,
    nsamples: int,
    config_name: str | None = None,
) -> None:
    """Initialize a wandb run and log per-metric evaluation summaries."""
    log_cfg = cfg.get("log", {})
    if not log_cfg.get("log_wandb", False):
        return

    if log_cfg.get("wandb_log_only_job0", True) and job_id != 0:
        logger.info(f"Skipping wandb logging for job_id={job_id} (wandb_log_only_job0=true)")
        return

    try:
        import wandb
    except ImportError:
        logger.warning("wandb logging requested but wandb is not installed")
        return

    config_name = config_name or cfg.get("base_config_name") or cfg.get("config_name") or "evaluate"
    run_name = log_cfg.get("wandb_run_name") or cfg.get("run_name") or config_name
    if cfg.get("eval_njobs", 1) > 1:
        run_name = f"{run_name}_job{job_id}"

    wandb.init(
        project=log_cfg.get("wandb_project", "complexa_eval"),
        entity=log_cfg.get("wandb_entity"),
        name=run_name,
        config={
            "config_name": config_name,
            "protein_type": cfg.get("protein_type"),
            "input_mode": cfg.get("input_mode"),
            "job_id": job_id,
            "eval_njobs": cfg.get("eval_njobs", 1),
        },
        reinit=True,
    )

    payload = _collect_wandb_payload(all_results, evaluation_time, job_id, nsamples)
    wandb.log(payload)
    logger.info(f"Logged {len(payload)} evaluation summary metrics to wandb (run={run_name})")
    wandb.finish()
