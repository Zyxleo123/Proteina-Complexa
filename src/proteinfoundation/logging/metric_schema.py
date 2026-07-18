"""Central W&B validation metric schema (Part 4).

Goal: every training run logs the *same* set of validation metric keys,
regardless of which modules are enabled (frozen vs joint AE, cyclization head
on/off, reconstruction/four-way-decode diagnostics on/off). Metrics that do
not apply to a given run are logged as NaN rather than omitted, so W&B
comparisons across runs never have "missing panel" gaps.

Usage (see `Proteina.log_unified_validation_metrics`):

    metrics = init_metric_dict()
    metrics.update(existing_loss_metrics)
    metrics.update(reconstruction_metrics_or_nan)
    metrics.update(four_way_decode_metrics_or_nan)
    metrics = finalize_metrics(metrics)
    for k, v in metrics.items():
        self.log(k, v, ...)
"""

from __future__ import annotations

from loguru import logger

from proteinfoundation.eval.cyclic_reconstruction_metrics import RECON_METRIC_SUFFIXES

# The four decode-branch-isolation combinations (Part 3) plus the standalone
# AE-only reconstruction diagnostic (Part 2) all share the same suffix set
# (`RECON_METRIC_SUFFIXES`, defined once in `cyclic_reconstruction_metrics`).
RECON_PREFIXES = [
    "ae_recon",
    "decode_gtca_gtz",
    "decode_predca_gtz",
    "decode_gtca_predz",
    "decode_predca_predz",
]

FLOW_CLEAN_KEYS = [
    "val/flow_clean/ca_coord_mae_A",
    "val/flow_clean/ca_rmse_A",
    "val/flow_clean/z_mse",
    "val/flow_clean/z_l1",
]

# Existing metrics, renamed/kept under the new uniform `val/...` namespace.
# These are *additional* aliases -- the original `validation_loss/...` keys
# logged elsewhere in `Proteina.training_step` are left untouched.
EXISTING_RENAMED_KEYS = [
    "val/loss/latent",
    "val/loss/bb_ca",
    "val/cyclization/acc",
    "val/cyclization/ce",
]

ALL_VALIDATION_METRIC_KEYS: list[str] = (
    [f"val/{prefix}/{suffix}" for prefix in RECON_PREFIXES for suffix in RECON_METRIC_SUFFIXES]
    + FLOW_CLEAN_KEYS
    + EXISTING_RENAMED_KEYS
)

_warned_missing_keys: set[str] = set()


def init_metric_dict() -> dict[str, float]:
    """Returns a fresh dict with every key in `ALL_VALIDATION_METRIC_KEYS` set to NaN."""
    return {k: float("nan") for k in ALL_VALIDATION_METRIC_KEYS}


def finalize_metrics(metrics: dict[str, float], warn_missing: bool = True) -> dict[str, float]:
    """Fills any key from `ALL_VALIDATION_METRIC_KEYS` missing (or `None`) in `metrics` with NaN.

    Extra keys already present in `metrics` (e.g. run-specific diagnostics not part of the
    universal schema) are preserved as-is. Warns (once per key, via `loguru`) the first time a
    schema key is found missing, so silent drift between this schema and the callers that are
    supposed to populate it is caught without spamming logs every validation step.
    """
    out = dict(metrics)
    for key in ALL_VALIDATION_METRIC_KEYS:
        if out.get(key, None) is None:
            out[key] = float("nan")
            if warn_missing and key not in _warned_missing_keys:
                _warned_missing_keys.add(key)
                logger.warning(f"metric_schema: key '{key}' missing from validation metrics; logging NaN.")
    return out
