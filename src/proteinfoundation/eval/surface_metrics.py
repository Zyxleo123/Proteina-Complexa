"""Oracle peptide-surface agreement metrics (Stage 5).

Compare a generated peptide (CA trace or extracted surface) against a ground-truth
interface surface cache. Complements existing binder / cyclization / Rosetta metrics;
does not replace them.

Typical use after generation (nm or Å — pass consistent units):

    from proteinfoundation.eval.surface_metrics import surface_agreement_metrics
    from proteinfoundation.surface.peptide_surface import load_surface_cache

    oracle = load_surface_cache(cache_path)
    metrics = surface_agreement_metrics(
        pred_xyz=sampled_ca_nm * 10.0,  # to Å if model is in nm
        oracle_xyz=oracle.sampled_xyz[oracle.sampled_valid_mask],
        oracle_normals=oracle.sampled_normals[oracle.sampled_valid_mask],
    )
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1, 3)


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between point sets ``a`` and ``b``."""
    a = _as_f64(a)
    b = _as_f64(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # (Na, Nb)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(0.5 * (np.sqrt(d2.min(axis=1)).mean() + np.sqrt(d2.min(axis=0)).mean()))


def normal_consistency(
    pred_xyz: np.ndarray,
    oracle_xyz: np.ndarray,
    oracle_normals: np.ndarray,
    pred_normals: np.ndarray | None = None,
) -> float:
    """Mean |n·n̂| between each oracle normal and the nearest predicted point's normal.

    If ``pred_normals`` is None, returns NaN (caller must supply predicted normals for
    a full consistency score). When only oracle normals are available, reports the
    mean absolute cosine between each oracle normal and the direction from the oracle
    point to its nearest predicted point (a weak proxy for facing agreement).
    """
    pred_xyz = _as_f64(pred_xyz)
    oracle_xyz = _as_f64(oracle_xyz)
    oracle_normals = _as_f64(oracle_normals)
    if pred_xyz.size == 0 or oracle_xyz.size == 0:
        return float("nan")

    d2 = ((oracle_xyz[:, None, :] - pred_xyz[None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(axis=1)
    if pred_normals is not None:
        pred_normals = _as_f64(pred_normals)
        pn = pred_normals[nn]
        pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-8)
        on = oracle_normals / np.maximum(np.linalg.norm(oracle_normals, axis=1, keepdims=True), 1e-8)
        return float(np.abs((on * pn).sum(1)).mean())

    # Proxy: alignment of oracle normal with vector toward nearest predicted point.
    vec = pred_xyz[nn] - oracle_xyz
    vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8)
    on = oracle_normals / np.maximum(np.linalg.norm(oracle_normals, axis=1, keepdims=True), 1e-8)
    return float(np.abs((on * vec).sum(1)).mean())


def surface_coverage(
    oracle_xyz: np.ndarray,
    pred_xyz: np.ndarray,
) -> tuple[float, float]:
    """(mean, max) distance from each oracle point to its nearest predicted point."""
    oracle_xyz = _as_f64(oracle_xyz)
    pred_xyz = _as_f64(pred_xyz)
    if oracle_xyz.size == 0 or pred_xyz.size == 0:
        return float("nan"), float("nan")
    d2 = ((oracle_xyz[:, None, :] - pred_xyz[None, :, :]) ** 2).sum(-1)
    d = np.sqrt(d2.min(axis=1))
    return float(d.mean()), float(d.max())


def surface_agreement_metrics(
    pred_xyz,
    oracle_xyz,
    oracle_normals=None,
    pred_normals=None,
) -> Mapping[str, float]:
    """Bundle Chamfer / normal consistency / coverage for one complex."""
    pred_xyz = _as_f64(pred_xyz)
    oracle_xyz = _as_f64(oracle_xyz)
    cov_mean, cov_max = surface_coverage(oracle_xyz, pred_xyz)
    out = {
        "surface_chamfer": chamfer_distance(pred_xyz, oracle_xyz),
        "surface_coverage_mean": cov_mean,
        "surface_coverage_max": cov_max,
    }
    if oracle_normals is not None:
        out["surface_normal_consistency"] = normal_consistency(
            pred_xyz, oracle_xyz, oracle_normals, pred_normals=pred_normals
        )
    return out


def batch_surface_agreement_from_ca(
    ca_nm,
    binder_mask,
    surface_xyz_nm,
    surface_mask,
    surface_normals=None,
    *,
    prefix: str = "val_gen/surface",
) -> dict[str, float]:
    """Mean surface agreement of sampled binder CA against batch surface caches.

    Units are converted to Å at the boundary (caches / metrics are Å-native). Returns a
    fixed key set; missing / empty samples contribute NaN and are skipped in the mean.
    """
    nm_to_ang = 10.0
    ca = ca_nm.detach().float().cpu().numpy() * nm_to_ang
    bmask = binder_mask.detach().bool().cpu().numpy()
    sxyz = surface_xyz_nm.detach().float().cpu().numpy() * nm_to_ang
    smask = surface_mask.detach().bool().cpu().numpy()
    snorm = None
    if surface_normals is not None:
        snorm = surface_normals.detach().float().cpu().numpy()

    keys = (
        "chamfer_A",
        "coverage_mean_A",
        "coverage_max_A",
        "normal_consistency",
    )
    acc: dict[str, list[float]] = {k: [] for k in keys}
    b = ca.shape[0]
    for i in range(b):
        pred = ca[i][bmask[i]]
        oracle = sxyz[i][smask[i]]
        if pred.size == 0 or oracle.size == 0:
            continue
        on = snorm[i][smask[i]] if snorm is not None else None
        m = surface_agreement_metrics(pred, oracle, oracle_normals=on)
        acc["chamfer_A"].append(m["surface_chamfer"])
        acc["coverage_mean_A"].append(m["surface_coverage_mean"])
        acc["coverage_max_A"].append(m["surface_coverage_max"])
        if "surface_normal_consistency" in m:
            acc["normal_consistency"].append(m["surface_normal_consistency"])

    out: dict[str, float] = {}
    for k in keys:
        vals = acc[k]
        out[f"{prefix}/{k}"] = float(np.nanmean(vals)) if vals else float("nan")
    return out
