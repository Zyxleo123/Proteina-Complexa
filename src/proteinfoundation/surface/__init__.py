"""Molecular-surface point clouds for peptide-receptor complexes.

Ported from PepBridge (MIT, Copyright (c) 2026 Guanlue Li); see
`proteinfoundation.surface.peptide_surface` for the full notice, the list of what was
ported, and the behavioural changes.
"""

from proteinfoundation.surface.peptide_surface import (
    DEFAULT_BACKEND,
    DEFAULT_CUTOFF,
    DEFAULT_NUM_POINTS,
    DEFAULT_SAS_POINTS_PER_ATOM,
    DEFAULT_SEED,
    EXTRACTOR_VERSION,
    PeptideSurface,
    SurfaceExtractionError,
    cache_path_for,
    extract_peptide_surface,
    farthest_point_sample,
    generate_surface_obj,
    interface_mask,
    is_cache_valid,
    load_mesh,
    load_surface_cache,
    nearest_receptor_distance,
    normalize_normals,
    patch_components,
    resolve_cache_path,
    resolve_chain_assignment,
    sampling_coverage,
    save_surface_cache,
    split_chains,
    transform_surface,
)

__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_CUTOFF",
    "DEFAULT_NUM_POINTS",
    "DEFAULT_SAS_POINTS_PER_ATOM",
    "DEFAULT_SEED",
    "EXTRACTOR_VERSION",
    "PeptideSurface",
    "SurfaceExtractionError",
    "cache_path_for",
    "extract_peptide_surface",
    "farthest_point_sample",
    "generate_surface_obj",
    "interface_mask",
    "is_cache_valid",
    "load_mesh",
    "load_surface_cache",
    "nearest_receptor_distance",
    "normalize_normals",
    "patch_components",
    "resolve_cache_path",
    "resolve_chain_assignment",
    "sampling_coverage",
    "save_surface_cache",
    "split_chains",
    "transform_surface",
]
