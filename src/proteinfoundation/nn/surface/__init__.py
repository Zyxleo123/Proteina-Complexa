"""Peptide-interface surface conditioning modules for LocalLatentsTransformer."""

from proteinfoundation.nn.surface.encoder import (
    BinderSurfacePairFeatures,
    GatedSurfaceCrossAttention,
    SurfaceEncoder,
    IntraSurfacePairFeatures,
)

__all__ = [
    "BinderSurfacePairFeatures",
    "GatedSurfaceCrossAttention",
    "IntraSurfacePairFeatures",
    "SurfaceEncoder",
]
