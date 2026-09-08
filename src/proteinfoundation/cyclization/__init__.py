"""CPSea cyclization-linkage prediction head.

A lightweight, purely classification-based auxiliary head that predicts the
single cyclization edge `(i, j, type)` of a cyclic peptide binder from the
flow model's predicted clean state. This module intentionally does not
enforce any chemical geometry and never modifies coordinates: it only
produces typed pairwise logits and a cross-entropy training signal.
"""

from proteinfoundation.cyclization.constants import (
    COND_TYPE_TO_NAME,
    CYCLIZATION_TYPE_TO_NAME,
    DISULFIDE,
    ISOPEPTIDE,
    LINEAR,
    MAINCHAIN,
    NAME_TO_COND_TYPE,
    NAME_TO_CYCLIZATION_TYPE,
    NO_CYCLIZATION_INDEX,
    NON_RING_COND_TYPES,
    NUM_CYCLIZATION_COND_TYPES,
    NUM_CYCLIZATION_TYPES,
    UNSPECIFIED,
    is_ring_request,
)
from proteinfoundation.cyclization.head import CyclizationLinkHead
from proteinfoundation.cyclization.loss import cyclization_link_loss, decode_cyclization_prediction
from proteinfoundation.cyclization.mask import build_cyclization_validity_mask

__all__ = [
    "MAINCHAIN",
    "DISULFIDE",
    "ISOPEPTIDE",
    "UNSPECIFIED",
    "LINEAR",
    "NON_RING_COND_TYPES",
    "is_ring_request",
    "NUM_CYCLIZATION_TYPES",
    "NUM_CYCLIZATION_COND_TYPES",
    "CYCLIZATION_TYPE_TO_NAME",
    "NAME_TO_CYCLIZATION_TYPE",
    "COND_TYPE_TO_NAME",
    "NAME_TO_COND_TYPE",
    "NO_CYCLIZATION_INDEX",
    "CyclizationLinkHead",
    "build_cyclization_validity_mask",
    "cyclization_link_loss",
    "decode_cyclization_prediction",
]
