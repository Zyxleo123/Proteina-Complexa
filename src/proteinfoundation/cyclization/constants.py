"""Constants for the CPSea cyclization-linkage prediction head.

Linkage types describe how the two ends of a cyclic peptide binder are
chemically joined. This is a *classification* target only: we predict which
single `(i, j, type)` triple describes the cyclization edge of the binder.
Nothing here enforces geometry or modifies coordinates.
"""

from openfold.np import residue_constants

MAINCHAIN = 0
DISULFIDE = 1
ISOPEPTIDE = 2
NUM_CYCLIZATION_TYPES = 3

CYCLIZATION_TYPE_TO_NAME = {
    MAINCHAIN: "mainchain",
    DISULFIDE: "disulfide",
    ISOPEPTIDE: "isopeptide",
}
NAME_TO_CYCLIZATION_TYPE = {v: k for k, v in CYCLIZATION_TYPE_TO_NAME.items()}

# Sentinel used when a sample has no (or an unparsable) cyclization label.
NO_CYCLIZATION_INDEX = -1

# Conditioning vocabulary. When the desired cyclization type is given as an
# *input* (see `cyclization.type_conditioning`), it needs values the head's 3-way
# output space cannot express. These extra indices are only ever embedding inputs;
# `NUM_CYCLIZATION_TYPES` stays 3 so the head's output width -- and every existing
# checkpoint -- is untouched.
#
# UNSPECIFIED means "no type requested": the classifier-free-guidance null, and what
# `cyclization_type_dropout_rate` rewrites rows to.
#
# LINEAR means the opposite of a request for "any ring": an explicit request for *no*
# ring at all. It exists so linear-peptide training data (PepBench / ProtFrag) can be
# mixed into a cyclic-peptide run without being labeled UNSPECIFIED -- if it were, the
# CFG null branch would come to mean "linear peptide" rather than "any peptide", which
# silently changes what `guidance_w > 1` pushes away from. It is also what you request
# at generation time to sample a linear binder in-distribution.
#
# Neither index is a ring request. Anything gating on "is a cycle being asked for"
# must use `is_ring_request`, never `!= UNSPECIFIED`.
UNSPECIFIED = 3
LINEAR = 4
NUM_CYCLIZATION_COND_TYPES = 5

# Conditioning indices that are NOT a request for a specific ring chemistry. Both are
# excluded from the cycle-graph pair features and from the head's candidate-set
# restriction; see `is_ring_request`.
NON_RING_COND_TYPES = (UNSPECIFIED, LINEAR)

COND_TYPE_TO_NAME = dict(CYCLIZATION_TYPE_TO_NAME) | {
    UNSPECIFIED: "unspecified",
    LINEAR: "linear",
}
NAME_TO_COND_TYPE = {v: k for k, v in COND_TYPE_TO_NAME.items()}


def is_ring_request(cond_type):
    """[B] bool: does this row actually ask for a cyclization edge?

    True only for a concrete chemistry (MAINCHAIN / DISULFIDE / ISOPEPTIDE). Both
    UNSPECIFIED ("any") and LINEAR ("none") are False, so the ring positional encoding
    and the typed pair channels switch off for them together -- a linear row must not be
    handed a cycle graph, and an unconditional row must not be told where the ring is.

    Args:
        cond_type: integer tensor of conditioning indices, any shape.

    Returns:
        Boolean tensor of the same shape.
    """
    return (cond_type != UNSPECIFIED) & (cond_type != LINEAR)

# Amino-acid integer ids, using the project's canonical ordering
# (`openfold.np.residue_constants.restype_order`, alphabetical one-letter
# codes 0..19). This matches `residue_type` / `aatype` tensors used
# throughout the flow model and CPSea dataset pipeline.
AA_CYS = residue_constants.restype_order["C"]
AA_LYS = residue_constants.restype_order["K"]
AA_ASP = residue_constants.restype_order["D"]
AA_GLU = residue_constants.restype_order["E"]
AA_ASN = residue_constants.restype_order["N"]
AA_GLN = residue_constants.restype_order["Q"]

# Isopeptide acid/amide partner atom name, by residue three-letter code. CONECT
# records in CPSea bond to the side-chain carbonyl carbon (CG for Asp/Asn, CD for
# Glu/Gln), never the oxygens -- see `cyclization.parse_labels`, the single source
# of truth for which real bonds exist in the dataset. ASN/GLN are included here
# because they ARE accepted at label-parsing time; every downstream consumer of an
# isopeptide `(i, j, type)` triple (geometry loss, strict reconstruction metrics,
# generation-time closure scoring) must recognize the same four residues, or a
# sample labeled `has_cyclization=True` at parse time silently gets zero geometric
# supervision / a silently-`None` observed chemistry downstream.
ISOPEPTIDE_ACID_CARBON_ATOM = {
    "ASP": "CG",
    "ASN": "CG",
    "GLU": "CD",
    "GLN": "CD",
}
