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
# *input* (see `cyclization.type_conditioning`), it needs a value meaning "no
# type requested", which the head's 3-way output space cannot express. This
# extra index is only ever an embedding input; `NUM_CYCLIZATION_TYPES` stays 3
# so the head's output width -- and every existing checkpoint -- is untouched.
UNSPECIFIED = 3
NUM_CYCLIZATION_COND_TYPES = 4

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
