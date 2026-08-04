"""Cyclic sequence-offset patch for refolding macrocyclic binders.

AF2 (via ColabDesign) has no notion of a covalent bond between the two ends of a
chain. It reads topology out of ``residue_index``, and everything downstream --
the relative positional encoding, and therefore pAE and pLDDT -- follows from the
pairwise offset ``idx[i] - idx[j]``. For a head-to-tail 12-mer that offset says
residues 0 and 11 are **11 apart**, so AF2 refolds and scores a *linear* peptide.

That is the instrument problem: ``i_pAE``, ``pLDDT`` and ``scRMSD`` for a generated
macrocycle are currently computed on a molecule that is not the one that was
generated -- the ring is silently cut before scoring. A design can close perfectly
and still be scored as the floppy linear peptide it is not.

This module wraps the offset around the ring for the binder block only, so the
termini read as separation +/-1 and AF2 folds the macrocycle. It is the same
construction used for cyclic-peptide prediction in ColabDesign.

**Only chemically honest for head-to-tail.** The wrap asserts that residues 0 and
L-1 are *backbone* neighbours, which is true exactly when the closing bond is the
peptide bond itself. A disulfide or isopeptide ring closes through **side chains**;
its backbone termini are not bonded, and wrapping the backbone offset would state
something false about the molecule -- trading one misdescription for another.
So those types are deliberately left linear, and
`cyclic_offset_supported` reports that, rather than quietly patching everything.
The residual limitation for those types (their crosslink is still invisible to the
refold) is real and is reported, not hidden.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# The wrap is only valid where the closing bond IS the backbone peptide bond.
CYCLIC_OFFSET_SUPPORTED_TYPES = frozenset({"mainchain"})


def cyclic_offset_supported(linkage_type: str | None) -> bool:
    """Whether wrapping the backbone offset is chemically valid for this linkage."""
    return linkage_type in CYCLIC_OFFSET_SUPPORTED_TYPES


def cyclic_offset_block(binder_len: int) -> np.ndarray:
    """Signed offset matrix on a ring of `binder_len` residues, shape [L, L].

    Entry ``[i, j]`` is the signed shortest displacement from j to i walking around
    the ring, i.e. the linear offset ``i - j`` replaced by the short way round
    whenever that is strictly shorter.

    Ties (exactly antipodal residues, only possible when `binder_len` is even) are
    resolved deterministically **in favour of the linear value**: the condition is a
    strict ``>``, so at ``|off| == L // 2`` both directions are equally short and the
    sign of the linear offset is kept. Deterministic matters more than which branch
    wins -- an arbitrary tie-break that flipped between calls would make the feature
    non-reproducible for even-length rings.
    """
    if binder_len < 0:
        raise ValueError(f"binder_len must be non-negative, got {binder_len}")
    idx = np.arange(binder_len)
    off = idx[:, None] - idx[None, :]
    abs_off = np.abs(off)
    # Strict '>' keeps the antipodal tie on the linear branch; see docstring.
    return np.where(abs_off > binder_len // 2, -np.sign(off) * (binder_len - abs_off), off)


def build_complex_cyclic_offset(residue_index, binder_len: int) -> np.ndarray:
    """Offset matrix for a target+binder complex with the binder block made cyclic.

    ColabDesign's binder protocol lays the complex out as ``[target, binder]``, so the
    binder occupies the trailing `binder_len` rows/columns. Only that diagonal block
    is wrapped: target-target and target-binder offsets are left exactly as they were.

    **Must be built from the complex's real `residue_index`, not a fresh
    ``arange(total_len)``.** ColabDesign's `prep_pdb`/`prep_inputs` deliberately puts a
    large gap in `residue_index` at the target/binder chain boundary (e.g. binder start
    = target's last index + 50) precisely so AF2's relative positional encoding reads
    the two chains as unrelated rather than sequence-adjacent. Rebuilding the whole
    matrix from a contiguous `arange` silently collapses that gap back down to the
    ordinary linear separation (e.g. an intended boundary offset of -50 becomes -1),
    which tells AF2 the receptor and binder are backbone neighbours -- the exact
    instrument problem this module exists to avoid, just relocated to the wrong chain
    boundary instead of the binder's own termini. `residue_index` here should be
    whatever `inputs["residue_index"]` already is for the prepared complex.

    Args:
        residue_index: The complex's real per-residue index, array-like of length
            `total_len`. For convenience (e.g. tests that don't need a real chain-break
            gap), a plain int is also accepted and treated as a contiguous
            ``arange(int)`` -- but production callers must pass the actual prepared
            array, never a bare length.
        binder_len: Number of trailing binder residues to wrap into a ring.
    """
    if np.isscalar(residue_index):
        residue_index = np.arange(int(residue_index))
    residue_index = np.asarray(residue_index).reshape(-1)
    total_len = residue_index.shape[0]
    if binder_len > total_len:
        raise ValueError(f"binder_len ({binder_len}) exceeds total_len ({total_len})")
    offset = residue_index[:, None] - residue_index[None, :]
    if binder_len > 0:
        offset[-binder_len:, -binder_len:] = cyclic_offset_block(binder_len)
    return offset


def apply_cyclic_offset(af_model, binder_len: int, linkage_type: str | None) -> bool:
    """Patch a prepared ColabDesign model so the binder folds as a macrocycle.

    Call **after** ``prep_inputs`` and **before** ``run``: ``prep_inputs`` is what
    populates ``_inputs``, and ``run`` is what consumes the offset.

    Args:
        af_model: A prepared ColabDesign model (``mk_afdesign_model``).
        binder_len: Number of binder residues (trailing block of the complex).
        linkage_type: Requested linkage; the patch is skipped unless it is one the
            wrap is chemically valid for (see module docstring).

    Returns:
        True if the offset was patched, False if it was left linear. A False return
        is a normal, expected outcome -- callers should record it so the resulting
        metrics are not mistaken for cyclic-aware ones.
    """
    if not cyclic_offset_supported(linkage_type):
        logger.info(
            "Cyclic offset NOT applied (linkage_type=%s): the wrap is only valid when the "
            "closing bond is the backbone peptide bond. Refold metrics for this design "
            "describe a chain without its side-chain crosslink.",
            linkage_type,
        )
        return False

    if binder_len <= 2:
        logger.warning("Cyclic offset skipped: binder_len=%d is too short to form a ring.", binder_len)
        return False

    inputs = getattr(af_model, "_inputs", None)
    if inputs is None or "residue_index" not in inputs:
        logger.warning("Cyclic offset skipped: model has no prepared inputs (call prep_inputs first).")
        return False

    residue_index = np.asarray(inputs["residue_index"]).reshape(-1)
    total_len = int(residue_index.shape[0])
    if binder_len > total_len:
        logger.warning(
            "Cyclic offset skipped: binder_len=%d exceeds prepared complex length %d.", binder_len, total_len
        )
        return False

    # Pass the model's own `residue_index`, NOT `total_len` -- see
    # `build_complex_cyclic_offset`'s docstring. Rebuilding from a bare length would
    # silently erase whatever target/binder chain-break gap `prep_inputs` already
    # encoded there.
    inputs["offset"] = build_complex_cyclic_offset(residue_index, binder_len)
    logger.info("Applied cyclic offset to binder block (binder_len=%d, total_len=%d).", binder_len, total_len)
    return True
