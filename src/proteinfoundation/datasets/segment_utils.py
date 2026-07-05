"""Continuous-segment detection for (potentially discontinuous) polymer chains.

Some datasets (e.g. CPSea) provide receptor/target chains that are, in
reality, receptor/pocket *crops*: a single PDB chain letter can be made up of
several physically disconnected stretches of backbone. Treating such a chain
as one continuous polymer corrupts sequence-order-dependent features
(relative sequence separation, chain identity, backbone torsions), since two
residues that are adjacent in tensor/file order need not be adjacent in the
actual protein backbone.

This module detects the underlying continuous segments, in file/tensor
order, so that feature construction can treat each segment as its own
chain-like object. See ``scripts/check_pdb_residue_jumps.py`` for the
equivalent standalone PDB-level diagnostic (``REAL_BREAK`` /
``NUMBERING_ONLY`` / ``CLEAN`` classification), which this module mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from openfold.np.residue_constants import atom_order

N_ATOM_IDX = atom_order["N"]
C_ATOM_IDX = atom_order["C"]

# Break classification labels for the boundary immediately *before* a residue.
BREAK_NONE = "NONE"  # first residue overall, no previous residue to compare against
BREAK_NEW_CHAIN = "NEW_CHAIN"  # different chain than the previous residue
BREAK_CLEAN = "CLEAN"  # consecutive numbering and a normal peptide bond
BREAK_NUMBERING_ONLY = "NUMBERING_ONLY"  # residue-number jump, but bond geometry is normal
BREAK_REAL_BREAK = "REAL_BREAK"  # missing or too-long C_i -> N_{i+1}

DEFAULT_CN_BREAK_CUTOFF = 2.0  # Angstrom


@dataclass
class SegmentInfo:
    """Per-residue segment metadata for a single structure (un-batched)."""

    segment_id: torch.Tensor  # [n] long, 0-indexed continuous-segment index *within its chain*
    pos_in_segment: torch.Tensor  # [n] long, 0-indexed position of the residue within its segment
    effective_chain_id: torch.Tensor  # [n] long, unique per (chain_id, segment_id) pair
    break_type: list  # length n, break_type[i] classifies the boundary before residue i (break_type[0] == BREAK_NONE)


def compute_segment_info(
    chains: torch.Tensor,
    residue_pdb_idx: torch.Tensor,
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    cn_break_cutoff: float = DEFAULT_CN_BREAK_CUTOFF,
) -> SegmentInfo:
    """Detects continuous backbone segments for every residue, in file/tensor order.

    For each chain, a new segment starts at residue ``i`` (``i > 0``, within
    the same chain as residue ``i - 1``) whenever, relative to residue
    ``i - 1``:
        - the PDB residue number does not increase by exactly 1, or
        - the previous residue's C atom or this residue's N atom is missing, or
        - the C_{i-1} -> N_i distance exceeds ``cn_break_cutoff`` Angstrom.

    A chain change (``chains[i] != chains[i - 1]``) always starts a new
    segment as well (segments live within a single chain).

    Args:
        chains: [n] integer per-residue chain index (e.g. ``Data.chains``), in file order.
        residue_pdb_idx: [n] integer original PDB residue numbers, in file order.
        coords: [n, 37, 3] atom37 coordinates, in Angstrom (not nanometers).
        coord_mask: [n, 37] boolean atom presence mask.
        cn_break_cutoff: distance (Angstrom) above which a C_i -> N_{i+1} bond
            is considered physically broken. Default 2.0 (see
            ``docs/README_DATA_SEQ_BREAK_ISSUE.md``: a normal peptide bond is
            ~1.3 Å).

    Returns:
        SegmentInfo with per-residue tensors, on the same device as ``chains``.
    """
    n_res = int(chains.shape[0])
    device = chains.device

    segment_id = torch.zeros(n_res, dtype=torch.long, device=device)
    pos_in_segment = torch.zeros(n_res, dtype=torch.long, device=device)
    effective_chain_id = torch.zeros(n_res, dtype=torch.long, device=device)
    break_type: list = [BREAK_NONE] * n_res

    if n_res == 0:
        return SegmentInfo(segment_id, pos_in_segment, effective_chain_id, break_type)

    chains_list = chains.tolist()
    residue_pdb_idx_list = residue_pdb_idx.tolist()

    cur_chain_segment = 0
    cur_pos_in_segment = 0
    cur_effective_id = 0

    for i in range(1, n_res):
        if chains_list[i] != chains_list[i - 1]:
            btype = BREAK_NEW_CHAIN
            cur_chain_segment = 0
            cur_pos_in_segment = 0
            cur_effective_id += 1
        else:
            numbering_jump = residue_pdb_idx_list[i] != residue_pdb_idx_list[i - 1] + 1
            c_prev_ok = bool(coord_mask[i - 1, C_ATOM_IDX])
            n_curr_ok = bool(coord_mask[i, N_ATOM_IDX])
            if not c_prev_ok or not n_curr_ok:
                physical_break = True
            else:
                cn_dist = torch.norm(coords[i - 1, C_ATOM_IDX] - coords[i, N_ATOM_IDX]).item()
                physical_break = cn_dist > cn_break_cutoff

            if physical_break:
                btype = BREAK_REAL_BREAK
            elif numbering_jump:
                btype = BREAK_NUMBERING_ONLY
            else:
                btype = BREAK_CLEAN

            if physical_break or numbering_jump:
                cur_chain_segment += 1
                cur_pos_in_segment = 0
                cur_effective_id += 1
            else:
                cur_pos_in_segment += 1

        break_type[i] = btype
        segment_id[i] = cur_chain_segment
        pos_in_segment[i] = cur_pos_in_segment
        effective_chain_id[i] = cur_effective_id

    return SegmentInfo(segment_id, pos_in_segment, effective_chain_id, break_type)


def summarize_breaks(break_type: list, chains: torch.Tensor | None = None, chain_id_of_interest: int | None = None):
    """Counts break types, optionally restricted to a single chain.

    Mirrors the summary counters in ``scripts/check_pdb_residue_jumps.py``
    (``REAL_BREAK`` / ``NUMBERING_ONLY``), useful for validation/tests, e.g.
    asserting a binder chain has zero real/numbering breaks.

    Args:
        break_type: list of per-residue break classifications, as returned by
            ``compute_segment_info``.
        chains: optional [n] per-residue chain index, required if
            ``chain_id_of_interest`` is given.
        chain_id_of_interest: if given, only count breaks occurring at a
            residue belonging to this chain (i.e. the break *into* this
            chain's residue, which is what matters for "is this chain
            contiguous").

    Returns:
        dict with counts for each break type present.
    """
    from collections import Counter

    if chain_id_of_interest is not None:
        assert chains is not None, "Must pass `chains` when filtering by chain_id_of_interest"
        chains_list = chains.tolist()
        filtered = [bt for bt, c in zip(break_type, chains_list, strict=True) if c == chain_id_of_interest]
        return dict(Counter(filtered))
    return dict(Counter(break_type))
