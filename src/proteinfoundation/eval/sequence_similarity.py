"""BLOSUM scoring for generated sequences, and the per-position records behind it.

WHY EXACT-MATCH RECOVERY IS THE WRONG PRIMARY SCORE HERE. This project already measured the
entropy floor of the task: on 2.44M train peptides the best lookup table reaches 2.557 nats
and the whole learnable span above it is 0.63 nats, because residue identity in a 5-16-mer is
*substitutable* -- several residues do the same job at the same site. Exact match charges a
full miss for ILE where the native has LEU, which is not a design error. BLOSUM62 scores the
substitution instead of the symbol, so a conservative swap is credited as one.

It does NOT replace exact match, it brackets it:

    free_recovery            exact symbol agreement -- the strict floor
    free_blosum_mean         mean BLOSUM62 over free positions, conservative swaps credited
    free_conservative_frac   fraction scoring >= 0, i.e. "a substitution matrix would accept it"
    free_blosum_mean_miss    mean BLOSUM62 over free positions that exact match got WRONG

The last one is the load-bearing one and the reason this module exists. A model whose misses
are conservative and a model whose misses are random have the SAME `free_recovery`; only the
score on the misses separates them. Read it against the null in `blosum_null_expectation`,
never on its own -- BLOSUM62's off-diagonal mean is negative, so "mean score on misses is
negative" is the expected case and not a finding.

TWO TRAPS.

**The matrix must be reindexed, not assumed.** biopython's BLOSUM62 carries its own alphabet
(with B, Z, X, * and a different letter order). `residue_type` here is OpenFold
`residue_constants.restypes` order -- the alphabet `cpseq/seq_inject.py` verifies rather than
assumes, and which is reused from there so there is one verified copy and not two.

**A raw mean BLOSUM is not comparable across residue compositions.** Self-scores range from
4 (LEU) to 11 (TRP), so a peptide rich in TRP scores higher while predicting nothing better.
Any comparison across cells must either hold the composition fixed -- which the paired cells
here do, since every cell scores the same complexes -- or use the null expectation.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from proteinfoundation.cpseq.seq_inject import _AA_BY_INDEX


@functools.lru_cache(maxsize=1)
def blosum62_by_restype_index() -> np.ndarray:
    """BLOSUM62 as a [20, 20] float array indexed by OpenFold `restypes` order.

    Raises rather than falling back to a hand-copied matrix: a silently wrong substitution
    matrix would shift every score by a plausible-looking amount.
    """
    from Bio.Align import substitution_matrices

    raw = substitution_matrices.load("BLOSUM62")
    letters = list(raw.alphabet)
    out = np.zeros((20, 20), dtype=np.float32)
    for i, a in enumerate(_AA_BY_INDEX):
        for j, b in enumerate(_AA_BY_INDEX):
            out[i, j] = float(raw[letters.index(a), letters.index(b)])
    return out


def blosum_metrics(
    pred_aatype: torch.Tensor,
    true_aatype: torch.Tensor,
    mask: torch.Tensor,
    cond_mask: torch.Tensor,
    prefix: str = "val_gen/seq",
) -> dict[str, float]:
    """BLOSUM62 aggregates over the free and conditioned populations.

    Same NaN convention as `sequence_recovery_metrics`: a rate or mean over an empty
    population is NaN, never 0.0 -- 0.0 is a real BLOSUM score (e.g. ALA/THR), so a zero
    standing for "nothing to measure" would be indistinguishable from a measurement.
    """
    mat = torch.as_tensor(blosum62_by_restype_index(), device=pred_aatype.device)

    mask = mask.bool()
    cond_mask = cond_mask.bool() & mask
    free_mask = mask & ~cond_mask

    # Clamp only to keep indexing safe; padded slots are masked out below either way.
    p = pred_aatype.long().clamp_(0, 19)
    t = true_aatype.long().clamp_(0, 19)
    score = mat[p.reshape(-1), t.reshape(-1)].reshape(p.shape)
    correct = pred_aatype == true_aatype

    def _mean(sel: torch.Tensor) -> float:
        n = int(sel.sum().item())
        if n == 0:
            return float("nan")
        return float(score[sel].sum().item()) / n

    def _frac(sel: torch.Tensor, cond: torch.Tensor) -> float:
        n = int(sel.sum().item())
        if n == 0:
            return float("nan")
        return float((cond & sel).sum().item()) / n

    free_miss = free_mask & ~correct
    out = {
        f"{prefix}/free_blosum_mean": _mean(free_mask),
        f"{prefix}/free_blosum_mean_miss": _mean(free_miss),
        f"{prefix}/free_conservative_frac": _frac(free_mask, score >= 0),
        f"{prefix}/free_positive_frac": _frac(free_mask, score > 0),
        f"{prefix}/cond_blosum_mean": _mean(cond_mask),
        f"{prefix}/n_free_miss": float(int(free_miss.sum().item())),
    }
    return out


def blosum_null_expectation(true_aatype: np.ndarray, composition: np.ndarray) -> float:
    """E[BLOSUM62(r, true)] with r drawn from `composition`, averaged over the truths given.

    The null a measured `free_blosum_mean` has to beat. Computed from the SAME truths being
    scored, so it carries their composition rather than a corpus-average one.

    Args:
        true_aatype: 1-D array of true residue indices (restypes order).
        composition: length-20 probability vector to draw the guess from.
    """
    mat = blosum62_by_restype_index()
    if true_aatype.size == 0:
        return float("nan")
    comp = np.asarray(composition, dtype=np.float64)
    comp = comp / comp.sum()
    return float(np.mean(mat[:, true_aatype].T @ comp))


def per_position_records(
    pred_aatype: torch.Tensor,
    true_aatype: torch.Tensor,
    mask: torch.Tensor,
    cond_mask: torch.Tensor,
    cyc_type: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Flatten one batch to one record per real residue, for offline analysis.

    This is what makes positional recovery, BLOSUM and the `P(r | pos, L, chemistry)` prior
    computable from disk without re-sampling. Aggregate rates cannot be decomposed after the
    fact -- a pooled 0.37 does not say whether the terminal anchor positions, which a lookup
    gets for free, are carrying it.

    `pos` is the index within the binder and `length` its real residue count, so (pos, length)
    keys the prior directly. Padding is dropped rather than flagged.
    """
    mask = mask.bool()
    cond_mask = cond_mask.bool() & mask
    b, n = mask.shape

    lengths = mask.sum(dim=-1)  # [b]
    pos = torch.arange(n, device=mask.device)[None, :].expand(b, n)
    length_b = lengths[:, None].expand(b, n)
    if cyc_type is None:
        cyc = torch.full((b, n), -1, dtype=torch.long, device=mask.device)
    else:
        c = cyc_type.reshape(-1).long()
        cyc = c[:, None].expand(b, n)

    sel = mask
    rec = {
        "pos": pos[sel],
        "length": length_b[sel],
        "cyc_type": cyc[sel],
        "true_aa": true_aatype.long()[sel],
        "pred_aa": pred_aatype.long()[sel],
        "revealed": cond_mask[sel],
    }
    return {k: v.detach().cpu().numpy() for k, v in rec.items()}
