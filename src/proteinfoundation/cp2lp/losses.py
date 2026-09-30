"""Generator-side losses for CP -> LP.

Four jobs beyond the adversarial term:

contact retention
    the generated LP must keep the source CP's *important* peptide-target contacts.
    One-sided by construction -- losing a contact is penalised, gaining one is free --
    because the LP is allowed to sit differently in the pocket, it is only not allowed
    to fall out of it.

sequence retention
    the LP must be the same sequence as the CP. Supervised with cross-entropy on the
    decoded logits so it is differentiable, and enforced again as a hard gate at export.

open-chain geometry
    peptide-bond and intra-residue bond lengths, scored WITH THE TERMINAL BOND ABSENT.
    Including the ``(L-1, 0)`` closure would tell the generator that the ring it was
    asked to open is a defect to repair, which is the opposite of the objective.

clash
    steric overlap inside the peptide and against the pocket. The threshold is 0.20 nm,
    not the 0.27 nm that reads as the "obvious" hard-sphere value: 0.27 nm rejects 43 of
    60 native crystal complexes, so a generator trained against it is being pushed away
    from native-like packing rather than towards it.

All terms are flat-bottomed where a flat bottom makes sense: once a quantity is inside
its tolerance the gradient is exactly zero, so the generator cannot buy adversarial slack
by over-optimising a geometry term past the point where it means anything.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from openfold.np.residue_constants import restype_num

from proteinfoundation.cp2lp.geometry import (
    heavy_atom_distances,
    intra_residue_bond_deviation_nm,
    pairwise_distances,
    safe_norm,
    peptide_bond_deviation_nm,
    residue_lengths,
    soft_contact,
    terminal_gap_nm,
)

#: Calibrated against native crystal complexes -- see the module docstring.
CLASH_THRESHOLD_NM = 0.20

#: Below this CA(0)-CA(L-1) separation an "open" peptide is not meaningfully open. Set
#: from the mainchain closing-bond window: a ring counts as closed near 0.13 nm, and
#: anything under 0.45 nm is still stacked end-on-end rather than extended.
MIN_OPEN_TERMINAL_GAP_NM = 0.45


def contact_retention_loss(
    gen_atom37: torch.Tensor,
    gen_atom_mask: torch.Tensor,
    src_atom37: torch.Tensor,
    src_atom_mask: torch.Tensor,
    pep_mask: torch.Tensor,
    target_atom37: torch.Tensor,
    target_atom_mask: torch.Tensor,
    target_mask: torch.Tensor,
    cutoff_nm: float = 0.8,
    sharpness_nm: float = 0.1,
    min_strength: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalises source-CP peptide-target contacts that the generated LP has lost.

    Contacts are soft (a sigmoid of the min heavy-atom distance) on both sides, so the
    loss has gradient through the decision boundary. A hard ``d < cutoff`` indicator
    would be exactly flat and would teach the generator nothing about a contact that
    drifted to 0.9 nm.

    Each source contact is weighted by its own strength, so the term is dominated by the
    contacts that actually define the binding mode rather than by the long tail of
    marginal ones. ``min_strength`` drops that tail entirely.

    Returns ``(loss, metrics)``. Loss is a scalar; metrics are per-batch diagnostics.
    """
    pm = pep_mask.bool()
    tm = target_mask.bool()

    d_src = heavy_atom_distances(src_atom37, src_atom_mask, target_atom37, target_atom_mask)
    d_gen = heavy_atom_distances(gen_atom37, gen_atom_mask, target_atom37, target_atom_mask)

    pair_mask = (pm[:, :, None] & tm[:, None, :]).float()
    c_src = soft_contact(d_src, cutoff_nm, sharpness_nm) * pair_mask
    c_gen = soft_contact(d_gen, cutoff_nm, sharpness_nm) * pair_mask

    weight = torch.where(c_src >= min_strength, c_src, torch.zeros_like(c_src)).detach()
    lost = F.relu(c_src.detach() - c_gen)  # one-sided: only losses are penalised
    denom = weight.sum(dim=(1, 2)).clamp(min=1e-6)
    per_example = (weight * lost).sum(dim=(1, 2)) / denom

    with torch.no_grad():
        n_src = (c_src >= min_strength).float().sum(dim=(1, 2))
        n_kept = ((c_src >= min_strength) & (c_gen >= min_strength)).float().sum(dim=(1, 2))
        retention = n_kept / n_src.clamp(min=1.0)

    metrics = {
        "contact_retention_frac": retention.mean(),
        "contact_n_source": n_src.mean(),
        "contact_n_kept": n_kept.mean(),
    }
    return per_example.mean(), metrics


def sequence_weight_scale(
    exact_match_ema: float,
    start: float = 0.9,
    floor: float = 0.05,
) -> float:
    """Multiplier that retires the sequence term once it has essentially converged.

    MEASURED (smoke 68421, gradient attribution into the flow network): the sequence term
    contributes ``max|g| = 2.40`` against contact retention's ``2.30e-02`` -- a hundredfold
    difference, with contact carrying the LARGER weight. Because the generator gradient is
    clipped to a fixed norm, the terms compete for one budget rather than summing freely,
    so the strongest term sets how much of a step everything else gets.

    Sequence identity is solved early and stays solved (exact match 0.93-0.96 by 39k steps
    in both arms), yet it keeps consuming that budget. Retiring it is what frees the step
    for contact retention -- in the contact-only arm retention only began to move AFTER
    exact match saturated (0.150 at 10-12k -> 0.435 at 37-39k), which is the same effect
    arriving by accident.

    Linear in the remaining error so the hand-off is gradual rather than a cliff: 1.0 at
    ``start``, 0.5 at 0.95, 0.1 at 0.99, ``floor`` at 1.0. Never reaches zero -- the term
    still has to DEFEND the sequence it won, and identity is a premise of the export gate.
    """
    if not (0.0 <= start < 1.0):
        raise ValueError(f"start must be in [0, 1), got {start}")
    if exact_match_ema <= start:
        return 1.0
    return max(floor, (1.0 - exact_match_ema) / (1.0 - start))


def sequence_retention_loss(
    seq_logits: torch.Tensor,
    target_aatype: torch.Tensor,
    pep_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-entropy pinning the decoded LP sequence to the source CP sequence.

    The generator is only allowed to move geometry and latents; the sequence is held
    fixed by construction of the task. This is the differentiable half of that
    constraint -- the hard half is the export gate, which rejects any sample whose
    argmax sequence does not match.

    Args:
        seq_logits: ``[b, n, C]`` decoder logits. ``C`` is 20 or 21.
        target_aatype: ``[b, n]`` source CP residue types.
        pep_mask: ``[b, n]``.
    """
    pm = pep_mask.bool()
    n_classes = seq_logits.shape[-1]
    tgt = target_aatype.long().clamp(0, n_classes - 1)
    ce = F.cross_entropy(
        seq_logits.reshape(-1, n_classes),
        tgt.reshape(-1),
        reduction="none",
    ).reshape(pm.shape)
    loss = (ce * pm.float()).sum() / pm.float().sum().clamp(min=1.0)

    with torch.no_grad():
        pred = seq_logits.argmax(dim=-1)
        correct = ((pred == tgt) & pm).float().sum(dim=1)
        total = pm.float().sum(dim=1).clamp(min=1.0)
        per_res = (correct / total)
        exact = (correct == total).float()

    return loss, {"seq_identity": per_res.mean(), "seq_exact_match": exact.mean()}


def open_chain_geometry_loss(
    atom37: torch.Tensor,
    atom_mask: torch.Tensor,
    pep_mask: torch.Tensor,
    bond_tol_nm: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backbone bond-length penalty for an OPEN chain, flat-bottomed at ``bond_tol_nm``.

    Only ``(i, i+1)`` peptide bonds and intra-residue N-CA / CA-C bonds are scored. The
    ``(L-1, 0)`` closure has no term -- see :mod:`proteinfoundation.cp2lp.geometry`.
    """
    pm = pep_mask.bool()
    bond_dev, bond_pair_mask = peptide_bond_deviation_nm(atom37, pm)
    intra_dev, _ = intra_residue_bond_deviation_nm(atom37, pm)

    bond_pen = F.relu(bond_dev - bond_tol_nm) * bond_pair_mask
    intra_pen = F.relu(intra_dev - bond_tol_nm) * pm[..., None].float()

    n_bond = bond_pair_mask.float().sum().clamp(min=1.0)
    n_intra = (pm.float().sum() * 2).clamp(min=1.0)
    loss = bond_pen.sum() / n_bond + intra_pen.sum() / n_intra

    with torch.no_grad():
        ok = ((bond_dev <= bond_tol_nm) | ~bond_pair_mask).all(dim=1).float()
        metrics = {
            "geom_peptide_bond_mae_nm": (bond_dev.sum() / n_bond),
            "geom_chain_intact_frac": ok.mean(),
        }
    return loss, metrics


def clash_loss(
    atom37: torch.Tensor,
    atom_mask: torch.Tensor,
    pep_mask: torch.Tensor,
    target_atom37: torch.Tensor | None = None,
    target_atom_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    threshold_nm: float = CLASH_THRESHOLD_NM,
    min_seq_sep: int = 2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Steric overlap inside the peptide and against the pocket.

    Intra-peptide pairs closer than ``min_seq_sep`` in sequence are excluded: residues
    ``i`` and ``i+1`` share a peptide bond and are *supposed* to have atoms at bonding
    distance, so penalising them would fight the geometry term.
    """
    pm = pep_mask.bool()
    b, n = pm.shape

    d_self = heavy_atom_distances(atom37, atom_mask, atom37, atom_mask)  # [b, n, n]
    sep = (torch.arange(n, device=d_self.device)[None, :] - torch.arange(n, device=d_self.device)[:, None]).abs()
    self_pair = (sep >= min_seq_sep)[None].expand(b, n, n) & pm[:, :, None] & pm[:, None, :]
    intra = (F.relu(threshold_nm - d_self) * self_pair.float()).sum() / 2.0
    n_intra_pairs = self_pair.float().sum().clamp(min=1.0) / 2.0

    total = intra / n_intra_pairs
    metrics = {"clash_intra_sum_nm": intra.detach()}

    if target_atom37 is not None:
        tm = target_mask.bool()
        d_t = heavy_atom_distances(atom37, atom_mask, target_atom37, target_atom_mask)
        t_pair = pm[:, :, None] & tm[:, None, :]
        inter = (F.relu(threshold_nm - d_t) * t_pair.float()).sum()
        n_inter_pairs = t_pair.float().sum().clamp(min=1.0)
        total = total + inter / n_inter_pairs
        metrics["clash_inter_sum_nm"] = inter.detach()
        with torch.no_grad():
            metrics["clash_min_inter_nm"] = torch.where(t_pair, d_t, torch.full_like(d_t, 1e4)).amin()

    return total, metrics


def terminal_opening_reward(
    atom37: torch.Tensor,
    pep_mask: torch.Tensor,
    min_gap_nm: float = MIN_OPEN_TERMINAL_GAP_NM,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One-sided hinge pushing the termini apart until they are genuinely open.

    This is NOT a target separation -- there is no term pulling a well-separated pair of
    termini to any particular distance, and the gradient is exactly zero once the gap
    clears ``min_gap_nm``. It exists only to escape the failure mode where the generator
    satisfies every other term by returning the source CP unchanged, which trivially
    retains all contacts and all sequence.

    How far apart real bound LP termini actually sit is a question for the adversarial
    term, which has the real distribution to compare against; hard-coding a target here
    would override it with a guess.
    """
    gap = terminal_gap_nm(atom37, pep_mask)
    loss = F.relu(min_gap_nm - gap).mean()
    with torch.no_grad():
        metrics = {
            "terminal_gap_nm": gap.mean(),
            "terminal_open_frac": (gap >= min_gap_nm).float().mean(),
        }
    return loss, metrics


def linear_terminus_valid(
    atom37: torch.Tensor,
    pep_mask: torch.Tensor,
    min_gap_nm: float = MIN_OPEN_TERMINAL_GAP_NM,
    bond_window_nm: float = 0.20,
) -> torch.Tensor:
    """[b] bool -- is this peptide chemically a LINEAR chain?

    Two conditions, both necessary:

    * the C(L-1)-N(0) distance is outside anything that could be read as a peptide bond,
      so no structure-parsing tool will infer a cyclic connection;
    * the CA(0)-CA(L-1) separation clears ``min_gap_nm``, so the chain is open in fact
      and not merely missing one bond while sitting in a ring conformation.

    The export gate uses this; nothing in training does.
    """
    from proteinfoundation.cp2lp.geometry import C_IDX, N_IDX

    pm = pep_mask.bool()
    lengths = residue_lengths(pm)
    idx = torch.arange(atom37.shape[0], device=atom37.device)
    c_last = atom37[idx, (lengths - 1).clamp(min=0), C_IDX]
    n_first = atom37[idx, 0, N_IDX]
    closure = safe_norm(c_last - n_first, dim=-1)
    gap = terminal_gap_nm(atom37, pm)
    return (closure > bond_window_nm) & (gap >= min_gap_nm) & (lengths > 1)


def sequence_matches_source(
    seq_logits: torch.Tensor,
    source_aatype: torch.Tensor,
    pep_mask: torch.Tensor,
) -> torch.Tensor:
    """[b] bool -- does the decoded argmax sequence equal the source CP sequence exactly?"""
    pm = pep_mask.bool()
    pred = seq_logits.argmax(dim=-1)
    tgt = source_aatype.long().clamp(0, restype_num)
    agree = (pred == tgt) | ~pm
    return agree.all(dim=1) & (pm.sum(dim=1) > 0)


def pairwise_ca_rmsd_nm(atom37: torch.Tensor, pep_mask: torch.Tensor) -> torch.Tensor:
    """Mean pairwise CA distance-matrix RMSD across a set of samples. Scalar.

    Diversity measured on the internal distance matrix rather than on superposed
    coordinates: it needs no alignment, and two samples that differ only by a rigid
    motion in the pocket frame are correctly scored as identical in conformation.
    """
    from proteinfoundation.cp2lp.geometry import ca_coords

    ca = ca_coords(atom37)
    d = pairwise_distances(ca, ca)  # [b, n, n]
    pm = pep_mask.bool()
    pair = (pm[:, :, None] & pm[:, None, :]).float()
    b = d.shape[0]
    if b < 2:
        return torch.zeros((), device=d.device)
    diff = (d[:, None] - d[None, :]) ** 2  # [b, b, n, n]
    common = (pair[:, None] * pair[None, :])
    per_pair = torch.sqrt((diff * common).sum(dim=(2, 3)) / common.sum(dim=(2, 3)).clamp(min=1.0) + 1e-12)
    iu = torch.triu_indices(b, b, offset=1, device=d.device)
    return per_pair[iu[0], iu[1]].mean()
