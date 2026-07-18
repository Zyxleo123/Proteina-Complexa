"""Reference-free metrics on a binder the model actually SAMPLED (ODE integrated from t=0).

Why this module exists
----------------------
Every other metric logged during CPSea training is TEACHER-FORCED: `x_t` is built by
interpolating the *true* `x_1` with noise, so the model is always handed most of the answer.
`run_four_way_decode_eval` is teacher-forced (one-shot `x_1 = x_t + (1-t) v` from a real
interpolant), and `decode_gtca_*` literally hands the decoder the native Ca trace.

Sampling never sees a true interpolant -- it consumes its own accumulated output. A model can
therefore look healthy on every teacher-forced metric and still emit garbage when integrated,
and no teacher-forced diagnostic can detect that, by construction. These metrics score the
structure the model actually produces.

Deliberately cheap: no AF2, no folding, no MPNN, no search. One ODE integration + AE decode +
geometry. Meant to run inside the training validation loop, so it must cost seconds, not minutes.

Metric groups
-------------
geom      Is it even a peptide? Consecutive Ca-Ca should be ~0.38 nm.
iface     Is it docking, or floating off into solvent? Contacts against target / hotspots.
seq       Sequence mode collapse (a model emitting poly-Gly scores fine on coordinate MSE).
div       Structural mode collapse: spread across repeats of the SAME complex. Needs n_repeat>1.
place     Unaligned Ca RMSD to the native binder. The frame is target-centered, so this is a
          PLACEMENT metric (did it dock in the right pocket), not a fold-quality one. It is a
          weak signal -- many peptides bind one receptor -- so read it as a floor, not a target.
"""

from __future__ import annotations

import math

import torch
from openfold.np import residue_constants as rc
from openfold.np.residue_constants import atom_order

from proteinfoundation.eval.cyclic_reconstruction_metrics import CYCLIC_COUNT_SUFFIXES

CA_IDX = atom_order["CA"]

_RESTYPE_ATOM37_MASK: torch.Tensor | None = None


def atom37_mask_from_aatype(aatype: torch.Tensor) -> torch.Tensor:
    """[B, L] residue types -> [B, L, 37] valid-atom mask, from the SAMPLED sequence.

    Cyclization chemistry (which atoms exist, and so whether a CYS-SG / LYS-NZ closing bond is even
    possible) is a property of the residues the MODEL emitted, not of the native ones. Scoring a
    sampled structure against the native atom mask would credit bonds between atoms the sampled
    sequence does not have.
    """
    global _RESTYPE_ATOM37_MASK
    if _RESTYPE_ATOM37_MASK is None:
        t = torch.zeros(21, 37, dtype=torch.bool)
        for restype, letter in enumerate(rc.restypes):
            for atom_name in rc.residue_atoms[rc.restype_1to3[letter]]:
                t[restype, rc.atom_order[atom_name]] = True
        _RESTYPE_ATOM37_MASK = t
    table = _RESTYPE_ATOM37_MASK.to(aatype.device)
    return table[aatype.long().clamp(0, 20)]


# Reference-free subset of `cyclic_reconstruction_metrics.CYCLIC_METRIC_SUFFIXES`: these score the
# SAMPLED structure on its own terms (is the closing bond at a chemically valid distance). The
# `*_gt_A` / `*_abs_err_A` keys are dropped -- a sampled binder has no native counterpart to its
# own predicted (i, j) endpoints, so an "error vs GT" there would be comparing different bonds.
CYCLIC_PRED_ONLY_SUFFIXES = [
    "cyc_cb_dist_pred_A",
    "cyc_cb_window_success",
    "mainchain_cn_dist_pred_A",
    "mainchain_cn_bond_success",
    "disulfide_sg_dist_pred_A",
    "disulfide_bond_success",
    "isopeptide_n_c_dist_pred_A",
    "isopeptide_bond_success",
    # The counts backing each rate above. Not optional bookkeeping: the per-type subsets differ by
    # construction (a disulfide is only scorable where the SAMPLED structure carries two CYS, while
    # mainchain is scorable on every sample), so without these you cannot tell whether
    # disulfide_bond_success >> mainchain_cn_bond_success means the model is better at disulfides or
    # merely that it was scored on three lucky samples.
    *CYCLIC_COUNT_SUFFIXES,
]

# Consecutive Ca-Ca distance in a real peptide, nm (measured on CPSea: 0.3835). A sampled chain
# outside this window is not a protein, whatever its MSE says.
CA_CA_NM = 0.3835
CA_CA_TOL_NM = 0.05

CONTACT_NM = 0.8       # binder-target Ca-Ca contact threshold (8 A)
HOTSPOT_NM = 1.0       # a hotspot counts as engaged if any binder Ca is within 10 A


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
    m = m.bool()
    return float(x[m].mean().item()) if m.any() else float("nan")


def _target_ca(x_target: torch.Tensor) -> torch.Tensor:
    """Target coords -> Ca only, [B, T, 3]. Accepts [B, T, 3] or [B, T, 37, 3] (pair_feats.py:569
    does the same dim-based dispatch)."""
    return x_target[:, :, CA_IDX, :] if x_target.dim() == 4 else x_target


def _as_residue_mask(m: torch.Tensor) -> torch.Tensor:
    """Reduce an ATOM-level target mask [B, T, 37] to a RESIDUE-level one [B, T]; pass [B, T] through.

    `ExtractTargetCoordinatesTransform` in `compact_mode` REASSIGNS `target_mask` to
    `graph.target_mask[target_residue_mask]` -> `[n_target, 37]` (transforms.py:2536), i.e. it is
    atom-level, not the [B, T] residue mask the name suggests. `target_hotspot_mask` is atom-level
    too in the non-compact branch (`hotspot_mask * target_mask`, transforms.py:2616). Consuming
    either as [B, T] silently misaligns every interface metric, so normalise here rather than
    trusting the caller.
    """
    return m.any(dim=-1) if m.dim() == 3 else m


@torch.no_grad()
def sampled_binder_metrics(
    coors: torch.Tensor,
    aatype: torch.Tensor,
    mask: torch.Tensor,
    *,
    gt_coors: torch.Tensor | None = None,
    x_target: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    target_hotspot_mask: torch.Tensor | None = None,
    n_repeat: int = 1,
    prefix: str = "val_gen",
) -> dict[str, float]:
    """Scores sampled binders. All tensors batch-first; coordinates in nm.

    Args:
        coors: [B, L, 37, 3] sampled binder coordinates, nm.
        aatype: [B, L] sampled residue type ids.
        mask: [B, L] valid binder residues.
        gt_coors: [B, L, 37, 3] native binder, nm. Enables the `place/*` group.
        x_target: [B, T, 3] or [B, T, 37, 3] receptor coordinates, nm. Enables `iface/*`.
        target_mask: [B, T] valid receptor residues.
        target_hotspot_mask: [B, T] receptor hotspots.
        n_repeat: samples per complex. The batch must be laid out so that repeats of one complex
            are CONTIGUOUS (i.e. `repeat_interleave`, NOT `repeat`) -- `div/*` slices on that
            assumption and will silently compare different complexes if it is violated.
        prefix: metric key prefix.

    Returns:
        Flat {f"{prefix}/..." : float} dict. Always the same key set; NaN where not computable,
        never omitted (DDP `sync_dist` needs a stable key set across ranks).
    """
    m = mask.bool()
    b, ell = m.shape
    out: dict[str, float] = {}
    ca = coors[..., CA_IDX, :]  # [B, L, 3]

    # --- geom: consecutive Ca-Ca -------------------------------------------------------------
    step = torch.linalg.norm(ca[:, 1:] - ca[:, :-1], dim=-1)          # [B, L-1]
    step_valid = m[:, 1:] & m[:, :-1]
    out[f"{prefix}/geom/ca_ca_nm"] = _masked_mean(step, step_valid)
    viol = (step - CA_CA_NM).abs() > CA_CA_TOL_NM
    out[f"{prefix}/geom/ca_ca_viol_frac"] = _masked_mean(viol.float(), step_valid)

    # --- iface: is it actually on the receptor? ----------------------------------------------
    if x_target is not None and target_mask is not None:
        tca = _target_ca(x_target)                                     # [B, T, 3]
        tm = _as_residue_mask(target_mask.bool())                      # [B, T]
        d = torch.linalg.norm(ca[:, :, None, :] - tca[:, None, :, :], dim=-1)  # [B, L, T]
        pair = m[:, :, None] & tm[:, None, :]
        big = torch.finfo(d.dtype).max
        dm = d.masked_fill(~pair, big)

        per_min = dm.reshape(b, -1).min(dim=-1).values                 # [B]
        ok = per_min < big
        out[f"{prefix}/iface/min_dist_nm"] = float(per_min[ok].mean().item()) if ok.any() else float("nan")
        out[f"{prefix}/iface/n_contacts"] = float(
            ((dm < CONTACT_NM) & pair).reshape(b, -1).sum(-1).float().mean().item()
        )
        if target_hotspot_mask is not None:
            hs = _as_residue_mask(target_hotspot_mask.bool()) & tm     # [B, T]
            near = ((dm < HOTSPOT_NM) & pair).any(dim=1)               # [B, T] hotspot engaged
            hit = (near & hs).sum(-1).float()
            tot = hs.sum(-1).float()
            frac = torch.where(tot > 0, hit / tot.clamp(min=1), torch.full_like(tot, float("nan")))
            out[f"{prefix}/iface/hotspot_frac"] = float(torch.nanmean(frac).item())
        else:
            out[f"{prefix}/iface/hotspot_frac"] = float("nan")
    else:
        for k in ("min_dist_nm", "n_contacts", "hotspot_frac"):
            out[f"{prefix}/iface/{k}"] = float("nan")

    # --- seq: mode collapse ------------------------------------------------------------------
    # A model that emits poly-Gly can still post a respectable coordinate MSE. This catches it.
    toks = aatype[m].long()
    if toks.numel():
        counts = torch.bincount(toks.clamp(min=0), minlength=21).float()
        p = counts / counts.sum()
        nz = p[p > 0]
        out[f"{prefix}/seq/top1_frac"] = float(p.max().item())
        out[f"{prefix}/seq/entropy_bits"] = float(-(nz * nz.log2()).sum().item())
        out[f"{prefix}/seq/n_types"] = float((counts > 0).sum().item())
    else:
        for k in ("top1_frac", "entropy_bits", "n_types"):
            out[f"{prefix}/seq/{k}"] = float("nan")

    # --- div: structural mode collapse across repeats of the SAME complex ---------------------
    # Mean pairwise unaligned Ca RMSD within each repeat group. ~0 => the model ignores the noise
    # and emits one answer per target, which is mode collapse even if that answer scores well.
    if n_repeat > 1 and b % n_repeat == 0:
        g = b // n_repeat
        ca_g = ca.view(g, n_repeat, ell, 3)
        m_g = m.view(g, n_repeat, ell)
        # Pair repeats within a group; compare on residues valid in BOTH members of the pair.
        d2 = (ca_g[:, :, None] - ca_g[:, None, :]).pow(2).sum(-1)      # [g, r, r, L]
        pm = (m_g[:, :, None] & m_g[:, None, :]).float()               # [g, r, r, L]
        cnt = pm.sum(-1)                                               # [g, r, r]
        rmsd = torch.sqrt((d2 * pm).sum(-1) / cnt.clamp(min=1))
        iu = torch.triu_indices(n_repeat, n_repeat, offset=1)
        pairs = rmsd[:, iu[0], iu[1]]                                  # [g, n_pairs]
        valid = cnt[:, iu[0], iu[1]] > 0
        out[f"{prefix}/div/ca_rmsd_nm"] = float(pairs[valid].mean().item()) if valid.any() else float("nan")
    else:
        out[f"{prefix}/div/ca_rmsd_nm"] = float("nan")

    # --- place: unaligned Ca RMSD to the native binder ----------------------------------------
    if gt_coors is not None:
        gca = gt_coors[..., CA_IDX, :]
        sq = (ca - gca).pow(2).sum(-1)                                 # [B, L]
        cnt = m.float().sum(-1)
        rmsd = torch.sqrt((sq * m.float()).sum(-1) / cnt.clamp(min=1))
        ok = cnt > 0
        out[f"{prefix}/place/ca_rmsd_nm"] = float(rmsd[ok].mean().item()) if ok.any() else float("nan")
    else:
        out[f"{prefix}/place/ca_rmsd_nm"] = float("nan")

    return out


SAMPLED_METRIC_SUFFIXES = [
    "geom/ca_ca_nm",
    "geom/ca_ca_viol_frac",
    "iface/min_dist_nm",
    "iface/n_contacts",
    "iface/hotspot_frac",
    "seq/top1_frac",
    "seq/entropy_bits",
    "seq/n_types",
    "div/ca_rmsd_nm",
    "place/ca_rmsd_nm",
]
