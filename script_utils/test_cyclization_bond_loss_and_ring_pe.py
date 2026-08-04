#!/usr/bin/env python3
"""Unit tests for the cyclization bond-distance loss and the cycle-aware graph PE.

Pure-tensor tests: no GPU, no dataset, no model required. Covers
`proteinfoundation.cyclization.bond_loss`,
`proteinfoundation.eval.cyclic_reconstruction_metrics.per_sample_requested_bond_distance`,
and `proteinfoundation.nn.feature_factory.pair_feats.CyclizationGraphPositionalPairFeat`.

Usage (as a standalone script):
    python script_utils/test_cyclization_bond_loss_and_ring_pe.py

Usage (via pytest, if installed):
    pytest script_utils/test_cyclization_bond_loss_and_ring_pe.py -v
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from proteinfoundation.cyclization.bond_loss import cyclization_bond_loss, flat_bottom_penalty
from proteinfoundation.cyclization.constants import (
    AA_ASP,
    AA_CYS,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
    NO_CYCLIZATION_INDEX,
    UNSPECIFIED,
)
from proteinfoundation.eval.cyclic_reconstruction_metrics import (
    DISULFIDE_SG_BOND_WINDOW_A,
    MAINCHAIN_CN_BOND_WINDOW_A,
    cyclic_geometry_metrics,
    per_sample_requested_bond_distance,
)
from proteinfoundation.nn.feature_factory.pair_feats import CyclizationGraphPositionalPairFeat

AA_OTHER = 0  # never CYS/LYS/ASP/GLU
N_IDX, CA_IDX, C_IDX = 0, 1, 2
SG_IDX, NZ_IDX, CG_IDX = 10, 35, 5

ANG_TO_NM = 0.1


def _zeros_atom37(batch_size: int, length: int):
    coords = torch.zeros(batch_size, length, 37, 3)
    mask = torch.zeros(batch_size, length, 37, dtype=torch.bool)
    return coords, mask


def _meta(i, j, cyc_type, has_cyc=True, batch_size=1):
    """Cyclization metadata dict in the shape `extract_cyclization_metadata` returns."""
    return {
        "i": torch.full((batch_size,), i, dtype=torch.long),
        "j": torch.full((batch_size,), j, dtype=torch.long),
        "type": torch.full((batch_size,), cyc_type, dtype=torch.long),
        "has_cyclization": torch.full((batch_size,), has_cyc, dtype=torch.bool),
    }


def _disulfide_at(sg_sep_A: float, length: int = 4):
    """Single-sample CYS/CYS structure whose two SG atoms sit `sg_sep_A` apart."""
    coords, mask = _zeros_atom37(1, length)
    j = length - 1
    mask[0, 0, SG_IDX] = True
    mask[0, j, SG_IDX] = True
    coords[0, 0, SG_IDX] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, j, SG_IDX] = torch.tensor([sg_sep_A * ANG_TO_NM, 0.0, 0.0])
    aa = torch.full((1, length), AA_OTHER, dtype=torch.long)
    aa[0, 0] = AA_CYS
    aa[0, j] = AA_CYS
    return coords, mask, aa, _meta(0, j, DISULFIDE)


# ---------------------------------------------------------------------------
# A. flat_bottom_penalty -- the two-sidedness is the whole point
# ---------------------------------------------------------------------------
def test_flat_bottom_zero_inside_window():
    lo = torch.tensor([1.1, 1.1, 1.1])
    hi = torch.tensor([1.6, 1.6, 1.6])
    dist = torch.tensor([1.1, 1.33, 1.6])  # both edges + a real amide C-N
    assert torch.allclose(flat_bottom_penalty(dist, lo, hi), torch.zeros(3))


def test_flat_bottom_penalises_fused_ring_below_window():
    # 0.76 A: an observed FUSED design. A one-sided "pull together" term scores this
    # as perfect; the flat-bottom must not.
    lo, hi = torch.tensor([1.1]), torch.tensor([1.6])
    pen = flat_bottom_penalty(torch.tensor([0.76]), lo, hi)
    assert math.isclose(float(pen), (1.1 - 0.76) ** 2, abs_tol=1e-6)
    assert float(pen) > 0


def test_flat_bottom_penalises_open_ring_above_window():
    lo, hi = torch.tensor([1.1]), torch.tensor([1.6])
    pen = flat_bottom_penalty(torch.tensor([5.0]), lo, hi)
    assert math.isclose(float(pen), (5.0 - 1.6) ** 2, abs_tol=1e-6)


def test_flat_bottom_gradient_is_zero_inside_and_signed_outside():
    lo, hi = torch.tensor([1.1]), torch.tensor([1.6])

    inside = torch.tensor([1.33], requires_grad=True)
    flat_bottom_penalty(inside, lo, hi).sum().backward()
    assert math.isclose(float(inside.grad), 0.0, abs_tol=1e-8)

    # Too far apart -> gradient must pull the distance DOWN.
    openr = torch.tensor([5.0], requires_grad=True)
    flat_bottom_penalty(openr, lo, hi).sum().backward()
    assert float(openr.grad) > 0

    # Fused -> gradient must push the distance UP.
    fused = torch.tensor([0.5], requires_grad=True)
    flat_bottom_penalty(fused, lo, hi).sum().backward()
    assert float(fused.grad) < 0


# ---------------------------------------------------------------------------
# B. per_sample_requested_bond_distance
# ---------------------------------------------------------------------------
def test_per_sample_disulfide_distance_and_window():
    coords, mask, aa, meta = _disulfide_at(2.05)
    out = per_sample_requested_bond_distance(coords, mask, aa, meta["i"], meta["j"], meta["type"])
    assert math.isclose(float(out["dist_A"][0]), 2.05, abs_tol=1e-4)
    assert math.isclose(float(out["window_lo_A"][0]), DISULFIDE_SG_BOND_WINDOW_A[0], abs_tol=1e-6)
    assert math.isclose(float(out["window_hi_A"][0]), DISULFIDE_SG_BOND_WINDOW_A[1], abs_tol=1e-6)
    assert bool(out["atoms_valid"][0])


def test_per_sample_disulfide_invalid_when_endpoints_not_cysteine():
    coords, mask, aa, meta = _disulfide_at(2.05)
    aa[0, 0] = AA_OTHER  # chemistry no longer supports a disulfide
    out = per_sample_requested_bond_distance(coords, mask, aa, meta["i"], meta["j"], meta["type"])
    assert not bool(out["atoms_valid"][0])


def test_per_sample_mainchain_uses_only_the_real_n_i_c_j_orientation():
    """N(first, i) <-> C(last, j) is the only real closing bond: it's the free
    N-terminus amine bonding to the free C-terminus carbonyl. C(i)<->N(j) is a
    different, chemically meaningless pair (both atoms are already engaged in
    their own in-chain peptide bonds), and must play no part in the distance --
    not even via a min() over both orientations, which would let a
    coincidentally-close meaningless pair masquerade as ring closure.
    """
    coords, mask = _zeros_atom37(1, 4)
    for res in (0, 3):
        for atom in (N_IDX, C_IDX):
            mask[0, res, atom] = True
    # Real bond N_0 <-> C_3: 1.33 A. Meaningless pair C_0 <-> N_3: far apart.
    coords[0, 0, N_IDX] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 3, C_IDX] = torch.tensor([1.33 * ANG_TO_NM, 0.0, 0.0])
    coords[0, 0, C_IDX] = torch.tensor([0.0, 5.0, 0.0])
    coords[0, 3, N_IDX] = torch.tensor([0.0, -5.0, 0.0])
    aa = torch.full((1, 4), AA_OTHER, dtype=torch.long)
    meta = _meta(0, 3, MAINCHAIN)

    out = per_sample_requested_bond_distance(coords, mask, aa, meta["i"], meta["j"], meta["type"])
    assert math.isclose(float(out["dist_A"][0]), 1.33, abs_tol=1e-4)
    assert math.isclose(float(out["window_lo_A"][0]), MAINCHAIN_CN_BOND_WINDOW_A[0], abs_tol=1e-6)
    assert bool(out["atoms_valid"][0])


def test_per_sample_isopeptide_lys_asp_uses_cg():
    coords, mask = _zeros_atom37(1, 4)
    mask[0, 0, NZ_IDX] = True
    mask[0, 3, CG_IDX] = True
    coords[0, 0, NZ_IDX] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 3, CG_IDX] = torch.tensor([1.33 * ANG_TO_NM, 0.0, 0.0])
    aa = torch.full((1, 4), AA_OTHER, dtype=torch.long)
    aa[0, 0], aa[0, 3] = AA_LYS, AA_ASP
    meta = _meta(0, 3, ISOPEPTIDE)

    out = per_sample_requested_bond_distance(coords, mask, aa, meta["i"], meta["j"], meta["type"])
    assert math.isclose(float(out["dist_A"][0]), 1.33, abs_tol=1e-4)
    assert bool(out["atoms_valid"][0])


def test_per_sample_unknown_type_is_invalid():
    coords, mask, aa, _ = _disulfide_at(2.05)
    meta = _meta(0, 3, NO_CYCLIZATION_INDEX)
    out = per_sample_requested_bond_distance(coords, mask, aa, meta["i"], meta["j"], meta["type"])
    assert not bool(out["atoms_valid"][0])


def test_per_sample_distance_matches_the_closure_metric():
    """Anti-drift guard: the loss's distance must equal the metric's, per type.

    If these two ever disagree, the loss is optimising something the eval does not
    measure -- exactly the failure the shared helper exists to prevent.
    """
    cases = [
        (_disulfide_at(2.4)[:3] + (_meta(0, 3, DISULFIDE),), "disulfide_sg_dist_pred_A"),
    ]
    # Mainchain case.
    coords, mask = _zeros_atom37(1, 4)
    for res in (0, 3):
        for atom in (N_IDX, C_IDX):
            mask[0, res, atom] = True
    coords[0, 0, C_IDX] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 3, N_IDX] = torch.tensor([0.42, 0.0, 0.0])  # 4.2 A -> an open ring
    coords[0, 0, N_IDX] = torch.tensor([0.0, 9.0, 0.0])
    coords[0, 3, C_IDX] = torch.tensor([0.0, -9.0, 0.0])
    aa_mc = torch.full((1, 4), AA_OTHER, dtype=torch.long)
    cases.append(((coords, mask, aa_mc, _meta(0, 3, MAINCHAIN)), "mainchain_cn_dist_pred_A"))

    for (coords, mask, aa, meta), metric_key in cases:
        from_loss = per_sample_requested_bond_distance(
            coords, mask, aa, meta["i"], meta["j"], meta["type"]
        )["dist_A"]
        from_metric = cyclic_geometry_metrics(
            pred_atom37=coords,
            gt_atom37=coords,
            atom37_mask=mask,
            seq_tokens=aa,
            cyclization_metadata=meta,
            prefix="t",
        )[f"t/{metric_key}"]
        assert math.isclose(float(from_loss.mean()), from_metric, abs_tol=1e-4), metric_key


# ---------------------------------------------------------------------------
# C. cyclization_bond_loss
# ---------------------------------------------------------------------------
def test_bond_loss_zero_for_closed_ring():
    coords, mask, aa, meta = _disulfide_at(2.05)  # inside [1.8, 2.3]
    loss, metrics = cyclization_bond_loss(coords, mask, aa, meta)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-8)
    assert math.isclose(metrics["window_success"], 1.0, abs_tol=1e-8)
    assert metrics["n_valid"] == 1.0


def test_bond_loss_positive_for_open_ring():
    coords, mask, aa, meta = _disulfide_at(6.0)
    loss, metrics = cyclization_bond_loss(coords, mask, aa, meta)
    assert math.isclose(float(loss), (6.0 - DISULFIDE_SG_BOND_WINDOW_A[1]) ** 2, abs_tol=1e-4)
    assert math.isclose(metrics["window_success"], 0.0, abs_tol=1e-8)


def test_bond_loss_positive_for_fused_ring():
    """The regression this loss shape exists for: 3/50 sampled designs came out fused."""
    coords, mask, aa, meta = _disulfide_at(0.8)
    loss, metrics = cyclization_bond_loss(coords, mask, aa, meta)
    assert float(loss) > 0
    assert math.isclose(float(loss), (DISULFIDE_SG_BOND_WINDOW_A[0] - 0.8) ** 2, abs_tol=1e-4)
    assert math.isclose(metrics["window_success"], 0.0, abs_tol=1e-8)


def test_bond_loss_t_weight_gates_low_t_to_zero():
    coords, mask, aa, meta = _disulfide_at(6.0)
    hot, _ = cyclization_bond_loss(coords, mask, aa, meta, t_weight=torch.tensor([1.0]))
    cold, _ = cyclization_bond_loss(coords, mask, aa, meta, t_weight=torch.tensor([0.0]))
    assert float(hot) > 0
    assert math.isclose(float(cold), 0.0, abs_tol=1e-8)


def test_bond_loss_metrics_exclude_t_zeroed_samples():
    """A sample predicted from pure noise must not be counted in the closure readout,
    or the metric becomes a readout of the sampled `t` distribution instead of the model."""
    coords, mask, aa, meta = _disulfide_at(6.0)
    loss, metrics = cyclization_bond_loss(coords, mask, aa, meta, t_weight=torch.tensor([0.0]))
    assert math.isclose(float(loss), 0.0, abs_tol=1e-8)
    assert metrics["n_valid"] == 0.0
    assert math.isnan(metrics["window_success"])


def test_bond_loss_gradient_reaches_coordinates():
    coords, mask, aa, meta = _disulfide_at(6.0)
    coords.requires_grad_(True)
    loss, _ = cyclization_bond_loss(coords, mask, aa, meta)
    loss.backward()
    assert coords.grad is not None
    # Gradient lands on the two SG anchors and nowhere else.
    assert torch.any(coords.grad[0, 0, SG_IDX] != 0)
    assert torch.any(coords.grad[0, 3, SG_IDX] != 0)
    assert torch.all(coords.grad[0, 0, CA_IDX] == 0)


def test_bond_loss_zero_when_no_usable_label():
    coords, mask, aa, _ = _disulfide_at(6.0)
    coords.requires_grad_(True)
    loss, metrics = cyclization_bond_loss(coords, mask, aa, _meta(0, 3, DISULFIDE, has_cyc=False))
    assert math.isclose(float(loss), 0.0, abs_tol=1e-8)
    assert metrics["n_valid"] == 0.0
    loss.backward()  # must stay differentiable for DDP


def test_bond_loss_no_metadata_is_noop():
    coords, mask, aa, _ = _disulfide_at(2.05)
    loss, metrics = cyclization_bond_loss(coords, mask, aa, None)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-8)
    assert metrics["n_valid"] == 0.0


def test_bond_loss_averages_only_over_valid_samples():
    """An unlabeled sample must not dilute the mean toward zero."""
    coords, mask = _zeros_atom37(2, 4)
    aa = torch.full((2, 4), AA_OTHER, dtype=torch.long)
    for b in (0, 1):
        mask[b, 0, SG_IDX] = True
        mask[b, 3, SG_IDX] = True
        aa[b, 0], aa[b, 3] = AA_CYS, AA_CYS
        coords[b, 3, SG_IDX] = torch.tensor([6.0 * ANG_TO_NM, 0.0, 0.0])
    meta = _meta(0, 3, DISULFIDE, batch_size=2)
    meta["has_cyclization"] = torch.tensor([True, False])

    loss, metrics = cyclization_bond_loss(coords, mask, aa, meta)
    assert metrics["n_valid"] == 1.0
    assert math.isclose(float(loss), (6.0 - DISULFIDE_SG_BOND_WINDOW_A[1]) ** 2, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# D. CyclizationGraphPositionalPairFeat
# ---------------------------------------------------------------------------
def _pair_batch(n=6, cond=None, cyc_i=None, cyc_j=None, mask=None):
    batch = {"x_t": {"bb_ca": torch.zeros(1, n, 3)}}
    batch["mask"] = torch.ones(1, n, dtype=torch.bool) if mask is None else mask
    if cond is not None:
        batch["cyclization_type_cond"] = torch.tensor([cond], dtype=torch.long)
    if cyc_i is not None:
        batch["cyclization_i"] = torch.tensor([cyc_i], dtype=torch.long)
        batch["cyclization_j"] = torch.tensor([cyc_j], dtype=torch.long)
    return batch


def test_ring_pe_shape_and_dim():
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=8)
    out = feat(_pair_batch(n=6, cond=MAINCHAIN))
    assert out.shape == (1, 6, 6, 9)  # ring_sep_dim + 1 closing-edge channel
    assert feat.get_dim() == 9


def test_ring_pe_is_zero_when_no_cyclization_requested():
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=8)
    assert torch.all(feat(_pair_batch(n=6, cond=UNSPECIFIED)) == 0)
    # No cyclization keys at all (non-CPSea batch) -> also a no-op.
    assert torch.all(feat(_pair_batch(n=6)) == 0)


def test_ring_pe_makes_termini_topological_neighbours():
    """The fix itself: residues 0 and n-1 must read as distance 1, not n-1."""
    n = 6
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=8)
    out = feat(_pair_batch(n=n, cond=MAINCHAIN))
    geo = out[0, :, :, :8].argmax(dim=-1)
    assert int(geo[0, n - 1]) == 1
    assert int(geo[0, 1]) == 1  # ordinary chain neighbour, unchanged


def test_ring_pe_geodesic_equals_ring_distance_for_head_to_tail():
    """For a termini closure the geodesic must reduce to min(|a-b|, L-|a-b|)."""
    n = 6
    out = CyclizationGraphPositionalPairFeat(ring_sep_dim=16)(_pair_batch(n=n, cond=MAINCHAIN))
    geo = out[0, :, :, :16].argmax(dim=-1)
    for a in range(n):
        for b in range(n):
            expected = min(abs(a - b), n - abs(a - b))
            assert int(geo[a, b]) == expected, (a, b, int(geo[a, b]), expected)


def test_ring_pe_marks_the_closing_edge_both_ways():
    n = 6
    out = CyclizationGraphPositionalPairFeat(ring_sep_dim=8)(_pair_batch(n=n, cond=MAINCHAIN))
    closing = out[0, :, :, -1]
    assert float(closing[0, n - 1]) == 1.0
    assert float(closing[n - 1, 0]) == 1.0
    assert float(closing.sum()) == 2.0  # and nowhere else


def test_ring_pe_uses_labeled_endpoints_when_present():
    n = 6
    out = CyclizationGraphPositionalPairFeat(ring_sep_dim=8)(
        _pair_batch(n=n, cond=DISULFIDE, cyc_i=1, cyc_j=4)
    )
    closing = out[0, :, :, -1]
    assert float(closing[1, 4]) == 1.0 and float(closing[4, 1]) == 1.0
    assert float(closing[0, n - 1]) == 0.0  # NOT the termini
    geo = out[0, :, :, :8].argmax(dim=-1)
    assert int(geo[1, 4]) == 1
    assert int(geo[0, 5]) == 3  # 0->1, closing edge 1-4, 4->5


def test_ring_pe_falls_back_to_termini_for_unlabeled_endpoints():
    n = 6
    out = CyclizationGraphPositionalPairFeat(ring_sep_dim=8)(
        _pair_batch(n=n, cond=MAINCHAIN, cyc_i=NO_CYCLIZATION_INDEX, cyc_j=NO_CYCLIZATION_INDEX)
    )
    closing = out[0, :, :, -1]
    assert float(closing[0, n - 1]) == 1.0


def test_ring_pe_termini_respect_padding():
    """A right-padded binder must close on its last REAL residue, not the pad."""
    n = 6
    mask = torch.ones(1, n, dtype=torch.bool)
    mask[0, 4:] = False  # binder is residues 0..3
    out = CyclizationGraphPositionalPairFeat(ring_sep_dim=8)(_pair_batch(n=n, cond=MAINCHAIN, mask=mask))
    closing = out[0, :, :, -1]
    assert float(closing[0, 3]) == 1.0
    assert float(closing[0, 5]) == 0.0


def test_ring_pe_is_constant_across_flow_time():
    """The cycle graph is conditioning, not a prediction: it must not depend on x_t/t."""
    feat = CyclizationGraphPositionalPairFeat(ring_sep_dim=8)
    b_noisy = _pair_batch(n=6, cond=MAINCHAIN)
    b_noisy["x_t"]["bb_ca"] = torch.randn(1, 6, 3) * 10
    b_noisy["t"] = {"bb_ca": torch.tensor([0.01])}
    b_clean = _pair_batch(n=6, cond=MAINCHAIN)
    b_clean["t"] = {"bb_ca": torch.tensor([0.99])}
    assert torch.allclose(feat(b_noisy), feat(b_clean))


# ---------------------------------------------------------------------------
# E. Proteina wiring (stubbed -- no checkpoints, no GPU, no data)
# ---------------------------------------------------------------------------
def _t_weight_stub(t_lower=0.5, data_modes=("bb_ca", "local_latents")):
    stub = SimpleNamespace()
    stub.fm = SimpleNamespace(data_modes=list(data_modes))
    stub.cyclization_bond_loss_t_lower = t_lower
    return stub


def test_t_weight_ramps_linearly_from_t_lower_lim():
    from proteinfoundation.proteina import Proteina

    batch = {"t": {"bb_ca": torch.tensor([0.0, 0.5, 0.75, 1.0]), "local_latents": torch.ones(4)}}
    w = Proteina._bond_loss_t_weight(_t_weight_stub(0.5), batch)
    assert torch.allclose(w, torch.tensor([0.0, 0.0, 0.5, 1.0]), atol=1e-6)


def test_t_weight_takes_min_over_modalities():
    """A clean Ca trace with noise-level latents still has no placeable anchor atom."""
    from proteinfoundation.proteina import Proteina

    batch = {"t": {"bb_ca": torch.tensor([1.0]), "local_latents": torch.tensor([0.1])}}
    assert math.isclose(float(Proteina._bond_loss_t_weight(_t_weight_stub(0.5), batch)), 0.0, abs_tol=1e-8)


def _bond_loss_stub(coords, atom_mask, aa, n):
    from proteinfoundation.proteina import Proteina

    stub = _t_weight_stub(0.5)
    # Real implementation, just bound onto the stub.
    stub._bond_loss_t_weight = lambda batch: Proteina._bond_loss_t_weight(stub, batch)
    stub.fm.nn_out_to_clean_sample_prediction = lambda batch, nn_out: {
        "bb_ca": torch.zeros(1, n, 3),
        "local_latents": torch.zeros(1, n, 8),
    }
    stub.autoencoder = SimpleNamespace(
        decode=lambda z_latent, ca_coors_nm, mask: {
            "coors_nm": coords,
            "atom_mask": atom_mask,
            "residue_type": aa,
        }
    )
    stub.logged = {}
    stub.log = lambda k, v, **kw: stub.logged.__setitem__(k, float(v))
    return stub


def test_compute_cyclization_bond_loss_wiring_and_logging():
    from proteinfoundation.proteina import Proteina

    n = 4
    coords, atom_mask, aa, meta = _disulfide_at(6.0, length=n)  # open ring
    stub = _bond_loss_stub(coords, atom_mask, aa, n)
    batch = {
        "mask": torch.ones(1, n, dtype=torch.bool),
        "t": {"bb_ca": torch.tensor([1.0]), "local_latents": torch.tensor([1.0])},
        "cyclization_i": meta["i"],
        "cyclization_j": meta["j"],
        "cyclization_type": meta["type"],
        "has_cyclization": meta["has_cyclization"],
    }
    loss, metrics = Proteina.compute_cyclization_bond_loss(stub, batch, nn_out={}, log_prefix="train", bs=1)

    assert float(loss) > 0
    assert metrics["n_valid"] == 1.0
    assert stub.logged["train/loss_cyclization_bond"] > 0
    for key in ("window_success", "mean_dist_A", "n_valid"):
        assert f"train/cyclization_bond_{key}" in stub.logged, key


def test_compute_cyclization_bond_loss_noop_without_cyclization_keys():
    """A non-CPSea batch must not break training."""
    from proteinfoundation.proteina import Proteina

    n = 4
    coords, atom_mask, aa, _ = _disulfide_at(6.0, length=n)
    stub = _bond_loss_stub(coords, atom_mask, aa, n)
    batch = {"mask": torch.ones(1, n, dtype=torch.bool), "t": {"bb_ca": torch.tensor([1.0])}}
    loss, metrics = Proteina.compute_cyclization_bond_loss(stub, batch, nn_out={}, log_prefix="train", bs=1)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-8)
    assert metrics == {}


# ---------------------------------------------------------------------------
# F. val_generation samples the way the DESIGN pipeline samples
# ---------------------------------------------------------------------------
def test_val_generation_uses_the_design_sampler_not_the_monomer_one():
    """Regression guard for the bug that made val_gen report shredded samples.

    `generation.model["ode"]` is Proteina's *unconditional monomer* sampler: it zero-CoMs the
    sample every ODE step. CPSea centres on the TARGET, so the binder legitimately sits
    off-origin at its binding site and centering drags it to the receptor's core every step --
    producing Ca-Ca ~4 nm (vs 0.38 real) that says nothing about the model. If this test fails,
    val_gen has silently gone back to measuring a sampler nobody runs.
    """
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    cfg_dir = Path(__file__).resolve().parent.parent / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name="example/training_cpsea_peptide_cyc_typecond")
    GlobalHydra.instance().clear()

    design = cfg.val_generation.design_sampling
    reference = OmegaConf.load(cfg_dir / "pipeline" / "model_sampling.yaml")

    # Pulled from the design pipeline itself, so the two cannot drift apart.
    assert OmegaConf.to_container(design.model) == OmegaConf.to_container(reference.model)
    # The specific knob that was wrong, called out by name so a regression is unambiguous.
    assert design.model.bb_ca.simulation_step_params.center_every_step is False
    assert design.model.bb_ca.simulation_step_params.sampling_mode == "sc"
    # Pinning these would silently override design's values.
    assert cfg.val_generation.nsteps is None
    assert cfg.val_generation.self_cond is None


def test_sample_formatting_returns_angstrom_so_val_gen_must_convert():
    """Regression guard: `sample_formatting` returns ANGSTROM, the metrics consume NANOMETERS.

    Feeding one to the other is silent -- no shape or dtype error, just every geometry metric
    wrong by 10x (and Angstrom-vs-nm comparisons in place/*, iface/*). This pins the contract at
    both ends so a change to either is caught here rather than in a week of meaningless val_gen.
    """
    import inspect

    from proteinfoundation.utils import sample_utils

    # End 1: sample_formatting converts nm -> Angstrom on the way out.
    src = inspect.getsource(sample_utils.format_sample_local_latents)
    assert 'nm_to_ang(output_decoder["coors_nm"])' in src, (
        "format_sample_local_latents no longer returns Angstrom; validation_step_generate's "
        "`/ NM_TO_ANG` conversion must be revisited."
    )

    # End 2: validation_step_generate converts back before metering.
    from proteinfoundation.proteina import Proteina

    vsg = inspect.getsource(Proteina.validation_step_generate)
    assert '/ NM_TO_ANG' in vsg, "validation_step_generate must convert Angstrom -> nm before metrics"

    # End 3: the consumer really is nm-based, at the scale that makes this a 10x error.
    from proteinfoundation.eval.sampled_binder_metrics import CA_CA_NM, CA_CA_TOL_NM

    assert abs(CA_CA_NM - 0.3835) < 1e-6
    # An Angstrom Ca-Ca (~3.8) must fall far outside the nm tolerance -- i.e. the bug this guards
    # against would indeed have pinned ca_ca_viol_frac at 1.0.
    assert abs(CA_CA_NM * 10 - CA_CA_NM) > CA_CA_TOL_NM


# ---------------------------------------------------------------------------
# G. terminal_only closes the disulfide/isopeptide escape hatch
# ---------------------------------------------------------------------------
def _cys_peptide(length=8, cys_at=(0, 7, 3, 5)):
    """Binder with CYS at several positions -- i.e. several candidate disulfide pairs."""
    aa = torch.full((1, length), AA_OTHER, dtype=torch.long)
    for p in cys_at:
        aa[0, p] = AA_CYS
    return aa, torch.ones(1, length, dtype=torch.bool)


def test_mask_disulfide_hatch_exists_by_default():
    """Pins the CURRENT default: any CYS-CYS pair is a candidate, not just the termini."""
    from proteinfoundation.cyclization.mask import build_cyclization_validity_mask

    aa, binder_mask = _cys_peptide()
    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask)
    # The non-terminal (3, 5) CYS pair is allowed -- this is the escape hatch.
    assert bool(valid[0, 3, 5, DISULFIDE])
    assert int(valid[0, :, :, DISULFIDE].sum()) > 1
    # ...while mainchain was always held to the single terminal pair.
    assert int(valid[0, :, :, MAINCHAIN].sum()) == 1
    assert bool(valid[0, 0, 7, MAINCHAIN])


def test_mask_terminal_only_closes_the_hatch_for_all_types():
    from proteinfoundation.cyclization.mask import build_cyclization_validity_mask

    aa, binder_mask = _cys_peptide()
    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask, terminal_only=True)
    # Only the terminal disulfide survives; the (3, 5) shortcut is gone.
    assert not bool(valid[0, 3, 5, DISULFIDE])
    assert bool(valid[0, 0, 7, DISULFIDE])
    assert int(valid[0, :, :, DISULFIDE].sum()) == 1
    # Mainchain is unchanged -- it was already terminal-only.
    assert int(valid[0, :, :, MAINCHAIN].sum()) == 1


def test_mask_terminal_only_respects_padding():
    """The terminal pair is the last REAL residue, not the last tensor slot."""
    from proteinfoundation.cyclization.mask import build_cyclization_validity_mask

    aa, binder_mask = _cys_peptide(length=8, cys_at=(0, 4, 7))
    binder_mask[0, 5:] = False  # binder is residues 0..4
    valid = build_cyclization_validity_mask(aa=aa, binder_mask=binder_mask, terminal_only=True)
    assert bool(valid[0, 0, 4, DISULFIDE])
    assert not bool(valid[0, 0, 7, DISULFIDE])


def test_mask_terminal_only_never_masks_out_a_gold_label():
    """force_gold_valid must still punch a (hypothetical) non-terminal gold through,
    so training is unaffected even if a rare non-terminal label exists."""
    from proteinfoundation.cyclization.mask import build_cyclization_validity_mask

    aa, binder_mask = _cys_peptide()
    valid = build_cyclization_validity_mask(
        aa=aa,
        binder_mask=binder_mask,
        gold_i=torch.tensor([3]),
        gold_j=torch.tensor([5]),
        gold_type=torch.tensor([DISULFIDE]),
        force_gold_valid=True,
        terminal_only=True,
    )
    assert bool(valid[0, 3, 5, DISULFIDE])


def test_val_gen_logs_the_counts_behind_each_rate():
    """A per-type success rate without its n is unreadable, and the per-type subsets differ."""
    from proteinfoundation.eval.cyclic_reconstruction_metrics import CYCLIC_COUNT_SUFFIXES
    from proteinfoundation.eval.sampled_binder_metrics import CYCLIC_PRED_ONLY_SUFFIXES

    for suffix in ("n_valid_mainchain", "n_valid_disulfide", "n_valid_isopeptide"):
        assert suffix in CYCLIC_PRED_ONLY_SUFFIXES, suffix
        assert suffix in CYCLIC_COUNT_SUFFIXES, suffix


ALL_TESTS = [
    test_flat_bottom_zero_inside_window,
    test_flat_bottom_penalises_fused_ring_below_window,
    test_flat_bottom_penalises_open_ring_above_window,
    test_flat_bottom_gradient_is_zero_inside_and_signed_outside,
    test_per_sample_disulfide_distance_and_window,
    test_per_sample_disulfide_invalid_when_endpoints_not_cysteine,
    test_per_sample_mainchain_uses_only_the_real_n_i_c_j_orientation,
    test_per_sample_isopeptide_lys_asp_uses_cg,
    test_per_sample_unknown_type_is_invalid,
    test_per_sample_distance_matches_the_closure_metric,
    test_bond_loss_zero_for_closed_ring,
    test_bond_loss_positive_for_open_ring,
    test_bond_loss_positive_for_fused_ring,
    test_bond_loss_t_weight_gates_low_t_to_zero,
    test_bond_loss_metrics_exclude_t_zeroed_samples,
    test_bond_loss_gradient_reaches_coordinates,
    test_bond_loss_zero_when_no_usable_label,
    test_bond_loss_no_metadata_is_noop,
    test_bond_loss_averages_only_over_valid_samples,
    test_ring_pe_shape_and_dim,
    test_ring_pe_is_zero_when_no_cyclization_requested,
    test_ring_pe_makes_termini_topological_neighbours,
    test_ring_pe_geodesic_equals_ring_distance_for_head_to_tail,
    test_ring_pe_marks_the_closing_edge_both_ways,
    test_ring_pe_uses_labeled_endpoints_when_present,
    test_ring_pe_falls_back_to_termini_for_unlabeled_endpoints,
    test_ring_pe_termini_respect_padding,
    test_ring_pe_is_constant_across_flow_time,
    test_t_weight_ramps_linearly_from_t_lower_lim,
    test_t_weight_takes_min_over_modalities,
    test_compute_cyclization_bond_loss_wiring_and_logging,
    test_compute_cyclization_bond_loss_noop_without_cyclization_keys,
    test_val_generation_uses_the_design_sampler_not_the_monomer_one,
    test_sample_formatting_returns_angstrom_so_val_gen_must_convert,
    test_mask_disulfide_hatch_exists_by_default,
    test_mask_terminal_only_closes_the_hatch_for_all_types,
    test_mask_terminal_only_respects_padding,
    test_mask_terminal_only_never_masks_out_a_gold_label,
    test_val_gen_logs_the_counts_behind_each_rate,
]


if __name__ == "__main__":
    print(f"Running {len(ALL_TESTS)} tests...")
    failures = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  OK {test_fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append(test_fn.__name__)
            print(f"  FAIL {test_fn.__name__}: {e}")

    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
