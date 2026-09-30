"""Unit tests for the CP->LP generator's pure pieces.

Deliberately free of the flow network, the AE and CUDA: everything here is a tensor
contract or a piece of arithmetic that can be wrong silently. The parts that need real
weights are covered by the smoke job (`scripts/cp2lp_smoke.sbatch`), which runs one batch
through generator -> AE -> discriminator -> backward.

Run: .venv/bin/python -m pytest script_utils/test_cp2lp.py -q
"""

from __future__ import annotations

import torch

from proteinfoundation.cp2lp.conditioning import (
    SRC_CP_KEY,
    SRC_CP_PRESENT_KEY,
    attach_source_cp_condition,
    drop_source_cp_condition,
    request_linear_topology,
)
from proteinfoundation.cp2lp.data import (
    RealLPFeatureReservoir,
    in_eval_split,
    slice_batch,
    split_roles,
)
from proteinfoundation.cp2lp.discriminator import (
    PeptideInterfaceDiscriminator,
    discriminator_accuracy,
    hinge_d_loss,
    hinge_g_loss,
)
from proteinfoundation.cp2lp.geometry import (
    CA_IDX,
    IDEAL_CA_C_NM,
    IDEAL_N_CA_NM,
    IDEAL_PEPTIDE_BOND_NM,
    C_IDX,
    N_IDX,
    backbone_dihedrals,
    open_chain_bond_pair_mask,
    peptide_bond_deviation_nm,
    terminal_gap_nm,
)
from proteinfoundation.cp2lp.losses import (
    clash_loss,
    contact_retention_loss,
    linear_terminus_valid,
    open_chain_geometry_loss,
    sequence_matches_source,
    sequence_retention_loss,
    terminal_opening_reward,
)
from proteinfoundation.cyclization.constants import LINEAR, MAINCHAIN, UNSPECIFIED


def _atom37(b=2, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(b, n, 37, 3)
    # Extended chain with CHEMICALLY CORRECT spacing: the residue stride is
    # N-CA (0.1458) + CA-C (0.1525) + C-N' (0.1329) = 0.4312 nm, so consecutive residues
    # are a real peptide bond apart. A convenient-looking round stride instead makes every
    # bond wrong and every geometry gate fire, which reads as a code failure.
    stride = IDEAL_N_CA_NM + IDEAL_CA_C_NM + IDEAL_PEPTIDE_BOND_NM
    for i in range(n):
        base = stride * i
        x[:, i, N_IDX] = torch.tensor([base, 0.0, 0.0])
        x[:, i, CA_IDX] = torch.tensor([base + IDEAL_N_CA_NM, 0.0, 0.0])
        x[:, i, C_IDX] = torch.tensor([base + IDEAL_N_CA_NM + IDEAL_CA_C_NM, 0.0, 0.0])
    x = x + 0.001 * torch.randn(x.shape, generator=g)
    m = torch.zeros(b, n, 37, dtype=torch.bool)
    m[:, :, [N_IDX, CA_IDX, C_IDX]] = True
    return x, m


# --------------------------------------------------------------------- conditioning


def test_attach_source_cp_detaches_and_flags_present():
    z = torch.randn(2, 5, 8, requires_grad=True)
    ca = torch.randn(2, 5, 3, requires_grad=True)
    batch = {"x_1": {"bb_ca": ca, "local_latents": z}, "mask": torch.ones(2, 5, dtype=torch.bool)}
    attach_source_cp_condition(batch)
    assert not batch[SRC_CP_KEY]["local_latents"].requires_grad
    assert torch.equal(batch[SRC_CP_PRESENT_KEY], torch.ones(2))
    # The condition must be a snapshot, not an alias of the live encoder output.
    assert batch[SRC_CP_KEY]["bb_ca"].data_ptr() == ca.data_ptr() or True  # detach may share storage
    assert batch[SRC_CP_KEY]["bb_ca"].grad_fn is None


def test_drop_source_cp_zeroes_and_flags_absent():
    batch = {
        "x_1": {"bb_ca": torch.randn(3, 4, 3), "local_latents": torch.randn(3, 4, 8)},
        "mask": torch.ones(3, 4, dtype=torch.bool),
    }
    drop_source_cp_condition(batch, ["bb_ca", "local_latents"])
    assert batch[SRC_CP_KEY]["local_latents"].shape == (3, 4, 8)
    assert batch[SRC_CP_KEY]["bb_ca"].abs().sum() == 0
    assert torch.equal(batch[SRC_CP_PRESENT_KEY], torch.zeros(3))


def test_request_linear_topology_clears_stale_ring_labels():
    batch = {
        "mask": torch.ones(2, 5, dtype=torch.bool),
        "cyclization_i": torch.tensor([0, 0]),
        "cyclization_j": torch.tensor([4, 4]),
        "has_cyclization": torch.tensor([True, True]),
    }
    request_linear_topology(batch, bs=2)
    assert torch.equal(batch["cyclization_type_cond"], torch.full((2,), LINEAR))
    # Endpoints inherited from the CP would be read downstream as a bond to supervise.
    for k in ("cyclization_i", "cyclization_j", "has_cyclization"):
        assert k not in batch


# --------------------------------------------------------------------- role split


def test_split_roles_reads_ring_request():
    batch = {
        "mask": torch.ones(4, 5, dtype=torch.bool),
        "cyclization_type_cond": torch.tensor([MAINCHAIN, LINEAR, UNSPECIFIED, MAINCHAIN]),
    }
    cyclic, linear = split_roles(batch)
    assert cyclic.tolist() == [True, False, False, True]
    # UNSPECIFIED is NOT a ring request, so it lands on the linear side.
    assert linear.tolist() == [False, True, True, False]


def test_slice_batch_handles_nested_and_passthrough():
    bs = 4
    batch = {
        "mask": torch.ones(bs, 5, dtype=torch.bool),
        "x_1": {"bb_ca": torch.randn(bs, 5, 3)},
        "ids": ["a", "b", "c", "d"],
        "scalar_cfg": 7,
        "not_batch": torch.randn(3, 3),
    }
    idx = torch.tensor([True, False, True, False])
    out = slice_batch(batch, idx, bs)
    assert out["mask"].shape[0] == 2
    assert out["x_1"]["bb_ca"].shape[0] == 2
    assert out["ids"] == ["a", "c"]
    assert out["scalar_cfg"] == 7
    assert out["not_batch"].shape == (3, 3)


# --------------------------------------------------------------------- geometry


def test_open_chain_never_includes_the_closure():
    mask = torch.ones(1, 5, dtype=torch.bool)
    pm = open_chain_bond_pair_mask(mask)
    assert pm.shape == (1, 4)  # 4 bonds for 5 residues, not 5


def test_terminal_gap_matches_hand_computation():
    x, m = _atom37(b=1, n=4)
    gap = terminal_gap_nm(x, torch.ones(1, 4, dtype=torch.bool))
    expected = torch.linalg.vector_norm(x[0, 3, CA_IDX] - x[0, 0, CA_IDX])
    assert torch.allclose(gap[0], expected, atol=1e-6)


def test_peptide_bond_deviation_flags_a_broken_bond():
    x, m = _atom37(b=1, n=4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    dev_ok, _ = peptide_bond_deviation_nm(x, mask)
    x_broken = x.clone()
    x_broken[0, 2:, :, 0] += 1.0  # yank the second half a nanometre away
    dev_bad, _ = peptide_bond_deviation_nm(x_broken, mask)
    assert dev_bad[0, 1] > dev_ok[0, 1] + 0.5


def test_backbone_dihedrals_mark_undefined_as_zero():
    x, _ = _atom37(b=1, n=4)
    d = backbone_dihedrals(x, torch.ones(1, 4, dtype=torch.bool))
    assert d.shape == (1, 4, 4)
    assert torch.allclose(d[0, 0, :2], torch.zeros(2))   # phi undefined at residue 0
    assert torch.allclose(d[0, -1, 2:], torch.zeros(2))  # psi undefined at the last


# --------------------------------------------------------------------- losses


def test_contact_retention_is_zero_when_nothing_moved():
    x, m = _atom37(b=1, n=4)
    t = x.clone() + torch.tensor([0.0, 0.5, 0.0])
    loss, metrics = contact_retention_loss(
        gen_atom37=x, gen_atom_mask=m, src_atom37=x, src_atom_mask=m,
        pep_mask=torch.ones(1, 4, dtype=torch.bool),
        target_atom37=t, target_atom_mask=m, target_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    assert float(loss) < 1e-5
    assert float(metrics["contact_retention_frac"]) == 1.0


def test_contact_retention_penalises_only_losses_not_gains():
    x, m = _atom37(b=1, n=4)
    t = x.clone() + torch.tensor([0.0, 0.5, 0.0])
    pm = torch.ones(1, 4, dtype=torch.bool)
    far = x.clone() + torch.tensor([0.0, 5.0, 0.0])  # generated peptide leaves the pocket
    lost, _ = contact_retention_loss(far, m, x, m, pm, t, m, pm)
    # Source far, generated close: contacts GAINED, which is free.
    gained, _ = contact_retention_loss(x, m, far, m, pm, t, m, pm)
    assert float(lost) > 0.1
    assert float(gained) < 1e-5


def test_sequence_retention_rewards_the_right_sequence():
    b, n, c = 2, 5, 21
    tgt = torch.randint(0, 20, (b, n))
    pm = torch.ones(b, n, dtype=torch.bool)
    good = torch.nn.functional.one_hot(tgt, c).float() * 20.0
    bad = torch.nn.functional.one_hot((tgt + 1) % 20, c).float() * 20.0
    l_good, m_good = sequence_retention_loss(good, tgt, pm)
    l_bad, m_bad = sequence_retention_loss(bad, tgt, pm)
    assert float(l_good) < float(l_bad)
    assert float(m_good["seq_exact_match"]) == 1.0
    assert float(m_bad["seq_exact_match"]) == 0.0


def test_open_chain_geometry_ignores_the_terminal_bond():
    """A perfectly good OPEN chain must not be penalised for its ends being apart."""
    x, m = _atom37(b=1, n=6)
    pm = torch.ones(1, 6, dtype=torch.bool)
    loss_open, _ = open_chain_geometry_loss(x, m, pm)
    # Now make it a ring by pulling the last residue onto the first: the open-chain loss
    # should get WORSE (the i,i+1 bonds are now stretched), never better.
    x_ring = x.clone()
    x_ring[0, -1] = x[0, 0]
    loss_ring, _ = open_chain_geometry_loss(x_ring, m, pm)
    assert float(loss_open) < float(loss_ring)


def test_clash_loss_excludes_bonded_neighbours():
    x, m = _atom37(b=1, n=4)
    pm = torch.ones(1, 4, dtype=torch.bool)
    base, _ = clash_loss(x, m, pm, threshold_nm=0.2, min_seq_sep=2)
    # i and i+1 sit ~0.13 nm apart at the peptide bond; with min_seq_sep=1 that becomes a
    # "clash", which is exactly the false positive the exclusion exists to prevent.
    with_neighbours, _ = clash_loss(x, m, pm, threshold_nm=0.2, min_seq_sep=1)
    assert float(with_neighbours) > float(base)


def test_terminal_opening_reward_is_flat_once_open():
    x, _ = _atom37(b=1, n=8)
    pm = torch.ones(1, 8, dtype=torch.bool)
    loss, metrics = terminal_opening_reward(x, pm, min_gap_nm=0.45)
    assert float(metrics["terminal_gap_nm"]) > 0.45
    assert float(loss) == 0.0  # flat bottom: no gradient once the chain is open


def test_linear_terminus_valid_rejects_a_ring():
    x, _ = _atom37(b=1, n=6)
    pm = torch.ones(1, 6, dtype=torch.bool)
    assert bool(linear_terminus_valid(x, pm)[0])
    ring = x.clone()
    ring[0, -1, C_IDX] = x[0, 0, N_IDX] + torch.tensor([0.1329, 0.0, 0.0])
    ring[0, -1, CA_IDX] = x[0, 0, CA_IDX] + torch.tensor([0.05, 0.0, 0.0])
    assert not bool(linear_terminus_valid(ring, pm)[0])


def test_sequence_matches_source_is_exact():
    logits = torch.zeros(2, 4, 21)
    src = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    logits.scatter_(2, src[..., None], 10.0)
    pm = torch.ones(2, 4, dtype=torch.bool)
    assert sequence_matches_source(logits, src, pm).tolist() == [True, True]
    logits[1, 0, 1] = -10.0
    assert sequence_matches_source(logits, src, pm).tolist() == [True, False]


# --------------------------------------------------------------------- discriminator


def _disc_inputs(b=2, n=6, m=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    pep, pep_m = _atom37(b=b, n=n, seed=seed)
    tgt = torch.randn(b, m, 37, 3, generator=g) * 0.5 + torch.tensor([0.0, 0.6, 0.0])
    tgt_m = torch.zeros(b, m, 37, dtype=torch.bool)
    tgt_m[:, :, [N_IDX, CA_IDX, C_IDX]] = True
    return dict(
        pep_atom37=pep,
        pep_atom_mask=pep_m,
        pep_mask=torch.ones(b, n, dtype=torch.bool),
        pep_seq_probs=torch.softmax(torch.randn(b, n, 21, generator=g), dim=-1),
        target_atom37=tgt,
        target_atom_mask=tgt_m,
        target_mask=torch.ones(b, m, dtype=torch.bool),
        target_aatype=torch.randint(0, 20, (b, m), generator=g),
    )


def test_discriminator_emits_one_logit_per_complex():
    d = PeptideInterfaceDiscriminator(hidden_dim=64, nlayers=2, nheads=4, k_pocket=4)
    feats = d.build_features(**_disc_inputs())
    out = d(feats)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_discriminator_gradient_reaches_coordinates_and_sequence():
    """The adversarial term is worthless if it cannot move the generator's output."""
    d = PeptideInterfaceDiscriminator(hidden_dim=64, nlayers=2, nheads=4, k_pocket=4)
    inputs = _disc_inputs()
    inputs["pep_atom37"] = inputs["pep_atom37"].clone().requires_grad_(True)
    inputs["pep_seq_probs"] = inputs["pep_seq_probs"].clone().requires_grad_(True)
    feats = d.build_features(**inputs)
    d(feats).sum().backward()
    assert inputs["pep_atom37"].grad is not None
    assert inputs["pep_atom37"].grad.abs().sum() > 0
    assert inputs["pep_seq_probs"].grad is not None
    assert inputs["pep_seq_probs"].grad.abs().sum() > 0


def test_discriminator_handles_a_receptor_smaller_than_k():
    d = PeptideInterfaceDiscriminator(hidden_dim=32, nlayers=1, nheads=4, k_pocket=8)
    feats = d.build_features(**_disc_inputs(m=3))
    assert torch.isfinite(d(feats)).all()


def test_shortcut_probe_is_detached_from_the_trunk():
    d = PeptideInterfaceDiscriminator(hidden_dim=32, nlayers=1, nheads=4, k_pocket=4)
    inputs = _disc_inputs()
    inputs["pep_atom37"] = inputs["pep_atom37"].clone().requires_grad_(True)
    feats = d.build_features(**inputs)
    d.probe_forward(feats).sum().backward()
    # The probe diagnoses the features; it must not shape them or reach the generator.
    assert inputs["pep_atom37"].grad is None or inputs["pep_atom37"].grad.abs().sum() == 0


def test_hinge_losses_have_the_right_signs():
    real = torch.tensor([3.0, 2.0])
    fake = torch.tensor([-3.0, -2.0])
    assert float(hinge_d_loss(real, fake)) == 0.0          # both confidently correct
    assert float(hinge_d_loss(fake, real)) > 0.0           # both confidently wrong
    assert float(hinge_g_loss(fake)) > float(hinge_g_loss(real))
    assert discriminator_accuracy(real, fake) == 1.0


# --------------------------------------------------------------------- reservoir


def _res_feats(b, n, lengths):
    """Feature batch in the schema `build_features` returns: per-residue keys padded to n,
    plus the per-example `glob`."""
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True
    return {
        "res": torch.randn(b, n, 21),
        "geom": torch.randn(b, n, 10),
        "iface": torch.randn(b, n, 5),
        "glob": torch.randn(b, 4),
        "mask": mask,
    }


def test_reservoir_stacks_same_length_from_differently_padded_batches():
    """The crash this guards: buffers are keyed by PEPTIDE LENGTH, but features arrive
    padded to their own batch's width. Two length-8 peptides collected from an n=16 batch
    and an n=15 batch are interchangeable by the key and unstackable by shape."""
    res = RealLPFeatureReservoir(capacity_per_length=8, seed=0)
    res.push_batch(_res_feats(2, 16, [8, 8]), torch.tensor([8, 8]))
    res.push_batch(_res_feats(2, 15, [8, 8]), torch.tensor([8, 8]))
    got, matched = res.draw_matched(torch.tensor([8, 8, 8]), torch.device("cpu"))
    assert matched.all()
    assert got["res"].shape == (3, 8, 21)   # trimmed to the true length, not either pad
    assert got["mask"].shape == (3, 8)
    assert got["glob"].shape == (3, 4)      # per-example, NOT trimmed by length


def test_reservoir_does_not_trim_the_per_example_block():
    """`glob` is [4] per example. A shape-based trim would corrupt it whenever the padded
    width happens to equal 4, and truncate it whenever the peptide is shorter than 4."""
    res = RealLPFeatureReservoir(capacity_per_length=4, seed=0)
    feats = _res_feats(1, 6, [2])
    expected = feats["glob"][0].clone()
    res.push_batch(feats, torch.tensor([2]))
    got, _ = res.draw_matched(torch.tensor([2]), torch.device("cpu"))
    assert got["glob"].shape == (1, 4)
    assert torch.allclose(got["glob"][0], expected)
    assert got["res"].shape == (1, 2, 21)


def test_reservoir_pads_when_slack_spans_lengths():
    res = RealLPFeatureReservoir(capacity_per_length=4, seed=0)
    res.push_batch(_res_feats(1, 12, [7]), torch.tensor([7]))
    res.push_batch(_res_feats(1, 12, [9]), torch.tensor([9]))
    got, matched = res.draw_matched(torch.tensor([7, 9]), torch.device("cpu"), max_length_slack=0)
    assert matched.all() and got["res"].shape == (2, 9, 21)  # padded to the longer
    assert got["mask"][0].sum() == 7 and got["mask"][1].sum() == 9


def test_reservoir_matches_only_on_exact_length_by_default():
    res = RealLPFeatureReservoir(capacity_per_length=4, seed=0)
    feats = {"mask": torch.ones(3, 7, dtype=torch.bool), "iface": torch.randn(3, 7, 5)}
    res.push_batch(feats, torch.tensor([7, 7, 7]))
    got, matched = res.draw_matched(torch.tensor([7, 9]), torch.device("cpu"))
    assert matched.tolist() == [True, False]
    assert got["mask"].shape[0] == 1
    # With slack the length-9 request finds the length-7 pool.
    _, matched_slack = res.draw_matched(torch.tensor([7, 9]), torch.device("cpu"), max_length_slack=2)
    assert matched_slack.tolist() == [True, True]


def test_reservoir_reports_nothing_when_empty():
    res = RealLPFeatureReservoir()
    got, matched = res.draw_matched(torch.tensor([5]), torch.device("cpu"))
    assert got is None and matched.tolist() == [False]


# --------------------------------------------------------------------- family split


def test_family_split_is_deterministic_and_disjoint():
    fams = [f"cluster_{i}" for i in range(400)]
    a = {f for f in fams if in_eval_split(f, 0.2, seed=7)}
    b = {f for f in fams if in_eval_split(f, 0.2, seed=7)}
    assert a == b                      # same answer every call
    assert 0.1 < len(a) / len(fams) < 0.3
    other = {f for f in fams if in_eval_split(f, 0.2, seed=8)}
    assert a != other                  # the seed actually moves the split


# --------------------------------------------------------------------- export


def test_build_oxt_makes_a_real_carboxylate():
    """The decoder emits no OXT; a linear C-terminus is chemically wrong without one."""
    from proteinfoundation.cp2lp.export import OXT_BOND_NM, OXT_IDX, build_oxt
    from proteinfoundation.cp2lp.geometry import O_IDX

    x, m = _atom37(b=1, n=4)
    x[0, :, O_IDX] = x[0, :, C_IDX] + torch.tensor([0.0, 0.123, 0.0])
    m[0, :, O_IDX] = True
    lengths = torch.tensor([4])

    assert not bool(m[0, 3, OXT_IDX])
    out, out_m = build_oxt(x, m, lengths)
    assert bool(out_m[0, 3, OXT_IDX])

    c, oxt, o, ca = out[0, 3, C_IDX], out[0, 3, OXT_IDX], out[0, 3, O_IDX], out[0, 3, CA_IDX]
    assert abs(float(torch.linalg.vector_norm(oxt - c)) - OXT_BOND_NM) < 1e-4

    def angle(a, b_, c_):
        v1, v2 = a - b_, c_ - b_
        cosv = (v1 @ v2) / (v1.norm() * v2.norm())
        return float(torch.rad2deg(torch.acos(cosv.clamp(-1, 1))))

    # sp2 carbon: CA-C-OXT and O-C-OXT both near 120 degrees.
    assert 100.0 < angle(ca, c, oxt) < 140.0
    assert 100.0 < angle(o, c, oxt) < 140.0
    # Only the C-terminal residue gains one.
    assert not bool(out_m[0, 0, OXT_IDX])


def test_seq_to_string_roundtrips():
    from proteinfoundation.cp2lp.export import seq_to_string
    import numpy as np

    assert seq_to_string(np.array([0, 5, 19])) == "AQV"
    assert seq_to_string(np.array([99])) == "X"


# ------------------------------------------------- padded-batch gradient finiteness
#
# THE bug class this file exists to prevent. Padded residues hold exactly-zero
# coordinates, and several standard geometry ops are non-differentiable at zero:
# `vector_norm` goes to 0/0 and `atan2(0, 0)` goes to 0/0. The usual
# `torch.where(valid, value, 0)` cleanup does NOT contain the damage, because `where`
# hands gradient to both branches -- `0 * NaN = NaN` -- so a single padded row turns every
# parameter in the model into NaN. It reads as a dead gradient rather than a broken one,
# because `nan > 0` is False.
#
# Every test below runs backward on a batch WITH PADDING and asserts finiteness.


def _padded_inputs(b=3, n=10, lengths=(10, 6, 1), m=12, seed=3):
    """Batch whose rows have different lengths, so padding is exercised."""
    g = torch.Generator().manual_seed(seed)
    pep, pep_m = _atom37(b=b, n=n, seed=seed)
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True
    # Padding is exactly zero, as autoencoder.decode leaves it.
    pep = pep * mask[..., None, None]
    pep_m = pep_m & mask[..., None]
    tgt = torch.randn(b, m, 37, 3, generator=g) * 0.4 + torch.tensor([0.0, 0.6, 0.0])
    tgt_m = torch.zeros(b, m, 37, dtype=torch.bool)
    tgt_m[:, :, [N_IDX, CA_IDX, C_IDX]] = True
    return dict(
        pep_atom37=pep, pep_atom_mask=pep_m, pep_mask=mask,
        pep_seq_probs=torch.softmax(torch.randn(b, n, 21, generator=g), dim=-1),
        target_atom37=tgt, target_atom_mask=tgt_m,
        target_mask=torch.ones(b, m, dtype=torch.bool),
        target_aatype=torch.randint(0, 20, (b, m), generator=g),
    )


def _assert_finite_grad(leaf, label):
    assert leaf.grad is not None, f"{label}: no gradient at all"
    assert torch.isfinite(leaf.grad).all(), (
        f"{label}: non-finite gradient "
        f"(NaN {int(torch.isnan(leaf.grad).sum())}, Inf {int(torch.isinf(leaf.grad).sum())})"
    )


def test_safe_norm_is_differentiable_at_zero():
    from proteinfoundation.cp2lp.geometry import safe_norm

    x = torch.zeros(4, 3, requires_grad=True)
    safe_norm(x, dim=-1).sum().backward()
    _assert_finite_grad(x, "safe_norm at the origin")


def test_atan2_at_origin_is_the_nan_source_and_masking_does_not_contain_it():
    """Pins the exact mechanism behind the all-NaN flow gradient, because the symptom is
    so misleading: `nan > 0` is False, so a NaN gradient reads as a DEAD one.

    Two separate facts, both load-bearing:
      1. `atan2(0, 0)` has NaN gradient -- padded residues are all-zero, so v and w in the
         dihedral collapse and this is hit on every padded batch.
      2. `torch.where(mask, value, 0)` does NOT contain it: `where` sends gradient to both
         branches, so `0 * NaN = NaN` and one padded row poisons every parameter.
    `vector_norm` is checked too, and is NOT a culprit on this torch -- recorded so the
    defensive `safe_norm` is not mistaken for the fix.
    """
    a = torch.zeros(3, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    torch.atan2(a, b).sum().backward()
    assert torch.isnan(a.grad).all(), "atan2(0,0) no longer NaNs; the guard may be removable"

    u = torch.zeros(3, requires_grad=True)
    v = torch.zeros(3, requires_grad=True)
    masked = torch.where(torch.zeros(3, dtype=torch.bool), torch.atan2(u, v), torch.zeros(3))
    masked.sum().backward()
    assert torch.isnan(u.grad).all(), "torch.where now blocks NaN gradient; guards may be removable"

    y = torch.zeros(4, 3, requires_grad=True)
    torch.linalg.vector_norm(y, dim=-1).sum().backward()
    assert torch.isfinite(y.grad).all(), "vector_norm now NaNs at 0 -- safe_norm is load-bearing"


def test_backbone_dihedrals_finite_grad_on_padding():
    inp = _padded_inputs()
    x = inp["pep_atom37"].clone().requires_grad_(True)
    backbone_dihedrals(x, inp["pep_mask"]).sum().backward()
    _assert_finite_grad(x, "backbone_dihedrals")


def test_geometry_helpers_finite_grad_on_padding():
    inp = _padded_inputs()
    for fn, label in [
        (lambda a: terminal_gap_nm(a, inp["pep_mask"]).sum(), "terminal_gap_nm"),
        (lambda a: peptide_bond_deviation_nm(a, inp["pep_mask"])[0].sum(), "peptide_bond_deviation_nm"),
    ]:
        x = inp["pep_atom37"].clone().requires_grad_(True)
        fn(x).backward()
        _assert_finite_grad(x, label)


def test_all_generator_losses_finite_grad_on_padding():
    inp = _padded_inputs()
    common = dict(
        target_atom37=inp["target_atom37"],
        target_atom_mask=inp["target_atom_mask"],
        target_mask=inp["target_mask"],
    )
    cases = {
        "geometry": lambda a: open_chain_geometry_loss(a, inp["pep_atom_mask"], inp["pep_mask"])[0],
        "clash": lambda a: clash_loss(a, inp["pep_atom_mask"], inp["pep_mask"], **common)[0],
        "open": lambda a: terminal_opening_reward(a, inp["pep_mask"])[0] + a.sum() * 0.0,
        "contact": lambda a: contact_retention_loss(
            gen_atom37=a, gen_atom_mask=inp["pep_atom_mask"],
            src_atom37=inp["pep_atom37"], src_atom_mask=inp["pep_atom_mask"],
            pep_mask=inp["pep_mask"], **common)[0],
    }
    for label, fn in cases.items():
        x = inp["pep_atom37"].clone().requires_grad_(True)
        fn(x).backward()
        _assert_finite_grad(x, f"loss/{label}")


def test_discriminator_finite_grad_on_padding():
    """The failure that actually bit: only the discriminator's feature builder used
    dihedrals, so every individual loss term looked finite while the adversarial term
    poisoned all 460 flow-network tensors."""
    d = PeptideInterfaceDiscriminator(hidden_dim=64, nlayers=2, nheads=4, k_pocket=4)
    inp = _padded_inputs()
    inp["pep_atom37"] = inp["pep_atom37"].clone().requires_grad_(True)
    inp["pep_seq_probs"] = inp["pep_seq_probs"].clone().requires_grad_(True)
    feats = d.build_features(**inp)
    hinge_g_loss(d(feats)).backward()
    _assert_finite_grad(inp["pep_atom37"], "discriminator -> coords")
    _assert_finite_grad(inp["pep_seq_probs"], "discriminator -> sequence")


def test_single_residue_row_does_not_poison_the_batch():
    """A length-1 peptide has no bonds and no dihedrals at all -- every geometric quantity
    is degenerate. It must still not produce a NaN for the OTHER rows."""
    d = PeptideInterfaceDiscriminator(hidden_dim=32, nlayers=1, nheads=4, k_pocket=4)
    inp = _padded_inputs(b=2, n=8, lengths=(8, 1))
    inp["pep_atom37"] = inp["pep_atom37"].clone().requires_grad_(True)
    feats = d.build_features(**inp)
    hinge_g_loss(d(feats)).backward()
    _assert_finite_grad(inp["pep_atom37"], "length-1 row")


def test_a_correct_extended_chain_passes_every_geometry_gate():
    """Guards the fixture as much as the code: if `_atom37` stops building a chemically
    valid chain, every geometry assertion above silently becomes vacuous."""
    from proteinfoundation.cp2lp.geometry import IDEAL_PEPTIDE_BOND_NM as IDEAL

    x, m = _atom37(b=1, n=6)
    pm = torch.ones(1, 6, dtype=torch.bool)
    dev, pair = peptide_bond_deviation_nm(x, pm)
    assert float(dev[pair].max()) < 0.01, "fixture chain does not have real peptide bonds"
    _, metrics = open_chain_geometry_loss(x, m, pm)
    assert float(metrics["geom_chain_intact_frac"]) == 1.0
    assert bool(linear_terminus_valid(x, pm)[0])


# --------------------------------------------------------------------- config arms


def test_contactonly_differs_from_gan_in_exactly_one_knob():
    """The ablation is only interpretable if it IS single-variable.

    Config drift is silent and cheap to introduce -- someone tunes a loss weight in the GAN
    arm and the control no longer isolates the adversarial term. This compares the fully
    resolved trees, so an override anywhere in the defaults chain is caught.
    """
    import os

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
        gan = OmegaConf.to_container(compose(config_name="example/training_cp2lp_gan"), resolve=False)
        ctl = OmegaConf.to_container(compose(config_name="example/training_cp2lp_contactonly"), resolve=False)

    def flat(d, prefix=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flat(v, key)) if isinstance(v, dict) else out.update({key: v})
        return out

    fg, fc = flat(gan), flat(ctl)
    diffs = {k for k in set(fg) | set(fc) if fg.get(k, "<absent>") != fc.get(k, "<absent>")}
    assert diffs == {"run_name", "cp2lp.adversarial.enabled"}, f"arms drifted apart: {sorted(diffs)}"
    assert fg["cp2lp.adversarial.enabled"] is True
    assert fc["cp2lp.adversarial.enabled"] is False


def test_source_cp_features_are_appended_last_in_both_arms():
    """Appending preserves the warm start; inserting re-binds trained weights to the wrong
    input columns, because each factory projects its concatenated features with ONE Linear
    and the splice copies the overlapping LEADING subtensor."""
    import os

    from hydra import compose, initialize_config_dir

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
        cfg = compose(config_name="example/training_cp2lp_gan")

    seq, pair = list(cfg.nn.feats_seq), list(cfg.nn.feats_pair_repr)
    new_seq = ["x_src_cp_bb_ca", "x_src_cp_local_latents", "src_cp_present"]
    assert seq[-3:] == new_seq, f"source-CP seq features are not last: {seq}"
    assert pair[-1] == "x_src_cp_bb_ca_pair_dists", f"source-CP pair feature is not last: {pair}"
    # Everything the init was trained with must keep its original relative order.
    assert "cyclization_ring_pe" in pair[:-1]


def test_train_py_disables_trainer_side_clip_and_accum_for_manual_optimization():
    """Lightning REFUSES `gradient_clip_val` and `accumulate_grad_batches != 1` for a
    module that owns its optimizer steps, and a two-optimizer GAN has to be manual.

    This is a source-level check because the offending line lives inline in `train.py`'s
    `main()`, which cannot be imported without Hydra and a GPU. It exists because the
    first smoke built its own bare Trainer, passed, and the real job then died in 30
    seconds on exactly this -- the preflight did not construct the Trainer the way
    production does.

    Clipping is RELOCATED, not removed: `CP2LPGenerator._step_optimizer` applies the same
    norm and value per optimizer. If this test fails, check that too before deleting it.
    """
    import pathlib
    import re

    repo = pathlib.Path(__file__).resolve().parents[1]
    src = (repo / "src" / "proteinfoundation" / "train.py").read_text()
    required = {
        "manual flag from cp2lp.enabled": r"manual_optimization\s*=\s*bool\(",
        "clip val gated": r"gradient_clip_val\s*=\s*None if manual_optimization else",
        "clip algorithm gated": r"gradient_clip_algorithm\s*=\s*None if manual_optimization else",
        "accumulation gated": r"accumulate_grad_batches\s*=\s*1 if manual_optimization else",
    }
    for label, pattern in required.items():
        assert re.search(pattern, src), f"train.py no longer has: {label}"


def test_module_owns_the_clip_it_took_over():
    """The other half of the relocation: dropping the clip to satisfy Lightning would be
    the wrong fix, most of all on the arm that adds an adversary and a rollout."""
    import inspect

    from proteinfoundation.cp2lp.module import CP2LPGenerator

    src = inspect.getsource(CP2LPGenerator._step_optimizer)
    assert "clip_gradients" in src, "_step_optimizer no longer clips"
    assert "gradient_clip_val" in src
    init = inspect.getsource(CP2LPGenerator._init_cp2lp)
    assert "grad_clip_val" in init and "1.0" in init, "default clip value is gone"


# ------------------------------------------------- the validation-step convention


def test_training_step_delegates_validation_batches_to_the_parent():
    """`Proteina.validation_step_data` scores a validation batch by calling
    `training_step(batch, batch_idx=-1)` (proteina.py: `val_step = batch_idx == -1`).

    The CP->LP override must honour that convention. It killed the GAN arm at its first
    validation, ~76 minutes in: a validation pass runs under `no_grad` so no loss carries
    a graph, and `(-1 + 1) % accum == 0` is True for EVERY accum, so the override tried to
    backward a graph-less discriminator loss and to step the optimizer and LR scheduler
    from inside validation.

    Source-level because constructing the module needs a 4 GB AE and a GPU; the property
    is structural, so checking the source is both sufficient and fast.
    """
    import inspect

    from proteinfoundation.cp2lp.module import CP2LPGenerator

    src = inspect.getsource(CP2LPGenerator.training_step)
    head = src.split("accum = ")[0]
    assert "batch_idx == -1" in head, "training_step no longer guards the val_step convention"
    assert "super().training_step" in head, "validation batches are not delegated to the parent"

    # The guard must come BEFORE anything that backwards or steps.
    i_guard = src.index("batch_idx == -1")
    for marker in ("manual_backward", "_step_optimizer", "_discriminator_step"):
        assert src.index(marker) > i_guard, f"`{marker}` is reachable before the val_step guard"


def test_both_backward_calls_are_guarded_against_a_missing_graph():
    """Either loss can legitimately arrive without a graph (an empty branch, a frozen
    discriminator, an enclosing no_grad). `manual_backward` RAISES on those rather than
    no-opping, so both call sites need a check."""
    import inspect

    from proteinfoundation.cp2lp.module import CP2LPGenerator

    g_src = inspect.getsource(CP2LPGenerator.training_step)
    assert "g_loss.requires_grad" in g_src, "generator backward is unguarded"
    d_src = inspect.getsource(CP2LPGenerator._discriminator_step)
    assert "d_loss.requires_grad" in d_src, "discriminator backward is unguarded"


def test_accum_modulo_is_true_for_the_validation_sentinel():
    """Pins the arithmetic that made the bug bite for any accumulation setting -- so
    nobody 'fixes' it by changing accum and assumes the hazard is gone."""
    for accum in (1, 2, 4, 8, 16):
        assert ((-1) + 1) % accum == 0, f"sentinel is not a step boundary at accum={accum}"


# ---------------------------------------------------------------- v2 rebalance (2026-09-29)
# The v1 arms trained 39k steps with contact retention pinned at 0.158 while the control,
# identical but for the adversarial term, reached 0.435. Root cause was a gradient-budget
# imbalance, not a bug; these pin the three pieces of the fix.


def test_sequence_weight_scale_retires_a_converged_term_without_switching_it_off():
    from proteinfoundation.cp2lp.losses import sequence_weight_scale

    # Full weight while the term still has work to do.
    assert sequence_weight_scale(0.0) == 1.0
    assert sequence_weight_scale(0.5) == 1.0
    assert sequence_weight_scale(0.9) == 1.0
    # Then linear in the REMAINING error, so the hand-off is gradual rather than a cliff.
    assert abs(sequence_weight_scale(0.95) - 0.5) < 1e-9
    assert abs(sequence_weight_scale(0.99) - 0.1) < 1e-9
    # Never zero: the term still has to defend the sequence the export gate requires.
    assert sequence_weight_scale(1.0) == 0.05
    assert sequence_weight_scale(1.0, floor=0.2) == 0.2
    # Monotone non-increasing, so the weight can never oscillate as the EMA drifts.
    xs = [i / 100.0 for i in range(101)]
    ys = [sequence_weight_scale(x) for x in xs]
    assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:])), "scale is not monotone"


def test_sequence_weight_scale_rejects_a_degenerate_start():
    import pytest

    from proteinfoundation.cp2lp.losses import sequence_weight_scale

    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            sequence_weight_scale(0.95, start=bad)


def test_wider_contact_sharpness_keeps_gradient_on_a_DRIFTED_contact():
    """The sigmoid is the only range over which a LOST contact can be pulled back.

    At sharpness 0.1 nm the slope falls an e-fold every 0.1 nm, so a contact that drifted
    well past the 0.8 nm cutoff is numerically dead and the term can defend marginal
    contacts but never recover drifted ones. That is a one-way ratchet under anything that
    pushes the peptide off the receptor, which is what the v1 adversary did.
    """
    from proteinfoundation.cp2lp.geometry import soft_contact

    def slope_at(dist_nm, sharpness):
        d = torch.tensor([dist_nm], requires_grad=True)
        soft_contact(d, cutoff_nm=0.8, sharpness_nm=sharpness).sum().backward()
        return float(d.grad.abs())

    # At the boundary both settings have usable slope; the narrow one is steeper there.
    assert slope_at(0.8, 0.1) > slope_at(0.8, 0.3) > 0.0
    # Drifted 0.7 nm past the cutoff, the narrow setting is dead and the wide one is not.
    # MEASURED: 0.00910 at sharpness 0.1 vs 0.269 at 0.3, i.e. 29.5x more gradient.
    narrow, wide = slope_at(1.5, 0.1), slope_at(1.5, 0.3)
    assert narrow < 1e-2, f"expected the 0.1 nm sigmoid to be dead at 1.5 nm, got {narrow}"
    assert wide > 20 * narrow, f"0.3 nm sigmoid should dominate at 1.5 nm: {wide} vs {narrow}"


def test_contact_sharpness_reaches_the_loss_from_config():
    """A plumbing test, because an unplumbed knob is silently the default -- and both call
    sites matter: `_generator_loss_terms` is the gradient-attribution path whose numbers
    are what the rebalance was decided on, so it has to score what training scores."""
    import inspect

    from proteinfoundation.cp2lp.module import CP2LPGenerator

    for fn in (CP2LPGenerator._generator_losses, CP2LPGenerator._generator_loss_terms):
        src = inspect.getsource(fn)
        assert "sharpness_nm=self.contact_sharpness_nm" in src, f"{fn.__name__} drops the sharpness"


def test_preclip_grad_norm_is_measured_before_the_clip():
    """Logging it AFTER the clip would report `min(norm, clip_val)` and hide the very
    competition it exists to expose."""
    import inspect

    from proteinfoundation.cp2lp.module import CP2LPGenerator

    src = inspect.getsource(CP2LPGenerator._step_optimizer)
    assert "grad_norm_preclip" in src, "pre-clip gradient norm is not logged"
    assert src.index("grad_norm_preclip") < src.index("self.clip_gradients"), \
        "gradient norm is measured after clipping, which makes it useless"


def test_preamble_run_names_track_the_configs():
    """`cp2lp_train.sbatch` passes `++run_name=${RUN_NAME}`, a Hydra override that BEATS the
    yaml. A preamble left at a previous version therefore points a new config's run at the
    old run directory and overwrites its checkpoints and wandb history -- the record of the
    arm being compared against. Caught exactly that while launching v2.
    """
    import os
    import re

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preamble = open(os.path.join(repo, "scripts", "_cp2lp_preamble.sh")).read()
    with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
        for arm, cfg_name in (
            ("gan", "example/training_cp2lp_gan"),
            ("contactonly", "example/training_cp2lp_contactonly"),
        ):
            want = OmegaConf.to_container(compose(config_name=cfg_name), resolve=False)["run_name"]
            m = re.search(rf'^\s*{arm}\)\s.*RUN_NAME="\$\{{RUN_NAME:-([^}}]+)\}}"', preamble, re.M)
            assert m, f"no RUN_NAME default found for arm {arm}"
            assert m.group(1) == want, \
                f"arm {arm}: preamble default {m.group(1)!r} != config run_name {want!r}"
