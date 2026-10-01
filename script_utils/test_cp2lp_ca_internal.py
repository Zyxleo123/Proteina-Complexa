"""Checks on the CA-trace internal-coordinate edit space.

The whole point of this parameterisation is a GUARANTEE -- that no reachable point is an
invalid backbone -- so the tests are written as attempts to violate it, not as spot checks.
The three that matter most:

* ``rebuild(extract(x)) == x`` exactly, because the editor's zero-initialised heads must still
  produce precisely the AE round trip of the source CP. An approximate identity would silently
  move every peptide on step one and invalidate the pretrained splice.
* an edit with a HUGE requested shape change still lands inside both native windows, because
  that is the property the measured failure needs.
* the same huge edit, expressed as a free Cartesian residual of matched size, lands far
  outside them -- the contrast is what makes the guarantee meaningful rather than tautological.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from proteinfoundation.cp2lp.ca_internal import (  # noqa: E402
    BEND_MAX_DEG,
    BEND_MIN_DEG,
    CA_CA_NATIVE_NM,
    apply_internal_residual,
    ca_to_internal,
    internal_to_ca,
    rotation_from_vector,
)

torch.manual_seed(0)


def _native_trace(b: int = 6, n: int = 14, seed: int = 0) -> torch.Tensor:
    """A random trace with native bond lengths and bend angles in the native window."""
    g = torch.Generator().manual_seed(seed)
    d = torch.full((b, n - 1), CA_CA_NATIVE_NM)
    theta = torch.deg2rad(torch.empty(b, n - 2).uniform_(85.0, 145.0, generator=g))
    tau = torch.empty(b, n - 3).uniform_(-math.pi, math.pi, generator=g)
    seed_pts = torch.zeros(b, 3, 3)
    seed_pts[:, 1, 0] = CA_CA_NATIVE_NM
    ang = theta[:, 0]
    seed_pts[:, 2, 0] = CA_CA_NATIVE_NM - CA_CA_NATIVE_NM * torch.cos(ang)
    seed_pts[:, 2, 1] = CA_CA_NATIVE_NM * torch.sin(ang)
    return internal_to_ca(d, theta, tau, seed_pts)


def _bonds(ca):
    return (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)


def _bends_deg(ca):
    a = ca[:, :-2] - ca[:, 1:-1]
    c = ca[:, 2:] - ca[:, 1:-1]
    a = a / a.norm(dim=-1, keepdim=True)
    c = c / c.norm(dim=-1, keepdim=True)
    return torch.rad2deg(torch.arccos((a * c).sum(-1).clamp(-1, 1)))


# ----------------------------------------------------------------- round trip


def test_round_trip_is_exact():
    ca = _native_trace(seed=1)
    i = ca_to_internal(ca)
    back = internal_to_ca(i["d"], i["theta"], i["tau"], i["seed"])
    err = (back - ca).abs().max()
    assert float(err) < 1e-5, f"round trip error {float(err)} nm"


def test_round_trip_exact_on_a_badly_broken_trace():
    """Extraction/rebuild must not assume validity -- it is applied to whatever arrives."""
    g = torch.Generator().manual_seed(2)
    ca = _native_trace(seed=3) + torch.randn(6, 14, 3, generator=g) * 0.17
    i = ca_to_internal(ca)
    back = internal_to_ca(i["d"], i["theta"], i["tau"], i["seed"])
    assert float((back - ca).abs().max()) < 1e-5


def test_zero_residual_is_the_identity():
    """The property the pretrained splice depends on."""
    ca = _native_trace(seed=4)
    b, n, _ = ca.shape
    out, _ = apply_internal_residual(
        ca, torch.ones(b, n, dtype=torch.bool),
        delta_theta=torch.zeros(b, n - 2), delta_tau=torch.zeros(b, n - 3),
        delta_trans=torch.zeros(b, 3), delta_rot=torch.zeros(b, 3),
    )
    err = (out - ca).abs().max()
    assert float(err) < 1e-5, f"zero residual moved the trace by {float(err)} nm"


def test_rotation_at_zero_is_exactly_identity():
    R = rotation_from_vector(torch.zeros(4, 3))
    assert torch.allclose(R, torch.eye(3).expand(4, 3, 3), atol=0.0), R


def test_rotation_is_a_rotation():
    g = torch.Generator().manual_seed(5)
    R = rotation_from_vector(torch.randn(8, 3, generator=g))
    eye = torch.eye(3).expand(8, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(8), atol=1e-5)


# ------------------------------------------------------- the guarantee itself


def test_a_huge_edit_still_lands_inside_both_native_windows():
    """THE POINT. A shape request far larger than anything training would produce must still
    come out as a valid backbone, in bond length AND bend angle."""
    ca = _native_trace(b=32, n=16, seed=6)
    b, n, _ = ca.shape
    g = torch.Generator().manual_seed(7)
    out, diag = apply_internal_residual(
        ca, torch.ones(b, n, dtype=torch.bool),
        delta_theta=torch.randn(b, n - 2, generator=g) * 2.0,   # radians: enormous
        delta_tau=torch.randn(b, n - 3, generator=g) * 2.0,
        delta_trans=torch.randn(b, 3, generator=g) * 1.0,
        delta_rot=torch.randn(b, 3, generator=g) * 1.0,
    )
    bd = _bonds(out)
    assert float((bd - CA_CA_NATIVE_NM).abs().max()) < 1e-5, "bond lengths moved"
    bn = _bends_deg(out)
    assert float(bn.min()) >= BEND_MIN_DEG - 0.5, f"bend floor violated: {float(bn.min())}"
    assert float(bn.max()) <= BEND_MAX_DEG + 0.5, f"bend ceiling violated: {float(bn.max())}"
    assert float(diag["internal_bend_clamp_frac"]) > 0.1, "clamp should be active at this size"


def test_the_contrast_a_cartesian_residual_of_matched_size_destroys_the_trace():
    """Without this, the guarantee above is untestable: it could hold simply because the edit
    was small. Matched RMS displacement in Cartesian space, and the trace is wrecked."""
    ca = _native_trace(b=32, n=16, seed=8)
    b, n, _ = ca.shape
    g = torch.Generator().manual_seed(9)
    out_int, _ = apply_internal_residual(
        ca, torch.ones(b, n, dtype=torch.bool),
        delta_theta=torch.randn(b, n - 2, generator=g) * 0.5,
        delta_tau=torch.randn(b, n - 3, generator=g) * 0.5,
    )
    moved = (out_int - ca).pow(2).sum(-1).mean().sqrt()      # RMS displacement achieved
    cart = ca + torch.randn(ca.shape, generator=g) * (moved / math.sqrt(3))
    bi, bc = _bonds(out_int), _bonds(cart)
    assert float((bi - CA_CA_NATIVE_NM).abs().max()) < 1e-5
    assert float((bc - CA_CA_NATIVE_NM).abs().max()) > 0.05, (
        "the Cartesian control should break bonds badly; "
        f"max deviation was only {float((bc - CA_CA_NATIVE_NM).abs().max())}"
    )
    assert float(_bends_deg(cart).min()) < BEND_MIN_DEG, "Cartesian control should break bends"


def test_the_edit_space_contains_ring_opening():
    """The task is ring opening, so the edit space has to contain it.

    Extension is a JOINT (theta, tau) move, not a torsion-only one: a CA trace with all
    torsions trans but bend angles near 90 deg is a planar zigzag that curls rather than
    extends (measured -- it made the gap SHORTER, 1.63 -> 0.42 nm, which is what first failed
    this test). The extended state is the beta-strand corner, theta ~125 deg with tau ~180,
    and both are inside the clamp window, which is the property being checked.
    """
    ca = _native_trace(b=8, n=16, seed=10)
    b, n, _ = ca.shape
    i = ca_to_internal(ca)
    want_theta = torch.full_like(i["theta"], math.radians(120.0))
    want_tau = torch.full_like(i["tau"], math.pi)   # trans = extended, see the helix test
    out, diag = apply_internal_residual(
        ca, torch.ones(b, n, dtype=torch.bool),
        delta_theta=want_theta - i["theta"], delta_tau=want_tau - i["tau"],
    )
    gap_before = float((ca[:, -1] - ca[:, 0]).norm(dim=-1).median())
    gap_after = float((out[:, -1] - out[:, 0]).norm(dim=-1).median())
    assert gap_after > gap_before, f"{gap_before} -> {gap_after}"
    # a 16-mer beta-strand rises ~0.33 nm/residue, so ~4.5 nm end to end
    assert gap_after > 4.0, f"extended 16-mer should exceed 4 nm, got {gap_after}"
    # and it is still a valid backbone, which is the whole point
    assert float((_bonds(out) - CA_CA_NATIVE_NM).abs().max()) < 1e-5
    bn = _bends_deg(out)
    assert float(bn.min()) >= BEND_MIN_DEG - 0.5 and float(bn.max()) <= BEND_MAX_DEG + 0.5
    assert float(diag["internal_bend_clamp_frac"]) == 0.0, "120 deg should need no clamping"


def test_bond_lengths_move_only_when_delta_d_is_passed():
    ca = _native_trace(b=8, n=12, seed=11)
    b, n, _ = ca.shape
    kw = dict(delta_theta=torch.zeros(b, n - 2), delta_tau=torch.zeros(b, n - 3))
    m = torch.ones(b, n, dtype=torch.bool)
    out_no, _ = apply_internal_residual(ca, m, **kw)
    assert float((_bonds(out_no) - CA_CA_NATIVE_NM).abs().max()) < 1e-5
    out_yes, _ = apply_internal_residual(ca, m, delta_d=torch.full((b, n - 1), 0.01), **kw)
    assert float((_bonds(out_yes) - CA_CA_NATIVE_NM).abs().max()) > 5e-3
    # ...and even then it is clamped into the native window
    out_wild, _ = apply_internal_residual(ca, m, delta_d=torch.full((b, n - 1), 5.0), **kw)
    assert float(_bonds(out_wild).max()) <= 0.400 + 1e-5


def test_secondary_structure_corners_reproduce_published_rises():
    """The convention check that actually validates (theta, tau) against real protein geometry.

    A CA trace's secondary structure is fixed by the (bend, virtual torsion) pair, and the
    rise per residue is a measured number for each: ~1.5 A for an alpha-helix at
    (theta ~91 deg, tau ~50 deg) and ~3.3 A for a beta-strand at (theta ~120, tau ~180). If
    the dihedral convention were offset -- which it WAS, by exactly pi, until the negation in
    `_dihedral` was fixed -- these two corners would land on the wrong conformations and this
    test is what notices.
    """
    n = 16
    def rise(theta_deg, tau_deg):
        th = torch.full((1, n - 2), math.radians(theta_deg))
        ta = torch.full((1, n - 3), math.radians(tau_deg))
        d = torch.full((1, n - 1), CA_CA_NATIVE_NM)
        seed = torch.zeros(1, 3, 3)
        seed[0, 1, 0] = CA_CA_NATIVE_NM
        a = math.radians(theta_deg)
        seed[0, 2, 0] = CA_CA_NATIVE_NM * (1.0 - math.cos(a))
        seed[0, 2, 1] = CA_CA_NATIVE_NM * math.sin(a)
        ca = internal_to_ca(d, th, ta, seed)
        return float((ca[0, -1] - ca[0, 0]).norm()) / (n - 1)

    helix = rise(91.0, 50.0)
    strand = rise(120.0, 180.0)
    assert 0.13 < helix < 0.17, f"alpha-helix rise {helix * 10:.2f} A, expected ~1.5"
    assert 0.31 < strand < 0.36, f"beta-strand rise {strand * 10:.2f} A, expected ~3.3"
    assert strand > 2 * helix, "a strand must be far more extended than a helix"


def test_extended_chain_is_near_trans_not_near_zero():
    """Guards the pi offset directly: the most extended torsion must be ~180, not ~0."""
    n = 14
    def gap(tau_deg):
        th = torch.full((1, n - 2), math.radians(120.0))
        ta = torch.full((1, n - 3), math.radians(tau_deg))
        d = torch.full((1, n - 1), CA_CA_NATIVE_NM)
        seed = torch.zeros(1, 3, 3)
        seed[0, 1, 0] = CA_CA_NATIVE_NM
        a = math.radians(120.0)
        seed[0, 2, 0] = CA_CA_NATIVE_NM * (1.0 - math.cos(a))
        seed[0, 2, 1] = CA_CA_NATIVE_NM * math.sin(a)
        ca = internal_to_ca(d, th, ta, seed)
        return float((ca[0, -1] - ca[0, 0]).norm())
    assert gap(180.0) > 3.5 * gap(0.0), f"trans {gap(180.0):.2f} vs cis {gap(0.0):.2f} nm"


def test_the_clamp_is_source_relative_so_identity_survives_an_out_of_window_source():
    """Encoded source traces DO sit outside the nominal window -- measured up to 160.4 deg
    against a 159 ceiling -- so a fixed clamp made a zero-delta edit move the trace by 0.49 nm.
    The bound must be the window OR the source's own angle, whichever is more permissive."""
    ca = _native_trace(b=4, n=12, seed=20)
    i = ca_to_internal(ca)
    # force one angle well outside the window on both ends
    bad_theta = i["theta"].clone()
    bad_theta[:, 2] = math.radians(168.0)
    bad_theta[:, 4] = math.radians(62.0)
    ca_bad = internal_to_ca(i["d"], bad_theta, i["tau"], i["seed"])
    b, n, _ = ca_bad.shape
    m = torch.ones(b, n, dtype=torch.bool)

    out, _ = apply_internal_residual(
        ca_bad, m, delta_theta=torch.zeros(b, n - 2), delta_tau=torch.zeros(b, n - 3)
    )
    err = float((out - ca_bad).abs().max())
    assert err < 1e-5, f"zero delta moved an out-of-window source by {err} nm"

    # and the editor still cannot push those angles FURTHER out
    push = torch.zeros(b, n - 2)
    push[:, 2] = math.radians(20.0)     # 168 -> would be 188
    push[:, 4] = math.radians(-20.0)    # 62  -> would be 42
    out2, _ = apply_internal_residual(
        ca_bad, m, delta_theta=push, delta_tau=torch.zeros(b, n - 3)
    )
    th2 = ca_to_internal(out2)["theta"]
    assert float(torch.rad2deg(th2[:, 2]).max()) <= 168.1, torch.rad2deg(th2[:, 2]).max()
    assert float(torch.rad2deg(th2[:, 4]).min()) >= 61.9, torch.rad2deg(th2[:, 4]).min()


# --------------------------------------------------------------- plumbing safety


def test_gradients_reach_every_delta_and_are_finite():
    ca = _native_trace(b=4, n=12, seed=12)
    b, n, _ = ca.shape
    dth = torch.zeros(b, n - 2, requires_grad=True)
    dta = torch.zeros(b, n - 3, requires_grad=True)
    dtr = torch.zeros(b, 3, requires_grad=True)
    dro = torch.zeros(b, 3, requires_grad=True)
    out, _ = apply_internal_residual(
        ca, torch.ones(b, n, dtype=torch.bool),
        delta_theta=dth, delta_tau=dta, delta_trans=dtr, delta_rot=dro,
    )
    out.pow(2).sum().backward()
    for name, t in (("theta", dth), ("tau", dta), ("trans", dtr), ("rot", dro)):
        assert t.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite gradient for {name}"
        assert float(t.grad.abs().sum()) > 0, f"zero gradient for {name}"


def test_padding_is_masked_and_produces_no_nan():
    ca = _native_trace(b=4, n=14, seed=13)
    m = torch.ones(4, 14, dtype=torch.bool)
    m[:, 9:] = False                       # short peptides, padded tail
    ca = ca * m[..., None]
    out, diag = apply_internal_residual(
        ca, m, delta_theta=torch.randn(4, 12) * 0.3, delta_tau=torch.randn(4, 11) * 0.3,
        delta_trans=torch.zeros(4, 3), delta_rot=torch.zeros(4, 3),
    )
    assert torch.isfinite(out).all(), "NaN/inf in the edited trace"
    assert float(out[~m].abs().max()) == 0.0, "padding not zeroed"
    for k, v in diag.items():
        assert torch.isfinite(v).all(), f"non-finite diagnostic {k}"
    # the valid prefix must still be a valid chain
    bd = (out[:, 1:9] - out[:, :8]).norm(dim=-1)
    assert float((bd - CA_CA_NATIVE_NM).abs().max()) < 1e-5


def test_short_peptides_do_not_crash():
    for n in (2, 3, 4):
        ca = torch.zeros(2, n, 3)
        for i in range(1, n):
            ca[:, i, 0] = CA_CA_NATIVE_NM * i
        i = ca_to_internal(ca)
        back = internal_to_ca(i["d"], i["theta"], i["tau"], i["seed"])
        assert back.shape == ca.shape, (n, back.shape)
        assert float((back - ca).abs().max()) < 1e-5, n


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:  # a crash is a failure too, and more informative raw
                fails += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    raise SystemExit(1 if fails else 0)
