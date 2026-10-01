"""Checks on the CA-spacing probe's geometry, on cases whose answer is known in advance.

The probe's whole claim rests on one quantity -- the share of the editor's residual that
actually changes a CA-CA distance -- so that quantity is tested against the two limits it is
meant to separate: a coherent motion (which must read ~0) and an independent per-residue
residual (which must read ~1). Without those two anchors the number is uninterpretable.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cp2lp_ca_spacing_probe import (  # noqa: E402
    NATIVE_CA_CA_NM,
    bend_angles_deg,
    bend_triple_mask,
    bond_lengths,
    pearson,
    project_ca_spacing,
    spearman,
)


def _chain(b: int = 4, n: int = 12, d0: float = NATIVE_CA_CA_NM, seed: int = 0) -> torch.Tensor:
    """A random chain with every CA-CA separation exactly ``d0``."""
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn(b, n - 1, 3, generator=g)
    steps = steps / steps.norm(dim=-1, keepdim=True) * d0
    return torch.cat([torch.zeros(b, 1, 3), steps.cumsum(1)], dim=1)


def _incoherence(src_ca, delta, bond_mask):
    """Exactly the probe's formula, so the test pins the probe and not a paraphrase."""
    edit = src_ca + delta
    ds_exact = bond_lengths(edit) - bond_lengths(src_ca)
    rms = ds_exact[bond_mask].double().pow(2).mean().sqrt()
    sigma_c = (delta**2).sum().double().div(delta[..., 0].numel() * 3).sqrt()
    return float(rms / (math.sqrt(2) * sigma_c)), float(rms)


def test_bond_lengths_exact():
    ca = _chain()
    L = bond_lengths(ca)
    assert torch.allclose(L, torch.full_like(L, NATIVE_CA_CA_NM), atol=1e-6), L


def test_projection_converges_from_a_badly_broken_chain():
    """The counterfactual is only readable if the projection actually lands on d0."""
    ca = _chain(seed=1)
    g = torch.Generator().manual_seed(2)
    broken = ca + torch.randn(ca.shape, generator=g) * 0.17   # the gan arm's sigma
    bm = torch.ones(ca.shape[0], ca.shape[1] - 1, dtype=torch.bool)
    out = project_ca_spacing(broken, bm, NATIVE_CA_CA_NM, iters=600, omega=0.5)
    resid = (bond_lengths(out) - NATIVE_CA_CA_NM).abs().max()
    assert float(resid) < 1e-4, f"projection did not converge: max |L-d0| = {float(resid)}"


def test_projection_introduces_no_net_translation():
    ca = _chain(seed=3)
    g = torch.Generator().manual_seed(4)
    broken = ca + torch.randn(ca.shape, generator=g) * 0.1
    bm = torch.ones(ca.shape[0], ca.shape[1] - 1, dtype=torch.bool)
    out = project_ca_spacing(broken, bm, NATIVE_CA_CA_NM, iters=400, omega=0.5)
    drift = (out.mean(1) - broken.mean(1)).norm(dim=-1).max()
    assert float(drift) < 1e-5, f"projection translated the peptide by {float(drift)} nm"


def test_coherent_motion_reads_as_zero_incoherence():
    """Translation + rotation of the whole peptide changes no CA-CA distance.

    This is the edit the task WANTS, and it must not be charged as chain breakage.
    """
    ca = _chain(seed=5)
    bm = torch.ones(ca.shape[0], ca.shape[1] - 1, dtype=torch.bool)
    # a rigid motion: rotate about z by 0.3 rad and translate
    c, s = math.cos(0.3), math.sin(0.3)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    moved = ca @ R.T + torch.tensor([0.5, -0.2, 0.1])
    inc, rms = _incoherence(ca, moved - ca, bm)
    assert rms < 1e-5, f"a rigid motion changed CA-CA by {rms} nm"
    assert inc < 1e-4, f"rigid motion scored incoherence {inc}"


def test_independent_residual_reads_as_unit_incoherence():
    """An iid per-residue residual must score ~1: that is the definition of the baseline."""
    ca = _chain(b=64, n=24, seed=6)
    bm = torch.ones(ca.shape[0], ca.shape[1] - 1, dtype=torch.bool)
    g = torch.Generator().manual_seed(7)
    # small amplitude so the first-order identity holds and the exact RMS matches sqrt(2)*sigma
    delta = torch.randn(ca.shape, generator=g) * 0.005
    inc, _ = _incoherence(ca, delta, bm)
    assert 0.85 < inc < 1.15, f"iid residual scored incoherence {inc}, expected ~1"


def test_smooth_residual_is_between_the_two_limits():
    """A low-spatial-frequency residual moves every CA but barely stretches the chain."""
    ca = _chain(b=32, n=24, seed=8)
    bm = torch.ones(ca.shape[0], ca.shape[1] - 1, dtype=torch.bool)
    i = torch.arange(ca.shape[1], dtype=torch.float32)
    # one slow sine along the chain, same amplitude per component as the iid case
    smooth = torch.stack([torch.sin(i * 0.2), torch.cos(i * 0.2), torch.sin(i * 0.1)], -1)
    delta = (smooth * 0.005).expand(ca.shape[0], -1, -1).contiguous()
    inc, _ = _incoherence(ca, delta, bm)
    assert inc < 0.5, f"a smooth residual scored incoherence {inc}, expected well below 1"


def test_bend_angle_on_a_known_geometry():
    """A right-angle and a straight-line trace, where the answer is arithmetic."""
    d = NATIVE_CA_CA_NM
    right = torch.tensor([[[0.0, 0.0, 0.0], [d, 0.0, 0.0], [d, d, 0.0]]])
    assert abs(float(bend_angles_deg(right)[0, 0]) - 90.0) < 1e-3
    straight = torch.tensor([[[0.0, 0.0, 0.0], [d, 0.0, 0.0], [2 * d, 0.0, 0.0]]])
    assert abs(float(bend_angles_deg(straight)[0, 0]) - 180.0) < 1e-2


def test_bend_triple_mask_needs_all_three():
    m = torch.tensor([[True, True, True, False, True]])
    got = bend_triple_mask(m)
    assert got.tolist() == [[True, False, False]], got


def test_projection_does_not_fix_bend_angles():
    """The point of adding the bend measurement: a bond-length projection leaves the
    angles wherever the residual put them, so it cannot be the whole repair."""
    ca = _chain(b=16, n=16, seed=11)
    bm = torch.ones(ca.shape[0], ca.shape[1] - 1, dtype=torch.bool)
    g = torch.Generator().manual_seed(12)
    broken = ca + torch.randn(ca.shape, generator=g) * 0.11   # the gan arm's sigma
    out = project_ca_spacing(broken, bm, NATIVE_CA_CA_NM, iters=600, omega=0.5)
    # bond lengths are repaired...
    assert float((bond_lengths(out) - NATIVE_CA_CA_NM).abs().max()) < 1e-4
    # ...while the bend angles stay far from the source trace's
    src_bend = bend_angles_deg(ca)
    proj_bend = bend_angles_deg(out)
    shift = float((proj_bend - src_bend).abs().median())
    assert shift > 10.0, f"expected the projection to leave bend angles disturbed, got {shift} deg"


def test_correlation_helpers():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(pearson(x, 2 * x + 1) - 1.0) < 1e-9
    assert abs(spearman(x, x.exp()) - 1.0) < 1e-9   # monotone but not linear
    assert abs(pearson(x, -x) + 1.0) < 1e-9


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
    print("all passed" if not fails else f"{fails} failed")
    raise SystemExit(1 if fails else 0)
