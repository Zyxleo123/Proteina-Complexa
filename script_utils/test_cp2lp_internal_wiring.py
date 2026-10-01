"""Checks on the `ca_residual_space` wiring, without building a model.

Two things need pinning and neither needs a GPU:

* **the default path is untouched.** Five arms are mid-flight against this code, so
  `cartesian` must behave exactly as before -- same keys consumed, same arithmetic, and no
  dependence on the new head existing.
* **the internal path refuses rather than degrades.** Asking for `internal` without the head
  that feeds it must raise, because the silent alternative is falling back to the Cartesian
  residual and running precisely the arm this mode exists to replace, under the new name.

`apply_residuals` is exercised directly with hand-built `nn_out` dicts: it is the single
function both paths funnel through, and testing it needs no trunk, no AE and no config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from proteinfoundation.cp2lp.ca_internal import CA_CA_NATIVE_NM  # noqa: E402
from proteinfoundation.cp2lp.editor import apply_residuals  # noqa: E402

DMS = ["bb_ca", "local_latents"]


def _state(b=3, n=10, dz=8):
    """A source state whose CA track is a valid native chain."""
    steps = torch.randn(b, n - 1, 3, generator=torch.Generator().manual_seed(0))
    steps = steps / steps.norm(dim=-1, keepdim=True) * CA_CA_NATIVE_NM
    ca = torch.cat([torch.zeros(b, 1, 3), steps.cumsum(1)], dim=1)
    return {"bb_ca": ca, "local_latents": torch.randn(b, n, dz)}


def _cart_out(b=3, n=10, dz=8, scale=0.05):
    g = torch.Generator().manual_seed(1)
    return {
        "bb_ca": {"delta": torch.randn(b, n, 3, generator=g) * scale},
        "local_latents": {"delta": torch.randn(b, n, dz, generator=g) * scale},
    }


def _int_out(b=3, n=10, dz=8, scale=0.1):
    g = torch.Generator().manual_seed(2)
    o = _cart_out(b, n, dz)
    o["bb_ca"]["delta_internal"] = torch.randn(b, n, 2, generator=g) * scale
    o["bb_ca"]["delta_rigid"] = torch.randn(b, 6, generator=g) * scale
    return o


def _bonds(ca):
    return (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)


def test_cartesian_is_plain_addition_and_unchanged():
    src, out = _state(), _cart_out()
    mask = torch.ones(3, 10, dtype=torch.bool)
    edited, deltas = apply_residuals(src, out, mask, DMS)
    for dm in DMS:
        assert torch.allclose(edited[dm], src[dm] + out[dm]["delta"], atol=0.0), dm
        assert torch.allclose(deltas[dm], out[dm]["delta"], atol=0.0), dm
    assert "_internal_diag" not in deltas


def test_cartesian_default_ignores_the_internal_keys_entirely():
    """A checkpoint carrying the new head must not change the old path's output."""
    src = _state()
    mask = torch.ones(3, 10, dtype=torch.bool)
    a, _ = apply_residuals(src, _cart_out(), mask, DMS)
    b, _ = apply_residuals(src, _int_out(), mask, DMS)          # extra keys present
    assert torch.allclose(a["bb_ca"], b["bb_ca"], atol=0.0)


def test_internal_mode_keeps_the_chain_valid():
    src, out = _state(), _int_out(scale=0.8)
    mask = torch.ones(3, 10, dtype=torch.bool)
    edited, deltas = apply_residuals(src, out, mask, DMS, ca_residual_space="internal")
    bd = _bonds(edited["bb_ca"])
    assert float((bd - CA_CA_NATIVE_NM).abs().max()) < 1e-5, float((bd - CA_CA_NATIVE_NM).abs().max())
    # the latent edit still goes through untouched -- this mode changes the CA channel only
    assert torch.allclose(edited["local_latents"],
                          src["local_latents"] + out["local_latents"]["delta"], atol=0.0)


def test_internal_mode_reports_the_realised_displacement():
    """`delta_ca_rms_nm` has to stay comparable across the two spaces, so the reported bb_ca
    delta must be `edited - source`, not a head output in angle units."""
    src, out = _state(), _int_out(scale=0.4)
    mask = torch.ones(3, 10, dtype=torch.bool)
    edited, deltas = apply_residuals(src, out, mask, DMS, ca_residual_space="internal")
    assert deltas["bb_ca"].shape == src["bb_ca"].shape
    assert torch.allclose(deltas["bb_ca"], edited["bb_ca"] - src["bb_ca"], atol=1e-6)
    assert float(deltas["bb_ca"].abs().sum()) > 0, "a nonzero edit reported a zero displacement"


def test_internal_mode_surfaces_its_diagnostics():
    src, out = _state(), _int_out(scale=0.8)
    mask = torch.ones(3, 10, dtype=torch.bool)
    _, deltas = apply_residuals(src, out, mask, DMS, ca_residual_space="internal")
    d = deltas["_internal_diag"]
    for k in ("internal_dtheta_deg", "internal_dtau_deg", "internal_bend_clamp_frac"):
        assert k in d, k
        assert torch.isfinite(d[k]).all(), k


def test_zero_internal_residual_is_the_identity():
    src = _state()
    mask = torch.ones(3, 10, dtype=torch.bool)
    out = _cart_out(scale=0.0)
    out["bb_ca"]["delta_internal"] = torch.zeros(3, 10, 2)
    out["bb_ca"]["delta_rigid"] = torch.zeros(3, 6)
    edited, _ = apply_residuals(src, out, mask, DMS, ca_residual_space="internal")
    assert float((edited["bb_ca"] - src["bb_ca"]).abs().max()) < 1e-5


def test_internal_without_the_head_raises_rather_than_falling_back():
    src, out = _state(), _cart_out()
    mask = torch.ones(3, 10, dtype=torch.bool)
    try:
        apply_residuals(src, out, mask, DMS, ca_residual_space="internal")
    except KeyError as e:
        assert "enable_editor_internal" in str(e), str(e)
        return
    raise AssertionError("internal mode silently used the Cartesian head")


def test_an_unknown_space_raises():
    src, out = _state(), _cart_out()
    mask = torch.ones(3, 10, dtype=torch.bool)
    try:
        apply_residuals(src, out, mask, DMS, ca_residual_space="torsion")
    except ValueError as e:
        assert "cartesian" in str(e) and "internal" in str(e)
        return
    raise AssertionError("an unknown ca_residual_space was accepted")


def test_cartesian_still_raises_when_its_own_delta_is_missing():
    src = _state()
    mask = torch.ones(3, 10, dtype=torch.bool)
    out = {"bb_ca": {"v": torch.zeros(3, 10, 3)}, "local_latents": {"delta": torch.zeros(3, 10, 8)}}
    try:
        apply_residuals(src, out, mask, DMS)
    except KeyError as e:
        assert "enable_editor_residual" in str(e)
        return
    raise AssertionError("a missing Cartesian delta was accepted")


def test_internal_mode_handles_padding():
    src, out = _state(b=3, n=12), _int_out(b=3, n=12, scale=0.5)
    mask = torch.ones(3, 12, dtype=torch.bool)
    mask[:, 8:] = False
    src["bb_ca"] = src["bb_ca"] * mask[..., None]
    edited, _ = apply_residuals(src, out, mask, DMS, ca_residual_space="internal")
    assert torch.isfinite(edited["bb_ca"]).all()
    assert float(edited["bb_ca"][~mask].abs().max()) == 0.0
    bd = (edited["bb_ca"][:, 1:8] - edited["bb_ca"][:, :7]).norm(dim=-1)
    assert float((bd - CA_CA_NATIVE_NM).abs().max()) < 1e-5


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
            except Exception as e:
                fails += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    raise SystemExit(1 if fails else 0)
