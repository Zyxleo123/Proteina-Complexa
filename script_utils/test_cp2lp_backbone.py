"""Checks on the backbone container and Arm B's torsion builder.

Arm B's claim is a GUARANTEE -- every backbone it can emit has correct local geometry and an
intact chain -- so these are written as attempts to violate it rather than as spot checks.
The ones that carry real weight:

* ``rebuild(extract(x)) == x`` under `ANGLE_SOURCE_SOURCE`, exactly. This is the test that
  catches a wrong NeRF sign choice, and it is the ONLY test that does: three of the CA-trace
  guarantee tests passed while that round trip was out by 3.6 nm, because a wrong sign still
  produces a valid-LOOKING chain. Same trap, same test, one representation up.
* under `ANGLE_SOURCE_TEMPLATE` every emitted bond length and bond angle equals the template
  to float precision NO MATTER what torsions are requested, including absurd ones.
* the torch builder agrees with the INDEPENDENT numpy builder in the calibration script. Two
  implementations of the same math, so a sign or indexing error has to be made twice to hide.
* a padded batch produces no NaN anywhere, including through a backward pass -- the failure
  mode `atan2` at the origin creates, which `torch.where` does not fix after the fact.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from proteinfoundation.cp2lp.backbone import (  # noqa: E402
    ANGLE_SOURCE_SOURCE,
    ANGLE_SOURCE_TEMPLATE,
    BB_NAMES,
    BB_SLOT,
    CIS_OMEGA_CUT_RAD,
    TEMPLATE_C_N_NM,
    TEMPLATE_CA_C_N_DEG,
    TEMPLATE_CA_C_NM,
    TEMPLATE_CA_C_O_DEG,
    TEMPLATE_C_N_CA_DEG,
    TEMPLATE_C_O_NM,
    TEMPLATE_N_CA_C_DEG,
    TEMPLATE_N_CA_NM,
    TEMPLATE_O_DIHEDRAL_OFFSET_DEG,
    apply_angle_residual,
    backbone_to_atom37,
    backbone_to_torsions,
    backbone_validity,
    bond_angle,
    extract_backbone,
    resolve_chain_geometry,
    torsions_to_backbone,
    wrap_to_pi,
)
from proteinfoundation.cp2lp.ca_internal import _dihedral  # noqa: E402

torch.manual_seed(0)
TOL = 1e-5


# --------------------------------------------------------------------------- fixtures


def synthetic_backbone(b: int = 4, n: int = 12, seed: int = 0, cis_at: int | None = None):
    """A chain built FROM random torsions, so it is exactly on the template manifold.

    Built rather than randomised: a random cloud of N/CA/C points has no bond lengths and no
    bond angles, so every "the template is preserved" test would pass trivially against it.
    """
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(b, n, dtype=torch.bool)
    phi = (torch.rand(b, n, generator=g) * 2 - 1) * math.pi
    psi = (torch.rand(b, n, generator=g) * 2 - 1) * math.pi
    omega = torch.full((b, n - 1), math.pi) + 0.05 * torch.randn(b, n - 1, generator=g)
    if cis_at is not None:
        omega[:, cis_at] = 0.08  # a genuine cis bond
    root = torch.zeros(b, 3, 3)
    root[:, 0] = torch.tensor([0.0, 0.0, 0.0])
    root[:, 1] = torch.tensor([TEMPLATE_N_CA_NM, 0.0, 0.0])
    # Place C so that N-CA-C is exactly the template angle.
    a = math.radians(TEMPLATE_N_CA_C_DEG)
    root[:, 2] = root[:, 1] + TEMPLATE_CA_C_NM * torch.tensor(
        [-math.cos(a), math.sin(a), 0.0]
    )
    bb = torsions_to_backbone(phi, psi, omega, root, mask)
    return bb, mask, {"phi": phi, "psi": psi, "omega": omega, "root": root}


def _backbone_with_varied_angles(b: int = 8, n: int = 14, seed: int = 14):
    """A chain with template bond LENGTHS but per-residue varying bond ANGLES.

    This is the configuration natives are in -- lengths tight, angles spread by ~3 deg -- and
    it is the configuration that distinguishes the two angle sources. Built by handing the
    builder a `geom` dict of randomised angles, so the result is exactly on the manifold
    source mode can reproduce and exactly off the one template mode can.
    """
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(b, n, dtype=torch.bool)
    phi = (torch.rand(b, n, generator=g) * 2 - 1) * math.pi
    psi = (torch.rand(b, n, generator=g) * 2 - 1) * math.pi
    omega = torch.full((b, n - 1), math.pi) + 0.05 * torch.randn(b, n - 1, generator=g)
    root = synthetic_backbone(b=b, n=n, seed=seed)[2]["root"]
    geom = {
        "ang_n_ca_c": torch.deg2rad(TEMPLATE_N_CA_C_DEG + 3.0 * torch.randn(b, n, generator=g)),
        "ang_ca_c_n": torch.deg2rad(TEMPLATE_CA_C_N_DEG + 1.7 * torch.randn(b, n - 1, generator=g)),
        "ang_c_n_ca": torch.deg2rad(TEMPLATE_C_N_CA_DEG + 2.1 * torch.randn(b, n - 1, generator=g)),
    }
    return torsions_to_backbone(phi, psi, omega, root, mask, geom), mask


def padded_backbone(b: int = 3, n: int = 10):
    """A batch with REAL padding: row i has i+4 valid residues, the rest are zeros."""
    bb, mask, _ = synthetic_backbone(b=b, n=n, seed=7)
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i in range(b):
        mask[i, : min(i + 4, n)] = True
    bb = bb * mask[:, :, None, None]
    return bb, mask


# ---------------------------------------------------------------- the round-trip test


def test_round_trip_is_exact_under_angle_source_source():
    """extract -> resolve(source) -> rebuild reproduces N, CA, C to float precision.

    THE test. A wrong NeRF sign, a transposed dihedral argument or an off-by-one in the
    angle indexing all produce a chain that passes every geometry check and is not the input.
    """
    bb, mask, _ = synthetic_backbone(seed=1)
    internal = backbone_to_torsions(bb, mask)
    geom = resolve_chain_geometry(internal, ANGLE_SOURCE_SOURCE)
    rebuilt = torsions_to_backbone(
        internal["phi"], internal["psi"], internal["omega"], internal["root"], mask, geom
    )
    for name in ("N", "CA", "C"):
        err = (rebuilt[:, :, BB_SLOT[name]] - bb[:, :, BB_SLOT[name]]).abs().max()
        assert err < TOL, f"{name} round trip out by {float(err):.3e} nm"


def test_round_trip_reproduces_oxygen():
    """O too -- it is determined by the residue frame plus psi, so it must come back as well.

    Every residue but the LAST: psi(L-1) is a fallback constant, not a measurement, so the
    terminal O is not a round-trippable quantity and claiming it would be false.
    """
    bb, mask, _ = synthetic_backbone(seed=2)
    internal = backbone_to_torsions(bb, mask)
    geom = resolve_chain_geometry(internal, ANGLE_SOURCE_SOURCE)
    rebuilt = torsions_to_backbone(
        internal["phi"], internal["psi"], internal["omega"], internal["root"], mask, geom
    )
    err = (rebuilt[:, :-1, BB_SLOT["O"]] - bb[:, :-1, BB_SLOT["O"]]).abs().max()
    assert err < TOL, f"O round trip out by {float(err):.3e} nm"


def test_cis_bond_survives_the_round_trip():
    """A cis peptide bond is carried through, not silently forced trans.

    0.41% of native bonds are cis. An arm that trans-ifies them is wrong on 1 residue in 243,
    which is invisible in a mean and fatal to the peptide it happens in.
    """
    bb, mask, built = synthetic_backbone(seed=3, cis_at=4)
    internal = backbone_to_torsions(bb, mask)
    assert internal["omega"][:, 4].abs().max() < CIS_OMEGA_CUT_RAD, "cis bond not read as cis"
    geom = resolve_chain_geometry(internal, ANGLE_SOURCE_SOURCE)
    rebuilt = torsions_to_backbone(
        internal["phi"], internal["psi"], internal["omega"], internal["root"], mask, geom
    )
    assert (rebuilt[:, :, BB_SLOT["CA"]] - bb[:, :, BB_SLOT["CA"]]).abs().max() < TOL


# ------------------------------------------------------- the construction guarantee


def test_template_geometry_holds_for_absurd_torsions():
    """Arm B's guarantee: every bond length and chain angle is the template, always.

    Requested torsions are deliberately insane (uniform over many turns). If the guarantee
    held only for reasonable inputs it would not be a guarantee, and a GAN generator is
    exactly the thing that produces unreasonable inputs.
    """
    b, n = 5, 16
    mask = torch.ones(b, n, dtype=torch.bool)
    g = torch.Generator().manual_seed(11)
    phi = (torch.rand(b, n, generator=g) * 40 - 20) * math.pi
    psi = (torch.rand(b, n, generator=g) * 40 - 20) * math.pi
    omega = (torch.rand(b, n - 1, generator=g) * 40 - 20) * math.pi
    root = synthetic_backbone(b=b, n=n, seed=4)[2]["root"]

    bb = torsions_to_backbone(phi, psi, omega, root, mask)
    nat, caat, cat = (bb[:, :, BB_SLOT[k]] for k in ("N", "CA", "C"))

    assert ((caat - nat).norm(dim=-1) - TEMPLATE_N_CA_NM).abs().max() < 1e-6
    assert ((cat - caat).norm(dim=-1) - TEMPLATE_CA_C_NM).abs().max() < 1e-6
    assert ((nat[:, 1:] - cat[:, :-1]).norm(dim=-1) - TEMPLATE_C_N_NM).abs().max() < 1e-6

    tau = bond_angle(nat, caat, cat).rad2deg()
    assert (tau - TEMPLATE_N_CA_C_DEG).abs().max() < 1e-3, f"tau drifted {float((tau - TEMPLATE_N_CA_C_DEG).abs().max())}"
    a1 = bond_angle(caat[:, :-1], cat[:, :-1], nat[:, 1:]).rad2deg()
    assert (a1 - TEMPLATE_CA_C_N_DEG).abs().max() < 1e-3
    a2 = bond_angle(cat[:, :-1], nat[:, 1:], caat[:, 1:]).rad2deg()
    assert (a2 - TEMPLATE_C_N_CA_DEG).abs().max() < 1e-3


def test_oxygen_geometry_is_the_template():
    """C=O length, CA-C-O angle and the O dihedral offset all hold by construction."""
    bb, mask, built = synthetic_backbone(seed=5)
    nat, caat, cat, oat = (bb[:, :, BB_SLOT[k]] for k in BB_NAMES)
    assert ((oat - cat).norm(dim=-1) - TEMPLATE_C_O_NM).abs().max() < 1e-6
    ang = bond_angle(caat, cat, oat).rad2deg()
    assert (ang - TEMPLATE_CA_C_O_DEG).abs().max() < 1e-3
    o_di = _dihedral(nat, caat, cat, oat)
    off = wrap_to_pi(o_di - built["psi"] - math.radians(TEMPLATE_O_DIHEDRAL_OFFSET_DEG))
    assert off.abs().max() < 1e-4, f"O dihedral offset wrong by {float(off.abs().max())} rad"


def test_validity_passes_everything_the_builder_makes():
    """The construction guarantee, measured through the same function the arms report with.

    "By construction" is a claim about code. This runs the independent validity checker over
    the builder's own output and insists it agrees -- so a builder bug and a checker bug have
    to coincide to pass.
    """
    bb, mask, _ = synthetic_backbone(b=6, n=14, seed=6)
    v = backbone_validity(bb, mask)
    assert v["bb_chain_intact"].min() == 1.0, "builder emitted a broken chain"
    for key in ("bb_peptide_bond_pass", "bb_intra_bond_pass", "bb_tau_pass",
                "bb_peptide_angle_pass", "bb_carbonyl_pass"):
        assert v[key].min() > 1.0 - 1e-6, f"{key} = {float(v[key].min()):.4f}"


def test_arm_a_style_free_residual_breaks_what_arm_b_guarantees():
    """The contrast that makes the guarantee meaningful rather than tautological.

    A free Cartesian residual of a few tenths of a nanometre -- the Arm A parameterisation --
    destroys the chain. If this passed, Arm B's construction would be solving a non-problem.
    """
    bb, mask, _ = synthetic_backbone(b=6, n=14, seed=8)
    noisy = bb + 0.05 * torch.randn(bb.shape, generator=torch.Generator().manual_seed(9))
    v = backbone_validity(noisy * mask[:, :, None, None], mask)
    assert v["bb_chain_intact"].max() == 0.0, "a 0.05 nm free residual left a chain intact"


# ------------------------------------------------------- agreement with the calibration


def test_torch_builder_matches_the_numpy_builder():
    """The torch builder and the calibration script's numpy builder place the same atoms.

    Two independent implementations of the same NeRF math. The calibration's `source` mode
    rebuilding natives to exactly 0.0000 nm is what certifies the numpy side, so agreement
    here transfers that certification to the torch side.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cp2lp_bb_calibrate", REPO / "script_utils" / "cp2lp_bb_calibrate.py"
    )
    calib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calib)

    a = np.array([0.0, 0.0, 0.0])
    bpt = np.array([0.15, 0.0, 0.0])
    c = np.array([0.20, 0.14, 0.0])
    for bond, ang, tor in ((0.1348, 117.8, 61.0), (0.1455, 123.8, -179.0), (0.1525, 113.0, -47.0)):
        want = calib.place(a, bpt, c, bond, ang, tor)
        from proteinfoundation.cp2lp.backbone import _place

        got = _place(
            torch.tensor(a)[None].double(), torch.tensor(bpt)[None].double(),
            torch.tensor(c)[None].double(), torch.tensor(bond).double(),
            torch.tensor(math.radians(ang)).double(),
            torch.tensor([math.radians(tor)]).double(),
        )[0].numpy()
        # 1e-7 nm, not float64 roundoff. The two differ by ~5e-9 nm because `safe_norm` adds
        # 1e-12 under the sqrt where numpy's norm does not -- MEASURED as 2.1e-11 relative in
        # the unit vectors -- plus a different order of operations. That is 5e-8 angstrom, a
        # thousand times tighter than any geometric tolerance in this project, and chasing it
        # to zero would mean removing the epsilon that keeps padded frames differentiable.
        assert np.abs(got - want).max() < 1e-7, f"placers disagree by {np.abs(got - want).max():.2e}"


# ------------------------------------------------------------------- angular residual


def test_zero_angle_residual_is_the_exact_identity():
    """A zero-initialised head must leave every torsion untouched, exactly.

    This is why the delta is encoded as atan2(s, 1 + c) and not atan2(s, c): the latter puts
    the identity at (0, 1), which a zero-init head cannot emit, so the arm's first output
    would not be the source.
    """
    base = (torch.rand(4, 11) * 2 - 1) * math.pi
    out = apply_angle_residual(base, torch.zeros(4, 11), torch.zeros(4, 11))
    assert (wrap_to_pi(out - base)).abs().max() < 1e-7


def test_angle_residual_is_continuous_across_the_seam():
    """Two angles either side of +-pi get nearly equal residual output. No discontinuity.

    The failure a plain scalar regression has: at the seam, two angles a thousandth of a
    radian apart differ in squared error by (2 pi)^2 and the gradient points the wrong way.
    """
    s = torch.full((1, 1), 0.3)
    c = torch.full((1, 1), 0.1)
    lo = apply_angle_residual(torch.full((1, 1), math.pi - 1e-4), s, c)
    hi = apply_angle_residual(torch.full((1, 1), -math.pi + 1e-4), s, c)
    assert wrap_to_pi(lo - hi).abs().max() < 1e-3


def test_angle_residual_gradient_is_finite_at_the_origin():
    """Gradients at s = c = 0, where a zero-init head starts and where atan2 is fragile."""
    base = torch.zeros(2, 5)
    s = torch.zeros(2, 5, requires_grad=True)
    c = torch.zeros(2, 5, requires_grad=True)
    apply_angle_residual(base, s, c).sum().backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(c.grad).all()
    assert s.grad.abs().max() > 0, "the residual head would get no gradient at all"


# ------------------------------------------------------------------------- container


def test_atom37_round_trip_and_mask_discipline():
    """Only N/CA/C/O are present; every other slot is zero AND masked off.

    Sidechain slots left unmasked at the origin would read as a catastrophic clash at the
    receptor centre and poison every steric metric in the evaluation.
    """
    bb, mask = padded_backbone()
    atom37, atom_mask = backbone_to_atom37(bb, mask)
    from proteinfoundation.cp2lp.geometry import CA_IDX, C_IDX, N_IDX, O_IDX

    present = {N_IDX, CA_IDX, C_IDX, O_IDX}
    for slot in range(37):
        if slot in present:
            assert atom_mask[:, :, slot][mask].all(), f"slot {slot} masked off on a real residue"
        else:
            assert not atom_mask[:, :, slot].any(), f"slot {slot} left unmasked"
            assert atom37[:, :, slot].abs().max() == 0.0, f"slot {slot} not zeroed"
    assert not atom_mask[~mask].any(), "a padded residue carries atoms"

    back, bb_mask = extract_backbone(atom37, atom_mask)
    assert (back[mask] - bb[mask]).abs().max() < 1e-7
    assert bb_mask[mask].all() and not bb_mask[~mask].any()


# ---------------------------------------------------------------------- NaN discipline


def test_padded_batch_is_nan_free_forward_and_backward():
    """Padding must not produce a NaN anywhere, including through backward.

    `atan2` differentiates to 0/0 at the origin, which is exactly where padded residues land.
    Sanitising afterwards is NOT equivalent: by then the NaN is in the graph and `torch.where`
    hands gradient to both branches, so multiplying by a zero mask gives 0 * NaN = NaN.
    """
    bb, mask = padded_backbone()
    bb = bb.clone().requires_grad_(True)
    internal = backbone_to_torsions(bb, mask)
    for k, v in internal.items():
        assert torch.isfinite(v).all(), f"{k} has a non-finite value"
    geom = resolve_chain_geometry(internal, ANGLE_SOURCE_SOURCE)
    out = torsions_to_backbone(
        internal["phi"], internal["psi"], internal["omega"], internal["root"], mask, geom
    )
    assert torch.isfinite(out).all(), "builder emitted a non-finite coordinate"
    v = backbone_validity(out, mask)
    for k, t in v.items():
        assert torch.isfinite(t).all(), f"validity metric {k} is non-finite"
    out.sum().backward()
    assert torch.isfinite(bb.grad).all(), "backward produced a NaN gradient"


def test_all_zero_input_does_not_nan():
    """The fully degenerate case: every atom at the origin. Finite, not correct."""
    b, n = 2, 6
    bb = torch.zeros(b, n, 4, 3, requires_grad=True)
    mask = torch.ones(b, n, dtype=torch.bool)
    internal = backbone_to_torsions(bb, mask)
    geom = resolve_chain_geometry(internal, ANGLE_SOURCE_SOURCE)
    out = torsions_to_backbone(
        internal["phi"], internal["psi"], internal["omega"], internal["root"], mask, geom
    )
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(bb.grad).all()


# --------------------------------------------------------------- the clamp and the floor


def test_source_angle_clamp_is_source_relative():
    """A source angle outside the native window survives unchanged, so identity stays exact.

    A plain clamp would move the chain at zero delta -- measured as a 0.49 nm "identity" edit
    when the CA editor made this mistake. The bound has to be the window OR the source's own
    value, whichever is more permissive.
    """
    bb, mask, _ = synthetic_backbone(seed=12)
    internal = backbone_to_torsions(bb, mask)
    # Force one angle far outside the window and check it comes back untouched.
    internal["ang_n_ca_c"] = internal["ang_n_ca_c"].clone()
    internal["ang_n_ca_c"][:, 3] = math.radians(150.0)
    geom = resolve_chain_geometry(internal, ANGLE_SOURCE_SOURCE)
    assert abs(float(geom["ang_n_ca_c"][0, 3].rad2deg()) - 150.0) < 1e-4


def test_template_mode_ignores_source_angles_and_source_mode_does_not():
    """The two modes are genuinely different, and template mode really does flatten."""
    bb, mask, _ = synthetic_backbone(seed=13)
    internal = backbone_to_torsions(bb, mask)
    t = resolve_chain_geometry(internal, ANGLE_SOURCE_TEMPLATE)
    # Equality against the constant, not `std() == 0`: the std of N identical float32 values
    # is one ULP (measured 1.2e-7 rad here), so a std threshold tests float layout rather
    # than whether the mode flattened the angles.
    assert (t["ang_n_ca_c"] - math.radians(TEMPLATE_N_CA_C_DEG)).abs().max() == 0.0, (
        "template mode kept per-residue variation"
    )
    s = resolve_chain_geometry(internal, ANGLE_SOURCE_SOURCE)
    assert torch.allclose(s["ang_n_ca_c"], internal["ang_n_ca_c"], atol=1e-6)


def test_template_mode_has_a_real_floor_and_source_mode_does_not():
    """Quantifies, in-test, the 1.18 A finding the calibration measured on natives.

    Not a duplicate of the calibration: this is the TORCH path on synthetic input, so if
    someone later changes the builder to secretly carry source angles, the calibration would
    still report 0.1180 from its own numpy code while this test would catch the change.
    """
    # Per-residue VARYING bond angles, template bond lengths -- the realistic case, and the
    # one the natives are in. Jittering the Cartesian coordinates instead would also move the
    # bond LENGTHS, which source mode does not carry, so it would floor BOTH modes and test
    # nothing. (That is a real property, pinned separately below.)
    bb, mask = _backbone_with_varied_angles(b=8, n=14, seed=14)
    internal = backbone_to_torsions(bb, mask)
    args = (internal["phi"], internal["psi"], internal["omega"], internal["root"], mask)

    src = torsions_to_backbone(*args, resolve_chain_geometry(internal, ANGLE_SOURCE_SOURCE))
    tpl = torsions_to_backbone(*args, resolve_chain_geometry(internal, ANGLE_SOURCE_TEMPLATE))
    err_src = (src[:, :, :3] - bb[:, :, :3]).norm(dim=-1)[mask].pow(2).mean().sqrt()
    err_tpl = (tpl[:, :, :3] - bb[:, :, :3]).norm(dim=-1)[mask].pow(2).mean().sqrt()
    assert float(err_src) < 1e-5, f"source mode has a floor of {float(err_src):.2e} nm"
    assert float(err_tpl) > 10 * float(err_src) + 1e-3, (
        f"template mode floor {float(err_tpl):.4f} nm is not meaningfully above source mode's"
    )


def test_carrying_lengths_is_what_removes_the_last_of_the_floor():
    """Carrying angles alone leaves a floor; carrying lengths too removes it. Both pinned.

    THE CONDITIONAL-INDEPENDENCE TRAP this encodes. The leave-one-out decomposition measured
    each bond LENGTH at ~0% of the floor, so "lengths are free" looked settled -- but every
    one of those rows was measured with the ANGLES STILL TEMPLATED, and the 1.18 A angle error
    swamped them. Once the angles are carried, the lengths are all that is left: MEASURED at
    0.21 A RMSD (2.04 A worst atom) on 12 real complexes, from per-bond deviations of only
    0.004-0.008 A that compound along the chain.

    So two factors each measured at ~0% of a total accounted, between them, for everything
    that remained when the dominant one was removed. The test pins both regimes so neither
    half of that can be forgotten.
    """
    from proteinfoundation.cp2lp.backbone import (
        LENGTH_SOURCE_SOURCE,
        LENGTH_SOURCE_TEMPLATE,
    )

    bb, mask, _ = synthetic_backbone(b=6, n=12, seed=21)
    # Perturb so the source's lengths AND angles are both off the template manifold, which is
    # the state a real crystal or AE decode is in.
    stretched = bb + 0.004 * torch.randn(bb.shape, generator=torch.Generator().manual_seed(22))
    internal = backbone_to_torsions(stretched, mask)
    args = (internal["phi"], internal["psi"], internal["omega"], internal["root"], mask)

    def rebuild(length_source):
        geom = resolve_chain_geometry(
            internal, ANGLE_SOURCE_SOURCE, length_source=length_source
        )
        out = torsions_to_backbone(*args, geom)
        return float((out[:, :, :3] - stretched[:, :, :3]).norm(dim=-1)[mask].pow(2).mean().sqrt())

    err_tmpl = rebuild(LENGTH_SOURCE_TEMPLATE)
    err_src = rebuild(LENGTH_SOURCE_SOURCE)
    assert err_tmpl > 1e-3, (
        f"templated lengths show no floor ({err_tmpl:.2e} nm) -- they are not being applied"
    )
    assert err_src < 1e-5, f"carried lengths still floor at {err_src:.2e} nm"
    assert err_tmpl > 100 * err_src


def test_no_closing_bond_is_ever_built():
    """The terminal C(L-1)-N(0) pair is not near a bond distance. No ring, by construction."""
    bb, mask, _ = synthetic_backbone(b=6, n=13, seed=16)
    c_last = bb[:, -1, BB_SLOT["C"]]
    n_first = bb[:, 0, BB_SLOT["N"]]
    gap = (c_last - n_first).norm(dim=-1)
    assert (gap > 2 * TEMPLATE_C_N_NM).all(), f"a terminal closure appeared: {gap.tolist()}"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "--no-header", "-x"]))
