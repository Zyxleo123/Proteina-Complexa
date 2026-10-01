"""CA-trace internal coordinates: an edit space whose every point is a valid backbone.

WHY THIS EXISTS. The one-pass editor adds a free per-residue Cartesian residual to the CA
track (`editor.apply_residuals`) and a frozen decoder then builds N, C and O from it.
MEASURED (docs/README_CP2LP.md, "Why the chain is not intact"): ~30-40% of that residual is
*incoherent*, i.e. it changes CA(i)-CA(i+1) separations, and the resulting trace is off its
native manifold in BOTH of the trace's constraints at once -- bonds from 1.0 to 7.2 A against
a native 3.838 +- 0.054 A, and bend angles down to 25.6 deg against a native floor of 74.5.
`chain_intact` is consequently 0.0000 in the gan arm against an AE round-trip ceiling of
0.9844.

Two things that measurement ruled out as fixes, and they are why this module is shaped the
way it is:

* **A penalty term does not work.** The geometry loss already acts on the DECODED atoms, i.e.
  downstream of the frozen decoder, while the cause is upstream of it; and a new CA-CA term
  would be a seventh term competing in a clipped gradient budget that already retires the
  sequence term to make room.
* **Repairing one constraint does not work.** Projecting the edited trace back onto valid
  CA-CA lengths and re-decoding recovered only 0.000 -> 0.141 (gan) and 0.516 -> 0.672
  (control), because it leaves the bend angles violated and in fact nudges the floor DOWN
  (74.5 -> 25.6 -> 21.8 deg). `chain_intact` is an all-bonds test, so a surviving tail costs
  almost everything: reaching 0.95 of peptides needs a per-bond pass rate of 0.9945.

So the edit has to be *unable* to produce an invalid trace. Here a CA trace is represented as

    (seed frame)  CA(0), CA(1), CA(2) -- carried through unchanged, so they fix position and
                  orientation and nothing else
    d[i]          |CA(i+1) - CA(i)|,                       i = 0 .. L-2
    theta[i]      angle CA(i-1), CA(i), CA(i+1),           i = 1 .. L-2
    tau[i]        dihedral CA(i-1), CA(i), CA(i+1), CA(i+2), i = 1 .. L-3

and an edit moves `theta`, `tau` and a rigid transform of the whole trace. Three properties
follow, each load-bearing:

* **Zero deltas reproduce the input EXACTLY.** The seed frame is the source's own first three
  atoms and `d` is the source's own bond lengths, so `rebuild(extract(ca)) == ca`. The editor
  keeps the zero-initialised-head property that its first output is precisely the AE round
  trip of the source CP -- the thing that made the pretrained splice safe.
* **Bond lengths cannot move at all** unless `delta_d` is passed, which it is not by default.
  The constraint the free residual violated 88% of the time is simply not in the edit space.
* **Bend angles are clamped to the measured native window**, so the second constraint cannot
  be violated either. `tau` is left free: the virtual torsion is unconstrained in real
  peptides, and it is the degree of freedom that actually opens a ring.

The coherent/incoherent split the probe had to *measure* becomes structural here: a rigid
motion is the seed-frame delta, shape change is theta/tau, and there is no parameter whose
job is to stretch the chain.

ON THE LEVER ARM, HONESTLY. Internal coordinates are not free of problems: a torsion change
at i swings the whole chain downstream of i, so gradients through `tau` are badly scaled
relative to gradients through a Cartesian residual, and the effect grows with L. This is the
known cost of torsion-space parameterisations and it is NOT fixed here -- it is traded for
exact constraint satisfaction. `internal_residual_scale` exists to damp it and is a knob to
tune, not a solved problem.

Everything is in NANOMETRES, matching the rest of cp2lp.
"""

from __future__ import annotations

import math

import torch

from proteinfoundation.cp2lp.geometry import safe_norm

#: Native CA(i)-CA(i+1) separation window, nm. MEASURED on the 1,233 bonded pairs of the
#: peptide chains in `calibration_native/`: median 3.838 A, sd 0.054, min 3.034 (a cis bond),
#: max 3.980, and NOTHING above 4.00. The window is the observed range, padded outward.
CA_CA_MIN_NM = 0.300
CA_CA_MAX_NM = 0.400
CA_CA_NATIVE_NM = 0.3838

#: Native CA(i-1)-CA(i)-CA(i+1) bend window, degrees. MEASURED on the 1,132 native peptide
#: triples: median 97.3, sd 16.2, min 74.47, p1 83.3, p99 146.7, max 157.50. Nothing below
#: 74. The window is the observed range, padded outward by ~1 deg on each side.
BEND_MIN_DEG = 73.0
BEND_MAX_DEG = 159.0

#: Floor for every normalisation, so a degenerate (padded) frame has a finite derivative.
_EPS = 1e-8


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / safe_norm(v, dim=-1, keepdim=True).clamp(min=_EPS)


def _dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """Signed dihedral about the p1-p2 axis, with the degenerate case handled BEFORE atan2.

    `atan2` differentiates to 0/0 = NaN at the origin, which is exactly where padded residues
    land. Sanitising afterwards is not equivalent: by then the NaN is in the graph and
    `torch.where` hands gradient to both branches, so multiplying by a zero mask gives
    0 * NaN = NaN. Same discipline as `geometry.backbone_dihedrals`.
    """
    # `-(p1 - p0)`, not `p1 - p0`. Without the negation the result is offset by exactly pi
    # from the IUPAC convention, which is self-consistent but mislabels every angle: anti
    # (trans) reads 0 instead of 180, and an extended CA trace would come out at tau=0.
    # Verified against anti-butane by hand and against the extended-chain scan in
    # `test_extended_chain_is_near_trans`. NOTE: `geometry.backbone_dihedrals` carries the
    # un-negated form, so its phi/psi are offset by pi too -- harmless there, because the
    # discriminator sees the same relabelling on real and generated peptides alike, but those
    # values are NOT Ramachandran phi/psi and should not be read as such.
    b0, b1, b2 = -(p1 - p0), p2 - p1, p3 - p2
    b1n = _unit(b1)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    degenerate = (x * x + y * y) < 1e-12
    x = torch.where(degenerate, torch.ones_like(x), x)
    y = torch.where(degenerate, torch.zeros_like(y), y)
    return torch.atan2(y, x)


def ca_to_internal(ca: torch.Tensor) -> dict[str, torch.Tensor]:
    """CA trace -> internal coordinates plus the seed frame.

    Args:
        ca: ``[b, n, 3]`` CA coordinates in nm.

    Returns:
        ``d`` ``[b, n-1]``, ``theta`` ``[b, n-2]`` (radians), ``tau`` ``[b, n-3]`` (radians),
        and ``seed`` ``[b, 3, 3]`` -- the first three CA positions, carried through unchanged.

    Padded positions produce arbitrary-but-finite values; the caller masks.
    """
    b, n, _ = ca.shape
    if n < 2:
        raise ValueError(f"a CA trace needs at least 2 residues, got n={n}")
    d = safe_norm(ca[:, 1:] - ca[:, :-1], dim=-1)

    if n >= 3:
        a = _unit(ca[:, :-2] - ca[:, 1:-1])
        c = _unit(ca[:, 2:] - ca[:, 1:-1])
        theta = torch.arccos((a * c).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    else:
        theta = ca.new_zeros((b, 0))

    if n >= 4:
        tau = _dihedral(ca[:, :-3], ca[:, 1:-2], ca[:, 2:-1], ca[:, 3:])
    else:
        tau = ca.new_zeros((b, 0))

    seed = ca[:, : min(3, n), :]
    return {"d": d, "theta": theta, "tau": tau, "seed": seed}


def internal_to_ca(
    d: torch.Tensor,
    theta: torch.Tensor,
    tau: torch.Tensor,
    seed: torch.Tensor,
) -> torch.Tensor:
    """Internal coordinates + seed frame -> CA trace, by sequential NeRF placement.

    The inverse of :func:`ca_to_internal`, exactly: feeding that function's output back here
    returns the original trace to float precision, which
    :mod:`script_utils.test_cp2lp_ca_internal` pins.

    Args:
        d: ``[b, n-1]`` bond lengths, theta: ``[b, n-2]`` radians, tau: ``[b, n-3]`` radians,
        seed: ``[b, 3, 3]`` (or ``[b, k, 3]`` for k<3 when the trace is shorter).

    Returns:
        ``[b, n, 3]``.
    """
    b = d.shape[0]
    n = d.shape[1] + 1
    pts = [seed[:, i, :] for i in range(min(seed.shape[1], n))]
    # A 3-residue trace is fully determined by its seed; only i>=3 needs placing.
    for i in range(len(pts), n):
        p_a, p_b, p_c = pts[i - 3], pts[i - 2], pts[i - 1]
        bond = d[:, i - 1]
        ang = theta[:, i - 2]
        tor = tau[:, i - 3]

        bc = _unit(p_c - p_b)
        # The normal's sign must agree with the dihedral convention `ca_to_internal` extracted
        # in, and the two published NeRF forms differ here. Determined by searching all four
        # sign choices for the one that round-trips a concrete trace (7e-08 nm) and pinned by
        # `test_round_trip_is_exact`. The wrong choices produce valid-LOOKING traces that are
        # not the input -- the failure a weaker test would have missed, and did: three of the
        # guarantee tests passed while the round trip was out by 3.6 nm.
        nrm = _unit(torch.cross(p_b - p_a, bc, dim=-1))
        m2 = torch.cross(nrm, bc, dim=-1)
        # Local offset in the (bc, n x bc, n) frame. The leading minus continues the chain
        # forward rather than folding it back onto p_b.
        off = (
            (-bond * torch.cos(ang))[:, None] * bc
            + (bond * torch.sin(ang) * torch.cos(tor))[:, None] * m2
            + (bond * torch.sin(ang) * torch.sin(tor))[:, None] * nrm
        )
        pts.append(p_c + off)
    return torch.stack(pts, dim=1) if pts else d.new_zeros((b, 0, 3))


def rotation_from_vector(omega: torch.Tensor) -> torch.Tensor:
    """``[b, 3]`` axis-angle -> ``[b, 3, 3]`` rotation, exactly the identity at omega=0.

    Rodrigues with the small-angle series substituted in, because ``sin|w|/|w|`` is 0/0 at the
    origin -- and the origin is where a zero-initialised head starts, so the identity case is
    the one that must be exact rather than the one that may be approximate.
    """
    b = omega.shape[0]
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(b, 3, 3)
    ang = safe_norm(omega, dim=-1)                       # [b]
    small = ang < 1e-6
    safe_ang = torch.where(small, torch.ones_like(ang), ang)
    axis = omega / safe_ang[:, None]
    zero = torch.zeros_like(ang)
    K = torch.stack(
        [
            torch.stack([zero, -axis[:, 2], axis[:, 1]], -1),
            torch.stack([axis[:, 2], zero, -axis[:, 0]], -1),
            torch.stack([-axis[:, 1], axis[:, 0], zero], -1),
        ],
        dim=1,
    )
    s = torch.where(small, ang, torch.sin(safe_ang))     # ~ang for small, exact otherwise
    c = torch.where(small, zero, 1.0 - torch.cos(safe_ang))
    R = eye + s[:, None, None] * K + c[:, None, None] * (K @ K)
    return torch.where(small[:, None, None], eye, R)


def apply_internal_residual(
    src_ca: torch.Tensor,
    mask: torch.Tensor,
    delta_theta: torch.Tensor,
    delta_tau: torch.Tensor,
    delta_trans: torch.Tensor | None = None,
    delta_rot: torch.Tensor | None = None,
    delta_d: torch.Tensor | None = None,
    scale: float = 1.0,
    bend_min_deg: float = BEND_MIN_DEG,
    bend_max_deg: float = BEND_MAX_DEG,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Edit a CA trace in internal coordinates. The output is always a valid trace.

    Args:
        src_ca: ``[b, n, 3]`` source CA, nm.
        mask: ``[b, n]`` residue mask.
        delta_theta: ``[b, n-2]`` bend-angle change, radians (before ``scale``).
        delta_tau: ``[b, n-3]`` virtual-torsion change, radians (before ``scale``).
        delta_trans: ``[b, 3]`` rigid translation, nm. The COHERENT part of the edit.
        delta_rot: ``[b, 3]`` axis-angle rigid rotation.
        delta_d: ``[b, n-1]`` bond-length change, nm. **None by default, and that is the
            point** -- the source bond lengths are already native and the measured failure was
            a residual that moved them, so the default edit space cannot move them at all.
        scale: damping on the shape deltas. See the lever-arm note in the module docstring.
        bend_min_deg / bend_max_deg: the native window theta is clamped into.

    Returns:
        ``(edited_ca, diagnostics)``. Diagnostics report what the edit actually did, in the
        coordinates that matter: how much bend angle moved, how much torsion moved, and how
        much of the clamp was hit (a clamp that saturates means the head is asking for
        something the window forbids, which is information rather than a failure).
    """
    m = mask.bool()
    internal = ca_to_internal(src_ca)
    lo = math.radians(bend_min_deg)
    hi = math.radians(bend_max_deg)

    # THE CLAMP IS SOURCE-RELATIVE, and it has to be. A fixed window breaks the exact-identity
    # property whenever the source itself sits outside it, and the sources DO: encoded source CP
    # traces reach a bend angle of 160.4 deg against this window's 159 ceiling, so a plain
    # `clamp` moved the trace at zero delta -- measured as a 0.49 nm "identity" edit before this
    # was fixed. Widening the window instead would only move the problem to whatever the next
    # batch's maximum happens to be.
    #
    # So the bound is the window OR the source's own angle, whichever is more permissive. Zero
    # delta is then exactly the identity for every input, while the editor still cannot push an
    # angle further out than it already was. `internal_bend_clamp_frac` reports how often the
    # bound binds.
    lo_i = torch.minimum(internal["theta"], torch.full_like(internal["theta"], lo))
    hi_i = torch.maximum(internal["theta"], torch.full_like(internal["theta"], hi))

    theta_raw = internal["theta"] + scale * delta_theta
    theta_new = torch.minimum(torch.maximum(theta_raw, lo_i), hi_i)
    tau_new = internal["tau"] + scale * delta_tau
    d_new = internal["d"] if delta_d is None else (internal["d"] + scale * delta_d).clamp(
        CA_CA_MIN_NM, CA_CA_MAX_NM
    )

    ca = internal_to_ca(d_new, theta_new, tau_new, internal["seed"])

    if delta_rot is not None or delta_trans is not None:
        # Rigid motion about the peptide's own centroid, so a rotation does not also translate.
        mf = m[..., None].to(ca.dtype)
        cen = (ca * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1.0)
        out = ca - cen
        if delta_rot is not None:
            out = torch.einsum("bij,bnj->bni", rotation_from_vector(delta_rot), out)
        out = out + cen
        if delta_trans is not None:
            out = out + delta_trans[:, None, :]
        ca = out

    ca = ca * m[..., None].to(ca.dtype)

    with torch.no_grad():
        tri = m[:, :-2] & m[:, 1:-1] & m[:, 2:] if m.shape[1] > 2 else m[:, :0]
        quad = m[:, :-3] & m[:, 1:-2] & m[:, 2:-1] & m[:, 3:] if m.shape[1] > 3 else m[:, :0]
        n_tri = tri.float().sum().clamp(min=1.0)
        diag = {
            "internal_dtheta_deg": (
                torch.rad2deg((theta_new - internal["theta"]).abs() * tri).sum() / n_tri
            ),
            "internal_dtau_deg": (
                torch.rad2deg((tau_new - internal["tau"]).abs() * quad).sum()
                / quad.float().sum().clamp(min=1.0)
            ),
            "internal_bend_clamp_frac": (
                ((theta_raw < lo_i) | (theta_raw > hi_i)) & tri
            ).float().sum() / n_tri,
        }
    return ca, diag
