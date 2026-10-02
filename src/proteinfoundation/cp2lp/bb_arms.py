"""The two backbone representation arms: one function each, same signature, same container.

THE COMPARISON THIS FILE EXISTS TO MAKE FAIR. Arm A and Arm B differ in exactly one thing --
how a network output becomes a backbone -- and everything else about them is shared: the
trunk, the conditioner, the discriminator, the losses, the packing, the evaluation panel and
the optimizer budget. So both arms are written as one function with one signature

    bb_lp, diag = apply_<arm>(src_bb, nn_out["backbone"], mask, ...)

taking the same source backbone and returning the same ``[b, n, 4, 3]``, and the module calls
whichever the config names. Anything that would belong to only one arm either goes in that
arm's branch here or it is a confound.

WHAT EACH ARM CAN AND CANNOT DO, STATED UP FRONT because the whole experiment is this
contrast:

    Arm A   a free Cartesian residual per backbone atom. Can express ANY backbone, including
            every invalid one. Local geometry is not constrained by the parameterisation at
            all, so bond lengths, bond angles, chain connectivity and carbonyl chirality are
            all things the arm has to LEARN. `backbone_validity` is its scoreboard, and a
            0.05 nm free residual was measured to leave 0 of 6 chains intact.
    Arm B   residual rotations on phi/psi/omega plus a rigid root transform. CANNOT express an
            invalid bond length, bond angle or chain break -- those constraints are absent
            from the edit space rather than penalised in it. Pays for that with a torsion
            lever arm: a rotation at residue i swings everything downstream of i, so the
            gradient scaling is poor and gets worse with length. Not solved, damped by
            `residual_scale`.

The movement budget is applied IDENTICALLY in both arms (see `clip_to_budget`), because
"Arm A moved further" and "Arm A is better" are not the same claim and a per-atom clip in one
arm against a per-torsion clip in the other would silently conflate them.

Nanometres, receptor frame, matching the rest of cp2lp.
"""

from __future__ import annotations

import torch

from proteinfoundation.cp2lp.backbone import (
    BB_SLOT,
    DEFAULT_ANGLE_SOURCE,
    DEFAULT_LENGTH_SOURCE,
    apply_angle_residual,
    backbone_to_torsions,
    resolve_chain_geometry,
    torsions_to_backbone,
)
from proteinfoundation.cp2lp.ca_internal import rotation_from_vector
from proteinfoundation.cp2lp.geometry import safe_norm

#: The two arm names. These are the strings a config sets and the strings that appear in a
#: manifest row, so they are defined once.
ARM_CARTESIAN = "cartesian"
ARM_TORSION = "torsion"
ARMS = (ARM_CARTESIAN, ARM_TORSION)

#: Default damping on each arm's raw head output. Separate numbers because the two heads emit
#: different QUANTITIES in different units -- nanometres of displacement against radians of
#: rotation -- so a shared scalar would not be a shared budget, it would just be the same
#: number applied to two incomparable things. The shared, comparable budget is the one
#: `clip_to_budget` enforces, in nanometres of actual atom movement, after the fact.
DEFAULT_CARTESIAN_SCALE = 1.0
DEFAULT_TORSION_SCALE = 0.3

#: Per-atom movement cap, nm. Applied in BOTH arms, measured in the same units, on the actual
#: displacement rather than on the parameterisation. `None` disables it.
DEFAULT_MAX_ATOM_SHIFT_NM = None


def _masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero every padded residue of a ``[b, n, ...]`` tensor."""
    m = mask.bool()
    while m.dim() < x.dim():
        m = m[..., None]
    return x * m.to(x.dtype)


def clip_to_budget(
    bb_new: torch.Tensor,
    bb_src: torch.Tensor,
    mask: torch.Tensor,
    max_shift_nm: float | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cap each atom's displacement from the source at `max_shift_nm`, identically per arm.

    THE BUDGET HAS TO BE DEFINED ON THE DISPLACEMENT, NOT THE PARAMETERS. Clipping Arm A's
    Cartesian residual and Arm B's torsion residual to the same number would define two
    different movement budgets wearing one name: 0.1 of a nanometre and 0.1 of a radian have
    nothing to do with each other, and 0.1 rad at residue 2 of a 20-mer moves the far end of
    the chain by nanometres. Measuring the actual atom shift and capping that is the only
    version of this that is the same constraint in both arms.

    The clip is a direction-preserving radial projection, so it never rotates an edit, only
    shortens it. Gradient flows through the unclipped direction.
    """
    diag: dict[str, torch.Tensor] = {}
    disp = bb_new - bb_src
    with torch.no_grad():
        d = safe_norm(disp, dim=-1)  # [b, n, 4]
        m4 = mask.bool()[..., None].expand_as(d)
        n = m4.float().sum().clamp(min=1.0)
        diag["bb_shift_mean_nm"] = (d * m4).sum() / n
        diag["bb_shift_max_nm"] = (d * m4).max()
    if max_shift_nm is None:
        diag["bb_shift_clip_frac"] = torch.zeros((), device=bb_new.device, dtype=bb_new.dtype)
        return bb_new, diag
    norm = safe_norm(disp, dim=-1, keepdim=True)
    factor = (max_shift_nm / norm.clamp(min=1e-8)).clamp(max=1.0)
    with torch.no_grad():
        diag["bb_shift_clip_frac"] = (
            ((factor < 1.0).squeeze(-1) & mask.bool()[..., None]).float().sum()
            / mask.bool()[..., None].expand(factor.squeeze(-1).shape).float().sum().clamp(min=1.0)
        )
    return bb_src + disp * factor, diag


# ------------------------------------------------------------------- Arm A: cartesian


def apply_cartesian(
    src_bb: torch.Tensor,
    head: dict[str, torch.Tensor],
    mask: torch.Tensor,
    scale: float = DEFAULT_CARTESIAN_SCALE,
    max_shift_nm: float | None = DEFAULT_MAX_ATOM_SHIFT_NM,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Arm A. ``bb_lp = src_bb + delta``, a free residual on all four backbone atoms.

    Nothing here constrains geometry, and that is the arm's premise rather than an oversight:
    Cartesian coordinates are not locally valid by construction, so whether a 12-channel head
    can learn to keep a chain intact is the question Arm A asks. The raw output is what gets
    packed and scored -- no repair step, because a repair step would measure the repairer.

    Args:
        src_bb: ``[b, n, 4, 3]`` source backbone, nm.
        head: the trunk's ``nn_out["backbone"]``; needs ``delta_cartesian`` ``[b, n, 4, 3]``.
        mask: ``[b, n]``.
        scale: damping on the raw head output.
        max_shift_nm: shared movement budget, see :func:`clip_to_budget`.
    """
    if "delta_cartesian" not in head:
        raise KeyError(
            "the network produced no `delta_cartesian`. Arm A needs "
            "`nn.editor_backbone_mode: cartesian`; without it the trunk emits only a flow "
            "velocity and a CA residual, neither of which is a full-backbone edit."
        )
    delta = _masked(head["delta_cartesian"] * scale, mask)
    bb_new, diag = clip_to_budget(_masked(src_bb, mask) + delta, _masked(src_bb, mask), mask, max_shift_nm)
    with torch.no_grad():
        m = mask.bool()[..., None].expand(delta.shape[:-1])
        diag["bb_delta_norm_nm"] = (
            safe_norm(delta, dim=-1) * m
        ).sum() / m.float().sum().clamp(min=1.0)
    return _masked(bb_new, mask), diag


# --------------------------------------------------------------------- Arm B: torsion


def apply_torsion(
    src_bb: torch.Tensor,
    head: dict[str, torch.Tensor],
    mask: torch.Tensor,
    scale: float = DEFAULT_TORSION_SCALE,
    max_shift_nm: float | None = DEFAULT_MAX_ATOM_SHIFT_NM,
    angle_source: str = DEFAULT_ANGLE_SOURCE,
    length_source: str = DEFAULT_LENGTH_SOURCE,
    aatype: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Arm B. Residual rotations on phi/psi/omega plus a rigid root transform, then rebuild.

    Every output of this function has template bond lengths and (under the default
    `angle_source`) the source's own bond angles, both within float precision, and an intact
    chain -- not because a loss encourages it but because the edit space contains no point
    that violates them.

    THE ROOT TRANSFORM IS NOT OPTIONAL. Arm A can translate and rotate the whole peptide
    through its per-atom residual; if Arm B could only change torsions, the comparison would
    partly be "which arm was allowed to move the peptide". The root delta restores that
    degree of freedom, applied about the peptide's own centroid so a rotation does not also
    translate.

    Args:
        src_bb: ``[b, n, 4, 3]`` source backbone, nm.
        head: needs ``delta_torsion`` ``[b, n, 6]``; ``delta_root`` ``[b, 6]`` is optional.
        mask: ``[b, n]``.
        scale: damping on the angular residual. The torsion lever arm grows with chain
            length and this is the knob that damps it -- a tuning parameter, not a fix.
        max_shift_nm: shared movement budget, see :func:`clip_to_budget`.
        angle_source / length_source: see :mod:`proteinfoundation.cp2lp.backbone`. The
            defaults carry the source's own bond angles AND lengths, which is what makes zero
            delta the EXACT identity -- angles alone leave a measured 0.21 A residual on real
            complexes, because small per-bond length errors compound along the chain.
        aatype: ``[b, n]`` residue types, used only for PRO/GLY templates under
            `ANGLE_SOURCE_TEMPLATE`.
    """
    if "delta_torsion" not in head:
        raise KeyError(
            "the network produced no `delta_torsion`. Arm B needs "
            "`nn.editor_backbone_mode: torsion`."
        )
    d = head["delta_torsion"]
    if d.shape[-1] != 6:
        raise ValueError(f"delta_torsion must have 6 channels (3 angles x sin/cos), got {d.shape[-1]}")

    src_bb = _masked(src_bb, mask)
    internal = backbone_to_torsions(src_bb, mask)
    geom = resolve_chain_geometry(internal, angle_source, aatype, length_source=length_source)

    # Channels: (phi_s, phi_c, psi_s, psi_c, omega_s, omega_c). omega is indexed by BOND, so
    # it reads the first n-1 residues' channels -- residue i's omega channel is the bond
    # i -> i+1, and the last residue has no outgoing bond.
    phi = apply_angle_residual(internal["phi"], d[..., 0], d[..., 1], scale)
    psi = apply_angle_residual(internal["psi"], d[..., 2], d[..., 3], scale)
    n = mask.shape[1]
    if n >= 2:
        omega = apply_angle_residual(
            internal["omega"], d[:, :-1, 4], d[:, :-1, 5], scale
        )
    else:
        omega = internal["omega"]

    root = internal["root"]
    bb_new = torsions_to_backbone(phi, psi, omega, root, mask, geom)

    if "delta_root" in head:
        r = head["delta_root"] * scale
        trans, omega_ax = r[:, :3], r[:, 3:6]
        # Centroid over ATOMS of real residues, so the rotation pivot is the peptide's own
        # centre rather than the origin of the receptor frame.
        #
        # `expand_as` IS LOAD-BEARING. `mask[..., None, None]` has shape [b, n, 1, 1], and
        # summing it over dims (1, 2) counts n_valid, while summing the coordinates over the
        # same dims covers 4 * n_valid atoms -- so the centroid came out FOUR TIMES too far
        # out, and a "pure rotation" moved the peptide by 7 nm. Caught by
        # `test_arm_b_root_rotation_is_about_the_centroid`; a broadcast that silently agrees
        # on rank while disagreeing on count is exactly the bug a shape check does not catch.
        mf = mask.bool()[..., None, None].expand_as(bb_new[..., :1]).to(bb_new.dtype)
        cen = (bb_new * mf).sum(dim=(1, 2), keepdim=True) / mf.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        out = bb_new - cen
        R = rotation_from_vector(omega_ax)  # exactly the identity at zero
        out = torch.einsum("bij,bnaj->bnai", R, out)
        bb_new = out + cen + trans[:, None, None, :]

    bb_new, diag = clip_to_budget(_masked(bb_new, mask), src_bb, mask, max_shift_nm)

    with torch.no_grad():
        m = mask.bool()
        nres = m.float().sum().clamp(min=1.0)
        pair = (m[:, :-1] & m[:, 1:]) if n > 1 else m[:, :0]
        diag["bb_dphi_deg"] = (
            (phi - internal["phi"]).abs().rad2deg() * m
        ).sum() / nres
        diag["bb_dpsi_deg"] = (
            (psi - internal["psi"]).abs().rad2deg() * m
        ).sum() / nres
        diag["bb_domega_deg"] = (
            (omega - internal["omega"]).abs().rad2deg() * pair
        ).sum() / pair.float().sum().clamp(min=1.0)
        if "delta_root" in head:
            diag["bb_root_trans_nm"] = safe_norm(head["delta_root"][:, :3] * scale, dim=-1).mean()
            diag["bb_root_rot_deg"] = (
                safe_norm(head["delta_root"][:, 3:6] * scale, dim=-1).rad2deg().mean()
            )
    return _masked(bb_new, mask), diag


# ------------------------------------------------------------------------- dispatch


def apply_backbone_arm(
    arm: str,
    src_bb: torch.Tensor,
    nn_out: dict,
    mask: torch.Tensor,
    aatype: torch.Tensor | None = None,
    **kw,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run whichever arm `arm` names. The one place the config string becomes behaviour."""
    head = nn_out.get("backbone", None)
    if not head:
        raise KeyError(
            "the network produced no `backbone` outputs. Set `nn.editor_backbone_mode` to "
            f"{ARM_CARTESIAN!r} or {ARM_TORSION!r} to match cp2lp.backbone_arm."
        )
    if arm == ARM_CARTESIAN:
        kw.pop("angle_source", None)
        return apply_cartesian(src_bb, head, mask, **kw)
    if arm == ARM_TORSION:
        return apply_torsion(src_bb, head, mask, aatype=aatype, **kw)
    raise ValueError(f"unknown backbone arm {arm!r}; expected one of {ARMS}")


def init_reconstruction_error(
    arm: str,
    src_bb: torch.Tensor,
    mask: torch.Tensor,
    angle_source: str = DEFAULT_ANGLE_SOURCE,
    length_source: str = DEFAULT_LENGTH_SOURCE,
    aatype: torch.Tensor | None = None,
) -> dict[str, float]:
    """What each arm emits at ZERO head output, against the source. Recorded once per arm.

    Arm A is exact by construction. Arm B is exact only under `ANGLE_SOURCE_SOURCE` and
    carries the measured template floor otherwise, so this is the number that says which
    regime a run is actually in -- rather than trusting the config to mean what it says.
    """
    b, n = mask.shape
    zeros_head = {
        ARM_CARTESIAN: lambda: {"delta_cartesian": src_bb.new_zeros((b, n, 4, 3))},
        ARM_TORSION: lambda: {
            "delta_torsion": src_bb.new_zeros((b, n, 6)),
            "delta_root": src_bb.new_zeros((b, 6)),
        },
    }[arm]()
    kw = (
        {"angle_source": angle_source, "length_source": length_source}
        if arm == ARM_TORSION
        else {}
    )
    bb0, _ = apply_backbone_arm(arm, src_bb, {"backbone": zeros_head}, mask, aatype=aatype, **kw)
    with torch.no_grad():
        m = mask.bool()[..., None].expand(bb0.shape[:-1])
        err = safe_norm(bb0 - _masked(src_bb, mask), dim=-1)
        rmsd = ((err * err * m).sum() / m.float().sum().clamp(min=1.0)).sqrt()
        ca = safe_norm(
            bb0[:, :, BB_SLOT["CA"]] - _masked(src_bb, mask)[:, :, BB_SLOT["CA"]], dim=-1
        )
        ca_rmsd = ((ca * ca * mask.bool()).sum() / mask.bool().float().sum().clamp(min=1.0)).sqrt()
    return {
        "init_bb_rmsd_nm": float(rmsd),
        "init_ca_rmsd_nm": float(ca_rmsd),
        "init_bb_max_nm": float((err * m).max()),
    }
