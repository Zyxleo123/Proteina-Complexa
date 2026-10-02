"""The output container both backbone arms emit, and Arm B's differentiable chain builder.

WHAT CHANGES HERE RELATIVE TO THE CA EDITOR. The one-pass editor edits a CA track plus the
local latents and hands the sum to the FROZEN AE DECODER, which invents N, C, O and a
sequence. These two arms delete that decoder from the output path: the generator emits
N, CA, C, O itself, the sequence is the source's and is never predicted, and nothing
downstream of the generator reconstructs geometry. Two consequences, both load-bearing:

* **The AE round trip stops being the ceiling.** `chain_intact` was measured against an AE
  round-trip ceiling of 0.9844 because the decoder sat between the edit and the metric. With
  the decoder gone, a backbone's validity is a property of the generator's own output. Arm B
  reaches 1.0 by construction; Arm A has to earn it, and that asymmetry IS the experiment.
* **The CA-latent consistency constraint is gone.** The measured binding constraint on the
  CA editor was that a perfect CA trace still decodes badly unless the latents agree with it
  (docs: "CA-latent consistency BINDS"). No latent is decoded here, so that constraint cannot
  bind. Source sidechain latents remain INPUT conditioning only.

THE CONTAINER IS atom37, DELIBERATELY. The repository's geometry, clash, contact, terminal-gap
and export code all speak `[b, n, 37, 3]` plus a `[b, n, 37]` atom mask, and all of it is
correct for a backbone-only structure provided the mask says only four atoms are present.
Inventing a parallel `[b, n, 4, 3]` convention for the arms would mean reimplementing every
one of those functions, so instead:

    bb  [b, n, 4, 3]   the DENSE arm-facing tensor, atoms ordered (N, CA, C, O)
    atom37             the SHARED container, produced by `backbone_to_atom37`, with exactly
                       those four slots filled and every other slot masked off

`backbone_to_atom37` is therefore the one place "backbone-only" is enforced, and it is a
mask, not a convention anyone has to remember.

ON THE TEMPLATE LIBRARY. Arm B fixes every bond length and bond angle to a measured constant,
so its local geometry is correct within float precision and its torsions are the only degrees
of freedom. The price is that it CANNOT reproduce a native backbone exactly, and that residual
is the arm's template reconstruction floor -- measured by `script_utils/cp2lp_bb_calibrate.py`
on the same 101 native complexes the CA-CA window came from, and recorded here beside the
constants so the two can never drift apart. An Arm B edit RMSD at or below the floor is
template drift, not an edit.

Everything is in NANOMETRES and in the receptor frame the data pipeline centres on, matching
the rest of cp2lp. Angles are stored in RADIANS and reported in degrees.
"""

from __future__ import annotations

import math

import torch

from openfold.np.residue_constants import restype_order

from proteinfoundation.cp2lp.ca_internal import _dihedral, _unit
from proteinfoundation.cp2lp.geometry import (
    CA_IDX,
    C_IDX,
    N_IDX,
    O_IDX,
    safe_norm,
)

#: Arm-facing atom order. Index into the `[b, n, 4, 3]` dense tensor.
BB_NAMES = ("N", "CA", "C", "O")
BB_IDX = (N_IDX, CA_IDX, C_IDX, O_IDX)
N_BB_ATOMS = len(BB_NAMES)

#: Position of each backbone atom inside the dense tensor, by name.
BB_SLOT = {name: i for i, name in enumerate(BB_NAMES)}

# ----------------------------------------------------------------- template library
# MEASURED by `script_utils/cp2lp_bb_calibrate.py --report-floor` on the peptide chains of
# the 101 native complexes in `calibration_native/` -- the same set the CA-CA window and the
# clash threshold were calibrated on, so all three agree about what "native" means. Medians,
# not means: omega has a cis tail and the angles have a few strained outliers, and a median
# is what a typical residue actually has.
#
# PROVENANCE: slurm job 70313. Re-measure with
#   sbatch scripts/cp2lp_bb_calibrate.sbatch
# and the job prints a paste-ready block of exactly these names.
#
# NOTE how far two of these are from the textbook values this file first carried: C-N is
# 0.1348 nm here against the 0.1329 in `geometry.IDEAL_PEPTIDE_BOND_NM`, and N-CA-C is 113.03
# deg against the usual 111.2. The textbook numbers describe a relaxed protein backbone; these
# describe the short receptor-bound peptides this project actually generates.
TEMPLATE_N_CA_NM = 0.14553
TEMPLATE_CA_C_NM = 0.15251
TEMPLATE_C_N_NM = 0.13476
TEMPLATE_C_O_NM = 0.12290
TEMPLATE_N_CA_C_DEG = 113.030
TEMPLATE_CA_C_N_DEG = 117.812
TEMPLATE_C_N_CA_DEG = 123.792
TEMPLATE_CA_C_O_DEG = 121.038

#: Per-residue N-CA-C, for the two residues whose ring / missing CB shifts it. MEASURED, and
#: measured to be WORTH ALMOST NOTHING: substituting them moves the reconstruction floor from
#: 0.1180 to 0.1172 nm, i.e. 0.7% of it. Kept because they are free and correct, but a
#: per-residue template is not the lever anyone should reach for.
TEMPLATE_N_CA_C_PRO_DEG = 114.623
TEMPLATE_N_CA_C_GLY_DEG = 114.116

#: dihedral(N, CA, C, O) minus psi. Close to 180 deg -- the carbonyl points opposite the next
#: residue's N -- but measured rather than assumed, because the builder stores the OFFSET and
#: a few degrees of systematic error here tilts every carbonyl in the arm.
#:
#: MEASURING THIS WRONG IS EASY and the first run of the calibration did: the true value sits
#: at ~180, wrapping into (-180, 180] splits one tight cluster across both ends of the range,
#: and the reported spread was sd = 166 deg for a quantity whose real spread is 2.34 deg. The
#: calibration now measures the residual ABOUT 180 and adds it back.
TEMPLATE_O_DIHEDRAL_OFFSET_DEG = 179.030

# ------------------------------------------------------------- the floor, and its address
#
# FIXING BOND LENGTHS IS FREE. FIXING BOND ANGLES COSTS 1.18 ANGSTROM. Measured as a
# leave-one-out decomposition of the reconstruction floor -- extract a native's torsions,
# rebuild it, and see how far the rebuild lands from where it started:
#
#     all template            0.1180 nm      <- what `ANGLE_SOURCE_TEMPLATE` gives
#     + PRO/GLY angles        0.1172 nm         0.7% better, i.e. nothing
#     source ang_N_CA_C       0.0812 nm        31.2% of the floor lives in this one angle
#     source ang_C_N_CA       0.0874 nm        25.9%
#     source ang_CA_C_N       0.0919 nm        22.1%
#     source len_C_N          0.1181 nm        -0.1%   <- a bond LENGTH buys nothing
#     source len_CA_C         0.1181 nm        -0.1%
#     source len_N_CA         0.1186 nm        -0.5%
#     all source              0.0000 nm      <- exact identity, no floor at all
#
# So the three bond lengths can be pinned to constants at zero cost, and essentially the
# entire floor is the three bond angles. That matters because a 1.18 A floor is not a
# rounding error in this comparison: it is larger than the 0.5 A "identity map" edit the
# export gate was measured to select for, so under `ANGLE_SOURCE_TEMPLATE` every edit bin
# below 1.18 A is empty for Arm B *by construction* -- and edit-binned comparison against
# Arm A is exactly how this experiment avoids rewarding a generator for doing nothing.
#
# Hence `ANGLE_SOURCE_*` below. Both are implemented; which one an arm uses is a config knob,
# not a decision buried in this file.
#: Bond angles come from the measured template library. Local geometry is then ideal whatever
#: the generator emits -- including when the source backbone handed to it is an AE decode with
#: off-manifold angles. Costs the 1.18 nm floor above.
ANGLE_SOURCE_TEMPLATE = "template"
#: Bond angles are carried through from the SOURCE backbone, per residue, as fixed
#: non-editable constants, clamped into the native window the way `ca_internal` clamps its
#: bend angle -- the bound is the window OR the source's own value, whichever is more
#: permissive. Zero delta is then the EXACT identity and there is no floor. The guarantee
#: becomes "the edit cannot break local geometry" rather than "the output has ideal local
#: geometry", which is the same claim `ca_internal` makes about CA-CA bonds, and it is still
#: a guarantee: no reachable point of the edit space has a worse angle than its input.
ANGLE_SOURCE_SOURCE = "source"

#: DECIDED 2026-10-01, on the decomposition above: Arm B carries source angles by default.
#: Three reasons, in the order they weighed:
#:   1. A 1.18 A floor is larger than the 0.5 A "identity map" edit the export gate was
#:      measured to select for, so under the template every edit bin below 1.18 A is empty for
#:      Arm B BY CONSTRUCTION -- and edit-binned comparison against Arm A is precisely how
#:      this experiment avoids rewarding a generator for doing nothing.
#:   2. The zero-initialised head's whole purpose is that the arm's first output IS the
#:      source; the template can only deliver that to 1.18 A.
#:   3. Arm A starts from the SAME AE-decoded source backbone. Templating only Arm B's angles
#:      would make the two arms less comparable, not more -- the AE's angle quality would be
#:      a confound present in one arm and absent in the other.
#: `ANGLE_SOURCE_TEMPLATE` stays implemented and tested; it is one config key away.
DEFAULT_ANGLE_SOURCE = ANGLE_SOURCE_SOURCE

#: Native windows for the three chain angles, degrees. MEASURED p1/p99 above, padded outward.
#: Used only to clamp carried-through source angles under `ANGLE_SOURCE_SOURCE`.
ANGLE_WINDOW_DEG = {
    "N_CA_C": (100.0, 125.0),
    "CA_C_N": (112.0, 126.0),
    "C_N_CA": (117.0, 134.0),
}

# ------------------------------------------------------- and the SAME QUESTION for lengths
#
# THE FLOOR DECOMPOSITION'S "LENGTHS ARE FREE" WAS CONDITIONAL, and the condition is easy to
# miss: each `src:len_*` row was measured with the ANGLES STILL TEMPLATED, so the angle error
# (1.18 A) swamped the length error and the length rows read as ~0% of the total. Once the
# angles are carried from the source, the lengths are the ONLY remaining error -- and MEASURED
# on 12 native complexes they are not zero:
#
#     per-bond deviation from the template, mean / max
#         N-CA   0.0063 A / 0.0369 A
#         CA-C   0.0080 A / 0.0309 A
#         C-N    0.0044 A / 0.0234 A
#     resulting whole-peptide reconstruction   0.210 A RMSD, 2.037 A worst atom
#
# Individually negligible, and they still compound to 0.21 A because a length error early in
# the chain levers the far end. That is a conditional-independence trap: two factors each
# measured as ~0% of a total, which between them account for all of what is left when the
# dominant one is removed.
#
# So the lengths get the same treatment as the angles, for the same reason, and zero delta
# becomes the EXACT identity rather than a 0.21 A approximation of it.
LENGTH_SOURCE_TEMPLATE = "template"
LENGTH_SOURCE_SOURCE = "source"
DEFAULT_LENGTH_SOURCE = LENGTH_SOURCE_SOURCE

#: Native windows for the three bond lengths, nm. MEASURED p1/p99, padded well outward --
#: these only bound a pathological source, since no delta can move a bond length at all.
LENGTH_WINDOW_NM = {
    "N_CA": (0.130, 0.160),
    "CA_C": (0.138, 0.168),
    "C_N": (0.118, 0.152),
}

#: Canonical fallbacks for the torsions a terminal residue does not have: psi at the C
#: terminus (no N(i+1) to measure against) and phi at the N terminus (no C(i-1)). Only ever
#: used to place an O and, in Arm B, never used at all for a chain atom -- the first residue's
#: N/CA/C come from the root frame. Beta-region values; a terminal residue is solvent exposed.
FALLBACK_PSI_DEG = -45.0
FALLBACK_PHI_DEG = -60.0

#: Trans omega, for the degenerate case where a source omega cannot be measured. NOT applied
#: to measurable bonds: the source's own omega is carried through, so a cis bond stays cis.
FALLBACK_OMEGA_DEG = 180.0

#: Below this |omega| a peptide bond is cis. Reported, never enforced.
CIS_OMEGA_CUT_RAD = math.pi / 2.0

#: Native windows for the RAW validity check, in nm and degrees. Arm A is scored against
#: these; Arm B satisfies them by construction and is scored anyway, because "by construction"
#: is a claim about code and this is the measurement that tests it.
BOND_TOL_NM = 0.02
ANGLE_TOL_DEG = 15.0

_EPS = 1e-8


def _deg(x: float) -> float:
    return math.radians(x)


# ------------------------------------------------------------------ container plumbing


def extract_backbone(
    atom37: torch.Tensor, atom_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[b, n, 37, 3]`` -> dense ``[b, n, 4, 3]`` in (N, CA, C, O) order.

    Args:
        atom37: coordinates, nm.
        atom_mask: ``[b, n, 37]`` atom presence. When omitted every backbone atom is
            assumed present, which is right for a generated structure and wrong for a
            crystal one -- so callers handling real data pass it.

    Returns:
        ``(bb, bb_atom_mask)`` with shapes ``[b, n, 4, 3]`` and ``[b, n, 4]``.
    """
    idx = torch.as_tensor(BB_IDX, device=atom37.device, dtype=torch.long)
    bb = atom37.index_select(-2, idx)
    if atom_mask is None:
        m = torch.ones(bb.shape[:-1], dtype=torch.bool, device=atom37.device)
    else:
        m = atom_mask.index_select(-1, idx).bool()
    return bb, m


def backbone_to_atom37(
    bb: torch.Tensor,
    res_mask: torch.Tensor,
    bb_atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense ``[b, n, 4, 3]`` -> the shared ``[b, n, 37, 3]`` container plus its atom mask.

    THIS FUNCTION IS WHERE "BACKBONE ONLY" IS ENFORCED. Every atom37 slot other than
    N/CA/C/O comes back zero AND masked off, so each existing geometry/clash/contact/export
    function sees a structure that simply has no sidechain -- rather than one with sidechain
    atoms sitting at the origin, which would read as a catastrophic clash at the receptor
    centre and quietly poison every steric metric.

    Args:
        bb: ``[b, n, 4, 3]`` nm, (N, CA, C, O).
        res_mask: ``[b, n]`` residue mask. Padded residues get no atoms at all.
        bb_atom_mask: ``[b, n, 4]`` per-atom presence, optional; defaults to all present.

    Returns:
        ``(atom37, atom_mask)``.
    """
    b, n = bb.shape[0], bb.shape[1]
    atom37 = bb.new_zeros((b, n, 37, 3))
    atom_mask = torch.zeros((b, n, 37), dtype=torch.bool, device=bb.device)
    rm = res_mask.bool()[..., None]
    am = torch.ones(bb.shape[:-1], dtype=torch.bool, device=bb.device) if bb_atom_mask is None else bb_atom_mask.bool()
    am = am & rm
    for slot, a37 in enumerate(BB_IDX):
        atom37[:, :, a37, :] = bb[:, :, slot, :] * am[:, :, slot, None].to(bb.dtype)
        atom_mask[:, :, a37] = am[:, :, slot]
    return atom37, atom_mask


# ------------------------------------------------------------------------- Arm B: read


def bond_angle(a: torch.Tensor, vertex: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Angle a-vertex-c, radians. THE VERTEX IS THE MIDDLE ARGUMENT.

    Spelled out because the repository has already been bitten by exactly this: the
    `bond_angles` vertex trap recorded in the typed-topology work silently supervised the
    wrong angle by passing the three atoms in chain order instead of vertex-centred order.
    """
    return torch.arccos(
        (_unit(a - vertex) * _unit(c - vertex)).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    )


def backbone_to_torsions(bb: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Open-chain backbone -> (phi, psi, omega), the source's bond angles, and the root.

    No closing bond is read: ``omega`` and the two peptide-bond angles have ``n-1`` entries,
    so the terminal pair C(L-1)-N(0) is not a quantity this function can even express. A
    cyclic input is read as the linear chain it also is.

    Args:
        bb: ``[b, n, 4, 3]`` nm, (N, CA, C, O).
        mask: ``[b, n]`` residue mask, contiguous from index 0.

    Returns:
        ``phi`` ``[b, n]`` (entry 0 is the fallback -- there is no C(-1)),
        ``psi`` ``[b, n]`` (entry L-1 is the fallback -- there is no N(L)),
        ``omega`` ``[b, n-1]``, all radians,
        ``ang_n_ca_c`` ``[b, n]``, ``ang_ca_c_n`` / ``ang_c_n_ca`` ``[b, n-1]`` -- the
        source's OWN bond angles, radians, for `ANGLE_SOURCE_SOURCE`, and
        ``root`` ``[b, 3, 3]`` -- residue 0's N, CA, C, carried through unchanged.

    Padded positions hold arbitrary-but-finite values; the caller masks.
    """
    b, n = mask.shape
    nat, caat, cat = bb[:, :, BB_SLOT["N"]], bb[:, :, BB_SLOT["CA"]], bb[:, :, BB_SLOT["C"]]
    oat = bb[:, :, BB_SLOT["O"]]

    # phi(i) = dihedral(C(i-1), N(i), CA(i), C(i)); undefined at i=0.
    phi = torch.full((b, n), _deg(FALLBACK_PHI_DEG), device=bb.device, dtype=bb.dtype)
    if n >= 2:
        phi = torch.cat(
            [phi[:, :1], _dihedral(cat[:, :-1], nat[:, 1:], caat[:, 1:], cat[:, 1:])], dim=1
        )

    # psi(i) = dihedral(N(i), CA(i), C(i), N(i+1)); undefined at i=L-1. Held at n entries with
    # the tail as a fallback so phi/psi index by residue, which is what the head emits.
    # The LAST residue's psi is recovered from its OWN CARBONYL, not from a fallback. It has
    # no N(i+1), but it does have an O, and O sits at dihedral(N, CA, C, O) = psi + offset by
    # construction -- so psi(L-1) = dihedral(N, CA, C, O) - offset, and the information is
    # present in the input after all.
    #
    # USING THE FALLBACK HERE IS A REAL BUG, not a cosmetic one, and the arm tests caught it:
    # with a constant psi(L-1) the builder re-places the terminal O wherever the fallback
    # points, so Arm B reported a spurious ~0.14 nm "edit" on the C-terminal oxygen of every
    # peptide at ZERO head output. That would have shown up in evaluation as an edit the
    # generator never requested, on the residue whose chemistry the packing adapter then has
    # to get right.
    # AT EACH ROW'S OWN LAST REAL RESIDUE, not at column n-1. The batch is padded to the
    # widest peptide, so `bb[:, -1]` is a block of ZEROS on every shorter row -- reading psi
    # there fed a degenerate dihedral into the terminal O placement and put that O 2.0 A from
    # where it belongs, on exactly the residue whose terminal chemistry the packer then has to
    # get right. MEASURED before the fix: N, CA and C exact to 0.000 A, O out by 2.007 A and
    # 1.673 A on the two padded rows and 0.096 A on the unpadded one.
    if n >= 2:
        psi = _dihedral(nat[:, :-1], caat[:, :-1], cat[:, :-1], nat[:, 1:])
        psi = torch.cat(
            [psi, torch.full((b, 1), _deg(FALLBACK_PSI_DEG), device=bb.device, dtype=bb.dtype)],
            dim=1,
        )
        last = (mask.bool().sum(dim=1) - 1).clamp(min=0)  # [b] each row's own last residue
        rows = torch.arange(b, device=bb.device)
        psi_last = wrap_to_pi(
            _dihedral(
                nat[rows, last], caat[rows, last], cat[rows, last], oat[rows, last]
            )
            - _deg(TEMPLATE_O_DIHEDRAL_OFFSET_DEG)
        )
        psi = psi.clone()
        psi[rows, last] = psi_last
    else:
        psi = torch.full((b, n), _deg(FALLBACK_PSI_DEG), device=bb.device, dtype=bb.dtype)

    # omega(i) = dihedral(CA(i), C(i), N(i+1), CA(i+1)); one per peptide BOND.
    if n >= 2:
        omega = _dihedral(caat[:, :-1], cat[:, :-1], nat[:, 1:], caat[:, 1:])
    else:
        omega = bb.new_zeros((b, 0))

    # The source's own bond angles. Under ANGLE_SOURCE_SOURCE these replace the template
    # constants, which the floor decomposition showed removes the floor entirely.
    ang_n_ca_c = bond_angle(nat, caat, cat)  # [b, n], vertex CA
    len_n_ca = safe_norm(caat - nat, dim=-1)  # [b, n]
    len_ca_c = safe_norm(cat - caat, dim=-1)  # [b, n]
    # The carbonyl's own geometry. Does NOT propagate along the chain -- O is a leaf off C --
    # but it is carried anyway so that zero delta reproduces the source's O exactly too.
    len_c_o = safe_norm(oat - cat, dim=-1)  # [b, n]
    ang_ca_c_o = bond_angle(caat, cat, oat)  # [b, n], vertex C
    # The O dihedral as an OFFSET from psi, per residue. The template constant is the
    # median 179.03 deg and the real spread is +-2.34 deg, which on a 1.23 A bond is ~0.04 A
    # of O displacement -- MEASURED as the 0.096 A residual that survived after lengths and
    # angles were already carried. Carried too, so the identity is exact on all four atoms.
    dihe_o_offset = wrap_to_pi(_dihedral(nat, caat, cat, oat) - psi)  # [b, n]
    if n >= 2:
        ang_ca_c_n = bond_angle(caat[:, :-1], cat[:, :-1], nat[:, 1:])  # vertex C(i)
        ang_c_n_ca = bond_angle(cat[:, :-1], nat[:, 1:], caat[:, 1:])  # vertex N(i+1)
        len_c_n = safe_norm(nat[:, 1:] - cat[:, :-1], dim=-1)  # [b, n-1]
    else:
        ang_ca_c_n = bb.new_zeros((b, 0))
        ang_c_n_ca = bb.new_zeros((b, 0))
        len_c_n = bb.new_zeros((b, 0))

    root = torch.stack([nat[:, 0], caat[:, 0], cat[:, 0]], dim=1)  # [b, 3, 3]
    return {
        "phi": phi,
        "psi": psi,
        "omega": omega,
        "ang_n_ca_c": ang_n_ca_c,
        "ang_ca_c_n": ang_ca_c_n,
        "ang_c_n_ca": ang_c_n_ca,
        "len_n_ca": len_n_ca,
        "len_ca_c": len_ca_c,
        "len_c_n": len_c_n,
        "len_c_o": len_c_o,
        "ang_ca_c_o": ang_ca_c_o,
        "dihe_o_offset": dihe_o_offset,
        "root": root,
    }


def resolve_chain_geometry(
    internal: dict[str, torch.Tensor],
    angle_source: str = DEFAULT_ANGLE_SOURCE,
    aatype: torch.Tensor | None = None,
    length_source: str = DEFAULT_LENGTH_SOURCE,
) -> dict[str, torch.Tensor]:
    """Decide the fixed lengths/angles the builder constructs with. Returns radians and nm.

    Bond LENGTHS follow `length_source`, for exactly the reasons the angles do -- see the
    conditional-independence note above the `LENGTH_SOURCE_*` constants. Carrying them is what
    takes the reconstruction floor from 0.21 A to zero on real complexes.

    Bond ANGLES follow `angle_source`:
      * `ANGLE_SOURCE_TEMPLATE` -- the library constants, optionally PRO/GLY-specific when
        `aatype` is given. Ideal local geometry, 1.18 A floor.
      * `ANGLE_SOURCE_SOURCE` -- the source's own angles, clamped source-relatively into the
        native window. Exact identity at zero delta, no floor.

    THE CLAMP IS SOURCE-RELATIVE, for the reason `ca_internal` documents at length: a plain
    clamp breaks the exact-identity property on every input that already sits outside the
    window, and encoded-then-decoded sources DO sit outside it. The bound is the window or
    the source's own angle, whichever is more permissive, so zero delta is exactly the
    identity while the arm still cannot push an angle further out than it already was.
    """
    out: dict[str, torch.Tensor] = {}

    # --- bond lengths. Same source-relative clamp discipline as the angles.
    if length_source == LENGTH_SOURCE_SOURCE:
        for key, win in (("len_n_ca", "N_CA"), ("len_ca_c", "CA_C"), ("len_c_n", "C_N")):
            d = internal[key]
            lo, hi = LENGTH_WINDOW_NM[win]
            lo_i = torch.minimum(d, torch.full_like(d, lo))
            hi_i = torch.maximum(d, torch.full_like(d, hi))
            out[key] = torch.minimum(torch.maximum(d, lo_i), hi_i)
        out["len_c_o"] = internal["len_c_o"]
        out["ang_ca_c_o"] = internal["ang_ca_c_o"]
        out["dihe_o_offset"] = internal["dihe_o_offset"]
    elif length_source == LENGTH_SOURCE_TEMPLATE:
        out["len_n_ca"] = torch.full_like(internal["len_n_ca"], TEMPLATE_N_CA_NM)
        out["len_ca_c"] = torch.full_like(internal["len_ca_c"], TEMPLATE_CA_C_NM)
        out["len_c_n"] = torch.full_like(internal["len_c_n"], TEMPLATE_C_N_NM)
        out["len_c_o"] = torch.full_like(internal["len_c_o"], TEMPLATE_C_O_NM)
        out["ang_ca_c_o"] = torch.full_like(internal["ang_ca_c_o"], _deg(TEMPLATE_CA_C_O_DEG))
        out["dihe_o_offset"] = torch.full_like(
            internal["dihe_o_offset"], _deg(TEMPLATE_O_DIHEDRAL_OFFSET_DEG)
        )
    else:
        raise ValueError(
            f"unknown length_source {length_source!r}; expected "
            f"{LENGTH_SOURCE_TEMPLATE!r} or {LENGTH_SOURCE_SOURCE!r}"
        )

    # --- bond angles
    if angle_source == ANGLE_SOURCE_SOURCE:
        for key, win in (
            ("ang_n_ca_c", "N_CA_C"), ("ang_ca_c_n", "CA_C_N"), ("ang_c_n_ca", "C_N_CA"),
        ):
            a = internal[key]
            lo, hi = (_deg(v) for v in ANGLE_WINDOW_DEG[win])
            lo_i = torch.minimum(a, torch.full_like(a, lo))
            hi_i = torch.maximum(a, torch.full_like(a, hi))
            out[key] = torch.minimum(torch.maximum(a, lo_i), hi_i)
        return out

    if angle_source != ANGLE_SOURCE_TEMPLATE:
        raise ValueError(
            f"unknown angle_source {angle_source!r}; expected "
            f"{ANGLE_SOURCE_TEMPLATE!r} or {ANGLE_SOURCE_SOURCE!r}"
        )

    ref = internal["ang_n_ca_c"]
    tau = torch.full_like(ref, _deg(TEMPLATE_N_CA_C_DEG))
    if aatype is not None:
        # PRO/GLY substitution. Measured to be worth 0.7% of the floor -- applied because it
        # is free and correct, not because it is a lever.
        for code, deg in (("P", TEMPLATE_N_CA_C_PRO_DEG), ("G", TEMPLATE_N_CA_C_GLY_DEG)):
            idx = restype_order.get(code)
            if idx is not None:
                tau = torch.where(aatype == idx, torch.full_like(tau, _deg(deg)), tau)
    out["ang_n_ca_c"] = tau
    out["ang_ca_c_n"] = torch.full_like(internal["ang_ca_c_n"], _deg(TEMPLATE_CA_C_N_DEG))
    out["ang_c_n_ca"] = torch.full_like(internal["ang_c_n_ca"], _deg(TEMPLATE_C_N_CA_DEG))
    return out


# ------------------------------------------------------------------------ Arm B: build


def _place(
    p_a: torch.Tensor,
    p_b: torch.Tensor,
    p_c: torch.Tensor,
    bond: float | torch.Tensor,
    angle: float | torch.Tensor,
    torsion: torch.Tensor,
) -> torch.Tensor:
    """NeRF: place d with |c-d|=bond, angle(b,c,d)=angle, dihedral(a,b,c,d)=torsion.

    Sign convention matches `ca_internal.internal_to_ca` exactly -- that one was fixed by
    searching all four sign choices for the one that round-trips a concrete trace, and
    getting it wrong here produces valid-LOOKING backbones that are not the input. The
    round-trip test is the only thing that catches it, and `test_cp2lp_backbone.py` runs it.
    """
    bc = _unit(p_c - p_b)
    nrm = _unit(torch.cross(p_b - p_a, bc, dim=-1))
    m2 = torch.cross(nrm, bc, dim=-1)
    if not torch.is_tensor(bond):
        bond = torch.as_tensor(bond, device=p_c.device, dtype=p_c.dtype)
    if not torch.is_tensor(angle):
        angle = torch.as_tensor(angle, device=p_c.device, dtype=p_c.dtype)
    sin_a, cos_a = torch.sin(angle), torch.cos(angle)
    off = (
        (-bond * cos_a)[..., None] * bc
        + (bond * sin_a * torch.cos(torsion))[..., None] * m2
        + (bond * sin_a * torch.sin(torsion))[..., None] * nrm
    )
    return p_c + off


def _expand(x: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    return torch.full(like.shape[:1], float(x), device=like.device, dtype=like.dtype)


def place_oxygen(
    n_i: torch.Tensor,
    ca_i: torch.Tensor,
    c_i: torch.Tensor,
    psi_i: torch.Tensor,
    bond: float | torch.Tensor = TEMPLATE_C_O_NM,
    angle: float | torch.Tensor | None = None,
    offset: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Carbonyl O from the residue's own N, CA, C and its psi.

    O is NOT a free output in either arm's chemistry: it is rigidly determined by the
    residue's own frame plus psi, and placing it any other way is how a backbone acquires a
    non-planar carbonyl that no scorer will accept. Arm A predicts it freely anyway -- that
    is Arm A's whole premise, and its raw validity check is where the difference shows up.
    """
    if offset is None:
        offset = _deg(TEMPLATE_O_DIHEDRAL_OFFSET_DEG)
    if not torch.is_tensor(offset):
        offset = torch.as_tensor(offset, device=psi_i.device, dtype=psi_i.dtype)
    if angle is None:
        angle = _deg(TEMPLATE_CA_C_O_DEG)
    return _place(n_i, ca_i, c_i, bond, angle, psi_i + offset)


def torsions_to_backbone(
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
    root: torch.Tensor,
    mask: torch.Tensor,
    geom: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Torsions + root segment -> full ``[b, n, 4, 3]`` backbone, differentiably.

    Builds the N-CA-C chain by sequential NeRF placement and then places every O from its own
    residue's frame. Bond lengths and the three chain angles are the template library, so
    they are correct within float precision for every output this function can produce -- the
    guarantee that is Arm B's reason to exist.

    NO CLOSING BOND IS BUILT. The chain runs 0 -> L-1 and stops; `omega` is indexed by bond
    and has L-1 entries, so there is no index at which a terminal bond could be placed.

    Args:
        phi: ``[b, n]`` radians. ``phi[0]`` is unused -- residue 0's C comes from the root.
        psi: ``[b, n]`` radians. ``psi[i]`` places N(i+1); ``psi[L-1]`` places only O(L-1).
        omega: ``[b, n-1]`` radians. ``omega[i]`` places CA(i+1). The source's own value, so
            a cis bond is preserved rather than forced trans.
        root: ``[b, 3, 3]`` residue 0's N, CA, C.
        mask: ``[b, n]`` residue mask, contiguous from index 0.
        geom: the fixed bond angles, radians, from :func:`resolve_chain_geometry` --
            ``ang_n_ca_c`` ``[b, n]`` and ``ang_ca_c_n`` / ``ang_c_n_ca`` ``[b, n-1]``.
            ``None`` means the template constants, i.e. `ANGLE_SOURCE_TEMPLATE` with no
            PRO/GLY handling.

    Returns:
        ``[b, n, 4, 3]``, zero on padded residues.
    """
    b, n = mask.shape
    dt = root.dtype
    if n < 1:
        return root.new_zeros((b, 0, N_BB_ATOMS, 3))

    def fixed(key: str, value: float, width: int, is_angle: bool) -> torch.Tensor:
        if geom is not None and key in geom:
            return geom[key]
        v = _deg(value) if is_angle else value
        return torch.full((b, width), v, device=root.device, dtype=dt)

    tau = fixed("ang_n_ca_c", TEMPLATE_N_CA_C_DEG, n, True)
    a_cacn = fixed("ang_ca_c_n", TEMPLATE_CA_C_N_DEG, max(n - 1, 0), True)
    a_cnca = fixed("ang_c_n_ca", TEMPLATE_C_N_CA_DEG, max(n - 1, 0), True)
    l_nca = fixed("len_n_ca", TEMPLATE_N_CA_NM, n, False)
    l_cac = fixed("len_ca_c", TEMPLATE_CA_C_NM, n, False)
    l_cn = fixed("len_c_n", TEMPLATE_C_N_NM, max(n - 1, 0), False)

    ns = [root[:, 0]]
    cas = [root[:, 1]]
    cs = [root[:, 2]]
    for i in range(n - 1):
        # N(i+1): along the peptide bond, oriented by psi(i).
        n_next = _place(ns[i], cas[i], cs[i], l_cn[:, i], a_cacn[:, i], psi[:, i])
        # CA(i+1): oriented by omega(i). This is where cis/trans lives -- the source's own
        # omega is carried through by the caller, so a cis bond stays cis rather than being
        # forced trans.
        ca_next = _place(cas[i], cs[i], n_next, l_nca[:, i + 1], a_cnca[:, i], omega[:, i])
        # C(i+1): oriented by phi(i+1), and closing the intra-residue angle tau(i+1).
        c_next = _place(cs[i], n_next, ca_next, l_cac[:, i + 1], tau[:, i + 1], phi[:, i + 1])
        ns.append(n_next)
        cas.append(ca_next)
        cs.append(c_next)

    nat = torch.stack(ns, dim=1)
    caat = torch.stack(cas, dim=1)
    cat = torch.stack(cs, dim=1)
    oat = place_oxygen(
        nat, caat, cat, psi,
        bond=fixed("len_c_o", TEMPLATE_C_O_NM, n, False),
        angle=fixed("ang_ca_c_o", TEMPLATE_CA_C_O_DEG, n, True),
        offset=fixed("dihe_o_offset", TEMPLATE_O_DIHEDRAL_OFFSET_DEG, n, True),
    )

    bb = torch.stack([nat, caat, cat, oat], dim=2)  # [b, n, 4, 3]
    return bb * mask.bool()[:, :, None, None].to(dt)


# ----------------------------------------------------------------- angular residuals


def apply_angle_residual(base: torch.Tensor, delta_sin: torch.Tensor, delta_cos: torch.Tensor,
                         scale: float = 1.0) -> torch.Tensor:
    """Rotate `base` by the angle that ``(delta_sin, delta_cos)`` encodes. Radians in, out.

    WHY NOT A PLAIN SCALAR. An angle lives on a circle, and a head that regresses it as a
    real number has a discontinuity at +-pi: two angles a thousandth of a radian apart get
    squared errors differing by (2 pi)^2, so the gradient at the seam points the wrong way.
    Encoding the DELTA as an unnormalised (sin, cos) pair and composing the rotation makes the
    parameterisation continuous everywhere and makes zero the exact identity:

        delta_angle = atan2(s, 1 + c)      with the head emitting s, c about zero

    ``1 + c`` rather than ``c`` is what puts the identity at the ORIGIN of the head's output
    instead of at ``(0, 1)``, which a zero-initialised head cannot reach. At ``s = c = 0``
    this is ``atan2(0, 1) = 0`` exactly, so the arm's first output is the source's own
    torsions and the builder reproduces the source to the template floor.

    `scale` damps the residual the way `internal_residual_scale` does: a torsion at i swings
    the entire chain downstream of i, so the lever arm grows with L and the raw head output is
    badly scaled against a Cartesian residual. Not solved here, only damped.
    """
    s = scale * delta_sin
    c = 1.0 + scale * delta_cos
    # No degenerate branch needed: `c` is 1 at the origin, and reaching atan2(0, 0) requires
    # the head to emit exactly c = -1/scale with s = 0, which is a measure-zero point that
    # `wrap_to_pi` below would handle anyway. Guarded so a pathological head cannot NaN.
    deg = (s * s + c * c) < 1e-12
    s = torch.where(deg, torch.zeros_like(s), s)
    c = torch.where(deg, torch.ones_like(c), c)
    return wrap_to_pi(base + torch.atan2(s, c))


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Map an angle into (-pi, pi]. Differentiable almost everywhere, identity near 0."""
    return x - 2.0 * math.pi * torch.floor((x + math.pi) / (2.0 * math.pi))


# ------------------------------------------------------------------- raw validity check


def backbone_validity(
    bb: torch.Tensor,
    mask: torch.Tensor,
    bond_tol_nm: float = BOND_TOL_NM,
    angle_tol_deg: float = ANGLE_TOL_DEG,
) -> dict[str, torch.Tensor]:
    """RAW open-chain geometry of a generated backbone. No repair, no tolerance tuning.

    Scores the four things a linear peptide backbone has to get right, each as a deviation
    from the template and as a pass fraction against the native-calibrated window:

        peptide bond   C(i)-N(i+1) length, the bond the 24-step rollout broke by ~70x
        intra bonds    N-CA and CA-C
        chain angles   CA-C-N, C-N-CA, N-CA-C
        chirality      the sign of dihedral(N, CA, C, O) relative to psi -- an L-amino-acid
                       backbone has a consistent carbonyl handedness, and a Cartesian head
                       free to place O anywhere can produce the mirror image, which every
                       length and angle check passes

    `chain_intact` is the ALL-BONDS test: one surviving broken bond costs the whole peptide,
    because a peptide with a break in it is not a peptide. Reaching 0.95 of peptides needs a
    per-bond pass rate of 0.9945, which is why a near-miss distribution is no comfort.

    Returns per-example tensors ``[b]`` plus scalar fractions, all detached -- this is a
    measurement, never a loss.
    """
    with torch.no_grad():
        b, n = mask.shape
        m = mask.bool()
        nat, caat, cat, oat = (bb[:, :, BB_SLOT[k]] for k in BB_NAMES)
        out: dict[str, torch.Tensor] = {}

        pair = m[:, :-1] & m[:, 1:] if n > 1 else m[:, :0]
        npair = pair.float().sum(dim=1).clamp(min=1.0)

        # --- peptide bond
        if n > 1:
            d_cn = safe_norm(nat[:, 1:] - cat[:, :-1], dim=-1)
            dev_cn = (d_cn - TEMPLATE_C_N_NM).abs()
            ok_cn = (dev_cn <= bond_tol_nm) | ~pair
            out["bb_peptide_bond_dev_nm"] = (dev_cn * pair).sum(1) / npair
            out["bb_peptide_bond_pass"] = (ok_cn & pair).float().sum(1) / npair
            intact = ok_cn.all(dim=1)
        else:
            out["bb_peptide_bond_dev_nm"] = bb.new_zeros(b)
            out["bb_peptide_bond_pass"] = bb.new_ones(b)
            intact = torch.ones(b, dtype=torch.bool, device=bb.device)

        # --- intra-residue bonds
        nres = m.float().sum(dim=1).clamp(min=1.0)
        d_nca = safe_norm(caat - nat, dim=-1)
        d_cac = safe_norm(cat - caat, dim=-1)
        dev_intra = (d_nca - TEMPLATE_N_CA_NM).abs() + (d_cac - TEMPLATE_CA_C_NM).abs()
        ok_intra = ((d_nca - TEMPLATE_N_CA_NM).abs() <= bond_tol_nm) & (
            (d_cac - TEMPLATE_CA_C_NM).abs() <= bond_tol_nm
        )
        out["bb_intra_bond_dev_nm"] = (dev_intra * m).sum(1) / nres
        out["bb_intra_bond_pass"] = ((ok_intra | ~m) & m).float().sum(1) / nres
        intact = intact & (ok_intra | ~m).all(dim=1)

        # --- chain angles
        def ang(a, v, c):
            return torch.arccos(
                (_unit(a - v) * _unit(c - v)).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            ).rad2deg()

        tau = ang(nat, caat, cat)
        dev_tau = (tau - TEMPLATE_N_CA_C_DEG).abs()
        ok_tau = dev_tau <= angle_tol_deg
        out["bb_tau_dev_deg"] = (dev_tau * m).sum(1) / nres
        out["bb_tau_pass"] = ((ok_tau | ~m) & m).float().sum(1) / nres

        if n > 1:
            a_cacn = ang(caat[:, :-1], cat[:, :-1], nat[:, 1:])
            a_cnca = ang(cat[:, :-1], nat[:, 1:], caat[:, 1:])
            dev_pep = (a_cacn - TEMPLATE_CA_C_N_DEG).abs() + (a_cnca - TEMPLATE_C_N_CA_DEG).abs()
            ok_pep = ((a_cacn - TEMPLATE_CA_C_N_DEG).abs() <= angle_tol_deg) & (
                (a_cnca - TEMPLATE_C_N_CA_DEG).abs() <= angle_tol_deg
            )
            out["bb_peptide_angle_dev_deg"] = (dev_pep * pair).sum(1) / npair
            out["bb_peptide_angle_pass"] = ((ok_pep | ~pair) & pair).float().sum(1) / npair
            intact = intact & (ok_pep | ~pair).all(dim=1)
            # omega: reported, not gated. A cis bond is rare but real and not an error.
            om = _dihedral(caat[:, :-1], cat[:, :-1], nat[:, 1:], caat[:, 1:])
            out["bb_cis_frac"] = ((om.abs() < CIS_OMEGA_CUT_RAD) & pair).float().sum(1) / npair
        else:
            out["bb_peptide_angle_dev_deg"] = bb.new_zeros(b)
            out["bb_peptide_angle_pass"] = bb.new_ones(b)
            out["bb_cis_frac"] = bb.new_zeros(b)
        intact = intact & (ok_tau | ~m).all(dim=1)

        # --- carbonyl handedness. The O dihedral should sit near psi + 180; the MIRROR
        # backbone sits near psi - 180, which is the same angle, so the discriminating
        # quantity is the out-of-plane component: O must be close to the N-CA-C-defined
        # plane's expected side. Measured as |dihedral(N,CA,C,O) - (psi + offset)|.
        if n > 1:
            psi = torch.cat(
                [
                    _dihedral(nat[:, :-1], caat[:, :-1], cat[:, :-1], nat[:, 1:]),
                    torch.full((b, 1), _deg(FALLBACK_PSI_DEG), device=bb.device, dtype=bb.dtype),
                ],
                dim=1,
            )
            o_di = _dihedral(nat, caat, cat, oat)
            off = wrap_to_pi(o_di - psi - _deg(TEMPLATE_O_DIHEDRAL_OFFSET_DEG)).abs().rad2deg()
            # The last residue's psi is a fallback, so its offset is meaningless: drop it.
            valid_o = m.clone()
            valid_o[:, -1] = False
            n_o = valid_o.float().sum(dim=1).clamp(min=1.0)
            out["bb_carbonyl_dev_deg"] = (off * valid_o).sum(1) / n_o
            out["bb_carbonyl_pass"] = (
                ((off <= angle_tol_deg) | ~valid_o) & valid_o
            ).float().sum(1) / n_o
        else:
            out["bb_carbonyl_dev_deg"] = bb.new_zeros(b)
            out["bb_carbonyl_pass"] = bb.new_ones(b)

        out["bb_chain_intact"] = intact.float()
        return out
