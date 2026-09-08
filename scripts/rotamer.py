"""Place a complete, ideal side chain on an existing backbone.

Why this exists
---------------
`anchor_graft.py` changes a terminal residue's identity so the cyclization head has a
candidate anchor pair to bond. The first version of that graft kept the backbone plus CB and
dropped every other side-chain atom, on the argument that inventing a rotamer would be
dishonest. Measured on the first grafted rows, **every grafted endpoint decoded as alanine**:

    graft  disulfide  A.VQGGAAGH.S    (wanted C ... C)
    graft  disulfide  A.GLTIYAQKQ.A   (wanted C ... C)

The reason is arithmetic, not subtlety. ALA's complete atom37 set is exactly
``[N, CA, C, CB, O]`` -- precisely what that truncation leaves behind. The AE encoder reads
`x1_aatype` (a 20-d one-hot), `x1_a37coors_nm` (coordinates **and** the 37-d occupancy mask)
and `x1_sidechain_angles` (chi angles recomputed from the coordinates, so a masked-out SG
means "no chi1") *together*, and it was only ever trained on residues where those three
agree. "CYS with no SG and no chi1" is off-manifold; the decoder resolves the contradiction
in favour of the geometry, and the geometry is an alanine. The graft was erased by the
encode/decode round trip.

So the side chain has to be built, not dropped. That is also what a chemist means by "mutate
the termini to cysteine": PyMOL's mutagenesis wizard and Rosetta's `MutateResidue` both place
a full side chain at a library rotamer. The honest description of the input is "a cysteine at
a standard rotamer", which is a real residue, rather than "a cysteine with no side chain",
which is not.

How
---
Straight AlphaFold2 all-atom geometry, via openfold's own `torsion_angles_to_frames` and
`frames_and_literature_positions_to_atom14_pos` (generic in the atom axis, so passing the
atom**37** constants yields atom37 directly). The backbone rigid-group frame is built from
the residue's *real* N/CA/C, so the side chain is attached to the actual backbone rather than
to an idealised one.

What is kept and what is replaced
---------------------------------
Only CB and the chi-group atoms (rigid groups >= 4) are taken from the build. N, CA and C
define the frame -- rebuilding them would silently substitute ideal internal geometry for the
measured backbone -- and O lives in the psi group, which we are not predicting. Those four
are copied through untouched.

The occupancy mask is likewise not simply set to the target residue's full atom set: the
built atoms are certain, but N/CA/C/O are only present if the *source* residue had them (a
terminal residue with missing density does happen). The returned mask is therefore
``built | (source & allowed-for-the-new-type)``, so the graft never claims an atom nobody
ever observed or placed.

Rotamer choice
--------------
`MODAL_CHI_DEG` is the most-populated backbone-independent rotamer for each residue the
anchor spec can ask for. It is deliberately *not* tuned to bring the two anchor atoms
together: the whole point of the grafted arm is to measure whether the bond-distance DPS term
can close the ring, and pre-orienting the side chains toward each other would be optimising
the thing under test. Callers that want a different rotamer pass `chi_rad` explicitly.
"""

from __future__ import annotations

import torch
from openfold.np import residue_constants as rc
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.rigid_utils import Rigid, Rotation

N_IDX, CA_IDX, C_IDX, O_IDX, CB_IDX = (rc.atom_order[a] for a in ("N", "CA", "C", "O", "CB"))
# `restype_atom37_mask` gives OXT to no residue at all, so intersecting with it would strip
# the terminal carboxylate off every grafted C-terminal residue. OXT is a backbone atom whose
# presence depends on the residue being terminal, not on which residue it is -- the same
# footing as N/CA/C/O -- so it is carried through from the source instead.
OXT_IDX = rc.atom_order["OXT"]

# Most-populated backbone-independent rotamer per residue, in degrees, chi1 first.
# Only the residues `anchor_graft.ANCHOR_SPEC` can request are listed; anything else raises
# rather than silently defaulting to an eclipsed chi=0 conformation, which is not a rotamer
# any of these side chains actually adopts.
MODAL_CHI_DEG: dict[str, tuple[float, ...]] = {
    "CYS": (-65.0,),                      # m
    "ASP": (-70.0, -15.0),                # m-20
    "ASN": (-65.0, -20.0),                # m-40
    "GLU": (-67.0, 180.0, -10.0),         # mt-10
    "GLN": (-67.0, 180.0, -25.0),         # mt-30
    "LYS": (-67.0, 180.0, 180.0, 180.0),  # mttt
}

_CONST_CACHE: dict[tuple[torch.dtype, torch.device], dict[str, torch.Tensor]] = {}


def _consts(dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    """openfold's atom37 rigid-group constants, materialised once per (dtype, device)."""
    key = (dtype, device)
    if key not in _CONST_CACHE:
        _CONST_CACHE[key] = {
            "default_frames": torch.tensor(
                rc.restype_rigid_group_default_frame, dtype=dtype, device=device),
            "group_idx": torch.tensor(
                rc.restype_atom37_to_rigid_group, dtype=torch.long, device=device),
            "atom_mask": torch.tensor(
                rc.restype_atom37_mask, dtype=dtype, device=device),
            "lit_positions": torch.tensor(
                rc.restype_atom37_rigid_group_positions, dtype=dtype, device=device),
        }
    return _CONST_CACHE[key]


def n_chi(aa_idx: int) -> int:
    """How many chi angles this residue type has."""
    return int(sum(rc.chi_angles_mask[aa_idx]))


def modal_chi_rad(aatype: torch.Tensor) -> torch.Tensor:
    """[K] residue indices -> [K, 4] chi angles in radians, zero-padded past the last chi."""
    out = torch.zeros(aatype.shape[0], 4, dtype=torch.float32, device=aatype.device)
    for k, idx in enumerate(aatype.tolist()):
        if not 0 <= idx < 20:
            raise SystemExit(f"FATAL: cannot build a side chain for residue index {idx}")
        name = rc.restype_1to3[rc.restypes[idx]]
        if name not in MODAL_CHI_DEG:
            raise SystemExit(
                f"FATAL: no modal rotamer recorded for {name}. Add one to "
                f"rotamer.MODAL_CHI_DEG -- defaulting to chi=0 would place an eclipsed "
                f"side chain, which is not a conformation this residue adopts.")
        chi = MODAL_CHI_DEG[name]
        if len(chi) != n_chi(idx):
            raise SystemExit(
                f"FATAL: MODAL_CHI_DEG[{name}] has {len(chi)} angles but {name} has "
                f"{n_chi(idx)} chi angles.")
        out[k, :len(chi)] = torch.deg2rad(torch.tensor(chi, device=aatype.device))
    return out


def build_sidechain(
    coords_A: torch.Tensor,
    src_mask: torch.Tensor,
    aatype: torch.Tensor,
    chi_rad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach a complete side chain of type `aatype` to each residue's existing backbone.

    Args:
        coords_A: [K, 37, 3] atom37 coordinates in ANGSTROM. Only N, CA, C (frame) and O
            (copied through) are read; every other slot is overwritten or zeroed.
        src_mask: [K, 37] bool occupancy of the *source* residue, used to decide whether
            N/CA/C/O may be claimed for the new residue.
        aatype: [K] long, the TARGET residue types.
        chi_rad: [K, 4] chi angles in radians, or None for `MODAL_CHI_DEG`.

    Returns:
        (coords_A, mask, built): [K, 37, 3], [K, 37] bool and [K, 37] bool. `built` flags the
        slots this function actually placed, so a caller can write back only those and leave
        N/CA/C/O bit-identical -- which matters because `coords` and `coords_nm` are separate
        tensors and a round trip through either unit would perturb the backbone it copied.
    """
    if coords_A.ndim != 3 or coords_A.shape[-2:] != (37, 3):
        raise SystemExit(f"FATAL: build_sidechain wants [K, 37, 3]; got {tuple(coords_A.shape)}")
    dtype = coords_A.dtype if coords_A.is_floating_point() else torch.float32
    device = coords_A.device
    k = _consts(dtype, device)

    # A frame built from a missing atom is silently garbage -- and would place the side chain
    # at a plausible-looking wrong position rather than failing.
    need = torch.tensor([N_IDX, CA_IDX, C_IDX], device=device)
    if not bool(src_mask[:, need].all()):
        bad = (~src_mask[:, need].all(dim=-1)).nonzero().flatten().tolist()
        raise SystemExit(
            f"FATAL: cannot graft onto residue(s) {bad}: N, CA and C must all be present to "
            f"define the backbone frame.")

    if chi_rad is None:
        chi_rad = modal_chi_rad(aatype)
    chi_rad = chi_rad.to(dtype=dtype, device=device)

    # Rigid group 0's frame is (p_neg_x_axis=C, origin=CA, p_xy_plane=N) -- openfold sets
    # restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"] in
    # data_transforms.atom37_to_frames. Swapping C and N here mirrors every side chain into a
    # D-amino acid, which is why test_rotamer.py checks the CB chirality against real PDBs.
    r = Rigid.from_3_points(
        p_neg_x_axis=coords_A[:, C_IDX].to(dtype),
        origin=coords_A[:, CA_IDX].to(dtype),
        p_xy_plane=coords_A[:, N_IDX].to(dtype),
    )
    # ...and then openfold rotates group 0 by diag(-1, 1, -1) (a half turn about y) before the
    # literature positions are applied -- see the `rots[..., 0, 0, 0] = -1` block at the end of
    # atom37_to_frames. `from_3_points` is Algorithm 21, whose x axis runs CA->C, while
    # `restype_atom37_rigid_group_positions` is expressed in the opposite-handed convention.
    # Omitting this places CB a median 2.63 A from where it belongs (measured over 1945 real
    # residues; 0.05 A with it) -- with ideal bond LENGTHS and ANGLES throughout, so nothing
    # looks wrong locally. That is why the check that caught it compares against real crystal
    # side chains rather than against the constants the builder itself uses.
    fix = torch.eye(3, dtype=dtype, device=device).expand(coords_A.shape[0], 3, 3).clone()
    fix[:, 0, 0] = -1.0
    fix[:, 2, 2] = -1.0
    r = r.compose(Rigid(Rotation(rot_mats=fix), None))

    # [K, 7, 2] as (sin, cos): slots 0-2 are omega/phi/psi, which we are not changing, so they
    # get the identity rotation (sin 0, cos 1); slots 3-6 are chi1..chi4.
    alpha = torch.zeros(coords_A.shape[0], 7, 2, dtype=dtype, device=device)
    alpha[..., 1] = 1.0
    alpha[:, 3:, 0] = torch.sin(chi_rad)
    alpha[:, 3:, 1] = torch.cos(chi_rad)

    frames = torsion_angles_to_frames(r, alpha, aatype, k["default_frames"])
    built = frames_and_literature_positions_to_atom14_pos(
        frames, aatype, k["default_frames"], k["group_idx"], k["atom_mask"], k["lit_positions"],
    )  # [K, 37, 3] -- atom37 constants in, atom37 out

    allowed = k["atom_mask"][aatype].bool().clone()  # [K, 37] the new residue's full atom set
    allowed[:, OXT_IDX] = True                       # identity-independent; see OXT_IDX above
    # Take CB (rigid group 0, so it is exact given the real frame) and every chi-group atom.
    # N/CA/C define the frame and O belongs to the psi group: replacing those would swap the
    # measured backbone for an idealised one.
    take = (k["group_idx"][aatype] >= 4) & allowed
    take[:, CB_IDX] = allowed[:, CB_IDX]
    take[:, OXT_IDX] = False                         # copied through, never placed

    out = coords_A.clone().to(dtype)
    out[take] = built[take]
    # Backbone slots survive only if the source actually had them; everything else is dropped.
    mask = take | (src_mask.bool() & allowed)
    out[~mask] = 0.0
    return out, mask, take
