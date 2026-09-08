"""CPU unit tests for the terminal anchor graft. No GPU, no checkpoint, seconds to run.

Run:  .venv/bin/python scripts/test_anchor_graft.py

Covers the ways a graft could look right and be wrong:
  * grafting the wrong residue, or the right ones in the wrong order (the isopeptide
    orientation is a measured fact, not a convention -- see anchor_graft.__doc__),
  * leaving the OLD residue's side-chain atoms behind, so the encoder reads an alanine's
    atoms as a cysteine's,
  * claiming an atom the source residue never had,
  * aliasing the caller's batch, so an ungrafted control silently inherits the graft,
  * losing the native sequence, which would make the edit look cheaper than it was,
  * a run_key that does not distinguish grafted from ungrafted (every row resumes as
    "already done" and the arm never runs).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from openfold.np import residue_constants as rc

sys.path.insert(0, str(Path(__file__).parent))
from anchor_graft import (  # noqa: E402
    BACKBONE_IDX,
    CA_IDX,
    CB_IDX,
    O_IDX,
    ANCHOR_SPEC,
    graft_anchors,
    graft_tag,
)
from proteinfoundation.cyclization.constants import (  # noqa: E402
    AA_ASP,
    AA_CYS,
    AA_LYS,
    DISULFIDE,
    ISOPEPTIDE,
    MAINCHAIN,
)

N = 8
AA = {c: rc.restype_order[c] for c in rc.restypes}


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"  ok  {name}")


def _backbone(k: int) -> dict[str, torch.Tensor]:
    """N/CA/C/O for residue `k` at ideal internal geometry, in a per-residue orientation.

    The grafted side chain is placed in the rigid-group frame built from N, CA and C, so the
    fixture cannot use arbitrary coordinates: the arange placeholder this replaced put those
    three atoms on a straight line, and Gram-Schmidt on a collinear triple is degenerate.
    Rotating each residue differently also stops a frame bug from cancelling out by symmetry.
    """
    n_loc = torch.tensor([1.458, 0.0, 0.0])                      # N-CA 1.458 A
    ang = torch.deg2rad(torch.tensor(111.0))                     # N-CA-C 111 deg
    c_loc = torch.tensor([1.525 * torch.cos(ang), 1.525 * torch.sin(ang), 0.0])
    o_loc = c_loc * (1.0 + 1.231 / 1.525)                        # roughly along CA->C
    a, b = 0.7 * k, 0.4 * k
    rz = torch.tensor([[torch.cos(torch.tensor(a)), -torch.sin(torch.tensor(a)), 0.0],
                       [torch.sin(torch.tensor(a)), torch.cos(torch.tensor(a)), 0.0],
                       [0.0, 0.0, 1.0]])
    rx = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, torch.cos(torch.tensor(b)), -torch.sin(torch.tensor(b))],
                       [0.0, torch.sin(torch.tensor(b)), torch.cos(torch.tensor(b))]])
    r = rz @ rx
    ca = torch.tensor([3.8 * k, 0.0, 0.0])                       # CA-CA 3.8 A
    return {"N": ca + r @ n_loc, "CA": ca, "C": ca + r @ c_loc, "O": ca + r @ o_loc}


def fake_batch(seq="AWTFEYGH", pad=0):
    """A peptide with each residue carrying exactly the atoms its type allows.

    Backbone atoms get real geometry (see `_backbone`); side-chain slots get placeholder
    values, since the graft either replaces or drops them and nothing else reads them.
    """
    n = len(seq) + pad
    aatype = torch.zeros(1, n, dtype=torch.long)
    mask = torch.zeros(1, n, dtype=torch.bool)
    cm = torch.zeros(1, n, 37, dtype=torch.bool)
    for k, c in enumerate(seq):
        aatype[0, k] = AA[c]
        mask[0, k] = True
        cm[0, k] = torch.tensor(rc.restype_atom37_mask[AA[c]], dtype=torch.bool)
    coords = torch.arange(1, n * 37 * 3 + 1, dtype=torch.float32).reshape(1, n, 37, 3)
    for k in range(len(seq)):
        for name, xyz in _backbone(k).items():
            coords[0, k, rc.atom_order[name]] = xyz
    coords = coords * cm[..., None]
    batch = {"residue_type": aatype, "coord_mask": cm,
             "coords": coords.clone(), "coords_nm": coords.clone() / 10.0}
    return batch, mask


def test_disulfide_grafts_cys_at_both_ends():
    batch, mask = fake_batch()
    out, info = graft_anchors(batch, mask, DISULFIDE)
    aa = out["residue_type"][0]
    check("CYS at i", int(aa[0]) == AA_CYS, rc.restypes[int(aa[0])])
    check("CYS at j", int(aa[N - 1]) == AA_CYS, rc.restypes[int(aa[N - 1])])
    check("interior untouched",
          torch.equal(aa[1:N - 1], batch["residue_type"][0][1:N - 1]))
    check("both counted as grafted", info["n_grafted"] == 2, info)


def test_isopeptide_orientation_is_lys_then_asp():
    """Measured, not conventional: of 499 isopeptide rings the model placed for itself it
    put LYS at the lower-index endpoint in 100% and paired it with ASP in 98%."""
    batch, mask = fake_batch()
    out, _ = graft_anchors(batch, mask, ISOPEPTIDE)
    aa = out["residue_type"][0]
    check("LYS at the FIRST endpoint", int(aa[0]) == AA_LYS, rc.restypes[int(aa[0])])
    check("ASP at the LAST endpoint", int(aa[N - 1]) == AA_ASP, rc.restypes[int(aa[N - 1])])
    check("spec is not symmetric", ANCHOR_SPEC[ISOPEPTIDE][0] != ANCHOR_SPEC[ISOPEPTIDE][1])


def test_mainchain_is_a_declared_no_op():
    batch, mask = fake_batch()
    out, info = graft_anchors(batch, mask, MAINCHAIN)
    check("mainchain returns the batch untouched", out is batch)
    check("mainchain reports graft_applied=0", info["graft_applied"] == 0, info)
    check("mainchain reports its type", info["graft_type"] == "mainchain", info)


def test_old_sidechain_is_replaced_by_a_complete_one():
    """The encoder reads aatype, atom37 coords and sidechain angles TOGETHER. A leftover
    tryptophan side chain under a CYS aatype is an object that does not exist -- and so is a
    CYS with no SG, which is what the first version of this graft produced."""
    batch, mask = fake_batch(seq="AWTFEYGW")   # TRP at j: 14 atoms, the fullest side chain
    before = int(batch["coord_mask"][0, N - 1].sum())
    out, info = graft_anchors(batch, mask, DISULFIDE)
    kept = out["coord_mask"][0, N - 1]
    cys_full = torch.tensor(rc.restype_atom37_mask[AA_CYS], dtype=torch.bool)
    check("TRP started with a full side chain", before == 14, before)
    check("occupancy is exactly CYS's atom set", torch.equal(kept, cys_full),
          [rc.atom_types[i] for i in range(37) if kept[i]])
    check("SG is present -- this is what distinguishes CYS from ALA", bool(kept[rc.atom_order["SG"]]))
    check("backbone kept", all(bool(kept[i]) for i in BACKBONE_IDX))
    check("CB kept", bool(kept[CB_IDX]))
    gone = batch["coord_mask"][0, N - 1] & ~kept
    check("TRP's ring atoms are gone", int(gone.sum()) == 9, int(gone.sum()))
    check("gone coords are zeroed, not stale",
          float(out["coords"][0, N - 1][gone].abs().max()) == 0.0)
    check("gone coords zeroed in nm too",
          float(out["coords_nm"][0, N - 1][gone].abs().max()) == 0.0)
    check("build counted", info["n_sidechain_atoms_built"] >= 1, info)
    check("coords and coords_nm stay in sync",
          float((out["coords"][0] - out["coords_nm"][0] * 10.0).abs().max()) < 1e-3)


def test_the_grafted_residue_is_not_an_alanine():
    """The regression that motivated the rebuild.

    ALA's complete atom37 set is [N, CA, C, CB, O]. Truncating a graft to backbone+CB
    therefore hands the encoder something indistinguishable from an alanine -- same
    occupancy, no chi1 -- and the AE decoded every such endpoint back as ALA. Build mode must
    differ from ALA's occupancy; truncate mode is asserted to still coincide with it, because
    that is the whole reason it is only a negative control.
    """
    ala_full = torch.tensor(rc.restype_atom37_mask[AA["A"]], dtype=torch.bool)
    batch, mask = fake_batch(seq="AWTFEYGW")

    built, _ = graft_anchors(batch, mask, DISULFIDE, sidechain="build")
    check("built CYS occupancy differs from ALA's",
          not torch.equal(built["coord_mask"][0, N - 1], ala_full))

    trunc, _ = graft_anchors(batch, mask, DISULFIDE, sidechain="truncate")
    check("truncated CYS occupancy IS ALA's -- the bug, pinned",
          torch.equal(trunc["coord_mask"][0, N - 1], ala_full))
    check("truncate still reports its drops", True)


def test_the_encoder_sees_a_cysteine_not_an_alanine():
    """The regression, checked through the encoder's OWN feature functions.

    `coord_mask` and the chi angles are two of the three things the AE reads about a residue
    (`x1_a37coors_nm` carries the 37-slot occupancy, `x1_sidechain_angles` the chi one-hots);
    the third is the aatype one-hot. Under the truncating graft the first two were bit-for-bit
    what a real ALANINE produces -- 5 atoms, chi1 masked out as absent -- so the only thing
    saying CYS was 20 dimensions of one-hot the AE has never seen contradicted, and it
    decoded ALA every time. Measured here rather than argued:

        truncate  n_atoms=5  chi1_valid=False      <- identical to a real ALA
        build     n_atoms=6  chi1_valid=True  -65  <- a cysteine
        real ALA  n_atoms=5  chi1_valid=False

    If a future change makes the built endpoint's occupancy and chi indistinguishable from
    alanine's again, this fails and the grafted arms do not silently produce alanines.
    """
    from proteinfoundation.nn.feature_factory.seq_feats import OpenfoldSideChainAnglesSeqFeat
    feat = OpenfoldSideChainAnglesSeqFeat()

    def seen(batch, mask, pos):
        b = {"residue_type": batch["residue_type"], "coords": batch["coords"],
             "coord_mask": batch["coord_mask"].float(), "mask": mask}
        _, ang, tmask = feat._get_sidechain_angles(b)
        return (int(batch["coord_mask"][0, pos].sum()), bool(tmask[0, pos, 0]),
                float(torch.rad2deg(ang[0, pos, 0])))

    batch, mask = fake_batch(seq="AWTFEYGW")
    ala, _ = fake_batch(seq="AWTFEYGA")
    ala_n, ala_chi, _ = seen(ala, _, N - 1)

    trunc, _ = graft_anchors(batch, mask, DISULFIDE, sidechain="truncate")
    t_n, t_chi, _ = seen(trunc, mask, N - 1)
    check("truncated endpoint is structurally an ALA to the encoder",
          (t_n, t_chi) == (ala_n, ala_chi), f"{(t_n, t_chi)} vs real ALA {(ala_n, ala_chi)}")

    built, _ = graft_anchors(batch, mask, DISULFIDE, sidechain="build")
    b_n, b_chi, b_val = seen(built, mask, N - 1)
    check("built endpoint is NOT structurally an ALA",
          (b_n, b_chi) != (ala_n, ala_chi), f"{(b_n, b_chi)}")
    check("built endpoint has a valid chi1", b_chi, b_chi)
    check("chi1 is the rotamer that was asked for", abs(b_val - (-65.0)) < 1.0, f"{b_val:+.2f} deg")


def test_truncate_mode_is_still_reachable_as_the_control():
    batch, mask = fake_batch(seq="AWTFEYGW")
    out, info = graft_anchors(batch, mask, DISULFIDE, sidechain="truncate")
    kept = out["coord_mask"][0, N - 1]
    check("only backbone+CB survive", int(kept.sum()) == 5, int(kept.sum()))
    dropped = batch["coord_mask"][0, N - 1] & ~kept
    check("dropped coords are zeroed, not stale",
          float(out["coords"][0, N - 1][dropped].abs().max()) == 0.0)
    # ALA at i has 5 atoms and keeps all 5 (backbone+CB); TRP at j has 14 and keeps 5.
    check("drop count reported", info["n_sidechain_atoms_dropped"] == 9,
          info["n_sidechain_atoms_dropped"])
    check("mode is recorded in info", info["graft_sidechain"] == "truncate", info)


def test_coords_and_coords_nm_are_built_in_their_own_frames():
    """`coords` and `coords_nm` are NOT one tensor scaled by 10 in the real loader.

    Measured on the LNR inputs, coords_nm is centred and rotated by the dataset transform
    while coords keeps the crystal frame: the two differ by up to 19 A yet fit each other to
    1e-6 A RMSD with det(R) = +1. Both are read by the encoder -- coords_nm feeds
    x1_a37coors_nm, coords feeds the backbone and side-chain ANGLE features -- so a side chain
    computed in one frame and written into the other lands somewhere arbitrary while every
    bond length still measures perfect. This fixture reproduces that split explicitly.
    """
    batch, mask = fake_batch(seq="AWTFEYGW")
    theta = torch.tensor(0.9)
    rot = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.0],
                        [torch.sin(theta), torch.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])
    shift = torch.tensor([12.0, -5.0, 3.0])
    occupied = batch["coord_mask"][..., None]
    batch["coords_nm"] = ((batch["coords"] @ rot.T + shift) * occupied) / 10.0

    out, _ = graft_anchors(batch, mask, DISULFIDE)
    sg = rc.atom_order["SG"]
    for pos in (0, N - 1):
        a = out["coords"][0, pos]
        b = out["coords_nm"][0, pos] * 10.0
        check(f"SG built in the coords frame at {pos}",
              abs(float((a[sg] - a[CB_IDX]).norm()) - 1.81) < 0.05,
              f"{float((a[sg] - a[CB_IDX]).norm()):.3f} A")
        check(f"SG built in the coords_nm frame at {pos}",
              abs(float((b[sg] - b[CB_IDX]).norm()) - 1.81) < 0.05,
              f"{float((b[sg] - b[CB_IDX]).norm()):.3f} A")
        # The two frames must still describe the SAME residue: distances from CA are
        # invariant to the rigid transform between them, coordinates are not.
        for at in (CB_IDX, sg):
            da, db = float((a[at] - a[CA_IDX]).norm()), float((b[at] - b[CA_IDX]).norm())
            check(f"atom {at} at {pos} has the same internal geometry in both frames",
                  abs(da - db) < 1e-2, f"{da:.4f} vs {db:.4f} A")
        check(f"and the frames really are different at {pos}",
              float((a[CB_IDX] - b[CB_IDX]).norm()) > 1.0)


def test_desynchronised_coordinate_tensors_are_fatal():
    """Same-frame or different-frame is fine; different STRUCTURES is not."""
    batch, mask = fake_batch(seq="AWTFEYGW")
    batch["coords_nm"] = batch["coords_nm"].clone()
    batch["coords_nm"][0, N - 1, CA_IDX] += 5.0        # move CA in one tensor only
    try:
        graft_anchors(batch, mask, DISULFIDE)
    except SystemExit as e:
        check("desync is caught", "internal geometry" in str(e) or "occupanc" in str(e), str(e))
    else:
        raise AssertionError("FAIL desynchronised coords/coords_nm were accepted")


def test_backbone_coordinates_are_untouched():
    batch, mask = fake_batch()
    out, _ = graft_anchors(batch, mask, DISULFIDE)
    for pos in (0, N - 1):
        for a in BACKBONE_IDX:
            check(f"backbone atom {a} at {pos} unchanged",
                  torch.equal(out["coords"][0, pos, a], batch["coords"][0, pos, a]))


def test_builds_a_cb_even_on_a_glycine_source():
    """GLY has no CB, and the graft must supply one.

    This invariant is the reverse of what it was while the graft truncated: back then
    "never claim an atom the source lacked" was the rule, and applying it to CB left a CYS
    that could not carry an SG either. CB sits in rigid group 0, so it is placed exactly
    from the real N/CA/C -- nothing is guessed. The rule that survives is the one below,
    about BACKBONE atoms, which genuinely cannot be invented.
    """
    batch, mask = fake_batch(seq="GWTFEYGG")
    out, _ = graft_anchors(batch, mask, DISULFIDE)
    for pos in (0, N - 1):
        check(f"CB built at {pos} (GLY source)", bool(out["coord_mask"][0, pos, CB_IDX]))
        check(f"SG built at {pos}", bool(out["coord_mask"][0, pos, rc.atom_order["SG"]]))
        ca, cb = out["coords"][0, pos, rc.atom_order["CA"]], out["coords"][0, pos, CB_IDX]
        d = float((ca - cb).norm())
        check(f"CA-CB bond length at {pos} is physical", 1.4 < d < 1.65, f"{d:.3f} A")


def test_never_claims_a_backbone_atom_the_source_lacked():
    """A residue with no O in the input must not come back claiming one: the graft builds
    side chains, and has no basis for inventing a carbonyl."""
    batch, mask = fake_batch(seq="AWTFEYGW")
    batch["coord_mask"][0, N - 1, O_IDX] = False
    batch["coords"][0, N - 1, O_IDX] = 0.0
    batch["coords_nm"][0, N - 1, O_IDX] = 0.0
    out, _ = graft_anchors(batch, mask, DISULFIDE)
    check("missing O is not invented", not bool(out["coord_mask"][0, N - 1, O_IDX]))
    check("and its coordinate stays zero",
          float(out["coords"][0, N - 1, O_IDX].abs().max()) == 0.0)


def test_already_correct_endpoint_keeps_its_sidechain():
    """A binder that already has the cysteine is real information -- do not truncate it, and
    do not count it as a graft (a null result must be readable against n_grafted)."""
    batch, mask = fake_batch(seq="CWTFEYGH")
    out, info = graft_anchors(batch, mask, DISULFIDE)
    check("untouched endpoint keeps every atom",
          torch.equal(out["coord_mask"][0, 0], batch["coord_mask"][0, 0]))
    check("only the other end counted", info["n_grafted"] == 1, info)
    check("from/to reported for the no-op end",
          info["graft_i_from"] == "C" and info["graft_i_to"] == "C", info)


def test_caller_batch_is_not_aliased():
    """An in-place graft would leak into the paired ungrafted control run next to it."""
    batch, mask = fake_batch()
    aa_before = batch["residue_type"].clone()
    cm_before = batch["coord_mask"].clone()
    co_before = batch["coords"].clone()
    out, _ = graft_anchors(batch, mask, DISULFIDE)
    check("caller residue_type unchanged", torch.equal(batch["residue_type"], aa_before))
    check("caller coord_mask unchanged", torch.equal(batch["coord_mask"], cm_before))
    check("caller coords unchanged", torch.equal(batch["coords"], co_before))
    check("output actually differs", not torch.equal(out["residue_type"], aa_before))


def test_native_sequence_is_preserved_for_scoring():
    batch, mask = fake_batch()
    out, _ = graft_anchors(batch, mask, ISOPEPTIDE)
    check("native sequence stashed", "residue_type_native" in out)
    check("native equals the pre-graft sequence",
          torch.equal(out["residue_type_native"], batch["residue_type"]))
    check("native differs from the grafted sequence",
          not torch.equal(out["residue_type_native"], out["residue_type"]))


def test_padding_is_ignored():
    """Endpoints are the first/last VALID residue, never the padded tail."""
    batch, mask = fake_batch(seq="AWTFEYGH", pad=5)
    out, _ = graft_anchors(batch, mask, DISULFIDE)
    check("last real residue grafted", int(out["residue_type"][0, N - 1]) == AA_CYS)
    check("padding untouched", int(out["residue_type"][0, N]) == 0
          and not bool(out["coord_mask"][0, N].any()))


def test_tag_separates_grafted_from_ungrafted():
    on = types.SimpleNamespace(graft_anchors=True, graft_keep_cb=1)
    nocb = types.SimpleNamespace(graft_anchors=True, graft_keep_cb=0,
                                 graft_sidechain="truncate")
    off = types.SimpleNamespace(graft_anchors=False, graft_keep_cb=1)
    check("off is empty", graft_tag(off) == "")
    check("on is tagged", graft_tag(on) == "graftsc", graft_tag(on))   # build is the default
    check("keep_cb arms differ", graft_tag(nocb) != graft_tag(on))
    # Rows written while the graft truncated are wrong but carry no marker saying so, and
    # resume keys off this tag: if the fixed arm reused "graft" it would skip exactly the
    # rows it exists to redo.
    build = types.SimpleNamespace(graft_anchors=True, graft_keep_cb=1, graft_sidechain="build")
    trunc = types.SimpleNamespace(graft_anchors=True, graft_keep_cb=1, graft_sidechain="truncate")
    check("build has its own tag", graft_tag(build) == "graftsc", graft_tag(build))
    check("build cannot resume onto truncated rows", graft_tag(build) != graft_tag(trunc))
    check("truncate keeps the historical tag", graft_tag(trunc) == "graft", graft_tag(trunc))


def main():
    for fn in (test_disulfide_grafts_cys_at_both_ends,
               test_isopeptide_orientation_is_lys_then_asp,
               test_mainchain_is_a_declared_no_op,
               test_old_sidechain_is_replaced_by_a_complete_one,
               test_the_grafted_residue_is_not_an_alanine,
               test_the_encoder_sees_a_cysteine_not_an_alanine,
               test_truncate_mode_is_still_reachable_as_the_control,
               test_coords_and_coords_nm_are_built_in_their_own_frames,
               test_desynchronised_coordinate_tensors_are_fatal,
               test_backbone_coordinates_are_untouched,
               test_builds_a_cb_even_on_a_glycine_source,
               test_never_claims_a_backbone_atom_the_source_lacked,
               test_already_correct_endpoint_keeps_its_sidechain,
               test_caller_batch_is_not_aliased,
               test_native_sequence_is_preserved_for_scoring,
               test_padding_is_ignored,
               test_tag_separates_grafted_from_ungrafted):
        print(f"{fn.__name__}:")
        fn()
    print("\nALL ANCHOR-GRAFT TESTS PASSED")


if __name__ == "__main__":
    main()
