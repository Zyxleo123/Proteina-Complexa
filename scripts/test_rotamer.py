"""CPU unit tests for the side-chain builder. No GPU, no checkpoint, seconds to run.

Run:  .venv/bin/python scripts/test_rotamer.py

The thing worth testing here is not that the code runs -- it is that the side chain lands
where a real side chain lands. An all-atom builder assembled from openfold's own constants
can be self-consistently wrong: the first version of `rotamer.py` omitted the diag(-1, 1, -1)
fixup openfold applies to rigid group 0, and produced side chains with *ideal bond lengths
and ideal bond angles* attached to the backbone in the wrong orientation -- CB about 2.4 A
from where it belongs. Nothing checkable against the constants themselves would have caught
it, because the constants were being used consistently.

So the load-bearing test rebuilds REAL crystal side chains from their OWN measured chi angles
and compares atom for atom. That is an external reference: it fails on a wrong frame, a wrong
handedness, a wrong group composition, or a wrong literature position, none of which the
cheaper tests can distinguish.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import torch
from openfold.data import data_transforms
from openfold.np import protein as ofp
from openfold.np import residue_constants as rc

sys.path.insert(0, str(Path(__file__).parent))
from rotamer import (  # noqa: E402
    CA_IDX,
    CB_IDX,
    C_IDX,
    MODAL_CHI_DEG,
    N_IDX,
    O_IDX,
    OXT_IDX,
    build_sidechain,
    modal_chi_rad,
    n_chi,
)

PDB_GLOB = "CPSea_data/lnr_staged/pdbs/*.pdb"
GRAFTABLE = sorted(MODAL_CHI_DEG)
AO = rc.atom_order


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"  ok  {name}")


def measured_chi(aatype, coords, mask):
    """chi1..chi4 in radians, straight from coordinates -- an independent code path from the
    frame composition the builder uses, which is the point of comparing against it."""
    p = data_transforms.atom37_to_torsion_angles(prefix="")(
        {"aatype": aatype, "all_atom_positions": coords.double(),
         "all_atom_mask": mask.double()})
    sc = p["torsion_angles_sin_cos"][..., 3:, :]
    return torch.atan2(sc[..., 0], sc[..., 1]).float()


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    return torch.rad2deg(torch.atan2((torch.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1)))


def real_residues(limit_files=20):
    """Complete graftable residues from the staged LNR inputs, as (aatype, coords, mask)."""
    files = sorted(glob.glob(PDB_GLOB))[:limit_files]
    if not files:
        return None
    out = []
    for f in files:
        prot = ofp.from_pdb_string(Path(f).read_text())
        aat = torch.tensor(prot.aatype).long()
        xyz = torch.tensor(prot.atom_positions).float()
        msk = torch.tensor(prot.atom_mask).bool()
        full = (msk == torch.tensor(rc.restype_atom37_mask)[aat].bool()).all(-1)
        sel = [i for i in range(len(aat))
               if full[i] and rc.restype_1to3[rc.restypes[aat[i]]] in MODAL_CHI_DEG]
        if sel:
            s = torch.tensor(sel)
            out.append((aat[s], xyz[s], msk[s]))
    if not out:
        return None
    return (torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]),
            torch.cat([o[2] for o in out]))


def test_rebuilds_real_crystal_sidechains():
    """THE test. Rebuild real side chains from their own chi and compare atom for atom.

    Tolerance is set by ideal-vs-real internal geometry, not by the builder: a residue's real
    bond lengths and angles differ from the library's by a few hundredths of an Angstrom, and
    that error compounds along the chain, so LYS (four bonds past CB) drifts further than CYS
    (one). Measured over ~1900 residues the worst single atom is ~2.5 A on a LYS NZ; a wrong
    frame put the FIRST atom, CB, 2.4 A out on every residue, so a median gate separates the
    two cases cleanly.
    """
    data = real_residues()
    if data is None:
        print(f"  SKIP no PDBs at {PDB_GLOB} -- this is the load-bearing check, "
              f"do not read a pass without it")
        return
    aat, xyz, msk = data
    chi = measured_chi(aat[None], xyz[None], msk[None])[0]
    built, _, placed = build_sidechain(xyz, msk, aat, chi)
    dev = (built - xyz).norm(dim=-1)

    cb = dev[:, CB_IDX]
    check(f"CB lands on the real CB (n={len(aat)})", float(cb.median()) < 0.10,
          f"median {float(cb.median()):.3f} A, max {float(cb.max()):.3f} A")
    for name in GRAFTABLE:
        idx = rc.restype_order[rc.restype_3to1[name]]
        sel = aat == idx
        if not bool(sel.any()):
            continue
        d = dev[sel][placed[sel]]
        med = float(d.median())
        check(f"{name} side chain reproduced (n={int(sel.sum())})", med < 0.25, f"median {med:.3f} A")
    allplaced = dev[placed]
    check("every built atom is within 3 A of the real one",
          float(allplaced.max()) < 3.0, f"max {float(allplaced.max()):.3f} A")


def test_cb_chirality_matches_real_l_amino_acids():
    """A mirrored group-0 frame builds a D-amino acid: ideal bonds, ideal angles, wrong hand.

    The improper N-C-CA-CB is the direct statement of that handedness. The reference is not a
    remembered constant -- it is measured from the same crystal structures in the line below,
    so this test cannot pass by agreeing with a typo.
    """
    data = real_residues()
    if data is None:
        print(f"  SKIP no PDBs at {PDB_GLOB}")
        return
    aat, xyz, msk = data
    built, _, _ = build_sidechain(xyz, msk, aat)

    def improper(t):
        return dihedral(t[:, N_IDX], t[:, C_IDX], t[:, CA_IDX], t[:, CB_IDX])

    real, made = improper(xyz), improper(built)
    check("real crystal residues are L (improper is negative)", float(real.mean()) < 0,
          f"{float(real.mean()):+.2f} deg")
    check("built residues have the same hand", float(made.mean()) < 0, f"{float(made.mean()):+.2f} deg")
    check("and the same value", abs(float(made.mean() - real.mean())) < 3.0,
          f"built {float(made.mean()):+.2f} vs real {float(real.mean()):+.2f} deg")


def test_requested_chi_is_the_chi_you_get():
    """Round trip through an independent dihedral calculation: ask for a chi, read it back.

    chi2 and beyond come back exact, because every atom defining them is one this function
    placed. chi1 does not, and should not: it is measured as N-CA-CB-SG, and N belongs to the
    real backbone. The group-0 frame is fixed by C, CA and N together, so when a residue's
    real N-CA-C angle differs from the ideal 111 degrees, N sits slightly off its ideal
    position within that frame and chi1 picks up the difference. Measured over 468 real
    residues the correlation between (N-CA-C - 111 deg) and the chi1 offset is **0.998**, with
    the backbone angle spanning 98-120 degrees; the offset is that deviation showing through,
    not a placement error. Attaching an ideal side chain to a real backbone cannot do better
    without also moving the backbone.
    """
    data = real_residues(limit_files=4)
    if data is None:
        print(f"  SKIP no PDBs at {PDB_GLOB}")
        return
    aat, xyz, msk = data
    want = modal_chi_rad(aat)
    built, newmask, _ = build_sidechain(xyz, msk, aat, want)
    got = measured_chi(aat[None], built[None], newmask[None].float())[0]
    for name in GRAFTABLE:
        idx = rc.restype_order[rc.restype_3to1[name]]
        sel = aat == idx
        if not bool(sel.any()):
            continue
        k = n_chi(idx)
        # ASP chi2 and GLU chi3 are pi-periodic (the carboxylate oxygens are equivalent), so
        # the recovered angle may legitimately differ by 180 deg. Compare mod pi there.
        per = rc.chi_pi_periodic[idx][:k]
        d = torch.remainder(got[sel][:, :k] - want[sel][:, :k] + torch.pi, 2 * torch.pi) - torch.pi
        d = d.abs()
        for c in range(k):
            if per[c]:
                d[:, c] = torch.minimum(d[:, c], (torch.pi - d[:, c]).abs())
        deg1 = float(torch.rad2deg(d[:, 0].max()))
        check(f"{name} chi1 within the backbone's own tau deviation (n={int(sel.sum())})",
              deg1 < 15.0, f"max {deg1:.3f} deg")
        if k > 1:
            deg2 = float(torch.rad2deg(d[:, 1:].max()))
            check(f"{name} chi2+ round-trips exactly", deg2 < 0.05, f"max {deg2:.4f} deg")


def test_chi1_offset_is_the_backbone_not_the_builder():
    """Pins the explanation above, so a future frame bug cannot hide inside chi1's tolerance.

    A wrong frame would put chi1 off by an amount unrelated to the residue's own N-CA-C
    angle. The near-perfect correlation is what distinguishes "ideal side chain on a real
    backbone" from "side chain in the wrong place".
    """
    data = real_residues(limit_files=8)
    if data is None:
        print(f"  SKIP no PDBs at {PDB_GLOB}")
        return
    aat, xyz, msk = data
    want = modal_chi_rad(aat)
    built, newmask, _ = build_sidechain(xyz, msk, aat, want)
    got = measured_chi(aat[None], built[None], newmask[None].float())[0]
    err = torch.rad2deg(
        torch.remainder(got[:, 0] - want[:, 0] + torch.pi, 2 * torch.pi) - torch.pi)
    v1, v2 = xyz[:, N_IDX] - xyz[:, CA_IDX], xyz[:, C_IDX] - xyz[:, CA_IDX]
    tau = torch.rad2deg(torch.acos(
        (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1)))) - 111.0
    e, t = err - err.mean(), tau - tau.mean()
    r = float((e * t).sum() / (e.norm() * t.norm()))
    check(f"chi1 error tracks the real N-CA-C angle (n={len(aat)})", r > 0.95, f"r = {r:.3f}")
    check("and is centred on zero", abs(float(err.mean())) < 1.0, f"{float(err.mean()):+.2f} deg")


def test_backbone_is_not_rebuilt():
    """N, CA, C define the frame and O is in the psi group. Replacing any of them would swap
    the measured backbone for an idealised one, which is exactly what the graft preserves."""
    data = real_residues(limit_files=4)
    if data is None:
        print(f"  SKIP no PDBs at {PDB_GLOB}")
        return
    aat, xyz, msk = data
    built, _, placed = build_sidechain(xyz, msk, aat)
    for name, i in (("N", N_IDX), ("CA", CA_IDX), ("C", C_IDX), ("O", O_IDX)):
        check(f"{name} is never marked as placed", not bool(placed[:, i].any()))
        check(f"{name} coordinates are bit-identical", torch.equal(built[:, i], xyz[:, i]))


def test_mask_and_coords_agree():
    data = real_residues(limit_files=4)
    if data is None:
        print(f"  SKIP no PDBs at {PDB_GLOB}")
        return
    aat, xyz, msk = data
    tgt = torch.full_like(aat, rc.restype_order["C"])      # graft everything to CYS
    built, newmask, _ = build_sidechain(xyz, msk, tgt, modal_chi_rad(tgt))
    allowed = torch.tensor(rc.restype_atom37_mask)[tgt].bool()
    check("mask never exceeds the target residue's atom set", not bool((newmask & ~allowed).any()))
    check("cleared slots are exactly zero", float(built[~newmask].abs().max()) == 0.0)
    check("kept slots are not all zero", float(built[newmask].abs().max()) > 0.0)
    check("SG present on every grafted residue", bool(newmask[:, AO["SG"]].all()))


def test_terminal_oxt_survives_the_graft():
    """openfold's restype_atom37_mask gives OXT to NO residue, so a naive intersection strips
    the terminal carboxylate off every grafted C-terminal residue -- which is every `j`
    endpoint this graft touches. OXT depends on being terminal, not on residue identity."""
    xyz = torch.zeros(1, 37, 3)
    xyz[0, N_IDX] = torch.tensor([1.458, 0.0, 0.0])
    xyz[0, CA_IDX] = torch.tensor([0.0, 0.0, 0.0])
    xyz[0, C_IDX] = torch.tensor([-0.546, 1.424, 0.0])
    xyz[0, O_IDX] = torch.tensor([-1.0, 2.5, 0.0])
    xyz[0, OXT_IDX] = torch.tensor([-1.2, 1.1, 0.9])
    msk = torch.zeros(1, 37, dtype=torch.bool)
    msk[0, [N_IDX, CA_IDX, C_IDX, O_IDX, OXT_IDX]] = True
    aat = torch.tensor([rc.restype_order["C"]])
    built, mask, placed = build_sidechain(xyz, msk, aat)
    check("OXT is kept", bool(mask[0, OXT_IDX]))
    check("OXT is copied, not placed", not bool(placed[0, OXT_IDX]))
    check("OXT coordinate is bit-identical", torch.equal(built[0, OXT_IDX], xyz[0, OXT_IDX]))
    msk[0, OXT_IDX] = False
    _, mask2, _ = build_sidechain(xyz, msk, aat)
    check("and not invented when the source lacked it", not bool(mask2[0, OXT_IDX]))


def test_missing_backbone_atom_is_fatal():
    """A frame built from a missing atom is silently garbage -- it would place the side chain
    somewhere plausible-looking and wrong rather than failing."""
    xyz = torch.zeros(1, 37, 3)
    xyz[0, N_IDX] = torch.tensor([1.458, 0.0, 0.0])
    xyz[0, CA_IDX] = torch.tensor([0.0, 0.0, 0.0])
    xyz[0, C_IDX] = torch.tensor([-0.546, 1.424, 0.0])
    msk = torch.zeros(1, 37, dtype=torch.bool)
    msk[0, [N_IDX, CA_IDX, C_IDX]] = True
    aat = torch.tensor([rc.restype_order["C"]])
    built, mask, _ = build_sidechain(xyz, msk, aat)
    check("a bare N/CA/C is enough to build from", bool(mask[0, AO["SG"]]))
    msk[0, C_IDX] = False
    try:
        build_sidechain(xyz, msk, aat)
    except SystemExit as e:
        check("missing C is fatal, not silent", "backbone frame" in str(e), str(e))
    else:
        raise AssertionError("FAIL missing C was accepted")


def test_unknown_residue_is_refused():
    """chi=0 is an eclipsed conformation, not a rotamer. Defaulting to it would look fine."""
    try:
        modal_chi_rad(torch.tensor([rc.restype_order["W"]]))
    except SystemExit as e:
        check("no silent chi=0 fallback", "MODAL_CHI_DEG" in str(e), str(e))
    else:
        raise AssertionError("FAIL TRP was accepted without a recorded rotamer")


def test_modal_table_matches_openfold_chi_counts():
    for name, chi in MODAL_CHI_DEG.items():
        idx = rc.restype_order[rc.restype_3to1[name]]
        check(f"{name} has {n_chi(idx)} chi", len(chi) == n_chi(idx), f"table has {len(chi)}")


def main():
    for fn in (test_rebuilds_real_crystal_sidechains,
               test_cb_chirality_matches_real_l_amino_acids,
               test_requested_chi_is_the_chi_you_get,
               test_chi1_offset_is_the_backbone_not_the_builder,
               test_backbone_is_not_rebuilt,
               test_mask_and_coords_agree,
               test_terminal_oxt_survives_the_graft,
               test_missing_backbone_atom_is_fatal,
               test_unknown_residue_is_refused,
               test_modal_table_matches_openfold_chi_counts):
        print(f"{fn.__name__}:")
        fn()
    print("\nALL ROTAMER TESTS PASSED")


if __name__ == "__main__":
    main()
