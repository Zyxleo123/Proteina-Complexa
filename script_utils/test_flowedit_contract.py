"""Unit tests for the LP->CP acceptance contract. No GPU, no checkpoint.

The tests that matter are the ones that try to sneak a bad candidate past the gates, because
that is the failure mode with consequences: a contract that accepts too much turns every
downstream success rate into an overstatement. So there are explicit cases for

  * a ring whose bond distance is perfect but whose ANCHORS cannot host the chemistry
    (the PRO/PRO isopeptide -- the measured dominant isopeptide failure mode), and
  * an input with NO contacts, whose retention is NaN and must NOT pass.

Run:  .venv/bin/python script_utils/test_flowedit_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from openfold.np import residue_constants as rc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.editing.contract import (  # noqa: E402
    CONTRACT,
    ca_rmsd_no_align,
    contact_retention,
    evaluate,
    heavy_atom_contacts,
    ring_geometry,
    sequence_budget,
    severe_clashes,
)

FAILURES: list[str] = []
AA = {rc.restype_1to3[c]: i for i, c in enumerate(rc.restypes)}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(name)


def blank(aatype: list[int], spacing: float = 12.0) -> np.ndarray:
    """atom37 coords for a chain whose residues are far apart, so nothing touches by accident.

    Only slots that really exist for each residue type are filled; the rest stay at the
    origin and are ignored by the atom masks.
    """
    L = len(aatype)
    xyz = np.zeros((L, 37, 3), dtype=float)
    m = rc.restype_atom37_mask[np.clip(aatype, 0, 20)].astype(bool)
    for i in range(L):
        for s in np.nonzero(m[i])[0]:
            # spread atoms slightly within a residue so intra-residue distances are sane
            xyz[i, s] = [i * spacing, 0.4 * (s % 5), 0.4 * (s // 5)]
    return xyz


def put(xyz: np.ndarray, res: int, atom: str, pos) -> None:
    xyz[res, rc.atom_order[atom]] = pos


# --------------------------------------------------------------------------- tests


def test_ring_requires_hostable_anchors():
    """Perfect bond distance + wrong anchor residues == NOT a closed ring."""
    # PRO/PRO at 1.33 A: the geometry is ideal, the chemistry is impossible.
    aatype = [AA["PRO"]] * 5
    xyz = blank(aatype)
    # NZ does not exist on PRO, so the isopeptide measurement must refuse outright.
    out = ring_geometry(xyz, aatype, 0, 4, "isopeptide")
    check("PRO/PRO isopeptide is not closed", out["ring_closed"] is False,
          f"can_host={out['anchors_can_host']}, d={out['bond_dist_A']}")

    # LYS/ASP with the bond atoms placed at 1.33 A: this one must pass.
    aatype2 = [AA["LYS"]] + [AA["GLY"]] * 3 + [AA["ASP"]]
    xyz2 = blank(aatype2)
    put(xyz2, 0, "NZ", [0.0, 0.0, 0.0])
    put(xyz2, 4, "CG", [1.33, 0.0, 0.0])
    out2 = ring_geometry(xyz2, aatype2, 0, 4, "isopeptide")
    check("LYS/ASP isopeptide at 1.33 A is closed", out2["ring_closed"] is True,
          f"d={out2['bond_dist_A']:.3f}, atoms={out2['bond_atoms']}")

    # Same residues, bond far too long.
    put(xyz2, 4, "CG", [6.0, 0.0, 0.0])
    out3 = ring_geometry(xyz2, aatype2, 0, 4, "isopeptide")
    check("LYS/ASP at 6 A is not closed", out3["ring_closed"] is False,
          f"d={out3['bond_dist_A']:.3f}")


def test_disulfide_needs_two_cys():
    aatype = [AA["CYS"]] + [AA["GLY"]] * 3 + [AA["CYS"]]
    xyz = blank(aatype)
    put(xyz, 0, "SG", [0.0, 0.0, 0.0])
    put(xyz, 4, "SG", [2.05, 0.0, 0.0])
    ok = ring_geometry(xyz, aatype, 0, 4, "disulfide")
    check("CYS/CYS at 2.05 A is closed", ok["ring_closed"] is True, f"d={ok['bond_dist_A']:.3f}")

    aatype_bad = [AA["SER"]] + [AA["GLY"]] * 3 + [AA["CYS"]]
    xyz_bad = blank(aatype_bad)
    put(xyz_bad, 4, "SG", [2.05, 0.0, 0.0])
    bad = ring_geometry(xyz_bad, aatype_bad, 0, 4, "disulfide")
    check("SER/CYS disulfide is not closed", bad["ring_closed"] is False,
          f"can_host={bad['anchors_can_host']}")


def test_mainchain_uses_n_to_c():
    aatype = [AA["ALA"]] * 6
    xyz = blank(aatype)
    put(xyz, 0, "N", [0.0, 0.0, 0.0])
    put(xyz, 5, "C", [1.33, 0.0, 0.0])
    out = ring_geometry(xyz, aatype, 0, 5, "mainchain")
    check("mainchain N-C at 1.33 A is closed", out["ring_closed"] is True,
          f"atoms={out['bond_atoms']}, d={out['bond_dist_A']:.3f}")


def test_heavy_atom_contacts_beat_the_ca_proxy():
    """A side-chain-mediated contact must register even though the CAs are far apart."""
    pep_aa = [AA["LYS"], AA["GLY"]]
    tgt_aa = [AA["ASP"]]
    pep = blank(pep_aa, spacing=30.0)
    tgt = blank(tgt_aa, spacing=30.0)
    # CAs 14 A apart (beyond any reasonable CA contact), side-chain tips 3.5 A apart.
    put(pep, 0, "CA", [0.0, 0.0, 0.0])
    put(tgt, 0, "CA", [14.0, 0.0, 0.0])
    put(pep, 0, "NZ", [6.0, 0.0, 0.0])
    put(tgt, 0, "CG", [9.5, 0.0, 0.0])
    got = heavy_atom_contacts(pep, pep_aa, tgt, tgt_aa, cutoff_A=4.5)
    ca_dist = float(np.linalg.norm(pep[0, rc.atom_order["CA"]] - tgt[0, rc.atom_order["CA"]]))
    check("side-chain contact detected at 14 A CA separation", (0, 0) in got,
          f"CA-CA = {ca_dist:.1f} A, contacts = {sorted(got)}")


def test_retention_and_nan_behaviour():
    check("retention of a subset", contact_retention({(0, 0), (1, 1)}, {(0, 0)}) == 0.5)
    nan = contact_retention(set(), {(0, 0)})
    check("empty 'before' gives NaN", nan != nan, str(nan))
    # And the gate must reject it rather than let NaN through.
    pep_aa = [AA["ALA"]] * 4
    pep = blank(pep_aa, spacing=50.0)
    tgt_aa = [AA["ALA"]]
    tgt = blank(tgt_aa)
    tgt[0] += 500.0  # receptor nowhere near the peptide -> no contacts at all
    put(pep, 0, "N", [0.0, 0.0, 0.0])
    put(pep, 3, "C", [1.33, 0.0, 0.0])
    row = evaluate(
        before_xyz_A=pep, before_aatype=pep_aa,
        after_xyz_A=pep, after_aatype=pep_aa,
        tgt_xyz_A=tgt, tgt_aatype=tgt_aa,
        cyc_type="mainchain", anchor_i=0, anchor_j=3, ddg_REU=-1.0,
    )
    check("NaN retention does not pass the contact gate", row["pass_contacts"] is False,
          f"retention={row['contact_retention']}")
    check("and the candidate is not accepted", row["accepted"] is False)


def test_sequence_budget():
    a = [AA["ALA"], AA["GLY"], AA["LYS"], AA["SER"], AA["ASP"]]
    b = [AA["ALA"], AA["GLY"], AA["LYS"], AA["SER"], AA["ASP"]]
    same = sequence_budget(a, b)
    check("identical sequence: 0 subs, identity 1.0",
          same["n_substitutions"] == 0 and same["seq_identity"] == 1.0)

    c = list(a)
    c[0], c[4] = AA["CYS"], AA["CYS"]  # two anchor mutations
    two = sequence_budget(a, c)
    check("two anchor mutations counted", two["n_substitutions"] == 2, str(two["substitutions"]))
    check("identity is 3/5", abs(two["seq_identity"] - 0.6) < 1e-9, f"{two['seq_identity']}")

    prot = sequence_budget(a, c, protected=[0])
    check("protected position violation caught", prot["protected_ok"] is False,
          str(prot["protected_violations"]))
    prot_ok = sequence_budget(a, c, protected=[1, 2, 3])
    check("untouched protected positions pass", prot_ok["protected_ok"] is True)

    try:
        sequence_budget(a, a[:-1])
        check("length change raises", False, "no exception")
    except ValueError:
        check("length change raises", True)


def test_rmsd_is_not_superposed():
    """A rigid translation must show up as RMSD, not be fitted away."""
    aa = [AA["ALA"]] * 5
    x = blank(aa)
    y = x.copy()
    y[:, :, 0] += 3.0  # shift the whole peptide 3 A along x
    r = ca_rmsd_no_align(x, y)
    check("rigid 3 A shift gives 3 A RMSD", abs(r - 3.0) < 1e-6, f"{r:.4f}")
    sub = ca_rmsd_no_align(x, y, subset=[1, 2])
    check("subset RMSD also 3 A", abs(sub - 3.0) < 1e-6, f"{sub:.4f}")
    check("empty subset is NaN", ca_rmsd_no_align(x, y, subset=[]) != ca_rmsd_no_align(x, y, subset=[]))


def test_clashes_exclude_the_ring_bond():
    aa = [AA["ALA"]] * 5
    xyz = blank(aa, spacing=12.0)
    # Put residue 0 and 4 bond atoms at 1.33 A -- the ring bond, which must be exempt.
    put(xyz, 0, "N", [0.0, 0.0, 0.0])
    put(xyz, 4, "C", [1.33, 0.0, 0.0])
    with_ring = severe_clashes(xyz, aa, ring=(0, 4))
    without = severe_clashes(xyz, aa, ring=None)
    check("declared ring bond is not a clash", with_ring["clash_intra_peptide"] == 0,
          f"{with_ring['clash_intra_peptide']}")
    check("the same pair IS a clash when undeclared", without["clash_intra_peptide"] > 0,
          f"{without['clash_intra_peptide']}")


def test_accept_path_end_to_end():
    """A candidate that satisfies everything must be accepted, and lose it one gate at a time."""
    pep_aa = [AA["LYS"], AA["ALA"], AA["ALA"], AA["ALA"], AA["ASP"]]
    tgt_aa = [AA["ALA"]] * 3
    pep = blank(pep_aa, spacing=6.0)
    tgt = blank(tgt_aa, spacing=6.0)
    # Receptor placed to make a few genuine heavy-atom contacts at ~4 A.
    tgt[:, :, 1] += 4.0
    after = pep.copy()
    put(after, 0, "NZ", [0.0, 0.0, 0.0])
    put(after, 4, "CG", [1.33, 0.0, 0.0])

    row = evaluate(
        before_xyz_A=pep, before_aatype=pep_aa,
        after_xyz_A=after, after_aatype=pep_aa,
        tgt_xyz_A=tgt, tgt_aatype=tgt_aa,
        cyc_type="isopeptide", anchor_i=0, anchor_j=4, ddg_REU=-2.0,
    )
    check("ring closed", row["pass_ring"] is True, f"d={row['bond_dist_A']:.2f}")
    check("dG gate passes at -2 REU", row["pass_ddg"] is True)
    check("dG non-worsening flagged", row["ddg_non_worsening"] is True)

    row_hi = evaluate(
        before_xyz_A=pep, before_aatype=pep_aa,
        after_xyz_A=after, after_aatype=pep_aa,
        tgt_xyz_A=tgt, tgt_aatype=tgt_aa,
        cyc_type="isopeptide", anchor_i=0, anchor_j=4, ddg_REU=7.0,
    )
    check("dG gate fails at +7 REU", row_hi["pass_ddg"] is False)

    row_none = evaluate(
        before_xyz_A=pep, before_aatype=pep_aa,
        after_xyz_A=after, after_aatype=pep_aa,
        tgt_xyz_A=tgt, tgt_aatype=tgt_aa,
        cyc_type="isopeptide", anchor_i=0, anchor_j=4, ddg_REU=None,
    )
    check("no Rosetta => pass_ddg is None", row_none["pass_ddg"] is None)
    check("no Rosetta => not accepted", row_none["accepted"] is False)
    check("but geometry/sequence verdict still reported",
          isinstance(row_none["pass_geometry_and_sequence"], bool))


def test_thresholds_are_the_stated_ones():
    check("identity min 0.80", CONTRACT["seq_identity_min"] == 0.80)
    check("max substitutions 2", CONTRACT["max_substitutions"] == 2)
    check("retention min 0.80", CONTRACT["contact_retention_min"] == 0.80)
    check("contact cutoff 4.5 A", CONTRACT["contact_cutoff_A"] == 4.5)
    check("CA RMSD max 2.0 A", CONTRACT["ca_rmsd_max_A"] == 2.0)
    check("interface CA RMSD max 1.0 A", CONTRACT["interface_ca_rmsd_max_A"] == 1.0)
    check("ddG max +5 REU", CONTRACT["ddg_max_REU"] == 5.0)


def main() -> int:
    tests = [
        ("ring requires hostable anchors", test_ring_requires_hostable_anchors),
        ("disulfide needs two CYS", test_disulfide_needs_two_cys),
        ("mainchain uses N-to-C", test_mainchain_uses_n_to_c),
        ("heavy-atom contacts beat the CA proxy", test_heavy_atom_contacts_beat_the_ca_proxy),
        ("retention and NaN behaviour", test_retention_and_nan_behaviour),
        ("sequence budget", test_sequence_budget),
        ("RMSD is not superposed", test_rmsd_is_not_superposed),
        ("clashes exclude the ring bond", test_clashes_exclude_the_ring_bond),
        ("accept path end to end", test_accept_path_end_to_end),
        ("thresholds are the stated ones", test_thresholds_are_the_stated_ones),
    ]
    for name, fn in tests:
        print(f"\n{name}")
        fn()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print(f"all {len(tests)} test groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
