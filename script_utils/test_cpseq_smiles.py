#!/usr/bin/env python3
"""Checks for the cyclic-peptide SMILES builder. No GPU, no network, no model.

What these are for: the SMILES is the only place the closure constraint is expressed in
the AF3 arm, so a SMILES that is subtly the wrong molecule produces a prediction of
something nobody asked for -- and it looks like a result. Each check below corresponds
to a way that happens: a ring that did not close, one that closed through a sidechain, a
fragmented molecule, an unspecified stereocentre, or a D-residue.

The canonical-SMILES regression at the end pins RDKit's output for one sequence. It is
there so an RDKit upgrade that changes atom ordering or output conventions is visible as
a failing test rather than as predictions that quietly stop matching earlier ones.

Run:  python script_utils/test_cpseq_smiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402

from proteinfoundation.cpseq.smiles import (  # noqa: E402
    CANONICAL_AA,
    SmilesBuildError,
    cyclic_peptide,
    sequence_from_records,
)

RDLogger.DisableLog("rdApp.*")

PASS, FAIL = 0, 0

# The four mainchain crystal controls this project's gate is derived from, plus two
# cohort parents. Real sequences, so a failure here is a failure on the actual inputs.
CONTROL_MAINCHAIN = ["INPYLYP", "GEPETGE", "GAEETGE", "GRCTKSIPPICFPD"]
COHORT = ["MEEVD", "DTSRMEEVD", "VPLRARNLPPSFFTEP", "NSKLLSLLRSKT", "KYWWKN",
          "RIIYDRKFLMECRN"]


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f"  -- {detail}" if detail else ""))


def test_every_residue_builds() -> None:
    print("\n[every canonical residue appears in a buildable peptide]")
    for aa in CANONICAL_AA:
        # Length 5 so proline-rich and glycine-only cases still make a real macrocycle.
        seq = f"A{aa}GA{aa}"
        try:
            pep = cyclic_peptide(seq)
            ok = pep.macrocycle_ring_size == 3 * len(seq)
            detail = ""
        except SmilesBuildError as exc:
            ok, detail = False, str(exc)
        check(f"{aa}: {seq} builds and closes", ok, detail)


def test_ring_size_is_three_per_residue() -> None:
    print("\n[the macrocycle is exactly 3 atoms per residue]")
    for seq in CONTROL_MAINCHAIN + COHORT:
        pep = cyclic_peptide(seq)
        check(f"{seq} (L={len(seq)}) ring == {3 * len(seq)}",
              pep.macrocycle_ring_size == 3 * len(seq),
              f"got {pep.macrocycle_ring_size}")


def test_single_molecule() -> None:
    print("\n[the result is one molecule, not two]")
    for seq in CONTROL_MAINCHAIN:
        pep = cyclic_peptide(seq)
        n = len(Chem.GetMolFrags(pep.mol))
        check(f"{seq} is a single fragment", n == 1, f"{n} fragments")


def test_closing_bond_exists_and_is_terminal() -> None:
    print("\n[the closing bond joins C of the last residue to N of the first]")
    for seq in CONTROL_MAINCHAIN:
        pep = cyclic_peptide(seq)
        c_idx, n_idx = pep.closure_atom_indices()
        bond = pep.mol.GetBondBetweenAtoms(c_idx, n_idx)
        check(f"{seq} has a C{len(seq)}-N1 bond", bond is not None)
        c_rec = pep.atom_records[c_idx]
        n_rec = pep.atom_records[n_idx]
        check(f"{seq} closure atoms are named C/N at the termini",
              c_rec.atom_name == "C" and c_rec.residue_number == len(seq)
              and n_rec.atom_name == "N" and n_rec.residue_number == 1,
              f"{c_rec} / {n_rec}")


def test_condensation_removed_exactly_one_oxygen() -> None:
    print("\n[cyclisation condenses: one oxygen fewer than the linear peptide]")
    for seq in CONTROL_MAINCHAIN:
        linear = Chem.MolFromSequence(seq)
        pep = cyclic_peptide(seq)
        check(f"{seq} loses exactly 1 heavy atom vs linear",
              pep.n_heavy_atoms == linear.GetNumAtoms() - 1,
              f"linear {linear.GetNumAtoms()}, cyclic {pep.n_heavy_atoms}")
        # And it must be the hydroxyl, so no carbonyl was eaten: every residue keeps its
        # backbone carbonyl O.
        carbonyls = sum(1 for r in pep.atom_records if r.atom_name == "O")
        check(f"{seq} keeps {len(seq)} backbone carbonyl oxygens",
              carbonyls == len(seq), f"got {carbonyls}")


def test_stereochemistry_is_L() -> None:
    print("\n[every alpha carbon is the L configuration]")
    for seq in CONTROL_MAINCHAIN + ["GRCTKSIPPICFPD"]:
        pep = cyclic_peptide(seq)
        centres = dict(Chem.FindMolChiralCenters(
            pep.mol, includeUnassigned=True, useLegacyImplementation=False))
        bad = []
        for rec in pep.atom_records:
            if rec.atom_name != "CA":
                continue
            tag = centres.get(rec.index)
            if rec.residue_name == "GLY":
                if tag is not None:
                    bad.append((rec.residue_number, rec.residue_name, tag))
            else:
                want = "R" if rec.residue_name == "CYS" else "S"
                if tag != want:
                    bad.append((rec.residue_number, rec.residue_name, tag, want))
        check(f"{seq} all CA are L (Gly none, Cys R, rest S)", not bad, str(bad))


def test_no_unassigned_stereocentres() -> None:
    print("\n[no stereocentre is left unspecified]")
    for seq in CONTROL_MAINCHAIN + COHORT:
        pep = cyclic_peptide(seq)
        unassigned = [i for i, t in Chem.FindMolChiralCenters(
            pep.mol, includeUnassigned=True, useLegacyImplementation=False) if t == "?"]
        check(f"{seq} has no '?' centre", not unassigned, str(unassigned))


def test_atom_records_cover_the_peptide() -> None:
    print("\n[the atom annotation covers every atom and every residue]")
    for seq in CONTROL_MAINCHAIN:
        pep = cyclic_peptide(seq)
        check(f"{seq} one record per atom",
              len(pep.atom_records) == pep.n_heavy_atoms,
              f"{len(pep.atom_records)} vs {pep.n_heavy_atoms}")
        check(f"{seq} record indices are 0..n-1 in order",
              [r.index for r in pep.atom_records] == list(range(pep.n_heavy_atoms)))
        numbers = sorted({r.residue_number for r in pep.atom_records})
        check(f"{seq} residues are 1..{len(seq)} with no gaps",
              numbers == list(range(1, len(seq) + 1)), str(numbers))
        for residue in numbers:
            names = {r.atom_name for r in pep.atom_records
                     if r.residue_number == residue}
            check(f"{seq} residue {residue} has N, CA, C",
                  {"N", "CA", "C"} <= names, str(sorted(names)))


def test_sequence_round_trips() -> None:
    print("\n[the sequence is recoverable from the annotation]")
    for seq in CONTROL_MAINCHAIN + COHORT:
        pep = cyclic_peptide(seq)
        got = sequence_from_records(pep.atom_records)
        check(f"{seq} round-trips", got == seq, f"got {got}")


def test_refusals() -> None:
    print("\n[it refuses rather than returning the wrong molecule]")
    cases = [
        ("empty sequence", ""),
        ("non-canonical letter X", "AAXAA"),
        ("non-canonical letter B", "ABBA"),
        ("lowercase is accepted by normalisation", None),
    ]
    for name, seq in cases[:3]:
        try:
            cyclic_peptide(seq)
            check(f"refuses {name}", False, "built something")
        except SmilesBuildError:
            check(f"refuses {name}", True)

    # An unsupported chemistry must raise, NOT silently return head-to-tail: a disulfide
    # macrocycle is a different molecule and this is the whole point of the check.
    for chem in ("disulfide", "isopeptide", "unspecified", ""):
        try:
            cyclic_peptide("CHPQFC", chemistry=chem)
            check(f"refuses chemistry {chem!r}", False, "returned head-to-tail instead")
        except SmilesBuildError:
            check(f"refuses chemistry {chem!r}", True)

    # Case normalisation is a convenience, not a silent reinterpretation.
    check("lowercase normalises to the same molecule",
          cyclic_peptide("inpylyp").smiles == cyclic_peptide("INPYLYP").smiles)


def test_token_cost_is_reported() -> None:
    print("\n[the AF3 token cost of the ligand is visible]")
    # AF3 tokenises a ligand per atom. These numbers are the reason the arm is not
    # free: a 14-mer goes from 14 tokens as a polymer to ~105 as a ligand.
    for seq in ["MEEVD", "INPYLYP", "NSKLLSLLRSKT", "RIIYDRKFLMECRN",
                "VPLRARNLPPSFFTEP"]:
        pep = cyclic_peptide(seq)
        per_residue = pep.n_heavy_atoms / len(seq)
        check(f"L={len(seq):2d} -> {pep.n_heavy_atoms:3d} atoms "
              f"({per_residue:.1f}/residue)",
              4.0 <= per_residue <= 12.0, f"{per_residue:.2f} per residue")


def test_smiles_output_order_is_a_real_permutation() -> None:
    print("\n[the SMILES output order maps back to residues]")
    for seq in CONTROL_MAINCHAIN + COHORT:
        pep = cyclic_peptide(seq)
        order = pep.smiles_atom_order
        check(f"{seq} order is a permutation of all atoms",
              sorted(order) == list(range(pep.n_heavy_atoms)),
              f"len {len(order)} vs {pep.n_heavy_atoms}")
        check(f"{seq} predicted atom names are unique and complete",
              len(set(pep.predicted_atom_names())) == pep.n_heavy_atoms)
        check(f"{seq} records_in_smiles_order round-trips",
              tuple(r.index for r in pep.records_in_smiles_order()) == tuple(order))
        # Every SMILES position must resolve back to the atom it came from.
        ok = all(pep.smiles_position(order[k]) == k for k in range(len(order)))
        check(f"{seq} smiles_position inverts the order", ok)

    # The permutation is NOT the identity, which is the whole reason it has to be
    # carried: mapping output position k to molecule atom k would be wrong from atom 0.
    pep = cyclic_peptide("INPYLYP")
    check("INPYLYP order is not the identity",
          pep.smiles_atom_order != tuple(range(pep.n_heavy_atoms)),
          "identity would make the permutation unnecessary")
    first = pep.atom_records[pep.smiles_atom_order[0]]
    check("INPYLYP SMILES position 0 is ILE1 CD1, not ILE1 N",
          (first.residue_name, first.residue_number, first.atom_name) == ("ILE", 1, "CD1"),
          f"got {first.residue_name}{first.residue_number} {first.atom_name}")


def test_af3_atom_names_regression() -> None:
    print("\n[AF3's observed atom naming, measured not assumed]")
    # Pinned from AF3 131ab9e7's own output for the 5TU6 preflight: it named the ligand's
    # atoms <element><running index per element> in SMILES order, and all 62 matched.
    # If a later AF3 changes this, the converter must fail rather than mis-assign, so the
    # expectation is pinned here.
    pep = cyclic_peptide("INPYLYP")
    names = pep.predicted_atom_names()
    check("first ten names are AF3's observed sequence",
          names[:10] == ("C1", "C2", "C3", "C4", "C5", "N1", "C6", "O1", "C7", "C8"),
          str(names[:10]))
    counts: dict[str, int] = {}
    for n in names:
        el = n.rstrip("0123456789")
        counts[el] = counts.get(el, 0) + 1
    # C44 N8 O10 = 62. Each part is accountable: 7 backbone N plus Asn's side-chain N;
    # 7 backbone carbonyl O plus Asn's side-chain O plus two Tyr hydroxyls -- and NOT a
    # C-terminal hydroxyl, which cyclisation condensed away.
    check("INPYLYP element tally C44 N8 O10", counts == {"C": 44, "N": 8, "O": 10},
          str(counts))
    check("names cover every atom exactly once", len(set(names)) == len(names) == 62)


def test_canonical_smiles_regression() -> None:
    print("\n[canonical SMILES regression, pinning RDKit's output]")
    # 5TU6's peptide. If RDKit changes canonicalisation or residue templates, this
    # fails here rather than silently changing what gets folded.
    expected = (
        "CC[C@H](C)[C@@H]1NC(=O)[C@@H]2CCCN2C(=O)[C@H](Cc2ccc(O)cc2)NC(=O)"
        "[C@H](CC(C)C)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)"
        "[C@H](CC(N)=O)NC1=O"
    )
    got = cyclic_peptide("INPYLYP").smiles
    check("INPYLYP canonical SMILES unchanged", got == expected,
          f"\n      expected {expected}\n      got      {got}")
    # Whatever the string, it must parse back to the same molecule.
    reparsed = Chem.MolFromSmiles(got)
    check("the emitted SMILES reparses",
          reparsed is not None and Chem.MolToSmiles(reparsed) == got)
    check("rdkit version recorded", True, f"rdkit {Chem.rdBase.rdkitVersion}")


def main() -> int:
    print(f"rdkit {Chem.rdBase.rdkitVersion}")
    test_every_residue_builds()
    test_ring_size_is_three_per_residue()
    test_single_molecule()
    test_closing_bond_exists_and_is_terminal()
    test_condensation_removed_exactly_one_oxygen()
    test_stereochemistry_is_L()
    test_no_unassigned_stereocentres()
    test_atom_records_cover_the_peptide()
    test_sequence_round_trips()
    test_refusals()
    test_token_cost_is_reported()
    test_smiles_output_order_is_a_real_permutation()
    test_af3_atom_names_regression()
    test_canonical_smiles_regression()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
