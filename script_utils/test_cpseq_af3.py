#!/usr/bin/env python3
"""Checks for the AF3 adapter. No GPU, no model, no network.

The important ones here are the REFUSALS. The failure this module exists to prevent is
silent: AF3 returns ligand coordinates in SMILES-string order, which is not the order of
the molecule RDKit built, and attaching them to residues by position produces a file that
looks like a peptide, parses, scores, and yields a dG. Nothing downstream can detect it.
So `map_ligand_atoms` asserts the mapping on names AND on bond lengths, and the tests
below drive each assertion with the wrong input on purpose.

The synthetic mmCIF is generated FROM the peptide, so the expected answer is known
exactly. A real AF3 file is also checked when one is present (the 5TU6 preflight), but
the real file cannot exercise the failure paths -- it is correct.

Run:  python script_utils/test_cpseq_af3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rdkit import RDLogger  # noqa: E402

from proteinfoundation.cpseq.adapters.af3 import (  # noqa: E402
    MAX_TOKENS,
    AF3Config,
    AF3InputError,
    AF3MappingError,
    ReceptorChain,
    build_input_json,
    input_hash,
    map_ligand_atoms,
    read_model_cif,
    token_estimate,
    write_complex_pdb,
)
from proteinfoundation.cpseq.smiles import cyclic_peptide  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PASS, FAIL = 0, 0
REAL_CIF = Path("/zfsauton/scratch/yixiz/cpseq/af3_preflight/5TU6/af3_out/"
                "preflight_5tu6/preflight_5tu6_model.cif")


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f"  -- {detail}" if detail else ""))


_CIF_HEADER = """data_synthetic
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def synthetic_cif(
    sequence: str, *, receptor_chain: str = "A", names: list[str] | None = None,
    scramble_coords: bool = False, drop_last: bool = False,
) -> tuple[str, object]:
    """An AF3-shaped mmCIF whose ligand coordinates ARE the peptide's own 3D conformer.

    Using a real embedded conformer matters: the bond-length assertion is only a
    meaningful test if the correct mapping actually passes it.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    peptide = cyclic_peptide(sequence)
    mol = Chem.Mol(peptide.mol)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()

    atom_names = names or list(peptide.predicted_atom_names())
    order = list(peptide.smiles_atom_order)
    if scramble_coords:
        # Rotate the coordinate assignment by one. Names stay correct, so ONLY the
        # bond-length assertion can catch this -- which is the point of having it.
        order = order[1:] + order[:1]
    if drop_last:
        atom_names = atom_names[:-1]
        order = order[:-1]

    rows = []
    serial = 1
    # A small receptor, so chain handling is exercised.
    for i in range(3):
        rows.append(
            f"ATOM {serial} N N . GLY {receptor_chain} 1 {i + 1} ? "
            f"{10.0 + i:.3f} 0.000 0.000 1.00 50.00 {i + 1} {receptor_chain} 1"
        )
        serial += 1
    for k, mol_idx in enumerate(order):
        p = conf.GetAtomPosition(mol_idx)
        element = peptide.mol.GetAtomWithIdx(mol_idx).GetSymbol()
        rows.append(
            f"HETATM {serial} {element} {atom_names[k]} . LIG_L L 2 . ? "
            f"{p.x:.3f} {p.y:.3f} {p.z:.3f} 1.00 40.00 1 L 1"
        )
        serial += 1
    return _CIF_HEADER + "\n".join(rows) + "\n#\n", peptide


def test_input_json() -> None:
    print("\n[input JSON]")
    rec = (ReceptorChain(chain_id="A", sequence="M" * 120, unpaired_msa=""),)
    text, pep, budget = build_input_json(
        name="t", receptor_chains=rec, sequence="INPYLYP", chemistry="mainchain")
    import json
    job = json.loads(text)
    check("dialect/version are AF3's", job["dialect"] == "alphafold3" and job["version"] == 1)
    check("one protein and one ligand entity",
          [list(s)[0] for s in job["sequences"]] == ["protein", "ligand"])
    lig = [s for s in job["sequences"] if "ligand" in s][0]["ligand"]
    check("ligand carries the macrocyclic SMILES", lig["smiles"] == pep.smiles)
    check("no bondedAtomPairs emitted (the ring is internal to the ligand)",
          "bondedAtomPairs" not in job)
    prot = [s for s in job["sequences"] if "protein" in s][0]["protein"]
    check("empty MSA is passed through as a deliberate single-sequence request",
          prot["unpairedMsa"] == "" and prot["pairedMsa"] == "")
    check("templates default to an empty list, never omitted", prot["templates"] == [])

    check("token budget counts the ligand per atom",
          budget["ligand_tokens"] == pep.n_heavy_atoms == 62, str(budget))
    check("multiplier against a polymer chain is reported",
          budget["ligand_tokenisation_multiplier"] > 8.0)


def test_input_refusals() -> None:
    print("\n[input refusals]")
    rec = (ReceptorChain(chain_id="A", sequence="M" * 50, unpaired_msa=""),)
    for chem in ("disulfide", "isopeptide", "nonsense"):
        try:
            build_input_json(name="t", receptor_chains=rec, sequence="CHPQFC", chemistry=chem)
            check(f"refuses chemistry {chem!r}", False, "built a job")
        except (AF3InputError, Exception):
            check(f"refuses chemistry {chem!r}", True)
    try:
        build_input_json(name="t", receptor_chains=(), sequence="INPYLYP")
        check("refuses a job with no receptor", False)
    except AF3InputError:
        check("refuses a job with no receptor", True)
    try:
        build_input_json(name="t", sequence="INPYLYP",
                         receptor_chains=(ReceptorChain(chain_id="A", sequence="MKV"),))
        check("refuses an unset MSA rather than running AF3's data pipeline", False)
    except AF3InputError:
        check("refuses an unset MSA rather than running AF3's data pipeline", True)
    # The token ceiling must bite, since a ligand-tokenised peptide inflates it.
    try:
        build_input_json(name="t", sequence="INPYLYP", receptor_chains=(
            ReceptorChain(chain_id="A", sequence="M" * (MAX_TOKENS + 10), unpaired_msa=""),))
        check("refuses past the token ceiling", False)
    except AF3InputError:
        check("refuses past the token ceiling", True)


def test_input_hash() -> None:
    print("\n[prediction identity]")
    rec = (ReceptorChain(chain_id="A", sequence="M" * 60, unpaired_msa=""),)
    a, _, _ = build_input_json(name="t", receptor_chains=rec, sequence="INPYLYP")
    b, _, _ = build_input_json(name="t", receptor_chains=rec, sequence="INPYLYG")
    cfg1 = AF3Config(model_seeds=(1,))
    cfg2 = AF3Config(model_seeds=(2,))
    check("same input and config -> same hash", input_hash(a, cfg1) == input_hash(a, cfg1))
    check("a different sequence -> different hash", input_hash(a, cfg1) != input_hash(b, cfg1))
    check("a different seed set -> different hash", input_hash(a, cfg1) != input_hash(a, cfg2))
    check("the compile cache does NOT change prediction identity",
          input_hash(a, AF3Config(model_seeds=(1,), jax_compilation_cache_dir="/x"))
          == input_hash(a, cfg1),
          "it changes speed, not output")


def test_mapping_accepts_the_truth(tmp: Path) -> None:
    print("\n[mapping accepts a correct AF3-shaped file]")
    text, pep = synthetic_cif("INPYLYP")
    p = tmp / "good.cif"
    p.write_text(text)
    atoms = read_model_cif(p)
    lig = [a for a in atoms if a.comp_id.startswith("LIG_")]
    check("reader finds 3 receptor + 62 ligand atoms",
          len(atoms) == 65 and len(lig) == 62, f"{len(atoms)}/{len(lig)}")
    pairs = map_ligand_atoms(lig, pep)
    check("every ligand atom is mapped", len(pairs) == 62)
    first = pep.atom_records[pairs[0][1]]
    check("position 0 maps to ILE1 CD1 (not ILE1 N)",
          (first.residue_name, first.residue_number, first.atom_name) == ("ILE", 1, "CD1"),
          f"{first}")


def test_mapping_refusals(tmp: Path) -> None:
    print("\n[mapping refuses wrong input instead of mis-assigning]")
    # Wrong atom count.
    text, pep = synthetic_cif("INPYLYP", drop_last=True)
    p = tmp / "short.cif"; p.write_text(text)
    lig = [a for a in read_model_cif(p) if a.comp_id.startswith("LIG_")]
    try:
        map_ligand_atoms(lig, pep)
        check("refuses a different number of ligand atoms", False)
    except AF3MappingError:
        check("refuses a different number of ligand atoms", True)

    # Right count, WRONG names.
    names = list(pep.predicted_atom_names())
    names[5], names[6] = names[6], names[5]
    text, pep2 = synthetic_cif("INPYLYP", names=names)
    p = tmp / "names.cif"; p.write_text(text)
    lig = [a for a in read_model_cif(p) if a.comp_id.startswith("LIG_")]
    try:
        map_ligand_atoms(lig, pep2)
        check("refuses when atom NAMES do not match the expected convention", False)
    except AF3MappingError:
        check("refuses when atom NAMES do not match the expected convention", True)

    # Right count, right names, coordinates rotated by one: only the bond-length
    # assertion can catch this. This is the silent failure the module exists to stop.
    text, pep3 = synthetic_cif("INPYLYP", scramble_coords=True)
    p = tmp / "scrambled.cif"; p.write_text(text)
    lig = [a for a in read_model_cif(p) if a.comp_id.startswith("LIG_")]
    try:
        map_ligand_atoms(lig, pep3)
        check("refuses a permuted mapping that names cannot catch", False,
              "this is the silent mis-assignment case")
    except AF3MappingError:
        check("refuses a permuted mapping that names cannot catch", True)


def test_pdb_writing(tmp: Path) -> None:
    print("\n[PDB writing]")
    text, pep = synthetic_cif("INPYLYP")
    cif = tmp / "w.cif"; cif.write_text(text)
    out = tmp / "w.pdb"
    rec = write_complex_pdb(cif_path=cif, peptide=pep, out_path=out)
    check("receptor chain and peptide chain are distinct",
          rec["receptor_chains"] == ["A"] and rec["peptide_chain"] == "B")
    check("all 62 peptide atoms written", rec["n_peptide_atoms"] == 62)
    body = out.read_text()
    check("peptide is chain B with real residue names",
          any(l.startswith("ATOM") and l[21] == "B" and l[17:20].strip() == "ILE"
              for l in body.splitlines()))
    check("a CONECT records the closing bond",
          any(l.startswith("CONECT") for l in body.splitlines()),
          "without it the file asserts a linear peptide to any bond-reading parser")
    names = {l[12:16].strip() for l in body.splitlines()
             if l.startswith("ATOM") and l[21] == "B"}
    check("peptide atom names are PDB names, not C1/N1/O1",
          {"N", "CA", "C", "O"} <= names and not {"C1", "N1"} & names, str(sorted(names))[:120])

    # A receptor that already occupies chain B must be refused, not silently merged.
    text, pep2 = synthetic_cif("INPYLYP", receptor_chain="B")
    cif2 = tmp / "collide.cif"; cif2.write_text(text)
    try:
        write_complex_pdb(cif_path=cif2, peptide=pep2, out_path=tmp / "c.pdb")
        check("refuses a receptor chain colliding with the peptide chain", False)
    except AF3MappingError:
        check("refuses a receptor chain colliding with the peptide chain", True)


def test_against_real_af3_output(tmp: Path) -> None:
    print("\n[the real AF3 output, when present]")
    if not REAL_CIF.exists():
        print(f"  skip  {REAL_CIF} not present (run scripts/cpseq_af3_preflight.sbatch)")
        return
    pep = cyclic_peptide("INPYLYP")
    rec = write_complex_pdb(cif_path=REAL_CIF, peptide=pep, out_path=tmp / "real.pdb")
    check("real AF3 output maps and converts", rec["n_peptide_atoms"] == 62)
    check("AF3's closing bond is inside the v3 amide window 1.23-1.43 A",
          1.23 <= rec["af3_closure_bond_A"] <= 1.43,
          f"{rec['af3_closure_bond_A']} A")
    # And the project's own geometry module must agree with the adapter.
    from proteinfoundation.cpseq.geometry import closure_geometry
    geo = closure_geometry(tmp / "real.pdb", "B", "mainchain")
    check("project geometry module reads the converted file",
          geo.get("status") == "ok", str(geo.get("status")))
    check("adapter and geometry module agree on the closing bond",
          abs(rec["af3_closure_bond_A"] - geo["bond_length_A"]) < 0.002,
          f"{rec['af3_closure_bond_A']} vs {geo['bond_length_A']}")


def main() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_input_json()
        test_input_refusals()
        test_input_hash()
        test_mapping_accepts_the_truth(tmp)
        test_mapping_refusals(tmp)
        test_pdb_writing(tmp)
        test_against_real_af3_output(tmp)
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
